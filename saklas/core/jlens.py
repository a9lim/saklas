"""Jacobian lens: corpus-averaged residual→output transport + readout.

Implements the Jacobian lens of Gurnee et al., "Verbalizable Representations
Form a Global Workspace in Language Models" (Transformer Circuits, 2026):
per-layer matrices ``J_l = E[∂h_final/∂h_l]`` — the average first-order effect
of a layer-l residual on the final-layer residual, over positions and a text
corpus. ``lens(h) = softmax(W_U · norm(J_l h))`` ranks the vocabulary by what
an activation is disposed to make the model say; the J-lens *direction* for
vocab id v at layer l is ``W_U[v] @ J_l`` — a per-layer d-vector with the same
shape contract as any saklas steering direction, which is what lets a
``jlens/<word>`` term ride the ordinary profile registry.

Estimator (reference: github.com/anthropics/jacobian-lens): consecutive prompts
share one right-padded autograd graph, then batched VJPs recover
``dim_batch`` output rows per backward without replicating the forward graph.
For output dimension ``r``, the cotangent is injected at every valid target
position, so the gradient at source position ``t`` is
``Σ_{t'≥t} ∂h_final[t',r]/∂h_l[t]``; source positions are averaged within
each prompt and prompt Jacobians are then summed, preserving exact equal-prompt
weighting even for ragged lengths. The first ``SKIP_FIRST_POSITIONS`` positions
(attention sinks) and the final position are excluded from both cotangents and
the source mean. Backends without batched-VJP coverage fall back to exact
unreplicated scalar VJPs; replicated VJPs remain an explicit reference mode.

This is the ONLY module in saklas that runs backward passes. The fit builds
its own autograd-enabled forward (``torch.enable_grad()`` + a grad-seeding
pre-hook on the first block) — the ``inference_mode`` capture machinery in
``vectors.py`` cannot be reused, because inference tensors never re-enter
autograd. Per-layer grads come from ``torch.autograd.grad(final, sources)``
— NOT ``backward()`` + ``retain_grad()`` (``.grad`` accumulates across the
multi-backward loop and would corrupt the one-hot-cotangent rows) — which
also stops the graph walk at the shallowest requested source layer, so a
band-restricted fit never backprops below its lowest source. A terminal-layer
hook captures the target residual and aborts the rest of the forward before
the final norm and full-vocabulary head. Row blocks are staged in bounded
per-layer stripes, validated after transfer, and committed directly into the
CPU fp32 cross-prompt accumulator. If a later VJP OOMs, the graph is rebuilt at
the first uncommitted row instead of discarding earlier work. The MPS sync
budget is bounded from both sides: a fully unsynced loop can exhaust Metal's
asynchronous command queue, so periodic drains plus the zero-row guard remain.
The fit is compute-bound (each prompt ≈ ``d_model × 2`` forward-equivalents
of backward, dim_batch-invariant); restricting ``source_layers`` is the one
lever that removes work. Entries are O(1), so fp16 storage downstream is
safe.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from saklas.core.errors import SaklasError, is_out_of_memory_error

log = logging.getLogger(__name__)

#: Positions before this index are excluded from the Jacobian average — early
#: positions act as attention sinks with atypical residual statistics.
SKIP_FIRST_POSITIONS = 16
DEFAULT_SEQ_LEN = 128
#: Output dims per backward pass. Total backward FLOPs are dim_batch-invariant
#: (pass count halves as pass width doubles), so this knob trades memory for
#: per-pass overhead and barely moves wall time — measured on an M5 Max /
#: gemma-3-4b, 8 is the sweet spot (93.6s/prompt vs 96.9s at 32, 102.5s at
#: 64, identical output). Halves automatically on OOM.
DEFAULT_DIM_BATCH = 8
# Consecutive corpus prompts per autograd graph on CPU/CUDA.  Their Jacobians
# remain equal-prompt weighted (not equal-token weighted); MPS defaults to one
# until real-device measurements justify the larger graph there.  OOM backoff
# reduces this independently of ``dim_batch``.
DEFAULT_PROMPT_BATCH = 4
#: Checkpoint cadence (prompts) for resumable fits.
DEFAULT_CHECKPOINT_EVERY = 25
#: Backward passes between queue drains on MPS. Metal reports command-queue
#: exhaustion as an *asynchronous* command-buffer error — no Python exception,
#: the encoded ops silently never complete — so an unsynchronized pass loop
#: that runs ahead of the device turns into all-zero gradients, not an OOM.
#: A bounded drain every few passes caps the in-flight transients; the
#: all-zero fold guard below catches whatever still slips through.
_MPS_SYNC_EVERY_PASSES = 4
#: A complete fp32 lens per staging tier is a multi-GiB cliff. Keep only this
#: many output rows per layer on device + host, validate/commit the stripe, and
#: resume from the first uncommitted row after an OOM.
_ROW_STRIPE = 256


class JacobianLensError(RuntimeError, SaklasError):
    """Raised when a Jacobian-lens fit or readout cannot proceed."""

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


class LensNotFittedError(JacobianLensError):
    """Raised when a lens artifact is required but absent for the model."""

    def user_message(self) -> tuple[int, str]:
        return (404, str(self) or self.__class__.__name__)


class MultiTokenWordError(ValueError, SaklasError):
    """Raised when a ``jlens/<word>`` atom has no single-token vocabulary id."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def resolve_word_token(tokenizer: Any, word: str) -> int:
    """Map a word to the single vocab id its J-lens direction should use.

    Tries the leading-space piece first (``"▁word"``/``"Ġword"`` — the form
    the model actually emits in running text), then the bare piece. A
    candidate only counts when it round-trips: ``decode([id]).strip()`` must
    equal the word, so a normalizing tokenizer can't silently match a merge
    artifact. Raises :class:`MultiTokenWordError` (listing the pieces) when
    neither form is a single token.
    """
    pieces: list[str] = []
    for cand in (f" {word}", word):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1 and tokenizer.decode(ids).strip() == word:
            return int(ids[0])
        if not pieces and len(ids) > 1:
            pieces = [tokenizer.decode([i]) for i in ids]
    raise MultiTokenWordError(
        f"{word!r} is not a single token in this vocabulary"
        + (f" (pieces: {pieces})" if pieces else "")
        + " — the Jacobian lens can only address single-token words"
    )


class JacobianLens:
    """Fitted per-layer ``J_l`` matrices plus the readout primitives.

    ``jacobians`` maps source layer index → ``[d_model, d_model]`` fp32
    matrix transporting that layer's residual into the final-layer basis.
    """

    def __init__(
        self,
        jacobians: Mapping[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.jacobians = {int(l): j.to(torch.float32) for l, j in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = int(n_prompts)
        self.d_model = int(d_model)
        self._atom_norm_cache: dict[tuple[int, str, str, int, tuple[int, ...]], torch.Tensor] = {}

    def __repr__(self) -> str:
        span = (
            f"[{self.source_layers[0]}..{self.source_layers[-1]}]"
            if self.source_layers
            else "[]"
        )
        return (
            f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"source_layers={span} ({len(self.source_layers)} layers))"
        )

    def transport(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a layer-``layer`` residual ``[..., d]`` into the final basis."""
        if layer not in self.jacobians:
            raise LensNotFittedError(
                f"layer {layer} not in fitted lens layers "
                f"{self.source_layers[:3]}..{self.source_layers[-3:]}"
            )
        J = self.jacobians[layer].to(hidden.device)
        return hidden.to(torch.float32) @ J.T

    def token_direction(
        self,
        token_id: int,
        unembed: torch.Tensor,
        *,
        layers: Sequence[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Per-layer J-lens direction for one vocab id: ``W_U[v] @ J_l``.

        Returns fp32 CPU tensors in the ``dict[int, Tensor]`` shape every
        saklas profile consumer (``fold_directions_to_subspace``,
        ``Profile``) expects.
        """
        w = unembed[token_id].detach().to(torch.float32).cpu()
        requested = self.source_layers if layers is None else [int(l) for l in layers]
        missing = [l for l in requested if l not in self.jacobians]
        if missing:
            raise LensNotFittedError(
                f"layers {missing} not in fitted lens layers "
                f"{self.source_layers[:3]}..{self.source_layers[-3:]}"
            )
        return {l: w @ self.jacobians[l] for l in requested}

    def select_layers(self, layers: Sequence[int]) -> "JacobianLens":
        """Return a view-like lens containing only ``layers``.

        The tensors are shared with ``self``; callers that persist the result
        will materialize their own fp16 copy through ``save_lens``.
        """
        requested = sorted(set(int(l) for l in layers))
        missing = [l for l in requested if l not in self.jacobians]
        if missing:
            raise LensNotFittedError(
                f"layers {missing} not in fitted lens layers "
                f"{self.source_layers[:3]}..{self.source_layers[-3:]}"
            )
        return JacobianLens(
            {layer: self.jacobians[layer] for layer in requested},
            n_prompts=self.n_prompts,
            d_model=self.d_model,
        )

    def atom_norms(self, layer: int, unembed: torch.Tensor) -> torch.Tensor:
        """Cached per-token norms of the layer's J-lens dictionary atoms."""
        if layer not in self.jacobians:
            raise LensNotFittedError(
                f"layer {layer} not in fitted lens layers "
                f"{self.source_layers[:3]}..{self.source_layers[-3:]}"
            )
        key = (
            int(layer),
            str(unembed.device),
            str(unembed.dtype),
            int(unembed.data_ptr()),
            tuple(int(x) for x in unembed.shape),
        )
        cached = self._atom_norm_cache.get(key)
        if cached is not None:
            return cached
        norms = _atom_norms(
            self.jacobians[layer].to(device=unembed.device, dtype=torch.float32),
            unembed,
        )
        self._atom_norm_cache[key] = norms
        return norms

    @classmethod
    def merge(cls, lenses: Sequence["JacobianLens"]) -> "JacobianLens":
        """Combine lenses fitted on disjoint prompt subsets
        (``n_prompts``-weighted mean)."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.source_layers != first.source_layers
                or other.d_model != first.d_model
            ):
                raise ValueError("lenses disagree on source_layers / d_model")
        total = sum(lens.n_prompts for lens in lenses)
        if total <= 0:
            raise ValueError("merge() needs lenses with n_prompts > 0")
        merged: dict[int, torch.Tensor] = {}
        for layer in first.source_layers:
            acc = first.jacobians[layer].clone().mul_(first.n_prompts / total)
            for lens in lenses[1:]:
                acc.add_(lens.jacobians[layer], alpha=lens.n_prompts / total)
            merged[layer] = acc
        return cls(merged, n_prompts=total, d_model=first.d_model)

    @classmethod
    def merge_into(
        cls, lenses: Sequence["JacobianLens"], *, target: int = -1,
    ) -> "JacobianLens":
        """Prompt-weighted merge using one caller-owned lens as the destination.

        Resume fitting owns the newly fitted tail, so allocating another complete
        fp32 artifact at final merge only multiplies peak RAM.  This explicit
        mutating variant preserves :meth:`merge`'s non-mutating public behavior
        while letting orchestration recycle that tail in place.
        """
        if not lenses:
            raise ValueError("merge_into() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.source_layers != first.source_layers
                or other.d_model != first.d_model
            ):
                raise ValueError("lenses disagree on source_layers / d_model")
        total = sum(lens.n_prompts for lens in lenses)
        if total <= 0:
            raise ValueError("merge_into() needs lenses with n_prompts > 0")
        owner = lenses[target]
        owner_weight = owner.n_prompts / total
        for layer in owner.source_layers:
            dst = owner.jacobians[layer]
            dst.mul_(owner_weight)
            for lens in lenses:
                if lens is owner:
                    continue
                dst.add_(lens.jacobians[layer], alpha=lens.n_prompts / total)
        owner.n_prompts = total
        owner._atom_norm_cache.clear()
        return owner

    @classmethod
    def union_layers(cls, lenses: Sequence["JacobianLens"]) -> "JacobianLens":
        """Combine same-corpus lenses that cover different source layers.

        Unlike :meth:`merge`, this is not a prompt-weighted average; every input
        must describe the same prompt set and ``d_model``.  Later inputs replace
        duplicate layers, which lets a missing-layer top-up overwrite a stale
        partial layer cleanly.
        """
        if not lenses:
            raise ValueError("union_layers() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if other.n_prompts != first.n_prompts or other.d_model != first.d_model:
                raise ValueError("lenses disagree on n_prompts / d_model")
        union: dict[int, torch.Tensor] = {}
        for lens in lenses:
            union.update(lens.jacobians)
        return cls(union, n_prompts=first.n_prompts, d_model=first.d_model)


def lens_logits(
    lens: JacobianLens,
    hidden_per_layer: Mapping[int, torch.Tensor],
    *,
    unembed: torch.Tensor,
    final_norm: nn.Module,
    layers: Sequence[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Full-vocabulary lens readout ``W_U · norm(J_l h)`` per requested layer.

    ``hidden_per_layer`` maps layer index → residual ``[..., d]``. The matvec
    runs in the unembedding's own dtype (a fp32 copy of a ~256k-row W_U would
    be gigabytes); ranking precision matches the model's own logit path.
    Returns fp32 logits ``[..., vocab]`` per layer, on the unembed's device.
    """
    requested = list(layers) if layers is not None else lens.source_layers
    out: dict[int, torch.Tensor] = {}
    for layer in requested:
        h = hidden_per_layer[layer]
        transported = lens.transport(h.to(unembed.device), layer)
        normed = final_norm(transported)
        out[layer] = (normed.to(unembed.dtype) @ unembed.T).float()
    return out


def topk_logprobs(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k log-probabilities without materializing a full log-softmax tensor.

    ``topk(log_softmax(x))`` has the same indices as ``topk(x)``.  Computing
    only the selected log-probabilities saves one vocab-sized tensor allocation
    per readout row, which matters for large vocabularies and multi-layer
    J-lens sweeps.
    """
    logits_f = logits.float()
    vals, idxs = logits_f.topk(k, dim=-1)
    vals = vals - logits_f.logsumexp(dim=-1, keepdim=True)
    return vals, idxs


def aggregate_readout(
    logits: torch.Tensor,
    depths: "Sequence[float]",
    *,
    top_k: int = 8,
) -> list[tuple[int, float, float, float]]:
    """Layer-aggregate a per-layer lens readout into one ranked token list.

    ``logits`` is ``[L, vocab]`` — one full-vocabulary lens readout row per
    layer (:func:`lens_logits` output stacked); ``depths`` the matching
    normalized layer depths in ``[0, 1]`` (``layer / (n_layers − 1)``).
    Returns ``[(vocab_id, strength, com, spread), ...]`` sorted by
    descending strength.

    Raw lens logits are uncalibrated across layers, so each layer is first
    put through its own softmax.  From the per-layer probabilities
    ``p_l(v)``, two statistics per token:

    - ``strength = mean_l p_l(v)`` — the mean probability over the layer
      band, in ``[0, 1]``.  Uniform layer weights: the softmax already
      lets a confident layer dominate the ranking, so extra confidence
      weighting would double-count.
    - ``com``/``spread`` — the depth center of mass (+ std) weighted by
      the same per-layer probability ``p_l(v)``.  The workspace-band
      readout is sharp, not diffuse (median per-layer max ≈ 0.8 on
      gemma-3-4b) — what changes over depth is *which* token leads, so a
      token's probability profile over depth IS its depth signal.
      Probability mass also discounts a genuinely diffuse (noise) layer
      automatically; the former within-layer salience ``p_l/max_v' p_l``
      handed such a layer's relative-top token a full vote regardless of
      absolute mass (in band the two weightings agree to ≲0.01 — one
      channel, ``p_l``, now backs every readout statistic).

    Top-k selection runs on the aggregated full-vocab strengths — a
    per-layer top-k union would miss a token that ranks mid-pack at every
    layer but top at none.
    """
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError(
            f"aggregate_readout expects [layers, vocab] logits, got shape "
            f"{tuple(logits.shape)}"
        )
    if len(depths) != logits.shape[0]:
        raise ValueError(
            f"aggregate_readout: {logits.shape[0]} logit rows but "
            f"{len(depths)} depths"
        )
    probs = logits.float().softmax(dim=-1)                      # [L, V]
    strength = probs.mean(dim=0)                                # [V]
    d = torch.tensor(
        [float(x) for x in depths], dtype=torch.float32, device=probs.device,
    ).unsqueeze(-1)                                             # [L, 1]
    mass = probs.sum(dim=0).clamp_min(1e-12)                    # [V]
    com = (probs * d).sum(dim=0) / mass                         # [V]
    var = (probs * (d - com.unsqueeze(0)) ** 2).sum(dim=0) / mass
    spread = var.clamp_min(0.0).sqrt()
    k = min(int(top_k), int(strength.shape[-1]))
    vals, idxs = strength.topk(k)
    # one batched host transfer for the whole readout
    stats = torch.stack([vals, com[idxs], spread[idxs]]).cpu()
    idx_cpu = idxs.cpu()
    return [
        (
            int(idx_cpu[j]),
            float(stats[0, j]),
            float(stats[1, j]),
            float(stats[2, j]),
        )
        for j in range(k)
    ]


def token_readout_stats(
    logits: torch.Tensor,
    depths: "Sequence[float]",
    token_ids: "Sequence[int]",
) -> list[tuple[float, float, float, list[float]]]:
    """Per-token readout statistics for pinned vocabulary ids.

    The single-token restriction of :func:`aggregate_readout`: the same
    per-layer softmax calibration, read at the requested ``token_ids``
    instead of selected top-k.  For each id:

    - ``strength = mean_l p_l(v)`` — mean band probability, the aggregate
      readout's ranking stat and the ONE probe/gate/display channel
      (objective and apples-to-apples across tokens and layers, unlike a
      within-layer max normalization).
    - ``com`` / ``spread`` — depth center of mass (+ std), weighted by the
      same per-layer probability ``p_l(v)`` exactly like
      :func:`aggregate_readout` — the one channel backs every statistic
      (the band readout is sharp, so a token's probability profile over
      depth is its depth signal; a diffuse noise layer's vote is
      discounted by its own lack of mass).
    - ``per_layer`` — ``[p_l, ...]`` aligned with the logit rows.

    Returns one ``(strength, com, spread, per_layer)`` tuple per requested
    id, with a single batched host transfer.
    """
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError(
            f"token_readout_stats expects [layers, vocab] logits, got shape "
            f"{tuple(logits.shape)}"
        )
    if len(depths) != logits.shape[0]:
        raise ValueError(
            f"token_readout_stats: {logits.shape[0]} logit rows but "
            f"{len(depths)} depths"
        )
    if not token_ids:
        return []
    probs = logits.float().softmax(dim=-1)                      # [L, V]
    ids = torch.tensor(
        [int(v) for v in token_ids], dtype=torch.long, device=probs.device,
    )
    p = probs.index_select(-1, ids)                             # [L, K]
    strength = p.mean(dim=0)                                    # [K]
    d = torch.tensor(
        [float(x) for x in depths], dtype=torch.float32, device=probs.device,
    ).unsqueeze(-1)                                             # [L, 1]
    mass = p.sum(dim=0).clamp_min(1e-12)                        # [K]
    com = (p * d).sum(dim=0) / mass                             # [K]
    var = (p * (d - com.unsqueeze(0)) ** 2).sum(dim=0) / mass
    spread = var.clamp_min(0.0).sqrt()
    # one batched host transfer: 3 aggregate rows + the per-layer block
    host = torch.cat(
        [torch.stack([strength, com, spread]), p], dim=0,
    ).cpu()                                                     # [3+L, K]
    n_layers = int(logits.shape[0])
    out: list[tuple[float, float, float, list[float]]] = []
    for j in range(len(token_ids)):
        per_layer = [float(host[3 + l, j]) for l in range(n_layers)]
        out.append(
            (
                float(host[0, j]), float(host[1, j]), float(host[2, j]),
                per_layer,
            )
        )
    return out


class JSpaceDecomposition:
    """One layer's sparse nonnegative split of a direction against the J-lens
    dictionary: ``share`` = fraction of the direction's variance carried by
    the selected atoms, ``tokens`` = ``[(vocab_id, coeff), ...]`` sorted by
    descending coefficient."""

    __slots__ = ("layer", "share", "tokens")

    def __init__(self, layer: int, share: float, tokens: list[tuple[int, float]]) -> None:
        self.layer = layer
        self.share = share
        self.tokens = tokens

    def __repr__(self) -> str:
        return (
            f"JSpaceDecomposition(layer={self.layer}, share={self.share:.3f}, "
            f"k={len(self.tokens)})"
        )


def sparse_nonneg_decompose(
    target: torch.Tensor,
    jacobian: torch.Tensor,
    unembed: torch.Tensor,
    *,
    layer: int,
    k: int = 16,
    nnls_iters: int = 200,
    atom_norms: torch.Tensor | None = None,
) -> JSpaceDecomposition:
    """Greedy sparse nonnegative pursuit of ``target`` against the J-lens
    dictionary ``D = W_U @ J_l`` (the paper's gradient-pursuit decomposition).

    The dictionary is never materialized (``[vocab, d]`` would be gigabytes
    on a real model): atom scores are the composed matvec
    ``W_U @ (J_l @ residual)``, normalized by chunk-computed atom norms
    (unnormalized inner products would bias selection toward large-norm
    atoms), and only the ≤k selected rows are formed. After each selection
    the coefficients re-solve as a k-dim nonnegative least squares
    (projected gradient with the exact Lipschitz step — the problem is
    tiny). Selection stops early when no atom correlates positively with
    the residual.
    """
    device = unembed.device
    J = jacobian.to(device=device, dtype=torch.float32)
    t = target.detach().to(device=device, dtype=torch.float32)
    t_norm_sq = float(t.pow(2).sum())
    if t_norm_sq == 0.0:
        return JSpaceDecomposition(layer, 0.0, [])
    norms = (
        _atom_norms(J, unembed)
        if atom_norms is None
        else atom_norms.to(device=device, dtype=torch.float32)
    )
    norms = norms.clamp(min=1e-12)

    selected: list[int] = []
    rows: list[torch.Tensor] = []
    coeffs = torch.zeros(0, device=device)
    residual = t.clone()
    for _ in range(k):
        # normalized correlation over the vocabulary, D never materialized
        scores = (unembed @ (J @ residual).to(unembed.dtype)).float() / norms
        if selected:
            scores[torch.tensor(selected, device=device)] = -torch.inf
        best = int(scores.argmax())
        if float(scores[best]) <= 0.0:
            break
        selected.append(best)
        rows.append(unembed[best].float() @ J)
        A = torch.stack(rows)  # [s, d]
        gram = A @ A.T
        b = A @ t
        c = torch.cat([coeffs, coeffs.new_zeros(1)])
        solved = _try_unconstrained_nonnegative(gram, b, device=device)
        if solved is None:
            # CPU hop: eigvalsh is unimplemented on MPS, and the gram is ≤ k×k.
            lipschitz = float(torch.linalg.eigvalsh(gram.cpu())[-1].clamp(min=1e-12))
            for _ in range(nnls_iters):
                c = torch.clamp(c - (gram @ c - b) / lipschitz, min=0.0)
        else:
            c = solved
        coeffs = c
        residual = t - A.T @ coeffs

    share = max(0.0, 1.0 - float(residual.pow(2).sum()) / t_norm_sq)
    pairs = sorted(
        ((tok, float(cf)) for tok, cf in zip(selected, coeffs) if float(cf) > 0.0),
        key=lambda p: -p[1],
    )
    return JSpaceDecomposition(layer, share, pairs)


def _try_unconstrained_nonnegative(
    gram: torch.Tensor,
    b: torch.Tensor,
    *,
    device: torch.device,
    tol: float = 1e-7,
) -> torch.Tensor | None:
    """Exact tiny least-squares solve when its coefficients are already >= 0.

    The greedy J-space step solves ``min ||A^T c - t||²`` over the selected
    atoms.  If the unconstrained normal-equation solution is nonnegative, it is
    also the NNLS optimum, so the 200-step projected-gradient loop is pure
    overhead.  If any coefficient is negative or the tiny system is singular, the
    caller keeps the existing PGD fallback.
    """
    g_cpu = gram.detach().to("cpu", torch.float32)
    b_cpu = b.detach().to("cpu", torch.float32)
    eye = torch.eye(g_cpu.shape[0], dtype=torch.float32)
    try:
        sol = torch.linalg.solve(g_cpu + 1e-7 * eye, b_cpu)
    except RuntimeError:
        return None
    if not bool(torch.isfinite(sol).all()):
        return None
    if bool((sol < -tol).any()):
        return None
    return sol.clamp(min=0.0).to(device=device, dtype=torch.float32)


def _atom_norms(
    jacobian: torch.Tensor, unembed: torch.Tensor, *, chunk: int = 8192
) -> torch.Tensor:
    """Per-atom norms ``‖W_U[v] @ J‖`` for the whole vocabulary, computed in
    chunks so the ``[vocab, d]`` dictionary never exists in full."""
    norms = torch.empty(unembed.shape[0], device=unembed.device)
    for start in range(0, unembed.shape[0], chunk):
        block = unembed[start : start + chunk].float() @ jacobian
        norms[start : start + chunk] = block.norm(dim=-1)
    return norms


def _output_tensor(output: Any) -> torch.Tensor:
    """A transformer block's residual output (blocks return tuples or tensors)."""
    return output[0] if isinstance(output, tuple) else output


class _BatchedVjpUnavailable(RuntimeError):
    """Internal signal: retry the current prompt with scalar VJPs."""

    def __init__(self, committed_until: int) -> None:
        super().__init__("batched VJP unavailable")
        self.committed_until = committed_until


class _FitForwardComplete(BaseException):
    """Private control flow: final residual captured; skip norm + LM head."""


class _PromptRowsCommitted(RuntimeError):
    """OOM after durable-in-accumulator rows; resume at ``committed_until``."""

    def __init__(self, committed_until: int, message: str) -> None:
        super().__init__(message)
        self.committed_until = committed_until


def _resolve_vjp_mode(mode: str) -> str:
    env = os.environ.get("SAKLAS_JLENS_VJP")
    raw = (env or mode).strip().lower()
    if raw not in {"auto", "batched", "scalar", "replicated"}:
        raise ValueError(
            "vjp_mode must be 'auto', 'batched', 'scalar', or 'replicated' "
            f"(got {mode!r})"
        )
    return raw


def _looks_like_batched_vjp_unsupported(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    if is_out_of_memory_error(exc):
        return False
    return any(
        needle in msg
        for needle in (
            "is_grads_batched",
            "vmap",
            "batching rule",
            "batched grad",
            "not implemented",
            "not supported",
        )
    )


def _install_fit_hooks(
    layer_modules: Sequence[nn.Module],
    sources: Sequence[int],
    final_idx: int,
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def seed_hook(
        _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # With frozen params and integer inputs no autograd graph exists at
        # all — seed a leaf into the residual stream at the first fitted block.
        if args:
            seeded = args[0].detach().clone().requires_grad_(True)
            return (seeded, *args[1:]), kwargs
        seeded = kwargs["hidden_states"].detach().clone().requires_grad_(True)
        return args, {**kwargs, "hidden_states": seeded}

    def make_capture(idx: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            captured[idx] = _output_tensor(output)
            if idx == final_idx:
                raise _FitForwardComplete()

        return hook

    # Seed at the LOWEST source block, not block 0: everything below it then
    # runs graph-free (frozen params + a detached input build no autograd
    # state), so a band-restricted fit pays neither graph memory nor backward
    # depth below its band. For the default all-layer fit this is block 0.
    handles.append(
        layer_modules[min(sources)].register_forward_pre_hook(
            seed_hook, with_kwargs=True,
        )
    )
    for idx in {*sources, final_idx}:
        handles.append(layer_modules[idx].register_forward_hook(make_capture(idx)))
    return captured, handles


def fit_jacobian_lens(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layer_modules: Sequence[nn.Module],
    *,
    source_layers: Sequence[int] | None = None,
    dim_batch: int = DEFAULT_DIM_BATCH,
    prompt_batch: int | None = None,
    max_seq_len: int = DEFAULT_SEQ_LEN,
    skip_first: int = SKIP_FIRST_POSITIONS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    checkpoint_cb: Callable[[JacobianLens], None] | None = None,
    checkpoint_accumulator_cb: (
        Callable[[Mapping[int, torch.Tensor], int, int], None] | None
    ) = None,
    on_progress: Callable[[str], None] | None = None,
    progress_base: int = 0,
    input_id_rows: Sequence[Sequence[int]] | None = None,
    vjp_mode: str = "auto",
) -> JacobianLens:
    """Fit ``J_l`` for every source layer over ``prompts``.

    One graph per prompt microbatch + ``ceil(d_model/dim_batch)`` backwards
    (see the module docstring for the estimator). ``checkpoint_cb`` receives the partial lens
    every ``checkpoint_every`` prompts — the io layer uses it for resumable
    fits (callers merging with a prior shard do so outside this function).
    ``checkpoint_accumulator_cb`` is the allocation-light persistence seam: it
    receives the raw fp32 sums, completed prompt count, and hidden size without
    materializing a second full averaged lens.  ``checkpoint_cb`` remains the
    public compatibility callback. ``input_id_rows`` is the pre-tokenized sibling
    of ``prompts``.  When supplied
    it must be aligned 1:1 with ``prompts`` and already reflect the caller's
    truncation/hash policy; the fit still applies ``max_seq_len`` defensively.
    This lets session-level resume/filtering reuse the IDs it already computed
    instead of tokenizing the corpus a second time.  ``vjp_mode="batched"``
    computes each output-dim block with ``is_grads_batched=True`` from a single
    prompt forward instead of replicating the forward graph; ``"auto"`` tries
    that path first and falls back to exact scalar VJPs if the backend lacks
    vmap coverage. ``prompt_batch`` controls consecutive ragged prompts per
    graph (CPU/CUDA default 4, MPS default 1); both prompt and output-dimension
    batch widths back off independently on device OOM and stay below a proven
    failure ceiling for the rest of the fit.

    Raises :class:`JacobianLensError` when no prompt in the corpus is long
    enough (each needs > ``skip_first + 1`` tokens).
    """
    device = next(model.parameters()).device
    if dim_batch <= 0:
        raise ValueError("dim_batch must be > 0")
    if prompt_batch is not None and prompt_batch <= 0:
        raise ValueError("prompt_batch must be > 0")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be > 0")
    if progress_base < 0:
        raise ValueError("progress_base must be >= 0")
    n_layers = len(layer_modules)
    final_idx = n_layers - 1
    sources = (
        sorted(set(source_layers)) if source_layers is not None else list(range(final_idx))
    )
    if not sources:
        raise ValueError("source_layers must name at least one source layer")
    if any(l < 0 or l >= final_idx for l in sources):
        raise ValueError(
            f"source_layers must lie in [0, {final_idx}) — the final layer is "
            f"the transport target, not a source; got {sources}"
        )
    if input_id_rows is not None and len(input_id_rows) != len(prompts):
        raise ValueError(
            "input_id_rows must be aligned with prompts "
            f"({len(input_id_rows)} != {len(prompts)})"
        )
    requested_vjp_mode = _resolve_vjp_mode(vjp_mode)

    # Cross-prompt state, threaded through every per-prompt sweep: the CPU
    # fp32 accumulator, plus per-layer on-device row buffers reused across
    # prompts (allocated once d_model is known).
    state: dict[str, Any] = {
        "acc": None,
        "d_model": 0,
        "stripe_rows": None,
        "host_stripes": None,
        "vjp_mode": "batched" if requested_vjp_mode == "auto" else requested_vjp_mode,
        "requested_vjp_mode": requested_vjp_mode,
    }
    # Prepare once. Session orchestration normally supplies token IDs it already
    # hashed for resume; direct callers get the same one-time tokenization here.
    prepared_rows: list[tuple[int, torch.Tensor]] = []
    for prompt_idx, prompt in enumerate(prompts):
        if input_id_rows is None:
            ids_cpu = tokenizer(prompt, return_tensors="pt")["input_ids"][
                0, :max_seq_len
            ].to(dtype=torch.long, device="cpu")
        else:
            row = [int(tok) for tok in input_id_rows[prompt_idx]][:max_seq_len]
            ids_cpu = torch.tensor(row, dtype=torch.long)
        if int(ids_cpu.numel()) < skip_first + 2:
            log.warning(
                "jlens: skipping prompt %d — %d tokens, need > %d",
                prompt_idx, int(ids_cpu.numel()), skip_first + 1,
            )
            continue
        prepared_rows.append((prompt_idx, ids_cpu))

    n_done = 0
    fit_started = time.perf_counter()
    active_dim_batch = int(dim_batch)
    target_prompt_batch = int(
        1 if requested_vjp_mode in {"scalar", "replicated"}
        else (
            prompt_batch
            if prompt_batch is not None
            else (1 if device.type == "mps" else DEFAULT_PROMPT_BATCH)
        )
    )
    active_prompt_batch = target_prompt_batch
    prompt_bad_ceiling: int | None = None
    cursor = 0

    def _partial() -> JacobianLens:
        acc = state["acc"]
        assert acc is not None
        return JacobianLens(
            {l: a / max(n_done, 1) for l, a in acc.items()},
            n_prompts=n_done,
            d_model=state["d_model"],
        )

    captured, handles = _install_fit_hooks(layer_modules, sources, final_idx)
    try:
        with torch.enable_grad():
            while cursor < len(prepared_rows):
                # Never cross a checkpoint boundary: every persisted count is a
                # prefix address into the corpus, even with prompt microbatches.
                until_checkpoint = checkpoint_every - (n_done % checkpoint_every)
                width = min(
                    active_prompt_batch, until_checkpoint,
                    len(prepared_rows) - cursor,
                )
                chunk = prepared_rows[cursor : cursor + width]
                lengths_cpu = torch.tensor(
                    [int(row.numel()) for _idx, row in chunk], dtype=torch.long,
                )
                pad_id = getattr(tokenizer, "pad_token_id", None)
                if pad_id is None:
                    pad_id = getattr(tokenizer, "eos_token_id", None) or 0
                ids_cpu = torch.nn.utils.rnn.pad_sequence(
                    [row for _idx, row in chunk], batch_first=True,
                    padding_value=int(pad_id),
                )
                ids = ids_cpu.to(device)
                lengths = lengths_cpu.to(device)
                max_len = int(ids.shape[1])
                attention_mask = (
                    (
                        torch.arange(max_len, device=device).unsqueeze(0)
                        < lengths.unsqueeze(1)
                    ).to(torch.long)
                    if bool((lengths_cpu != max_len).any()) else None
                )
                batch = max(1, min(active_dim_batch, state["d_model"] or active_dim_batch))
                committed_row = 0
                restart_with_smaller_prompts = False
                while True:
                    retry_batch: int | None = None
                    try:
                        _accumulate_prompt_jacobian(
                            model, ids, layer_modules, sources, final_idx, state,
                            captured=captured, batch=batch, skip_first=skip_first,
                            attention_mask=attention_mask, lengths=lengths,
                            row_start=committed_row,
                        )
                        break
                    except _BatchedVjpUnavailable as exc:
                        assert requested_vjp_mode == "auto"
                        state["vjp_mode"] = "scalar"
                        target_prompt_batch = 1
                        active_prompt_batch = 1
                        log.warning(
                            "jlens: batched VJP is unavailable on this backend — "
                            "falling back to unreplicated scalar VJPs"
                        )
                        if exc.committed_until > committed_row:
                            # A backend normally reports missing vmap coverage
                            # on the first VJP. If it fails late, some rows of
                            # this prompt batch may already be in the corpus
                            # accumulator. Restart this suffix fit from zero in
                            # scalar mode rather than double-counting them.
                            for accumulator in state["acc"].values():
                                accumulator.zero_()
                            n_done = 0
                            cursor = 0
                            committed_row = 0
                            restart_with_smaller_prompts = True
                            log.warning(
                                "jlens: batched VJP failed after committed "
                                "rows; restarting the current fit shard in "
                                "scalar mode"
                            )
                            break
                        if width > 1:
                            restart_with_smaller_prompts = True
                            break
                        continue
                    except _PromptRowsCommitted as exc:
                        committed_row = exc.committed_until
                        if batch <= 1:
                            if width <= 1:
                                raise
                            # The smallest row stripe still cannot fit the
                            # multi-prompt graph.  Some rows are already folded
                            # into the shard accumulator, so restart this shard
                            # at a narrower prompt width rather than double
                            # count them or escape without trying B=1.
                            for accumulator in state["acc"].values():
                                accumulator.zero_()
                            n_done = 0
                            cursor = 0
                            committed_row = 0
                            prompt_bad_ceiling = (
                                width if prompt_bad_ceiling is None
                                else min(prompt_bad_ceiling, width)
                            )
                            active_prompt_batch = max(1, width // 2)
                            restart_with_smaller_prompts = True
                            _empty_device_cache(device)
                            log.warning(
                                "jlens: OOM after committed rows at "
                                "dim_batch=1 — restarting the fit shard with "
                                "prompt_batch=%d", active_prompt_batch,
                            )
                            break
                        batch = max(1, batch // 2)
                        active_dim_batch = batch
                        _empty_device_cache(device)
                        log.warning(
                            "jlens: OOM after row %d — resuming prompt batch "
                            "with dim_batch=%d",
                            committed_row, batch,
                        )
                        continue
                    except RuntimeError as exc:  # OOM → halve dim_batch and retry
                        if not is_out_of_memory_error(exc):
                            raise
                        if width > 1:
                            prompt_bad_ceiling = (
                                width if prompt_bad_ceiling is None
                                else min(prompt_bad_ceiling, width)
                            )
                            active_prompt_batch = max(1, width // 2)
                            restart_with_smaller_prompts = True
                            _empty_device_cache(device)
                            log.warning(
                                "jlens: OOM — retrying with prompt_batch=%d",
                                active_prompt_batch,
                            )
                            break
                        if batch <= 1:
                            raise
                        retry_batch = max(1, batch // 2)
                    assert retry_batch is not None
                    batch = retry_batch
                    active_dim_batch = batch
                    _empty_device_cache(device)
                    log.warning("jlens: OOM — retrying prompt with dim_batch=%d", batch)
                if restart_with_smaller_prompts:
                    continue
                n_done += width
                cursor += width
                if on_progress is not None:
                    mode = str(state["vjp_mode"])
                    elapsed = max(time.perf_counter() - fit_started, 1e-9)
                    on_progress(
                        f"prompt {progress_base + n_done}/"
                        f"{progress_base + len(prepared_rows)} "
                        f"(prompt_batch={width}, dim_batch="
                        f"{state.get('effective_dim_batch', batch)}, vjp={mode}, "
                        f"elapsed={elapsed:.1f}s, rate={n_done / elapsed:.3f}/s)"
                    )
                # Do not oscillate back into a width already proven bad during
                # this run. The dim-batch work is nearly width-invariant, so its
                # reduced value likewise stays put after an OOM.
                if prompt_bad_ceiling is None:
                    active_prompt_batch = target_prompt_batch
                if n_done % checkpoint_every == 0:
                    if checkpoint_accumulator_cb is not None:
                        checkpoint_accumulator_cb(
                            state["acc"], n_done, state["d_model"],
                        )
                    if checkpoint_cb is not None:
                        checkpoint_cb(_partial())
                    # Allocator hygiene at checkpoint cadence only — a per-prompt
                    # empty_cache forces a sync and dumps the pool the very next
                    # prompt re-allocates.
                    if device.type == "mps":
                        _empty_device_cache(device)
    finally:
        for handle in handles:
            handle.remove()

    if state["acc"] is None or n_done == 0:
        raise JacobianLensError(
            f"no usable prompts: every prompt had <= {skip_first + 1} tokens"
        )
    # Finalization owns the accumulator now; normalize it in place instead of
    # allocating one additional fp32 artifact through ``_partial()``.
    acc: dict[int, torch.Tensor] = state["acc"]
    inv_n = 1.0 / n_done
    for tensor in acc.values():
        tensor.mul_(inv_n)
    return JacobianLens(
        acc, n_prompts=n_done, d_model=state["d_model"],
    )


def _accumulate_prompt_jacobian(
    model: Any,
    ids: torch.Tensor,
    layer_modules: Sequence[nn.Module],
    sources: Sequence[int],
    final_idx: int,
    state: dict[str, Any],
    *,
    captured: dict[int, torch.Tensor],
    batch: int,
    skip_first: int,
    attention_mask: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    row_start: int = 0,
) -> None:
    """Run one prompt microbatch's exact VJP sweep into bounded row stripes.

    Fully transferred/validated stripes are added directly to the persistent
    CPU accumulator. If a later VJP OOMs, ``_PromptRowsCommitted`` carries the
    first unfinished output row; the caller rebuilds the graph and continues
    there, so neither the completed backward work nor a full lens-sized staging
    matrix is needed.
    """
    prompt_count, seq_len = ids.shape
    del layer_modules
    device = ids.device
    committed_until = int(row_start)
    if lengths is None:
        lengths = torch.full(
            (prompt_count,), seq_len, dtype=torch.long, device=device,
        )
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    valid_mask = (
        (positions >= skip_first)
        & (positions < (lengths.to(device=device).unsqueeze(1) - 1))
    )
    captured.clear()
    try:
        vjp_mode = str(state["vjp_mode"])
        if vjp_mode != "batched" and prompt_count != 1:
            raise ValueError(f"{vjp_mode} VJP requires prompt_batch=1")
        if vjp_mode == "scalar":
            batch = 1
        if vjp_mode == "replicated":
            batch = min(batch, _ROW_STRIPE)
        forward_batch = prompt_count if vjp_mode in {"batched", "scalar"} else batch
        replicated = (
            ids if vjp_mode in {"batched", "scalar"}
            else ids.expand(forward_batch, -1)
        )
        replicated_mask = (
            attention_mask
            if vjp_mode in {"batched", "scalar"} or attention_mask is None
            else attention_mask.expand(forward_batch, -1)
        )
        try:
            try:
                model(
                    input_ids=replicated, attention_mask=replicated_mask,
                    use_cache=False,
                )
            except _FitForwardComplete:
                pass
        except TypeError:  # toy/CPU-test models without a use_cache kwarg
            try:
                model(input_ids=replicated)
            except _FitForwardComplete:
                pass
        final = captured[final_idx]
        d_model = final.shape[-1]
        batch = min(batch, d_model)
        batch = min(batch, _ROW_STRIPE)
        state["effective_dim_batch"] = batch
        if not 0 <= row_start < d_model:
            if row_start == d_model:
                return
            raise ValueError(f"row_start must lie in [0, {d_model}], got {row_start}")
        if state["acc"] is None:
            state["acc"] = {
                l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in sources
            }
            state["d_model"] = d_model
        stripe_capacity = min(_ROW_STRIPE, d_model)
        if device.type != "cpu" and state["stripe_rows"] is None:
            state["stripe_rows"] = {
                l: torch.empty(
                    stripe_capacity, d_model, dtype=torch.float32, device=device,
                )
                for l in sources
            }
        if device.type != "cpu" and state["host_stripes"] is None:
            try:
                state["host_stripes"] = {
                    l: torch.empty(
                        stripe_capacity, d_model, dtype=torch.float32,
                        pin_memory=device.type == "cuda",
                    )
                    for l in sources
                }
            except RuntimeError:
                state["host_stripes"] = {
                    l: torch.empty(stripe_capacity, d_model, dtype=torch.float32)
                    for l in sources
                }
        stripe_rows: dict[int, torch.Tensor] | None = state["stripe_rows"]
        host_stripes: dict[int, torch.Tensor] | None = state["host_stripes"]
        stripe_start = int(row_start)
        source_tensors: list[torch.Tensor] = []
        for l in sources:
            if not captured[l].requires_grad:
                raise JacobianLensError(
                    f"layer {l} output carries no grad — the seed hook did not "
                    "reach the residual stream (unsupported block call shape?)"
                )
            source_tensors.append(captured[l])

        def flush_stripe(end: int) -> None:
            nonlocal stripe_start, committed_until
            if device.type == "cpu" or end <= stripe_start:
                return
            assert host_stripes is not None and stripe_rows is not None
            n = end - stripe_start
            for layer in sources:
                host_stripes[layer][:n].copy_(
                    stripe_rows[layer][:n], non_blocking=device.type == "cuda",
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            # Validate every layer before committing any: a failed asynchronous
            # transfer must not leave a partially counted stripe.
            for layer in sources:
                rows = host_stripes[layer][:n]
                if bool((rows.abs().sum(dim=1) == 0).any()):
                    raise JacobianLensError(
                        f"layer {layer} came back with zero rows from the device — "
                        "likely an asynchronous out of memory on the command "
                        "queue; retrying at a smaller dim_batch"
                    )
            for layer in sources:
                state["acc"][layer][stripe_start:end].add_(
                    host_stripes[layer][:n]
                )
            committed_until = end
            stripe_start = end

        cot = None
        prev_dims: torch.Tensor | None = None
        batch_rows = torch.arange(batch, device=device)
        batched_outputs = (
            (final * valid_mask.unsqueeze(-1)).sum(dim=(0, 1))
            if vjp_mode in {"batched", "scalar"} else None
        )
        batched_eye = (
            torch.eye(batch, dtype=final.dtype, device=device)
            if vjp_mode == "batched" else None
        )
        dim_start = int(row_start)
        pass_idx = 0
        while dim_start < d_model:
            n_dims = min(batch, d_model - dim_start)
            dims = dim_start + batch_rows[:n_dims]
            # grad(final, sources) rather than backward(): the grads return
            # directly (no hooks), and the walk stops at the shallowest
            # requested layer instead of descending to the seed leaf.
            try:
                grads = _grad_row_block(
                    final, source_tensors, valid_mask, dims,
                    mode=vjp_mode,
                    retain_graph=dim_start + n_dims < d_model,
                    cotangent=cot,
                    prev_dims=prev_dims,
                    batched_outputs=batched_outputs,
                    batched_eye=batched_eye,
                )
            except RuntimeError as exc:
                if (
                    state["requested_vjp_mode"] == "auto"
                    and vjp_mode == "batched"
                    and _looks_like_batched_vjp_unsupported(exc)
                ):
                    raise _BatchedVjpUnavailable(committed_until) from exc
                raise
            if vjp_mode == "replicated":
                if cot is None:
                    cot = torch.zeros_like(final)
                prev_dims = dims
            write_end = dim_start + n_dims
            stripe_offset = 0
            if device.type != "cpu":
                assert stripe_rows is not None
                if write_end - stripe_start > stripe_capacity:
                    flush_stripe(dim_start)
                stripe_offset = dim_start - stripe_start
            for l, g in zip(sources, grads):
                block = _source_grad_block(
                    g, mode=vjp_mode, n_dims=n_dims,
                    valid_mask=valid_mask,
                )
                if device.type != "cpu":
                    assert stripe_rows is not None
                    stripe_rows[l][stripe_offset : stripe_offset + n_dims] = block
                else:
                    state["acc"][l][dim_start:write_end].add_(block)
            if device.type == "cpu":
                committed_until = write_end
            pass_idx += 1
            if device.type == "mps" and pass_idx % _MPS_SYNC_EVERY_PASSES == 0:
                torch.mps.synchronize()
            dim_start = write_end
        flush_stripe(d_model)
    except RuntimeError as exc:
        if is_out_of_memory_error(exc) and committed_until > row_start:
            raise _PromptRowsCommitted(committed_until, str(exc)) from exc
        raise
    finally:
        captured.clear()


def _grad_row_block(
    final: torch.Tensor,
    source_tensors: Sequence[torch.Tensor],
    valid_mask: torch.Tensor,
    dims: torch.Tensor,
    *,
    mode: str,
    retain_graph: bool,
    cotangent: torch.Tensor | None,
    prev_dims: torch.Tensor | None,
    batched_outputs: torch.Tensor | None,
    batched_eye: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    if mode == "batched":
        assert batched_outputs is not None and batched_eye is not None
        outputs = batched_outputs[dims]
        eye = batched_eye[:dims.numel(), :dims.numel()]
        return torch.autograd.grad(
            outputs, source_tensors, grad_outputs=eye,
            retain_graph=retain_graph, is_grads_batched=True,
        )
    if mode == "scalar":
        assert batched_outputs is not None and dims.numel() == 1
        return torch.autograd.grad(
            batched_outputs[dims[0]], source_tensors,
            retain_graph=retain_graph,
        )

    if cotangent is None:
        cotangent = torch.zeros_like(final)
    valid = valid_mask[0].nonzero(as_tuple=False).reshape(-1)
    rows = torch.arange(dims.numel(), device=final.device).unsqueeze(1)
    if prev_dims is not None:
        prev_rows = torch.arange(prev_dims.numel(), device=final.device).unsqueeze(1)
        cotangent[prev_rows, valid.unsqueeze(0), prev_dims.unsqueeze(1)] = 0.0
    cotangent[rows, valid.unsqueeze(0), dims.unsqueeze(1)] = 1.0
    return torch.autograd.grad(
        final, source_tensors, grad_outputs=cotangent,
        retain_graph=retain_graph,
    )


def _source_grad_block(
    grad: torch.Tensor,
    *,
    mode: str,
    n_dims: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if mode == "batched":
        # [n_dims, B, T, D] from batched VJP. Preserve the estimator's
        # equal-prompt weighting: mean source positions *within* each prompt,
        # then sum prompt Jacobians into the raw corpus accumulator.
        mask = valid_mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        counts = valid_mask.sum(dim=1).clamp_min(1).to(torch.float32)
        per_prompt = (
            (grad[:n_dims].to(torch.float32) * mask).sum(dim=2)
            / counts.reshape(1, -1, 1)
        )
        return per_prompt.sum(dim=1)
    # [replicated_batch, T, D] from the reference replicated-prompt path.
    valid = valid_mask[0].nonzero(as_tuple=False).reshape(-1)
    return grad[:n_dims, valid].mean(dim=1, dtype=torch.float32)


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
