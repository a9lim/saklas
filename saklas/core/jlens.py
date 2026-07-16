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
output hook on the first fitted block) — the ``inference_mode`` capture machinery in
``vectors.py`` cannot be reused, because inference tensors never re-enter
autograd. Per-layer grads come from ``torch.autograd.grad(final, sources)``
— NOT ``backward()`` + ``retain_grad()`` (``.grad`` accumulates across the
multi-backward loop and would corrupt the one-hot-cotangent rows) — which
also stops the graph walk at the shallowest requested source layer, so a
band-restricted fit never backprops below its lowest source. A terminal-layer
hook captures the target residual and aborts the rest of the forward before
the final norm and full-vocabulary head. Row blocks are staged in bounded
per-layer stripes, validated after transfer, and committed directly into the
CPU fp32 cross-prompt accumulator; CUDA double-buffers those stripes so D2H for
one can overlap the next backward block. If a later VJP OOMs, the graph is
rebuilt at the first uncommitted row, and prompt-width backoff splits only that
group while preserving its exact committed prefix and every prior microbatch.
The MPS sync
budget is bounded from both sides: a fully unsynced loop can exhaust Metal's
asynchronous command queue, so periodic drains plus the zero-row guard remain.
The fit is compute-bound (each prompt ≈ ``d_model × 2`` forward-equivalents
of backward, dim_batch-invariant); restricting ``source_layers`` is the one
lever that removes work. The fp32 accumulator is persisted losslessly so saved,
resumed, and loaded lenses retain the estimator's precision.
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
# remain equal-prompt weighted (not equal-token weighted).  MPS defaults to two
# after the M5 Max / gemma-3-4b sweep measured 1.72x over one with unchanged
# peak RSS. OOM backoff reduces this independently of ``dim_batch``.
DEFAULT_PROMPT_BATCH = 4
DEFAULT_MPS_PROMPT_BATCH = 2
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
#: Maximum fp32 bytes held by one device/host stripe slot.  A row spans every
#: fitted source layer, so a fixed 256-row allocation grows into GiBs on large
#: hidden sizes / all-layer fits and can OOM independently of ``dim_batch``.
#: This is a per-slot budget (CUDA may use two slots for overlap); the active
#: VJP block remains the hard minimum so one result block always fits.
_ROW_STRIPE_BYTES_PER_SLOT = 128 * 1024**2


def _row_stripe_capacity(
    d_model: int,
    n_sources: int,
    dim_batch: int,
    *,
    byte_budget: int = _ROW_STRIPE_BYTES_PER_SLOT,
) -> int:
    """Largest bounded stripe that holds at least one complete VJP block."""
    if d_model <= 0 or n_sources <= 0 or dim_batch <= 0 or byte_budget <= 0:
        raise ValueError("stripe dimensions and byte_budget must be positive")
    row_bytes = int(d_model) * int(n_sources) * 4  # fp32 staging
    budget_rows = max(1, int(byte_budget) // row_bytes)
    return min(int(d_model), _ROW_STRIPE, max(int(dim_batch), budget_rows))


def _smaller_row_stripe_capacity(capacity: int, dim_batch: int) -> int | None:
    """Next allocation-backoff capacity, or ``None`` at the VJP-block floor."""
    floor = max(1, int(dim_batch))
    if int(capacity) <= floor:
        return None
    return max(floor, int(capacity) // 2)


class JacobianLensError(RuntimeError, SaklasError):
    """Raised when a Jacobian-lens fit or readout cannot proceed."""

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


class LensNotFittedError(JacobianLensError):
    """Raised when a lens artifact is required but absent for the model."""

    def user_message(self) -> tuple[int, str]:
        return (404, str(self) or self.__class__.__name__)


class JacobianLensCancelled(JacobianLensError):
    """Raised after a cooperative stop at a safe estimator boundary."""

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or "Jacobian-lens fit cancelled")


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
        will materialize their own fp32 shard through ``save_lens``.
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


def readout_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Calibrate per-layer lens logits into the shared probability unit.

    Kept as one explicit primitive so a live decode step can normalize its
    full ``[layers, vocab]`` matrix once, then share the result between pinned
    probes, per-layer cards, and the aggregate readout.
    """
    return logits.float().softmax(dim=-1)


def aggregate_readout_from_probabilities(
    probabilities: torch.Tensor,
    depths: "Sequence[float]",
    *,
    top_k: int = 8,
    depth_tensor: torch.Tensor | None = None,
) -> list[tuple[int, float, float, float]]:
    """Aggregate already-calibrated ``[layers, vocab]`` probabilities.

    Strength must be computed over the whole vocabulary before selection, but
    depth CoM/spread are needed only for the selected tokens.  Gathering those
    columns first preserves the exact statistic while avoiding full-vocabulary
    depth tensors.
    """
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError(
            "aggregate_readout_from_probabilities expects [layers, vocab] "
            f"probabilities, got shape {tuple(probabilities.shape)}"
        )
    if len(depths) != probabilities.shape[0]:
        raise ValueError(
            "aggregate_readout_from_probabilities: "
            f"{probabilities.shape[0]} probability rows but {len(depths)} depths"
        )
    idxs, stats = aggregate_readout_tensors_from_probabilities(
        probabilities,
        depths,
        top_k=top_k,
        depth_tensor=depth_tensor,
    )
    # Public list surface: one packed host transfer. Keep the device-side
    # payload in fp32: MPS does not support float64, while fp32 still represents
    # every practical vocabulary id exactly (up to 2**24) and avoids a second
    # accelerator synchronization.
    host = torch.cat(
        [stats.float(), idxs.reshape(1, -1).to(torch.float32)],
        dim=0,
    ).cpu()
    return [
        (
            int(host[3, j]),
            float(host[0, j]),
            float(host[1, j]),
            float(host[2, j]),
        )
        for j in range(int(idxs.numel()))
    ]


def aggregate_readout_tensors_from_probabilities(
    probabilities: torch.Tensor,
    depths: "Sequence[float]",
    *,
    top_k: int = 8,
    depth_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-resident aggregate selection and statistics.

    Returns ``(token_ids[K], stats[3,K])`` where the statistic rows are
    strength, depth center-of-mass, and depth spread. Keeping this tensor form
    separate from the public list conversion lets the live readout combine its
    per-layer and aggregate payloads into one GPU-to-host synchronization per
    token.
    """
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError(
            "aggregate_readout_tensors_from_probabilities expects "
            f"[layers, vocab] probabilities, got shape {tuple(probabilities.shape)}"
        )
    if len(depths) != probabilities.shape[0]:
        raise ValueError(
            "aggregate_readout_tensors_from_probabilities: "
            f"{probabilities.shape[0]} probability rows but {len(depths)} depths"
        )
    strength = probabilities.mean(dim=0)                       # [V]
    k = min(max(int(top_k), 0), int(strength.shape[-1]))
    vals, idxs = strength.topk(k)
    p = probabilities.index_select(-1, idxs)                   # [L, K]
    d = _depth_column(
        depths,
        device=probabilities.device,
        rows=int(probabilities.shape[0]),
        depth_tensor=depth_tensor,
    )                                                           # [L, 1]
    mass = p.sum(dim=0).clamp_min(1e-12)                        # [K]
    com = (p * d).sum(dim=0) / mass                             # [K]
    var = (p * (d - com.unsqueeze(0)) ** 2).sum(dim=0) / mass
    spread = var.clamp_min(0.0).sqrt()
    return idxs, torch.stack([vals, com, spread])


def aggregate_readout(
    logits: torch.Tensor,
    depths: "Sequence[float]",
    *,
    top_k: int = 8,
    depth_tensor: torch.Tensor | None = None,
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
      the same per-layer probability ``p_l(v)``. The selected-layer
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
    return aggregate_readout_from_probabilities(
        readout_probabilities(logits),
        depths,
        top_k=top_k,
        depth_tensor=depth_tensor,
    )


def token_readout_stats_from_probabilities(
    probabilities: torch.Tensor,
    depths: "Sequence[float]",
    token_ids: "Sequence[int]",
    *,
    token_ids_tensor: torch.Tensor | None = None,
    depth_tensor: torch.Tensor | None = None,
) -> list[tuple[float, float, float, list[float]]]:
    """Per-token statistics from already-calibrated readout probabilities."""
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError(
            "token_readout_stats_from_probabilities expects [layers, vocab] "
            f"probabilities, got shape {tuple(probabilities.shape)}"
        )
    if len(depths) != probabilities.shape[0]:
        raise ValueError(
            "token_readout_stats_from_probabilities: "
            f"{probabilities.shape[0]} probability rows but {len(depths)} depths"
        )
    if not token_ids:
        return []
    ids = _token_id_tensor(
        token_ids,
        device=probabilities.device,
        token_ids_tensor=token_ids_tensor,
    )
    p = probabilities.index_select(-1, ids)                    # [L, K]
    return _token_readout_stats_from_probability_columns(
        p, depths, depth_tensor=depth_tensor,
    )


def _token_probability_columns_from_logits(
    logits: torch.Tensor,
    token_ids: "Sequence[int]",
    *,
    token_ids_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact softmax probabilities for selected token ids only.

    The denominator remains the full-vocabulary logsumexp for each layer; only
    the returned probability columns are narrowed. This preserves gate/display
    thresholds while avoiding a full ``[layers, vocab]`` probability matrix for
    fixed-token probes.
    """
    ids = _token_id_tensor(
        token_ids,
        device=logits.device,
        token_ids_tensor=token_ids_tensor,
    )
    logits_f = logits.float()
    selected = logits_f.index_select(-1, ids)                  # [L, K]
    log_z = logits_f.logsumexp(dim=-1, keepdim=True)           # [L, 1]
    return (selected - log_z).exp()


def _depth_column(
    depths: "Sequence[float]",
    *,
    device: torch.device,
    rows: int,
    depth_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return depths as a device ``[L, 1]`` tensor, reusing a caller cache."""
    if depth_tensor is None:
        d = torch.tensor(
            [float(x) for x in depths],
            dtype=torch.float32,
            device=device,
        )
    else:
        d = depth_tensor.to(device=device, dtype=torch.float32)
        if d.ndim > 2:
            raise ValueError(
                f"depth_tensor must be [layers] or [layers, 1], got {tuple(d.shape)}"
            )
        d = d.reshape(-1)
    if int(d.numel()) != rows:
        raise ValueError(
            f"depth tensor has {int(d.numel())} rows, expected {rows}"
        )
    return d.reshape(rows, 1)


def _token_id_tensor(
    token_ids: "Sequence[int]",
    *,
    device: torch.device,
    token_ids_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return token ids as a device long tensor, reusing a caller cache."""
    if token_ids_tensor is None:
        return torch.tensor(
            [int(v) for v in token_ids],
            dtype=torch.long,
            device=device,
        )
    ids = token_ids_tensor.to(device=device, dtype=torch.long).reshape(-1)
    if int(ids.numel()) != len(token_ids):
        raise ValueError(
            f"token id tensor has {int(ids.numel())} ids, expected {len(token_ids)}"
        )
    return ids


def _token_readout_stats_from_probability_columns(
    p: torch.Tensor,
    depths: "Sequence[float]",
    *,
    depth_tensor: torch.Tensor | None = None,
) -> list[tuple[float, float, float, list[float]]]:
    """Shared stats body for selected token probability columns ``[L, K]``."""
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError(
            "token readout stats expect [layers, tokens] probabilities, "
            f"got shape {tuple(p.shape)}"
        )
    if len(depths) != p.shape[0]:
        raise ValueError(
            f"token readout stats: {p.shape[0]} probability rows but "
            f"{len(depths)} depths"
        )
    if p.shape[1] == 0:
        return []
    strength = p.mean(dim=0)                                    # [K]
    d = _depth_column(
        depths,
        device=p.device,
        rows=int(p.shape[0]),
        depth_tensor=depth_tensor,
    )                                                           # [L, 1]
    mass = p.sum(dim=0).clamp_min(1e-12)                        # [K]
    com = (p * d).sum(dim=0) / mass                             # [K]
    var = (p * (d - com.unsqueeze(0)) ** 2).sum(dim=0) / mass
    spread = var.clamp_min(0.0).sqrt()
    # one batched host transfer: 3 aggregate rows + the per-layer block
    host = torch.cat(
        [torch.stack([strength, com, spread]), p], dim=0,
    ).cpu()                                                     # [3+L, K]
    n_layers = int(p.shape[0])
    out: list[tuple[float, float, float, list[float]]] = []
    for j in range(int(p.shape[1])):
        per_layer = [float(host[3 + l, j]) for l in range(n_layers)]
        out.append(
            (
                float(host[0, j]), float(host[1, j]), float(host[2, j]),
                per_layer,
            )
        )
    return out


def token_readout_stats(
    logits: torch.Tensor,
    depths: "Sequence[float]",
    token_ids: "Sequence[int]",
    *,
    token_ids_tensor: torch.Tensor | None = None,
    depth_tensor: torch.Tensor | None = None,
) -> list[tuple[float, float, float, list[float]]]:
    """Per-token readout statistics for pinned vocabulary ids.

    The single-token restriction of :func:`aggregate_readout`: the same
    per-layer softmax calibration, read at the requested ``token_ids``
    instead of selected top-k.  For each id:

    - ``strength = mean_l p_l(v)`` — mean fitted-layer probability, the aggregate
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
    return _token_readout_stats_from_probability_columns(
        _token_probability_columns_from_logits(
            logits, token_ids, token_ids_tensor=token_ids_tensor,
        ),
        depths,
        depth_tensor=depth_tensor,
    )


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


class _MeanSourceProbe(torch.autograd.Function):
    """Transparent residual identity with a mean-position probe derivative.

    The Jacobian estimator needs only the mean source-position derivative, not
    the full ``[T,D]`` gradient at every fitted layer.  A zero-valued ``[B,D]``
    shared perturbation over the valid source positions has exactly that
    derivative.  This custom identity leaves the forward byte-for-byte
    unchanged while collapsing each source gradient inside autograd, before it
    can become a retained ``[rows,B,T,D]`` result.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(
        residual: torch.Tensor,
        probe: torch.Tensor,
        counts: torch.Tensor,
        source_start: int,
    ) -> torch.Tensor:
        del probe, counts, source_start
        return residual

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        _residual, _probe, counts, source_start = inputs
        ctx.save_for_backward(counts)
        ctx.source_start = int(source_start)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        if len(grad_outputs) != 1:
            raise JacobianLensError("mean-source probe expected one gradient output")
        grad_output = grad_outputs[0]
        (counts,) = ctx.saved_tensors
        # Target cotangents exist only at valid content positions. In a causal
        # LM, excluded final/padded source positions cannot influence an earlier
        # selected target, so their gradients are exactly zero; slicing the
        # deliberately excluded prefix is therefore the exact valid-source sum.
        pooled = grad_output[..., ctx.source_start :, :].sum(
            dim=-2, dtype=torch.float32,
        )
        shape = [1] * (pooled.ndim - 2) + [int(counts.numel()), 1]
        probe_grad = pooled / counts.reshape(shape)
        return grad_output, probe_grad, None, None


def _install_fit_hooks(
    layer_modules: Sequence[nn.Module],
    sources: Sequence[int],
    final_idx: int,
    hook_state: dict[str, Any],
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    first_source = min(sources)

    def make_capture(idx: int) -> Callable[..., Any]:
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
            residual = _output_tensor(output)
            if idx in sources:
                # Seed at the source block's OUTPUT. Autograd gradients target
                # this residual, so retaining the block's QKV/MLP internals
                # cannot affect J but costs a full block of graph memory.
                if idx == first_source:
                    residual = residual.detach().clone()
                counts = hook_state.get("source_counts")
                source_start = hook_state.get("source_start")
                if not isinstance(counts, torch.Tensor) or source_start is None:
                    raise JacobianLensError("J-lens source probe state was not prepared")
                probe = torch.zeros(
                    residual.shape[0], residual.shape[-1],
                    device=residual.device, dtype=torch.float32,
                    requires_grad=True,
                )
                residual = _MeanSourceProbe.apply(
                    residual, probe, counts, int(source_start),
                )
                captured[idx] = probe
                if isinstance(output, tuple):
                    return (residual, *output[1:])
                return residual
            captured[idx] = residual
            if idx == final_idx:
                raise _FitForwardComplete()
            return None

        return hook

    # Everything through the LOWEST source block runs graph-free. The returned
    # leaf then seeds later blocks, so restricted fits retain no graph below
    # the fitted band and no internals for its first block.
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
    checkpoint_accumulator_cb: (
        Callable[[Mapping[int, torch.Tensor], int, int], None] | None
    ) = None,
    on_progress: Callable[[str], None] | None = None,
    progress_base: int = 0,
    input_id_rows: Sequence[Sequence[int]] | None = None,
    vjp_mode: str = "auto",
    cancel_event: Any | None = None,
    suppress_terminal_checkpoint: bool = False,
    initial_lens: JacobianLens | None = None,
) -> JacobianLens:
    """Fit ``J_l`` for every source layer over ``prompts``.

    One graph per prompt microbatch + ``ceil(d_model/dim_batch)`` backwards
    (see the module docstring for the estimator). ``initial_lens`` transfers
    ownership of an existing averaged prefix
    to the estimator: its fp32 matrices are converted to weighted sums in
    place and become the accumulator, so resume never allocates a second full
    matrix set.
    ``checkpoint_accumulator_cb`` is the allocation-light persistence seam: it
    receives the raw fp32 sums, completed prompt count, and hidden size without
    materializing a second full averaged lens. ``input_id_rows`` is the
    pre-tokenized sibling of ``prompts``. When supplied
    it must be aligned 1:1 with ``prompts`` and already reflect the caller's
    truncation/hash policy; the fit still applies ``max_seq_len`` defensively.
    This lets session-level resume/filtering reuse the IDs it already computed
    instead of tokenizing the corpus a second time.  ``vjp_mode="batched"``
    computes each output-dim block with ``is_grads_batched=True`` from a single
    prompt forward instead of replicating the forward graph; ``"auto"`` tries
    that path first and falls back to exact scalar VJPs if the backend lacks
    vmap coverage. ``prompt_batch`` controls consecutive ragged prompts per
    graph (CPU/CUDA default 4, MPS default 2); both prompt and output-dimension
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
    initial_n_prompts = 0
    initial_acc: dict[int, torch.Tensor] | None = None
    if initial_lens is not None:
        if initial_lens.source_layers != sources:
            raise ValueError(
                "initial_lens source layers must exactly match the fit request"
            )
        if initial_lens.n_prompts <= 0:
            raise ValueError("initial_lens must contain at least one prompt")
        initial_n_prompts = int(initial_lens.n_prompts)
        initial_acc = initial_lens.jacobians
    if input_id_rows is not None and len(input_id_rows) != len(prompts):
        raise ValueError(
            "input_id_rows must be aligned with prompts "
            f"({len(input_id_rows)} != {len(prompts)})"
        )
    requested_vjp_mode = _resolve_vjp_mode(vjp_mode)

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

    if not prepared_rows:
        raise JacobianLensError(
            f"no usable prompts: every prompt had <= {skip_first + 1} tokens"
        )
    if cancel_event is not None and cancel_event.is_set():
        raise JacobianLensCancelled("Jacobian-lens fit cancelled before start")
    if initial_acc is not None:
        for tensor in initial_acc.values():
            tensor.mul_(initial_n_prompts)
        assert initial_lens is not None
        initial_lens._atom_norm_cache.clear()

    # Cross-prompt state, threaded through every per-prompt sweep: the CPU
    # fp32 accumulator, plus per-layer on-device row buffers reused across
    # prompts (allocated once d_model is known).
    state: dict[str, Any] = {
        "acc": initial_acc,
        "d_model": initial_lens.d_model if initial_lens is not None else 0,
        "stripe_rows": None,
        "host_stripes": None,
        "cuda_transfer_stream": None,
        "vjp_mode": "batched" if requested_vjp_mode == "auto" else requested_vjp_mode,
        "requested_vjp_mode": requested_vjp_mode,
    }

    n_done = 0
    fit_started = time.perf_counter()
    active_dim_batch = int(dim_batch)
    target_prompt_batch = int(
        1 if requested_vjp_mode in {"scalar", "replicated"}
        else (
            prompt_batch
            if prompt_batch is not None
            else (
                DEFAULT_MPS_PROMPT_BATCH
                if device.type == "mps"
                else DEFAULT_PROMPT_BATCH
            )
        )
    )
    active_prompt_batch = target_prompt_batch
    prompt_bad_ceiling: int | None = None
    cursor = 0
    next_checkpoint = checkpoint_every
    # A task says: contributions for prompts [start:end] below ``row_start``
    # are already in the accumulator; fit only the remaining output rows.  A
    # prompt-width OOM can therefore split the task without erasing earlier
    # microbatches or double-counting its committed row prefix.  ``n_done`` is
    # advanced only after every task in the original group completes, keeping
    # progress/checkpoints aligned to internally consistent prompt boundaries.
    pending_tasks: list[tuple[int, int, int]] = []
    pending_group_width = 0

    hook_state: dict[str, Any] = {}
    captured, handles = _install_fit_hooks(
        layer_modules, sources, final_idx, hook_state,
    )
    state["hook_state"] = hook_state
    try:
        with torch.enable_grad():
            while cursor < len(prepared_rows):
                if not pending_tasks:
                    # Cancellation is observed only between complete prompt
                    # groups.  During a split recovery the accumulator carries
                    # committed rows for the whole group but ``n_done`` quite
                    # deliberately does not count it yet.
                    if cancel_event is not None and cancel_event.is_set():
                        if n_done > 0:
                            if checkpoint_accumulator_cb is not None:
                                checkpoint_accumulator_cb(
                                    state["acc"],
                                    initial_n_prompts + n_done,
                                    state["d_model"],
                                )
                        raise JacobianLensCancelled(
                            f"Jacobian-lens fit cancelled after {n_done} prompts"
                        )
                    pending_group_width = min(
                        active_prompt_batch, len(prepared_rows) - cursor,
                    )
                    pending_tasks.append(
                        (cursor, cursor + pending_group_width, 0),
                    )

                task_start, task_end, task_row_start = pending_tasks.pop(0)
                width = task_end - task_start
                chunk = prepared_rows[task_start:task_end]
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
                committed_row = task_row_start
                restart_with_smaller_prompts = False

                def split_current_task(*, row_start: int, max_width: int) -> None:
                    """Prepend exact narrower tasks for this prompt group.

                    Rows below ``row_start`` already contain the aggregate for
                    every prompt in the current task, so each child starts at
                    that same boundary.  Their suffix contributions add to the
                    same result as the original ragged microbatch.
                    """
                    child_width = max(1, min(int(max_width), width - 1))
                    children = [
                        (start, min(start + child_width, task_end), row_start)
                        for start in range(task_start, task_end, child_width)
                    ]
                    pending_tasks[0:0] = children

                while True:
                    retry_batch: int | None = None
                    try:
                        _accumulate_prompt_jacobian(
                            model, ids, layer_modules, sources, final_idx, state,
                            captured=captured, batch=batch, skip_first=skip_first,
                            attention_mask=attention_mask, lengths=lengths,
                            row_start=committed_row,
                            cancel_event=cancel_event,
                        )
                        # Close the tiny race after the final VJP block: a
                        # cancellation that lands after that block's internal
                        # check must still prevent this group (especially the
                        # terminal group) from being counted and published.
                        if cancel_event is not None and cancel_event.is_set():
                            raise JacobianLensCancelled(
                                "Jacobian-lens fit cancelled after the active "
                                "prompt group"
                            )
                        break
                    except _BatchedVjpUnavailable as exc:
                        assert requested_vjp_mode == "auto"
                        state["vjp_mode"] = "replicated"
                        target_prompt_batch = 1
                        active_prompt_batch = 1
                        durable = max(committed_row, exc.committed_until)
                        log.warning(
                            "jlens: batched VJP is unavailable on this backend — "
                            "falling back to replicated output-row VJPs"
                        )
                        if width > 1:
                            split_current_task(row_start=durable, max_width=1)
                            restart_with_smaller_prompts = True
                            if durable > task_row_start:
                                log.warning(
                                    "jlens: batched VJP failed after committed "
                                    "rows; preserving the row prefix while "
                                    "splitting the prompt group for replicated "
                                    "mode"
                                )
                            break
                        committed_row = durable
                        continue
                    except _PromptRowsCommitted as exc:
                        committed_row = exc.committed_until
                        if _downgrade_cuda_stripe_buffers(state, device):
                            _empty_device_cache(device)
                            log.warning(
                                "jlens: CUDA OOM — dropping the second transfer "
                                "slot and retrying from row %d", committed_row,
                            )
                            continue
                        if batch <= 1:
                            if width <= 1:
                                raise
                            # The committed prefix is the exact aggregate for
                            # this whole task. Split only its unfinished suffix;
                            # prior prompt groups remain durable in memory.
                            prompt_bad_ceiling = (
                                width if prompt_bad_ceiling is None
                                else min(prompt_bad_ceiling, width)
                            )
                            active_prompt_batch = max(1, width // 2)
                            split_current_task(
                                row_start=committed_row,
                                max_width=active_prompt_batch,
                            )
                            restart_with_smaller_prompts = True
                            _empty_device_cache(device)
                            log.warning(
                                "jlens: OOM after committed rows at "
                                "dim_batch=1 — preserving them and splitting "
                                "the prompt group with "
                                "prompt_batch=%d", active_prompt_batch,
                            )
                            break
                        batch = max(1, batch // 2)
                        active_dim_batch = batch
                        if _shrink_device_stripe_buffers(state, device, batch):
                            log.warning(
                                "jlens: releasing the one-slot staging tier at "
                                "the reduced dim_batch"
                            )
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
                        if _downgrade_cuda_stripe_buffers(state, device):
                            _empty_device_cache(device)
                            log.warning(
                                "jlens: CUDA OOM — dropping the second transfer "
                                "slot before narrowing estimator batches"
                            )
                            continue
                        if width > 1:
                            prompt_bad_ceiling = (
                                width if prompt_bad_ceiling is None
                                else min(prompt_bad_ceiling, width)
                            )
                            active_prompt_batch = max(1, width // 2)
                            split_current_task(
                                row_start=committed_row,
                                max_width=active_prompt_batch,
                            )
                            restart_with_smaller_prompts = True
                            _empty_device_cache(device)
                            log.warning(
                                "jlens: OOM — preserving committed rows and "
                                "retrying with prompt_batch=%d",
                                active_prompt_batch,
                            )
                            break
                        if batch <= 1:
                            raise
                        retry_batch = max(1, batch // 2)
                    assert retry_batch is not None
                    batch = retry_batch
                    active_dim_batch = batch
                    if _shrink_device_stripe_buffers(state, device, batch):
                        log.warning(
                            "jlens: releasing the one-slot staging tier at the "
                            "reduced dim_batch"
                        )
                    _empty_device_cache(device)
                    log.warning("jlens: OOM — retrying prompt with dim_batch=%d", batch)
                if restart_with_smaller_prompts:
                    continue
                if pending_tasks:
                    continue
                completed_width = pending_group_width
                n_done += completed_width
                cursor += completed_width
                if on_progress is not None:
                    mode = str(state["vjp_mode"])
                    elapsed = max(time.perf_counter() - fit_started, 1e-9)
                    on_progress(
                        f"prompt {progress_base + n_done}/"
                        f"{progress_base + len(prepared_rows)} "
                        f"(prompt_batch={completed_width}, dim_batch="
                        f"{state.get('effective_dim_batch', batch)}, vjp={mode}, "
                        f"elapsed={elapsed:.1f}s, rate={n_done / elapsed:.3f}/s)"
                    )
                # Do not oscillate back into a width already proven bad during
                # this run. The dim-batch work is nearly width-invariant, so its
                # reduced value likewise stays put after an OOM.
                if prompt_bad_ceiling is None:
                    active_prompt_batch = target_prompt_batch
                # Persist after the first complete microbatch that crosses the
                # cadence.  Do not fracture an otherwise healthy prompt batch
                # merely to land on an exact count, and do not write a complete
                # terminal checkpoint immediately before the durable artifact.
                if (
                    n_done >= next_checkpoint
                    and (
                        cursor < len(prepared_rows)
                        or not suppress_terminal_checkpoint
                    )
                ):
                    if checkpoint_accumulator_cb is not None:
                        checkpoint_accumulator_cb(
                            state["acc"],
                            initial_n_prompts + n_done,
                            state["d_model"],
                        )
                    # Allocator hygiene at checkpoint cadence only — a per-prompt
                    # empty_cache forces a sync and dumps the pool the very next
                    # prompt re-allocates.
                    if device.type == "mps":
                        _empty_device_cache(device)
                    next_checkpoint = (
                        (n_done // checkpoint_every) + 1
                    ) * checkpoint_every
    finally:
        for handle in handles:
            handle.remove()
        # The hooks deliberately retain the latest source/final activations.
        # On accelerator fits those tensors own the final autograd graph, and
        # the reusable stripe buffers own additional device allocations.  A
        # long-lived server otherwise carries the whole fit working set into
        # the next generation; on MPS that can terminate the process instead
        # of raising a recoverable OOM.  Break every reference before flushing
        # the allocator.  The CPU accumulator is intentionally preserved.
        captured.clear()
        hook_state.clear()
        state["hook_state"] = None
        state["stripe_rows"] = None
        state["host_stripes"] = None
        state["cuda_transfer_stream"] = None
        if device.type in {"mps", "cuda"}:
            # Autograd can leave Python cycles around custom/vmap nodes; make
            # those unreachable allocations visible to empty_cache now, at the
            # artifact boundary, rather than during the user's next decode.
            import gc

            gc.collect()
            _empty_device_cache(device)

    if state["acc"] is None or n_done == 0:
        raise JacobianLensError(
            f"no usable prompts: every prompt had <= {skip_first + 1} tokens"
        )
    # Finalization owns the accumulator now; normalize it in place.
    acc: dict[int, torch.Tensor] = state["acc"]
    total_done = initial_n_prompts + n_done
    inv_n = 1.0 / total_done
    for tensor in acc.values():
        tensor.mul_(inv_n)
    return JacobianLens(
        acc, n_prompts=total_done, d_model=state["d_model"],
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
    cancel_event: Any | None = None,
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
    cuda_pending: list[tuple[int, int, torch.cuda.Event] | None] = []
    drain_cuda_slot: Callable[[int], None] = lambda _slot: None
    drain_cuda_pending_in_order: Callable[[], None] = lambda: None

    if lengths is None:
        lengths = torch.full(
            (prompt_count,), seq_len, dtype=torch.long, device=device,
        )
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    valid_mask = (
        (positions >= skip_first)
        & (positions < (lengths.to(device=device).unsqueeze(1) - 1))
    )
    source_counts = valid_mask.sum(dim=1).clamp_min(1).to(torch.float32)
    captured.clear()
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise JacobianLensCancelled(
                "Jacobian-lens fit cancelled before the next prompt group"
            )
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
        hook_state: dict[str, Any] = state["hook_state"]
        hook_state["source_counts"] = (
            source_counts
            if vjp_mode in {"batched", "scalar"}
            else source_counts.expand(forward_batch)
        )
        hook_state["source_start"] = int(skip_first)
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
        elif (
            int(state["d_model"]) != int(d_model)
            or set(state["acc"]) != set(sources)
        ):
            raise ValueError(
                "initial_lens shape/layers do not match the loaded model"
            )
        stripe_capacity = _row_stripe_capacity(
            d_model, len(sources), batch,
        )
        capacity_limit = state.get("stripe_capacity_limit")
        if capacity_limit is not None:
            stripe_capacity = max(
                batch, min(stripe_capacity, int(capacity_limit)),
            )
        if device.type == "cuda" and state["stripe_rows"] is None:
            # Two device/host slots let stripe N's D2H transfer overlap the
            # backward passes that fill stripe N+1. Allocation pressure first
            # drops overlap, then halves the stripe itself. Prompt/dim backoff
            # cannot cure a fixed staging allocation, so do that recovery here.
            while True:
                device_rows: list[dict[int, torch.Tensor]] | None = None
                host_rows: list[dict[int, torch.Tensor]] | None = None
                try:
                    device_rows = [
                        {
                            l: torch.empty(
                                stripe_capacity, d_model, dtype=torch.float32,
                                device=device,
                            )
                            for l in sources
                        }
                        for _ in range(2)
                    ]
                    host_rows = [
                        {
                            l: torch.empty(
                                stripe_capacity, d_model, dtype=torch.float32,
                                pin_memory=True,
                            )
                            for l in sources
                        }
                        for _ in range(2)
                    ]
                    transfer_stream = torch.cuda.Stream(device=device)
                except RuntimeError:
                    device_rows = None
                    host_rows = None
                    _empty_device_cache(device)
                    one_device: dict[int, torch.Tensor] | None = None
                    one_host: dict[int, torch.Tensor] | None = None
                    try:
                        one_device = {
                            l: torch.empty(
                                stripe_capacity, d_model, dtype=torch.float32,
                                device=device,
                            )
                            for l in sources
                        }
                        try:
                            one_host = {
                                l: torch.empty(
                                    stripe_capacity, d_model, dtype=torch.float32,
                                    pin_memory=True,
                                )
                                for l in sources
                            }
                        except RuntimeError:
                            one_host = {
                                l: torch.empty(
                                    stripe_capacity, d_model, dtype=torch.float32,
                                )
                                for l in sources
                            }
                        transfer_stream = torch.cuda.Stream(device=device)
                        assert one_device is not None and one_host is not None
                        device_rows = [one_device]
                        host_rows = [one_host]
                        log.warning(
                            "jlens: CUDA stripe double-buffer allocation failed; "
                            "using one transfer slot (%d rows)", stripe_capacity,
                        )
                    except RuntimeError:
                        one_device = None
                        one_host = None
                        device_rows = None
                        host_rows = None
                        _empty_device_cache(device)
                        smaller = _smaller_row_stripe_capacity(
                            stripe_capacity, batch,
                        )
                        if smaller is None:
                            raise
                        log.warning(
                            "jlens: CUDA stripe allocation failed — retrying "
                            "with %d rows instead of %d",
                            smaller, stripe_capacity,
                        )
                        stripe_capacity = smaller
                        continue
                assert device_rows is not None and host_rows is not None
                state["stripe_rows"] = device_rows
                state["host_stripes"] = host_rows
                state["cuda_transfer_stream"] = transfer_stream
                state["stripe_capacity"] = stripe_capacity
                break
        elif device.type == "mps" and state["stripe_rows"] is None:
            while True:
                device_rows_mps: dict[int, torch.Tensor] | None = None
                host_rows_mps: dict[int, torch.Tensor] | None = None
                try:
                    device_rows_mps = {
                        l: torch.empty(
                            stripe_capacity, d_model, dtype=torch.float32,
                            device=device,
                        )
                        for l in sources
                    }
                    host_rows_mps = {
                        l: torch.empty(
                            stripe_capacity, d_model, dtype=torch.float32,
                        )
                        for l in sources
                    }
                except RuntimeError:
                    device_rows_mps = None
                    host_rows_mps = None
                    _empty_device_cache(device)
                    smaller = _smaller_row_stripe_capacity(
                        stripe_capacity, batch,
                    )
                    if smaller is None:
                        raise
                    log.warning(
                        "jlens: MPS stripe allocation failed — retrying with "
                        "%d rows instead of %d", smaller, stripe_capacity,
                    )
                    stripe_capacity = smaller
                    continue
                assert device_rows_mps is not None and host_rows_mps is not None
                state["stripe_rows"] = device_rows_mps
                state["host_stripes"] = host_rows_mps
                state["stripe_capacity"] = stripe_capacity
                break
        if device.type in {"cuda", "mps"}:
            stripe_capacity = int(state["stripe_capacity"])
        stripe_rows = state["stripe_rows"]
        host_stripes = state["host_stripes"]
        stripe_start = int(row_start)
        cuda_slot = 0
        cuda_pending = (
            [None] * len(stripe_rows)
            if device.type == "cuda" and isinstance(stripe_rows, list)
            else []
        )
        source_tensors: list[torch.Tensor] = []
        for l in sources:
            if not captured[l].requires_grad:
                raise JacobianLensError(
                    f"layer {l} output carries no grad — the seed hook did not "
                    "reach the residual stream (unsupported block call shape?)"
                )
            source_tensors.append(captured[l])

        def commit_host_stripe(
            host: Mapping[int, torch.Tensor], start: int, end: int,
        ) -> None:
            nonlocal committed_until
            n = end - start
            # Validate every layer before committing any: a failed asynchronous
            # transfer must not leave a partially counted stripe.
            for layer in sources:
                rows = host[layer][:n]
                if bool((rows.abs().sum(dim=1) == 0).any()):
                    raise JacobianLensError(
                        f"layer {layer} came back with zero rows from the device — "
                        "likely an asynchronous out of memory on the command "
                        "queue; retrying at a smaller dim_batch"
                    )
            for layer in sources:
                state["acc"][layer][start:end].add_(host[layer][:n])
            committed_until = end

        def _drain_cuda_slot(slot: int) -> None:
            if not cuda_pending or cuda_pending[slot] is None:
                return
            pending = cuda_pending[slot]
            assert pending is not None
            start, end, done = pending
            done.synchronize()
            assert isinstance(host_stripes, list)
            commit_host_stripe(host_stripes[slot], start, end)
            cuda_pending[slot] = None

        def _drain_cuda_pending_in_order() -> None:
            """Commit completed transfers monotonically; discard suffix on error."""
            pending_order = sorted(
                (pending[0], slot)
                for slot, pending in enumerate(cuda_pending)
                if pending is not None
            )
            try:
                for _start, slot in pending_order:
                    _drain_cuda_slot(slot)
            except BaseException:
                # A failed event wait/validation leaves its slot record intact.
                # Quiesce all copy work before clearing the uncommitted suffix;
                # no later stripe may cross the failed predecessor.
                transfer_stream = state.get("cuda_transfer_stream")
                try:
                    if transfer_stream is not None:
                        transfer_stream.synchronize()
                except BaseException:
                    # Preserve the original drain/validation failure. CUDA may
                    # surface the same asynchronous error again while quiescing.
                    pass
                finally:
                    for slot in range(len(cuda_pending)):
                        cuda_pending[slot] = None
                raise

        drain_cuda_pending_in_order = _drain_cuda_pending_in_order

        drain_cuda_slot = _drain_cuda_slot

        def flush_stripe(end: int) -> None:
            nonlocal stripe_start, cuda_slot
            if device.type == "cpu" or end <= stripe_start:
                return
            assert host_stripes is not None and stripe_rows is not None
            start = stripe_start
            n = end - start
            if device.type == "cuda":
                assert isinstance(stripe_rows, list)
                assert isinstance(host_stripes, list)
                transfer_stream = state["cuda_transfer_stream"]
                assert isinstance(transfer_stream, torch.cuda.Stream)
                next_slot = (cuda_slot + 1) % len(stripe_rows)
                # The previous transfer in the slot we are about to reuse has
                # overlapped all computation that filled the current stripe.
                # Commit it before enqueueing a later stripe, preserving row
                # order even when synchronization/validation fails.
                drain_cuda_pending_in_order()
                assert cuda_pending[cuda_slot] is None
                current_stream = torch.cuda.current_stream(device)
                transfer_stream.wait_stream(current_stream)
                try:
                    with torch.cuda.stream(transfer_stream):
                        for layer in sources:
                            host_stripes[cuda_slot][layer][:n].copy_(
                                stripe_rows[cuda_slot][layer][:n],
                                non_blocking=True,
                            )
                        done = torch.cuda.Event()
                        done.record(transfer_stream)
                except BaseException:
                    # Some layer copies may already be queued, but no pending
                    # record exists until the whole stripe + event succeeds.
                    # Quiesce and discard this uncommitted stripe before retry.
                    transfer_stream.synchronize()
                    raise
                cuda_pending[cuda_slot] = (start, end, done)
                if len(stripe_rows) == 1:
                    # No second device buffer exists to fill while D2H runs.
                    # Drain before the caller reuses this sole slot.
                    drain_cuda_slot(cuda_slot)
                cuda_slot = next_slot
            else:
                assert isinstance(stripe_rows, dict)
                assert isinstance(host_stripes, dict)
                for layer in sources:
                    host_stripes[layer][:n].copy_(stripe_rows[layer][:n])
                torch.mps.synchronize()
                commit_host_stripe(host_stripes, start, end)
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
            if cancel_event is not None and cancel_event.is_set():
                # A cancellation must not leave accelerator work or an async
                # transfer alive after the fit worker releases the model. The
                # current prompt group's partial CPU sums are intentionally
                # abandoned; only complete-group checkpoints are resumable.
                drain_cuda_pending_in_order()
                if device.type == "mps":
                    torch.mps.synchronize()
                raise JacobianLensCancelled(
                    "Jacobian-lens fit cancelled during an active prompt group"
                )
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
            if cancel_event is not None and cancel_event.is_set():
                drain_cuda_pending_in_order()
                if device.type == "mps":
                    torch.mps.synchronize()
                raise JacobianLensCancelled(
                    "Jacobian-lens fit cancelled during an active prompt group"
                )
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
                )
                if device.type != "cpu":
                    assert stripe_rows is not None
                    if device.type == "cuda":
                        assert isinstance(stripe_rows, list)
                        target_rows = stripe_rows[cuda_slot]
                    else:
                        assert isinstance(stripe_rows, dict)
                        target_rows = stripe_rows
                    target_rows[l][stripe_offset : stripe_offset + n_dims] = block
                else:
                    state["acc"][l][dim_start:write_end].add_(block)
            if device.type == "cpu":
                committed_until = write_end
            pass_idx += 1
            if device.type == "mps" and pass_idx % _MPS_SYNC_EVERY_PASSES == 0:
                torch.mps.synchronize()
            dim_start = write_end
        flush_stripe(d_model)
        drain_cuda_pending_in_order()
    except RuntimeError as exc:
        # A later VJP can fail while the previous CUDA stripe is still in
        # flight. Finish and commit that known-good transfer before reporting
        # the durable row boundary to the retry scheduler.
        drain_cuda_pending_in_order()
        if isinstance(exc, _BatchedVjpUnavailable):
            exc.committed_until = max(exc.committed_until, committed_until)
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
) -> torch.Tensor:
    if mode == "batched":
        # [n_dims, B, D] — source-position means were collapsed by the
        # transparent probe identity inside autograd. Preserve equal-prompt
        # weighting by summing those per-prompt Jacobians.
        return grad[:n_dims].to(torch.float32).sum(dim=1)
    # [replicated_batch,D] (or [1,D] scalar) is already mean-position reduced.
    return grad[:n_dims].to(torch.float32)


def _downgrade_cuda_stripe_buffers(
    state: dict[str, Any], device: torch.device,
) -> bool:
    """Release the overlap slot after a CUDA OOM and keep one viable slot."""
    if device.type != "cuda":
        return False
    device_rows = state.get("stripe_rows")
    host_rows = state.get("host_stripes")
    if (
        not isinstance(device_rows, list) or len(device_rows) <= 1
        or not isinstance(host_rows, list) or len(host_rows) <= 1
    ):
        return False
    state["stripe_rows"] = [device_rows[0]]
    state["host_stripes"] = [host_rows[0]]
    # Drop the local references before emptying the allocator in the caller.
    del device_rows[1:]
    del host_rows[1:]
    return True


def _shrink_device_stripe_buffers(
    state: dict[str, Any], device: torch.device, dim_batch: int,
) -> bool:
    """Release a quiescent one-slot staging tier and lower its next capacity.

    ``_accumulate_prompt_jacobian`` drains every pending CUDA transfer before an
    OOM escapes, so the retry scheduler can safely discard these persistent
    buffers without losing a committed row prefix.
    """
    if device.type not in {"cuda", "mps"}:
        return False
    capacity = int(state.get("stripe_capacity", 0) or 0)
    smaller = _smaller_row_stripe_capacity(capacity, dim_batch)
    if smaller is None:
        return False
    state["stripe_rows"] = None
    state["host_stripes"] = None
    state["cuda_transfer_stream"] = None
    state["stripe_capacity"] = 0
    state["stripe_capacity_limit"] = smaller
    return True


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
