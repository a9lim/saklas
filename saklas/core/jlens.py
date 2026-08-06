"""Jacobian lens artifact + readout: transport, readout math, decomposition.

Implements the Jacobian lens of Gurnee et al., "Verbalizable Representations
Form a Global Workspace in Language Models" (Transformer Circuits, 2026):
per-layer matrices ``J_l = E[∂h_final/∂h_l]`` — the average first-order effect
of a layer-l residual on the final-layer residual, over positions and a text
corpus.  ``lens(h) = softmax(W_U · norm(J_l h))`` ranks
the vocabulary by what an activation is disposed to make the model say; the
J-lens *direction* for vocab id v at layer l is ``W_U[v] @ J_l`` — a per-layer
d-vector with the same shape contract as any saklas steering direction, which
is what lets a ``jlens/<word>`` term ride the ordinary profile registry.

This module holds everything on a runtime path: the :class:`JacobianLens`
artifact (transport, token directions, merge/union), the readout math every
probe / live-workspace / token-replay surface calls per decode step, and the
J-space decomposition.  It imports nothing beyond torch and the error
taxonomy, so a per-step import costs a ``sys.modules`` lookup.

The **estimator** — the only backward-pass code in saklas — lives in
:mod:`saklas.core.jlens_fit`.  ``fit_jacobian_lens`` remains reachable as
``saklas.core.jlens.fit_jacobian_lens`` through a lazy module ``__getattr__``,
so importing this module never drags the estimator in.

One unit runs through every readout surface: the per-layer softmax probability
``p_l(v)``.  ``readout_probabilities`` is the single calibration primitive; a
token's ``strength`` is ``mean_l p_l(v)`` and its depth center of mass is
weighted by the same ``p_l``, so per-layer cards, pinned probes, gate scalars,
and the layer aggregate are all apples-to-apples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from saklas.core.errors import SaklasError

#: Positions before this index are excluded from the Jacobian average — early
#: positions act as attention sinks with atypical residual statistics.  Part of
#: the corpus-spec identity the sidecar records, so it lives with the artifact
#: rather than the estimator.
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

# The fit-configuration defaults above stay with the artifact: io/cli/session
# read them to stamp and validate sidecars without wanting the estimator, and
# they are plain ints, so keeping them here costs the split nothing.


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


def topk_probabilities(
    logits: torch.Tensor, k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k readout probabilities without a full softmax tensor.

    ``topk(softmax(x))`` has the same indices as ``topk(x)``, so only the
    selected columns need normalizing — saving one vocab-sized allocation per
    readout row, which matters for large vocabularies and multi-layer J-lens
    sweeps.  The unit is the per-layer probability ``p_l(v)``, the same one
    ``readout_probabilities`` produces and every other lens surface reports.
    """
    logits_f = logits.float()
    vals, idxs = logits_f.topk(k, dim=-1)
    vals = (vals - logits_f.logsumexp(dim=-1, keepdim=True)).exp()
    return vals, idxs


def readout_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Calibrate per-layer lens logits into the shared probability unit.

    Kept as one explicit primitive so a live decode step can normalize its
    full ``[layers, vocab]`` matrix once, then share the result between pinned
    probes, per-layer cards, and the aggregate readout.
    """
    return logits.float().softmax(dim=-1)


def pack_readout_rows_to_host(*rows: torch.Tensor) -> torch.Tensor:
    """Pack selected readout rows for one accelerator-to-host transfer.

    The packed payload stays fp32 on the accelerator because MPS does not
    support float64.  fp32 still represents every practical vocabulary id
    exactly (up to 2**24), so integer token-id rows can share the transfer with
    probability/statistic rows without changing the public values.
    """
    return torch.cat([row.float() for row in rows], dim=0).detach().cpu()


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
    host = pack_readout_rows_to_host(stats, idxs.reshape(1, -1))
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


def __getattr__(name: str) -> Any:
    """Lazy compatibility alias for the estimator entry point.

    ``fit_jacobian_lens`` moved to :mod:`saklas.core.jlens_fit`; resolving it
    on demand keeps ``from saklas.core.jlens import fit_jacobian_lens``
    working without importing the estimator when only the readout is wanted.
    """
    if name == "fit_jacobian_lens":
        from saklas.core.jlens_fit import fit_jacobian_lens

        return fit_jacobian_lens
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
