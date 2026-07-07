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

Estimator (reference: github.com/anthropics/jacobian-lens, followed exactly):
one forward per prompt with the prompt replicated ``dim_batch``× along the
batch axis and the graph retained, then ``ceil(d_model/dim_batch)`` backward
passes. Batch element ``b`` of pass ``p`` carries a one-hot cotangent at
output dim ``p·dim_batch + b`` injected at every valid target position, so
the gradient at source position ``t`` is ``Σ_{t'≥t} ∂h_final[t']/∂h_l[t]``;
the mean over source positions gives ``dim_batch`` rows of every layer's
``J_l`` per backward. The first ``SKIP_FIRST_POSITIONS`` positions (attention
sinks) and the final position are excluded from both cotangents and the
source mean.

This is the ONLY module in saklas that runs backward passes. The fit builds
its own autograd-enabled forward (``torch.enable_grad()`` + a grad-seeding
pre-hook on the first block) — the ``inference_mode`` capture machinery in
``vectors.py`` cannot be reused, because inference tensors never re-enter
autograd. Per-layer grads come from ``torch.autograd.grad(final, sources)``
— NOT ``backward()`` + ``retain_grad()`` (``.grad`` accumulates across the
multi-backward loop and would corrupt the one-hot-cotangent rows) — which
also stops the graph walk at the shallowest requested source layer, so a
band-restricted fit never backprops below its lowest source. Each pass
writes its row block into a per-layer ON-DEVICE fp32 buffer; the only
device→host transfer is one fold per prompt into the CPU fp32 cross-prompt
accumulator (a per-pass ``.cpu()`` is a blocking sync per source layer —
worth ~6% on MPS). The sync budget is bounded from BOTH sides: a fully
unsynced pass loop lets the CPU enqueue arbitrarily far ahead of the device,
and Metal reports the resulting queue exhaustion asynchronously — no Python
exception, the work silently never runs, the fold reads zeros — hence the
periodic drain (``_MPS_SYNC_EVERY_PASSES``) plus the zero-row fold guard.
The fit is compute-bound (each prompt ≈ ``d_model × 2`` forward-equivalents
of backward, dim_batch-invariant); restricting ``source_layers`` is the one
lever that removes work. Entries are O(1), so fp16 storage downstream is
safe.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from saklas.core.errors import SaklasError

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
#: Checkpoint cadence (prompts) for resumable fits.
DEFAULT_CHECKPOINT_EVERY = 25
#: Backward passes between queue drains on MPS. Metal reports command-queue
#: exhaustion as an *asynchronous* command-buffer error — no Python exception,
#: the encoded ops silently never complete — so an unsynchronized pass loop
#: that runs ahead of the device turns into all-zero gradients, not an OOM.
#: A bounded drain every few passes caps the in-flight transients; the
#: all-zero fold guard below catches whatever still slips through.
_MPS_SYNC_EVERY_PASSES = 4


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
        self, token_id: int, unembed: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Per-layer J-lens direction for one vocab id: ``W_U[v] @ J_l``.

        Returns fp32 CPU tensors in the ``dict[int, Tensor]`` shape every
        saklas profile consumer (``fold_directions_to_subspace``,
        ``Profile``) expects.
        """
        w = unembed[token_id].detach().to(torch.float32).cpu()
        return {l: w @ J for l, J in self.jacobians.items()}

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
            acc = torch.zeros(first.d_model, first.d_model, dtype=torch.float32)
            for lens in lenses:
                acc += lens.jacobians[layer] * (lens.n_prompts / total)
            merged[layer] = acc
        return cls(merged, n_prompts=total, d_model=first.d_model)


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
    atom_norms = _atom_norms(J, unembed).clamp(min=1e-12)

    selected: list[int] = []
    rows: list[torch.Tensor] = []
    coeffs = torch.zeros(0, device=device)
    residual = t.clone()
    for _ in range(k):
        # normalized correlation over the vocabulary, D never materialized
        scores = (unembed @ (J @ residual).to(unembed.dtype)).float() / atom_norms
        if selected:
            scores[torch.tensor(selected, device=device)] = -torch.inf
        best = int(scores.argmax())
        if float(scores[best]) <= 0.0:
            break
        selected.append(best)
        rows.append(unembed[best].float() @ J)
        A = torch.stack(rows)  # [s, d]
        gram = A @ A.T
        # CPU hop: eigvalsh is unimplemented on MPS, and the gram is ≤ k×k.
        lipschitz = float(torch.linalg.eigvalsh(gram.cpu())[-1].clamp(min=1e-12))
        b = A @ t
        c = torch.cat([coeffs, coeffs.new_zeros(1)])
        for _ in range(nnls_iters):
            c = torch.clamp(c - (gram @ c - b) / lipschitz, min=0.0)
        coeffs = c
        residual = t - A.T @ coeffs

    share = max(0.0, 1.0 - float(residual.pow(2).sum()) / t_norm_sq)
    pairs = sorted(
        ((tok, float(cf)) for tok, cf in zip(selected, coeffs) if float(cf) > 0.0),
        key=lambda p: -p[1],
    )
    return JSpaceDecomposition(layer, share, pairs)


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


def fit_jacobian_lens(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layer_modules: Sequence[nn.Module],
    *,
    source_layers: Sequence[int] | None = None,
    dim_batch: int = DEFAULT_DIM_BATCH,
    max_seq_len: int = DEFAULT_SEQ_LEN,
    skip_first: int = SKIP_FIRST_POSITIONS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    checkpoint_cb: Callable[[JacobianLens], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> JacobianLens:
    """Fit ``J_l`` for every source layer over ``prompts``.

    One forward + ``ceil(d_model/dim_batch)`` backwards per prompt (see module
    docstring for the estimator). ``checkpoint_cb`` receives the partial lens
    every ``checkpoint_every`` prompts — the io layer uses it for resumable
    fits (callers merging with a prior shard do so outside this function).
    ``dim_batch`` halves automatically on device OOM.

    Raises :class:`JacobianLensError` when no prompt in the corpus is long
    enough (each needs > ``skip_first + 1`` tokens).
    """
    device = next(model.parameters()).device
    n_layers = len(layer_modules)
    final_idx = n_layers - 1
    sources = (
        sorted(set(source_layers)) if source_layers is not None else list(range(final_idx))
    )
    if any(l < 0 or l >= final_idx for l in sources):
        raise ValueError(
            f"source_layers must lie in [0, {final_idx}) — the final layer is "
            f"the transport target, not a source; got {sources}"
        )

    # Cross-prompt state, threaded through every per-prompt sweep: the CPU
    # fp32 accumulator, plus per-layer on-device row buffers reused across
    # prompts (allocated once d_model is known).
    state: dict[str, Any] = {"acc": None, "d_model": 0, "dev_rows": None}
    n_done = 0

    def _partial() -> JacobianLens:
        acc = state["acc"]
        assert acc is not None
        return JacobianLens(
            {l: a / max(n_done, 1) for l, a in acc.items()},
            n_prompts=n_done,
            d_model=state["d_model"],
        )

    with torch.enable_grad():
        for prompt_idx, prompt in enumerate(prompts):
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"][:, :max_seq_len]
            seq_len = ids.shape[1]
            if seq_len < skip_first + 2:
                log.warning(
                    "jlens: skipping prompt %d — %d tokens, need > %d",
                    prompt_idx, seq_len, skip_first + 1,
                )
                continue
            ids = ids.to(device)
            batch = max(1, min(dim_batch, state["d_model"] or dim_batch))
            while True:
                try:
                    _accumulate_prompt_jacobian(
                        model, ids, layer_modules, sources, final_idx, state,
                        batch=batch, skip_first=skip_first,
                    )
                    break
                except RuntimeError as exc:  # OOM → halve dim_batch and retry
                    if "out of memory" not in str(exc).lower() or batch <= 1:
                        raise
                    batch = max(1, batch // 2)
                    _empty_device_cache(device)
                    log.warning("jlens: OOM — retrying prompt with dim_batch=%d", batch)
            dim_batch = batch
            n_done += 1
            if on_progress is not None:
                on_progress(f"prompt {n_done}/{len(prompts)} (dim_batch={batch})")
            if checkpoint_cb is not None and n_done % checkpoint_every == 0:
                checkpoint_cb(_partial())
                # Allocator hygiene at checkpoint cadence only — a per-prompt
                # empty_cache forces a sync and dumps the pool the very next
                # prompt re-allocates.
                _empty_device_cache(device)

    if state["acc"] is None or n_done == 0:
        raise JacobianLensError(
            f"no usable prompts: every prompt had <= {skip_first + 1} tokens"
        )
    return _partial()


def _accumulate_prompt_jacobian(
    model: Any,
    ids: torch.Tensor,
    layer_modules: Sequence[nn.Module],
    sources: Sequence[int],
    final_idx: int,
    state: dict[str, Any],
    *,
    batch: int,
    skip_first: int,
) -> None:
    """Run one prompt's forward + backward sweep, adding into ``state``.

    The fold into ``state["acc"]`` happens once at the end, all-or-nothing,
    so a mid-prompt OOM retry never double-counts. Row blocks land in the
    reused per-layer device buffers (``state["dev_rows"]``) — each
    ``(pass, layer)`` writes a disjoint block, and the device→host transfer
    happens once per prompt, off the backward critical path.
    """
    seq_len = ids.shape[1]
    device = ids.device
    valid = torch.arange(skip_first, seq_len - 1, device=device)
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def seed_hook(
        _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # With frozen params and integer inputs no autograd graph exists at
        # all — seed a leaf into the residual stream at the first block.
        if args:
            seeded = args[0].detach().clone().requires_grad_(True)
            return (seeded, *args[1:]), kwargs
        seeded = kwargs["hidden_states"].detach().clone().requires_grad_(True)
        return args, {**kwargs, "hidden_states": seeded}

    def make_capture(idx: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            captured[idx] = _output_tensor(output)
        return hook

    # Seed at the LOWEST source block, not block 0: everything below it then
    # runs graph-free (frozen params + a detached input build no autograd
    # state), so a band-restricted fit pays neither graph memory nor backward
    # depth below its band. For the default all-layer fit this is block 0.
    handles.append(
        layer_modules[min(sources)].register_forward_pre_hook(
            seed_hook, with_kwargs=True
        )
    )
    for idx in {*sources, final_idx}:
        handles.append(layer_modules[idx].register_forward_hook(make_capture(idx)))

    try:
        replicated = ids.expand(batch, -1)
        try:
            model(input_ids=replicated, use_cache=False)
        except TypeError:  # toy/CPU-test models without a use_cache kwarg
            model(input_ids=replicated)
        final = captured[final_idx]
        d_model = final.shape[-1]
        if state["acc"] is None:
            state["acc"] = {
                l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in sources
            }
            state["d_model"] = d_model
        if state["dev_rows"] is None:
            try:
                state["dev_rows"] = {
                    l: torch.zeros(d_model, d_model, dtype=torch.float32, device=device)
                    for l in sources
                }
            except RuntimeError as exc:  # device too tight for the buffers
                if "out of memory" not in str(exc).lower():
                    raise
                state["dev_rows"] = {
                    l: torch.zeros(d_model, d_model, dtype=torch.float32)
                    for l in sources
                }
                log.warning(
                    "jlens: device row buffers do not fit — accumulating on "
                    "CPU (per-pass sync transfers; the fit will be slower)"
                )
        dev_rows: dict[int, torch.Tensor] = state["dev_rows"]
        on_device = next(iter(dev_rows.values())).device == device
        source_tensors: list[torch.Tensor] = []
        for l in sources:
            if not captured[l].requires_grad:
                raise JacobianLensError(
                    f"layer {l} output carries no grad — the seed hook did not "
                    "reach the residual stream (unsupported block call shape?)"
                )
            source_tensors.append(captured[l])
            dev_rows[l].zero_()

        n_passes = math.ceil(d_model / batch)
        cot = torch.zeros_like(final)
        batch_rows = torch.arange(batch, device=device)
        for p in range(n_passes):
            dim_start = p * batch
            n_dims = min(batch, d_model - dim_start)
            cot.zero_()
            rows = batch_rows[:n_dims].unsqueeze(1)
            cot[rows, valid.unsqueeze(0), (dim_start + batch_rows[:n_dims]).unsqueeze(1)] = 1.0
            # grad(final, sources) rather than backward(): the grads return
            # directly (no hooks), and the walk stops at the shallowest
            # requested layer instead of descending to the seed leaf.
            grads = torch.autograd.grad(
                final, source_tensors, grad_outputs=cot,
                retain_graph=p < n_passes - 1,
            )
            for l, g in zip(sources, grads):
                block = g[:n_dims, skip_first : seq_len - 1].mean(
                    dim=1, dtype=torch.float32
                )
                dev_rows[l][dim_start : dim_start + n_dims] = (
                    block if on_device else block.cpu()
                )
            if device.type == "mps" and (p + 1) % _MPS_SYNC_EVERY_PASSES == 0:
                torch.mps.synchronize()
        # Two-phase fold: transfer everything first, then add — a transfer
        # failure mid-fold must not leave some layers already accumulated
        # (the OOM retry would double-count them).
        host_rows = {l: dev_rows[l].cpu() for l in sources}
        for l in sources:
            # A zero ROW of J_l is impossible for a real transformer (the
            # residual identity path alone makes every output dim depend on
            # every layer) — it means the device dropped that pass's work,
            # i.e. an asynchronous command-buffer OOM. Raise with the
            # "out of memory" phrasing so the dim_batch-halving retry fires.
            if bool((host_rows[l].abs().sum(dim=1) == 0).any()):
                raise JacobianLensError(
                    f"layer {l} came back with zero rows from the device — "
                    "likely an asynchronous out of memory on the command "
                    "queue; retrying at a smaller dim_batch"
                )
        for l in sources:
            state["acc"][l] += host_rows[l]
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
