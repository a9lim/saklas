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
import os
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
#: MPS unified memory is most fragile when every fitted layer owns a full
#: ``[d_model, d_model]`` fp32 device buffer.  Keep a small per-layer stripe on
#: device, validate it, then fold into a local CPU prompt buffer.
_MPS_ROW_STRIPE = 256
#: After an OOM halves ``dim_batch``, try the original width again only after a
#: few successful prompts.  One unusually long prompt should not punish the
#: remainder of the corpus forever.
_DIM_BATCH_GROW_AFTER_PROMPTS = 4


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
            acc = torch.zeros(first.d_model, first.d_model, dtype=torch.float32)
            for lens in lenses:
                acc += lens.jacobians[layer] * (lens.n_prompts / total)
            merged[layer] = acc
        return cls(merged, n_prompts=total, d_model=first.d_model)

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
    """Internal signal: retry the current prompt with replicated VJPs."""


def _resolve_vjp_mode(mode: str) -> str:
    env = os.environ.get("SAKLAS_JLENS_VJP")
    raw = (env or mode).strip().lower()
    if raw not in {"auto", "batched", "replicated"}:
        raise ValueError(
            "vjp_mode must be 'auto', 'batched', or 'replicated' "
            f"(got {mode!r})"
        )
    return raw


def _looks_like_batched_vjp_unsupported(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    if "out of memory" in msg:
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
    max_seq_len: int = DEFAULT_SEQ_LEN,
    skip_first: int = SKIP_FIRST_POSITIONS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    checkpoint_cb: Callable[[JacobianLens], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
    input_id_rows: Sequence[Sequence[int]] | None = None,
    vjp_mode: str = "auto",
) -> JacobianLens:
    """Fit ``J_l`` for every source layer over ``prompts``.

    One forward + ``ceil(d_model/dim_batch)`` backwards per prompt (see module
    docstring for the estimator). ``checkpoint_cb`` receives the partial lens
    every ``checkpoint_every`` prompts — the io layer uses it for resumable
    fits (callers merging with a prior shard do so outside this function).
    ``input_id_rows`` is the pre-tokenized sibling of ``prompts``.  When supplied
    it must be aligned 1:1 with ``prompts`` and already reflect the caller's
    truncation/hash policy; the fit still applies ``max_seq_len`` defensively.
    This lets session-level resume/filtering reuse the IDs it already computed
    instead of tokenizing the corpus a second time.  ``vjp_mode="batched"``
    computes each output-dim block with ``is_grads_batched=True`` from a single
    prompt forward instead of replicating the forward graph; ``"auto"`` tries
    that path first and falls back to the reference replicated estimator if the
    backend lacks vmap coverage.  ``dim_batch`` halves automatically on device
    OOM and cautiously grows back toward the requested value after successful
    prompts.

    Raises :class:`JacobianLensError` when no prompt in the corpus is long
    enough (each needs > ``skip_first + 1`` tokens).
    """
    device = next(model.parameters()).device
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
        "dev_rows": None,
        "row_norms": None,
        "stripe_rows": None,
        "stripe_norms": None,
        "vjp_mode": "batched" if requested_vjp_mode == "auto" else requested_vjp_mode,
        "requested_vjp_mode": requested_vjp_mode,
    }
    n_done = 0
    target_dim_batch = int(dim_batch)
    active_dim_batch = int(dim_batch)
    successes_since_resize = 0

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
            for prompt_idx, prompt in enumerate(prompts):
                if input_id_rows is None:
                    ids = tokenizer(prompt, return_tensors="pt")["input_ids"][
                        :, :max_seq_len
                    ]
                else:
                    row = [int(tok) for tok in input_id_rows[prompt_idx]][:max_seq_len]
                    ids = torch.tensor([row], dtype=torch.long)
                seq_len = ids.shape[1]
                if seq_len < skip_first + 2:
                    log.warning(
                        "jlens: skipping prompt %d — %d tokens, need > %d",
                        prompt_idx, seq_len, skip_first + 1,
                    )
                    continue
                ids = ids.to(device)
                batch = max(1, min(active_dim_batch, state["d_model"] or active_dim_batch))
                while True:
                    retry_batch: int | None = None
                    try:
                        _accumulate_prompt_jacobian(
                            model, ids, layer_modules, sources, final_idx, state,
                            captured=captured, batch=batch, skip_first=skip_first,
                        )
                        break
                    except _BatchedVjpUnavailable:
                        assert requested_vjp_mode == "auto"
                        state["vjp_mode"] = "replicated"
                        log.warning(
                            "jlens: batched VJP is unavailable on this backend — "
                            "falling back to replicated-prompt VJPs"
                        )
                        continue
                    except RuntimeError as exc:  # OOM → halve dim_batch and retry
                        msg = str(exc).lower()
                        if "out of memory" not in msg or batch <= 1:
                            raise
                        retry_batch = max(1, batch // 2)
                    assert retry_batch is not None
                    batch = retry_batch
                    active_dim_batch = batch
                    successes_since_resize = 0
                    _empty_device_cache(device)
                    log.warning("jlens: OOM — retrying prompt with dim_batch=%d", batch)
                n_done += 1
                successes_since_resize += 1
                if on_progress is not None:
                    mode = str(state["vjp_mode"])
                    on_progress(
                        f"prompt {n_done}/{len(prompts)} "
                        f"(dim_batch={batch}, vjp={mode})"
                    )
                if (
                    active_dim_batch < target_dim_batch
                    and successes_since_resize >= _DIM_BATCH_GROW_AFTER_PROMPTS
                ):
                    active_dim_batch = min(target_dim_batch, active_dim_batch * 2)
                    successes_since_resize = 0
                if checkpoint_cb is not None and n_done % checkpoint_every == 0:
                    checkpoint_cb(_partial())
                    # Allocator hygiene at checkpoint cadence only — a per-prompt
                    # empty_cache forces a sync and dumps the pool the very next
                    # prompt re-allocates.
                    _empty_device_cache(device)
    finally:
        for handle in handles:
            handle.remove()

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
    captured: dict[int, torch.Tensor],
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
    del layer_modules
    device = ids.device
    valid = torch.arange(skip_first, seq_len - 1, device=device)
    captured.clear()
    try:
        vjp_mode = str(state["vjp_mode"])
        forward_batch = 1 if vjp_mode == "batched" else batch
        replicated = ids.expand(forward_batch, -1)
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
        if state["dev_rows"] is None and device.type == "mps":
            state["dev_rows"] = {
                l: torch.empty(0, dtype=torch.float32, device=device) for l in sources
            }
            state["row_norms"] = {
                l: torch.empty(0, dtype=torch.float32, device=device) for l in sources
            }
        elif state["dev_rows"] is None:
            try:
                state["dev_rows"] = {
                    l: torch.zeros(d_model, d_model, dtype=torch.float32, device=device)
                    for l in sources
                }
                state["row_norms"] = {
                    l: torch.zeros(d_model, dtype=torch.float32, device=device)
                    for l in sources
                }
            except RuntimeError as exc:  # device too tight for the buffers
                if "out of memory" not in str(exc).lower():
                    raise
                state["dev_rows"] = {
                    l: torch.zeros(d_model, d_model, dtype=torch.float32)
                    for l in sources
                }
                state["row_norms"] = {
                    l: torch.zeros(d_model, dtype=torch.float32)
                    for l in sources
                }
                log.warning(
                    "jlens: device row buffers do not fit — accumulating on "
                    "CPU (per-pass sync transfers; the fit will be slower)"
                )
        dev_rows: dict[int, torch.Tensor] = state["dev_rows"]
        row_norms: dict[int, torch.Tensor] = state["row_norms"]
        on_device = next(iter(dev_rows.values())).device == device
        use_striped_rows = device.type == "mps" and on_device
        if use_striped_rows and state["stripe_rows"] is None:
            stripe = min(_MPS_ROW_STRIPE, d_model)
            state["stripe_rows"] = {
                l: torch.empty(stripe, d_model, dtype=torch.float32, device=device)
                for l in sources
            }
            state["stripe_norms"] = {
                l: torch.empty(stripe, dtype=torch.float32, device=device)
                for l in sources
            }
        host_rows: dict[int, torch.Tensor] | None = None
        host_norms: dict[int, torch.Tensor] | None = None
        stripe_rows: dict[int, torch.Tensor] | None = state["stripe_rows"]
        stripe_norms: dict[int, torch.Tensor] | None = state["stripe_norms"]
        stripe_start = 0
        source_tensors: list[torch.Tensor] = []
        for l in sources:
            if not captured[l].requires_grad:
                raise JacobianLensError(
                    f"layer {l} output carries no grad — the seed hook did not "
                    "reach the residual stream (unsupported block call shape?)"
                )
            source_tensors.append(captured[l])
            if use_striped_rows:
                if host_rows is None:
                    host_rows = {
                        layer: torch.empty(d_model, d_model, dtype=torch.float32)
                        for layer in sources
                    }
                    host_norms = {
                        layer: torch.empty(d_model, dtype=torch.float32)
                        for layer in sources
                    }
            else:
                row_norms[l].zero_()

        def flush_stripe(end: int) -> None:
            nonlocal stripe_start
            if not use_striped_rows or end <= stripe_start:
                return
            assert host_rows is not None and host_norms is not None
            assert stripe_rows is not None and stripe_norms is not None
            n = end - stripe_start
            for layer in sources:
                norms = stripe_norms[layer][:n].cpu()
                if bool((norms == 0).any()):
                    raise JacobianLensError(
                        f"layer {layer} came back with zero rows from the device — "
                        "likely an asynchronous out of memory on the command "
                        "queue; retrying at a smaller dim_batch"
                    )
                host_rows[layer][stripe_start:end] = stripe_rows[layer][:n].cpu()
                host_norms[layer][stripe_start:end] = norms
            stripe_start = end

        n_passes = math.ceil(d_model / batch)
        cot = None
        prev_dims: torch.Tensor | None = None
        batch_rows = torch.arange(batch, device=device)
        for p in range(n_passes):
            dim_start = p * batch
            n_dims = min(batch, d_model - dim_start)
            dims = dim_start + batch_rows[:n_dims]
            # grad(final, sources) rather than backward(): the grads return
            # directly (no hooks), and the walk stops at the shallowest
            # requested layer instead of descending to the seed leaf.
            try:
                grads = _grad_row_block(
                    final, source_tensors, valid, dims,
                    mode=vjp_mode,
                    retain_graph=p < n_passes - 1,
                    cotangent=cot,
                    prev_dims=prev_dims,
                )
            except RuntimeError as exc:
                if (
                    state["requested_vjp_mode"] == "auto"
                    and vjp_mode == "batched"
                    and _looks_like_batched_vjp_unsupported(exc)
                ):
                    raise _BatchedVjpUnavailable() from exc
                raise
            if vjp_mode == "replicated":
                if cot is None:
                    cot = torch.zeros_like(final)
                prev_dims = dims
            write_end = dim_start + n_dims
            stripe_offset = 0
            if use_striped_rows:
                assert stripe_rows is not None and stripe_norms is not None
                if write_end - stripe_start > stripe_rows[sources[0]].shape[0]:
                    flush_stripe(dim_start)
                stripe_offset = dim_start - stripe_start
            for l, g in zip(sources, grads):
                block = _source_grad_block(
                    g, mode=vjp_mode, n_dims=n_dims,
                    skip_first=skip_first, seq_len=seq_len,
                )
                norm = block.abs().sum(dim=1)
                if use_striped_rows:
                    assert stripe_rows is not None and stripe_norms is not None
                    stripe_rows[l][stripe_offset : stripe_offset + n_dims] = block
                    stripe_norms[l][stripe_offset : stripe_offset + n_dims] = norm
                else:
                    dev_rows[l][dim_start : write_end] = (
                        block if on_device else block.cpu()
                    )
                    row_norms[l][dim_start : write_end] = (
                        norm if on_device else norm.cpu()
                    )
            if device.type == "mps" and (p + 1) % _MPS_SYNC_EVERY_PASSES == 0:
                torch.mps.synchronize()
        flush_stripe(d_model)
        # Two-phase fold: transfer everything first, then add — a transfer
        # failure mid-fold must not leave some layers already accumulated
        # (the OOM retry would double-count them).
        if use_striped_rows:
            assert host_rows is not None and host_norms is not None
        else:
            host_rows = {l: dev_rows[l].cpu() for l in sources}
            host_norms = {l: row_norms[l].cpu() for l in sources}
        for l in sources:
            # A zero ROW of J_l is impossible for a real transformer (the
            # residual identity path alone makes every output dim depend on
            # every layer) — it means the device dropped that pass's work,
            # i.e. an asynchronous command-buffer OOM. Raise with the
            # "out of memory" phrasing so the dim_batch-halving retry fires.
            if bool((host_norms[l] == 0).any()):
                raise JacobianLensError(
                    f"layer {l} came back with zero rows from the device — "
                    "likely an asynchronous out of memory on the command "
                    "queue; retrying at a smaller dim_batch"
                )
        for l in sources:
            state["acc"][l] += host_rows[l]
    finally:
        captured.clear()


def _grad_row_block(
    final: torch.Tensor,
    source_tensors: Sequence[torch.Tensor],
    valid: torch.Tensor,
    dims: torch.Tensor,
    *,
    mode: str,
    retain_graph: bool,
    cotangent: torch.Tensor | None,
    prev_dims: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    if mode == "batched":
        outputs = final[0, valid[:, None], dims[None, :]].sum(dim=0)
        eye = torch.eye(dims.numel(), dtype=final.dtype, device=final.device)
        return torch.autograd.grad(
            outputs, source_tensors, grad_outputs=eye,
            retain_graph=retain_graph, is_grads_batched=True,
        )

    if cotangent is None:
        cotangent = torch.zeros_like(final)
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
    skip_first: int,
    seq_len: int,
) -> torch.Tensor:
    if mode == "batched":
        # [n_dims, 1, T, D] from batched VJP.
        return grad[:n_dims, 0, skip_first : seq_len - 1].mean(
            dim=1, dtype=torch.float32,
        )
    # [replicated_batch, T, D] from the reference replicated-prompt path.
    return grad[:n_dims, skip_first : seq_len - 1].mean(
        dim=1, dtype=torch.float32,
    )


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
