"""Extraction, saving, and loading of activation steering/probe vectors."""

from __future__ import annotations

import functools
import json
import logging
import warnings
from collections.abc import Sequence
from contextlib import suppress
from importlib import resources as _resources
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file

from saklas.core.stats import median_or_zero

if TYPE_CHECKING:
    from saklas.core.sae import SaeBackend

log = logging.getLogger(__name__)

# Per-layer probe-quality diagnostics: thresholds for the soft warning the
# extractor emits at end-of-extraction.  Fired against the median across
# retained layers — a single dim layer with rough metrics is normal; a
# concept whose median is degenerate is the failure mode users care about.
_DIAG_DEGENERATE_EVR = 0.95         # ~all variance in one direction
_DIAG_DEGENERATE_INTRA_VAR = 0.01   # almost-identical pos/neg pairs
_DIM_DIAGNOSTIC_SAMPLE_MAX = 32

# Skip the chat template for extraction when it adds more than this many
# tokens of overhead (e.g. Ministral injects a ~500-token system prompt).
# The overhead cancels in contrastive diffs but wastes memory per pass.
_MAX_TEMPLATE_OVERHEAD = 100

# Keyed by id(tokenizer).  Object IDs can be reused after GC, so this
# cache is only safe when a single tokenizer lives for the session lifetime
# (which is the case in both the TUI and the API server).
_template_overhead_cache: dict[int, int] = {}


def _chat_template_overhead(tokenizer: Any, template_kwargs: dict[str, Any]) -> int:
    """Return the number of extra tokens the chat template adds beyond content."""
    cache_key = id(tokenizer)
    cached = _template_overhead_cache.get(cache_key)
    if cached is not None:
        return cached

    probe = "X"
    raw_len = len(tokenizer.encode(probe, add_special_tokens=False))
    messages = [{"role": "user", "content": "."}, {"role": "assistant", "content": probe}]
    try:
        wrapped = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **template_kwargs,
        )
        wrapped_len = len(tokenizer.encode(wrapped, add_special_tokens=False))
    except Exception:
        wrapped_len = raw_len  # can't measure, assume no overhead
    overhead = wrapped_len - raw_len
    _template_overhead_cache[cache_key] = overhead
    if overhead > _MAX_TEMPLATE_OVERHEAD:
        log.info("chat template adds %d tokens of overhead, skipping for extraction", overhead)
    return overhead


def _normalize(v: torch.Tensor, ref_norm: float | None = None) -> torch.Tensor:
    """Normalize a direction vector.

    If *ref_norm* is given the vector is scaled so that its norm equals
    *ref_norm* (i.e. it lives at the same magnitude as the hidden states
    it was derived from).  Otherwise the vector is L2-normalized to unit
    norm — which is fine for models without per-layer output scaling, but
    catastrophic for architectures like Gemma 4 whose cumulative
    ``layer_scalar`` shrinks the residual stream by orders of magnitude.
    """
    unit = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    if ref_norm is not None:
        return unit * ref_norm
    return unit


def _capture_all_hidden_states(model: torch.nn.Module, layers: torch.nn.ModuleList, input_ids: torch.Tensor):
    """Run a single-sequence forward pass capturing hidden states at ALL layers.

    Uses ``use_cache=False`` to avoid polluting any persistent KV cache.

    Returns:
        dict mapping layer index to (1, seq, dim) tensors.
    """
    captured_hidden: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def _hook(module: torch.nn.Module, input: Any, output: Any):
            h = output if isinstance(output, torch.Tensor) else output[0]
            # No .clone() — with use_cache=False and inference_mode() the
            # residual-stream tensors are fresh allocations at each layer
            # boundary (residual add produces a new tensor).  Detach severs
            # the autograd graph reference so the rest of the forward pass
            # can't invalidate the data.
            captured_hidden[idx] = h.detach()
        return _hook

    handles = [layers[idx].register_forward_hook(_make_hook(idx)) for idx in range(len(layers))]
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
        # Single sync after the full forward pass — lazy backends (MPS)
        # may not have materialised tensor data yet.
        if input_ids.device.type == "mps":
            torch.mps.synchronize()
    finally:
        for h in handles:
            h.remove()

    return captured_hidden


def special_token_ids(tokenizer: Any) -> set[int]:
    """Token ids that are structural rather than content.

    The canonical "non-content" set every last-content-token pooling site
    shares: the tokenizer's ``all_special_ids`` *plus* everything in
    ``added_tokens_encoder``.  The second arm matters for tokenizers that
    don't promote chat-boundary markers to ``all_special_ids`` (talkie's
    ``<|user|>``/``<|end|>``/``<|assistant|>`` are added tokens but not
    "special") — without it, pooling lands on a structural turn marker
    whose hidden state is disconnected from the content.
    """
    skip = set(getattr(tokenizer, "all_special_ids", []) or [])
    added = getattr(tokenizer, "added_tokens_encoder", None) or {}
    skip.update(int(v) for v in added.values())
    return skip


def last_content_index(token_ids: Sequence[int], tokenizer: Any) -> int:
    """Index of the last non-special token in ``token_ids``.

    Walks backward from the final position past every id in
    :func:`special_token_ids` (``all_special_ids`` + ``added_tokens_encoder``),
    flooring at 0 so a sequence that is entirely special tokens still
    yields a valid index.  This is the single definition of "last content
    token" shared by extraction (:func:`_encode_and_capture_all`), the
    aggregate vector probe (:meth:`TraitMonitor.score_per_token`), the
    incremental scoring path (``session._score_incremental``), and the
    manifold aggregate (:meth:`ManifoldMonitor.score_aggregate`) — every
    reported single-state value pools from here so the discipline can't
    drift per-site.
    """
    idx = len(token_ids) - 1
    if idx < 0:
        return 0
    skip = special_token_ids(tokenizer)
    if skip:
        while idx > 0 and int(token_ids[idx]) in skip:
            idx -= 1
    return idx


def _encode_and_capture_all(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    response: str,
    layers: torch.nn.ModuleList,
    device: torch.device,
    *,
    role: str | None = None,
    model_type: str | None = None,
):
    """Capture the last-content-token hidden state per layer for a turn pair, fp32.

    Conversational (4.0 / A2) capture: the corpus item is an assistant
    *response* to a fixed baseline *prompt*, so extraction pools the model in
    its real generation regime — ``[user: prompt, assistant: response]`` with
    no system turn, standard assistant label.  ``role`` overrides the assistant
    label only when an explicit per-node role is set (the persona-baselined
    fit); ``role=None`` is the swap-back default.

    Pools from the response's last non-special token — chat templates append
    trailing markers (Llama's <|eot_id|>, Gemma's <end_of_turn>, Qwen's
    <|im_end|>) whose hidden states are disconnected from content; the last
    content token is the attention-weighted summary the model uses for
    next-token prediction.

    Returns:
        dict mapping layer_idx -> pooled vector (dim,) in fp32.
    """
    ids, content_end = _render_and_tokenize_for_capture(
        tokenizer, prompt, response, device, role=role, model_type=model_type,
    )
    hidden_per_layer = _capture_all_hidden_states(model, layers, ids)
    return {
        idx: h[0, min(content_end, h.shape[1] - 1)].float()
        for idx, h in hidden_per_layer.items()
    }


def _render_and_tokenize_for_capture(
    tokenizer: Any,
    prompt: str,
    response: str,
    device: torch.device,
    *,
    role: str | None = None,
    model_type: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Render a ``[user, assistant]`` pair + tokenize, locating the last content token.

    The shared front half of :func:`_encode_and_capture_all`.  Conversational
    (4.0 / A2) capture: ``response`` is an assistant turn answering ``prompt``,
    rendered with no system turn and the standard assistant label.  ``role``
    (when set) substitutes a custom assistant-role label via
    :func:`saklas.core.role_templates.apply_with_role` for the persona-baselined
    fit; ``role=None`` is the swap-back default.  Returns ``(ids [1, T] on
    device, content_end)`` where ``content_end`` is the response's last
    non-special token — the canonical pooling position.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        # Disable thinking/reasoning mode for models that support it
        # (Qwen 3.5, QwQ, etc.) — thinking tokens would contaminate pooling.
        kwargs: dict[str, Any] = {}
        if "enable_thinking" in (getattr(tokenizer, "chat_template", "") or ""):
            kwargs["enable_thinking"] = False

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        if role is None:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs,
            )
        else:
            if model_type is None:
                raise ValueError(
                    "_render_and_tokenize_for_capture: role= requires model_type= "
                    "so the family's role-header registry entry can be looked up"
                )
            from saklas.core.role_templates import apply_with_role
            text = apply_with_role(
                tokenizer, messages,
                role=role, model_type=model_type,
                add_generation_prompt=False, tokenize=False, **kwargs,
            )
    else:
        # Base model (no chat template) — there are no turn roles to render and
        # A2 role-swap cannot apply; capture the bare prompt+response
        # continuation as raw text.
        text = f"{prompt}\n{response}"
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    if ids.numel() == 0:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.eos_token_id or 0
        ids = torch.tensor([[bos_id]])
    ids = ids.to(device)

    # Last non-special token — chat templates append trailing markers (Llama's
    # <|eot_id|>, Gemma's <end_of_turn>, Qwen's <|im_end|>) whose hidden states
    # are disconnected from content.  ``last_content_index`` is the canonical
    # walkback (skips ``all_special_ids`` + ``added_tokens_encoder``), shared by
    # every single-state readout so the pooling position is defined once.
    content_end = last_content_index(ids[0].tolist(), tokenizer)
    return ids, content_end


@functools.cache
def _load_neutral_prompts() -> list[str]:
    """Load neutral prompts, preferring a user override at ~/.saklas/neutral_statements.json."""
    from saklas.io.paths import neutral_statements_path
    user_path = neutral_statements_path()
    if user_path.exists():
        with open(user_path) as f:
            return json.load(f)
    with _resources.files("saklas.data").joinpath("neutral_statements.json").open() as f:
        return json.load(f)


@functools.cache
def _load_baseline_prompts() -> list[str]:
    """Load the shared A2 baseline user prompts.

    Prefers a user override at ``~/.saklas/baseline_prompts.json``, else the
    bundled ``saklas/data/baseline_prompts.json`` (64 affect-neutral prompts).
    Conversational capture pairs each node response with its prompt by
    ``response[i] -> prompt[i % len(prompts)]``.
    """
    from saklas.io.paths import baseline_prompts_path
    user_path = baseline_prompts_path()
    if user_path.exists():
        with open(user_path) as f:
            return json.load(f)
    with _resources.files("saklas.data").joinpath("baseline_prompts.json").open() as f:
        return json.load(f)


def _neutral_pairs() -> list[tuple[str, str]]:
    """The neutral baseline as ``(prompt, response)`` pairs for conversational capture.

    The neutral corpus (``neutral_statements.json``) is the model's organic,
    no-system / no-role responses to the shared baseline prompts; pairing is
    positional, ``response[i] -> baseline[i % k]`` — the same alignment a node
    corpus uses.  Raises if the corpus length is not a multiple of the prompt
    set (regenerate neutrals against the shared baseline).
    """
    responses = _load_neutral_prompts()
    baseline = _load_baseline_prompts()
    k = len(baseline)
    if k == 0:
        raise ValueError("no baseline prompts available for neutral capture")
    if len(responses) % k != 0:
        raise ValueError(
            f"neutral corpus ({len(responses)}) must be a multiple of the "
            f"baseline prompt set ({k}); regenerate neutrals against it"
        )
    return [(baseline[i % k], r) for i, r in enumerate(responses)]


def compute_layer_means(
    model: torch.nn.Module,
    tokenizer: Any,
    layers: torch.nn.ModuleList,
    device: torch.device | None = None,
) -> dict[int, torch.Tensor]:
    """Compute mean hidden state per layer over neutral prompts.

    Returns dict mapping layer_idx -> mean vector (dim,) in fp32.
    Used to center activations before probe cosine similarity,
    removing the baseline projection bias.
    """
    if device is None:
        device = next(model.parameters()).device
    assert device is not None  # device is always set by this point

    n_layers = len(layers)
    sums: dict[int, torch.Tensor] = {}

    _mps = device.type == "mps"

    pairs = _neutral_pairs()
    for prompt, response in pairs:
        per_layer = _encode_and_capture_all(
            model, tokenizer, prompt, response, layers, device,
        )
        if not sums:
            for idx in range(n_layers):
                sums[idx] = per_layer[idx].clone()
        else:
            for idx in range(n_layers):
                sums[idx] += per_layer[idx]
        del per_layer
        if _mps:
            torch.mps.empty_cache()

    n = len(pairs)
    return {idx: sums[idx] / n for idx in range(n_layers)}


def compute_neutral_activations(
    model: torch.nn.Module,
    tokenizer: Any,
    layers: torch.nn.ModuleList,
    device: torch.device | None = None,
) -> dict[int, torch.Tensor]:
    """Per-layer ``[N, D]`` stack across the 90 neutral prompts.

    Same forward-pass discipline as :func:`compute_layer_means` — last
    non-special-token pooling, fp32, MPS-friendly.  Returns one stacked
    tensor per layer (rows = prompts).  Used by cross-model alignment
    (:func:`saklas.io.alignment.fit_alignment`) which needs paired
    observations to fit Procrustes; the means alone (N=1) are degenerate
    for that fit.

    Storage cost: ~90 · n_layers · hidden_dim · 4B in fp32 (≈ 56MB on
    a 4096-dim, 80-layer model).  Callers persist this through
    :func:`saklas.io.alignment.load_or_compute_neutral_activations`.
    """
    if device is None:
        device = next(model.parameters()).device
    assert device is not None  # device is always set by this point

    n_layers = len(layers)
    rows: list[dict[int, torch.Tensor]] = []
    _mps = device.type == "mps"

    for prompt, response in _neutral_pairs():
        per_layer = _encode_and_capture_all(
            model, tokenizer, prompt, response, layers, device,
        )
        # Move each layer's vector to CPU before discarding the rest of
        # the captured dict — same MPS discipline as ``compute_layer_means``.
        rows.append({idx: per_layer[idx].detach().to("cpu") for idx in range(n_layers)})
        del per_layer
        if _mps:
            torch.mps.empty_cache()

    return {
        idx: torch.stack([row[idx] for row in rows])  # (N, D), fp32 on cpu
        for idx in range(n_layers)
    }


def compute_dls_axes(
    node_centroids: dict[int, torch.Tensor],
    bases: dict[int, torch.Tensor],
    layer_means: dict[int, torch.Tensor] | None,
) -> dict[int, set[int]]:
    """Per-axis Discriminative-Layer-Selection over **N node centroids**.

    The N-node generalization of the bipolar opposite-sign test (Dang & Ngo
    2026, Eq. 9) — the form a flat *subspace* of any node count uses.  Each
    layer carries a basis ``[R, D]`` (its ``LayerSubspace.basis`` rows) and
    ``node_centroids[L]`` is the ``[K, D]`` stack of that layer's per-node
    mean activations.  Every row ``d̂_r`` is kept iff the node projections
    relative to the neutral baseline, ``{(c_i − ν)·d̂_r : i}``, **straddle
    zero** (both a negative and a positive present) — the axis then
    discriminates node *position* across the baseline rather than encoding a
    common offset / intensity every node shares.

    **N=2 parity (the spine invariant).**  With
    ``node_centroids[L] = stack([μ_pos, μ_neg])`` the straddle
    ``min < 0 < max`` is exactly the bipolar opposite-sign product test, bit
    for bit — :func:`compute_dls_mask_per_axis` is the 2-node sugar over this
    core, and the scalar :func:`compute_dls_mask` the ``R = 1`` face of that.
    So a 2-node subspace reproduces today's vector DLS keep set exactly, and
    an N-node subspace (e.g. ``personas``) prunes per-axis the same way.

    Returns ``{layer: {kept axis indices}}``; a layer absent / mapped to an
    empty set is fully dropped.  ``layer_means`` ``None``/empty disables DLS
    (every axis of every layer kept).  A layer whose baseline ``ν`` is missing
    keeps every checkable axis conservatively (over-include rather than drop a
    real discriminative axis for missing data).  Degenerate (zero-norm) rows
    are dropped and excluded from the all-fail fallback; when no axis anywhere
    passes, falls back to the full checkable set with a one-time warning.

    Args:
        node_centroids: per-layer ``[K, D]`` stack of node mean activations.
        bases: per-layer ``[R, D]`` basis (rows are the candidate axes).
        layer_means: per-layer neutral baseline; ``None``/empty disables.
    """
    if layer_means is None or not layer_means:
        out: dict[int, set[int]] = {}
        for L in node_centroids:
            B = bases.get(L)
            r = int(B.shape[0]) if B is not None else 1
            out[L] = set(range(r))
        return out
    keep: dict[int, set[int]] = {}
    # ``checkable`` collects the (layer, axis) pairs we *attempted* the test
    # on — rows with a valid (non-degenerate) direction.  Rows skipped for
    # zero norm are excluded from the all-failed fallback.
    checkable: dict[int, set[int]] = {}
    any_pass = False
    for L in node_centroids:
        B = bases.get(L)
        if B is None:
            continue
        B32 = B.to(dtype=torch.float32, device="cpu").reshape(int(B.shape[0]), -1)
        mu_n = layer_means.get(L)
        mu_n_cpu = (
            mu_n.to(dtype=torch.float32, device="cpu").reshape(-1)
            if mu_n is not None
            else None
        )
        C = node_centroids[L].to(dtype=torch.float32, device="cpu")
        if C.ndim == 1:
            C = C.reshape(1, -1)
        C = C.reshape(int(C.shape[0]), -1)              # (K, D)
        layer_checkable: set[int] = set()
        layer_keep: set[int] = set()
        for r in range(int(B32.shape[0])):
            row = B32[r]
            row_norm = float(row.norm())
            if row_norm < 1e-12:
                continue  # degenerate axis — drop, exclude from fallback
            layer_checkable.add(r)
            if mu_n_cpu is None:
                # Baseline doesn't cover this layer.  Conservative: keep —
                # over-include rather than drop a real discriminative axis
                # for missing baseline data.
                layer_keep.add(r)
                continue
            projs = (C - mu_n_cpu) @ (row / row_norm)   # (K,) node projections
            # Straddle zero: both a negative and a positive projection ⇒ the
            # axis separates nodes across the baseline.  At K=2 this is the
            # bipolar ``proj_pos · proj_neg < 0`` opposite-sign test exactly.
            if float(projs.min()) < 0.0 and float(projs.max()) > 0.0:
                layer_keep.add(r)
        if layer_checkable:
            checkable[L] = layer_checkable
        if layer_keep:
            keep[L] = layer_keep
            any_pass = True
    if not any_pass:
        # Every checkable axis failed — probe degenerate on this model (the
        # diagnostics warning fires separately).  Keep the checkable set
        # rather than every layer; when nothing is checkable the fallback is
        # empty and the share-bake all-zero handler kicks in.
        warnings.warn(
            "DLS: no layers pass the discriminative check; falling back "
            "to keep-all-checkable.  Probe likely degenerate on this "
            "model — review the diagnostics warning above.",
            UserWarning,
            stacklevel=3,
        )
        return checkable
    return keep


def compute_dls_mask_per_axis(
    mu_pos: dict[int, torch.Tensor],
    mu_neg: dict[int, torch.Tensor],
    bases: dict[int, torch.Tensor],
    layer_means: dict[int, torch.Tensor] | None,
) -> dict[int, set[int]]:
    """Bipolar (2-node) per-axis DLS — sugar over :func:`compute_dls_axes`.

    Stacks each layer's ``[μ_pos, μ_neg]`` into the ``[2, D]`` node-centroid
    form and delegates to the N-node straddle core.  At ``K = 2`` the straddle
    ``min < 0 < max`` *is* the centered opposite-sign product test
    ``(μ_pos − ν)·d̂_r · (μ_neg − ν)·d̂_r < 0``, so this reproduces the
    historical bipolar keep set bit for bit; the scalar :func:`compute_dls_mask`
    is the ``R = 1`` face (a single basis row).  The consumer slices the
    effective basis to the kept rows at apply time, so DLS survives the
    vector→subspace fold without a separate scalar path.

    ``mu_neg`` must cover every layer of ``mu_pos`` (the caller builds them
    aligned).  ``layer_means`` ``None``/empty disables DLS; see
    :func:`compute_dls_axes` for the centering rationale, the missing-baseline
    conservative-keep, and the all-fail fallback.
    """
    node_centroids = {
        L: torch.stack([mu_pos[L].reshape(-1), mu_neg[L].reshape(-1)])
        for L in mu_pos
    }
    return compute_dls_axes(node_centroids, bases, layer_means)


def compute_dls_mask(
    mu_pos: dict[int, torch.Tensor],
    mu_neg: dict[int, torch.Tensor],
    directions: dict[int, torch.Tensor],
    layer_means: dict[int, torch.Tensor] | None,
) -> set[int]:
    """Discriminative-Layer-Selection keep set (scalar / single-direction).

    The R=1 face of :func:`compute_dls_mask_per_axis`: each layer's single
    ``directions[L]`` is a one-row basis, and a layer is kept iff that row
    passes the centered opposite-sign check (or is conservatively kept on a
    missing baseline).  Bit-identical to the historical implementation — it
    is now a thin wrapper so the vector and subspace paths share one DLS
    kernel.

    Layers where both pos- and neg-class means project to the same side of
    the neutral baseline along ``d̂`` are non-discriminative (they encode
    concept *intensity*, not *polarity*); dropping them concentrates share
    on layers that genuinely carry the contrast.  See
    :func:`compute_dls_mask_per_axis` for the centering rationale and the
    all-fail fallback.

    Args:
        mu_pos: per-layer positive-class mean, fp32 ``[D]`` per layer.
        mu_neg: same for the negative class (same layer set as ``mu_pos``).
        directions: per-layer unsigned direction (``unit(μ_pos − μ_neg)``
            for DiM, principal component for PCA).  Magnitude is irrelevant
            (only projection sign matters); unit input recommended.
        layer_means: per-layer neutral baseline; ``None`` disables DLS
            (returns every layer).

    Returns:
        ``set[int]`` of layers passing the discriminative check; the
        keep-all-checkable fallback fires (with a warning) if every layer
        fails.
    """
    bases = {L: d.reshape(1, -1) for L, d in directions.items()}
    per_axis = compute_dls_mask_per_axis(mu_pos, mu_neg, bases, layer_means)
    return {L for L, axes in per_axis.items() if axes}


def _fold_centroids_to_affine_manifold(
    name: str,
    mean_pos_per_layer: dict[int, torch.Tensor],
    mean_neg_per_layer: dict[int, torch.Tensor],
    *,
    pos_label: str,
    neg_label: str,
    whitener: "Any | None" = None,
    layer_means: dict[int, torch.Tensor] | None = None,
    dls: bool = True,
    feature_space: str = "raw",
) -> "Any":  # -> saklas.core.manifold.Manifold
    """Fold per-layer pos/neg centroids into an affine ``R = 1`` manifold.

    The folded-vector representation (saklas 4.0 §5): a steering vector *is*
    the ``K = 2`` case of a flat affine subspace — a line through the two pole
    centroids that fills its 1-D span (``H_n ≡ 0``).  Per retained layer
    ``L``, :func:`~saklas.core.manifold.fit_affine_subspace` (pos = node 0,
    so the basis orients ``+δ̂``) gives:

    * ``basis`` = unit ``δ̂_L`` where ``δ_L = mean_pos_L − mean_neg_L`` — the
      **raw** difference-of-means direction (PCA@2 ≡ DiM exactly).  This is the
      legacy fold path (test-only, retiring in 6b); the live 2-node-vector read
      whitens the basis via extraction's ``fit_affine_subspace``.  Only the
      share is whitened here (the ``covers_all`` gate below).
    * ``mean`` = ``P_basis(ν_L)`` — the neutral mean projected into the span
      (neutral-anchored frame, §5); falls back to the pole midpoint ``μ`` when
      no neutral baseline is supplied.
    * ``LayerSubspace.node_coords`` = the **real**, neutral-anchored pole
      coords ``(c_± − ν_L)·δ̂_L`` ``(2, 1)``.  The per-layer magnitude lives
      *here* now (``coord_+ − coord_− = ‖δ_L‖``), not in the share — neutral →
      coord 0, so a pole sits at distance ∝ ‖δ_L‖ from the origin.

    **Share — the per-layer budget (§5).**  ``mahalanobis_share[L]`` is the
    μ-centered whitened spread :func:`~saklas.core.manifold.subspace_share`
    (``‖δ_L‖_M / √2`` at K=2, whitened) when the whitener covers every
    retained layer, else the Euclidean ``‖δ_L‖₂ / √2``.  The ``√2`` is a
    constant that **cancels under the apply-time ``Σ_L share_L = 1``
    normalization**, so ``hooks._manifold_layer_shares`` reproduces today's
    DiM hook profile exactly (the normalized share is the DiM one — coords
    carry the position, share carries the budget).

    **Origin** is implicitly ``0`` (neutral → coord 0 by the neutral anchor),
    so no origin is stored — the flat ``!`` ablation target is coord 0 (§2).

    **DLS** is per-axis at ``R = 1`` (keep-or-drop the layer): the basis row
    is kept iff the pole projections straddle the neutral baseline —
    bit-identical to the historical scalar test, dropped layers absent from
    the returned manifold.

    Returns a :class:`~saklas.core.manifold.Manifold`; the caller persists it.
    ``share_metric`` is recorded on ``manifold.metadata`` for that save.
    """
    from saklas.core.manifold import (
        CustomDomain, LayerSubspace, Manifold, fit_affine_subspace,
        subspace_share,
    )

    shared = sorted(set(mean_pos_per_layer) & set(mean_neg_per_layer))

    # All-or-nothing whitener gate (parity with the DiM bake): Mahalanobis
    # share only when the whitener covers every candidate layer, else
    # Euclidean for all.
    maha_w = (
        whitener
        if whitener is not None and whitener.covers_all(shared)
        else None
    )

    # Per-layer affine fit over [pos, neg] (pos = node 0 ⇒ +δ̂ orientation).
    # Degenerate pairs (zero diff — no direction to steer) drop out first so
    # the K=2 fit never sees a zero scatter.  This is the legacy fold path
    # (test-only, retiring in 6b) and keeps the **raw** δ̂ basis (PCA@2 ≡ DiM)
    # — the live 2-node-vector read goes through extraction's whitened
    # ``fit_affine_subspace``; only the share is whitened here.
    fits: dict[int, tuple["LayerSubspace", torch.Tensor, torch.Tensor]] = {}
    for idx in shared:
        mp = mean_pos_per_layer[idx].to(torch.float32).reshape(-1)
        mn = mean_neg_per_layer[idx].to(torch.float32).reshape(-1)
        if float((mp - mn).norm()) <= 1e-12:
            continue
        cent = torch.stack([mp, mn])               # (2, D); node 0 = pos
        nu = (
            layer_means[idx].to(torch.float32).reshape(-1)
            if layer_means is not None and idx in layer_means
            else None
        )
        sub, mu_coords, _ev = fit_affine_subspace(
            cent, neutral_mean=nu, orient_to=0,
        )
        fits[idx] = (sub, mu_coords, cent)

    # Per-axis DLS at R=1 (collapses to the scalar keep-or-drop).
    if dls:
        node_centroids = {idx: fits[idx][2] for idx in fits}
        bases = {idx: fits[idx][0].basis for idx in fits}
        per_axis = compute_dls_axes(node_centroids, bases, layer_means)
        kept = [idx for idx in fits if per_axis.get(idx)]
    else:
        kept = list(fits)

    layers: dict[int, "LayerSubspace"] = {}
    mahalanobis_share: dict[int, float] = {}
    for idx in kept:
        sub, mu_coords, _cent = fits[idx]
        layers[idx] = sub
        # Per-layer budget = μ-centered whitened/Euclidean spread.  No origin
        # store — affine origin is coord 0 (§2).
        mahalanobis_share[idx] = subspace_share(
            mu_coords, sub.basis, whitener=maha_w, layer=idx,
        )

    # Shared display layout: pos at +1, neg at −1 on a 1-D CustomDomain (the
    # label/display frame; the real per-layer steer coords live on each
    # ``LayerSubspace.node_coords``).
    node_coords = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    manifold = Manifold(
        name=name,
        domain=CustomDomain(1),
        node_labels=[pos_label, neg_label],
        node_coords=node_coords,
        layers=layers,
        feature_space=feature_space,
        mahalanobis_share=mahalanobis_share,
    )
    manifold.metadata["share_metric"] = (
        "mahalanobis" if maha_w is not None else "euclidean"
    )
    return manifold


def fold_directions_to_subspace(
    name: str,
    directions: dict[int, torch.Tensor],
    neutral_means: dict[int, torch.Tensor] | None,
    *,
    whitener: "Any | None" = None,
    label: str = "+",
    feature_space: str = "raw",
) -> "Any":  # -> saklas.core.manifold.Manifold
    """Fold an arbitrary per-layer *direction* into a neutral-anchored affine
    ``R = 1`` manifold (a one-pole ray).

    The monopolar sibling of :func:`_fold_centroids_to_affine_manifold`: where
    a bipolar concept folds a line through two pole centroids, a derived
    direction (a ``merge`` linear-combination, a `~`/`|` projection of one
    concept onto another) has no pole pair — just a direction to push along.
    So the subspace is neutral-anchored at ``mean = P_basis(ν_L) = (ν_L·d̂)d̂``
    (the neutral mean projected into the 1-D span, §5) with a single ``+`` pole
    node; steering ``along`` toward the pole pushes the running activation's
    ``d̂`` component away from neutral, and ``!`` (``along``-to-coord-0) slides
    it back to neutral, i.e. ablates the direction.

    Per layer: ``basis = d̂_L``; ``LayerSubspace.node_coords = [[‖d_L‖]]`` —
    the real coord of the pole, a step of ``‖d_L‖`` along ``d̂`` from the
    neutral origin (coord 0).  ``share = ‖d_L‖_M`` (whitened) / ``‖d_L‖₂``
    (Euclidean) — the direction magnitude itself is the budget (a single
    direction has no node cloud to take a μ-centered spread over, so this is
    *not* :func:`~saklas.core.manifold.subspace_share`'s √2-scaled form; the
    apply-time normalization handles the cross-layer scale either way).  No
    DLS — a derived direction has no polarity for the straddle test; the
    caller's layer set folds verbatim.  No origin store (coord 0, §2).
    ``neutral_means=None`` (CPU stubs) anchors at coord 0.
    """
    from saklas.core.manifold import (
        CustomDomain, LayerSubspace, Manifold,
    )

    present = sorted(directions)
    maha_w = (
        whitener
        if whitener is not None and whitener.covers_all(present)
        else None
    )

    layers: dict[int, "LayerSubspace"] = {}
    mahalanobis_share: dict[int, float] = {}
    for idx in present:
        d = directions[idx].to(torch.float32).reshape(-1)
        norm = float(d.norm())
        if norm <= 1e-12:
            continue
        basis = (d / norm).reshape(1, -1)          # (1, D) unit d̂
        nu = (
            neutral_means[idx].to(torch.float32).reshape(-1)
            if neutral_means is not None and idx in neutral_means
            else None
        )
        # Neutral-anchored: mean = P_basis(ν) (off-span part of ν dropped);
        # the pole sits at the real coord ‖d‖ along d̂ from the origin.
        mean = (nu @ basis.T) @ basis if nu is not None else torch.zeros_like(d)
        node_coords_L = torch.tensor([[norm]], dtype=torch.float32)
        layers[idx] = LayerSubspace.affine(mean, basis, node_coords=node_coords_L)
        mahalanobis_share[idx] = (
            float(maha_w.mahalanobis_norm(idx, d))
            if maha_w is not None else norm
        )

    # Shared display layout: a single ``+`` pole at coord 1 (the real per-layer
    # pole distance ‖d_L‖ lives on each ``LayerSubspace.node_coords``).
    node_coords = torch.tensor([[1.0]], dtype=torch.float32)
    manifold = Manifold(
        name=name,
        domain=CustomDomain(1),
        node_labels=[label],
        node_coords=node_coords,
        layers=layers,
        feature_space=feature_space,
        mahalanobis_share=mahalanobis_share,
    )
    manifold.metadata["share_metric"] = (
        "mahalanobis" if maha_w is not None else "euclidean"
    )
    return manifold


def folded_vector_directions(manifold: "Any") -> dict[int, torch.Tensor]:
    """Baked-direction view of a folded (affine ``R = 1``) vector manifold.

    Returns ``{L: δ̂_L · share_L}`` — the steering-vector-equivalent baked
    directions, proportional *per layer* to what
    :func:`extract_difference_of_means` bakes (``‖baked_L‖ ∝ ‖δ_L‖_M``).  The
    global per-concept scale differs (the folded share is the un-normalized
    ``‖δ_L‖_M``, not the ``ref_norm``-weighted normalized share today's bake
    folds in), so this view is *exact* for scale-invariant ops — per-layer and
    aggregate cosine, ``vector compare``/``why`` — and merely
    proportional for cross-concept magnitude (``merge`` / GGUF export, which
    migrate to read the folded artifact natively).

    Lets the unified folded Manifold back the legacy direction-math surface
    (``Profile``-returning ``extract()``, ``vector compare``/``why``) without a
    second *stored* representation — the concept's only on-disk artifact stays
    the folded Manifold; this is a downstream in-memory view.

    Raises :class:`ValueError` on a curved or multi-dim manifold — the
    single-direction view is only meaningful for a folded ``R = 1`` vector.
    """
    out: dict[int, torch.Tensor] = {}
    share_map = getattr(manifold, "mahalanobis_share", None) or {}
    for idx, sub in manifold.layers.items():
        if not sub.is_affine or sub.rank != 1:
            raise ValueError(
                "folded_vector_directions requires affine R=1 layers; "
                f"layer {idx} is rank {sub.rank}, affine={sub.is_affine}"
            )
        share = float(share_map.get(idx, 1.0))
        out[idx] = sub.basis.reshape(-1).to(torch.float32) * share
    return out


def save_profile(
    profile: dict[int, torch.Tensor],
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Save a baked vector profile as .safetensors with a slim .json sidecar.

    ``metadata`` must contain at minimum:
        method            - str, e.g. "difference_of_means" / "merge" / "layer_means"

    Optional keys honored:
        statements_sha256 - str, hash of source statements at extraction time
        components        - dict, merge provenance (method="merge" only)
        diagnostics       - dict[int, dict[str, float]], per-layer probe-quality
                            metrics (see ``_compute_layer_diagnostics``).
                            Persisted as ``diagnostics_by_layer`` on the
                            sidecar with stringified layer keys.

    The safetensors file contains keys ``"layer_{i}"`` for each active layer.
    Tensors are already baked (share pre-multiplied into magnitude) — the
    sidecar carries only method/saklas_version plus the optional fields above.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # fp32 write invariant: every saklas safetensor writer enforces fp32
    # on disk (matches gguf_io.py's ``.to(dtype=torch.float32)``), so the
    # stored dtype is a guarantee rather than a coincidence of the caller.
    tensors = {
        f"layer_{idx}": vec.to(dtype=torch.float32).contiguous().cpu()
        for idx, vec in profile.items()
    }
    save_file(tensors, str(path))

    from saklas import __version__ as _saklas_version
    from saklas.io.packs import PACK_FORMAT_VERSION
    sidecar: dict[str, Any] = {
        "format_version": PACK_FORMAT_VERSION,
        "method": metadata.get("method", "difference_of_means"),
        "saklas_version": _saklas_version,
    }
    if "statements_sha256" in metadata:
        sidecar["statements_sha256"] = metadata["statements_sha256"]
    if "components" in metadata:
        sidecar["components"] = metadata["components"]
    # v2.1: bake method records which scoring metric drove share allocation
    # (``"euclidean"`` = legacy ``||m||_2 / r``; ``"mahalanobis"`` =
    # ``||m||_M / r`` via per-layer activation covariance).  Loaders read
    # this only for diagnostics; the runtime hook reads tensor magnitudes
    # regardless of bake flavor.  Default ``"euclidean"`` is back-compat
    # for tensors written before the bake field existed.
    if "bake" in metadata:
        sidecar["bake"] = metadata["bake"]
    # SAE provenance — present only when extraction used an SAE backend.
    for key in ("sae_release", "sae_revision", "sae_ids_by_layer"):
        if key in metadata:
            sidecar[key] = metadata[key]
    # Transfer provenance — present only on transferred profiles
    # (method="procrustes_transfer").  ``alignment_map_hash`` pins the
    # specific Procrustes fit; ``transfer_quality_estimate`` is the
    # median per-layer R² across shared layers.
    for key in (
        "source_model_id",
        "alignment_map_hash",
        "transfer_quality_estimate",
    ):
        if key in metadata:
            sidecar[key] = metadata[key]
    # Diagnostics: stringify layer keys so the JSON round-trips through
    # standard parsers (JSON object keys must be strings).  Reader inverts.
    diagnostics = metadata.get("diagnostics")
    if diagnostics:
        sidecar["diagnostics_by_layer"] = {
            str(layer): {k: float(v) for k, v in metrics.items()}
            for layer, metrics in diagnostics.items()
        }

    from saklas.io.atomic import write_json_atomic
    meta_path = path.with_suffix(".json")
    write_json_atomic(meta_path, sidecar)

    log.info("Saved profile (%d layers) to %s", len(profile), path)


def load_profile(path: str | Path) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Load a baked vector profile and its metadata.

    Dispatches on file extension: ``.safetensors`` reads the companion
    ``.json`` sidecar; ``.gguf`` reads the control-vector metadata embedded
    in the GGUF header (see :mod:`saklas.gguf_io`). Both paths yield the
    same ``(profile, metadata)`` shape — callers don't need to branch.

    Returns:
        (profile dict mapping layer_idx -> baked vector, metadata dict)
    """
    path = Path(path)
    if path.suffix == ".gguf":
        from saklas.io.gguf_io import read_gguf_profile
        return read_gguf_profile(path)

    tensors = load_file(str(path))
    meta_path = path.with_suffix(".json")
    with open(meta_path) as f:
        metadata = json.load(f)

    from saklas.io.packs import PACK_FORMAT_VERSION
    from saklas.core.profile import ProfileError
    fmt_ver = metadata.get("format_version", 1)
    if not isinstance(fmt_ver, int) or fmt_ver < PACK_FORMAT_VERSION:
        raise ProfileError(
            f"pack format is from saklas < 2.0 "
            f"(sidecar {meta_path} format_version={fmt_ver!r}, "
            f"need >= {PACK_FORMAT_VERSION}); "
            f"run `python scripts/upgrade_packs.py {path.parent}` to migrate"
        )

    profile = {int(key.split("_", 1)[1]): tensor for key, tensor in tensors.items()}

    # Invert the layer-key stringification done at save time so diagnostics
    # are addressable by ``int`` consistently with the profile dict.
    raw_diag = metadata.get("diagnostics_by_layer")
    if isinstance(raw_diag, dict) and raw_diag:
        # Malformed diagnostics leave the raw dict in place; downstream readers
        # can decide whether to fall back. The tensors themselves are still valid.
        with suppress(TypeError, ValueError):
            metadata["diagnostics"] = {
                int(layer): dict(metrics)
                for layer, metrics in raw_diag.items()
            }

    return profile, metadata


def project_profile(
    base: dict[int, torch.Tensor],
    onto: dict[int, torch.Tensor],
    operator: str,
    *,
    whitener: "Any | None" = None,
) -> dict[int, torch.Tensor]:
    """Per-layer projection of ``base`` against ``onto``.

    Default (Euclidean), per shared layer (fp32)::

        proj = (dot(base, onto) / dot(onto, onto)) * onto

    With ``whitener`` (a :class:`saklas.core.mahalanobis.LayerWhitener`),
    switches to LEACE-style projection in the Mahalanobis metric::

        coef = <base, onto>_M / <onto, onto>_M
        proj = coef * onto                # direction is ``onto``; metric is M

    The output direction is still ``onto``, but the *amount* removed is
    the component along ``onto`` measured in the whitened space.  For
    operator ``"|"``, this is the closed-form LEACE projector for a
    single direction (Belrose et al. 2023, arXiv 2306.03819) — provably
    erases linearly-decodable information along ``onto`` from ``base``
    with minimum collateral damage.  Reduces to plain Gram-Schmidt when
    ``Σ = I``.

    Operator semantics (both metrics):

    - ``operator == "~"``   returns ``proj``       (component of base aligned with onto).
    - ``operator == "|"`` returns ``base - proj`` (component of base orthogonal to onto).

    Layers in ``base`` without a matching layer in ``onto``: for ``"|"``
    they pass through unchanged (nothing to project away); for ``"~"`` they
    are dropped (projection onto an absent direction is undefined).

    Near-zero ``||onto|| < 1e-12`` layers are treated the same way: ``"|"``
    passes base through unchanged, ``"~"`` drops the layer.

    The metric is decided **all-or-nothing** via
    :meth:`LayerWhitener.covers_all` over the projected layer set
    (``base ∩ onto``): LEACE only when the whitener covers *every*
    projected layer, else plain Gram-Schmidt for *all* layers — never a
    per-layer mix of LEACE and Euclidean.  Mixing would compare
    incommensurable magnitudes (``‖·‖_M`` carries a ``1/√λ_L`` factor that
    ``‖·‖_2`` doesn't, and it doesn't cancel across layers).
    Result tensors are cast back to the source dtype of ``base``.

    The returned dict shape matches :func:`extract_difference_of_means` so
    it plugs into ``SteeringManager.add_vector`` without adaptation.
    """
    if operator not in ("~", "|"):
        raise ValueError(f"unknown projection operator: {operator!r}")
    # All-or-nothing metric gate over the layers that actually get
    # projected (``base ∩ onto``): LEACE on every covered layer or
    # Gram-Schmidt on all of them (see ``covers_all`` — incommensurable
    # per-layer scales forbid a mix).
    projected_layers = sorted(set(base) & set(onto))
    # Bind the narrowed whitener once (mirrors ``extraction.py``'s
    # ``maha_whitener`` idiom) so the metric decision is made a single time.
    maha = whitener if (whitener is not None and whitener.covers_all(projected_layers)) else None
    out: dict[int, torch.Tensor] = {}
    for layer, base_t in base.items():
        onto_t = onto.get(layer)
        if onto_t is None:
            if operator == "|":
                out[layer] = base_t
            continue
        # LEACE branch: whitener covers the full projected set.
        if maha is not None:
            projected = maha.leace_project(layer, base_t, onto_t, operator)
            # Drop the layer for ``~`` when ``onto`` is degenerate under
            # the Mahalanobis metric — leace_project returns a zero
            # tensor in that case, mirroring the Euclidean drop rule.
            if operator == "~" and torch.all(projected == 0):
                continue
            out[layer] = projected
            continue
        # Euclidean Gram-Schmidt (no whitener, or partial coverage).
        a_f = base_t.to(dtype=torch.float32)
        b_f = onto_t.to(dtype=torch.float32)
        b_dot = torch.dot(b_f, b_f).item()
        if b_dot < 1e-12:
            if operator == "|":
                out[layer] = base_t
            continue
        proj = (torch.dot(a_f, b_f) / b_dot) * b_f
        if operator == "~":
            out[layer] = proj.to(dtype=base_t.dtype)
        else:
            out[layer] = (a_f - proj).to(dtype=base_t.dtype)
    if not out:
        raise ValueError(
            f"project_profile: no layers produced for operator {operator!r} "
            f"(base layers: {sorted(base.keys())}, "
            f"onto layers: {sorted(onto.keys())})"
        )
    return out
