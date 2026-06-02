"""Extraction, saving, and loading of activation steering/probe vectors."""

from __future__ import annotations

import functools
import json
import logging
import warnings
from collections.abc import Sequence
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
    text: str,
    layers: torch.nn.ModuleList,
    device: torch.device,
    *,
    role: str | None = None,
    model_type: str | None = None,
):
    """Tokenize text, run forward pass, return last-content-token hidden state per layer in fp32.

    For instruction-tuned models (those with a chat template), wraps the text
    as an assistant response so the extraction happens in the model's actual
    generation regime.  Base models get the raw string.

    Pools from the last non-special token — chat templates append trailing
    markers (Llama's <|eot_id|>, Gemma's <end_of_turn>, Qwen's <|im_end|>)
    whose hidden states are disconnected from content.  The last content
    token's hidden state is itself an attention-weighted summary of prior
    positions and is exactly what the model uses for next-token prediction.

    ``role`` (optional): substitute a custom assistant-role label into the
    chat template via :func:`saklas.core.role_templates.apply_with_role` —
    the extraction baseline shifts from "model speaking as assistant" to
    "model speaking as <role>".  Requires ``model_type``.  ``role=None``
    keeps the standard assistant-baseline path with zero overhead.

    Returns:
        dict mapping layer_idx -> pooled vector (dim,) in fp32.
    """
    ids, content_end = _render_and_tokenize_for_capture(
        tokenizer, text, device, role=role, model_type=model_type,
    )
    hidden_per_layer = _capture_all_hidden_states(model, layers, ids)
    return {
        idx: h[0, min(content_end, h.shape[1] - 1)].float()
        for idx, h in hidden_per_layer.items()
    }


def encode_and_capture_stack(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    layers: torch.nn.ModuleList,
    device: torch.device,
    *,
    role: str | None = None,
    model_type: str | None = None,
) -> tuple[dict[int, torch.Tensor], int]:
    """Capture the full ``[T, D]`` per-layer hidden stack over *text*.

    The stack-returning companion to :func:`_encode_and_capture_all`:
    same chat-template rendering, tokenization, and last-content-token
    discipline, but it returns the whole sequence stack per layer plus the
    ``agg_index`` of the last content token rather than collapsing to a
    single pooled state.  Used by :meth:`ManifoldMonitor.measure` (one-shot
    manifold text scoring), where ``score_aggregate`` pools the
    ``agg_index`` row itself.

    Returns ``({layer_idx: [T, D] fp32}, agg_index)``.
    """
    ids, content_end = _render_and_tokenize_for_capture(
        tokenizer, text, device, role=role, model_type=model_type,
    )
    hidden_per_layer = _capture_all_hidden_states(model, layers, ids)
    stacks = {
        idx: h[0].float()  # [T, D]
        for idx, h in hidden_per_layer.items()
    }
    # content_end is already the last-content index in this sequence;
    # clamp defensively against the captured stack length.
    seq_len = next(iter(stacks.values())).shape[0] if stacks else 0
    agg_index = min(content_end, seq_len - 1) if seq_len else 0
    return stacks, agg_index


def _render_and_tokenize_for_capture(
    tokenizer: Any,
    text: str,
    device: torch.device,
    *,
    role: str | None = None,
    model_type: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Render + tokenize *text* and locate the last content token.

    The shared front half of :func:`_encode_and_capture_all` /
    :func:`encode_and_capture_stack` — chat-template wrapping (with the
    optional role substitution and the template-overhead fallback) plus
    the canonical last-content-token walkback.  Returns ``(ids [1, T] on
    device, content_end)`` so both pooling shapes share one definition of
    how text becomes model input and where "content" ends.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        # Disable thinking/reasoning mode for models that support it
        # (Qwen 3.5, QwQ, etc.) — thinking tokens would contaminate pooling.
        kwargs: dict[str, Any] = {}
        if "enable_thinking" in (getattr(tokenizer, "chat_template", "") or ""):
            kwargs["enable_thinking"] = False

        def _render(msgs: list[dict[str, str]]) -> str:
            if role is None:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False, **kwargs,
                )
            if model_type is None:
                raise ValueError(
                    "_encode_and_capture_all: role= requires model_type= "
                    "so the family's role-header registry entry can be "
                    "looked up"
                )
            from saklas.core.role_templates import apply_with_role
            return apply_with_role(
                tokenizer, msgs,
                role=role, model_type=model_type,
                add_generation_prompt=False,
                tokenize=False,
                **kwargs,
            )

        messages = [{"role": "assistant", "content": text}]
        try:
            text = _render(messages)
        except Exception:
            # Some chat templates require a user turn before assistant.
            # The filler must be semantically empty — "." triggers
            # model-specific greeting/help responses whose template
            # tokens contaminate pooling.
            messages = [
                {"role": "user", "content": "Continue:"},
                {"role": "assistant", "content": text},
            ]
            text = _render(messages)
        # Some chat templates inject a large system prompt (e.g.
        # Ministral adds ~500 tokens).  For contrastive extraction the
        # overhead cancels in the diff but wastes memory on every
        # forward pass.  Fall back to raw tokenization when excessive.
        # The template-overhead probe is role-agnostic — splicing a
        # different role label can't shift the overhead by more than a
        # few characters, and the cache key intentionally ignores role
        # so the probe runs once per tokenizer.
        overhead = _chat_template_overhead(tokenizer, kwargs)
        if overhead > _MAX_TEMPLATE_OVERHEAD:
            text = messages[-1]["content"]  # use raw text
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    if ids.numel() == 0:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.eos_token_id or 0
        ids = torch.tensor([[bos_id]])
    ids = ids.to(device)

    # Find the last non-template-token position.  Chat templates append
    # trailing markers like Llama's <|eot_id|>, Gemma's <end_of_turn>,
    # Qwen's <|im_end|> — pooling from those positions yields degenerate
    # signals disconnected from the content.  Some tokenizers don't
    # promote chat boundary tokens to ``all_special_ids`` (talkie's
    # ``<|user|>``/``<|end|>``/``<|assistant|>`` are added tokens but
    # not "special"), so we also skip everything in
    # ``added_tokens_encoder``.  Without this, extraction pools at the
    # structural turn marker — talkie's outlier channels then dominate
    # the captured ref_norm, baking 100×-too-large probe magnitudes
    # that produce gibberish at any nonzero alpha.
    skip_ids = special_token_ids(tokenizer)
    content_end = ids.shape[1] - 1
    if skip_ids:
        id_list = ids[0].tolist()
        while content_end > 0 and id_list[content_end] in skip_ids:
            content_end -= 1

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

    prompts = _load_neutral_prompts()
    for text in prompts:
        per_layer = _encode_and_capture_all(model, tokenizer, text, layers, device)
        if not sums:
            for idx in range(n_layers):
                sums[idx] = per_layer[idx].clone()
        else:
            for idx in range(n_layers):
                sums[idx] += per_layer[idx]
        del per_layer
        if _mps:
            torch.mps.empty_cache()

    n = len(prompts)
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

    for text in _load_neutral_prompts():
        per_layer = _encode_and_capture_all(model, tokenizer, text, layers, device)
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


def _compute_layer_diagnostics(
    diff_matrix: torch.Tensor,
    principal_direction: torch.Tensor,
    evr: float,
) -> dict[str, float]:
    """Compute per-layer probe-quality metrics from contrastive diffs.

    All inputs in fp32.  ``diff_matrix`` is ``(N, dim)`` of pos-neg pair
    diffs; ``principal_direction`` is the first PC ``(dim,)`` (unsigned,
    pre-orientation); ``evr`` is the explained-variance ratio computed at
    the SVD site.

    Returns a small dict with four scalars:

    * ``evr`` — passes through the input.  Captures how concentrated the
      contrastive signal is along its principal direction; values near 1.0
      with low intra-pair variance indicate one-sided / repetitive pair sets.
    * ``intra_pair_variance_mean`` / ``intra_pair_variance_std`` — stats over
      ``||diff_i||`` across pairs.  Mean near zero with EVR near 1.0 is the
      "all pairs identical" pathology.
    * ``inter_pair_alignment`` — mean off-diagonal absolute cosine across
      pairs, batched as ``D̂ @ D̂^T``.  Low values mean pairs disagree on
      direction; the principal component still emerges from SVD but is
      weakly grounded.
    * ``diff_principal_projection`` — mean of ``|cos(diff_i, v)|`` across
      pairs.  How much of each pair's diff lives along the principal
      direction; complements ``inter_pair_alignment`` (the former measures
      pairs vs each other, the latter pairs vs the chosen direction).

    Cost is O(N·d + N²) per layer, dominated by the ``D @ D.T`` for
    ``inter_pair_alignment`` — at typical N=45, d=4096 this is ~50µs on
    CPU.  Negligible against the SVD it follows.
    """
    n_pairs = diff_matrix.shape[0]
    if n_pairs < 2:
        # Single-pair: most metrics degenerate.  Return minimal info so
        # callers can still distinguish "computed but degenerate" from
        # "not computed at all".
        diff_norm = float(diff_matrix.norm(dim=-1)[0].item())
        return {
            "evr": float(evr),
            "intra_pair_variance_mean": diff_norm,
            "intra_pair_variance_std": 0.0,
            "inter_pair_alignment": 1.0,
            "diff_principal_projection": 1.0,
        }

    diff_norms = diff_matrix.norm(dim=-1)  # (N,)
    intra_mean = float(diff_norms.mean().item())
    intra_std = float(diff_norms.std().item())

    # Unit-normalize diffs in fp32; clamp avoids NaN on a zero diff.
    unit_diffs = diff_matrix / diff_norms.clamp(min=1e-12).unsqueeze(-1)

    # Inter-pair alignment: mean |cos| of off-diagonal pairs.
    # D̂ @ D̂^T is symmetric with 1.0 on the diagonal; we want the mean
    # absolute value of the off-diagonal entries.
    cos_matrix = unit_diffs @ unit_diffs.transpose(0, 1)  # (N, N)
    abs_cos = cos_matrix.abs()
    n = abs_cos.shape[0]
    # Subtract the diagonal (self-cosine, always 1.0) and average the rest.
    off_diag_sum = abs_cos.sum() - abs_cos.diagonal().sum()
    inter_alignment = float((off_diag_sum / max(n * (n - 1), 1)).item())

    # Diff-to-PC projection: mean |cos(diff_i, v)|.
    v_norm = principal_direction.norm().clamp(min=1e-12)
    v_unit = principal_direction / v_norm
    proj_cos = (unit_diffs @ v_unit).abs()  # (N,)
    diff_pc_proj = float(proj_cos.mean().item())

    return {
        "evr": float(evr),
        "intra_pair_variance_mean": intra_mean,
        "intra_pair_variance_std": intra_std,
        "inter_pair_alignment": inter_alignment,
        "diff_principal_projection": diff_pc_proj,
    }


def _emit_diagnostics_warning(
    diagnostics: dict[int, dict[str, float]],
    *,
    concept_label: str | None = None,
) -> None:
    """Soft-warn when the median across layers looks degenerate.

    Catches one-sided / repetitive pair sets — high EVR (one direction
    explains nearly all variance) combined with near-zero intra-pair
    variance (all pos/neg pairs end up at the same activation point).
    The extracted profile is still usable; the warning is
    informational, not a block.

    The diagnostics themselves (``evr``, ``intra_pair_variance_mean``,
    ``inter_pair_alignment``, ``diff_principal_projection``) are
    always computed and persisted to the sidecar regardless of whether
    this warning fires — use ``saklas vector why <concept>`` to
    inspect them.
    """
    if not diagnostics:
        return

    evrs = [d["evr"] for d in diagnostics.values() if "evr" in d]
    intras = [
        d["intra_pair_variance_mean"]
        for d in diagnostics.values()
        if "intra_pair_variance_mean" in d
    ]
    if not evrs:
        return

    med_evr = median_or_zero(evrs)
    med_intra = median_or_zero(intras) if intras else float("inf")

    label = concept_label or "probe"
    if med_evr > _DIAG_DEGENERATE_EVR and med_intra < _DIAG_DEGENERATE_INTRA_VAR:
        warnings.warn(
            f"{label}: probe likely one-sided "
            f"(median EVR={med_evr:.2f}, intra-pair variance={med_intra:.4f}); "
            f"contrastive pairs may be too similar. Diversify the negative "
            f"pole and re-extract for a stronger direction.",
            UserWarning,
            stacklevel=3,
        )


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


def _capture_diffs_for_pairs(
    model: torch.nn.Module,
    tokenizer: Any,
    pairs: list[dict[str, str]],
    layers: torch.nn.ModuleList,
    device: torch.device,
    *,
    sae: "SaeBackend | None" = None,
    role: str | None = None,
    model_type: str | None = None,
) -> tuple[
    int,
    dict[int, list[torch.Tensor]],
    dict[int, list[torch.Tensor]],
    dict[int, list[torch.Tensor]],
    list[float],
    set[int] | None,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    """Run the contrastive forward-pass capture loop.

    Shared by the PCA and DiM extractors — both consume the same set of
    per-layer diffs, raw activation norm sums, per-layer pos/neg means
    (for centered DLS), and (when an SAE is wired) per-layer pos/neg
    activation stacks.  Only the post-capture per-layer direction
    computation differs between methods.

    SAE coverage is enforced here so callers don't repeat the check; an
    empty intersection raises :class:`SaeCoverageError` before the forward
    loop burns time.

    Per-layer pos/neg running sums are tracked at O(D) per layer
    regardless of N_pairs (cheap; no per-pair tensor list to manage)
    and converted to means on return.  These feed
    :func:`compute_dls_mask` for the discriminative-layer check; SAE
    paths get the per-pair pos/neg stacks too via ``pos_per_layer`` /
    ``neg_per_layer`` so feature-space encoding doesn't need a second
    pass.

    Returns:
        ``(n_layers, diffs_per_layer, pos_per_layer, neg_per_layer,
        norm_sums_cpu, sae_layer_set, mean_pos_per_layer,
        mean_neg_per_layer)``.  ``pos_per_layer`` and ``neg_per_layer``
        are empty dicts when ``sae is None``.  ``sae_layer_set`` is
        ``None`` when ``sae is None``.  ``mean_pos_per_layer`` and
        ``mean_neg_per_layer`` always cover every layer in ``[0,
        n_layers)`` in fp32 on CPU.
    """
    n_layers = len(layers)

    sae_layer_set: set[int] | None
    if sae is not None:
        from saklas.core.errors import SaeCoverageError
        covered = sae.layers & set(range(n_layers))
        if not covered:
            raise SaeCoverageError(
                f"SAE release '{sae.release}' covers no layers for a "
                f"{n_layers}-layer model"
            )
        sae_layer_set = set(sorted(covered))
    else:
        sae_layer_set = None

    # Accumulate per-layer diffs and running norm sums.
    # norm_sums is a GPU tensor to avoid per-layer .item() sync points
    # (was 2 * N_pairs * N_layers GPU→CPU syncs, now 0 during the loop).
    diffs_per_layer: dict[int, list[torch.Tensor]] = {
        i: [] for i in range(n_layers)
    }
    # SAE path: keep the pos/neg tensors themselves (pca_center needs both,
    # not just their diff), but only for layers the SAE actually covers —
    # non-covered layers would allocate O(N · d_model) fp32 tensors for
    # nothing. Raw path: these dicts stay empty, no cost.
    pos_per_layer: dict[int, list[torch.Tensor]] = (
        {i: [] for i in sae_layer_set} if sae_layer_set is not None else {}
    )
    neg_per_layer: dict[int, list[torch.Tensor]] = (
        {i: [] for i in sae_layer_set} if sae_layer_set is not None else {}
    )
    # Per-layer pos/neg running sums for centered DLS.  fp32 on CPU
    # throughout — adds N_pairs * D per pair to the sum, where N_pairs
    # at the bundled n=45 is well within fp32 precision for any
    # reasonable D.  None initially; first pair seeds, subsequent pairs
    # accumulate in place.
    sum_pos: dict[int, torch.Tensor | None] = {i: None for i in range(n_layers)}
    sum_neg: dict[int, torch.Tensor | None] = {i: None for i in range(n_layers)}
    norm_sums = torch.zeros(n_layers, device=device, dtype=torch.float32)

    # On MPS, keep diffs on CPU — SVD runs there anyway, and the
    # model already occupies most of the unified memory budget.
    _mps = device.type == "mps"
    diff_device = "cpu" if _mps else device

    for pair in pairs:
        pos_all = _encode_and_capture_all(
            model, tokenizer, pair["positive"], layers, device,
            role=role, model_type=model_type,
        )
        neg_all = _encode_and_capture_all(
            model, tokenizer, pair["negative"], layers, device,
            role=role, model_type=model_type,
        )
        for idx in range(n_layers):
            p, n = pos_all[idx], neg_all[idx]
            norm_sums[idx] += p.norm() + n.norm()
            p_d = p.to(diff_device)
            n_d = n.to(diff_device)
            diffs_per_layer[idx].append(p_d - n_d)
            # Centered-DLS prep: running fp32 sums on CPU.  Per-layer
            # cost: one float-cast + one in-place add per pair, vs. the
            # diff path's stack-then-svd which dominates anyway.
            p_cpu = p_d.to(dtype=torch.float32, device="cpu")
            n_cpu = n_d.to(dtype=torch.float32, device="cpu")
            sp = sum_pos[idx]
            if sp is None:
                sum_pos[idx] = p_cpu.clone()
                sum_neg[idx] = n_cpu.clone()
            else:
                sp += p_cpu
                neg_acc = sum_neg[idx]
                assert neg_acc is not None
                neg_acc += n_cpu
            if sae_layer_set is not None and idx in sae_layer_set:
                # fp32 matches the diff dtype discipline; avoids fp16 overflow.
                pos_per_layer[idx].append(p_d.float())
                neg_per_layer[idx].append(n_d.float())
        # Free forward-pass intermediates (attention maps, hidden states)
        # before the next pair — MPS doesn't release memory eagerly.
        del pos_all, neg_all
        if _mps:
            torch.mps.empty_cache()

    norm_sums_cpu = norm_sums.tolist()

    n_pairs = len(pairs)
    mean_pos_per_layer: dict[int, torch.Tensor] = {}
    mean_neg_per_layer: dict[int, torch.Tensor] = {}
    if n_pairs > 0:
        for idx in range(n_layers):
            sp = sum_pos[idx]
            sn = sum_neg[idx]
            if sp is not None and sn is not None:
                mean_pos_per_layer[idx] = sp / float(n_pairs)
                mean_neg_per_layer[idx] = sn / float(n_pairs)

    return (
        n_layers,
        diffs_per_layer,
        pos_per_layer,
        neg_per_layer,
        norm_sums_cpu,
        sae_layer_set,
        mean_pos_per_layer,
        mean_neg_per_layer,
    )


def _capture_dim_stats_for_pairs(
    model: torch.nn.Module,
    tokenizer: Any,
    pairs: list[dict[str, str]],
    layers: torch.nn.ModuleList,
    device: torch.device,
    *,
    role: str | None = None,
    model_type: str | None = None,
) -> tuple[
    int,
    dict[int, torch.Tensor],
    list[float],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, list[torch.Tensor]],
]:
    """Capture running DiM statistics without retaining every pair diff.

    PCA needs the full ``N × D`` diff matrix.  DiM only needs the mean diff
    per layer, so the default extractor keeps O(layers × dim) running sums
    on the model device (CPU on MPS) and a bounded CPU sample for diagnostics.
    """
    n_layers = len(layers)
    accum_device = torch.device("cpu") if device.type == "mps" else device
    sum_diffs: dict[int, torch.Tensor | None] = {i: None for i in range(n_layers)}
    sum_pos: dict[int, torch.Tensor | None] = {i: None for i in range(n_layers)}
    sum_neg: dict[int, torch.Tensor | None] = {i: None for i in range(n_layers)}
    diagnostic_samples: dict[int, list[torch.Tensor]] = {
        i: [] for i in range(n_layers)
    }
    norm_sums = torch.zeros(n_layers, device=device, dtype=torch.float32)

    for pair in pairs:
        pos_all = _encode_and_capture_all(
            model, tokenizer, pair["positive"], layers, device,
            role=role, model_type=model_type,
        )
        neg_all = _encode_and_capture_all(
            model, tokenizer, pair["negative"], layers, device,
            role=role, model_type=model_type,
        )
        for idx in range(n_layers):
            p, n = pos_all[idx], neg_all[idx]
            norm_sums[idx] += p.norm() + n.norm()
            p_acc = p.to(dtype=torch.float32, device=accum_device)
            n_acc = n.to(dtype=torch.float32, device=accum_device)
            diff = p_acc - n_acc
            sd = sum_diffs[idx]
            if sd is None:
                sum_diffs[idx] = diff.clone()
            else:
                sd += diff

            p_cpu = p_acc.to("cpu")
            n_cpu = n_acc.to("cpu")
            sp = sum_pos[idx]
            if sp is None:
                sum_pos[idx] = p_cpu.clone()
                sum_neg[idx] = n_cpu.clone()
            else:
                sp += p_cpu
                sn = sum_neg[idx]
                assert sn is not None
                sn += n_cpu

            samples = diagnostic_samples[idx]
            if len(samples) < _DIM_DIAGNOSTIC_SAMPLE_MAX:
                samples.append(diff.detach().to("cpu"))
        del pos_all, neg_all
        if device.type == "mps":
            torch.mps.empty_cache()

    n_pairs = len(pairs)
    mean_diffs: dict[int, torch.Tensor] = {}
    mean_pos_per_layer: dict[int, torch.Tensor] = {}
    mean_neg_per_layer: dict[int, torch.Tensor] = {}
    if n_pairs > 0:
        for idx in range(n_layers):
            sd = sum_diffs[idx]
            if sd is not None:
                mean_diffs[idx] = sd / float(n_pairs)
            sp = sum_pos[idx]
            sn = sum_neg[idx]
            if sp is not None and sn is not None:
                mean_pos_per_layer[idx] = sp / float(n_pairs)
                mean_neg_per_layer[idx] = sn / float(n_pairs)

    return (
        n_layers,
        mean_diffs,
        norm_sums.tolist(),
        mean_pos_per_layer,
        mean_neg_per_layer,
        diagnostic_samples,
    )


def _share_bake_and_warn(
    raw: dict[int, tuple[torch.Tensor, float]],
    diagnostics_per_layer: dict[int, dict[str, float]],
    keep_set: set[int] | None,
    *,
    concept_label: str | None,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    """Apply layer mask + share-baking + emit the diagnostics warning.

    Both extractors close on this — the math is identical regardless of
    whether the per-layer directions came from PCA SVD or from the
    mean-of-diffs.  Mutates ``raw`` and ``diagnostics_per_layer`` in
    place by removing layers not in ``keep_set``.

    ``keep_set=None`` means "keep every layer in ``raw``" (no DLS) — the
    fast path for tests / mock paths that bypass the discriminative
    check entirely.  When provided, layers absent from ``keep_set`` are
    dropped from both ``raw`` and ``diagnostics_per_layer`` before
    share-baking.  The dropped-layer indices are simply absent from the
    returned profile dict; downstream hook attachment iterates the
    keys.
    """
    if keep_set is not None:
        drop = [i for i in raw if i not in keep_set]
        for i in drop:
            raw.pop(i, None)
            diagnostics_per_layer.pop(i, None)

    # Bake shares into the stored tensors. Total share is 1.0 across retained
    # layers, so sum(||baked_i||) ≈ sum(ref_norm_i * share_i): the collective
    # magnitude budget is fixed by the reference activation norms and
    # distributed in proportion to per-layer signal quality.  At apply time
    # the unified subspace kernel reads these baked magnitudes as per-layer
    # weights.
    total_score = sum(score for _, score in raw.values())
    if total_score <= 0:
        # Pathological extraction (all-zero diffs).  Fall back to uniform
        # across retained layers.
        shares = {idx: 1.0 / len(raw) for idx in raw}
    else:
        shares = {idx: score / total_score for idx, (_, score) in raw.items()}

    baked = {idx: direction * shares[idx] for idx, (direction, _) in raw.items()}
    _emit_diagnostics_warning(diagnostics_per_layer, concept_label=concept_label)
    return baked, diagnostics_per_layer


def extract_difference_of_means(
    model: torch.nn.Module,
    tokenizer: Any,
    pairs: list[dict[str, str]],
    layers: torch.nn.ModuleList,
    device: torch.device | None = None,
    *,
    sae: "SaeBackend | None" = None,
    concept_label: str | None = None,
    whitener: "Any | None" = None,
    dls: bool = True,
    layer_means: dict[int, torch.Tensor] | None = None,
    role: str | None = None,
    model_type: str | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    """Contrastive direction extraction via **difference of means** (DiM).

    Per-layer direction is the mean over pos-neg diffs ``mean_i (h_pos_i -
    h_neg_i)``, computed in fp32.

    **Score (default since v2.1, opt-out):** ``||direction||_M / ref_norm``
    where ``||·||_M`` is the Mahalanobis norm against the per-layer
    activation covariance built from cached neutral activations.  The
    ``/ ref_norm`` normalization is what makes the existing share-bake
    pipeline give pure-Mahalanobis hook shares: ``share_L_hook =
    ||m_L||_M / Σ ||m_L'||_M``, with ``ref_norm_L`` cancelling from the
    cross-layer ratio (preserves the Euclidean bake's algebraic shape).
    Layers where the contrastive signal sits in low-variance directions
    score higher than under Euclidean — the metric directly measures
    "how much linearly-decodable signal does this layer carry."

    **Score (Euclidean fallback):** when ``whitener=None``, score is
    ``||direction||_2 / ref_norm`` — the v1.x form.  Used by tests and
    by sessions that haven't populated ``neutral_activations`` yet.
    Pure Euclidean magnitude weighting at hook time.

    Theoretical motivation: Im & Li (2025, arXiv 2502.02716) prove that
    the mean-of-differences direction is optimal for the linear-steering
    objective under squared error.  PCA-of-diffs picks the axis of maximum
    variance among the diffs, which can be near-orthogonal to the actual
    class-separation axis on noisy / inconsistent pair sets — DiM picks
    the class-separation axis directly.  AxBench (Wu et al., ICML 2025)
    corroborates empirically.  The Mahalanobis score is the natural
    extension of LEACE-style metric awareness (Belrose et al. 2023,
    arXiv 2306.03819) to the share-allocation problem: under anisotropic
    activation distributions, Euclidean magnitude over-weights layers
    whose mean-diff happens to align with high-variance noise axes; the
    Mahalanobis form measures signal strength against the activation
    distribution itself.

    Returns the tuple ``(profile, diagnostics)``; ``profile`` maps each
    DLS-retained layer to its baked direction tensor (the unit direction
    pre-multiplied by its cross-layer share), ``diagnostics`` maps each
    layer to its probe-quality fields.  The ``sae=...`` branch runs the
    same mean-of-diffs in feature space and decodes back through the SAE
    before share-baking; no SVD is performed.  The Mahalanobis score is
    computed on the *decoded* model-space direction, where the
    residual-stream whitener applies; SAE feature-space norms don't have a
    meaningful Mahalanobis interpretation under the same covariance.

    The ``whitener`` parameter is a :class:`saklas.core.mahalanobis.LayerWhitener`
    (or ``None`` for the Euclidean fallback).  The metric choice is
    **all-or-nothing**: Mahalanobis scoring is used only when the whitener
    covers *every* scored layer (``LayerWhitener.covers_all``); on partial
    coverage (or no whitener) every layer falls back to Euclidean.  Mixing
    the two across layers would compare incommensurable scales — ``‖·‖_M``
    carries a per-layer ``1/√λ_L`` factor that ``‖·‖_2`` lacks, and that
    factor doesn't cancel from the cross-layer-normalized share.  In
    practice a session whitener covers all layers, so this is full
    Mahalanobis; the gate only bites on a degenerate/partial cache.

    **DLS (Selective Steering, Dang & Ngo 2026).**  Layer selection is
    data-driven via :func:`compute_dls_mask`: a layer is kept only when the
    pos- and neg-class means project to *opposite* sides of the neutral
    baseline along ``d̂`` — same-side layers encode concept *intensity*, not
    *polarity*, and inflate share without aiding discrimination.  Dropped
    layers are simply absent from the returned profile dict.  Pass
    ``dls=False`` to skip the mask (tests / mock paths).
    """
    if device is None:
        device = next(model.parameters()).device
    assert device is not None  # device is always set by this point

    n_pairs = len(pairs)
    n_norm_samples = n_pairs * 2  # pos + neg per pair

    # SAE branch: mean of (F_pos - F_neg) in feature space, decode back.
    if sae is not None:
        (n_layers, _diffs_per_layer, pos_per_layer, neg_per_layer,
         norm_sums_cpu, sae_layer_set,
         mean_pos_per_layer, mean_neg_per_layer) = _capture_diffs_for_pairs(
            model, tokenizer, pairs, layers, device, sae=sae,
            role=role, model_type=model_type,
        )
        assert sae_layer_set is not None
        sae_layers = sorted(sae_layer_set)
        # All-or-nothing metric gate (see ``LayerWhitener.covers_all``): the
        # whitened and Euclidean per-layer scores are on different scales,
        # so we whiten every scored layer or none.  Partial coverage (or no
        # whitener) → Euclidean across the board.  Bind a coverage-gated
        # whitener so the metric decision narrows the type for the branch.
        maha_w = (
            whitener if whitener is not None
            and whitener.covers_all(sae_layers) else None
        )
        directions: dict[int, torch.Tensor] = {}
        score_tensors: list[torch.Tensor] = []
        diagnostics_per_layer: dict[int, dict[str, float]] = {}
        for idx in sae_layers:
            pos_stack = torch.stack(pos_per_layer[idx])  # (N, d_model), fp32
            neg_stack = torch.stack(neg_per_layer[idx])
            ref_norm = norm_sums_cpu[idx] / n_norm_samples

            with torch.no_grad():
                F_pos = sae.encode_layer(idx, pos_stack.to(device)).float()
                F_neg = sae.encode_layer(idx, neg_stack.to(device)).float()

            # DiM in feature space: mean of paired diffs.  No SVD, no
            # orientation step — pos-minus-neg already points pos-ward.
            v_feat = (F_pos - F_neg).mean(dim=0)

            with torch.no_grad():
                v_model = sae.decode_layer(idx, v_feat).float()

            # Score: Mahalanobis norm of the decoded model-space direction
            # (same shape as the raw branch — see ``score`` docstring).
            # Whitener-absent or partial coverage → Euclidean for every
            # layer (``maha_w`` is None unless coverage is complete) so SAE
            # extraction without a populated neutral_activations cache works.
            if maha_w is not None:
                m_norm = maha_w.mahalanobis_norm(idx, v_model)
                score_value = m_norm / max(ref_norm, 1e-8)
                score_tensors.append(torch.tensor(score_value, dtype=torch.float32))
            else:
                v_model_norm = v_model.norm().clamp(min=1e-8)
                score_tensors.append(v_model_norm / max(ref_norm, 1e-8))

            directions[idx] = _normalize(v_model, ref_norm=ref_norm)

            # Diagnostics in model space against the contrastive diffs —
            # same shape as PCA so consumers don't branch.  Principal
            # direction is the decoded DiM vector, EVR-as-score-proxy is
            # the same diff-norm-vs-activation ratio used elsewhere for
            # mean-based scoring (matches single-pair PCA's ``score``).
            diff_model = (pos_stack - neg_stack).to("cpu")
            diff_norms = diff_model.norm(dim=-1)
            score_proxy = float(
                (diff_norms.mean() / max(ref_norm * 2, 1e-8)).item()
            )
            diagnostics_per_layer[idx] = _compute_layer_diagnostics(
                diff_model,
                v_model.detach().to("cpu"),
                score_proxy,
            )

        scores = torch.stack(score_tensors).tolist()
        raw: dict[int, tuple[torch.Tensor, float]] = {
            idx: (directions[idx], score)
            for idx, score in zip(sae_layers, scores)
        }
        # Centered DLS on the SAE-decoded directions, restricted to
        # SAE-covered layers (the means the helper consumes are also
        # restricted there — feature-space encoding only touched those).
        sae_directions_unit = {
            idx: directions[idx] / max(float(directions[idx].norm()), 1e-12)
            for idx in sae_layers
        }
        sae_pos_means = {
            idx: torch.stack(pos_per_layer[idx]).mean(dim=0).cpu()
            for idx in sae_layers
        }
        sae_neg_means = {
            idx: torch.stack(neg_per_layer[idx]).mean(dim=0).cpu()
            for idx in sae_layers
        }
        keep_set = compute_dls_mask(
            sae_pos_means, sae_neg_means, sae_directions_unit,
            layer_means,
        ) if dls else None
        return _share_bake_and_warn(
            raw, diagnostics_per_layer, keep_set,
            concept_label=concept_label,
        )

    (
        n_layers,
        mean_diffs,
        norm_sums_cpu,
        mean_pos_per_layer,
        mean_neg_per_layer,
        diagnostic_samples,
    ) = _capture_dim_stats_for_pairs(
        model, tokenizer, pairs, layers, device,
        role=role, model_type=model_type,
    )

    # Per-layer DiM in residual-stream space.  All-or-nothing metric gate
    # over the full scored-layer set (``range(n_layers)``): whiten every
    # layer or none (see ``LayerWhitener.covers_all``).
    raw = {}
    diagnostics_per_layer = {}
    maha_w = (
        whitener if whitener is not None
        and whitener.covers_all(range(n_layers)) else None
    )

    if n_pairs < 2:
        # Single pair degenerates: mean over one element is just the
        # element.  Use the same scoring as single-pair PCA so share-bake
        # math is unaffected.
        diff_stack = torch.stack([mean_diffs[idx] for idx in range(n_layers)])
        diff_norms_cpu = diff_stack.norm(dim=-1).tolist()
        for idx in range(n_layers):
            diff_vec = mean_diffs[idx]
            ref_norm = norm_sums_cpu[idx] / n_norm_samples
            direction = _normalize(diff_vec, ref_norm=ref_norm)
            activation_norm = norm_sums_cpu[idx]
            if maha_w is not None:
                # Mahalanobis on the single diff vector; ``activation_norm``
                # is pos+neg sum, mirrors the Euclidean form's denominator.
                m_norm = maha_w.mahalanobis_norm(idx, diff_vec)
                score = m_norm / max(activation_norm, 1e-8)
            else:
                score = diff_norms_cpu[idx] / max(activation_norm, 1e-8)
            raw[idx] = (direction, score)
            diagnostics_per_layer[idx] = _compute_layer_diagnostics(
                diff_vec.unsqueeze(0),
                direction,
                score,
            )
    else:
        # Multi-pair DiM: running mean diffs are already available, so no
        # full ``layers × pairs × dim`` diff matrix is materialized.
        ref_norms = [
            norm_sums_cpu[idx] / n_norm_samples for idx in range(n_layers)
        ]
        means = torch.stack([mean_diffs[idx] for idx in range(n_layers)])
        means_norms = means.norm(dim=-1)

        if maha_w is not None:
            # Mahalanobis branch: per-layer matvec via Woodbury through
            # ``LayerWhitener.mahalanobis_norm``.  Loop instead of batch
            # because each layer has its own ``Σ_L^{-1}`` and ``X_L``;
            # extraction is one-shot, not a hot path.  ``maha_w`` is only
            # set when the whitener covers every layer (all-or-nothing), so
            # no per-layer coverage check is needed here.
            scores_cpu = []
            for idx in range(n_layers):
                ref_L = max(ref_norms[idx], 1e-8)
                m_norm = maha_w.mahalanobis_norm(idx, means[idx])
                scores_cpu.append(m_norm / ref_L)
        else:
            # Euclidean fallback: original batched path, single GPU→CPU
            # transfer.  Score = ||mean_diff||_2 / ref_norm — lands in
            # the same range as PCA's EVR (~0.01–0.4).
            scores_t = means_norms / torch.tensor(
                ref_norms, device=means_norms.device, dtype=means_norms.dtype,
            ).clamp(min=1e-8)
            scores_cpu = scores_t.tolist()

        for idx in range(n_layers):
            direction = means[idx].to(device)
            raw[idx] = (
                _normalize(direction, ref_norm=ref_norms[idx]),
                scores_cpu[idx],
            )
            # Diagnostics use the unit-direction so EVR-as-score-proxy
            # and the alignment metric stay scale-invariant.
            diagnostics_per_layer[idx] = _compute_layer_diagnostics(
                torch.stack(diagnostic_samples[idx])
                if diagnostic_samples[idx]
                else means[idx].detach().to("cpu").unsqueeze(0),
                means[idx].detach().to("cpu"),
                scores_cpu[idx],
            )

    # Centered-DLS mask via the per-layer mean-of-diffs direction.
    # ``mean_pos - mean_neg`` is exactly the DiM direction (linearity of
    # expectation), so the projection check works directly on the
    # CPU-side means without re-computing.  Unit-norm so the projection
    # check stays scale-invariant.
    if dls:
        unit_dirs: dict[int, torch.Tensor] = {}
        for idx in range(n_layers):
            mp = mean_pos_per_layer.get(idx)
            mn = mean_neg_per_layer.get(idx)
            if mp is None or mn is None:
                continue
            d = mp - mn
            d_n = float(d.norm())
            if d_n > 1e-12:
                unit_dirs[idx] = d / d_n
        keep_set = compute_dls_mask(
            mean_pos_per_layer, mean_neg_per_layer, unit_dirs, layer_means,
        )
    else:
        keep_set = None
    return _share_bake_and_warn(
        raw, diagnostics_per_layer, keep_set, concept_label=concept_label,
    )


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


def fold_vector_to_subspace(
    model: torch.nn.Module,
    tokenizer: Any,
    pairs: list[dict[str, str]],
    layers: torch.nn.ModuleList,
    device: torch.device | None = None,
    *,
    concept_label: str,
    pos_label: str,
    neg_label: str,
    whitener: "Any | None" = None,
    dls: bool = True,
    layer_means: dict[int, torch.Tensor] | None = None,
    role: str | None = None,
    model_type: str | None = None,
) -> "Any":  # -> saklas.core.manifold.Manifold
    """Extract a folded steering vector as an affine ``R = 1`` manifold.

    The subspace-native replacement for :func:`extract_difference_of_means`:
    captures the per-layer pos/neg centroids over the contrastive ``pairs``
    (the same forward-pass capture the DiM path uses), then folds them into
    an affine manifold via :func:`_fold_centroids_to_affine_manifold`.  See
    that function for the geometry, the exact-parity share, and the origin /
    DLS handling.  Pure extraction — the caller persists and routes the
    result.
    """
    if device is None:
        device = next(model.parameters()).device
    assert device is not None
    (
        _n_layers, _mean_diffs, _norm_sums,
        mean_pos_per_layer, mean_neg_per_layer, _diag,
    ) = _capture_dim_stats_for_pairs(
        model, tokenizer, pairs, layers, device,
        role=role, model_type=model_type,
    )
    return _fold_centroids_to_affine_manifold(
        concept_label,
        mean_pos_per_layer, mean_neg_per_layer,
        pos_label=pos_label, neg_label=neg_label,
        whitener=whitener, layer_means=layer_means, dls=dls,
    )


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
        try:
            metadata["diagnostics"] = {
                int(layer): dict(metrics)
                for layer, metrics in raw_diag.items()
            }
        except (TypeError, ValueError):
            # Leave the raw dict in place; downstream readers can decide
            # whether to fall back.  Don't fail the load over malformed
            # diagnostics — the tensors themselves are still valid.
            pass

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


def load_contrastive_pairs(dataset_path: str) -> dict[str, Any]:
    """Load a contrastive-pairs JSON file.

    Expects a bare list: ``[{"positive": ..., "negative": ...}, ...]``
    (the shape written to ``statements.json`` in concept folders).
    Returns a dict ``{"pairs": [...]}``.
    """
    with open(dataset_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"{dataset_path}: expected a JSON list of pairs, got {type(data).__name__}"
        )
    return {"pairs": data}
