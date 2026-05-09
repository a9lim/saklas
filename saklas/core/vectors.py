"""Extraction, saving, and loading of activation steering/probe vectors."""

from __future__ import annotations

import functools
import json
import logging
import warnings
from importlib import resources as _resources
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file

if TYPE_CHECKING:
    from saklas.core.sae import SaeBackend

log = logging.getLogger(__name__)

# Per-layer probe-quality diagnostics: thresholds for the soft warning the
# extractor emits at end-of-extraction.  Fired against the median across
# retained layers — a single dim layer with rough metrics is normal; a
# concept whose median is degenerate is the failure mode users care about.
_DIAG_DEGENERATE_EVR = 0.95         # ~all variance in one direction
_DIAG_DEGENERATE_INTRA_VAR = 0.01   # almost-identical pos/neg pairs
_DIAG_LOW_ALIGNMENT = 0.2           # pairs disagree on direction

# Skip the chat template for extraction when it adds more than this many
# tokens of overhead (e.g. Ministral injects a ~500-token system prompt).
# The overhead cancels in contrastive diffs but wastes memory per pass.
_MAX_TEMPLATE_OVERHEAD = 100

# Keyed by id(tokenizer).  Object IDs can be reused after GC, so this
# cache is only safe when a single tokenizer lives for the session lifetime
# (which is the case in both the TUI and the API server).
_template_overhead_cache: dict[int, int] = {}


def _chat_template_overhead(tokenizer, template_kwargs: dict) -> int:
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


def _capture_all_hidden_states(model, layers, input_ids):
    """Run a single-sequence forward pass capturing hidden states at ALL layers.

    Uses ``use_cache=False`` to avoid polluting any persistent KV cache.

    Returns:
        dict mapping layer index to (1, seq, dim) tensors.
    """
    captured_hidden: dict[int, torch.Tensor] = {}

    def _make_hook(idx):
        def _hook(module, input, output):
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


def _encode_and_capture_all(model, tokenizer, text, layers, device):
    """Tokenize text, run forward pass, return last-content-token hidden state per layer in fp32.

    For instruction-tuned models (those with a chat template), wraps the text
    as an assistant response so the extraction happens in the model's actual
    generation regime.  Base models get the raw string.

    Pools from the last non-special token — chat templates append trailing
    markers (Llama's <|eot_id|>, Gemma's <end_of_turn>, Qwen's <|im_end|>)
    whose hidden states are disconnected from content.  The last content
    token's hidden state is itself an attention-weighted summary of prior
    positions and is exactly what the model uses for next-token prediction.

    Returns:
        dict mapping layer_idx -> pooled vector (dim,) in fp32.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        # Disable thinking/reasoning mode for models that support it
        # (Qwen 3.5, QwQ, etc.) — thinking tokens would contaminate pooling.
        kwargs: dict = {}
        if "enable_thinking" in (getattr(tokenizer, "chat_template", "") or ""):
            kwargs["enable_thinking"] = False
        messages = [{"role": "assistant", "content": text}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs,
            )
        except Exception:
            # Some chat templates require a user turn before assistant.
            # The filler must be semantically empty — "." triggers
            # model-specific greeting/help responses whose template
            # tokens contaminate pooling.
            messages = [
                {"role": "user", "content": "Continue:"},
                {"role": "assistant", "content": text},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs,
            )
        # Some chat templates inject a large system prompt (e.g.
        # Ministral adds ~500 tokens).  For contrastive extraction the
        # overhead cancels in the diff but wastes memory on every
        # forward pass.  Fall back to raw tokenization when excessive.
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
    skip_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    added = getattr(tokenizer, "added_tokens_encoder", None) or {}
    skip_ids.update(int(v) for v in added.values())
    content_end = ids.shape[1] - 1
    if skip_ids:
        id_list = ids[0].tolist()
        while content_end > 0 and id_list[content_end] in skip_ids:
            content_end -= 1

    hidden_per_layer = _capture_all_hidden_states(model, layers, ids)
    return {
        idx: h[0, min(content_end, h.shape[1] - 1)].float()
        for idx, h in hidden_per_layer.items()
    }


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
    model,
    tokenizer,
    layers,
    device=None,
) -> dict[int, torch.Tensor]:
    """Compute mean hidden state per layer over neutral prompts.

    Returns dict mapping layer_idx -> mean vector (dim,) in fp32.
    Used to center activations before probe cosine similarity,
    removing the baseline projection bias.
    """
    if device is None:
        device = next(model.parameters()).device

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
    model,
    tokenizer,
    layers,
    device=None,
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

    Fires at most once per call.  Threshold pair (a) catches one-sided /
    repetitive pair sets (high EVR, near-zero intra variance); threshold
    (b) catches incoherent pair sets (low inter-pair alignment).  Both
    leave the extracted profile usable — the warning is informational,
    not a block.
    """
    if not diagnostics:
        return

    def _median(values: list[float]) -> float:
        s = sorted(values)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    evrs = [d["evr"] for d in diagnostics.values() if "evr" in d]
    intras = [
        d["intra_pair_variance_mean"]
        for d in diagnostics.values()
        if "intra_pair_variance_mean" in d
    ]
    aligns = [
        d["inter_pair_alignment"]
        for d in diagnostics.values()
        if "inter_pair_alignment" in d
    ]
    if not evrs:
        return

    med_evr = _median(evrs)
    med_intra = _median(intras) if intras else float("inf")
    med_align = _median(aligns) if aligns else 1.0

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
    elif med_align < _DIAG_LOW_ALIGNMENT:
        warnings.warn(
            f"{label}: pair directions disagree "
            f"(median inter-pair alignment={med_align:.2f}); "
            f"the principal component still extracts but pairs point in "
            f"conflicting directions. Review statements.json for "
            f"semantically orthogonal pairs.",
            UserWarning,
            stacklevel=3,
        )


def compute_dls_mask(
    mu_pos: dict[int, torch.Tensor],
    mu_neg: dict[int, torch.Tensor],
    directions: dict[int, torch.Tensor],
    layer_means: dict[int, torch.Tensor] | None,
) -> set[int]:
    """Discriminative-Layer-Selection (Dang & Ngo 2026, Eq. 9) keep set.

    Returns the set of layer indices that pass the centered-DLS check::

        μ̃_pos = (μ_pos − μ_neutral) · d̂
        μ̃_neg = (μ_neg − μ_neutral) · d̂
        keep iff μ̃_pos · μ̃_neg < 0

    Layers where both pos- and neg-class means project to the same side of
    the neutral baseline along ``d̂`` are non-discriminative — they encode
    something correlated with concept *intensity* but not concept
    *polarity*, and steering through them rotates a residual that's
    already aligned with the same pole as the neutral baseline (i.e.
    they don't carry the contrast).  Dropping them concentrates share
    on layers that genuinely encode the pos/neg axis.

    **Centering** is required.  Without subtracting ``μ_neutral`` raw
    activations all project positively along ``d̂`` at most layers
    because of shared base-rate variance, and the sign-of-projection
    test fires near-uniformly across layers (verified empirically: the
    literal-paper uncentered version on gemma-4-e4b-it / angry.calm
    keeps 17/42 in fragmented patterns vs. centered 31/42 in the
    expected mid-stack band).  When ``layer_means`` is ``None``,
    centering is undefined and the helper returns *all* layers — DLS
    is disabled silently and the caller falls back to "keep
    everything."  Real session-driven extraction always passes
    ``layer_means`` (built before extraction during ``bootstrap_probes``);
    fixture / mock paths skip it.

    Args:
        mu_pos: per-layer mean of positive-class final-content-token
            hidden states, fp32, shape ``[D]`` per layer.
        mu_neg: same for the negative class.  Must cover the same layer
            set as ``mu_pos``.
        directions: per-layer unsigned direction vectors (typically
            ``unit(μ_pos − μ_neg)`` for DiM, or the principal component
            for PCA).  Same layer set as ``mu_pos``.  Magnitude doesn't
            matter — only sign of projection — but unit-normed input
            is recommended for numerical sanity.
        layer_means: per-layer neutral baseline (90-prompt mean, cached
            under ``~/.saklas/models/<id>/layer_means.safetensors``).
            ``None`` disables DLS (returns all layers).

    Returns:
        ``set[int]`` of layers that pass the discriminative check.  Empty
        set is *not* returned — if every layer fails, returns the full
        layer set with a warning, on the principle that something is
        better than nothing for a degenerate concept (the caller's
        downstream extractor diagnostics will already flag the probe
        quality).
    """
    # Empty-dict is treated identically to ``None`` — both mean "no
    # baseline available, fall back to keep-all silently."  Without
    # the explicit early-out, every layer in the loop would hit
    # ``mu_n = layer_means.get(L) → None`` and fall through the
    # conservative-keep branch, which produces the same observable
    # result but routes through confusing per-layer logic.  This
    # guard also catches the v2.1 footgun where ``probes=[]``
    # sessions had ``self._layer_means = {}`` propagate to the
    # helper, silently disabling DLS even when neutrals were
    # cacheable — fixed at the session layer via the lazy
    # ``layer_means`` property, but the helper-level guard is cheap
    # insurance against future leak paths.
    if layer_means is None or not layer_means:
        return set(mu_pos)
    keep: set[int] = set()
    # ``checkable`` is the set of layers we *attempted* the
    # discriminative test on — i.e. layers with a valid direction.
    # Layers explicitly skipped for missing direction or zero-norm
    # direction are excluded from the all-failed fallback (they're
    # degenerate by construction; re-including them via the fallback
    # would silently undo the skip).
    checkable: set[int] = set()
    for L in mu_pos:
        d = directions.get(L)
        if d is None:
            continue
        d32 = d.to(dtype=torch.float32, device="cpu").reshape(-1)
        d_norm = float(d32.norm())
        if d_norm < 1e-12:
            continue  # degenerate direction — drop, do not include in fallback
        checkable.add(L)
        mu_n = layer_means.get(L)
        if mu_n is None:
            # Layer-means doesn't cover this layer.  Conservative: keep —
            # we'd rather over-include than mistakenly drop a real
            # discriminative layer due to missing baseline data.
            keep.add(L)
            continue
        d_hat = d32 / d_norm
        mu_n_cpu = mu_n.to(dtype=torch.float32, device="cpu").reshape(-1)
        mu_p_cpu = mu_pos[L].to(dtype=torch.float32, device="cpu").reshape(-1)
        mu_g_cpu = mu_neg[L].to(dtype=torch.float32, device="cpu").reshape(-1)
        proj_pos = float(((mu_p_cpu - mu_n_cpu) * d_hat).sum())
        proj_neg = float(((mu_g_cpu - mu_n_cpu) * d_hat).sum())
        if proj_pos * proj_neg < 0.0:
            keep.add(L)
    if not keep:
        # All checkable layers failed the discriminative check.  Probe
        # is degenerate on this model (the diagnostics warning will
        # fire separately).  Keep the *checkable* set rather than every
        # layer in ``mu_pos`` — re-including layers explicitly skipped
        # for missing/zero-norm directions would silently undo the
        # skip.  When no layers are checkable at all (caller passed
        # all-degenerate directions) the fallback returns an empty
        # set; the share-bake step's all-zero handler kicks in.
        warnings.warn(
            "DLS: no layers pass the discriminative check; falling back "
            "to keep-all-checkable.  Probe likely degenerate on this "
            "model — review the diagnostics warning above.",
            UserWarning,
            stacklevel=3,
        )
        return checkable
    return keep


def _capture_diffs_for_pairs(
    model,
    tokenizer,
    pairs: list[dict],
    layers,
    device,
    *,
    sae: "SaeBackend | None" = None,
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
        pos_all = _encode_and_capture_all(model, tokenizer, pair["positive"], layers, device)
        neg_all = _encode_and_capture_all(model, tokenizer, pair["negative"], layers, device)
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
    # the additive hook does ``alpha * _STEER_GAIN * sum(baked)``; the
    # angular hook reads the same baked magnitudes as per-layer weights.
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


def extract_contrastive(
    model,
    tokenizer,
    pairs: list[dict],
    layers,
    device=None,
    *,
    sae: "SaeBackend | None" = None,
    concept_label: str | None = None,
    dls: bool = True,
    layer_means: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    """Contrastive direction extraction via PCA across all layers.

    Hooks every layer in the same 2N forward passes. For each layer,
    computes the first principal component of pos-neg differences.

    Per-layer extraction yields a raw direction (scaled to the mean
    activation norm of that layer) and a raw score: explained variance
    ratio for multi-pair, diff_norm/activation_norm for single-pair.
    The returned tensors are "baked": each direction is pre-multiplied
    by its share ``score_i / sum(scores)`` so the layer-weighting math
    that used to live in the steering hook collapses to a flat
    ``user_alpha * _STEER_GAIN * sum(directions)``. All invariances
    (layer-count, score-magnitude) are preserved — they just move from
    apply-time to extract-time.

    Note: as of v2.1 PCA is the **legacy** extraction method; the default
    is :func:`extract_difference_of_means`.  PCA picks the axis of maximum
    contrastive variance, which can be near-orthogonal to the actual
    class-separation axis on noisy pair sets (Im & Li 2025); DiM picks
    the class-separation axis directly.  Both share the share-baking and
    diagnostics machinery — only the per-layer direction computation
    differs.

    **DLS replaces edge-drop in v2.1.**  The `drop_edges` parameter is
    gone; the layer mask is now derived from the data itself via
    :func:`compute_dls_mask` (centered Selective-Steering, Dang & Ngo
    2026).  Layers where both pos- and neg-class means project to the
    same side of the neutral baseline along ``d̂`` are dropped — they
    encode concept *intensity* rather than concept *polarity* and
    inflate share without aiding discrimination.  Empirical incidence
    (gemma-4-e4b-it / angry.calm: 11/42 dropped; Qwen3.6-27B / same:
    13/64 dropped, mostly contiguous L49–L60).  Dropped layers are
    simply absent from the returned profile dict.

    Args:
        pairs: List of {"positive": str, "negative": str} prompt pairs.
        sae: Optional SAE backend. When provided, extraction runs a
            feature-space ``pca_center`` branch restricted to the layers
            covered by the SAE, and decodes the principal feature-space
            direction back into model space before baking shares.
        concept_label: Optional human-readable name surfaced in the
            soft-warning text when diagnostics flag a degenerate probe.
            Defaults to a generic label.
        dls: When ``True`` (default since v2.1), apply the centered
            DLS mask via :func:`compute_dls_mask`.  When ``False``,
            keep every layer — the path tests use for small mock models
            (DLS on synthetic 4-layer data is degenerate).
        layer_means: Per-layer neutral baseline (``{layer: [D] fp32}``)
            used by DLS centering.  Real session-driven extraction always
            passes this from ``self._layer_means``; ``None`` falls back
            to "keep all layers" silently (with a warning logged from
            ``compute_dls_mask`` when DLS is enabled but layer_means is
            missing).

    Returns:
        ``(profile, diagnostics)``:

        * ``profile`` — dict mapping layer_idx → baked direction vector.
          Dropped edge layers are simply absent from the dict; downstream
          hook attachment iterates over present keys.
        * ``diagnostics`` — dict mapping layer_idx → ``{evr,
          intra_pair_variance_mean, intra_pair_variance_std,
          inter_pair_alignment, diff_principal_projection}``.  Same key
          set as ``profile`` (no diagnostics for dropped edges).  Empty
          dict when ``len(pairs) < 2`` and the SVD path is skipped.
    """
    if device is None:
        device = next(model.parameters()).device

    (n_layers, diffs_per_layer, pos_per_layer, neg_per_layer,
     norm_sums_cpu, sae_layer_set,
     mean_pos_per_layer, mean_neg_per_layer) = _capture_diffs_for_pairs(
        model, tokenizer, pairs, layers, device, sae=sae,
    )

    n_pairs = len(pairs)
    n_norm_samples = n_pairs * 2  # pos + neg per pair

    # SAE branch: feature-space pca_center, decode back to model space.
    # Runs only on the covered-layer subset; returns early via the shared
    # share-baking helper with tensors restricted to those layers.
    if sae is not None:
        assert sae_layer_set is not None  # capture helper guarantees this
        sae_layers = sorted(sae_layer_set)
        directions: dict[int, torch.Tensor] = {}
        # Accumulate EVR scalars as tensors, one .tolist() after the loop
        # (matches the raw-PCA path's discipline — one GPU→CPU transfer).
        evr_tensors: list[torch.Tensor] = []
        # Diagnostics computed in *model space* on the contrastive diffs
        # — gives a cross-comparable readout against raw PCA, even though
        # the principal direction itself was fit in SAE feature space.
        diagnostics_per_layer: dict[int, dict[str, float]] = {}
        for idx in sae_layers:
            pos_stack = torch.stack(pos_per_layer[idx])  # (N, d_model), fp32
            neg_stack = torch.stack(neg_per_layer[idx])
            ref_norm = norm_sums_cpu[idx] / n_norm_samples

            with torch.no_grad():
                F_pos = sae.encode_layer(idx, pos_stack.to(device)).float()
                F_neg = sae.encode_layer(idx, neg_stack.to(device)).float()

            center = (F_pos + F_neg) / 2.0
            stacked = torch.cat([F_pos - center, F_neg - center], dim=0)  # (2N, d_feat)

            _, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
            v_feat = Vh[0]
            evr_layer_t = S[0] / S.sum()
            evr_tensors.append(evr_layer_t)

            # Orient by majority vote: pos-minus-neg projected onto v_feat
            # should majority-positive.
            dots = (F_pos - F_neg) @ v_feat
            if (dots < 0).sum() > (dots > 0).sum():
                v_feat = -v_feat

            with torch.no_grad():
                v_model = sae.decode_layer(idx, v_feat).float()

            directions[idx] = _normalize(v_model, ref_norm=ref_norm)

            # Per-layer diagnostics: model-space diffs are pos_stack -
            # neg_stack (already fp32 on CPU when MPS, on device otherwise);
            # principal direction in model space is v_model (pre-share-bake).
            diff_model = (pos_stack - neg_stack).to("cpu")
            diagnostics_per_layer[idx] = _compute_layer_diagnostics(
                diff_model,
                v_model.detach().to("cpu"),
                float(evr_layer_t.item()),
            )

        evrs = torch.stack(evr_tensors).tolist()
        raw: dict[int, tuple[torch.Tensor, float]] = {
            idx: (directions[idx], evr) for idx, evr in zip(sae_layers, evrs)
        }
        # SAE-path DLS: feature-space directions decoded back to model
        # space; centered DLS check uses the *decoded* direction against
        # the same neutral baseline.  Unit-norm the decoded vector for
        # the projection check (magnitude carries no sign information).
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

    # Per-layer: compute direction and score via PCA, then bake shares.
    raw = {}
    diagnostics_per_layer = {}

    if n_pairs < 2:
        # Single pair: score as diff norm relative to activation magnitude.
        # This produces values in roughly the same range as the
        # explained-variance-ratio used for multi-pair extraction
        # (typically 0.01–0.4), so single-pair and multi-pair profiles
        # contribute comparably when baked into shares.
        # Stack single diffs and batch the norm compute so we only do one
        # GPU→CPU transfer instead of n_layers individual .item() syncs.
        diff_stack = torch.stack([diffs_per_layer[idx][0] for idx in range(n_layers)])
        diff_norms_cpu = diff_stack.norm(dim=-1).tolist()
        for idx in range(n_layers):
            diff_vec = diffs_per_layer[idx][0]
            ref_norm = norm_sums_cpu[idx] / n_norm_samples
            direction = _normalize(diff_vec, ref_norm=ref_norm)
            activation_norm = norm_sums_cpu[idx]  # pos_norm + neg_norm
            score = diff_norms_cpu[idx] / max(activation_norm, 1e-8)
            raw[idx] = (direction, score)
            # Single-pair diagnostics carry only what's defined for N=1
            # (intra mean = the single diff norm, std = 0, alignment / proj
            # tautologically 1.0).  Helps the JSON sidecar stay shape-stable
            # across pair counts.
            diagnostics_per_layer[idx] = _compute_layer_diagnostics(
                diff_vec.unsqueeze(0),
                direction,
                score,
            )
    else:
        # Multi-pair: batched SVD across all layers.
        # Stack into (n_layers, N, dim) and run one batched SVD call —
        # amortizes LAPACK dispatch overhead vs. n_layers individual calls,
        # and matters most on CPU (MPS SVD fallback path).
        diff_matrices = []  # one (N, dim) per layer
        ref_norms = []
        for idx in range(n_layers):
            diff_matrices.append(torch.stack(diffs_per_layer[idx]))  # (N, dim)
            ref_norms.append(norm_sums_cpu[idx] / n_norm_samples)

        # Diffs are already float32 and on CPU for MPS — run SVD directly.
        batched = torch.stack(diff_matrices)  # (n_layers, N, dim)
        _, S, Vh = torch.linalg.svd(batched, full_matrices=False)
        # S: (n_layers, min(N,dim)), Vh: (n_layers, min(N,dim), dim)

        # Batch EVR compute: one vector-wide op and a single GPU→CPU
        # transfer instead of n_layers scalar .item() calls.
        scores_cpu = (S[:, 0] / S.sum(dim=-1)).tolist()

        for idx in range(n_layers):
            direction = Vh[idx, 0].to(device)  # (dim,)

            # Orient so "positive" stays positive: majority vote across pairs.
            dots = diff_matrices[idx] @ direction.to(diff_matrices[idx].device)
            if (dots < 0).sum() > (dots > 0).sum():
                direction = -direction

            raw[idx] = (_normalize(direction, ref_norm=ref_norms[idx]), scores_cpu[idx])

            # Per-layer diagnostics on the same diff matrix the SVD ran on.
            # Use the unsigned principal direction — sign convention doesn't
            # affect any of the diagnostic metrics (all are absolute-value
            # or magnitude based).
            diagnostics_per_layer[idx] = _compute_layer_diagnostics(
                diff_matrices[idx],
                Vh[idx, 0].detach().to(diff_matrices[idx].device),
                scores_cpu[idx],
            )

    # Centered-DLS mask.  PCA's per-layer principal components feed the
    # discriminative check; layer means come from the session's cached
    # neutrals.  ``dls=False`` skips the mask entirely (tests on small
    # mock models where the synthetic data is too small for the
    # discriminative test to be meaningful).
    if dls:
        # Build the unit-normed direction dict from raw (the share-bake
        # multiplies by score; we want the direction shape only).
        unit_dirs = {
            idx: tup[0] / max(float(tup[0].norm()), 1e-12)
            for idx, tup in raw.items()
        }
        keep_set = compute_dls_mask(
            mean_pos_per_layer, mean_neg_per_layer, unit_dirs, layer_means,
        )
    else:
        keep_set = None
    return _share_bake_and_warn(
        raw, diagnostics_per_layer, keep_set, concept_label=concept_label,
    )


def extract_difference_of_means(
    model,
    tokenizer,
    pairs: list[dict],
    layers,
    device=None,
    *,
    sae: "SaeBackend | None" = None,
    concept_label: str | None = None,
    whitener: "Any | None" = None,
    dls: bool = True,
    layer_means: dict[int, torch.Tensor] | None = None,
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

    Shape and metadata are identical to :func:`extract_contrastive`:
    same returned tuple ``(profile, diagnostics)``, same edge-drop and
    share-baking discipline, same diagnostics fields.  The only behavior
    delta is the per-layer score (and therefore share allocation) when
    a whitener is provided; downstream hook math sees a baked direction
    tensor either way.

    Args / Returns: see :func:`extract_contrastive`.  The ``sae=...``
    branch runs the same mean-of-diffs in feature space and decodes back
    through the SAE before share-baking; no SVD is performed.  The
    Mahalanobis score is computed on the *decoded* model-space direction,
    where the residual-stream whitener applies; SAE feature-space norms
    don't have a meaningful Mahalanobis interpretation under the same
    covariance.

    The ``whitener`` parameter is a :class:`saklas.core.mahalanobis.LayerWhitener`
    (or ``None`` for the Euclidean fallback).  Layers absent from the
    whitener fall back to Euclidean per-layer — the whitener may
    legitimately cover a subset of layers in edge cases.

    **DLS replaces edge-drop in v2.1.**  The ``drop_edges`` parameter
    is gone; layer selection is data-driven via :func:`compute_dls_mask`.
    Pass ``dls=False`` to skip the mask (tests / mock paths).  See
    :func:`extract_contrastive` for the rationale.
    """
    if device is None:
        device = next(model.parameters()).device

    (n_layers, diffs_per_layer, pos_per_layer, neg_per_layer,
     norm_sums_cpu, sae_layer_set,
     mean_pos_per_layer, mean_neg_per_layer) = _capture_diffs_for_pairs(
        model, tokenizer, pairs, layers, device, sae=sae,
    )

    n_pairs = len(pairs)
    n_norm_samples = n_pairs * 2  # pos + neg per pair

    # SAE branch: mean of (F_pos - F_neg) in feature space, decode back.
    if sae is not None:
        assert sae_layer_set is not None
        sae_layers = sorted(sae_layer_set)
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
            # Whitener-absent or layer-uncovered → Euclidean fallback so
            # SAE extraction without a populated neutral_activations cache
            # still works.
            if whitener is not None and whitener.covers(idx):
                m_norm = whitener.mahalanobis_norm(idx, v_model)
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

    # Per-layer DiM in residual-stream space.
    raw = {}
    diagnostics_per_layer = {}

    if n_pairs < 2:
        # Single pair degenerates: mean over one element is just the
        # element.  Use the same scoring as single-pair PCA so share-bake
        # math is unaffected.
        diff_stack = torch.stack(
            [diffs_per_layer[idx][0] for idx in range(n_layers)]
        )
        diff_norms_cpu = diff_stack.norm(dim=-1).tolist()
        for idx in range(n_layers):
            diff_vec = diffs_per_layer[idx][0]
            ref_norm = norm_sums_cpu[idx] / n_norm_samples
            direction = _normalize(diff_vec, ref_norm=ref_norm)
            activation_norm = norm_sums_cpu[idx]
            if whitener is not None and whitener.covers(idx):
                # Mahalanobis on the single diff vector; ``activation_norm``
                # is pos+neg sum, mirrors the Euclidean form's denominator.
                m_norm = whitener.mahalanobis_norm(idx, diff_vec)
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
        # Multi-pair: stack diffs, take mean across pairs in fp32.
        # One stacked tensor + one ``mean(dim=1)`` per shape — single
        # GPU→CPU transfer for the per-layer norms (= scores).
        diff_matrices = [
            torch.stack(diffs_per_layer[idx]) for idx in range(n_layers)
        ]  # each (N, dim) fp32
        ref_norms = [
            norm_sums_cpu[idx] / n_norm_samples for idx in range(n_layers)
        ]

        batched = torch.stack(diff_matrices)        # (n_layers, N, dim)
        means = batched.mean(dim=1)                 # (n_layers, dim)
        means_norms = means.norm(dim=-1)            # (n_layers,)

        if whitener is not None:
            # Mahalanobis branch: per-layer matvec via Woodbury through
            # ``LayerWhitener.mahalanobis_norm``.  Loop instead of batch
            # because each layer has its own ``Σ_L^{-1}`` and ``X_L``;
            # extraction is one-shot, not a hot path.  Layers absent from
            # the whitener fall back to Euclidean for that layer.
            scores_cpu = []
            for idx in range(n_layers):
                ref_L = max(ref_norms[idx], 1e-8)
                if whitener.covers(idx):
                    m_norm = whitener.mahalanobis_norm(idx, means[idx])
                    scores_cpu.append(m_norm / ref_L)
                else:
                    scores_cpu.append(float(means_norms[idx].item()) / ref_L)
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
                diff_matrices[idx],
                means[idx].detach().to(diff_matrices[idx].device),
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


def save_profile(
    profile: dict[int, torch.Tensor],
    path: str,
    metadata: dict,
) -> None:
    """Save a baked vector profile as .safetensors with a slim .json sidecar.

    ``metadata`` must contain at minimum:
        method            - str, e.g. "contrastive_pca" / "single_pair" / "merge" / "layer_means"

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

    tensors = {f"layer_{idx}": vec.contiguous().cpu() for idx, vec in profile.items()}
    save_file(tensors, str(path))

    from saklas import __version__ as _saklas_version
    from saklas.io.packs import PACK_FORMAT_VERSION
    sidecar: dict = {
        "format_version": PACK_FORMAT_VERSION,
        "method": metadata.get("method", "contrastive_pca"),
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


def load_profile(path: str) -> tuple[dict[int, torch.Tensor], dict]:
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
    passes base through unchanged, ``"~"`` drops the layer.  Layers absent
    from the whitener fall back to the Euclidean projection — the whitener
    may legitimately cover a subset of layers (e.g. SAE-only releases).
    Result tensors are cast back to the source dtype of ``base``.

    The returned dict shape matches :func:`extract_contrastive` so it
    plugs into ``SteeringManager.add_vector`` without adaptation.
    """
    if operator not in ("~", "|"):
        raise ValueError(f"unknown projection operator: {operator!r}")
    out: dict[int, torch.Tensor] = {}
    for layer, base_t in base.items():
        onto_t = onto.get(layer)
        if onto_t is None:
            if operator == "|":
                out[layer] = base_t
            continue
        # LEACE branch: whitener available + covers this layer.
        if whitener is not None and whitener.covers(layer):
            projected = whitener.leace_project(layer, base_t, onto_t, operator)
            # Drop the layer for ``~`` when ``onto`` is degenerate under
            # the Mahalanobis metric — leace_project returns a zero
            # tensor in that case, mirroring the Euclidean drop rule.
            if operator == "~" and torch.all(projected == 0):
                continue
            out[layer] = projected
            continue
        # Euclidean fallback (no whitener, or layer not covered).
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


def load_contrastive_pairs(dataset_path: str) -> dict:
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
