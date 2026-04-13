"""Extraction, saving, and loading of activation steering/probe vectors."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

log = logging.getLogger(__name__)

# Skip the chat template for extraction when it adds more than this many
# tokens of overhead (e.g. Ministral injects a ~500-token system prompt).
# The overhead cancels in contrastive diffs but wastes memory per pass.
_MAX_TEMPLATE_OVERHEAD = 100

# Keyed by id(tokenizer).  Object IDs can be reused after GC, so this
# cache is only safe when a single tokenizer lives for the session lifetime
# (which is the case in both the TUI and the API server).
_template_overhead_cache: dict[int, int] = {}
_SAFE_CONCEPT_RE = re.compile(r'[^\w\-.]')


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
    # Compute norm in float32 to avoid fp16 overflow: for hidden_dim=2048
    # with element magnitudes ~6, the sum-of-squares (73728) exceeds
    # fp16 max (65504), producing Inf and zeroing the entire vector.
    v_f32 = v.float()
    unit = (v_f32 / v_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)).to(v.dtype)
    if ref_norm is not None:
        return unit * ref_norm
    return unit


def _capture_all_hidden_states(model, layers, input_ids, output_attentions=False):
    """Run a single-sequence forward pass capturing hidden states at ALL layers.

    Uses ``use_cache=False`` to avoid polluting any persistent KV cache.
    When *output_attentions* is True, temporarily switches the model to
    eager attention (SDPA cannot return attention weights), then restores
    the original implementation after the forward pass.

    Returns:
        dict with ``"hidden"`` mapping layer index to (1, seq, dim) tensors,
        and optionally ``"attention"`` mapping layer index to
        (1, heads, seq, seq) tensors.
    """
    captured_hidden: dict[int, torch.Tensor] = {}
    captured_attn: dict[int, torch.Tensor] = {}

    def _make_hook(idx):
        def _hook(module, input, output):
            if isinstance(output, torch.Tensor):
                h = output
            else:
                h = output[0]
                if output_attentions and len(output) > 1:
                    captured_attn[idx] = output[1]
            # No .clone() — with use_cache=False and inference_mode() the
            # residual-stream tensors are fresh allocations at each layer
            # boundary (residual add produces a new tensor).  Detach severs
            # the autograd graph reference so the rest of the forward pass
            # can't invalidate the data.
            captured_hidden[idx] = h.detach()
        return _hook

    # SDPA doesn't support output_attentions; swap to eager for extraction.
    prev_attn = None
    if output_attentions and hasattr(model, "set_attn_implementation"):
        prev_attn = getattr(model.config, "_attn_implementation_internal",
                            getattr(model.config, "_attn_implementation", None))
        if prev_attn and prev_attn != "eager":
            model.set_attn_implementation("eager")
        else:
            prev_attn = None  # already eager, nothing to restore

    n_layers = len(layers)
    handles = []
    for idx in range(n_layers):
        handles.append(layers[idx].register_forward_hook(_make_hook(idx)))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False,
                  output_attentions=output_attentions)
        # Single sync after the full forward pass — lazy backends (MPS)
        # may not have materialised tensor data yet.  One flush here
        # replaces the per-layer sync that used to stall the pipeline
        # N_layers times per pass.
        if input_ids.device.type == "mps":
            torch.mps.synchronize()
    finally:
        for h in handles:
            h.remove()
        if prev_attn is not None:
            model.set_attn_implementation(prev_attn)

    result = {"hidden": captured_hidden}
    if captured_attn:
        result["attention"] = captured_attn
    return result


def _encode_and_capture_all(model, tokenizer, text, layers, device):
    """Tokenize text, run forward pass, return attention-weighted hidden state per layer in fp32.

    For instruction-tuned models (those with a chat template), wraps the text
    as an assistant response so the extraction happens in the model's actual
    generation regime.  Base models get the raw string.

    Uses the last token's self-attention distribution (averaged across heads,
    from the last layer) as pooling weights — the model's own saliency signal
    for which positions matter.  Falls back to last-token pooling if attention
    capture fails.

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
            # tokens contaminate the attention-weighted pooling.
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
    enc = tokenizer(text, return_tensors="pt", return_attention_mask=True, add_special_tokens=False)
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    if ids.numel() == 0 or (mask.sum() == 0):
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.eos_token_id or 0
        ids = torch.tensor([[bos_id]])
        mask = torch.ones_like(ids)
    ids = ids.to(device)
    mask = mask.to(device)

    try:
        captured = _capture_all_hidden_states(model, layers, ids, output_attentions=True)
    except RuntimeError:
        # Attention capture can OOM on memory-constrained devices (MPS
        # with large models / long sequences).  Fall back to last-token
        # pooling which skips attention storage entirely.
        if ids.device.type == "mps":
            torch.mps.empty_cache()
        captured = _capture_all_hidden_states(model, layers, ids, output_attentions=False)
    hidden_per_layer = captured["hidden"]  # {idx: (1, seq, dim)}

    attn_per_layer = captured.get("attention")  # {idx: (1, heads, seq, seq)}
    if attn_per_layer:
        result = {}
        for idx, h in hidden_per_layer.items():
            h_f32 = h.float()  # (1, seq, dim)
            attn = attn_per_layer.get(idx)
            if attn is not None:
                # Use this layer's own attention: last token's view, averaged across heads
                weights = attn[0, :, -1, :].mean(dim=0)  # (seq,)
                # Build mask from hidden-state length — VLM wrappers may
                # reshape the sequence before the language-model layers,
                # so the tokenizer mask length can differ.
                seq = h_f32.shape[1]
                mask_f = mask[0, :seq].float() if mask.shape[1] >= seq else torch.ones(seq, device=h.device)
                weights = weights * mask_f
                weights = weights / weights.sum().clamp(min=1e-8)
                result[idx] = (h_f32[0] * weights.unsqueeze(-1)).sum(dim=0)  # (dim,)
            else:
                # Layer didn't produce attention weights — last-token fallback
                result[idx] = h_f32[0, -1]
        return result

    # Fallback: last-token pooling
    return {idx: h.float()[0, -1] for idx, h in hidden_per_layer.items()}


import functools as _functools
from importlib import resources as _resources


@_functools.cache
def _load_neutral_prompts() -> list[str]:
    """Load neutral prompts, preferring a user override at ~/.saklas/neutral_statements.json."""
    from saklas.paths import neutral_statements_path
    user_path = neutral_statements_path()
    if user_path.exists():
        with open(user_path) as f:
            return json.load(f)
    with _resources.files("saklas.data").joinpath("neutral_statements.json").open() as f:
        return json.load(f)


class _NeutralPromptsProxy:
    """Sequence-like proxy so existing call sites (``for p in _NEUTRAL_PROMPTS``,
    ``len(_NEUTRAL_PROMPTS)``) keep working while the source moves to a JSON file."""

    def __iter__(self):
        return iter(_load_neutral_prompts())

    def __len__(self):
        return len(_load_neutral_prompts())

    def __getitem__(self, i):
        return _load_neutral_prompts()[i]


_NEUTRAL_PROMPTS = _NeutralPromptsProxy()


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

    for text in _NEUTRAL_PROMPTS:
        per_layer = _encode_and_capture_all(model, tokenizer, text, layers, device)
        for idx in range(n_layers):
            if idx not in sums:
                sums[idx] = per_layer[idx]
            else:
                sums[idx] = sums[idx] + per_layer[idx]
        del per_layer
        if _mps:
            torch.mps.empty_cache()

    n = len(_NEUTRAL_PROMPTS)
    return {idx: sums[idx] / n for idx in range(n_layers)}


def extract_contrastive(
    model,
    tokenizer,
    pairs: list[dict],
    layers,
    device=None,
) -> dict[int, tuple[torch.Tensor, float]]:
    """Contrastive direction extraction via PCA across all layers.

    Hooks every layer in the same 2N forward passes. For each layer,
    computes the first principal component of pos-neg differences and
    scores it by explained variance ratio (sigma_1 / sum(sigma)).

    Args:
        pairs: List of {"positive": str, "negative": str} prompt pairs.

    Returns:
        Profile dict mapping layer_idx -> (direction_vector, score)
        for all layers.
    """
    if device is None:
        device = next(model.parameters()).device

    n_layers = len(layers)
    # Accumulate per-layer diffs and running norm sums.
    # norm_sums is a GPU tensor to avoid per-layer .item() sync points
    # (was 2 * N_pairs * N_layers GPU→CPU syncs, now 0 during the loop).
    diffs_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    norm_sums = torch.zeros(n_layers, device=device, dtype=torch.float32)

    # On MPS, keep diffs on CPU — SVD runs there anyway, and the
    # model already occupies most of the unified memory budget.
    _mps = device.type == "mps"
    diff_device = "cpu" if _mps else device

    for pair in pairs:
        pos_all = _encode_and_capture_all(model, tokenizer, pair["positive"], layers, device)
        neg_all = _encode_and_capture_all(model, tokenizer, pair["negative"], layers, device)
        for idx in range(n_layers):
            # Cast to float32 before differencing — fp16 subtraction
            # loses precision for close vectors, producing degenerate
            # diff matrices that cause LAPACK SVD errors (SLASCL).
            p, n = pos_all[idx].float(), neg_all[idx].float()
            diffs_per_layer[idx].append((p - n).to(diff_device))
            norm_sums[idx] += p.norm() + n.norm()
        # Free forward-pass intermediates (attention maps, hidden states)
        # before the next pair — MPS doesn't release memory eagerly.
        del pos_all, neg_all
        if _mps:
            torch.mps.empty_cache()

    # Per-layer: compute direction and score
    profile: dict[int, tuple[torch.Tensor, float]] = {}
    n_pairs = len(pairs)

    n_norm_samples = n_pairs * 2  # pos + neg per pair
    # Single GPU→CPU transfer for all layer norms
    norm_sums_cpu = norm_sums.tolist()

    if n_pairs < 2:
        # Single pair: score as diff norm relative to activation magnitude.
        # This produces values in roughly the same range as the
        # explained-variance-ratio used for multi-pair extraction
        # (typically 0.01–0.4), so single-pair and multi-pair profiles
        # contribute comparably when used as probes or weighted by score.
        for idx in range(n_layers):
            diff_vec = diffs_per_layer[idx][0]
            ref_norm = norm_sums_cpu[idx] / n_norm_samples
            direction = _normalize(diff_vec, ref_norm=ref_norm)
            diff_norm = diff_vec.float().norm().item()
            activation_norm = norm_sums_cpu[idx]  # pos_norm + neg_norm
            score = diff_norm / max(activation_norm, 1e-8)
            profile[idx] = (direction, score)
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
        svd_input = batched
        _, S, Vh = torch.linalg.svd(svd_input, full_matrices=False)
        # S: (n_layers, min(N,dim)), Vh: (n_layers, min(N,dim), dim)

        for idx in range(n_layers):
            direction = Vh[idx, 0].to(device)  # (dim,)

            # Orient so "positive" stays positive: majority vote across pairs.
            dots = diff_matrices[idx] @ direction.to(diff_matrices[idx].device)
            if (dots < 0).sum() > (dots > 0).sum():
                direction = -direction

            score = (S[idx, 0] / S[idx].sum()).item()
            profile[idx] = (_normalize(direction, ref_norm=ref_norms[idx]), score)

    return profile


def save_profile(
    profile: dict[int, tuple[torch.Tensor, float]],
    path: str,
    metadata: dict,
) -> None:
    """Save a vector profile as .safetensors with .json metadata sidecar.

    The safetensors file contains keys ``"layer_{i}"`` for each active layer.
    The JSON sidecar contains ``metadata`` plus ``"scores"`` mapping layer
    indices to their signal strength scores.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensors, scores = {}, {}
    for idx, (vec, score) in profile.items():
        tensors[f"layer_{idx}"] = vec.contiguous().cpu()
        scores[str(idx)] = score
    save_file(tensors, str(path))
    meta = {**metadata, "scores": scores}
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Saved profile (%d layers) to %s", len(profile), path)


def load_profile(path: str) -> tuple[dict[int, tuple[torch.Tensor, float]], dict]:
    """Load a vector profile and its metadata.

    Returns:
        (profile dict mapping layer_idx -> (vector, score), metadata dict)
    """
    path = Path(path)
    tensors = load_file(str(path))

    meta_path = path.with_suffix(".json")
    with open(meta_path) as f:
        metadata = json.load(f)

    scores = metadata.get("scores", {})
    profile = {}
    for key, tensor in tensors.items():
        # Keys are "layer_{i}"
        idx = int(key.split("_", 1)[1])
        score = float(scores.get(str(idx), 1.0))
        profile[idx] = (tensor, score)

    return profile, metadata


def get_cache_path(
    cache_dir: str,
    model_id: str,
    concept: str,
) -> str:
    """Deterministic cache path for a vector profile.

    Returns:
        Path like ``{cache_dir}/{model_name}/{concept}.safetensors``
    """
    model_name = model_id.replace("/", "_")
    safe_concept = _SAFE_CONCEPT_RE.sub('_', concept)
    filename = f"{safe_concept}.safetensors"
    return str(Path(cache_dir) / model_name / filename)


def load_contrastive_pairs(dataset_path: str) -> dict:
    """Load a contrastive-pairs JSON dataset.

    Expected schema::

        {
            "name": str,
            "description": str,
            "category": str,
            "pairs": [{"positive": str, "negative": str}, ...]
        }

    Returns:
        The parsed dict.
    """
    with open(dataset_path) as f:
        return json.load(f)
