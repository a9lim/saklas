"""Extraction, saving, and loading of activation steering/probe vectors."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

log = logging.getLogger(__name__)


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
        and optionally ``"attention"`` (1, heads, seq, seq) from the last layer.
    """
    captured_hidden: dict[int, torch.Tensor] = {}
    captured_attn: dict[str, torch.Tensor] = {}

    def _make_hook(idx, is_last):
        def _hook(module, input, output):
            if isinstance(output, torch.Tensor):
                h = output
            else:
                h = output[0]
                if is_last and output_attentions and len(output) > 1:
                    captured_attn["attention"] = output[1]
            if h.device.type == "mps":
                torch.mps.synchronize()
            captured_hidden[idx] = h.clone()
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
        handles.append(layers[idx].register_forward_hook(
            _make_hook(idx, idx == n_layers - 1)
        ))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False,
                  output_attentions=output_attentions)
    finally:
        for h in handles:
            h.remove()
        if prev_attn is not None:
            model.set_attn_implementation(prev_attn)

    result = {"hidden": captured_hidden}
    if "attention" in captured_attn:
        result["attention"] = captured_attn["attention"]
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
        messages = [{"role": "assistant", "content": text}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            # Some chat templates require a user turn before assistant.
            # Fall back to minimal filler rather than contaminating with
            # a semantically loaded prompt.
            messages = [
                {"role": "user", "content": "."},
                {"role": "assistant", "content": text},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
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

    captured = _capture_all_hidden_states(model, layers, ids, output_attentions=True)
    hidden_per_layer = captured["hidden"]  # {idx: (1, seq, dim)}

    attn = captured.get("attention")
    if attn is not None:
        # attn: (1, heads, seq, seq) — last token's attention over all positions
        weights = attn[0, :, -1, :].mean(dim=0)  # (seq,)
        weights = weights * mask[0].float()
        weights = weights / weights.sum().clamp(min=1e-8)

        result = {}
        for idx, h in hidden_per_layer.items():
            h_f32 = h.float()  # (1, seq, dim)
            result[idx] = (h_f32[0] * weights.unsqueeze(-1)).sum(dim=0)  # (dim,)
        return result

    # Fallback: last-token pooling
    seq_len = mask[0].sum() - 1
    result = {}
    for idx, h in hidden_per_layer.items():
        result[idx] = h.float()[0, seq_len]  # (dim,)
    return result


def extract_contrastive(
    model,
    tokenizer,
    pairs: list[dict],
    layers=None,
    device=None,
) -> dict[int, tuple[torch.Tensor, float]]:
    """Contrastive direction extraction via PCA across all layers.

    Hooks every layer in the same 2N forward passes. For each layer,
    computes the first principal component of pos-neg differences and
    scores it by explained variance ratio (sigma_1 / sum(sigma)).

    Layers where score < 0.1 * max_score are dropped.

    Args:
        pairs: List of {"positive": str, "negative": str} prompt pairs.

    Returns:
        Profile dict mapping layer_idx -> (direction_vector, score)
        for layers above the signal threshold.
    """
    if device is None:
        device = next(model.parameters()).device

    n_layers = len(layers)
    # Accumulate per-layer diffs and norms
    diffs_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    norms_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    for pair in pairs:
        pos_all = _encode_and_capture_all(model, tokenizer, pair["positive"], layers, device)
        neg_all = _encode_and_capture_all(model, tokenizer, pair["negative"], layers, device)
        for idx in range(n_layers):
            diffs_per_layer[idx].append(pos_all[idx] - neg_all[idx])
            norms_per_layer[idx].append(pos_all[idx].norm())
            norms_per_layer[idx].append(neg_all[idx].norm())

    # Per-layer: compute direction and score
    profile: dict[int, tuple[torch.Tensor, float]] = {}
    scores: dict[int, float] = {}

    for idx in range(n_layers):
        diffs = diffs_per_layer[idx]
        ref_norm = torch.stack(norms_per_layer[idx]).mean().item() * 0.1
        diff_matrix = torch.stack(diffs)  # (N, dim)

        if len(diffs) < 2:
            # Single pair: use raw diff, score by norm (normalized to [0,1] below)
            direction = _normalize(diff_matrix.squeeze(0), ref_norm=ref_norm)
            scores[idx] = diff_matrix.squeeze(0).float().norm().item()
            profile[idx] = (direction, scores[idx])
            continue  # scores normalized after the loop

        # SVD on uncentered difference matrix
        svd_input = diff_matrix.float().cpu() if diff_matrix.device.type == "mps" else diff_matrix.float()
        _, s, Vh = torch.linalg.svd(svd_input, full_matrices=False)
        direction = Vh[0].to(diff_matrix.device)  # (dim,)

        # Orient so "positive" stays positive: majority vote across pairs.
        # Mean-diff orientation is fragile — one outlier pair with large
        # magnitude can flip the entire vector.  Majority vote counts how
        # many individual diffs agree with the current sign and flips only
        # when the majority disagree.
        dots = diff_matrix @ direction  # (N,)
        if (dots < 0).sum() > (dots > 0).sum():
            direction = -direction

        score = (s[0] / s.sum()).item()
        scores[idx] = score
        profile[idx] = (_normalize(direction, ref_norm=ref_norm), score)

    # Single-pair scores are raw diff norms — normalize to [0, 1] so they're
    # on the same scale as multi-pair explained-variance-ratio scores.
    if len(pairs) < 2 and scores:
        max_raw = max(scores.values())
        if max_raw > 1e-8:
            for idx in scores:
                scores[idx] /= max_raw
                vec, _ = profile[idx]
                profile[idx] = (vec, scores[idx])

    # Adaptive threshold: mean score adapts to the distribution shape.
    # Peaked (one dominant layer): mean is low, keeps just the peak.
    # Flat (signal spread evenly): mean ≈ each score, keeps ~half.
    # The fixed 0.1 * max approach over-prunes flat distributions and
    # under-prunes peaked ones.
    if scores:
        threshold = sum(scores.values()) / len(scores)
        profile = {idx: v for idx, v in profile.items() if v[1] >= threshold}

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

    tensors = {f"layer_{idx}": vec.contiguous().cpu() for idx, (vec, _) in profile.items()}
    save_file(tensors, str(path))

    scores = {str(idx): score for idx, (_, score) in profile.items()}
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
    safe_concept = re.sub(r'[^\w\-.]', '_', concept)
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
