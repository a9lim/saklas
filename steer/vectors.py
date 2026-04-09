"""Extraction, saving, and loading of activation steering/probe vectors."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

log = logging.getLogger(__name__)


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states over non-padding token positions.

    Args:
        hidden_states: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len), 1 for real tokens, 0 for padding.

    Returns:
        (batch, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


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


def _capture_hidden_states(model, layer, enc):
    """Run a forward pass and capture hidden states at a specific layer via hook.

    Using a hook is more robust than ``output_hidden_states`` because
    multimodal wrappers (Gemma 3/4, etc.) can re-index the hidden-state
    list, causing an off-by-N mismatch between the extracted vector and
    the layer the steering hook is attached to.
    """
    captured = {}

    def _hook(module, input, output):
        # Handle both bare-tensor (Gemma 4) and tuple outputs
        h = output if isinstance(output, torch.Tensor) else output[0]
        # Clone to avoid corruption from downstream in-place ops
        captured["hidden"] = h.clone()

    handle = layer.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model(**enc)
    finally:
        handle.remove()
    return captured["hidden"]


def extract_actadd(
    model,
    tokenizer,
    concept: str,
    layer_idx: int,
    baseline: str = "",
    layers=None,
) -> torch.Tensor:
    """Single-concept ActAdd extraction (Turner et al., 2023).

    Tokenizes concept and baseline in a single batch, extracts hidden states
    at the given layer, mean-pools across token positions, mean-centers, and
    returns the L2-normalized difference vector.
    """
    device = next(model.parameters()).device

    enc = tokenizer(
        [concept, baseline],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    if layers is not None:
        hidden = _capture_hidden_states(model, layers[layer_idx], enc)
    else:
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[layer_idx]

    mask = enc["attention_mask"]  # (2, seq)
    pooled = _mean_pool(hidden, mask)  # (2, dim)

    # Use the mean hidden-state norm as reference so the steering vector
    # lives at the same scale as the activations (critical for architectures
    # with per-layer output scaling like Gemma 4's layer_scalar).
    ref_norm = pooled.norm(dim=-1).mean().item()

    pos_mean = pooled[0:1]  # (1, dim)
    neg_mean = pooled[1:2]

    diff = pos_mean - neg_mean  # (1, dim)
    return _normalize(diff, ref_norm=ref_norm).squeeze(0)  # (dim,)


def extract_actadd_batched(
    model,
    tokenizer,
    concepts: list[str],
    layer_idx: int,
    baseline: str = "",
    layers=None,
) -> dict[str, torch.Tensor]:
    """Batch multiple ActAdd extractions into fewer forward passes.

    Tokenizes all concepts plus the baseline as a single batch, runs one
    forward pass, and extracts per-concept steering vectors.

    Returns:
        Dict mapping concept string -> unit steering vector.
    """
    device = next(model.parameters()).device
    texts = concepts + [baseline]

    enc = tokenizer(
        texts,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    if layers is not None:
        hidden = _capture_hidden_states(model, layers[layer_idx], enc)
    else:
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[layer_idx]

    mask = enc["attention_mask"]  # (batch, seq)
    pooled = _mean_pool(hidden, mask)  # (batch, dim)

    ref_norm = pooled.norm(dim=-1).mean().item()
    neg_mean = pooled[-1]  # baseline is last in batch
    result: dict[str, torch.Tensor] = {}

    for i, concept in enumerate(concepts):
        diff = pooled[i] - neg_mean
        result[concept] = _normalize(diff.unsqueeze(0), ref_norm=ref_norm).squeeze(0)

    return result


def extract_caa(
    model,
    tokenizer,
    pairs: list[dict],
    layer_idx: int,
    layers=None,
) -> torch.Tensor:
    """Contrastive Activation Addition (Rimsky et al., 2023).

    Args:
        pairs: List of {"positive": str, "negative": str} prompt pairs.
        layer_idx: Which layer to extract from.

    Returns:
        L2-normalized mean contrastive vector.
    """
    device = next(model.parameters()).device
    n = len(pairs)

    positives = [p["positive"] for p in pairs]
    negatives = [p["negative"] for p in pairs]

    # Single batch: positives then negatives
    enc = tokenizer(
        positives + negatives,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    if layers is not None:
        hidden = _capture_hidden_states(model, layers[layer_idx], enc)
    else:
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[layer_idx]

    mask = enc["attention_mask"]  # (2*n, seq)
    pooled = _mean_pool(hidden, mask)  # (2*n, dim)

    ref_norm = pooled.norm(dim=-1).mean().item()
    pos_pooled = pooled[:n]  # (n, dim)
    neg_pooled = pooled[n:]  # (n, dim)

    diffs = pos_pooled - neg_pooled  # (n, dim)
    mean_diff = diffs.mean(dim=0, keepdim=True)  # (1, dim)

    return _normalize(mean_diff, ref_norm=ref_norm).squeeze(0)  # (dim,)


def save_vector(vector: torch.Tensor, path: str, metadata: dict) -> None:
    """Save a steering vector as .safetensors with .json metadata sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_file({"vector": vector.contiguous().cpu()}, str(path))

    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Saved vector to %s", path)


def load_vector(path: str) -> tuple[torch.Tensor, dict]:
    """Load a steering vector and its metadata.

    Returns:
        (vector tensor, metadata dict)
    """
    path = Path(path)
    tensors = load_file(str(path))
    vector = tensors["vector"]

    meta_path = path.with_suffix(".json")
    with open(meta_path) as f:
        metadata = json.load(f)

    return vector, metadata


def get_cache_path(
    cache_dir: str,
    model_id: str,
    concept: str,
    layer_idx: int,
    method: str,
) -> str:
    """Deterministic cache path for a steering vector.

    Returns:
        Path like ``{cache_dir}/{model_name}/{concept}_{layer}_{method}.safetensors``
    """
    model_name = model_id.replace("/", "_")
    filename = f"{concept}_{layer_idx}_{method}.safetensors"
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
