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



def _capture_hidden_states_single(model, layer, input_ids):
    """Run a single-sequence forward pass and capture hidden states at *layer*.

    Uses ``use_cache=False`` to avoid polluting any persistent KV cache.
    """
    captured = {}

    def _hook(module, input, output):
        h = output if isinstance(output, torch.Tensor) else output[0]
        if h.device.type == "mps":
            torch.mps.synchronize()
        captured["hidden"] = h.clone()

    handle = layer.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    return captured["hidden"]  # (1, seq, dim)


def _encode_and_capture(model, tokenizer, text, layer_idx, layers, device):
    """Tokenize text, ensure at least 1 real token, run forward pass, return mean-pooled hidden state in fp32."""
    enc = tokenizer(text, return_tensors="pt", return_attention_mask=True, add_special_tokens=True)
    ids = enc["input_ids"]
    if ids.numel() == 0 or (enc["attention_mask"].sum() == 0):
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.eos_token_id or 0
        ids = torch.tensor([[bos_id]])
    ids = ids.to(device)
    h = _capture_hidden_states_single(model, layers[layer_idx], ids)
    return h.float().mean(dim=1).squeeze(0)  # (dim,)


def extract_actadd(
    model,
    tokenizer,
    concept: str,
    layer_idx: int,
    baseline: str = "",
    layers=None,
    device=None,
) -> torch.Tensor:
    """Single-concept ActAdd extraction (Turner et al., 2023).

    Tokenizes concept and baseline **separately** (no batching) to avoid
    degenerate attention from fully-masked padding when the baseline is
    shorter.  Each text gets its own forward pass.
    """
    if device is None:
        device = next(model.parameters()).device

    pos_mean = _encode_and_capture(model, tokenizer, concept, layer_idx, layers, device)
    neg_mean = _encode_and_capture(model, tokenizer, baseline, layer_idx, layers, device)

    diff = pos_mean - neg_mean  # (dim,)

    # Scale to 10% of the mean hidden-state norm.
    ref_norm = (pos_mean.norm().item() + neg_mean.norm().item()) / 2 * 0.1

    return _normalize(diff, ref_norm=ref_norm)


def extract_caa(
    model,
    tokenizer,
    pairs: list[dict],
    layer_idx: int,
    layers=None,
    device=None,
) -> torch.Tensor:
    """Contrastive Activation Addition (Rimsky et al., 2023).

    Runs each prompt through a separate forward pass to avoid
    padding-induced attention corruption on multimodal models.

    Args:
        pairs: List of {"positive": str, "negative": str} prompt pairs.
        layer_idx: Which layer to extract from.

    Returns:
        L2-normalized mean contrastive vector.
    """
    if device is None:
        device = next(model.parameters()).device

    diffs = []
    norms = []
    for pair in pairs:
        pos_mean = _encode_and_capture(model, tokenizer, pair["positive"], layer_idx, layers, device)
        neg_mean = _encode_and_capture(model, tokenizer, pair["negative"], layer_idx, layers, device)
        diffs.append(pos_mean - neg_mean)
        norms.append(pos_mean.norm())
        norms.append(neg_mean.norm())

    mean_diff = torch.stack(diffs).mean(dim=0)  # (dim,)
    ref_norm = torch.stack(norms).mean().item() * 0.1

    return _normalize(mean_diff, ref_norm=ref_norm)


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
    safe_concept = re.sub(r'[^\w\-.]', '_', concept)
    filename = f"{safe_concept}_{layer_idx}_{method}.safetensors"
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
