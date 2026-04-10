"""Extraction, saving, and loading of activation steering/probe vectors."""

from __future__ import annotations

import json
import logging
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


def extract_actadd(
    model,
    tokenizer,
    concept: str,
    layer_idx: int,
    baseline: str = "",
    layers=None,
) -> torch.Tensor:
    """Single-concept ActAdd extraction (Turner et al., 2023).

    Tokenizes concept and baseline **separately** (no batching) to avoid
    degenerate attention from fully-masked padding when the baseline is
    shorter.  Each text gets its own forward pass.
    """
    device = next(model.parameters()).device

    def _encode_single(text: str) -> torch.Tensor:
        """Tokenize a single string, guaranteeing ≥1 real token."""
        enc = tokenizer(
            text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )
        ids = enc["input_ids"]
        # If the tokenizer produced nothing (empty string with no BOS),
        # fall back to the BOS token so the model has valid input.
        if ids.numel() == 0 or (enc["attention_mask"].sum() == 0):
            bos = tokenizer.bos_token_id
            if bos is None:
                bos = tokenizer.eos_token_id or 0
            ids = torch.tensor([[bos]])
        return ids.to(device)

    concept_ids = _encode_single(concept)
    baseline_ids = _encode_single(baseline)

    if layers is not None:
        layer_mod = layers[layer_idx]
        h_concept = _capture_hidden_states_single(model, layer_mod, concept_ids)
        h_baseline = _capture_hidden_states_single(model, layer_mod, baseline_ids)
    else:
        with torch.inference_mode():
            out_c = model(input_ids=concept_ids, output_hidden_states=True, use_cache=False)
            out_b = model(input_ids=baseline_ids, output_hidden_states=True, use_cache=False)
        h_concept = out_c.hidden_states[layer_idx + 1]
        h_baseline = out_b.hidden_states[layer_idx + 1]

    # Mean-pool over token positions (no padding, so simple mean).
    pos_mean = h_concept.float().mean(dim=1, keepdim=True)   # (1, 1, dim)
    neg_mean = h_baseline.float().mean(dim=1, keepdim=True)
    # Squeeze out seq dim → (1, dim)
    pos_mean = pos_mean.squeeze(1)
    neg_mean = neg_mean.squeeze(1)

    diff = pos_mean - neg_mean  # (1, dim)

    # Scale to 10% of the mean hidden-state norm.
    ref_norm = (
        torch.stack([pos_mean, neg_mean]).norm(dim=-1).mean().item() * 0.1
    )

    return _normalize(diff.to(h_concept.dtype), ref_norm=ref_norm).squeeze(0)


def extract_actadd_batched(
    model,
    tokenizer,
    concepts: list[str],
    layer_idx: int,
    baseline: str = "",
    layers=None,
) -> dict[str, torch.Tensor]:
    """Batch multiple ActAdd extractions as separate forward passes.

    Runs each concept and the baseline through individual forward passes
    to avoid padding-induced attention corruption on multimodal models.

    Returns:
        Dict mapping concept string -> unit steering vector.
    """
    device = next(model.parameters()).device
    # Ensure baseline produces real tokens (empty string → BOS fallback).
    if not baseline.strip():
        bos = tokenizer.bos_token
        baseline = bos if bos else " "

    def _encode_and_capture(text: str) -> torch.Tensor:
        enc = tokenizer(
            text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )
        ids = enc["input_ids"]
        if ids.numel() == 0 or (enc["attention_mask"].sum() == 0):
            bos_id = tokenizer.bos_token_id
            if bos_id is None:
                bos_id = tokenizer.eos_token_id or 0
            ids = torch.tensor([[bos_id]])
        ids = ids.to(device)
        if layers is not None:
            h = _capture_hidden_states_single(model, layers[layer_idx], ids)
        else:
            with torch.inference_mode():
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[layer_idx + 1]
        return h.float().mean(dim=1).squeeze(0)  # (dim,)

    neg_mean = _encode_and_capture(baseline)
    neg_norm = neg_mean.norm().item()
    result: dict[str, torch.Tensor] = {}

    for concept in concepts:
        pos_mean = _encode_and_capture(concept)
        diff = pos_mean - neg_mean
        ref_norm = (pos_mean.norm().item() + neg_norm) / 2 * 0.1
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

    Runs each prompt through a separate forward pass to avoid
    padding-induced attention corruption on multimodal models.

    Args:
        pairs: List of {"positive": str, "negative": str} prompt pairs.
        layer_idx: Which layer to extract from.

    Returns:
        L2-normalized mean contrastive vector.
    """
    device = next(model.parameters()).device

    def _encode_and_capture(text: str) -> torch.Tensor:
        enc = tokenizer(
            text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )
        ids = enc["input_ids"]
        if ids.numel() == 0 or (enc["attention_mask"].sum() == 0):
            bos_id = tokenizer.bos_token_id
            if bos_id is None:
                bos_id = tokenizer.eos_token_id or 0
            ids = torch.tensor([[bos_id]])
        ids = ids.to(device)
        if layers is not None:
            h = _capture_hidden_states_single(model, layers[layer_idx], ids)
        else:
            with torch.inference_mode():
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[layer_idx + 1]
        return h.float().mean(dim=1).squeeze(0)  # (dim,)

    diffs = []
    norms = []
    for pair in pairs:
        pos_mean = _encode_and_capture(pair["positive"])
        neg_mean = _encode_and_capture(pair["negative"])
        diffs.append(pos_mean - neg_mean)
        norms.append(pos_mean.norm().item())
        norms.append(neg_mean.norm().item())

    mean_diff = torch.stack(diffs).mean(dim=0, keepdim=True)  # (1, dim)
    ref_norm = sum(norms) / len(norms) * 0.1

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
