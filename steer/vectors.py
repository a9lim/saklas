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



def _capture_hidden_states_single(model, layer, input_ids, output_attentions=False):
    """Run a single-sequence forward pass and capture hidden states at *layer*.

    Uses ``use_cache=False`` to avoid polluting any persistent KV cache.
    When *output_attentions* is True, temporarily switches the model to
    eager attention (SDPA cannot return attention weights), then restores
    the original implementation after the forward pass.

    Returns:
        dict with ``"hidden"`` (1, seq, dim) and optionally ``"attention"``
        (1, heads, seq, seq).
    """
    captured = {}

    def _hook(module, input, output):
        if isinstance(output, torch.Tensor):
            h = output
        else:
            h = output[0]
            if output_attentions and len(output) > 1:
                captured["attention"] = output[1]
        if h.device.type == "mps":
            torch.mps.synchronize()
        captured["hidden"] = h.clone()

    # SDPA doesn't support output_attentions; swap to eager for extraction.
    prev_attn = None
    if output_attentions and hasattr(model, "set_attn_implementation"):
        prev_attn = getattr(model.config, "_attn_implementation_internal",
                            getattr(model.config, "_attn_implementation", None))
        if prev_attn and prev_attn != "eager":
            model.set_attn_implementation("eager")
        else:
            prev_attn = None  # already eager, nothing to restore

    handle = layer.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False,
                  output_attentions=output_attentions)
    finally:
        handle.remove()
        if prev_attn is not None:
            model.set_attn_implementation(prev_attn)
    return captured


def _encode_and_capture(model, tokenizer, text, layer_idx, layers, device):
    """Tokenize text, run forward pass, return attention-weighted hidden state in fp32.

    For instruction-tuned models (those with a chat template), wraps the text
    as an assistant response so the extraction happens in the model's actual
    generation regime.  Base models get the raw string.

    Uses the last token's self-attention distribution (averaged across heads)
    as pooling weights — the model's own saliency signal for which positions
    matter.  Falls back to last-token pooling if attention capture fails.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        messages = [
            {"role": "user", "content": "Continue the conversation."},
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

    captured = _capture_hidden_states_single(
        model, layers[layer_idx], ids, output_attentions=True,
    )
    h = captured["hidden"].float()  # (1, seq, dim)

    attn = captured.get("attention")
    if attn is not None:
        # attn: (1, heads, seq, seq) — last token's attention over all positions
        weights = attn[0, :, -1, :].mean(dim=0)  # (seq,)
        weights = weights * mask[0].float()
        weights = weights / weights.sum().clamp(min=1e-8)
        return (h[0] * weights.unsqueeze(-1)).sum(dim=0)  # (dim,)

    # Fallback: last-token pooling (still better than mean for causal LMs)
    seq_len = mask[0].sum() - 1
    return h[0, seq_len]  # (dim,)


def extract_contrastive(
    model,
    tokenizer,
    pairs: list[dict],
    layer_idx: int,
    layers=None,
    device=None,
) -> torch.Tensor:
    """Contrastive direction extraction via PCA (Zou et al., 2023).

    Computes pos−neg difference vectors for each pair, then takes
    the first principal component — the direction of maximum variance
    across the differences.  More robust than mean-difference (CAA)
    when individual pairs are noisy.

    Runs each prompt through a separate forward pass to avoid
    padding-induced attention corruption on multimodal models.

    Args:
        pairs: List of {"positive": str, "negative": str} prompt pairs.
        layer_idx: Which layer to extract from.

    Returns:
        Direction vector scaled to 10% of mean hidden-state norm.
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

    diff_matrix = torch.stack(diffs)  # (N, dim)
    ref_norm = torch.stack(norms).mean().item() * 0.1

    if len(diffs) < 2:
        # Can't do PCA with a single vector; fall back to the diff itself.
        return _normalize(diff_matrix.squeeze(0), ref_norm=ref_norm)

    # PCA: first principal component of centered difference vectors.
    # MPS lacks QR decomposition (aten::linalg_qr), so move to CPU there.
    diff_matrix = diff_matrix - diff_matrix.mean(dim=0)
    pca_input = diff_matrix.cpu() if diff_matrix.device.type == "mps" else diff_matrix
    _, _, V = torch.pca_lowrank(pca_input, q=1, niter=5)
    direction = V[:, 0].to(diff_matrix.device)  # (dim,)

    # pca_lowrank returns an unsigned direction; orient it so that
    # "positive" stays positive (align with the mean difference).
    mean_diff = torch.stack(diffs).mean(dim=0)
    if direction @ mean_diff < 0:
        direction = -direction

    return _normalize(direction, ref_norm=ref_norm)


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
) -> str:
    """Deterministic cache path for a steering vector.

    Returns:
        Path like ``{cache_dir}/{model_name}/{concept}_{layer}.safetensors``
    """
    model_name = model_id.replace("/", "_")
    safe_concept = re.sub(r'[^\w\-.]', '_', concept)
    filename = f"{safe_concept}_{layer_idx}.safetensors"
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
