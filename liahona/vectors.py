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
    result = {}
    for idx, h in hidden_per_layer.items():
        result[idx] = h.float()[0, -1]  # (dim,)
    return result


_NEUTRAL_PROMPTS = [
    # Simple facts
    "Water freezes at zero degrees Celsius.",
    "There are seven days in a week.",
    "The chemical symbol for gold is Au.",
    # Spatial / physical descriptions
    "The book is on the table next to the lamp.",
    "A narrow path winds between the two hills.",
    "The parking lot wraps around the back of the building.",
    # Everyday actions (varied subjects)
    "She walked to the store and bought some groceries.",
    "The technician replaced the filter and restarted the system.",
    "He sorted the mail into three piles before lunch.",
    # Temporal / scheduling
    "The library closes at nine on weekdays.",
    "Flights to Denver depart twice daily, morning and evening.",
    "The semester runs from September through mid-December.",
    # Process / how-things-work
    "Bread dough rises when yeast converts sugar into carbon dioxide.",
    "A compiler translates source code into machine instructions.",
    "Seeds germinate when moisture and temperature reach the right levels.",
    # Quantitative / measurement
    "The bridge spans roughly four hundred meters across the river.",
    "An adult human skeleton contains two hundred and six bones.",
    "Light travels about three hundred thousand kilometers per second.",
    # Conditional / if-then
    "If the temperature drops below freezing, the pipes may need insulation.",
    "Connecting the cables in the wrong order can trip the breaker.",
    "The sensor activates when motion is detected within five meters.",
    # Multi-clause / compound
    "The train was delayed by twenty minutes, so most passengers waited on the platform.",
    "After the rain stopped, the crew resumed paving the road.",
    "She finished the report, emailed it to the team, and left for the day.",
    # Questions
    "What time does the next bus arrive at the central station?",
    "How many liters of paint are needed to cover the back wall?",
    "Which floor is the records office on?",
    # Instructions / imperatives
    "Preheat the oven to one hundred and eighty degrees before adding the dish.",
    "Turn left at the second intersection, then continue straight for two blocks.",
    "Press and hold the reset button for five seconds until the light turns green.",
    # Abstract / categorical
    "Mammals are warm-blooded vertebrates that nurse their young.",
    "Most programming languages distinguish between integers and floating-point numbers.",
    "Supply and demand determine the market price of a commodity.",
    # Sensory / environmental
    "The hum of the air conditioner was the only sound in the room.",
    "Dry leaves scraped across the pavement in the wind.",
    "The smell of fresh paint lingered in the hallway all morning.",
    # Historical / past-tense narrative
    "The canal was completed in 1914 after a decade of construction.",
    "Early telephone networks connected operators manually with patch cords.",
    "The expedition reached the summit on the third attempt.",
    # Comparative / relational
    "Aluminum is lighter than steel but less rigid under the same load.",
    "The express route is shorter, though the local road has less traffic.",
    "Handwritten notes tend to aid recall more than typed ones.",
    # Hypothetical / generic conditional
    "A fully charged battery should last roughly eight hours under normal use.",
    "Most packages arrive within three to five business days.",
    "The average commute in the metro area takes about forty minutes.",
]


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

    for text in _NEUTRAL_PROMPTS:
        per_layer = _encode_and_capture_all(model, tokenizer, text, layers, device)
        for idx in range(n_layers):
            if idx not in sums:
                sums[idx] = per_layer[idx]
            else:
                sums[idx] = sums[idx] + per_layer[idx]

    n = len(_NEUTRAL_PROMPTS)
    return {idx: sums[idx] / n for idx in range(n_layers)}


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

    for pair in pairs:
        pos_all = _encode_and_capture_all(model, tokenizer, pair["positive"], layers, device)
        neg_all = _encode_and_capture_all(model, tokenizer, pair["negative"], layers, device)
        for idx in range(n_layers):
            # Cast to float32 before differencing — fp16 subtraction
            # loses precision for close vectors, producing degenerate
            # diff matrices that cause LAPACK SVD errors (SLASCL).
            p, n = pos_all[idx].float(), neg_all[idx].float()
            diffs_per_layer[idx].append(p - n)
            norm_sums[idx] += p.norm() + n.norm()

    # Per-layer: compute direction and score
    profile: dict[int, tuple[torch.Tensor, float]] = {}
    scores: dict[int, float] = {}
    n_pairs = len(pairs)

    n_norm_samples = n_pairs * 2  # pos + neg per pair
    # Single GPU→CPU transfer for all layer norms
    norm_sums_cpu = norm_sums.tolist()

    if n_pairs < 2:
        # Single pair: use raw diff, score by norm (normalized to [0,1] below)
        for idx in range(n_layers):
            diff_vec = diffs_per_layer[idx][0]
            ref_norm = norm_sums_cpu[idx] / n_norm_samples
            direction = _normalize(diff_vec, ref_norm=ref_norm)
            scores[idx] = diff_vec.float().norm().item()
            profile[idx] = (direction, scores[idx])
    else:
        # Multi-pair: batched SVD across all layers.
        # Stack into (n_layers, N, dim) and run one batched SVD call —
        # amortizes LAPACK dispatch overhead vs. n_layers individual calls,
        # and matters most on CPU (MPS SVD fallback path).
        diff_matrices = []  # one (N, dim) per layer
        ref_norms = []
        src_device = diffs_per_layer[0][0].device
        for idx in range(n_layers):
            diff_matrices.append(torch.stack(diffs_per_layer[idx]))  # (N, dim)
            ref_norms.append(norm_sums_cpu[idx] / n_norm_samples)

        batched = torch.stack(diff_matrices).float()  # (n_layers, N, dim)
        # SVD on MPS must fall back to CPU
        svd_input = batched.cpu() if src_device.type == "mps" else batched
        _, S, Vh = torch.linalg.svd(svd_input, full_matrices=False)
        # S: (n_layers, min(N,dim)), Vh: (n_layers, min(N,dim), dim)

        for idx in range(n_layers):
            direction = Vh[idx, 0].to(src_device)  # (dim,)

            # Orient so "positive" stays positive: majority vote across pairs.
            dots = diff_matrices[idx] @ direction  # (N,)
            if (dots < 0).sum() > (dots > 0).sum():
                direction = -direction

            score = (S[idx, 0] / S[idx].sum()).item()
            scores[idx] = score
            profile[idx] = (_normalize(direction, ref_norm=ref_norms[idx]), score)

    # Single-pair scores are raw diff norms — normalize to [0, 1] so they're
    # on the same scale as multi-pair explained-variance-ratio scores.
    if len(pairs) < 2 and scores:
        max_raw = max(scores.values())
        if max_raw > 1e-8:
            for idx in scores:
                scores[idx] /= max_raw
                vec, _ = profile[idx]
                profile[idx] = (vec, scores[idx])

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
