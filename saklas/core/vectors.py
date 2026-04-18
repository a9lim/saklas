"""Extraction, saving, and loading of activation steering/probe vectors."""

from __future__ import annotations

import functools
import json
import logging
from importlib import resources as _resources
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file

if TYPE_CHECKING:
    from saklas.core.sae import SaeBackend

log = logging.getLogger(__name__)

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

    # Find the last non-special-token position.  Chat templates append
    # trailing markers like Llama's <|eot_id|>, Gemma's <end_of_turn>,
    # Qwen's <|im_end|> — pooling from those positions yields degenerate
    # signals disconnected from the content.
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    content_end = ids.shape[1] - 1
    if special_ids:
        id_list = ids[0].tolist()
        while content_end > 0 and id_list[content_end] in special_ids:
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


def extract_contrastive(
    model,
    tokenizer,
    pairs: list[dict],
    layers,
    device=None,
    *,
    sae: "SaeBackend | None" = None,
    drop_edges: tuple[int, int] = (2, 2),
) -> dict[int, torch.Tensor]:
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

    Args:
        pairs: List of {"positive": str, "negative": str} prompt pairs.
        sae: Optional SAE backend. When provided, extraction runs a
            feature-space ``pca_center`` branch restricted to the layers
            covered by the SAE, and decodes the principal feature-space
            direction back into model space before baking shares.
        drop_edges: ``(n_lo, n_hi)`` — exclude the first ``n_lo`` and last
            ``n_hi`` model layers from the share distribution. Early layers
            carry tokenization / lexical features; late layers are strongly
            aligned with the unembedding head. Steering at either end tends
            to corrupt surface form rather than latent meaning — on some
            architectures (e.g. ministral-3, loaded via
            :func:`_load_text_from_multimodal`) L0 PCA share inflates to
            3–4× the model median, which produces immediate grammar collapse
            at otherwise-coherent α. Dropping the edges suppresses the
            pathology uniformly; the remaining share budget redistributes
            over retained layers automatically. Default ``(2, 2)``; pass
            ``(0, 0)`` to recover pre-fix behavior (useful for tests on
            small mock models and for A/B comparisons).

    Returns:
        Profile dict mapping layer_idx -> baked direction vector. Dropped
        edge layers are simply absent from the dict — downstream hook
        attachment iterates over present keys.
    """
    if device is None:
        device = next(model.parameters()).device

    n_layers = len(layers)

    n_drop_lo, n_drop_hi = drop_edges
    if n_drop_lo < 0 or n_drop_hi < 0:
        raise ValueError(f"drop_edges must be non-negative, got {drop_edges}")
    if n_drop_lo + n_drop_hi >= n_layers:
        raise ValueError(
            f"drop_edges={drop_edges} would leave no retained layers "
            f"(n_layers={n_layers})"
        )
    edge_idx = set(range(n_drop_lo)) | set(
        range(n_layers - n_drop_hi, n_layers)
    )

    # Coverage check for the SAE branch — raise early before any forward
    # passes if the backend covers none of this model's layers.
    if sae is not None:
        from saklas.core.errors import SaeCoverageError
        covered = sae.layers & set(range(n_layers))
        if not covered:
            raise SaeCoverageError(
                f"SAE release '{sae.release}' covers no layers for a "
                f"{n_layers}-layer model"
            )
        sae_layers = sorted(covered)
        sae_layer_set = set(sae_layers)  # O(1) membership for the inner loop
    else:
        sae_layers = None
        sae_layer_set = None

    # Accumulate per-layer diffs and running norm sums.
    # norm_sums is a GPU tensor to avoid per-layer .item() sync points
    # (was 2 * N_pairs * N_layers GPU→CPU syncs, now 0 during the loop).
    diffs_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    # SAE path: keep the pos/neg tensors themselves (pca_center needs both,
    # not just their diff), but only for layers the SAE actually covers —
    # non-covered layers would allocate O(N · d_model) fp32 tensors for
    # nothing. Raw path: these dicts stay empty, no cost.
    pos_per_layer: dict[int, list[torch.Tensor]] = (
        {i: [] for i in sae_layer_set} if sae is not None else {}
    )
    neg_per_layer: dict[int, list[torch.Tensor]] = (
        {i: [] for i in sae_layer_set} if sae is not None else {}
    )
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
            if sae is not None and idx in sae_layer_set:
                # fp32 matches the diff dtype discipline; avoids fp16 overflow.
                pos_per_layer[idx].append(p_d.float())
                neg_per_layer[idx].append(n_d.float())
        # Free forward-pass intermediates (attention maps, hidden states)
        # before the next pair — MPS doesn't release memory eagerly.
        del pos_all, neg_all
        if _mps:
            torch.mps.empty_cache()

    # SAE branch: feature-space pca_center, decode back to model space.
    # Runs only on the covered-layer subset; returns early with baked
    # tensors restricted to those layers.
    if sae is not None:
        norm_sums_cpu = norm_sums.tolist()
        n_norm_samples = len(pairs) * 2
        directions: dict[int, torch.Tensor] = {}
        # Accumulate EVR scalars as tensors, one .tolist() after the loop
        # (matches the raw-PCA path's discipline — one GPU→CPU transfer).
        evr_tensors: list[torch.Tensor] = []
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
            evr_tensors.append(S[0] / S.sum())

            # Orient by majority vote: pos-minus-neg projected onto v_feat
            # should majority-positive.
            dots = (F_pos - F_neg) @ v_feat
            if (dots < 0).sum() > (dots > 0).sum():
                v_feat = -v_feat

            with torch.no_grad():
                v_model = sae.decode_layer(idx, v_feat).float()

            directions[idx] = _normalize(v_model, ref_norm=ref_norm)

        evrs = torch.stack(evr_tensors).tolist()
        raw: dict[int, tuple[torch.Tensor, float]] = {
            idx: (directions[idx], evr) for idx, evr in zip(sae_layers, evrs)
        }

        for i in edge_idx:
            raw.pop(i, None)

        # Bake shares across the covered subset (same logic as raw path).
        total_score = sum(score for _, score in raw.values())
        # Defensive: evr is always positive when any singular value exists;
        # the fallback never triggers in practice.
        if total_score <= 0:
            shares = {i: 1.0 / len(raw) for i in raw}
        else:
            shares = {i: score / total_score for i, (_, score) in raw.items()}

        return {i: direction * shares[i] for i, (direction, _) in raw.items()}

    # Per-layer: compute direction and score, then bake shares into magnitude.
    n_pairs = len(pairs)
    n_norm_samples = n_pairs * 2  # pos + neg per pair
    # Single GPU→CPU transfer for all layer norms
    norm_sums_cpu = norm_sums.tolist()

    # First pass: extract raw (direction, score) per layer.
    raw: dict[int, tuple[torch.Tensor, float]] = {}

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

    for i in edge_idx:
        raw.pop(i, None)

    # Bake shares into the stored tensors. Total share is 1.0 across retained
    # layers, so sum(||baked_i||) ≈ sum(ref_norm_i * share_i): the collective
    # magnitude budget is fixed by the reference activation norms and
    # distributed in proportion to per-layer signal quality. At apply time
    # the hook just does alpha * _STEER_GAIN * sum(baked) — no shares,
    # no sums, no per-layer weights.
    total_score = sum(score for _, score in raw.values())
    if total_score <= 0:
        # Pathological extraction (all-zero diffs). Fall back to uniform
        # across retained layers.
        shares = {idx: 1.0 / len(raw) for idx in raw}
    else:
        shares = {idx: score / total_score for idx, (_, score) in raw.items()}

    return {idx: direction * shares[idx] for idx, (direction, _) in raw.items()}


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
    # SAE provenance — present only when extraction used an SAE backend.
    for key in ("sae_release", "sae_revision", "sae_ids_by_layer"):
        if key in metadata:
            sidecar[key] = metadata[key]

    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(sidecar, f, indent=2)

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
    return profile, metadata


def project_profile(
    base: dict[int, torch.Tensor],
    onto: dict[int, torch.Tensor],
    operator: str,
) -> dict[int, torch.Tensor]:
    """Per-layer projection of ``base`` against ``onto``.

    For each shared layer (fp32)::

        proj = (dot(base, onto) / dot(onto, onto)) * onto

    - ``operator == "~"``   returns ``proj``       (component of base aligned with onto).
    - ``operator == "|"`` returns ``base - proj`` (component of base orthogonal to onto).

    Layers in ``base`` without a matching layer in ``onto``: for ``"|"``
    they pass through unchanged (nothing to project away); for ``"~"`` they
    are dropped (projection onto an absent direction is undefined).

    Near-zero ``||onto|| < 1e-12`` layers are treated the same way: ``"|"``
    passes base through unchanged, ``"~"`` drops the layer. Result tensors
    are cast back to the source dtype of ``base``.

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

    Accepts two shapes:
      - bare list: [{"positive": ..., "negative": ...}, ...]
        (new format — statements.json in concept folders)
      - legacy object: {"name": ..., "pairs": [...]}
        (old saklas/datasets/<name>.json schema)

    Returns a dict with at least a ``"pairs"`` key.
    """
    with open(dataset_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"pairs": data}
    return data
