"""Cross-model probe alignment via per-layer Procrustes (v1.6).

Probes are extracted per (model, concept).  A probe extracted on one
model isn't directly usable on a different model — different tokenizers,
different hidden dims, different basis rotation in the residual stream.
This module fits a per-layer linear map between two models' neutral
activations (``Procrustes`` for matched dim, low-rank PCA-and-lift for
mismatched dim) and uses it to transfer a probe's per-layer baked
direction from source-space to target-space.

Public surface:

* :func:`load_or_compute_neutral_activations` — disk-cached per-model
  neutral-statement activations; ``[N=90, D]`` per layer, stored fp32.
  Its metadata-returning sibling lets dependent artifact builders reuse the
  sidecar validated in that same cache transaction instead of hashing the
  payload again merely to recover its identity.
* :func:`fit_alignment` — per-layer alignment map ``M_L : ℝ^D_src → ℝ^D_tgt``.
* :func:`transfer_profile` — apply the alignment map to a profile.
* :func:`alignment_cache_path` — disk cache for the fitted map keyed by
  the source model id.

The transferred profile lands at the target model's tensor path with a
``_from-<safe_src>`` suffix — uses the same variant-suffix machinery as
SAE variants, so the rest of saklas (selectors, packs, monitor) sees a
transferred probe as just another tensor on disk.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import hashlib
import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from saklas.core.errors import SaklasError
from saklas.core.profile import Profile
from saklas.io.atomic import write_json_atomic
from saklas.io.paths import model_dir, safe_model_id
from saklas.io.packs import hash_file

log = logging.getLogger(__name__)

_NEUTRAL_ACTS_NAME = "neutral_activations"
_NEUTRAL_CACHE_FORMAT_VERSION = 2
_NEUTRAL_CAPTURE_VERSION = 1


# ---------------------------------------------------------------------------
# Per-model neutral activation cache.
# ---------------------------------------------------------------------------


def _neutral_acts_paths(model_id: str) -> tuple[Path, Path]:
    """Return ``(safetensors_path, sidecar_path)`` for cached activations."""
    md = model_dir(model_id)
    return (
        md / f"{_NEUTRAL_ACTS_NAME}.safetensors",
        md / f"{_NEUTRAL_ACTS_NAME}.json",
    )


@contextmanager
def neutral_fit_lock(model_id: str) -> Iterator[None]:
    """Single-flight lock for one model's expensive neutral forward pass."""
    from saklas.io.atomic import artifact_lock

    ts_path, _ = _neutral_acts_paths(model_id)
    with artifact_lock(ts_path.with_name(f"{ts_path.stem}.fit")):
        yield


def validate_neutral_cache_metadata(model_id: str) -> dict[str, Any]:
    """Validate a neutral cache's identity and tensor header without paging data.

    The payload digest proves bytes, while ``safe_open`` checks the declared
    key/shape/dtype schema from the mmap header. This is the cheap preflight for
    exact transfer repeats; numerical finite checks remain in the materializing
    loader used by fits and whiteners.
    """
    from saklas.io.atomic import artifact_lock

    ts_path, _ = _neutral_acts_paths(model_id)
    with artifact_lock(ts_path):
        return _validate_neutral_cache_metadata_locked(model_id)


def _validate_neutral_cache_metadata_locked(model_id: str) -> dict[str, Any]:
    """Header/digest validation with the neutral pair lock already held."""
    ts_path, sc_path = _neutral_acts_paths(model_id)
    if not ts_path.exists() or not sc_path.exists():
        raise FileNotFoundError(f"neutral activation cache missing for {model_id}")
    with open(sc_path) as handle:
        sidecar = json.load(handle)
    if (
        sidecar.get("format_version") != _NEUTRAL_CACHE_FORMAT_VERSION
        or sidecar.get("capture_version") != _NEUTRAL_CAPTURE_VERSION
        or sidecar.get("tensor_sha256") != hash_file(ts_path)
    ):
        raise ValueError("neutral activation cache identity or payload digest mismatch")
    layers = sidecar.get("layers")
    schema = sidecar.get("tensor_schema")
    if not isinstance(layers, list) or not isinstance(schema, dict):
        raise ValueError("neutral activation cache sidecar has no layer schema")
    expected_keys = {f"layer_{int(layer)}" for layer in layers}
    expected_n = int(sidecar.get("n_prompts", -1))
    with safe_open(str(ts_path), framework="pt", device="cpu") as tensors:
        if set(tensors.keys()) != expected_keys:
            raise ValueError("neutral activation cache layer keys do not match sidecar")
        for key in tensors.keys():
            layer = int(key.split("_", 1)[1])
            spec = schema.get(str(layer), {})
            view = tensors.get_slice(key)
            shape = list(view.get_shape())
            if (
                view.get_dtype() != "F32"
                or spec.get("dtype") != "torch.float32"
                or shape != spec.get("shape")
                or len(shape) != 2
                or shape[0] != expected_n
            ):
                raise ValueError(
                    f"neutral activation cache layer {layer} header failed validation"
                )
    return sidecar


def load_validated_neutral_cache(
    model_id: str,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Materialize and fully validate a neutral cache without loading the model."""
    from saklas.io.atomic import artifact_lock

    ts_path, _ = _neutral_acts_paths(model_id)
    with artifact_lock(ts_path):
        sidecar = _validate_neutral_cache_metadata_locked(model_id)
        tensors = load_file(str(ts_path))
    out: dict[int, torch.Tensor] = {}
    schema = sidecar["tensor_schema"]
    expected_n = int(sidecar.get("n_prompts", -1))
    for key, tensor in tensors.items():
        layer = int(key.split("_", 1)[1])
        expected_shape = schema.get(str(layer), {}).get("shape")
        if (
            tensor.dtype != torch.float32
            or list(tensor.shape) != expected_shape
            or tensor.ndim != 2
            or int(tensor.shape[0]) != expected_n
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"neutral activation cache layer {layer} failed validation")
        out[layer] = tensor
    return out, sidecar


def neutral_cache_identity(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Stable identity subset embedded into dependent alignment artifacts."""
    keys = (
        "format_version", "capture_version", "model_fingerprint",
        "model_source_fingerprint",
        "capture_sha256", "tensor_sha256", "layers", "tensor_schema",
        "n_prompts",
    )
    return {key: sidecar.get(key) for key in keys}


def load_or_compute_neutral_activations(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
) -> dict[int, torch.Tensor]:
    """Single-flight neutral-cache load/capture, returning activations only."""
    activations, _sidecar = load_or_compute_neutral_activations_with_metadata(
        model, tokenizer, layers, model_id=model_id, force=force,
    )
    return activations


def load_or_compute_neutral_activations_with_metadata(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Return activations plus the sidecar proven in the same transaction.

    Alignment fitting needs both the materialized neutral rows and their stable
    cache identity.  Returning the already-validated sidecar avoids an immediate
    second full-file digest pass through
    :func:`validate_neutral_cache_metadata`.  The activation-only public API
    above remains the compatibility surface for ordinary whitener/probe callers.
    """
    with neutral_fit_lock(model_id):
        return _load_or_compute_neutral_activations_with_metadata_locked(
            model, tokenizer, layers, model_id=model_id, force=force,
        )


def _load_or_compute_neutral_activations_with_metadata_locked(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Disk-cached per-statement activations for one model.

    Cache identity binds the exact rendered token rows and pooling positions,
    the loaded model weights, and the fitted layer set. The tensor payload is
    digest-verified before reuse. Stale/legacy cache → recompute and replace.

    Returns ``({layer_idx: [N, D] fp32 CPU tensor}, validated_sidecar)``.
    Stored fp32 on disk so the whitener covariance is built and inverted at
    full precision and the compute path (cache miss) and cache-hit path return
    bit-identical tensors (no precision seam — bf16's ~0.4% input error landed
    on exactly the inverted quantity).  fp16 was abandoned because its 65504
    ceiling overflows gemma-3's extreme late-layer channels to ±inf (poisoning
    ``λ`` / ``K``); fp32 has the range and, unlike the former bf16 store, no
    input-precision loss, at the cost of ~2× disk on this one cache.
    """
    from saklas.core.model import loaded_model_fingerprint
    from saklas.core.vectors import (
        _neutral_pairs,
        _prepare_capture_batch,
        compute_neutral_activations,
    )

    ts_path, sc_path = _neutral_acts_paths(model_id)
    md = model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)

    pairs = _neutral_pairs()
    prompts = [prompt for prompt, _ in pairs]
    responses = [response for _, response in pairs]
    prepared = _prepare_capture_batch(
        tokenizer, prompts, responses, torch.device("cpu"),
    )
    rendered_payload = [
        {"input_ids": ids[0].tolist(), "content_end": int(content_end)}
        for ids, content_end in prepared
    ]
    capture_sha = hashlib.sha256(json.dumps(
        {"capture_version": _NEUTRAL_CAPTURE_VERSION, "rows": rendered_payload},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    model_fingerprint = loaded_model_fingerprint(model, model_id)
    base_model = getattr(model, "_orig_mod", model)
    model_source_fingerprint = getattr(
        base_model, "_saklas_source_fingerprint", None,
    )
    expected_layers = list(range(len(layers)))

    if not force and ts_path.exists() and sc_path.exists():
        try:
            out, sc = load_validated_neutral_cache(model_id)
            identity_matches = (
                sc.get("format_version") == _NEUTRAL_CACHE_FORMAT_VERSION
                and sc.get("capture_version") == _NEUTRAL_CAPTURE_VERSION
                and sc.get("capture_sha256") == capture_sha
                and sc.get("model_fingerprint") == model_fingerprint
                and sc.get("model_source_fingerprint") == model_source_fingerprint
                and sc.get("layers") == expected_layers
            )
            if identity_matches:
                log.debug("Loaded cached neutral activations for %s", model_id)
                return out, sc
            else:
                log.info(
                    "Neutral activations stale (model/token/corpus/layer identity changed); "
                    "recomputing for %s",
                    model_id,
                )
        except Exception as e:
            log.warning("Corrupt neutral activations cache for %s, recomputing: %s", model_id, e)

    log.info("Computing neutral activations (one-time per model)...")
    activations = compute_neutral_activations(
        model, tokenizer, layers, rendered=prepared,
    )

    # Persist fp32 — the whitener covariance is built and inverted from these,
    # so the compute path (this fresh fp32 store) and the cache-hit path stay
    # bit-identical (no precision seam) and the inversion is full-precision.
    # fp32 carries the exponent range fp16 lacked (gemma-3's late layers exceed
    # the fp16 max of 65504 and overflowed to ±inf) without bf16's input loss,
    # at ~2× disk on this one cache.
    fp32 = {f"layer_{idx}": t.contiguous().to(torch.float32).cpu() for idx, t in activations.items()}
    from saklas.io.atomic import artifact_lock

    with artifact_lock(ts_path):
        tmp_path = ts_path.with_suffix(ts_path.suffix + ".tmp")
        save_file(fp32, str(tmp_path))
        os.replace(tmp_path, ts_path)
        sidecar: dict[str, Any] = {
            "method": "neutral_activations",
            "format_version": _NEUTRAL_CACHE_FORMAT_VERSION,
            "capture_version": _NEUTRAL_CAPTURE_VERSION,
            "capture_sha256": capture_sha,
            "model_fingerprint": model_fingerprint,
            "model_source_fingerprint": model_source_fingerprint,
            "tensor_sha256": hash_file(ts_path),
            "layers": sorted(activations),
            "tensor_schema": {
                str(idx): {"shape": list(t.shape), "dtype": str(t.dtype)}
                for idx, t in sorted(activations.items())
            },
            "n_prompts": (
                next(iter(activations.values())).shape[0] if activations else 0
            ),
            "n_layers": len(activations),
        }
        write_json_atomic(sc_path, sidecar)
    return activations, sidecar


# ---------------------------------------------------------------------------
# Alignment fit.
# ---------------------------------------------------------------------------


class AlignmentError(ValueError, SaklasError):
    """Raised when an alignment can't be fit (insufficient shared layers)."""

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


def fit_alignment(
    src_acts: dict[int, torch.Tensor],
    tgt_acts: dict[int, torch.Tensor],
    *,
    min_shared_layers: int = 10,
) -> dict[int, torch.Tensor]:
    """Per-layer Procrustes mapping ``M_L : ℝ^D_src → ℝ^D_tgt``.

    For each layer the two models share, fits ``M_L`` such that
    ``X_tgt ≈ X_src @ M_L^T`` over the centered neutral activations.

    * **Same hidden dim:** orthogonal Procrustes via SVD —
      ``X_tgt^T @ X_src = U @ S @ V^T`` then ``M_L = U @ V^T`` (rotation
      that minimises ``||X_tgt - X_src @ M^T||``).
    * **Different hidden dim:** general least-squares form
      ``M_L = (X_src^+ @ X_tgt)^T`` shaped ``(D_tgt, D_src)``.  The
      pseudoinverse handles rectangular systems cleanly; centring on
      both sides keeps the fit translation-invariant.

    Returns ``{layer_idx: M_L}`` over the shared-layer set.  Tensors are
    fp32, on CPU.  Raises :class:`AlignmentError` when fewer than
    ``min_shared_layers`` layers are common to both models — partial
    alignment under that threshold tends to produce noisy transfers.
    """
    shared = sorted(set(src_acts.keys()) & set(tgt_acts.keys()))
    if len(shared) < min_shared_layers:
        raise AlignmentError(
            f"alignment requires >= {min_shared_layers} shared layers; "
            f"src={sorted(src_acts.keys())}, tgt={sorted(tgt_acts.keys())}"
        )

    out: dict[int, torch.Tensor] = {}
    for layer in shared:
        X_src = src_acts[layer].to(torch.float32)
        X_tgt = tgt_acts[layer].to(torch.float32)
        if X_src.shape[0] != X_tgt.shape[0]:
            raise AlignmentError(
                f"layer {layer}: src has N={X_src.shape[0]} prompts, "
                f"tgt has N={X_tgt.shape[0]}; expected matched."
            )

        # Center both sides — Procrustes is translation-invariant only
        # under shared origin.
        X_src_c = X_src - X_src.mean(dim=0, keepdim=True)
        X_tgt_c = X_tgt - X_tgt.mean(dim=0, keepdim=True)

        D_src = X_src_c.shape[1]
        D_tgt = X_tgt_c.shape[1]

        if D_src == D_tgt:
            # Orthogonal Procrustes: closest rotation that maps src → tgt.
            U, _S, Vh = torch.linalg.svd(
                X_tgt_c.transpose(0, 1) @ X_src_c, full_matrices=False,
            )
            M = U @ Vh  # (D_tgt, D_src), orthogonal
        else:
            # Least-squares: solve X_src_c @ A = X_tgt_c, store M = A^T
            # so the apply step is M @ v_src.  ``torch.linalg.lstsq`` is
            # the stable rectangular solver.
            sol = torch.linalg.lstsq(X_src_c, X_tgt_c).solution  # (D_src, D_tgt)
            M = sol.transpose(0, 1).contiguous()  # (D_tgt, D_src)

        out[layer] = M
    return out


def alignment_quality(
    M: dict[int, torch.Tensor],
    src_acts: dict[int, torch.Tensor],
    tgt_acts: dict[int, torch.Tensor],
) -> dict[int, float]:
    """Per-layer R² between the transferred src activations and tgt.

    For each layer ``L`` shared by ``M``, computes
    ``1 - ||X_tgt - X_src @ M_L^T||^2 / ||X_tgt||^2`` on centered data.
    Values near 1.0 mean the linear map captures the cross-model
    geometry; values near 0.0 mean transferred probes will be noisy.

    Surfaced through ``Sidecar.transfer_quality_estimate`` (median over
    shared layers) so users see one summary number per transfer; the full
    per-layer dict is exported in transfer sidecars and CLI JSON output for
    callers that want it.
    """
    out: dict[int, float] = {}
    for layer, M_L in M.items():
        X_src = src_acts[layer].to(torch.float32)
        X_tgt = tgt_acts[layer].to(torch.float32)
        X_src_c = X_src - X_src.mean(dim=0, keepdim=True)
        X_tgt_c = X_tgt - X_tgt.mean(dim=0, keepdim=True)
        # Apply M to each row: X_src_c @ M_L^T
        X_pred = X_src_c @ M_L.transpose(0, 1)
        residual = (X_tgt_c - X_pred).pow(2).sum().item()
        total = X_tgt_c.pow(2).sum().item()
        if total <= 1e-12:
            out[layer] = 0.0
        else:
            out[layer] = float(1.0 - residual / total)
    return out


# ---------------------------------------------------------------------------
# Profile transfer.
# ---------------------------------------------------------------------------


def transfer_profile(
    profile: Profile,
    alignment_map: dict[int, torch.Tensor],
    *,
    source_model_id: str,
    transfer_quality_estimate: float | None = None,
    whitener: "Any | None" = None,
) -> Profile:
    """Apply the alignment map to a source-space profile.

    For each layer in the profile that the alignment covers, computes
    ``v_tgt = M_L @ v_src``.  Layers not covered by the alignment are
    dropped — partial transfer is the only sensible behavior, since a
    direction with no map can't be lifted into target space.

    **Target-metric re-bake (mandatory).**  The source tensor is
    share-baked in the *source* model's metric — its per-layer Euclidean
    magnitude is the source Mahalanobis norm of the raw mean-diff (the
    unified subspace hook reads ``‖baked_L‖₂ / Σ‖baked‖₂`` back out as
    the layer share).  The orthogonal Procrustes map preserves Euclidean norm, so a
    bare transfer would carry the *source* cross-layer share into target
    space — where it no longer matches the target's anisotropy.  The
    ``whitener`` for the **target** model is **required** and must cover
    every transferred layer (all-or-nothing, mirroring the DiM-bake /
    monitor / manifold-fit gate); each layer is rescaled so its magnitude
    becomes its *target* Mahalanobis norm::

        v_tgt'_L = v_tgt_L · (‖v_tgt_L‖_M(target) / ‖v_tgt_L‖₂)

    The direction is untouched; only the per-layer magnitude — and hence
    the hook-recovered share — changes.  ``‖v_tgt_L‖₂`` carries
    the transported source-signal strength (the best target-signal proxy
    available without target contrastive pairs) and ``‖v̂_tgt_L‖_M(target)``
    applies the target anisotropy correction, so the composite is the
    target-metric analogue of a native DiM bake.  ``bake: "mahalanobis"``
    is stamped on the result.  A missing or non-covering whitener raises
    :class:`WhitenerError` — there is no Euclidean transfer.

    Carries provenance through ``Profile.metadata``:

    * ``method = "procrustes_transfer"``
    * ``source_model_id`` — HF coord of the source model.
    * ``transfer_quality_estimate`` — median R² across shared layers, if
      known.
    * ``bake`` — always ``"mahalanobis"``.

    Existing diagnostics fields on the source profile pass through
    unchanged; users can still reason about source-side separation when
    judging whether to trust a transferred probe.
    """
    if not alignment_map:
        from saklas.core.profile import ProfileError

        raise ProfileError("transfer_profile: alignment_map is empty")

    # Stage the transferred directions in fp32 (keyed by layer) plus the
    # original per-layer dtype, so the target re-bake operates at full
    # precision before the dtype restore.
    staged: dict[int, torch.Tensor] = {}
    orig_dtype: dict[int, torch.dtype] = {}
    for layer, src_vec in profile.items():
        M_L = alignment_map.get(layer)
        if M_L is None:
            continue
        v_src = src_vec.to(torch.float32).to(M_L.device)
        staged[layer] = (M_L @ v_src).cpu()
        orig_dtype[layer] = src_vec.dtype

    if not staged:
        from saklas.core.profile import ProfileError

        raise ProfileError(
            "transfer_profile: alignment covered no layers in the source profile"
        )

    # Mahalanobis-only target re-bake: the per-layer ‖·‖_M and ‖·‖₂ scales
    # differ by a 1/√λ_L factor that doesn't cancel from the cross-layer
    # share, so the target whitener must cover every transferred layer.
    # No Euclidean transfer — a missing / partial whitener is an error.
    from saklas.core.mahalanobis import WhitenerError

    if whitener is None or not whitener.covers_all(staged.keys()):
        raise WhitenerError(
            "transfer_profile requires a Mahalanobis whitener covering every "
            f"transferred layer {sorted(staged.keys())}; generate neutral "
            "activations for the TARGET model first (the Euclidean path is gone)"
        )
    for layer, v_tgt in staged.items():
        eucl = float(v_tgt.norm().item())
        if eucl < 1e-8:
            # Degenerate direction — leave it; rescaling a zero vector
            # is undefined and it carries no share anyway.
            continue
        m_norm = whitener.mahalanobis_norm(layer, v_tgt)
        staged[layer] = v_tgt * (m_norm / eucl)

    out_tensors = {
        layer: v.to(dtype=orig_dtype[layer]) for layer, v in staged.items()
    }

    metadata = dict(profile.metadata)
    metadata["method"] = "procrustes_transfer"
    metadata["source_model_id"] = source_model_id
    metadata["bake"] = "mahalanobis"
    if transfer_quality_estimate is not None:
        metadata["transfer_quality_estimate"] = float(transfer_quality_estimate)
    return Profile(out_tensors, metadata=metadata)


# ---------------------------------------------------------------------------
# Alignment-map disk cache.
# ---------------------------------------------------------------------------


def alignment_cache_path(src_model_id: str, tgt_model_id: str) -> tuple[Path, Path]:
    """``(safetensors, sidecar)`` paths for a cached alignment map.

    Layout: ``~/.saklas/models/<safe_tgt>/alignments/<safe_src>.{safetensors,json}``.
    Lives under the *target* model's directory so deleting a target
    model's cache (``rm -rf models/<tgt>``) wipes its alignments too.
    """
    md = model_dir(tgt_model_id)
    al_dir = md / "alignments"
    src_safe = safe_model_id(src_model_id)
    return (al_dir / f"{src_safe}.safetensors", al_dir / f"{src_safe}.json")


@contextmanager
def alignment_fit_lock(src_model_id: str, tgt_model_id: str) -> Iterator[None]:
    """Single-flight a complete directional alignment fit, including loads."""
    from saklas.io.atomic import artifact_lock

    ts_path, _ = alignment_cache_path(src_model_id, tgt_model_id)
    with artifact_lock(ts_path.with_name(f"{ts_path.stem}.fit")):
        yield


def save_alignment_map(
    M: dict[int, torch.Tensor],
    src_model_id: str,
    tgt_model_id: str,
    *,
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
    quality_per_layer: dict[int, float] | None = None,
) -> Path:
    """Persist a fitted alignment map.

    Returns the path to the written safetensors file.  Sidecar JSON
    carries provenance: source/target model ids, per-layer quality
    estimates if available, and a fit method tag.
    """
    ts_path, sc_path = alignment_cache_path(src_model_id, tgt_model_id)
    ts_path.parent.mkdir(parents=True, exist_ok=True)

    tensors = {f"layer_{idx}": M_L.contiguous().cpu() for idx, M_L in M.items()}
    from saklas.io.atomic import artifact_lock

    with artifact_lock(ts_path):
        tmp_path = ts_path.with_suffix(ts_path.suffix + ".tmp")
        save_file(tensors, str(tmp_path))
        os.replace(tmp_path, ts_path)
        sidecar: dict[str, Any] = {
            "format_version": 2,
            "method": "procrustes_alignment",
            "source_model_id": src_model_id,
            "target_model_id": tgt_model_id,
            "shared_layers": sorted(M.keys()),
            "tensor_schema": {
                str(layer): list(tensor.shape)
                for layer, tensor in sorted(M.items())
            },
            "source_neutral_identity": source_identity,
            "target_neutral_identity": target_identity,
            "tensor_sha256": hash_file(ts_path),
        }
        if quality_per_layer:
            sidecar["quality_per_layer"] = {
                str(layer): round(float(q), 6)
                for layer, q in quality_per_layer.items()
            }
        write_json_atomic(sc_path, sidecar)
    return ts_path


def load_alignment_map(
    src_model_id: str, tgt_model_id: str,
    *,
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
) -> tuple[dict[int, torch.Tensor], dict[str, Any]] | None:
    """Load a cached alignment map.  Returns ``None`` when not on disk."""
    import json

    ts_path, sc_path = alignment_cache_path(src_model_id, tgt_model_id)
    from saklas.io.atomic import artifact_lock

    with artifact_lock(ts_path):
        if not ts_path.exists() or not sc_path.exists():
            return None
        try:
            tensors = load_file(str(ts_path))
            with open(sc_path) as f:
                sidecar = json.load(f)
            if (
                sidecar.get("format_version") != 2
                or sidecar.get("source_neutral_identity") != source_identity
                or sidecar.get("target_neutral_identity") != target_identity
                or sidecar.get("tensor_sha256") != hash_file(ts_path)
            ):
                return None
            M = {int(k.split("_", 1)[1]): v for k, v in tensors.items()}
            expected_layers = sidecar.get("shared_layers")
            schema = sidecar.get("tensor_schema") or {}
            if sorted(M) != expected_layers or any(
                list(tensor.shape) != schema.get(str(layer))
                or tensor.ndim != 2
                or not bool(torch.isfinite(tensor).all())
                for layer, tensor in M.items()
            ):
                return None
            return M, sidecar
        except Exception as e:
            log.warning(
                "Corrupt alignment map cache (%s → %s): %s",
                src_model_id, tgt_model_id, e,
            )
            return None
