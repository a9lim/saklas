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
  neutral-statement activations; ``[N=90, D]`` per layer in fp16.
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
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from saklas.core.profile import Profile
from saklas.io.atomic import write_json_atomic
from saklas.io.paths import model_dir, neutral_statements_path, safe_model_id
from saklas.io.packs import hash_file

log = logging.getLogger(__name__)

_NEUTRAL_ACTS_NAME = "neutral_activations"


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


def load_or_compute_neutral_activations(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
) -> dict[int, torch.Tensor]:
    """Disk-cached per-statement activations for one model.

    Cache key: sha256 of ``~/.saklas/neutral_statements.json`` (same
    discipline as ``layer_means``).  Stale cache → recompute, write,
    return.  Fresh cache → load and return.

    Returns ``{layer_idx: [N, D] fp32 CPU tensor}``.  Stored as fp16 on
    disk to keep the artifact ~30MB instead of ~60MB; promoted back to
    fp32 in memory for the Procrustes fit.
    """
    from saklas.core.vectors import compute_neutral_activations

    ts_path, sc_path = _neutral_acts_paths(model_id)
    md = model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)

    current_ns_hash: str | None = None
    if neutral_statements_path().exists():
        current_ns_hash = hash_file(neutral_statements_path())

    if not force and ts_path.exists() and sc_path.exists():
        try:
            import json

            with open(sc_path) as f:
                sc = json.load(f)
            if current_ns_hash is None or sc.get("statements_sha256") == current_ns_hash:
                tensors = load_file(str(ts_path))
                # Tensors stored as ``layer_<idx>``; lift fp16 → fp32 in
                # memory because Procrustes wants fp32 precision.
                out = {
                    int(k.split("_", 1)[1]): v.float()
                    for k, v in tensors.items()
                }
                log.debug("Loaded cached neutral activations for %s", model_id)
                return out
            log.info(
                "Neutral activations stale (neutral_statements changed); recomputing for %s",
                model_id,
            )
        except Exception as e:
            log.warning("Corrupt neutral activations cache for %s, recomputing: %s", model_id, e)

    log.info("Computing neutral activations (one-time per model)...")
    activations = compute_neutral_activations(model, tokenizer, layers)

    # Persist as fp16 — the alignment fit doesn't benefit from fp32
    # precision in the input observations themselves; the Procrustes
    # SVD lifts to fp32 internally.
    fp16 = {f"layer_{idx}": t.contiguous().to(torch.float16).cpu() for idx, t in activations.items()}
    save_file(fp16, str(ts_path))
    write_json_atomic(sc_path, {
        "method": "neutral_activations",
        "statements_sha256": current_ns_hash or "",
        "n_prompts": next(iter(activations.values())).shape[0] if activations else 0,
        "n_layers": len(activations),
    })
    return activations


# ---------------------------------------------------------------------------
# Alignment fit.
# ---------------------------------------------------------------------------


class AlignmentError(ValueError):
    """Raised when an alignment can't be fit (insufficient shared layers)."""


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
    shared layers) so users see one summary number per transfer; the
    full per-layer dict is exported through the `pack ls -v` JSON path
    for callers that want it.
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
) -> Profile:
    """Apply the alignment map to a source-space profile.

    For each layer in the profile that the alignment covers, computes
    ``v_tgt = M_L @ v_src``.  Layers not covered by the alignment are
    dropped — partial transfer is the only sensible behavior, since a
    direction with no map can't be lifted into target space.

    Carries provenance through ``Profile.metadata``:

    * ``method = "procrustes_transfer"``
    * ``source_model_id`` — HF coord of the source model.
    * ``transfer_quality_estimate`` — median R² across shared layers, if
      known.

    Existing diagnostics fields on the source profile pass through
    unchanged; users can still reason about source-side separation when
    judging whether to trust a transferred probe.
    """
    if not alignment_map:
        from saklas.core.profile import ProfileError

        raise ProfileError("transfer_profile: alignment_map is empty")

    out_tensors: dict[int, torch.Tensor] = {}
    for layer, src_vec in profile.items():
        M_L = alignment_map.get(layer)
        if M_L is None:
            continue
        v_src = src_vec.to(torch.float32).to(M_L.device)
        v_tgt = M_L @ v_src
        # Restore the source dtype convention so the new profile is
        # bit-comparable with native ones at the same layer.
        out_tensors[layer] = v_tgt.to(dtype=src_vec.dtype).cpu()

    if not out_tensors:
        from saklas.core.profile import ProfileError

        raise ProfileError(
            "transfer_profile: alignment covered no layers in the source profile"
        )

    metadata = dict(profile.metadata)
    metadata["method"] = "procrustes_transfer"
    metadata["source_model_id"] = source_model_id
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


def save_alignment_map(
    M: dict[int, torch.Tensor],
    src_model_id: str,
    tgt_model_id: str,
    *,
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
    save_file(tensors, str(ts_path))

    sidecar: dict[str, Any] = {
        "method": "procrustes_alignment",
        "source_model_id": src_model_id,
        "target_model_id": tgt_model_id,
        "shared_layers": sorted(M.keys()),
    }
    if quality_per_layer:
        sidecar["quality_per_layer"] = {
            str(layer): round(float(q), 6) for layer, q in quality_per_layer.items()
        }
    write_json_atomic(sc_path, sidecar)
    return ts_path


def load_alignment_map(
    src_model_id: str, tgt_model_id: str,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]] | None:
    """Load a cached alignment map.  Returns ``None`` when not on disk."""
    import json

    ts_path, sc_path = alignment_cache_path(src_model_id, tgt_model_id)
    if not ts_path.exists() or not sc_path.exists():
        return None
    try:
        tensors = load_file(str(ts_path))
        with open(sc_path) as f:
            sidecar = json.load(f)
        M = {int(k.split("_", 1)[1]): v for k, v in tensors.items()}
        return M, sidecar
    except Exception as e:
        log.warning("Corrupt alignment map cache (%s → %s): %s", src_model_id, tgt_model_id, e)
        return None
