"""Per-model Jacobian-lens artifact: save/load under ``models/<safe_id>/``.

The lens is a per-model transport (one ``J_l`` matrix per source layer), not a
per-concept artifact, so it lives next to the neutral-activation cache rather
than under ``manifolds/``:

    ~/.saklas/models/<safe_model_id>/jlens.safetensors   # layer_<idx>, fp16
    ~/.saklas/models/<safe_model_id>/jlens.json          # sidecar

fp16 on disk (the reference-repo convention — J entries are O(1), so range is
no constraint and fp16's extra mantissa bits beat bf16; this deliberately
differs from the neutral cache's fp32 invariant, which exists because that
cache feeds a covariance inversion). Promoted to fp32 on load.

The sidecar records the corpus spec + sha256 so a re-fit against a different
corpus reads as stale, and ``n_prompts`` so an interrupted fit can resume
(load → fit the remaining prompts → ``JacobianLens.merge`` → save).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from saklas.core.jlens import JacobianLens
from saklas.io.atomic import write_json_atomic
from saklas.io.paths import model_dir

log = logging.getLogger(__name__)

LENS_FORMAT_VERSION = 1
_LENS_NAME = "jlens"
_LENS_METHOD = "jlens_cotangent_sum"


def lens_paths(model_id: str) -> tuple[Path, Path]:
    """Return ``(safetensors_path, sidecar_path)`` for a model's lens."""
    md = model_dir(model_id)
    return md / f"{_LENS_NAME}.safetensors", md / f"{_LENS_NAME}.json"


def save_lens(
    lens: JacobianLens,
    model_id: str,
    *,
    corpus_spec: str,
    corpus_sha256: str,
    seq_len: int,
    dim_batch: int,
    skip_first: int,
) -> Path:
    """Persist a fitted lens (fp16 tensors + atomic JSON sidecar)."""
    ts_path, sc_path = lens_paths(model_id)
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"layer_{idx}": J.contiguous().to(torch.float16).cpu()
        for idx, J in lens.jacobians.items()
    }
    save_file(tensors, str(ts_path))
    write_json_atomic(sc_path, {
        "format_version": LENS_FORMAT_VERSION,
        "method": _LENS_METHOD,
        "n_prompts": lens.n_prompts,
        "d_model": lens.d_model,
        "source_layers": lens.source_layers,
        "dtype": "float16",
        "corpus_spec": corpus_spec,
        "corpus_sha256": corpus_sha256,
        "seq_len": seq_len,
        "dim_batch": dim_batch,
        "skip_first_positions": skip_first,
    })
    return ts_path


def load_lens(model_id: str) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load a model's fitted lens, or ``None`` when absent or unusable.

    Self-healing like the neutral-activation cache: a wrong format version,
    non-finite tensors, or any parse failure logs a warning and reads as
    "no lens" (the caller decides whether to error or re-fit) rather than
    crashing the session.
    """
    ts_path, sc_path = lens_paths(model_id)
    if not (ts_path.exists() and sc_path.exists()):
        return None
    try:
        with open(sc_path) as f:
            sidecar = json.load(f)
        version = sidecar.get("format_version")
        if version != LENS_FORMAT_VERSION:
            log.warning(
                "jlens cache for %s has format_version %r (need %d); ignoring "
                "— re-fit with `saklas lens fit`", model_id, version, LENS_FORMAT_VERSION,
            )
            return None
        tensors = load_file(str(ts_path))
        jacobians = {
            int(k.split("_", 1)[1]): v.to(torch.float32) for k, v in tensors.items()
        }
        if not all(bool(torch.isfinite(j).all()) for j in jacobians.values()):
            log.warning(
                "jlens cache for %s contains non-finite values; ignoring — "
                "re-fit with `saklas lens fit`", model_id,
            )
            return None
        lens = JacobianLens(
            jacobians,
            n_prompts=int(sidecar.get("n_prompts", 0)),
            d_model=int(sidecar.get("d_model", 0)),
        )
        return lens, sidecar
    except Exception as exc:
        log.warning("Corrupt jlens cache for %s; ignoring: %s", model_id, exc)
        return None


def remove_lens(model_id: str) -> bool:
    """Delete a model's lens artifact. Returns True when anything was removed."""
    removed = False
    for path in lens_paths(model_id):
        if path.exists():
            path.unlink()
            removed = True
    return removed
