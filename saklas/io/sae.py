"""Small local metadata cache for a session-resident SAE release.

Weights remain in the normal Hugging Face cache owned by SAELens.  Saklas only
persists the resolved release/layer identity and optional per-feature metadata
(Neuronpedia display labels + ``maxActApprox``, the corpus-max activation that
normalizes the readout channel to a 0..1 strength) under ``models/<safe>/sae``
so a UI can describe the last successful load without copying model-sized
tensors into a second cache.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from saklas.io.atomic import write_json_atomic
from saklas.io.paths import model_dir

SAE_RUNTIME_FORMAT_VERSION = 1
_UNSAFE = re.compile(r"[^a-z0-9._-]+")


def safe_release_id(release: str) -> str:
    slug = _UNSAFE.sub("_", release.lower()).strip("_")
    return slug or "sae"


def sae_runtime_dir(model_id: str) -> Path:
    return model_dir(model_id) / "sae"


def sae_metadata_path(model_id: str, release: str) -> Path:
    return sae_runtime_dir(model_id) / f"{safe_release_id(release)}.json"


def sae_features_path(model_id: str, release: str) -> Path:
    return sae_runtime_dir(model_id) / f"{safe_release_id(release)}-features.json"


def save_sae_metadata(model_id: str, release: str, payload: dict[str, Any]) -> Path:
    path = sae_metadata_path(model_id, release)
    write_json_atomic(path, {
        "format_version": SAE_RUNTIME_FORMAT_VERSION,
        "model_id": model_id,
        "release": release,
        **payload,
    })
    return path


def load_sae_metadata(model_id: str, release: str) -> dict[str, Any] | None:
    import json

    path = sae_metadata_path(model_id, release)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != SAE_RUNTIME_FORMAT_VERSION
        or payload.get("model_id") != model_id
        or payload.get("release") != release
    ):
        return None
    return payload


def _coerce_feature_entry(value: Any) -> dict[str, Any] | None:
    """Normalize one cached feature row to ``{label, max_act}``."""
    if not isinstance(value, dict):
        return None
    label = value.get("label")
    if not (isinstance(label, str) and label.strip()):
        label = None
    max_act = value.get("max_act")
    if not (isinstance(max_act, (int, float)) and float(max_act) > 0):
        max_act = None
    return {"label": label, "max_act": float(max_act) if max_act else None}


def load_sae_feature_meta(model_id: str, release: str) -> dict[str, dict[str, Any]]:
    """Load the current ``{feature_id: {label, max_act}}`` metadata cache."""
    import json

    path = sae_features_path(model_id, release)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in features.items():
        entry = _coerce_feature_entry(value)
        if entry is not None:
            out[str(key)] = entry
    return out


def save_sae_feature_meta(
    model_id: str, release: str, features: dict[str, dict[str, Any]],
) -> Path:
    path = sae_features_path(model_id, release)
    write_json_atomic(path, {
        "format_version": SAE_RUNTIME_FORMAT_VERSION,
        "model_id": model_id,
        "release": release,
        "features": features,
    })
    return path
