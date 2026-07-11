"""Small local metadata cache for a session-resident SAE release.

Weights remain in the normal Hugging Face cache owned by SAELens.  Saklas only
persists the resolved release/layer identity and optional display labels under
``models/<safe>/sae`` so a UI can describe the last successful load without
copying model-sized tensors into a second cache.
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


def sae_labels_path(model_id: str, release: str) -> Path:
    return sae_runtime_dir(model_id) / f"{safe_release_id(release)}-labels.json"


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


def load_sae_labels(model_id: str, release: str) -> dict[str, str]:
    import json

    path = sae_labels_path(model_id, release)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    labels = payload.get("labels", payload) if isinstance(payload, dict) else {}
    if not isinstance(labels, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in labels.items()
        if isinstance(value, str) and value.strip()
    }


def save_sae_labels(model_id: str, release: str, labels: dict[str, str]) -> Path:
    path = sae_labels_path(model_id, release)
    write_json_atomic(path, {
        "format_version": SAE_RUNTIME_FORMAT_VERSION,
        "model_id": model_id,
        "release": release,
        "labels": labels,
    })
    return path
