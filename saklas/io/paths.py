"""Filesystem path helpers for the ~/.saklas/ tree.

All paths resolve through saklas_home(), which honors the SAKLAS_HOME
environment variable for testing and non-default installs.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Variant filename convention: `<safe_model_id>[_sae-<release>].safetensors`.
# The literal `_sae-` is the separator — no HF model id contains it, and
# SAELens release names are ASCII lower/digits/./-, all fitting our slug rules.
_VARIANT_SEP = "_sae-"
_UNSAFE_VARIANT_CHARS = re.compile(r"[^a-z0-9._-]+")


def saklas_home() -> Path:
    """Return the root ~/.saklas/ directory. Honors $SAKLAS_HOME override."""
    override = os.environ.get("SAKLAS_HOME")
    if override:
        return Path(override)
    return Path.home() / ".saklas"


def vectors_dir() -> Path:
    return saklas_home() / "vectors"


def models_dir() -> Path:
    return saklas_home() / "models"


def neutral_statements_path() -> Path:
    return saklas_home() / "neutral_statements.json"


def safe_model_id(model_id: str) -> str:
    """Flatten an HF-style model ID for filesystem use: '/' -> '__'."""
    return model_id.replace("/", "__")


def concept_dir(namespace: str, concept: str) -> Path:
    return vectors_dir() / namespace / concept


def model_dir(model_id: str) -> Path:
    return models_dir() / safe_model_id(model_id)


def safe_variant_suffix(release: str | None) -> str:
    """Render the filename suffix for a variant. ``None``/``""`` = raw (no suffix)."""
    if not release:
        return ""
    slug = _UNSAFE_VARIANT_CHARS.sub("_", release.lower())
    return f"{_VARIANT_SEP}{slug}"


def tensor_filename(model_id: str, *, release: str | None = None) -> str:
    """Construct the canonical tensor filename for a (model, variant) pair."""
    return f"{safe_model_id(model_id)}{safe_variant_suffix(release)}.safetensors"


def sidecar_filename(model_id: str, *, release: str | None = None) -> str:
    """Sidecar JSON partner for a tensor filename."""
    return f"{safe_model_id(model_id)}{safe_variant_suffix(release)}.json"


def parse_tensor_filename(filename: str) -> tuple[str, str | None] | None:
    """Reverse of :func:`tensor_filename`. Returns ``(safe_model_id, release)``.

    Returns ``None`` for filenames that aren't ``.safetensors``. Release is
    ``None`` for raw-PCA filenames (no ``_sae-`` marker).
    """
    if not filename.endswith(".safetensors"):
        return None
    stem = filename[: -len(".safetensors")]
    if _VARIANT_SEP in stem:
        model, release = stem.split(_VARIANT_SEP, 1)
        return model, release
    return stem, None
