"""Source registry + local metadata cache for a session-resident SAE.

Weights remain in the normal Hugging Face cache owned by SAELens. Saklas only
persists the resolved release/layer binding and optional per-feature metadata
(Neuronpedia display labels + ``maxActApprox``) under
``models/<safe>/sae/bindings``. Saklas-trained weights are a separate local
artifact family under ``sae/local`` (:mod:`saklas.io.sae_artifacts`).

The active-source half is the J-lens family's twin and shares its
implementation: :data:`SAE_SOURCES` is a
:class:`~saklas.io.source_registry.ActiveSourceRegistry`, so both families get
the same locking, format-version, kind-whitelist, name-grammar, and
existence-precondition discipline. :func:`load_active_sae` is the counterpart
of ``io.lens.load_lens`` — the ONE place the ``local:`` / ``saelens:`` source
grammar is parsed, so callers stop re-deriving the prefix convention.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import re
from typing import Any

from saklas.io.atomic import artifact_lock, write_json_atomic
from saklas.io.integrity import NAME_REGEX
from saklas.io.paths import decode_release_id, encode_release_id, model_dir
from saklas.io.source_registry import ActiveSourceRegistry

SAE_RUNTIME_FORMAT_VERSION = 3
SAE_SOURCE_FORMAT_VERSION = 1
_RUNTIME_FIELDS = {
    "layer", "width", "revision", "fingerprint", "sae_id", "repo_id",
    "neuronpedia_id",
}
# The two source-grammar prefixes, spelled once.
LOCAL_SOURCE_PREFIX = "local:"
PROVIDER_SOURCE_PREFIX = "saelens:"


def safe_release_id(release: str) -> str:
    return encode_release_id(release)


def sae_runtime_dir(model_id: str) -> Path:
    return model_dir(model_id) / "sae"


def sae_bindings_dir(model_id: str) -> Path:
    return sae_runtime_dir(model_id) / "bindings"


def sae_active_path(model_id: str) -> Path:
    return sae_runtime_dir(model_id) / "active.json"


def sae_metadata_path(model_id: str, release: str) -> Path:
    return sae_bindings_dir(model_id) / f"{safe_release_id(release)}.json"


def sae_features_path(model_id: str, release: str) -> Path:
    return sae_bindings_dir(model_id) / f"{safe_release_id(release)}-features.json"


def _validate_source_name(kind: str, name: str) -> None:
    """Name grammar per kind.

    A local artifact is Saklas-owned, so it obeys the shared artifact-name
    grammar.  A provider release id is provider-shaped (slashes, dots, mixed
    case are all legitimate), so it only has to be a non-blank string.
    """
    if kind == "local":
        if NAME_REGEX.fullmatch(name) is None:
            raise ValueError(f"local SAE name must match {NAME_REGEX.pattern}")
    elif not name.strip():
        raise ValueError("SAE source name must not be empty")


def _source_exists(root: Path, kind: str, name: str) -> bool:
    """Whether the named source is on disk, so a selection can never dangle."""
    if kind == "local":
        return (root / "local" / name / "manifest.json").exists()
    return (
        root / "bindings" / f"{safe_release_id(name)}.json"
    ).exists()


#: The active-source selection, shared implementation with the J-lens family.
SAE_SOURCES = ActiveSourceRegistry(
    label="SAE",
    format_version=SAE_SOURCE_FORMAT_VERSION,
    kinds=("local", "saelens"),
    validate_name=_validate_source_name,
    exists=_source_exists,
)


def set_active_sae_source(model_id: str, kind: str, name: str) -> Path:
    """Select the session-resident SAE source (locked, validated, existing)."""
    return SAE_SOURCES.write(sae_runtime_dir(model_id), model_id, kind, name)


def load_active_sae_source(model_id: str) -> dict[str, Any] | None:
    """The validated ``{format_version, model_id, kind, name}`` selection."""
    return SAE_SOURCES.read(sae_runtime_dir(model_id), model_id)


def sae_source_release(active: dict[str, Any]) -> str:
    """Render a selection back into the public ``<kind>:<name>`` grammar.

    ``local:<name>`` for a Saklas-trained artifact; a bare release id for a
    provider source (what ``load_sae_backend`` and the metadata cache key on).
    This is the ONE place the prefix convention is applied.
    """
    if active["kind"] == "local":
        return f"{LOCAL_SOURCE_PREFIX}{active['name']}"
    return str(active["name"])


def load_active_sae(model_id: str) -> tuple[str, dict[str, Any] | None] | None:
    """Resolve the active SAE selection to ``(release, provider_metadata)``.

    The counterpart of :func:`saklas.io.lens.load_lens`: source dispatch lives
    in io, so callers do not re-derive the ``local:`` / ``saelens:`` prefix
    convention or re-decide when to read the provider binding.  Returns
    ``None`` when nothing is selected; ``metadata`` is the validated runtime
    binding for a provider source (carrying the explicitly selected hook
    layer) and ``None`` for a local artifact, whose manifest is the binding.
    """
    active = load_active_sae_source(model_id)
    if active is None:
        return None
    release = sae_source_release(active)
    metadata = (
        load_sae_metadata(model_id, release)
        if active["kind"] == "saelens" else None
    )
    return release, metadata


def save_sae_metadata(
    model_id: str, release: str, payload: dict[str, Any], *,
    activate: bool = True,
) -> Path:
    """Persist a provider release's runtime binding under the artifact lock.

    ``activate=False`` records the binding without selecting it — the same
    opt-out ``fetch_neuronpedia_lens`` has, for a caller that fetches or
    refreshes one source while another stays resident.
    """
    if set(payload) != _RUNTIME_FIELDS:
        raise ValueError(
            f"SAE runtime metadata fields must be {sorted(_RUNTIME_FIELDS)}"
        )
    full = {
        **payload,
        "format_version": SAE_RUNTIME_FORMAT_VERSION,
        "model_id": model_id,
        "release": release,
    }
    if not _validate_runtime_payload(full, model_id, release):
        raise ValueError("invalid SAE runtime metadata values")
    path = sae_metadata_path(model_id, release)
    with artifact_lock(path):
        write_json_atomic(path, full)
    if activate:
        set_active_sae_source(model_id, "saelens", release)
    return path


def _validate_runtime_payload(
    payload: Any, model_id: str, release: str,
) -> bool:
    expected = {"format_version", "model_id", "release", *_RUNTIME_FIELDS}
    if not isinstance(payload, dict) or set(payload) != expected:
        return False
    if (
        payload["format_version"] != SAE_RUNTIME_FORMAT_VERSION
        or payload["model_id"] != model_id
        or payload["release"] != release
        or isinstance(payload["layer"], bool)
        or not isinstance(payload["layer"], int)
        or payload["layer"] < 0
        or isinstance(payload["width"], bool)
        or not isinstance(payload["width"], int)
        or payload["width"] <= 0
        or not isinstance(payload["revision"], str)
        or not payload["revision"]
        or not isinstance(payload["fingerprint"], str)
        or not payload["fingerprint"]
    ):
        return False
    return all(
        payload[key] is None or (
            isinstance(payload[key], str) and bool(payload[key].strip())
        )
        for key in ("sae_id", "repo_id", "neuronpedia_id")
    )


def load_sae_metadata(model_id: str, release: str) -> dict[str, Any] | None:
    path = sae_metadata_path(model_id, release)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not _validate_runtime_payload(payload, model_id, release):
        return None
    return payload


def _validate_feature_entry(value: Any) -> dict[str, Any] | None:
    """Validate one exact current ``{label, max_act}`` feature row."""
    if not isinstance(value, dict) or set(value) != {"label", "max_act"}:
        return None
    label = value["label"]
    if label is not None and not (isinstance(label, str) and label.strip()):
        return None
    max_act = value["max_act"]
    if max_act is not None and (
        isinstance(max_act, bool)
        or not isinstance(max_act, (int, float))
        or not math.isfinite(float(max_act))
        or float(max_act) <= 0
    ):
        return None
    return {"label": label, "max_act": None if max_act is None else float(max_act)}


def load_sae_feature_meta(model_id: str, release: str) -> dict[str, dict[str, Any]]:
    """Load the current ``{feature_id: {label, max_act}}`` metadata cache."""
    path = sae_features_path(model_id, release)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if (
        not isinstance(payload, dict)
        or set(payload) != {"format_version", "model_id", "release", "features"}
        or payload["format_version"] != SAE_RUNTIME_FORMAT_VERSION
        or payload["model_id"] != model_id
        or payload["release"] != release
    ):
        return {}
    features = payload["features"]
    if not isinstance(features, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in features.items():
        if not isinstance(key, str) or not re.fullmatch(r"0|[1-9][0-9]*", key):
            return {}
        entry = _validate_feature_entry(value)
        if entry is not None:
            out[key] = entry
        else:
            return {}
    return out


def save_sae_feature_meta(
    model_id: str, release: str, features: dict[str, dict[str, Any]],
) -> Path:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in features.items():
        key_value: Any = key
        if (
            not isinstance(key_value, str)
            or not re.fullmatch(r"0|[1-9][0-9]*", key_value)
        ):
            raise ValueError(f"invalid SAE feature id {key!r}")
        # The resident session may carry ephemeral lookup state (currently
        # ``checked``). Persist only the two fields in the cache contract.
        row = {"label": value.get("label"), "max_act": value.get("max_act")}
        entry = _validate_feature_entry(row)
        if entry is None:
            raise ValueError(f"invalid SAE feature metadata row {key!r}")
        normalized[key] = entry
    path = sae_features_path(model_id, release)
    with artifact_lock(path):
        write_json_atomic(path, {
            "format_version": SAE_RUNTIME_FORMAT_VERSION,
            "model_id": model_id,
            "release": release,
            "features": normalized,
        })
    return path


def use_sae_source(model_id: str, source: str) -> Path:
    """Select a source from the public ``local:NAME`` / ``saelens:RELEASE`` grammar."""
    source = source.strip()
    if source.startswith(LOCAL_SOURCE_PREFIX):
        from saklas.io.sae_artifacts import (
            load_local_sae_manifest,
            normalize_local_sae_name,
        )

        name = normalize_local_sae_name(source)
        if load_local_sae_manifest(model_id, name) is None:
            raise FileNotFoundError(f"local SAE {name!r} is not trained")
        return set_active_sae_source(model_id, "local", name)
    if source.startswith(PROVIDER_SOURCE_PREFIX):
        release = source[len(PROVIDER_SOURCE_PREFIX):]
        if load_sae_metadata(model_id, release) is None:
            raise FileNotFoundError(
                f"SAELens release {release!r} has not been fetched for {model_id}"
            )
        return set_active_sae_source(model_id, "saelens", release)
    raise ValueError("SAE source must be local:NAME or saelens:RELEASE")


def remove_sae_binding(model_id: str, release: str) -> bool:
    """Forget an external release without deleting SAELens/HF cache bytes."""
    metadata = sae_metadata_path(model_id, release)
    features = sae_features_path(model_id, release)
    with artifact_lock(metadata):
        removed = metadata.exists() or features.exists()
        metadata.unlink(missing_ok=True)
        features.unlink(missing_ok=True)
    # Unpublish after the binding is gone, so the selection can never outlive
    # what it points at.
    SAE_SOURCES.clear_if_active(
        sae_runtime_dir(model_id), model_id, "saelens", release,
    )
    return removed


def list_sae_sources(model_id: str) -> list[dict[str, Any]]:
    from saklas.io.sae_artifacts import list_local_saes

    rows = list_local_saes(model_id)
    active = load_active_sae_source(model_id)
    for path in sorted(sae_bindings_dir(model_id).glob("_z*.json")):
        if path.name.endswith("-features.json"):
            continue
        try:
            release = decode_release_id(path.stem)
        except ValueError:
            continue
        metadata = load_sae_metadata(model_id, release)
        if metadata is None:
            continue
        rows.append({
            "source": f"saelens:{release}",
            "kind": "saelens",
            "name": release,
            "active": bool(
                active is not None
                and active["kind"] == "saelens"
                and active["name"] == release
            ),
            "path": str(path),
            "layer": metadata["layer"],
            "features": metadata["width"],
        })
    return rows
