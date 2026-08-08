"""Immutable per-layer shard generations behind an atomic JSON pointer.

Three per-model artifact families store the same shape: one immutable
``<stem>.layer-<L>.gen-<uuid>.safetensors`` payload per layer, plus a JSON
sidecar that names the live generation for each layer under ``tensor_files``.
Publication is a single atomic pointer replace; a reader only ever sees a
complete generation set, and superseded payloads are collected *after* the new
pointer is durable.

The families are the neutral-activation cache and the Procrustes alignment
cache (:mod:`saklas.io.alignment`) and the Saklas-fitted Jacobian lens
(:mod:`saklas.io.lens`).  Their sidecar *schemas* and format versions stay
private to each module — what lives here is only the path/GC/lock primitive
they share, so crash-recovery-critical behavior cannot drift between them.
``label`` names the artifact in error and log messages; it is the sole
per-family difference.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
import logging
from pathlib import Path
from typing import Any
import uuid

log = logging.getLogger(__name__)


def generation_path(anchor: Path, layer: int) -> Path:
    """Name a fresh, never-reused payload generation for one layer.

    ``anchor`` is the family's stable path (also the lock anchor); the returned
    sibling carries the layer and a random generation id, so publishing a new
    fit never overwrites bytes a concurrent reader may have mapped.
    """
    return anchor.with_name(
        f"{anchor.stem}.layer-{int(layer)}.gen-{uuid.uuid4().hex}{anchor.suffix}",
    )


def shard_paths(
    anchor: Path,
    sidecar: Mapping[str, Any],
    layers: list[int],
    *,
    label: str,
    require_exact_keys: bool = True,
) -> dict[int, Path]:
    """Resolve every requested layer to its declared immutable generation.

    A ``tensor_files`` entry must be a bare filename — a manifest-supplied
    ``..`` or absolute path is rejected rather than resolved off-tree.  With
    ``require_exact_keys`` the sidecar's key set must equal ``layers`` exactly
    (a full-artifact read); a scoped read passes ``False`` to materialize a
    subset of a wider pointer.
    """
    files = sidecar.get("tensor_files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{label}: sidecar has no tensor shard map")
    if require_exact_keys and (
        {str(layer) for layer in layers} != {str(key) for key in files}
    ):
        raise ValueError(f"{label}: tensor shard keys do not match the declared layers")
    out: dict[int, Path] = {}
    for layer in layers:
        filename = files.get(str(layer))
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"{label}: invalid tensor shard for layer {layer}")
        out[layer] = anchor.parent / filename
    return out


def representative_shard_path(
    anchor: Path, sidecar: Mapping[str, Any], *, label: str,
) -> Path:
    """Resolve the lowest-layer generation, for path-reporting callers.

    Surfaces one concrete payload path where an API historically returned the
    single pre-sharded file.  Raises when the pointer carries no usable shard
    map — callers that must not fail fall back to ``anchor`` themselves.
    """
    files = sidecar.get("tensor_files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"{label}: sidecar has no tensor shard map")
    try:
        first_key = min(files, key=lambda key: int(str(key)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid tensor shard keys") from exc
    filename = files[first_key]
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise ValueError(f"{label}: invalid tensor shard filename")
    return anchor.parent / filename


def json_pointer_matches(path: Path, payload: Mapping[str, Any]) -> bool:
    """Whether an exception escaped after this exact pointer was replaced.

    A failure between the pointer write and the end of the publishing
    transaction is only recoverable-as-published when the durable pointer is
    byte-for-byte the payload the writer intended.
    """
    try:
        with open(path) as handle:
            current = json.load(handle)
        return current == payload
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def cleanup_generations(
    anchor: Path, sidecar: Mapping[str, Any], *, label: str,
) -> None:
    """Collect superseded generations and crash-left streaming temporaries.

    Only ever called once the pointer naming the surviving generations is
    durable, so a power loss between publication and collection leaves extra
    payloads rather than a pointer whose bytes are gone.  Removal failures are
    logged, never raised — an uncollected generation is inert.
    """
    files = sidecar.get("tensor_files")
    keep = (
        {str(filename) for filename in files.values() if isinstance(filename, str)}
        if isinstance(files, Mapping) else set()
    )
    for path in (
        *anchor.parent.glob(f"{anchor.stem}.layer-*.gen-*{anchor.suffix}"),
        *anchor.parent.glob(f"{anchor.stem}.layer-*.gen-*{anchor.suffix}.tmp"),
        anchor,
        anchor.with_suffix(anchor.suffix + ".tmp"),
    ):
        if path.name not in keep:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("%s: could not remove old generation %s: %s", label, path, exc)


@contextmanager
def fit_lock(anchor: Path) -> Iterator[None]:
    """Single-flight the expensive fit/lifecycle transaction for one artifact.

    Keyed on a ``<stem>.fit`` sibling of the stable anchor rather than the
    anchor itself, so the fit lock is held across model construction and
    publication without blocking the short pair-lock transactions that read
    the current pointer.
    """
    from saklas.io.atomic import artifact_lock

    with artifact_lock(anchor.with_name(f"{anchor.stem}.fit")):
        yield
