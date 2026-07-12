"""On-disk format for manifold-steering artifacts (re-export shim).

This module is the stable public import surface for the manifold on-disk
format.  The implementation is split across three submodules — the bodies
moved out, the names re-export here, so ``from saklas.io.manifolds import X``
keeps working unchanged everywhere:

- :mod:`saklas.io.manifold_folder` — the on-disk format core: the
  dataclasses, integrity/sha256, ``manifold.json`` load/save, and the
  shared module-private validators + payload builders.
- :mod:`saklas.io.manifold_authoring` — folder discovery and the
  create/init/append/plan/merge/update authoring path.
- :mod:`saklas.io.manifold_lifecycle` — lifecycle (rm/clear/refresh),
  cross-model transfer, and the shared summary serializer.

Bundled materialization (:func:`materialize_bundled_manifolds`,
:func:`bundled_manifold_names`, and the ``_materialized_this_process``
process-scope flag) lives **physically in this module**, not a submodule.
It is the one stateful, monkeypatched-by-path surface here — tests reset
``saklas.io.manifolds._materialized_this_process`` directly — so keeping
the real flag and the function that reads it (via this module's ``global``)
in one place preserves that contract with zero edits at any import site.

A *manifold* is a set of labeled nodes — each node a small corpus of
statements — placed at authoring coordinates on a :class:`ManifoldDomain`
(an n-dimensional intrinsic manifold of some topology: a box/disk, a
cylinder, a torus, a sphere, or an explicit immersion).  Fitting a
manifold against a model produces a per-model RBF artifact (see
:mod:`saklas.core.manifold`).  Manifolds live under their own root,
``~/.saklas/manifolds/<ns>/<name>/``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from importlib import resources as _resources
from pathlib import Path
from typing import Any

from saklas.io.atomic import write_bytes_atomic, write_json_atomic
from saklas.io.paths import manifolds_dir, saklas_home

# -- format core ---------------------------------------------------------
from saklas.io.manifold_folder import (
    MANIFOLD_FORMAT_VERSION,
    BakedManifoldError,
    DiscoverGenerationPlan,
    ManifoldFolder,
    ManifoldFormatError,
    ManifoldSidecar,
    _canonical_json,
    _node_filename,
    sanitize_hyperparams,
    domain_label,
    hash_manifold_files,
    min_nodes,
)

# -- discovery + authoring ----------------------------------------------
from saklas.io.manifold_authoring import (
    append_discover_manifold_node,
    create_baked_manifold_folder,
    create_discover_manifold_folder,
    create_manifold_folder,
    create_manifold_from_template,
    init_discover_manifold_folder,
    iter_manifold_folders,
    merge_discover_manifolds,
    plan_discover_generation,
    save_baked_manifold_tensor,
    update_manifold_folder,
)

# -- lifecycle / transfer / summary -------------------------------------
from saklas.io.manifold_lifecycle import (
    TransferSourceProof,
    clear_manifold_tensors,
    manifold_summary,
    preflight_transfer_manifold,
    refresh_manifold,
    remove_manifold_folder,
    transfer_manifold,
)

_log = logging.getLogger("saklas.io.manifolds")

# Process-scope flag: set True after the first ``materialize_bundled_manifolds``
# call so subsequent calls within the same Python process are no-ops.  See
# the docstring on that function for rationale (avoids clobbering CLI-set
# hyperparams on session re-init within the same invocation).  This flag and
# the function that reads it stay physically in this module so tests can reset
# it via ``saklas.io.manifolds._materialized_this_process`` (see the module
# docstring).
_materialized_this_process: bool = False


# ====================================================== bundled materialization ===
#
# Parallel to ``saklas.io.packs.materialize_bundled`` but for the
# manifold artifact kind.  Bundled manifolds live under
# ``saklas/data/manifolds/<name>/`` in the wheel and materialize into
# ``~/.saklas/manifolds/default/<name>/`` on session startup.  JSON-only
# on the shipped side — per-model ``.safetensors`` fits are produced on
# the user's machine via ``saklas manifold discover``.


def _bundled_manifest_node_filenames(pkg_root: Any) -> set[str] | None:
    """Expected node corpus filenames for a bundled manifest, or ``None`` if invalid."""
    try:
        with pkg_root.joinpath("manifold.json").open(encoding="utf-8") as f:
            payload = json.load(f)
    except (
        AttributeError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
    ):
        return None
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return None
    names: set[str] = set()
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            return None
        label = node.get("label")
        if not isinstance(label, str):
            return None
        names.add(_node_filename(idx, label))
    return names


def _bundled_manifest_complete(pkg_root: Any) -> bool:
    """True when a package-data manifold has every node corpus it declares."""
    node_files = _bundled_manifest_node_filenames(pkg_root)
    if node_files is None:
        return False
    if not node_files:
        return True
    nodes_root = pkg_root.joinpath("nodes")
    return nodes_root.is_dir() and all(
        nodes_root.joinpath(name).is_file() for name in node_files
    )


def bundled_manifold_names() -> list[str]:
    """List complete manifolds shipped under ``saklas/data/manifolds/``."""
    try:
        root = _resources.files("saklas.data.manifolds")
    except (ModuleNotFoundError, FileNotFoundError):
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and _bundled_manifest_complete(p)
    )


def _manifest_content_sha256(data: bytes) -> str:
    """Bundle-drift sha of a manifest payload — canonical JSON (sorted
    keys, no whitespace, so cosmetic differences compare equal) with local
    fit-transaction state (``files`` / ``artifact_id`` / ``fit_epochs``)
    stripped before hashing.

    ``files`` accumulates per-model fit proofs *locally*
    (:meth:`ManifoldFolder.update_file_hashes` after every fit), so it is
    local state, not bundle content.  Comparing the raw manifest against
    the shipped one misreads every fit as a bundle update; the refresh
    then clobbered the manifest with the shipped bytes (whose ``files`` is
    empty), orphaning the fitted tensors — the strict per-tensor loader
    refuses a tensor with no proof.  Falls back to a raw sha256 if the
    bytes don't parse as JSON, so unparseable on-disk content is treated
    as "user edited" rather than silently overwritten.
    """
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return hashlib.sha256(data).hexdigest()
    if isinstance(parsed, dict):
        for key in ("files", "artifact_id", "fit_epochs"):
            parsed.pop(key, None)
    return hashlib.sha256(_canonical_json(parsed)).hexdigest()


def _is_bundled_json_file(entry: Any) -> bool:
    """True for package-data payloads materialized from a bundled manifold."""
    return entry.is_file() and entry.name.endswith(".json")


def _copy_bundled_manifold_fresh(pkg_root: Any, target: Path) -> None:
    """Fresh install of a bundled manifold — copy JSON payloads only."""
    target.mkdir(parents=True, exist_ok=True)
    for entry in pkg_root.iterdir():
        if entry.is_file() and entry.name == "manifold.json":
            write_bytes_atomic(target / entry.name, entry.read_bytes())
        elif entry.is_dir() and entry.name == "nodes":
            nodes_dir = target / "nodes"
            nodes_dir.mkdir(parents=True, exist_ok=True)
            for node_file in entry.iterdir():
                if _is_bundled_json_file(node_file):
                    write_bytes_atomic(
                        nodes_dir / node_file.name, node_file.read_bytes(),
                    )


def _refresh_all_bundled_nodes(pkg_root: Any, target: Path) -> None:
    """Re-copy every shipped node file unconditionally.

    Bundle-update path — the manifest moved under the user, so any
    node-level "edits" are stale-against-old-bundle (the corpus that
    statement at position N referred to no longer matches what bundle-
    position-N currently is).  Better to drop them and have the user
    re-edit against the new bundle than to silently mix two corpora.

    Stale node files from the old bundle are removed if they don't
    exist in the new bundle (label set change).
    """
    pkg_nodes = pkg_root.joinpath("nodes")
    if not pkg_nodes.is_dir():
        return
    target_nodes = target / "nodes"
    target_nodes.mkdir(parents=True, exist_ok=True)
    bundled_names: set[str] = set()
    for node_file in pkg_nodes.iterdir():
        if not _is_bundled_json_file(node_file):
            continue
        bundled_names.add(node_file.name)
        write_bytes_atomic(target_nodes / node_file.name, node_file.read_bytes())
    # Drop any on-disk node files that aren't in the new bundle (label
    # set shrank or rename happened).  Without this, an old roster's
    # files would linger and confuse the loader.
    for stale in target_nodes.iterdir():
        if stale.is_file() and stale.name not in bundled_names:
            stale.unlink()


def _materialize_one_bundled_manifold(default_dir: Path, name: str) -> None:
    """Refresh one bundled folder under the same lock used by fitting."""
    from saklas.io.manifold_folder import _locked_manifest

    target = default_dir / name
    pkg_root = _resources.files("saklas.data.manifolds").joinpath(name)
    with _locked_manifest(target):
        if not target.exists():
            _copy_bundled_manifold_fresh(pkg_root, target)
            return

        on_disk_manifest = target / "manifold.json"
        if not on_disk_manifest.exists():
            return
        try:
            with open(on_disk_manifest) as f:
                on_disk_payload = json.load(f)
        except Exception:
            return

        bundled_manifest_bytes = (pkg_root / "manifold.json").read_bytes()
        on_disk_manifest_bytes = on_disk_manifest.read_bytes()
        # Drift-compare with the local ``files`` integrity map stripped —
        # fit proofs are local state, and comparing them against the
        # shipped manifest made every fitted bundled manifold read as a
        # bundle update on the next launch (the refresh then wiped the
        # proofs, orphaning the tensors).
        manifest_changed = (
            _manifest_content_sha256(on_disk_manifest_bytes)
            != _manifest_content_sha256(bundled_manifest_bytes)
        )
        fmt = on_disk_payload.get("format_version")
        format_stale = isinstance(fmt, int) and fmt < MANIFOLD_FORMAT_VERSION
        if not manifest_changed and not format_stale:
            return

        write_bytes_atomic(
            on_disk_manifest.with_suffix(".json.bak"), on_disk_manifest_bytes,
        )
        # A genuine bundle update: take the shipped manifest, but carry the
        # per-model fit proofs forward for artifacts still on disk that the
        # bundle doesn't ship (fitted tensors + sidecars).  The tensors
        # deliberately stay put across bundle updates; without their proofs
        # the strict loader can't read them, and re-verification against
        # the carried hash still runs on every load — nothing is laundered,
        # while the per-tensor ``nodes_sha256`` staleness check remains the
        # thing that decides whether an old fit is still current.
        try:
            merged_payload = json.loads(bundled_manifest_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            merged_payload = None
        old_files = on_disk_payload.get("files", {})
        if isinstance(merged_payload, dict) and isinstance(old_files, dict):
            shipped = {
                entry.name for entry in pkg_root.iterdir()
                if _is_bundled_json_file(entry)
            }
            carried = {
                fname: digest
                for fname, digest in old_files.items()
                if isinstance(fname, str)
                and fname not in shipped
                and (target / fname).is_file()
            }
            bundled_files = merged_payload.get("files", {})
            merged_payload["files"] = {
                **carried,
                **(bundled_files if isinstance(bundled_files, dict) else {}),
            }
            write_json_atomic(on_disk_manifest, merged_payload)
        else:
            write_bytes_atomic(on_disk_manifest, bundled_manifest_bytes)
        _refresh_all_bundled_nodes(pkg_root, target)
        for entry in pkg_root.iterdir():
            if not _is_bundled_json_file(entry) or entry.name == "manifold.json":
                continue
            write_bytes_atomic(target / entry.name, entry.read_bytes())

        reason = (
            f"v{fmt}->v{MANIFOLD_FORMAT_VERSION} (format_version)"
            if format_stale else "manifest content changed"
        )
        warnings.warn(
            f"materialize_bundled_manifolds: refreshed default/{name} — "
            f"{reason}; any local edits to its node corpus were overwritten "
            f"(fork under local/ to keep a custom corpus)",
            UserWarning,
            stacklevel=3,
        )
        _log.warning(
            "materialize_bundled_manifolds: refreshed default/%s — %s "
            "(node corpus re-copied, local edits overwritten)",
            name, reason,
        )


def materialize_bundled_manifolds() -> None:
    """Copy bundled manifolds into ``~/.saklas/manifolds/default/``.

    For each complete ``saklas/data/manifolds/<name>/`` in the wheel, ensure
    ``~/.saklas/manifolds/default/<name>/`` is current.  Mirrors
    :func:`saklas.io.packs.materialize_bundled` for the manifold artifact
    kind; only touches ``manifold.json`` and ``nodes/*.json`` since
    bundled manifolds ship JSON-only (no per-model ``.safetensors`` —
    those are user-side fits). Non-JSON filesystem metadata is ignored.

    Three paths:

    - **Fresh install** (target dir doesn't exist) — copy every shipped
      file atomically.
    - **Bundle update** (canonical-JSON hash of bundled ``manifold.json``
      differs from materialized, OR on-disk ``format_version`` is older
      than :data:`MANIFOLD_FORMAT_VERSION`) — re-copy ``manifold.json``
      in place (writing a ``.bak``), re-copy every node file
      unconditionally, re-copy any other top-level shipped files.
    - **No change** (manifest hashes match AND format_version is
      current) — skip.

    Bundle-update intentionally does NOT preserve user edits to node
    files.  A node-level "user edit" is meaningful only relative to a
    specific bundle version; once the bundle has moved (manifest hash
    differs), the edit is stale-against-old-bundle and silently keeping
    it would mix corpora from two versions.  Users who want to override
    a bundled node corpus should fork it under a different namespace
    (``saklas manifold generate ...`` or hand-author under
    ``local/<name>/``) rather than edit the default-namespace copy.

    Per-model ``.safetensors`` tensor files stay put on bundle update —
    they're expensive to refit and the per-tensor ``nodes_sha256``
    check invalidates them automatically on next discover/fit.

    **Process-scoped no-op after first call.**  Subsequent calls within
    the same process return immediately without touching disk.  This
    prevents a second materialize (from e.g. ``SaklasSession.from_pretrained``
    later in the same CLI invocation) from clobbering CLI-set
    hyperparams that the runner wrote between the two calls — the
    materialize-detects-bundle-update logic can't distinguish
    "bundle changed under user" from "user changed manifest via CLI
    override", and process-scope caching sidesteps the entire
    ambiguity.  A long-running server that wants to pick up a bundle
    update mid-process would need a restart; this is not a real use
    case (bundle updates ship via pip and require restart anyway).
    """
    global _materialized_this_process
    if _materialized_this_process:
        return
    _materialized_this_process = True

    home = saklas_home()
    home.mkdir(parents=True, exist_ok=True)

    default_dir = manifolds_dir() / "default"
    default_dir.mkdir(parents=True, exist_ok=True)
    for name in bundled_manifold_names():
        _materialize_one_bundled_manifold(default_dir, name)


__all__ = [
    "MANIFOLD_FORMAT_VERSION",
    "min_nodes",
    "ManifoldFormatError",
    "BakedManifoldError",
    "ManifoldSidecar",
    "ManifoldFolder",
    "hash_manifold_files",
    "iter_manifold_folders",
    "create_manifold_folder",
    "create_manifold_from_template",
    "create_discover_manifold_folder",
    "create_baked_manifold_folder",
    "save_baked_manifold_tensor",
    "init_discover_manifold_folder",
    "append_discover_manifold_node",
    "plan_discover_generation",
    "DiscoverGenerationPlan",
    "merge_discover_manifolds",
    "update_manifold_folder",
    "clear_manifold_tensors",
    "remove_manifold_folder",
    "refresh_manifold",
    "TransferSourceProof",
    "preflight_transfer_manifold",
    "transfer_manifold",
    "manifold_summary",
    "domain_label",
    "bundled_manifold_names",
    "materialize_bundled_manifolds",
    "sanitize_hyperparams",
]
