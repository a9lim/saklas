"""Lifecycle, cross-model transfer, and the shared summary serializer.

Every function addressed by ``(namespace, name)`` that deletes / relocates
/ re-pulls a manifold's fitted tensors (rm / clear / refresh), the
cross-model Procrustes transfer, and the session-independent
``manifold_summary`` serializer.  Format primitives come from
:mod:`saklas.io.manifold_folder`.

Bundled materialization (``materialize_bundled_manifolds`` and the
``_materialized_this_process`` flag) deliberately does **not** live here —
it stays physically in :mod:`saklas.io.manifolds` so the process-scope
flag is monkeypatchable by its public attribute path with zero test edits.
``refresh_manifold`` reaches it through a lazy import of that module.
"""
from __future__ import annotations

import json
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Optional

from saklas.io.atomic import write_json_atomic
from saklas.core.errors import ManifoldExistsError, ManifoldNotFoundError, SaklasError
from saklas.core.manifold import domain_from_spec
from saklas.io.manifold_folder import (
    BakedManifoldError,
    ManifoldFolder,
    ManifoldFormatError,
    domain_label,
    manifold_pair_lock,
    min_nodes,
)
from saklas.io.paths import manifold_dir


# ============================================================ lifecycle (rm/clear/refresh) ===
#
# The manifold analogue of pack lifecycle in ``saklas.io.cache_ops``
# (``uninstall`` / ``delete_tensors`` / ``refresh``).  Manifolds don't
# go through the concept ``Selector``/``resolve`` machinery — they're
# addressed by ``(namespace, name)`` and discovered through
# ``iter_manifold_folders`` — so these are folder-level functions rather
# than selector-driven sweeps.  Source-tier semantics mirror packs: the
# ``manifold.json::source`` field (``"local"`` / ``"bundled"`` /
# ``"hf://..."``) decides refresh behavior.


def _manifold_tensor_variant_matches(key: str, filter_: str) -> bool:
    """Mirror ``cache_ops._variant_matches_key`` for manifold tensors.

    ``key`` is the variant slug a manifold tensor filename parses to:
    ``"raw"`` (unsuffixed, canonical), ``"sae-<release>"``, or
    ``"from-<safe_src>"`` (transferred).  ``filter_`` is one of ``"raw"``
    / ``"sae"`` / ``"from"`` / ``"all"`` — ``"from"`` selects transferred
    tensors, so a ``clear --variant from`` drops only the cross-model
    transfer variants while keeping the native fit.  Twin of
    ``cache_ops._variant_matches_key`` — kept in sync so the pack and
    manifold clear-filters recognize the same variant slugs (both match
    the ``_from-<safe_src>`` variant transfers produce).
    """
    if filter_ == "all":
        return True
    if filter_ == "raw":
        return key == "raw"
    if filter_ == "sae":
        return key.startswith("sae-")
    if filter_ == "from":
        return key.startswith("from-")
    return False


def _manifold_tensor_files(
    folder: Path, *, variant: str = "all", model_scope: Optional[str] = None,
) -> list[Path]:
    """Per-model fitted tensors + their ``.json`` sidecars under ``folder``.

    Scans fitted tensors and sidecars independently, filters by ``variant``
    (``raw`` / ``sae`` / ``from`` / ``all``), and keeps orphaned halves so
    ``clear`` can repair an interrupted or corrupted fit.  The
    node corpus and ``manifold.json`` are never touched — this is the
    fitted-artifact layer only.

    ``model_scope`` (a raw model id, e.g. ``"google/gemma-3-4b-it"``)
    narrows the result to a single model's tensors — the filename's
    parsed safe-model-id must equal ``safe_model_id(model_scope)``.
    ``None`` (default) keeps every model's tensors.  Mirrors the
    ``model_scope`` filter in :func:`saklas.io.cache_ops._tensor_files_for`,
    which does the same safe-id conversion at the io boundary.
    """
    from saklas.io.paths import parse_tensor_filename, safe_model_id

    target_safe = safe_model_id(model_scope) if model_scope is not None else None
    out: set[Path] = set()
    candidates = list(folder.glob("*.safetensors")) + [
        sidecar
        for sidecar in folder.glob("*.json")
        if sidecar.name != "manifold.json"
    ]
    for candidate in candidates:
        tensor_name = (
            candidate.with_suffix(".safetensors").name
            if candidate.suffix == ".json"
            else candidate.name
        )
        parsed = parse_tensor_filename(tensor_name)
        if parsed is None:
            continue
        model, var = parsed
        if target_safe is not None and model != target_safe:
            continue
        key = "raw" if var is None else var
        if not _manifold_tensor_variant_matches(key, variant):
            continue
        out.add(candidate)
    return sorted(out)


def _pair_tensor_paths(files: list[Path]) -> list[Path]:
    """Canonical tensor paths for a mixed tensor/sidecar file list."""

    return sorted({
        path.with_suffix(".safetensors") if path.suffix == ".json" else path
        for path in files
    })


@contextmanager
def _locked_manifold_pairs(files: list[Path]):
    """Acquire fitted pair locks in one deterministic global order."""

    with ExitStack() as stack:
        for tensor_path in _pair_tensor_paths(files):
            stack.enter_context(manifold_pair_lock(tensor_path))
        yield


def _capture_group_is_referenced(safe_model: str, capture_sha: str) -> bool:
    """Whether any surviving fitted sidecar still owns a shared capture stem."""
    from saklas.io.paths import manifolds_dir, parse_tensor_filename

    for sidecar_path in manifolds_dir().glob("*/*/*.json"):
        if sidecar_path.name == "manifold.json":
            continue
        parsed = parse_tensor_filename(
            sidecar_path.with_suffix(".safetensors").name,
        )
        if parsed is None or parsed[0] != safe_model:
            continue
        try:
            with open(sidecar_path) as handle:
                sidecar = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if sidecar.get("capture_sha256") == capture_sha:
            return True
    return False


def _remove_unreferenced_capture_groups(
    capture_groups: set[tuple[str, str]],
) -> None:
    """Garbage-collect content-addressed captures only after the last owner."""
    if not capture_groups:
        return
    from saklas.io.atomic import artifact_has_live_lease, artifact_lock
    from saklas.io.paths import models_dir

    for safe_model, capture_sha in capture_groups:
        cache_dir = models_dir() / safe_model / "manifold_capture"
        # The capture lock spans cache publication through fitted-sidecar
        # publication.  Scan and delete under the same lock so lifecycle GC
        # cannot race a different folder that is about to establish ownership.
        with artifact_lock(cache_dir / capture_sha):
            # A fit may have released the short cache transaction while still
            # consuming mapped row shards.  Its process lease is the temporary
            # owner; a later prune/clear pass reaps the group after the lease is
            # gone (including stale PID markers after a crash).
            if artifact_has_live_lease(cache_dir / capture_sha):
                continue
            if _capture_group_is_referenced(safe_model, capture_sha):
                continue
            for cached in cache_dir.glob(f"{capture_sha}.*"):
                cached.unlink()


def clear_manifold_tensors(
    namespace: str, name: str, model_scope: Optional[str] = None, *, variant: str = "all",
) -> int:
    """Atomically clear selected fitted pairs and their manifest entries."""
    from saklas.io.manifold_folder import _locked_manifest

    folder = manifold_dir(namespace, name)
    with _locked_manifest(folder):
        return _clear_manifold_tensors_locked(
            namespace, name, model_scope, variant=variant,
        )


def _clear_manifold_tensors_locked(
    namespace: str, name: str, model_scope: Optional[str] = None, *, variant: str = "all",
) -> int:
    """Delete a manifold's per-model fitted tensors, keeping the corpus.

    Mirrors ``saklas.io.cache_ops.delete_tensors`` for packs: removes the
    fitted ``<safe>*.safetensors`` + ``.json`` sidecars (so they re-fit
    on next use) while leaving ``manifold.json`` and the ``nodes/`` corpus
    in place.  ``variant`` filters by tensor flavor — ``"raw"`` only the
    unsuffixed canonical tensors, ``"sae"`` only ``_sae-*`` variants,
    ``"from"`` only ``_from-*`` transfer variants, ``"all"`` (default)
    every flavor.  ``model_scope`` (a raw model id) narrows
    deletion to that one model's tensors (safe-id-matched, the same
    convention ``delete_tensors`` uses); ``None`` (default) clears every
    model.  Returns the number of files deleted.

    Re-hashes ``manifold.json::files`` afterward (via ``write_metadata``)
    so the integrity manifest no longer references the removed files.
    Raises :class:`FileNotFoundError` when the manifold isn't installed.
    """
    folder = manifold_dir(namespace, name)
    if not (folder / "manifold.json").exists():
        raise ManifoldNotFoundError(f"manifold {namespace}/{name} not found at {folder}")
    # This is a repair operation: even the non-integrity folder loader parses
    # every fitted sidecar and rejects orphan halves. Read only the authoring
    # manifest so the corrupt selected artifacts remain deletable.
    manifest_path = folder / "manifold.json"
    try:
        with open(manifest_path) as handle:
            manifest_data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifoldFormatError(
            f"manifold {namespace}/{name} manifest is unreadable: {exc}"
        ) from exc
    if manifest_data.get("fit_mode") == "baked":
        # A baked manifold has no corpus to re-fit from — deleting its tensor
        # would destroy the only copy of its geometry.  Refuse; ``manifold rm``
        # is the way to remove it wholesale.
        raise BakedManifoldError(
            f"manifold {namespace}/{name} is baked (corpus-less, cannot "
            f"re-fit); clearing its tensors is irreversible — use "
            f"`manifold rm {namespace}/{name}` to delete it"
        )
    files = _manifold_tensor_files(folder, variant=variant, model_scope=model_scope)
    capture_groups: set[tuple[str, str]] = set()
    from saklas.io.paths import parse_tensor_filename

    with _locked_manifold_pairs(files):
        for sidecar_path in (path for path in files if path.suffix == ".json"):
            try:
                with open(sidecar_path) as handle:
                    sidecar = json.load(handle)
                capture_sha = sidecar.get("capture_sha256")
                parsed = parse_tensor_filename(
                    sidecar_path.with_suffix(".safetensors").name,
                )
                if isinstance(capture_sha, str) and len(capture_sha) == 64 and parsed:
                    capture_groups.add((parsed[0], capture_sha))
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        for f in files:
            f.unlink(missing_ok=True)
        # Repair only the selected manifest entries. Re-hashing every survivor
        # here would bless an unrelated corrupt model/variant while clearing
        # this one. Preserve untouched expected digests byte-for-byte.
        from saklas.io.paths import safe_model_id

        target_safe = safe_model_id(model_scope) if model_scope is not None else None

        def _selected_manifest_entry(filename: str) -> bool:
            candidate = (
                Path(filename).with_suffix(".safetensors").name
                if filename.endswith(".json")
                else filename
            )
            parsed = parse_tensor_filename(candidate)
            if parsed is None:
                return False
            safe_model, parsed_variant = parsed
            key = "raw" if parsed_variant is None else parsed_variant
            return (
                (target_safe is None or safe_model == target_safe)
                and _manifold_tensor_variant_matches(key, variant)
            )

        manifest_files = manifest_data.get("files", {})
        if not isinstance(manifest_files, dict):
            raise ManifoldFormatError("manifold files manifest must be an object")
        manifest_data["files"] = {
            name: digest
            for name, digest in manifest_files.items()
            if not _selected_manifest_entry(name)
        }
        # Invalidate only fitting targets selected by this clear.  A global
        # epoch would throw away unrelated model/variant compute.
        epochs_raw = manifest_data.get("fit_epochs", {})
        epochs = dict(epochs_raw) if isinstance(epochs_raw, dict) else {}
        selector_key = f"{target_safe or '*'}:{variant}"
        epochs[selector_key] = int(epochs.get(selector_key, 0)) + 1
        manifest_data["fit_epochs"] = epochs
        write_json_atomic(manifest_path, manifest_data)
    _remove_unreferenced_capture_groups(capture_groups)
    return len(files)


def remove_manifold_folder(namespace: str, name: str) -> dict[str, Any]:
    """Remove a whole folder while excluding concurrent fits/mutations."""
    from saklas.io.manifold_folder import _locked_manifest

    folder = manifold_dir(namespace, name)
    with _locked_manifest(folder):
        return _remove_manifold_folder_locked(namespace, name)


def _remove_manifold_folder_locked(namespace: str, name: str) -> dict[str, Any]:
    """Remove a whole manifold folder (rm), bundled-respawn semantics.

    The manifold analogue of ``saklas.io.cache_ops.uninstall`` for a
    single concept: ``rmtree`` the folder so the manifold ceases to
    exist.  Bundled manifolds (``default/`` namespace) re-materialize on
    next session init via :func:`materialize_bundled_manifolds`, exactly
    as bundled concepts do — the returned ``rematerializes_on_restart``
    flag lets a caller pick a friendlier message for that case.

    Returns ``{namespace, name, source, removed, rematerializes_on_restart}``.
    Raises :class:`FileNotFoundError` when the manifold isn't installed.
    """
    folder = manifold_dir(namespace, name)
    if not (folder / "manifold.json").exists():
        raise ManifoldNotFoundError(f"manifold {namespace}/{name} not found at {folder}")
    # Read the source tier before deleting (best-effort — a corrupt
    # manifest just reports the namespace-implied tier).
    try:
        source = ManifoldFolder.load(folder, verify_manifest=False).source
    except ManifoldFormatError:
        source = "bundled" if namespace == "default" else "local"
    rematerializes = namespace == "default" or source == "bundled"
    capture_groups: set[tuple[str, str]] = set()
    from saklas.io.paths import parse_tensor_filename

    fitted_files = _manifold_tensor_files(folder)
    with _locked_manifold_pairs(fitted_files):
        for sidecar_path in folder.glob("*.json"):
            if sidecar_path.name == "manifold.json":
                continue
            try:
                with open(sidecar_path) as handle:
                    sidecar = json.load(handle)
                capture_sha = sidecar.get("capture_sha256")
                parsed = parse_tensor_filename(
                    sidecar_path.with_suffix(".safetensors").name,
                )
                if isinstance(capture_sha, str) and len(capture_sha) == 64 and parsed:
                    capture_groups.add((parsed[0], capture_sha))
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        shutil.rmtree(folder)
    _remove_unreferenced_capture_groups(capture_groups)
    return {
        "namespace": namespace,
        "name": name,
        "source": source,
        "removed": True,
        "rematerializes_on_restart": rematerializes,
    }


def refresh_manifold(
    namespace: str, name: str, *, model_scope: Optional[str] = None, force: bool = True,
) -> str:
    """Refresh a folder while excluding concurrent fits/mutations."""
    from saklas.io.manifold_folder import _locked_manifest

    folder = manifold_dir(namespace, name)
    with _locked_manifest(folder):
        return _refresh_manifold_locked(
            namespace, name, model_scope=model_scope, force=force,
        )


def _refresh_manifold_locked(
    namespace: str, name: str, *, model_scope: Optional[str] = None, force: bool = True,
) -> str:
    """Re-pull / re-materialize a manifold from its source.

    Mirrors ``saklas.io.cache_ops.refresh`` per source tier:

    - ``local`` (or any source other than the two below) — nothing
      upstream to re-pull from; silently skipped, returns ``"skipped"``.
    - ``bundled`` (or ``default/`` namespace) — re-materialized from
      package data; returns ``"bundled"``.  Delegates to
      :func:`materialize_bundled_manifolds`, which is process-scoped, so
      a fresh process re-copy is the practical path.
    - ``hf://<owner>/<name>[@rev]`` — re-pulled into the same folder via
      :func:`saklas.io.hf_manifolds.pull_manifold`; returns ``"hf"``.

    When ``model_scope`` (a raw model id) is given the source tier is
    *bypassed* — exactly as ``cache_ops.refresh``'s scoped path does for
    packs: delete just that model's fitted tensor pair (via
    :func:`clear_manifold_tensors`) so it re-fits on next use, and do NOT
    re-pull from the upstream source.  Returns ``"scoped"``.

    ``force`` is threaded into ``pull_manifold`` (overwrite the existing
    install).  Raises :class:`FileNotFoundError` when the manifold isn't
    installed, :class:`ManifoldFormatError` on a corrupt manifest.
    """
    folder = manifold_dir(namespace, name)
    if not (folder / "manifold.json").exists():
        raise ManifoldNotFoundError(f"manifold {namespace}/{name} not found at {folder}")

    if model_scope is not None:
        # Scoped refresh: drop just that model's fitted tensor pair so it
        # re-fits from the node corpus on next use.  Mirrors the pack-side
        # tensors-only scoped refresh — a whole-repo re-pull for one model
        # makes no sense (HF pulls are whole-folder).
        clear_manifold_tensors(namespace, name, model_scope, variant="all")
        return "scoped"

    source = ManifoldFolder.load(folder, verify_manifest=False).source

    if namespace == "default" or source == "bundled":
        # Bundled tier — re-copy from package data.  Process-scoped, so
        # this is a no-op after the first materialize within a process.
        from saklas.io.manifolds import materialize_bundled_manifolds

        with _locked_manifold_pairs(_manifold_tensor_files(folder)):
            materialize_bundled_manifolds()
        return "bundled"

    if source.startswith("hf://"):
        from saklas.io.hf import split_revision
        from saklas.io.hf_manifolds import pull_manifold

        coord, revision = split_revision(source[len("hf://"):])
        with _locked_manifold_pairs(_manifold_tensor_files(folder)):
            pull_manifold(coord, target_folder=folder, force=force, revision=revision)
        return "hf"

    # local (or anything without an upstream): nothing to do.
    return "skipped"


# ============================================================ cross-model transfer ===
#
# Writes a ``_from-<safe_src>`` variant tensor for cross-model manifold transfer.
# Pure-io: the caller builds
# the per-layer Procrustes alignment map — that needs both models loaded,
# which lives in the session/CLI layer — and passes it in; this function
# only *applies* it to a fitted manifold's per-layer subspace and writes
# the transferred tensor.  Do not rebuild the Procrustes solver here.


def _transfer_preflight_locked(
    folder: Path, *, from_model: str, to_model: str, force: bool,
) -> tuple[Path, Path, dict[str, Any], dict[str, str], bool]:
    """Prove the source pair and classify the target with locks already held."""
    from saklas.io.packs import verify_integrity
    from saklas.io.paths import tensor_filename

    folder = Path(folder)
    source = folder / tensor_filename(from_model)
    target = folder / tensor_filename(to_model, transferred_from=from_model)
    with open(folder / "manifold.json") as handle:
        manifest = json.load(handle)
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise ManifoldFormatError("manifold files manifest must be an object")

    def trusted_pair(tensor: Path) -> bool:
        sidecar = tensor.with_suffix(".json")
        expected = (
            {tensor.name: files[tensor.name], sidecar.name: files[sidecar.name]}
            if tensor.name in files and sidecar.name in files else {}
        )
        return bool(expected) and verify_integrity(folder, expected)[0]

    if not trusted_pair(source):
        raise ManifoldNotFoundError(
            f"manifold {folder.name!r} has no trusted fit for source model "
            f"{from_model!r} at {source}"
        )
    target_trusted = trusted_pair(target)
    if target_trusted and not force:
        raise ManifoldExistsError(
            f"{target} already exists; pass force=True to overwrite"
        )
    return source, target, manifest, dict(files), target_trusted


def preflight_transfer_manifold(
    folder: Path, *, from_model: str, to_model: str, force: bool = False,
) -> None:
    """Cheap proof-aware transfer preflight before alignment/model work."""
    from saklas.io.manifold_folder import _locked_manifest
    from saklas.io.paths import tensor_filename

    folder = Path(folder)
    source = folder / tensor_filename(from_model)
    target = folder / tensor_filename(to_model, transferred_from=from_model)
    with _locked_manifest(folder):
        with _locked_manifold_pairs([source, target]):
            _transfer_preflight_locked(
                folder, from_model=from_model, to_model=to_model, force=force,
            )


def transfer_manifold(
    folder: Path,
    *,
    from_model: str,
    to_model: str,
    alignment: dict[int, Any],
    transfer_quality_estimate: Optional[float] = None,
    source_model_fingerprint: str,
    target_model_fingerprint: str,
    whitener: "Any | None" = None,
    force: bool = False,
) -> Path:
    """Publish one transferred tensor/sidecar/manifest update atomically."""
    from saklas.io.manifold_folder import _locked_manifest
    from saklas.io.paths import tensor_filename

    source = Path(folder) / tensor_filename(from_model)
    target = Path(folder) / tensor_filename(
        to_model, transferred_from=from_model,
    )
    with _locked_manifest(Path(folder)):
        with _locked_manifold_pairs([source, target]):
            return _transfer_manifold_locked(
                folder, from_model=from_model, to_model=to_model,
                alignment=alignment,
                transfer_quality_estimate=transfer_quality_estimate,
                source_model_fingerprint=source_model_fingerprint,
                target_model_fingerprint=target_model_fingerprint,
                whitener=whitener, force=force,
            )


def _transfer_manifold_locked(
    folder: Path,
    *,
    from_model: str,
    to_model: str,
    alignment: dict[int, Any],
    transfer_quality_estimate: Optional[float] = None,
    source_model_fingerprint: str,
    target_model_fingerprint: str,
    whitener: "Any | None" = None,
    force: bool = False,
) -> Path:
    """Apply a per-layer alignment map to a fitted manifold, target-side.

    Reads the source-model fit at ``<folder>/<safe_from>.safetensors``,
    maps each layer's affine subspace (``mean`` + ``basis`` rows, both in
    model space) through the supplied compact factorized affine alignment
    (the shape :func:`saklas.io.alignment.fit_alignment` produces), and writes a
    transferred per-model tensor at the ``_from-<safe_src>`` filename
    variant (``<safe_to>_from-<safe_from>.safetensors`` —
    :func:`saklas.io.paths.tensor_filename`'s transfer suffix).

    Translation applies to points/means but not directions.  The mapped basis
    is orthonormalized and the affine/RBF reduced coefficients are transformed
    exactly into that new frame; a rank collapse is rejected.  ``node_coords``
    (the intrinsic authoring layout) remains model-independent.  Layers the
    alignment doesn't cover are dropped,
    mirroring :func:`saklas.io.alignment.transfer_profile`.

    ``alignment`` is supplied by the caller (building it needs both
    models loaded — a session/CLI concern), keeping this function
    pure-io.  ``transfer_quality_estimate`` (median per-layer R², if the
    caller computed it) rides into the sidecar provenance.

    The subspace re-mapping + target-space share re-bake is the pure-tensor
    :func:`saklas.core.manifold.transfer_manifold_subspaces`; this function only
    loads the source tensor, calls it, and writes the transferred tensor +
    sidecar.  (It used to do the tensor math inline, which is why it lazily
    reached into ``core`` for ``LayerSubspace`` / ``eval_rbf`` / ``subspace_share``
    / ``WhitenerError``; those moved with the compute.)

    **Target-metric re-bake (mandatory).**  The fitted manifold's
    per-layer Mahalanobis share is a per-model quantity (``Σ`` belongs to
    ``from_model``), so a bare transfer can't carry it.  The ``whitener``
    for the **target** model is **required** and must cover every
    transferred layer (all-or-nothing, mirroring the fit gate); the share
    is recomputed in target space via
    ``sqrt(Σ_k coordsᵀ (B_tgt Σ_tgt⁻¹ B_tgtᵀ) coords)`` (the RBF reduced
    node values × the target subspace-restricted inverse covariance — the
    same formula the fit pipeline bakes).  The sidecar then records
    ``share_metric: "mahalanobis"``.  A missing or non-covering whitener
    raises :class:`~saklas.core.mahalanobis.WhitenerError` — there is no
    Euclidean rebake.  ``subspace_metric`` always carries the source
    value: the basis was *selected* on the source model and only rotated
    here.

    Returns the path to the written transferred tensor.  Raises
    :class:`FileNotFoundError` when the source fit is missing,
    :class:`ManifoldFormatError` when ``alignment`` is empty or covers no
    fitted layer, :class:`FileExistsError` when a transferred tensor
    already exists and ``force`` is ``False``.
    """
    from saklas.core.manifold import (
        load_manifold,
        save_manifold,
        transfer_manifold_subspaces,
    )

    folder = Path(folder)
    (
        src_tensor, out_path, manifest, files, target_trusted,
    ) = _transfer_preflight_locked(
        folder, from_model=from_model, to_model=to_model, force=force,
    )
    if not alignment:
        raise ManifoldFormatError(
            f"transfer_manifold: alignment map for {from_model!r} → "
            f"{to_model!r} is empty"
        )

    # Classify the destination before loading/folding the source manifold.  A
    # proven pair preserves ordinary exists semantics; an unproven pair is a
    # crash-left publication that this exact producer may repair.  The public
    # wrapper holds manifest + source/target pair locks, so this proof remains
    # authoritative until publication completes.
    sidecar_path = out_path.with_suffix(".json")

    src = load_manifold(src_tensor)
    # ``load_manifold`` returns metadata from the same locked pair read.  Do
    # not reopen the sidecar outside that transaction and create a TOCTOU gap.
    source_sidecar: dict[str, Any] = dict(src.metadata)
    if not source_model_fingerprint or not target_model_fingerprint:
        raise ManifoldFormatError(
            "transfer requires proven source and target model fingerprints"
        )
    if source_sidecar.get("model_fingerprint") != source_model_fingerprint:
        raise ManifoldFormatError(
            "transfer source manifold was fitted for different loaded weights; "
            "refit the source manifold before transfer"
        )
    # The subspace re-mapping + target-space share re-bake is pure-tensor
    # compute owned by ``core.manifold``; this io function only orchestrates the
    # folder read/write around it.  ``transfer_manifold_subspaces`` raises
    # ``WhitenerError`` (a ``SaklasError``) on a missing / partial target
    # whitener — let that propagate verbatim — and a plain ``ValueError`` when
    # the alignment covers no fitted layer, which we surface as the io-level
    # ``ManifoldFormatError`` (preserving the message) so the CLI / server
    # format-error handling is unchanged.  The ``SaklasError`` guard runs first
    # so the ``WhitenerError`` (a ``ValueError`` subclass) isn't rewrapped.
    try:
        transferred = transfer_manifold_subspaces(
            src, alignment, whitener=whitener,
            from_model=from_model, to_model=to_model,
        )
    except SaklasError:
        raise
    except ValueError as exc:
        raise ManifoldFormatError(f"transfer_manifold: {exc}") from exc

    # Carry the discover-mode per-model layout (``node_coords``) and the
    # source sidecar's provenance fields into the transferred tensor, then
    # stamp the transfer method + source id.  ``save_manifold`` reads
    # provenance keys off this metadata dict.
    metadata: dict[str, object] = dict(src.metadata)
    metadata.update(source_sidecar)
    metadata["method"] = "manifold_procrustes_transfer"
    metadata["source_model_id"] = from_model
    metadata["source_model_fingerprint"] = source_model_fingerprint
    metadata["model_fingerprint"] = target_model_fingerprint
    # Record the *target* share metric (the source sidecar's value rode in
    # via the ``metadata.update`` above and would be misleading).
    # ``subspace_metric`` is left as the source carried it — the basis was
    # selected on the source model and only rotated here, so its selection
    # metric is unchanged by the transfer.
    metadata["share_metric"] = "mahalanobis"
    if transfer_quality_estimate is not None:
        metadata["transfer_quality_estimate"] = float(transfer_quality_estimate)
    if not target_trusted:
        out_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        files = dict(files)
        files.pop(out_path.name, None)
        files.pop(sidecar_path.name, None)
        manifest["files"] = files
        write_json_atomic(folder / "manifold.json", manifest)
    current = ManifoldFolder.load(folder, verify_manifest=False)
    metadata["nodes_sha256"] = current.nodes_sha256()
    save_manifold(transferred, out_path, metadata)
    current.update_file_hashes(out_path, sidecar_path)
    return out_path


# ============================================================ shared summary serializer ===


def manifold_summary(folder: Path) -> dict[str, Any]:
    """Session-independent summary of a manifold folder.

    The shared serializer behind ``manifold show -j`` (CLI) and
    ``GET /saklas/v1/manifolds/{ns}/{name}`` (server) — the keys both can
    render without a loaded session.  Pure-io: reads the folder off disk
    and reports its identity, geometry, node layout, hyperparameters, and
    which models have a fitted tensor present.

    For an *authored* manifold the ``domain`` / ``intrinsic_dim`` /
    ``node_coords`` come straight off ``manifold.json``.  For a *discover*
    manifold the on-disk folder carries none of these — coords are derived
    per-model at fit time and live in the per-model safetensors — so the
    top-level ``domain`` is ``{}``, ``intrinsic_dim`` is ``0``, and
    ``node_coords`` is ``[]`` here; a session-aware caller can lift the
    derived layout from the fitted sidecar/tensor if it wants the per-model
    geometry (that read needs the safetensors, kept out of this pure-io
    summary).

    Returns a dict with keys: ``namespace`` / ``name`` / ``description`` /
    ``source`` / ``tags`` / ``fit_mode`` / ``is_discover`` / ``domain`` /
    ``domain_label`` / ``intrinsic_dim`` / ``min_nodes`` / ``node_count`` /
    ``node_labels`` / ``node_coords`` / ``node_roles`` / ``hyperparams`` /
    ``fitted_models`` / ``tensor_variants``.

    ``namespace`` is read off the folder's parent directory name.  Raises
    :class:`ManifoldFormatError` on a malformed folder.
    """
    folder = Path(folder)
    # A summary reports authoring metadata and fitted filenames only.  Payload
    # integrity remains mandatory when a tensor is installed, pushed, or used;
    # hashing every historical fit here makes ordinary show/detail routing
    # scale with artifact bytes rather than metadata size.
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    namespace = folder.parent.name

    if mf.fit_mode == "authored" and mf.domain:
        domain = mf.domain
        domain_lbl = domain_label(domain)
        intrinsic = domain_from_spec(domain).intrinsic_dim
    else:
        domain = {}
        domain_lbl = f"discover-{mf.fit_mode}"
        intrinsic = 0

    # Each fitted tensor stem → its variant key, so a caller can tell a
    # canonical fit apart from an SAE or transferred variant.
    from saklas.io.paths import parse_tensor_filename

    fitted_models: list[str] = []
    tensor_variants: dict[str, list[str]] = {}
    for ts in sorted(folder.glob("*.safetensors")):
        parsed = parse_tensor_filename(ts.name)
        if parsed is None:
            continue
        safe_model, variant = parsed
        key = "raw" if variant is None else variant
        tensor_variants.setdefault(safe_model, []).append(key)
        if safe_model not in fitted_models:
            fitted_models.append(safe_model)

    return {
        "namespace": namespace,
        "name": mf.name,
        "description": mf.description,
        "source": mf.source,
        "tags": list(mf.tags),
        "fit_mode": mf.fit_mode,
        "is_discover": mf.is_discover,
        "domain": domain,
        "domain_label": domain_lbl,
        "intrinsic_dim": intrinsic,
        "min_nodes": min_nodes(intrinsic) if intrinsic else None,
        "node_count": len(mf.node_labels),
        "node_labels": list(mf.node_labels),
        "node_coords": [list(c) for c in mf.node_coords],
        "node_roles": list(mf._roles_padded()),
        "node_kinds": list(mf._kinds_padded()),
        "hyperparams": dict(mf.hyperparams),
        "fitted_models": fitted_models,
        "tensor_variants": tensor_variants,
    }
