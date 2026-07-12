"""ManifoldExtractionPipeline — fit a steering manifold from an authored corpus.

The single extraction pipeline (4.0): a steering vector is the 2-node ``pca``
case of a manifold, so concept extraction (``session.extract``) and manifold
fitting share this one pipeline.  It pools per-node centroids, fits the
per-layer PCA (+ RBF for curved manifolds), bakes the per-layer share, and
writes the per-model tensor.

Dependencies are passed structurally (not as a back-reference to the session)
via the runtime-checkable :class:`ModelHandle` protocol plus an
:class:`EventBus` for ``ManifoldExtracted`` emission.  ``SaklasSession``
satisfies the protocol implicitly, so construction reads as
``ManifoldExtractionPipeline(self, self.events)``.

The session gates re-entry against ``GenState.IDLE`` before forwarding; the
pipeline itself does not touch generation state.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any, Callable, Protocol, runtime_checkable

import torch
from safetensors.torch import load_file, save_file

from saklas.core.events import EventBus, ManifoldExtracted
from saklas.core.manifold import MANIFOLD_FIT_POLICY_VERSION
from saklas.core.model import loaded_model_fingerprint, workspace_layer_indices
from saklas.core.sae import SaeBackend
from saklas.io.paths import model_dir, tensor_filename


_CAPTURE_CACHE_FORMAT_VERSION = 4
_ROW_DIGEST_CACHE_MAX = 512
_row_digest_cache: set[tuple[object, ...]] = set()
_row_digest_cache_lock = threading.Lock()


def _apply_fit_overrides_locked(
    folder: pathlib.Path,
    *,
    fit_mode: str | None,
    hyperparams: Mapping[str, object] | None,
) -> None:
    """Apply discover-fit overrides inside the folder transaction.

    Callers pass only the values supplied by this request.  The merge happens
    against the manifest reloaded under ``_locked_manifest``, so two fit
    requests cannot publish a stale read/modify/write snapshot before either
    fit computes its cache identity.
    """
    if fit_mode is None and hyperparams is None:
        return

    from saklas.io.atomic import write_json_atomic
    from saklas.io.manifolds import ManifoldFolder, sanitize_hyperparams

    mf = ManifoldFolder.load(folder, verify_manifest=False)
    if not mf.is_discover:
        raise ValueError(
            "fit_mode/hyperparams overrides are discover-mode only; "
            f"{folder} is authored"
        )
    requested_mode = fit_mode or mf.fit_mode
    if requested_mode not in {"pca", "spectral", "auto"}:
        raise ValueError(f"invalid discover fit mode {requested_mode!r}")
    requested_hyperparams = dict(mf.hyperparams)
    if hyperparams is not None:
        requested_hyperparams.update(hyperparams)
    requested_hyperparams = sanitize_hyperparams(
        requested_mode, requested_hyperparams,
    )
    if (
        requested_mode == mf.fit_mode
        and requested_hyperparams == mf.hyperparams
    ):
        return

    with open(folder / "manifold.json") as handle:
        data = json.load(handle)
    data["fit_mode"] = requested_mode
    data["hyperparams"] = requested_hyperparams
    # Discover nodes are label-only.  Preserve role/kind provenance while
    # stripping any authored geometry left by a mode transition.
    data["nodes"] = [
        {
            "label": label,
            **(
                {"role": mf.node_roles[idx]}
                if idx < len(mf.node_roles) and mf.node_roles[idx] is not None
                else {}
            ),
            **(
                {"kind": mf.node_kinds[idx]}
                if idx < len(mf.node_kinds) and mf.node_kinds[idx] is not None
                else {}
            ),
        }
        for idx, label in enumerate(mf.node_labels)
    ]
    data.pop("domain", None)
    write_json_atomic(folder / "manifold.json", data)


def _repair_incomplete_target_locked(
    folder: pathlib.Path, tensor_path: pathlib.Path,
) -> None:
    """Remove an interrupted one-sided fitted pair and its stale proofs."""

    sidecar_path = tensor_path.with_suffix(".json")
    if tensor_path.exists() == sidecar_path.exists():
        return
    tensor_path.unlink(missing_ok=True)
    sidecar_path.unlink(missing_ok=True)
    manifest_path = folder / "manifold.json"
    with open(manifest_path) as handle:
        payload = json.load(handle)
    files = payload.get("files", {})
    if isinstance(files, dict):
        files = dict(files)
        files.pop(tensor_path.name, None)
        files.pop(sidecar_path.name, None)
        payload["files"] = files
        from saklas.io.atomic import write_json_atomic

        write_json_atomic(manifest_path, payload)


class ManifoldAuthoringChangedError(RuntimeError):
    """The authoring inputs changed while a target was being fitted."""


def _assert_authoring_revision(
    folder: pathlib.Path, target_path: pathlib.Path, expected_revision: str,
) -> None:
    from saklas.io.manifold_folder import _locked_manifest

    with _locked_manifest(folder):
        _mf, _nodes_sha, actual, _groups, _baseline = (
            _authoring_snapshot_locked(folder, target_path)
        )
    if actual != expected_revision:
        raise ManifoldAuthoringChangedError(
            f"manifold authoring changed during fit ({expected_revision[:12]}"
            f" -> {actual[:12]}); retry against the current revision"
        )


def _publish_fit_if_current(
    folder: pathlib.Path,
    *,
    expected_revision: str,
    manifold: Any,
    tensor_path: pathlib.Path,
    metadata: dict[str, Any],
) -> None:
    """CAS-publish a fitted pair while the authoring revision is unchanged."""

    from saklas.core.manifold import save_manifold
    from saklas.io.manifold_folder import _locked_manifest

    with _locked_manifest(folder):
        current, _nodes_sha, actual, _groups, _baseline = (
            _authoring_snapshot_locked(folder, tensor_path)
        )
        if actual != expected_revision:
            raise ManifoldAuthoringChangedError(
                f"manifold authoring changed during fit ({expected_revision[:12]}"
                f" -> {actual[:12]}); retry against the current revision"
            )
        save_manifold(manifold, tensor_path, metadata)
        current.update_file_hashes(
            tensor_path, tensor_path.with_suffix(".json"),
        )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Exact original-bit digest without materializing the whole payload."""
    digest = hashlib.sha256()
    flat = tensor.detach().reshape(-1)
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(dim) for dim in tensor.shape)).encode("utf-8"))
    for start in range(0, flat.numel(), 1_048_576):
        chunk = flat[start:start + 1_048_576].to(device="cpu").contiguous()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _row_tensor_matches_cache_metadata(
    path: pathlib.Path,
    layer: int,
    tensor: torch.Tensor,
    *,
    expected_shape: object,
    expected_dtype: object,
    expected_digest: object,
) -> bool:
    """Validate one row-cache tensor, memoizing an unchanged exact digest.

    Row containers can be multi-GiB and geometry-only refits intentionally read
    the same layer mappings repeatedly.  The first access still hashes every
    byte; later accesses reuse that proof only while the container's full stat
    identity and the expected per-layer digest remain unchanged.
    """
    if (
        expected_shape != list(tensor.shape)
        or expected_dtype != str(tensor.dtype)
        or not isinstance(expected_digest, str)
        or len(expected_digest) != 64
    ):
        return False
    try:
        before = path.stat()
    except OSError:
        return False
    stat_key = (
        str(path.resolve()), before.st_size, before.st_mtime_ns,
        before.st_ctime_ns, before.st_dev, before.st_ino,
        int(layer), expected_digest,
    )
    with _row_digest_cache_lock:
        if stat_key in _row_digest_cache:
            return True
    if _tensor_sha256(tensor) != expected_digest:
        return False
    try:
        after = path.stat()
    except OSError:
        return False
    after_key = (
        str(path.resolve()), after.st_size, after.st_mtime_ns,
        after.st_ctime_ns, after.st_dev, after.st_ino,
        int(layer), expected_digest,
    )
    if after_key != stat_key:
        return False
    with _row_digest_cache_lock:
        if len(_row_digest_cache) >= _ROW_DIGEST_CACHE_MAX:
            _row_digest_cache.clear()
        _row_digest_cache.add(stat_key)
    return True


def _save_safetensors_atomic(
    tensors: dict[str, torch.Tensor], path: pathlib.Path,
) -> None:
    """Publish safetensors through a unique same-directory staging file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    os.close(fd)
    tmp = pathlib.Path(tmp_name)
    try:
        save_file(tensors, str(tmp))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _centroid_shard_path(
    folder: pathlib.Path, stem: str, layer: int,
) -> pathlib.Path:
    return folder / f"{stem}.centroids.layer_{int(layer)}.safetensors"


def _row_shard_path(
    folder: pathlib.Path, stem: str, layer: int,
) -> pathlib.Path:
    return folder / f"{stem}.rows.layer_{int(layer)}.safetensors"


def _capture_group_paths(
    folder: pathlib.Path, stem: str,
) -> list[pathlib.Path]:
    """Current files for one content-addressed capture group."""
    return [path for path in folder.glob(f"{stem}.*") if path.is_file()]


def _publish_deferred_row_shards(
    capture_dir: pathlib.Path,
    cache_stem: str,
    cache_meta: pathlib.Path,
    retained_rows: Any,
    node_sizes: list[int],
    row_layers: set[int],
    row_files: dict[str, str],
    row_shapes: dict[str, object],
    row_dtypes: dict[str, object],
    row_digests: dict[str, object],
) -> None:
    """Top up auto-curved row shards in a short stem transaction."""

    from saklas.io.atomic import artifact_lock, write_json_atomic

    with artifact_lock(capture_dir / cache_stem):
        with open(cache_meta) as handle:
            payload = json.load(handle)
        if (
            int(payload.get("format_version", 0)) != _CAPTURE_CACHE_FORMAT_VERSION
            or payload.get("capture_sha256") != cache_stem
            or [int(x) for x in payload.get("node_sizes", [])] != node_sizes
        ):
            raise ValueError("capture cache changed identity during deferred row top-up")
        current_layers = {int(i) for i in payload.get("row_layers", [])}
        current_files = dict(payload.get("row_files", {}))
        current_shapes = dict(payload.get("row_shapes", {}))
        current_dtypes = dict(payload.get("row_dtypes", {}))
        current_digests = dict(payload.get("row_tensor_sha256", {}))
        for idx in current_layers:
            key = str(idx)
            if not all(
                key in mapping for mapping in (
                    current_files, current_shapes, current_dtypes, current_digests,
                )
            ):
                raise ValueError("capture row pointer changed incompletely")
            row_files[key] = current_files[key]
            row_shapes[key] = current_shapes[key]
            row_dtypes[key] = current_dtypes[key]
            row_digests[key] = current_digests[key]
        row_layers.update(current_layers)
        newly_durable = sorted(set(retained_rows.layer_indices) - current_layers)
        for idx in newly_durable:
            row_path = _row_shard_path(capture_dir, cache_stem, idx)
            rows = retained_rows.flat_rows(idx)
            _save_safetensors_atomic({f"layer_{idx}": rows}, row_path)
            row_files[str(idx)] = row_path.name
            row_shapes[str(idx)] = list(rows.shape)
            row_dtypes[str(idx)] = str(rows.dtype)
            row_digests[str(idx)] = _tensor_sha256(rows)
        row_layers.update(newly_durable)
        durable = sorted(row_layers)
        payload["row_layers"] = durable
        payload["row_files"] = {str(i): row_files[str(i)] for i in durable}
        payload["row_shapes"] = {str(i): row_shapes[str(i)] for i in durable}
        payload["row_dtypes"] = {str(i): row_dtypes[str(i)] for i in durable}
        payload["row_tensor_sha256"] = {
            str(i): row_digests[str(i)] for i in durable
        }
        write_json_atomic(cache_meta, payload)


def _prune_manifold_capture_cache(
    folder: pathlib.Path, *, keep_stem: str,
) -> None:
    """Bound capture-cache disk use without invalidating an active fit.

    Callers hold no capture-stem lock here.  A directory-wide prune lock
    serializes eviction decisions, then each victim's own logical stem lock is
    acquired and its files are re-read before deletion.  No code holds one stem
    while acquiring another, avoiding the A->B / B->A deadlock that a naive
    lock-aware pruner would introduce.
    """
    try:
        limit_gb = float(os.environ.get("SAKLAS_MANIFOLD_CAPTURE_CACHE_GB", "8"))
    except ValueError:
        limit_gb = 8.0
    if limit_gb <= 0.0:
        return
    limit = int(limit_gb * 1024**3)
    from saklas.io.atomic import artifact_has_live_lease, artifact_lock

    folder.mkdir(parents=True, exist_ok=True)
    with artifact_lock(folder / "capture-cache-prune"):
        groups: dict[str, list[pathlib.Path]] = {}
        for path in folder.iterdir():
            if not path.is_file():
                continue
            stem = path.name.split(".", 1)[0]
            if len(stem) == 64:
                groups.setdefault(stem, []).append(path)
        sizes: dict[str, int] = {}
        mtimes: dict[str, int] = {}
        for stem, paths in groups.items():
            try:
                stats = [path.stat() for path in paths]
            except FileNotFoundError:
                continue
            sizes[stem] = sum(stat.st_size for stat in stats)
            mtimes[stem] = min(stat.st_mtime_ns for stat in stats)
        total = sum(sizes.values())
        if total <= limit:
            return
        oldest = sorted(
            (stem for stem in sizes if stem != keep_stem),
            key=mtimes.__getitem__,
        )
        for stem in oldest:
            with artifact_lock(folder / stem):
                if artifact_has_live_lease(folder / stem):
                    continue
                # The group may have been topped up or removed while we waited.
                # Re-stat under its transaction lock and charge exactly what is
                # about to be evicted.
                paths = _capture_group_paths(folder, stem)
                for path in paths:
                    path.unlink(missing_ok=True)
            # Other stem transactions do not take the directory prune lock and
            # may have grown while we waited.  Recompute the advisory total
            # instead of subtracting a stale pre-lock size.
            total = 0
            for path in folder.iterdir():
                if not path.is_file() or len(path.name.split(".", 1)[0]) != 64:
                    continue
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass
            if total <= limit:
                break


# ----------------------------------------------------------------------
# Structural protocols.  Runtime-checkable so callers (and tests) can
# ``isinstance(session, ModelHandle)`` to sanity-check the implicit
# implementation.  ``SaklasSession`` satisfies each by virtue of carrying
# the listed attributes / methods.
# ----------------------------------------------------------------------

@runtime_checkable
class ModelHandle(Protocol):
    """Read-only surface the pipeline needs from the live HF session.

    Held as a *handle*, not a copy — the pipeline must see the same
    model object the session uses; otherwise device-side state diverges.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def model(self) -> torch.nn.Module: ...

    @property
    def tokenizer(self) -> Any: ...  # PreTrainedTokenizerBase

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def layers(self) -> Any: ...  # ``get_layers`` returns ``nn.ModuleList`` — list-like

    def _run_generator(
        self, system_msg: str, prompt: str, max_new_tokens: int,
    ) -> str:
        """Single-turn LLM call backing conversational corpus generation.

        Underscore-prefixed because the override site is per-session
        (subclass-and-override is the established test pattern).  The
        protocol shape mirrors the existing ``SaklasSession._run_generator``
        signature exactly so the session satisfies it implicitly.
        """
        ...

    def generate_responses(
        self,
        concepts: list[str],
        kinds: list[str | None],
        *,
        roles: dict[str, str | None] | None = None,
        custom_system: str | None = None,
        samples_per_prompt: int = ...,
        max_new_tokens: int = ...,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, list[str]]: ...


def prepare_manifold_capture_identity(
    handle: ModelHandle,
    mf: Any,
    model_fingerprint: str,
    *,
    node_groups_snapshot: list[tuple[str, list[str]]] | None = None,
    baseline_prompts_snapshot: list[str | list[dict[str, str]]] | None = None,
) -> tuple[
    list[tuple[str, list[str]]],
    list[str | list[dict[str, str]]],
    list[tuple[torch.Tensor, int]] | None,
    str | None,
    list[str | None],
    list[str | None],
    str | None,
]:
    """Render/tokenize all fit rows and return their exact capture identity."""
    model = handle.model
    base_model = getattr(model, "_orig_mod", model)
    config = getattr(base_model, "config", getattr(model, "config", None))
    node_groups = (
        [(label, list(rows)) for label, rows in node_groups_snapshot]
        if node_groups_snapshot is not None else mf.node_groups()
    )
    node_roles = mf._roles_padded()
    node_kinds = mf._kinds_padded()
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if any(role is not None for role in node_roles) and model_type is None:
        raise ValueError(
            f"manifold {mf.name!r} carries per-node roles but model config "
            "has no 'model_type' — cannot resolve role headers"
        )
    if baseline_prompts_snapshot is not None:
        baseline_prompts = list(baseline_prompts_snapshot)
    elif mf.template_ref is not None:
        from saklas.io.templates import resolve_template

        tmpl = resolve_template(mf.template_ref)
        baseline_prompts: list[str | list[dict[str, str]]] = [
            context.messages() for context in tmpl.contexts
        ]
    else:
        from saklas.core.vectors import _load_baseline_prompts

        baseline_prompts = list(_load_baseline_prompts())
    tokenizer = handle.tokenizer
    prepared_rows: list[tuple[torch.Tensor, int]] | None = None
    capture_sha: str | None = None
    if callable(tokenizer):
        from saklas.core.vectors import _prepare_capture_batch

        flat_prompts: list[str | list[dict[str, str]]] = []
        flat_responses: list[str] = []
        flat_roles: list[str | None] = []
        k_prompts = len(baseline_prompts)
        for (_label, responses), role in zip(
            node_groups, node_roles, strict=True,
        ):
            for row_idx, response in enumerate(responses):
                flat_prompts.append(baseline_prompts[row_idx % k_prompts])
                flat_responses.append(response)
                flat_roles.append(role)
        prepared_rows = _prepare_capture_batch(
            tokenizer, flat_prompts, flat_responses, handle.device,
            roles=flat_roles, model_type=model_type,
        )
        capture_model_fingerprint = {
            "model_id": handle.model_id,
            "model_class": (
                f"{type(base_model).__module__}.{type(base_model).__qualname__}"
            ),
            "commit": getattr(config, "_commit_hash", None),
            "name_or_path": getattr(config, "_name_or_path", None),
            "model_type": getattr(config, "model_type", None),
            "capture_version": 3,
            "fingerprint": model_fingerprint,
        }
        capture_hash = hashlib.sha256(json.dumps(
            capture_model_fingerprint, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"))
        capture_hash.update(json.dumps(
            {"node_sizes": [len(rows) for _label, rows in node_groups]},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"))
        for ids, content_end in prepared_rows:
            capture_hash.update(
                repr((ids[0].tolist(), int(content_end))).encode("utf-8")
            )
        capture_sha = capture_hash.hexdigest()
    return (
        node_groups, baseline_prompts, prepared_rows, capture_sha,
        node_roles, node_kinds, model_type,
    )


def _authoring_snapshot_locked(
    folder: pathlib.Path,
    target_path: pathlib.Path,
) -> tuple[
    Any,
    str,
    str,
    list[tuple[str, list[str]]],
    list[str | list[dict[str, str]]],
]:
    """Materialize template inputs and snapshot one exact authoring revision."""

    from saklas.io.manifolds import ManifoldFolder

    manifest_path = folder / "manifold.json"
    with open(manifest_path) as handle:
        identity_payload = json.load(handle)
    artifact_id = identity_payload.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        artifact_id = uuid.uuid4().hex
        identity_payload["artifact_id"] = artifact_id
        identity_payload.setdefault("fit_epochs", {})
        from saklas.io.atomic import write_json_atomic

        write_json_atomic(manifest_path, identity_payload)
    from saklas.io.paths import parse_tensor_filename

    parsed = parse_tensor_filename(target_path.name)
    if parsed is None:
        raise ValueError(f"invalid manifold tensor target {target_path.name!r}")
    safe_model, raw_variant = parsed
    variant = "raw" if raw_variant is None else raw_variant.split("-", 1)[0]
    fit_epochs_raw = identity_payload.get("fit_epochs", {})
    fit_epochs = fit_epochs_raw if isinstance(fit_epochs_raw, dict) else {}
    epoch_keys = (
        "*:all", f"*:{variant}", f"{safe_model}:all",
        f"{safe_model}:{variant}", target_path.name,
    )
    target_epochs = {key: int(fit_epochs.get(key, 0)) for key in epoch_keys}
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    template_sha: str | None = None
    if mf.template_ref is not None:
        from saklas.io.manifold_authoring import _write_node_corpus
        from saklas.io.templates import resolve_template

        tmpl = resolve_template(mf.template_ref)
        template_sha = tmpl.sha256()
        expected_corpora = tmpl.node_corpora()
        if dict(mf.node_groups()) != expected_corpora:
            _write_node_corpus(
                folder,
                [
                    {"label": label, "statements": statements}
                    for label, statements in expected_corpora.items()
                ],
            )
            mf.node_labels = list(expected_corpora)
            mf.node_roles = [None] * len(mf.node_labels)
            mf.node_kinds = [None] * len(mf.node_labels)
            mf.write_metadata()
            mf = ManifoldFolder.load(folder, verify_manifest=False)
        baseline: list[str | list[dict[str, str]]] = [
            context.messages() for context in tmpl.contexts
        ]
    else:
        from saklas.core.vectors import _load_baseline_prompts

        baseline = list(_load_baseline_prompts())
    groups = mf.node_groups()
    nodes_sha = mf.nodes_sha256(resolved_template_sha256=template_sha)
    revision_payload = json.dumps(
        {
            "artifact_id": artifact_id,
            "target_epochs": target_epochs,
            "nodes_sha256": nodes_sha,
            "baseline_prompts": baseline,
        },
        sort_keys=True, separators=(",", ":"),
    ).encode()
    revision = hashlib.sha256(revision_payload).hexdigest()
    return mf, nodes_sha, revision, groups, baseline


def _diagnostics_to_dict(diag: Any) -> dict[str, Any]:
    """Convert a discover-mode diagnostics dataclass into a JSON-safe dict.

    Both :class:`PcaDiagnostics` and :class:`SpectralDiagnostics` carry
    one or more tensor fields; the sidecar is JSON, so tensors are
    converted to plain Python lists and floats here.  The dispatcher is
    structural — duck-typed on the dataclass fields — so adding a third
    method later doesn't require touching this helper as long as the
    dataclass stays JSON-serializable in this way.
    """
    out: dict[str, Any] = {}
    for name in (
        "per_component_variance", "cumulative_variance", "eigenvalues",
    ):
        if hasattr(diag, name):
            t = getattr(diag, name)
            if isinstance(t, torch.Tensor):
                out[name] = [float(x) for x in t.tolist()]
            else:
                out[name] = list(t)
    for name in (
        "picked_k", "gap_index", "k_nn", "component_count", "heuristic_k",
    ):
        if hasattr(diag, name):
            out[name] = int(getattr(diag, name))
    for name in ("threshold", "gap_magnitude", "bandwidth"):
        if hasattr(diag, name):
            out[name] = float(getattr(diag, name))
    # Spectral dimensionality-floor provenance (optional / bool).
    if hasattr(diag, "pinned"):
        out["pinned"] = bool(diag.pinned)
    if getattr(diag, "min_dim", None) is not None:
        out["min_dim"] = int(diag.min_dim)
    return out


class ManifoldExtractionPipeline:
    """Fit an RBF-based steering manifold from an authored corpus.

    THE 4.0 extraction pipeline: concept extraction and manifold fitting are
    the same operation — a steering vector is just the 2-node ``pca`` case (N
    labeled node groups, no contrastive pairs, no scenario generation).  It
    reuses the :class:`ModelHandle` protocol (it needs ``model`` /
    ``tokenizer`` / ``layers`` / ``device`` / ``model_id``) and emits
    :class:`ManifoldExtracted`.

    Feature space: ``sae=None`` fits per-layer PCA + RBF directly on
    residual-stream centroids.  ``sae="<release>"`` reconstructs each
    centroid through the SAE (encode then decode) before the fit — a
    denoised, sparse-feature-supported centroid — and restricts the
    manifold to the SAE's covered layers.  Either way the fitted
    :class:`~saklas.core.manifold.LayerSubspace` is model-space, so the
    steering hook never touches the SAE.
    """

    __slots__ = ("_events", "_handle")

    def __init__(self, model_handle: ModelHandle, events: EventBus) -> None:
        self._handle = model_handle
        self._events = events

    def fit(
        self,
        folder: str | pathlib.Path,
        *,
        sae: str | SaeBackend | None = None,
        sae_revision: str | None = None,
        layer_indices: "Sequence[int] | str | None" = None,
        fit_mode: str | None = None,
        hyperparams: Mapping[str, object] | None = None,
        force: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ):
        """Fit against a stable authoring snapshot, publishing by revision CAS.

        The long-running lock is target-scoped, so two model/variant targets in
        one folder may compute concurrently and readers keep using the prior
        published artifact.  The folder lock is held only while taking the exact
        authoring snapshot and again during final compare-and-swap publication.
        """
        from saklas.io.atomic import artifact_lock
        from saklas.io.manifold_folder import _locked_manifest, manifold_pair_lock

        folder_path = pathlib.Path(folder)
        sae_release = (
            sae if isinstance(sae, str)
            else (sae.release if sae is not None else None)
        )
        target_path = folder_path / tensor_filename(
            self._handle.model_id, release=sae_release,
        )
        # Lock outside the removable folder.  ``rm`` + recreate must not mint a
        # fresh lock inode while a stale fit still owns the old one.
        fit_lock_key = hashlib.sha256(
            f"{folder_path.resolve(strict=False)}\0{target_path.name}".encode()
        ).hexdigest()
        fit_lock = folder_path.parent / f".saklas-fit-{fit_lock_key}"
        with artifact_lock(fit_lock):
            with _locked_manifest(folder_path):
                with manifold_pair_lock(target_path):
                    _repair_incomplete_target_locked(folder_path, target_path)
                _apply_fit_overrides_locked(
                    folder_path, fit_mode=fit_mode, hyperparams=hyperparams,
                )
                mf, nodes_sha, revision, groups, baseline = _authoring_snapshot_locked(
                    folder_path, target_path,
                )
            return self._fit_locked(
                folder_path, sae=sae, sae_revision=sae_revision,
                layer_indices=layer_indices, force=force,
                on_progress=on_progress,
                _snapshot_mf=mf,
                _snapshot_nodes_sha=nodes_sha,
                _authoring_revision=revision,
                _node_groups_snapshot=groups,
                _baseline_prompts_snapshot=baseline,
            )

    def _fit_locked(
        self,
        folder: str | pathlib.Path,
        *,
        sae: str | SaeBackend | None = None,
        sae_revision: str | None = None,
        layer_indices: "Sequence[int] | str | None" = None,
        force: bool = False,
        on_progress: Callable[[str], None] | None = None,
        _capture_lock_held: bool = False,
        _prepared_identity: tuple[Any, ...] | None = None,
        _model_fingerprint: str | None = None,
        _verified_cache_miss: bool = False,
        _snapshot_mf: Any | None = None,
        _snapshot_nodes_sha: str | None = None,
        _authoring_revision: str | None = None,
        _node_groups_snapshot: list[tuple[str, list[str]]] | None = None,
        _baseline_prompts_snapshot: list[str | list[dict[str, str]]] | None = None,
        _release_capture_lock: Callable[[], None] | None = None,
    ):
        """Fit (or load from cache) a manifold for the session's model.

        ``folder`` is a manifold pack directory — either authored (the
        user supplied a domain spec + per-node coordinates) or
        discover-mode (the user supplied only labeled node corpora; the
        coords are derived per-model via
        :func:`saklas.core.manifold.discover_coords`).  Returns a
        :class:`~saklas.core.manifold.Manifold`.

        A cache hit — the per-model tensor exists and its sidecar
        ``nodes_sha256`` still matches the folder's current state —
        short-circuits the forward passes.  For discover-mode folders
        the staleness key folds in ``fit_mode`` + ``hyperparams``, so
        a refit with different hyperparameters reliably misses cache.
        ``force=True`` bypasses the cache and re-pools/re-fits unconditionally
        — the discover/multi-node analogue of ``manifold extract -f`` (which
        forces a refit by re-authoring the corpus); without it a code change to
        the fit itself, e.g. a topology-selection fix, can't be picked up while
        the corpus is unchanged.
        """
        from saklas.core.manifold import (
            DEFAULT_N_COMPONENTS,
            CustomDomain,
            Manifold,
            compute_manifold_node_stats,
            compute_store_reduced_covariances,
            discover_coords,
            domain_from_spec,
            fit_affine_subspace,
            fit_layer_subspace,
            fit_sigma_field,
            invert_parameterization,
            neutral_layout_coord,
            prepare_rbf_fit_plan,
            subspace_share,
        )
        from saklas.core.vectors import compute_dls_axes
        from saklas.io.manifolds import (
            ManifoldFolder, ManifoldSidecar, min_nodes,
        )
        from saklas.core.errors import SaeCoverageError
        from saklas.core.mahalanobis import WhitenerError

        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        if sae_revision is not None and sae is None:
            raise ValueError("sae_revision requires an SAE release")
        fit_started = time.perf_counter()
        mf = (
            _snapshot_mf if _snapshot_mf is not None
            else ManifoldFolder.load(pathlib.Path(folder), verify_manifest=False)
        )
        nodes_sha = _snapshot_nodes_sha or mf.nodes_sha256()
        model = self._handle.model
        layers = self._handle.layers
        n_layers = len(layers)
        if layer_indices is None or layer_indices == "all":
            requested_fit_layers = list(range(n_layers))
        elif layer_indices == "workspace":
            requested_fit_layers = workspace_layer_indices(
                range(n_layers), n_layers,
            )
        elif isinstance(layer_indices, str):
            try:
                requested_fit_layers = sorted({
                    int(part.strip()) for part in layer_indices.split(",")
                    if part.strip()
                })
            except ValueError as exc:
                raise ValueError(
                    "layer_indices must be 'all', 'workspace', or a "
                    "comma-separated integer list"
                ) from exc
        else:
            requested_fit_layers = sorted({int(idx) for idx in layer_indices})
        if not requested_fit_layers:
            raise ValueError("layer_indices must name at least one layer")
        if any(idx < 0 or idx >= n_layers for idx in requested_fit_layers):
            raise ValueError(
                f"layer_indices must lie in [0, {n_layers}); got "
                f"{requested_fit_layers}"
            )
        model_fingerprint = (
            _model_fingerprint
            if _model_fingerprint is not None
            else loaded_model_fingerprint(model, self._handle.model_id)
        )

        # Resolve string requests through SAELens metadata before accepting a
        # hit: transform identity is part of the cache key. Registry/import
        # work is therefore allowed here, but the backend remains one-layer
        # lazy and no SAE weights are loaded on a valid fitted-tensor hit.
        sae_backend: SaeBackend | None
        sae_release: str | None
        if sae is None:
            sae_backend = None
            sae_release = None
        elif isinstance(sae, str):
            sae_backend = None
            sae_release = sae
        else:
            sae_backend = sae
            sae_release = sae.release

        if isinstance(sae, str):
            from saklas.core.sae import load_sae_backend

            sae_backend = load_sae_backend(
                sae,
                revision=sae_revision,
                model_id=self._handle.model_id,
                device=self._handle.device,
            )
        sae_fingerprint = (
            getattr(sae_backend, "fingerprint", None)
            if sae_backend is not None else None
        )

        tensor_path = pathlib.Path(folder) / tensor_filename(
            self._handle.model_id, release=sae_release,
        )

        # Compute the token-exact capture identity before accepting the final
        # tensor cache.  Source-text hashes alone miss baseline-prompt edits and
        # tokenizer/chat-template changes; both alter the actual residuals even
        # when the authored node files are byte-identical.
        prepared_identity = (
            _prepared_identity
            if _prepared_identity is not None
            else prepare_manifold_capture_identity(
                self._handle, mf, model_fingerprint,
                node_groups_snapshot=_node_groups_snapshot,
                baseline_prompts_snapshot=_baseline_prompts_snapshot,
            )
        )
        (
            node_groups, baseline_prompts, prepared_rows, capture_sha,
            node_roles, node_kinds, model_type,
        ) = prepared_identity
        tokenizer = self._handle.tokenizer
        device = self._handle.device
        K = len(node_groups)
        any_role = any(r is not None for r in node_roles)
        any_kind = any(k is not None for k in node_kinds)
        if any_role and sae is not None:
            raise ValueError(
                "SAE-backed and role-baselined manifold fits are mutually exclusive"
            )

        # Cache hit: tensor present + every fit-affecting input unchanged.
        # ``nodes_sha256`` folds in the corpus, plus either the domain
        # spec + node coords (authored) or the fit_mode + hyperparams
        # (discover); ``sae_revision`` is the SAE the centroids are
        # reconstructed through and does not ride the filename, so it is
        # checked here or a stale tensor is served.
        sidecar_path = tensor_path.with_suffix(".json")
        cached_revision = (
            sae_revision if isinstance(sae, str)
            else sae_backend.revision if sae_backend is not None
            else None
        )
        cached_manifold: Manifold | None = None
        if not force and not _verified_cache_miss:
            from saklas.core.manifold import _load_manifold_locked
            from saklas.io.manifold_folder import manifold_pair_lock
            from saklas.io.packs import verify_integrity

            # Clear/rm/refresh take the folder lock and then this same stable
            # pair lock.  The fast path deliberately takes only the pair lock:
            # it proves the *current* manifest pair, reads the sidecar, and
            # loads the tensor in one transaction, then releases it before the
            # authoring-CAS check takes the folder lock.  This avoids both the
            # old proof->load deletion window and a folder/pair lock inversion.
            with manifold_pair_lock(tensor_path):
                try:
                    with open(pathlib.Path(folder) / "manifold.json") as handle:
                        current_manifest = json.load(handle)
                    current_files = current_manifest.get("files", {})
                    expected_target_files = {
                        name: current_files[name]
                        for name in (tensor_path.name, sidecar_path.name)
                        if isinstance(current_files, dict) and name in current_files
                    }
                    pair_ok = (
                        len(expected_target_files) == 2
                        and tensor_path.exists()
                        and sidecar_path.exists()
                        and verify_integrity(
                            pathlib.Path(folder), expected_target_files,
                        )[0]
                    )
                    sc = ManifoldSidecar.load(sidecar_path) if pair_ok else None
                except (OSError, json.JSONDecodeError, KeyError, ValueError):
                    sc = None
                if (
                    sc is not None
                    and sc.nodes_sha256 == nodes_sha
                    and sc.sae_release == sae_release
                    and (
                        sae_release is None
                        or (
                            sae_fingerprint is not None
                            and sc.sae_fingerprint == sae_fingerprint
                        )
                    )
                    and (
                        sc.sae_revision == cached_revision
                        if cached_revision is not None or sae_release is None
                        else True
                    )
                    and sc.model_fingerprint == model_fingerprint
                    and sc.capture_sha256 == capture_sha
                    and sc.fit_policy_version == MANIFOLD_FIT_POLICY_VERSION
                    and (
                        sc.fitted_layers == requested_fit_layers
                        if sae_release is None or layer_indices is not None
                        else sc.sae_full_coverage
                    )
                ):
                    cached_manifold = _load_manifold_locked(
                        tensor_path, verify_manifest=False,
                    )
        if cached_manifold is not None:
            _assert_authoring_revision(
                pathlib.Path(folder), tensor_path,
                _authoring_revision or nodes_sha,
            )
            _progress(f"Loaded cached manifold '{mf.name}'.")
            self._events.emit(ManifoldExtracted(
                name=mf.name, manifold=cached_manifold,
                metadata=dict(cached_manifold.metadata),
            ))
            return cached_manifold

        # SAE coverage — fail-fast.  Validate the fit-layer set here, before
        # the expensive per-node centroid pooling, so an SAE release that
        # covers none of the model's layers errors immediately instead of
        # after K node passes.
        if sae_backend is not None:
            covered = set(sae_backend.layers) & set(range(n_layers))
            if layer_indices is not None:
                missing = sorted(set(requested_fit_layers) - covered)
                if missing:
                    raise SaeCoverageError(
                        f"SAE release {sae_backend.release!r} does not cover "
                        f"requested layers {missing}"
                    )
                fit_layers = requested_fit_layers
            else:
                fit_layers = sorted(covered)
            if not fit_layers:
                raise SaeCoverageError(
                    f"SAE release {sae_backend.release!r} covers no layers "
                    f"of {self._handle.model_id}"
                )
            feature_space = f"sae-{sae_backend.release}"
        else:
            fit_layers = requested_fit_layers
            feature_space = "raw"

        # Capture payloads are shared per model and token identity, not per
        # manifold folder. Re-enter once with an explicitly releasable stem
        # transaction: cache read/top-up/publish is exclusive, then downstream
        # fitting runs under a process lease that prune/GC observes. Passing an
        # already-resolved SAE backend avoids a second registry resolution.
        if capture_sha is not None and not _capture_lock_held:
            from saklas.io.atomic import (
                ReleasableArtifactLock,
                artifact_process_lease,
            )

            capture_lock_stem = (
                model_dir(self._handle.model_id)
                / "manifold_capture"
                / capture_sha
            )
            transaction = ReleasableArtifactLock(capture_lock_stem).acquire()
            lease_context = artifact_process_lease(capture_lock_stem)
            lease_entered = False
            try:
                lease_context.__enter__()
                lease_entered = True
                manifold = self._fit_locked(
                    folder,
                    sae=(sae_backend if isinstance(sae, str) else sae),
                    sae_revision=sae_revision,
                    layer_indices=layer_indices,
                    force=force,
                    on_progress=on_progress,
                    _capture_lock_held=True,
                    _prepared_identity=prepared_identity,
                    _model_fingerprint=model_fingerprint,
                    _verified_cache_miss=True,
                    _snapshot_mf=mf,
                    _snapshot_nodes_sha=nodes_sha,
                    _authoring_revision=_authoring_revision,
                    _node_groups_snapshot=_node_groups_snapshot,
                    _baseline_prompts_snapshot=_baseline_prompts_snapshot,
                    _release_capture_lock=transaction.release,
                )
            finally:
                # The inner cache stage normally releases early. Reacquire before
                # removing the lease marker so creation/removal and prune's live-
                # lease decision are all ordered by the same stem transaction.
                transaction.acquire()
                try:
                    if lease_entered:
                        lease_context.__exit__(None, None, None)
                finally:
                    transaction.release()
            # Eviction must run after this stem transaction is fully complete.
            # The pruner takes victim locks one at a time; calling it while the
            # current stem were held would permit cross-stem lock inversion.
            try:
                _prune_manifold_capture_cache(
                    capture_lock_stem.parent, keep_stem=capture_sha,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                # Cache eviction is advisory.  The fitted tensor and its capture
                # group are already durably published; a prune failure must not
                # turn successful model work into a failed fit.
                _progress(f"Activation capture cache prune skipped: {exc}")
            return manifold

        # 1. Per-node centroids (one forward pass per response) — shared
        #    between authored and discover paths.  Conversational (4.0 / A2):
        #    each node corpus is a list of in-character responses to the shared
        #    baseline prompts, pooled as ``[user: prompt, assistant: response]``
        #    pairs (response[i] -> prompt[i % k]).  Per-node role rides through
        #    ``compute_node_centroid`` only when set (persona-baselined fit);
        #    a ``None`` role pools under the standard assistant (swap-back)
        #    baseline.
        # Curved raw fits need within-node reduced covariance
        # after the basis exists to fit the fuzzy sigma field.  Retain the
        # first-pass per-response rows in a temporary mmap-backed layer-major
        # spool.  Auto participates too: a resolved-flat fit deletes the spool,
        # while auto-curved avoids repeating every model forward.  The spool
        # keeps source dtype, so speculative auto retention costs bounded RSS
        # and half the disk traffic for fp16/bf16 models.
        retain_node_rows = (
            sae_backend is None and mf.fit_mode in {"authored", "spectral", "auto"}
        )
        # Capture identity is deliberately narrower than ``nodes_sha``: geometry
        # and fit hyperparameters do not change residual activations. Hash the
        # actual rendered token rows + pool positions, node partition, and a
        # loaded-model fingerprint so ordinary geometry/smoothing/layer-scope
        # refits can reuse or top up capture without trusting raw source text or
        # tokenizer identity.
        capture_dir = model_dir(self._handle.model_id) / "manifold_capture"
        cache_stem = (
            capture_sha if capture_sha is not None else None
        )
        cache_meta = (
            capture_dir / f"{cache_stem}.json"
            if cache_stem is not None else None
        )
        node_sizes = [len(responses) for _label, responses in node_groups]
        stacks_cached: dict[int, torch.Tensor] = {}
        cached_rows = None
        centroid_layers: set[int] = set()
        row_layers: set[int] = set()
        loaded_row_layers: set[int] = set()
        cached_centroid_files: dict[str, str] = {}
        cached_row_files: dict[str, str] = {}
        cached_file_digests: dict[str, str] = {}
        cached_centroid_shapes: dict[str, object] = {}
        cached_row_shapes: dict[str, object] = {}
        cached_row_dtypes: dict[str, object] = {}
        cached_row_digests: dict[str, object] = {}
        # Always read a valid v4 pointer, including on force.  Force invalidates
        # only the selected fit layers; unselected immutable shards remain part
        # of the durable union and must not be silently unlinked by a scoped fit.
        if cache_meta is not None and cache_meta.exists():
            try:
                from saklas.core.manifold import ActivationRowStore
                from saklas.io.packs import verify_integrity

                with open(cache_meta) as handle:
                    meta = json.load(handle)
                format_version = int(meta.get("format_version", 0))
                if format_version != _CAPTURE_CACHE_FORMAT_VERSION:
                    raise ValueError("legacy capture cache needs v4 reshaping")
                if (
                    meta.get("capture_sha256") != capture_sha
                    or [int(size) for size in meta.get("node_sizes", [])] != node_sizes
                ):
                    raise ValueError("capture cache metadata identity mismatch")
                cache_files = dict(meta.get("files", {}))
                cached_file_digests = {
                    str(filename): str(digest)
                    for filename, digest in cache_files.items()
                }
                centroid_layers = {
                    int(idx) for idx in meta.get("centroid_layers", [])
                }
                row_layers = {int(idx) for idx in meta.get("row_layers", [])}
                loaded_row_layers = set(row_layers)
                cached_centroid_files = dict(meta.get("centroid_files", {}))
                cached_row_files = dict(meta.get("row_files", {}))
                cached_centroid_shapes = dict(meta.get("centroid_shapes", {}))
                cached_row_shapes = dict(meta.get("row_shapes", {}))
                cached_row_dtypes = dict(meta.get("row_dtypes", {}))
                cached_row_digests = dict(meta.get("row_tensor_sha256", {}))
                assert cache_stem is not None
                if any(
                    str(idx) not in cached_centroid_files
                    or str(idx) not in cached_centroid_shapes
                    or cached_centroid_files.get(str(idx)) not in cache_files
                    or cached_centroid_files.get(str(idx)) != _centroid_shard_path(
                        capture_dir, cache_stem, idx,
                    ).name
                    for idx in centroid_layers
                ) or any(
                    str(idx) not in cached_row_files
                    or str(idx) not in cached_row_shapes
                    or str(idx) not in cached_row_dtypes
                    or str(idx) not in cached_row_digests
                    or cached_row_files.get(str(idx)) != _row_shard_path(
                        capture_dir, cache_stem, idx,
                    ).name
                    for idx in row_layers
                ):
                    raise ValueError("capture cache metadata omits layer shards")
                if not row_layers <= centroid_layers:
                    raise ValueError("row-cache layers exceed centroid coverage")

                if force:
                    forced = set(fit_layers)
                    centroid_layers.difference_update(forced)
                    row_layers.difference_update(forced)

                # Verify and load only the layers this fit can consume.  A bad
                # selected shard invalidates that layer, not unrelated durable
                # coverage, so the capture pass tops up exactly the damage.
                for idx in sorted(centroid_layers & set(fit_layers)):
                    filename = cached_centroid_files[str(idx)]
                    path = capture_dir / filename
                    expected_file_digest = cache_files.get(filename, "")
                    ok, _bad = verify_integrity(
                        capture_dir, {filename: expected_file_digest},
                    )
                    if not ok:
                        centroid_layers.discard(idx)
                        row_layers.discard(idx)
                        continue
                    try:
                        shard = load_file(str(path), device="cpu")
                        key = f"layer_{idx}"
                        if set(shard) != {key}:
                            raise ValueError("centroid shard key mismatch")
                        value = shard[key].to(torch.float32)
                        if (
                            cached_centroid_shapes.get(str(idx)) != list(value.shape)
                            or value.ndim != 2 or int(value.shape[0]) != K
                            or not bool(torch.isfinite(value).all())
                        ):
                            raise ValueError("centroid shard shape/finite mismatch")
                        stacks_cached[idx] = value
                    except (OSError, RuntimeError, ValueError, KeyError):
                        centroid_layers.discard(idx)
                        row_layers.discard(idx)

                if row_layers and retain_node_rows:
                    selected_row_layers = sorted(row_layers & set(fit_layers))
                    if selected_row_layers:
                        valid_stores: list[ActivationRowStore] = []
                        for idx in selected_row_layers:
                            path = capture_dir / cached_row_files[str(idx)]
                            row_store: ActivationRowStore | None = None
                            try:
                                row_store = ActivationRowStore.load_shards(
                                    {idx: path}, node_sizes,
                                )
                                if not _row_tensor_matches_cache_metadata(
                                    path, idx, row_store.flat_rows(idx),
                                    expected_shape=cached_row_shapes.get(str(idx)),
                                    expected_dtype=cached_row_dtypes.get(str(idx)),
                                    expected_digest=cached_row_digests.get(str(idx)),
                                ):
                                    raise ValueError(
                                        "row cache metadata shape/dtype mismatch"
                                    )
                                valid_stores.append(row_store)
                                row_store = None  # ownership moved to valid_stores
                            except (OSError, RuntimeError, ValueError, KeyError):
                                if row_store is not None:
                                    row_store.close()
                                row_layers.discard(idx)
                        if len(valid_stores) > 1:
                            cached_rows = ActivationRowStore.combine_disjoint(
                                valid_stores,
                            )
                        elif valid_stores:
                            cached_rows = valid_stores[0]
                loaded_row_layers = set(row_layers)
            except (
                OSError, RuntimeError, ValueError, KeyError,
                json.JSONDecodeError, TypeError,
            ):
                stacks_cached = {}
                centroid_layers = set()
                row_layers = set()
                loaded_row_layers = set()
                cached_centroid_files = {}
                cached_row_files = {}
                cached_file_digests = {}
                cached_centroid_shapes = {}
                cached_row_shapes = {}
                cached_row_dtypes = {}
                cached_row_digests = {}
                if cached_rows is not None:
                    cached_rows.close()
                    cached_rows = None

        needed_layers = set(fit_layers) - centroid_layers
        if retain_node_rows and mf.fit_mode != "auto":
            needed_layers |= set(fit_layers) - row_layers
        capture_layers = sorted(needed_layers)
        new_rows = None
        if capture_layers:
            capture_started = time.perf_counter()
            _progress(
                f"Pooling {len(node_groups)} nodes in fit-wide batches "
                f"({sum(node_sizes)} responses, layers {capture_layers})..."
            )
            from saklas.core.vectors import _ReusablePooledCapture

            # Protocol-only test/dummy handles can expose opaque layer
            # sentinels while monkeypatching the encoder seam.  Production HF
            # blocks are Modules; only they can host the fit-scoped reusable
            # hooks.  The null context preserves that explicit encoder seam.
            reusable = all(
                hasattr(layers[idx], "register_forward_hook")
                for idx in capture_layers
            )
            capture_scope = (
                _ReusablePooledCapture(model, layers, capture_layers)
                if reusable else nullcontext(None)
            )
            with capture_scope as capture_context:
                newly_captured, new_rows = compute_manifold_node_stats(
                    model, tokenizer, layers, device, node_groups, baseline_prompts,
                    roles=node_roles, model_type=model_type,
                    layer_indices=capture_layers, retain_rows=retain_node_rows,
                    prepared_rows=prepared_rows,
                    capture_context=capture_context,
                )
            for idx in capture_layers:
                stacks_cached[idx] = newly_captured[idx].to(
                    torch.float32,
                ).contiguous()
            # ``to(...).contiguous()`` aliases an already-contiguous fp32
            # capture. Drop the producer mapping now so SAE reconstruction can
            # release each raw layer as soon as its decoded replacement exists.
            del newly_captured
            centroid_layers.update(capture_layers)
            capture_elapsed = time.perf_counter() - capture_started
            _progress(
                f"Captured {sum(node_sizes)} rows across "
                f"{len(capture_layers)} layers in "
                f"{capture_elapsed:.1f}s."
            )
        elif fit_layers:
            _progress("Reusing token-exact manifold activation capture cache...")

        retained_rows = None
        captured_row_layers = (
            set(new_rows.layer_indices) if new_rows is not None else set()
        )
        if retain_node_rows:
            from saklas.core.manifold import ActivationRowStore

            if cached_rows is not None and new_rows is None:
                retained_rows = cached_rows
                cached_rows = None
            elif new_rows is not None and cached_rows is None:
                retained_rows = new_rows
                new_rows = None
            else:
                assert cached_rows is not None and new_rows is not None
                retained_rows = ActivationRowStore.combine_disjoint(
                    [cached_rows, new_rows],
                )
                cached_rows = None
                new_rows = None

        if capture_layers and cache_stem is not None and cache_meta is not None:
            from saklas.io.atomic import write_json_atomic
            from saklas.io.packs import hash_file

            try:
                capture_dir.mkdir(parents=True, exist_ok=True)
                persist_rows_now = mf.fit_mode != "auto"
                files = {
                    cached_centroid_files[str(idx)]: cached_file_digests[
                        cached_centroid_files[str(idx)]
                    ]
                    for idx in centroid_layers
                    if (
                        str(idx) in cached_centroid_files
                        and cached_centroid_files[str(idx)] in cached_file_digests
                    )
                }
                for idx in capture_layers:
                    centroid_path = _centroid_shard_path(capture_dir, cache_stem, idx)
                    _save_safetensors_atomic(
                        {f"layer_{idx}": stacks_cached[idx]}, centroid_path,
                    )
                    cached_centroid_files[str(idx)] = centroid_path.name
                    cached_centroid_shapes[str(idx)] = list(
                        stacks_cached[idx].shape,
                    )
                    files[centroid_path.name] = hash_file(centroid_path)

                if persist_rows_now and retained_rows is not None:
                    for idx in sorted(captured_row_layers):
                        row_path = _row_shard_path(capture_dir, cache_stem, idx)
                        rows = retained_rows.flat_rows(idx)
                        _save_safetensors_atomic({f"layer_{idx}": rows}, row_path)
                        cached_row_files[str(idx)] = row_path.name
                        cached_row_shapes[str(idx)] = list(rows.shape)
                        cached_row_dtypes[str(idx)] = str(rows.dtype)
                        cached_row_digests[str(idx)] = _tensor_sha256(rows)
                    row_layers.update(captured_row_layers)
                metadata_row_layers = (
                    sorted(row_layers) if persist_rows_now
                    else sorted(loaded_row_layers)
                )
                write_json_atomic(cache_meta, {
                    "format_version": _CAPTURE_CACHE_FORMAT_VERSION,
                    "capture_sha256": capture_sha,
                    "node_sizes": node_sizes,
                    "centroid_layers": sorted(centroid_layers),
                    "row_layers": metadata_row_layers,
                    "centroid_files": {
                        str(idx): cached_centroid_files[str(idx)]
                        for idx in sorted(centroid_layers)
                    },
                    "row_files": {
                        str(idx): cached_row_files[str(idx)]
                        for idx in metadata_row_layers
                    },
                    "centroid_shapes": {
                        str(idx): cached_centroid_shapes[str(idx)]
                        for idx in sorted(centroid_layers)
                    },
                    "row_shapes": {
                        str(idx): cached_row_shapes[str(idx)]
                        for idx in metadata_row_layers
                    },
                    "row_dtypes": {
                        str(idx): cached_row_dtypes[str(idx)]
                        for idx in metadata_row_layers
                    },
                    "row_tensor_sha256": {
                        str(idx): cached_row_digests[str(idx)]
                        for idx in metadata_row_layers
                    },
                    "files": files,
                })
                keep_names = {
                    cache_meta.name,
                    *files,
                    *(
                        cached_row_files[str(idx)]
                        for idx in metadata_row_layers
                    ),
                }
                for old_path in _capture_group_paths(capture_dir, cache_stem):
                    if old_path.name not in keep_names:
                        old_path.unlink(missing_ok=True)
            except (OSError, RuntimeError, ValueError) as exc:
                # The fit result is already in memory and remains valid. A cache
                # write failure must not turn successful model work into a
                # failed fit; next run simply captures again.
                _progress(f"Activation capture cache write skipped: {exc}")

        # Centroid tensors are resident and row payloads are protected by the
        # process lease from this point onward.  Hand the short cache transaction
        # lock back before topology/PCA/RBF/covariance work; prune/GC observes the
        # lease until this fit returns (including exceptional returns).
        if _release_capture_lock is not None:
            _release_capture_lock()

        # 1b. Monopolar (concept-vs-neutral).  A 1-node ``pca`` folder has no
        #     second pole to span an affine subspace, so it can only mean one
        #     thing: the concept against the **neutral baseline**.  Its steering
        #     direction is the displacement of the single concept centroid from
        #     the model's neutral activation mean ν (``layer_means``) — sourced
        #     fresh per model, never a stored corpus — folded into a 1-node
        #     neutral-anchored ray via :func:`fold_directions_to_subspace` (the
        #     same primitive ``merge`` uses).  No discover-coords, no DLS, no
        #     synthetic second node; ``concept − ν`` already cancels common-mode
        #     like DiM, so the raw δ̂ basis is appropriate.  The engine
        #     recognizes the shape structurally (a flat ``pca`` fit otherwise
        #     needs ``k + 1 ≥ 2`` poised nodes).
        if mf.fit_mode == "pca" and K == 1:
            from saklas.core.vectors import fold_directions_to_subspace
            per_node = [
                {idx: stacks_cached[idx][0] for idx in fit_layers}
            ]
            means = getattr(self._handle, "layer_means", None) or {}
            if not means:
                raise ValueError(
                    f"monopolar manifold {mf.name!r} (a 1-node concept-vs-"
                    f"neutral fit) needs the model's neutral activation mean "
                    f"(layer_means) as its negative pole, but none is "
                    f"available; regenerate the neutral corpus so layer_means "
                    f"can build"
                )
            _wh = getattr(self._handle, "whitener", None)
            if _wh is None or not _wh.covers_all(fit_layers):
                raise WhitenerError(
                    f"monopolar manifold {mf.name!r}: the Mahalanobis "
                    f"whitener must cover every fit layer to bake the share "
                    f"(regenerate the neutral activation cache for "
                    f"{self._handle.model_id!r})"
                )
            maha = _wh
            concept_label = node_groups[0][0]
            directions: dict[int, torch.Tensor] = {}
            for idx in fit_layers:
                nu = means.get(idx)
                if nu is None:
                    continue  # no neutral anchor at this layer → can't fold it
                c = per_node[0][idx].to(torch.float32).reshape(-1)
                if sae_backend is not None:
                    with torch.no_grad():
                        feat = sae_backend.encode_layer(
                            idx, c.reshape(1, -1).to(device),
                        )
                        c = sae_backend.decode_layer(idx, feat).detach().to(
                            "cpu", torch.float32,
                        ).reshape(-1)
                directions[idx] = c - nu.to(torch.float32).reshape(-1)
            _progress(
                f"Folding monopolar '{concept_label}' − neutral across "
                f"{len(directions)} layers..."
            )
            manifold = fold_directions_to_subspace(
                mf.name, directions, means, whitener=maha,
                label=concept_label, feature_space=feature_space,
            )
            # Carry the per-node role/kind onto the fitted ray so steer-time
            # role substitution (``nearest_node_role``) and provenance work; the
            # fold primitive itself is role-agnostic.
            manifold.node_roles = list(node_roles)
            manifold.node_kinds = list(node_kinds)
            metadata: dict[str, Any] = {
                "method": (
                    "manifold_monopolar_sae" if sae_backend is not None
                    else "manifold_monopolar"
                ),
                "nodes_sha256": nodes_sha,
                "model_fingerprint": model_fingerprint,
                "capture_sha256": capture_sha,
                "fitted_layers": list(fit_layers),
                "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
                "monopolar": True,
                # The fold keeps the raw δ̂ basis (``concept − ν`` cancels
                # common-mode like DiM by differencing), so the subspace is
                # metric-free: ``subspace_metric`` is "euclidean" as a basis
                # *label* (no whitened-PCA selection ran), not a fallback.
                # The share is always whitened (the gate above guarantees it).
                "share_metric": "mahalanobis",
                "subspace_metric": "euclidean",
            }
            if sae_backend is not None:
                metadata["sae_release"] = sae_backend.release
                metadata["sae_revision"] = sae_backend.revision
                metadata["sae_fingerprint"] = sae_backend.fingerprint
                metadata["sae_ids_by_layer"] = dict(
                    getattr(sae_backend, "sae_ids_by_layer", {})
                )
                metadata["sae_full_coverage"] = layer_indices is None
            if any_role:
                metadata["node_roles"] = list(node_roles)
            if any_kind:
                metadata["node_kinds"] = list(node_kinds)
            _publish_fit_if_current(
                pathlib.Path(folder),
                expected_revision=_authoring_revision or nodes_sha,
                manifold=manifold, tensor_path=tensor_path, metadata=metadata,
            )
            manifold.metadata.update(metadata)
            self._events.emit(ManifoldExtracted(
                name=mf.name, manifold=manifold, metadata=metadata,
            ))
            _progress(
                f"Fit complete in {time.perf_counter() - fit_started:.1f}s."
            )
            return manifold

        # 2. Resolve the Mahalanobis whitener once.  It is mandatory for an
        #    activation-space fit — without it the basis, mean, and steering
        #    direction get dominated by rogue (massive-activation) channels —
        #    and it now also drives the layer-agnostic coordinate derivation
        #    below, so the discover-coord consensus and the per-layer fit share
        #    one resolution.  Resolved *after* the cache-hit return and the
        #    fail-fast SAE/role checks since ``handle.whitener`` can trigger a
        #    lazy neutral-activation build.  It must cover *every* fit layer (so
        #    the cross-layer-normalized share — and the consensus Gram — compare
        #    like with like); partial coverage or a handle without a whitener is
        #    a hard error, not a Euclidean fallback.  The basis is model-space
        #    for both raw and SAE fits (centroids are decoded back before the
        #    fit), so the residual-stream whitener applies to the SAE variant
        #    unchanged.
        _whitener = getattr(self._handle, "whitener", None)
        if _whitener is None or not _whitener.covers_all(fit_layers):
            raise WhitenerError(
                f"manifold {mf.name!r}: the Mahalanobis whitener must cover "
                f"every fit layer (regenerate the neutral activation cache "
                f"for {self._handle.model_id!r})"
            )
        maha_whitener = _whitener

        # 2b. Stack per-node centroids per layer once (SAE-reconstructed when a
        #     backend is set), shared by the consensus-Gram coord derivation and
        #     the per-layer subspace fit so the SAE round-trip runs a single
        #     time.  (K, D) fp32 on CPU per layer.
        def _stacked_centroids(idx: int) -> torch.Tensor:
            # The capture cache already owns the contiguous (K,D) stack. Alias
            # it instead of rebuilding the same ~K*D payload from row views.
            # On an SAE fit, transfer ownership out of ``stacks_cached`` one
            # layer at a time. Once the decoded replacement is returned, no raw
            # centroid roster remains for that layer.
            s = (
                stacks_cached.pop(idx)
                if sae_backend is not None else stacks_cached[idx]
            )
            if sae_backend is not None:
                with torch.no_grad():
                    feat = sae_backend.encode_layer(idx, s.to(device))
                    recon = sae_backend.decode_layer(idx, feat)
                s = recon.detach().to("cpu", torch.float32)
            return s

        stacks: dict[int, torch.Tensor] = {
            idx: _stacked_centroids(idx) for idx in fit_layers
        }
        if sae_backend is not None:
            # All requested raw stacks were popped above. Clear the now-dead
            # container explicitly so future edits cannot accidentally retain a
            # raw K×D×L roster beside the reconstructed and whitened rosters.
            stacks_cached.clear()

        # 2c. Per-layer whitened between-node spread — the concept's signal
        #     concentration across the stack.  ``G_L = X̃_L Σ_L⁻¹ X̃_Lᵀ`` is the
        #     whitened (K, K) Gram of the node-mean-centered centroids;
        #     ``tr(G_L) = Σ_k ‖x̃_k‖²_M`` is the total whitened node spread at
        #     layer L, in background-σ² units (whitening makes it comparable
        #     across layers).  This is a *diagnostic* layer profile — where does
        #     this concept live? — distinct from the apply-time
        #     ``mahalanobis_share``, which restricts the same idea to the
        #     steerable subspace; a layer where ``tr(G_L)`` is large but the
        #     share is small is one whose fitted subspace is dropping concept
        #     signal (low explained variance).  The per-layer Grams are also the
        #     summands of the discover consensus Gram, so the discover branch
        #     reuses ``layer_grams`` instead of recomputing them.
        whitened_rows: dict[int, torch.Tensor] = {}
        layer_grams: dict[int, torch.Tensor] = {}
        for idx in fit_layers:
            xc = stacks[idx].to(torch.float32)
            xc = xc - xc.mean(dim=0, keepdim=True)
            sinv_xc = maha_whitener.apply_inv(idx, xc).to(torch.float32)
            gram = xc @ sinv_xc.transpose(0, 1)
            whitened_rows[idx] = sinv_xc
            layer_grams[idx] = 0.5 * (gram + gram.transpose(0, 1))  # (K, K)
        node_spread_per_layer: dict[int, float] = {
            idx: float(layer_grams[idx].diagonal().sum()) for idx in fit_layers
        }

        # 3. Resolve domain + node_params — the only step that differs
        #    between authored and discover paths.
        discover_metadata: dict[str, Any] = {}
        # ``max_subspace_dim`` caps the per-layer PCA subspace in the
        # **curved** fit (``fit_layer_subspace``; default
        # :data:`DEFAULT_N_COMPONENTS` = 64).  Smaller values constrain the
        # dim count that ``subspace_inject`` displaces at steer time — finer-
        # grained steering control at the cost of representing less per-layer
        # activation variance.  It is a *curved-only* knob: a flat (``pca``)
        # fit's per-layer subspace is exactly its k-dim layout span (the
        # affine span *is* the layout), so ``max_subspace_dim`` is not a
        # ``pca`` hyperparam — the affine branch hard-sets ``n_components`` to
        # the derived intrinsic dim and ignores any override.  Authored
        # manifolds don't currently route hyperparams so they inherit 64.
        max_subspace_dim_override: int | None = None
        # ``smoothing`` (curved discover only): penalized-RBF λ selection for
        # the per-layer surface.  ``None`` ⇒ exact interpolation — the authored
        # contract (node = exact steering target) and the flat ``pca`` path
        # (no RBF).  A curved ``spectral`` discover fit defaults to GCV
        # (``"auto"``), trading exactness for a surface that doesn't chase
        # noise in the per-node centroids.
        curved_smoothing: float | str | None = None

        # ``effective_fit_mode`` is the *resolved* geometry the per-layer fit
        # branches on: it equals ``mf.fit_mode`` for authored / pca / spectral,
        # and the topology ``select_topology`` picks for ``fit_mode="auto"``.
        effective_fit_mode = mf.fit_mode
        auto_rbf_plan = None
        auto_fisher_bases: dict[int, torch.Tensor] = {}
        if mf.fit_mode == "authored":
            domain = domain_from_spec(mf.domain)
            node_coords = torch.tensor(mf.node_coords, dtype=torch.float32)
            node_params = domain.embed(node_coords)
            method = "manifold_sae" if sae_backend is not None else "manifold_pca"
        elif mf.fit_mode == "auto":
            # Auto: pick the discover geometry per-model — flat (pca) vs curved
            # (spectral) by GCV, plus periodic (BoxDomain) axes via persistent
            # homology.  ``select_topology`` returns the resolved fit_mode +
            # coords + domain; the per-layer fit below runs unchanged on the
            # resolved mode.  Sphere is authored-only (not an auto candidate).
            from saklas.io.manifolds import sanitize_hyperparams
            from saklas.core.manifold import select_topology
            st_hyper = sanitize_hyperparams("auto", dict(mf.hyperparams))
            consensus_gram = torch.stack(
                [layer_grams[idx] for idx in fit_layers]
            ).mean(dim=0)
            _progress(
                f"Selecting topology across {len(fit_layers)} layers "
                f"({K} centroids)..."
            )
            choice = select_topology(
                {idx: stacks[idx] for idx in fit_layers},
                {idx: layer_grams[idx] for idx in fit_layers},
                consensus_gram,
                whitener=maha_whitener,
                whitened_rows=whitened_rows,
                max_dim=int(st_hyper.get("max_dim", 8)),
                smoothing=st_hyper.get("smoothing", "auto"),
                persistence_frac=float(st_hyper.get("persistence_frac", 0.5)),
                score_dim=int(
                    st_hyper.get("max_subspace_dim", DEFAULT_N_COMPONENTS)
                ),
            )
            effective_fit_mode = choice.fit_mode
            auto_rbf_plan = choice.rbf_plan
            auto_fisher_bases = choice.fisher_bases
            domain = choice.domain
            node_coords = choice.coords
            node_params = domain.embed(node_coords)
            k = int(node_coords.shape[1])
            floor = (k + 1) if effective_fit_mode == "pca" else min_nodes(k)
            if floor > K:
                raise ValueError(
                    f"auto manifold {mf.name!r}: resolved topology "
                    f"{choice.winner_name!r} (k={k}) needs >= {floor} nodes, "
                    f"got K={K}"
                )
            if effective_fit_mode == "spectral":
                curved_smoothing = st_hyper.get("smoothing", "auto")
            method = (
                "manifold_discover_sae" if sae_backend is not None
                else "manifold_discover_auto"
            )
            discover_metadata = {
                "fit_mode": "auto",
                "resolved_fit_mode": effective_fit_mode,
                "hyperparams": dict(st_hyper),
                "topology_winner": choice.winner_name,
                "topology_candidates": [
                    {
                        "name": c.name,
                        "fit_mode": c.fit_mode,
                        "intrinsic_dim": c.intrinsic_dim,
                        "score": (c.score if math.isfinite(c.score) else None),
                        "viable": c.viable,
                        "reason": c.reason,
                    }
                    for c in choice.candidates
                ],
            }
            # Emit the winner's coordinate diagnostics (pca per-component
            # variance / spectral eigenvalues) so the inspector renders the
            # same bars a pinned pca/spectral fit does — the auto resolution
            # otherwise leaves the panel blank.
            if choice.diagnostics is not None:
                discover_metadata["diagnostics"] = _diagnostics_to_dict(
                    choice.diagnostics
                )
        else:
            # Discover: derive coords from the per-node centroids, layer-
            # agnostically — there is no reference layer.  The same coords feed
            # the per-layer RBF as the manifold's intrinsic coordinates —
            # wrapped in a ``CustomDomain(k)`` with identity embed so the
            # existing fit machinery handles them unchanged.
            # Sanitize against the per-mode whitelist (single source of truth
            # in ``io.manifolds``) so a stale on-disk ``manifold.json`` carrying
            # a since-removed key (e.g. the old ``anchor_origin``) can't reach
            # ``discover_coords`` as an unexpected kwarg.  Author/CLI paths
            # already sanitize; this guards legacy + hand-edited folders.
            from saklas.io.manifolds import sanitize_hyperparams
            hyperparams = sanitize_hyperparams(mf.fit_mode, dict(mf.hyperparams))
            # ``max_subspace_dim`` is consumed by the curved per-layer fit,
            # not by ``discover_coords`` — pop it before the discover call so
            # the dispatcher doesn't get an unexpected kwarg.  (Sanitized out
            # of ``pca`` folders above; the affine branch ignores it regardless
            # — see the affine fit below.)
            if "max_subspace_dim" in hyperparams:
                max_subspace_dim_override = int(hyperparams.pop("max_subspace_dim"))
            # ``smoothing`` is consumed by the penalized curved fit
            # (``fit_layer_subspace`` → ``fit_rbf_smoothed``), not by
            # ``discover_coords`` — pop it before the dispatch so it isn't an
            # unexpected kwarg.  Only the curved (``spectral``) path has an RBF
            # surface to smooth; the flat (``pca``) path's affine span has no
            # interpolant, so smoothing never applies there.
            if mf.fit_mode == "spectral":
                curved_smoothing = hyperparams.pop("smoothing", "auto")
            else:
                hyperparams.pop("smoothing", None)
            # Consensus Gram: the mean over every fit layer of that layer's
            # whitened, node-mean-centered (K, K) Gram ``X̃_L Σ_L⁻¹ X̃_Lᵀ``
            # (the ``layer_grams`` already computed for the spread profile).
            # Whitening puts each layer in common (background-σ) units, so the
            # raw average is **signal-weighted** — a layer where the nodes
            # aren't separated contributes a near-zero Gram and drops out, while
            # a layer where the concept is strongly represented dominates.  PCA
            # eigendecomposes this Gram; spectral reads its pairwise distances.
            # The (K, K) Gram is the layer-invariant object, so this is exactly
            # the single-reference-layer derivation generalized to the whole
            # stack — ``%`` positions and node labels live in one coordinate
            # system distilled from all layers, wherever the concept's signal
            # happens to concentrate.
            _progress(
                f"Deriving {mf.fit_mode} coords across {len(fit_layers)} "
                f"layers ({K} centroids)..."
            )
            consensus_gram = torch.stack(
                [layer_grams[idx] for idx in fit_layers]
            ).mean(dim=0)
            derived_coords, diagnostics = discover_coords(
                consensus_gram, method=mf.fit_mode, **hyperparams,
            )
            k = derived_coords.shape[1]
            # Node-count floor.  The curved (spectral) path fits an RBF
            # surface and needs the poisedness floor ``min_nodes(k) = 2k+1``.
            # The flat (``pca``) path fits an *affine* subspace with no RBF —
            # it only needs ``k+1`` affinely-independent centroids to span a
            # k-dim subspace, so a rank-1 (k=1) fit is valid at K=2: that is a
            # difference-of-means steering vector (ARCHITECTURE §1/§5, "a
            # vector = a 2-node fit_mode=pca folder").
            if mf.fit_mode == "pca":
                floor = k + 1
                floor_reason = "to span the affine subspace"
            else:
                floor = min_nodes(k)
                floor_reason = "for the RBF fit"
            if floor > K:
                raise ValueError(
                    f"discover manifold {mf.name!r}: picked k={k} needs "
                    f">= {floor} nodes {floor_reason}, got K={K}"
                )
            # The derived coords come out PCA-mean-centered (origin = the node
            # centroid).  The per-layer fit below neutral-anchors each layer's
            # *real* reduced frame (coord 0 = neutral), and the shared display
            # layout is re-anchored on neutral in step 4a after the fit, so the
            # display/`%`-authoring origin matches the steer origin.
            domain = CustomDomain(k)
            node_coords = derived_coords
            node_params = derived_coords  # identity embedding
            method = (
                "manifold_discover_sae" if sae_backend is not None
                else f"manifold_discover_{mf.fit_mode}"
            )
            # Record what we ran with — fit_mode + hyperparams (with
            # any derived defaults filled in, e.g. spectral's resolved
            # ``k_nn``/``bandwidth``) + the diagnostics for the sidecar
            # and the inspector surfaces.
            resolved_hyperparams = dict(hyperparams)
            if max_subspace_dim_override is not None:
                resolved_hyperparams["max_subspace_dim"] = max_subspace_dim_override
            if curved_smoothing is not None:
                resolved_hyperparams["smoothing"] = curved_smoothing
            if hasattr(diagnostics, "k_nn"):  # SpectralDiagnostics
                resolved_hyperparams["k_nn"] = int(diagnostics.k_nn)  # pyright: ignore[reportAttributeAccessIssue]  # SpectralDiagnostics only; guarded by hasattr
                resolved_hyperparams["bandwidth"] = float(
                    diagnostics.bandwidth,  # pyright: ignore[reportAttributeAccessIssue]  # SpectralDiagnostics only; guarded by hasattr
                )
            discover_metadata = {
                "fit_mode": mf.fit_mode,
                "hyperparams": resolved_hyperparams,
                "diagnostics": _diagnostics_to_dict(diagnostics),
            }

        # 4. Per-layer fit over the centroid stacks built in step 2b (SAE
        #    reconstruction, when set, already folded in there).  The whitener
        #    (mandatory; resolved in step 2) selects the whitened/Fisher basis
        #    and bakes the per-layer share.
        _progress(
            f"Fitting RBF interpolant across {len(fit_layers)} layers..."
        )
        layer_subs = {}
        mahalanobis_share: dict[int, float] = {}
        rbf_plan = auto_rbf_plan
        # Per-layer penalized-RBF provenance (curved + ``smoothing`` only):
        # ``{layer: {"lambda", "edf", "gcv"}}`` from the GCV select, for the
        # sidecar + inspector.  Empty for exact / flat fits.
        rbf_smoothing_per_layer: dict[int, dict[str, float]] = {}
        # Neutral baseline (probe-centering means), **ungated** by the
        # whitener: the curved fit's neutral-anchor (``mean = P_basis(ν)``,
        # §5) consumes it.  ``None`` on a CPU-stub handle ⇒ the fit falls back
        # to the centroid-mean anchor.  (Also drives the origin foot below.)
        _handle_means = getattr(self._handle, "layer_means", None)
        fit_kwargs: dict[str, Any] = {}
        if max_subspace_dim_override is not None:
            fit_kwargs["n_components"] = max_subspace_dim_override

        def _neutral_for(idx: int) -> "torch.Tensor | None":
            return (
                _handle_means[idx]
                if _handle_means is not None and idx in _handle_means
                else None
            )

        def _bake_share(
            idx: int, sub: Any, mu_coords: torch.Tensor,
        ) -> None:
            # Per-layer budget = the **μ-centered** (anchor-independent)
            # whitened spread (§5 / ``subspace_share``): the share measures
            # signal spread, not where neutral sits.  The whitener is
            # guaranteed to cover every fit layer (checked above).
            mahalanobis_share[idx] = subspace_share(
                mu_coords, sub.basis, whitener=maha_whitener, layer=idx,
            )

        if effective_fit_mode == "pca":
            # FLAT affine fit — the discover-``pca`` path is a flat rank-``k``
            # subspace, not an RBF surface (ARCHITECTURE §1/§5).  Per layer:
            # ``fit_affine_subspace`` (μ-centered PCA basis at the derived
            # intrinsic dim ``k``, neutral-anchored frame, real per-layer node
            # coords).  **Whitened/Fisher basis**, always — the de-rogued basis
            # is what makes the no-lever gain coherent, and the whitener is
            # mandatory (checked above), so there is no Euclidean fit path here.
            # The shared ``node_coords`` stays the derived
            # PCA layout (display/labels); the real per-layer steer coords live
            # on each ``LayerSubspace.node_coords``.
            # The per-layer steerable subspace is exactly the k-dim layout
            # span — ``n_components`` is the derived intrinsic dim (the shared
            # layout's width, set for every fit_mode, so always bound here
            # unlike the discover-only ``k``).  There is no separate
            # ``max_subspace_dim`` knob for a flat fit: the affine span *is*
            # the layout, so any stray curved-fit override is ignored here.
            affine_kwargs = {"n_components": int(node_coords.shape[1])}
            raw_fits: dict[int, tuple[Any, torch.Tensor]] = {}
            for idx in fit_layers:
                stacked = stacks[idx]
                sub, mu_coords, _ev_ratio = fit_affine_subspace(
                    stacked, neutral_mean=_neutral_for(idx),
                    whitener=maha_whitener, layer=idx,
                    whitened_gram=layer_grams[idx],
                    whitened_rows=whitened_rows[idx],
                    orient_to=0,
                    basis_override=auto_fisher_bases.get(idx),
                    **affine_kwargs,
                )
                raw_fits[idx] = (sub, mu_coords)
            # Per-axis DLS straddle over all fit layers at once (flat → DLS;
            # the global all-fail fallback matches the folded-vector path).
            dls_kept = compute_dls_axes(
                {idx: stacks[idx] for idx in raw_fits},
                {idx: raw_fits[idx][0].basis for idx in raw_fits},
                _handle_means,
            )
            for idx, (sub, mu_coords) in raw_fits.items():
                kept = sorted(dls_kept.get(idx, set()))
                if not kept:
                    continue  # no axis straddles the baseline → drop the layer
                if len(kept) < sub.rank:
                    sub = sub.select_axes(kept)
                    mu_coords = mu_coords[:, kept]
                layer_subs[idx] = sub
                _bake_share(idx, sub, mu_coords)
        else:
            # CURVED (authored / spectral): RBF surface fit, neutral-anchored.
            # ``curved_smoothing`` is ``None`` for authored (exact-interpolation
            # contract) and the GCV / fixed-λ selector for ``spectral``.
            if rbf_plan is None:
                rbf_plan = prepare_rbf_fit_plan(
                    node_params, smoothing=curved_smoothing,
                )
            for idx in fit_layers:
                stacked = stacks[idx]
                # ``neutral_mean`` neutral-anchors the frame; ``maha_whitener``
                # (mandatory, checked above) selects the whitened/Fisher basis.
                _rbf_info: dict[str, float] = {}
                sub, _ev_ratio = fit_layer_subspace(
                    stacked, node_params,
                    whitener=maha_whitener, layer=idx,
                    neutral_mean=_neutral_for(idx),
                    whitened_gram=layer_grams[idx],
                    whitened_rows=whitened_rows[idx],
                    smoothing=curved_smoothing,
                    rbf_info=_rbf_info,
                    rbf_plan=rbf_plan,
                    basis_override=auto_fisher_bases.get(idx),
                    **fit_kwargs,
                )
                layer_subs[idx] = sub
                if _rbf_info:
                    rbf_smoothing_per_layer[idx] = _rbf_info
                # μ-centered share (NOT ``eval_rbf(node_params)`` — the surface
                # is neutral-anchored, so its node values aren't μ-centered).
                mu_centered = stacked.to(torch.float32)
                mu_centered = mu_centered - mu_centered.mean(dim=0)
                mu_coords = mu_centered @ sub.basis.to(torch.float32).T  # (K, R)
                _bake_share(idx, sub, mu_coords)

        # 4a. Neutral-center the flat (pca) display layout.  ``discover_coords``
        #     returns node coords centered on the **node centroid** (PCA removes
        #     the node mean), so the shared layout's origin is "the average
        #     node", not the neutral baseline — a coord-form ``% 0,…,0`` would
        #     mean the centroid and the rack sliders read displacements from it.
        #     Re-anchor the layout on neutral so ``% 0,…,0`` reads as neutral and
        #     the sliders share the geometry plot's origin (whose ``neutral_white``
        #     is already the whitened origin).  ``neutral_layout_coord`` is the
        #     landmark-MDS projection of the neutral baseline into the consensus-
        #     PCA layout from neutral's whitened, node-mean-centered cross-Gram
        #     column ``gᵢ = mean_L (ν_L − μ_L)ᵀ Σ_L⁻¹ (c_{L,i} − μ_L)`` (the same
        #     layer-averaged metric the consensus Gram itself uses).  Subtracting
        #     it is a **pure translation**: steering is unchanged because
        #     ``_affine_manifold_push`` blends per-layer *neutral-anchored* coords
        #     with cardinal weights, which are translation-invariant — the new
        #     origin maps to neutral's in-span projection, so a coord-form ``%`` at
        #     (0,...,0) pushes by neutral's (small) off-hull residual, not the full
        #     slide toward the node centroid it used to.  Flat only: a curved
        #     layout carries authored coords / a
        #     per-layer ``origin`` foot, not a layout coordinate for neutral.
        #     Needs the neutral baseline — a CPU-stub handle without
        #     ``layer_means`` keeps the centroid origin (graceful, fit still valid).
        if effective_fit_mode == "pca" and _handle_means is not None:
            g_cols: list[torch.Tensor] = []
            for idx in fit_layers:
                nu = _neutral_for(idx)
                if nu is None:
                    g_cols = []
                    break
                xc = stacks[idx].to(torch.float32)
                xc = xc - xc.mean(dim=0, keepdim=True)
                mu_L = stacks[idx].to(torch.float32).mean(dim=0, keepdim=True)
                nu_c = nu.to(torch.float32).reshape(1, -1) - mu_L  # (1, D) ν − μ
                g_cols.append(whitened_rows[idx] @ nu_c.reshape(-1))  # (K,) x̃ᵀΣ⁻¹ν̃
            if g_cols:
                g_nu = torch.stack(g_cols).mean(dim=0)             # (K,) layer-avg
                neutral_coords = neutral_layout_coord(node_coords, g_nu)
                node_coords = (node_coords - neutral_coords).contiguous()
                node_params = domain.embed(node_coords)

        if effective_fit_mode == "pca" and retained_rows is not None:
            retained_rows.close()
            retained_rows = None

        # 4b. Fuzzy-manifold σ-field (curved + raw only).  The source-dtype
        #     activation-row spool retained from the first capture accumulates
        #     each node's within-node reduced ``(R, R)`` covariance after the
        #     basis exists, then
        #     ``fit_sigma_field`` reduces it to one off-surface ``σ`` per node and
        #     fits a ``log σ`` RBF onto each ``LayerSubspace`` (mutated in place).
        #     This gives the surface a *tube thickness* that soft-``onto`` steers
        #     into and the monitor reads as a soft node-assignment bandwidth.
        #     Skipped for SAE fits (the σ would mix raw activation spread with an
        #     SAE-reconstructed mean surface) and flat fits (``H_n ≡ 0``, the tube
        #     is vacuous) — both leave ``σ`` absent ⇒ exact zero-thickness legacy.
        sigma_field_per_layer: dict[int, dict[str, float]] = {}
        if effective_fit_mode != "pca" and sae_backend is None and layer_subs:
            if mf.fit_mode == "auto":
                from saklas.core.manifold import ActivationRowStore

                stores: list[ActivationRowStore] = []
                if retained_rows is not None:
                    stores.append(retained_rows)
                already_loaded = {
                    idx for store in stores for idx in store.layer_indices
                }
                selected_cached_layers = sorted(
                    (row_layers & set(fit_layers)) - already_loaded
                )
                if selected_cached_layers and cache_stem is not None:
                    for idx in selected_cached_layers:
                        path = capture_dir / cached_row_files[str(idx)]
                        auto_store: ActivationRowStore | None = None
                        try:
                            auto_store = ActivationRowStore.load_shards(
                                {idx: path}, node_sizes,
                            )
                            if not _row_tensor_matches_cache_metadata(
                                path, idx, auto_store.flat_rows(idx),
                                expected_shape=cached_row_shapes.get(str(idx)),
                                expected_dtype=cached_row_dtypes.get(str(idx)),
                                expected_digest=cached_row_digests.get(str(idx)),
                            ):
                                raise ValueError("row cache metadata mismatch")
                            stores.append(auto_store)
                            auto_store = None  # ownership moved to stores
                        except (OSError, RuntimeError, ValueError, KeyError):
                            if auto_store is not None:
                                auto_store.close()
                            row_layers.discard(idx)
                covered_rows = {
                    idx for store in stores for idx in store.layer_indices
                }
                missing_row_layers = sorted(set(fit_layers) - covered_rows)
                if missing_row_layers:
                    from saklas.core.vectors import _ReusablePooledCapture

                    reusable = all(
                        hasattr(layers[idx], "register_forward_hook")
                        for idx in missing_row_layers
                    )
                    capture_scope = (
                        _ReusablePooledCapture(model, layers, missing_row_layers)
                        if reusable else nullcontext(None)
                    )
                    with capture_scope as capture_context:
                        _unused, missing_rows = compute_manifold_node_stats(
                            model, tokenizer, layers, device,
                            node_groups, baseline_prompts,
                            roles=node_roles, model_type=model_type,
                            layer_indices=missing_row_layers, retain_rows=True,
                            prepared_rows=prepared_rows,
                            capture_context=capture_context,
                        )
                    assert missing_rows is not None
                    stores.append(missing_rows)
                if len(stores) > 1:
                    retained_rows = ActivationRowStore.combine_disjoint(stores)
                elif stores:
                    retained_rows = stores[0]
            if retained_rows is None:
                raise RuntimeError(
                    "curved raw manifold fit lost its retained activation-row "
                    "spool before the sigma-field pass"
                )
            # Auto kept a temporary mmap during its one model pass but did not
            # publish/hash the multi-GiB row cache speculatively. Curvature has
            # now won, so make that reusable payload durable before consuming
            # the spool; flat auto fits closed it above without this I/O.
            if (
                mf.fit_mode == "auto"
                and cache_stem is not None
                and cache_meta is not None
                and cache_meta.exists()
            ):
                try:
                    _publish_deferred_row_shards(
                        capture_dir, cache_stem, cache_meta, retained_rows,
                        node_sizes,
                        row_layers, cached_row_files, cached_row_shapes,
                        cached_row_dtypes, cached_row_digests,
                    )
                except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                    _progress(f"Deferred activation-row cache write skipped: {exc}")
            _progress(
                f"Fitting within-node σ-field across {len(layer_subs)} layers "
                f"({K} nodes, retained activation rows)..."
            )
            node_covs = compute_store_reduced_covariances(
                retained_rows, layer_subs,
            )
            retained_rows.close()
            retained_rows = None
            sigma_field_per_layer = fit_sigma_field(
                layer_subs, domain, node_coords, node_covs,
                smoothing=(
                    curved_smoothing if curved_smoothing is not None else "auto"
                ),
                rbf_plan=(rbf_plan if effective_fit_mode != "pca" else None),
            )

        # Origin ``O_L`` — the per-layer foot of the neutral mean on ``M``, in
        # authoring coords ``(n,)``.  **Curved only** — a flat affine subspace's
        # surface fills its span, so neutral's foot is reduced-coord 0 (the
        # ``!`` ablation target) with no stored origin (§2); routing an affine
        # subspace through ``invert_parameterization`` would also hit
        # ``rbf_params()`` and raise.  Each layer's cold-start foot seed; layers
        # whose mean isn't resolvable (CPU stub) are simply absent.
        origin: dict[int, torch.Tensor] = {}
        if effective_fit_mode != "pca" and _handle_means is not None:
            for idx, sub in layer_subs.items():
                if idx not in _handle_means:
                    continue
                mu = _handle_means[idx].to(
                    device="cpu", dtype=torch.float32,
                ).reshape(-1)
                # Reduced coords of the neutral mean in this layer's subspace,
                # then nearest-point invert onto ``M``.
                q = (mu - sub.mean) @ sub.basis.T  # (R,)
                O_coords, _dist = invert_parameterization(
                    sub, domain, q, node_coords,
                )
                origin[idx] = O_coords.reshape(-1).to(torch.float32)

        manifold = Manifold(
            name=mf.name,
            domain=domain,
            node_labels=[label for label, _ in node_groups],
            node_coords=node_coords,
            layers=layer_subs,
            feature_space=feature_space,
            node_roles=list(node_roles),
            node_kinds=list(node_kinds),
            mahalanobis_share=mahalanobis_share,
            origin=origin,
        )

        # 5. Persist + refresh the folder integrity manifest.
        metadata: dict[str, Any] = {
            "method": method,
            "nodes_sha256": nodes_sha,
            "model_fingerprint": model_fingerprint,
            "capture_sha256": capture_sha,
            "fitted_layers": list(fit_layers),
            "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
            # Provenance only (nothing branches on these at load).  The
            # whitener is mandatory for an activation-space fit, so both the
            # per-layer share weighting and the PCA *subspace selection* are
            # always whitened/Fisher — no Euclidean fit path survives.
            "share_metric": "mahalanobis",
            "subspace_metric": "mahalanobis",
            # Diagnostic layer profile (see step 2c): the whitened between-node
            # spread per layer, ``{str(L): tr(G_L)}``.  Not consumed by any
            # runtime path — surfaced by `manifold show` as the concept's
            # signal-by-layer curve.
            "node_spread_per_layer": {
                str(idx): node_spread_per_layer[idx] for idx in fit_layers
            },
        }
        if rbf_smoothing_per_layer:
            # Penalized-RBF provenance: the GCV-chosen λ + effective dof per
            # layer (curved discover fits only).  Diagnostic — surfaced by
            # `manifold show`, nothing branches on it at load.
            metadata["rbf_smoothing_per_layer"] = {
                str(idx): info for idx, info in rbf_smoothing_per_layer.items()
            }
        if sigma_field_per_layer:
            # Fuzzy-manifold σ-field provenance: per-layer within-node off-surface
            # spread summary (``sigma_mean``/``sigma_min``/``sigma_max`` + the
            # log-σ RBF's smoothing λ).  Diagnostic — the σ-RBF tensors themselves
            # ride the per-model safetensors; this is the inspector-facing
            # summary.  Absent on flat / SAE / legacy fits (no tube).
            metadata["sigma_field_per_layer"] = {
                str(idx): info for idx, info in sigma_field_per_layer.items()
            }
        if sae_backend is not None:
            metadata["sae_release"] = sae_backend.release
            metadata["sae_revision"] = sae_backend.revision
            metadata["sae_fingerprint"] = sae_backend.fingerprint
            metadata["sae_ids_by_layer"] = dict(
                getattr(sae_backend, "sae_ids_by_layer", {})
            )
            metadata["sae_full_coverage"] = layer_indices is None
        if any_role:
            # Per-node roles ride into the sidecar so `manifold
            # show` and the inspector surfaces can report "this node was
            # pooled as <role>" without re-reading manifold.json.  The
            # order matches ``node_labels``; a missing entry means
            # ``None`` (standard assistant baseline).
            metadata["node_roles"] = list(node_roles)
        if any_kind:
            # Per-node kind rides into the sidecar for inspector/provenance,
            # ``node_labels`` order; absent when no node carries a kind.
            metadata["node_kinds"] = list(node_kinds)
        metadata.update(discover_metadata)
        _publish_fit_if_current(
            pathlib.Path(folder),
            expected_revision=_authoring_revision or nodes_sha,
            manifold=manifold, tensor_path=tensor_path, metadata=metadata,
        )
        manifold.metadata.update(metadata)

        self._events.emit(ManifoldExtracted(
            name=mf.name, manifold=manifold, metadata=metadata,
        ))
        _progress(f"Fit complete in {time.perf_counter() - fit_started:.1f}s.")
        return manifold
