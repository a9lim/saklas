"""Discovery + the authoring path for manifold-steering artifacts.

Folder discovery (:func:`iter_manifold_folders`) and every function that
*writes* a fresh or extended manifold folder from corpora — the
create/init/append/plan/merge/update surface.  Format primitives come
from :mod:`saklas.io.manifold_folder`; cross-model transfer and lifecycle
live in :mod:`saklas.io.manifold_lifecycle` (not corpus authoring).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Iterator, Optional

from saklas.io.atomic import write_json_atomic
from saklas.io.manifold_folder import (
    _FIT_MODES_DISCOVER,
    _LABEL_REGEX,
    MANIFOLD_FORMAT_VERSION,
    DiscoverGenerationPlan,
    ManifoldFolder,
    ManifoldFormatError,
    _canonical_json,
    _node_filename,
    _node_payload_authored,
    _node_payload_discover,
    reset_manifold_folder,
    sanitize_hyperparams,
    _validate_node_kind,
    _validate_node_role,
    min_nodes,
)
from saklas.core.manifold import domain_from_spec
from saklas.io.packs import NAME_REGEX
from saklas.io.paths import manifold_dir, manifolds_dir


def _lock_namespace_name(func: Any) -> Any:
    """Serialize a mutation whose first arguments are namespace and name."""
    signature = inspect.signature(func)
    first, second = tuple(signature.parameters)[:2]

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from saklas.io.manifold_folder import _locked_manifest

        bound = signature.bind(*args, **kwargs)
        namespace = str(bound.arguments[first])
        name = str(bound.arguments[second])
        with _locked_manifest(manifold_dir(namespace, name)):
            return func(*args, **kwargs)
    return wrapped


def _lock_folder_arg(func: Any) -> Any:
    """Serialize a mutation whose first argument is a manifold folder."""
    signature = inspect.signature(func)
    first = next(iter(signature.parameters))

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from saklas.io.manifold_folder import _locked_manifest

        bound = signature.bind(*args, **kwargs)
        folder = Path(bound.arguments[first])
        with _locked_manifest(Path(folder)):
            return func(*args, **kwargs)
    return wrapped


# ===================================================== discovery + authoring ===
#
# The functions below are the shared backend for `saklas manifold`
# (CLI) and the `/saklas/v1/manifolds` HTTP routes — folder discovery and
# the create/update authoring path live here in `io` so neither `cli` nor
# `server` re-implements the on-disk format, and `server` need not import
# `cli`.


def iter_manifold_folders(
    namespace: Optional[str] = None,
) -> Iterator[tuple[str, "ManifoldFolder"]]:
    """Yield ``(namespace, ManifoldFolder)`` for every installed manifold.

    Walks ``~/.saklas/manifolds/<ns>/<name>/``; malformed folders are
    skipped rather than raising, so one bad manifold does not break a
    listing.  Discovery is metadata-only: fitted payload hashes are verified
    at explicit use/publish/install boundaries, not while routing selectors or
    rendering inventories.  Optionally filtered to a single ``namespace``.
    """
    root = manifolds_dir()
    if not root.exists():
        return
    for ns_dir in sorted(root.iterdir()):
        if not ns_dir.is_dir():
            continue
        if namespace is not None and ns_dir.name != namespace:
            continue
        for mdir in sorted(ns_dir.iterdir()):
            if not (mdir / "manifold.json").exists():
                continue
            try:
                yield ns_dir.name, ManifoldFolder.load(
                    mdir, verify_manifest=False,
                )
            except ManifoldFormatError:
                continue


#: Namespaces no manifold may be authored under. ``jlens`` and ``sae`` are
#: lazily-resolved per-model steering tiers; a manifold folder under either
#: name would shadow that resolution.
RESERVED_NAMESPACES = frozenset({"jlens", "sae"})


def _validate_ns_name(namespace: str, name: str) -> None:
    """Shared name/namespace gate for every folder-creation entry point."""
    if not NAME_REGEX.match(name):
        raise ManifoldFormatError(
            f"manifold name {name!r} invalid; must match {NAME_REGEX.pattern}"
        )
    if not NAME_REGEX.match(namespace):
        raise ManifoldFormatError(
            f"manifold namespace {namespace!r} invalid; "
            f"must match {NAME_REGEX.pattern}"
        )
    if namespace in RESERVED_NAMESPACES:
        raise ManifoldFormatError(
            f"manifold namespace {namespace!r} is reserved — "
            f"'jlens/<word>' resolves through the model's Jacobian lens, "
            f"not a manifold folder"
        )


def _validate_authored_nodes(name: str, domain: Any, nodes: Any) -> None:
    """Validate a webui/CLI-authored node list against the domain.

    ``nodes`` is ``[{label, coords, statements}, ...]`` — the authoring
    shape, statements inline (on disk they live in ``nodes/*.json``).
    Raises :class:`ManifoldFormatError` on any problem so the caller
    surfaces a clean 400 instead of a half-written folder.
    """
    n = domain.intrinsic_dim
    floor = min_nodes(n)
    if not isinstance(nodes, list) or len(nodes) < floor:
        raise ManifoldFormatError(
            f"manifold {name!r} ({n}-D domain) needs >= {floor} nodes, "
            f"got {len(nodes) if isinstance(nodes, list) else nodes!r}"
        )
    labels: list[str] = []
    for entry in nodes:
        if not isinstance(entry, dict):
            raise ManifoldFormatError(
                f"manifold {name!r} node {entry!r} must be an object"
            )
        label = entry.get("label")
        if not isinstance(label, str) or not _LABEL_REGEX.match(label):
            raise ManifoldFormatError(
                f"manifold {name!r} node label {label!r} invalid; "
                f"a node label is a grammar-addressable identifier "
                f"(no '.', reserved as the bipolar separator) — "
                f"must match {_LABEL_REGEX.pattern}"
            )
        coords = entry.get("coords")
        if (
            not isinstance(coords, list)
            or len(coords) != n
            or not all(isinstance(c, (int, float)) for c in coords)
        ):
            raise ManifoldFormatError(
                f"manifold {name!r} node {label!r} needs 'coords' of "
                f"{n} number(s), got {coords!r}"
            )
        statements = entry.get("statements")
        if (
            not isinstance(statements, list)
            or not statements
            or not all(isinstance(s, str) and s.strip() for s in statements)
        ):
            raise ManifoldFormatError(
                f"manifold {name!r} node {label!r} needs a non-empty list "
                f"of non-blank statement strings"
            )
        # ``role`` is optional; validate the slug shape when set, no-op
        # otherwise.  Family-unsupported is a fit-time concern.
        _validate_node_role(name, label, entry.get("role"))
        # ``kind`` is optional; validate the enum when set, no-op otherwise.
        _validate_node_kind(name, label, entry.get("kind"))
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ManifoldFormatError(f"manifold {name!r} has duplicate node labels")


def _write_node_corpus(folder: Path, nodes: list[dict[str, Any]]) -> None:
    """(Re)write the ``nodes/`` corpus from an authored node list.

    Staged: the new corpus is written to ``nodes.staging/`` in full,
    then swapped in.  A write or IO error mid-corpus therefore leaves
    the existing ``nodes/`` untouched rather than half-rewritten — the
    destructive window shrinks to one ``rmtree`` + one ``rename``.
    """
    nodes_dir = folder / "nodes"
    staging = folder / "nodes.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for idx, entry in enumerate(nodes):
        write_json_atomic(
            staging / _node_filename(idx, entry["label"]),
            [str(s) for s in entry["statements"]],
        )
    if nodes_dir.exists():
        shutil.rmtree(nodes_dir)
    staging.rename(nodes_dir)


def _load_with_advisories(folder: Path) -> tuple["ManifoldFolder", list[str]]:
    """Load a manifold folder, returning it plus any authoring-quality warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mf = ManifoldFolder.load(folder, verify_manifest=False)
    return mf, [str(w.message) for w in caught]


@_lock_namespace_name
def create_manifold_folder(
    namespace: str,
    name: str,
    description: str,
    domain_spec: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> tuple[Path, list[str]]:
    """Author a fresh manifold artifact folder on disk.

    ``domain_spec`` is the ``manifold.json`` ``domain`` tagged union;
    ``nodes`` is ``[{label, coords, statements}, ...]``.  Writes
    ``manifold.json`` (with an empty ``files`` manifest — there are no
    fitted tensors yet; ``fit`` back-fills it) and the ``nodes/`` corpus.
    Returns ``(folder, advisories)`` where ``advisories`` are the soft
    poisedness/flat-axis warnings so the UI can flag a deficient layout
    before a fit is paid for.

    Raises :class:`ManifoldFormatError` on any validation failure and
    :class:`FileExistsError` when a manifold already lives at the path.
    """
    _validate_ns_name(namespace, name)
    try:
        domain = domain_from_spec(domain_spec)
    except (ValueError, KeyError) as e:
        raise ManifoldFormatError(f"invalid manifold domain: {e}") from e
    _validate_authored_nodes(name, domain, nodes)

    folder = manifold_dir(namespace, name)
    if (folder / "manifold.json").exists():
        raise FileExistsError(f"manifold {namespace}/{name} already exists")

    folder.mkdir(parents=True, exist_ok=True)
    _write_node_corpus(folder, nodes)
    payload: dict[str, Any] = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": description,
        "domain": domain.to_spec(),
        "nodes": [
            _node_payload_authored(
                entry["label"], entry["coords"], entry.get("role"),
                entry.get("kind"),
            )
            for entry in nodes
        ],
        "files": {},
    }
    write_json_atomic(folder / "manifold.json", payload)

    _, advisories = _load_with_advisories(folder)
    return folder, advisories


def _validate_discover_corpora(name: str, node_corpora: object) -> None:
    """Validate a discover-mode node corpus dict.

    ``node_corpora`` is ``{label: [statement, ...]}`` — the authoring
    shape for discover mode, where coords are derived at fit time and
    the user only supplies labels + statements.  Labels must match
    :data:`_LABEL_REGEX` (the grammar-addressable identifier subset —
    no ``.``) and every statement list must be non-empty strings.
    """
    if not isinstance(node_corpora, dict) or not node_corpora:
        raise ManifoldFormatError(
            f"discover manifold {name!r} needs a non-empty "
            f"{{label: [statements]}} dict"
        )
    for label, statements in node_corpora.items():
        if not isinstance(label, str) or not _LABEL_REGEX.match(label):
            raise ManifoldFormatError(
                f"discover manifold {name!r} label {label!r} invalid; "
                f"a node label is a grammar-addressable identifier "
                f"(no '.', reserved as the bipolar separator) — "
                f"must match {_LABEL_REGEX.pattern}"
            )
        if (
            not isinstance(statements, list)
            or not statements
            or not all(isinstance(s, str) and s.strip() for s in statements)
        ):
            raise ManifoldFormatError(
                f"discover manifold {name!r} node {label!r} needs a "
                f"non-empty list of non-blank statement strings"
            )


def _validate_discover_labels(name: str, labels: object) -> list[str]:
    """Return validated discover labels, preserving order."""
    if not isinstance(labels, list) or not labels:
        raise ManifoldFormatError(
            f"discover manifold {name!r} needs at least one node label"
        )
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if not isinstance(label, str) or not _LABEL_REGEX.match(label):
            raise ManifoldFormatError(
                f"discover manifold {name!r} label {label!r} invalid; "
                f"a node label is a grammar-addressable identifier "
                f"(no '.', reserved as the bipolar separator) — "
                f"must match {_LABEL_REGEX.pattern}"
            )
        if label in seen:
            raise ManifoldFormatError(
                f"discover manifold {name!r} duplicate node label {label!r}"
            )
        seen.add(label)
        out.append(label)
    return out


@_lock_namespace_name
def create_manifold_from_template(
    namespace: str,
    name: str,
    description: str,
    *,
    template_ref: str,
    fit_mode: str,
    hyperparams: Optional[dict[str, Any]] = None,
    force: bool = False,
) -> Path:
    """Author a discover manifold whose node corpora derive from a template.

    Resolves the standalone template ``template_ref`` (a
    :class:`saklas.io.templates.TemplateFolder` selector), expands its
    ``values × contexts`` into per-value node corpora (the slot-filled assistant
    turns, ``corpus[i]`` aligned to ``contexts[i]``), and writes a discover
    folder that stores **both** the derived corpus (``nodes/``, like any discover
    folder) and the ``template_ref`` — so a later ``manifold fit`` resolves the
    template's multi-turn contexts as the per-node elicitation prefixes. The
    template is the authoring source of truth; the corpus is its materialization.
    Mirrors the ``generate → fit`` split: fit it next with
    ``saklas manifold fit <name> -m MODEL``.
    """
    from saklas.io.templates import resolve_template

    _validate_ns_name(namespace, name)
    if fit_mode not in _FIT_MODES_DISCOVER:
        raise ManifoldFormatError(
            f"discover manifold {name!r} fit_mode {fit_mode!r} invalid; "
            f"expected one of {sorted(_FIT_MODES_DISCOVER)}"
        )
    tmpl = resolve_template(template_ref)
    node_corpora = tmpl.node_corpora()
    _validate_discover_labels(name, list(node_corpora))
    sanitized_hyperparams = sanitize_hyperparams(fit_mode, hyperparams)
    ref = f"{tmpl.path.parent.name}/{tmpl.name}" if tmpl.path else tmpl.name
    folder = manifold_dir(namespace, name)
    if folder.exists():
        if force:
            reset_manifold_folder(folder)
        elif not (folder / "manifold.json").exists():
            raise FileExistsError(
                f"manifold {namespace}/{name} has an incomplete folder; "
                "pass force=True to replace it"
            )
    return create_discover_manifold_folder(
        namespace, name, description,
        fit_mode=fit_mode,
        node_corpora=node_corpora,
        hyperparams=sanitized_hyperparams,
        template_ref=ref,
    )


@_lock_namespace_name
def create_discover_manifold_folder(
    namespace: str,
    name: str,
    description: str,
    *,
    fit_mode: str,
    node_corpora: dict[str, list[str]],
    hyperparams: Optional[dict[str, Any]] = None,
    node_roles: Optional[dict[str, str | None]] = None,
    node_kinds: Optional[dict[str, str | None]] = None,
    template_ref: Optional[str] = None,
) -> Path:
    """Author a fresh discover-mode manifold artifact folder on disk.

    ``fit_mode`` is one of ``"pca"`` / ``"spectral"``;
    ``node_corpora`` is the authoring shape ``{label: [statement, ...]}``.
    Writes ``manifold.json`` (no ``domain``, no per-node ``coords``,
    empty ``files`` manifest) and the ``nodes/`` corpus.  Returns the
    folder path.

    Current A2 conversational corpora use the bundled global baseline prompts,
    not per-manifold scenario lists.

    Coords are derived per-model at fit time
    (:func:`saklas.core.manifold.discover_coords` runs over the per-node
    centroids), so authoring quality cannot be advised here the way it
    is for authored manifolds — the spectral connectivity check and PCA
    variance-floor diagnostics surface only on the fit itself.

    Raises :class:`ManifoldFormatError` on any validation failure and
    :class:`FileExistsError` when a manifold already lives at the path.

    Cross-model transfer of a fitted layout is handled by
    :func:`saklas.io.manifold_lifecycle.transfer_manifold` — it reuses the
    same per-layer Procrustes map (:mod:`saklas.io.alignment`) the
    ``vector transfer`` path builds and writes a ``_from-<safe_src>``
    variant tensor, mirroring how transferred steering vectors land.
    """
    return _create_discover_manifold_folder(
        namespace, name, description,
        fit_mode=fit_mode,
        node_corpora=node_corpora,
        hyperparams=hyperparams,
        node_roles=node_roles,
        node_kinds=node_kinds,
        template_ref=template_ref,
    )


def _create_discover_manifold_folder(
    namespace: str,
    name: str,
    description: str,
    *,
    fit_mode: str,
    node_corpora: dict[str, list[str]],
    hyperparams: Optional[dict[str, Any]] = None,
    node_roles: Optional[dict[str, str | None]] = None,
    node_kinds: Optional[dict[str, str | None]] = None,
    template_ref: Optional[str] = None,
    target_folder: Path | None = None,
) -> Path:
    """Internal discover writer."""
    _validate_ns_name(namespace, name)
    if fit_mode not in _FIT_MODES_DISCOVER:
        raise ManifoldFormatError(
            f"discover manifold {name!r} fit_mode {fit_mode!r} invalid; "
            f"expected one of {sorted(_FIT_MODES_DISCOVER)}"
        )
    _validate_discover_corpora(name, node_corpora)
    # Validate the optional ``node_roles`` mapping up front, before any
    # corpus is written — keeps the failure mode consistent (no
    # half-built folder on a bad role slug).  Extra keys outside the
    # corpus labels raise; missing keys default to ``None``.
    roles_resolved: dict[str, str | None] = dict.fromkeys(node_corpora)
    if node_roles is not None:
        unknown = set(node_roles) - set(node_corpora)
        if unknown:
            raise ManifoldFormatError(
                f"discover manifold {name!r} node_roles carries labels "
                f"not in node_corpora: {sorted(unknown)}"
            )
        for label, role in node_roles.items():
            roles_resolved[label] = _validate_node_role(name, label, role)
    kinds_resolved: dict[str, str | None] = dict.fromkeys(node_corpora)
    if node_kinds is not None:
        unknown_k = set(node_kinds) - set(node_corpora)
        if unknown_k:
            raise ManifoldFormatError(
                f"discover manifold {name!r} node_kinds carries labels "
                f"not in node_corpora: {sorted(unknown_k)}"
            )
        for label, kind in node_kinds.items():
            kinds_resolved[label] = _validate_node_kind(name, label, kind)

    folder = (
        Path(target_folder) if target_folder is not None
        else manifold_dir(namespace, name)
    )
    if (folder / "manifold.json").exists():
        raise FileExistsError(f"manifold {namespace}/{name} already exists")

    folder.mkdir(parents=True, exist_ok=True)
    # ``_write_node_corpus`` takes the authored-shape list; convert.
    authored_shape = [
        {"label": label, "statements": statements}
        for label, statements in node_corpora.items()
    ]
    _write_node_corpus(folder, authored_shape)
    # Drop cross-method hyperparam keys at the IO boundary so a stray
    # ``k_nn`` on a PCA fit (or a stray ``var_threshold`` on a spectral
    # fit) never lands in the manifest.  Otherwise the dispatcher would
    # raise ``TypeError`` at fit time and the folder would be unusable
    # until manually edited.
    payload: dict[str, Any] = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": description,
        "fit_mode": fit_mode,
        "hyperparams": sanitize_hyperparams(fit_mode, hyperparams),
        "nodes": [
            _node_payload_discover(label, roles_resolved[label], kinds_resolved[label])
            for label in node_corpora
        ],
        "files": {},
    }
    if template_ref is not None:
        payload["template_ref"] = template_ref
    write_json_atomic(folder / "manifold.json", payload)
    return folder


@_lock_namespace_name
def create_baked_manifold_folder(
    namespace: str,
    name: str,
    description: str,
    manifold: "Any",
    model_id: str,
    *,
    method: str,
    tags: Optional[list[str]] = None,
    source: str = "local",
    force: bool = False,
    components: Optional[dict[str, Any]] = None,
    model_fingerprint: str,
    model_id_is_safe: bool = False,
) -> tuple[Path, "ManifoldFolder"]:
    """Persist a corpus-less pre-baked Manifold as a ``fit_mode="baked"`` folder.

    The single producer for corpus-less directions — ``merge`` outputs and
    imported control vectors.  ``manifold`` is an already-fitted
    :class:`saklas.core.manifold.Manifold` (typically the affine ``R = 1`` ray
    from :func:`saklas.core.vectors.fold_directions_to_subspace`); its geometry
    is frozen into the per-model ``<safe_model>.safetensors`` tensor, the folder
    carries no ``nodes/`` corpus, and it can never re-fit.

    Writes ``manifold.json`` (``fit_mode="baked"``, the manifold's display
    ``domain``, label-only nodes) + the per-model tensor + sidecar, then
    back-fills the ``files`` integrity manifest.  ``method`` is the sidecar
    provenance tag (``"merge"`` / ``"imported"``); ``components`` rides the
    sidecar as merge provenance (``merge`` only).  Returns
    ``(folder, ManifoldFolder)``.

    For a multi-model merge, call this once for the first model and
    :func:`save_baked_manifold_tensor` for each subsequent model (sharing the
    one ``manifold.json``); each call publishes that pair's ``files`` proof
    incrementally, so there is no final scan/rewrite over historical tensors.

    Raises :class:`ManifoldFormatError` on an invalid name/namespace and
    :class:`FileExistsError` when a manifold already lives at the path and
    ``force`` is ``False``.
    """
    _validate_ns_name(namespace, name)
    if not model_fingerprint:
        raise ValueError("baked manifold tensors require a proven model fingerprint")
    from saklas.io.paths import tensor_filename

    # Validate the destination filename before a force reset can remove the
    # prior corpus-less geometry.
    tensor_filename(model_id, model_id_is_safe=model_id_is_safe)

    labels = list(manifold.node_labels)
    payload: dict[str, Any] = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": description,
        "fit_mode": "baked",
        "domain": manifold.domain.to_spec(),
        "nodes": [_node_payload_discover(label, None) for label in labels],
        "files": {},
    }
    if tags:
        payload["tags"] = [str(t) for t in tags]
    if source and source != "local":
        payload["source"] = source

    folder = manifold_dir(namespace, name)
    manifest_path = folder / "manifold.json"
    if folder.exists() and not manifest_path.exists():
        if force:
            reset_manifold_folder(folder)
        else:
            raise FileExistsError(
                f"manifold {namespace}/{name} has an incomplete folder; "
                "pass force=True to replace it"
            )
    if manifest_path.exists():
        if force:
            reset_manifold_folder(folder)
        elif not _recoverable_baked_first_publication(
            folder, payload, model_id, model_id_is_safe=model_id_is_safe,
        ):
            raise FileExistsError(f"manifold {namespace}/{name} already exists")
    if not manifest_path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        write_json_atomic(manifest_path, payload)

    save_baked_manifold_tensor(
        folder, manifold, model_id, method=method, components=components,
        model_fingerprint=model_fingerprint,
        model_id_is_safe=model_id_is_safe,
    )

    # Publication already proved the newly written pair in ``files``.  The
    # returned object is metadata-only; do not immediately reread every fitted
    # payload merely to reconstruct it.
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    return folder, mf


def _recoverable_baked_first_publication(
    folder: Path,
    expected_manifest: dict[str, Any],
    model_id: str,
    *,
    model_id_is_safe: bool,
) -> bool:
    """Whether ``folder`` is the interrupted first write of this baked target."""

    from saklas.io.manifold_folder import manifold_folder_tensor_paths
    from saklas.io.packs import verify_integrity
    from saklas.io.paths import parse_tensor_filename, tensor_filename

    try:
        payload = json.loads((folder / "manifold.json").read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    # Transaction-only fields are allowed to differ; the authoring identity must
    # otherwise be exactly the skeleton this request would have created.
    actual_identity = {
        key: value for key, value in payload.items()
        if key not in {"files", "artifact_id", "fit_epochs"}
    }
    expected_identity = {
        key: value for key, value in expected_manifest.items()
        if key not in {"files", "artifact_id", "fit_epochs"}
    }
    if actual_identity != expected_identity:
        return False

    target = folder / tensor_filename(
        model_id, model_id_is_safe=model_id_is_safe,
    )
    if any(path != target for path in manifold_folder_tensor_paths(folder)):
        return False
    pair_names = {target.name, target.with_suffix(".json").name}
    files = payload.get("files", {})
    if not isinstance(files, dict):
        return False
    expected = {name: files[name] for name in pair_names if name in files}
    if len(expected) == 2 and verify_integrity(folder, expected)[0]:
        return False  # a complete, trusted artifact preserves exists semantics
    # Do not adopt an interrupted skeleton that also carries some other model's
    # fitted pair; that is a multi-model merge concern with stronger provenance
    # checks in ``io.merge``.
    for filename in files:
        candidate = (
            Path(filename).with_suffix(".safetensors").name
            if filename.endswith(".json") else filename
        )
        if parse_tensor_filename(candidate) is not None and filename not in pair_names:
            return False
    return True


@_lock_folder_arg
def save_baked_manifold_tensor(
    folder: Path,
    manifold: "Any",
    model_id: str,
    *,
    method: str,
    components: Optional[dict[str, Any]] = None,
    model_fingerprint: str,
    model_id_is_safe: bool = False,
) -> Path:
    """Write one per-model tensor + sidecar into a ``fit_mode="baked"`` folder.

    Factored out of :func:`create_baked_manifold_folder` so a multi-model merge
    can share one ``manifold.json`` across every target model's tensor.  The
    caller is responsible for the surrounding ``manifold.json`` write once.  Pair
    publication and its manifest proof happen in this same folder transaction;
    an interruption between them is repaired by overwriting/adopting the target
    on the next producer retry.  Returns the tensor path.
    """
    from saklas.core.manifold import save_manifold
    from saklas.io.paths import tensor_filename

    # ``nodes_sha256`` is provenance-only for baked (never re-fits), computed
    # from the same identity ``ManifoldFolder.nodes_sha256`` hashes.
    nodes_sha = hashlib.sha256(
        _canonical_json(
            {"fit_mode": "baked", "node_labels": list(manifold.node_labels)}
        )
    ).hexdigest()
    save_meta: dict[str, object] = {
        "method": method,
        "fit_mode": "baked",
        "nodes_sha256": nodes_sha,
    }
    if not model_fingerprint:
        raise ValueError("baked manifold tensors require a proven model fingerprint")
    save_meta["model_fingerprint"] = model_fingerprint
    share_metric = manifold.metadata.get("share_metric")
    if share_metric:
        save_meta["share_metric"] = share_metric
    if components:
        save_meta["components"] = components
    if method == "merge":
        from saklas.io.manifold_folder import MERGE_BAKE_POLICY

        save_meta["bake_policy"] = MERGE_BAKE_POLICY
    tensor_path = folder / tensor_filename(
        model_id, model_id_is_safe=model_id_is_safe,
    )
    save_manifold(manifold, tensor_path, save_meta)
    ManifoldFolder.load(
        folder, verify_manifest=False,
    ).update_file_hashes(tensor_path, tensor_path.with_suffix(".json"))
    return tensor_path


@_lock_folder_arg
def write_manifold_scenarios(folder: Path, scenarios: list[str]) -> None:
    """Persist the shared scenario list to ``<folder>/scenarios.json``.

    Discover-mode generation provenance — the domains the node corpora
    were generated against — mirroring the ``{"scenarios": [...]}``
    schema the discover pipeline writes.  The all-at-once
    :func:`create_discover_manifold_folder` (via its ``scenarios`` kwarg)
    routes through here; under 4.0 conversational generation the shared
    baseline prompts are global, so generation no longer writes per-manifold
    scenarios.
    """
    write_json_atomic(
        folder / "scenarios.json",
        {"scenarios": [str(s) for s in scenarios]},
    )


@_lock_namespace_name
def init_discover_manifold_folder(
    namespace: str,
    name: str,
    description: str,
    *,
    fit_mode: str,
    labels: list[str],
    hyperparams: Optional[dict[str, Any]] = None,
    node_roles: Optional[dict[str, str | None]] = None,
    node_kinds: Optional[dict[str, str | None]] = None,
) -> Path:
    """Create a discover-mode skeleton for *streaming* node writes.

    Writes ``manifold.json`` (label-only nodes, empty ``files``
    manifest) and an empty ``nodes/`` dir, then returns the folder.
    Node corpora are written incrementally via
    :func:`append_discover_manifold_node` (one file per completed node)
    and the scenario provenance via :func:`write_manifold_scenarios` —
    the streaming companion to :func:`create_discover_manifold_folder`
    for big-roster generation, where holding every corpus in memory at
    once is wasteful and a crash should keep the nodes already finished.

    ``labels`` is the full node label list in on-disk order; each must
    be unique and match :data:`_LABEL_REGEX`.  Raises
    :class:`ManifoldFormatError` on a bad name / namespace / fit_mode /
    label / role, and :class:`FileExistsError` when a manifold already
    lives at the path.
    """
    _validate_ns_name(namespace, name)
    if fit_mode not in _FIT_MODES_DISCOVER:
        raise ManifoldFormatError(
            f"discover manifold {name!r} fit_mode {fit_mode!r} invalid; "
            f"expected one of {sorted(_FIT_MODES_DISCOVER)}"
        )
    labels = _validate_discover_labels(name, labels)

    roles_resolved: dict[str, str | None] = dict.fromkeys(labels)
    if node_roles is not None:
        unknown = set(node_roles) - set(labels)
        if unknown:
            raise ManifoldFormatError(
                f"discover manifold {name!r} node_roles carries labels "
                f"not in the roster: {sorted(unknown)}"
            )
        for label, role in node_roles.items():
            roles_resolved[label] = _validate_node_role(name, label, role)
    kinds_resolved: dict[str, str | None] = dict.fromkeys(labels)
    if node_kinds is not None:
        unknown_k = set(node_kinds) - set(labels)
        if unknown_k:
            raise ManifoldFormatError(
                f"discover manifold {name!r} node_kinds carries labels "
                f"not in the roster: {sorted(unknown_k)}"
            )
        for label, kind in node_kinds.items():
            kinds_resolved[label] = _validate_node_kind(name, label, kind)

    folder = manifold_dir(namespace, name)
    if (folder / "manifold.json").exists():
        raise FileExistsError(f"manifold {namespace}/{name} already exists")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "nodes").mkdir(exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": description,
        "fit_mode": fit_mode,
        "hyperparams": sanitize_hyperparams(fit_mode, hyperparams),
        "nodes": [
            _node_payload_discover(label, roles_resolved[label], kinds_resolved[label])
            for label in labels
        ],
        "files": {},
    }
    write_json_atomic(folder / "manifold.json", payload)
    return folder


@_lock_folder_arg
def append_discover_manifold_node(
    folder: Path, index: int, label: str, statements: list[str],
) -> None:
    """Write one discover-mode node corpus to ``nodes/NN_<label>.json``.

    Streaming companion to :func:`init_discover_manifold_folder`:
    ``index`` is the node's position in the skeleton's label order, so
    the ``NN_`` filename prefix matches ``manifold.json``.  Each file is
    written atomically; unlike :func:`_write_node_corpus` there is no
    whole-corpus staging swap — that is the point, so a run that dies
    part-way keeps the nodes it already finished.
    """
    label_in: Any = label
    if not isinstance(label_in, str) or not _LABEL_REGEX.match(label_in):
        raise ManifoldFormatError(
            f"discover manifold node label {label!r} invalid; "
            f"must match {_LABEL_REGEX.pattern}"
        )
    statements_in: Any = statements
    if (
        not isinstance(statements_in, list)
        or not statements_in
        or not all(isinstance(s, str) and s.strip() for s in statements_in)
    ):
        raise ManifoldFormatError(
            f"discover manifold node {label!r} needs a non-empty list "
            f"of non-blank statement strings"
        )
    nodes_dir = folder / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        nodes_dir / _node_filename(index, label),
        [str(s) for s in statements],
    )


def read_manifold_scenarios(folder: Path) -> list[str] | None:
    """Return the persisted shared scenario list, or ``None`` if absent.

    Reads the ``{"scenarios": [...]}`` provenance file
    :func:`write_manifold_scenarios` writes; tolerates the richer
    CLI/server shape (extra keys alongside ``"scenarios"``).  Returns
    ``None`` when there is no ``scenarios.json`` or it carries no list —
    the signal a streaming run uses to lock resumed/added nodes onto the
    original domains.
    """
    p = Path(folder) / "scenarios.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    scn = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(scn, list):
        return None
    return [str(s) for s in scn]


def _discover_manifest_payload(
    name: str,
    description: str,
    fit_mode: str,
    labels: list[str],
    roles: dict[str, str | None],
    hyperparams: Optional[dict[str, Any]],
    kinds: Optional[dict[str, str | None]] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build the label-only discover ``manifold.json`` dict."""
    kinds = kinds or {}
    payload: dict[str, Any] = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": description,
        "fit_mode": fit_mode,
        "hyperparams": sanitize_hyperparams(fit_mode, hyperparams),
        "nodes": [
            _node_payload_discover(label, roles.get(label), kinds.get(label))
            for label in labels
        ],
        "files": {},
    }
    # Category tags ride manifold.json; written only when non-empty so a
    # tagless manifold stays byte-identical to the pre-tags shape.
    if tags:
        payload["tags"] = [str(t) for t in tags]
    return payload


@_lock_folder_arg
def plan_discover_generation(
    folder: Path,
    name: str,
    description: str,
    *,
    fit_mode: str,
    labels: list[str],
    hyperparams: Optional[dict[str, Any]] = None,
    node_roles: Optional[dict[str, str | None]] = None,
    node_kinds: Optional[dict[str, str | None]] = None,
    tags: Optional[list[str]] = None,
    force: bool = False,
) -> DiscoverGenerationPlan:
    """Ensure a streamable discover skeleton at ``folder`` covering every
    label in ``labels``, and report which node corpora still need writing.

    Resume + add-nodes in one — the single planner every discover-generate
    surface (the bundled regen scripts, ``manifold generate``, the
    HTTP generate route) calls:

    - A fresh ``folder`` gets a label-only skeleton (`manifold.json` +
      empty ``nodes/``) and every label is ``pending``.
    - An existing discover folder keeps its node order, **appends** any
      labels new to it (the add-nodes case — the existing fit goes stale
      on the next read since the corpus hash changes), and reports as
      ``pending`` the declared labels whose ``nodes/NN_<label>.json`` is
      absent — so a run killed half-way resumes the missing nodes instead
      of starting over.
    - ``scenarios.json`` (when present) is read back into
      ``plan.scenarios`` so the caller can lock the resumed/added nodes
      onto the original domains rather than regenerating fresh ones.
    - ``force=True`` removes the prior folder and publishes the fresh skeleton
      in the same manifest transaction.

    The read is deliberately lenient — it does **not** route through
    :meth:`ManifoldFolder.load`, which rejects a partially-written folder
    by design (the missing-node-corpus guard) — so the planner can
    inspect the very partial it is resuming.  Description (and ``tags``,
    when supplied) is refreshed to the caller's value; fit hyperparams on
    an existing folder are kept (a resume fills corpus, it does not silently
    re-spec the fit — use ``--force`` / a fresh folder to change those).

    Raises :class:`ManifoldFormatError` on a non-discover ``fit_mode``, a
    bad/duplicate label, a ``node_roles`` key outside the roster, or an
    existing folder whose ``manifold.json`` is not discover-mode.
    """
    folder = Path(folder)
    if fit_mode not in _FIT_MODES_DISCOVER:
        raise ManifoldFormatError(
            f"discover manifold {name!r} fit_mode {fit_mode!r} invalid; "
            f"expected one of {sorted(_FIT_MODES_DISCOVER)}"
        )
    labels = _validate_discover_labels(name, labels)
    roles_in = node_roles or {}
    unknown = set(roles_in) - set(labels)
    if unknown:
        raise ManifoldFormatError(
            f"discover manifold {name!r} node_roles carries labels not in "
            f"the roster: {sorted(unknown)}"
        )
    kinds_in = node_kinds or {}
    unknown_k = set(kinds_in) - set(labels)
    if unknown_k:
        raise ManifoldFormatError(
            f"discover manifold {name!r} node_kinds carries labels not in "
            f"the roster: {sorted(unknown_k)}"
        )
    # Validate every replacement input before a force reset can remove the
    # prior folder.  The resolved maps are also reused by the fresh path.
    roles_resolved = {
        label: _validate_node_role(name, label, roles_in.get(label))
        for label in labels
    }
    kinds_resolved = {
        label: _validate_node_kind(name, label, kinds_in.get(label))
        for label in labels
    }
    sanitized_hyperparams = sanitize_hyperparams(fit_mode, hyperparams)

    # Reset and skeleton publication are one manifest transaction.  Keeping
    # this inside the planner prevents a fitter from snapshotting between an
    # external ``rmtree`` and the replacement manifest, and prevents a stale
    # fit from publishing across a concurrent force-regeneration.  Remove even
    # a partial folder: force is the recovery path for a missing manifest.
    if force and folder.exists():
        reset_manifold_folder(folder)

    meta_path = folder / "manifold.json"
    nodes_dir = folder / "nodes"

    if not meta_path.exists():
        # Fresh skeleton — everything is pending.
        folder.mkdir(parents=True, exist_ok=True)
        nodes_dir.mkdir(exist_ok=True)
        write_json_atomic(
            meta_path,
            _discover_manifest_payload(
                name, description, fit_mode, labels, roles_resolved,
                sanitized_hyperparams,
                kinds_resolved, tags,
            ),
        )
        return DiscoverGenerationPlan(
            folder=folder,
            index_of={label: i for i, label in enumerate(labels)},
            pending=tuple(labels),
            scenarios=None,
            added=(),
            resumed=False,
        )

    # Resume / extend an existing folder.
    try:
        data = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ManifoldFormatError(
            f"manifold.json in {folder} is unreadable: {e}"
        ) from e
    if data.get("fit_mode", "authored") not in _FIT_MODES_DISCOVER:
        raise ManifoldFormatError(
            f"manifold at {folder} is not discover-mode; cannot "
            f"stream-generate into it"
        )
    existing_nodes = data.get("nodes") or []
    existing_labels = [n["label"] for n in existing_nodes]
    existing_roles: dict[str, str | None] = {
        n["label"]: n.get("role") for n in existing_nodes
    }
    existing_kinds: dict[str, str | None] = {
        n["label"]: n.get("kind") for n in existing_nodes
    }
    new_labels = [label for label in labels if label not in existing_labels]
    full_labels = existing_labels + new_labels
    # Category tags refresh to the caller's value when supplied (None leaves
    # whatever the folder already carries); written only when non-empty so a
    # tagless manifold stays byte-identical to the pre-tags shape.
    desired_tags = [str(t) for t in tags] if tags else None
    if new_labels:
        # Add-nodes: append the new labels (validating their roles/kinds) and
        # rewrite manifold.json atomically.  Existing roles/kinds/hyperparams
        # are preserved; description (and tags, when supplied) refresh to the
        # caller's value.
        merged_roles: dict[str, str | None] = dict(existing_roles)
        merged_kinds: dict[str, str | None] = dict(existing_kinds)
        for label in new_labels:
            merged_roles[label] = _validate_node_role(
                name, label, roles_in.get(label),
            )
            merged_kinds[label] = _validate_node_kind(
                name, label, kinds_in.get(label),
            )
        data["description"] = description
        if desired_tags is not None:
            data["tags"] = desired_tags
        data["nodes"] = [
            _node_payload_discover(
                label, merged_roles.get(label), merged_kinds.get(label),
            )
            for label in full_labels
        ]
        write_json_atomic(meta_path, data)
    else:
        changed = False
        if data.get("description") != description:
            data["description"] = description
            changed = True
        if desired_tags is not None and data.get("tags") != desired_tags:
            data["tags"] = desired_tags
            changed = True
        if changed:
            write_json_atomic(meta_path, data)

    nodes_dir.mkdir(exist_ok=True)
    index_of = {label: i for i, label in enumerate(full_labels)}
    pending = tuple(
        label for label in full_labels
        if not (nodes_dir / _node_filename(index_of[label], label)).exists()
    )
    scn = read_manifold_scenarios(folder)
    return DiscoverGenerationPlan(
        folder=folder,
        index_of=index_of,
        pending=pending,
        scenarios=tuple(scn) if scn is not None else None,
        added=tuple(new_labels),
        resumed=True,
    )


@_lock_namespace_name
def merge_discover_manifolds(
    target_namespace: str,
    target_name: str,
    target_description: str,
    *,
    sources: list[tuple[str, str]],
    fit_mode: Optional[str] = None,
    hyperparams: Optional[dict[str, Any]] = None,
    force: bool = False,
) -> Path:
    """Union N discover-mode manifolds' nodes into one fresh discover folder.

    The vector-side counterpart is :func:`saklas.io.merge.merge_into_manifold`,
    but the manifold semantics are different: vector merge composes a
    new direction from a steering expression; manifold merge unions
    *node corpora* and lets the next fit derive coords from the
    combined heap.

    Restricted to discover-mode sources by design — authored manifolds
    carry user-declared geometry (a specific ``BoxDomain`` / ``SphereDomain``
    + per-node coords on that domain), and reconciling two unrelated
    coordinate systems without a shared frame isn't meaningful.  Discover
    folders derive coords per-model from centroids at fit time, so the
    merge is just "pool the centroids" — the operation that's natural
    for the autofit pipeline.

    Source label collisions raise :class:`ManifoldFormatError`; the
    caller should rename in source folders before merging (we don't
    auto-prefix since labels are user-visible and silent renames hide
    provenance).

    Fit-mode reconciliation:
      * sources agree → that's the default
      * sources disagree → an explicit ``fit_mode`` is required
      * caller supplies ``fit_mode`` → that wins

    Hyperparams default to the first source's; an explicit
    ``hyperparams`` arg replaces them wholesale (matches
    ``create_discover_manifold_folder``'s shape).  ``_sanitize_hyperparams``
    drops cross-method keys at the IO boundary.

    The merged folder is written *unfitted* — no per-model tensor is
    materialized.  Run ``saklas manifold discover`` or
    ``POST .../fit`` against the merged folder to derive coords + fit.
    """
    if len(sources) < 2:
        raise ValueError(
            f"merge needs >= 2 sources, got {len(sources)}: {sources!r}",
        )

    folders: list[tuple[str, str, ManifoldFolder]] = []
    for ns, name in sources:
        folder_path = manifold_dir(ns, name)
        if not (folder_path / "manifold.json").exists():
            raise FileNotFoundError(
                f"merge source {ns}/{name} not found at {folder_path}",
            )
        mf = ManifoldFolder.load(folder_path, verify_manifest=False)
        if not mf.is_discover:
            raise ManifoldFormatError(
                f"merge source {ns}/{name} is authored ({mf.fit_mode!r}); "
                f"merge supports discover-mode (autofitted) manifolds only — "
                f"authored manifolds carry user-declared geometry that "
                f"isn't mergeable without a shared coordinate system.",
            )
        folders.append((ns, name, mf))

    # Detect label collisions across the union.  Refuse rather than
    # silently rename — labels carry provenance the user cares about.
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for ns, name, mf in folders:
        src_key = f"{ns}/{name}"
        for label in mf.node_labels:
            if label in seen and seen[label] != src_key:
                collisions.append((label, seen[label], src_key))
            seen[label] = src_key
    if collisions:
        details = "; ".join(
            f"{label!r} in {a} and {b}" for label, a, b in collisions
        )
        raise ManifoldFormatError(
            f"merge {target_namespace}/{target_name}: label collisions — "
            f"{details}. Rename one side before merging.",
        )

    # Reconcile fit_mode.
    source_modes = sorted({mf.fit_mode for _, _, mf in folders})
    if fit_mode is None:
        if len(source_modes) > 1:
            raise ManifoldFormatError(
                f"merge {target_namespace}/{target_name}: sources have "
                f"mixed fit_modes ({source_modes}); pass fit_mode= to pick one.",
            )
        fit_mode = source_modes[0]
    elif fit_mode not in _FIT_MODES_DISCOVER:
        raise ManifoldFormatError(
            f"merge {target_namespace}/{target_name}: fit_mode "
            f"{fit_mode!r} invalid; expected one of "
            f"{sorted(_FIT_MODES_DISCOVER)}.",
        )

    # Reconcile hyperparams — default to the first source's, caller may
    # override wholesale.
    effective_hyperparams: dict[str, Any]
    effective_hyperparams = dict(hyperparams) if hyperparams is not None else dict(folders[0][2].hyperparams)

    # Pool the corpus + roles.  Iteration order: source order, then
    # per-source label order (matches how the source authored them).
    node_corpora: dict[str, list[str]] = {}
    node_roles: dict[str, str | None] = {}
    node_kinds: dict[str, str | None] = {}
    for _ns, _name, mf in folders:
        groups = dict(mf.node_groups())
        roles_padded = mf._roles_padded()
        kinds_padded = mf._kinds_padded()
        for idx, label in enumerate(mf.node_labels):
            node_corpora[label] = list(groups.get(label, []))
            node_roles[label] = roles_padded[idx]
            node_kinds[label] = kinds_padded[idx]

    target_folder = manifold_dir(target_namespace, target_name)
    if target_folder.exists():
        if not force:
            raise FileExistsError(
                f"manifold {target_namespace}/{target_name} already exists; "
                f"pass force=True to overwrite",
            )
        reset_manifold_folder(target_folder)

    return create_discover_manifold_folder(
        target_namespace,
        target_name,
        target_description,
        fit_mode=fit_mode,
        node_corpora=node_corpora,
        hyperparams=effective_hyperparams,
        node_roles=node_roles,
        node_kinds=node_kinds,
    )


@_lock_folder_arg
def update_manifold_folder(
    folder: Path,
    *,
    description: Optional[str] = None,
    nodes: Optional[list[dict[str, Any]]] = None,
) -> tuple[Path, list[str]]:
    """Re-author an existing manifold folder.

    ``nodes``, when given, fully replaces the node list (labels, coords
    and corpus).  Existing fitted tensors are left in place — they become
    stale (``nodes_sha256`` no longer matches) and the next fit overwrites
    them.  Returns ``(folder, advisories)``.
    """
    folder = Path(folder)
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    if description is not None:
        mf.description = description
    if nodes is not None:
        domain = domain_from_spec(mf.domain)
        _validate_authored_nodes(mf.name, domain, nodes)
        _write_node_corpus(folder, nodes)
        mf.node_labels = [entry["label"] for entry in nodes]
        mf.node_coords = [[float(c) for c in entry["coords"]] for entry in nodes]
        # Roles ride per-entry; defaults to None for entries that
        # don't carry the field.  ``_validate_authored_nodes`` already
        # checked the slug shape, so this is a pure copy.
        mf.node_roles = [entry.get("role") for entry in nodes]
        mf.node_kinds = [entry.get("kind") for entry in nodes]
    mf.write_metadata()
    _, advisories = _load_with_advisories(folder)
    return folder, advisories
