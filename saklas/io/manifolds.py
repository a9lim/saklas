"""On-disk format for manifold-steering artifacts.

A *manifold* is a set of labeled nodes — each node a small corpus of
statements — placed at authoring coordinates on a :class:`ManifoldDomain`
(an n-dimensional intrinsic manifold of some topology: a box/disk, a
cylinder, a torus, a sphere, or an explicit immersion).  Fitting a
manifold against a model produces a per-model RBF artifact (see
:mod:`saklas.core.manifold`).  Manifolds live under their own root,
``~/.saklas/manifolds/<ns>/<name>/``, parallel to ``vectors/`` — a
manifold is not a single bipolar concept, so it is not a
:class:`saklas.io.packs.ConceptFolder`.

```
~/.saklas/manifolds/<ns>/<name>/
  manifold.json               # name, description, domain spec, nodes
  nodes/NN_<label>.json       # one zero-padded file per node: a JSON list
                              # of statement strings
  <safe_model>.safetensors    # fitted per-model RBF (+ .json sidecar)
  <safe_model>_sae-<rel>.safetensors   # SAE-feature-space variant
```

The user authors ``manifold.json`` (each node carries a ``label`` and its
authoring ``coords``) and the ``nodes/*.json`` corpus files by hand;
``saklas vector manifold fit`` produces the per-model tensors and
back-fills the ``files`` integrity manifest.  Tensor save/load itself
lives in :mod:`saklas.core.manifold` (``save_manifold`` / ``load_manifold``);
this module owns folder discovery, the node corpus, and integrity.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from saklas.core.errors import SaklasError
from saklas.core.manifold import BoxDomain, domain_from_spec
from saklas.io.atomic import write_json_atomic
from saklas.io.packs import NAME_REGEX, verify_integrity
from saklas.io.paths import manifold_dir, manifolds_dir

# Manifold artifact format version.  Decoupled from concept packs'
# ``PACK_FORMAT_VERSION`` so the two formats can churn independently.
# v3 is the arbitrary-dimensional / arbitrary-topology format (domain
# spec + per-node coordinates); v2 and earlier were the 1-D cyclic-spline
# format and must be converted with ``scripts/upgrade_manifolds.py``.
MANIFOLD_FORMAT_VERSION = 3


class ManifoldFormatError(ValueError, SaklasError):
    """Raised when a manifold folder is malformed or fails integrity."""


def min_nodes(n: int) -> int:
    """Minimum node count for an ``n``-dimensional manifold.

    ``2*n + 1`` — one center node plus two extremes per authoring axis,
    so every axis is sampled at three levels and curvature is fittable.
    At ``n == 1`` this is 3, the historic floor.  This is *necessary* but
    not *sufficient*: the nodes must also be poised (affinely span the
    embedding), which :func:`saklas.core.manifold.fit_rbf_interpolant`
    enforces.
    """
    return 2 * n + 1


def _node_filename(index: int, label: str) -> str:
    """Canonical zero-padded node corpus filename: ``NN_<label>.json``."""
    return f"{index:02d}_{label}.json"


def hash_manifold_files(folder: Path) -> dict[str, str]:
    """Return ``{relative_posix_path: sha256}`` for a manifold's fitted files.

    Covers the top-level fitted artifacts (the per-model ``.safetensors``
    tensors and their ``.json`` sidecars); skips ``manifold.json`` itself
    and the ``nodes/`` corpus.  The node corpus is user-authored and
    *expected* to be edited — editing it is the re-fit trigger, tracked
    by the per-tensor ``nodes_sha256`` sidecar field — so hashing it into
    an integrity manifest would reject the normal workflow.  Used to
    (re)populate ``manifold.json.files`` after a fit.
    """
    out: dict[str, str] = {}
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.name != "manifold.json":
            out[entry.name] = _hash_file(entry)
    return out


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: Any) -> bytes:
    """Stable canonical-JSON encoding for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class ManifoldSidecar:
    """JSON sidecar beside a fitted per-model manifold tensor.

    Lean by design — the concept-extraction fields on
    :class:`saklas.io.packs.Sidecar` (``statements_sha256``,
    ``diagnostics_by_layer`` ...) are meaningless for a manifold, so this
    is a separate type rather than a reuse.
    """

    method: str            # "manifold_pca" | "manifold_sae"
    saklas_version: str
    domain: dict[str, Any]
    node_count: int
    node_labels: list[str]
    feature_space: str = "raw"
    nodes_sha256: Optional[str] = None
    sae_release: Optional[str] = None
    sae_revision: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "ManifoldSidecar":
        with open(path) as f:
            data = json.load(f)
        domain = data.get("domain")
        if not isinstance(domain, dict):
            raise ManifoldFormatError(
                f"manifold sidecar {path} has no 'domain' object"
            )
        return cls(
            method=data.get("method", "manifold_pca"),
            saklas_version=data.get("saklas_version", "0"),
            domain=domain,
            node_count=int(data.get("node_count", 0)),
            node_labels=list(data.get("node_labels", [])),
            feature_space=data.get("feature_space", "raw"),
            nodes_sha256=data.get("nodes_sha256"),
            sae_release=data.get("sae_release"),
            sae_revision=data.get("sae_revision"),
        )


@dataclass
class ManifoldFolder:
    """A manifold artifact folder on disk.

    Discovery + corpus + integrity only — the fitted RBF tensors are
    loaded through :func:`saklas.core.manifold.load_manifold`.
    """

    folder: Path
    name: str
    description: str
    domain: dict[str, Any]
    node_labels: list[str]
    node_coords: list[list[float]]
    files: dict[str, str]
    # tensor stem (``<safe_model>`` or ``<safe_model>_sae-<rel>``) -> sidecar.
    _sidecars: dict[str, ManifoldSidecar] = field(default_factory=dict)

    @classmethod
    def load(cls, folder: Path) -> "ManifoldFolder":
        folder = Path(folder)
        meta_path = folder / "manifold.json"
        if not meta_path.exists():
            raise ManifoldFormatError(f"no manifold.json in {folder}")
        try:
            with open(meta_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # A malformed manifest must surface as ManifoldFormatError so
            # callers (iter_manifold_folders, the HTTP routes) catch it
            # rather than letting a bare JSONDecodeError become a 500.
            raise ManifoldFormatError(
                f"manifold.json in {folder} is unreadable: {e}"
            ) from e

        fmt = data.get("format_version", 1)
        if not isinstance(fmt, int) or fmt < MANIFOLD_FORMAT_VERSION:
            raise ManifoldFormatError(
                f"manifold.json in {folder} has format_version={fmt!r}; "
                f"need >= {MANIFOLD_FORMAT_VERSION}. Pre-v3 manifolds use "
                f"the old 1-D cyclic-spline format — convert them with "
                f"scripts/upgrade_manifolds.py."
            )

        name = data.get("name", folder.name)
        if not isinstance(name, str) or not NAME_REGEX.match(name):
            raise ManifoldFormatError(
                f"manifold name {name!r} invalid; must match {NAME_REGEX.pattern}"
            )

        domain_spec = data.get("domain")
        if not isinstance(domain_spec, dict):
            raise ManifoldFormatError(
                f"manifold {name!r} needs a 'domain' object"
            )
        try:
            domain = domain_from_spec(domain_spec)
        except (ValueError, KeyError) as e:
            raise ManifoldFormatError(
                f"manifold {name!r} has an invalid domain: {e}"
            ) from e
        n = domain.intrinsic_dim

        nodes = data.get("nodes")
        floor = min_nodes(n)
        if not isinstance(nodes, list) or len(nodes) < floor:
            raise ManifoldFormatError(
                f"manifold {name!r} ({n}-D domain) needs a 'nodes' list of "
                f">= {floor} entries, got {nodes!r}"
            )

        node_labels: list[str] = []
        node_coords: list[list[float]] = []
        for entry in nodes:
            if not isinstance(entry, dict):
                raise ManifoldFormatError(
                    f"manifold {name!r} node {entry!r} must be an object "
                    f"with 'label' and 'coords'"
                )
            label = entry.get("label")
            if not isinstance(label, str) or not NAME_REGEX.match(label):
                raise ManifoldFormatError(
                    f"manifold {name!r} node label {label!r} invalid; "
                    f"must match {NAME_REGEX.pattern}"
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
            node_labels.append(label)
            node_coords.append([float(c) for c in coords])
        if len(set(node_labels)) != len(node_labels):
            raise ManifoldFormatError(
                f"manifold {name!r} has duplicate node labels"
            )

        _warn_authoring_quality(name, domain, node_coords)

        files = data.get("files", {})
        if not isinstance(files, dict):
            raise ManifoldFormatError(
                f"manifold {name!r} 'files' must be an object"
            )
        # Verify only a populated manifest — a freshly hand-authored
        # manifold has no hashes yet; `fit` back-fills them.
        if files:
            ok, bad = verify_integrity(folder, files)
            if not ok:
                raise ManifoldFormatError(
                    f"manifold integrity check failed in {folder}: "
                    f"tampered/missing {bad}"
                )

        inst = cls(
            folder=folder,
            name=name,
            description=data.get("description", ""),
            domain=domain_spec,
            node_labels=node_labels,
            node_coords=node_coords,
            files=files,
        )

        # Every node file must be present.
        for idx, label in enumerate(node_labels):
            p = inst.node_path(idx)
            if not p.exists():
                raise ManifoldFormatError(
                    f"manifold {name!r} missing node corpus file {p}"
                )

        for t in sorted(folder.glob("*.safetensors")):
            sc_path = t.with_suffix(".json")
            if sc_path.exists():
                inst._sidecars[t.stem] = ManifoldSidecar.load(sc_path)
        return inst

    # -- node corpus -------------------------------------------------------

    def node_path(self, index: int) -> Path:
        return self.folder / "nodes" / _node_filename(
            index, self.node_labels[index],
        )

    def node_groups(self) -> list[tuple[str, list[str]]]:
        """Return ``[(label, statements), ...]`` in node order."""
        groups: list[tuple[str, list[str]]] = []
        for idx, label in enumerate(self.node_labels):
            with open(self.node_path(idx)) as f:
                statements = json.load(f)
            if not isinstance(statements, list) or not statements:
                raise ManifoldFormatError(
                    f"manifold {self.name!r} node {label!r} must be a "
                    f"non-empty JSON list of statements"
                )
            if not all(isinstance(s, str) for s in statements):
                raise ManifoldFormatError(
                    f"manifold {self.name!r} node {label!r} statements "
                    f"must all be strings"
                )
            groups.append((label, list(statements)))
        return groups

    def nodes_sha256(self) -> str:
        """Stable hash of the ordered node corpus, domain, and node coords.

        The staleness key: a fitted tensor's sidecar records this, and a
        re-fit is needed when it no longer matches.  Folding the domain
        spec and the authoring coordinates into the hash means any
        geometry edit — moving a node, flipping an axis to periodic —
        triggers a re-fit alongside corpus edits.
        """
        h = hashlib.sha256()
        for idx in range(len(self.node_labels)):
            h.update(self.node_path(idx).read_bytes())
        h.update(_canonical_json(self.domain))
        h.update(_canonical_json(self.node_coords))
        return h.hexdigest()

    # -- fitted tensors ----------------------------------------------------

    def tensor_models(self) -> list[str]:
        """Tensor stems present on disk (``<safe_model>[_sae-<rel>]``)."""
        return sorted(self._sidecars)

    def tensor_path(self, stem: str) -> Path:
        return self.folder / f"{stem}.safetensors"

    def sidecar(self, stem: str) -> ManifoldSidecar:
        return self._sidecars[stem]

    # -- manifest ----------------------------------------------------------

    def write_metadata(self, *, files: Optional[dict[str, str]] = None) -> None:
        """Rewrite ``manifold.json``, re-hashing the ``files`` manifest.

        Called by the fit step after writing a new per-model tensor so the
        integrity manifest covers every fitted artifact.
        """
        if files is None:
            files = hash_manifold_files(self.folder)
        self.files = files
        payload: dict[str, Any] = {
            "format_version": MANIFOLD_FORMAT_VERSION,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "nodes": [
                {"label": label, "coords": list(coords)}
                for label, coords in zip(self.node_labels, self.node_coords)
            ],
            "files": files,
        }
        write_json_atomic(self.folder / "manifold.json", payload)


def _warn_authoring_quality(
    name: str, domain: Any, node_coords: list[list[float]],
) -> None:
    """Soft advisory warnings on node placement — never raises.

    Poisedness (the embedded coordinates failing to affinely span the
    embedding) is a hard error at fit time; flagging it here gives the
    author the feedback before they run a fit.  A near-flat non-periodic
    axis (fewer than three distinct values) cannot show curvature.
    """
    coords = torch.tensor(node_coords, dtype=torch.float32)
    embedded = domain.embed(coords)
    centered = embedded - embedded.mean(dim=0, keepdim=True)
    rank = int(torch.linalg.matrix_rank(centered))
    if rank != domain.embed_dim:
        warnings.warn(
            f"manifold {name!r}: node coordinates have affine rank {rank} "
            f"but the embedding is {domain.embed_dim}-dimensional — the fit "
            f"will fail as not poised; spread the nodes across every axis",
            UserWarning,
            stacklevel=3,
        )
    if isinstance(domain, BoxDomain):
        for i, ax in enumerate(domain.axes):
            if ax.periodic:
                continue
            distinct = {round(row[i], 9) for row in node_coords}
            if len(distinct) < 3:
                warnings.warn(
                    f"manifold {name!r}: axis {ax.name!r} has only "
                    f"{len(distinct)} distinct coordinate value(s) — "
                    f"curvature along it cannot be fitted",
                    UserWarning,
                    stacklevel=3,
                )


# ===================================================== discovery + authoring ===
#
# The functions below are the shared backend for `saklas vector manifold`
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
    listing.  Optionally filtered to a single ``namespace``.
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
                yield ns_dir.name, ManifoldFolder.load(mdir)
            except ManifoldFormatError:
                continue


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
        if not isinstance(label, str) or not NAME_REGEX.match(label):
            raise ManifoldFormatError(
                f"manifold {name!r} node label {label!r} invalid; "
                f"must match {NAME_REGEX.pattern}"
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
        mf = ManifoldFolder.load(folder)
    return mf, [str(w.message) for w in caught]


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
    if not NAME_REGEX.match(name):
        raise ManifoldFormatError(
            f"manifold name {name!r} invalid; must match {NAME_REGEX.pattern}"
        )
    if not NAME_REGEX.match(namespace):
        raise ManifoldFormatError(
            f"manifold namespace {namespace!r} invalid; "
            f"must match {NAME_REGEX.pattern}"
        )
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
            {"label": entry["label"], "coords": [float(c) for c in entry["coords"]]}
            for entry in nodes
        ],
        "files": {},
    }
    write_json_atomic(folder / "manifold.json", payload)

    _, advisories = _load_with_advisories(folder)
    return folder, advisories


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
    mf = ManifoldFolder.load(folder)
    if description is not None:
        mf.description = description
    if nodes is not None:
        domain = domain_from_spec(mf.domain)
        _validate_authored_nodes(mf.name, domain, nodes)
        _write_node_corpus(folder, nodes)
        mf.node_labels = [entry["label"] for entry in nodes]
        mf.node_coords = [[float(c) for c in entry["coords"]] for entry in nodes]
    mf.write_metadata()
    _, advisories = _load_with_advisories(folder)
    return folder, advisories


__all__ = [
    "MANIFOLD_FORMAT_VERSION",
    "min_nodes",
    "ManifoldFormatError",
    "ManifoldSidecar",
    "ManifoldFolder",
    "hash_manifold_files",
    "iter_manifold_folders",
    "create_manifold_folder",
    "update_manifold_folder",
]
