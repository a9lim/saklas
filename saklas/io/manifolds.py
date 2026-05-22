"""On-disk format for manifold-steering artifacts.

A *manifold* is an ordered (optionally cyclic) set of labeled nodes, each
node a small corpus of statements.  Fitting a manifold against a model
produces a per-model spline artifact (see :mod:`saklas.core.manifold`).
Manifolds live under their own root, ``~/.saklas/manifolds/<ns>/<name>/``,
parallel to ``vectors/`` — a manifold is not a single bipolar concept, so
it is not a :class:`saklas.io.packs.ConceptFolder`.

```
~/.saklas/manifolds/<ns>/<name>/
  manifold.json               # name, description, cyclic flag, node order
  nodes/NN_<label>.json       # one zero-padded file per node: a JSON list
                              # of statement strings
  <safe_model>.safetensors    # fitted per-model spline (+ .json sidecar)
  <safe_model>_sae-<rel>.safetensors   # SAE-feature-space variant
```

The user authors ``manifold.json`` and the ``nodes/*.json`` corpus files
by hand; ``saklas vector manifold fit`` produces the per-model tensors and
back-fills the ``files`` integrity manifest.  Tensor save/load itself
lives in :mod:`saklas.core.manifold` (``save_manifold`` / ``load_manifold``);
this module owns folder discovery, the node corpus, and integrity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from saklas.core.errors import SaklasError
from saklas.io.atomic import write_json_atomic
from saklas.io.packs import NAME_REGEX, PACK_FORMAT_VERSION, verify_integrity

# Minimum nodes for a manifold to define a curve rather than a chord/point.
MIN_MANIFOLD_NODES = 3


class ManifoldFormatError(ValueError, SaklasError):
    """Raised when a manifold folder is malformed or fails integrity."""


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
    cyclic: bool
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
        return cls(
            method=data.get("method", "manifold_pca"),
            saklas_version=data.get("saklas_version", "0"),
            cyclic=bool(data.get("cyclic", False)),
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

    Discovery + corpus + integrity only — the fitted spline tensors are
    loaded through :func:`saklas.core.manifold.load_manifold`.
    """

    folder: Path
    name: str
    description: str
    cyclic: bool
    node_labels: list[str]
    files: dict[str, str]
    # tensor stem (``<safe_model>`` or ``<safe_model>_sae-<rel>``) -> sidecar.
    _sidecars: dict[str, ManifoldSidecar] = field(default_factory=dict)

    @classmethod
    def load(cls, folder: Path) -> "ManifoldFolder":
        folder = Path(folder)
        meta_path = folder / "manifold.json"
        if not meta_path.exists():
            raise ManifoldFormatError(f"no manifold.json in {folder}")
        with open(meta_path) as f:
            data = json.load(f)

        fmt = data.get("format_version", 1)
        if not isinstance(fmt, int) or fmt < PACK_FORMAT_VERSION:
            raise ManifoldFormatError(
                f"manifold.json in {folder} has format_version={fmt!r}; "
                f"need >= {PACK_FORMAT_VERSION}"
            )

        name = data.get("name", folder.name)
        if not isinstance(name, str) or not NAME_REGEX.match(name):
            raise ManifoldFormatError(
                f"manifold name {name!r} invalid; must match {NAME_REGEX.pattern}"
            )

        nodes = data.get("nodes")
        if not isinstance(nodes, list) or len(nodes) < MIN_MANIFOLD_NODES:
            raise ManifoldFormatError(
                f"manifold {name!r} needs a 'nodes' list of >= "
                f"{MIN_MANIFOLD_NODES} labels, got {nodes!r}"
            )
        node_labels: list[str] = []
        for label in nodes:
            if not isinstance(label, str) or not NAME_REGEX.match(label):
                raise ManifoldFormatError(
                    f"manifold {name!r} node label {label!r} invalid; "
                    f"must match {NAME_REGEX.pattern}"
                )
            node_labels.append(label)
        if len(set(node_labels)) != len(node_labels):
            raise ManifoldFormatError(
                f"manifold {name!r} has duplicate node labels"
            )

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
            cyclic=bool(data.get("cyclic", False)),
            node_labels=node_labels,
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
        """Stable hash of the ordered node corpus.

        The staleness key: a fitted tensor's sidecar records this, and a
        re-fit is needed when it no longer matches.
        """
        h = hashlib.sha256()
        for idx in range(len(self.node_labels)):
            h.update(self.node_path(idx).read_bytes())
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
        integrity manifest covers the corpus and every fitted artifact.
        """
        if files is None:
            files = hash_manifold_files(self.folder)
        self.files = files
        payload: dict[str, Any] = {
            "format_version": PACK_FORMAT_VERSION,
            "name": self.name,
            "description": self.description,
            "cyclic": self.cyclic,
            "nodes": list(self.node_labels),
            "files": files,
        }
        write_json_atomic(self.folder / "manifold.json", payload)


__all__ = [
    "MIN_MANIFOLD_NODES",
    "ManifoldFormatError",
    "ManifoldSidecar",
    "ManifoldFolder",
    "hash_manifold_files",
]
