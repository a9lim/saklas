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
import logging
import re
import shutil
import warnings
from dataclasses import dataclass, field
from importlib import resources as _resources
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from saklas.core.errors import SaklasError
from saklas.core.manifold import BoxDomain, domain_from_spec
from saklas.core.role_templates import _ROLE_SLUG_RE
from saklas.io.atomic import write_bytes_atomic, write_json_atomic
from saklas.io.packs import NAME_REGEX, verify_integrity
from saklas.io.paths import manifold_dir, manifolds_dir, saklas_home

_log = logging.getLogger("saklas.io.manifolds")

# Manifold node *labels* are stricter than the artifact ``NAME_REGEX``
# (``io.packs``): they drop ``.`` because the dot is reserved as the
# bipolar-pole separator (``deer.wolf``) and the steering-expr lexer
# addresses a label via ``%label`` — a dotted label could neither be
# typed unambiguously (``persona%a.b`` reads ``a`` dotted with pole ``b``)
# nor resolved through the bare-name pipeline.  Labels are therefore the
# grammar-addressable identifier subset of the name grammar: lowercase
# start, then ``[a-z0-9_-]`` only, ≤64 chars.  The manifold *name*
# itself keeps ``NAME_REGEX`` (it is never used as a ``%`` operand).
_LABEL_REGEX = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

# Process-scope flag: set True after the first ``materialize_bundled_manifolds``
# call so subsequent calls within the same Python process are no-ops.  See
# the docstring on that function for rationale (avoids clobbering CLI-set
# hyperparams on session re-init within the same invocation).
_materialized_this_process: bool = False

# Discover-mode fit modes — set as ``manifold.json::fit_mode`` for
# manifolds whose node coordinates are derived from the model's
# activations rather than authored by hand.  Authored manifolds carry
# ``fit_mode == "authored"`` (or omit the field, which means the same).
_FIT_MODES_DISCOVER: frozenset[str] = frozenset({"pca", "spectral"})
_FIT_MODES_ALL: frozenset[str] = frozenset({"authored"}) | _FIT_MODES_DISCOVER

# Per-method hyperparameter whitelists.  Anything outside the whitelist
# for a given fit_mode is dropped at folder-create time so a user
# POSTing ``{fit_mode: "pca", hyperparams: {"k_nn": 5}}`` doesn't land
# a foreign key that would then crash ``derive_pca_coords`` at fit
# time with ``TypeError: unexpected keyword argument``.  ``max_dim``
# is shared.  ``reference_layer`` is honored by both methods (consumed
# in ``ManifoldExtractionPipeline.fit`` before the dispatch).
_HYPERPARAMS_BY_MODE: dict[str, frozenset[str]] = {
    "pca": frozenset({
        "max_dim", "var_threshold", "reference_layer", "max_subspace_dim",
        # ``anchor_origin`` is the discover-pca origin-anchoring toggle.
        # When set to a node label (or the literal ``true`` — sugar for
        # the canonical ``"default"`` label), the fit pipeline locates
        # that label among the K nodes, takes its derived authoring coord
        # as the anchor, and translates *all* node coords so the anchor
        # sits at ``(0, ..., 0)``.  The RBF interpolant is exact at fit
        # nodes, so steering ``<manifold>%0,0,...`` reproduces the
        # anchor's per-layer behavior — giving the manifold's origin a
        # principled meaning ("no behavioral shift from the anchor")
        # instead of "centroid of the K corpora in PCA space".  Spectral
        # mode is not yet supported — Laplacian eigenmaps need Nyström-
        # style out-of-sample extension to project a held-out anchor.
        "anchor_origin",
    }),
    "spectral": frozenset({
        "max_dim", "k_nn", "bandwidth", "reference_layer", "max_subspace_dim",
    }),
}


def domain_label(spec: dict[str, Any]) -> str:
    """Short ``type(Nd)`` label for a manifold domain spec dict."""
    kind = spec.get("type", "?")
    if kind == "box":
        n = len(spec.get("axes", []))
    elif kind == "sphere":
        n = int(spec.get("dim", 0))
    elif kind == "custom":
        n = int(spec.get("embed_dim", 0))
    else:
        n = 0
    return f"{kind}({n}d)"


def _sanitize_hyperparams(
    fit_mode: str, hyperparams: dict[str, Any] | None,
) -> dict[str, Any]:
    """Drop hyperparam keys that don't apply to ``fit_mode``.

    Single source of truth for the per-method whitelist; both the create
    and the fit-override paths funnel through this so the folder
    manifest never carries a key that would crash the dispatcher.  An
    unknown ``fit_mode`` (already validated upstream) passes through
    unchanged — better to land a soft "extra key" warning at fit time
    than to silently drop everything.
    """
    if hyperparams is None:
        return {}
    allowed = _HYPERPARAMS_BY_MODE.get(fit_mode)
    if allowed is None:
        return dict(hyperparams)
    return {k: v for k, v in hyperparams.items() if k in allowed}

# Manifold artifact format version.  Decoupled from concept packs'
# ``PACK_FORMAT_VERSION`` so the two formats can churn independently.
# v3 is the arbitrary-dimensional / arbitrary-topology format (domain
# spec + per-node coordinates); v2 and earlier were the 1-D cyclic-spline
# format and must be converted with ``scripts/upgrade_manifolds.py``.
# v4 adds the per-layer ``explained_variance_per_layer`` sidecar field
# used by additive-mode manifold steering's quality normalization — the
# field rides on optional sidecar metadata, so a v3 manifold loads at
# v4 fine *except* that we want users to refit so the EV value is
# populated and the normalizer can kick in.  Bumping the version forces
# materialize_bundled_manifolds to refresh the bundled fit on next
# session start; user-fit v3 manifolds will need ``saklas vector
# manifold fit`` to pick up the new field.
# v5 adds the per-layer ``origin_per_layer`` sidecar field (the per-layer
# authoring-coordinate foot of the neutral mean, ``{str(L): [coord, ...]}``,
# the cold-start foot seed).  Loading stays back-compatible — an absent
# field loads as an empty dict (the apply path seeds a zero-coord foot per
# layer), so no migration is needed; the bump just forces
# materialize_bundled_manifolds to refresh the bundled fit so the origins
# get baked on the next fit.
MANIFOLD_FORMAT_VERSION = 5


def _validate_node_role(name: str, label: str, role: Any) -> str | None:
    """Validate an optional per-node ``role`` field.

    ``None`` / missing means "use the standard assistant baseline" (the
    legacy shape, same as today).  A non-empty string must match
    :data:`saklas.core.role_templates._ROLE_SLUG_RE`
    (``[a-z0-9._-]+``).  Family-unsupported (Mistral-3, talkie) is *not*
    checked here — the folder is model-agnostic; the check fires when
    :func:`saklas.core.role_templates.apply_with_role` runs at fit time.
    """
    if role is None:
        return None
    if not isinstance(role, str) or not _ROLE_SLUG_RE.match(role):
        raise ManifoldFormatError(
            f"manifold {name!r} node {label!r} role {role!r} invalid; "
            f"must match {_ROLE_SLUG_RE.pattern}"
        )
    return role


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
    # Twin: :func:`saklas.io.packs.hash_file` is byte-identical and kept
    # separate by design (the manifold format is decoupled from packs);
    # mirror any change to one in the other.
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

    Discover-mode fits (``fit_mode != "authored"``) additionally carry
    the derived ``coords``, the hyperparameters used, and the per-method
    diagnostics block (PCA variance bars or spectral spectrum) so the
    sidecar is self-describing for downstream inspection.
    """

    # ``manifold_pca`` / ``manifold_sae`` for authored fits;
    # ``manifold_discover_pca`` / ``manifold_discover_spectral`` /
    # ``manifold_discover_sae`` for discover-mode fits (the SAE label
    # collapses across pca/spectral because the SAE reconstruction
    # rides into both before the coord derivation runs).
    method: str
    saklas_version: str
    domain: dict[str, Any]
    node_count: int
    node_labels: list[str]
    feature_space: str = "raw"
    nodes_sha256: Optional[str] = None
    sae_release: Optional[str] = None
    sae_revision: Optional[str] = None
    # Discover-mode-only fields.  ``None`` on authored fits.
    fit_mode: str = "authored"
    hyperparams: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Per-node assistant-role substitution used at fit time, in
    # ``node_labels`` index order.  ``None`` for a given node (and an
    # empty list as a whole) = "standard assistant baseline" — the
    # default, byte-identical to today's non-role manifolds.  The same
    # information rides ``ManifoldFolder.node_roles`` but the sidecar
    # carries an independent copy so a downstream consumer
    # (``vector manifold show``, the webui inspector) doesn't have to
    # round-trip through the folder to know which role each node was
    # pooled under.
    node_roles: list[str | None] = field(default_factory=list)

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
            fit_mode=data.get("fit_mode", "authored"),
            hyperparams=dict(data.get("hyperparams", {})),
            diagnostics=dict(data.get("diagnostics", {})),
            node_roles=list(data.get("node_roles", [])),
        )


@dataclass
class ManifoldFolder:
    """A manifold artifact folder on disk.

    Discovery + corpus + integrity only — the fitted RBF tensors are
    loaded through :func:`saklas.core.manifold.load_manifold`.

    Two folder shapes share this class via the ``fit_mode`` field:

    - ``fit_mode == "authored"`` (default for legacy v3 manifolds): the
      user supplied per-node ``coords`` on a declared ``domain``.  The
      fit pipeline embeds the coords and runs straight into
      ``fit_layer_subspace``.
    - ``fit_mode in {"pca", "spectral"}`` (discover mode): nodes carry
      ``{label}`` only — no coords, no top-level ``domain``.  The fit
      pipeline pools per-node centroids, derives coords via
      :func:`saklas.core.manifold.discover_coords` (per-model, since
      different models embed the same heap differently), wraps them in
      a ``CustomDomain(k)`` with identity embedding, and fits.

    For authored folders ``domain`` is the tagged-union spec and
    ``node_coords`` is the K×n list; for discover folders both are
    empty placeholders (``{}`` and ``[]``) — the real geometry lives
    in the per-model sidecar after the fit runs.  ``hyperparams``
    captures the discover-mode knobs (``max_dim``, ``var_threshold``
    for PCA; ``max_dim``, ``k_nn``, ``bandwidth`` for spectral) and is
    empty for authored folders.
    """

    folder: Path
    name: str
    description: str
    domain: dict[str, Any]
    node_labels: list[str]
    node_coords: list[list[float]]
    files: dict[str, str]
    fit_mode: str = "authored"
    hyperparams: dict[str, Any] = field(default_factory=dict)
    # Provenance tier, mirroring :attr:`saklas.io.packs.PackMetadata.source`.
    # ``"local"`` (default) — hand-authored / generated under ``local/``;
    # ``"bundled"`` — shipped under ``saklas/data/manifolds/`` and
    # materialized into ``default/`` (set by namespace at refresh time,
    # not stored); ``"hf://<owner>/<name>[@rev]"`` — pulled from the HF
    # hub (stamped by :func:`saklas.io.hf_manifolds.pull_manifold`).  The
    # ``refresh_manifold`` lifecycle reads it to decide where to re-pull:
    # ``local`` is silently skipped (nothing upstream), ``hf://`` re-pulls.
    source: str = "local"
    # Per-node assistant-role substitution for role-augmented manifolds
    # (e.g. a persona manifold where each node is a persona).  Aligned
    # with ``node_labels`` index-by-index.  ``None`` for a given node =
    # "use the standard assistant baseline" (the legacy shape, what every
    # pre-role-differential manifold carries).  An all-``None`` list is
    # semantically identical to today's behavior — the centroid pooling
    # just goes through the default chat-template branch.
    node_roles: list[str | None] = field(default_factory=list)
    # tensor stem (``<safe_model>`` or ``<safe_model>_sae-<rel>``) -> sidecar.
    _sidecars: dict[str, ManifoldSidecar] = field(default_factory=dict)

    @property
    def is_discover(self) -> bool:
        """True when this folder's coords are derived per-model rather than authored."""
        return self.fit_mode in _FIT_MODES_DISCOVER

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
        if fmt > MANIFOLD_FORMAT_VERSION:
            # Symmetric upper bound, mirroring ``PackMetadata.load`` — a
            # manifold authored by a newer saklas may use fields this
            # reader can't safely interpret, so refuse rather than load
            # it silently.
            raise ManifoldFormatError(
                f"manifold.json in {folder} was created by a newer saklas "
                f"(format v{fmt} > local v{MANIFOLD_FORMAT_VERSION}); "
                f"upgrade saklas."
            )

        name = data.get("name", folder.name)
        if not isinstance(name, str) or not NAME_REGEX.match(name):
            raise ManifoldFormatError(
                f"manifold name {name!r} invalid; must match {NAME_REGEX.pattern}"
            )

        # Authored vs discover.  Authored = the historical shape:
        # ``domain`` + per-node ``coords``.  Discover = no ``domain`` and
        # nodes carry ``{label}`` only; coords are derived per-model at
        # fit time.  Default ``"authored"`` keeps every pre-discover v3
        # manifold loading unchanged.
        fit_mode = data.get("fit_mode", "authored")
        if fit_mode not in _FIT_MODES_ALL:
            raise ManifoldFormatError(
                f"manifold {name!r} fit_mode {fit_mode!r} invalid; "
                f"expected one of {sorted(_FIT_MODES_ALL)}"
            )

        nodes = data.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ManifoldFormatError(
                f"manifold {name!r} needs a non-empty 'nodes' list"
            )

        node_labels: list[str] = []
        node_coords: list[list[float]] = []
        node_roles: list[str | None] = []
        hyperparams: dict[str, Any] = {}
        domain_spec: dict[str, Any]

        if fit_mode == "authored":
            domain_spec = data.get("domain") or {}
            if not isinstance(domain_spec, dict) or not domain_spec:
                raise ManifoldFormatError(
                    f"authored manifold {name!r} needs a 'domain' object"
                )
            try:
                domain = domain_from_spec(domain_spec)
            except (ValueError, KeyError) as e:
                raise ManifoldFormatError(
                    f"manifold {name!r} has an invalid domain: {e}"
                ) from e
            n = domain.intrinsic_dim
            floor = min_nodes(n)
            if len(nodes) < floor:
                raise ManifoldFormatError(
                    f"manifold {name!r} ({n}-D domain) needs a 'nodes' "
                    f"list of >= {floor} entries, got {len(nodes)}"
                )
            for entry in nodes:
                if not isinstance(entry, dict):
                    raise ManifoldFormatError(
                        f"manifold {name!r} node {entry!r} must be an "
                        f"object with 'label' and 'coords'"
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
                        f"manifold {name!r} node {label!r} needs 'coords' "
                        f"of {n} number(s), got {coords!r}"
                    )
                node_labels.append(label)
                node_coords.append([float(c) for c in coords])
                node_roles.append(_validate_node_role(name, label, entry.get("role")))
            _warn_authoring_quality(name, domain, node_coords)
        else:
            # Discover mode: no ``domain`` field, no per-node ``coords``.
            # The fit pipeline derives coords per-model from the
            # activations and wraps them in a ``CustomDomain(k)``.  We
            # do not even know the intrinsic dimension until after the
            # fit, so the ``min_nodes(k)`` floor is enforced at fit time
            # (once ``k`` is picked) rather than here.
            if "domain" in data and data["domain"]:
                raise ManifoldFormatError(
                    f"discover-mode manifold {name!r} must not carry a "
                    f"'domain' field — coords are derived per-model at "
                    f"fit time"
                )
            domain_spec = {}
            hyperparams = dict(data.get("hyperparams", {}))
            for entry in nodes:
                if not isinstance(entry, dict):
                    raise ManifoldFormatError(
                        f"discover manifold {name!r} node {entry!r} must "
                        f"be an object with 'label'"
                    )
                label = entry.get("label")
                if not isinstance(label, str) or not _LABEL_REGEX.match(label):
                    raise ManifoldFormatError(
                        f"discover manifold {name!r} node label "
                        f"{label!r} invalid; a node label is a grammar-"
                        f"addressable identifier (no '.', reserved as the "
                        f"bipolar separator) — must match "
                        f"{_LABEL_REGEX.pattern}"
                    )
                if "coords" in entry:
                    raise ManifoldFormatError(
                        f"discover manifold {name!r} node {label!r} "
                        f"must not carry 'coords' — coords are derived "
                        f"at fit time"
                    )
                node_labels.append(label)
                node_roles.append(_validate_node_role(name, label, entry.get("role")))

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
            domain=domain_spec,
            node_labels=node_labels,
            node_coords=node_coords,
            files=files,
            fit_mode=fit_mode,
            hyperparams=hyperparams,
            source=str(data.get("source", "local")),
            node_roles=node_roles,
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
            # Every fitted tensor must carry its ``.json`` sidecar, the
            # same invariant ``ConceptFolder.load`` enforces.  ``fit``
            # always writes one via ``save_manifold``, so a missing
            # sidecar means a genuinely-corrupt folder, not a legitimate
            # shape — refuse rather than silently accept a sidecar-less
            # tensor.
            if not sc_path.exists():
                raise ManifoldFormatError(
                    f"manifold {name!r} tensor {t.name} has no sidecar "
                    f"{sc_path.name}"
                )
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
        """Stable hash of the inputs that determine a fit's output.

        The staleness key: a fitted tensor's sidecar records this, and a
        re-fit is needed when it no longer matches.  For authored
        manifolds the hash folds in the corpus, the domain spec, and
        the authoring coordinates — any geometry edit (moving a node,
        flipping an axis to periodic) triggers a re-fit alongside
        corpus edits.  For discover manifolds the corresponding inputs
        are the corpus plus the fit mode plus the hyperparameters
        (``max_dim``, ``var_threshold`` for PCA; ``max_dim``,
        ``k_nn``, ``bandwidth`` for spectral) — changing any of those
        invalidates a cached fit.

        The field name is unchanged for backward compat with v3
        sidecars; the semantics naturally extends in the discover case.
        """
        h = hashlib.sha256()
        for idx in range(len(self.node_labels)):
            h.update(self.node_path(idx).read_bytes())
        if self.fit_mode == "authored":
            h.update(_canonical_json(self.domain))
            h.update(_canonical_json(self.node_coords))
        else:
            h.update(_canonical_json({
                "fit_mode": self.fit_mode,
                "hyperparams": self.hyperparams,
            }))
        # Per-node roles are inputs that determine the fit's geometry
        # (each node's centroid is pooled under its role's chat-template
        # substitution), so a role edit must invalidate a cached fit.
        # All-``None`` (legacy / non-role) hashes to the same value
        # whether the field is missing or explicit-None — same shape.
        if any(r is not None for r in self.node_roles):
            h.update(_canonical_json(self.node_roles))
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
            "fit_mode": self.fit_mode,
            "files": files,
        }
        # Preserve provenance across re-fits, mirroring how a pack's
        # ``source`` survives ``PackMetadata.write``.  Only the default
        # ``"local"`` is omitted, so a hand-authored / generated folder
        # stays byte-identical to the pre-source shape; an ``hf://`` or
        # ``bundled`` tier is written so ``refresh_manifold`` can find it
        # after a fit has rewritten the manifest.
        if self.source and self.source != "local":
            payload["source"] = self.source
        # Per-node ``role`` is written only when set — keeps the legacy
        # shape (every node carries ``{label, coords}`` or ``{label}``
        # only) byte-identical for non-role manifolds, and a stray
        # ``role: null`` doesn't leak into the manifest for a node that
        # opted out.
        if self.fit_mode == "authored":
            payload["domain"] = self.domain
            payload["nodes"] = [
                _node_payload_authored(label, coords, role)
                for label, coords, role in zip(
                    self.node_labels, self.node_coords, self._roles_padded(),
                )
            ]
        else:
            payload["hyperparams"] = self.hyperparams
            payload["nodes"] = [
                _node_payload_discover(label, role)
                for label, role in zip(self.node_labels, self._roles_padded())
            ]
        write_json_atomic(self.folder / "manifold.json", payload)

    def _roles_padded(self) -> list[str | None]:
        """Return ``node_roles`` padded to ``len(node_labels)`` with ``None``s.

        ``ManifoldFolder`` constructed via :meth:`load` always carries a
        full-length ``node_roles``, but in-memory mutations (e.g.
        :func:`update_manifold_folder` swapping the node list) might
        leave the roles list out of sync.  Padding here is defensive.
        """
        if len(self.node_roles) == len(self.node_labels):
            return list(self.node_roles)
        return [None] * len(self.node_labels)


def _node_payload_authored(
    label: str, coords: list[float], role: str | None,
) -> dict[str, Any]:
    """Build one authored-mode node entry for ``manifold.json``.

    ``role`` is emitted only when set, so the legacy ``{label, coords}``
    shape stays byte-identical for non-role manifolds.
    """
    out: dict[str, Any] = {"label": label, "coords": [float(c) for c in coords]}
    if role is not None:
        out["role"] = role
    return out


def _node_payload_discover(label: str, role: str | None) -> dict[str, Any]:
    """Build one discover-mode node entry for ``manifold.json``."""
    out: dict[str, Any] = {"label": label}
    if role is not None:
        out["role"] = role
    return out


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
            _node_payload_authored(
                entry["label"], entry["coords"], entry.get("role"),
            )
            for entry in nodes
        ],
        "files": {},
    }
    write_json_atomic(folder / "manifold.json", payload)

    _, advisories = _load_with_advisories(folder)
    return folder, advisories


def _validate_discover_corpora(
    name: str, node_corpora: dict[str, list[str]],
) -> None:
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


def create_discover_manifold_folder(
    namespace: str,
    name: str,
    description: str,
    *,
    fit_mode: str,
    node_corpora: dict[str, list[str]],
    hyperparams: Optional[dict[str, Any]] = None,
    node_roles: Optional[dict[str, str | None]] = None,
    scenarios: Optional[list[str]] = None,
) -> Path:
    """Author a fresh discover-mode manifold artifact folder on disk.

    ``fit_mode`` is one of ``"pca"`` / ``"spectral"``;
    ``node_corpora`` is the authoring shape ``{label: [statement, ...]}``.
    Writes ``manifold.json`` (no ``domain``, no per-node ``coords``,
    empty ``files`` manifest) and the ``nodes/`` corpus.  Returns the
    folder path.

    ``scenarios`` (when given) is persisted to ``scenarios.json`` as the
    generation-provenance record — the discover-manifold analogue of the
    vector pipeline's ``scenarios.json``, so a later re-fit/refresh can
    regenerate against the same shared domains instead of drifting.

    Coords are derived per-model at fit time
    (:func:`saklas.core.manifold.discover_coords` runs over the per-node
    centroids), so authoring quality cannot be advised here the way it
    is for authored manifolds — the spectral connectivity check and PCA
    variance-floor diagnostics surface only on the fit itself.

    Raises :class:`ManifoldFormatError` on any validation failure and
    :class:`FileExistsError` when a manifold already lives at the path.

    Cross-model transfer of a fitted layout is handled by
    :func:`transfer_manifold` — it reuses the same per-layer Procrustes
    map (:mod:`saklas.io.alignment`) the ``vector transfer`` path builds
    and writes a ``_from-<safe_src>`` variant tensor, mirroring how
    transferred steering vectors land.
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
    roles_resolved: dict[str, str | None] = {label: None for label in node_corpora}
    if node_roles is not None:
        unknown = set(node_roles) - set(node_corpora)
        if unknown:
            raise ManifoldFormatError(
                f"discover manifold {name!r} node_roles carries labels "
                f"not in node_corpora: {sorted(unknown)}"
            )
        for label, role in node_roles.items():
            roles_resolved[label] = _validate_node_role(name, label, role)

    folder = manifold_dir(namespace, name)
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
        "hyperparams": _sanitize_hyperparams(fit_mode, hyperparams),
        "nodes": [
            _node_payload_discover(label, roles_resolved[label])
            for label in node_corpora
        ],
        "files": {},
    }
    write_json_atomic(folder / "manifold.json", payload)
    if scenarios is not None:
        write_manifold_scenarios(folder, scenarios)
    return folder


def write_manifold_scenarios(folder: Path, scenarios: list[str]) -> None:
    """Persist the shared scenario list to ``<folder>/scenarios.json``.

    Discover-mode generation provenance — the domains the node corpora
    were generated against — mirroring the ``{"scenarios": [...]}``
    schema the vector extraction pipeline writes.  Both the all-at-once
    :func:`create_discover_manifold_folder` (via its ``scenarios``
    kwarg) and the streaming path (as the ``on_scenarios`` sink of
    :meth:`SaklasSession.generate_statements`) route through here.
    """
    write_json_atomic(
        folder / "scenarios.json",
        {"scenarios": [str(s) for s in scenarios]},
    )


def init_discover_manifold_folder(
    namespace: str,
    name: str,
    description: str,
    *,
    fit_mode: str,
    labels: list[str],
    hyperparams: Optional[dict[str, Any]] = None,
    node_roles: Optional[dict[str, str | None]] = None,
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
    if not NAME_REGEX.match(name):
        raise ManifoldFormatError(
            f"manifold name {name!r} invalid; must match {NAME_REGEX.pattern}"
        )
    if not NAME_REGEX.match(namespace):
        raise ManifoldFormatError(
            f"manifold namespace {namespace!r} invalid; "
            f"must match {NAME_REGEX.pattern}"
        )
    if fit_mode not in _FIT_MODES_DISCOVER:
        raise ManifoldFormatError(
            f"discover manifold {name!r} fit_mode {fit_mode!r} invalid; "
            f"expected one of {sorted(_FIT_MODES_DISCOVER)}"
        )
    if not labels:
        raise ManifoldFormatError(
            f"discover manifold {name!r} needs at least one node label"
        )
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

    roles_resolved: dict[str, str | None] = {label: None for label in labels}
    if node_roles is not None:
        unknown = set(node_roles) - set(labels)
        if unknown:
            raise ManifoldFormatError(
                f"discover manifold {name!r} node_roles carries labels "
                f"not in the roster: {sorted(unknown)}"
            )
        for label, role in node_roles.items():
            roles_resolved[label] = _validate_node_role(name, label, role)

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
        "hyperparams": _sanitize_hyperparams(fit_mode, hyperparams),
        "nodes": [
            _node_payload_discover(label, roles_resolved[label])
            for label in labels
        ],
        "files": {},
    }
    write_json_atomic(folder / "manifold.json", payload)
    return folder


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
    if not isinstance(label, str) or not _LABEL_REGEX.match(label):
        raise ManifoldFormatError(
            f"discover manifold node label {label!r} invalid; "
            f"must match {_LABEL_REGEX.pattern}"
        )
    if (
        not isinstance(statements, list)
        or not statements
        or not all(isinstance(s, str) and s.strip() for s in statements)
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
) -> dict[str, Any]:
    """Build the label-only discover ``manifold.json`` dict."""
    return {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": description,
        "fit_mode": fit_mode,
        "hyperparams": _sanitize_hyperparams(fit_mode, hyperparams),
        "nodes": [_node_payload_discover(label, roles.get(label)) for label in labels],
        "files": {},
    }


@dataclass(frozen=True)
class DiscoverGenerationPlan:
    """Resume / extend plan for streaming discover-manifold generation.

    Returned by :func:`plan_discover_generation`.  ``index_of`` maps each
    label to its on-disk node index (the ``NN_`` filename prefix);
    ``pending`` is the declared labels whose corpus file is not yet on
    disk (the ones a run still needs to generate, in node order);
    ``scenarios`` is the locked domain list read back from
    ``scenarios.json`` (``None`` on a fresh folder); ``added`` is the
    labels appended to ``manifold.json`` this call (the add-nodes case);
    ``resumed`` is ``True`` when an existing ``manifold.json`` was found.
    """

    folder: Path
    index_of: dict[str, int]
    pending: tuple[str, ...]
    scenarios: tuple[str, ...] | None
    added: tuple[str, ...]
    resumed: bool


def plan_discover_generation(
    folder: Path,
    name: str,
    description: str,
    *,
    fit_mode: str,
    labels: list[str],
    hyperparams: Optional[dict[str, Any]] = None,
    node_roles: Optional[dict[str, str | None]] = None,
) -> DiscoverGenerationPlan:
    """Ensure a streamable discover skeleton at ``folder`` covering every
    label in ``labels``, and report which node corpora still need writing.

    Resume + add-nodes in one — the single planner every discover-generate
    surface (the bundled regen scripts, ``vector manifold generate``, the
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

    The read is deliberately lenient — it does **not** route through
    :meth:`ManifoldFolder.load`, which rejects a partially-written folder
    by design (the missing-node-corpus guard) — so the planner can
    inspect the very partial it is resuming.  Description is refreshed to
    the caller's value; fit hyperparams on an existing folder are kept
    (a resume fills corpus, it does not silently re-spec the fit — use
    ``--force`` / a fresh folder to change those).

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
    if not labels:
        raise ManifoldFormatError(
            f"discover manifold {name!r} needs at least one node label"
        )
    seen: set[str] = set()
    for label in labels:
        if not isinstance(label, str) or not _LABEL_REGEX.match(label):
            raise ManifoldFormatError(
                f"discover manifold {name!r} label {label!r} invalid; "
                f"must match {_LABEL_REGEX.pattern}"
            )
        if label in seen:
            raise ManifoldFormatError(
                f"discover manifold {name!r} duplicate node label {label!r}"
            )
        seen.add(label)
    roles_in = node_roles or {}
    unknown = set(roles_in) - set(labels)
    if unknown:
        raise ManifoldFormatError(
            f"discover manifold {name!r} node_roles carries labels not in "
            f"the roster: {sorted(unknown)}"
        )

    meta_path = folder / "manifold.json"
    nodes_dir = folder / "nodes"

    if not meta_path.exists():
        # Fresh skeleton — everything is pending.
        roles_resolved = {
            label: _validate_node_role(name, label, roles_in.get(label))
            for label in labels
        }
        folder.mkdir(parents=True, exist_ok=True)
        nodes_dir.mkdir(exist_ok=True)
        write_json_atomic(
            meta_path,
            _discover_manifest_payload(
                name, description, fit_mode, labels, roles_resolved, hyperparams,
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
    new_labels = [label for label in labels if label not in existing_labels]
    full_labels = existing_labels + new_labels
    if new_labels:
        # Add-nodes: append the new labels (validating their roles) and
        # rewrite manifold.json atomically.  Existing roles/hyperparams
        # are preserved; description refreshes to the caller's value.
        merged_roles: dict[str, str | None] = dict(existing_roles)
        for label in new_labels:
            merged_roles[label] = _validate_node_role(
                name, label, roles_in.get(label),
            )
        data["description"] = description
        data["nodes"] = [
            _node_payload_discover(label, merged_roles.get(label))
            for label in full_labels
        ]
        write_json_atomic(meta_path, data)
    elif data.get("description") != description:
        data["description"] = description
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

    The vector-side counterpart is :func:`saklas.io.merge.merge_into_pack`,
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
    materialized.  Run ``saklas vector manifold discover`` or
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
        mf = ManifoldFolder.load(folder_path)
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
    if hyperparams is not None:
        effective_hyperparams = dict(hyperparams)
    else:
        effective_hyperparams = dict(folders[0][2].hyperparams)

    # Pool the corpus + roles.  Iteration order: source order, then
    # per-source label order (matches how the source authored them).
    node_corpora: dict[str, list[str]] = {}
    node_roles: dict[str, str | None] = {}
    for _ns, _name, mf in folders:
        groups = dict(mf.node_groups())
        for idx, label in enumerate(mf.node_labels):
            node_corpora[label] = list(groups.get(label, []))
            role = (
                mf.node_roles[idx]
                if idx < len(mf.node_roles)
                else None
            )
            node_roles[label] = role

    target_folder = manifold_dir(target_namespace, target_name)
    if target_folder.exists():
        if not force:
            raise FileExistsError(
                f"manifold {target_namespace}/{target_name} already exists; "
                f"pass force=True to overwrite",
            )
        shutil.rmtree(target_folder)

    return create_discover_manifold_folder(
        target_namespace,
        target_name,
        target_description,
        fit_mode=fit_mode,
        node_corpora=node_corpora,
        hyperparams=effective_hyperparams,
        node_roles=node_roles,
    )


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
        # Roles ride per-entry; defaults to None for entries that
        # don't carry the field.  ``_validate_authored_nodes`` already
        # checked the slug shape, so this is a pure copy.
        mf.node_roles = [entry.get("role") for entry in nodes]
    mf.write_metadata()
    _, advisories = _load_with_advisories(folder)
    return folder, advisories


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

    Globs ``*.safetensors``, filters by ``variant`` (``raw`` / ``sae`` /
    ``from`` / ``all``), and pairs each kept tensor with its sidecar.  The
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
    out: list[Path] = []
    for ts in sorted(folder.glob("*.safetensors")):
        parsed = parse_tensor_filename(ts.name)
        if parsed is None:
            continue
        model, var = parsed
        if target_safe is not None and model != target_safe:
            continue
        key = "raw" if var is None else var
        if not _manifold_tensor_variant_matches(key, variant):
            continue
        out.append(ts)
        sc = ts.with_suffix(".json")
        if sc.exists():
            out.append(sc)
    return out


def clear_manifold_tensors(
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
        raise FileNotFoundError(f"manifold {namespace}/{name} not found at {folder}")
    # Load *before* unlinking — once the tensors are gone the populated
    # ``files`` manifest would fail the integrity check on a reload.  Keep
    # the in-memory folder and re-hash from disk afterward, the same shape
    # ``cache_ops.delete_tensors`` uses (load, mutate, re-hash).
    mf = ManifoldFolder.load(folder)
    files = _manifold_tensor_files(folder, variant=variant, model_scope=model_scope)
    for f in files:
        f.unlink()
    if files:
        # ``write_metadata`` defaults to re-hashing the now-smaller
        # on-disk tensor set via ``hash_manifold_files``.
        mf.write_metadata()
    return len(files)


def remove_manifold_folder(namespace: str, name: str) -> dict[str, Any]:
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
        raise FileNotFoundError(f"manifold {namespace}/{name} not found at {folder}")
    # Read the source tier before deleting (best-effort — a corrupt
    # manifest just reports the namespace-implied tier).
    try:
        source = ManifoldFolder.load(folder).source
    except ManifoldFormatError:
        source = "bundled" if namespace == "default" else "local"
    rematerializes = namespace == "default" or source == "bundled"
    shutil.rmtree(folder)
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
        raise FileNotFoundError(f"manifold {namespace}/{name} not found at {folder}")

    if model_scope is not None:
        # Scoped refresh: drop just that model's fitted tensor pair so it
        # re-fits from the node corpus on next use.  Mirrors the pack-side
        # tensors-only scoped refresh — a whole-repo re-pull for one model
        # makes no sense (HF pulls are whole-folder).
        clear_manifold_tensors(namespace, name, model_scope, variant="all")
        return "scoped"

    source = ManifoldFolder.load(folder).source

    if namespace == "default" or source == "bundled":
        # Bundled tier — re-copy from package data.  Process-scoped, so
        # this is a no-op after the first materialize within a process.
        materialize_bundled_manifolds()
        return "bundled"

    if source.startswith("hf://"):
        from saklas.io.hf import split_revision
        from saklas.io.hf_manifolds import pull_manifold

        coord, revision = split_revision(source[len("hf://"):])
        pull_manifold(coord, target_folder=folder, force=force, revision=revision)
        return "hf"

    # local (or anything without an upstream): nothing to do.
    return "skipped"


# ============================================================ cross-model transfer ===
#
# The manifold analogue of ``saklas vector transfer`` (which writes a
# ``_from-<safe_src>`` variant tensor for steering vectors via
# ``saklas.io.alignment.transfer_profile``).  Pure-io: the caller builds
# the per-layer Procrustes alignment map — that needs both models loaded,
# which lives in the session/CLI layer — and passes it in; this function
# only *applies* it to a fitted manifold's per-layer subspace and writes
# the transferred tensor.  Do not rebuild the Procrustes solver here.


def transfer_manifold(
    folder: Path,
    *,
    from_model: str,
    to_model: str,
    alignment: dict[int, torch.Tensor],
    transfer_quality_estimate: Optional[float] = None,
    whitener: "Any | None" = None,
    layer_means: "dict[int, torch.Tensor] | None" = None,
    force: bool = False,
) -> Path:
    """Apply a per-layer alignment map to a fitted manifold, target-side.

    Reads the source-model fit at ``<folder>/<safe_from>.safetensors``,
    maps each layer's affine subspace (``mean`` + ``basis`` rows, both in
    model space) through the supplied ``alignment`` map
    (``{layer: M_L}`` where ``v_tgt = M_L @ v_src``, the shape
    :func:`saklas.io.alignment.fit_alignment` produces), and writes a
    transferred per-model tensor at the ``_from-<safe_src>`` filename
    variant (``<safe_to>_from-<safe_from>.safetensors`` —
    :func:`saklas.io.paths.tensor_filename`'s transfer suffix).

    The RBF interpolant fields (``node_params`` / ``rbf_weights`` /
    ``poly_coeffs`` / ``coord_offset`` / ``coord_scale``) live in
    subspace/authoring-coordinate space, not model space, so they ride
    through untouched — the subspace itself relocates via the transformed
    ``mean``/``basis`` and the in-subspace parameterization is invariant.
    ``node_coords`` (the intrinsic authoring layout) is likewise
    model-independent.  Layers the alignment doesn't cover are dropped,
    mirroring :func:`saklas.io.alignment.transfer_profile`.

    ``alignment`` is supplied by the caller (building it needs both
    models loaded — a session/CLI concern), keeping this function
    pure-io.  ``transfer_quality_estimate`` (median per-layer R², if the
    caller computed it) rides into the sidecar provenance.

    **Target-metric re-bake (``whitener`` + ``layer_means`` given).**  The
    fitted manifold's per-layer Mahalanobis share and steering lever are
    per-model quantities (``Σ`` and the neutral activations both belong to
    ``from_model``), so a bare transfer can't carry them.  When a
    ``whitener`` for the **target** model is supplied and covers every
    transferred layer (all-or-nothing, mirroring the fit gate), both are
    recomputed in target space: the share via
    ``sqrt(Σ_k coordsᵀ (B_tgt Σ_tgt⁻¹ B_tgtᵀ) coords)`` (the RBF reduced
    node values × the target subspace-restricted inverse covariance) and
    the lever via :func:`~saklas.core.manifold.layer_lever` over the target
    neutral activations (the whitener's centered observations uncentered by
    ``layer_means``).  The sidecar then records ``share_metric:
    "mahalanobis"``.  Without coverage (or no whitener), both are left
    empty — the apply path falls back to the Euclidean centroid-spread
    share + ``N = 1`` lever (``share_metric: "euclidean"``).
    ``subspace_metric`` always carries the source value: the basis was
    *selected* on the source model and only rotated here.

    Returns the path to the written transferred tensor.  Raises
    :class:`FileNotFoundError` when the source fit is missing,
    :class:`ManifoldFormatError` when ``alignment`` is empty or covers no
    fitted layer, :class:`FileExistsError` when a transferred tensor
    already exists and ``force`` is ``False``.
    """
    from dataclasses import replace as _dc_replace

    from saklas.core.manifold import (
        LayerSubspace,
        eval_rbf,
        layer_lever,
        load_manifold,
        save_manifold,
    )
    from saklas.io.paths import safe_model_id, tensor_filename

    folder = Path(folder)
    safe_from = safe_model_id(from_model)
    src_tensor = folder / f"{safe_from}.safetensors"
    if not src_tensor.exists():
        raise FileNotFoundError(
            f"manifold {folder.name!r} has no fit for source model "
            f"{from_model!r} at {src_tensor}"
        )
    if not alignment:
        raise ManifoldFormatError(
            f"transfer_manifold: alignment map for {from_model!r} → "
            f"{to_model!r} is empty"
        )

    src = load_manifold(src_tensor)

    # Map each covered layer's subspace into target space.  ``M_L`` is
    # ``(D_tgt, D_src)`` so ``mean_tgt = M_L @ mean_src`` and each basis
    # row transforms the same way → ``basis_tgt = basis_src @ M_L^T``.
    new_layers: dict[int, LayerSubspace] = {}
    new_ev: dict[int, float] = {}
    for layer, sub in src.layers.items():
        M_L = alignment.get(layer)
        if M_L is None:
            continue
        M = M_L.to(dtype=torch.float32)
        mean_f = sub.mean.to(torch.float32)
        basis_f = sub.basis.to(torch.float32)
        mean_tgt = (M @ mean_f).to(dtype=sub.mean.dtype)
        basis_tgt = (basis_f @ M.transpose(0, 1)).to(dtype=sub.basis.dtype)
        new_layers[layer] = _dc_replace(sub, mean=mean_tgt, basis=basis_tgt)
        if layer in src.explained_variance:
            new_ev[layer] = src.explained_variance[layer]

    if not new_layers:
        raise ManifoldFormatError(
            f"transfer_manifold: alignment for {from_model!r} → {to_model!r} "
            f"covered none of the source manifold's fitted layers "
            f"({sorted(src.layers)})"
        )

    # The source model's Mahalanobis share + steering lever are per-model
    # (Σ and the neutral activations are both ``from_model`` quantities),
    # so they're invalid in ``to_model`` space.  When a **target** whitener
    # is supplied and covers every transferred layer (all-or-nothing,
    # mirroring the fit gate), recompute both in target space; otherwise
    # clear both — the apply path then falls back to the Euclidean
    # centroid-spread share + ``N = 1`` lever, which is metric-valid for
    # the target.  (EV is a fit-quality ratio, not a per-model metric, so
    # it carries either way.)
    rebake = whitener is not None and whitener.covers_all(new_layers.keys())
    new_share: dict[int, float] = {}
    new_lever: dict[int, float] = {}
    if rebake:
        assert whitener is not None  # narrowed by ``rebake``
        for layer, sub_tgt in new_layers.items():
            sub_f = sub_tgt.to(device=torch.device("cpu"), dtype=torch.float32)
            # ``coords`` are the reduced node values the RBF interpolates
            # (subspace-coordinate space — invariant under the model-space
            # alignment, so identical to the source fit).  ``M_R`` restricts
            # the *target* Σ⁻¹ to the transferred basis; the whitened share
            # is ``sqrt(Σ_k coords_kᵀ M_R coords_k)`` — the same formula the
            # fit pipeline bakes, now in target space.
            coords = eval_rbf(
                sub_f.node_params, sub_f.rbf_weights, sub_f.poly_coeffs,
                sub_f.node_params,
            )  # (K, R)
            gram = whitener.subspace_gram(layer, sub_f.basis)  # (R, R)
            quad = float(
                (coords @ gram * coords).sum().clamp_min(0.0).item()
            )
            new_share[layer] = quad ** 0.5
            # Lever needs the raw target neutral activations — uncenter the
            # whitener's centered observations by the target layer mean.
            # Skipped (left absent → that layer contributes ``N`` via the
            # share-weighted mean) when the layer mean isn't resolvable.
            if layer_means is not None and layer in layer_means:
                X_c, _K, _lam = whitener.woodbury_factors(
                    layer, device=torch.device("cpu"), dtype=torch.float32,
                )
                mu = layer_means[layer].to(
                    device="cpu", dtype=torch.float32,
                ).reshape(-1)
                new_lever[layer] = layer_lever(
                    X_c + mu, sub_f.mean, sub_f.basis,
                )

    transferred = _dc_replace(
        src, layers=new_layers, explained_variance=new_ev,
        mahalanobis_share=new_share, lever=new_lever,
        # ``origin`` is the per-layer foot of the *source* model's neutral
        # mean — a per-model quantity invalid in target space (same reason
        # share + lever are cleared); the apply path falls back to a
        # zero-coord seed per layer.
        origin={},
    )

    out_path = folder / tensor_filename(to_model, transferred_from=from_model)
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists; pass force=True to overwrite"
        )

    # Carry the discover-mode per-model layout (``node_coords``) and the
    # source sidecar's provenance fields into the transferred tensor, then
    # stamp the transfer method + source id.  ``save_manifold`` reads
    # provenance keys off this metadata dict.
    metadata: dict[str, object] = dict(src.metadata)
    src_sidecar_path = src_tensor.with_suffix(".json")
    if src_sidecar_path.exists():
        with open(src_sidecar_path) as f:
            metadata.update(json.load(f))
    metadata["method"] = "manifold_procrustes_transfer"
    metadata["source_model_id"] = from_model
    # Record the *target* share metric (the source sidecar's value rode in
    # via the ``metadata.update`` above and would be misleading).
    # ``subspace_metric`` is left as the source carried it — the basis was
    # selected on the source model and only rotated here, so its selection
    # metric is unchanged by the transfer.
    metadata["share_metric"] = "mahalanobis" if rebake else "euclidean"
    metadata["nodes_sha256"] = ManifoldFolder.load(folder).nodes_sha256()
    if transfer_quality_estimate is not None:
        metadata["transfer_quality_estimate"] = float(transfer_quality_estimate)

    save_manifold(transferred, out_path, metadata)
    # ``save_manifold`` only persists a fixed sidecar key allow-list
    # (``method`` / ``nodes_sha256`` / sae / fit_mode / hyperparams /
    # diagnostics / node_roles) — the transfer-provenance fields aren't
    # in it.  Patch them in afterward so a consumer can see where the
    # tensor came from, mirroring ``Sidecar.source_model_id`` /
    # ``transfer_quality_estimate`` on the vector path.
    sidecar_path = out_path.with_suffix(".json")
    with open(sidecar_path) as f:
        sc_data = json.load(f)
    sc_data["source_model_id"] = from_model
    if transfer_quality_estimate is not None:
        sc_data["transfer_quality_estimate"] = float(transfer_quality_estimate)
    write_json_atomic(sidecar_path, sc_data)
    # Refresh the folder integrity manifest so the new tensor + sidecar
    # are covered (mirrors the fit path).  The sidecar patch above must
    # happen *before* this re-hash so the manifest covers the final bytes.
    ManifoldFolder.load(folder).write_metadata()
    return out_path


# ============================================================ shared summary serializer ===


def manifold_summary(folder: Path) -> dict[str, Any]:
    """Session-independent summary of a manifold folder.

    The shared serializer behind ``vector manifold show -j`` (CLI) and
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
    ``source`` / ``fit_mode`` / ``is_discover`` / ``domain`` /
    ``domain_label`` / ``intrinsic_dim`` / ``min_nodes`` / ``node_count`` /
    ``node_labels`` / ``node_coords`` / ``node_roles`` / ``hyperparams`` /
    ``fitted_models`` / ``tensor_variants``.

    ``namespace`` is read off the folder's parent directory name.  Raises
    :class:`ManifoldFormatError` on a malformed folder.
    """
    folder = Path(folder)
    mf = ManifoldFolder.load(folder)
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
        "hyperparams": dict(mf.hyperparams),
        "fitted_models": fitted_models,
        "tensor_variants": tensor_variants,
    }


# ====================================================== bundled materialization ===
#
# Parallel to ``saklas.io.packs.materialize_bundled`` but for the
# manifold artifact kind.  Bundled manifolds live under
# ``saklas/data/manifolds/<name>/`` in the wheel and materialize into
# ``~/.saklas/manifolds/default/<name>/`` on session startup.  JSON-only
# on the shipped side — per-model ``.safetensors`` fits are produced on
# the user's machine via ``saklas vector manifold discover``.


def bundled_manifold_names() -> list[str]:
    """List every manifold shipped under ``saklas/data/manifolds/``."""
    try:
        root = _resources.files("saklas.data.manifolds")
    except (ModuleNotFoundError, FileNotFoundError):
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "manifold.json").is_file()
    )


def _canonical_json_sha256(data: bytes) -> str:
    """Content-stable sha256 of a JSON byte payload.

    Hashes the canonical-JSON form (sorted keys, no surrounding
    whitespace) so cosmetic-only differences (key order, indent, trailing
    newline) compare equal.  Falls back to a raw sha256 if the bytes don't
    parse as JSON, so unparseable on-disk content is treated as "user
    edited" rather than silently overwritten.

    Twin: :func:`saklas.io.packs._canonical_json_sha256` is the
    byte-equivalent helper for the decoupled pack format — kept separate
    (rather than importing the private cross-module name) so the two
    formats can churn independently; mirror any change to one in the other.
    """
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(_canonical_json(parsed)).hexdigest()


def _copy_bundled_manifold_fresh(pkg_root: Any, target: Path) -> None:
    """Fresh install of a bundled manifold — copy every shipped file."""
    target.mkdir(parents=True, exist_ok=True)
    for entry in pkg_root.iterdir():
        if entry.is_file():
            write_bytes_atomic(target / entry.name, entry.read_bytes())
        elif entry.is_dir() and entry.name == "nodes":
            nodes_dir = target / "nodes"
            nodes_dir.mkdir(parents=True, exist_ok=True)
            for node_file in entry.iterdir():
                if node_file.is_file():
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
        if not node_file.is_file():
            continue
        bundled_names.add(node_file.name)
        write_bytes_atomic(target_nodes / node_file.name, node_file.read_bytes())
    # Drop any on-disk node files that aren't in the new bundle (label
    # set shrank or rename happened).  Without this, an old roster's
    # files would linger and confuse the loader.
    for stale in target_nodes.iterdir():
        if stale.is_file() and stale.name not in bundled_names:
            stale.unlink()


def materialize_bundled_manifolds() -> None:
    """Copy bundled manifolds into ``~/.saklas/manifolds/default/``.

    For each ``saklas/data/manifolds/<name>/`` in the wheel, ensure
    ``~/.saklas/manifolds/default/<name>/`` is current.  Mirrors
    :func:`saklas.io.packs.materialize_bundled` for the manifold artifact
    kind; only touches ``manifold.json`` and ``nodes/*.json`` since
    bundled manifolds ship JSON-only (no per-model ``.safetensors`` —
    those are user-side fits).

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
    (``saklas vector manifold generate ...`` or hand-author under
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
        target = default_dir / name
        pkg_root = _resources.files("saklas.data.manifolds").joinpath(name)

        if not target.exists():
            _copy_bundled_manifold_fresh(pkg_root, target)
            continue

        on_disk_manifest = target / "manifold.json"
        if not on_disk_manifest.exists():
            # Folder exists without a manifold.json — refuse to fabricate one.
            continue

        try:
            with open(on_disk_manifest) as f:
                on_disk_payload = json.load(f)
        except Exception:
            # Corrupt; don't stomp user state.
            continue

        bundled_manifest_bytes = (pkg_root / "manifold.json").read_bytes()
        on_disk_manifest_bytes = on_disk_manifest.read_bytes()
        manifest_changed = (
            _canonical_json_sha256(on_disk_manifest_bytes)
            != _canonical_json_sha256(bundled_manifest_bytes)
        )
        fmt = on_disk_payload.get("format_version")
        format_stale = isinstance(fmt, int) and fmt < MANIFOLD_FORMAT_VERSION

        if not manifest_changed and not format_stale:
            # Same bundle, current format — nothing to do.
            continue

        # Bundle update — manifest moved or format_version bumped.  Both
        # cases want the new bundled state to win; user node-edits are
        # interpreted as stale-against-old-bundle and replaced.
        write_bytes_atomic(
            on_disk_manifest.with_suffix(".json.bak"), on_disk_manifest_bytes,
        )
        write_bytes_atomic(on_disk_manifest, bundled_manifest_bytes)

        _refresh_all_bundled_nodes(pkg_root, target)

        # Re-copy other top-level shipped files (e.g. scenarios.json
        # provenance) that aren't manifold.json or under nodes/.
        for entry in pkg_root.iterdir():
            if not entry.is_file():
                continue
            if entry.name == "manifold.json":
                continue
            write_bytes_atomic(target / entry.name, entry.read_bytes())

        reason = (
            f"v{fmt}->v{MANIFOLD_FORMAT_VERSION} (format_version)"
            if format_stale
            else "manifest content changed"
        )
        # Unlike ``packs.materialize_bundled`` (which preserves a
        # user-edited ``statements.json``), the bundle-update path here
        # re-copies every node file unconditionally — a node "edit" is
        # stale-against-old-bundle once the manifest moves.  That clobber
        # is intentional, but a user who hand-edited a bundled node corpus
        # deserves to *see* that their edit was overwritten, so this is a
        # user-facing warning rather than an INFO log they'd never notice.
        warnings.warn(
            f"materialize_bundled_manifolds: refreshed default/{name} — "
            f"{reason}; any local edits to its node corpus were overwritten "
            f"(fork under local/ to keep a custom corpus)",
            UserWarning,
            stacklevel=2,
        )
        _log.warning(
            "materialize_bundled_manifolds: refreshed default/%s — %s "
            "(node corpus re-copied, local edits overwritten)",
            name, reason,
        )


__all__ = [
    "MANIFOLD_FORMAT_VERSION",
    "min_nodes",
    "ManifoldFormatError",
    "ManifoldSidecar",
    "ManifoldFolder",
    "hash_manifold_files",
    "iter_manifold_folders",
    "create_manifold_folder",
    "create_discover_manifold_folder",
    "init_discover_manifold_folder",
    "append_discover_manifold_node",
    "write_manifold_scenarios",
    "read_manifold_scenarios",
    "plan_discover_generation",
    "DiscoverGenerationPlan",
    "merge_discover_manifolds",
    "update_manifold_folder",
    "clear_manifold_tensors",
    "remove_manifold_folder",
    "refresh_manifold",
    "transfer_manifold",
    "manifold_summary",
    "domain_label",
    "bundled_manifold_names",
    "materialize_bundled_manifolds",
    "_sanitize_hyperparams",
]
