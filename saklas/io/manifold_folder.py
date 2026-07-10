"""On-disk format core for manifold-steering artifacts.

The dataclasses (:class:`ManifoldFolder` / :class:`ManifoldSidecar`),
integrity/sha256 helpers, ``manifold.json`` load/save, and the shared
module-private validators + payload builders that the authoring and
lifecycle layers both reuse.  This is the dependency root of the split —
both :mod:`saklas.io.manifold_authoring` and
:mod:`saklas.io.manifold_lifecycle` import from here.

A *manifold* is a set of labeled nodes — each node a small corpus of
statements — placed at authoring coordinates on a :class:`ManifoldDomain`
(an n-dimensional intrinsic manifold of some topology: a box/disk, a
cylinder, a torus, a sphere, or an explicit immersion).  Fitting a
manifold against a model produces a per-model RBF artifact (see
:mod:`saklas.core.manifold`).  Manifolds live under their own root,
``~/.saklas/manifolds/<ns>/<name>/``, parallel to the legacy ``vectors/``
port root — a manifold carries N labeled nodes on a domain, not a single
bipolar concept (the retired pack/concept folder shape).

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
``saklas manifold fit`` produces the per-model tensors and
back-fills the ``files`` integrity manifest.  Tensor save/load itself
lives in :mod:`saklas.core.manifold` (``save_manifold`` / ``load_manifold``);
this module owns folder discovery, the node corpus, and integrity.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from saklas.core.errors import SaklasError
from saklas.core.manifold import BoxDomain, domain_from_spec
from saklas.core.role_templates import _ROLE_SLUG_RE
from saklas.io.atomic import write_json_atomic
from saklas.io.packs import NAME_REGEX, hash_file, verify_integrity

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

# Discover-mode fit modes — set as ``manifold.json::fit_mode`` for
# manifolds whose node coordinates are derived from the model's
# activations rather than authored by hand.  Authored manifolds carry
# ``fit_mode == "authored"`` (or omit the field, which means the same).
# ``auto`` is a discover mode whose *resolved* geometry (flat ``pca`` vs curved
# ``spectral``, plus periodic ``BoxDomain`` axes) is chosen per-model at fit
# time by ``core.manifold.select_topology`` — the folder declares only the
# corpus + hyperparams, exactly like ``pca``/``spectral``.
_FIT_MODES_DISCOVER: frozenset[str] = frozenset({"pca", "spectral", "auto"})
# A baked manifold is a pre-fitted, corpus-less artifact: its geometry is
# frozen in the per-model tensor and it can never re-fit (no node corpus to
# pool from).  Merge outputs and imported control vectors land this way — a
# steering *direction* is the K=1 affine case (see
# ``core/vectors.fold_directions_to_subspace``).  It is deliberately NOT a
# discover mode (``is_discover`` stays False — there are no per-model coords
# to derive).
_FIT_MODES_BAKED: frozenset[str] = frozenset({"baked"})
_FIT_MODES_ALL: frozenset[str] = (
    frozenset({"authored"}) | _FIT_MODES_DISCOVER | _FIT_MODES_BAKED
)

# Per-method hyperparameter whitelists.  Anything outside the whitelist
# for a given fit_mode is dropped at folder-create time so a user
# POSTing ``{fit_mode: "pca", hyperparams: {"k_nn": 5}}`` doesn't land
# a foreign key that would then crash ``derive_pca_coords`` at fit
# time with ``TypeError: unexpected keyword argument``.  ``max_dim``
# is shared.  Coordinate derivation is layer-agnostic — the consensus
# Gram averages every fit layer (``ManifoldExtractionPipeline.fit``), so
# there is no reference-layer knob.
# ``max_dim`` caps the derived intrinsic (layout) dim for both modes.  A
# flat (``pca``) fit's per-layer steerable subspace *is* its k-dim layout
# span, so ``max_dim`` is its only dim knob — there is no separate
# ``max_subspace_dim`` for pca (the subspace dim equals the layout dim).
# ``max_subspace_dim`` survives only for the curved (``spectral``) fit,
# where the per-layer RBF subspace can carry off-surface dims beyond the
# intrinsic coordinate count.  ``min_dim`` (spectral only) floors the
# layout dim the eigenvalue-ratio cliff derives — for an authored-
# dimensionality manifold (PAD's P×A×D) the cliff undershoots when one
# mode dominates the spectrum, so the floor keeps the declared geometry
# (set ``min_dim == max_dim`` to pin the dim exactly).  The steer-time
# origin is always the projection of the per-model neutral mean onto the
# subspace (the affine fit neutral-anchors the frame), so there is no
# ``anchor_origin`` knob.
_HYPERPARAMS_BY_MODE: dict[str, frozenset[str]] = {
    "pca": frozenset({
        "max_dim", "var_threshold",
    }),
    "spectral": frozenset({
        "max_dim", "min_dim", "k_nn", "bandwidth", "max_subspace_dim",
        "smoothing",
    }),
    # ``auto`` routes to ``select_topology`` (flat/curved by GCV + periodic by
    # persistent homology); it accepts the union of the knobs its candidate
    # fits consume — ``max_dim`` (layout dim cap), ``smoothing`` (curved-fit
    # GCV λ), ``persistence_frac`` (H1 loop-significance threshold), plus the
    # spectral graph knobs and ``max_subspace_dim`` for a resolved curved fit.
    "auto": frozenset({
        "max_dim", "var_threshold", "min_dim", "k_nn", "bandwidth",
        "max_subspace_dim", "smoothing", "persistence_frac",
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


def sanitize_hyperparams(
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


# Back-compat alias: the function was renamed to the public ``sanitize_hyperparams``.
_sanitize_hyperparams = sanitize_hyperparams

# Manifold artifact format version.  Decoupled from concept packs'
# ``PACK_FORMAT_VERSION`` so the two formats can churn independently.
# v3 is the arbitrary-dimensional / arbitrary-topology format (domain
# spec + per-node coordinates); v2 and earlier were the 1-D cyclic-spline
# format and must be converted with ``scripts/upgrade_manifolds.py``.
# v4 added a per-layer ``explained_variance_per_layer`` sidecar field
# (a fit-quality ratio).  It is no longer read — cross-layer read
# weighting now rides the Mahalanobis ``share`` (the same per-layer
# budget that drives steering), so EV is neither baked nor loaded.  An
# old sidecar that still carries the key loads fine (it's ignored); a
# refit simply drops it.
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
    (``[a-z0-9._-]+``).  Family-unsupported (Mistral-3) is *not*
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


_NODE_KINDS = ("abstract", "concrete", "custom")


def _validate_node_kind(name: str, label: str, kind: Any) -> str | None:
    """Validate an optional per-node ``kind`` field.

    ``None`` / missing means "unspecified".  A non-empty value must be one of
    :data:`_NODE_KINDS` (``"abstract"`` / ``"concrete"`` / ``"custom"``).
    Generation-time provenance only — it selects the system template and
    elicitation role label when authoring a node's conversational corpus
    (``custom`` = a caller-supplied system prompt, no role swap), and never
    feeds the fit.
    """
    if kind is None:
        return None
    if not isinstance(kind, str) or kind not in _NODE_KINDS:
        raise ManifoldFormatError(
            f"manifold {name!r} node {label!r} kind {kind!r} invalid; "
            f"must be one of {_NODE_KINDS}"
        )
    return kind


class ManifoldFormatError(ValueError, SaklasError):
    """Raised when a manifold folder is malformed or fails integrity."""


class BakedManifoldError(ValueError, SaklasError):
    """Raised when an operation invalid for a corpus-less baked manifold runs.

    A baked manifold (``fit_mode == "baked"``) has no node corpus and its
    geometry lives only in the per-model tensor, so a re-fit is impossible.
    Tensor-deleting lifecycle ops (``clear``, scoped ``refresh``) would
    therefore destroy the only copy of the geometry irreversibly — they
    refuse with this error and point the caller at ``manifold rm`` instead.
    """


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
            out[entry.name] = hash_file(entry)
    return out


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
    # Per-layer whitened between-node spread ``{str(L): tr(G_L)}`` — the
    # concept's signal-concentration profile across the stack, in
    # background-σ² units (comparable across layers).  A diagnostic readout
    # of "where does this concept live", distinct from the apply-time
    # ``mahalanobis_share`` (which restricts the same whitened spread to the
    # steerable subspace).  Empty on fits that predate it.
    node_spread_per_layer: dict[str, Any] = field(default_factory=dict)
    # Merge provenance on a ``fit_mode="baked"`` manifold — the
    # ``{coord: {alpha, project_away, tensor_sha256}}`` map written by
    # :func:`saklas.io.merge.merge_into_manifold`.  ``None`` on every fit that
    # isn't a merge (the common case).  Informational only — a baked
    # manifold never re-fits, so nothing branches on it.
    components: Optional[dict[str, Any]] = None
    # Per-node assistant-role substitution used at fit time, in
    # ``node_labels`` index order.  ``None`` for a given node (and an
    # empty list as a whole) = "standard assistant baseline" — the
    # default, byte-identical to today's non-role manifolds.  The same
    # information rides ``ManifoldFolder.node_roles`` but the sidecar
    # carries an independent copy so a downstream consumer
    # (``manifold show``, the webui inspector) doesn't have to
    # round-trip through the folder to know which role each node was
    # pooled under.
    node_roles: list[str | None] = field(default_factory=list)
    # Per-node conceptual kind ("abstract"/"concrete"), ``node_labels`` order.
    # Mirrors ``node_roles``' independent-copy rationale; generation provenance.
    node_kinds: list[str | None] = field(default_factory=list)

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
            node_spread_per_layer=dict(data.get("node_spread_per_layer", {})),
            node_roles=list(data.get("node_roles", [])),
            node_kinds=list(data.get("node_kinds", [])),
            components=data.get("components"),
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
    # Per-node conceptual ``kind`` — ``"abstract"`` (trait) or ``"concrete"``
    # (entity), aligned with ``node_labels`` index-by-index.  ``None`` =
    # unspecified.  Generation-time only (system template + elicitation role
    # label); never consumed by the fit.  An all-``None`` list is byte-identical
    # to a folder authored before the distinction existed.
    node_kinds: list[str | None] = field(default_factory=list)
    # Category tags, mirroring :attr:`saklas.io.packs.PackMetadata.tags`.
    # Carried so category-grouped probe bootstrap
    # (``load_default_manifolds`` -> ``_bootstrap_manifold_probes``) keeps
    # working once a steering vector lives as a 2-node ``pca`` manifold.
    # Optional/additive — the
    # loader defaults ``[]``; a tagless manifold stays byte-identical.
    tags: list[str] = field(default_factory=list)
    # Reference to a standalone template artifact
    # (``saklas.io.templates.TemplateFolder``, ``<ns>/<name>`` or bare name).
    # A *templated* discover manifold's node corpora are the slot-filled
    # assistant turns derived from that template; the manifold stores the derived
    # corpus in ``nodes/`` (like any discover folder) and keeps this reference so
    # the fit can resolve the template's **multi-turn contexts** as the
    # per-manifold elicitation prefixes (the template analogue of the shared
    # baseline prompts). The template is the single authoring source; the corpus
    # is its materialization. ``None`` for every non-templated manifold, which
    # keeps the manifest byte-identical to the pre-template shape. The resolved
    # template's content hash folds into ``nodes_sha256`` (a context edit re-fits)
    # and the ref is preserved across re-fits by ``write_metadata``.
    template_ref: str | None = None
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
                f"need >= {MANIFOLD_FORMAT_VERSION}. Regenerate it with the "
                f"current saklas — run scripts/upgrade_manifolds.py for a legacy "
                f"pack, or re-fit a discover manifold."
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
        node_kinds: list[str | None] = []
        hyperparams: dict[str, Any] = {}
        domain_spec: dict[str, Any]

        if fit_mode == "authored":
            domain_spec_in: Any = data.get("domain") or {}
            if not isinstance(domain_spec_in, dict) or not domain_spec_in:
                raise ManifoldFormatError(
                    f"authored manifold {name!r} needs a 'domain' object"
                )
            domain_spec = domain_spec_in
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
                node_kinds.append(_validate_node_kind(name, label, entry.get("kind")))
            _warn_authoring_quality(name, domain, node_coords)
        elif fit_mode == "baked":
            # Baked mode: a pre-fitted, corpus-less artifact (merge output /
            # imported control vector).  Like authored it carries a display
            # ``domain`` (the ``CustomDomain`` the fold produced) so the
            # inspector/summary have geometry to report; like discover its
            # nodes are label-only (no ``coords`` — the real per-layer node
            # coords live baked in the tensor).  There is no corpus and no
            # re-fit, so ``min_nodes`` and the authoring-quality advisory
            # are both inapplicable.
            domain_spec_in = data.get("domain") or {}
            if not isinstance(domain_spec_in, dict) or not domain_spec_in:
                raise ManifoldFormatError(
                    f"baked manifold {name!r} needs a 'domain' object"
                )
            domain_spec = domain_spec_in
            try:
                domain_from_spec(domain_spec)
            except (ValueError, KeyError) as e:
                raise ManifoldFormatError(
                    f"manifold {name!r} has an invalid domain: {e}"
                ) from e
            for entry in nodes:
                if not isinstance(entry, dict):
                    raise ManifoldFormatError(
                        f"baked manifold {name!r} node {entry!r} must be an "
                        f"object with 'label'"
                    )
                label = entry.get("label")
                if not isinstance(label, str) or not _LABEL_REGEX.match(label):
                    raise ManifoldFormatError(
                        f"baked manifold {name!r} node label {label!r} "
                        f"invalid; must match {_LABEL_REGEX.pattern}"
                    )
                if "coords" in entry:
                    raise ManifoldFormatError(
                        f"baked manifold {name!r} node {label!r} must not "
                        f"carry 'coords' — baked geometry lives in the tensor"
                    )
                node_labels.append(label)
                node_roles.append(_validate_node_role(name, label, entry.get("role")))
                node_kinds.append(_validate_node_kind(name, label, entry.get("kind")))
        else:
            # Discover mode: no ``domain`` field, no per-node ``coords``.
            # The fit pipeline derives coords per-model from the
            # activations and wraps them in a ``CustomDomain(k)``.  We
            # do not even know the intrinsic dimension until after the
            # fit, so the ``min_nodes(k)`` floor is enforced at fit time
            # (once ``k`` is picked) rather than here.
            if data.get("domain"):
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
                node_kinds.append(_validate_node_kind(name, label, entry.get("kind")))

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

        raw_tags = data.get("tags", [])
        if not isinstance(raw_tags, list) or not all(
            isinstance(t, str) for t in raw_tags
        ):
            raise ManifoldFormatError(
                f"manifold {name!r} 'tags' must be a list of strings"
            )

        # Optional reference to a standalone template artifact — present only on
        # templated discover manifolds. The node corpora were derived from it at
        # authoring time and live in ``nodes/`` like any discover folder; this
        # ref lets the fit resolve the template's multi-turn contexts as the
        # elicitation prefixes. Only discover folders may carry it.
        raw_template_ref = data.get("template_ref")
        template_ref: str | None = None
        if raw_template_ref is not None:
            if not isinstance(raw_template_ref, str) or not raw_template_ref:
                raise ManifoldFormatError(
                    f"manifold {name!r} 'template_ref' must be a non-empty string"
                )
            if fit_mode not in _FIT_MODES_DISCOVER:
                raise ManifoldFormatError(
                    f"manifold {name!r} carries a 'template_ref' but fit_mode is "
                    f"{fit_mode!r}; templated corpora are a discover-mode feature "
                    f"({sorted(_FIT_MODES_DISCOVER)})"
                )
            template_ref = raw_template_ref

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
            node_kinds=node_kinds,
            tags=[str(t) for t in raw_tags],
            template_ref=template_ref,
        )

        # Every node file must be present — except for a baked manifold,
        # which has no ``nodes/`` corpus at all (its geometry is the tensor).
        if fit_mode != "baked":
            for idx, _label in enumerate(node_labels):
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

        # A baked manifold's tensor is its entire reason to exist (no corpus
        # to re-fit from), so a tensor-less baked folder is corrupt, not a
        # legitimate pre-fit shape the way a fresh authored/discover folder is.
        if fit_mode == "baked" and not inst._sidecars:
            raise ManifoldFormatError(
                f"baked manifold {name!r} has no fitted tensor — a corpus-less "
                f"manifold carries its geometry only in the per-model tensor"
            )
        return inst

    # -- node corpus -------------------------------------------------------

    def node_path(self, index: int) -> Path:
        return self.folder / "nodes" / _node_filename(
            index, self.node_labels[index],
        )

    def node_groups(self) -> list[tuple[str, list[str]]]:
        """Return ``[(label, statements), ...]`` in node order."""
        if self.fit_mode == "baked":
            raise ManifoldFormatError(
                f"baked manifold {self.name!r} has no node corpus to read"
            )
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
        if self.fit_mode == "baked":
            # No corpus and no re-fit: the key is provenance-only (the
            # sidecar records it, and it always matches since baked never
            # re-fits).  Hash the identity that distinguishes it.
            h.update(_canonical_json({
                "fit_mode": "baked",
                "node_labels": list(self.node_labels),
            }))
            return h.hexdigest()
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
        # Per-node kind selects the generation system template + elicitation
        # role label, so it shapes the corpus a re-fit would pool — a kind edit
        # must invalidate a cached fit.  All-``None`` hashes identically to a
        # missing field (same legacy shape).
        if any(k is not None for k in self.node_kinds):
            h.update(_canonical_json(self.node_kinds))
        # A templated manifold's elicitation prefixes are the referenced
        # template's multi-turn contexts (a fit input that the node corpus files
        # above don't capture — they're only the slotted assistant turns), so a
        # context/value edit must invalidate a cached fit. Fold the resolved
        # template's content hash; fall back to the ref string if it can't be
        # resolved (best-effort — a missing template fails loudly at fit time).
        # ``None`` (every non-templated manifold) hashes identically to a missing
        # field.
        if self.template_ref is not None:
            from saklas.io.templates import (
                AmbiguousTemplateError,
                TemplateNotFoundError,
                resolve_template,
            )
            try:
                h.update(resolve_template(self.template_ref).sha256().encode())
            except (TemplateNotFoundError, AmbiguousTemplateError):
                # The template genuinely can't be resolved (missing / a
                # cross-namespace collision); the ref string is the best
                # available staleness key, and the fit fails loudly at fit
                # time.  Any *other* error (a corrupt template, an IO fault)
                # is a real bug — let it propagate rather than silently
                # degrade the staleness key into a stale-but-passing hash.
                h.update(_canonical_json({"template_ref": self.template_ref}))
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
        # Category tags survive re-fit, written only when non-empty so a
        # tagless manifold stays byte-identical to the pre-tags shape.
        if self.tags:
            payload["tags"] = list(self.tags)
        # The template reference is fit-time provenance (its multi-turn contexts
        # are the elicitation prefixes), so it must survive the post-fit manifest
        # rewrite — written only when set, keeping non-templated manifests
        # byte-identical.
        if self.template_ref is not None:
            payload["template_ref"] = self.template_ref
        # Per-node ``role`` is written only when set — keeps the legacy
        # shape (every node carries ``{label, coords}`` or ``{label}``
        # only) byte-identical for non-role manifolds, and a stray
        # ``role: null`` doesn't leak into the manifest for a node that
        # opted out.
        if self.fit_mode == "authored":
            payload["domain"] = self.domain
            payload["nodes"] = [
                _node_payload_authored(label, coords, role, kind)
                for label, coords, role, kind in zip(
                    self.node_labels, self.node_coords,
                    self._roles_padded(), self._kinds_padded(),
                    strict=True,
                )
            ]
        elif self.fit_mode == "baked":
            # Display domain + label-only nodes (no coords, no hyperparams).
            payload["domain"] = self.domain
            payload["nodes"] = [
                _node_payload_discover(label, role, kind)
                for label, role, kind in zip(
                    self.node_labels, self._roles_padded(), self._kinds_padded(),
                    strict=True,
                )
            ]
        else:
            payload["hyperparams"] = self.hyperparams
            payload["nodes"] = [
                _node_payload_discover(label, role, kind)
                for label, role, kind in zip(
                    self.node_labels, self._roles_padded(), self._kinds_padded(),
                    strict=True,
                )
            ]
        write_json_atomic(self.folder / "manifold.json", payload)

    def update_file_hashes(self, *paths: Path) -> None:
        """Refresh only newly written fitted artifacts in the integrity manifest.

        A fit replaces one tensor/sidecar pair.  Re-reading every historical
        model and variant in the folder makes persistence scale with old
        artifacts rather than the work just completed; the existing manifest
        was already verified by :meth:`load`, so unchanged entries can be kept.
        A legacy/unfitted folder with no manifest gets one full population on
        its first write so pre-existing files do not become silently untracked.
        """
        if not self.files:
            self.write_metadata()
            return
        files = dict(self.files)
        for path in paths:
            resolved = Path(path)
            if resolved.parent != self.folder or not resolved.is_file():
                raise ValueError(
                    f"manifest update path must be a fitted file in {self.folder}: "
                    f"{resolved}"
                )
            files[resolved.name] = hash_file(resolved)
        self.write_metadata(files=files)

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

    def _kinds_padded(self) -> list[str | None]:
        """Return ``node_kinds`` padded to ``len(node_labels)`` with ``None``s.

        Defensive twin of :meth:`_roles_padded` — keeps the kind list aligned
        with the labels under in-memory node-list mutations.
        """
        if len(self.node_kinds) == len(self.node_labels):
            return list(self.node_kinds)
        return [None] * len(self.node_labels)


def _node_payload_authored(
    label: str, coords: list[float], role: str | None, kind: str | None = None,
) -> dict[str, Any]:
    """Build one authored-mode node entry for ``manifold.json``.

    ``role`` / ``kind`` are emitted only when set, so the legacy
    ``{label, coords}`` shape stays byte-identical for plain manifolds.
    """
    out: dict[str, Any] = {"label": label, "coords": [float(c) for c in coords]}
    if role is not None:
        out["role"] = role
    if kind is not None:
        out["kind"] = kind
    return out


def _node_payload_discover(
    label: str, role: str | None, kind: str | None = None,
) -> dict[str, Any]:
    """Build one discover-mode node entry for ``manifold.json``."""
    out: dict[str, Any] = {"label": label}
    if role is not None:
        out["role"] = role
    if kind is not None:
        out["kind"] = kind
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


@dataclass(frozen=True)
class DiscoverGenerationPlan:
    """Resume / extend plan for streaming discover-manifold generation.

    Returned by :func:`saklas.io.manifold_authoring.plan_discover_generation`.
    ``index_of`` maps each label to its on-disk node index (the ``NN_``
    filename prefix); ``pending`` is the declared labels whose corpus file
    is not yet on disk (the ones a run still needs to generate, in node
    order); ``scenarios`` is the locked domain list read back from
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
