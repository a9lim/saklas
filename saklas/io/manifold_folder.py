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

from contextlib import ExitStack, contextmanager
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
# ``fit_mode == "authored"``. Current manifests always carry the discriminator.
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
MERGE_BAKE_POLICY = "additive_union_v1"


@contextmanager
def _locked_manifest(folder: Path):
    """Serialize cross-process manifest read-modify-write for one folder."""
    from saklas.io.atomic import artifact_lock

    with artifact_lock(folder.parent / f"{folder.name}.manifest"):
        yield


def manifold_pair_lock_path(tensor_path: Path) -> Path:
    """Stable, bounded logical lock path for one fitted tensor pair.

    The lock lives under the manifold namespace rather than inside the
    removable manifold folder.  Consequently ``rm`` and stage-swap refreshes
    cannot unlink the held lock inode and let a later process acquire a fresh
    inode for the same logical pair.  The digest also bounds the filename for
    long model/release-derived tensor names.
    """

    tensor_path = Path(tensor_path).expanduser().resolve(strict=False)
    identity = f"{tensor_path.parent}\0{tensor_path.name}".encode()
    digest = hashlib.sha256(identity).hexdigest()
    return tensor_path.parent.parent / f".saklas-pair-{digest}"


@contextmanager
def manifold_pair_lock(tensor_path: Path):
    """Lock a fitted tensor/sidecar pair using stable external identity."""

    from saklas.io.atomic import artifact_lock

    with artifact_lock(manifold_pair_lock_path(tensor_path)):
        yield


def manifold_folder_tensor_paths(folder: Path) -> list[Path]:
    """Return canonical tensor paths for every fitted/orphan pair in ``folder``.

    Destructive folder replacement cannot trust ``manifold.json::files``: the
    operation is also the recovery path for an interrupted publication whose
    tensor or sidecar never acquired a manifest proof.  Scan both halves and
    canonicalize sidecars back to their tensor path so each logical pair is
    locked exactly once.
    """

    from saklas.io.paths import parse_tensor_filename

    folder = Path(folder)
    candidates = list(folder.glob("*.safetensors")) + [
        sidecar
        for sidecar in folder.glob("*.json")
        if sidecar.name != "manifold.json"
    ]
    tensors: set[Path] = set()
    for candidate in candidates:
        tensor = (
            candidate.with_suffix(".safetensors")
            if candidate.suffix == ".json" else candidate
        )
        if parse_tensor_filename(tensor.name) is not None:
            tensors.add(tensor)
    return sorted(tensors)


@contextmanager
def destructive_manifold_folder_transaction(folder: Path):
    """Quiesce fitted readers before a whole-folder reset or stage swap.

    Global mutation order is manifest then stable pair locks, with pair locks in
    deterministic tensor-path order.  The pair locks live outside the removable
    folder, so holding this context across ``rmtree``/rename/recreation prevents
    a new inode from bypassing a reader that still owns the old logical pair.
    The manifest lock is re-entrant, allowing authoring entry points that already
    serialize their destination to use this one shared destructive seam.
    """

    folder = Path(folder)
    with _locked_manifest(folder):
        with ExitStack() as stack:
            for tensor_path in manifold_folder_tensor_paths(folder):
                stack.enter_context(manifold_pair_lock(tensor_path))
            yield


def reset_manifold_folder(folder: Path) -> None:
    """Remove ``folder`` only after all extant fitted pairs are quiescent."""

    import shutil

    folder = Path(folder)
    with destructive_manifold_folder_transaction(folder):
        if folder.exists():
            shutil.rmtree(folder)

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
    """Validate and return hyperparameters for one exact fit mode.

    Single source of truth for the per-method whitelist; both the create
    and the fit-override paths funnel through this so the folder
    manifest never accepts a key that the selected dispatcher would ignore.
    """
    if hyperparams is None:
        return {}
    allowed = _HYPERPARAMS_BY_MODE.get(fit_mode)
    if allowed is None:
        raise ManifoldFormatError(f"unknown fit_mode {fit_mode!r}")
    invalid = set(hyperparams) - allowed
    if invalid:
        names = ", ".join(sorted(invalid))
        raise ManifoldFormatError(
            f"fit_mode {fit_mode!r} does not accept hyperparameter(s): {names}"
        )
    return dict(hyperparams)


# Current manifold artifact format. v7 makes manifest defaults and per-node
# provenance explicit, yielding one canonical payload shape. v6 added an optional per-layer
# ``affine_map`` tensor for flat fitted
# subspaces.  Ordinary fits omit it (identity); rectangular/non-isometric
# cross-model transfers persist the exact authoring-to-orthonormal-reduced
# reparameterization.  Old readers would otherwise ignore that tensor and
# silently move manifold world points, so this is a real format boundary.
MANIFOLD_FORMAT_VERSION = 7


def _validate_node_role(name: str, label: str, role: Any) -> str | None:
    """Validate an optional per-node ``role`` field.

    ``None`` means "use the standard assistant baseline". A non-empty string must match
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

    ``None`` means "unspecified". A non-empty value must be one of
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


def validate_manifold_format_version(
    value: Any, *, location: str = "manifold",
) -> int:
    """Validate the shared readable/writable manifold format boundary."""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value != MANIFOLD_FORMAT_VERSION
    ):
        raise ManifoldFormatError(
            f"{location} has format_version={value!r}; "
            f"need exactly {MANIFOLD_FORMAT_VERSION}. Regenerate or re-fit it "
            "with the current saklas."
        )
    return value


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
    sae_fingerprint: Optional[str] = None
    sae_ids_by_layer: dict[str, str] = field(default_factory=dict)
    sae_full_coverage: bool = False
    model_fingerprint: Optional[str] = None
    capture_sha256: Optional[str] = None
    fitted_layers: list[int] = field(default_factory=list)
    fit_policy_version: Optional[int] = None
    # Discover-mode-only fields.  ``None`` on authored fits.
    fit_mode: str = "authored"
    hyperparams: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Per-layer whitened between-node spread ``{str(L): tr(G_L)}`` — the
    # concept's signal-concentration profile across the stack, in
    # background-σ² units (comparable across layers).  A diagnostic readout
    # of "where does this concept live", distinct from the apply-time
    # ``mahalanobis_share`` (which restricts the same whitened spread to the
    # steerable subspace).  Empty when the current fit has no measurements.
    node_spread_per_layer: dict[str, Any] = field(default_factory=dict)
    # Merge provenance on a ``fit_mode="baked"`` manifold — the
    # ``{coord: {alpha, tensor_sha256}}`` map written by
    # :func:`saklas.io.merge.merge_into_manifold`.  ``None`` on every fit that
    # isn't a merge (the common case).  Informational only — a baked
    # manifold never re-fits, so nothing branches on it.
    components: Optional[dict[str, Any]] = None
    bake_policy: Optional[str] = None
    # Per-node assistant-role substitution used at fit time, in
    # ``node_labels`` index order.  ``None`` for a given node (and an
    # empty list as a whole) = "standard assistant baseline" — the
    # default. The same
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
        data = load_manifold_sidecar_data(path)
        return cls(
            method=data["method"],
            saklas_version=data["saklas_version"],
            domain=data["domain"],
            node_count=data["node_count"],
            node_labels=data["node_labels"],
            feature_space=data["feature_space"],
            nodes_sha256=data.get("nodes_sha256"),
            sae_release=data.get("sae_release"),
            sae_revision=data.get("sae_revision"),
            sae_fingerprint=data.get("sae_fingerprint"),
            sae_ids_by_layer=dict(data.get("sae_ids_by_layer", {})),
            sae_full_coverage=bool(data.get("sae_full_coverage", False)),
            model_fingerprint=data.get("model_fingerprint"),
            capture_sha256=data.get("capture_sha256"),
            fitted_layers=sorted(int(idx) for idx in data.get("fitted_layers", [])),
            fit_policy_version=data.get("fit_policy_version"),
            fit_mode=data["fit_mode"],
            hyperparams=dict(data.get("hyperparams", {})),
            diagnostics=dict(data.get("diagnostics", {})),
            node_spread_per_layer=dict(data.get("node_spread_per_layer", {})),
            node_roles=list(data.get("node_roles", [])),
            node_kinds=list(data.get("node_kinds", [])),
            components=data.get("components"),
            bake_policy=data.get("bake_policy"),
        )


def load_manifold_sidecar_data(path: Path) -> dict[str, Any]:
    """Read and validate the exact current fitted-manifold sidecar shape."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ManifoldFormatError(
            f"manifold sidecar {path} is unreadable: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ManifoldFormatError(
            f"manifold sidecar {path} must be a JSON object"
        )
    validate_manifold_format_version(
        data.get("format_version"), location=f"manifold sidecar {path}",
    )
    allowed = {
        "format_version", "name", "method", "saklas_version", "domain",
        "node_count", "node_labels", "feature_space", "fit_mode",
        "hyperparams", "diagnostics", "node_spread_per_layer",
        "mahalanobis_share_per_layer", "origin_per_layer", "nodes_sha256",
        "sae_release", "sae_revision", "sae_fingerprint", "sae_ids_by_layer",
        "sae_full_coverage", "model_fingerprint", "capture_sha256",
        "fitted_layers", "fit_policy_version", "share_metric",
        "subspace_metric", "rbf_smoothing_per_layer", "sigma_field_per_layer",
        "resolved_fit_mode", "topology_winner", "topology_candidates",
        "node_roles", "node_kinds", "components", "bake_policy",
        "source_model_id", "source_model_fingerprint",
        "transfer_quality_estimate",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ManifoldFormatError(
            f"manifold sidecar {path} has unknown field(s): {sorted(unknown)}"
        )
    required_types: dict[str, type] = {
        "name": str,
        "method": str,
        "saklas_version": str,
        "domain": dict,
        "node_count": int,
        "node_labels": list,
        "feature_space": str,
        "fit_mode": str,
        "hyperparams": dict,
        "diagnostics": dict,
        "node_spread_per_layer": dict,
    }
    for key, expected in required_types.items():
        value = data.get(key)
        if not isinstance(value, expected) or (
            expected is int and isinstance(value, bool)
        ):
            raise ManifoldFormatError(
                f"manifold sidecar {path} field {key!r} must be "
                f"{expected.__name__}"
            )
    if data["fit_mode"] not in {
        "authored", "pca", "spectral", "auto", "baked",
    }:
        raise ManifoldFormatError(
            f"manifold sidecar {path} has invalid "
            f"fit_mode={data['fit_mode']!r}"
        )
    labels = data["node_labels"]
    if any(not isinstance(label, str) for label in labels):
        raise ManifoldFormatError(
            f"manifold sidecar {path} node_labels must contain only strings"
        )
    if data["node_count"] != len(labels):
        raise ManifoldFormatError(
            f"manifold sidecar {path} node_count does not match node_labels"
        )
    for key in (
        "sae_ids_by_layer", "hyperparams", "diagnostics",
        "node_spread_per_layer",
    ):
        if key in data and not isinstance(data[key], dict):
            raise ManifoldFormatError(
                f"manifold sidecar {path} field {key!r} must be an object"
            )
    for key in ("fitted_layers", "node_roles", "node_kinds"):
        if key in data and not isinstance(data[key], list):
            raise ManifoldFormatError(
                f"manifold sidecar {path} field {key!r} must be an array"
            )
    return data


@dataclass
class ManifoldFolder:
    """A manifold artifact folder on disk.

    Discovery + corpus + integrity only — the fitted RBF tensors are
    loaded through :func:`saklas.core.manifold.load_manifold`.

    Two folder shapes share this class via the ``fit_mode`` field:

    - ``fit_mode == "authored"``: the
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
    # "use the standard assistant baseline". An all-``None`` list keeps the
    # ordinary behavior — the centroid pooling
    # just goes through the default chat-template branch.
    node_roles: list[str | None] = field(default_factory=list)
    # Per-node conceptual ``kind`` — ``"abstract"`` (trait) or ``"concrete"``
    # (entity), aligned with ``node_labels`` index-by-index.  ``None`` =
    # unspecified.  Generation-time only (system template + elicitation role
    # label); never consumed by the fit.
    node_kinds: list[str | None] = field(default_factory=list)
    # Category tags, mirroring :attr:`saklas.io.packs.PackMetadata.tags`.
    # Carried so category-grouped probe bootstrap
    # (``load_default_manifolds`` -> ``_bootstrap_manifold_probes``) keeps
    # working once a steering vector lives as a 2-node ``pca`` manifold.
    # Always explicit in the manifest; empty means uncategorized.
    tags: list[str] = field(default_factory=list)
    # Reference to a standalone template artifact
    # (``saklas.io.templates.TemplateFolder``, ``<ns>/<name>`` or bare name).
    # A *templated* discover manifold's node corpora are the slot-filled
    # assistant turns derived from that template; the manifold stores the derived
    # corpus in ``nodes/`` (like any discover folder) and keeps this reference so
    # the fit can resolve the template's **multi-turn contexts** as the
    # per-manifold elicitation prefixes (the template analogue of the shared
    # baseline prompts). The template is the single authoring source; the corpus
    # is its materialization. ``None`` marks every non-templated manifold. The resolved
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
    def load(
        cls, folder: Path, *, verify_manifest: bool = True,
    ) -> "ManifoldFolder":
        """Parse a manifold folder.

        Runtime/install/push callers keep the default full integrity walk.
        Metadata-only inventory, authoring, and lifecycle routing callers may set
        ``verify_manifest=False``; fit callers do likewise, hashing the live
        corpus into ``nodes_sha256`` and validating only the requested fitted
        tensor.  This avoids rereading every historical model/SAE payload when no
        payload is being consumed.  Structural checks and tensor/sidecar pairing
        still run either way.
        """
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
        if not isinstance(data, dict):
            raise ManifoldFormatError(
                f"manifold.json in {folder} must be a JSON object"
            )

        validate_manifold_format_version(
            data.get("format_version"),
            location=f"manifold.json in {folder}",
        )

        common_fields = {
            "format_version", "name", "description", "fit_mode", "nodes",
            "files", "source", "tags", "artifact_id", "fit_epochs",
            "template_ref",
        }

        name = data.get("name")
        if not isinstance(name, str) or not NAME_REGEX.match(name):
            raise ManifoldFormatError(
                f"manifold name {name!r} invalid; must match {NAME_REGEX.pattern}"
            )

        # Authored manifolds carry ``domain`` + per-node ``coords``. Discover
        # manifolds omit ``domain`` and derive coordinates per model at fit time.
        fit_mode = data.get("fit_mode")
        if fit_mode not in _FIT_MODES_ALL:
            raise ManifoldFormatError(
                f"manifold {name!r} fit_mode {fit_mode!r} invalid; "
                f"expected one of {sorted(_FIT_MODES_ALL)}"
            )
        mode_fields = (
            {"domain"} if fit_mode in {"authored", "baked"}
            else {"hyperparams"}
        )
        unknown = set(data) - common_fields - mode_fields
        if unknown:
            raise ManifoldFormatError(
                f"manifold {name!r} has unknown field(s): {sorted(unknown)}"
            )
        for key, expected in (
            ("description", str), ("files", dict), ("source", str),
            ("tags", list),
        ):
            if not isinstance(data.get(key), expected):
                raise ManifoldFormatError(
                    f"manifold {name!r} field {key!r} must be {expected.__name__}"
                )
        if not data["source"]:
            raise ManifoldFormatError(
                f"manifold {name!r} field 'source' must be non-empty"
            )
        if "artifact_id" in data and (
            not isinstance(data["artifact_id"], str) or not data["artifact_id"]
        ):
            raise ManifoldFormatError(
                f"manifold {name!r} field 'artifact_id' must be a non-empty str"
            )
        if "fit_epochs" in data and not isinstance(data["fit_epochs"], dict):
            raise ManifoldFormatError(
                f"manifold {name!r} field 'fit_epochs' must be dict"
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
                expected_node = {"label", "coords", "role", "kind"}
                if set(entry) != expected_node:
                    raise ManifoldFormatError(
                        f"manifold {name!r} node fields must be "
                        f"{sorted(expected_node)}"
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
                expected_node = {"label", "role", "kind"}
                if set(entry) != expected_node:
                    raise ManifoldFormatError(
                        f"baked manifold {name!r} node fields must be "
                        f"{sorted(expected_node)}"
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
            raw_hyperparams = data.get("hyperparams")
            if not isinstance(raw_hyperparams, dict):
                raise ManifoldFormatError(
                    f"discover manifold {name!r} field 'hyperparams' must be dict"
                )
            hyperparams = dict(raw_hyperparams)
            for entry in nodes:
                if not isinstance(entry, dict):
                    raise ManifoldFormatError(
                        f"discover manifold {name!r} node {entry!r} must "
                        f"be an object with 'label'"
                    )
                expected_node = {"label", "role", "kind"}
                if set(entry) != expected_node:
                    raise ManifoldFormatError(
                        f"discover manifold {name!r} node fields must be "
                        f"{sorted(expected_node)}"
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

        files = data["files"]
        if not isinstance(files, dict):
            raise ManifoldFormatError(
                f"manifold {name!r} 'files' must be an object"
            )
        # Verify only a populated manifest — a freshly hand-authored
        # manifold has no hashes yet; `fit` back-fills them.
        if files and verify_manifest:
            ok, bad = verify_integrity(folder, files)
            if not ok:
                raise ManifoldFormatError(
                    f"manifold integrity check failed in {folder}: "
                    f"tampered/missing {bad}"
                )

        raw_tags = data["tags"]
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
        if "template_ref" not in data:
            raise ManifoldFormatError(
                f"manifold {name!r} needs explicit 'template_ref'"
            )
        raw_template_ref = data["template_ref"]
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
            description=data["description"],
            domain=domain_spec,
            node_labels=node_labels,
            node_coords=node_coords,
            files=files,
            fit_mode=fit_mode,
            hyperparams=hyperparams,
            source=data["source"],
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

    def nodes_sha256(
        self, *, resolved_template_sha256: str | None = None,
    ) -> str:
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

        The current hash covers the complete canonical fit input shape.
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
        # Labels are part of the fitted artifact (steering-by-label, nearest
        # reads, roles/kinds alignment).  Node filenames are index-based, so a
        # rename can leave every corpus byte unchanged unless labels themselves
        # participate in the staleness key.
        h.update(_canonical_json(list(self.node_labels)))
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
        h.update(_canonical_json(self.node_roles))
        # Per-node kind selects the generation system template + elicitation
        # role label, so it shapes the corpus a re-fit would pool — a kind edit
        # must invalidate a cached fit.
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
            if resolved_template_sha256 is not None:
                h.update(resolved_template_sha256.encode())
                return h.hexdigest()
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
        """Rewrite authoring metadata while preserving trusted fitted hashes.

        Fitted writers must call :meth:`update_file_hashes` with their exact
        outputs. A metadata-only rewrite never scans and blesses unrelated
        top-level files. The locked latest manifest supplies ``files`` when an
        explicit trusted mapping is not provided.
        """
        with _locked_manifest(self.folder):
            manifest_path = self.folder / "manifold.json"
            try:
                with open(manifest_path) as handle:
                    latest_payload = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ManifoldFormatError(
                    f"cannot update unreadable manifest at {manifest_path}: {exc}"
                ) from exc
            if not isinstance(latest_payload, dict):
                raise ManifoldFormatError(
                    f"manifold manifest at {manifest_path} must be a JSON object"
                )
            validate_manifold_format_version(
                latest_payload.get("format_version"), location=str(manifest_path),
            )
            if files is None:
                latest = latest_payload.get("files", {})
                if not isinstance(latest, dict):
                    raise ManifoldFormatError(
                        "manifold files manifest is not an object"
                    )
                files = dict(latest)
            self.files = files
            payload: dict[str, Any] = {
                "format_version": MANIFOLD_FORMAT_VERSION,
                "name": self.name,
                "description": self.description,
                "fit_mode": self.fit_mode,
                "files": files,
                "source": self.source or "local",
                "tags": list(self.tags),
                "template_ref": self.template_ref,
            }
            for key in ("artifact_id", "fit_epochs"):
                if key in latest_payload:
                    payload[key] = latest_payload[key]
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
                        self.node_labels, self._roles_padded(),
                        self._kinds_padded(), strict=True,
                    )
                ]
            else:
                payload["hyperparams"] = self.hyperparams
                payload["nodes"] = [
                    _node_payload_discover(label, role, kind)
                    for label, role, kind in zip(
                        self.node_labels, self._roles_padded(),
                        self._kinds_padded(), strict=True,
                    )
                ]
            write_json_atomic(self.folder / "manifold.json", payload)

    def update_file_hashes(self, *paths: Path) -> None:
        """Refresh only newly written fitted artifacts in the integrity manifest.

        A fit replaces one tensor/sidecar pair.  Re-reading every historical
        model and variant in the folder makes persistence scale with old
        artifacts rather than the work just completed; the existing manifest
        was already verified by :meth:`load`, so unchanged entries can be kept.
        Starting from an empty manifest still records only the paths named by
        the successful writer.  Unrelated pre-existing pairs remain untrusted;
        implicitly hashing them here would launder an interrupted or tampered
        artifact merely because another model was fitted.
        """
        resolved_paths = [Path(path) for path in paths]
        for resolved in resolved_paths:
            if resolved.parent != self.folder or not resolved.is_file():
                raise ValueError(
                    f"manifest update path must be a fitted file in {self.folder}: "
                    f"{resolved}"
                )
        with _locked_manifest(self.folder):
            manifest_path = self.folder / "manifold.json"
            try:
                with open(manifest_path) as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ManifoldFormatError(
                    f"cannot update unreadable manifest at {manifest_path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ManifoldFormatError(
                    f"manifold manifest at {manifest_path} must be a JSON object"
                )
            # Reject an unknown schema before interpreting even familiar fields:
            # a future writer may have changed their shape or semantics.
            validate_manifold_format_version(
                payload.get("format_version"), location=str(manifest_path),
            )
            latest = payload.get("files", {})
            if not isinstance(latest, dict):
                raise ManifoldFormatError("manifold files manifest is not an object")
            # A newly published pair is always written by the current tensor
            # writer. Preserve the exact current format in the same locked CAS.
            files = dict(latest)
            for resolved in resolved_paths:
                files[resolved.name] = hash_file(resolved)
            payload["format_version"] = MANIFOLD_FORMAT_VERSION
            payload["files"] = files
            write_json_atomic(manifest_path, payload)
            self.files = files

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

    ``role`` and ``kind`` are explicit, including ``None``.
    """
    return {
        "label": label, "coords": [float(c) for c in coords],
        "role": role, "kind": kind,
    }


def _node_payload_discover(
    label: str, role: str | None, kind: str | None = None,
) -> dict[str, Any]:
    """Build one discover-mode node entry for ``manifold.json``."""
    return {"label": label, "role": role, "kind": kind}


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
    order); ``added`` is the labels appended to ``manifold.json`` this call
    (the add-nodes case);
    ``resumed`` is ``True`` when an existing ``manifold.json`` was found.
    """

    folder: Path
    index_of: dict[str, int]
    pending: tuple[str, ...]
    added: tuple[str, ...]
    resumed: bool
