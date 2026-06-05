"""Monitor — read-side counterpart to manifold steering.

Pure CPU tests: build a toy manifold with hand-controllable per-layer
geometry, verify the three channels (subspace_fraction, nearest_nodes,
inverse_projection) match hand-computed expectations, and confirm the
flat_scalars adapter produces the namespaced keys the grammar / gate
callback consumes.
"""
from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from saklas.core.manifold import (
    BoxAxis,
    BoxDomain,
    CustomDomain,
    LayerSubspace,
    Manifold,
    fit_layer_subspace as _fit_layer_subspace_with_ev,
)
from saklas.core.monitor import (
    DEFAULT_NEAREST_TOP_N,
    AttachedManifoldProbe,
    Monitor,
)
from saklas.core.results import ManifoldReading


def fit_layer_subspace(*args: Any, **kwargs: Any) -> Any:
    """Test helper: drop the EV ratio for callers that don't care."""
    sub, _ev = _fit_layer_subspace_with_ev(*args, **kwargs)
    return sub


def _iso_monitor(m: "Manifold") -> Monitor:
    """A ``Monitor`` wired with an isotropic whitener over ``m``'s
    fit layers.

    Manifold reads are Mahalanobis-only (4.0 collapse): ``add_probe`` →
    ``_build_whitened`` requires a covering whitener.  An *isotropic* one
    (Σ ≈ σ²·I) makes the whitened readout reduce to the Euclidean readout, so
    the geometric value assertions below (fraction = 1 inside / 0 outside,
    nearest-node distances) still hold to a loose tolerance.
    """
    from tests._whitener import isotropic_whitener
    dim = next(iter(m.layers.values())).mean.shape[0]
    whitener = isotropic_whitener(list(m.layers.keys()), dim)
    return Monitor(whitener=whitener)


def _node_world(m: Manifold, layer_idx: int) -> torch.Tensor:
    """World-space ``(K, D)`` node activations for a layer.

    The monitor used to pre-cache this as ``AttachedManifoldProbe
    .node_values_world``; that field was dead (only the reduced cache is
    read in scoring) and was removed.  Tests recompute it on demand from
    the manifold's RBF — exactly the ``sub.eval_at(domain.embed(coords))``
    the cache builder used.
    """
    sub = m.layers[layer_idx]
    coords = m.node_coords.to(torch.float32)
    embedded = m.domain.embed(m.domain.clamp_position(coords))
    return sub.eval_at(embedded)  # (K, D)


def _toy_manifold(
    *,
    n_layers: int = 2,
    n_nodes: int = 3,
    dim: int = 8,
    seed: int = 0,
) -> Manifold:
    """Build a 1-D BoxDomain manifold with 3 nodes at coords [-1, 0, 1].

    Each layer is fit on a hand-controlled (K, D) centroid matrix so the
    per-node activations are predictable.  All centroids are constructed
    to lie in a known 2-D subspace of the ambient D-dim space; PCA
    recovers that exactly, so ``subspace_fraction`` on an activation
    placed inside that subspace is 1.0 and the nearest-node geometry
    matches the centroid layout.
    """
    torch.manual_seed(seed)
    domain = BoxDomain([BoxAxis("u", periodic=False, lo=-1.0, hi=1.0)])
    coords = torch.tensor([[-1.0], [0.0], [1.0]])

    layers: dict[int, LayerSubspace] = {}
    ev: dict[int, float] = {}
    # Same orthonormal frame across layers so each layer's geometry is
    # easy to reason about (the fraction = 1.0 case lands on a single
    # well-known set of D-dim directions).
    e1 = torch.zeros(dim)
    e1[0] = 1.0
    e2 = torch.zeros(dim)
    e2[1] = 1.0
    for layer_idx in range(n_layers):
        # Centroids: well-separated in (e1, e2), per layer scaled so the
        # 3 nodes don't degenerate.  Per-layer scaling makes the EV
        # weighting non-trivial.
        scale = 1.0 + 0.5 * layer_idx
        centroids = torch.stack([
            -scale * e1,
            torch.zeros(dim),
            scale * e1,
        ])
        # Tiny e2 component so the embedded coords poise affinely
        # against the affine polynomial term (the RBF wants ≥ embed_dim+1
        # affinely-independent node params; an open 1-D axis has embed
        # dim = 1, so 3 distinct coords poise trivially — the e2
        # perturbation is belt-and-braces against degenerate ranks).
        centroids = centroids + 0.01 * torch.stack([
            -e2, torch.zeros(dim), e2,
        ])
        sub, ev_ratio = _fit_layer_subspace_with_ev(
            centroids, domain.embed(coords),
        )
        layers[layer_idx] = sub
        ev[layer_idx] = ev_ratio
    return Manifold(
        name="toy",
        domain=domain,
        node_labels=["a", "b", "c"],
        node_coords=coords,
        layers=layers,
        explained_variance=ev,
    )


# ============================================ attach / cache + accessors ===

def test_add_probe_registers_and_precaches():
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    assert mon.probe_names == ["toy"]
    probes = mon.attached_probes()
    assert "toy" in probes
    p = probes["toy"]
    assert isinstance(p, AttachedManifoldProbe)
    # Per-layer reduced node-value cache present for every fitted layer,
    # with shape (K, R).  (The former world-space cache was dead code and
    # has been removed — only the reduced cache feeds scoring.)
    for layer_idx in m.layers:
        assert layer_idx in p.node_values_reduced
        assert p.node_values_reduced[layer_idx].shape[0] == len(m.node_labels)
    # EV weights normalized to sum ≈ 1.
    assert sum(p.ev_weights.values()) == pytest.approx(1.0, abs=1e-5)


def test_attached_layers_is_union():
    m = _toy_manifold(n_layers=3)
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    assert mon.attached_layers() == {0, 1, 2}


def test_remove_probe():
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    mon.remove_probe("toy")
    assert mon.probe_names == []
    assert mon.attached_layers() == set()


def test_add_probe_top_n_default():
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    assert mon.attached_probes()["toy"].top_n == DEFAULT_NEAREST_TOP_N


def test_add_probe_top_n_override():
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m, top_n=2)
    assert mon.attached_probes()["toy"].top_n == 2


def test_add_probe_rejects_empty_manifold():
    domain = BoxDomain([BoxAxis("u", periodic=False, lo=0.0, hi=1.0)])
    empty = Manifold(
        name="empty", domain=domain, node_labels=[],
        node_coords=torch.zeros(0, 1), layers={},
    )
    mon = Monitor()
    with pytest.raises(ValueError):
        mon.add_probe("empty", empty)


# ====================================================== subspace_fraction ===

def test_fraction_inside_subspace_is_one():
    """An activation that lives entirely in the manifold's PCA subspace
    should report ``fraction`` near 1 on every layer the manifold
    covers."""
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)

    hidden = {}
    for layer_idx, sub in m.layers.items():
        # h = mean + basis row 0 ⇒ centered fully in-subspace.
        h = sub.mean + sub.basis[0]
        hidden[layer_idx] = h

    readings = mon.score_single_token(hidden)
    assert readings["toy"].fraction == pytest.approx(1.0, abs=1e-5)


def test_fraction_outside_subspace_is_zero():
    """An activation whose centered component is orthogonal to the
    PCA subspace should report ``fraction`` ≈ 0."""
    m = _toy_manifold(dim=8)
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)

    hidden = {}
    for layer_idx, sub in m.layers.items():
        # Build a unit vector orthogonal to every basis row.  Start with
        # ``e_last`` (the highest-index canonical axis); project out the
        # basis components to make it strictly orthogonal.
        v = torch.zeros(sub.mean.shape[0])
        v[-1] = 1.0
        v = v - (v @ sub.basis.T) @ sub.basis
        v = v / torch.linalg.vector_norm(v).clamp(min=1e-8)
        hidden[layer_idx] = sub.mean + v

    readings = mon.score_single_token(hidden)
    # Whitened M-orthogonal fraction under an *isotropic* whitener ≈ the old
    # Euclidean fraction; finite-N Σ has small off-diagonal leakage, so the
    # near-zero value is loose rather than exact.
    assert readings["toy"].fraction == pytest.approx(0.0, abs=2e-2)


def test_fraction_partial_inside_half():
    """A 45-degree blend of in-subspace + out-of-subspace components
    should report ``fraction`` ≈ ``cos(45°)`` = 1/sqrt(2)."""
    m = _toy_manifold(dim=8)
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)

    hidden = {}
    for layer_idx, sub in m.layers.items():
        in_dir = sub.basis[0]  # unit norm by construction
        # Orthogonal direction.
        out_dir = torch.zeros(sub.mean.shape[0])
        out_dir[-1] = 1.0
        out_dir = out_dir - (out_dir @ sub.basis.T) @ sub.basis
        out_dir = out_dir / torch.linalg.vector_norm(out_dir).clamp(min=1e-8)
        h = sub.mean + in_dir + out_dir  # equal in/out, fraction = 1/sqrt(2)
        hidden[layer_idx] = h

    readings = mon.score_single_token(hidden)
    # Isotropic-whitener readout ≈ Euclidean fraction (1/√2); loose tolerance
    # for the finite-N Σ approximation (see test_fraction_outside_subspace).
    assert readings["toy"].fraction == pytest.approx(
        1.0 / math.sqrt(2), abs=2e-2,
    )


# ========================================================== nearest_nodes ===

def test_nearest_finds_correct_node_at_centroid():
    """An activation placed exactly at node k's centroid should rank
    node k first."""
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)

    for k in range(len(m.node_labels)):
        hidden = {}
        for layer_idx, sub in m.layers.items():
            # World-space activation at node k = sub.eval_at(coord_k).
            v_world_k = _node_world(m, layer_idx)[k]
            hidden[layer_idx] = v_world_k
        readings = mon.score_single_token(hidden)
        nearest = readings["toy"].nearest
        assert nearest[0][0] == m.node_labels[k], (
            f"expected node {m.node_labels[k]}, got {nearest[0][0]}"
        )
        # Distance to itself is ≈ 0.
        assert nearest[0][1] == pytest.approx(0.0, abs=1e-3)


def test_nearest_top_n_length():
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m, top_n=2)
    hidden = {layer_idx: sub.mean for layer_idx, sub in m.layers.items()}
    readings = mon.score_single_token(hidden)
    assert len(readings["toy"].nearest) == 2


def test_nearest_sorted_ascending():
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    hidden = {layer_idx: sub.mean for layer_idx, sub in m.layers.items()}
    readings = mon.score_single_token(hidden)
    dists = [d for _, d in readings["toy"].nearest]
    assert dists == sorted(dists)


# =========================================================== flat_scalars ===

def test_flat_scalars_keys_and_signs():
    """flat_scalars must emit ``<name>:fraction`` (positive ∈ [0, 1])
    and ``<name>@<label>`` (negative — encodes -distance so larger =
    closer)."""
    mon = Monitor()
    reading = ManifoldReading(
        fraction=0.7, nearest=[("a", 0.5), ("b", 1.2)],
    )
    flat = mon.flat_scalars({"toy": reading})
    assert flat["toy:fraction"] == pytest.approx(0.7)
    assert flat["toy@a"] == pytest.approx(-0.5)
    assert flat["toy@b"] == pytest.approx(-1.2)
    # ``toy@a`` larger than ``toy@b`` because a is closer.
    assert flat["toy@a"] > flat["toy@b"]


def test_flat_scalars_namespaces_per_probe():
    mon = Monitor()
    r1 = ManifoldReading(fraction=0.5, nearest=[("x", 0.1)])
    r2 = ManifoldReading(fraction=0.8, nearest=[("y", 0.2)])
    flat = mon.flat_scalars({"p1": r1, "p2": r2})
    assert set(flat) == {"p1:fraction", "p1@x", "p2:fraction", "p2@y"}


# ============================================== inverse_projection / aggregate ===

def test_score_aggregate_at_node_recovers_authoring_coord():
    """Aggregating over a captured stack where every row is node k's
    centroid should recover the authoring coord ``node_coords[k]``."""
    m = _toy_manifold(dim=8)
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)

    # Pick node 0 (coord = -1.0).  Captured stack has T=3 rows, all the
    # same centroid — the pooled (last-non-special) row stays on it.
    target_k = 0
    captured: dict[int, torch.Tensor] = {}
    for layer_idx, _sub in m.layers.items():
        v_world_k = _node_world(m, layer_idx)[target_k]
        captured[layer_idx] = v_world_k.unsqueeze(0).repeat(3, 1)

    agg = mon.score_aggregate(captured)
    assert "toy" in agg
    r: ManifoldReading = agg["toy"]
    # Coords near the authoring coord (-1.0) under the grid's resolution.
    assert len(r.coords) == 1
    assert r.coords[0] == pytest.approx(-1.0, abs=0.1)
    # Per-layer dict carries one coord per fitted layer.
    assert set(r.coords_per_layer.keys()) == set(m.layers.keys())
    for layer_idx, coord in r.coords_per_layer.items():
        assert len(coord) == 1
        assert coord[0] == pytest.approx(-1.0, abs=0.1)
    # Residual is small at a node.
    assert r.residual < 0.1


def test_score_aggregate_pools_last_non_special_row_not_mean():
    """The aggregate pools the chosen row (default last; ``agg_index``
    overrides), NOT a mean across the trajectory.

    Build a stack whose rows sit at *different* nodes so the two
    behaviors are distinguishable: a mean would land between the nodes
    (coord ≈ 0, label "b"), while row-selection recovers the specific
    node at that row.  This pins the fix for the former
    ``stack.mean(dim=0)`` pooling.
    """
    m = _toy_manifold(dim=8)
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)

    # T=2: row 0 = node 2 (coord +1, "c"), row 1 = node 0 (coord -1, "a").
    # row 1 stands in for a trailing special token the session walks past.
    captured: dict[int, torch.Tensor] = {}
    for layer_idx, _sub in m.layers.items():
        world = _node_world(m, layer_idx)
        node2 = world[2]
        node0 = world[0]
        captured[layer_idx] = torch.stack([node2, node0])

    # agg_index=0 → the non-special row → node 2 (coord +1, "c").
    agg0 = mon.score_aggregate(captured, agg_index=0)["toy"]
    assert agg0.coords[0] == pytest.approx(1.0, abs=0.1)
    assert agg0.nearest[0][0] == "c"
    # A mean would have landed at coord ≈ 0 / label "b" — guard against
    # regressing to mean(dim=0).
    assert agg0.nearest[0][0] != "b"

    # Default (no agg_index) → final row → node 0 (coord -1, "a").
    agg_last = mon.score_aggregate(captured)["toy"]
    assert agg_last.coords[0] == pytest.approx(-1.0, abs=0.1)
    assert agg_last.nearest[0][0] == "a"


def test_score_aggregate_fraction_matches_per_token():
    """For a stack of identical hidden states the aggregate fraction
    should match the per-token fraction (mean of identical values = the
    value itself).
    """
    m = _toy_manifold(dim=8)
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)

    # All-in-subspace hidden state.
    hidden = {}
    captured: dict[int, torch.Tensor] = {}
    for layer_idx, sub in m.layers.items():
        h = sub.mean + sub.basis[0]
        hidden[layer_idx] = h
        captured[layer_idx] = h.unsqueeze(0).repeat(4, 1)

    per_token = mon.score_single_token(hidden)
    agg = mon.score_aggregate(captured)
    assert per_token["toy"].fraction == pytest.approx(
        agg["toy"].fraction, abs=1e-5,
    )


def test_score_aggregate_to_dict_round_trip():
    """``ManifoldReading.to_dict`` produces a JSON-serializable dict
    with the expected structure."""
    import json

    m = _toy_manifold(dim=8)
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    captured: dict[int, torch.Tensor] = {}
    for layer_idx, _sub in m.layers.items():
        v_world_k = _node_world(m, layer_idx)[1]
        captured[layer_idx] = v_world_k.unsqueeze(0).repeat(2, 1)

    agg = mon.score_aggregate(captured)
    d = agg["toy"].to_dict()
    js = json.dumps(d)  # raises if any value is non-serializable
    assert "fraction" in d
    assert "coords" in d
    assert "nearest" in d
    # Re-decodable, and the round-trip shape is stable.
    decoded = json.loads(js)
    assert decoded["fraction"] == d["fraction"]


def test_empty_inputs_return_empty():
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    assert mon.score_single_token({}) == {}
    assert mon.score_aggregate({}) == {}


def test_pending_per_token_flag():
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    assert not mon.has_pending_per_token()
    hidden = {layer_idx: sub.mean for layer_idx, sub in m.layers.items()}
    mon.score_single_token(hidden)
    assert mon.has_pending_per_token()
    mon.consume_pending_per_token()
    assert not mon.has_pending_per_token()


# ============================================== affine (flat) batched path ===
#
# The curved tests above exercise the per-probe foot-solve path.  These build
# *flat* (affine) probes so they ride the batched coordinate readout
# (one stacked matmul + one shared Woodbury per layer; rank is the block size,
# rank-1 has no separate path).  An affine subspace's immersion is pure-affine
# (``eval_at(c) = c @ basis + mean``), so a hand-built ``LayerSubspace.affine``
# over a ``CustomDomain`` is self-consistent: node k's activation is
# ``rc[k] @ basis + mean``, and the readout must recover the authoring coord
# ``rc[k]``.


def _flat_manifold(
    *,
    reduced_coords: torch.Tensor,   # (K, R) real reduced node coords == domain coords
    n_layers: int = 2,
    dim: int = 16,
    seed: int = 0,
    name: str = "flat",
    labels: list[str] | None = None,
) -> Manifold:
    """Hand-build a flat ``fit_mode=pca``-shaped manifold of arbitrary rank.

    Each layer gets a fresh random orthonormal ``(R, D)`` basis + mean; the
    node activations are ``rc @ basis + mean`` so the affine readout is exact.
    The domain is the identity ``CustomDomain(R)`` and ``node_coords`` are the
    real reduced coords (shared across layers — fine for the test, the affine
    map is fit per layer regardless).
    """
    torch.manual_seed(seed)
    rc = reduced_coords.to(torch.float32)
    K, R = rc.shape
    layers: dict[int, LayerSubspace] = {}
    ev: dict[int, float] = {}
    for layer_idx in range(n_layers):
        a = torch.randn(dim, R)
        q, _ = torch.linalg.qr(a)          # (D, R) orthonormal columns
        basis = q[:, :R].T.contiguous()    # (R, D) orthonormal rows
        mean = torch.randn(dim) * 0.1 + float(layer_idx + 1)
        layers[layer_idx] = LayerSubspace.affine(mean, basis, node_coords=rc)
        ev[layer_idx] = 1.0 + 0.3 * layer_idx
    return Manifold(
        name=name,
        domain=CustomDomain(R),
        node_labels=labels or [f"n{k}" for k in range(K)],
        node_coords=rc,
        layers=layers,
        explained_variance=ev,
    )


def _flat_node_hidden(m: Manifold, k: int) -> dict[int, torch.Tensor]:
    """Per-layer activation sitting exactly on node ``k`` (``rc[k] @ basis + mean``)."""
    rc = m.node_coords.to(torch.float32)
    return {
        L: sub.eval_at(rc[k]) for L, sub in m.layers.items()
    }


def test_affine_rank1_recovers_pole_coord():
    """A 2-node (rank-1) flat probe places the activation on a pole and
    recovers that pole's authoring coord."""
    rc = torch.tensor([[1.0], [-1.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["pos", "neg"])
    mon = _iso_monitor(m)
    mon.add_probe("ax", m)
    for k, want, label in ((0, 1.0, "pos"), (1, -1.0, "neg")):
        r = mon.score_single_token(_flat_node_hidden(m, k))["ax"]
        assert len(r.coords) == 1
        assert r.coords[0] == pytest.approx(want, abs=1e-3)
        # In-subspace → fraction ≈ 1.  Under the unified full reading flat
        # probes now fill nearest per token too (was deferred); residual ≈ 0.
        assert r.fraction == pytest.approx(1.0, abs=2e-2)
        assert r.nearest[0][0] == label
        assert r.residual == pytest.approx(0.0, abs=1e-5)


def test_affine_rankR_recovers_multidim_coords():
    """A 3-node 2-D flat probe (rank-2) recovers full 2-D authoring coords
    at each node — the generalized affine coord map, not just axis 0."""
    rc = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["x", "y", "z"])
    mon = _iso_monitor(m)
    mon.add_probe("tri", m)
    for k in range(3):
        r = mon.score_single_token(_flat_node_hidden(m, k))["tri"]
        assert len(r.coords) == 2
        assert r.coords[0] == pytest.approx(float(rc[k][0]), abs=1e-3)
        assert r.coords[1] == pytest.approx(float(rc[k][1]), abs=1e-3)


def test_affine_mixed_rank_batched_together():
    """A rank-1 and a rank-2 affine probe attached to ONE monitor are scored
    in the same batched pass — validates the block-diagonal reduced solve +
    the global per-probe coord layout across mixed ranks."""
    rc1 = torch.tensor([[2.0], [-2.0]])
    rc2 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    m1 = _flat_manifold(reduced_coords=rc1, dim=16, seed=1, name="ax", labels=["p", "n"])
    m2 = _flat_manifold(reduced_coords=rc2, dim=16, seed=2, name="tri", labels=["x", "y", "z"])
    # Share one whitener covering both probes' layers (same layer set).
    from tests._whitener import isotropic_whitener
    whitener = isotropic_whitener(list(m1.layers.keys()), 16)
    mon = Monitor(whitener=whitener)
    mon.add_probe("ax", m1)
    mon.add_probe("tri", m2)

    # Build a combined hidden: node 0 of ax, node 2 of tri — but they live on
    # different bases, so score them one probe at a time against its own node
    # activation and confirm the batched dict carries both correctly.
    r_ax = mon.score_single_token(_flat_node_hidden(m1, 0))
    assert r_ax["ax"].coords[0] == pytest.approx(2.0, abs=1e-3)
    assert len(r_ax["ax"].coords) == 1
    # ``tri`` is also scored in the same call (different basis → arbitrary
    # coord), but its slice must be length-2 (global coord layout intact).
    assert len(r_ax["tri"].coords) == 2

    r_tri = mon.score_single_token(_flat_node_hidden(m2, 2))
    assert r_tri["tri"].coords[0] == pytest.approx(-1.0, abs=1e-3)
    assert r_tri["tri"].coords[1] == pytest.approx(-1.0, abs=1e-3)
    assert len(r_tri["ax"].coords) == 1


def test_affine_aggregate_fills_nearest_and_coords():
    """End-of-gen aggregate for a flat probe recovers coords via the affine
    map AND fills nearest (deferred from the per-token path), residual ≈ 0
    (the surface fills its subspace)."""
    rc = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["x", "y", "z"])
    mon = _iso_monitor(m)
    mon.add_probe("tri", m)
    target = 1  # node "y"
    captured = {
        L: h.unsqueeze(0).repeat(3, 1)
        for L, h in _flat_node_hidden(m, target).items()
    }
    agg = mon.score_aggregate(captured)["tri"]
    assert agg.coords[0] == pytest.approx(0.0, abs=1e-3)
    assert agg.coords[1] == pytest.approx(1.0, abs=1e-3)
    assert agg.nearest[0][0] == "y"
    assert agg.nearest[0][1] == pytest.approx(0.0, abs=1e-3)
    assert agg.residual == pytest.approx(0.0, abs=1e-5)


def test_affine_fraction_outside_subspace_is_zero():
    """A rank-2 flat probe reports fraction ≈ 0 for an off-subspace activation."""
    rc = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    m = _flat_manifold(reduced_coords=rc)
    mon = _iso_monitor(m)
    mon.add_probe("tri", m)
    hidden = {}
    for L, sub in m.layers.items():
        v = torch.zeros(sub.mean.shape[0])
        v[-1] = 1.0
        v = v - (v @ sub.basis.T) @ sub.basis
        v = v / torch.linalg.vector_norm(v).clamp(min=1e-8)
        hidden[L] = sub.mean + v
    r = mon.score_single_token(hidden)["tri"]
    assert r.fraction == pytest.approx(0.0, abs=2e-2)
