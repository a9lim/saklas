"""ManifoldMonitor — read-side counterpart to manifold steering.

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
    LayerSubspace,
    Manifold,
    fit_layer_subspace as _fit_layer_subspace_with_ev,
)
from saklas.core.monitor import (
    DEFAULT_NEAREST_TOP_N,
    AttachedManifoldProbe,
    ManifoldMonitor,
)
from saklas.core.results import ManifoldAggregate, ManifoldTokenReading


def fit_layer_subspace(*args: Any, **kwargs: Any) -> Any:
    """Test helper: drop the EV ratio for callers that don't care."""
    sub, _ev = _fit_layer_subspace_with_ev(*args, **kwargs)
    return sub


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
    mon = ManifoldMonitor()
    mon.add_probe("toy", m)
    assert mon.probe_names == ["toy"]
    probes = mon.attached_probes()
    assert "toy" in probes
    p = probes["toy"]
    assert isinstance(p, AttachedManifoldProbe)
    # Per-layer node-value caches present for every fitted layer, with
    # shape (K, D) for world and (K, R) for reduced.
    for layer_idx in m.layers:
        assert layer_idx in p.node_values_world
        assert layer_idx in p.node_values_reduced
        assert p.node_values_world[layer_idx].shape[0] == len(m.node_labels)
        assert p.node_values_reduced[layer_idx].shape[0] == len(m.node_labels)
    # EV weights normalized to sum ≈ 1.
    assert sum(p.ev_weights.values()) == pytest.approx(1.0, abs=1e-5)


def test_attached_layers_is_union():
    m = _toy_manifold(n_layers=3)
    mon = ManifoldMonitor()
    mon.add_probe("toy", m)
    assert mon.attached_layers() == {0, 1, 2}


def test_remove_probe():
    m = _toy_manifold()
    mon = ManifoldMonitor()
    mon.add_probe("toy", m)
    mon.remove_probe("toy")
    assert mon.probe_names == []
    assert mon.attached_layers() == set()


def test_add_probe_top_n_default():
    m = _toy_manifold()
    mon = ManifoldMonitor()
    mon.add_probe("toy", m)
    assert mon.attached_probes()["toy"].top_n == DEFAULT_NEAREST_TOP_N


def test_add_probe_top_n_override():
    m = _toy_manifold()
    mon = ManifoldMonitor()
    mon.add_probe("toy", m, top_n=2)
    assert mon.attached_probes()["toy"].top_n == 2


def test_add_probe_rejects_empty_manifold():
    domain = BoxDomain([BoxAxis("u", periodic=False, lo=0.0, hi=1.0)])
    empty = Manifold(
        name="empty", domain=domain, node_labels=[],
        node_coords=torch.zeros(0, 1), layers={},
    )
    mon = ManifoldMonitor()
    with pytest.raises(ValueError):
        mon.add_probe("empty", empty)


# ====================================================== subspace_fraction ===

def test_fraction_inside_subspace_is_one():
    """An activation that lives entirely in the manifold's PCA subspace
    should report ``fraction`` near 1 on every layer the manifold
    covers."""
    m = _toy_manifold()
    mon = ManifoldMonitor()
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
    mon = ManifoldMonitor()
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
    assert readings["toy"].fraction == pytest.approx(0.0, abs=1e-5)


def test_fraction_partial_inside_half():
    """A 45-degree blend of in-subspace + out-of-subspace components
    should report ``fraction`` ≈ ``cos(45°)`` = 1/sqrt(2)."""
    m = _toy_manifold(dim=8)
    mon = ManifoldMonitor()
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
    assert readings["toy"].fraction == pytest.approx(
        1.0 / math.sqrt(2), abs=1e-4,
    )


# ========================================================== nearest_nodes ===

def test_nearest_finds_correct_node_at_centroid():
    """An activation placed exactly at node k's centroid should rank
    node k first."""
    m = _toy_manifold()
    mon = ManifoldMonitor()
    mon.add_probe("toy", m)

    for k in range(len(m.node_labels)):
        hidden = {}
        for layer_idx, sub in m.layers.items():
            # World-space activation at node k = mean + (centered node).
            # The pre-cached node_values_world matches sub.eval_at —
            # use that directly.
            probe = mon.attached_probes()["toy"]
            v_world_k = probe.node_values_world[layer_idx][k]
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
    mon = ManifoldMonitor()
    mon.add_probe("toy", m, top_n=2)
    hidden = {layer_idx: sub.mean for layer_idx, sub in m.layers.items()}
    readings = mon.score_single_token(hidden)
    assert len(readings["toy"].nearest) == 2


def test_nearest_sorted_ascending():
    m = _toy_manifold()
    mon = ManifoldMonitor()
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
    mon = ManifoldMonitor()
    reading = ManifoldTokenReading(
        fraction=0.7, nearest=[("a", 0.5), ("b", 1.2)],
    )
    flat = mon.flat_scalars({"toy": reading})
    assert flat["toy:fraction"] == pytest.approx(0.7)
    assert flat["toy@a"] == pytest.approx(-0.5)
    assert flat["toy@b"] == pytest.approx(-1.2)
    # ``toy@a`` larger than ``toy@b`` because a is closer.
    assert flat["toy@a"] > flat["toy@b"]


def test_flat_scalars_namespaces_per_probe():
    mon = ManifoldMonitor()
    r1 = ManifoldTokenReading(fraction=0.5, nearest=[("x", 0.1)])
    r2 = ManifoldTokenReading(fraction=0.8, nearest=[("y", 0.2)])
    flat = mon.flat_scalars({"p1": r1, "p2": r2})
    assert set(flat) == {"p1:fraction", "p1@x", "p2:fraction", "p2@y"}


# ============================================== inverse_projection / aggregate ===

def test_score_aggregate_at_node_recovers_authoring_coord():
    """Aggregating over a captured stack where every row is node k's
    centroid should recover the authoring coord ``node_coords[k]``."""
    m = _toy_manifold(dim=8)
    mon = ManifoldMonitor()
    mon.add_probe("toy", m)
    probe = mon.attached_probes()["toy"]

    # Pick node 0 (coord = -1.0).  Captured stack has T=3 rows, all the
    # same centroid — pooled mean stays on the centroid.
    target_k = 0
    captured: dict[int, torch.Tensor] = {}
    for layer_idx, _sub in m.layers.items():
        v_world_k = probe.node_values_world[layer_idx][target_k]
        captured[layer_idx] = v_world_k.unsqueeze(0).repeat(3, 1)

    agg = mon.score_aggregate(captured)
    assert "toy" in agg
    r: ManifoldAggregate = agg["toy"]
    # Coords near the authoring coord (-1.0) under the grid's resolution.
    assert len(r.coords) == 1
    assert r.coords[0] == pytest.approx(-1.0, abs=0.1)
    # Per-layer dict carries one coord per fitted layer.
    assert set(r.coords_per_layer.keys()) == set(m.layers.keys())
    for layer_idx, coord in r.coords_per_layer.items():
        assert len(coord) == 1
        assert coord[0] == pytest.approx(-1.0, abs=0.1)
    # Residual is small at a node.
    assert r.residual_mean < 0.1


def test_score_aggregate_fraction_matches_per_token():
    """For a stack of identical hidden states the aggregate fraction
    should match the per-token fraction (mean of identical values = the
    value itself).
    """
    m = _toy_manifold(dim=8)
    mon = ManifoldMonitor()
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
        agg["toy"].fraction_mean, abs=1e-5,
    )


def test_score_aggregate_to_dict_round_trip():
    """``ManifoldAggregate.to_dict`` produces a JSON-serializable dict
    with the expected structure."""
    import json

    m = _toy_manifold(dim=8)
    mon = ManifoldMonitor()
    mon.add_probe("toy", m)
    probe = mon.attached_probes()["toy"]
    captured: dict[int, torch.Tensor] = {}
    for layer_idx, _sub in m.layers.items():
        v_world_k = probe.node_values_world[layer_idx][1]
        captured[layer_idx] = v_world_k.unsqueeze(0).repeat(2, 1)

    agg = mon.score_aggregate(captured)
    d = agg["toy"].to_dict()
    js = json.dumps(d)  # raises if any value is non-serializable
    assert "fraction_mean" in d
    assert "coords" in d
    assert "nearest" in d
    # Re-decodable, and the round-trip shape is stable.
    decoded = json.loads(js)
    assert decoded["fraction_mean"] == d["fraction_mean"]


def test_empty_inputs_return_empty():
    m = _toy_manifold()
    mon = ManifoldMonitor()
    mon.add_probe("toy", m)
    assert mon.score_single_token({}) == {}
    assert mon.score_aggregate({}) == {}


def test_pending_per_token_flag():
    m = _toy_manifold()
    mon = ManifoldMonitor()
    mon.add_probe("toy", m)
    assert not mon.has_pending_per_token()
    hidden = {layer_idx: sub.mean for layer_idx, sub in m.layers.items()}
    mon.score_single_token(hidden)
    assert mon.has_pending_per_token()
    mon.consume_pending_per_token()
    assert not mon.has_pending_per_token()
