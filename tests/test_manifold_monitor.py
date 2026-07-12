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
    NEUTRAL_LABEL,
    AttachedManifoldProbe,
    Monitor,
)
from saklas.core.results import ProbeReading


def fit_layer_subspace(*args: Any, **kwargs: Any) -> Any:
    """Test helper: drop the EV ratio for callers that don't care."""
    sub, _ev = _fit_layer_subspace_with_ev(*args, **kwargs)
    return sub


def _iso_monitor(m: "Manifold", *, n_layers: int | None = None) -> Monitor:
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
    return Monitor(whitener=whitener, n_layers=n_layers)


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
    share: dict[int, float] = {}
    # Same orthonormal frame across layers so each layer's geometry is
    # easy to reason about (the fraction = 1.0 case lands on a single
    # well-known set of D-dim directions).
    e1 = torch.zeros(dim)
    e1[0] = 1.0
    e2 = torch.zeros(dim)
    e2[1] = 1.0
    for layer_idx in range(n_layers):
        # Centroids: well-separated in (e1, e2), per layer scaled so the
        # 3 nodes don't degenerate.  Per-layer scaling makes the share
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
        sub.sigma_rbf_weights = torch.zeros(coords.shape[0], 1)
        sub.sigma_poly_coeffs = torch.zeros(domain.embed(coords).shape[1] + 1, 1)
        layers[layer_idx] = sub
        share[layer_idx] = ev_ratio
    return Manifold(
        name="toy",
        domain=domain,
        node_labels=["a", "b", "c"],
        node_coords=coords,
        layers=layers,
        mahalanobis_share=share,
        origin={layer: torch.zeros(domain.intrinsic_dim) for layer in layers},
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
    # Share weights normalized to sum ≈ 1.
    assert sum(p.share_weights.values()) == pytest.approx(1.0, abs=1e-5)


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


def test_neutral_competes_in_nearest_at_origin():
    """An activation at the neutral-anchored frame origin (reduced coord 0,
    world ``sub.mean``) ranks the synthetic ``neutral`` candidate first — it is
    never a stored node, only a competitor in the readout (flat/batched path)."""
    rc = torch.tensor([[1.0, 1.0], [2.0, -1.0], [-1.0, 2.0]])  # none at origin
    m = _flat_manifold(reduced_coords=rc, labels=["a", "b", "c"])
    mon = _iso_monitor(m)
    mon.add_probe("tri", m)
    # World activation at the origin = the subspace mean (reduced 0 = neutral).
    hidden = {L: sub.mean for L, sub in m.layers.items()}
    nearest = mon.score_single_token(hidden)["tri"].nearest
    assert nearest[0][0] == NEUTRAL_LABEL
    assert nearest[0][1] == pytest.approx(0.0, abs=1e-3)
    # ``neutral`` never leaks into the stored node set.
    assert NEUTRAL_LABEL not in m.node_labels


def test_neutral_competes_in_aggregate_path():
    """The end-of-gen aggregate (the per-probe ``_score_probe_full`` path, not
    the batched one) also lets neutral compete — last content token at origin."""
    rc = torch.tensor([[1.0, 1.0], [2.0, -1.0], [-1.0, 2.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["a", "b", "c"])
    mon = _iso_monitor(m)
    mon.add_probe("tri", m)
    captured = {
        L: sub.mean.unsqueeze(0).repeat(3, 1) for L, sub in m.layers.items()
    }
    agg = mon.score_aggregate(captured)["tri"]
    assert agg.nearest[0][0] == NEUTRAL_LABEL
    assert agg.nearest[0][1] == pytest.approx(0.0, abs=1e-3)


def test_neutral_does_not_displace_a_closer_node():
    """On a node centroid the real node still wins; neutral is farther."""
    rc = torch.tensor([[1.0, 1.0], [2.0, -1.0], [-1.0, 2.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["a", "b", "c"])
    mon = _iso_monitor(m)
    mon.add_probe("tri", m)
    nearest = mon.score_single_token(_flat_node_hidden(m, 1))["tri"].nearest
    assert nearest[0][0] == "b"
    assert nearest[0][1] == pytest.approx(0.0, abs=1e-3)


def test_neutral_suppressed_when_node_named_neutral():
    """A real corpus node labeled ``neutral`` keeps sole ownership of the
    label — the synthetic competitor is not injected."""
    rc = torch.tensor([[1.0, 1.0], [2.0, -1.0], [-1.0, 2.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["a", "neutral", "c"])
    mon = _iso_monitor(m)
    mon.add_probe("tri", m, top_n=-1)  # all candidates
    assert mon._probes["tri"].inject_neutral is False
    # The real ``neutral`` node wins at its own centroid; exactly one entry.
    nearest = mon.score_single_token(_flat_node_hidden(m, 1))["tri"].nearest
    assert nearest[0][0] == "neutral"
    assert nearest[0][1] == pytest.approx(0.0, abs=1e-3)
    assert sum(1 for label, _ in nearest if label == "neutral") == 1


# =========================================================== flat_scalars ===

def test_flat_scalars_keys_and_signs():
    """flat_scalars must emit ``<name>:fraction`` (positive ∈ [0, 1])
    and ``<name>@<label>`` (negative — encodes -distance so larger =
    closer)."""
    mon = Monitor()
    reading = ProbeReading(
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
    r1 = ProbeReading(fraction=0.5, nearest=[("x", 0.1)])
    r2 = ProbeReading(fraction=0.8, nearest=[("y", 0.2)])
    flat = mon.flat_scalars({"p1": r1, "p2": r2})
    # ``:membership`` is always emitted (defaults to 1.0); ``~<label>`` only when
    # the reading carries a soft assignment (these have none).
    assert set(flat) == {
        "p1:fraction", "p1@x", "p1:membership",
        "p2:fraction", "p2@y", "p2:membership",
    }


def test_flat_scalars_emits_assignment_and_membership():
    mon = Monitor()
    r = ProbeReading(
        fraction=0.5, nearest=[("x", 0.1)],
        assignment=[("x", 0.7), ("y", 0.3)], membership=0.42,
    )
    flat = mon.flat_scalars({"p": r})
    assert flat["p~x"] == 0.7
    assert flat["p~y"] == 0.3
    assert flat["p:membership"] == 0.42
    # distance gate untouched (additive)
    assert flat["p@x"] == -0.1


def _attach_const_sigma(m: "Manifold", value: float) -> None:
    """Attach a constant-σ field to every curved layer of ``m`` (test helper)."""
    from saklas.core.manifold import fit_rbf_smoothed
    for sub in m.layers.values():
        np_, _, _ = sub.rbf_params()
        K = np_.shape[0]
        log_sigma = torch.full((K, 1), math.log(value))
        w, c, _ = fit_rbf_smoothed(np_, log_sigma, smoothing=0.0)
        sub.sigma_rbf_weights = w
        sub.sigma_poly_coeffs = c


def _attach_per_node_sigma(m: "Manifold", sigmas: list[float]) -> None:
    """Attach a σ-field interpolating the given per-node thicknesses (test helper)."""
    from saklas.core.manifold import fit_rbf_smoothed
    for sub in m.layers.values():
        np_, _, _ = sub.rbf_params()
        K = np_.shape[0]
        assert K == len(sigmas)
        log_sigma = torch.tensor(
            [[math.log(s)] for s in sigmas], dtype=torch.float32,
        )
        w, c, _ = fit_rbf_smoothed(np_, log_sigma, smoothing=0.0)
        sub.sigma_rbf_weights = w
        sub.sigma_poly_coeffs = c


def _curved_toy(dim: int = 8, seed: int = 0) -> "Manifold":
    """A genuinely curved 1-D manifold (a parabola) in a rank-2 subspace.

    5 nodes along ``u ∈ [-1, 1]`` placed at ``s·(u·e1 + u²·e2)`` so the surface
    bends into the ``e2`` direction — R = 2, n = 1, so a real off-surface
    (normal) direction exists for membership tests (unlike the near-collinear
    ``_toy_manifold``, which collapses to rank 1).
    """
    torch.manual_seed(seed)
    domain = BoxDomain([BoxAxis("u", periodic=False, lo=-1.0, hi=1.0)])
    coords = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    u = coords.reshape(-1)
    e1 = torch.zeros(dim)
    e1[0] = 1.0
    e2 = torch.zeros(dim)
    e2[1] = 1.0
    layers: dict[int, LayerSubspace] = {}
    share: dict[int, float] = {}
    for layer_idx in range(2):
        s = 1.0 + 0.5 * layer_idx
        centroids = s * (u.unsqueeze(1) * e1 + (u ** 2).unsqueeze(1) * e2)
        sub, ev_ratio = _fit_layer_subspace_with_ev(centroids, domain.embed(coords))
        sub.sigma_rbf_weights = torch.zeros(coords.shape[0], 1)
        sub.sigma_poly_coeffs = torch.zeros(domain.embed(coords).shape[1] + 1, 1)
        layers[layer_idx] = sub
        share[layer_idx] = ev_ratio
    return Manifold(
        name="curve", domain=domain,
        node_labels=["a", "b", "c", "d", "e"],
        node_coords=coords, layers=layers, mahalanobis_share=share,
        origin={layer: torch.zeros(domain.intrinsic_dim) for layer in layers},
    )


def test_soft_assignment_peaks_at_nearest_node():
    # An activation at node 0's centroid → assignment is a valid distribution
    # whose mass concentrates on node "a"; nearest argmax agrees.
    m = _toy_manifold(dim=8)
    mon = _iso_monitor(m)
    mon.add_probe("toy", m, top_n=3)
    hidden = {L: _node_world(m, L)[0] for L in m.layers}
    reading = mon.score_single_token(hidden)["toy"]
    assert reading.assignment
    probs = [p for _, p in reading.assignment]
    assert all(0.0 <= p <= 1.0 + 1e-6 for p in probs)
    assert sum(probs) <= 1.0 + 1e-5            # top-N head of the simplex
    top_label = max(reading.assignment, key=lambda kv: kv[1])[0]
    assert top_label == "a"
    assert reading.nearest[0][0] == "a"


def test_soft_assignment_does_not_let_wide_node_win_from_far_away():
    # Regression: the bare ``softmax(−d²/2τ²)`` form lets a diffuse-τ candidate
    # swallow all the mass regardless of distance — the gemma-4-12B emotions eval
    # surfaced this with ``triumphant`` winning 99.7% despite not being in the
    # top-4 nearest.  Fix: add the Gaussian log-volume bias ``−R·log(τ)`` to the
    # logit (proper isotropic R-D mixture posterior).
    #
    # Setup: node "a" is wide (σ=2.0), all other nodes are tight (σ=0.1).  Place
    # the query AT node "c" (the middle of the curve, far from "a").  Under the
    # bare softmax, "a" would dominate by virtue of its wide τ; under the fixed
    # softmax, "c" wins (the query is on it) and "a" is penalized for its
    # log-volume.
    m = _curved_toy(dim=8)
    _attach_per_node_sigma(m, [2.0, 0.1, 0.1, 0.1, 0.1])  # only "a" is diffuse
    mon = _iso_monitor(m)
    mon.add_probe("curve", m, top_n=5)
    at_c = {L: _node_world(m, L)[2] for L in m.layers}
    reading = mon.score_single_token(at_c)["curve"]
    assert reading.assignment
    top = max(reading.assignment, key=lambda kv: kv[1])
    assert top[0] == "c", (
        f"broadest-node-wins regression: query at 'c' assigned to {top[0]!r} "
        f"with mass {top[1]:.3f} — the wide-σ 'a' is dominating despite distance. "
        f"Full assignment: {reading.assignment}"
    )
    a_prob = next((p for L, p in reading.assignment if L == "a"), 0.0)
    c_prob = top[1]
    assert c_prob > 10.0 * a_prob, (
        f"wide-σ 'a' still has comparable mass ({a_prob:.3f}) to the true "
        f"nearest 'c' ({c_prob:.3f}); log-volume bias not strong enough"
    )


def test_logvol_bias_attached_and_finite():
    # The precomputed bias rides on the probe alongside ``assign_bandwidth``.
    m = _curved_toy(dim=8)
    _attach_const_sigma(m, 0.3)
    mon = _iso_monitor(m)
    mon.add_probe("curve", m)
    probe = mon.attached_probes()["curve"]
    assert probe.assign_bandwidth is not None
    assert probe.assign_logvol_bias is not None
    assert probe.assign_bandwidth.shape == probe.assign_logvol_bias.shape
    assert torch.isfinite(probe.assign_logvol_bias).all()
    # For constant τ across nodes the bias is constant too (one number repeated).
    bias = probe.assign_logvol_bias
    assert (bias.max() - bias.min()).abs() < 1e-5


def test_gate_scalar_fraction_label_assignment_skip_curved_foot(
    monkeypatch: pytest.MonkeyPatch,
):
    """Gate-only curved fraction/label/assignment channels avoid foot solves."""
    m = _curved_toy(dim=8)
    _attach_const_sigma(m, 0.3)
    mon = _iso_monitor(m)
    mon.add_probe("curve", m, top_n=5)
    hidden = {L: _node_world(m, L)[2] for L in m.layers}
    full = mon.flat_scalars(mon.score_single_token(hidden))

    def _fail_foot(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("gate scalar path should not solve a curved foot")

    monkeypatch.setattr("saklas.core.monitor.invert_parameterization", _fail_foot)
    keys = {"curve:fraction", "curve@c", "curve~c"}
    scalars = mon.score_gate_scalars(hidden, keys)
    assert scalars["curve:fraction"] == pytest.approx(full["curve:fraction"])
    assert scalars["curve@c"] == pytest.approx(full["curve@c"])
    assert scalars["curve~c"] == pytest.approx(full["curve~c"])


def test_planned_gate_scalars_match_public_and_full_reading():
    m = _curved_toy(dim=8)
    _attach_const_sigma(m, 0.3)
    mon = _iso_monitor(m)
    mon.add_probe("curve", m, top_n=5)
    hidden = {L: _node_world(m, L)[2] for L in m.layers}
    keys = {
        "curve",
        "curve:fraction",
        "curve@c",
        "curve~c",
        "curve:membership",
        "curve@missing",
        "curve~missing",
    }

    plan = mon.plan_gate_scalars(keys)
    planned = mon.score_planned_gate_scalars(hidden, plan)
    public = mon.score_gate_scalars(hidden, keys)
    full = mon.flat_scalars(mon.score_single_token(hidden))

    assert len(plan) == 1
    assert plan[0].dist_index is not None
    assert plan[0].assign_index is not None
    assert "curve@missing" not in planned
    assert "curve~missing" not in planned
    assert planned == pytest.approx(public)
    for key in keys - {"curve@missing", "curve~missing"}:
        assert planned[key] == pytest.approx(full[key])
    axis_plan = mon.plan_gate_scalars({"curve[0]"})
    axis_planned = mon.score_planned_gate_scalars(hidden, axis_plan)
    assert axis_planned["curve[0]"] == pytest.approx(full["curve[0]"])


def test_gate_scalar_requested_labels_ignore_probe_top_n():
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m, top_n=1)
    hidden = {L: _node_world(m, L)[2] for L in m.layers}  # nearest is "c"

    full = mon.flat_scalars(mon.score_single_token(hidden))
    assert "toy@c" in full
    assert "toy@a" not in full
    assert "toy~a" not in full

    scalars = mon.score_gate_scalars(hidden, {"toy@a", "toy~a"})
    assert "toy@a" in scalars
    assert scalars["toy@a"] < 0.0
    assert "toy~a" in scalars
    assert 0.0 <= scalars["toy~a"] <= 1.0


def test_gate_label_plan_uses_attached_candidate_metadata():
    rc = torch.tensor([[1.0, 1.0], [2.0, -1.0], [-1.0, 2.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["a", "b", "c"])
    mon = _iso_monitor(m)
    mon.add_probe("tri", m)
    probe = mon.attached_probes()["tri"]

    assert probe.candidate_labels == ("a", "b", "c", NEUTRAL_LABEL)
    assert probe.label_to_candidate_idx == {
        "a": 0,
        "b": 1,
        "c": 2,
        NEUTRAL_LABEL: 3,
    }
    neutral_scalars = mon.score_gate_scalars(
        {L: sub.mean for L, sub in m.layers.items()},
        {"tri@neutral", "tri~neutral"},
    )
    assert neutral_scalars["tri@neutral"] == pytest.approx(0.0, abs=1e-3)
    assert 0.0 <= neutral_scalars["tri~neutral"] <= 1.0

    real_neutral = _flat_manifold(
        reduced_coords=rc,
        labels=["a", NEUTRAL_LABEL, "c"],
    )
    mon2 = _iso_monitor(real_neutral)
    mon2.add_probe("tri", real_neutral)
    probe2 = mon2.attached_probes()["tri"]
    assert probe2.inject_neutral is False
    assert probe2.candidate_labels == ("a", NEUTRAL_LABEL, "c")
    assert probe2.label_to_candidate_idx[NEUTRAL_LABEL] == 1


def test_gate_label_plan_duplicate_labels_resolve_last_occurrence():
    rc = torch.tensor([[0.0], [1.0], [2.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["dup", "other", "dup"])
    mon = _iso_monitor(m)
    mon.add_probe("dup_probe", m, top_n=-1)

    probe = mon.attached_probes()["dup_probe"]
    assert probe.label_to_candidate_idx["dup"] == 2
    scalars = mon.score_gate_scalars(
        _flat_node_hidden(m, 2),
        {"dup_probe@dup", "dup_probe~dup"},
    )
    assert scalars["dup_probe@dup"] == pytest.approx(0.0, abs=1e-3)
    assert 0.0 <= scalars["dup_probe~dup"] <= 1.0


def test_membership_high_on_surface_low_off_tube():
    # With a σ-field, membership is ~1 on the surface and collapses far off it.
    m = _curved_toy(dim=8)
    _attach_const_sigma(m, 0.3)
    mon = _iso_monitor(m)
    mon.add_probe("curve", m)
    on = {L: _node_world(m, L)[2] for L in m.layers}   # node "c", on-surface
    mem_on = mon.score_single_token(on)["curve"].membership
    assert mem_on > 0.9
    # push along basis row 1 (≈ the surface normal at the apex) → off-tube
    off = {
        L: _node_world(m, L)[2] + 5.0 * sub.basis[1]
        for L, sub in m.layers.items()
    }
    mem_off = mon.score_single_token(off)["curve"].membership
    assert mem_off < 0.5
    assert mem_off < mem_on


def test_curved_probe_without_sigma_field_is_rejected():
    # Raw curved runtime geometry requires a complete tube field.
    m = _curved_toy(dim=8)
    for sub in m.layers.values():
        sub.sigma_rbf_weights = None
        sub.sigma_poly_coeffs = None
    mon = _iso_monitor(m)
    with pytest.raises(ValueError, match="requires a sigma field"):
        mon.add_probe("curve", m)


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
    r: ProbeReading = agg["toy"]
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
    """``ProbeReading.to_dict`` produces a JSON-serializable dict
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
    share: dict[int, float] = {}
    for layer_idx in range(n_layers):
        a = torch.randn(dim, R)
        q, _ = torch.linalg.qr(a)          # (D, R) orthonormal columns
        basis = q[:, :R].T.contiguous()    # (R, D) orthonormal rows
        mean = torch.randn(dim) * 0.1 + float(layer_idx + 1)
        layers[layer_idx] = LayerSubspace.affine(mean, basis, node_coords=rc)
        share[layer_idx] = 1.0 + 0.3 * layer_idx
    return Manifold(
        name=name,
        domain=CustomDomain(R),
        node_labels=labels or [f"n{k}" for k in range(K)],
        node_coords=rc,
        layers=layers,
        mahalanobis_share=share,
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


def test_affine_batched_topk_width_tracks_requested_top_n():
    """The flat batched scorer should not sort/copy every padded candidate when
    a probe only asks for a small nearest/assignment head.
    """
    rc = torch.randn(32, 2)
    labels = [f"n{k}" for k in range(rc.shape[0])]
    m = _flat_manifold(
        reduced_coords=rc, dim=16, seed=3, name="wide", labels=labels,
    )
    mon = _iso_monitor(m)
    mon.add_probe("wide", m, top_n=3)

    reading = mon.score_single_token(_flat_node_hidden(m, 0))["wide"]

    assert mon._flat_global["topk_width"] == 3
    assert len(reading.nearest) == 3
    assert len(reading.assignment) == 3
    assert reading.nearest[0][0] == "n0"


def test_affine_batched_top_n_zero_skips_ranked_heads():
    rc = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["x", "y", "z"])
    mon = _iso_monitor(m)
    mon.add_probe("tri", m, top_n=0)

    reading = mon.score_single_token(_flat_node_hidden(m, 1))["tri"]

    assert mon._flat_global["topk_width"] == 0
    assert reading.nearest == []
    assert reading.assignment == []


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


# ================================================ curved-probe warm-start ===

def test_curved_warm_start_matches_cold_path():
    """Warm-started curved-probe foot reads track the cold solve.

    A curved probe's per-token nearest-point foot is warm-started from the
    previous token's foot when the session enables it for the (sequential)
    incremental live-scoring pass.  For a smoothly drifting activation the
    warm path (``warm_iter`` steps from the carried foot + one nearest-node
    safety restart) must land at the same nearest point the cold solve
    (``max_iter`` steps from the nearest nodes) finds.
    """
    m = _toy_manifold(dim=8)
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)

    # A smoothly drifting sequence: slide the foot from node a → c along the
    # domain, plus a small off-surface nudge so the residual channel is
    # exercised too.
    dim = next(iter(m.layers.values())).mean.shape[0]
    off = torch.zeros(dim)
    off[-1] = 1.0
    seq = []
    for t in range(8):
        frac = t / 7.0  # 0 → 1
        coord = torch.tensor([2.0 * frac - 1.0])  # slide -1 → 1, on-domain
        embedded = m.domain.embed(m.domain.clamp_position(coord))
        hidden = {
            # World activation at the drifting foot + a small off-surface nudge.
            L: sub.eval_at(embedded) + 0.05 * off
            for L, sub in m.layers.items()
        }
        seq.append(hidden)

    # Cold pass (warm disabled, the default).
    mon.enable_curved_warm(False)
    cold = [mon.score_single_token(h)["toy"] for h in seq]

    # Warm pass: cold-start the feet, enable warm, score the same sequence
    # in order.
    mon.reset_curved_feet()
    mon.enable_curved_warm(True)
    warm = [mon.score_single_token(h)["toy"] for h in seq]

    # The warm path must have populated foot state (it actually ran).
    assert mon._curved_feet, "warm path did not stash any foot"

    for t, (c, w) in enumerate(zip(cold, warm, strict=True)):
        assert w.coords[0] == pytest.approx(c.coords[0], abs=2e-2), (
            f"token {t}: warm coord {w.coords[0]} vs cold {c.coords[0]}"
        )
        assert w.fraction == pytest.approx(c.fraction, abs=2e-2)
        assert w.nearest[0][0] == c.nearest[0][0]


# ========================================== aggregate-only tail capture ===

def test_aggregate_tail_pools_last_content_token():
    """Aggregate-only capture pools the last *content* token, not the EOS slice.

    The bounded tail ring keeps the last ``depth`` forward slices and scores
    nothing per token.  ``tail_slice_at(forward_index)`` must map the
    last-content-token forward index to the correct ring slot — including the
    EOS off-by-one (the terminal EOS forward captures but its token is not in
    ``generated_ids``), so the aggregate matches the full-stack path pooled at
    the same index.
    """
    import torch.nn as nn

    from saklas.core.hooks import HiddenCapture

    m = _toy_manifold(dim=8)  # curved, layers {0, 1}
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    D = next(iter(m.layers.values())).mean.shape[0]

    class _Pass(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return x

    layers = nn.ModuleList([_Pass(), _Pass()])

    # Five forwards; each layer's hidden is a distinct slide along basis[0].
    # generated_ids has 4 tokens (forward 4 = terminal EOS, not appended), so
    # the last content token is forward index 3.
    def _forward_state(t: int) -> dict[int, torch.Tensor]:
        return {
            L: (m.layers[L].mean + float(t + 1) * m.layers[L].basis[0]).reshape(1, 1, D)
            for L in m.layers
        }

    forwards = [_forward_state(t) for t in range(5)]
    last_content_forward = 3

    # Full-retention capture → score_aggregate at the last-content index.
    cap_full = HiddenCapture()
    cap_full.attach(layers, [0, 1])
    for fw in forwards:
        for L in (0, 1):
            layers[L](fw[L])
    full_stack = cap_full.stacked()
    cap_full.detach()
    assert all(s.shape[0] == 5 for s in full_stack.values())
    agg_full = mon.score_aggregate(full_stack, agg_index=last_content_forward)["toy"]

    # Aggregate-tail capture (depth 3) → tail_slice_at(last_content_forward).
    cap_tail = HiddenCapture()
    cap_tail.attach(layers, [0, 1])
    cap_tail.set_aggregate_tail(3)
    for fw in forwards:
        for L in (0, 1):
            layers[L](fw[L])
    assert cap_tail._forward_count == 5
    assert all(len(b) == 3 for b in cap_tail._per_layer.values())  # ring capped
    pooled = cap_tail.tail_slice_at(last_content_forward)
    cap_tail.detach()
    agg_tail = mon.score_aggregate(pooled)["toy"]

    # Same pooled token ⇒ identical aggregate (not the EOS/forward-4 slice).
    assert agg_tail.coords[0] == pytest.approx(agg_full.coords[0], abs=1e-5)
    assert agg_tail.fraction == pytest.approx(agg_full.fraction, abs=1e-5)
    assert agg_tail.nearest[0][0] == agg_full.nearest[0][0]


def test_aggregate_tail_clamps_when_walkback_exceeds_depth():
    """A walk-back deeper than the ring clamps to the oldest retained slice."""
    import torch.nn as nn

    from saklas.core.hooks import HiddenCapture

    m = _toy_manifold(dim=8)
    D = next(iter(m.layers.values())).mean.shape[0]

    class _Pass(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    layers = nn.ModuleList([_Pass(), _Pass()])
    cap = HiddenCapture()
    cap.attach(layers, [0, 1])
    cap.set_aggregate_tail(2)  # tiny ring
    for t in range(5):
        for L in (0, 1):
            layers[L]((m.layers[L].mean + float(t) * m.layers[L].basis[0]).reshape(1, 1, D))
    cap.detach()
    # Ask for forward 0 (older than the depth-2 ring holding forwards {3, 4}):
    # clamps to the oldest retained slice rather than indexing out of range.
    pooled = cap.tail_slice_at(0)
    assert set(pooled) == {0, 1}


def test_tail_with_sink_can_keep_deep_tail_on_selected_layers_only():
    import torch.nn as nn

    from saklas.core.hooks import HiddenCapture

    class _Pass(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    layers = nn.ModuleList([_Pass(), _Pass(), _Pass()])
    cap = HiddenCapture()
    rows: list[dict[int, torch.Tensor]] = []
    cap.attach(layers, [0, 1, 2])
    cap.set_tail_with_sink(
        3,
        lambda latest: rows.append({
            layer: row.clone() for layer, row in latest.items()
        }),
        tail_layers={2},
    )

    for step in range(5):
        for layer in range(3):
            value = torch.full((1, 1, 4), float(layer * 100 + step))
            layers[layer](value)
        cap.fire_step_sink()

    assert [len(cap._per_layer[layer]) for layer in range(3)] == [1, 1, 3]
    assert set(cap.latest_per_layer()) == {0, 1, 2}
    assert len(rows) == 5
    assert all(set(row) == {0, 1, 2} for row in rows)

    pooled = cap.tail_slice_at(2)
    assert set(pooled) == {2}
    assert pooled[2][0].item() == pytest.approx(202.0)


# ===== probe-inspector live-point subspace coords (gated stamping) =========

def test_subspace_coords_gated_off_by_default():
    """Live-point coords are NOT stamped unless the session opted in via
    ``set_subspace_coords(True)`` — the default hot path neither computes
    nor serializes ``subspace_coords_per_layer``."""
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    hidden = {li: sub.mean + sub.basis[0] for li, sub in m.layers.items()}
    reading = mon.score_single_token(hidden)["toy"]
    assert reading.subspace_coords_per_layer == {}
    assert reading.to_dict()["subspace_coords_per_layer"] == {}


def test_subspace_coords_stamped_when_enabled():
    """With the gate on, every probed layer carries its (R,) whitened query
    coords — the live point the inspector plots, same frame as the geometry
    endpoint's ``node_white`` — and they serialize with string layer keys."""
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    mon.set_subspace_coords(True)
    hidden = {li: sub.mean + sub.basis[0] for li, sub in m.layers.items()}
    reading = mon.score_single_token(hidden)["toy"]
    sc = reading.subspace_coords_per_layer
    assert set(sc.keys()) == set(m.layers.keys())
    for layer_idx, coords in sc.items():
        assert len(coords) == m.layers[layer_idx].basis.shape[0]
        assert all(isinstance(c, float) for c in coords)
    wire = reading.to_dict()["subspace_coords_per_layer"]
    assert set(wire.keys()) == {str(li) for li in m.layers}


# ===================================================== coords_only (FIX F2) ===
#
# The lean per-token read (``coords_only=True``) must produce the SAME coords +
# fraction as the full read — the lean live consumers (trait stream, loom probe
# row) read only the cross-layer axis-0 coord, so the lean path skips nearest /
# assignment / per-layer traces but must not perturb the coordinate it keeps.


def _assert_lean_matches_full(mon: Monitor, hidden: dict[int, torch.Tensor]) -> None:
    mon.reset_curved_feet()
    full = mon.score_single_token(hidden)
    mon.reset_curved_feet()
    lean = mon.score_single_token(hidden, coords_only=True)
    assert set(full) == set(lean)
    for name in full:
        f, l = full[name], lean[name]
        assert l.coords == pytest.approx(f.coords, abs=1e-4), name
        assert l.fraction == pytest.approx(f.fraction, abs=1e-4), name
        # Lean drops the richer fields the live axis-0 consumers don't read.
        assert l.nearest == []
        assert l.assignment == []
        assert l.membership == 1.0
        assert l.residual == 0.0


def test_coords_only_matches_full_flat_on_node():
    rc = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["x", "y", "z"])
    mon = _iso_monitor(m)
    mon.add_probe("tri", m)
    for k in range(3):
        _assert_lean_matches_full(mon, _flat_node_hidden(m, k))


def test_coords_only_matches_full_flat_off_node():
    rc = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["x", "y", "z"])
    mon = _iso_monitor(m)
    mon.add_probe("tri", m)
    h0 = _flat_node_hidden(m, 0)
    h1 = _flat_node_hidden(m, 1)
    blend = {L: 0.4 * h0[L] + 0.6 * h1[L] for L in h0}
    _assert_lean_matches_full(mon, blend)


def test_coords_only_matches_full_curved():
    m = _toy_manifold()
    mon = _iso_monitor(m)
    mon.add_probe("toy", m)
    for k in range(3):
        _assert_lean_matches_full(mon, {L: _node_world(m, L)[k] for L in m.layers})


def test_coords_only_mixed_flat_and_curved_roster():
    flat = _flat_manifold(
        reduced_coords=torch.tensor([[1.0], [-1.0]]), dim=8,
        name="ax", labels=["p", "n"],
    )
    curved = _toy_manifold(dim=8)
    from tests._whitener import isotropic_whitener
    whitener = isotropic_whitener(sorted(set(flat.layers) | set(curved.layers)), 8)
    mon = Monitor(whitener=whitener)
    mon.add_probe("ax", flat)
    mon.add_probe("toy", curved)
    # Both probes read the same per-layer hidden (different bases) — we only
    # assert lean == full per probe, not specific values.
    hidden = dict(_flat_node_hidden(flat, 0))
    for L in curved.layers:
        hidden[L] = _node_world(curved, L)[1]
    _assert_lean_matches_full(mon, hidden)


# ==================================================== depth statistics ===
# depth_com / depth_spread: per-axis depth center of mass of the per-layer
# coordinate trace, mass = share_weight_L · |coord_L[axis]|, depths
# normalized layer/(n_layers−1).


def test_depth_stats_math():
    from saklas.core.monitor import _depth_stats

    coords = {0: (1.0,), 8: (3.0,)}
    weights = {0: 0.5, 8: 0.5}
    com, spread = _depth_stats(coords, weights, 8.0)
    # masses: L0 → 0.5·|1| = 0.5 at depth 0; L8 → 0.5·|3| = 1.5 at depth 1
    assert com == pytest.approx((1.5 / 2.0,))
    expected_var = (0.5 * (0 - 0.75) ** 2 + 1.5 * (1 - 0.75) ** 2) / 2.0
    assert spread[0] == pytest.approx(expected_var ** 0.5)
    # sign-independent mass: flipping a coordinate's sign moves nothing
    com_neg, _ = _depth_stats({0: (-1.0,), 8: (-3.0,)}, weights, 8.0)
    assert com_neg == pytest.approx(com)


def test_depth_stats_empty_and_zero_mass():
    from saklas.core.monitor import _depth_stats

    assert _depth_stats({}, {}, 8.0) == ((), ())
    # denominator unset (monitor constructed without n_layers)
    assert _depth_stats({0: (1.0,)}, {0: 1.0}, 0.0) == ((), ())
    # zero-mass axis (activation at neutral): defined-but-degenerate
    com, spread = _depth_stats({0: (0.0,), 4: (0.0,)}, {0: 1.0, 4: 1.0}, 4.0)
    assert com == (0.0,)
    assert spread == (0.0,)


def test_depth_stats_per_axis_independent():
    from saklas.core.monitor import _depth_stats

    # axis 0 reads only at L0, axis 1 only at L4 → coms split to the ends
    coords = {0: (2.0, 0.0), 4: (0.0, 2.0)}
    weights = {0: 1.0, 4: 1.0}
    com, spread = _depth_stats(coords, weights, 4.0)
    assert com == pytest.approx((0.0, 1.0))
    assert spread == pytest.approx((0.0, 0.0))


def test_reading_carries_depth_stats_flat_batched_path():
    rc = torch.tensor([[1.0], [-1.0]])
    m = _flat_manifold(reduced_coords=rc, labels=["pos", "neg"])
    mon = _iso_monitor(m, n_layers=2)
    mon.add_probe("ax", m)
    r = mon.score_single_token(_flat_node_hidden(m, 0))["ax"]
    assert len(r.depth_com) == len(r.coords) == 1
    assert len(r.depth_spread) == 1
    assert 0.0 <= r.depth_com[0] <= 1.0
    assert r.depth_spread[0] >= 0.0
    # both layers read the same node coord, so with 2 layers at depths
    # {0, 1} the com must sit strictly inside the interval
    assert 0.0 < r.depth_com[0] < 1.0


def test_reading_carries_depth_stats_curved_path():
    m = _toy_manifold()
    mon = _iso_monitor(m, n_layers=2)
    mon.add_probe("toy", m)
    hidden = {L: _node_world(m, L)[2] for L in m.layers}
    r = mon.score_single_token(hidden)["toy"]
    assert len(r.depth_com) == len(r.coords)
    assert all(0.0 <= c <= 1.0 for c in r.depth_com)
    assert all(s >= 0.0 for s in r.depth_spread)


def test_reading_depth_stats_empty_without_n_layers():
    rc = torch.tensor([[1.0], [-1.0]])
    m = _flat_manifold(reduced_coords=rc)
    mon = _iso_monitor(m)   # no n_layers → no depth axis
    mon.add_probe("ax", m)
    r = mon.score_single_token(_flat_node_hidden(m, 0))["ax"]
    assert r.depth_com == ()
    assert r.depth_spread == ()


def test_depth_stats_serialize_in_to_dict():
    rc = torch.tensor([[1.0], [-1.0]])
    m = _flat_manifold(reduced_coords=rc)
    mon = _iso_monitor(m, n_layers=2)
    mon.add_probe("ax", m)
    d = mon.score_single_token(_flat_node_hidden(m, 0))["ax"].to_dict()
    assert "depth_com" in d and "depth_spread" in d
    assert isinstance(d["depth_com"], list)
