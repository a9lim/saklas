"""Angular budget water-fill: per-layer θ ≤ θ_max with budget conserved.

Both angular steering paths (vector and manifold) express a per-layer
rotation budget in θ_max units.  A peaked per-layer share can push a
single layer's budget past 1.0 (= θ > θ_max).  The historical vector
path clamped at 1.0 and *dropped* the excess; the manifold path didn't
clamp at all.  ``_redistribute_budget`` water-fills: cap each layer at
1.0 and spread the trimmed excess over the still-uncapped layers,
preserving the cumulative budget up to the n_layers ceiling.

These are pure-CPU tests on the helper plus an end-to-end assertion that
a peaked-share synthetic profile / manifold produces no per-layer
θ > θ_max while the cumulative rotation budget is conserved.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from saklas.core.hooks import (
    SteeringManager,
    _redistribute_budget,
)
from saklas.core.manifold import (
    BoxAxis,
    BoxDomain,
    LayerSubspace,
    Manifold,
    fit_layer_subspace,
)


# ---------------------------------------------------------------------------
# _redistribute_budget — the water-fill helper.
# ---------------------------------------------------------------------------


class TestRedistributeBudget:
    def test_empty(self) -> None:
        assert _redistribute_budget({}) == {}

    def test_no_overflow_passthrough(self) -> None:
        # All entries already ≤ 1.0 → unchanged (within fp slack).
        raw = {0: 0.2, 1: 0.5, 2: 0.3}
        out = _redistribute_budget(raw)
        for k in raw:
            assert out[k] == raw[k]

    def test_caps_at_one(self) -> None:
        raw = {0: 1.7, 1: 0.1, 2: 0.1}
        out = _redistribute_budget(raw)
        for v in out.values():
            assert v <= 1.0 + 1e-9

    def test_budget_conserved_when_room_remains(self) -> None:
        # Total 1.7 over 3 layers (ceiling 3) → all of it lands somewhere.
        raw = {0: 1.5, 1: 0.1, 2: 0.1}
        out = _redistribute_budget(raw)
        assert sum(out.values()) == pytest.approx(sum(raw.values()), abs=1e-6)
        # The over-budget layer pinned at 1.0; the 0.5 excess spread over
        # layers 1 and 2 in proportion (equal here) → 0.1 + 0.25 each.
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.35, abs=1e-6)
        assert out[2] == pytest.approx(0.35, abs=1e-6)

    def test_saturates_at_ceiling_on_overflow(self) -> None:
        # Total 5.0 over 3 layers (ceiling 3) → every layer pinned at 1.0,
        # the unabsorbable excess (2.0) is dropped.
        raw = {0: 3.0, 1: 1.0, 2: 1.0}
        out = _redistribute_budget(raw)
        assert sum(out.values()) == pytest.approx(3.0, abs=1e-6)
        for v in out.values():
            assert v == pytest.approx(1.0)

    def test_iterates_to_fixpoint(self) -> None:
        # One huge layer + two small: first pass caps layer 0 and pushes
        # excess that itself over-fills layer 1, requiring a second pass.
        raw = {0: 2.4, 1: 0.9, 2: 0.1}  # total 3.4 > ceiling 3 ⇒ all cap
        out = _redistribute_budget(raw)
        for v in out.values():
            assert v <= 1.0 + 1e-9
        assert sum(out.values()) == pytest.approx(3.0, abs=1e-6)

    def test_proportional_redistribution(self) -> None:
        # Excess spreads proportional to current sub-cap value.
        raw = {0: 1.6, 1: 0.2, 2: 0.4}  # total 2.2 ≤ ceiling 3
        out = _redistribute_budget(raw)
        assert sum(out.values()) == pytest.approx(2.2, abs=1e-6)
        assert out[0] == pytest.approx(1.0)
        # 0.6 excess over {1: 0.2, 2: 0.4} → +0.2, +0.4.
        assert out[1] == pytest.approx(0.2 + 0.6 * (0.2 / 0.6), abs=1e-6)
        assert out[2] == pytest.approx(0.4 + 0.6 * (0.4 / 0.6), abs=1e-6)

    def test_sign_preserved(self) -> None:
        raw = {0: -1.7, 1: -0.1, 2: 0.1}
        out = _redistribute_budget(raw)
        assert out[0] < 0.0
        assert abs(out[0]) <= 1.0 + 1e-9
        # Magnitudes water-fill; signs ride through.
        mags = _redistribute_budget({k: abs(v) for k, v in raw.items()})
        for k in raw:
            assert abs(out[k]) == pytest.approx(mags[k], abs=1e-6)


# ---------------------------------------------------------------------------
# End-to-end: peaked-share vector profile under angular mode never asks a
# layer to rotate past θ_max, and the cumulative budget is conserved.
# ---------------------------------------------------------------------------


class _Passthrough(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden,)


def _peaked_profile(dim: int = 16) -> dict[int, torch.Tensor]:
    """One layer dominates the share (large ||baked||), two are small."""
    torch.manual_seed(0)
    big = torch.randn(dim)
    big = big / big.norm() * 10.0      # ||baked|| = 10
    small1 = torch.randn(dim)
    small1 = small1 / small1.norm() * 1.0
    small2 = torch.randn(dim)
    small2 = small2 / small2.norm() * 1.0
    return {0: big, 1: small1, 2: small2}


def _effective_alphas(mgr: SteeringManager) -> dict[int, float]:
    """The per-layer angular strength the hooks will rotate by.

    Each peaked-share layer carries a single BOTH additive group; the
    hook stamps the angular strength (≈ |effective_alpha| for a single
    unit-direction term) into ``angular_strengths``.  We read it back to
    assert the per-layer rotation budget.
    """
    out: dict[int, float] = {}
    for idx, hook in mgr.hooks.items():
        # Fast path stamps _theta directly; slow path keeps strengths.
        if hook.composed is not None:
            out[idx] = hook._theta / hook.theta_max
        elif hook.angular_strengths:
            out[idx] = sum(min(1.0, s) for _t, s in hook.angular_strengths)
    return out


class TestVectorAngularBudget:
    def test_peaked_share_no_layer_past_theta_max(self) -> None:
        mgr = SteeringManager(injection_mode="angular")
        profile = _peaked_profile()
        # α large enough that the dominant layer's raw budget (≈ α·0.83)
        # would exceed 1.0 without the cap.
        mgr.add_vector("c", profile, alpha=2.0)
        layers = nn.ModuleList([_Passthrough() for _ in range(3)])
        mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)

        eff = _effective_alphas(mgr)
        assert eff, "expected angular budgets to be recorded"
        for idx, budget in eff.items():
            assert budget <= 1.0 + 1e-6, (
                f"layer {idx} rotates past θ_max: budget={budget}"
            )

    def test_budget_conserved_under_cap(self) -> None:
        # Cumulative budget = min(α, n_layers) when α fits under the
        # ceiling.  Here α=2.0 over 3 layers ⇒ Σ budget == 2.0.
        mgr = SteeringManager(injection_mode="angular")
        mgr.add_vector("c", _peaked_profile(), alpha=2.0)
        layers = nn.ModuleList([_Passthrough() for _ in range(3)])
        mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
        eff = _effective_alphas(mgr)
        assert sum(eff.values()) == pytest.approx(2.0, abs=1e-4)


# ---------------------------------------------------------------------------
# End-to-end: peaked-share manifold under angular mode never rotates a
# layer past θ_max.  Build a synthetic manifold whose per-layer Euclidean
# centroid spread is sharply peaked so one layer's raw share·gain blows
# past 1.0 without the redistribute.
# ---------------------------------------------------------------------------


def _peaked_manifold(n_layers: int = 4, dim: int = 8) -> Manifold:
    """1-D BoxDomain manifold with one layer's centroid spread dominant.

    Layer 0 gets a large centroid scale (high Euclidean share), the rest
    tiny — so layer 0's ``α · share_L · _MANIFOLD_GAIN_ANGULAR`` exceeds
    1.0 at the default gain (8.0) for any α above ~0.13.
    """
    torch.manual_seed(0)
    domain = BoxDomain([BoxAxis("u", periodic=False, lo=-1.0, hi=1.0)])
    coords = torch.tensor([[-1.0], [0.0], [1.0]])
    e1 = torch.zeros(dim)
    e1[0] = 1.0
    e2 = torch.zeros(dim)
    e2[1] = 1.0
    layers: dict[int, LayerSubspace] = {}
    ev: dict[int, float] = {}
    for layer_idx in range(n_layers):
        scale = 20.0 if layer_idx == 0 else 0.5
        centroids = torch.stack([-scale * e1, torch.zeros(dim), scale * e1])
        centroids = centroids + 0.01 * torch.stack(
            [-e2, torch.zeros(dim), e2]
        )
        sub, ev_ratio = fit_layer_subspace(centroids, domain.embed(coords))
        layers[layer_idx] = sub
        ev[layer_idx] = ev_ratio
    return Manifold(
        name="peaked",
        domain=domain,
        node_labels=["a", "b", "c"],
        node_coords=coords,
        layers=layers,
        explained_variance=ev,
    )


class TestManifoldAngularBudget:
    def test_peaked_share_no_layer_past_theta_max(self) -> None:
        mgr = SteeringManager(injection_mode="angular")
        m = _peaked_manifold()
        # α=1.0 with the dominant layer's share ≈ 1.0 and gain 8.0 would
        # ask for an 8× θ_max rotation on layer 0 absent the redistribute.
        mgr.add_manifold("peaked", m, (0.0,), alpha=1.0)
        n_layers = max(m.layers) + 2
        layers = nn.ModuleList([_Passthrough() for _ in range(n_layers)])
        mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)

        # Read the effective_alpha stamped on each manifold group: it is
        # the rotation budget subspace_rotate multiplies θ_max by.
        seen = 0
        for idx, hook in mgr.hooks.items():
            for _trig, _basis, _mean, _target, alpha in hook.manifold_groups:
                seen += 1
                assert alpha <= 1.0 + 1e-6, (
                    f"layer {idx} manifold rotation past θ_max: {alpha}"
                )
        assert seen == len(m.layers)

    def test_manifold_budget_conserved(self) -> None:
        mgr = SteeringManager(injection_mode="angular")
        m = _peaked_manifold()
        mgr.add_manifold("peaked", m, (0.0,), alpha=1.0)
        n_layers = max(m.layers) + 2
        layers = nn.ModuleList([_Passthrough() for _ in range(n_layers)])
        mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)

        total = 0.0
        for hook in mgr.hooks.values():
            for _t, _b, _m, _tg, alpha in hook.manifold_groups:
                total += alpha
        # Raw cumulative budget = α · _MANIFOLD_GAIN_ANGULAR (share sums to
        # 1) = 1.0 · 8.0 = 8.0, capped to the ceiling = n_fit_layers.
        n_fit = len(m.layers)
        assert total == pytest.approx(min(8.0, n_fit), abs=1e-4)
