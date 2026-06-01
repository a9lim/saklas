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

from typing import Any

import pytest
import torch
import torch.nn as nn

from saklas.core.hooks import (
    _MANIFOLD_GAIN,
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


# ``_Passthrough`` returns ``(hidden,)`` so a hook can mutate it in place —
# shared by the manifold-along budget tests below.  (The former vector
# angular-budget tests are gone: a vector no longer water-fills — the merged
# affine subspace clamps its per-layer slide.  ``_redistribute_budget`` survives
# only on the curved-manifold ``along`` path, exercised by
# ``TestManifoldAlongBudget``.)


class _Passthrough(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden,)


# ---------------------------------------------------------------------------
# End-to-end: peaked-share manifold under angular mode never rotates a
# layer past θ_max.  Build a synthetic manifold whose per-layer Euclidean
# centroid spread is sharply peaked so one layer's raw share·gain blows
# past 1.0 without the redistribute.
# ---------------------------------------------------------------------------


def _peaked_manifold(n_layers: int = 4, dim: int = 8) -> Manifold:
    """1-D BoxDomain manifold with one layer's centroid spread dominant.

    Layer 0 gets a large centroid scale (high Euclidean share), the rest
    tiny — so layer 0's ``along · share_L · _MANIFOLD_GAIN`` exceeds
    1.0 (with share_0 ≈ 1 and base gain 2.0, for any α above ~0.5),
    forcing the water-fill cap + redistribute.  No lever stamped, so the
    apply path uses ``N = 1`` (the un-normalized base gain).
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


def _group_along(hook: Any) -> list[float]:
    """The per-layer along budget (index 5 of the three-op group tuple)."""
    return [grp[5] for grp in hook.manifold_groups]


class TestManifoldAlongBudget:
    def test_peaked_share_no_layer_past_target(self) -> None:
        mgr = SteeringManager()
        m = _peaked_manifold()
        # along=1.0 with the dominant layer's share ≈ 1.0 and base gain 2.0
        # would ask for a 2× full-slide on layer 0 absent the water-fill.
        mgr.add_manifold("peaked", m, (0.0,), along=1.0, onto=0.0)
        n_layers = max(m.layers) + 2
        layers = nn.ModuleList([_Passthrough() for _ in range(n_layers)])
        mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)

        # The per-layer along budget is the geodesic-slide fraction; the
        # water-fill caps it at 1.0 (= fully on the target at that layer).
        seen = 0
        for idx, hook in mgr.hooks.items():
            for along in _group_along(hook):
                seen += 1
                assert along <= 1.0 + 1e-6, (
                    f"layer {idx} manifold slides past target: {along}"
                )
        assert seen == len(m.layers)

    def test_manifold_along_budget_conserved(self) -> None:
        mgr = SteeringManager()
        m = _peaked_manifold()
        mgr.add_manifold("peaked", m, (0.0,), along=1.0, onto=0.0)
        n_layers = max(m.layers) + 2
        layers = nn.ModuleList([_Passthrough() for _ in range(n_layers)])
        mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)

        total = sum(
            along for hook in mgr.hooks.values() for along in _group_along(hook)
        )
        # Raw cumulative budget = along · _MANIFOLD_GAIN (share sums to 1; no
        # lever ⇒ N=1) = 1.0 · 2.0, capped to the ceiling = n_fit.
        n_fit = len(m.layers)
        assert total == pytest.approx(
            min(1.0 * _MANIFOLD_GAIN, n_fit), abs=1e-4,
        )
