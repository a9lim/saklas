"""Topology auto-selection — ``fit_mode="auto"`` machinery (CPU only).

Covers the two decoupled decisions in :func:`select_topology`:

  (a) flat (``pca``) vs curved (``spectral``), by GCV in a shared whitened-
      reduced metric — validated on linearly-embedded (flat) vs nonlinearly-
      embedded (curved) synthetic manifolds;
  (b) periodic axes, by Vietoris–Rips H1 *persistent homology* counting the
      loops (ellipse/noise-robust) and the spectral eigenpairs coordinating
      them.

The PH loop counter is tested directly on distance matrices of known topology;
the full selector through a synthetic whitener + per-layer consensus Gram.
"""
from __future__ import annotations

import math

import pytest
import torch

from saklas.core.manifold import (
    BoxDomain,
    CustomDomain,
    _count_persistent_loops,
    _is_angular_harmonic,
    select_topology,
)
from tests._whitener import isotropic_whitener

_LAYERS = list(range(6))
_D = 48


def _circle(k: int) -> torch.Tensor:
    th = torch.linspace(0, 2 * math.pi, k + 1)[:-1]
    return torch.stack([torch.cos(th), torch.sin(th)], dim=1)


def _ellipse(k: int, a: float, b: float) -> torch.Tensor:
    th = torch.linspace(0, 2 * math.pi, k + 1)[:-1]
    rot = torch.tensor([[0.8, -0.6], [0.6, 0.8]])
    return torch.stack([a * torch.cos(th), b * torch.sin(th)], dim=1) @ rot


def _torus_t2(side: int) -> torch.Tensor:
    g = torch.linspace(0, 2 * math.pi, side + 1)[:-1]
    a, b = torch.meshgrid(g, g, indexing="ij")
    return torch.stack(
        [torch.cos(a).flatten(), torch.sin(a).flatten(),
         torch.cos(b).flatten(), torch.sin(b).flatten()], dim=1,
    )


def _sphere(k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(k, 3, generator=g)
    return v / v.norm(dim=1, keepdim=True)


# ----------------------------------------------------- PH H1 loop counter ---

@pytest.mark.parametrize("name,points,want", [
    ("circle", _circle(40), 1),
    ("ellipse-4:1", _ellipse(40, 3.0, 0.7), 1),
    ("ellipse-6:1", _ellipse(40, 6.0, 1.0), 1),
    ("torus-T2", _torus_t2(8), 2),
    ("blob", None, 0),          # filled in below (needs a seed)
    ("arc", torch.stack([torch.linspace(0, 1, 40), torch.linspace(0, 1, 40) ** 2], 1), 0),
    ("line", torch.stack([torch.linspace(0, 1, 30), 2 * torch.linspace(0, 1, 30)], 1), 0),
    ("sphere-S2", _sphere(80, 3), 0),
])
def test_ph_loop_count(name: str, points: torch.Tensor | None, want: int) -> None:
    if points is None:  # blob
        points = torch.randn(60, 2, generator=torch.Generator().manual_seed(2))
    got = _count_persistent_loops(torch.cdist(points, points))
    assert got == want, f"{name}: H1={got}, want {want}"


def test_ph_noisy_circle_is_one_loop() -> None:
    g = torch.Generator().manual_seed(1)
    pts = _circle(40) + 0.05 * torch.randn(40, 2, generator=g)
    assert _count_persistent_loops(torch.cdist(pts, pts)) == 1


# ------------------------------------------------------ harmonic dedup -------

def test_angular_harmonic_detects_multiples() -> None:
    side = 6
    g = torch.linspace(0, 2 * math.pi, side + 1)[:-1]
    a, b = torch.meshgrid(g, g, indexing="ij")
    theta_a = a.flatten()
    theta_b = b.flatten()              # an independent torus factor
    second = (2.0 * theta_a) % (2.0 * math.pi)   # 2nd harmonic of theta_a
    assert _is_angular_harmonic(second, [theta_a]) is True
    # The other torus factor is genuinely independent — not a harmonic.
    assert _is_angular_harmonic(theta_b, [theta_a]) is False


# ----------------------------------------------------- full select_topology -

def _stacks_from_coords(
    low: torch.Tensor, *, noise: float = 0.02,
) -> dict[int, torch.Tensor]:
    """Per-layer activation centroids: a random linear lift of ``low`` per layer."""
    k, p = low.shape
    out: dict[int, torch.Tensor] = {}
    for layer in _LAYERS:
        g = torch.Generator().manual_seed(100 + layer)
        proj = torch.randn(p, _D, generator=g)
        out[layer] = (low @ proj + noise * torch.randn(k, _D, generator=g)).float()
    return out


def _nonlinear_curve(k: int) -> torch.Tensor:
    """A genuinely curved (nonlinearly-embedded) 1-D manifold — not in any plane."""
    t = torch.linspace(0, 1, k)
    return torch.stack([torch.sin(5 * t), torch.cos(7 * t), torch.sin(11 * t)], dim=1)


def _choose(low: torch.Tensor, *, noise: float = 0.02):
    wh = isotropic_whitener(_LAYERS, _D)
    stacks = _stacks_from_coords(low, noise=noise)
    grams = {
        layer: wh.subspace_gram(layer, stacks[layer] - stacks[layer].mean(0, keepdim=True))
        for layer in _LAYERS
    }
    consensus = torch.stack([grams[layer] for layer in _LAYERS]).mean(0)
    return select_topology(stacks, grams, consensus, whitener=wh, max_dim=6)


def test_select_flat_blob_is_pca() -> None:
    g = torch.Generator().manual_seed(0)
    choice = _choose(torch.randn(50, 3, generator=g))
    assert choice.fit_mode == "pca"
    assert isinstance(choice.domain, CustomDomain)


def test_select_line_is_pca() -> None:
    t = torch.linspace(0, 1, 30)
    choice = _choose(torch.stack([t, 2 * t, -t], dim=1))
    assert choice.fit_mode == "pca"


def test_select_nonlinear_curve_is_spectral() -> None:
    choice = _choose(_nonlinear_curve(40), noise=0.005)
    assert choice.winner_name == "spectral"
    assert choice.fit_mode == "spectral"
    assert isinstance(choice.domain, CustomDomain)


def test_select_circle_is_periodic() -> None:
    choice = _choose(_circle(40))
    assert choice.winner_name == "torus-T1"
    assert choice.fit_mode == "spectral"
    assert isinstance(choice.domain, BoxDomain)
    assert choice.domain.axes[0].periodic
    assert choice.coords.shape[1] == 1


def test_select_ellipse_is_periodic() -> None:
    # The case that defeated the geometric heuristic: a linearly-mapped circle
    # is an ellipse, but PH still reads one loop.
    choice = _choose(_ellipse(40, 3.0, 1.0))
    assert choice.winner_name == "torus-T1"
    assert isinstance(choice.domain, BoxDomain)


def test_select_torus_t2_is_periodic_2d() -> None:
    choice = _choose(_torus_t2(8))
    assert choice.winner_name == "torus-T2"
    assert isinstance(choice.domain, BoxDomain)
    assert choice.coords.shape[1] == 2
    assert all(ax.periodic for ax in choice.domain.axes)


def test_select_records_candidate_ranking() -> None:
    g = torch.Generator().manual_seed(0)
    choice = _choose(torch.randn(50, 3, generator=g))
    names = {c.name for c in choice.candidates}
    assert "flat-pca" in names
    # Candidates are ranked viable-first by score (GCV); the winner is present.
    assert any(c.name == choice.winner_name for c in choice.candidates)
    assert choice.candidates[0].viable
