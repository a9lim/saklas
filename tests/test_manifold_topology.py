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
from typing import Any

import pytest
import torch

from saklas.core.manifold import (
    BoxDomain,
    CustomDomain,
    _count_persistent_loops,
    _faint_cycle_coords,
    _is_angular_harmonic,
    select_topology,
)
from saklas.core.mahalanobis import LayerWhitener
from tests._whitener import isotropic_whitener, rogue_whitener

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


def test_ph_dense_complete_complex_no_spurious_loops() -> None:
    """A (near-)complete Rips complex has trivial H1 — the triangle cap must be
    large enough not to *manufacture* loops by truncating the filling triangles.

    Regression for the ``personas`` 8-torus.  109 tightly-clustered points plus
    one far outlier force ``eps_c`` (the largest MST edge) up to the outlier
    distance, so ``eps_max = 2·eps_c`` puts *every* pair inside the ceiling — the
    complex is complete and its true H1 is 0.  But that is ``C(109,3) ≈ 210k``
    triangles; the old ``max_triangles=150_000`` cap dropped the largest-
    filtration ones, leaving ~800 cycles born-but-unfillable and miscounted as
    essential (which routed the 107-node ``personas`` heap to a spurious
    8-torus).  The raised cap keeps every triangle across the supported regime.
    """
    from saklas.core.manifold import _rips_h1_persistence

    g = torch.Generator().manual_seed(7)
    cluster = torch.randn(109, 6, generator=g) * 0.1
    outlier = torch.zeros(1, 6)
    outlier[0, 0] = 50.0
    D = torch.cdist(torch.cat([cluster, outlier]), torch.cat([cluster, outlier]))

    # The fix: no spurious loops on the dense complex.
    assert _count_persistent_loops(D) == 0

    # And the cap is genuinely load-bearing: at the same (complete) ceiling the
    # starved budget *does* manufacture essential cycles while the current
    # default keeps every triangle and reports none — so the regression is real,
    # not incidental to this particular heap.
    K = D.shape[0]
    iu = torch.triu_indices(K, K, offset=1)
    eps_max = 2.0 * float(D[iu[0], iu[1]].max())  # ≥ every pair ⇒ complete
    starved = _rips_h1_persistence(D, eps_max, max_triangles=150_000)
    ample = _rips_h1_persistence(D, eps_max, max_triangles=500_000)
    assert sum(1 for _b, death in starved if math.isinf(death)) > 0
    assert sum(1 for _b, death in ample if math.isinf(death)) == 0


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


# ------------------------------------ faint single-cycle fallback (S^1) ------
#
# H1 persistence counts loops by *hole size*, so a faint ring (a small cyclic
# modulation on a near-equidistant heap — e.g. day-of-week centroids) slips
# under its threshold. ``_faint_cycle_coords`` is the complementary
# graph-topological detector that fires only when PH found nothing.

def _faint_ring(k: int, mod: float = 0.16, common_dims: int = 30) -> torch.Tensor:
    """A faint ``S^1``: a small cyclic modulation + many constant dims, so the
    points are near-equidistant (thin hole that PH would miss)."""
    th = torch.linspace(0, 2 * math.pi, k + 1)[:-1]
    ring = mod * torch.stack([torch.cos(th), torch.sin(th)], dim=1)
    return torch.cat([ring, torch.ones(k, common_dims)], dim=1)


def _arc(k: int, deg: float) -> torch.Tensor:
    th = torch.linspace(0, math.radians(deg), k)
    return torch.stack([torch.cos(th), torch.sin(th)], dim=1)


def _theta_graph() -> torch.Tensor:
    """Two hubs joined by three parallel routes — maxdeg-3 but not S^1."""
    mids = torch.linspace(0.5, 3.5, 3)
    return torch.cat([
        torch.tensor([[0.0, 0.0], [4.0, 0.0]]),
        torch.stack([mids, torch.full((3,), 1.0)], dim=1),
        torch.stack([mids, torch.full((3,), -1.0)], dim=1),
        torch.stack([mids, torch.zeros(3)], dim=1),
    ], dim=0)


def _grid(side: int) -> torch.Tensor:
    g = torch.arange(side, dtype=torch.float32)
    a, b = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([a.flatten(), b.flatten()], dim=1)


def _persona_fan(k: int, rank: int, seed: int) -> torch.Tensor:
    """A high-D non-cyclic fan — the dangerous false-positive (personas)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, rank, generator=g).abs() @ torch.randn(rank, 40, generator=g)


@pytest.mark.parametrize("name,points,want", [
    ("clean-circle-K9", _circle(9), True),
    ("faint-ring-K7", _faint_ring(7), True),
    ("faint-ring-K12", _faint_ring(12), True),
    ("line-K7", torch.stack([torch.linspace(0, 1, 7), torch.zeros(7)], 1), False),
    ("arc-210-K7", _arc(7, 210), False),
    ("arc-300-K12", _arc(12, 300), False),
    ("theta", _theta_graph(), False),
    ("grid-4x4", _grid(4), False),
    ("persona-fan-K30", _persona_fan(30, 8, 1), False),
    ("persona-fan-K107", _persona_fan(107, 8, 2), False),
    ("too-few-K5", _circle(5), False),  # below the K>=7 reliability gate
])
def test_faint_cycle_detector(name: str, points: torch.Tensor, want: bool) -> None:
    got = _faint_cycle_coords(torch.cdist(points, points)) is not None
    assert got is want, name


def test_faint_cycle_recovers_uniform_ordered_angles() -> None:
    """The recovered S^1 coordinate is uniform and in the cyclic order."""
    coords = _faint_cycle_coords(torch.cdist(_faint_ring(8), _faint_ring(8)))
    assert coords is not None
    # Each node lands on a distinct 2*pi*i/8 grid point; sorted gaps are uniform.
    ordered = torch.sort(coords).values
    gaps = torch.diff(ordered)
    assert torch.allclose(gaps, gaps.mean(), atol=1e-4)


def test_faint_cycle_false_positive_rate_low() -> None:
    """Random Gaussian heaps must rarely read as cyclic (the FP guard)."""
    fp = 0
    for seed in range(120):
        g = torch.Generator().manual_seed(seed)
        pts = torch.randn(9, 6, generator=g)
        fp += _faint_cycle_coords(torch.cdist(pts, pts)) is not None
    assert fp <= 2, f"random-heap false positives too high: {fp}/120"


def test_select_faint_ring_is_periodic() -> None:
    """End-to-end: a faint ring PH misses still routes to a periodic BoxDomain."""
    choice = _choose(_faint_ring(9, mod=0.16, common_dims=0)[:, :2], noise=0.01)
    # (a plain faint circle through the synthetic whitener/consensus path)
    assert choice.winner_name == "torus-T1"
    assert isinstance(choice.domain, BoxDomain)
    assert choice.domain.axes[0].periodic


def test_select_blob_not_spuriously_periodic() -> None:
    """The fallback must not turn a random blob into a ring."""
    for seed in range(8):
        g = torch.Generator().manual_seed(seed)
        choice = _choose(torch.randn(9, 4, generator=g))
        assert not isinstance(choice.domain, BoxDomain), f"seed {seed} spurious ring"


# ----------------------------- clustered-ring fallback ----------------------
#
# Real concept families don't tile a ring uniformly — months cluster into
# seasons, days into weekday/weekend. A clustered ring (tight clumps spaced
# around the loop) makes the tour edges *bimodal*, so H1 counts no fat loop AND
# the uniform faint path's closure/recall guards fail, yet the loop is real. The
# clustered path recovers it from >=2 regular inter-cluster gaps + a real far
# antipode, while still rejecting the 2-D grids and high-D fans that the looser
# degree bound would otherwise admit.


def _clustered_ring(k_per: int, n_clusters: int, spread: float) -> torch.Tensor:
    """A ring sampled in ``n_clusters`` tight clumps of ``k_per`` (seasonal
    sampling); ``spread`` is each clump's angular half-width (radians)."""
    centers = torch.linspace(0, 2 * math.pi, n_clusters + 1)[:-1]
    th = torch.cat([c + torch.linspace(-spread, spread, k_per) for c in centers])
    return torch.stack([torch.cos(th), torch.sin(th)], dim=1)


def _gapped_ring(k: int, gap_sectors: int) -> torch.Tensor:
    """A uniform ring with one contiguous sector removed. Geometrically this is
    *identical* to an open arc — same point cloud — so it correctly stays
    NON-periodic; the closure guard rejecting both is right, not a miss."""
    full = k + gap_sectors
    th = torch.linspace(0, 2 * math.pi, full + 1)[:-1][:k]
    return torch.stack([torch.cos(th), torch.sin(th)], dim=1)


@pytest.mark.parametrize("name,points,want", [
    ("clustered-4x3", _clustered_ring(3, 4, 0.18), True),
    ("clustered-3x4", _clustered_ring(4, 3, 0.25), True),
    ("clustered-6x2", _clustered_ring(2, 6, 0.20), True),
    ("clustered-4x5", _clustered_ring(5, 4, 0.20), True),
    ("clustered-5x3", _clustered_ring(3, 5, 0.22), True),
    # must reject — the impostors the clustered path's guards screen out:
    ("grid-5x5", _grid(5), False),       # 2-D: gaps not decisively bimodal
    ("grid-6x6", _grid(6), False),
    ("grid-7x7", _grid(7), False),
    ("persona-fan-K60", _persona_fan(60, 8, 4), False),  # high-D: degree >> 4
    ("gapped-K22-g2", _gapped_ring(22, 2), False),   # == open arc (1 gap)
    ("gapped-K18-g6", _gapped_ring(18, 6), False),
])
def test_faint_cycle_clustered(name: str, points: torch.Tensor, want: bool) -> None:
    got = _faint_cycle_coords(torch.cdist(points, points)) is not None
    assert got is want, name


def test_faint_cycle_clustered_recovers_cyclic_order() -> None:
    """The clustered path recovers the true cyclic order: each input clump stays
    contiguous on the recovered loop. Walking the recovered cycle, the cluster id
    changes exactly ``n_clusters`` times (a scrambled tour would change far more).
    Robust to the tour's start point, rotation, and reflection."""
    k_per, n_clusters = 3, 4
    pts = _clustered_ring(k_per, n_clusters, 0.18)
    coords = _faint_cycle_coords(torch.cdist(pts, pts))
    assert coords is not None
    order = torch.argsort(coords).tolist()            # nodes in recovered cyclic order
    cluster_of = [i // k_per for i in order]
    transitions = sum(cluster_of[i] != cluster_of[(i + 1) % len(order)]
                      for i in range(len(order)))
    assert transitions == n_clusters


def test_select_clustered_ring_is_periodic() -> None:
    """End-to-end: a clustered ring routes to a periodic BoxDomain (the case the
    uniform faint path and H1 both miss)."""
    choice = _choose(_clustered_ring(3, 4, 0.18), noise=0.03)
    assert choice.winner_name == "torus-T1"
    assert isinstance(choice.domain, BoxDomain)
    assert choice.domain.axes[0].periodic


def test_select_grid_not_spuriously_periodic() -> None:
    """The looser clustered-path degree bound must not turn a 2-D grid (which now
    reaches the tour stage) into a ring."""
    for side in (5, 6, 7):
        choice = _choose(_grid(side), noise=0.03)
        assert not isinstance(choice.domain, BoxDomain), f"grid-{side} spurious ring"


# ====================== flat<->curved dim-match (flat-bias fix) =============
#
# The flat-vs-curved GCV comparison used to be unfair: the flat candidate took
# its PCA variance-threshold dim while the curved candidate took the spectral
# eigenvalue-ratio-cliff dim, and that cliff systematically *undershoots* (one
# dominant Fiedler mode picks k=1).  So a curved manifold linearly embedded in a
# k_flat-plane read flat -- the flat affine fit reconstructs the in-plane curve
# near-perfectly while the under-dimensioned curved fit can't match it, losing on
# reconstruction it would *win* at matched dim.  ``select_topology`` now floors
# the curved candidate to the flat dim (``min_dim=k_flat``).  These guard that
# fix and ensure it does not spuriously flip genuinely flat shapes.


def _curve_in_plane(k: int, c: float) -> torch.Tensor:
    """A 1-D curve whose nonlinear (curved) energy is set by ``c``.

    At ``c=0`` the embedding is affine in the intrinsic ``t`` (a straight line);
    as ``c`` grows the quadratic/cubic terms an affine map can't reproduce grow
    while the cloud stays inside a low-dim plane -- the geometry that read flat
    before the dim-match fix.
    """
    t = torch.linspace(0, 1, k)

    def s(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean()) / x.std().clamp(min=1e-9)

    return torch.stack(
        [s(t), c * s(t * t - t), c * s(t ** 3 - 1.5 * t ** 2 + 0.5 * t)], dim=1,
    )


@pytest.mark.parametrize("c", [0.8, 1.2, 2.0])
def test_select_curve_in_plane_is_curved(c: float) -> None:
    """A genuinely curved manifold embedded in a plane reads curved, not flat.

    The headline flat-bias regression: without the dim-match floor this routed to
    ``flat-pca`` for every ``c`` (the flat 2-affine fit reconstructs the in-plane
    curve, the spectral candidate undershoots to 1-D and loses).
    """
    choice = _choose(_curve_in_plane(40, c), noise=0.01)
    assert choice.fit_mode == "spectral", (
        f"c={c}: curved-in-plane mislabelled {choice.winner_name!r}"
    )


@pytest.mark.parametrize("c", [0.8, 1.2, 2.0])
def test_select_curved_candidate_dim_matches_flat(c: float) -> None:
    """The dim-match invariant: the viable curved candidate is floored to the flat dim.

    This is exactly what ``min_dim=k_flat`` guarantees; without it the spectral
    cliff undershoots and the GCV comparison is rigged toward flat.
    """
    choice = _choose(_curve_in_plane(40, c), noise=0.01)
    flat = next(x for x in choice.candidates if x.name == "flat-pca")
    spec = next((x for x in choice.candidates if x.name == "spectral"), None)
    assert spec is not None and spec.viable
    assert spec.intrinsic_dim >= flat.intrinsic_dim, (
        f"c={c}: curved dim {spec.intrinsic_dim} < flat dim {flat.intrinsic_dim}"
    )


@pytest.mark.parametrize("name,low", [
    ("line", torch.stack([torch.linspace(0, 1, 30),
                          2 * torch.linspace(0, 1, 30),
                          -torch.linspace(0, 1, 30)], 1)),
    ("blob", torch.randn(40, 3, generator=torch.Generator().manual_seed(0))),
    ("plane", torch.randn(40, 2, generator=torch.Generator().manual_seed(1))),
    ("grid", _grid(6)),
    ("persona-fan", _persona_fan(40, 8, 2)),
    ("arc-shallow", _arc(30, 90)),
])
def test_select_flat_shapes_unaffected_by_dim_match(name: str, low: torch.Tensor) -> None:
    """The dim-match fix must not turn genuinely flat shapes curved.

    A flat fan (personas), plane, grid, line, or shallow arc stays ``pca``: the
    floored curved candidate gains coordinates but no reconstruction it can't get
    affinely, so GCV's edf penalty keeps flat the winner.
    """
    choice = _choose(low, noise=0.02)
    assert choice.fit_mode == "pca", f"{name} spuriously curved -> {choice.winner_name!r}"


# ============================== rogue-channel invariance ====================
#
# Every topology test above runs under an isotropic whitener.  The whitened /
# Fisher metric exists for the opposite condition -- a few massive-activation
# channels at 100x+ the background -- which it must divide out.  These confirm
# the topology read is invariant to rogue channels (signal lifted into the clean
# dims; rogue dims are background-only).

_ROGUE_DIM = 48


def _stacks_clean(low: torch.Tensor, clean: list[int], *, noise: float,
                  dim: int = _ROGUE_DIM) -> dict[int, torch.Tensor]:
    """Per-layer lift restricted to ``clean`` dims (whitener-visible signal)."""
    k, p = low.shape
    low = low - low.mean(0, keepdim=True)
    low = low / low.std().clamp(min=1e-9)
    ct = torch.tensor(clean, dtype=torch.long)
    out: dict[int, torch.Tensor] = {}
    for layer in _LAYERS:
        g = torch.Generator().manual_seed(100 + layer)
        proj = torch.zeros(p, dim, dtype=torch.float32)
        proj[:, ct] = torch.randn(p, len(clean), generator=g) / math.sqrt(p)
        out[layer] = (low @ proj + noise * torch.randn(k, dim, generator=g)).float()
    return out


def _choose_clean(low: torch.Tensor, wh: LayerWhitener, clean: list[int],
                  *, noise: float):
    stacks = _stacks_clean(low, clean, noise=noise)
    grams = {
        L: wh.subspace_gram(L, stacks[L] - stacks[L].mean(0, keepdim=True))
        for L in _LAYERS
    }
    consensus = torch.stack([grams[L] for L in _LAYERS]).mean(0)
    return select_topology(stacks, grams, consensus, whitener=wh, max_dim=6)


@pytest.mark.parametrize("name,low,want_mode", [
    ("circle", _circle(36), "spectral"),       # periodic rides the curved path
    ("curve", _curve_in_plane(40, 1.2), "spectral"),
    ("blob", torch.randn(40, 3, generator=torch.Generator().manual_seed(0)), "pca"),
])
def test_select_invariant_to_rogue_channels(name: str, low: torch.Tensor,
                                            want_mode: str) -> None:
    """The topology read is invariant to massive-activation (rogue) channels.

    The Fisher metric divides them out, so an isotropic whitener and one with
    four channels at 200x the background must agree -- and on the right answer.
    """
    iso = isotropic_whitener(_LAYERS, _ROGUE_DIM)
    rogue, clean = rogue_whitener(_LAYERS, _ROGUE_DIM, rogue_mag=200.0)
    iso_choice = _choose_clean(low, iso, list(range(_ROGUE_DIM)), noise=0.03)
    rogue_choice = _choose_clean(low, rogue, clean, noise=0.03)
    assert iso_choice.fit_mode == rogue_choice.fit_mode == want_mode, (
        f"{name}: iso={iso_choice.winner_name!r} rogue={rogue_choice.winner_name!r}"
    )


# ================================= determinism ==============================


def test_select_topology_deterministic() -> None:
    """Repeated ``select_topology`` on identical input is bit-reproducible (CPU).

    Guards the sidecar / regression-test contract: the winner and every
    candidate's GCV score must match across calls (``inf == inf`` for an unviable
    candidate is fine; finite scores are bit-identical on CPU).
    """
    g = torch.Generator().manual_seed(0)
    shapes = [_circle(36), _curve_in_plane(40, 1.2), _torus_t2(7),
              torch.randn(40, 3, generator=g)]
    for low in shapes:
        a = _choose(low, noise=0.03)
        b = _choose(low, noise=0.03)
        assert a.winner_name == b.winner_name
        assert {c.name: c.score for c in a.candidates} == \
               {c.name: c.score for c in b.candidates}


def test_select_topology_reuses_laplacian_eigensystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import saklas.core.manifold as manifold_mod

    calls = 0
    real = manifold_mod._laplacian_eigen

    def _counted(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(manifold_mod, "_laplacian_eigen", _counted)
    choice = _choose(_circle(40))
    assert choice.winner_name == "torus-T1"
    assert calls == 1


def test_select_torus_t2_coarseness_floor() -> None:
    """T2 is reliably detected at >= 7 points per loop (the coarseness floor).

    Below ~7 the loops' holes fill inside the ``eps_max=2*eps_c`` window and read
    flat; side-7 is the validated floor (T3 and coarser tori are out of the
    practical envelope covered by the current topology selector).
    """
    choice = _choose(_torus_t2(7))
    assert choice.winner_name == "torus-T2"
