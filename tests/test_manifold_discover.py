"""Discover-mode coord derivation: derive_pca_coords / derive_spectral_coords.

Pure CPU, no model, no IO. The fit math is tensor over a centroid Gram; the
centroids are synthesized so every test exercises a known ground truth.
The Procrustes-agreement test in particular is the discriminating check
for "did we wire the methods up correctly": PCA and spectral must
agree on genuinely flat data and disagree on genuinely curved data.

The derive functions take a ``(K, K)`` consensus Gram, not raw centroids:
``derive_pca_coords`` eigendecomposes it and ``derive_spectral_coords`` reads
pairwise distances off it. For a single layer with ``Σ = I`` the Gram is the
centered Euclidean ``X̃ X̃ᵀ`` (``_centered_gram`` below), whose eigendecomposition
is exactly the PCA of the centroids and whose ``diag ⊕ diag − 2G`` distances are
``‖c_i − c_j‖`` — so these tests pin the single-layer behavior. The layer-agnostic
consensus (the mean of per-layer whitened Grams the pipeline actually feeds) is
covered by the signal-weighting tests at the bottom: a clean signal-layer Gram
averaged with an isotropic noise-layer Gram recovers the signal layout, because
whitened common-scale averaging is signal-weighted.
"""
from __future__ import annotations

import math

import pytest
import torch

from saklas.core.manifold import (
    PcaDiagnostics,
    SpectralDiagnostics,
    derive_pca_coords,
    derive_spectral_coords,
    discover_coords,
    neutral_layout_coord,
)


# ---------------------------------------------------------------- helpers ---

def _centered_gram(centroids: torch.Tensor) -> torch.Tensor:
    """The centered Euclidean Gram ``X̃ X̃ᵀ`` of ``(K, D)`` centroids.

    The single-layer (``Σ = I``) case of the consensus Gram the pipeline
    feeds the derive functions: its eigendecomposition is the PCA of the
    centroids, and the pairwise distances read off it
    (``diag ⊕ diag − 2 G``) equal ``‖c_i − c_j‖``, so it drives both methods.
    """
    xc = centroids.to(torch.float32)
    xc = xc - xc.mean(dim=0, keepdim=True)
    return xc @ xc.transpose(0, 1)


def _random_orthonormal_plane(d: int, k: int, *, seed: int) -> torch.Tensor:
    """A ``(d, k)`` matrix with orthonormal columns — a random k-plane in R^d."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, k, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q  # (d, k)


def _procrustes_align(
    a: torch.Tensor, b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonal Procrustes: return ``(a' , b')`` aligned and both centered.

    Both ``a`` and ``b`` are ``(N, k)``.  ``a'`` is ``a`` centered and
    then rotated/reflected by the orthogonal matrix minimizing
    ``||a R - b||_F``; ``b'`` is ``b`` centered.  Allows reflections —
    spectral coords are sign-ambiguous, so requiring a proper rotation
    would spuriously fail.
    """
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    M = a_c.T @ b_c  # (k, k)
    U, _, Vh = torch.linalg.svd(M)
    R = U @ Vh
    return a_c @ R, b_c


def _procrustes_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized Procrustes distance: ``||a' - b'||_F / ||b'||_F``."""
    a_aligned, b_c = _procrustes_align(a, b)
    num = torch.linalg.norm(a_aligned - b_c)
    den = torch.linalg.norm(b_c).clamp(min=1e-12)
    return float((num / den).item())


def _procrustes_distance_scaled(a: torch.Tensor, b: torch.Tensor) -> float:
    """Procrustes distance allowing uniform scale **and** rotation (similarity).

    A discover-coord layout is only defined up to a similarity transform —
    the absolute coord magnitude is meaningless (the per-layer fit recomputes
    real node coords, and RBF/affine interpolation re-normalizes to the unit
    box).  The rotation-only :func:`_procrustes_distance` is the right check
    when both layouts share a scale (orthonormal-plane embeddings); the
    consensus comparison must additionally absorb the global scale that the
    layer averaging introduces, which only this scaled form can.
    """
    a_aligned, b_c = _procrustes_align(a, b)
    s = float(
        (a_aligned * b_c).sum() / (a_aligned * a_aligned).sum().clamp(min=1e-12)
    )
    num = torch.linalg.norm(s * a_aligned - b_c)
    den = torch.linalg.norm(b_c).clamp(min=1e-12)
    return float((num / den).item())


def _circle_centroids(
    n: int, *, ambient: int, noise: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """N uniformly-spaced 2D circle points embedded into R^ambient + noise.

    Returns ``(centroids, theta)`` so the test can check the recovered
    layout against the generating angles.
    """
    g = torch.Generator().manual_seed(seed)
    theta = torch.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]  # exclude wrap
    plane = _random_orthonormal_plane(ambient, 2, seed=seed)  # (ambient, 2)
    circle_2d = torch.stack(
        [torch.cos(theta), torch.sin(theta)], dim=1,
    )                                                          # (N, 2)
    centroids = circle_2d @ plane.T                            # (N, ambient)
    centroids = centroids + noise * torch.randn(
        n, ambient, generator=g,
    )
    return centroids, theta


def _flat_centroids(
    n: int, *, ambient: int, rank: int, noise: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """N points in a random ``rank``-plane of R^ambient + isotropic noise.

    Returns ``(centroids, true_coords)`` — ``true_coords`` is the
    ``(N, rank)`` matrix the embedding came from, against which PCA's
    recovered layout must match (up to orthogonal rotation).
    """
    g = torch.Generator().manual_seed(seed)
    true_coords = torch.randn(n, rank, generator=g)
    plane = _random_orthonormal_plane(ambient, rank, seed=seed)
    centroids = true_coords @ plane.T
    centroids = centroids + noise * torch.randn(
        n, ambient, generator=g,
    )
    return centroids, true_coords


# ---------------------------------------------------------- PCA: headlines ---

def test_pca_recovers_flat_subspace_rank():
    """At a 0.95 threshold, PCA picks k=3 on rank-3 data and recovers the layout.

    The 0.70 default is the cumulative-variance picker, not a rank
    detector — random rank-3 data with one PC carrying ~45% and a
    second ~30% crosses 0.70 at k=2.  Threshold 0.95 forces the picker
    to keep going until the noise floor, which on a rank-3 dataset
    happens exactly at k=3.  The threshold-semantics test below
    exercises the lower-threshold contract directly.
    """
    centroids, true_coords = _flat_centroids(
        n=60, ambient=32, rank=3, noise=0.02, seed=11,
    )
    coords, diag = derive_pca_coords(
        _centered_gram(centroids), max_dim=8, var_threshold=0.95,
    )
    assert isinstance(diag, PcaDiagnostics)
    assert diag.picked_k == 3
    # Cumulative variance crosses the 95% threshold exactly at k=3.
    assert diag.cumulative_variance[2] >= 0.95
    assert diag.cumulative_variance[1] < 0.95
    # Coords have the right shape, and the recovered layout is an
    # orthogonal rotation of the generating coords (Procrustes distance
    # under noise floor).
    assert coords.shape == (60, 3)
    dist = _procrustes_distance(coords, true_coords)
    assert dist < 0.10, f"Procrustes distance {dist:.4f} too large"


def test_pca_threshold_semantics():
    """Picked k is the smallest prefix whose cumvar crosses the threshold.

    Direct test of the contract: ``cum_var[picked_k - 1] >= threshold``
    and ``cum_var[picked_k - 2] < threshold`` (when picked_k > 1).
    """
    centroids, _ = _flat_centroids(
        n=60, ambient=32, rank=3, noise=0.02, seed=11,
    )
    gram = _centered_gram(centroids)
    for threshold in (0.50, 0.70, 0.90, 0.99):
        _, diag = derive_pca_coords(
            gram, max_dim=8, var_threshold=threshold,
        )
        k = diag.picked_k
        assert diag.cumulative_variance[k - 1] >= threshold, (
            f"threshold {threshold}: cumvar at picked_k={k} is "
            f"{diag.cumulative_variance[k - 1]:.3f} < {threshold}"
        )
        if k > 1:
            assert diag.cumulative_variance[k - 2] < threshold, (
                f"threshold {threshold}: picked_k={k} but k-1 already "
                f"crossed ({diag.cumulative_variance[k - 2]:.3f})"
            )


def test_pca_picks_k1_when_one_direction_dominates():
    """A near-1D heap collapses to k=1 with cumvar already past threshold."""
    # 1D linear ramp dominates; everything else is tiny noise.
    g = torch.Generator().manual_seed(7)
    n, d = 40, 32
    ramp = torch.linspace(-5.0, 5.0, n).unsqueeze(1)  # (N, 1)
    plane = _random_orthonormal_plane(d, 1, seed=7)
    centroids = ramp @ plane.T + 0.01 * torch.randn(n, d, generator=g)
    coords, diag = derive_pca_coords(
        _centered_gram(centroids), max_dim=8, var_threshold=0.70,
    )
    assert diag.picked_k == 1
    assert diag.cumulative_variance[0] >= 0.70
    assert coords.shape == (40, 1)


def test_pca_caps_at_max_dim_when_threshold_unreachable():
    """Pure isotropic noise — no prefix crosses 70%, so we cap at max_dim."""
    g = torch.Generator().manual_seed(3)
    centroids = torch.randn(50, 32, generator=g)
    coords, diag = derive_pca_coords(
        _centered_gram(centroids), max_dim=4, var_threshold=0.99,
    )
    # Variance is spread across all 32 axes; threshold 0.99 unreachable
    # inside the cap.  Picked_k pins to max_dim.
    assert diag.picked_k == 4
    assert coords.shape == (50, 4)


def test_pca_rejects_too_few_centroids():
    """K=1 has no variance; the function refuses rather than divide by zero."""
    with pytest.raises(ValueError, match=">= 2 centroids"):
        derive_pca_coords(_centered_gram(torch.zeros(1, 32)))


def test_pca_rejects_non_square_gram():
    """A non-square input is a wiring error — the derive functions take Grams."""
    with pytest.raises(ValueError, match=r"square \(K, K\) Gram"):
        derive_pca_coords(torch.randn(10, 32))


# -------------------------------------------------- spectral: circle recovery ---

def test_spectral_recovers_circle_topology():
    """The headline test: S^1 → spectral picks k=2, recovers angular order."""
    centroids, theta = _circle_centroids(
        n=80, ambient=32, noise=0.02, seed=21,
    )
    coords, diag = derive_spectral_coords(_centered_gram(centroids), max_dim=6)
    assert isinstance(diag, SpectralDiagnostics)
    assert diag.component_count == 1
    # S^1 has two non-trivial Laplacian eigenvalues (the cos/sin pair)
    # before a clean gap to the next pair.  picked_k = 2.
    assert diag.picked_k == 2
    assert coords.shape == (80, 2)
    # The recovered 2-coord embedding wraps once around the origin.
    # Sort by atan2 of recovered coords; the original angular order
    # should be preserved up to reflection.
    rec_angle = torch.atan2(coords[:, 1], coords[:, 0])
    rec_order = torch.argsort(rec_angle)
    orig_order = torch.argsort(theta)
    # Cyclic shift + possible reflection: rotate orig_order until it
    # aligns with rec_order in either direction.
    rec_idx = rec_order.tolist()
    orig_idx = orig_order.tolist()

    def cyclic_match(a: list[int], b: list[int]) -> bool:
        n = len(a)
        for start in range(n):
            if all(a[(start + i) % n] == b[i] for i in range(n)):
                return True
        return False

    matches_forward = cyclic_match(orig_idx, rec_idx)
    matches_reflected = cyclic_match(orig_idx[::-1], rec_idx)
    assert matches_forward or matches_reflected, (
        "spectral embedding did not preserve circular order"
    )


def test_spectral_flat_subspace_picks_small_k():
    """On flat data, spectral still works — it just won't show a big gap.

    The recovered layout should still embed the rank, but the gap
    detection won't have a sharp peak.  This is the negative-space
    test for the headline: spectral isn't broken on flat input, it
    just doesn't have the structural advantage there.
    """
    centroids, _ = _flat_centroids(
        n=60, ambient=32, rank=3, noise=0.02, seed=33,
    )
    coords, diag = derive_spectral_coords(_centered_gram(centroids), max_dim=6)
    assert diag.component_count == 1
    # The fit succeeded and produced some k between 1 and max_dim.
    assert 1 <= diag.picked_k <= 6
    assert coords.shape == (60, diag.picked_k)


# ------------------------------------------------ agreement-iff-flat invariant ---

def test_pca_and_spectral_pick_same_k_on_flat_data():
    """Structural agreement on flat data: both methods pick the same intrinsic dim.

    PCA and spectral have different objectives — PCA preserves
    *variance*, spectral preserves *local distances* — so their
    coordinate values can disagree pointwise even on flat data,
    depending on the sampling pattern (a gaussian cloud and a uniform
    grid give different layouts under spectral, the same under PCA).
    The *structural* invariant — what intrinsic dimension each method
    recovers — should agree, and that's the discriminating wiring test:
    if PCA says rank-2 and spectral says rank-5 on data that genuinely
    lies on a flat 2-plane, one of them is miscomputed.
    """
    centroids, _ = _flat_centroids(
        n=60, ambient=32, rank=2, noise=0.02, seed=44,
    )
    gram = _centered_gram(centroids)
    _, pca_diag = derive_pca_coords(
        gram, max_dim=8, var_threshold=0.95,
    )
    _, spec_diag = derive_spectral_coords(gram, max_dim=8)
    assert pca_diag.picked_k == 2, (
        f"PCA picked k={pca_diag.picked_k} on rank-2 data"
    )
    # Spectral may pick 1 or 2 on a noisy 2-plane (low-rank noise can
    # blur the ratio cliff into the first eigengap); both are
    # structurally reasonable.  What's *not* reasonable is spectral
    # picking the max_dim cap on flat data — that flags a broken gap.
    assert spec_diag.picked_k <= 2, (
        f"spectral picked k={spec_diag.picked_k} on flat rank-2 data; "
        f"the eigengap heuristic is over-counting"
    )


def test_pca_and_spectral_diverge_on_curved_data():
    """On a circle, PCA flattens the loop and spectral preserves it.

    Inverse of the structural-agreement test: on genuinely curved
    input the *coordinate layouts* must differ substantially.  PCA's
    2-D embedding of a circle is the uniform 2-disk projection (a flat
    chord layout); spectral's is the unwrapped circle.  These should
    show a large Procrustes distance — the discriminator that the
    methods are actually behaving differently rather than collapsing
    to the same layout for trivial reasons.
    """
    centroids, _ = _circle_centroids(
        n=80, ambient=32, noise=0.02, seed=55,
    )
    gram = _centered_gram(centroids)
    pca_coords, _ = derive_pca_coords(
        gram, max_dim=2, var_threshold=0.95,
    )
    spec_coords, _ = derive_spectral_coords(gram, max_dim=2)
    if spec_coords.shape[1] != 2 or pca_coords.shape[1] != 2:
        pytest.skip(
            f"need k=2 from both for Procrustes match "
            f"(pca={pca_coords.shape[1]}, spec={spec_coords.shape[1]})"
        )
    dist = _procrustes_distance(spec_coords, pca_coords)
    assert dist > 0.40, (
        f"PCA and spectral agree on curved data (Procrustes dist "
        f"{dist:.3f}); spectral may be collapsing to PCA"
    )


# ----------------------------------------------------- error paths / safety ---

def test_spectral_rejects_disconnected_graph():
    """Two well-separated clusters with small k_nn → ValueError on n_components."""
    g = torch.Generator().manual_seed(99)
    cluster_a = torch.randn(20, 32, generator=g) * 0.3 + 10.0
    cluster_b = torch.randn(20, 32, generator=g) * 0.3 - 10.0
    centroids = torch.cat([cluster_a, cluster_b], dim=0)
    with pytest.raises(ValueError, match=r"2 connected components"):
        derive_spectral_coords(_centered_gram(centroids), max_dim=4, k_nn=3)


def test_spectral_rejects_too_few_centroids():
    """Below the floor K=4 the heuristics are pure noise; refuse early."""
    with pytest.raises(ValueError, match=">= 4 centroids"):
        derive_spectral_coords(_centered_gram(torch.zeros(3, 32)))


def test_spectral_diagnostics_record_data_driven_defaults():
    """The default ``k_nn`` and ``bandwidth`` ride into diagnostics for repro."""
    centroids, _ = _circle_centroids(
        n=50, ambient=16, noise=0.01, seed=66,
    )
    _, diag = derive_spectral_coords(_centered_gram(centroids), max_dim=4)
    assert diag.k_nn == max(5, math.ceil(math.log(50)))
    assert diag.bandwidth > 0.0
    assert diag.eigenvalues.shape[0] >= diag.picked_k


# ----------------------------------------------- min_dim dimensionality floor ---

def test_spectral_min_dim_floors_undershooting_cliff():
    """``min_dim`` honors a declared intrinsic dim over the ratio cliff.

    The PAD failure mode: when one mode dominates the spectrum the
    eigenvalue-ratio cliff undershoots (a small first non-trivial / Fiedler
    eigenvalue makes ``λ_2 / λ_1`` the largest ratio → ``k=1``), collapsing
    P×A×D to the valence axis.  A clean circle has intrinsic dim 2, so the
    cliff picks ``k=2`` — below a declared ``min_dim=3``, which the floor
    must raise the embedding back up to, recording that it fired.
    """
    centroids, _ = _circle_centroids(n=60, ambient=32, noise=0.02, seed=5)
    gram = _centered_gram(centroids)
    # Unpinned: the cliff picks the circle's intrinsic dim (2).
    _, free = derive_spectral_coords(gram, max_dim=6)
    assert free.pinned is False
    assert free.min_dim is None
    assert free.heuristic_k == free.picked_k  # records its own pick when free
    assert free.picked_k < 3, (
        f"precondition: circle should pick k<3, got {free.picked_k}"
    )
    # Pinned to 3: the floor raises picked_k and the coords widen to match.
    coords, diag = derive_spectral_coords(gram, max_dim=6, min_dim=3)
    assert diag.picked_k == 3
    assert diag.gap_index == 3          # the alias tracks the floored pick
    assert diag.min_dim == 3
    assert diag.pinned is True
    assert diag.heuristic_k == free.picked_k  # the heuristic still ran
    assert coords.shape == (60, 3)


def test_spectral_min_dim_at_or_below_pick_is_noop():
    """A floor at/below the heuristic pick leaves pick and shape untouched."""
    centroids, _ = _circle_centroids(n=60, ambient=32, noise=0.02, seed=5)
    gram = _centered_gram(centroids)
    coords_free, free = derive_spectral_coords(gram, max_dim=6)
    coords_pin, pinned = derive_spectral_coords(gram, max_dim=6, min_dim=1)
    assert pinned.picked_k == free.picked_k
    assert pinned.pinned is False
    assert pinned.min_dim == 1
    assert coords_pin.shape == coords_free.shape


def test_spectral_min_dim_clamped_to_eigenvector_budget():
    """``min_dim`` can't widen past the usable budget (``max_dim`` cap)."""
    centroids, _ = _circle_centroids(n=60, ambient=32, noise=0.02, seed=5)
    gram = _centered_gram(centroids)
    coords, diag = derive_spectral_coords(gram, max_dim=2, min_dim=5)
    assert diag.picked_k == 2          # capped at max_dim despite asking for 5
    assert coords.shape[1] == 2


# ------------------------------------------- layer-agnostic consensus Gram ---

def test_pca_consensus_drops_noise_layer():
    """Averaging a signal-layer Gram with an isotropic noise-layer Gram keeps
    the signal layout — the signal-weighting that makes the derivation
    layer-agnostic.

    The pipeline derives coords from ``mean_L`` of each layer's whitened
    ``(K, K)`` Gram.  Whitening puts every layer in common (background-σ)
    units, so a layer where the nodes aren't separated contributes a
    near-zero Gram and falls out of the mean.  Here the "noise layer" is
    isotropic at 1% the signal scale: the consensus must recover the
    signal layer's rank-2 layout rather than a blurred average.
    """
    signal, true_coords = _flat_centroids(
        n=60, ambient=32, rank=2, noise=0.02, seed=101,
    )
    g = torch.Generator().manual_seed(202)
    noise = 0.01 * torch.randn(60, 32, generator=g)  # ~isotropic, tiny
    signal_only, _ = derive_pca_coords(
        _centered_gram(signal), max_dim=8, var_threshold=0.95,
    )
    consensus = 0.5 * (_centered_gram(signal) + _centered_gram(noise))
    coords, diag = derive_pca_coords(
        consensus, max_dim=8, var_threshold=0.95,
    )
    assert diag.picked_k == 2, (
        f"consensus picked k={diag.picked_k}; the noise layer should not "
        f"add structure"
    )
    # The consensus layout is the *same* layout the signal layer recovers
    # alone (up to similarity) — the noise layer fell out of the average.
    # Compared scale-invariantly because the layer averaging rescales coords.
    dist = _procrustes_distance_scaled(coords, signal_only)
    assert dist < 0.05, (
        f"consensus layout diverged from signal-only by {dist:.4f}; the "
        f"noise layer perturbed it"
    )
    # And the signal layer's own recovery is faithful to ground truth (so the
    # comparison above isn't two-wrongs-agreeing).
    assert _procrustes_distance(signal_only, true_coords) < 0.10


def test_spectral_consensus_recovers_signal_topology():
    """A circle in the signal layer survives averaging with a noise layer.

    Spectral reads distances off the consensus Gram; the noise layer's
    near-uniform tiny distances don't change the k-NN structure, so the
    circle topology is recovered from the average just as from the signal
    layer alone.
    """
    circle, theta = _circle_centroids(
        n=80, ambient=32, noise=0.02, seed=303,
    )
    g = torch.Generator().manual_seed(404)
    noise = 0.01 * torch.randn(80, 32, generator=g)
    consensus = 0.5 * (_centered_gram(circle) + _centered_gram(noise))
    coords, diag = derive_spectral_coords(consensus, max_dim=6)
    assert diag.component_count == 1
    assert diag.picked_k == 2, (
        f"consensus picked k={diag.picked_k}; the circle topology should "
        f"survive the noise layer"
    )
    assert coords.shape == (80, 2)
    rec_order = torch.argsort(torch.atan2(coords[:, 1], coords[:, 0])).tolist()
    orig_order = torch.argsort(theta).tolist()

    def cyclic_match(a: list[int], b: list[int]) -> bool:
        n = len(a)
        return any(
            all(a[(start + i) % n] == b[i] for i in range(n))
            for start in range(n)
        )

    assert cyclic_match(orig_order, rec_order) or cyclic_match(
        orig_order[::-1], rec_order,
    ), "consensus spectral embedding did not preserve circular order"


def test_pca_consensus_signal_weights_across_layers():
    """When two layers carry *different* clean structure at different scales,
    the stronger one dominates the consensus layout.

    Two rank-1 signals on orthogonal axes: layer A at unit scale, layer B
    at 1/10 scale.  The consensus is rank-2 (both axes present), but the
    leading component must align with layer A — a direct read on
    "signal-weighted, not equal-weighted".
    """
    g = torch.Generator().manual_seed(505)
    n, d = 50, 24
    plane = _random_orthonormal_plane(d, 2, seed=505)  # two orthonormal axes
    axis_a, axis_b = plane[:, 0], plane[:, 1]
    ramp = torch.linspace(-3.0, 3.0, n)
    perm = torch.randperm(n, generator=g)
    # Layer A: strong rank-1 ramp on axis A.  Layer B: weak rank-1 ramp on
    # axis B (a different node ordering, so the axes are genuinely distinct).
    cents_a = ramp.unsqueeze(1) * axis_a.unsqueeze(0)
    cents_b = 0.1 * ramp[perm].unsqueeze(1) * axis_b.unsqueeze(0)
    consensus = 0.5 * (_centered_gram(cents_a) + _centered_gram(cents_b))
    coords, diag = derive_pca_coords(
        consensus, max_dim=8, var_threshold=0.999,
    )
    # Leading component carries far more variance than the second.
    assert diag.per_component_variance[0] > 0.9, (
        f"leading variance {diag.per_component_variance[0]:.3f} — the strong "
        f"layer should dominate"
    )
    # The leading coordinate axis tracks layer A's ramp (up to sign), not
    # layer B's permuted weak ramp.
    lead = coords[:, 0]
    corr_a = abs(float(torch.corrcoef(torch.stack([lead, ramp]))[0, 1]))
    assert corr_a > 0.98, f"leading axis corr with layer-A ramp {corr_a:.3f}"


# --------------------------------------------------------------- dispatcher ---

def test_discover_coords_routes_to_pca():
    centroids, _ = _flat_centroids(
        n=40, ambient=16, rank=2, noise=0.02, seed=77,
    )
    coords, diag = discover_coords(
        _centered_gram(centroids), method="pca", max_dim=4, var_threshold=0.70,
    )
    assert isinstance(diag, PcaDiagnostics)
    assert coords.shape[0] == 40


def test_discover_coords_routes_to_spectral():
    centroids, _ = _circle_centroids(
        n=40, ambient=16, noise=0.02, seed=88,
    )
    coords, diag = discover_coords(
        _centered_gram(centroids), method="spectral", max_dim=4,
    )
    assert isinstance(diag, SpectralDiagnostics)
    assert coords.shape[0] == 40


def test_discover_coords_rejects_unknown_method():
    with pytest.raises(ValueError, match=r"unknown discover method"):
        discover_coords(torch.randn(4, 4), method="banana")


def test_neutral_layout_coord_recovers_landmark():
    """The MDS out-of-sample projection: neutral's cross-Gram column against the
    nodes maps to its coordinate in the (U S) layout.  If neutral coincides with
    node j, its column is the Gram's j-th column and the projection returns node
    j's coordinate; a half-and-half column lands at the coordinate midpoint
    (the projection is linear in the cross-Gram)."""
    g = torch.Generator().manual_seed(0)
    node_coords = torch.randn(6, 3, generator=g)            # full column rank
    gram = node_coords @ node_coords.transpose(0, 1)        # (U S)(U S)ᵀ
    for j in range(node_coords.shape[0]):
        cj = neutral_layout_coord(node_coords, gram[:, j])
        assert torch.allclose(cj, node_coords[j], atol=1e-4)
    mid = 0.5 * (gram[:, 0] + gram[:, 1])
    c_mid = neutral_layout_coord(node_coords, mid)
    assert torch.allclose(
        c_mid, 0.5 * (node_coords[0] + node_coords[1]), atol=1e-4,
    )


def test_neutral_layout_coord_rejects_length_mismatch():
    with pytest.raises(ValueError, match=r"entries but the layout has"):
        neutral_layout_coord(torch.randn(5, 2), torch.randn(4))
