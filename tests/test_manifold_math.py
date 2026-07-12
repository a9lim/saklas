"""RBF + domain + manifold math for saklas.core.manifold.

Pure CPU tests — no model, no IO beyond a save/load round-trip in a
temp directory.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.core.manifold import (
    ActivationRowStore,
    BoxAxis,
    BoxDomain,
    CustomDomain,
    Manifold,
    SphereDomain,
    decompose,
    domain_from_spec,
    eval_rbf,
    eval_rbf_jacobian,
    fit_layer_subspace as _fit_layer_subspace_with_ev,  # returns (LayerSubspace, ev_ratio)
    fit_rbf_interpolant,
    fit_rbf_smoothed,
    fit_sigma_field,
    compute_node_reduced_covariance_from_rows,
    compute_manifold_node_stats,
    compute_store_reduced_covariances,
    prepare_rbf_fit_plan,
    _gcv_select_lambda,
    _off_surface_var,
    _off_surface_vars,
    _rbf_smoother_matrix,
    rbf_cardinal_weights,
    subspace_inject,
    LayerSubspace,
    invert_parameterization,
    load_manifold,
    save_manifold,
)


# ------------------------------------------------------------------ domains ---

def fit_layer_subspace(*args: Any, **kwargs: Any) -> Any:
    """Test alias dropping the EV ratio.  ``core.manifold.fit_layer_subspace``
    returns ``(LayerSubspace, ev_ratio)``; most tests don't care about the
    second half, so unpack here once."""
    sub, _ev = _fit_layer_subspace_with_ev(*args, **kwargs)
    return sub


def test_box_domain_dims():
    d = BoxDomain([
        BoxAxis("u", periodic=False, lo=-1.0, hi=1.0),
        BoxAxis("theta", periodic=True, period=1.0),
    ])
    assert d.intrinsic_dim == 2
    # one open axis (1) + one periodic axis (2) = 3
    assert d.embed_dim == 3


def test_box_domain_embed_periodic_wraps():
    d = BoxDomain([BoxAxis("theta", periodic=True, period=1.0)])
    a = d.embed(torch.tensor([[0.0]]))
    b = d.embed(torch.tensor([[1.0]]))      # one full period later
    assert torch.allclose(a, b, atol=1e-5)
    # the embedded circle point is a unit vector
    assert torch.allclose(
        torch.linalg.vector_norm(a, dim=-1), torch.ones(1), atol=1e-5,
    )
    # chordal distance across the seam is small
    near = d.distance(d.embed(torch.tensor([0.95])), d.embed(torch.tensor([0.05])))
    far = d.distance(d.embed(torch.tensor([0.0])), d.embed(torch.tensor([0.5])))
    # across the seam is short; half a period away is the diameter
    assert near.item() < 0.7
    assert far.item() == pytest.approx(2.0, abs=1e-4)


def test_box_domain_clamp():
    d = BoxDomain([
        BoxAxis("u", periodic=False, lo=0.0, hi=1.0),
        BoxAxis("t", periodic=True, period=1.0),
    ])
    out = d.clamp_position(torch.tensor([1.7, 2.3]))
    assert out[0].item() == pytest.approx(1.0)        # clamped
    assert out[1].item() == pytest.approx(0.3, abs=1e-6)  # wrapped


def test_sphere_domain_chordal_distance():
    d = SphereDomain(2)
    assert d.intrinsic_dim == 2
    assert d.embed_dim == 3
    pts = torch.tensor([[0.3, 0.5], [1.2, 2.0], [2.9, 5.0]])
    emb = d.embed(pts)
    norms = torch.linalg.vector_norm(emb, dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)
    # antipodal points (colatitude 0 vs pi) are chordal distance 2 apart
    north = d.embed(torch.tensor([0.0, 0.0]))
    south = d.embed(torch.tensor([math.pi, 0.0]))
    assert d.distance(north, south).item() == pytest.approx(2.0, abs=1e-4)


def test_custom_domain_identity():
    d = CustomDomain(4)
    assert d.intrinsic_dim == 4 and d.embed_dim == 4
    x = torch.randn(3, 4)
    assert torch.allclose(d.embed(x), x)
    assert torch.allclose(
        d.embed_jacobian(torch.randn(4)), torch.eye(4),
    )


def test_domain_spec_round_trip():
    for d in (
        BoxDomain([
            BoxAxis("v", periodic=False, lo=-1.0, hi=1.0),
            BoxAxis("a", periodic=True, period=2.0),
        ]),
        SphereDomain(3),
        CustomDomain(5, bounds=[[0.0, 1.0]] * 5),
    ):
        d2 = domain_from_spec(d.to_spec())
        assert type(d2) is type(d)
        assert d2.intrinsic_dim == d.intrinsic_dim
        assert d2.embed_dim == d.embed_dim


def test_domain_from_spec_rejects_unknown():
    with pytest.raises(ValueError):
        domain_from_spec({"type": "klein-bottle"})


# -------------------------------------------------------------------- RBF ---

def _reference_natural_cubic(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A reference natural-cubic interpolant, sampled densely on [0, 1].

    Used only to confirm the n=1 open-box RBF reproduces it.
    """
    K = t.shape[0]
    h = t[1:] - t[:-1]
    M = torch.zeros_like(y)
    n = K - 2
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, i] = 2.0 * (h[i] + h[i + 1])
        if i > 0:
            A[i, i - 1] = h[i]
        if i < n - 1:
            A[i, i + 1] = h[i + 1]
    rhs = 6.0 * (
        (y[2:] - y[1:-1]) / h[1:].unsqueeze(-1)
        - (y[1:-1] - y[:-2]) / h[:-1].unsqueeze(-1)
    )
    M[1:-1] = torch.linalg.solve(A, rhs)

    s = torch.linspace(0.0, 1.0, 50)
    idx = (torch.searchsorted(t, s.clamp(t[0], t[-1]), right=True) - 1)
    idx = idx.clamp(0, K - 2)
    i1 = idx + 1
    t0, t1 = t[idx], t[i1]
    hh = (t1 - t0).clamp(min=1e-12)
    a = ((t1 - s) / hh).unsqueeze(-1)
    b = ((s - t0) / hh).unsqueeze(-1)
    h2 = (hh * hh).unsqueeze(-1)
    return (
        a * y[idx] + b * y[i1]
        + ((a.pow(3) - a) * M[idx] + (b.pow(3) - b) * M[i1]) * h2 / 6.0
    )


def test_rbf_reproduces_natural_cubic():
    """The load-bearing property: 1-D open-box r^3 RBF == natural cubic."""
    torch.manual_seed(0)
    K = 8
    t = torch.linspace(0.0, 1.0, K)
    y = torch.randn(K, 3)
    reference = _reference_natural_cubic(t, y)

    w, c = fit_rbf_interpolant(t.unsqueeze(1), y)
    s = torch.linspace(0.0, 1.0, 50)
    rbf = eval_rbf(t.unsqueeze(1), w, c, s.unsqueeze(1))
    assert torch.allclose(rbf, reference, atol=1e-3)


def test_rbf_interpolates_nodes():
    torch.manual_seed(1)
    node = torch.rand(9, 2)
    val = torch.randn(9, 5)
    w, c = fit_rbf_interpolant(node, val)
    got = eval_rbf(node, w, c, node)
    assert torch.allclose(got, val, atol=1e-3)


def test_rbf_cardinal_weights_node_exact_and_partition_of_unity():
    # The cardinal weights are e_i at node i (RBF reproduces sampled values)
    # and sum to 1 everywhere (the affine polynomial reproduces constants).
    torch.manual_seed(7)
    nodes = torch.rand(8, 3)
    K = nodes.shape[0]
    for i in range(K):
        w = rbf_cardinal_weights(nodes, nodes[i])
        assert torch.allclose(w, torch.eye(K)[i], atol=1e-4)
    w_mid = rbf_cardinal_weights(nodes, nodes.mean(dim=0))
    assert w_mid.shape == (K,)
    assert abs(float(w_mid.sum()) - 1.0) < 1e-4


def test_rbf_cardinal_weights_reproduce_affine_off_node():
    # r**3 RBF + linear polynomial reproduces affine functions exactly, so
    # ``w(z) @ Y`` equals ``A z + b`` at any query when Y is affine in coords —
    # the property that makes the per-layer target a faithful layout blend.
    torch.manual_seed(11)
    nodes = torch.rand(12, 2)
    A = torch.randn(4, 2)
    b = torch.randn(4)
    Y = nodes @ A.T + b                       # (12, 4) affine in coords
    z = torch.tensor([0.37, 0.62])
    w = rbf_cardinal_weights(nodes, z)
    assert torch.allclose(w @ Y, A @ z + b, atol=1e-3)


def test_rbf_cardinal_weights_poisedness_rejection():
    # A degenerate (collinear) layout has no interpolant — propagates ValueError.
    line = torch.stack(
        [torch.linspace(0, 1, 6), torch.linspace(0, 1, 6)], dim=-1,
    )
    with pytest.raises(ValueError, match="poisedness"):
        rbf_cardinal_weights(line, torch.tensor([0.5, 0.5]))


def test_rbf_poisedness_rejection_collinear():
    # 6 nodes in a 2-D embedding but all on one line -> affine rank 1.
    line = torch.stack(
        [torch.linspace(0, 1, 6), torch.linspace(0, 1, 6)], dim=-1,
    )
    with pytest.raises(ValueError, match="poisedness"):
        fit_rbf_interpolant(line, torch.randn(6, 3))


def test_rbf_poisedness_rejection_too_few():
    # 2 nodes cannot determine an affine term in 2 dimensions.
    with pytest.raises(ValueError, match="poisedness"):
        fit_rbf_interpolant(torch.rand(2, 2), torch.randn(2, 3))


# ------------------------------------------------ penalized RBF (smoothing) ---

def test_fit_rbf_smoothed_lambda_zero_equals_interpolant():
    # smoothing=0 (and None) must reproduce the exact interpolant bit-for-bit:
    # the discover path can always recover authored-style exactness, and the
    # cardinal-weight / interpolation guarantees rest on this identity.
    torch.manual_seed(3)
    node = torch.rand(10, 2)
    val = torch.randn(10, 4)
    w0, c0 = fit_rbf_interpolant(node, val)
    for smoothing in (0, 0.0, None):
        ws, cs, info = fit_rbf_smoothed(node, val, smoothing=smoothing)
        assert torch.equal(w0, ws)
        assert torch.equal(c0, cs)
        assert info["lambda"] == 0.0
        assert info["edf"] == float(node.shape[0])


def test_fit_rbf_smoothed_gcv_picks_interior_lambda():
    # On noisy values GCV should select a positive λ and an effective dof
    # strictly below the node count (genuine smoothing, not interpolation).
    torch.manual_seed(4)
    K = 14
    x = torch.linspace(0.0, 1.0, K).unsqueeze(1)
    clean = torch.stack([torch.sin(6 * x[:, 0]), (x[:, 0] - 0.5) ** 2], dim=1)
    y = clean + 0.2 * torch.randn(K, 2)
    _w, _c, info = fit_rbf_smoothed(x, y, smoothing="auto")
    assert info["lambda"] > 0.0
    assert info["edf"] < K
    assert math.isfinite(info["gcv"]) and info["gcv"] > 0.0


@pytest.mark.parametrize("smoothing", [None, 0.25, "auto"])
def test_rbf_fit_plan_matches_standalone_fit(smoothing: float | str | None):
    torch.manual_seed(41)
    node = torch.rand(14, 2)
    values = torch.randn(14, 5)
    plan = prepare_rbf_fit_plan(node, smoothing=smoothing)

    expected = fit_rbf_smoothed(
        plan.node_params, values, smoothing=smoothing,
    )
    actual = fit_rbf_smoothed(
        plan.node_params, values, smoothing=smoothing, plan=plan,
    )

    assert torch.allclose(actual[0], expected[0], atol=2e-5, rtol=2e-5)
    assert torch.allclose(actual[1], expected[1], atol=2e-5, rtol=2e-5)
    assert actual[2] == pytest.approx(expected[2], rel=2e-5, abs=2e-5)


def test_fit_rbf_smoothed_edf_monotone_to_polynomial():
    # The rigorous correctness check on the penalty form: as λ grows the
    # effective dof tr(S_λ) must fall monotonically from K (interpolation) to
    # m+1 (the affine polynomial null space).  A wrong penalty system would
    # not converge to the polynomial dimension.
    torch.manual_seed(5)
    K, m = 12, 1
    x = torch.linspace(0.0, 1.0, K).unsqueeze(1)
    dist = torch.cdist(x, x)
    E = dist.pow(3)
    Q = torch.cat([torch.ones(K, 1), x], dim=1)
    lams = [0.0, 1e-4, 1e-2, 1e0, 1e2, 1e5]
    edfs = [float(_rbf_smoother_matrix(E, Q, lam).diagonal().sum()) for lam in lams]
    assert edfs[0] == pytest.approx(float(K), abs=1e-3)
    for a, b in zip(edfs, edfs[1:]):
        assert b <= a + 1e-4
    assert edfs[-1] == pytest.approx(float(m + 1), abs=1e-2)


def test_fit_rbf_smoothed_clean_data_keeps_low_lambda():
    # GCV on cleanly-interpolatable (affine) data should not over-smooth: the
    # surface stays close to the data and the smallest grid λ wins out.
    torch.manual_seed(6)
    K = 12
    x = torch.rand(K, 2)
    A = torch.randn(3, 2)
    b = torch.randn(3)
    y = x @ A.T + b  # exactly affine → reproduced by the polynomial term
    w, c, _info = fit_rbf_smoothed(x, y, smoothing="auto")
    fit = eval_rbf(x, w, c, x)
    assert torch.allclose(fit, y, atol=1e-2)


def test_fit_rbf_smoothed_fixed_lambda_shrinks_node_residual():
    # A fixed positive λ shrinks the surface toward the trend, so it no longer
    # passes exactly through the (noisy) nodes — the smoothing trade-off.
    torch.manual_seed(7)
    K = 12
    x = torch.linspace(0.0, 1.0, K).unsqueeze(1)
    y = torch.sin(5 * x) + 0.25 * torch.randn(K, 1)
    w0, c0 = fit_rbf_interpolant(x, y)
    ws, cs, _info = fit_rbf_smoothed(x, y, smoothing=1.0)
    exact_resid = float((eval_rbf(x, w0, c0, x) - y).abs().max())
    smooth_resid = float((eval_rbf(x, ws, cs, x) - y).abs().max())
    assert exact_resid < 1e-3 < smooth_resid


def test_fit_rbf_smoothed_poisedness_rejection():
    # The smoothing path validates poisedness too (the constraint Qᵀw=0 still
    # needs Q full column rank), raising the same error as the exact fit.
    line = torch.stack(
        [torch.linspace(0, 1, 6), torch.linspace(0, 1, 6)], dim=-1,
    )
    with pytest.raises(ValueError, match="poisedness"):
        fit_rbf_smoothed(line, torch.randn(6, 3), smoothing="auto")


def test_gcv_select_lambda_eigen_matches_smoother_reference():
    """The Demmler-Reinsch eigen reformulation of ``_gcv_select_lambda`` must
    reproduce the smoother-matrix GCV curve bit-comparably — the optimization
    is only valid if it selects the SAME lambda the saddle-solve loop did."""
    torch.manual_seed(7)
    for k, m, r in [(12, 2, 3), (20, 1, 1), (9, 3, 2), (15, 2, 5)]:
        node = torch.randn(k, m, dtype=torch.float32)
        values = torch.randn(k, r, dtype=torch.float32)
        e = torch.cdist(node, node).pow(3)
        q = torch.cat([torch.ones(k, 1), node], dim=1)

        # Reference: brute-force GCV over the same grid via the full smoother
        # matrix S_lambda (the pre-optimization formulation).
        denom = k * k - k
        e_scale = float(e.abs().sum() / denom)
        grid = e_scale * torch.logspace(-6.0, 3.0, 40, dtype=torch.float32)
        eye = torch.eye(k)
        ref_lam, ref_edf, ref_gcv = float(grid[0]), float(k), math.inf
        best = math.inf
        for lam_t in grid:
            lam = float(lam_t)
            im_s = eye - _rbf_smoother_matrix(e, q, lam)
            tr = float(im_s.diagonal().sum())
            if tr <= 0.0:
                continue
            gcv = k * float((im_s @ values).pow(2).sum()) / (tr * tr)
            if gcv < best:
                best = gcv
                ref_lam, ref_edf, ref_gcv = lam, float(k) - tr, gcv

        lam, edf, gcv = _gcv_select_lambda(e, q, values)
        # Same grid point selected (logspace steps are ~1.7x apart, so an exact
        # match means no argmin drift), and matching edf/gcv scalars.
        assert lam == pytest.approx(ref_lam, rel=1e-5)
        assert edf == pytest.approx(ref_edf, rel=1e-3)
        assert gcv == pytest.approx(ref_gcv, rel=1e-3)


def test_eval_rbf_jacobian_matches_finite_difference():
    torch.manual_seed(2)
    node = torch.rand(10, 2)
    val = torch.randn(10, 4)
    w, c = fit_rbf_interpolant(node, val)
    x = torch.tensor([0.4, 0.6])
    jac = eval_rbf_jacobian(node, w, c, x)  # (R, m)
    eps = 1e-3
    for k in range(2):
        d = torch.zeros(2)
        d[k] = eps
        fd = (
            eval_rbf(node, w, c, x + d) - eval_rbf(node, w, c, x - d)
        ) / (2 * eps)
        assert torch.allclose(jac[:, k], fd, atol=1e-2)


# --------------------------------------------------------- fit_layer_subspace ---

def _circle(k: int, dim: int) -> tuple[torch.Tensor, BoxDomain, torch.Tensor]:
    """K activation centroids on a planar circle + a 1-D periodic domain."""
    centroids = torch.zeros(k, dim)
    base = torch.full((dim,), 0.7)
    for i in range(k):
        theta = 2.0 * math.pi * i / k
        centroids[i] = base.clone()
        centroids[i, 0] += 2.0 * math.cos(theta)
        centroids[i, 1] += 2.0 * math.sin(theta)
    domain = BoxDomain([BoxAxis("t", periodic=True, period=1.0)])
    node_coords = torch.tensor([[i / k] for i in range(k)])
    node_params = domain.embed(node_coords)
    return centroids, domain, node_params


def _grid2d(steps: tuple[float, ...] = (0.0, 0.5, 1.0)) -> tuple[BoxDomain, torch.Tensor]:
    domain = BoxDomain([
        BoxAxis("u", periodic=False, lo=0.0, hi=1.0),
        BoxAxis("v", periodic=False, lo=0.0, hi=1.0),
    ])
    coords = torch.tensor([[x, y] for x in steps for y in steps])
    return domain, coords


def test_fit_layer_subspace_circle_spans_curve_plane():
    centroids, _domain, node_params = _circle(8, dim=16)
    sub = fit_layer_subspace(centroids, node_params)
    assert sub.rank == 2  # a planar circle has rank 2


def test_fit_layer_subspace_eval_at_nodes():
    centroids, domain, node_params = _circle(6, dim=12)
    sub = fit_layer_subspace(centroids, node_params)
    man = Manifold(
        name="c", domain=domain, node_labels=[f"n{i}" for i in range(6)],
        node_coords=torch.tensor([[i / 6] for i in range(6)]),
        layers={0: sub},
    )
    p0 = man.manifold_point(0, (0.0,))
    # The R-dim surface passes through each centroid's **in-span projection**
    # (neutral-anchor §5: with no neutral the anchor is μ and mean = P_basis(μ),
    # dropping the common off-span offset the old full-μ mean carried for free).
    p_basis = (centroids[0] @ sub.basis.T) @ sub.basis
    assert torch.allclose(p0, p_basis, atol=1e-3)


def test_fit_layer_subspace_n2_box():
    domain, coords = _grid2d()
    torch.manual_seed(3)
    centroids = torch.randn(9, 24)
    sub = fit_layer_subspace(centroids, domain.embed(coords))
    man = Manifold(
        name="g", domain=domain,
        node_labels=[f"n{i}" for i in range(9)],
        node_coords=coords, layers={0: sub},
    )
    got = man.manifold_point(0, (0.5, 0.5))
    assert got.shape == (24,)
    # a corner node reproduces its centroid's in-span projection (the surface
    # is R-dim; neutral-anchor §5 keeps only the in-span component of mean)
    assert torch.allclose(
        man.manifold_point(0, (0.0, 0.0)),
        (centroids[0] @ sub.basis.T) @ sub.basis, atol=1e-3,
    )


def test_fit_layer_subspace_neutral_anchored():
    """A neutral baseline anchors the curved frame: ``mean = P_basis(ν)`` and
    the neutral lands at reduced-coord 0 (§5).  The basis is unchanged — it
    stays μ-centered regardless of the anchor (the basis caveat)."""
    centroids, _domain, node_params = _circle(6, dim=12)
    nu = 3.0 + torch.randn(12)
    sub_n, _ = _fit_layer_subspace_with_ev(
        centroids, node_params, neutral_mean=nu,
    )
    sub_mu, _ = _fit_layer_subspace_with_ev(centroids, node_params)  # no neutral
    # Basis identical (μ-centered regardless of anchor); only mean/coords move.
    assert torch.allclose(sub_n.basis, sub_mu.basis, atol=1e-5)
    # mean = projection of neutral into the span.
    assert torch.allclose(
        sub_n.mean, (nu @ sub_n.basis.T) @ sub_n.basis, atol=1e-4,
    )
    # Neutral → reduced-coord 0.
    q = (nu - sub_n.mean) @ sub_n.basis.T
    assert torch.allclose(q, torch.zeros(sub_n.rank), atol=1e-4)


def test_fit_affine_subspace_reuses_precomputed_whitened_gram():
    """Extraction precomputes ``X Σ⁻¹ Xᵀ`` for diagnostics / discover coords;
    the affine fit should consume that Gram instead of recomputing it."""
    from saklas.core.manifold import fit_affine_subspace
    from tests._whitener import synthetic_means, synthetic_whitener

    torch.manual_seed(19)
    centroids = torch.randn(6, 14)
    means = synthetic_means([0], 14)
    base = synthetic_whitener([0], 14, means=means)
    X = centroids - centroids.mean(dim=0)
    gram = base.subspace_gram(0, X)

    class CountingWhitener:
        def __init__(self, inner: Any) -> None:
            self.inner = inner
            self.subspace_gram_calls = 0
            self.apply_inv_calls = 0

        def subspace_gram(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            self.subspace_gram_calls += 1
            return self.inner.subspace_gram(*args, **kwargs)

        def apply_inv(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            self.apply_inv_calls += 1
            return self.inner.apply_inv(*args, **kwargs)

    counting = CountingWhitener(base)
    sub, mu_coords, ev = fit_affine_subspace(
        centroids,
        neutral_mean=means[0],
        n_components=3,
        whitener=counting,  # type: ignore[arg-type]
        layer=0,
        whitened_gram=gram,
        whitened_rows=base.apply_inv(0, X),
    )
    ref_sub, ref_mu_coords, ref_ev = fit_affine_subspace(
        centroids,
        neutral_mean=means[0],
        n_components=3,
        whitener=base,
        layer=0,
    )

    assert counting.subspace_gram_calls == 0
    assert counting.apply_inv_calls == 0
    assert torch.allclose(sub.basis, ref_sub.basis, atol=1e-5)
    assert torch.allclose(mu_coords, ref_mu_coords, atol=1e-5)
    assert ev == pytest.approx(ref_ev, abs=1e-6)


def test_fit_layer_subspace_rejects_too_few_nodes():
    with pytest.raises(ValueError):
        fit_layer_subspace(torch.randn(2, 8), torch.rand(2, 1))


def test_fit_layer_subspace_returns_reusable_mu_centered_coords():
    generator = torch.Generator().manual_seed(91)
    centroids = 100_000.0 + torch.randn(7, 12, generator=generator)
    neutral = -70_000.0 + torch.randn(12, generator=generator)
    node_params = torch.linspace(0.0, 1.0, 7).reshape(-1, 1)
    fit_result: dict[str, torch.Tensor] = {}

    sub, _ = _fit_layer_subspace_with_ev(
        centroids, node_params, neutral_mean=neutral, fit_result=fit_result,
    )

    expected = (
        (centroids.float() - centroids.float().mean(dim=0))
        @ sub.basis.float().T
    )
    assert torch.equal(fit_result["mu_coords"], expected)
    expected_anchor_coords = (
        (centroids.float() - neutral.float()) @ sub.basis.float().T
    )
    fitted_anchor_coords = eval_rbf(
        sub.node_params, sub.rbf_weights, sub.poly_coeffs, sub.node_params,
    )
    assert torch.allclose(
        fitted_anchor_coords, expected_anchor_coords, atol=0.1, rtol=1e-6,
    )


def test_fit_layer_subspace_returns_ev_ratio():
    """The new tuple-return shape carries an explained-variance ratio
    alongside the LayerSubspace.  EV ∈ [0, 1]: when the chosen R
    captures all of the centered centroid variance EV ≈ 1.0; when R
    is capped below the centroids' effective rank, EV drops below 1.

    K centroids in any D form a rank-at-most-(K-1) centered matrix,
    so the default ``n_components=64`` retains everything at small K.
    To get EV < 1 we need to cap ``n_components`` below the centroids'
    effective rank — that's exactly the regime the bundled
    ``personas`` manifold uses (R=8, retained < total variance).
    """
    # Clean case: rank-2 planar circle, default R cap → EV = 1.0.
    centroids, _domain, node_params = _circle(8, dim=16)
    sub, ev = _fit_layer_subspace_with_ev(centroids, node_params)
    assert sub.rank == 2
    assert 0.99 <= ev <= 1.000001

    # Capped case: 9 centroids in dim=32, capped at R=2.  The full
    # centered rank is min(K-1, D) = 8, so retaining only the top 2
    # singular values discards the remaining 6 → EV well below 1.
    torch.manual_seed(11)
    K, D = 9, 32
    isotropic_centroids = torch.randn(K, D)
    _grid_coords = torch.tensor([[x, y] for x in (0.0, 0.5, 1.0)
                                          for y in (0.0, 0.5, 1.0)])
    _domain_g = BoxDomain([
        BoxAxis("u", periodic=False, lo=0.0, hi=1.0),
        BoxAxis("v", periodic=False, lo=0.0, hi=1.0),
    ])
    _node_params_g = _domain_g.embed(_grid_coords)
    _sub_capped, ev_capped = _fit_layer_subspace_with_ev(
        isotropic_centroids, _node_params_g, n_components=2,
    )
    assert 0.0 < ev_capped < 0.9   # 2/8 ≈ 25% on average for isotropic
    # The capped case retains less of the total centroid variance —
    # fit quality is captured by EV.
    assert ev_capped < ev


# ------------------------------------------------------------------- tangent ---

def test_tangent_matches_finite_difference():
    domain, coords = _grid2d()
    torch.manual_seed(4)
    sub = fit_layer_subspace(torch.randn(9, 20), domain.embed(coords))
    man = Manifold(
        name="g", domain=domain,
        node_labels=[f"n{i}" for i in range(9)],
        node_coords=coords, layers={0: sub},
    )
    pos = torch.tensor([0.3, 0.7])
    tan = man.tangent(0, pos)  # (2, D)
    eps = 1e-3
    for k in range(2):
        d = torch.zeros(2)
        d[k] = eps
        fd = (
            man.manifold_point(0, pos + d) - man.manifold_point(0, pos - d)
        ) / (2 * eps)
        assert torch.allclose(tan[k], fd, atol=5e-2)


# ---------------------------------------------------- inverse parameterization ---

def test_invert_parameterization_recovers_known_position():
    domain, coords = _grid2d()
    torch.manual_seed(5)
    sub = fit_layer_subspace(torch.randn(9, 16), domain.embed(coords))
    target_pos = torch.tensor([0.4, 0.65])
    embedded = domain.embed(target_pos)
    query = eval_rbf(
        sub.node_params, sub.rbf_weights, sub.poly_coeffs,
        sub._normalize(embedded),
    )
    pos, dist = invert_parameterization(sub, domain, query.unsqueeze(0), coords)
    assert torch.allclose(pos[0], target_pos, atol=5e-2)
    assert dist.item() < 1e-2




# ----------------------------------------------------------------- save/load ---

def test_save_load_manifold_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    ca, domain, node_params = _circle(7, dim=20)
    cb = ca * 1.3
    manifold = Manifold(
        name="mood",
        domain=domain,
        node_labels=[f"n{i}" for i in range(7)],
        node_coords=torch.tensor([[i / 7] for i in range(7)]),
        layers={
            4: fit_layer_subspace(ca, node_params),
            9: fit_layer_subspace(cb, node_params),
        },
        feature_space="raw",
        mahalanobis_share={4: 1.5, 9: 2.0},
        # Per-layer authoring-coordinate foot of the neutral mean (the circle
        # is 1-D, so each ``O_L`` is ``(1,)``).
        origin={4: torch.tensor([0.42]), 9: torch.tensor([0.55])},
    )
    for sub in manifold.layers.values():
        _attach_constant_sigma(sub, 0.1)
    path = tmp_path / "mood" / "model.safetensors"
    save_manifold(manifold, path, {"method": "manifold_pca",
                                   "nodes_sha256": "abc123",
                                   "share_metric": "mahalanobis"})
    loaded = load_manifold(path)

    assert loaded.name == "mood"
    assert loaded.domain.intrinsic_dim == 1
    assert loaded.node_labels == [f"n{i}" for i in range(7)]
    assert sorted(loaded.layers) == [4, 9]
    assert loaded.feature_space == "raw"
    assert loaded.metadata["nodes_sha256"] == "abc123"
    # New v4+ companion fields round-trip: the per-layer whitened share
    # and the metric label that records which weighting the fit used.
    assert loaded.mahalanobis_share == {4: 1.5, 9: 2.0}
    assert loaded.metadata["share_metric"] == "mahalanobis"
    # The per-layer ``origin`` round-trips through the sidecar.
    assert sorted(loaded.origin) == [4, 9]
    for idx in (4, 9):
        assert torch.allclose(loaded.origin[idx], manifold.origin[idx])
    assert torch.allclose(loaded.node_coords, manifold.node_coords)
    for idx in (4, 9):
        a, b = manifold.layers[idx], loaded.layers[idx]
        _, a_rw, a_pc = a.rbf_params()
        _, b_rw, b_pc = b.rbf_params()
        assert torch.allclose(a.mean, b.mean)
        assert torch.allclose(a.basis, b.basis)
        assert torch.allclose(a_rw, b_rw)
        assert torch.allclose(a_pc, b_pc)
    # evaluation matches after the round-trip
    assert torch.allclose(
        loaded.manifold_point(4, (0.3,)),
        manifold.manifold_point(4, (0.3,)),
        atol=1e-4,
    )


def test_save_rejects_manifold_with_empty_share_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence rejects geometry without complete Mahalanobis shares."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    ca, domain, node_params = _circle(7, dim=20)
    manifold = Manifold(
        name="mood",
        domain=domain,
        node_labels=[f"n{i}" for i in range(7)],
        node_coords=torch.tensor([[i / 7] for i in range(7)]),
        layers={3: fit_layer_subspace(ca, node_params)},
        feature_space="raw",
    )  # no mahalanobis_share
    path = tmp_path / "mood" / "model.safetensors"
    with pytest.raises(ValueError, match="Mahalanobis shares"):
        save_manifold(manifold, path, {"method": "manifold_pca",
                                       "nodes_sha256": "abc"})


def test_layer_subspace_to_device_dtype():
    centroids, _domain, node_params = _circle(5, dim=10)
    sub = fit_layer_subspace(centroids, node_params)
    moved = sub.to(device=torch.device("cpu"), dtype=torch.float64)
    assert moved.basis.dtype == torch.float64
    assert moved.mean.dtype == torch.float64
    assert moved.rbf_weights.dtype == torch.float64


# ----------------------------------------------------------------- geodesics ---

def test_geodesic_box_open_lerp():
    d = BoxDomain([
        BoxAxis("u", periodic=False, lo=-1.0, hi=1.0),
        BoxAxis("v", periodic=False, lo=-1.0, hi=1.0),
    ])
    a = torch.tensor([-1.0, 0.0])
    b = torch.tensor([1.0, 1.0])
    assert torch.allclose(d.geodesic(a, b, 0.0), a)
    assert torch.allclose(d.geodesic(a, b, 1.0), b)
    assert torch.allclose(d.geodesic(a, b, 0.5), torch.tensor([0.0, 0.5]))


def test_geodesic_box_periodic_takes_short_arc():
    d = BoxDomain([BoxAxis("t", periodic=True, period=1.0)])
    a = torch.tensor([0.9])
    b = torch.tensor([0.1])  # short way is +0.2 across the seam, not -0.8
    assert torch.allclose(d.geodesic(a, b, 0.5), torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(d.geodesic(a, b, 0.25), torch.tensor([0.95]), atol=1e-6)
    assert torch.allclose(d.geodesic(a, b, 1.0), torch.tensor([0.1]), atol=1e-6)


def test_geodesic_box_batched_frac():
    d = BoxDomain([
        BoxAxis("u", periodic=False, lo=-1.0, hi=1.0),
        BoxAxis("v", periodic=False, lo=-1.0, hi=1.0),
    ])
    a = torch.tensor([[-1.0, 0.0], [0.0, 0.0]])
    b = torch.tensor([[1.0, 1.0], [1.0, 0.5]])
    frac = torch.tensor([[0.0], [1.0]])
    out = d.geodesic(a, b, frac)
    assert torch.allclose(out[0], a[0]) and torch.allclose(out[1], b[1])


def test_geodesic_sphere_slerp_stays_on_sphere_and_bisects():
    sph = SphereDomain(2)
    a = torch.tensor([0.4, 0.5])
    b = torch.tensor([1.2, 2.0])
    assert torch.allclose(sph.embed(sph.geodesic(a, b, 0.0)), sph.embed(a), atol=1e-5)
    assert torch.allclose(sph.embed(sph.geodesic(a, b, 1.0)), sph.embed(b), atol=1e-5)
    mid = sph.geodesic(a, b, 0.5)
    emid = sph.embed(mid)
    assert torch.allclose((emid * emid).sum(), torch.tensor(1.0), atol=1e-5)
    ea, eb = sph.embed(a), sph.embed(b)
    full = torch.arccos((ea * eb).sum().clamp(-1, 1))
    half = torch.arccos((ea * emid).sum().clamp(-1, 1))
    assert torch.allclose(half, full / 2, atol=1e-4)


def test_geodesic_sphere_unembed_roundtrip():
    sph = SphereDomain(3)
    ang = torch.tensor([[0.7, 1.3, 2.1], [1.9, 0.4, 5.0]])
    assert torch.allclose(
        sph.embed(sph.clamp_position(sph._unembed(sph.embed(ang)))),
        sph.embed(ang), atol=1e-5,
    )


def test_geodesic_sphere_degenerate_fallback():
    sph = SphereDomain(2)
    a = torch.tensor([0.5, 1.0])
    assert torch.allclose(sph.embed(sph.geodesic(a, a, 0.5)), sph.embed(a), atol=1e-5)


def test_geodesic_custom_linear():
    cd = CustomDomain(3)
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(cd.geodesic(a, b, 0.5), torch.tensor([1.0, 2.0, 3.0]))


# ------------------------------------------------------------- subspace_inject ---

def _grid_manifold(dim: int = 16, seed: int = 0):
    """A curved 3x3-grid manifold with a realistic large common-mode norm.

    Returns ``(subspace, domain)``. The DC offset mirrors real LM activations
    (rogue/massive-activation channels dominate ‖h‖), so node centroids have
    comparable norm and the kernel's soft norm cap stays dormant.
    """
    torch.manual_seed(seed)
    coords = torch.tensor([[u, v] for u in (0.0, 0.5, 1.0) for v in (0.0, 0.5, 1.0)])
    domain = BoxDomain([
        BoxAxis("u", periodic=False, lo=0.0, hi=1.0),
        BoxAxis("v", periodic=False, lo=0.0, hi=1.0),
    ])
    w_lin = torch.randn(2, dim)
    w_quad = torch.randn(2, dim)
    centroids = 20.0 + coords @ w_lin + (coords ** 2) @ w_quad + 0.05 * torch.randn(9, dim)
    sub = fit_layer_subspace(centroids, domain.embed(coords))
    return sub, domain


def _on_surface(sub: Any, domain: Any, coord: Any) -> torch.Tensor:
    return sub.eval_at(domain.embed(torch.tensor(coord, dtype=torch.float32)))


def test_inject_identity_at_zero():
    sub, domain = _grid_manifold()
    h = _on_surface(sub, domain, [0.4, 0.6]) + 0.2 * torch.randn(16)
    out, _ = subspace_inject(
        h, sub, domain, torch.tensor([0.9, 0.1]), torch.tensor([0.4, 0.6]),
        0.0, 0.0, gn_steps=20,
    )
    assert torch.allclose(out, h, atol=1e-3)


def test_inject_identity_at_zero_off_surface_approximate_foot():
    """``along=0, onto=0`` is *exact* identity even for an activation far off
    the surface whose foot solve hasn't converged (one GN step, a deliberately
    wrong seed).

    This is the regression guard for the curved-steering gibberish: the former
    project-onto-normal + renorm transport silently discarded the residual's
    tangential-at-the-foot component on every fire, so it corrupted any
    off-surface activation by 20-150% *with zero steering applied* — compounding
    across layers into degenerate repetition.  ``test_inject_identity_at_zero``
    missed it because a near-surface input with a perfect seed leaves the
    residual already-normal.  The minimal orthogonal frame rotation is the
    identity whenever the foot doesn't move (``p_new == p``), so it cannot
    corrupt the activation regardless of foot accuracy.
    """
    sub, domain = _grid_manifold()
    torch.manual_seed(1)
    # A large *in-subspace* off-surface residual (pure H_n) + a far-off seed so
    # one GN step leaves the foot approximate (residual not normal at it).
    pert = torch.randn(16)
    pert = (pert @ sub.basis.T) @ sub.basis            # project into the subspace
    h = _on_surface(sub, domain, [0.5, 0.5]) + 3.0 * pert
    out, _ = subspace_inject(
        h, sub, domain, torch.tensor([0.9, 0.1]), torch.tensor([0.02, 0.97]),
        0.0, 0.0, gn_steps=1,
    )
    assert torch.allclose(out, h, atol=1e-4)


def test_inject_along_slides_on_surface():
    sub, domain = _grid_manifold()
    a, b = [0.0, 0.0], [1.0, 1.0]
    h = _on_surface(sub, domain, a)  # on the surface, H_n = 0
    out, _ = subspace_inject(
        h, sub, domain, torch.tensor(b), torch.tensor(a),
        1.0, 0.0, gn_steps=20,
    )
    assert torch.allclose(out, _on_surface(sub, domain, b), atol=5e-3)
    # half-slide lands on the geodesic midpoint foot
    out_h, _ = subspace_inject(
        h, sub, domain, torch.tensor(b), torch.tensor(a),
        0.5, 0.0, gn_steps=20,
    )
    assert torch.allclose(out_h, _on_surface(sub, domain, [0.5, 0.5]), atol=5e-3)


def test_inject_onto_collapses_to_surface():
    sub, domain = _grid_manifold()
    h = _on_surface(sub, domain, [0.5, 0.5]) + 0.3 * torch.randn(16)
    out, foot = subspace_inject(
        h, sub, domain, torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5]),
        0.0, 1.0, gn_steps=20,
    )
    # in-subspace part of the output lies on the surface at the found foot
    q_out = (out - sub.mean) @ sub.basis.T
    surface = sub.eval_at(domain.embed(foot)) - sub.mean
    assert torch.allclose(q_out @ sub.basis, surface, atol=5e-3)


def test_inject_onto_half_halves_off_manifold_residual():
    sub, domain = _grid_manifold()
    h = _on_surface(sub, domain, [0.4, 0.4]) + 0.3 * torch.randn(16)
    pos, seed = torch.tensor([0.4, 0.4]), torch.tensor([0.4, 0.4])
    out0, foot = subspace_inject(h, sub, domain, pos, seed, 0.0, 0.0, gn_steps=20)
    out_h, _ = subspace_inject(h, sub, domain, pos, seed, 0.0, 0.5, gn_steps=20)
    surf = sub.eval_at(domain.embed(foot)) - sub.mean
    hn0 = ((out0 - sub.mean) @ sub.basis.T @ sub.basis) - surf
    hnh = ((out_h - sub.mean) @ sub.basis.T @ sub.basis) - surf
    assert torch.allclose(hnh, 0.5 * hn0, atol=5e-3)


def test_inject_keeps_off_subspace_verbatim():
    # The off-subspace residual ``H_o`` is always kept verbatim — the old
    # ``toward`` op that scaled it is removed.  Even a strong along+onto slide
    # leaves the orthogonal complement untouched.
    sub, domain = _grid_manifold()
    h = _on_surface(sub, domain, [0.3, 0.7]) + 0.4 * torch.randn(16)
    _, perp_in = decompose(h, sub.mean, sub.basis)
    out, _ = subspace_inject(
        h, sub, domain, torch.tensor([0.9, 0.2]), torch.tensor([0.3, 0.7]),
        1.0, 1.0, gn_steps=20,
    )
    _, perp_out = decompose(out, sub.mean, sub.basis)
    assert torch.allclose(perp_out, perp_in, atol=1e-4)


def test_inject_along_onto_preserve_off_subspace():
    sub, domain = _grid_manifold()
    h = _on_surface(sub, domain, [0.2, 0.8]) + 0.3 * torch.randn(16)
    _, perp_in = decompose(h, sub.mean, sub.basis)
    out, _ = subspace_inject(
        h, sub, domain, torch.tensor([0.9, 0.2]), torch.tensor([0.2, 0.8]),
        0.7, 0.5, gn_steps=20,
    )
    _, perp_out = decompose(out, sub.mean, sub.basis)
    assert torch.allclose(perp_out, perp_in, atol=1e-4)


def test_inject_norm_cap_bounds_output():
    sub, domain = _grid_manifold()
    h = _on_surface(sub, domain, [0.5, 0.5]) + 0.3 * torch.randn(16)
    for a in (0.0, 0.5, 1.0):
        for o in (0.0, 1.0):
            out, _ = subspace_inject(
                h, sub, domain, torch.tensor([0.9, 0.1]),
                torch.tensor([0.5, 0.5]), a, o, gn_steps=20, norm_cap=3.0,
            )
            assert torch.isfinite(out).all()
            ratio = out.norm() / h.norm()
            assert ratio <= 3.0 + 1e-4


# ----------------------------------------------- fuzzy-manifold σ-field ---
#
# A curved manifold can carry a per-node within-node off-surface spread
# interpolated as a separate ``log σ`` RBF (``LayerSubspace.sigma_at``).  It
# gives the surface a *tube thickness*: soft ``onto`` shrinks the off-surface
# residual toward ``σ`` (the typical set) instead of to the zero-thickness
# wire.  Absent ⇒ ``sigma_at`` returns 0 ⇒ exact legacy behavior.


def test_activation_row_store_combines_disjoint_layers_without_copy() -> None:
    left = ActivationRowStore([2, 1])
    right = ActivationRowStore([2, 1])
    indices = torch.arange(3)
    left.write(0, indices, torch.randn(3, 5))
    right.write(2, indices, torch.randn(3, 5))
    left_ptr = left.flat_rows(0).data_ptr()
    right_ptr = right.flat_rows(2).data_ptr()

    combined = ActivationRowStore.combine_disjoint([left, right])

    assert combined.layer_indices == [0, 2]
    assert combined.flat_rows(0).data_ptr() == left_ptr
    assert combined.flat_rows(2).data_ptr() == right_ptr
    combined.close()
    assert left._closed and right._closed


def test_node_stats_returns_normalized_accumulator_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Centroid normalization does not allocate a second K x D roster."""
    import saklas.core.manifold as manifold_module
    from saklas.core import vectors

    captured_accumulators: list[torch.Tensor] = []
    real_zeros = torch.zeros

    def tracked_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = real_zeros(*args, **kwargs)
        if args[:2] == (2, 2) and kwargs.get("device") is None:
            captured_accumulators.append(result)
        return result

    def fake_capture(*_args: Any, **_kwargs: Any) -> dict[int, torch.Tensor]:
        return {0: torch.tensor([
            [1.0, 3.0], [3.0, 5.0], [10.0, 14.0], [14.0, 18.0],
        ])}

    monkeypatch.setattr(manifold_module.torch, "zeros", tracked_zeros)
    monkeypatch.setattr(vectors, "_encode_and_capture_all_batch", fake_capture)
    centroids, retained = compute_manifold_node_stats(
        torch.nn.Linear(1, 1), object(),
        torch.nn.ModuleList([torch.nn.Identity()]), torch.device("cpu"),
        [("a", ["a1", "a2"]), ("b", ["b1", "b2"])], ["prompt"],
        roles=[None, None], layer_indices=[0], retain_rows=False,
        prepared_rows=[(torch.tensor([[1]]), 0)] * 4,
    )

    assert retained is None
    assert len(captured_accumulators) == 1
    assert centroids[0] is captured_accumulators[0]
    assert torch.equal(centroids[0], torch.tensor([[2.0, 4.0], [12.0, 16.0]]))


def test_layer_major_store_covariances_match_node_reference() -> None:
    torch.manual_seed(4)
    node_sizes = [2, 3, 1]
    store = ActivationRowStore(node_sizes)
    total = sum(node_sizes)
    indices = torch.arange(total)
    layer_subs: dict[int, LayerSubspace] = {}
    for layer in (0, 1, 2):
        rows = (
            torch.full((total, 7), 1_000_000.0) + 10.0 * torch.randn(total, 7)
            if layer == 2
            else torch.randn(total, 7, dtype=torch.float16)
        )
        store.write(layer, indices, rows)
        basis = torch.linalg.qr(torch.randn(7, 3)).Q.T.contiguous()
        layer_subs[layer] = LayerSubspace.affine(
            (
                torch.full((7,), 1_000_000.0)
                if layer == 2 else torch.randn(7)
            ),
            basis,
        )

    before = {
        layer: store.flat_rows(layer).clone() for layer in store.layer_indices
    }
    expected = [
        compute_node_reduced_covariance_from_rows(rows, layer_subs)
        for rows in store
    ]
    actual = compute_store_reduced_covariances(
        store, layer_subs, row_chunk=4,
    )

    for expected_node, actual_node in zip(expected, actual, strict=True):
        for layer in layer_subs:
            assert torch.allclose(
                actual_node[layer], expected_node[layer], atol=1e-6,
            )
    for layer, original in before.items():
        assert torch.equal(store.flat_rows(layer), original), (
            "covariance projection mutated the activation-row store"
        )
    store.close()

def _attach_constant_sigma(sub: Any, value: float) -> None:
    """Fit a σ-RBF interpolating a constant ``σ = value`` over the layout.

    A constant value is reproduced exactly by the affine polynomial term (the
    RBF weights vanish), so ``sigma_at`` returns ``value`` everywhere — the
    clean, predictable σ-field for testing the soft-``onto`` arithmetic.
    """
    np_, _rw, _pc = sub.rbf_params()
    K = np_.shape[0]
    log_sigma = torch.full((K, 1), math.log(value), dtype=torch.float32)
    w, c, _ = fit_rbf_smoothed(np_, log_sigma, smoothing=0.0)
    sub.sigma_rbf_weights = w
    sub.sigma_poly_coeffs = c


def test_sigma_at_absent_is_zero():
    # A freshly fitted curved subspace carries no σ-field → zero-thickness wire.
    sub, domain = _grid_manifold()
    assert not sub.has_sigma
    z = domain.embed(torch.tensor([0.3, 0.7]))
    assert float(sub.sigma_at(z)) == 0.0


def test_sigma_at_constant_field():
    sub, domain = _grid_manifold()
    _attach_constant_sigma(sub, 0.7)
    assert sub.has_sigma
    for coord in ([0.1, 0.2], [0.5, 0.5], [0.9, 0.4]):
        z = domain.embed(torch.tensor(coord))
        assert abs(float(sub.sigma_at(z)) - 0.7) < 1e-3


def _curved_with_pure_residual(sigma: float | None, resid_norm: float = 3.0):
    """A curved sub + an on-surface point at [0.5,0.5] plus a pure in-subspace
    off-surface residual of norm ``resid_norm``; optionally σ-field-equipped."""
    sub, domain = _grid_manifold()
    if sigma is not None:
        _attach_constant_sigma(sub, sigma)
    torch.manual_seed(7)
    pert = torch.randn(16)
    pert = (pert @ sub.basis.T) @ sub.basis          # pure in-subspace (H_n)
    pert = resid_norm * pert / pert.norm()
    h = _on_surface(sub, domain, [0.5, 0.5]) + pert
    return sub, domain, h


def test_soft_onto_lands_one_sigma_off_the_wire():
    # With a σ-field, onto=1 shrinks ‖H_n‖ to σ (not 0): the activation lands on
    # the tube, one σ off the mean surface — direction preserved.
    sigma = 0.5
    sub, domain, h = _curved_with_pure_residual(sigma, resid_norm=3.0)
    pos = torch.tensor([0.5, 0.5])
    out, foot = subspace_inject(h, sub, domain, pos, pos, 0.0, 1.0, gn_steps=20)
    q_out = (out - sub.mean) @ sub.basis.T
    foot_red = (sub.eval_at(domain.embed(foot)) - sub.mean) @ sub.basis.T
    hn_out = q_out - foot_red
    assert abs(float(hn_out.norm()) - sigma) < 5e-3
    # direction preserved vs the hard-collapse direction (onto=0 keeps full H_n)
    out0, foot0 = subspace_inject(h, sub, domain, pos, pos, 0.0, 0.0, gn_steps=20)
    hn0 = (out0 - sub.mean) @ sub.basis.T - (
        sub.eval_at(domain.embed(foot0)) - sub.mean
    ) @ sub.basis.T
    cos = float((hn_out @ hn0) / (hn_out.norm() * hn0.norm()))
    assert cos > 0.999


def test_soft_onto_without_field_collapses_to_wire():
    # No σ-field → onto=1 is the exact legacy collapse: H_n → 0.
    sub, domain, h = _curved_with_pure_residual(None, resid_norm=3.0)
    pos = torch.tensor([0.5, 0.5])
    out, foot = subspace_inject(h, sub, domain, pos, pos, 0.0, 1.0, gn_steps=20)
    hn_out = (out - sub.mean) @ sub.basis.T - (
        sub.eval_at(domain.embed(foot)) - sub.mean
    ) @ sub.basis.T
    assert float(hn_out.norm()) < 5e-3


def test_soft_onto_never_expands_inside_tube():
    # A residual already smaller than σ is left untouched (the (·)_+ clamp).
    sigma = 5.0
    sub, domain, h = _curved_with_pure_residual(sigma, resid_norm=1.0)
    pos = torch.tensor([0.5, 0.5])
    out0, foot0 = subspace_inject(h, sub, domain, pos, pos, 0.0, 0.0, gn_steps=20)
    out1, foot1 = subspace_inject(h, sub, domain, pos, pos, 0.0, 1.0, gn_steps=20)
    hn0 = (out0 - sub.mean) @ sub.basis.T - (
        sub.eval_at(domain.embed(foot0)) - sub.mean
    ) @ sub.basis.T
    hn1 = (out1 - sub.mean) @ sub.basis.T - (
        sub.eval_at(domain.embed(foot1)) - sub.mean
    ) @ sub.basis.T
    assert torch.allclose(hn1, hn0, atol=5e-3)


def test_off_surface_var_isolates_normal_directions():
    # A 4-D reduced space, surface tangent spanning dims 0..1; a covariance with
    # variance only in the tangent directions has zero off-surface variance, and
    # one with variance only in the normal directions returns that variance.
    R, n = 4, 2
    tangent = torch.zeros(R, n)
    tangent[0, 0] = 1.0
    tangent[1, 1] = 1.0
    cov_tan = torch.diag(torch.tensor([3.0, 3.0, 0.0, 0.0]))
    assert _off_surface_var(cov_tan, tangent, R, n) < 1e-5
    cov_norm = torch.diag(torch.tensor([0.0, 0.0, 2.0, 4.0]))
    # off-surface mean variance = (2 + 4) / (R − n) = 3.0
    assert abs(_off_surface_var(cov_norm, tangent, R, n) - 3.0) < 1e-5


def test_off_surface_vars_matches_scalar_helper():
    R, n = 4, 2
    tangents = torch.zeros(3, R, n)
    tangents[:, 0, 0] = 1.0
    tangents[:, 1, 1] = 1.0
    covs = torch.stack([
        torch.diag(torch.tensor([1.0, 0.0, 2.0, 4.0])),
        torch.diag(torch.tensor([0.0, 3.0, 0.5, 1.5])),
        torch.diag(torch.tensor([2.0, 2.0, 0.0, 8.0])),
    ])

    batched = _off_surface_vars(covs, tangents, R, n)
    scalar = torch.tensor([
        _off_surface_var(covs[i], tangents[i], R, n)
        for i in range(covs.shape[0])
    ])

    assert torch.allclose(batched, scalar, atol=1e-6)


def test_off_surface_variance_uses_actual_rank_at_fold() -> None:
    # Nominal n=2, but a local fold collapses the tangent to rank 1.  The
    # actual normal complement is 2-D, so isotropic unit covariance stays 1.
    cov = torch.eye(3)
    tangent = torch.tensor([
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    assert _off_surface_var(cov, tangent, R=3, n=2) == pytest.approx(1.0)
    assert _off_surface_vars(
        cov.unsqueeze(0), tangent.unsqueeze(0), R=3, n=2,
    ).item() == pytest.approx(1.0)


def test_off_surface_variance_rank_deficient_when_r_le_n() -> None:
    # Even when nominal n fills R, a rank-1 tangent leaves a real normal axis;
    # sigma must isolate its variance instead of averaging tangent+normal.
    cov = torch.diag(torch.tensor([3.0, 7.0]))
    tangent = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert _off_surface_var(cov, tangent, R=2, n=2) == pytest.approx(7.0)
    assert _off_surface_vars(
        cov.unsqueeze(0), tangent.unsqueeze(0), R=2, n=2,
    ).item() == pytest.approx(7.0)


def test_off_surface_variance_fuses_rank_and_projector_svd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused SVD uses the matrix-rank tolerance at a near-rank fold."""
    R = n = 2
    eps = torch.finfo(torch.float32).eps
    threshold = max(R, n) * eps
    tangents = torch.stack([
        torch.diag(torch.tensor([1.0, threshold * 2.0])),
        torch.diag(torch.tensor([1.0, threshold * 0.5])),
        torch.zeros(2, 2),
    ])
    covs = torch.stack([
        torch.diag(torch.tensor([3.0, 7.0])),
        torch.diag(torch.tensor([3.0, 7.0])),
        torch.diag(torch.tensor([3.0, 7.0])),
    ])

    original_svd = torch.linalg.svd
    calls = 0

    def counted_svd(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted_svd)
    monkeypatch.setattr(
        torch.linalg, "pinv",
        lambda *_args, **_kwargs: pytest.fail("projector repeated the SVD"),
    )
    monkeypatch.setattr(
        torch.linalg, "matrix_rank",
        lambda *_args, **_kwargs: pytest.fail("rank repeated the SVD"),
    )

    result = _off_surface_vars(covs, tangents, R, n)

    assert calls == 1
    # Above tolerance the tangent fills R, selecting the isotropic fallback;
    # below it the second axis is normal and its variance is isolated.
    assert result.tolist() == pytest.approx([5.0, 7.0, 5.0])


def test_fit_sigma_field_interpolates_node_thickness():
    # fit_sigma_field reduces per-node covariances to off-surface σ and attaches
    # a log-σ RBF; sigma_at at a node recovers that node's off-surface std.
    sub, domain = _grid_manifold()
    coords = torch.tensor(
        [[u, v] for u in (0.0, 0.5, 1.0) for v in (0.0, 0.5, 1.0)]
    )
    R = sub.rank
    # Per-node isotropic covariance, distinct scale per node so interpolation is
    # non-trivial. Isotropic ⇒ off-surface σ == the isotropic scale (equal
    # variance in every direction, normal or tangent). Keyed by layer 0 like the
    # real pipeline's per-node ``{layer: (R,R)}``.
    layer_subs = {0: sub}
    node_covs = [
        {0: (0.1 + 0.05 * k) ** 2 * torch.eye(R)}
        for k in range(coords.shape[0])
    ]
    info = fit_sigma_field(layer_subs, domain, coords, node_covs, smoothing=0.0)
    assert 0 in info and info[0]["sigma_mean"] > 0
    assert sub.has_sigma
    # at node 0 (coord [0,0]) the off-surface σ ≈ sqrt(iso var on normal dims)
    # = the isotropic scale 0.1 (isotropic cov has equal var in every direction)
    z0 = domain.embed(coords[0])
    assert abs(float(sub.sigma_at(z0)) - 0.1) < 2e-2


def test_sigma_field_save_load_round_trip(tmp_path: Path):
    sub, domain = _grid_manifold()
    _attach_constant_sigma(sub, 0.42)
    coords = torch.tensor(
        [[u, v] for u in (0.0, 0.5, 1.0) for v in (0.0, 0.5, 1.0)]
    )
    man = Manifold(
        name="fuzzy", domain=domain,
        node_labels=[f"n{i}" for i in range(9)],
        node_coords=coords, layers={0: sub},
        mahalanobis_share={0: 1.0}, origin={0: torch.zeros(2)},
    )
    path = tmp_path / "fuzzy.safetensors"
    save_manifold(man, path, {
        "method": "manifold_discover_spectral", "fit_mode": "spectral",
    })
    loaded = load_manifold(path)
    lsub = loaded.layers[0]
    assert lsub.has_sigma
    for coord in ([0.2, 0.3], [0.8, 0.6]):
        z = domain.embed(torch.tensor(coord))
        assert abs(float(lsub.sigma_at(z)) - float(sub.sigma_at(z))) < 1e-4


def test_curved_manifold_without_sigma_is_rejected(tmp_path: Path):
    sub, domain = _grid_manifold()
    coords = torch.tensor(
        [[u, v] for u in (0.0, 0.5, 1.0) for v in (0.0, 0.5, 1.0)]
    )
    man = Manifold(
        name="legacy", domain=domain,
        node_labels=[f"n{i}" for i in range(9)],
        node_coords=coords, layers={0: sub},
        mahalanobis_share={0: 1.0}, origin={0: torch.zeros(2)},
    )
    path = tmp_path / "legacy.safetensors"
    with pytest.raises(ValueError, match="requires a sigma field"):
        save_manifold(man, path, {
            "method": "manifold_discover_spectral", "fit_mode": "spectral",
        })


# ------------------------------------------------- affine (flat) subspace ---
#
# A folded steering vector is the degenerate n = R = 1 case: an affine
# subspace whose "surface" fills its span, so H_n ≡ 0 and subspace_inject
# takes the analytic shortcut (no GN solve / RBF eval / tangent solve).

def _folded_vector(dim: int = 16, seed: int = 0):
    """An affine folded-vector subspace (n = R = 1) + its CustomDomain(1)."""
    torch.manual_seed(seed)
    raw = torch.randn(dim)
    basis = (raw / raw.norm()).reshape(1, dim)     # (1, D) unit direction
    mean = 20.0 + torch.randn(dim)                 # realistic DC offset
    return LayerSubspace.affine(mean, basis), CustomDomain(1)


def test_affine_factory_and_is_affine():
    sub, _ = _folded_vector()
    assert sub.is_affine
    assert sub.node_params is None
    assert sub.rank == 1
    # identity normalization so authoring coords == reduced coords
    assert torch.allclose(sub.coord_offset, torch.zeros(1))
    assert torch.allclose(sub.coord_scale, torch.ones(1))
    # a curved fit is not affine
    curved, _dom = _grid_manifold()
    assert not curved.is_affine


def test_affine_rbf_params_raises():
    sub, _ = _folded_vector()
    with pytest.raises(ValueError, match="affine"):
        sub.rbf_params()


def test_affine_eval_at_is_pure_affine():
    sub, _ = _folded_vector()
    for c in (-1.3, 0.0, 0.4, 2.1):
        coord = torch.tensor([c])
        assert torch.allclose(sub.eval_at(coord), coord @ sub.basis + sub.mean)
    # batched
    coords = torch.tensor([[-0.5], [0.0], [1.0]])
    assert torch.allclose(sub.eval_at(coords), coords @ sub.basis + sub.mean)


def test_affine_jacobian_is_basis_T():
    sub, _ = _folded_vector()
    # single point: (D, m) == basis.T, position-independent
    jac = sub.jacobian_at(torch.tensor([0.7]))
    assert torch.allclose(jac, sub.basis.transpose(-1, -2))
    # batched: broadcast across the leading dim
    jb = sub.jacobian_at(torch.tensor([[0.1], [0.9]]))
    assert jb.shape == (2, sub.basis.shape[1], 1)
    assert torch.allclose(jb[0], sub.basis.transpose(-1, -2))


def test_affine_inject_identity_at_zero():
    sub, domain = _folded_vector()
    h = sub.eval_at(torch.tensor([0.6])) + 0.2 * torch.randn(16)
    out, foot = subspace_inject(
        h, sub, domain, torch.tensor([0.9]), torch.tensor([0.6]),
        0.0, 0.0,
    )
    assert torch.allclose(out, h, atol=1e-4)
    # the returned foot is q exactly (the in-subspace reduced coord of h)
    assert torch.allclose(foot, (h - sub.mean) @ sub.basis.T, atol=1e-5)


def test_affine_inject_along_translates_coord_by_target():
    # Translate semantics (not collapse): ``along`` shifts the foot BY the fixed
    # offset ``along·target`` (origin = 0 in the ν-anchored affine frame), so the
    # per-token foot spread is preserved — it does NOT slide every foot onto the
    # absolute target (that collapse erased the spread and degenerated to looping).
    sub, domain = _folded_vector()
    h = sub.eval_at(torch.tensor([-0.4])) + 0.3 * torch.randn(16)
    target = torch.tensor([1.5])
    q_in = (h - sub.mean) @ sub.basis.T             # input reduced coord
    out, _ = subspace_inject(
        h, sub, domain, target, torch.tensor([-0.4]), 1.0, 0.0,
    )
    q_out = (out - sub.mean) @ sub.basis.T          # reduced coord of the output
    assert torch.allclose(q_out, q_in + target, atol=1e-4)
    # half-slide translates by half the offset
    out_h, _ = subspace_inject(
        h, sub, domain, target, torch.tensor([-0.4]), 0.5, 0.0,
    )
    q_half = (out_h - sub.mean) @ sub.basis.T
    assert torch.allclose(q_half, q_in + 0.5 * target, atol=1e-4)


def test_affine_inject_onto_is_vacuous():
    # H_n ≡ 0 on a flat subspace, so onto must have no effect.
    sub, domain = _folded_vector()
    h = sub.eval_at(torch.tensor([0.2])) + 0.3 * torch.randn(16)
    pos, seed = torch.tensor([0.2]), torch.tensor([0.2])
    out0, _ = subspace_inject(h, sub, domain, pos, seed, 0.3, 0.0)
    out1, _ = subspace_inject(h, sub, domain, pos, seed, 0.3, 1.0)
    assert torch.allclose(out0, out1, atol=1e-5)


def test_affine_inject_keeps_off_subspace_verbatim():
    # On a flat subspace the off-subspace residual ``H_o`` is kept verbatim —
    # the old ``toward`` op that scaled it is removed — even under a full slide.
    sub, domain = _folded_vector()
    h = sub.eval_at(torch.tensor([0.3])) + 0.4 * torch.randn(16)
    _, perp_in = decompose(h, sub.mean, sub.basis)
    out, _ = subspace_inject(
        h, sub, domain, torch.tensor([1.2]), torch.tensor([0.3]), 1.0, 1.0,
    )
    _, perp_out = decompose(out, sub.mean, sub.basis)
    assert torch.allclose(perp_out, perp_in, atol=1e-4)


def test_affine_manifold_point_round_trip():
    # A flat Manifold (folded vector) over two poles at ±s; manifold_point
    # recovers each pole centroid exactly.
    torch.manual_seed(3)
    dim = 12
    c_pos = 20.0 + torch.randn(dim)
    c_neg = 20.0 + torch.randn(dim)
    mean = 0.5 * (c_pos + c_neg)
    diff = c_pos - c_neg
    s = float(diff.norm()) / 2.0
    basis = (diff / diff.norm()).reshape(1, dim)
    sub = LayerSubspace.affine(mean, basis)
    mfld = Manifold(
        name="folded",
        domain=CustomDomain(1),
        node_labels=["pos", "neg"],
        node_coords=torch.tensor([[s], [-s]]),
        layers={7: sub},
    )
    assert torch.allclose(mfld.manifold_point(7, (s,)), c_pos, atol=1e-4)
    assert torch.allclose(mfld.manifold_point(7, (-s,)), c_neg, atol=1e-4)
    # midpoint coord lands at the mean
    assert torch.allclose(mfld.manifold_point(7, (0.0,)), mean, atol=1e-4)


def test_save_load_affine_manifold_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flat (folded-vector) manifold survives save/load: ``is_affine``
    preserved, mean/basis bit-identical, eval matches, and the on-disk
    payload carries *no* RBF triple (the absence-of-node_params marker)."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    sub_a, _ = _folded_vector(dim=18, seed=1)
    sub_b, _ = _folded_vector(dim=18, seed=2)
    real_node_coords = torch.tensor([[0.8], [-0.8]])
    sub_a.node_coords = real_node_coords.clone()
    sub_b.node_coords = real_node_coords.clone()
    manifold = Manifold(
        name="folded",
        domain=CustomDomain(1),
        node_labels=["pos", "neg"],
        node_coords=real_node_coords,
        layers={5: sub_a, 11: sub_b},
        feature_space="raw",
        mahalanobis_share={5: 1.2, 11: 0.7},
        origin={},
    )
    path = tmp_path / "folded" / "model.safetensors"
    save_manifold(manifold, path, {
        "method": "folded_vector", "fit_mode": "baked",
        "share_metric": "mahalanobis",
    })

    # The on-disk payload omits the RBF triple + coord normalization for the
    # flat layers — only mean + basis (plus the shared node_coords).
    from safetensors.torch import load_file as _load_file
    raw = _load_file(str(path))
    assert "node_coords" in raw
    for idx in (5, 11):
        assert f"layer_{idx}.mean" in raw
        assert f"layer_{idx}.basis" in raw
        assert f"layer_{idx}.node_params" not in raw
        assert f"layer_{idx}.rbf_weights" not in raw
        assert f"layer_{idx}.poly_coeffs" not in raw
        assert f"layer_{idx}.coord_offset" not in raw
        assert f"layer_{idx}.coord_scale" not in raw

    loaded = load_manifold(path)
    assert sorted(loaded.layers) == [5, 11]
    assert loaded.mahalanobis_share == {5: 1.2, 11: 0.7}
    assert torch.allclose(loaded.node_coords, manifold.node_coords)
    for idx in (5, 11):
        a, b = manifold.layers[idx], loaded.layers[idx]
        assert b.is_affine
        assert b.node_params is None
        assert torch.allclose(a.mean, b.mean)
        assert torch.allclose(a.basis, b.basis)
        # identity coord normalization is rebuilt from the basis shape
        assert torch.allclose(b.coord_offset, torch.zeros(b.rank))
        assert torch.allclose(b.coord_scale, torch.ones(b.rank))
        # evaluation matches after the round-trip
        for c in (-0.5, 0.0, 0.8):
            assert torch.allclose(
                loaded.manifold_point(idx, (c,)),
                manifold.manifold_point(idx, (c,)),
                atol=1e-4,
            )


@pytest.mark.parametrize("node_coords", [torch.zeros(1, 2), torch.zeros(2, 1)])
def test_affine_runtime_geometry_requires_exact_k_by_r(
    node_coords: torch.Tensor,
) -> None:
    sub = LayerSubspace.affine(
        torch.zeros(4), torch.eye(2, 4), node_coords=node_coords,
    )
    manifold = Manifold(
        name="bad-affine", domain=CustomDomain(2),
        node_labels=["a", "b"], node_coords=torch.zeros(2, 2),
        layers={0: sub}, mahalanobis_share={0: 1.0},
    )
    with pytest.raises(ValueError, match="node_coords must have shape"):
        manifold.validate_runtime_geometry()


def test_runtime_geometry_rejects_explicit_empty_node_roster() -> None:
    coords = torch.tensor([[1.0], [-1.0]])
    sub = LayerSubspace.affine(
        torch.zeros(4), torch.ones(1, 4) / 2, node_coords=coords,
    )
    manifold = Manifold(
        name="bad-roster", domain=CustomDomain(1),
        node_labels=["a", "b"], node_coords=coords, layers={0: sub},
        node_roles=[], node_kinds=[None, None],
        mahalanobis_share={0: 1.0},
    )
    with pytest.raises(ValueError, match="node_roles must align exactly"):
        manifold.validate_runtime_geometry()


def test_runtime_geometry_rejects_wrong_shared_authoring_arity() -> None:
    per_layer = torch.tensor([[1.0], [-1.0]])
    sub = LayerSubspace.affine(
        torch.zeros(4), torch.ones(1, 4) / 2, node_coords=per_layer,
    )
    manifold = Manifold(
        name="bad-domain", domain=CustomDomain(1),
        node_labels=["a", "b"], node_coords=torch.zeros(2, 2),
        layers={0: sub}, mahalanobis_share={0: 1.0},
    )
    with pytest.raises(ValueError, match=r"shape \(K, n\)"):
        manifold.validate_runtime_geometry()
