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
    BoxAxis,
    BoxDomain,
    CustomDomain,
    Manifold,
    SphereDomain,
    domain_from_spec,
    eval_rbf,
    eval_rbf_jacobian,
    fit_layer_subspace as _fit_layer_subspace_with_ev,  # returns (LayerSubspace, ev_ratio)
    fit_rbf_interpolant,
    invert_parameterization,
    load_manifold,
    save_manifold,
    subspace_replace,
    subspace_rotate,
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
    assert torch.allclose(p0, centroids[0], atol=1e-3)


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
    # a corner node reproduces its centroid
    assert torch.allclose(
        man.manifold_point(0, (0.0, 0.0)), centroids[0], atol=1e-3,
    )


def test_fit_layer_subspace_rejects_too_few_nodes():
    with pytest.raises(ValueError):
        fit_layer_subspace(torch.randn(2, 8), torch.rand(2, 1))


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


# ------------------------------------------------------------ subspace_replace ---

def _ortho_basis(r: int, dim: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, r))
    return q.T.contiguous()  # (r, dim) orthonormal rows


def test_subspace_replace_alpha_zero_is_identity():
    torch.manual_seed(0)
    dim = 32
    h = torch.randn(4, dim)
    mean = torch.randn(dim)
    basis = _ortho_basis(3, dim)
    target = torch.randn(3) @ basis + mean
    out = subspace_replace(h, mean, basis, target, alpha=0.0)
    assert torch.allclose(out, h, atol=1e-5)


def test_subspace_replace_preserves_norm():
    torch.manual_seed(1)
    dim = 48
    h = torch.randn(5, dim)
    mean = torch.randn(dim)
    basis = _ortho_basis(4, dim)
    target = torch.randn(4) @ basis + mean
    for alpha in (0.25, 0.5, 1.0):
        out = subspace_replace(h, mean, basis, target, alpha=alpha)
        assert torch.allclose(out.norm(dim=-1), h.norm(dim=-1), atol=1e-4)


def test_subspace_replace_alpha_one_lands_on_target():
    torch.manual_seed(2)
    dim = 24
    h = torch.randn(dim)
    mean = torch.randn(dim)
    basis = _ortho_basis(3, dim)
    target = torch.randn(3) @ basis + mean
    out = subspace_replace(h, mean, basis, target, alpha=1.0)
    h_par = (h - mean) @ basis.T @ basis + mean
    h_perp = h - h_par
    expected_dir = h_perp + target
    cos = torch.dot(out, expected_dir) / (out.norm() * expected_dir.norm())
    assert cos.item() == pytest.approx(1.0, abs=1e-4)


# ------------------------------------------------------------ subspace_rotate ---

def test_subspace_rotate_alpha_zero_is_identity():
    torch.manual_seed(0)
    dim = 32
    h = torch.randn(4, dim)
    mean = torch.randn(dim)
    basis = _ortho_basis(3, dim)
    target = torch.randn(3) @ basis + mean
    out = subspace_rotate(h, mean, basis, target, alpha=0.0, theta_max=math.pi / 2)
    assert torch.allclose(out, h, atol=1e-5)


def test_subspace_rotate_preserves_centered_norm():
    """``||h - mean||`` is invariant — rotation in the subspace plane
    leaves ``||h_par||`` exact and ``h_perp`` untouched, so the centered
    magnitude is conserved without a norm-restore step."""
    torch.manual_seed(1)
    dim = 48
    h = torch.randn(5, dim)
    mean = torch.randn(dim)
    basis = _ortho_basis(4, dim)
    target = torch.randn(4) @ basis + mean
    centered_before = (h - mean).norm(dim=-1)
    for alpha in (0.1, 0.25, 0.5, 1.0):
        out = subspace_rotate(
            h, mean, basis, target, alpha=alpha, theta_max=math.pi / 2,
        )
        centered_after = (out - mean).norm(dim=-1)
        assert torch.allclose(centered_after, centered_before, atol=1e-4)


def test_subspace_rotate_preserves_h_perp():
    """The orthogonal-to-subspace component is the part the manifold has
    no opinion about — keep it bit-stable through the rotation."""
    torch.manual_seed(2)
    dim = 40
    h = torch.randn(3, dim)
    mean = torch.randn(dim)
    basis = _ortho_basis(3, dim)
    target = torch.randn(3) @ basis + mean
    centered = h - mean
    h_par_before = (centered @ basis.T) @ basis
    h_perp_before = centered - h_par_before
    out = subspace_rotate(
        h, mean, basis, target, alpha=0.6, theta_max=math.pi / 2,
    )
    centered_after = out - mean
    h_par_after = (centered_after @ basis.T) @ basis
    h_perp_after = centered_after - h_par_after
    assert torch.allclose(h_perp_after, h_perp_before, atol=1e-4)


def test_subspace_rotate_alpha_one_aligns_h_par_with_target():
    """At α=1 with θ_max=π/2 the rotation lands ``h_par`` orthogonal to
    its starting direction, in the plane toward the target — its
    *centered direction* coincides with the in-plane perpendicular axis
    pointing from ``h_par`` toward the target.  This is the angular
    analogue of subspace_replace's "snap onto target" — magnitudes are
    preserved (h_par magnitude stays), directions are matched modulo the
    rotation plane's in-plane orthogonal vector."""
    torch.manual_seed(3)
    dim = 24
    h = torch.randn(dim)
    mean = torch.randn(dim)
    basis = _ortho_basis(3, dim)
    target = torch.randn(3) @ basis + mean

    centered = h - mean
    h_par_c = (centered @ basis.T) @ basis
    target_c = target - mean
    u = h_par_c / h_par_c.norm()
    target_unit = target_c / target_c.norm()
    cos0 = (u * target_unit).sum()
    w = target_unit - cos0 * u
    w_unit = w / w.norm()

    out = subspace_rotate(
        h, mean, basis, target, alpha=1.0, theta_max=math.pi / 2,
    )
    centered_after = out - mean
    h_par_after = (centered_after @ basis.T) @ basis
    # h_par_after should be norm * w_unit (cos(π/2)=0, sin(π/2)=1).
    cos = torch.dot(h_par_after, w_unit) / h_par_after.norm()
    assert cos.item() == pytest.approx(1.0, abs=1e-3)
    assert h_par_after.norm().item() == pytest.approx(
        h_par_c.norm().item(), abs=1e-3,
    )


def test_subspace_norms_fp32_accumulate_at_large_dim():
    """Defense-in-depth fp32 norm parity with the vector hot path.

    Both injection primitives now pass ``dtype=torch.float32`` to every
    ``vector_norm`` (matching ``hooks.py``).  With fp16 inputs at a large
    hidden dim the fp16 sum-of-squares would overflow / lose precision;
    the fp32 accumulation keeps the norm-preservation invariants exact to
    the fp16 round-trip tolerance.  This pins the regression so a future
    refactor can't silently drop the explicit dtype back to incidental.
    """
    torch.manual_seed(7)
    dim = 4096                       # >= 2048: fp16 sum-of-squares overflows
    h = (torch.randn(3, dim) * 4.0).to(torch.float16)
    mean = (torch.randn(dim) * 4.0).to(torch.float16)
    basis = _ortho_basis(4, dim).to(torch.float16)
    target = ((torch.randn(4) @ basis.float()) + mean.float()).to(torch.float16)

    # subspace_replace: ||h|| preserved.
    norm_pre = h.float().norm(dim=-1)
    out_r = subspace_replace(h, mean, basis, target, alpha=0.5)
    assert out_r.dtype == torch.float16
    assert torch.allclose(out_r.float().norm(dim=-1), norm_pre, rtol=2e-2)
    assert torch.isfinite(out_r.float()).all()

    # subspace_rotate: ||h - mean|| preserved.
    centered_pre = (h.float() - mean.float()).norm(dim=-1)
    out_a = subspace_rotate(
        h, mean, basis, target, alpha=0.5, theta_max=math.pi / 2,
    )
    assert out_a.dtype == torch.float16
    centered_post = (out_a.float() - mean.float()).norm(dim=-1)
    assert torch.allclose(centered_post, centered_pre, rtol=2e-2)
    assert torch.isfinite(out_a.float()).all()


def test_subspace_rotate_degenerate_h_par_is_identity():
    """At the manifold origin (h ≈ mean) the rotation plane is
    undefined — fall back to identity rather than emit NaN."""
    torch.manual_seed(4)
    dim = 16
    mean = torch.randn(dim)
    basis = _ortho_basis(3, dim)
    # Put h exactly on the manifold mean -- h_par_c is zero.
    h = mean.clone().unsqueeze(0)
    target = torch.randn(3) @ basis + mean
    out = subspace_rotate(
        h, mean, basis, target, alpha=0.7, theta_max=math.pi / 2,
    )
    assert torch.allclose(out, h, atol=1e-5)


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
    )
    path = tmp_path / "mood" / "model.safetensors"
    save_manifold(manifold, path, {"method": "manifold_pca",
                                   "nodes_sha256": "abc123"})
    loaded = load_manifold(path)

    assert loaded.name == "mood"
    assert loaded.domain.intrinsic_dim == 1
    assert loaded.node_labels == [f"n{i}" for i in range(7)]
    assert sorted(loaded.layers) == [4, 9]
    assert loaded.feature_space == "raw"
    assert loaded.metadata["nodes_sha256"] == "abc123"
    assert torch.allclose(loaded.node_coords, manifold.node_coords)
    for idx in (4, 9):
        a, b = manifold.layers[idx], loaded.layers[idx]
        assert torch.allclose(a.mean, b.mean)
        assert torch.allclose(a.basis, b.basis)
        assert torch.allclose(a.rbf_weights, b.rbf_weights)
        assert torch.allclose(a.poly_coeffs, b.poly_coeffs)
    # evaluation matches after the round-trip
    assert torch.allclose(
        loaded.manifold_point(4, (0.3,)),
        manifold.manifold_point(4, (0.3,)),
        atol=1e-4,
    )


def test_layer_subspace_to_device_dtype():
    centroids, _domain, node_params = _circle(5, dim=10)
    sub = fit_layer_subspace(centroids, node_params)
    moved = sub.to(device=torch.device("cpu"), dtype=torch.float64)
    assert moved.basis.dtype == torch.float64
    assert moved.mean.dtype == torch.float64
    assert moved.rbf_weights.dtype == torch.float64
