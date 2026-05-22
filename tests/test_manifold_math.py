"""RBF + domain + manifold math for saklas.core.manifold.

Pure CPU tests — no model, no IO beyond a save/load round-trip in a
temp directory.
"""
from __future__ import annotations

import math

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
    fit_layer_subspace,
    fit_rbf_interpolant,
    invert_parameterization,
    load_manifold,
    save_manifold,
    subspace_replace,
)


# ------------------------------------------------------------------ domains ---

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


def _grid2d(steps=(0.0, 0.5, 1.0)) -> tuple[BoxDomain, torch.Tensor]:
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
    pos, dist = invert_parameterization(sub, domain, query.unsqueeze(0))
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


# ----------------------------------------------------------------- save/load ---

def test_save_load_manifold_round_trip(tmp_path, monkeypatch):
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
