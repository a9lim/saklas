"""Spline + manifold math for saklas.core.manifold.

Pure CPU tests — no model, no IO beyond a save/load round-trip in a
temp directory.
"""
from __future__ import annotations

import math

import pytest
import torch

from saklas.core.manifold import (
    Manifold,
    eval_cubic,
    fit_layer_subspace,
    invert_parameterization,
    load_manifold,
    save_manifold,
    solve_natural_cubic,
    solve_periodic_cubic,
    subspace_replace,
)


# ----------------------------------------------------------- natural cubic ---

def test_natural_cubic_interpolates_knots():
    t = torch.tensor([0.0, 0.3, 0.55, 0.8, 1.0])
    y = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.5],
        [0.5, -1.0],
        [-1.0, 2.0],
        [2.0, 0.0],
    ])
    M = solve_natural_cubic(t, y)
    got = eval_cubic(t, y, M, t)
    assert torch.allclose(got, y, atol=1e-4)


def test_natural_cubic_boundary_condition():
    t = torch.tensor([0.0, 0.4, 0.7, 1.0])
    y = torch.tensor([[1.0], [3.0], [-2.0], [0.5]])
    M = solve_natural_cubic(t, y)
    # Natural boundary: zero second derivative at both endpoints.
    assert torch.allclose(M[0], torch.zeros_like(M[0]), atol=1e-6)
    assert torch.allclose(M[-1], torch.zeros_like(M[-1]), atol=1e-6)


def test_natural_cubic_first_derivative_continuous():
    # The interior knot should have a continuous first derivative.
    t = torch.tensor([0.0, 0.5, 1.0])
    y = torch.tensor([[0.0], [1.0], [0.0]])
    M = solve_natural_cubic(t, y)
    # One-sided finite differences each carry an O(eps * f'') bias, so
    # eps must be small enough that the bias gap stays under tolerance.
    eps = 1e-4
    left = (eval_cubic(t, y, M, torch.tensor([0.5]))
            - eval_cubic(t, y, M, torch.tensor([0.5 - eps]))) / eps
    right = (eval_cubic(t, y, M, torch.tensor([0.5 + eps]))
             - eval_cubic(t, y, M, torch.tensor([0.5]))) / eps
    assert torch.allclose(left, right, atol=1e-2)


def test_natural_cubic_two_knots_is_linear():
    t = torch.tensor([0.0, 1.0])
    y = torch.tensor([[0.0, 0.0], [2.0, -4.0]])
    M = solve_natural_cubic(t, y)
    assert torch.allclose(M, torch.zeros_like(M))
    mid = eval_cubic(t, y, M, torch.tensor([0.5]))
    assert torch.allclose(mid, torch.tensor([[1.0, -2.0]]), atol=1e-5)


# ---------------------------------------------------------- periodic cubic ---

def test_periodic_cubic_interpolates_knots():
    # K=4 distinct knots, P=5 with the wrap row repeating knot 0.
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    y = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],   # wrap == knot 0
    ])
    M = solve_periodic_cubic(t, y)
    got = eval_cubic(t, y, M, t)
    assert torch.allclose(got, y, atol=1e-4)


def test_periodic_cubic_seam_derivative_matches():
    # First derivative entering the seam equals the one leaving it.
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    y = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
    ])
    M = solve_periodic_cubic(t, y)
    # M wraps: second derivative at the seam matches.
    assert torch.allclose(M[0], M[-1], atol=1e-5)
    eps = 1e-4
    leaving = (eval_cubic(t, y, M, torch.tensor([eps]))
               - eval_cubic(t, y, M, torch.tensor([0.0]))) / eps
    entering = (eval_cubic(t, y, M, torch.tensor([1.0]))
                - eval_cubic(t, y, M, torch.tensor([1.0 - eps]))) / eps
    assert torch.allclose(leaving, entering, atol=1e-2)


def test_periodic_cubic_rejects_too_few_knots():
    t = torch.tensor([0.0, 0.5, 1.0])  # P=3 -> K=2 distinct
    y = torch.tensor([[0.0], [1.0], [0.0]])
    with pytest.raises(ValueError):
        solve_periodic_cubic(t, y)


# ---------------------------------------------------- inverse parameterization ---

def test_invert_parameterization_recovers_known_param():
    t = torch.tensor([0.0, 0.35, 0.7, 1.0])
    y = torch.tensor([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, -1.0],
        [4.0, 1.0],
    ])
    M = solve_natural_cubic(t, y)
    s_star = torch.tensor([0.1, 0.42, 0.63, 0.91])
    query = eval_cubic(t, y, M, s_star)
    s_rec, dist = invert_parameterization(t, y, M, query)
    assert torch.allclose(s_rec, s_star, atol=5e-3)
    assert torch.all(dist < 1e-2)


def test_invert_parameterization_off_curve_distance_positive():
    t = torch.tensor([0.0, 0.5, 1.0])
    y = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    M = solve_natural_cubic(t, y)
    # A point clearly off the (y=0) curve.
    query = torch.tensor([[1.0, 5.0]])
    _s, dist = invert_parameterization(t, y, M, query)
    assert dist.item() == pytest.approx(5.0, abs=0.1)


# --------------------------------------------------------- fit_layer_subspace ---

def _circle_centroids(k: int, dim: int) -> torch.Tensor:
    """K points on a circle living in the (e0, e1) plane of R^dim."""
    centroids = torch.zeros(k, dim)
    base = torch.full((dim,), 0.7)
    for i in range(k):
        theta = 2.0 * math.pi * i / k
        centroids[i] = base.clone()
        centroids[i, 0] += 2.0 * math.cos(theta)
        centroids[i, 1] += 2.0 * math.sin(theta)
    return centroids


def test_fit_layer_subspace_cyclic_spans_curve_plane():
    centroids = _circle_centroids(8, dim=16)
    sub = fit_layer_subspace(centroids, cyclic=True)
    # A planar circle has rank 2.
    assert sub.rank == 2
    # coords lift back to the original centroids.
    recon = sub.coords[:-1] @ sub.basis + sub.mean  # drop the wrap row
    assert torch.allclose(recon, centroids, atol=1e-3)


def test_fit_layer_subspace_eval_at_knots():
    centroids = _circle_centroids(6, dim=12)
    sub = fit_layer_subspace(centroids, cyclic=True)
    got = eval_cubic(sub.t_knots, sub.coords, sub.spline_M, sub.t_knots)
    assert torch.allclose(got, sub.coords, atol=1e-4)
    # spline_point lifts to world space.
    p0 = sub.spline_point(0.0)
    assert torch.allclose(p0, centroids[0], atol=1e-3)


def test_fit_layer_subspace_natural():
    # A non-closed sequence — sequential, not cyclic.
    centroids = torch.stack([
        torch.linspace(0.0, 1.0, 10) * c for c in (1.0, -2.0, 0.5)
    ], dim=1)  # (10, 3) along a straight line
    sub = fit_layer_subspace(centroids, cyclic=False)
    assert sub.t_knots.shape[0] == 10
    got = eval_cubic(sub.t_knots, sub.coords, sub.spline_M, sub.t_knots)
    assert torch.allclose(got, sub.coords, atol=1e-4)


def test_fit_layer_subspace_rejects_too_few_nodes():
    with pytest.raises(ValueError):
        fit_layer_subspace(torch.randn(2, 8), cyclic=False)


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
    target = torch.randn(3) @ basis + mean  # in-subspace
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
        assert torch.allclose(
            out.norm(dim=-1), h.norm(dim=-1), atol=1e-4,
        )


def test_subspace_replace_alpha_one_lands_on_target():
    torch.manual_seed(2)
    dim = 24
    h = torch.randn(dim)
    mean = torch.randn(dim)
    basis = _ortho_basis(3, dim)
    target = torch.randn(3) @ basis + mean  # in-subspace point

    out = subspace_replace(h, mean, basis, target, alpha=1.0)

    # At alpha=1 the pre-renorm result is (h_perp + target): in-subspace
    # component == target, residual == h's residual.  After renorm the
    # result is a positive multiple of that vector.
    h_par = (h - mean) @ basis.T @ basis + mean
    h_perp = h - h_par
    expected_dir = (h_perp + target)
    cos = torch.dot(out, expected_dir) / (
        out.norm() * expected_dir.norm()
    )
    assert cos.item() == pytest.approx(1.0, abs=1e-4)


# ----------------------------------------------------------------- save/load ---

def test_save_load_manifold_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    centroids_a = _circle_centroids(7, dim=20)
    centroids_b = _circle_centroids(7, dim=20) * 1.3
    manifold = Manifold(
        name="mood",
        cyclic=True,
        node_labels=[f"n{i}" for i in range(7)],
        layers={
            4: fit_layer_subspace(centroids_a, cyclic=True),
            9: fit_layer_subspace(centroids_b, cyclic=True),
        },
        feature_space="raw",
    )
    path = tmp_path / "mood" / "model.safetensors"
    save_manifold(manifold, path, {"method": "manifold_pca",
                                   "nodes_sha256": "abc123"})
    loaded = load_manifold(path)

    assert loaded.name == "mood"
    assert loaded.cyclic is True
    assert loaded.node_labels == [f"n{i}" for i in range(7)]
    assert sorted(loaded.layers) == [4, 9]
    assert loaded.feature_space == "raw"
    assert loaded.metadata["nodes_sha256"] == "abc123"
    for idx in (4, 9):
        a = manifold.layers[idx]
        b = loaded.layers[idx]
        assert torch.allclose(a.mean, b.mean)
        assert torch.allclose(a.basis, b.basis)
        assert torch.allclose(a.coords, b.coords)
        assert torch.allclose(a.spline_M, b.spline_M)


def test_layer_subspace_to_device_dtype():
    sub = fit_layer_subspace(_circle_centroids(5, dim=10), cyclic=True)
    moved = sub.to(device=torch.device("cpu"), dtype=torch.float64)
    assert moved.basis.dtype == torch.float64
    assert moved.mean.dtype == torch.float64
