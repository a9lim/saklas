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
    decompose,
    domain_from_spec,
    eval_rbf,
    eval_rbf_jacobian,
    fit_layer_subspace as _fit_layer_subspace_with_ev,  # returns (LayerSubspace, ev_ratio)
    fit_rbf_interpolant,
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

        def subspace_gram(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            self.subspace_gram_calls += 1
            return self.inner.subspace_gram(*args, **kwargs)

        def apply_inv(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            return self.inner.apply_inv(*args, **kwargs)

    counting = CountingWhitener(base)
    sub, mu_coords, ev = fit_affine_subspace(
        centroids,
        neutral_mean=means[0],
        n_components=3,
        whitener=counting,  # type: ignore[arg-type]
        layer=0,
        whitened_gram=gram,
    )
    ref_sub, ref_mu_coords, ref_ev = fit_affine_subspace(
        centroids,
        neutral_mean=means[0],
        n_components=3,
        whitener=base,
        layer=0,
    )

    assert counting.subspace_gram_calls == 0
    assert torch.allclose(sub.basis, ref_sub.basis, atol=1e-5)
    assert torch.allclose(mu_coords, ref_mu_coords, atol=1e-5)
    assert ev == pytest.approx(ref_ev, abs=1e-6)


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


def test_load_manifold_without_share_fields_defaults_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit with no whitener (no ``mahalanobis_share`` / ``share_metric``)
    round-trips with an empty share dict and no crash — the apply path
    then falls back to the Euclidean spread."""
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
    save_manifold(manifold, path, {"method": "manifold_pca",
                                   "nodes_sha256": "abc"})
    loaded = load_manifold(path)
    assert loaded.mahalanobis_share == {}
    assert "share_metric" not in loaded.metadata
    assert "mahalanobis_share_per_layer" not in loaded.metadata
    # A manifold saved with ``origin=None`` round-trips with absence
    # preserved — no ``origin`` tensor written, loads back as ``None``.
    assert loaded.origin == {}


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
    manifold = Manifold(
        name="folded",
        domain=CustomDomain(1),
        node_labels=["pos", "neg"],
        node_coords=torch.tensor([[0.8], [-0.8]]),
        layers={5: sub_a, 11: sub_b},
        feature_space="raw",
        mahalanobis_share={5: 1.2, 11: 0.7},
        origin={5: torch.tensor([0.0]), 11: torch.tensor([0.0])},
    )
    path = tmp_path / "folded" / "model.safetensors"
    save_manifold(manifold, path, {"method": "folded_vector",
                                   "share_metric": "mahalanobis"})

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
