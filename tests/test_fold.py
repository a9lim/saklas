"""Folded-vector geometry — a derived direction as a flat affine subspace.

Tests the production fold path (saklas 4.0 §5): ``fold_directions_to_subspace``
folds an arbitrary per-layer direction into a neutral-anchored one-pole ray —
basis = d̂, **neutral-anchored** ``mean = P_basis(ν)``, **real** per-layer pole
coord ``‖d‖``, the direction-magnitude share — and ``folded_directions``
is its reverse view ``{L: δ̂_L · share_L}`` (refusing a curved manifold).

The legacy bipolar-centroid fold (``_fold_centroids_to_affine_manifold``) was
retired in the Mahalanobis-only collapse; the live 2-node-vector read goes
through ``extraction.py``'s whitened ``fit_affine_subspace``.
"""

from __future__ import annotations

import torch
import pytest

from saklas.core.capture import (
    fold_directions_to_subspace,
    folded_directions,
    is_foldable_vector_manifold,
)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def _p_basis(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Projection of ``vec`` into the span of ``basis`` rows."""
    return (vec @ basis.T) @ basis


def _whitener(layers: list[int], dim: int):
    from tests._whitener import isotropic_whitener
    return isotropic_whitener(layers, dim)


# --------------------------------------------------------------- geometry ---

def test_folded_directions_rejects_curved():
    """A curved (RBF-fitted, non-affine) manifold has no single direction —
    the view must refuse it."""
    from saklas.core.manifold import (
        BoxAxis, BoxDomain, Manifold, fit_layer_subspace,
    )
    torch.manual_seed(0)
    K, dim = 5, 10
    centroids = torch.randn(K, dim)
    node_params = torch.linspace(0.0, 1.0, K).reshape(K, 1)
    curved, _ev = fit_layer_subspace(centroids, node_params)
    assert not curved.is_affine
    mfld = Manifold(
        name="m",
        domain=BoxDomain([BoxAxis("t", periodic=False, lo=0.0, hi=1.0)]),
        node_labels=[f"n{i}" for i in range(K)],
        node_coords=node_params,
        layers={0: curved},
        mahalanobis_share={0: 1.0},
        origin={0: torch.zeros(1)},
        feature_space="sae-test",
    )
    with pytest.raises(ValueError, match="affine"):
        folded_directions(mfld)
    # ...and the foldability predicate agrees, so analytics skip it.
    assert not is_foldable_vector_manifold(mfld)


def test_is_foldable_vector_manifold_rejects_multinode_affine():
    """A flat *rank-R>1* fit (the ``personas`` fan shape) is affine but has no
    single direction — the predicate must reject it so the direction-cosine
    analytics (correlation / pairwise) skip it instead of folding to a 500."""
    from saklas.core.manifold import (
        CustomDomain, LayerSubspace, Manifold,
    )
    torch.manual_seed(0)
    K, R, dim = 4, 3, 8           # 4 nodes, rank-3 affine subspace
    basis, _ = torch.linalg.qr(torch.randn(dim, R))
    basis = basis.T.contiguous()  # (R, D), orthonormal rows
    mean = torch.zeros(dim)
    node_coords = torch.randn(K, R)
    sub = LayerSubspace.affine(mean, basis, node_coords=node_coords)
    assert sub.is_affine and sub.rank == R
    mfld = Manifold(
        name="fan",
        domain=CustomDomain(R),
        node_labels=[f"n{i}" for i in range(K)],
        node_coords=node_coords,
        layers={0: sub},
        mahalanobis_share={0: 1.0},
    )
    assert not is_foldable_vector_manifold(mfld)
    with pytest.raises(ValueError, match="affine R=1"):
        folded_directions(mfld)


def test_is_foldable_vector_manifold_accepts_r1_and_rejects_empty():
    """The R=1 fold output is foldable; a layerless manifold is not."""
    from saklas.core.manifold import CustomDomain, Manifold

    d = 6
    directions = {0: torch.randn(d), 3: torch.randn(d)}
    mfld = fold_directions_to_subspace(
        "v", directions, {0: torch.zeros(d), 3: torch.zeros(d)},
        whitener=_whitener([0, 3], d),
    )
    assert is_foldable_vector_manifold(mfld)
    empty = Manifold(
        name="empty", domain=CustomDomain(1),
        node_labels=["x"], node_coords=torch.tensor([[1.0]]), layers={},
    )
    assert not is_foldable_vector_manifold(empty)


def test_fold_directions_to_subspace_neutral_anchored():
    """A derived direction folds to a one-pole ray: basis = d̂, mean =
    P_basis(ν), real pole coord = ‖d‖, share = ‖d‖, single + node, no
    stored origin."""
    d = 8
    dir0, dir1 = torch.randn(d), torch.randn(d)
    directions = {2: dir0, 5: dir1}
    neutral = {2: 10.0 + torch.randn(d), 5: 3.0 + torch.randn(d)}
    whitener = _whitener([2, 5], d)
    mfld = fold_directions_to_subspace(
        "merged", directions, neutral,
        whitener=whitener, label="merged",
    )
    assert mfld.node_labels == ["merged"]
    assert torch.allclose(mfld.node_coords, torch.tensor([[1.0]]))  # display layout
    assert mfld.metadata["share_metric"] == "mahalanobis"
    assert mfld.origin == {}
    for L, raw in ((2, dir0), (5, dir1)):
        sub = mfld.layers[L]
        assert sub.is_affine and sub.rank == 1
        assert torch.allclose(sub.basis.reshape(-1), raw / raw.norm(), atol=1e-5)
        # neutral-anchored: mean = P_basis(ν), not the raw neutral vector
        assert torch.allclose(sub.mean, _p_basis(neutral[L], sub.basis), atol=1e-5)
        assert mfld.mahalanobis_share[L] == pytest.approx(
            whitener.mahalanobis_norm(L, raw), abs=1e-4,
        )
        # real per-layer pole coord = ‖d‖ (a step of ‖d‖ along d̂ from origin)
        assert sub.node_coords.reshape(-1).item() == pytest.approx(
            float(raw.norm()), abs=1e-4,
        )
    # the pole eval lands at the in-span projection of ν + d
    sub2 = mfld.layers[2]
    assert torch.allclose(
        sub2.eval_at(sub2.node_coords[0]),
        _p_basis(neutral[2] + dir0, sub2.basis), atol=1e-4,
    )


def test_fold_directions_rejects_missing_neutral_mean():
    d = 5
    directions = {0: torch.randn(d), 1: torch.zeros(d)}   # layer 1 degenerate
    with pytest.raises(ValueError, match="neutral means"):
        fold_directions_to_subspace(
            "m", directions, {}, whitener=_whitener([0, 1], d),
        )


def test_fold_directions_rejects_all_zero_profile():
    directions = {3: torch.zeros(4)}
    with pytest.raises(ValueError, match="only zero vectors"):
        fold_directions_to_subspace(
            "zero", directions, {3: torch.zeros(4)},
            whitener=_whitener([3], 4),
        )
