"""Folded-vector extraction — a steering vector as a flat affine subspace.

Tests the pure geometry core (saklas 4.0 §5): the ``K = 2`` affine fit —
PCA@2 ≡ DiM basis, **neutral-anchored** ``mean = P_basis(ν)``, **real**
per-layer node coords ``(c± − ν)·δ̂`` (the per-layer ‖δ‖ now lives in the
coords, not the share), the μ-centered budget share, per-axis DLS at R=1, and
the affine save/load round-trip (now carrying per-layer ``node_coords``).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import saklas.core.vectors as V
from saklas.core.mahalanobis import LayerWhitener
from saklas.core.manifold import load_manifold, save_manifold
from saklas.core.vectors import (
    _fold_centroids_to_affine_manifold,
    extract_difference_of_means,
    fold_directions_to_subspace,
    fold_vector_to_subspace,
    folded_vector_directions,
)

_SQRT2 = math.sqrt(2.0)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def _p_basis(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Projection of ``vec`` into the span of ``basis`` rows."""
    return (vec @ basis.T) @ basis


# --------------------------------------------------------------- geometry ---

def test_fold_pca2_is_dim_basis():
    """PCA@2 ≡ DiM: the μ-centered fit's sole axis is ``unit(pos − neg)``,
    oriented pos-ward (node 0 = pos)."""
    torch.manual_seed(0)
    d = 8
    d0 = _unit(torch.randn(d))
    d1 = _unit(torch.randn(d))
    mid0, mid1 = 5.0 + torch.randn(d), 3.0 + torch.randn(d)
    pos = {0: mid0 + 1.5 * d0, 1: mid1 + 0.4 * d1}
    neg = {0: mid0 - 1.5 * d0, 1: mid1 - 0.4 * d1}

    mfld = _fold_centroids_to_affine_manifold(
        "happy.sad", pos, neg, pos_label="happy", neg_label="sad",
    )

    assert mfld.name == "happy.sad"
    assert mfld.node_labels == ["happy", "sad"]
    assert mfld.domain.intrinsic_dim == 1
    # Shared display layout stays canonical ±1 (the real geometry is per-layer).
    assert torch.allclose(mfld.node_coords, torch.tensor([[1.0], [-1.0]]))
    assert sorted(mfld.layers) == [0, 1]
    assert mfld.metadata["share_metric"] == "euclidean"

    for L, dir_unit in ((0, d0), (1, d1)):
        sub = mfld.layers[L]
        assert sub.is_affine and sub.rank == 1
        # basis is the *raw* δ̂, oriented pos-ward (node 0 = pos).
        assert torch.allclose(sub.basis.reshape(-1), dir_unit, atol=1e-5)


def test_fold_neutral_anchored_real_coords():
    """With a neutral baseline: ``mean = P_basis(ν)``, node coords are the
    real ``(c± − ν)·δ̂``, neutral → coord 0, and ``coord_+ − coord_- = ‖δ‖``."""
    d = 8
    d0 = _unit(torch.randn(d))
    pos = {0: 2.0 * d0}
    neg = {0: -2.0 * d0}                       # δ = 4·d0, ‖δ‖ = 4
    neutral = {0: 0.5 * d0 + torch.randn(d)}   # off-line neutral

    mfld = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n", layer_means=neutral,
    )
    sub = mfld.layers[0]
    nu = neutral[0]
    dhat = sub.basis.reshape(-1)

    # mean = projection of neutral into the span (off-span part dropped).
    assert torch.allclose(sub.mean, _p_basis(nu, sub.basis), atol=1e-5)
    # Real per-layer coords = (c − ν)·δ̂.
    nc = sub.node_coords.reshape(-1)
    assert nc[0].item() == pytest.approx(float((pos[0] - nu) @ dhat), abs=1e-4)
    assert nc[1].item() == pytest.approx(float((neg[0] - nu) @ dhat), abs=1e-4)
    # Neutral → coord 0; coord_+ − coord_- = ‖δ‖.
    assert float((nu - nu) @ dhat) == pytest.approx(0.0, abs=1e-6)
    assert float(nc[0] - nc[1]) == pytest.approx(4.0, abs=1e-4)
    # eval_at(real coord) reconstructs the in-span projection of the centroid.
    assert torch.allclose(
        sub.eval_at(sub.node_coords[0]), _p_basis(pos[0], sub.basis), atol=1e-4,
    )
    # Affine ⇒ no stored origin (the neutral foot is coord 0 by construction).
    assert mfld.origin == {}


def test_fold_no_neutral_anchors_at_midpoint():
    """No neutral ⇒ anchor at the centroid mean μ (= the midpoint at K=2):
    ``mean = P_basis(μ)``, coords = ±‖δ‖/2."""
    d = 6
    d0 = _unit(torch.randn(d))
    mid = 4.0 + torch.randn(d)
    pos = {0: mid + 1.0 * d0}
    neg = {0: mid - 1.0 * d0}                  # ‖δ‖ = 2
    mfld = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n",
    )
    sub = mfld.layers[0]
    assert torch.allclose(sub.mean, _p_basis(mid, sub.basis), atol=1e-5)
    nc = sub.node_coords.reshape(-1)
    assert nc[0].item() == pytest.approx(1.0, abs=1e-4)    # +‖δ‖/2
    assert nc[1].item() == pytest.approx(-1.0, abs=1e-4)


def test_fold_euclidean_share_is_mu_centered_spread():
    """No whitener ⇒ share = ‖δ_L‖₂/√2 (the μ-centered spread); *normalized*
    it is exactly today's DiM Euclidean hook share (the √2 cancels)."""
    d = 6
    d0, d1 = _unit(torch.randn(d)), _unit(torch.randn(d))
    pos = {0: 2.0 * d0, 1: 0.5 * d1}      # ‖δ_0‖ = 4, ‖δ_1‖ = 1
    neg = {0: -2.0 * d0, 1: -0.5 * d1}
    mfld = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n",
    )
    assert mfld.metadata["share_metric"] == "euclidean"
    assert mfld.mahalanobis_share[0] == pytest.approx(4.0 / _SQRT2, abs=1e-4)
    assert mfld.mahalanobis_share[1] == pytest.approx(1.0 / _SQRT2, abs=1e-4)
    # normalized share == normalize({L: ‖δ_L‖₂}) — the DiM hook-share profile
    total = sum(mfld.mahalanobis_share.values())
    assert mfld.mahalanobis_share[0] / total == pytest.approx(0.8, abs=1e-4)


def test_fold_whitened_share_proportional_to_mahalanobis_norm():
    """With a covering whitener, share = ‖δ_L‖_M/√2 (the μ-centered whitened
    spread), no stored origin (affine)."""
    d = 10
    layers = (0, 1)
    g = torch.Generator().manual_seed(7)
    acts = {L: torch.randn(60, d, generator=g) * (1.0 + 0.5 * L)
            for L in layers}
    means = {L: torch.zeros(d) for L in layers}
    whitener = LayerWhitener.from_neutral_activations(acts, means)
    assert whitener.covers_all(layers)

    d0, d1 = _unit(torch.randn(d)), _unit(torch.randn(d))
    pos = {0: 1.2 * d0, 1: 0.9 * d1}
    neg = {0: -1.2 * d0, 1: -0.9 * d1}
    delta = {0: pos[0] - neg[0], 1: pos[1] - neg[1]}

    mfld = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n",
        whitener=whitener, layer_means=means,
    )
    assert mfld.metadata["share_metric"] == "mahalanobis"
    assert mfld.origin == {}     # affine: no stored origin (coord 0)
    for L in layers:
        assert mfld.mahalanobis_share[L] == pytest.approx(
            whitener.mahalanobis_norm(L, delta[L]) / _SQRT2, abs=1e-4,
        )


def test_fold_per_axis_dls_drops_non_discriminative_layer():
    """At R=1, per-axis DLS keeps a layer iff (pos−neut)·d̂ and (neg−neut)·d̂
    straddle zero — same as the historical scalar DLS."""
    d = 5
    dir_k = _unit(torch.randn(d))
    dir_x = _unit(torch.randn(d))
    # layer 0: neutral between the poles ⇒ straddle ⇒ KEEP
    mid0 = torch.zeros(d)
    pos = {0: mid0 + 2.0 * dir_k}
    neg = {0: mid0 - 2.0 * dir_k}
    neut = {0: mid0}
    # layer 1: both poles on the same side of neutral ⇒ no straddle ⇒ DROP
    pos[1] = 3.0 * dir_x
    neg[1] = 1.0 * dir_x            # δ_1 = 2·dir_x ≠ 0, but both above neutral
    neut[1] = torch.zeros(d)

    mfld = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n", layer_means=neut,
    )
    assert sorted(mfld.layers) == [0]      # layer 1 dropped by DLS

    mfld_all = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n",
        layer_means=neut, dls=False,
    )
    assert sorted(mfld_all.layers) == [0, 1]


def test_fold_degenerate_pair_drops_layer():
    d = 4
    pos = {0: _unit(torch.randn(d)), 1: torch.ones(d)}
    neg = {0: -pos[0], 1: torch.ones(d)}   # layer 1: δ = 0
    mfld = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n", dls=False,
    )
    assert sorted(mfld.layers) == [0]


def test_folded_vector_directions_view():
    """The Profile-view = {L: δ̂_L · share_L}; per-layer cosine against the
    underlying δ̂ is ±1 (direction preserved), magnitude tracks the share.
    The cross-layer *ratio* (4:1) survives the √2 share rescale."""
    d = 8
    d0, d1 = _unit(torch.randn(d)), _unit(torch.randn(d))
    pos = {0: 2.0 * d0, 1: 0.5 * d1}      # ‖δ_0‖ = 4, ‖δ_1‖ = 1
    neg = {0: -2.0 * d0, 1: -0.5 * d1}
    mfld = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n",
    )
    dirs = folded_vector_directions(mfld)
    assert sorted(dirs) == [0, 1]
    for L, unit in ((0, d0), (1, d1)):
        v = dirs[L]
        # direction preserved (cosine ±1), magnitude = the (μ-centered) share
        assert torch.allclose(v / v.norm(), unit, atol=1e-5)
        assert float(v.norm()) == pytest.approx(mfld.mahalanobis_share[L], abs=1e-4)
    # ratio of layer magnitudes == ratio of ‖δ‖ (4:1) — the per-layer profile
    assert float(dirs[0].norm() / dirs[1].norm()) == pytest.approx(4.0, abs=1e-3)


def test_folded_vector_directions_rejects_curved():
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
    )
    with pytest.raises(ValueError, match="affine"):
        folded_vector_directions(mfld)


def test_fold_directions_to_subspace_neutral_anchored():
    """A derived direction folds to a one-pole ray: basis = d̂, mean =
    P_basis(ν), real pole coord = ‖d‖, share = ‖d‖, single + node, no
    stored origin."""
    d = 8
    dir0, dir1 = torch.randn(d), torch.randn(d)
    directions = {2: dir0, 5: dir1}
    neutral = {2: 10.0 + torch.randn(d), 5: 3.0 + torch.randn(d)}
    mfld = fold_directions_to_subspace(
        "merged", directions, neutral, label="merged",
    )
    assert mfld.node_labels == ["merged"]
    assert torch.allclose(mfld.node_coords, torch.tensor([[1.0]]))  # display layout
    assert mfld.metadata["share_metric"] == "euclidean"
    assert mfld.origin == {}
    for L, raw in ((2, dir0), (5, dir1)):
        sub = mfld.layers[L]
        assert sub.is_affine and sub.rank == 1
        assert torch.allclose(sub.basis.reshape(-1), raw / raw.norm(), atol=1e-5)
        # neutral-anchored: mean = P_basis(ν), not the raw neutral vector
        assert torch.allclose(sub.mean, _p_basis(neutral[L], sub.basis), atol=1e-5)
        assert mfld.mahalanobis_share[L] == pytest.approx(float(raw.norm()), abs=1e-4)
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


def test_fold_directions_drops_zero_and_anchors_at_origin_without_neutral():
    d = 5
    directions = {0: torch.randn(d), 1: torch.zeros(d)}   # layer 1 degenerate
    mfld = fold_directions_to_subspace("m", directions, None)  # no neutral
    assert sorted(mfld.layers) == [0]
    assert torch.allclose(mfld.layers[0].mean, torch.zeros(d))    # P_basis(∅) = 0


def test_fold_share_matches_dim_bake_exactly(monkeypatch: pytest.MonkeyPatch):
    """R=1 parity gate: feed *identical* centroids to ``fold_vector_to_subspace``
    and the production ``extract_difference_of_means``; their *normalized*
    per-layer hook shares match bit-for-bit.  The folded μ-centered share is
    ``‖δ_L‖_M/√2`` — proportional to the DiM bake's ``‖δ_L‖_M``, so the √2 (and
    ``ref_norm``) both cancel through the normalization."""
    d = 12
    layers = (0, 1, 2)
    g = torch.Generator().manual_seed(11)
    acts = {L: torch.randn(80, d, generator=g) * (1.0 + 0.3 * L) for L in layers}
    means = {L: torch.zeros(d) for L in layers}
    whitener = LayerWhitener.from_neutral_activations(acts, means)

    dirs = {L: _unit(torch.randn(d, generator=g)) for L in layers}
    scales = {0: 1.7, 1: 0.6, 2: 1.1}
    mean_pos = {L: scales[L] * dirs[L] for L in layers}
    mean_neg = {L: -scales[L] * dirs[L] for L in layers}
    mean_diffs = {L: mean_pos[L] - mean_neg[L] for L in layers}
    n_pairs = 6
    norm_sums = {L: float(n_pairs * 2) * (5.0 + L) for L in layers}  # ref_norm varies
    diag = {L: [] for L in layers}

    def _fake_capture(*_args: object, **_kwargs: object):
        return (len(layers), dict(mean_diffs), dict(norm_sums),
                dict(mean_pos), dict(mean_neg), dict(diag))

    monkeypatch.setattr(V, "_capture_dim_stats_for_pairs", _fake_capture)

    dummy_model = torch.nn.Module()       # unused (capture is mocked)
    dummy_pairs = [{"positive": "p", "negative": "n"}] * n_pairs
    dev = torch.device("cpu")
    # DiM path: hook share = ‖baked_L‖ normalized.
    profile, _ = extract_difference_of_means(
        dummy_model, None, dummy_pairs, torch.nn.ModuleList(), dev,
        whitener=whitener, layer_means=means, dls=False,
    )
    baked = {L: float(profile[L].norm()) for L in layers}
    bt = sum(baked.values())
    dim_share = {L: baked[L] / bt for L in layers}

    # Folded path: normalized mahalanobis_share.
    mfld = fold_vector_to_subspace(
        dummy_model, None, dummy_pairs, torch.nn.ModuleList(), dev,
        concept_label="c", pos_label="p", neg_label="n",
        whitener=whitener, layer_means=means, dls=False,
    )
    st = sum(mfld.mahalanobis_share.values())
    fold_share = {L: mfld.mahalanobis_share[L] / st for L in layers}

    for L in layers:
        assert fold_share[L] == pytest.approx(dim_share[L], abs=1e-5)


def test_fold_round_trips_through_save_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A folded vector persists and reloads as an affine subspace, with the
    per-layer real ``node_coords`` round-tripped alongside basis + mean +
    share."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    d = 7
    d0, d1 = _unit(torch.randn(d)), _unit(torch.randn(d))
    pos = {2: 5.0 + 1.1 * d0, 5: 3.0 + 0.6 * d1}
    neg = {2: 5.0 - 1.1 * d0, 5: 3.0 - 0.6 * d1}
    neutral = {2: 5.0 + 0.2 * d0, 5: 3.0 - 0.1 * d1}
    mfld = _fold_centroids_to_affine_manifold(
        "happy.sad", pos, neg, pos_label="happy", neg_label="sad",
        layer_means=neutral,
    )
    path = tmp_path / "happy.sad" / "model.safetensors"
    save_manifold(mfld, path, {"method": "folded_vector",
                               "share_metric": mfld.metadata["share_metric"]})
    loaded = load_manifold(path)
    assert sorted(loaded.layers) == [2, 5]
    assert loaded.mahalanobis_share == pytest.approx(mfld.mahalanobis_share)
    for L in (2, 5):
        lsub, osub = loaded.layers[L], mfld.layers[L]
        assert lsub.is_affine
        assert torch.allclose(lsub.basis, osub.basis)
        assert torch.allclose(lsub.mean, osub.mean)
        assert lsub.node_coords is not None
        assert torch.allclose(lsub.node_coords, osub.node_coords)
