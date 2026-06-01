"""Folded-vector extraction — a steering vector as an affine R=1 manifold.

Tests the pure geometry core ``_fold_centroids_to_affine_manifold`` (saklas
4.0 Phase 2 §1): basis = raw δ̂, midpoint mean, unit pole coords, the
exact-``R=1``-parity baked share, the neutral-foot origin, per-axis DLS at
R=1, and the affine save/load round-trip (commit 2.3).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import saklas.core.vectors as V
from saklas.core.mahalanobis import LayerWhitener
from saklas.core.manifold import load_manifold, save_manifold
from saklas.core.vectors import (
    _fold_centroids_to_affine_manifold,
    extract_difference_of_means,
    fold_vector_to_subspace,
    folded_vector_directions,
)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


# --------------------------------------------------------------- geometry ---

def test_fold_basic_affine_structure():
    torch.manual_seed(0)
    d = 8
    # Two layers with distinct directions + magnitudes.
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
    assert torch.allclose(mfld.node_coords, torch.tensor([[1.0], [-1.0]]))
    assert sorted(mfld.layers) == [0, 1]
    assert mfld.metadata["share_metric"] == "euclidean"

    for L, dir_unit, mid in ((0, d0, mid0), (1, d1, mid1)):
        sub = mfld.layers[L]
        assert sub.is_affine
        assert sub.rank == 1
        # basis is the *raw* δ̂ (sign: pos − neg points pos-ward, +1 coord)
        assert torch.allclose(sub.basis.reshape(-1), dir_unit, atol=1e-5)
        assert torch.allclose(sub.mean, mid, atol=1e-5)
        # +1 coord lands a unit step toward pos, −1 toward neg, 0 at midpoint
        assert torch.allclose(mfld.manifold_point(L, (1.0,)), mid + dir_unit, atol=1e-4)
        assert torch.allclose(mfld.manifold_point(L, (-1.0,)), mid - dir_unit, atol=1e-4)
        assert torch.allclose(mfld.manifold_point(L, (0.0,)), mid, atol=1e-4)


def test_fold_euclidean_share_is_delta_norm():
    """No whitener ⇒ share = ‖δ_L‖₂, which (normalized) is exactly today's
    DiM Euclidean hook share."""
    d = 6
    d0, d1 = _unit(torch.randn(d)), _unit(torch.randn(d))
    pos = {0: 2.0 * d0, 1: 0.5 * d1}      # ‖δ_0‖ = 4, ‖δ_1‖ = 1
    neg = {0: -2.0 * d0, 1: -0.5 * d1}
    mfld = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n",
    )
    assert mfld.metadata["share_metric"] == "euclidean"
    assert mfld.mahalanobis_share[0] == pytest.approx(4.0, abs=1e-4)
    assert mfld.mahalanobis_share[1] == pytest.approx(1.0, abs=1e-4)
    # normalized share == normalize({L: ‖δ_L‖₂}) — the DiM hook-share formula
    total = sum(mfld.mahalanobis_share.values())
    assert mfld.mahalanobis_share[0] / total == pytest.approx(0.8, abs=1e-4)


def test_fold_whitened_share_equals_mahalanobis_norm():
    """With a covering whitener, share = ‖δ_L‖_M exactly (the per-layer
    scalar today's DiM Mahalanobis bake uses) + lever/origin populated."""
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
    for L in layers:
        assert mfld.mahalanobis_share[L] == pytest.approx(
            whitener.mahalanobis_norm(L, delta[L]), abs=1e-4,
        )
        # lever ∈ (0, 1] and origin populated (neutral mean is 0 ⇒ foot ≈ 0)
        assert 0.0 < mfld.lever[L] <= 1.0
        assert L in mfld.origin
        assert mfld.origin[L].item() == pytest.approx(0.0, abs=1e-4)


def test_fold_origin_is_neutral_foot():
    """O_L = (μ_L − midpoint)·δ̂_L — the projection of neutral onto the line.
    Off-center neutral lands the foot off the midpoint."""
    d = 6
    dir0 = _unit(torch.randn(d))
    mid = 4.0 + torch.randn(d)
    pos = {0: mid + dir0}
    neg = {0: mid - dir0}
    # neutral sits 0.7 units toward pos along the line
    neutral = {0: mid + 0.7 * dir0}
    mfld = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n", layer_means=neutral,
    )
    assert mfld.origin[0].item() == pytest.approx(0.7, abs=1e-4)


def test_fold_per_axis_dls_drops_non_discriminative_layer():
    """At R=1, per-axis DLS keeps a layer iff (pos−neut)·d̂ and (neg−neut)·d̂
    have opposite signs — same as today's scalar DLS."""
    d = 5
    dir_k = _unit(torch.randn(d))
    dir_x = _unit(torch.randn(d))
    # layer 0: neutral between the poles ⇒ opposite signs ⇒ KEEP
    mid0 = torch.zeros(d)
    pos = {0: mid0 + 2.0 * dir_k}
    neg = {0: mid0 - 2.0 * dir_k}
    neut = {0: mid0}
    # layer 1: both poles on the same side of neutral ⇒ same sign ⇒ DROP
    pos[1] = 3.0 * dir_x
    neg[1] = 1.0 * dir_x            # δ_1 = 2·dir_x ≠ 0, but both above neutral
    neut[1] = torch.zeros(d)

    mfld = _fold_centroids_to_affine_manifold(
        "c", pos, neg, pos_label="p", neg_label="n", layer_means=neut,
    )
    assert sorted(mfld.layers) == [0]      # layer 1 dropped by DLS

    # dls=False keeps both
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
    underlying δ̂ is ±1 (direction preserved), magnitude tracks the share."""
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
        # direction preserved (cosine ±1), magnitude = share = ‖δ_L‖₂
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


def test_fold_share_matches_dim_bake_exactly(monkeypatch: pytest.MonkeyPatch):
    """R=1 parity gate: feed *identical* centroids to ``fold_vector_to_subspace``
    and the production ``extract_difference_of_means``; their normalized
    per-layer hook shares must match bit-for-bit (the folded share ``‖δ_L‖_M``
    *is* the DiM hook share once ``ref_norm`` cancels through the bake)."""
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
    """A folded vector persists and reloads as an affine manifold (commit
    2.3 save/load), share + basis preserved."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    d = 7
    d0, d1 = _unit(torch.randn(d)), _unit(torch.randn(d))
    pos = {2: 5.0 + 1.1 * d0, 5: 3.0 + 0.6 * d1}
    neg = {2: 5.0 - 1.1 * d0, 5: 3.0 - 0.6 * d1}
    mfld = _fold_centroids_to_affine_manifold(
        "happy.sad", pos, neg, pos_label="happy", neg_label="sad",
    )
    path = tmp_path / "happy.sad" / "model.safetensors"
    save_manifold(mfld, path, {"method": "folded_vector",
                               "share_metric": mfld.metadata["share_metric"]})
    loaded = load_manifold(path)
    assert sorted(loaded.layers) == [2, 5]
    assert loaded.mahalanobis_share == pytest.approx(mfld.mahalanobis_share)
    for L in (2, 5):
        assert loaded.layers[L].is_affine
        assert torch.allclose(loaded.layers[L].basis, mfld.layers[L].basis)
        assert torch.allclose(loaded.layers[L].mean, mfld.layers[L].mean)
