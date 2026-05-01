"""Cross-model probe alignment via per-layer Procrustes (v1.6).

Covers ``fit_alignment`` (matched + mismatched dim, layer-count
mismatch), ``transfer_profile`` (provenance + dropping uncovered
layers), ``alignment_quality`` (R² metric), and the disk cache shape.

CPU-only, no model load — alignment math runs over synthetic
activation tensors.
"""
from __future__ import annotations

import json

import pytest
import torch

from saklas.core.profile import Profile, ProfileError
from saklas.io.alignment import (
    AlignmentError,
    alignment_cache_path,
    alignment_quality,
    fit_alignment,
    load_alignment_map,
    save_alignment_map,
    transfer_profile,
)


def _random_activations(
    n_layers: int, n_prompts: int, dim: int, *, seed: int = 0,
) -> dict[int, torch.Tensor]:
    """Synthetic ``[N, D]`` activation stack per layer."""
    g = torch.Generator().manual_seed(seed)
    return {layer: torch.randn(n_prompts, dim, generator=g) for layer in range(n_layers)}


# ---------------------------------------------------------------------------
# fit_alignment.
# ---------------------------------------------------------------------------


class TestFitAlignment:
    def test_same_dim_orthogonal_procrustes_recovers_rotation(self) -> None:
        # Generate a known rotation, apply it, fit alignment — should
        # recover the rotation up to numerical error.
        torch.manual_seed(0)
        D = 8
        N = 30
        # Random orthogonal rotation via QR.
        A = torch.randn(D, D)
        Q, _ = torch.linalg.qr(A)
        X_src = torch.randn(N, D)
        X_tgt = X_src @ Q.transpose(0, 1)  # X_tgt = X_src @ Q^T

        M = fit_alignment(
            {0: X_src, 1: X_src + 0.01 * torch.randn(N, D)},
            {0: X_tgt, 1: X_tgt + 0.01 * torch.randn(N, D)},
            min_shared_layers=2,
        )

        # M[0] should approximately equal Q (the orthogonal rotation).
        assert torch.allclose(M[0], Q, atol=1e-2)

    def test_mismatched_dim_uses_least_squares(self) -> None:
        # D_src = 8, D_tgt = 4.  Procrustes can't operate on different
        # dims; the fit should fall back to least-squares.
        torch.manual_seed(1)
        N = 30
        D_src, D_tgt = 8, 4
        X_src = torch.randn(N, D_src)
        # Synthesize a known rectangular linear relation.
        A = torch.randn(D_src, D_tgt)
        X_tgt = X_src @ A

        M = fit_alignment(
            {layer: X_src for layer in range(12)},
            {layer: X_tgt for layer in range(12)},
        )

        # M is shaped (D_tgt, D_src) so v_tgt = M @ v_src.
        assert M[0].shape == (D_tgt, D_src)
        # Sanity: predict X_tgt and check residual is small.
        X_pred = (X_src - X_src.mean(dim=0)) @ M[0].transpose(0, 1)
        X_tgt_c = X_tgt - X_tgt.mean(dim=0)
        residual = (X_tgt_c - X_pred).norm().item()
        baseline = X_tgt_c.norm().item()
        # Linear relationship is exact up to centering; residual should
        # be a tiny fraction of the baseline.
        assert residual / baseline < 0.05

    def test_insufficient_shared_layers_raises(self) -> None:
        src = _random_activations(n_layers=5, n_prompts=10, dim=4)
        tgt = _random_activations(n_layers=3, n_prompts=10, dim=4, seed=1)
        # Shared layers: {0, 1, 2}, 3 < default min=10.
        with pytest.raises(AlignmentError, match="shared layers"):
            fit_alignment(src, tgt)

    def test_mismatched_n_prompts_raises(self) -> None:
        src = {0: torch.randn(20, 4)}
        tgt = {0: torch.randn(15, 4)}
        with pytest.raises(AlignmentError, match="N=20.*N=15|matched"):
            fit_alignment(src, tgt, min_shared_layers=1)

    def test_partial_layer_overlap_uses_intersection(self) -> None:
        src = _random_activations(n_layers=15, n_prompts=20, dim=4)
        tgt = {
            layer: torch.randn(20, 4) for layer in range(5, 20)  # overlap is layers 5-14
        }
        M = fit_alignment(src, tgt, min_shared_layers=10)
        assert sorted(M.keys()) == list(range(5, 15))


# ---------------------------------------------------------------------------
# transfer_profile.
# ---------------------------------------------------------------------------


class TestTransferProfile:
    def test_applies_per_layer_map_and_records_provenance(self) -> None:
        torch.manual_seed(0)
        # Identity alignment — transferred values should equal source.
        D = 4
        eye = torch.eye(D)
        M = {0: eye, 2: eye, 5: eye}

        src_profile = Profile(
            {0: torch.tensor([1.0, 0.0, 0.0, 0.0]),
             2: torch.tensor([0.5, 0.5, 0.0, 0.0]),
             5: torch.tensor([0.0, 0.0, 1.0, 0.0])},
            metadata={"method": "contrastive_pca"},
        )

        transferred = transfer_profile(
            src_profile, M,
            source_model_id="google/gemma-3-4b-it",
            transfer_quality_estimate=0.85,
        )

        assert transferred.layers == [0, 2, 5]
        for layer in transferred.layers:
            assert torch.allclose(transferred[layer], src_profile[layer], atol=1e-5)

        # Provenance survives the wrap.
        assert transferred.metadata["method"] == "procrustes_transfer"
        assert transferred.metadata["source_model_id"] == "google/gemma-3-4b-it"
        assert transferred.metadata["transfer_quality_estimate"] == pytest.approx(0.85)

    def test_drops_uncovered_layers(self) -> None:
        D = 4
        M = {0: torch.eye(D), 5: torch.eye(D)}  # Layer 2 not covered.
        src_profile = Profile(
            {0: torch.ones(D), 2: torch.ones(D), 5: torch.ones(D)},
            metadata={"method": "contrastive_pca"},
        )
        transferred = transfer_profile(
            src_profile, M, source_model_id="src/model",
        )
        assert transferred.layers == [0, 5]

    def test_empty_alignment_raises(self) -> None:
        src_profile = Profile({0: torch.ones(4)}, metadata={})
        with pytest.raises(ProfileError, match="empty"):
            transfer_profile(src_profile, {}, source_model_id="src/model")

    def test_fully_disjoint_layers_raises(self) -> None:
        src_profile = Profile({0: torch.ones(4)}, metadata={})
        # Alignment covers layers 5-9; profile is only at layer 0.
        M = {layer: torch.eye(4) for layer in range(5, 10)}
        with pytest.raises(ProfileError, match="covered no layers"):
            transfer_profile(src_profile, M, source_model_id="src/model")


# ---------------------------------------------------------------------------
# alignment_quality.
# ---------------------------------------------------------------------------


class TestAlignmentQuality:
    def test_perfect_fit_yields_r2_near_one(self) -> None:
        torch.manual_seed(0)
        D = 4
        N = 20
        eye = torch.eye(D)
        X = torch.randn(N, D)
        # Source = target → identity is a perfect map → R² ≈ 1.
        q = alignment_quality(
            {0: eye},
            {0: X},
            {0: X},
        )
        assert q[0] == pytest.approx(1.0, abs=1e-4)

    def test_random_map_yields_low_r2(self) -> None:
        torch.manual_seed(0)
        D = 8
        N = 30
        X_src = torch.randn(N, D)
        X_tgt = torch.randn(N, D)  # uncorrelated
        # Random alignment matrix — no structural relationship.
        q = alignment_quality(
            {0: torch.randn(D, D)},
            {0: X_src},
            {0: X_tgt},
        )
        # R² can go negative when the map fits worse than the mean
        # baseline; for uncorrelated random data we expect << 1.
        assert q[0] < 0.5


# ---------------------------------------------------------------------------
# Disk cache: save / load / path layout.
# ---------------------------------------------------------------------------


class TestAlignmentCache:
    def test_cache_path_layout(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        ts, sc = alignment_cache_path("google/gemma-3-4b-it", "Qwen/Qwen2.5-7B-Instruct")
        # Layout: under the *target* model's dir.
        assert "Qwen__Qwen2.5-7B-Instruct" in str(ts)
        assert "alignments" in str(ts)
        assert "google__gemma-3-4b-it" in str(ts)
        assert ts.suffix == ".safetensors"
        assert sc.suffix == ".json"

    def test_save_load_round_trip(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        D = 4
        M = {0: torch.eye(D), 5: torch.eye(D)}
        quality = {0: 0.85, 5: 0.72}

        save_alignment_map(M, "a/b", "c/d", quality_per_layer=quality)
        result = load_alignment_map("a/b", "c/d")
        assert result is not None
        loaded_M, sidecar = result

        assert sorted(loaded_M.keys()) == [0, 5]
        for layer in loaded_M:
            assert torch.allclose(loaded_M[layer], M[layer])
        assert sidecar["source_model_id"] == "a/b"
        assert sidecar["target_model_id"] == "c/d"
        assert sidecar["shared_layers"] == [0, 5]
        # Quality is stringified on disk; the load_alignment_map shape
        # is intentionally raw (caller decides whether to int-key it).
        assert "0" in sidecar["quality_per_layer"]
        assert sidecar["quality_per_layer"]["0"] == pytest.approx(0.85)

    def test_load_missing_returns_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        assert load_alignment_map("nope/src", "nope/tgt") is None


# ---------------------------------------------------------------------------
# Sidecar transfer fields round-trip through packs.Sidecar.
# ---------------------------------------------------------------------------


class TestSidecarTransferFields:
    def test_round_trip(self, tmp_path) -> None:
        from saklas.io.packs import Sidecar

        sc = Sidecar(
            method="procrustes_transfer",
            saklas_version="1.6.0",
            source_model_id="google/gemma-3-4b-it",
            alignment_map_hash="deadbeef",
            transfer_quality_estimate=0.78,
        )
        path = tmp_path / "sidecar.json"
        sc.write(path)

        with open(path) as f:
            raw = json.load(f)
        assert raw["source_model_id"] == "google/gemma-3-4b-it"
        assert raw["alignment_map_hash"] == "deadbeef"
        assert raw["transfer_quality_estimate"] == pytest.approx(0.78)

        loaded = Sidecar.load(path)
        assert loaded.method == "procrustes_transfer"
        assert loaded.source_model_id == "google/gemma-3-4b-it"
        assert loaded.alignment_map_hash == "deadbeef"
        assert loaded.transfer_quality_estimate == pytest.approx(0.78)
