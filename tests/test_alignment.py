"""Cross-model probe alignment via compact per-layer affine maps.

Covers ``fit_alignment`` (matched + mismatched dim, layer-count
mismatch), ``alignment_quality`` (R² metric), and the disk cache shape.

CPU-only, no model load — alignment math runs over synthetic
activation tensors.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from saklas.io.alignment import (
    AlignmentError,
    LayerAlignment,
    alignment_cache_path,
    alignment_quality,
    fit_alignment,
    load_alignment_map,
    save_alignment_map,
)


def _random_activations(
    n_layers: int, n_prompts: int, dim: int, *, seed: int = 0,
) -> dict[int, torch.Tensor]:
    """Synthetic ``[N, D]`` activation stack per layer."""
    g = torch.Generator().manual_seed(seed)
    return {layer: torch.randn(n_prompts, dim, generator=g) for layer in range(n_layers)}


def _alignment(dense: torch.Tensor) -> LayerAlignment:
    """Current affine alignment shape for tests authored from dense maps."""
    target_dim, source_dim = dense.shape
    return LayerAlignment(
        dense,
        torch.eye(source_dim),
        torch.zeros(target_dim),
    )


def _alignments(dense: dict[int, torch.Tensor]) -> dict[int, LayerAlignment]:
    return {layer: _alignment(value) for layer, value in dense.items()}


# ---------------------------------------------------------------------------
# fit_alignment.
# ---------------------------------------------------------------------------


class TestFitAlignment:
    def test_factorized_fit_is_low_rank_and_carries_affine_offset(self) -> None:
        torch.manual_seed(42)
        n, d_src, d_tgt = 12, 64, 48
        src = torch.randn(n, d_src)
        dense = torch.randn(d_tgt, d_src)
        offset = torch.linspace(-2.0, 2.0, d_tgt)
        tgt = src @ dense.transpose(0, 1) + offset

        fitted = fit_alignment(
            {layer: src for layer in range(10)},
            {layer: tgt for layer in range(10)},
        )[0]

        assert fitted.rank <= n - 1
        assert fitted.left.numel() + fitted.right.numel() < d_src * d_tgt
        assert torch.allclose(fitted.apply_points(src), tgt, atol=2e-4, rtol=2e-4)
        assert torch.allclose(
            fitted.offset,
            tgt.mean(0) - fitted.apply_vector(src.mean(0)),
            atol=1e-5,
        )

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
        assert torch.allclose(M[0].to_dense(), Q, atol=1e-2)

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
        X_pred = M[0].apply_vectors(X_src - X_src.mean(dim=0))
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
            _alignments({0: eye}),
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
            _alignments({0: torch.randn(D, D)}),
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
    _SRC_ID = {"model_fingerprint": "src", "capture_sha256": "a" * 64}
    _TGT_ID = {"model_fingerprint": "tgt", "capture_sha256": "b" * 64}

    def test_cache_path_layout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from saklas.io.paths import safe_model_id

        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        ts, sc = alignment_cache_path("google/gemma-3-4b-it", "Qwen/Qwen2.5-7B-Instruct")
        # Layout: under the *target* model's dir.
        assert safe_model_id("Qwen/Qwen2.5-7B-Instruct") in str(ts)
        assert "alignments" in str(ts)
        assert safe_model_id("google/gemma-3-4b-it") in str(ts)
        assert ts.suffix == ".safetensors"
        assert sc.suffix == ".json"

    def test_save_load_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        D = 4
        M = _alignments({0: torch.eye(D), 5: torch.eye(D)})
        quality = {0: 0.85, 5: 0.72}

        save_alignment_map(
            M, "a/b", "c/d", source_identity=self._SRC_ID,
            target_identity=self._TGT_ID, quality_per_layer=quality,
        )
        result = load_alignment_map(
            "a/b", "c/d", source_identity=self._SRC_ID,
            target_identity=self._TGT_ID,
        )
        assert result is not None
        loaded_M, sidecar = result

        assert sorted(loaded_M.keys()) == [0, 5]
        for layer in loaded_M:
            assert torch.allclose(loaded_M[layer].to_dense(), M[layer].to_dense())
        assert sidecar["source_model_id"] == "a/b"
        assert sidecar["target_model_id"] == "c/d"
        assert sidecar["shared_layers"] == [0, 5]
        # Quality is stringified on disk; the load_alignment_map shape
        # is intentionally raw (caller decides whether to int-key it).
        assert "0" in sidecar["quality_per_layer"]
        assert sidecar["quality_per_layer"]["0"] == pytest.approx(0.85)

    def test_load_missing_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        assert load_alignment_map(
            "nope/src", "nope/tgt", source_identity=self._SRC_ID,
            target_identity=self._TGT_ID,
        ) is None

    def test_identity_drift_invalidates_alignment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        save_alignment_map(
            _alignments({0: torch.eye(3)}), "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
        )
        changed = {**self._SRC_ID, "model_fingerprint": "changed"}
        assert load_alignment_map(
            "a/b", "c/d", source_identity=changed,
            target_identity=self._TGT_ID,
        ) is None

    def test_identity_drift_never_materializes_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        save_alignment_map(
            _alignments({0: torch.eye(3)}), "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
        )
        import saklas.io.alignment as alignment_mod

        monkeypatch.setattr(
            alignment_mod, "load_safetensors",
            lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("stale identity materialized alignment payload")
            ),
        )
        changed = {**self._SRC_ID, "model_fingerprint": "changed"}
        assert load_alignment_map(
            "a/b", "c/d", source_identity=changed,
            target_identity=self._TGT_ID,
        ) is None

    def test_cache_round_trips_factorized_offset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        alignment = LayerAlignment(
            left=torch.randn(6, 2), right=torch.randn(2, 4),
            offset=torch.arange(6, dtype=torch.float32),
        )
        save_alignment_map(
            {3: alignment}, "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
        )
        loaded = load_alignment_map(
            "a/b", "c/d", source_identity=self._SRC_ID,
            target_identity=self._TGT_ID,
        )
        assert loaded is not None
        restored, sidecar = loaded
        assert sidecar["format_version"] == 5
        assert torch.equal(restored[3].left, alignment.left)
        assert torch.equal(restored[3].right, alignment.right)
        assert torch.equal(restored[3].offset, alignment.offset)

    def test_sharded_save_avoids_payload_rehash_and_cleans_legacy_monolith(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        import saklas.io.alignment as alignment_mod

        anchor, _ = alignment_mod._alignment_anchor_paths("a/b", "c/d")
        anchor.parent.mkdir(parents=True, exist_ok=True)
        anchor.write_bytes(b"legacy-v3")
        monkeypatch.setattr(
            alignment_mod, "hash_file",
            lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("alignment save reread a written shard")
            ),
        )
        saved = save_alignment_map(
            _alignments({0: torch.eye(4), 5: torch.eye(4)}), "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
        )
        assert saved.exists()
        assert not anchor.exists()
        sidecar = json.loads(anchor.with_suffix(".json").read_text())
        assert set(sidecar["tensor_files"]) == {"0", "5"}
        assert len(set(sidecar["tensor_files"].values())) == 2

    def test_selective_load_reads_only_requested_factor_shard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        save_alignment_map(
            _alignments({0: torch.eye(4), 5: 2 * torch.eye(4)}), "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
        )
        real_read = Path.read_bytes
        read_names: list[str] = []

        def counted_read(path: Path) -> bytes:
            read_names.append(path.name)
            return real_read(path)

        monkeypatch.setattr(Path, "read_bytes", counted_read)
        loaded = load_alignment_map(
            "a/b", "c/d", source_identity=self._SRC_ID,
            target_identity=self._TGT_ID, requested_layers=[5, 99],
        )
        assert loaded is not None
        factors, sidecar = loaded
        assert set(factors) == {5}
        assert sidecar["shared_layers"] == [0, 5]
        assert len(read_names) == 1
        assert ".layer-5." in read_names[0]

    def test_failed_pointer_publication_preserves_prior_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        import saklas.io.alignment as alignment_mod

        save_alignment_map(
            _alignments({0: torch.eye(4)}), "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
        )
        anchor, pointer = alignment_mod._alignment_anchor_paths("a/b", "c/d")
        prior_pointer = pointer.read_bytes()
        prior_files = set(json.loads(prior_pointer)["tensor_files"].values())

        def _fail_pointer(_path: Path, _payload: object) -> None:
            raise OSError("simulated alignment pointer failure")

        monkeypatch.setattr(alignment_mod, "write_json_atomic", _fail_pointer)
        with pytest.raises(OSError, match="pointer failure"):
            save_alignment_map(
                _alignments({0: 2 * torch.eye(4)}), "a/b", "c/d",
                source_identity=self._SRC_ID, target_identity=self._TGT_ID,
            )

        assert pointer.read_bytes() == prior_pointer
        assert {
            path.name for path in anchor.parent.glob(
                f"{anchor.stem}.layer-*.gen-*.safetensors"
            )
        } == prior_files
        loaded = load_alignment_map(
            "a/b", "c/d", source_identity=self._SRC_ID,
            target_identity=self._TGT_ID,
        )
        assert loaded is not None
        assert torch.equal(loaded[0][0].to_dense(), torch.eye(4))

    def test_post_pointer_fsync_failure_preserves_new_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After pointer replace, a durability error must not delete its shards."""
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        import saklas.io.alignment as alignment_mod

        calls = 0

        def _fail_fsync(_folder: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated directory fsync failure")

        monkeypatch.setattr(alignment_mod, "fsync_directory", _fail_fsync)
        with pytest.raises(OSError, match="fsync failure"):
            save_alignment_map(
                _alignments({0: 2 * torch.eye(4)}), "a/b", "c/d",
                source_identity=self._SRC_ID, target_identity=self._TGT_ID,
            )

        anchor, pointer = alignment_mod._alignment_anchor_paths("a/b", "c/d")
        sidecar = json.loads(pointer.read_text())
        current = anchor.parent / sidecar["tensor_files"]["0"]
        assert current.is_file()
        loaded = load_alignment_map(
            "a/b", "c/d", source_identity=self._SRC_ID,
            target_identity=self._TGT_ID,
        )
        assert loaded is not None
        assert torch.equal(loaded[0][0].to_dense(), 2 * torch.eye(4))

    def test_payload_directory_barrier_precedes_alignment_pointer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        import saklas.io.alignment as alignment_mod

        events: list[str] = []
        real_write = alignment_mod.write_json_atomic

        def record_write(path: Path, payload: object) -> None:
            events.append("pointer")
            real_write(path, payload)

        monkeypatch.setattr(alignment_mod, "write_json_atomic", record_write)
        monkeypatch.setattr(
            alignment_mod, "fsync_directory",
            lambda _path: events.append("barrier"),
        )
        save_alignment_map(
            _alignments({0: torch.eye(4)}), "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
        )
        assert events[0:3] == ["barrier", "pointer", "barrier"]

    def test_exception_after_pointer_replace_preserves_new_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        import saklas.io.alignment as alignment_mod

        real_write = alignment_mod.write_json_atomic

        def write_then_fail(path: Path, payload: object) -> None:
            real_write(path, payload)
            raise OSError("after alignment pointer replace")

        monkeypatch.setattr(alignment_mod, "write_json_atomic", write_then_fail)
        with pytest.raises(OSError, match="after alignment pointer"):
            save_alignment_map(
                _alignments({0: 3 * torch.eye(4)}), "a/b", "c/d",
                source_identity=self._SRC_ID, target_identity=self._TGT_ID,
            )

        loaded = load_alignment_map(
            "a/b", "c/d", source_identity=self._SRC_ID,
            target_identity=self._TGT_ID,
        )
        assert loaded is not None
        assert torch.equal(loaded[0][0].to_dense(), 3 * torch.eye(4))

    def test_alignment_top_up_reuses_unrequested_generation_without_reading_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        save_alignment_map(
            _alignments({0: torch.eye(4)}), "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
            quality_per_layer={0: 0.8},
        )
        import saklas.io.alignment as alignment_mod

        anchor, pointer = alignment_mod._alignment_anchor_paths("a/b", "c/d")
        prior = json.loads(pointer.read_text())
        prior_file = prior["tensor_files"]["0"]
        real_read = Path.read_bytes

        def reject_old_read(path: Path) -> bytes:
            if path.name == prior_file:
                raise AssertionError("top-up reread unrelated factor payload")
            return real_read(path)

        monkeypatch.setattr(Path, "read_bytes", reject_old_read)
        save_alignment_map(
            _alignments({5: 2 * torch.eye(4)}), "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
            quality_per_layer={5: 0.9}, extend=True,
        )
        current = json.loads(pointer.read_text())
        assert current["shared_layers"] == [0, 5]
        assert current["tensor_files"]["0"] == prior_file
        assert (anchor.parent / prior_file).exists()

    def test_corrupt_requested_factor_repair_preserves_unrequested_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        import saklas.io.alignment as alignment_mod

        save_alignment_map(
            _alignments({0: torch.eye(4), 5: torch.eye(4)}), "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
            quality_per_layer={0: 0.8, 5: 0.7},
        )
        anchor, pointer = alignment_mod._alignment_anchor_paths("a/b", "c/d")
        before = json.loads(pointer.read_text())
        layer0_name = before["tensor_files"]["0"]
        corrupt = anchor.parent / before["tensor_files"]["5"]
        payload = bytearray(corrupt.read_bytes())
        payload[-1] ^= 0x01
        corrupt.write_bytes(payload)

        partial = load_alignment_map(
            "a/b", "c/d", source_identity=self._SRC_ID,
            target_identity=self._TGT_ID, requested_layers=[5],
        )
        assert partial is not None and partial[0] == {}
        save_alignment_map(
            _alignments({5: 2 * torch.eye(4)}), "a/b", "c/d",
            source_identity=self._SRC_ID, target_identity=self._TGT_ID,
            quality_per_layer={5: 0.9}, extend=True,
        )

        after = json.loads(pointer.read_text())
        assert after["shared_layers"] == [0, 5]
        assert after["tensor_files"]["0"] == layer0_name
        assert (anchor.parent / layer0_name).exists()
        loaded = load_alignment_map(
            "a/b", "c/d", source_identity=self._SRC_ID,
            target_identity=self._TGT_ID,
        )
        assert loaded is not None and set(loaded[0]) == {0, 5}
        assert torch.equal(loaded[0][5].to_dense(), 2 * torch.eye(4))
