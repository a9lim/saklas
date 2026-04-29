"""Probe-quality diagnostics computed at extract time.

Covers the four metrics ``_compute_layer_diagnostics`` returns, the
soft-warning behavior at degenerate inputs, sidecar round-trip, and
``Profile.diagnostics`` / ``Profile.has_diagnostics`` surface.
"""
from __future__ import annotations

import json
import warnings

import pytest
import torch

from saklas.core import vectors as V
from saklas.core.profile import Profile
from saklas.io.packs import Sidecar


# ---------------------------------------------------------------------------
# Per-layer metric helper.  Cheap to test in isolation since it takes a diff
# matrix + principal direction directly — no model forward needed.
# ---------------------------------------------------------------------------


class TestComputeLayerDiagnostics:
    def test_separable_pairs_high_evr_high_alignment(self) -> None:
        # All diffs point along +x with small noise.  Principal direction
        # should be +x; EVR near 1.0; intra mean close to 1.0; alignment
        # high; projection near 1.0.
        torch.manual_seed(0)
        diffs = torch.zeros(20, 4)
        diffs[:, 0] = 1.0
        diffs += 0.05 * torch.randn(20, 4)
        # Principal direction approximated by mean of unit-normalized diffs
        unit = diffs / diffs.norm(dim=-1, keepdim=True)
        v = unit.mean(dim=0)
        evr = 0.95  # caller-supplied — matches what SVD would report

        m = V._compute_layer_diagnostics(diffs, v, evr)

        assert m["evr"] == pytest.approx(0.95)
        assert m["intra_pair_variance_mean"] == pytest.approx(1.0, abs=0.1)
        assert m["intra_pair_variance_std"] < 0.1
        assert m["inter_pair_alignment"] > 0.85
        assert m["diff_principal_projection"] > 0.95

    def test_random_diffs_low_alignment(self) -> None:
        # Random unit-ish diffs in 8-d.  Inter-pair alignment should be low.
        torch.manual_seed(0)
        diffs = torch.randn(30, 8)
        v = diffs[0]  # arbitrary direction
        m = V._compute_layer_diagnostics(diffs, v, 0.2)

        # 30 vectors in 8-d sphere: expected |cos| ≈ 1/sqrt(d) ≈ 0.35.
        # Threshold here is generous to keep the test stable across seeds.
        assert m["inter_pair_alignment"] < 0.5

    def test_one_sided_pairs_near_zero_intra_variance(self) -> None:
        # All diffs identical: intra variance collapses to 0, alignment 1.
        diffs = torch.tensor([[1.0, 0.0]] * 10)
        v = torch.tensor([1.0, 0.0])
        m = V._compute_layer_diagnostics(diffs, v, 0.99)

        assert m["intra_pair_variance_std"] == pytest.approx(0.0, abs=1e-5)
        assert m["inter_pair_alignment"] == pytest.approx(1.0, abs=1e-4)
        assert m["diff_principal_projection"] == pytest.approx(1.0, abs=1e-4)

    def test_single_pair_returns_minimal_dict(self) -> None:
        # N=1 path: only intra mean is meaningful; the rest are tautological.
        diffs = torch.tensor([[2.0, 0.0]])
        v = torch.tensor([1.0, 0.0])
        m = V._compute_layer_diagnostics(diffs, v, 0.1)

        assert m["intra_pair_variance_mean"] == pytest.approx(2.0)
        assert m["intra_pair_variance_std"] == 0.0
        assert m["inter_pair_alignment"] == 1.0
        assert m["diff_principal_projection"] == 1.0


# ---------------------------------------------------------------------------
# End-to-end through ``extract_contrastive`` with a stubbed forward pass.
# ---------------------------------------------------------------------------


def _stub_encode(model, tokenizer, text, layers, device):
    """Per-layer activation that has clear pos/neg separation along axis 0.

    Reused by every full-extraction test below; matches the shape
    ``_encode_and_capture_all`` would produce on a real forward pass.
    """
    n = len(layers)
    sign = 1.0 if "pos" in text else -1.0
    out: dict[int, torch.Tensor] = {}
    for idx in range(n):
        base = torch.zeros(4)
        base[0] = sign * 1.0
        out[idx] = base + 0.05 * torch.randn(4)
    return out


class _FakeModel:
    def parameters(self):
        yield torch.zeros(1)


class _FakeTok:
    pass


class TestExtractContrastiveReturnsTuple:
    """``extract_contrastive`` returns ``(profile, diagnostics)``."""

    def test_multi_pair_diagnostics_shape(self, monkeypatch) -> None:
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode)

        pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(8)]
        profile, diagnostics = V.extract_contrastive(
            _FakeModel(), _FakeTok(), pairs, layers=[object()] * 6,
            device=torch.device("cpu"),
            drop_edges=(0, 0),
        )

        # Diagnostics layer set matches the profile layer set exactly so
        # downstream consumers can index either dict by layer index without
        # having to branch on missing entries.
        assert set(diagnostics.keys()) == set(profile.keys())

        for metrics in diagnostics.values():
            assert {
                "evr",
                "intra_pair_variance_mean",
                "intra_pair_variance_std",
                "inter_pair_alignment",
                "diff_principal_projection",
            } <= set(metrics.keys())

    def test_drop_edges_drops_diagnostics_layers(self, monkeypatch) -> None:
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode)

        pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(5)]
        profile, diagnostics = V.extract_contrastive(
            _FakeModel(), _FakeTok(), pairs, layers=[object()] * 8,
            device=torch.device("cpu"),
            drop_edges=(2, 2),
        )

        assert set(profile.keys()) == set(diagnostics.keys()) == set(range(2, 6))


class TestSoftWarning:
    """``UserWarning`` fires on degenerate metrics, never raises."""

    def test_one_sided_pairs_warn(self, monkeypatch) -> None:
        # Identical pos/neg activations except a tiny axis-0 separation —
        # high EVR, near-zero intra variance, no inter-pair disagreement.
        def _identical(model, tokenizer, text, layers, device):
            sign = 1.0 if "pos" in text else -1.0
            return {idx: torch.tensor([sign * 0.001, 0.0, 0.0, 0.0])
                    for idx in range(len(layers))}

        monkeypatch.setattr(V, "_encode_and_capture_all", _identical)
        pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(5)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            V.extract_contrastive(
                _FakeModel(), _FakeTok(), pairs, layers=[object()] * 4,
                device=torch.device("cpu"),
                drop_edges=(0, 0),
                concept_label="probe-test",
            )

        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("one-sided" in m for m in msgs)
        assert any("probe-test" in m for m in msgs)

    def test_aligned_pairs_no_warning(self, monkeypatch) -> None:
        torch.manual_seed(42)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode)
        pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(20)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            V.extract_contrastive(
                _FakeModel(), _FakeTok(), pairs, layers=[object()] * 4,
                device=torch.device("cpu"),
                drop_edges=(0, 0),
                concept_label="clean-probe",
            )

        diag_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and ("one-sided" in str(w.message) or "directions disagree" in str(w.message))
        ]
        assert diag_warnings == []


# ---------------------------------------------------------------------------
# Sidecar round-trip + Profile metadata.
# ---------------------------------------------------------------------------


class TestSidecarRoundTrip:
    def test_save_load_preserves_diagnostics(self, tmp_path) -> None:
        profile_dict = {0: torch.ones(4), 2: torch.ones(4) * 0.5}
        diagnostics = {
            0: {
                "evr": 0.62,
                "intra_pair_variance_mean": 1.05,
                "intra_pair_variance_std": 0.08,
                "inter_pair_alignment": 0.78,
                "diff_principal_projection": 0.85,
            },
            2: {
                "evr": 0.41,
                "intra_pair_variance_mean": 0.72,
                "intra_pair_variance_std": 0.12,
                "inter_pair_alignment": 0.55,
                "diff_principal_projection": 0.73,
            },
        }
        path = tmp_path / "test.safetensors"
        V.save_profile(profile_dict, str(path), {
            "method": "contrastive_pca",
            "diagnostics": diagnostics,
        })

        # Sidecar JSON: keys must be strings (JSON spec); reader converts back.
        with open(path.with_suffix(".json")) as f:
            raw = json.load(f)
        assert "diagnostics_by_layer" in raw
        assert set(raw["diagnostics_by_layer"].keys()) == {"0", "2"}

        loaded_tensors, meta = V.load_profile(str(path))
        assert "diagnostics" in meta
        # Round-trips with int layer keys, not strings.
        assert set(meta["diagnostics"].keys()) == {0, 2}
        assert meta["diagnostics"][0]["evr"] == pytest.approx(0.62)
        assert meta["diagnostics"][2]["inter_pair_alignment"] == pytest.approx(0.55)

    def test_old_sidecar_without_diagnostics_loads_clean(self, tmp_path) -> None:
        # Simulate a v1.5-era sidecar: no diagnostics_by_layer key at all.
        profile_dict = {0: torch.ones(4)}
        path = tmp_path / "old.safetensors"
        V.save_profile(profile_dict, str(path), {"method": "contrastive_pca"})

        with open(path.with_suffix(".json")) as f:
            raw = json.load(f)
        assert "diagnostics_by_layer" not in raw

        _, meta = V.load_profile(str(path))
        assert "diagnostics" not in meta

    def test_sidecar_dataclass_roundtrip(self, tmp_path) -> None:
        # Mirror the same shape through io.packs.Sidecar so packs.py
        # readers see the same field consistently.
        sc = Sidecar(
            method="contrastive_pca",
            saklas_version="1.6.0",
            diagnostics_by_layer={0: {"evr": 0.5, "inter_pair_alignment": 0.8}},
        )
        path = tmp_path / "sidecar.json"
        sc.write(path)

        loaded = Sidecar.load(path)
        assert loaded.diagnostics_by_layer is not None
        assert loaded.diagnostics_by_layer[0]["evr"] == pytest.approx(0.5)
        assert loaded.diagnostics_by_layer[0]["inter_pair_alignment"] == pytest.approx(0.8)


class TestProfileSurface:
    def test_profile_diagnostics_property(self) -> None:
        diagnostics = {0: {"evr": 0.7, "intra_pair_variance_mean": 1.2}}
        p = Profile(
            {0: torch.ones(4)},
            metadata={"method": "contrastive_pca", "diagnostics": diagnostics},
        )
        assert p.has_diagnostics is True
        out = p.diagnostics
        assert out is not None
        assert out[0]["evr"] == pytest.approx(0.7)

    def test_profile_diagnostics_absent_returns_none(self) -> None:
        p = Profile({0: torch.ones(4)}, metadata={"method": "contrastive_pca"})
        assert p.has_diagnostics is False
        assert p.diagnostics is None

    def test_profile_diagnostics_returns_defensive_copy(self) -> None:
        diagnostics = {0: {"evr": 0.5}}
        p = Profile(
            {0: torch.ones(4)},
            metadata={"method": "contrastive_pca", "diagnostics": diagnostics},
        )
        out = p.diagnostics
        assert out is not None
        out[0]["evr"] = 999.0
        # Cached metric dict is untouched by mutation through the surface.
        again = p.diagnostics
        assert again is not None
        assert again[0]["evr"] == pytest.approx(0.5)
