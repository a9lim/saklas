"""Probe-quality diagnostics computed at extract time.

Covers the four metrics ``_compute_layer_diagnostics`` returns, the
soft-warning behavior at degenerate inputs, sidecar round-trip, and
``Profile.diagnostics`` / ``Profile.has_diagnostics`` surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from saklas.core import vectors as V
from saklas.core.profile import Profile


# ---------------------------------------------------------------------------
# Per-layer metric helper.  Cheap to test in isolation since it takes a diff
# matrix + principal direction directly — no model forward needed.
# ---------------------------------------------------------------------------


class TestSidecarRoundTrip:
    def test_save_load_preserves_diagnostics(self, tmp_path: Path) -> None:
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

    def test_old_sidecar_without_diagnostics_loads_clean(self, tmp_path: Path) -> None:
        # Simulate a v1.5-era sidecar: no diagnostics_by_layer key at all.
        profile_dict = {0: torch.ones(4)}
        path = tmp_path / "old.safetensors"
        V.save_profile(profile_dict, str(path), {"method": "contrastive_pca"})

        with open(path.with_suffix(".json")) as f:
            raw = json.load(f)
        assert "diagnostics_by_layer" not in raw

        _, meta = V.load_profile(str(path))
        assert "diagnostics" not in meta


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
