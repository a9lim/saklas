"""Per-layer diagnostics carried as free-form Profile provenance.

The unified pipeline's ``PcaDiagnostics`` / ``SpectralDiagnostics`` ride the
*manifold* sidecar, not the profile one.  What these cover is what a caller
stashing its own metrics gets from the profile wire format: an arbitrary
provenance blob that survives the save/load round trip.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from saklas.core import profile as V


class TestProvenanceRoundTrip:
    def test_save_load_preserves_diagnostics_provenance(self, tmp_path: Path) -> None:
        profile_dict = {0: torch.ones(4), 2: torch.ones(4) * 0.5}
        # JSON object keys must be strings, so a producer stringifies its own
        # layer keys; the schema refuses a non-string key rather than silently
        # rewriting one.
        diagnostics = {
            "0": {"evr": 0.62, "inter_pair_alignment": 0.78},
            "2": {"evr": 0.41, "inter_pair_alignment": 0.55},
        }
        path = tmp_path / "test.safetensors"
        V.save_profile(profile_dict, str(path), {
            "method": "profile",
            "diagnostics": diagnostics,
        })

        # The blob lands nested under ``provenance`` on the wire; the exact
        # sidecar schema stays five keys wide.
        with open(path.with_suffix(".json")) as f:
            raw = json.load(f)
        assert set(raw) == {
            "format_version", "saklas_version", "method",
            "tensor_sha256", "provenance",
        }
        assert raw["provenance"]["diagnostics"] == diagnostics

        # ...and flattens back to the top level on load, so the metadata the
        # reader gets is the shape the writer accepts.
        _, meta = V.load_profile(str(path))
        assert meta["method"] == "profile"
        assert meta["diagnostics"]["0"]["evr"] == pytest.approx(0.62)

    def test_int_layer_keys_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "int-keys.safetensors"
        with pytest.raises(V.ProfileError, match="object keys must be strings"):
            V.save_profile({0: torch.ones(4)}, str(path), {
                "method": "profile", "diagnostics": {0: {"evr": 0.5}},
            })
        assert not path.exists()

    def test_sidecar_without_diagnostics_loads_clean(self, tmp_path: Path) -> None:
        profile_dict = {0: torch.ones(4)}
        path = tmp_path / "plain.safetensors"
        V.save_profile(profile_dict, str(path), {"method": "profile"})

        with open(path.with_suffix(".json")) as f:
            raw = json.load(f)
        assert raw["provenance"] == {}

        _, meta = V.load_profile(str(path))
        assert "diagnostics" not in meta
