"""Per-layer diagnostics carried as free-form Profile provenance.

Nothing in saklas writes profile diagnostics — the unified pipeline's
``PcaDiagnostics`` / ``SpectralDiagnostics`` ride the *manifold* sidecar
instead.  What these cover is the surface a caller stashing its own
metrics gets: the provenance blob round-trips them and
``Profile.diagnostics`` / ``Profile.has_diagnostics`` read them back.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from saklas.core import profile as V
from saklas.core.profile import Profile


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


class TestProfileSurface:
    def test_profile_diagnostics_property(self) -> None:
        diagnostics = {0: {"evr": 0.7, "intra_pair_variance_mean": 1.2}}
        p = Profile(
            {0: torch.ones(4)},
            metadata={"method": "profile", "diagnostics": diagnostics},
        )
        assert p.has_diagnostics is True
        out = p.diagnostics
        assert out is not None
        assert out[0]["evr"] == pytest.approx(0.7)

    def test_profile_diagnostics_absent_returns_none(self) -> None:
        p = Profile({0: torch.ones(4)}, metadata={"method": "profile"})
        assert p.has_diagnostics is False
        assert p.diagnostics is None

    def test_profile_diagnostics_returns_defensive_copy(self) -> None:
        diagnostics = {0: {"evr": 0.5}}
        p = Profile(
            {0: torch.ones(4)},
            metadata={"method": "profile", "diagnostics": diagnostics},
        )
        out = p.diagnostics
        assert out is not None
        out[0]["evr"] = 999.0
        # Cached metric dict is untouched by mutation through the surface.
        again = p.diagnostics
        assert again is not None
        assert again[0]["evr"] == pytest.approx(0.5)
