"""Pack + sidecar format_version gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from saklas.io.integrity import PROFILE_FORMAT_VERSION
from saklas.core.profile import Profile, ProfileError
from saklas.core.profile import load_profile, save_profile


def test_save_profile_writes_format_version_in_sidecar(tmp_path: Path):
    profile = {0: torch.zeros(4), 1: torch.ones(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "profile"})
    sidecar = json.loads((path.with_suffix(".json")).read_text())
    assert sidecar["format_version"] == PROFILE_FORMAT_VERSION
    assert sidecar["method"] == "profile"


def test_load_profile_rejects_missing_format_version(tmp_path: Path):
    profile = {0: torch.zeros(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "profile"})

    # The current reader never infers a version for an unstamped sidecar.
    sc_path = path.with_suffix(".json")
    data = json.loads(sc_path.read_text())
    data.pop("format_version", None)
    sc_path.write_text(json.dumps(data))

    with pytest.raises(ProfileError, match="need exactly"):
        load_profile(str(path))


@pytest.mark.parametrize("version", [1, PROFILE_FORMAT_VERSION + 1])
def test_load_profile_rejects_non_current_format_version(
    tmp_path: Path, version: int,
):
    profile = {0: torch.zeros(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "profile"})

    sc_path = path.with_suffix(".json")
    data = json.loads(sc_path.read_text())
    data["format_version"] = version
    sc_path.write_text(json.dumps(data))

    # A stale artifact says so by version, not by whichever field the current
    # exact-set schema happens to miss first, and names the remedy.
    with pytest.raises(ProfileError, match="need exactly"):
        load_profile(str(path))


def test_load_profile_rejects_pre_cut_sidecar_by_version(tmp_path: Path):
    """The pre-5.x fourteen-key sidecar is rejected on version, not shape.

    The field set was cut without a bump, so a file stamped ``5`` already
    failed the exact-set check; ``PROFILE_FORMAT_VERSION = 6`` makes that
    rejection say what actually happened.
    """
    profile = {0: torch.zeros(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "profile"})

    sc_path = path.with_suffix(".json")
    data = json.loads(sc_path.read_text())
    data.pop("provenance")
    data.update({
        "format_version": 5, "statements_sha256": "x", "components": None,
        "bake": None, "sae_release": None, "sae_revision": None,
        "sae_ids_by_layer": {}, "source_model_id": None,
        "alignment_map_hash": None, "transfer_quality_estimate": None,
        "diagnostics_by_layer": {},
    })
    sc_path.write_text(json.dumps(data))

    with pytest.raises(ProfileError, match="format_version=5"):
        load_profile(str(path))


def test_profile_save_roundtrip_uses_format_version(tmp_path: Path):
    p = Profile({0: torch.randn(4), 1: torch.randn(4)})
    path = tmp_path / "y.safetensors"
    p.save(path)
    sidecar = json.loads((path.with_suffix(".json")).read_text())
    assert sidecar["format_version"] == PROFILE_FORMAT_VERSION
    # Round-trip the Profile.load path too.
    back = Profile.load(path)
    assert back.layers == [0, 1]
