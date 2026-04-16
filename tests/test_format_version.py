"""Pack + sidecar format_version gate."""
from __future__ import annotations

import json

import pytest
import torch

from saklas.io import packs
from saklas.io.packs import PACK_FORMAT_VERSION, PackFormatError
from saklas.core.profile import Profile, ProfileError
from saklas.core.vectors import save_profile, load_profile


def test_save_profile_writes_format_version_in_sidecar(tmp_path):
    profile = {0: torch.zeros(4), 1: torch.ones(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "contrastive_pca"})
    sidecar = json.loads((path.with_suffix(".json")).read_text())
    assert sidecar["format_version"] == PACK_FORMAT_VERSION
    assert sidecar["method"] == "contrastive_pca"


def test_load_profile_rejects_missing_format_version(tmp_path):
    profile = {0: torch.zeros(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "contrastive_pca"})

    # Strip format_version to simulate a v1.x sidecar.
    sc_path = path.with_suffix(".json")
    data = json.loads(sc_path.read_text())
    data.pop("format_version", None)
    sc_path.write_text(json.dumps(data))

    with pytest.raises(ProfileError, match="upgrade_packs"):
        load_profile(str(path))


def test_load_profile_rejects_format_version_one(tmp_path):
    profile = {0: torch.zeros(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "contrastive_pca"})

    sc_path = path.with_suffix(".json")
    data = json.loads(sc_path.read_text())
    data["format_version"] = 1
    sc_path.write_text(json.dumps(data))

    with pytest.raises(ProfileError, match="saklas < 2.0"):
        load_profile(str(path))


def test_profile_save_roundtrip_uses_format_version(tmp_path):
    p = Profile({0: torch.randn(4), 1: torch.randn(4)})
    path = tmp_path / "y.safetensors"
    p.save(path)
    sidecar = json.loads((path.with_suffix(".json")).read_text())
    assert sidecar["format_version"] == PACK_FORMAT_VERSION
    # Round-trip the Profile.load path too.
    back = Profile.load(path)
    assert back.layers == [0, 1]


def test_pack_metadata_rejects_format_version_one(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    (d / "pack.json").write_text(json.dumps({
        "name": "p",
        "description": "x",
        "format_version": 1,
        "version": "1.0.0",
        "license": "MIT",
        "tags": [],
        "recommended_alpha": 0.5,
        "source": "local",
        "files": {},
    }))
    with pytest.raises(PackFormatError, match="upgrade_packs"):
        packs.PackMetadata.load(d)


def test_pack_metadata_rejects_missing_format_version(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    (d / "pack.json").write_text(json.dumps({
        "name": "p",
        "description": "x",
        "version": "1.0.0",
        "license": "MIT",
        "tags": [],
        "recommended_alpha": 0.5,
        "source": "local",
        "files": {},
    }))
    with pytest.raises(PackFormatError, match="upgrade_packs"):
        packs.PackMetadata.load(d)


def test_pack_metadata_write_includes_format_version(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    meta = packs.PackMetadata(
        name="p", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local", files={},
    )
    meta.write(d)
    data = json.loads((d / "pack.json").read_text())
    assert data["format_version"] == PACK_FORMAT_VERSION
