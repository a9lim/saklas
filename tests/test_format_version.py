"""Pack + sidecar format_version gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from saklas.io.packs import PACK_FORMAT_VERSION
from saklas.core.profile import Profile, ProfileError
from saklas.core.vectors import save_profile, load_profile


def test_save_profile_writes_format_version_in_sidecar(tmp_path: Path):
    profile = {0: torch.zeros(4), 1: torch.ones(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "contrastive_pca"})
    sidecar = json.loads((path.with_suffix(".json")).read_text())
    assert sidecar["format_version"] == PACK_FORMAT_VERSION
    assert sidecar["method"] == "contrastive_pca"


def test_load_profile_rejects_missing_format_version(tmp_path: Path):
    profile = {0: torch.zeros(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "contrastive_pca"})

    # Strip format_version to simulate a v1.x sidecar.
    sc_path = path.with_suffix(".json")
    data = json.loads(sc_path.read_text())
    data.pop("format_version", None)
    sc_path.write_text(json.dumps(data))

    with pytest.raises(ProfileError, match="regenerate"):
        load_profile(str(path))


def test_load_profile_rejects_format_version_one(tmp_path: Path):
    profile = {0: torch.zeros(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "contrastive_pca"})

    sc_path = path.with_suffix(".json")
    data = json.loads(sc_path.read_text())
    data["format_version"] = 1
    sc_path.write_text(json.dumps(data))

    with pytest.raises(ProfileError, match="saklas < 2.0"):
        load_profile(str(path))


def test_profile_save_roundtrip_uses_format_version(tmp_path: Path):
    p = Profile({0: torch.randn(4), 1: torch.randn(4)})
    path = tmp_path / "y.safetensors"
    p.save(path)
    sidecar = json.loads((path.with_suffix(".json")).read_text())
    assert sidecar["format_version"] == PACK_FORMAT_VERSION
    # Round-trip the Profile.load path too.
    back = Profile.load(path)
    assert back.layers == [0, 1]


# NOTE: the three ``test_pack_metadata_*`` tests were deleted in 4.0 —
# ``saklas.io.packs.PackMetadata`` (the ``vectors/`` ``pack.json`` dataclass
# and its ``format_version`` load gate) was removed.  Concepts ship as
# manifolds now; the manifold-side ``format_version`` gate is covered by the
# ``MANIFOLD_FORMAT_VERSION`` tests in ``test_manifold_format.py``.  The
# ``save_profile`` / ``Profile`` sidecar ``format_version`` checks above
# (still using ``PACK_FORMAT_VERSION``, which SURVIVES) stay.
