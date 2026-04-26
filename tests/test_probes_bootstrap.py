from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.core.errors import StaleSidecarError
from saklas.io import packs, probes_bootstrap


def _mk_concept(home: Path, ns: str, name: str, tags: list[str]) -> Path:
    d = home / "vectors" / ns / name
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    packs.PackMetadata(
        name=name, description="x", version="1.0.0", license="MIT",
        tags=tags, recommended_alpha=0.5, source="bundled",
        files={"statements.json": packs.hash_file(d / "statements.json")},
    ).write(d)
    return d


def _bake_fake_tensor(folder: Path, sid: str, statements_sha: str) -> None:
    """Save a minimal fake profile + sidecar so bootstrap_probes' cache
    short-circuit fires without needing a real model forward pass."""
    from saklas.core.vectors import save_profile
    profile = {0: torch.zeros(8, dtype=torch.float32)}
    save_profile(profile, str(folder / f"{sid}.safetensors"), {
        "method": "contrastive_pca",
        "statements_sha256": statements_sha,
    })


def test_load_defaults_groups_by_tag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    # Materialization will also run and copy the bundled 28; pre-create the
    # ones we're asserting on so they win the tag placement.
    _mk_concept(tmp_path, "default", "zz-custom-a", ["custom-tag"])
    _mk_concept(tmp_path, "default", "zz-custom-b", ["custom-tag"])
    d = probes_bootstrap.load_defaults()
    assert sorted(d["custom-tag"]) == ["zz-custom-a", "zz-custom-b"]
    # Bundled happy.sad should have materialized under the affect tag.
    # Skip until the bundled pack is regenerated to the new 21-probe layout
    # via scripts/regenerate_bundled_statements.py --purge.
    if "affect" not in d:
        pytest.skip("bundled pack pending regeneration to new 21-probe layout")
    assert "happy.sad" in d["affect"]


def test_bootstrap_raises_on_stale_statements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """A modified ``statements.json`` after extraction must raise
    StaleSidecarError, naming the concrete pack-refresh remediation
    rather than silently returning a stale tensor."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.delenv("SAKLAS_ALLOW_STALE", raising=False)

    cdir = _mk_concept(tmp_path, "default", "zz-custom", ["custom-tag"])
    stmts_path = cdir / "statements.json"
    extraction_time_sha = packs.hash_file(stmts_path)
    _bake_fake_tensor(cdir, "google__gemma-2-2b-it", extraction_time_sha)

    # User edits statements.json after extraction; refresh pack.json's
    # files map so the integrity check still passes — staleness is
    # specifically the sidecar-vs-statements mismatch, not a tampering one.
    stmts_path.write_text('[["my edit", "after extraction"]]')
    meta = packs.PackMetadata.load(cdir)
    meta.files["statements.json"] = packs.hash_file(stmts_path)
    meta.files[f"google__gemma-2-2b-it.safetensors"] = packs.hash_file(
        cdir / "google__gemma-2-2b-it.safetensors"
    )
    meta.files[f"google__gemma-2-2b-it.json"] = packs.hash_file(
        cdir / "google__gemma-2-2b-it.json"
    )
    meta.write(cdir)

    model_info: dict[str, Any] = {"model_id": "google/gemma-2-2b-it"}
    with pytest.raises(StaleSidecarError) as excinfo:
        probes_bootstrap.bootstrap_probes(
            None, None, [], model_info, ["custom-tag"],
        )
    msg = str(excinfo.value)
    assert "default/zz-custom" in msg
    assert "google/gemma-2-2b-it" in msg
    assert "saklas pack refresh" in msg
    assert "SAKLAS_ALLOW_STALE" in msg


def test_bootstrap_allow_stale_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """``SAKLAS_ALLOW_STALE=1`` escape-hatches the staleness check —
    the stale tensor loads, no exception."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.setenv("SAKLAS_ALLOW_STALE", "1")

    cdir = _mk_concept(tmp_path, "default", "zz-custom", ["custom-tag"])
    stmts_path = cdir / "statements.json"
    extraction_time_sha = packs.hash_file(stmts_path)
    _bake_fake_tensor(cdir, "google__gemma-2-2b-it", extraction_time_sha)
    stmts_path.write_text('[["my edit", "after extraction"]]')
    meta = packs.PackMetadata.load(cdir)
    meta.files["statements.json"] = packs.hash_file(stmts_path)
    meta.files["google__gemma-2-2b-it.safetensors"] = packs.hash_file(
        cdir / "google__gemma-2-2b-it.safetensors"
    )
    meta.files["google__gemma-2-2b-it.json"] = packs.hash_file(
        cdir / "google__gemma-2-2b-it.json"
    )
    meta.write(cdir)

    model_info: dict[str, Any] = {"model_id": "google/gemma-2-2b-it"}
    probes = probes_bootstrap.bootstrap_probes(
        None, None, [], model_info, ["custom-tag"],
    )
    assert "zz-custom" in probes


def test_bootstrap_matching_hash_loads_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """When the recorded statements_sha256 matches the live file, the
    cached tensor loads without warning or exception."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.delenv("SAKLAS_ALLOW_STALE", raising=False)

    cdir = _mk_concept(tmp_path, "default", "zz-custom", ["custom-tag"])
    extraction_time_sha = packs.hash_file(cdir / "statements.json")
    _bake_fake_tensor(cdir, "google__gemma-2-2b-it", extraction_time_sha)
    # No edit; hash still matches.

    model_info: dict[str, Any] = {"model_id": "google/gemma-2-2b-it"}
    probes = probes_bootstrap.bootstrap_probes(
        None, None, [], model_info, ["custom-tag"],
    )
    assert "zz-custom" in probes
