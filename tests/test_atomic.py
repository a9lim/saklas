"""Atomic-write helper + crash-recovery semantics for ~/.saklas state."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from saklas.io import packs
from saklas.io.atomic import _temp_path, write_bytes_atomic, write_json_atomic


# Synthetic bundled-vectors fixture — see the matching helper in
# ``tests/test_packs.py``.  4.0 step 6b emptied ``saklas/data/vectors/`` (the
# 26 bundled concepts ship as manifolds now), so ``materialize_bundled()`` has
# no real concept to upgrade.  These tests exercise the upgrade machinery
# (stale ``format_version`` -> v2, user-edited ``statements.json`` preservation
# vs. canonical-equal refresh), so we stand up a synthetic bundled concept and
# redirect the ``saklas.data.vectors`` resource lookup at it.

# A name NOT in ``bundled_manifold_names()`` so nothing routes elsewhere.
_SYNTH = "synthvec.alpha"


def _install_synthetic_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    statements: object,
) -> Path:
    """Build a fake ``saklas/data/vectors`` tree holding one concept and
    redirect ``packs._resources.files`` at it.  ``statements`` is written into
    the synthetic bundled ``statements.json``."""
    bundle_root = tmp_path / "_synth_data_vectors"
    cdir = bundle_root / _SYNTH
    cdir.mkdir(parents=True)
    (cdir / "pack.json").write_text(json.dumps({
        "name": _SYNTH,
        "description": f"synthetic bundled {_SYNTH}",
        "format_version": packs.PACK_FORMAT_VERSION,
        "version": "1.0.0",
        "license": "MIT",
        "tags": ["synthetic"],
        "recommended_alpha": 0.5,
        "source": "bundled",
        "files": {},
    }))
    (cdir / "statements.json").write_text(json.dumps(statements))

    real_files = packs._resources.files

    def _patched_files(anchor: str):
        if anchor == "saklas.data.vectors":
            return bundle_root
        return real_files(anchor)

    monkeypatch.setattr(packs._resources, "files", _patched_files)
    return bundle_root


def test_write_json_atomic_creates_file(tmp_path: Path):
    path = tmp_path / "x.json"
    write_json_atomic(path, {"a": 1, "b": [2, 3]})
    assert path.is_file()
    assert json.loads(path.read_text()) == {"a": 1, "b": [2, 3]}
    # Trailing newline matches the prior json.dump + f.write("\n") convention.
    assert path.read_text().endswith("\n")


def test_write_json_atomic_overwrites(tmp_path: Path):
    path = tmp_path / "x.json"
    write_json_atomic(path, {"v": 1})
    write_json_atomic(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 2}


def test_write_json_atomic_no_orphan_tmp(tmp_path: Path):
    path = tmp_path / "x.json"
    write_json_atomic(path, {"v": 1})
    # Successful write leaves no <path>.tmp behind.
    assert not _temp_path(path).exists()


def test_write_json_atomic_creates_parent(tmp_path: Path):
    path = tmp_path / "nested" / "deep" / "x.json"
    write_json_atomic(path, {"v": 1})
    assert path.is_file()


def test_write_bytes_atomic_basic(tmp_path: Path):
    path = tmp_path / "blob.bin"
    write_bytes_atomic(path, b"\x00\x01\x02")
    assert path.read_bytes() == b"\x00\x01\x02"
    assert not _temp_path(path).exists()


def test_temp_path_with_suffix(tmp_path: Path):
    p = tmp_path / "x.json"
    assert _temp_path(p) == tmp_path / "x.json.tmp"


def test_temp_path_no_suffix(tmp_path: Path):
    p = tmp_path / "Makefile"
    assert _temp_path(p) == tmp_path / "Makefile.tmp"


def test_temp_path_same_directory(tmp_path: Path):
    """Atomicity requires the tempfile sit on the same volume — same dir
    is a sufficient proxy."""
    p = tmp_path / "subdir" / "x.json"
    assert _temp_path(p).parent == p.parent


def _make_concept(tmp_path: Path,  name: str = "happy") -> Path:
    """Build a minimal valid concept folder for ConceptFolder.load tests."""
    d = tmp_path / name
    d.mkdir()
    stmts = d / "statements.json"
    stmts.write_text("[]")
    files = {"statements.json": packs.hash_file(stmts)}
    meta = packs.PackMetadata(
        name=name, description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local", files=files,
    )
    meta.write(d)
    return d


def test_orphan_tmp_does_not_break_concept_folder_load(tmp_path: Path):
    """Crash recovery: a half-written ``pack.json.tmp`` orphan from an
    interrupted atomic write must not cause ConceptFolder.load to fail.
    The prior good ``pack.json`` is still on disk; the orphan is outside
    the manifest's ``files`` map and the loader ignores it.
    """
    d = _make_concept(tmp_path)
    # Simulate a crash mid-write: half-written tempfile sitting next to
    # the good pack.json (no os.replace ever happened).
    orphan = d / "pack.json.tmp"
    orphan.write_text('{"name": "happy", "descript')  # truncated JSON
    cf = packs.ConceptFolder.load(d)
    assert cf.metadata.name == "happy"
    # The orphan is left on disk but harmless.
    assert orphan.exists()


def test_orphan_statements_tmp_ignored(tmp_path: Path):
    """A ``statements.json.tmp`` orphan from an interrupted write doesn't
    confuse the loader either — only files in the manifest matter."""
    d = _make_concept(tmp_path)
    (d / "statements.json.tmp").write_text("[broken")
    cf = packs.ConceptFolder.load(d)
    assert cf.has_statements is True


def test_atomic_overwrite_preserves_prior_on_simulated_crash(tmp_path: Path):
    """If the .tmp file is written but the ``os.replace`` step never lands
    (the canonical crash window), the original file is byte-identical to
    what it was before the write started."""
    path = tmp_path / "x.json"
    write_json_atomic(path, {"version": 1})
    original_bytes = path.read_bytes()

    # Simulate a partial write: stage a new tempfile but don't replace.
    tmp = _temp_path(path)
    tmp.write_text('{"version": 2, "trunc')

    # The "kill" window: tmp exists, original is untouched.
    assert path.read_bytes() == original_bytes
    assert tmp.exists()

    # And the original still loads cleanly.
    assert json.loads(path.read_text()) == {"version": 1}


def test_pack_metadata_future_format_version_message(tmp_path: Path):
    """A pack.json with format_version > PACK_FORMAT_VERSION must raise a
    PackFormatError pointing the user at upgrading saklas (or the
    --force-legacy escape hatch), not at scripts/upgrade_packs.py."""
    d = tmp_path / "future"
    d.mkdir()
    (d / "pack.json").write_text(json.dumps({
        "name": "future",
        "description": "from the future",
        "format_version": packs.PACK_FORMAT_VERSION + 1,
        "version": "1.0.0",
        "license": "MIT",
        "tags": [],
        "recommended_alpha": 0.5,
        "source": "local",
        "files": {},
    }))
    with pytest.raises(packs.PackFormatError) as excinfo:
        packs.ConceptFolder.load(d)
    msg = str(excinfo.value)
    assert "newer saklas" in msg
    assert "--force-legacy" in msg
    assert f"v{packs.PACK_FORMAT_VERSION + 1}" in msg
    assert f"local v{packs.PACK_FORMAT_VERSION}" in msg
    # Old "run upgrade_packs.py" hint is reserved for older-version case.
    assert "upgrade_packs.py" not in msg


def test_pack_metadata_future_format_version_via_pack_metadata_load(tmp_path: Path):
    """Same future-version branch must also raise from PackMetadata.load."""
    d = tmp_path / "future"
    d.mkdir()
    (d / "pack.json").write_text(json.dumps({
        "name": "future",
        "description": "from the future",
        "format_version": 99,
        "version": "1.0.0",
        "license": "MIT",
        "tags": [],
        "recommended_alpha": 0.5,
        "source": "local",
        "files": {},
    }))
    with pytest.raises(packs.PackFormatError, match="--force-legacy"):
        packs.PackMetadata.load(d)


def test_materialize_preserves_user_edited_statements(monkeypatch: pytest.MonkeyPatch,  tmp_path: Path,  caplog: pytest.LogCaptureFixture):
    """When the bundled format_version is stale and the on-disk
    ``statements.json`` differs from the bundled copy (canonical hash),
    materialize_bundled must NOT overwrite statements; only ``pack.json``
    upgrades. A ``pack.json.bak`` is left next to the new pack.json.
    """
    import logging
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    # Synthetic bundled statements that differ (canonically) from the user's.
    _install_synthetic_bundle(
        monkeypatch, tmp_path,
        statements=[["bundled pos", "bundled neg"]],
    )
    concept_dir = tmp_path / "vectors" / "default" / _SYNTH
    concept_dir.mkdir(parents=True)

    # Stale v1 pack.json; gets upgraded.
    stale_pack = concept_dir / "pack.json"
    stale_pack.write_text(
        '{"name": "' + _SYNTH + '", "description": "stale v1", "format_version": 1}'
    )
    stale_pack_text = stale_pack.read_text()

    # User-edited statements.json — guaranteed to differ from the bundled
    # copy because it's a fixed string we control.
    user_stmts = concept_dir / "statements.json"
    user_stmts_payload = '[["my hand-edited", "contrastive pair"]]'
    user_stmts.write_text(user_stmts_payload)

    with caplog.at_level(logging.INFO, logger="saklas.io.packs"):
        packs.materialize_bundled()

    # pack.json upgraded to v2.
    upgraded = json.loads(stale_pack.read_text())
    assert upgraded.get("format_version") == 2
    # User-edited statements left alone.
    assert user_stmts.read_text() == user_stmts_payload
    # .bak preserves the prior pack.json bytes.
    bak = concept_dir / "pack.json.bak"
    assert bak.is_file()
    assert bak.read_text() == stale_pack_text
    # INFO log captured the skip + the upgrade.
    messages = [r.getMessage() for r in caplog.records]
    assert any("preserving user-edited" in m for m in messages)
    assert any(f"upgraded default/{_SYNTH}" in m for m in messages)


def test_materialize_overwrites_unedited_statements(monkeypatch: pytest.MonkeyPatch,  tmp_path: Path):
    """When on-disk statements.json is canonically equal to the bundled
    one, the upgrade is allowed to refresh it — no skip log."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    bundled_payload = [["bundled pos", "bundled neg"], ["pos2", "neg2"]]
    _install_synthetic_bundle(monkeypatch, tmp_path, statements=bundled_payload)
    concept_dir = tmp_path / "vectors" / "default" / _SYNTH
    concept_dir.mkdir(parents=True)
    (concept_dir / "pack.json").write_text(
        '{"name": "' + _SYNTH + '", "description": "stale v1", "format_version": 1}'
    )

    # Mirror the bundled statements canonically (re-formatted to test
    # canonical-hash equivalence: pretty-printed should still compare equal).
    # Re-emit pretty so byte-equality fails but canonical hash matches.
    (concept_dir / "statements.json").write_text(json.dumps(bundled_payload, indent=4))

    packs.materialize_bundled()
    # Should have been refreshed to bundled bytes (no longer indent=4).
    on_disk = json.loads((concept_dir / "statements.json").read_text())
    assert on_disk == bundled_payload
