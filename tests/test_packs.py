"""Surviving pack-format primitives (4.0 collapse).

The pack *format/distribution* surface (``PackMetadata`` / ``Sidecar`` /
``ConceptFolder`` / ``enumerate_variants`` / ``materialize_bundled`` /
``is_stale`` / ``version_mismatch`` / ``bundled_concept_names``) was retired
when concepts collapsed into manifolds.  What remains in ``io.packs`` is the
cross-cutting integrity infra — ``hash_file`` / ``verify_integrity`` — plus the
``save_profile`` / ``load_profile`` tensor-dict cache helpers (owned by
``core.profile`` and still aliased from ``core.vectors``).  This module covers
exactly those.
"""
from pathlib import Path

from saklas.io import packs


def test_hash_file_sha256(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello")
    # echo -n hello | sha256sum
    assert packs.hash_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_hash_folder_files_skips_pack_json(tmp_path: Path):
    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "pack.json").write_bytes(b"{}")
    out = packs.hash_folder_files(tmp_path)
    assert set(out) == {"a.safetensors"}
    assert out["a.safetensors"] == packs.hash_file(tmp_path / "a.safetensors")


def test_verify_integrity_clean(tmp_path: Path):
    (tmp_path / "statements.json").write_bytes(b"data")
    files = {"statements.json": packs.hash_file(tmp_path / "statements.json")}
    ok, bad = packs.verify_integrity(tmp_path, files)
    assert ok is True
    assert bad == []


def test_verify_integrity_tampered(tmp_path: Path):
    (tmp_path / "statements.json").write_bytes(b"original")
    files = {"statements.json": packs.hash_file(tmp_path / "statements.json")}
    (tmp_path / "statements.json").write_bytes(b"tampered")
    ok, bad = packs.verify_integrity(tmp_path, files)
    assert ok is False
    assert bad == ["statements.json"]


def test_verify_integrity_missing_file(tmp_path: Path):
    files = {"statements.json": "deadbeef"}
    ok, bad = packs.verify_integrity(tmp_path, files)
    assert ok is False
    assert bad == ["statements.json"]


def test_verify_integrity_detects_same_size_rewrite_with_restored_mtime(
    tmp_path: Path,
) -> None:
    import os

    target = tmp_path / "x.bin"
    target.write_bytes(b"AAAA")
    expected = packs.hash_file(target)
    assert packs.verify_integrity(tmp_path, {"x.bin": expected}) == (True, [])
    original = target.stat()
    target.write_bytes(b"BBBB")
    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
    ok, bad = packs.verify_integrity(tmp_path, {"x.bin": expected})
    assert not ok
    assert bad == ["x.bin"]


def test_verify_integrity_rejects_path_traversal(tmp_path: Path):
    # A manifest entry resolving outside the folder fails rather than reading
    # off-tree (ensure_within is the path-traversal barrier).
    ok, bad = packs.verify_integrity(tmp_path, {"../escape": "deadbeef"})
    assert ok is False
    assert bad == ["../escape"]


def test_save_load_profile_roundtrip_slim_sidecar(tmp_path: Path):
    import torch
    from saklas.core.vectors import save_profile, load_profile
    profile = {
        0: torch.randn(8),
        14: torch.randn(8),
    }
    path = tmp_path / "google__gemma-2-2b-it.safetensors"
    save_profile(profile, str(path), {
        "method": "difference_of_means",
        "statements_sha256": "abc",
    })
    loaded, meta = load_profile(str(path))
    assert sorted(loaded.keys()) == [0, 14]
    assert meta["method"] == "difference_of_means"
    assert meta["statements_sha256"] == "abc"
    assert "saklas_version" in meta
    # Scores no longer live on disk — shares are baked into tensor magnitudes.
    assert "scores" not in meta
    # No legacy keys:
    assert "concept" not in meta
    assert "model_id" not in meta
    assert "num_pairs" not in meta
    # Round-trip preserves tensor values bit-for-bit.
    for idx in profile:
        assert torch.allclose(profile[idx], loaded[idx])
