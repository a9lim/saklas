import json
from pathlib import Path

import pytest

from saklas import packs


def _write_pack(tmp_path: Path, data: dict) -> Path:
    d = tmp_path / "happy"
    d.mkdir()
    (d / "pack.json").write_text(json.dumps(data))
    return d


def test_pack_metadata_parse_minimal(tmp_path):
    folder = _write_pack(tmp_path, {
        "name": "happy",
        "description": "Upbeat.",
        "version": "1.0.0",
        "license": "MIT",
        "tags": ["emotion"],
        "recommended_alpha": 0.5,
        "source": "bundled",
        "files": {},
        "signature": None,
        "signature_method": None,
    })
    meta = packs.PackMetadata.load(folder)
    assert meta.name == "happy"
    assert meta.description == "Upbeat."
    assert meta.tags == ["emotion"]
    assert meta.recommended_alpha == 0.5
    assert meta.source == "bundled"
    assert meta.files == {}
    assert meta.signature is None


def test_pack_metadata_missing_required_field_errors(tmp_path):
    folder = _write_pack(tmp_path, {"description": "no name"})
    with pytest.raises(packs.PackFormatError, match="name"):
        packs.PackMetadata.load(folder)


def test_pack_metadata_invalid_name_rejected(tmp_path):
    folder = _write_pack(tmp_path, {
        "name": "Has_Caps",
        "description": "x", "version": "1", "license": "x",
        "tags": [], "recommended_alpha": 0.5,
        "source": "local", "files": {},
        "signature": None, "signature_method": None,
    })
    with pytest.raises(packs.PackFormatError, match="name"):
        packs.PackMetadata.load(folder)


def test_pack_metadata_long_description_optional(tmp_path):
    folder = _write_pack(tmp_path, {
        "name": "happy",
        "description": "short",
        "long_description": "longer form",
        "version": "1.0.0", "license": "MIT",
        "tags": [], "recommended_alpha": 0.5,
        "source": "bundled", "files": {},
        "signature": None, "signature_method": None,
    })
    meta = packs.PackMetadata.load(folder)
    assert meta.long_description == "longer form"


def test_sidecar_parse_minimal(tmp_path):
    p = tmp_path / "google__gemma-2-2b-it.json"
    p.write_text(json.dumps({
        "method": "contrastive_pca",
        "scores": {"0": 0.02, "14": 0.31},
        "statements_sha256": "abc123",
        "saklas_version": "2.0.0",
    }))
    sc = packs.Sidecar.load(p)
    assert sc.method == "contrastive_pca"
    assert sc.scores == {0: 0.02, 14: 0.31}
    assert sc.statements_sha256 == "abc123"
    assert sc.saklas_version == "2.0.0"
    assert sc.components is None


def test_sidecar_merge_with_components(tmp_path):
    p = tmp_path / "merged.json"
    p.write_text(json.dumps({
        "method": "merge",
        "scores": {"0": 0.5},
        "saklas_version": "2.0.0",
        "components": {
            "default/happy": {"alpha": 0.3, "tensor_sha256": "aa"},
            "user/archaic": {"alpha": 0.4, "tensor_sha256": "bb"},
        },
    }))
    sc = packs.Sidecar.load(p)
    assert sc.method == "merge"
    assert sc.components == {
        "default/happy": {"alpha": 0.3, "tensor_sha256": "aa"},
        "user/archaic": {"alpha": 0.4, "tensor_sha256": "bb"},
    }
    assert sc.statements_sha256 is None


def test_sidecar_write_roundtrip(tmp_path):
    p = tmp_path / "x.json"
    sc = packs.Sidecar(
        method="contrastive_pca",
        scores={0: 0.1, 5: 0.4},
        statements_sha256="hash",
        saklas_version="2.0.0",
    )
    sc.write(p)
    loaded = packs.Sidecar.load(p)
    assert loaded.scores == sc.scores
    assert loaded.statements_sha256 == "hash"


def test_hash_file_sha256(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello")
    # echo -n hello | sha256sum
    assert packs.hash_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_verify_integrity_clean(tmp_path):
    (tmp_path / "statements.json").write_bytes(b"data")
    files = {"statements.json": packs.hash_file(tmp_path / "statements.json")}
    ok, bad = packs.verify_integrity(tmp_path, files)
    assert ok is True
    assert bad == []


def test_verify_integrity_tampered(tmp_path):
    (tmp_path / "statements.json").write_bytes(b"original")
    files = {"statements.json": packs.hash_file(tmp_path / "statements.json")}
    (tmp_path / "statements.json").write_bytes(b"tampered")
    ok, bad = packs.verify_integrity(tmp_path, files)
    assert ok is False
    assert bad == ["statements.json"]


def test_verify_integrity_missing_file(tmp_path):
    files = {"statements.json": "deadbeef"}
    ok, bad = packs.verify_integrity(tmp_path, files)
    assert ok is False
    assert bad == ["statements.json"]
