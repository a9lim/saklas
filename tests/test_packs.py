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
    })
    meta = packs.PackMetadata.load(folder)
    assert meta.name == "happy"
    assert meta.description == "Upbeat."
    assert meta.tags == ["emotion"]
    assert meta.recommended_alpha == 0.5
    assert meta.source == "bundled"
    assert meta.files == {}


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
    })
    meta = packs.PackMetadata.load(folder)
    assert meta.long_description == "longer form"


def test_sidecar_parse_minimal(tmp_path):
    p = tmp_path / "google__gemma-2-2b-it.json"
    p.write_text(json.dumps({
        "method": "contrastive_pca",
        "statements_sha256": "abc123",
        "saklas_version": "2.0.0",
    }))
    sc = packs.Sidecar.load(p)
    assert sc.method == "contrastive_pca"
    assert sc.statements_sha256 == "abc123"
    assert sc.saklas_version == "2.0.0"
    assert sc.components is None


def test_sidecar_merge_with_components(tmp_path):
    p = tmp_path / "merged.json"
    p.write_text(json.dumps({
        "method": "merge",
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
        statements_sha256="hash",
        saklas_version="2.0.0",
    )
    sc.write(p)
    loaded = packs.Sidecar.load(p)
    assert loaded.method == "contrastive_pca"
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


def _make_concept(tmp_path, name="happy", with_statements=True, with_tensor=False):
    d = tmp_path / name
    d.mkdir()
    files = {}
    if with_statements:
        stmts = d / "statements.json"
        stmts.write_text("[]")
        files["statements.json"] = packs.hash_file(stmts)
    if with_tensor:
        t = d / "google__gemma-2-2b-it.safetensors"
        t.write_bytes(b"\x00" * 16)
        files["google__gemma-2-2b-it.safetensors"] = packs.hash_file(t)
        sc = packs.Sidecar(
            method="contrastive_pca",
            saklas_version="2.0.0",
            statements_sha256=files.get("statements.json"),
        )
        sc.write(d / "google__gemma-2-2b-it.json")
        files["google__gemma-2-2b-it.json"] = packs.hash_file(
            d / "google__gemma-2-2b-it.json"
        )
    meta = packs.PackMetadata(
        name=name, description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local", files=files,
    )
    meta.write(d)
    return d


def test_concept_folder_load_statements_only(tmp_path):
    d = _make_concept(tmp_path, with_statements=True, with_tensor=False)
    cf = packs.ConceptFolder.load(d)
    assert cf.metadata.name == "happy"
    assert cf.has_statements is True
    assert cf.tensor_models() == []


def test_concept_folder_load_statements_and_tensor(tmp_path):
    d = _make_concept(tmp_path, with_statements=True, with_tensor=True)
    cf = packs.ConceptFolder.load(d)
    assert cf.has_statements is True
    assert cf.tensor_models() == ["google__gemma-2-2b-it"]
    sc = cf.sidecar("google__gemma-2-2b-it")
    assert sc.method == "contrastive_pca"


def test_concept_folder_load_empty_errors(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    meta = packs.PackMetadata(
        name="empty", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local", files={},
    )
    meta.write(d)
    with pytest.raises(packs.PackFormatError, match="at least one"):
        packs.ConceptFolder.load(d)


def test_concept_folder_load_gguf_only(tmp_path):
    """A concept folder with only a .gguf tensor (no safetensors) should load."""
    pytest.importorskip("gguf")
    import torch
    from saklas.gguf_io import write_gguf_profile

    d = tmp_path / "gguf_only"
    d.mkdir()
    profile = {0: torch.randn(8), 1: torch.randn(8)}
    write_gguf_profile(profile, d / "llama.gguf", model_hint="llama")
    files = {"llama.gguf": packs.hash_file(d / "llama.gguf")}
    packs.PackMetadata(
        name="gguf-only", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local", files=files,
    ).write(d)

    cf = packs.ConceptFolder.load(d)
    assert cf.has_statements is False
    assert cf.tensor_models() == ["llama"]
    assert cf.tensor_format("llama") == "gguf"
    assert cf.tensor_path("llama") == d / "llama.gguf"
    # GGUF tensors have no JSON sidecar.
    with pytest.raises(KeyError):
        cf.sidecar("llama")


def test_concept_folder_prefers_safetensors_over_gguf(tmp_path):
    """Both safetensors and gguf present for the same model → safetensors wins."""
    pytest.importorskip("gguf")
    import torch
    from saklas.gguf_io import write_gguf_profile
    from saklas.vectors import save_profile

    d = tmp_path / "dual"
    d.mkdir()
    profile = {0: torch.randn(8)}
    save_profile(profile, str(d / "llama.safetensors"),
                 {"method": "contrastive_pca"})
    write_gguf_profile(profile, d / "llama.gguf", model_hint="llama")

    files = {
        "llama.safetensors": packs.hash_file(d / "llama.safetensors"),
        "llama.json": packs.hash_file(d / "llama.json"),
        "llama.gguf": packs.hash_file(d / "llama.gguf"),
    }
    packs.PackMetadata(
        name="dual", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local", files=files,
    ).write(d)
    cf = packs.ConceptFolder.load(d)
    assert cf.tensor_format("llama") == "safetensors"
    assert cf.tensor_path("llama").suffix == ".safetensors"
    # Sidecar works because we picked the safetensors side.
    assert cf.sidecar("llama").method == "contrastive_pca"


def test_concept_folder_load_tampered_errors(tmp_path):
    d = _make_concept(tmp_path, with_statements=True, with_tensor=False)
    (d / "statements.json").write_text("[{}]")  # mutate, breaks hash
    with pytest.raises(packs.PackFormatError, match="integrity"):
        packs.ConceptFolder.load(d)


def test_is_stale_statements_changed(tmp_path):
    d = _make_concept(tmp_path, with_statements=True, with_tensor=True)
    cf = packs.ConceptFolder.load(d)
    sc = cf.sidecar("google__gemma-2-2b-it")
    stale = packs.is_stale(
        current_statements_sha=packs.hash_file(d / "statements.json"),
        sidecar=sc,
    )
    assert stale is False
    stale = packs.is_stale(current_statements_sha="deadbeef", sidecar=sc)
    assert stale is True


def test_is_stale_no_statements():
    sc = packs.Sidecar(method="merge", saklas_version="2.0.0")
    assert packs.is_stale(current_statements_sha=None, sidecar=sc) is False


def test_version_mismatch_detection():
    sc = packs.Sidecar(method="contrastive_pca", saklas_version="1.9.9")
    assert packs.version_mismatch(sc, current="2.0.0") is True
    sc2 = packs.Sidecar(method="contrastive_pca", saklas_version="2.0.3")
    assert packs.version_mismatch(sc2, current="2.0.0") is False
    sc3 = packs.Sidecar(method="contrastive_pca", saklas_version="2.1.0")
    assert packs.version_mismatch(sc3, current="2.0.0") is True


def test_save_load_profile_roundtrip_slim_sidecar(tmp_path):
    import torch
    from saklas.vectors import save_profile, load_profile
    profile = {
        0: torch.randn(8),
        14: torch.randn(8),
    }
    path = tmp_path / "google__gemma-2-2b-it.safetensors"
    save_profile(profile, str(path), {
        "method": "contrastive_pca",
        "statements_sha256": "abc",
    })
    loaded, meta = load_profile(str(path))
    assert sorted(loaded.keys()) == [0, 14]
    assert meta["method"] == "contrastive_pca"
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


def test_bundled_concept_names_includes_known():
    names = packs.bundled_concept_names()
    # `agentic` is a stable name across pre- and post-regen layouts.
    assert "agentic" in names
    assert len(names) >= 1


def test_materialize_empty_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    packs.materialize_bundled()
    assert (tmp_path / "neutral_statements.json").is_file()
    assert (tmp_path / "vectors" / "default" / "agentic" / "pack.json").is_file()


def test_materialize_does_not_overwrite(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    target = tmp_path / "vectors" / "default" / "agentic" / "pack.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"user": "edited"}')
    packs.materialize_bundled()
    assert target.read_text() == '{"user": "edited"}'


def test_materialize_partial_fills_gaps(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    # Pre-create `agentic` with user edits; materialization should still
    # populate other bundled packs (at least `angry.calm` in the current
    # data dir) without overwriting agentic.
    (tmp_path / "vectors" / "default" / "agentic").mkdir(parents=True)
    (tmp_path / "vectors" / "default" / "agentic" / "pack.json").write_text("{}")
    packs.materialize_bundled()
    assert (tmp_path / "vectors" / "default" / "angry.calm" / "pack.json").is_file()
    assert (tmp_path / "vectors" / "default" / "agentic" / "pack.json").read_text() == "{}"


