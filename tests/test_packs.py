from typing import Any
import json
from pathlib import Path

import pytest

from saklas.io import packs


def _write_pack(tmp_path: Path,  data: dict[str, Any]) -> Path:
    d = tmp_path / "happy"
    d.mkdir()
    (d / "pack.json").write_text(json.dumps(data))
    return d


def test_pack_metadata_parse_minimal(tmp_path: Path):
    folder = _write_pack(tmp_path, {
        "name": "happy",
        "description": "Upbeat.",
        "format_version": 2,
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


def test_pack_metadata_missing_required_field_errors(tmp_path: Path):
    folder = _write_pack(tmp_path, {"description": "no name"})
    with pytest.raises(packs.PackFormatError, match="name"):
        packs.PackMetadata.load(folder)


def test_pack_metadata_invalid_name_rejected(tmp_path: Path):
    folder = _write_pack(tmp_path, {
        "name": "Has_Caps",
        "description": "x", "format_version": 2, "version": "1", "license": "x",
        "tags": [], "recommended_alpha": 0.5,
        "source": "local", "files": {},
    })
    with pytest.raises(packs.PackFormatError, match="name"):
        packs.PackMetadata.load(folder)


def test_pack_metadata_long_description_optional(tmp_path: Path):
    folder = _write_pack(tmp_path, {
        "name": "happy",
        "description": "short",
        "long_description": "longer form",
        "format_version": 2,
        "version": "1.0.0", "license": "MIT",
        "tags": [], "recommended_alpha": 0.5,
        "source": "bundled", "files": {},
    })
    meta = packs.PackMetadata.load(folder)
    assert meta.long_description == "longer form"


def test_sidecar_parse_minimal(tmp_path: Path):
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


def test_sidecar_merge_with_components(tmp_path: Path):
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


def test_sidecar_write_roundtrip(tmp_path: Path):
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


def test_hash_file_sha256(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello")
    # echo -n hello | sha256sum
    assert packs.hash_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


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


def _make_concept(tmp_path: Path,  name: Any="happy",  with_statements: Any=True,  with_tensor: Any=False):
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


def test_concept_folder_load_statements_only(tmp_path: Path):
    d = _make_concept(tmp_path, with_statements=True, with_tensor=False)
    cf = packs.ConceptFolder.load(d)
    assert cf.metadata.name == "happy"
    assert cf.has_statements is True
    assert cf.tensor_models() == []


def test_concept_folder_load_statements_and_tensor(tmp_path: Path):
    d = _make_concept(tmp_path, with_statements=True, with_tensor=True)
    cf = packs.ConceptFolder.load(d)
    assert cf.has_statements is True
    assert cf.tensor_models() == ["google__gemma-2-2b-it"]
    sc = cf.sidecar("google__gemma-2-2b-it")
    assert sc.method == "contrastive_pca"


def test_concept_folder_load_empty_errors(tmp_path: Path):
    d = tmp_path / "empty"
    d.mkdir()
    meta = packs.PackMetadata(
        name="empty", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local", files={},
    )
    meta.write(d)
    with pytest.raises(packs.PackFormatError, match="at least one"):
        packs.ConceptFolder.load(d)


def test_concept_folder_load_gguf_only(tmp_path: Path):
    """A concept folder with only a .gguf tensor (no safetensors) should load."""
    pytest.importorskip("gguf")
    import torch
    from saklas.io.gguf_io import write_gguf_profile

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


def test_concept_folder_prefers_safetensors_over_gguf(tmp_path: Path):
    """Both safetensors and gguf present for the same model → safetensors wins."""
    pytest.importorskip("gguf")
    import torch
    from saklas.io.gguf_io import write_gguf_profile
    from saklas.core.vectors import save_profile

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


def test_concept_folder_load_tampered_errors(tmp_path: Path):
    d = _make_concept(tmp_path, with_statements=True, with_tensor=False)
    (d / "statements.json").write_text("[{}]")  # mutate, breaks hash
    with pytest.raises(packs.PackFormatError, match="integrity"):
        packs.ConceptFolder.load(d)


def test_is_stale_statements_changed(tmp_path: Path):
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


def test_save_load_profile_roundtrip_slim_sidecar(tmp_path: Path):
    import torch
    from saklas.core.vectors import save_profile, load_profile
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


def test_materialize_empty_home(monkeypatch: pytest.MonkeyPatch,  tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    packs.materialize_bundled()
    assert (tmp_path / "neutral_statements.json").is_file()
    assert (tmp_path / "vectors" / "default" / "agentic" / "pack.json").is_file()


def test_materialize_does_not_overwrite(monkeypatch: pytest.MonkeyPatch,  tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    target = tmp_path / "vectors" / "default" / "agentic" / "pack.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"user": "edited"}')
    packs.materialize_bundled()
    assert target.read_text() == '{"user": "edited"}'


def test_materialize_upgrades_stale_bundled(monkeypatch: pytest.MonkeyPatch,  tmp_path: Path):
    """Existing bundled folder with an explicit v1 format_version gets
    upgraded in place on materialize_bundled — the hard-break migration
    path for users who had saklas 1.x installed before the Profile refactor.
    Any per-model tensor files already on disk stay untouched (they don't
    exist here; the shipped bundled dir only contains pack.json +
    statements.json)."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    concept_dir = tmp_path / "vectors" / "default" / "agentic"
    concept_dir.mkdir(parents=True)
    stale_pack = concept_dir / "pack.json"
    # Minimal v1-shaped pack.json — missing files block, but carries an
    # explicit format_version=1 that identifies it as a stale install.
    stale_pack.write_text(
        '{"name": "agentic", "description": "stale v1", "format_version": 1}'
    )
    fake_tensor = concept_dir / "stale_model.safetensors"
    fake_tensor.write_bytes(b"\x00" * 8)
    packs.materialize_bundled()
    # pack.json is now the shipped v2.
    upgraded = json.loads(stale_pack.read_text())
    assert upgraded.get("format_version") == 2
    assert upgraded.get("description") != "stale v1"
    # statements.json got copied across.
    assert (concept_dir / "statements.json").is_file()
    # Pre-existing per-model tensor files are left alone.
    assert fake_tensor.is_file()
    assert fake_tensor.read_bytes() == b"\x00" * 8


def test_materialize_partial_fills_gaps(monkeypatch: pytest.MonkeyPatch,  tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    # Pre-create `agentic` with user edits; materialization should still
    # populate other bundled packs (at least `angry.calm` in the current
    # data dir) without overwriting agentic.
    (tmp_path / "vectors" / "default" / "agentic").mkdir(parents=True)
    (tmp_path / "vectors" / "default" / "agentic" / "pack.json").write_text("{}")
    packs.materialize_bundled()
    assert (tmp_path / "vectors" / "default" / "angry.calm" / "pack.json").is_file()
    assert (tmp_path / "vectors" / "default" / "agentic" / "pack.json").read_text() == "{}"


def test_variants_for_concept_raw_and_sae(tmp_path: Path):
    """enumerate_variants returns both raw and sae-* tensors in a folder."""
    from saklas.io.packs import enumerate_variants

    folder = tmp_path / "honest.deceptive"
    folder.mkdir()
    # Raw
    (folder / "google__gemma-2-2b-it.safetensors").write_bytes(b"")
    (folder / "google__gemma-2-2b-it.json").write_text("{}")
    # SAE
    (folder / "google__gemma-2-2b-it_sae-gemma-scope-2b-pt-res-canonical.safetensors").write_bytes(b"")
    (folder / "google__gemma-2-2b-it_sae-gemma-scope-2b-pt-res-canonical.json").write_text("{}")
    # Different model — must be excluded
    (folder / "mistralai__ministral-8b.safetensors").write_bytes(b"")

    result = enumerate_variants(folder, "google/gemma-2-2b-it")
    assert set(result.keys()) == {"raw", "sae-gemma-scope-2b-pt-res-canonical"}
    assert result["raw"].name == "google__gemma-2-2b-it.safetensors"
    assert result["sae-gemma-scope-2b-pt-res-canonical"].name.endswith(
        "_sae-gemma-scope-2b-pt-res-canonical.safetensors"
    )


def test_enumerate_variants_empty_folder(tmp_path: Path):
    from saklas.io.packs import enumerate_variants
    folder = tmp_path / "empty"
    folder.mkdir()
    assert enumerate_variants(folder, "google/gemma-2-2b-it") == {}


def test_enumerate_variants_nonexistent_folder(tmp_path: Path):
    from saklas.io.packs import enumerate_variants
    # Non-existent folder returns empty dict (not an error).
    assert enumerate_variants(tmp_path / "does-not-exist", "any/model") == {}


def test_save_profile_writes_sae_sidecar_fields(tmp_path: Path):
    """save_profile with sae metadata writes sae_release/sae_ids_by_layer to sidecar."""
    import json
    import torch
    from saklas.core.vectors import save_profile

    profile = {3: torch.zeros(4), 7: torch.zeros(4)}
    target = tmp_path / "x_sae-mock.safetensors"
    save_profile(profile, str(target), {
        "method": "pca_center_sae",
        "sae_release": "mock-release",
        "sae_revision": None,
        "sae_ids_by_layer": {"3": "layer_3/mock", "7": "layer_7/mock"},
    })

    with open(tmp_path / "x_sae-mock.json") as f:
        sidecar = json.load(f)
    assert sidecar["method"] == "pca_center_sae"
    assert sidecar["sae_release"] == "mock-release"
    assert sidecar["sae_revision"] is None
    assert sidecar["sae_ids_by_layer"] == {"3": "layer_3/mock", "7": "layer_7/mock"}
    assert sidecar["format_version"] == 2


def test_save_profile_omits_sae_fields_when_absent(tmp_path: Path):
    """Raw-PCA sidecars stay clean — no empty/null SAE fields."""
    import json
    import torch
    from saklas.core.vectors import save_profile

    profile = {0: torch.zeros(4)}
    target = tmp_path / "x.safetensors"
    save_profile(profile, str(target), {"method": "contrastive_pca"})
    with open(tmp_path / "x.json") as f:
        sidecar = json.load(f)
    assert "sae_release" not in sidecar
    assert "sae_revision" not in sidecar
    assert "sae_ids_by_layer" not in sidecar


