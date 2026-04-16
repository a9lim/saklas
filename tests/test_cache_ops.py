
import pytest

from saklas.io import cache_ops, packs
from saklas.cli import selectors as sel


def _mk(home, ns, name, models=(), tags=(), source="local"):
    d = home / "vectors" / ns / name
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    files = {"statements.json": packs.hash_file(d / "statements.json")}
    for m in models:
        ts = d / f"{m}.safetensors"
        ts.write_bytes(b"\x00" * 8)
        sc = packs.Sidecar(method="contrastive_pca", saklas_version="2.0.0")
        sc.write(d / f"{m}.json")
        files[f"{m}.safetensors"] = packs.hash_file(ts)
        files[f"{m}.json"] = packs.hash_file(d / f"{m}.json")
    packs.PackMetadata(
        name=name, description="x", version="1.0.0", license="MIT",
        tags=list(tags), recommended_alpha=0.5, source=source,
        files=files,
    ).write(d)
    return d


def test_delete_single_concept_tensors(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy", models=["gemma", "qwen"])
    n = cache_ops.delete_tensors(sel.parse("happy"), model_scope=None)
    assert n == 4
    d = tmp_path / "vectors" / "default" / "happy"
    assert (d / "statements.json").exists()
    assert not list(d.glob("*.safetensors"))


def test_delete_with_model_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy", models=["gemma", "qwen"])
    n = cache_ops.delete_tensors(sel.parse("happy"), model_scope="gemma")
    assert n == 2
    d = tmp_path / "vectors" / "default" / "happy"
    assert not (d / "gemma.safetensors").exists()
    assert (d / "qwen.safetensors").exists()


def test_delete_tag_scoped(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy", models=["gemma"], tags=["emotion"])
    _mk(tmp_path, "default", "calm", models=["gemma"], tags=["emotion"])
    _mk(tmp_path, "default", "honest", models=["gemma"], tags=["personality"])
    n = cache_ops.delete_tensors(sel.parse("tag:emotion"), model_scope=None)
    assert n == 4
    assert not (tmp_path / "vectors" / "default" / "happy" / "gemma.safetensors").exists()
    assert (tmp_path / "vectors" / "default" / "honest" / "gemma.safetensors").exists()


def test_delete_all(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy", models=["gemma"])
    _mk(tmp_path, "a9lim", "archaic", models=["gemma"])
    n = cache_ops.delete_tensors(sel.parse("all"), model_scope=None)
    assert n == 4
    assert (tmp_path / "vectors" / "default" / "happy" / "statements.json").exists()


def test_install_local_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    src = tmp_path / "src" / "archaic"
    src.mkdir(parents=True)
    (src / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="archaic", description="x", version="1.0.0", license="MIT",
        tags=["style"], recommended_alpha=0.4, source="local",
        files={"statements.json": packs.hash_file(src / "statements.json")},
    ).write(src)
    target = cache_ops.install_folder(src, namespace="local", as_=None)
    assert target == tmp_path / "home" / "vectors" / "local" / "archaic"
    assert (target / "pack.json").is_file()
    assert (target / "statements.json").is_file()


def test_install_conflict_without_force(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    src = tmp_path / "src" / "archaic"
    src.mkdir(parents=True)
    (src / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="archaic", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(src / "statements.json")},
    ).write(src)
    cache_ops.install_folder(src, namespace="local", as_=None)
    with pytest.raises(cache_ops.InstallConflict):
        cache_ops.install_folder(src, namespace="local", as_=None)


def test_install_force_overwrites(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    src = tmp_path / "src" / "archaic"
    src.mkdir(parents=True)
    (src / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="archaic", description="first", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(src / "statements.json")},
    ).write(src)
    cache_ops.install_folder(src, namespace="local", as_=None)
    packs.PackMetadata(
        name="archaic", description="second", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(src / "statements.json")},
    ).write(src)
    cache_ops.install_folder(src, namespace="local", as_=None, force=True)
    m = packs.PackMetadata.load(tmp_path / "home" / "vectors" / "local" / "archaic")
    assert m.description == "second"


def test_install_as_relocates(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    src = tmp_path / "src" / "archaic"
    src.mkdir(parents=True)
    (src / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="archaic", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(src / "statements.json")},
    ).write(src)
    target = cache_ops.install_folder(src, namespace="local", as_="community/ancient")
    assert target == tmp_path / "home" / "vectors" / "community" / "ancient"
    m = packs.PackMetadata.load(target)
    assert m.name == "ancient"


def test_refresh_bundled_restores_statements(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    packs.materialize_bundled()
    d = tmp_path / "vectors" / "default" / "angry.calm"
    if not (d / "statements.json").exists():
        import pytest
        pytest.skip("angry.calm statements.json not yet regenerated")
    original = (d / "statements.json").read_text()
    (d / "statements.json").write_text("[{}]")
    cache_ops.refresh(sel.parse("angry.calm"))
    assert (d / "statements.json").read_text() == original


def test_refresh_local_skipped(monkeypatch, tmp_path):
    """Locals have no upstream to re-pull from, so refresh skips them
    silently — this keeps `-r all` working when the user has their own
    vectors in the cache."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "local", "bard", models=[])
    assert cache_ops.refresh(sel.parse("local/bard")) == 0


def test_install_hf_routes_to_pull_pack(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    called = {}

    def fake_pull(coord, target_folder, *, force, revision=None):
        called["coord"] = coord
        called["target"] = target_folder
        called["force"] = force
        called["revision"] = revision
        target_folder.mkdir(parents=True, exist_ok=True)
        (target_folder / "statements.json").write_text("[]")
        src = f"hf://{coord}@{revision}" if revision else f"hf://{coord}"
        packs.PackMetadata(
            name="happy", description="x", version="1.0.0", license="MIT",
            tags=["emotion"], recommended_alpha=0.5,
            source=src,
            files={"statements.json": packs.hash_file(target_folder / "statements.json")},
        ).write(target_folder)
        return target_folder

    monkeypatch.setattr("saklas.io.hf.pull_pack", fake_pull)
    cache_ops.install("user/happy", as_=None, force=False)
    assert called["coord"] == "user/happy"
    assert called["revision"] is None
    assert called["target"] == tmp_path / "home" / "vectors" / "user" / "happy"


def test_install_hf_with_revision_pins_source(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    called = {}

    def fake_pull(coord, target_folder, *, force, revision=None):
        called["coord"] = coord
        called["revision"] = revision
        target_folder.mkdir(parents=True, exist_ok=True)
        (target_folder / "statements.json").write_text("[]")
        src = f"hf://{coord}@{revision}" if revision else f"hf://{coord}"
        packs.PackMetadata(
            name="happy", description="x", version="1.0.0", license="MIT",
            tags=["emotion"], recommended_alpha=0.5,
            source=src,
            files={"statements.json": packs.hash_file(target_folder / "statements.json")},
        ).write(target_folder)
        return target_folder

    monkeypatch.setattr("saklas.io.hf.pull_pack", fake_pull)
    result = cache_ops.install("user/happy@v1.2.0", as_=None, force=False)
    assert called["coord"] == "user/happy"
    assert called["revision"] == "v1.2.0"
    m = packs.PackMetadata.load(result)
    assert m.source == "hf://user/happy@v1.2.0"


def test_refresh_pinned_hf_source_passes_revision(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "user", "happy", source="hf://user/happy@v1.2.0")
    called = {}

    def fake_pull(coord, target_folder, *, force, revision=None):
        called["coord"] = coord
        called["revision"] = revision
        called["force"] = force
        return target_folder

    monkeypatch.setattr("saklas.io.hf.pull_pack", fake_pull)
    cache_ops.refresh(sel.parse("user/happy"))
    assert called["coord"] == "user/happy"
    assert called["revision"] == "v1.2.0"
    assert called["force"] is True


def test_list_local_all(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy", tags=["emotion"])
    _mk(tmp_path, "a9lim", "archaic", tags=["style"])
    cache_ops.list_concepts(selector=None, hf=False)
    out = capsys.readouterr().out
    assert "happy" in out and "[installed]" in out
    assert "archaic" in out


def test_list_local_info_mode(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy", tags=["emotion"], models=["gemma"])
    cache_ops.list_concepts(selector=sel.parse("default/happy"), hf=False)
    out = capsys.readouterr().out
    assert "happy" in out
    assert "emotion" in out
    assert "gemma" in out


def test_list_concepts_includes_hf_rows(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy", tags=["emotion"])
    monkeypatch.setattr(
        "saklas.io.hf.search_packs",
        lambda selector: [{
            "name": "calm",
            "namespace": "other",
            "description": "zen",
            "tags": ["emotion"],
            "recommended_alpha": 0.3,
        }],
    )
    cache_ops.list_concepts(selector=sel.parse("tag:emotion"), hf=True)
    out = capsys.readouterr().out
    assert "happy" in out
    assert "[installed]" in out
    assert "calm" in out
    assert "[hf]" in out
