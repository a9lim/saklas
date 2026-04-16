
import pytest

from saklas.cli import config_file as cfg


def test_parse_minimal(tmp_path):
    p = tmp_path / "setup.yaml"
    p.write_text("model: google/gemma-2-2b-it\n")
    c = cfg.ConfigFile.load(p)
    assert c.model == "google/gemma-2-2b-it"
    assert c.vectors == {}
    assert c.thinking is None


def test_parse_full(tmp_path):
    p = tmp_path / "setup.yaml"
    p.write_text("""
model: google/gemma-2-2b-it
vectors:
  default/happy: 0.4
  a9lim/calm: 0.3
thinking: true
temperature: 0.9
top_p: 0.95
max_tokens: 512
system_prompt: "You are helpful."
""".strip())
    c = cfg.ConfigFile.load(p)
    assert c.model == "google/gemma-2-2b-it"
    assert c.vectors == {"default/happy": 0.4, "a9lim/calm": 0.3}
    assert c.thinking is True
    assert c.temperature == 0.9
    assert c.top_p == 0.95
    assert c.max_tokens == 512
    assert c.system_prompt == "You are helpful."


def test_parse_unknown_keys_warn_but_accept(tmp_path, caplog):
    p = tmp_path / "setup.yaml"
    p.write_text("model: x\nsomething_new: 1\n")
    import logging
    caplog.set_level(logging.WARNING, logger="saklas.cli.config_file")
    c = cfg.ConfigFile.load(p)
    assert c.model == "x"
    assert any("unknown" in r.message for r in caplog.records)


def test_parse_invalid_types_raises(tmp_path):
    p = tmp_path / "setup.yaml"
    p.write_text("vectors:\n  default/happy: not-a-number\n")
    with pytest.raises(cfg.ConfigFileError, match="alpha"):
        cfg.ConfigFile.load(p)


def test_compose_later_overrides_earlier():
    a = cfg.ConfigFile(model="x", temperature=0.5)
    b = cfg.ConfigFile(model="y", top_p=0.9)
    out = cfg.compose([a, b])
    assert out.model == "y"
    assert out.temperature == 0.5
    assert out.top_p == 0.9


def test_compose_vectors_merge():
    a = cfg.ConfigFile(vectors={"default/happy": 0.3})
    b = cfg.ConfigFile(vectors={"default/calm": 0.5, "default/happy": 0.4})
    out = cfg.compose([a, b])
    assert out.vectors == {"default/happy": 0.4, "default/calm": 0.5}


def test_apply_flag_overrides():
    c = cfg.ConfigFile(model="from-yaml", temperature=0.5)
    out = cfg.apply_flag_overrides(c, model="from-flag", temperature=None, max_tokens=256)
    assert out.model == "from-flag"
    assert out.temperature == 0.5
    assert out.max_tokens == 256


def test_ensure_vectors_installed_all_present(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    d = tmp_path / "vectors" / "default" / "happy"
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="happy", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="bundled",
        files={"statements.json": packs.hash_file(d / "statements.json")},
    ).write(d)

    c = cfg.ConfigFile(vectors={"default/happy": 0.5})
    missing = cfg.ensure_vectors_installed(c, strict=False)
    assert missing == []


def test_ensure_vectors_installed_missing_hf(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    installed = {}

    def fake_install(target, as_=None, force=False):
        installed["target"] = target
        return tmp_path / "vectors" / "user" / "happy"

    monkeypatch.setattr("saklas.io.cache_ops.install", fake_install)
    c = cfg.ConfigFile(vectors={"user/happy": 0.5})
    missing = cfg.ensure_vectors_installed(c, strict=False)
    assert installed["target"] == "user/happy"
    assert missing == []


def test_ensure_vectors_installed_strict_raises_on_local_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    c = cfg.ConfigFile(vectors={"local/bard": 0.5})
    with pytest.raises(cfg.ConfigFileError, match="local/bard"):
        cfg.ensure_vectors_installed(c, strict=True)


def test_load_default_returns_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    assert cfg.ConfigFile.load_default() is None


def test_load_default_returns_file_when_present(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model: default-model\n")
    c = cfg.ConfigFile.load_default()
    assert c is not None
    assert c.model == "default-model"


def test_effective_composes_default_and_extras(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model: default-model\ntemperature: 0.5\n")
    extra = tmp_path / "extra.yaml"
    extra.write_text("model: extra-model\ntop_p: 0.9\n")
    c = cfg.ConfigFile.effective([extra])
    assert c.model == "extra-model"  # extra overrides default
    assert c.temperature == 0.5  # inherited from default
    assert c.top_p == 0.9


def test_effective_no_default(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model: default-model\n")
    c = cfg.ConfigFile.effective([], include_default=False)
    assert c.model is None


def test_to_yaml_roundtrip(tmp_path):
    c = cfg.ConfigFile(model="x", temperature=0.7, vectors={"default/happy.sad": 0.3})
    y = c.to_yaml(header="# header")
    assert y.startswith("# header\n")
    assert "model: x" in y
    assert "happy.sad" in y


def test_resolve_poles_bare_name(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    d = tmp_path / "vectors" / "local" / "deer.wolf"
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="deer.wolf", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(d / "statements.json")},
    ).write(d)
    from saklas.cli.selectors import invalidate
    invalidate()

    c = cfg.ConfigFile(vectors={"wolf": 0.5})
    resolved = c.resolve_poles()
    assert resolved.vectors == {"deer.wolf": -0.5}
