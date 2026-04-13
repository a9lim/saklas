from pathlib import Path

import pytest

from saklas import config_file as cfg


def test_parse_minimal(tmp_path):
    p = tmp_path / "setup.yaml"
    p.write_text("model: google/gemma-2-2b-it\n")
    c = cfg.ConfigFile.load(p)
    assert c.model == "google/gemma-2-2b-it"
    assert c.vectors == {}
    assert c.orthogonalize is None


def test_parse_full(tmp_path):
    p = tmp_path / "setup.yaml"
    p.write_text("""
model: google/gemma-2-2b-it
vectors:
  default/happy: 0.4
  a9lim/calm: 0.3
orthogonalize: true
thinking: true
temperature: 0.9
top_p: 0.95
max_tokens: 512
system_prompt: "You are helpful."
""".strip())
    c = cfg.ConfigFile.load(p)
    assert c.model == "google/gemma-2-2b-it"
    assert c.vectors == {"default/happy": 0.4, "a9lim/calm": 0.3}
    assert c.orthogonalize is True
    assert c.thinking is True
    assert c.temperature == 0.9
    assert c.top_p == 0.95
    assert c.max_tokens == 512
    assert c.system_prompt == "You are helpful."


def test_parse_unknown_keys_warn_but_accept(tmp_path, caplog):
    p = tmp_path / "setup.yaml"
    p.write_text("model: x\nsomething_new: 1\n")
    import logging
    caplog.set_level(logging.WARNING, logger="saklas.config_file")
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
    from saklas import packs
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

    monkeypatch.setattr("saklas.cache_ops.install", fake_install)
    c = cfg.ConfigFile(vectors={"user/happy": 0.5})
    missing = cfg.ensure_vectors_installed(c, strict=False)
    assert installed["target"] == "user/happy"
    assert missing == []


def test_ensure_vectors_installed_strict_raises_on_local_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    c = cfg.ConfigFile(vectors={"local/bard": 0.5})
    with pytest.raises(cfg.ConfigFileError, match="local/bard"):
        cfg.ensure_vectors_installed(c, strict=True)
