"""ConfigFile loading + composition — YAML accepts a single steering
expression string for the ``vectors:`` key.
"""

import pytest

from saklas.cli import config_file as cfg


def test_parse_minimal(tmp_path):
    p = tmp_path / "setup.yaml"
    p.write_text("model: google/gemma-2-2b-it\n")
    c = cfg.ConfigFile.load(p)
    assert c.model == "google/gemma-2-2b-it"
    assert c.vectors is None
    assert c.thinking is None


def test_parse_full(tmp_path):
    p = tmp_path / "setup.yaml"
    p.write_text("""
model: google/gemma-2-2b-it
vectors: "0.4 default/happy + 0.3 a9lim/calm"
thinking: true
temperature: 0.9
top_p: 0.95
max_tokens: 512
system_prompt: "You are helpful."
""".strip())
    c = cfg.ConfigFile.load(p)
    assert c.model == "google/gemma-2-2b-it"
    assert c.vectors == "0.4 default/happy + 0.3 a9lim/calm"
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


def test_parse_rejects_map_form(tmp_path):
    p = tmp_path / "setup.yaml"
    p.write_text("vectors:\n  default/happy: 0.5\n")
    with pytest.raises(cfg.ConfigFileError, match="expression string"):
        cfg.ConfigFile.load(p)


def test_parse_invalid_expression_raises(tmp_path):
    p = tmp_path / "setup.yaml"
    p.write_text('vectors: "0.5 default/happy +"\n')
    with pytest.raises(cfg.ConfigFileError, match="vectors"):
        cfg.ConfigFile.load(p)


def test_parse_empty_expression_is_none(tmp_path):
    p = tmp_path / "setup.yaml"
    p.write_text('vectors: ""\n')
    c = cfg.ConfigFile.load(p)
    assert c.vectors is None


def test_compose_later_overrides_earlier():
    a = cfg.ConfigFile(model="x", temperature=0.5)
    b = cfg.ConfigFile(model="y", top_p=0.9)
    out = cfg.compose([a, b])
    assert out.model == "y"
    assert out.temperature == 0.5
    assert out.top_p == 0.9


def test_compose_vectors_replace():
    """Vectors compose wholesale — later replaces earlier."""
    a = cfg.ConfigFile(vectors="0.3 default/happy")
    b = cfg.ConfigFile(vectors="0.4 default/happy + 0.5 default/calm")
    out = cfg.compose([a, b])
    assert out.vectors == "0.4 default/happy + 0.5 default/calm"


def test_apply_flag_overrides():
    c = cfg.ConfigFile(model="from-yaml", temperature=0.5)
    out = cfg.apply_flag_overrides(c, model="from-flag", temperature=None, max_tokens=256)
    assert out.model == "from-flag"
    assert out.temperature == 0.5
    assert out.max_tokens == 256


def test_ensure_vectors_installed_all_present(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    from saklas.cli.selectors import invalidate
    d = tmp_path / "vectors" / "default" / "happy"
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="happy", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="bundled",
        files={"statements.json": packs.hash_file(d / "statements.json")},
    ).write(d)
    invalidate()

    c = cfg.ConfigFile(vectors="0.5 default/happy")
    missing = cfg.ensure_vectors_installed(c, strict=False)
    assert missing == []


def test_ensure_vectors_installed_missing_hf(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.cli.selectors import invalidate
    invalidate()
    installed = {}

    def fake_install(target, as_=None, force=False):
        installed["target"] = target
        return tmp_path / "vectors" / "user" / "happy"

    monkeypatch.setattr("saklas.io.cache_ops.install", fake_install)
    c = cfg.ConfigFile(vectors="0.5 user/happy")
    missing = cfg.ensure_vectors_installed(c, strict=False)
    assert installed["target"] == "user/happy"
    assert missing == []


def test_ensure_vectors_installed_strict_raises_on_local_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.cli.selectors import invalidate
    invalidate()
    c = cfg.ConfigFile(vectors="0.5 local/bard")
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
    assert c.model == "extra-model"
    assert c.temperature == 0.5
    assert c.top_p == 0.9


def test_effective_no_default(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model: default-model\n")
    c = cfg.ConfigFile.effective([], include_default=False)
    assert c.model is None


def test_to_yaml_roundtrip(tmp_path):
    c = cfg.ConfigFile(model="x", temperature=0.7, vectors="0.3 default/happy.sad")
    y = c.to_yaml(header="# header")
    assert y.startswith("# header\n")
    assert "model: x" in y
    assert "happy.sad" in y


def test_bare_pole_validates_against_installed_packs(monkeypatch, tmp_path):
    """Bare-pole references in the YAML expression validate through the
    parser; install-time checks walk the raw AST via
    ``referenced_selectors`` so the namespace-less bare name doesn't flag
    as missing when the installed pack bipolar-matches it."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io import packs
    from saklas.cli.selectors import invalidate
    d = tmp_path / "vectors" / "local" / "deer.wolf"
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    packs.PackMetadata(
        name="deer.wolf", description="x", version="1.0.0", license="MIT",
        tags=[], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(d / "statements.json")},
    ).write(d)
    invalidate()

    c = cfg.ConfigFile(vectors="0.5 wolf")
    # Install check passes because wolf bipolar-matches deer.wolf.
    missing = cfg.ensure_vectors_installed(c, strict=True)
    assert missing == []
