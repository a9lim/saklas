
import pytest

from saklas import cli_selectors as sel, packs


def test_parse_bare_name():
    s = sel.parse("happy")
    assert s.kind == "name"
    assert s.value == "happy"
    assert s.namespace is None


def test_parse_namespaced():
    s = sel.parse("a9lim/happy")
    assert s.kind == "name"
    assert s.value == "happy"
    assert s.namespace == "a9lim"


def test_parse_tag():
    s = sel.parse("tag:emotion")
    assert s.kind == "tag"
    assert s.value == "emotion"


def test_parse_namespace_scope():
    s = sel.parse("namespace:a9lim")
    assert s.kind == "namespace"
    assert s.value == "a9lim"


def test_parse_model_scope():
    s = sel.parse("model:google/gemma-2-2b-it")
    assert s.kind == "model"
    assert s.value == "google/gemma-2-2b-it"


def test_parse_default_alias():
    s = sel.parse("default")
    assert s.kind == "namespace"
    assert s.value == "default"


def test_parse_all():
    s = sel.parse("all")
    assert s.kind == "all"
    assert s.value is None


def test_parse_invalid_name_raises():
    with pytest.raises(sel.SelectorError):
        sel.parse("HAS_CAPS")


def test_parse_invalid_prefix_raises():
    with pytest.raises(sel.SelectorError):
        sel.parse("unknown:foo")


def _mk(tmp_path, ns, name, tags=None):
    d = tmp_path / "vectors" / ns / name
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    meta = packs.PackMetadata(
        name=name, description="x", version="1.0.0", license="MIT",
        tags=tags or [], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(d / "statements.json")},
    )
    meta.write(d)
    return d


def test_resolve_bare_unique(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy")
    results = sel.resolve(sel.parse("happy"))
    assert len(results) == 1
    assert results[0].name == "happy"


def test_resolve_bare_ambiguous_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy")
    _mk(tmp_path, "a9lim", "happy")
    with pytest.raises(sel.AmbiguousSelectorError) as ei:
        sel.resolve(sel.parse("happy"))
    assert "default/happy" in str(ei.value)
    assert "a9lim/happy" in str(ei.value)


def test_resolve_namespaced(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy")
    _mk(tmp_path, "a9lim", "happy")
    results = sel.resolve(sel.parse("a9lim/happy"))
    assert len(results) == 1
    assert "a9lim" in str(results[0].folder)


def test_resolve_tag(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy", tags=["emotion"])
    _mk(tmp_path, "default", "calm", tags=["emotion"])
    _mk(tmp_path, "default", "honest", tags=["personality"])
    results = sel.resolve(sel.parse("tag:emotion"))
    names = sorted(r.name for r in results)
    assert names == ["calm", "happy"]


def test_resolve_namespace(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy")
    _mk(tmp_path, "a9lim", "archaic")
    results = sel.resolve(sel.parse("namespace:a9lim"))
    assert [r.name for r in results] == ["archaic"]


def test_resolve_all(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy")
    _mk(tmp_path, "a9lim", "archaic")
    results = sel.resolve(sel.parse("all"))
    assert len(results) == 2


def test_parse_args_concept_plus_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    args, model_scope = sel.parse_args(["tag:emotion", "model:google/gemma-2-2b-it"])
    assert args.kind == "tag"
    assert args.value == "emotion"
    assert model_scope == "google/gemma-2-2b-it"


def test_parse_args_concept_only():
    args, model_scope = sel.parse_args(["happy"])
    assert args.kind == "name"
    assert model_scope is None


def test_parse_args_two_concepts_raises():
    with pytest.raises(sel.SelectorError, match="one concept"):
        sel.parse_args(["happy", "tag:emotion"])


def test_parse_args_two_models_raises():
    with pytest.raises(sel.SelectorError, match="one model"):
        sel.parse_args(["happy", "model:a", "model:b"])


# --- resolve_pole alias resolution -----------------------------------------

class TestResolvePole:
    def test_monopolar_exact_match(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "default", "agentic")
        sel.invalidate()
        name, sign, m = sel.resolve_pole("agentic")
        assert name == "agentic"
        assert sign == 1
        assert m is not None

    def test_positive_pole_alias(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "default", "angry.calm")
        sel.invalidate()
        name, sign, m = sel.resolve_pole("angry")
        assert name == "angry.calm"
        assert sign == 1

    def test_negative_pole_alias(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "default", "angry.calm")
        sel.invalidate()
        name, sign, m = sel.resolve_pole("calm")
        assert name == "angry.calm"
        assert sign == -1

    def test_composite_literal(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "default", "angry.calm")
        sel.invalidate()
        name, sign, m = sel.resolve_pole("angry.calm")
        assert name == "angry.calm"
        assert sign == 1

    def test_slug_normalization(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "default", "high_context.low_context")
        sel.invalidate()
        name, sign, _ = sel.resolve_pole("High-Context")
        assert name == "high_context.low_context"
        assert sign == 1

    def test_unknown_falls_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        sel.invalidate()
        name, sign, m = sel.resolve_pole("xyzzy")
        assert name == "xyzzy"
        assert sign == 1
        assert m is None

    def test_collision_monopolar_vs_bipolar(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "alice", "angry")
        _mk(tmp_path, "default", "angry.calm")
        sel.invalidate()
        with pytest.raises(sel.AmbiguousSelectorError):
            sel.resolve_pole("angry")

    def test_collision_two_bipolars(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "default", "angry.calm")
        _mk(tmp_path, "default", "angry.fearful")
        sel.invalidate()
        with pytest.raises(sel.AmbiguousSelectorError):
            sel.resolve_pole("angry")

    def test_namespaced_scoped_resolve(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "bob", "deer.wolf")
        _mk(tmp_path, "alice", "wolf")
        sel.invalidate()
        # Scoped to bob/: wolf -> deer.wolf with sign -1
        name, sign, m = sel.resolve_pole("wolf", namespace="bob")
        assert name == "deer.wolf"
        assert sign == -1
        assert m.namespace == "bob"
        # Scoped to alice/: wolf is a monopolar exact match
        name, sign, m = sel.resolve_pole("wolf", namespace="alice")
        assert name == "wolf"
        assert sign == 1
        assert m.namespace == "alice"
