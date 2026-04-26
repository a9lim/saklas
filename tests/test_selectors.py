
import pytest

from saklas.cli import selectors as sel
from saklas.io import packs


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


def test_resolve_model_matches_raw_and_sae_tensors(monkeypatch, tmp_path):
    """``model:X`` matches any concept with a tensor for X — raw or SAE.

    Regression for the pre-fix bug where the filter only globbed
    ``<safe>.safetensors`` and missed concepts that shipped only a
    ``_sae-<release>`` tensor for that model.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    from saklas.io.paths import safe_model_id, tensor_filename

    model_id = "google/gemma-3-4b-it"
    sid = safe_model_id(model_id)

    # Concept A: has raw tensor only.
    a = _mk(tmp_path, "default", "a_raw_only")
    (a / f"{sid}.safetensors").write_bytes(b"x")

    # Concept B: has only an SAE tensor for this model.
    b = _mk(tmp_path, "default", "b_sae_only")
    (b / tensor_filename(model_id, release="my-release")).write_bytes(b"x")

    # Concept C: has a tensor for a different model — should not match.
    c = _mk(tmp_path, "default", "c_other_model")
    (c / f"{safe_model_id('meta/llama-3-8b')}.safetensors").write_bytes(b"x")

    sel.invalidate()
    results = sel.resolve(sel.parse(f"model:{model_id}"))
    names = sorted(r.name for r in results)
    assert names == ["a_raw_only", "b_sae_only"]


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
        name, sign, m, _v = sel.resolve_pole("agentic")
        assert name == "agentic"
        assert sign == 1
        assert m is not None

    def test_positive_pole_alias(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "default", "angry.calm")
        sel.invalidate()
        name, sign, m, _v = sel.resolve_pole("angry")
        assert name == "angry.calm"
        assert sign == 1

    def test_negative_pole_alias(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "default", "angry.calm")
        sel.invalidate()
        name, sign, m, _v = sel.resolve_pole("calm")
        assert name == "angry.calm"
        assert sign == -1

    def test_composite_literal(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "default", "angry.calm")
        sel.invalidate()
        name, sign, m, _v = sel.resolve_pole("angry.calm")
        assert name == "angry.calm"
        assert sign == 1

    def test_slug_normalization(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _mk(tmp_path, "default", "high_context.low_context")
        sel.invalidate()
        name, sign, _m, _v = sel.resolve_pole("High-Context")
        assert name == "high_context.low_context"
        assert sign == 1

    def test_unknown_falls_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        sel.invalidate()
        name, sign, m, _v = sel.resolve_pole("xyzzy")
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
        name, sign, m, _variant = sel.resolve_pole("wolf", namespace="bob")
        assert name == "deer.wolf"
        assert sign == -1
        assert m.namespace == "bob"
        # Scoped to alice/: wolf is a monopolar exact match
        name, sign, m, _variant = sel.resolve_pole("wolf", namespace="alice")
        assert name == "wolf"
        assert sign == 1
        assert m.namespace == "alice"


def _install_minimal_pack(saklas_home, name):
    """Lay down a minimal pack.json tree so _all_concepts finds the name."""
    import json
    folder = saklas_home / "vectors" / "default" / name
    folder.mkdir(parents=True)
    (folder / "pack.json").write_text(json.dumps({
        "name": name,
        "description": "test",
        "version": "0.0.0",
        "license": "MIT",
        "tags": [],
        "recommended_alpha": 0.3,
        "source": "local",
        "files": {},
        "format_version": 2,
    }))


def test_resolve_pole_strips_raw_variant(tmp_path, monkeypatch):
    _install_minimal_pack(tmp_path, "honest.deceptive")
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    from saklas.cli.selectors import resolve_pole, invalidate
    invalidate()

    canonical, sign, match, variant = resolve_pole("honest:raw")
    assert canonical == "honest.deceptive"
    assert sign == 1
    assert match is not None
    assert variant == "raw"


def test_resolve_pole_sae_variant(tmp_path, monkeypatch):
    _install_minimal_pack(tmp_path, "honest.deceptive")
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    from saklas.cli.selectors import resolve_pole, invalidate
    invalidate()

    canonical, sign, match, variant = resolve_pole("honest:sae")
    assert canonical == "honest.deceptive"
    assert variant == "sae"


def test_resolve_pole_sae_with_release(tmp_path, monkeypatch):
    _install_minimal_pack(tmp_path, "honest.deceptive")
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    from saklas.cli.selectors import resolve_pole, invalidate
    invalidate()

    canonical, sign, match, variant = resolve_pole("honest:sae-gemma-scope-2b-pt-res-canonical")
    assert variant == "sae-gemma-scope-2b-pt-res-canonical"


def test_resolve_pole_no_variant_defaults_to_raw(tmp_path, monkeypatch):
    _install_minimal_pack(tmp_path, "honest.deceptive")
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    from saklas.cli.selectors import resolve_pole, invalidate
    invalidate()

    canonical, sign, match, variant = resolve_pole("honest")
    assert variant == "raw"


def test_resolve_pole_variant_preserves_pole_sign(tmp_path, monkeypatch):
    _install_minimal_pack(tmp_path, "deer.wolf")
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    from saklas.cli.selectors import resolve_pole, invalidate
    invalidate()

    canonical, sign, match, variant = resolve_pole("wolf:sae")
    assert canonical == "deer.wolf"
    assert sign == -1
    assert variant == "sae"


def test_resolve_pole_rejects_invalid_variant(tmp_path, monkeypatch):
    _install_minimal_pack(tmp_path, "honest.deceptive")
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))

    from saklas.cli.selectors import resolve_pole, invalidate, SelectorError
    invalidate()

    with pytest.raises(SelectorError):
        resolve_pole("honest:weird-variant")


def test_parse_accepts_variant_suffix():
    """parse() with a :variant suffix strips the variant, keeps Selector.value as the bare name."""
    from saklas.cli.selectors import parse
    s = parse("honest.deceptive:sae")
    assert s.kind == "name"
    assert s.value == "honest.deceptive"


def test_parse_rejects_unknown_variant():
    import pytest as _pt
    from saklas.cli.selectors import parse, SelectorError
    with _pt.raises(SelectorError):
        parse("honest:garbage")


def test_materialize_then_invalidate_makes_bundled_visible(monkeypatch, tmp_path):
    """The contract `SaklasSession.__init__` relies on for bundled visibility.

    Regression: when bundled concepts are added (e.g. via
    `regenerate_bundled_statements.py`) but the user-cache hasn't been
    refreshed since, `_all_concepts()` doesn't see them — and
    `session.extract(name)` silently falls through to the local namespace
    and re-runs scenario+pair generation instead of using the bundled
    statements. `SaklasSession.__init__` calls `materialize_bundled()` +
    `selectors.invalidate()` to guarantee bundled visibility per session
    boot. This test pins that invariant at the helper level so the
    contract holds even when probes=[] skips probes_bootstrap entirely.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    sel.invalidate()

    # Prime the cache with an empty walk — bundled not visible yet.
    user_default = tmp_path / "vectors" / "default"
    assert not user_default.exists()
    initial = sel._all_concepts()
    assert all(c.namespace != "default" for c in initial)

    bundled = set(packs.bundled_concept_names())
    assert bundled, "test prereq: shipped saklas.data.vectors must be non-empty"

    # The session-init contract: materialize, then invalidate the cache.
    packs.materialize_bundled()
    sel.invalidate()

    # Bundled concepts are now in user cache and visible to the selector.
    assert user_default.is_dir()
    concepts = sel._all_concepts()
    names = {c.name for c in concepts if c.namespace == "default"}
    missing = bundled - names
    assert not missing, (
        f"_all_concepts() did not surface bundled concepts after "
        f"materialize+invalidate: {sorted(missing)}"
    )
