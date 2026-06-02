
from __future__ import annotations

import pytest
from pathlib import Path

import json

from saklas.io import selectors as sel
from saklas.io.manifolds import create_discover_manifold_folder


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


def _mk(tmp_path: Path, ns: str, name: str, tags: list[str] | None = None) -> Path:
    """Author an installed concept as a 2-node ``pca`` manifold (4.0).

    Concepts and steering manifolds are the same artifact now, so an
    installed concept is a manifold folder under ``manifolds/<ns>/<name>/``.
    ``tags`` are patched into ``manifold.json`` so ``tag:`` selectors resolve.
    """
    folder = create_discover_manifold_folder(
        ns, name, "x", fit_mode="pca",
        node_corpora={"pos": ["a statement."], "neg": ["b statement."]},
        hyperparams={"max_dim": 1},
    )
    if tags:
        mpath = folder / "manifold.json"
        data = json.loads(mpath.read_text())
        data["tags"] = list(tags)
        mpath.write_text(json.dumps(data))
    return folder


def test_resolve_bare_unique(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy")
    results = sel.resolve(sel.parse("happy"))
    assert len(results) == 1
    assert results[0].name == "happy"


def test_resolve_bare_ambiguous_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy")
    _mk(tmp_path, "a9lim", "happy")
    with pytest.raises(sel.AmbiguousSelectorError) as ei:
        sel.resolve(sel.parse("happy"))
    assert "default/happy" in str(ei.value)
    assert "a9lim/happy" in str(ei.value)


def test_resolve_namespaced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy")
    _mk(tmp_path, "a9lim", "happy")
    results = sel.resolve(sel.parse("a9lim/happy"))
    assert len(results) == 1
    assert "a9lim" in str(results[0].folder)


def test_resolve_tag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy", tags=["emotion"])
    _mk(tmp_path, "default", "calm", tags=["emotion"])
    _mk(tmp_path, "default", "honest", tags=["personality"])
    results = sel.resolve(sel.parse("tag:emotion"))
    names = sorted(r.name for r in results)
    assert names == ["calm", "happy"]


def test_resolve_namespace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy")
    _mk(tmp_path, "a9lim", "archaic")
    results = sel.resolve(sel.parse("namespace:a9lim"))
    assert [r.name for r in results] == ["archaic"]


def test_resolve_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk(tmp_path, "default", "happy")
    _mk(tmp_path, "a9lim", "archaic")
    results = sel.resolve(sel.parse("all"))
    assert len(results) == 2


def _fake_fitted_tensor(folder: Path, filename: str) -> None:
    """Write a placeholder tensor + minimal sidecar so the manifold loads.

    ``model:`` resolution only checks tensor *filenames*, never tensor
    contents, so the bytes can be junk — but ``ManifoldFolder.load`` demands
    a ``.json`` sidecar beside every ``.safetensors``, so we write a lean
    one (a ``domain`` object is the only hard requirement of
    ``ManifoldSidecar.load``).
    """
    (folder / filename).write_bytes(b"x")
    sidecar = Path(folder) / (Path(filename).stem + ".json")
    sidecar.write_text(json.dumps({
        "method": "manifold_pca",
        "saklas_version": "0",
        "domain": {"kind": "custom", "dim": 1},
        "node_count": 2,
        "node_labels": ["pos", "neg"],
        "fit_mode": "pca",
    }))


def test_resolve_model_matches_raw_and_sae_tensors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    _fake_fitted_tensor(a, f"{sid}.safetensors")

    # Concept B: has only an SAE tensor for this model.
    b = _mk(tmp_path, "default", "b_sae_only")
    _fake_fitted_tensor(b, tensor_filename(model_id, release="my-release"))

    # Concept C: has a tensor for a different model — should not match.
    c = _mk(tmp_path, "default", "c_other_model")
    _fake_fitted_tensor(c, f"{safe_model_id('meta/llama-3-8b')}.safetensors")

    sel.invalidate()
    results = sel.resolve(sel.parse(f"model:{model_id}"))
    names = sorted(r.name for r in results)
    assert names == ["a_raw_only", "b_sae_only"]


def test_parse_args_concept_plus_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    """4.0: ``resolve_pole`` no longer scans disk or resolves bipolar aliases.

    It always returns ``(canonical_slug, +1, None, variant)`` — peeling a
    ``:variant`` suffix and canonicalizing the name.  Bipolar-pole STEERING
    (``wolf`` → ``deer.wolf``) moved to the manifold tier
    (:func:`resolve_manifold_label` / ``resolve_bare_name`` in the steering
    grammar), so the old positive/negative-alias, collision, sign-flip, and
    namespaced-scoped tests are obsolete and were deleted.
    """

    def test_monopolar_exact_match(self) -> None:
        name, sign, m, _v = sel.resolve_pole("agentic")
        assert name == "agentic"
        assert sign == 1
        assert m is None

    def test_composite_literal(self) -> None:
        name, sign, m, _v = sel.resolve_pole("angry.calm")
        assert name == "angry.calm"
        assert sign == 1
        assert m is None

    def test_slug_normalization(self) -> None:
        name, sign, _m, _v = sel.resolve_pole("High-Context")
        assert name == "high_context"
        assert sign == 1

    def test_unknown_falls_through(self) -> None:
        name, sign, m, _v = sel.resolve_pole("xyzzy")
        assert name == "xyzzy"
        assert sign == 1
        assert m is None


def test_resolve_pole_strips_raw_variant() -> None:
    from saklas.io.selectors import resolve_pole

    canonical, sign, match, variant = resolve_pole("honest:raw")
    assert canonical == "honest"
    assert sign == 1
    assert match is None
    assert variant == "raw"


def test_resolve_pole_sae_variant() -> None:
    from saklas.io.selectors import resolve_pole

    canonical, sign, match, variant = resolve_pole("honest:sae")
    assert canonical == "honest"
    assert match is None
    assert variant == "sae"


def test_resolve_pole_sae_with_release() -> None:
    from saklas.io.selectors import resolve_pole

    canonical, sign, match, variant = resolve_pole("honest:sae-gemma-scope-2b-pt-res-canonical")
    assert variant == "sae-gemma-scope-2b-pt-res-canonical"


def test_resolve_pole_no_variant_defaults_to_raw() -> None:
    from saklas.io.selectors import resolve_pole

    canonical, sign, match, variant = resolve_pole("honest")
    assert variant == "raw"


def test_resolve_pole_variant_strips_suffix() -> None:
    """A ``:variant`` suffix peels off; the name canonicalizes, sign stays +1.

    (Pre-4.0 this asserted a bipolar sign flip for ``wolf:sae`` →
    ``deer.wolf @ -1``; that resolution moved to the manifold tier.)
    """
    from saklas.io.selectors import resolve_pole

    canonical, sign, match, variant = resolve_pole("wolf:sae")
    assert canonical == "wolf"
    assert sign == 1
    assert match is None
    assert variant == "sae"


def test_resolve_pole_rejects_invalid_variant() -> None:
    from saklas.io.selectors import resolve_pole, SelectorError

    with pytest.raises(SelectorError):
        resolve_pole("honest:weird-variant")


def test_parse_accepts_variant_suffix():
    """parse() with a :variant suffix strips the variant, keeps Selector.value as the bare name."""
    from saklas.io.selectors import parse
    s = parse("honest.deceptive:sae")
    assert s.kind == "name"
    assert s.value == "honest.deceptive"


def test_parse_rejects_unknown_variant():
    import pytest as _pt
    from saklas.io.selectors import parse, SelectorError
    with _pt.raises(SelectorError):
        parse("honest:garbage")


def test_parse_role_variant():
    """parse() with a :role-<id> suffix strips the variant, keeps Selector.value as the bare name."""
    from saklas.io.selectors import parse
    s = parse("honest:role-pirate")
    assert s.kind == "name"
    assert s.value == "honest"
    assert s.namespace is None


def test_resolve_pole_role_variant() -> None:
    from saklas.io.selectors import resolve_pole

    canonical, sign, match, variant = resolve_pole("angry:role-pirate")
    assert canonical == "angry"
    assert sign == 1
    assert match is None
    assert variant == "role-pirate"


def test_resolve_pole_role_with_dotted_id() -> None:
    from saklas.io.selectors import resolve_pole

    canonical, sign, match, variant = resolve_pole("happy.sad:role-mad-scientist")
    assert canonical == "happy.sad"
    assert sign == 1
    assert match is None
    assert variant == "role-mad-scientist"


def test_parse_role_variant_invalid_slug():
    """Uppercase id rejected — matches the SAE precedent for `sae-FOO`."""
    import pytest as _pt
    from saklas.io.selectors import parse, SelectorError
    with _pt.raises(SelectorError):
        parse("honest:role-PIRATE")


def test_parse_role_with_namespace():
    from saklas.io.selectors import parse
    s = parse("default/honest:role-pirate")
    assert s.kind == "name"
    assert s.value == "honest"
    assert s.namespace == "default"


# NOTE: ``test_materialize_then_invalidate_makes_bundled_visible`` was deleted
# in 4.0 — it pinned the now-removed ``packs.bundled_concept_names()`` /
# ``packs.materialize_bundled()`` ``vectors/``-pack visibility contract.
# Bundled concepts ship as manifolds now
# (``saklas.io.manifolds.materialize_bundled_manifolds``).
