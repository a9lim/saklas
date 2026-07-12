
from __future__ import annotations

import pytest
from pathlib import Path

import json

from saklas.io import selectors as sel
from saklas.io.manifolds import (
    create_discover_manifold_folder,
)
from saklas.io.manifold_folder import canonical_manifold_sidecar_payload


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
    """Write a placeholder tensor + current sidecar so the manifold loads.

    ``model:`` resolution only checks tensor *filenames*, never tensor
    contents, so the bytes can be junk — but ``ManifoldFolder.load`` demands
    a current ``.json`` sidecar beside every ``.safetensors``.
    """
    (folder / filename).write_bytes(b"x")
    sidecar = Path(folder) / (Path(filename).stem + ".json")
    sidecar.write_text(json.dumps(canonical_manifold_sidecar_payload(
        name=folder.name, method="manifold_pca", saklas_version="0",
        domain={"type": "custom", "embed_dim": 1, "bounds": None},
        node_labels=["pos", "neg"],
        feature_space="raw", fit_mode="pca",
    )))


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


# --- canonicalize_atom (the retired resolve_pole's surviving behavior) -------

class TestCanonicalizeAtom:
    """4.0: ``resolve_pole`` is retired; :func:`canonicalize_atom` carries its
    surviving behavior — peel a ``:variant`` suffix and canonicalize the name,
    returning ``(canonical_slug, variant)``.  Bipolar-pole STEERING
    (``wolf`` → ``deer.wolf``) moved to the manifold tier (the label tier of
    :func:`resolve_bare_atom`), so there is no sign-flip / match slot anymore.
    """

    def test_monopolar_exact_match(self) -> None:
        name, _v = sel.canonicalize_atom("agentic")
        assert name == "agentic"

    def test_composite_literal(self) -> None:
        name, _v = sel.canonicalize_atom("angry.calm")
        assert name == "angry.calm"

    def test_slug_normalization(self) -> None:
        name, _v = sel.canonicalize_atom("High-Context")
        assert name == "high_context"

    def test_unknown_falls_through(self) -> None:
        name, _v = sel.canonicalize_atom("xyzzy")
        assert name == "xyzzy"


def test_canonicalize_atom_strips_raw_variant() -> None:
    from saklas.io.selectors import canonicalize_atom

    canonical, variant = canonicalize_atom("honest:raw")
    assert canonical == "honest"
    assert variant == "raw"


def test_canonicalize_atom_sae_variant() -> None:
    from saklas.io.selectors import canonicalize_atom

    canonical, variant = canonicalize_atom("honest:sae")
    assert canonical == "honest"
    assert variant == "sae"


def test_canonicalize_atom_sae_with_release() -> None:
    from saklas.io.selectors import canonicalize_atom

    _canonical, variant = canonicalize_atom("honest:sae-gemma-scope-2b-pt-res-canonical")
    assert variant == "sae-gemma-scope-2b-pt-res-canonical"


def test_canonicalize_atom_no_variant_defaults_to_raw() -> None:
    from saklas.io.selectors import canonicalize_atom

    _canonical, variant = canonicalize_atom("honest")
    assert variant == "raw"


def test_canonicalize_atom_variant_strips_suffix() -> None:
    """A ``:variant`` suffix peels off; the name canonicalizes.

    (Pre-4.0 this asserted a bipolar sign flip for ``wolf:sae`` →
    ``deer.wolf @ -1``; that resolution moved to the manifold tier.)
    """
    from saklas.io.selectors import canonicalize_atom

    canonical, variant = canonicalize_atom("wolf:sae")
    assert canonical == "wolf"
    assert variant == "sae"


def test_canonicalize_atom_rejects_invalid_variant() -> None:
    from saklas.io.selectors import canonicalize_atom, SelectorError

    with pytest.raises(SelectorError):
        canonicalize_atom("honest:weird-variant")


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


def test_canonicalize_atom_role_variant() -> None:
    from saklas.io.selectors import canonicalize_atom

    canonical, variant = canonicalize_atom("angry:role-pirate")
    assert canonical == "angry"
    assert variant == "role-pirate"


def test_canonicalize_atom_role_with_dotted_id() -> None:
    from saklas.io.selectors import canonicalize_atom

    canonical, variant = canonicalize_atom("happy.sad:role-mad-scientist")
    assert canonical == "happy.sad"
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


# --- resolve_bare_atom: the single-owner bare-atom tier ladder ---------------

def _mk_nodes(ns: str, name: str, labels: list[str]) -> Path:
    """Author an installed discover manifold with the given node labels."""
    return create_discover_manifold_folder(
        ns, name, "x", fit_mode="pca",
        node_corpora={lbl: [f"{lbl} statement."] for lbl in labels},
        hyperparams={"max_dim": 1},
    )


def test_bare_atom_label_tier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A bare slug matching a multi-node manifold's node label → label hit."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk_nodes("default", "personas", ["pirate", "wizard", "vandal"])
    sel.invalidate()
    atom = sel.resolve_bare_atom("pirate")
    assert atom.kind == "label"
    assert atom.manifold is not None
    assert atom.manifold.manifold_key == "default/personas"
    assert atom.manifold.label == "pirate"


def test_bare_atom_name_tier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A dotted 2-node ``pca`` manifold *name* → composite-name hit (node 0).

    The ``.`` makes it skip the label tier and land on the name tier.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk_nodes("default", "deer.wolf", ["deer", "wolf"])
    sel.invalidate()
    atom = sel.resolve_bare_atom("deer.wolf")
    assert atom.kind == "name"
    assert atom.manifold_name is not None
    assert atom.manifold_name.manifold_key == "default/deer.wolf"
    assert atom.manifold_name.pole_label == "deer"  # node 0


def test_bare_atom_pole_fallthrough(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No manifold match → pole canonicalization (peel variant + slug)."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    sel.invalidate()
    atom = sel.resolve_bare_atom("Xyzzy-Thing", variant="sae")
    assert atom.kind == "pole"
    assert atom.pole == ("xyzzy_thing", "sae")


def test_bare_atom_variant_skips_label_tier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A non-``raw`` variant skips both manifold tiers (variant addressing)."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk_nodes("default", "personas", ["pirate"])
    sel.invalidate()
    atom = sel.resolve_bare_atom("pirate", variant="sae")
    assert atom.kind == "pole"
    assert atom.pole == ("pirate", "sae")


def test_bare_atom_typed_namespace_skips_label_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A user-typed namespace skips the bare-label tier (but not the name tier)."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk_nodes("alice", "personas", ["pirate"])
    sel.invalidate()
    atom = sel.resolve_bare_atom("pirate", typed_namespace="alice")
    assert atom.kind == "pole"
    assert atom.pole == ("pirate", "raw")


def test_bare_atom_cross_manifold_label_collision_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Two manifolds owning the same node label → AmbiguousSelectorError."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _mk_nodes("default", "personas", ["pirate"])
    _mk_nodes("default", "roles", ["pirate"])
    sel.invalidate()
    with pytest.raises(sel.AmbiguousSelectorError):
        sel.resolve_bare_atom("pirate")


# NOTE: ``test_materialize_then_invalidate_makes_bundled_visible`` was deleted
# in 4.0 — it pinned the now-removed ``packs.bundled_concept_names()`` /
# ``packs.materialize_bundled()`` ``vectors/``-pack visibility contract.
# Bundled concepts ship as manifolds now
# (``saklas.io.manifolds.materialize_bundled_manifolds``).
