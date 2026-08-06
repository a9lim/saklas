"""``io.source_registry`` — the active-source selection shared by lens + SAE.

The two families are documented as mirrors but used to diverge on every
load-bearing axis of this one artifact: one locked its writes and validated
names, the other did neither. These tests pin the shared contract and assert
both families are actually built on it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))


def _registry(**overrides):
    from saklas.io.source_registry import ActiveSourceRegistry

    def validate(kind: str, name: str) -> None:
        if not name.islower():
            raise ValueError(f"bad {kind} name {name!r}")

    kwargs: dict = {
        "label": "test",
        "format_version": 1,
        "kinds": ("local", "external"),
        "validate_name": validate,
        "exists": lambda root, kind, name: True,
    }
    kwargs.update(overrides)
    return ActiveSourceRegistry(**kwargs)


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    reg = _registry()
    root = tmp_path / "fam"
    root.mkdir()
    path = reg.write(root, "org/model", "local", "mine")
    assert path == root / "active.json"
    assert reg.read(root, "org/model") == reg.payload("org/model", "local", "mine")


def test_write_rejects_unknown_kind_and_bad_name(tmp_path: Path) -> None:
    reg = _registry()
    root = tmp_path / "fam"
    root.mkdir()
    with pytest.raises(ValueError, match="unknown test source kind"):
        reg.write(root, "org/model", "bogus", "mine")
    with pytest.raises(ValueError, match="bad local name"):
        reg.write(root, "org/model", "local", "MiXeD")
    assert not (root / "active.json").exists()


def test_write_requires_the_source_to_exist(tmp_path: Path) -> None:
    """A selection must never point at a source that is not on disk."""
    reg = _registry(exists=lambda root, kind, name: name == "present")
    root = tmp_path / "fam"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="local:absent"):
        reg.write(root, "org/model", "local", "absent")
    reg.write(root, "org/model", "local", "present")
    assert reg.read(root, "org/model") is not None


@pytest.mark.parametrize("mutate", [
    {"format_version": 2},
    {"model_id": "other/model"},
    {"kind": "bogus"},
    {"name": "MiXeD"},
    {"name": ""},
    {"extra": 1},
])
def test_read_rejects_any_deviation(tmp_path: Path, mutate: dict) -> None:
    """A stale or hand-edited selection reads as 'none', never raises."""
    reg = _registry()
    root = tmp_path / "fam"
    root.mkdir()
    payload = {**reg.payload("org/model", "local", "mine"), **mutate}
    (root / "active.json").write_text(json.dumps(payload))
    assert reg.read(root, "org/model") is None


def test_read_survives_corrupt_json(tmp_path: Path) -> None:
    reg = _registry()
    root = tmp_path / "fam"
    root.mkdir()
    (root / "active.json").write_text("{ not json")
    assert reg.read(root, "org/model") is None


def test_clear_if_active_only_matches_exactly(tmp_path: Path) -> None:
    reg = _registry()
    root = tmp_path / "fam"
    root.mkdir()
    reg.write(root, "org/model", "local", "mine")
    assert reg.clear_if_active(root, "org/model", "local", "other") is False
    assert reg.clear_if_active(root, "org/model", "external", "mine") is False
    assert reg.read(root, "org/model") is not None
    assert reg.clear_if_active(root, "org/model", "local", "mine") is True
    assert reg.read(root, "org/model") is None
    # Idempotent — nothing left to clear.
    assert reg.clear_if_active(root, "org/model", "local", "mine") is False


def test_both_families_use_the_shared_registry() -> None:
    """The mirror claim has to be structural, not a comment."""
    from saklas.io.lens_sources import LENS_SOURCES
    from saklas.io.sae import SAE_SOURCES
    from saklas.io.source_registry import ActiveSourceRegistry

    for reg in (LENS_SOURCES, SAE_SOURCES):
        assert isinstance(reg, ActiveSourceRegistry)
    assert LENS_SOURCES.kinds == {"local", "huggingface"}
    assert SAE_SOURCES.kinds == {"local", "saelens"}


def test_sae_selection_is_validated_and_precondition_checked() -> None:
    """The SAE half now carries the lens's discipline, not a bare non-empty check."""
    from saklas.io.sae import set_active_sae_source

    with pytest.raises(ValueError, match="unknown SAE source kind"):
        set_active_sae_source("org/model", "bogus", "x")
    with pytest.raises(ValueError, match="local SAE name must match"):
        set_active_sae_source("org/model", "local", "Not A Slug")
    with pytest.raises(ValueError, match="must not be empty"):
        set_active_sae_source("org/model", "saelens", "  ")
    # A well-formed name for a source that isn't on disk is still refused.
    with pytest.raises(FileNotFoundError, match="local:mine"):
        set_active_sae_source("org/model", "local", "mine")


def _weights():
    import torch

    return {
        "W_enc": torch.eye(3, 5),
        "W_dec": torch.eye(5, 3),
        "b_enc": torch.zeros(5),
        "b_dec": torch.ones(3),
    }


def _save_local(name: str = "mine", *, activate: bool = True):
    from saklas.io.sae_artifacts import save_local_sae

    return save_local_sae(
        "org/model", name, _weights(),
        model_fingerprint="model-fp", model_source_fingerprint="source-fp",
        layer=2, corpus_spec="test", corpus_sha256="a" * 64,
        tokens_trained=100, seq_len=16, batch_size=2,
        learning_rate=1e-3, l1_coefficient=1e-3, dead_feature_threshold=1e-6,
        activate=activate,
    )


def test_save_local_sae_activate_opt_out() -> None:
    """Publishing an artifact and selecting it are separable, like the lens fetch."""
    from saklas.io.sae import load_active_sae_source

    _save_local("first")
    assert load_active_sae_source("org/model")["name"] == "first"
    _save_local("second", activate=False)
    assert load_active_sae_source("org/model")["name"] == "first"


def test_save_sae_metadata_activate_opt_out() -> None:
    from saklas.io.sae import load_active_sae_source, save_sae_metadata

    payload = {
        "layer": 3, "width": 16, "revision": "rev", "fingerprint": "fp",
        "sae_id": None, "repo_id": None, "neuronpedia_id": None,
    }
    save_sae_metadata("org/model", "rel/one", payload)
    assert load_active_sae_source("org/model")["name"] == "rel/one"
    save_sae_metadata("org/model", "rel/two", payload, activate=False)
    assert load_active_sae_source("org/model")["name"] == "rel/one"


def test_load_active_sae_owns_the_prefix_convention() -> None:
    """``load_active_sae`` is the io-side mirror of ``io.lens.load_lens``."""
    from saklas.io.sae import load_active_sae, save_sae_metadata

    assert load_active_sae("org/model") is None

    _save_local("mine")
    release, metadata = load_active_sae("org/model")
    assert release == "local:mine"
    assert metadata is None, "a local artifact's manifest is its binding"

    save_sae_metadata("org/model", "rel/one", {
        "layer": 3, "width": 16, "revision": "rev", "fingerprint": "fp",
        "sae_id": None, "repo_id": None, "neuronpedia_id": None,
    })
    release, metadata = load_active_sae("org/model")
    assert release == "rel/one"
    assert metadata is not None and metadata["layer"] == 3


def test_removing_a_source_unpublishes_the_selection() -> None:
    """The selection can never outlive what it points at."""
    from saklas.io.sae import (
        load_active_sae_source, remove_sae_binding, save_sae_metadata,
    )
    from saklas.io.sae_artifacts import remove_local_sae

    _save_local("mine")
    assert remove_local_sae("org/model", "mine") is True
    assert load_active_sae_source("org/model") is None

    save_sae_metadata("org/model", "rel/one", {
        "layer": 3, "width": 16, "revision": "rev", "fingerprint": "fp",
        "sae_id": None, "repo_id": None, "neuronpedia_id": None,
    })
    assert remove_sae_binding("org/model", "rel/one") is True
    assert load_active_sae_source("org/model") is None


def test_removing_a_different_source_keeps_the_selection() -> None:
    from saklas.io.sae import load_active_sae_source
    from saklas.io.sae_artifacts import remove_local_sae

    _save_local("keep")
    _save_local("drop", activate=False)
    assert remove_local_sae("org/model", "drop") is True
    assert load_active_sae_source("org/model")["name"] == "keep"


def test_the_engine_does_not_respell_the_local_source_prefix() -> None:
    """The engine reads the source grammars; io owns spelling them.

    SAE: ``load_active_sae`` renders a selection, and
    ``is_local_sae_release`` / ``normalize_local_sae_name`` /
    ``local_sae_release`` parse and build one.  Lens: ``lens_source_label``
    is its twin.  So no module that resolves an artifact needs a literal
    ``local:``, and the convention cannot drift in one place and not the
    others.

    Scoped to artifact resolution: the *qualified* two-prefix display label
    (``local:<name>`` / ``saelens:<release>``) the instrument, the instrument
    routes, and the CLI render for clients is a separate grammar with its own
    duplication.
    """
    import saklas.core.sae as sae_mod
    import saklas.core.session as session_mod

    offenders = [
        f"{Path(mod.__file__).name}:{lineno}: {line.strip()}"
        for mod in (sae_mod, session_mod)
        for lineno, line in enumerate(
            Path(mod.__file__).read_text(encoding="utf-8").splitlines(), start=1,
        )
        if '"local:' in line or "'local:" in line
    ]
    assert offenders == []


def test_lens_source_label_inverts_use_lens_source() -> None:
    from saklas.io.lens_sources import (
        lens_source_label, load_active_lens_source, local_lens_dir,
        set_active_lens_source,
    )

    manifest = local_lens_dir("org/model") / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")
    set_active_lens_source("org/model", "local", "default")
    assert lens_source_label(load_active_lens_source("org/model")) == "local:default"
