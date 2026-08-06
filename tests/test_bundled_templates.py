"""Bundled-template materialization infra — CPU-only, content-independent.

Tests the ``materialize_bundled_templates`` / ``bundled_template_names`` mechanics
(the reusable mirror of the manifold materializer) against a synthetic fixture
template injected via ``_resources``, so coverage doesn't depend on whatever
template-derived manifold happens to ship.  No template-derived bundled manifold
currently ships (colors was the first candidate, pulled after it resolved to a
flat lexical scatter rather than a hue ring); the end-to-end ``template_ref``
wiring test returns to this suite when a real one lands.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import saklas.io.templates as templates_mod
from saklas.io.paths import templates_dir
from saklas.io.templates import (
    TemplateFolder,
    bundled_template_names,
    materialize_bundled_templates,
    resolve_template,
)

_DEMO = {
    "format_version": 2,
    "name": "demo",
    "slot": "[X]",
    "values": ["alpha", "beta"],
    "contexts": [
        {"turns": [{"role": "user", "content": "pick one"}], "assistant": "[X]"},
    ],
    "description": "",
    "source": "bundled",
    "tags": [],
}


class _FakeResources:
    """Stand-in for ``importlib.resources`` pointing at a temp package-data root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def files(self, pkg: str) -> Path:
        assert pkg == "saklas.data.templates"
        return self._root


def _write_template(
    pkg_root: Path, payload: dict[str, object], name: str = "demo",
) -> None:
    d = pkg_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "template.json").write_text(json.dumps(payload))


def _wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pkg_root: Path) -> None:
    pkg_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(templates_mod, "_resources", _FakeResources(pkg_root))
    monkeypatch.setattr(templates_mod, "_templates_materialized_home", None)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #


def test_empty_package_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _wire(monkeypatch, tmp_path, tmp_path / "pkg")  # no template written
    assert bundled_template_names() == []
    materialize_bundled_templates()  # must not crash, must not create default/
    assert not (templates_dir() / "default").exists()


def test_valid_template_advertised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, _DEMO)
    assert bundled_template_names() == ["demo"]


def test_invalid_template_not_advertised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, {**_DEMO, "values": ["only_one"]})  # < 2 values: invalid
    assert bundled_template_names() == []


# --------------------------------------------------------------------------- #
# materialization
# --------------------------------------------------------------------------- #


def test_fresh_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, _DEMO)
    materialize_bundled_templates()
    manifest = templates_dir() / "default" / "demo" / "template.json"
    assert manifest.exists()
    tmpl = TemplateFolder.load(manifest.parent)
    assert tmpl.name == "demo" and list(tmpl.values) == ["alpha", "beta"]
    # bare-name resolves to the materialized default copy
    assert resolve_template("demo").slot == "[X]"


def test_process_scope_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, _DEMO)
    materialize_bundled_templates()
    manifest = templates_dir() / "default" / "demo" / "template.json"
    manifest.unlink()
    materialize_bundled_templates()  # flag set → no-op, no re-copy
    assert not manifest.exists()


def test_process_scope_guard_is_per_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, _DEMO)
    materialize_bundled_templates()
    assert (templates_dir() / "default" / "demo" / "template.json").exists()

    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "second-home"))
    materialize_bundled_templates()
    assert (templates_dir() / "default" / "demo" / "template.json").exists()


def test_bundle_update_recopies_and_backs_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, _DEMO)  # bundled: 2 values
    # A stale on-disk copy with different content — canonical hash differs.
    target = templates_dir() / "default" / "demo"
    target.mkdir(parents=True)
    (target / "template.json").write_text(json.dumps({
        **_DEMO, "values": ["gamma", "delta", "epsilon"],
    }))
    with pytest.warns(UserWarning, match="refreshed default/demo"):
        materialize_bundled_templates()
    refreshed = TemplateFolder.load(target)
    assert list(refreshed.values) == ["alpha", "beta"]   # bundled content won
    assert (target / "template.json.bak").exists()        # old copy preserved


# --------------------------------------------------------------------------- #
# the ordering invariant lives in one place
# --------------------------------------------------------------------------- #


def test_bootstrap_entry_point_runs_templates_before_manifolds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The order is load-bearing, so exactly one function may own it.

    A bundled manifold may ``template_ref`` a bundled ``default/<name>``
    template; its fit resolves that ref and raises ``TemplateNotFoundError``
    if the template hasn't materialized yet.  Five bootstrap sites used to
    each re-derive this from a copied comment, so a sixth would silently
    violate it.
    """
    from saklas.io import bootstrap, manifolds as manifolds_mod
    from saklas.io import templates as templates_mod

    order: list[str] = []
    monkeypatch.setattr(
        templates_mod, "materialize_bundled_templates",
        lambda: order.append("templates"),
    )
    monkeypatch.setattr(
        manifolds_mod, "materialize_bundled_manifolds",
        lambda: order.append("manifolds"),
    )
    bootstrap.materialize_bundled_artifacts()
    assert order == ["templates", "manifolds"]


def test_every_bootstrap_site_uses_the_entry_point() -> None:
    """No site may call the two materializers directly and re-derive the order."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "saklas"
    owners = {
        root / "io" / "bootstrap.py",
        root / "io" / "manifolds.py",
        root / "io" / "templates.py",
        # ``refresh_manifold``'s bundled tier re-materializes one artifact
        # kind on purpose; it is not a bootstrap.
        root / "io" / "manifold_lifecycle.py",
    }
    offenders = [
        str(path.relative_to(root.parent))
        for path in root.rglob("*.py")
        if path not in owners
        and "materialize_bundled_templates(" in path.read_text()
    ]
    assert offenders == []

