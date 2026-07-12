"""Templated discover-manifold authoring (template_ref flow) — CPU-only, no model.

A *templated* manifold is a discover manifold whose node corpora derive from a
standalone template artifact (:mod:`saklas.io.templates`). The manifold stores the
derived corpus in ``nodes/`` like any discover folder and carries a
``template_ref`` so the fit can resolve the template's multi-turn contexts as the
per-node elicitation prefixes. These tests cover the authoring round-trip, the
staleness key (a template edit re-fits), and the manifest preservation. The
standalone template artifact itself is covered in ``test_templates.py``; the actual
multi-turn fit is exercised in the GPU verify path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from saklas.io.manifolds import (
    MANIFOLD_FORMAT_VERSION,
    ManifoldFolder,
    ManifoldFormatError,
    create_discover_manifold_folder,
    create_manifold_from_template,
)
from saklas.io.templates import create_template_folder

SLOT = "[DAY]"
VALUES = ["Monday", "Tuesday", "Sunday"]
CONTEXTS = [
    {"turns": [{"role": "user", "content": "what day is it?"}],
     "assistant": "today is [DAY]"},
    {"turns": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
        {"role": "user", "content": "remind me what day this is?"}],
     "assistant": "it's [DAY]"},
]


def _author_template(name: str = "weekday") -> None:
    create_template_folder(
        "local", name, slot=SLOT, values=VALUES, contexts=CONTEXTS,
        description="days of the week",
    )


def test_from_template_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_template()
    folder = create_manifold_from_template(
        "local", "weekday", "days", template_ref="weekday", fit_mode="auto",
    )
    mf = ManifoldFolder.load(folder)
    assert mf.fit_mode == "auto"
    assert mf.is_discover
    assert mf.node_labels == ["monday", "tuesday", "sunday"]
    assert mf.template_ref == "local/weekday"
    # Node corpora are the derived slot-filled assistant turns, one per context.
    groups = dict(mf.node_groups())
    assert groups["tuesday"] == ["today is Tuesday", "it's Tuesday"]
    assert groups["monday"] == ["today is Monday", "it's Monday"]


def test_from_template_default_namespace_and_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_template("colours")
    # ns/name form for the manifold; bare template name resolves cross-namespace.
    folder = create_manifold_from_template(
        "local", "colours", "", template_ref="colours", fit_mode="pca",
    )
    mf = ManifoldFolder.load(folder)
    assert mf.template_ref == "local/colours"


def test_templated_write_metadata_preserves_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_template()
    folder = create_manifold_from_template(
        "local", "weekday", "", template_ref="weekday", fit_mode="auto",
    )
    mf = ManifoldFolder.load(folder)
    sha_before = mf.nodes_sha256()
    mf.write_metadata()  # the post-fit manifest rewrite
    mf2 = ManifoldFolder.load(folder)
    assert mf2.template_ref == mf.template_ref == "local/weekday"
    assert mf2.nodes_sha256() == sha_before


def test_templated_sha_sensitive_to_template_context_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A context/value edit lives in the template, so the manifold sha folds it in.

    The history turns are an elicitation input the manifold's node corpus files
    don't capture (they're only the slotted assistant turns), so editing a
    context must invalidate the cached fit.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _author_template()
    folder = create_manifold_from_template(
        "local", "weekday", "", template_ref="weekday", fit_mode="auto",
    )
    mf = ManifoldFolder.load(folder)
    sha_before = mf.nodes_sha256()
    # Edit the template's first context user turn — only the template changes.
    edited = [dict(c) for c in CONTEXTS]
    edited[0] = {
        "turns": [{"role": "user", "content": "so, which day are we on?"}],
        "assistant": "today is [DAY]",
    }
    create_template_folder(
        "local", "weekday", slot=SLOT, values=VALUES, contexts=edited, force=True,
    )
    assert mf.nodes_sha256() != sha_before


def test_non_templated_discover_has_no_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """An ordinary discover folder carries an explicit null template_ref."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "plain", "", fit_mode="pca",
        node_corpora={"a": ["a says", "a too"], "b": ["b says", "b too"]},
        hyperparams={"max_dim": 1},
    )
    mf = ManifoldFolder.load(folder)
    assert mf.template_ref is None
    # The canonical manifest carries an explicit null reference.
    payload = json.loads((folder / "manifold.json").read_text())
    assert payload["template_ref"] is None


def test_load_rejects_template_ref_on_authored(tmp_path: Path):
    """A ``template_ref`` on a non-discover folder is a format error."""
    folder = tmp_path / "bad"
    (folder / "nodes").mkdir(parents=True)
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": MANIFOLD_FORMAT_VERSION,
            "name": "bad",
            "description": "",
            "fit_mode": "authored",
        "domain": {"type": "box", "axes": [
            {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0},
        ]},
        "nodes": [
            {"label": "a", "coords": [0.0], "role": None, "kind": None},
            {"label": "b", "coords": [0.5], "role": None, "kind": None},
            {"label": "c", "coords": [1.0], "role": None, "kind": None},
        ],
        "template_ref": "weekday",
        "files": {},
        "source": "local", "tags": [],
    }))
    with pytest.raises(ManifoldFormatError, match="discover-mode feature"):
        ManifoldFolder.load(folder)
