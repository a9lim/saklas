"""Templated discover-manifold authoring — CPU-only, no model.

A *templated* manifold is a discover manifold whose node corpora are
slot-filled assistant turns, plus a ``template`` block in ``manifold.json``
that records the authoring template and — load-bearing at fit time — the
per-manifold elicitation prompt set (the template's user turns). These tests
cover the expansion, the on-disk round-trip, the staleness key, and the
validation invariants. The actual fit (which pools each filled assistant
against ``pairs[*].user``) is exercised in the GPU verify path.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from saklas.io.manifolds import (
    ManifoldFolder,
    ManifoldFormatError,
    create_discover_manifold_folder,
    create_templated_manifold_folder,
    expand_template,
)

# A small two-template weekday set; the slot lives only in the assistant turn.
PAIRS = [
    {"user": "what day is it?", "assistant": "today is [DAY]"},
    {"user": "remind me what day this is?", "assistant": "it's [DAY]"},
]
VALUES = ["Monday", "Tuesday", "Sunday"]
SLOT = "[DAY]"


# ---- expand_template (pure) ------------------------------------------------

def test_expand_template_fills_and_slugs():
    corpora, block = expand_template(SLOT, PAIRS, VALUES, name="weekday")
    # One node per value, labels slugged from the raw (as-typed) value.
    assert list(corpora.keys()) == ["monday", "tuesday", "sunday"]
    # Corpus = each template's assistant with the slot filled by the RAW value
    # (so the captured text reads naturally), one entry per template.
    assert corpora["monday"] == ["today is Monday", "it's Monday"]
    assert corpora["sunday"] == ["today is Sunday", "it's Sunday"]
    # The block preserves raw values + normalized pairs for re-expansion.
    assert block["slot"] == SLOT
    assert block["values"] == VALUES
    assert block["pairs"] == PAIRS


def test_expand_template_multiword_value_slug():
    corpora, _ = expand_template(
        "[CITY]",
        [{"user": "where are you?", "assistant": "I'm in [CITY]"}],
        ["New York", "São Paulo"],
        name="cities",
    )
    assert list(corpora.keys()) == ["new_york", "s_o_paulo"]
    assert corpora["new_york"] == ["I'm in New York"]


def test_expand_template_rejects_assistant_without_slot():
    with pytest.raises(ManifoldFormatError, match="must contain the slot"):
        expand_template(
            SLOT,
            [{"user": "hi", "assistant": "no slot here"}],
            VALUES,
            name="weekday",
        )


def test_expand_template_rejects_slot_in_user_turn():
    with pytest.raises(ManifoldFormatError, match="must.*not contain the slot"):
        expand_template(
            SLOT,
            [{"user": "is it [DAY]?", "assistant": "yes, [DAY]"}],
            VALUES,
            name="weekday",
        )


def test_expand_template_requires_two_values():
    with pytest.raises(ManifoldFormatError, match=r"'values'"):
        expand_template(SLOT, PAIRS, ["Monday"], name="weekday")


def test_expand_template_requires_pairs():
    with pytest.raises(ManifoldFormatError, match=r"'pairs'"):
        expand_template(SLOT, [], VALUES, name="weekday")


def test_expand_template_rejects_label_collision():
    # Two values that slug to the same label.
    with pytest.raises(ManifoldFormatError, match="collides"):
        expand_template(SLOT, PAIRS, ["Monday", "monday!"], name="weekday")


def test_expand_template_rejects_value_slugging_to_invalid_label():
    # A value with no leading letter slugs to a label the grammar rejects.
    with pytest.raises(ManifoldFormatError, match="not a valid node label"):
        expand_template(
            "[N]",
            [{"user": "pick a number", "assistant": "the number is [N]"}],
            ["7", "8"],
            name="numbers",
        )


# ---- folder round-trip -----------------------------------------------------

def test_create_templated_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_templated_manifold_folder(
        "local", "weekday", "days of the week",
        fit_mode="auto", slot=SLOT, pairs=PAIRS, values=VALUES,
    )
    mf = ManifoldFolder.load(folder)
    assert mf.fit_mode == "auto"
    assert mf.is_discover
    assert mf.node_labels == ["monday", "tuesday", "sunday"]
    assert mf.template is not None
    assert mf.template["slot"] == SLOT
    assert mf.template["values"] == VALUES
    # The user turns are the per-manifold elicitation prompt set the fit uses.
    assert [p["user"] for p in mf.template["pairs"]] == [
        "what day is it?", "remind me what day this is?",
    ]
    # Node corpora are the filled assistant turns, one per template.
    groups = dict(mf.node_groups())
    assert groups["tuesday"] == ["today is Tuesday", "it's Tuesday"]


def test_templated_write_metadata_preserves_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_templated_manifold_folder(
        "local", "weekday", "", fit_mode="auto",
        slot=SLOT, pairs=PAIRS, values=VALUES,
    )
    mf = ManifoldFolder.load(folder)
    sha_before = mf.nodes_sha256()
    mf.write_metadata()  # the post-fit manifest rewrite
    mf2 = ManifoldFolder.load(folder)
    assert mf2.template == mf.template
    assert mf2.nodes_sha256() == sha_before


def test_templated_sha_sensitive_to_user_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A user-turn edit lives only in the block, so the sha must fold it in."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_templated_manifold_folder(
        "local", "weekday", "", fit_mode="auto",
        slot=SLOT, pairs=PAIRS, values=VALUES,
    )
    mf = ManifoldFolder.load(folder)
    assert mf.template is not None
    sha_before = mf.nodes_sha256()
    edited = copy.deepcopy(mf.template)
    edited["pairs"][0]["user"] = "so, which day are we on?"
    mf.template = edited
    assert mf.nodes_sha256() != sha_before


def test_non_templated_discover_has_no_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """An ordinary discover folder stays byte-identical: no template block."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "plain", "", fit_mode="pca",
        node_corpora={"a": ["a says", "a too"], "b": ["b says", "b too"]},
        hyperparams={"max_dim": 1},
    )
    mf = ManifoldFolder.load(folder)
    assert mf.template is None


def test_load_rejects_template_on_authored(tmp_path: Path):
    """A ``template`` block on a non-discover folder is a format error."""
    import json
    folder = tmp_path / "bad"
    (folder / "nodes").mkdir(parents=True)
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": 5,
        "name": "bad",
        "fit_mode": "authored",
        "domain": {"type": "box", "axes": [
            {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0},
        ]},
        "nodes": [
            {"label": "a", "coords": [0.0]},
            {"label": "b", "coords": [0.5]},
            {"label": "c", "coords": [1.0]},
        ],
        "template": {"slot": "[X]", "values": ["a", "b"], "pairs": [
            {"user": "q", "assistant": "[X]"},
        ]},
        "files": {},
    }))
    with pytest.raises(ManifoldFormatError, match="discover-mode feature"):
        ManifoldFolder.load(folder)
