"""Standalone template artifact (``saklas.io.templates``) — CPU-only, no model.

Covers the multi-turn schema, the ``expand``/``score_inputs`` derived views, the
on-disk round-trip + content hash, resolution (bare / ns-qualified / ambiguous),
and the validation invariants (slot only in the final assistant turn, last history
turn is user, slot appears exactly once).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from saklas.io.templates import (
    AmbiguousTemplateError,
    TemplateFolder,
    TemplateFormatError,
    TemplateNotFoundError,
    create_template_folder,
    iter_template_folders,
    remove_template_folder,
    resolve_template,
)

SLOT = "[DAY]"
VALUES = ["Monday", "Tuesday", "Sunday"]
CONTEXTS = [
    {"turns": [{"role": "user", "content": "what day is it?"}],
     "assistant": "today is [DAY]"},
    {"turns": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
        {"role": "user", "content": "remind me what day this is?"}],
     "assistant": "it's [DAY], friend"},
]


@pytest.fixture(autouse=True)
def _home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))


def _make(name: str = "weekday", **kw: Any) -> TemplateFolder:
    return create_template_folder(
        "local", name, slot=SLOT, values=VALUES, contexts=CONTEXTS, **kw,
    )


# ---- derived views ---------------------------------------------------------

def test_node_labels_and_corpora():
    t = _make()
    assert t.node_labels() == ["monday", "tuesday", "sunday"]
    corpora = t.node_corpora()
    # One entry per context, slot filled by the RAW value, corpus[i] ↔ context[i].
    assert corpora["monday"] == ["today is Monday", "it's Monday, friend"]
    assert corpora["sunday"] == ["today is Sunday", "it's Sunday, friend"]


def test_score_inputs_split_slot():
    t = _make()
    si = t.score_inputs()
    assert len(si) == 2
    # Context 0: single-turn history, slot at the end → empty suffix.
    assert si[0]["messages"] == [{"role": "user", "content": "what day is it?"}]
    assert si[0]["assistant_prefix"] == "today is "
    assert si[0]["suffix"] == ""
    assert si[0]["choices"] == VALUES
    assert si[0]["labels"] == ["monday", "tuesday", "sunday"]
    # Context 1: multi-turn history preserved; slot mid-string → non-empty suffix.
    assert si[1]["messages"][-1] == {"role": "user", "content": "remind me what day this is?"}
    assert si[1]["assistant_prefix"] == "it's "
    assert si[1]["suffix"] == ", friend"


def test_multiword_value_slug():
    t = create_template_folder(
        "local", "cities", slot="[CITY]", values=["New York", "São Paulo"],
        contexts=[{"turns": [{"role": "user", "content": "where?"}],
                   "assistant": "in [CITY]"}],
    )
    assert t.node_labels() == ["new_york", "s_o_paulo"]
    assert t.node_corpora()["new_york"] == ["in New York"]


# ---- round-trip + hash -----------------------------------------------------

def test_round_trip_and_sha_stable():
    t = _make(description="day ring", tags=["temporal"])
    r = resolve_template("weekday")
    assert r.slot == SLOT
    assert list(r.values) == VALUES
    assert len(r.contexts) == 2
    assert r.description == "day ring"
    assert list(r.tags) == ["temporal"]
    assert r.sha256() == t.sha256()


def test_sha_ignores_description_but_tracks_content():
    t = _make(description="one")
    sha = t.sha256()
    # Re-author with a different description, same content → same sha.
    t2 = create_template_folder(
        "local", "weekday", slot=SLOT, values=VALUES, contexts=CONTEXTS,
        description="two different", force=True,
    )
    assert t2.sha256() == sha
    # Edit a value → sha changes.
    t3 = create_template_folder(
        "local", "weekday", slot=SLOT, values=["Monday", "Tuesday", "Saturday"],
        contexts=CONTEXTS, force=True,
    )
    assert t3.sha256() != sha


# ---- resolution ------------------------------------------------------------

def test_resolve_bare_and_qualified():
    _make("weekday")
    assert resolve_template("weekday").name == "weekday"
    assert resolve_template("local/weekday").name == "weekday"


def test_resolve_missing_raises():
    with pytest.raises(TemplateNotFoundError):
        resolve_template("nope")


def test_resolve_ambiguous_raises():
    create_template_folder("local", "dup", slot=SLOT, values=VALUES, contexts=CONTEXTS)
    create_template_folder("other", "dup", slot=SLOT, values=VALUES, contexts=CONTEXTS)
    with pytest.raises(AmbiguousTemplateError):
        resolve_template("dup")
    # Qualifying disambiguates.
    assert resolve_template("other/dup").name == "dup"


def test_iter_and_remove():
    _make("weekday")
    names = {t.name for t in iter_template_folders()}
    assert "weekday" in names
    assert remove_template_folder("local", "weekday") is True
    assert remove_template_folder("local", "weekday") is False


# ---- validation invariants -------------------------------------------------

def test_rejects_slot_in_history_turn():
    with pytest.raises(TemplateFormatError, match="not contain the slot"):
        create_template_folder(
            "local", "bad", slot=SLOT, values=VALUES,
            contexts=[{"turns": [{"role": "user", "content": "is it [DAY]?"}],
                       "assistant": "yes, [DAY]"}],
        )


def test_rejects_assistant_without_slot():
    with pytest.raises(TemplateFormatError, match="exactly once"):
        create_template_folder(
            "local", "bad", slot=SLOT, values=VALUES,
            contexts=[{"turns": [{"role": "user", "content": "q"}],
                       "assistant": "no slot here"}],
        )


def test_rejects_slot_twice():
    with pytest.raises(TemplateFormatError, match="exactly once"):
        create_template_folder(
            "local", "bad", slot=SLOT, values=VALUES,
            contexts=[{"turns": [{"role": "user", "content": "q"}],
                       "assistant": "[DAY] and [DAY]"}],
        )


def test_rejects_last_history_turn_not_user():
    with pytest.raises(TemplateFormatError, match="last history turn must be 'user'"):
        create_template_folder(
            "local", "bad", slot=SLOT, values=VALUES,
            contexts=[{"turns": [{"role": "assistant", "content": "hi"}],
                       "assistant": "it's [DAY]"}],
        )


def test_requires_two_values():
    with pytest.raises(TemplateFormatError, match="'values'"):
        create_template_folder(
            "local", "bad", slot=SLOT, values=["Monday"], contexts=CONTEXTS,
        )


def test_requires_contexts():
    with pytest.raises(TemplateFormatError, match="'contexts'"):
        create_template_folder("local", "bad", slot=SLOT, values=VALUES, contexts=[])


def test_rejects_value_slugging_to_invalid_label():
    with pytest.raises(TemplateFormatError, match="not a valid node label"):
        create_template_folder(
            "local", "numbers", slot="[N]", values=["7", "8"],
            contexts=[{"turns": [{"role": "user", "content": "pick"}],
                       "assistant": "the number is [N]"}],
        )


def test_single_turn_sugar_via_explicit_user_turn():
    # The degenerate single-turn case: a one-element user history.
    t = create_template_folder(
        "local", "simple", slot=SLOT, values=VALUES,
        contexts=[{"turns": [{"role": "user", "content": "day?"}],
                   "assistant": "[DAY]"}],
    )
    si = t.score_inputs()
    assert si[0]["assistant_prefix"] == ""
    assert si[0]["suffix"] == ""
