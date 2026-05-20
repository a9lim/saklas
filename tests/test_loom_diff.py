"""Tests for cross-branch diff helpers (v2.3 phase 5)."""
from __future__ import annotations

import pytest

from saklas import (
    LoomTree,
    Recipe,
    per_token_diff,
    readings_diff,
    steering_delta,
    text_diff,
)
from saklas.core.loom_diff import NodeDiff


# ---------------------------------------------------------------------------
# text_diff
# ---------------------------------------------------------------------------


def test_text_diff_identical():
    spans = text_diff("hello world", "hello world")
    assert all(s.state == "equal" for s in spans)
    assert " ".join(s.text for s in spans) == "hello world"


def test_text_diff_insert():
    spans = text_diff("hello world", "hello there world")
    states = [s.state for s in spans]
    assert "insert" in states
    assert "equal" in states
    insert_span = next(s for s in spans if s.state == "insert")
    assert insert_span.text == "there"


def test_text_diff_delete():
    spans = text_diff("hello there world", "hello world")
    states = [s.state for s in spans]
    assert "delete" in states


def test_text_diff_replace_emits_delete_then_insert():
    spans = text_diff("hello world", "hello earth")
    states = [s.state for s in spans]
    # "world" becomes "earth" — splits to delete world + insert earth.
    assert "delete" in states
    assert "insert" in states


def test_text_diff_empty_both():
    assert text_diff("", "") == []


def test_text_diff_empty_b_is_pure_delete():
    spans = text_diff("hello world", "")
    assert all(s.state == "delete" for s in spans)


# ---------------------------------------------------------------------------
# readings_diff
# ---------------------------------------------------------------------------


def test_readings_diff_sorted_by_magnitude():
    a = {"angry": 0.3, "honest": 0.1, "warm": 0.0}
    b = {"angry": 0.4, "honest": -0.6, "warm": 0.05}
    out = readings_diff(a, b)
    # Sorted descending by |delta|.
    abs_deltas = [abs(r.delta) for r in out]
    assert abs_deltas == sorted(abs_deltas, reverse=True)
    # Honest had the biggest swing.
    assert out[0].name == "honest"
    assert pytest.approx(out[0].delta, abs=1e-9) == -0.7


def test_readings_diff_handles_missing_keys():
    a = {"angry": 0.3}
    b = {"warm": 0.2}
    out = readings_diff(a, b)
    names = {r.name for r in out}
    assert names == {"angry", "warm"}
    # The "missing" entries carry zero defaults on the absent side.
    angry = next(r for r in out if r.name == "angry")
    assert angry.a_value == 0.3 and angry.b_value == 0.0
    warm = next(r for r in out if r.name == "warm")
    assert warm.a_value == 0.0 and warm.b_value == 0.2


def test_readings_diff_empty_inputs():
    assert readings_diff({}, {}) == []


# ---------------------------------------------------------------------------
# per_token_diff
# ---------------------------------------------------------------------------


def test_per_token_diff_aligned_simple():
    a = ["hello", "world"]
    b = ["hello", "world"]
    spans = per_token_diff(a, b)
    assert len(spans) == 2
    assert all(s.aligned for s in spans)


def test_per_token_diff_diverges():
    a = ["hello", "world", "foo"]
    b = ["hello", "earth", "bar"]
    spans = per_token_diff(a, b)
    # First token aligns; subsequent ones diverge.
    assert spans[0].aligned is True
    assert any(not s.aligned for s in spans[1:])


def test_per_token_diff_reading_deltas_only_on_aligned():
    a = ["hi", "there"]
    b = ["hi", "there"]
    a_scores = {"angry": [0.1, 0.2]}
    b_scores = {"angry": [0.4, 0.3]}
    spans = per_token_diff(a, b, a_scores=a_scores, b_scores=b_scores)
    assert all(s.aligned for s in spans)
    assert spans[0].reading_deltas[0].delta == pytest.approx(0.3)
    assert spans[1].reading_deltas[0].delta == pytest.approx(0.1)


def test_per_token_diff_unequal_lengths():
    a = ["hi"]
    b = ["hi", "there"]
    spans = per_token_diff(a, b)
    assert len(spans) == 2
    assert spans[-1].aligned is False
    assert spans[-1].b_text == "there"


# ---------------------------------------------------------------------------
# session.diff_nodes
# ---------------------------------------------------------------------------


def _seed_tree_with_siblings():
    t = LoomTree()
    u = t.add_user_turn("ping")
    a1 = t.begin_assistant(u, recipe=Recipe(steering="0.3 honest"))
    t.finalize_assistant(
        a1, text="hello there", aggregate_readings={"angry": 0.2, "warm": 0.5},
    )
    a2 = t.begin_assistant(u, recipe=Recipe(steering="0.5 warm"))
    t.finalize_assistant(
        a2, text="hello world", aggregate_readings={"angry": 0.4, "warm": 0.3},
    )
    return t, u, a1, a2


def test_session_diff_nodes_via_synthetic_session():
    """The session method is a thin wrapper; exercise via a small shim."""
    # Build a minimal stand-in that exposes ``.tree`` so we can call the
    # method via __get__ binding.  Avoids loading a real model.
    from saklas.core.session import SaklasSession

    tree, u, a1, a2 = _seed_tree_with_siblings()

    class _Stub:
        def __init__(self, tree):
            self.tree = tree

    stub = _Stub(tree)
    diff = SaklasSession.diff_nodes.__get__(stub, _Stub)(a1, a2)
    assert isinstance(diff, NodeDiff)
    assert diff.a_id == a1
    assert diff.b_id == a2
    assert diff.parent_id == u
    # Both readings appear in the delta table.
    names = {r.name for r in diff.readings}
    assert "angry" in names and "warm" in names


# ---------------------------------------------------------------------------
# steering_delta edge label helper
# ---------------------------------------------------------------------------


def test_steering_delta_identical_empty():
    assert steering_delta("0.3 honest", "0.3 honest") == ""


def test_steering_delta_none_to_steered():
    label = steering_delta(None, "0.3 honest")
    # parsed "honest" canonicalizes via pole resolution; label exposes the
    # delta on whichever canonical name landed.
    assert label  # non-empty
    assert "0.3" in label or "+0.3" in label


def test_steering_delta_add_term():
    label = steering_delta("0.3 honest", "0.3 honest + 0.2 warm")
    assert "warm" in label


def test_steering_delta_unparseable_returns_empty():
    # Both unparseable expressions → no change → empty.
    assert steering_delta("???", "???") == ""
