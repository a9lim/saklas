"""Tests for the loom tree-pruning filter grammar (v2.3 phase 5)."""
from __future__ import annotations

import pytest

from saklas import (
    FilterParseError,
    LoomTree,
    Recipe,
    parse_filter,
)


# ---------------------------------------------------------------------------
# Grammar parsing
# ---------------------------------------------------------------------------


def test_parse_agg_op():
    fc = parse_filter("agg:angry.calm > 0.4")
    assert len(fc.clauses) == 1
    c = fc.clauses[0]
    assert c.agg == "agg"
    assert c.probe == "angry.calm"
    assert c.op == ">"
    assert c.threshold == 0.4


@pytest.mark.parametrize("op", [">", ">=", "<", "<="])
def test_parse_all_operators(op: str) -> None:
    fc = parse_filter(f"agg:honest {op} 0")
    assert fc.clauses[0].op == op


@pytest.mark.parametrize("agg", ["agg", "any", "last"])
def test_parse_all_agg_ops(agg: str) -> None:
    fc = parse_filter(f"{agg}:honest > 0.1")
    assert fc.clauses[0].agg == agg


def test_parse_multi_clause_and():
    fc = parse_filter(
        "any:hallucinating.grounded > 0.7, agg:honest > 0, last:refusal.compliant < 0"
    )
    assert len(fc.clauses) == 3
    assert [c.agg for c in fc.clauses] == ["any", "agg", "last"]


def test_parse_multi_word_probe_name():
    fc = parse_filter("agg:high_context.low_context >= 0.3")
    assert fc.clauses[0].probe == "high_context.low_context"


def test_parse_negative_threshold():
    fc = parse_filter("agg:angry.calm < -0.5")
    assert fc.clauses[0].threshold == -0.5


def test_parse_decimal_threshold():
    fc = parse_filter("agg:warm > 0.25")
    assert fc.clauses[0].threshold == 0.25


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------


def test_empty_raises():
    with pytest.raises(FilterParseError):
        parse_filter("")
    with pytest.raises(FilterParseError):
        parse_filter("   ")


def test_bare_clause_defaults_to_agg():
    """A bare ``<probe> <op> <num>`` clause defaults to ``agg:`` — the
    plan calls ``agg`` the default and the parser used to reject the
    prefix-less form (v2.3 fix).  Explicit prefixed forms keep working.
    """
    fc = parse_filter("angry.calm > 0.4")
    assert len(fc.clauses) == 1
    c = fc.clauses[0]
    assert c.agg == "agg"
    assert c.probe == "angry.calm"
    assert c.op == ">"
    assert c.threshold == 0.4

    # Multi-clause: bare + explicit, mixed.
    fc2 = parse_filter("angry.calm > 0.4, any:honest >= 0.2")
    assert [c.agg for c in fc2.clauses] == ["agg", "any"]


def test_unknown_agg_op():
    with pytest.raises(FilterParseError, match="unknown agg"):
        parse_filter("mean:angry > 0.4")


def test_missing_op():
    with pytest.raises(FilterParseError, match="missing comparison op"):
        parse_filter("agg:angry 0.4")


def test_missing_threshold():
    with pytest.raises(FilterParseError, match="missing threshold"):
        parse_filter("agg:angry >")


def test_non_numeric_threshold():
    with pytest.raises(FilterParseError, match="not a number"):
        parse_filter("agg:angry > foo")


def test_invalid_probe_name():
    with pytest.raises(FilterParseError, match="not a valid identifier"):
        parse_filter("agg:1bad > 0.4")


# ---------------------------------------------------------------------------
# Evaluate against LoomNodes
# ---------------------------------------------------------------------------


class _SyntheticNode:
    """Tiny stand-in for LoomNode — only ``aggregate_readings`` is read."""

    def __init__(self, readings: dict[str, float]) -> None:
        self.id = "n0"
        self.aggregate_readings = readings


def test_evaluate_agg_op_pass():
    fc = parse_filter("agg:angry.calm > 0.4")
    node = _SyntheticNode({"angry.calm": 0.7})
    assert fc.evaluate(node) is True


def test_evaluate_agg_op_fail():
    fc = parse_filter("agg:angry.calm > 0.4")
    node = _SyntheticNode({"angry.calm": 0.2})
    assert fc.evaluate(node) is False


def test_evaluate_missing_probe_is_false():
    """Documented: missing probe → clause is False under AND semantics."""
    fc = parse_filter("agg:angry.calm > 0.4")
    node = _SyntheticNode({"honest": 0.5})
    assert fc.evaluate(node) is False


def test_evaluate_multi_clause_and():
    fc = parse_filter("agg:angry > 0.4, agg:honest > 0")
    n1 = _SyntheticNode({"angry": 0.5, "honest": 0.1})
    n2 = _SyntheticNode({"angry": 0.5, "honest": -0.1})
    n3 = _SyntheticNode({"angry": 0.1, "honest": 0.5})
    assert fc.evaluate(n1) is True
    assert fc.evaluate(n2) is False
    assert fc.evaluate(n3) is False


def test_evaluate_any_op_uses_per_token():
    fc = parse_filter("any:angry > 0.5")
    node = _SyntheticNode({"angry": 0.2})
    # No per-token table → any clause fails.
    assert fc.evaluate(node) is False
    # With per-token scores where max > 0.5.
    assert fc.evaluate(node, per_token_scores={"angry": [0.1, 0.6, 0.2]}) is True
    # All below.
    assert fc.evaluate(node, per_token_scores={"angry": [0.1, 0.2]}) is False


def test_evaluate_last_op_uses_per_token():
    fc = parse_filter("last:refusal.compliant < 0")
    node = _SyntheticNode({"refusal.compliant": 0.2})
    assert fc.evaluate(node, per_token_scores={"refusal.compliant": [0.5, -0.2]}) is True
    assert fc.evaluate(node, per_token_scores={"refusal.compliant": [0.5, 0.1]}) is False


def test_evaluate_any_lt_uses_min():
    fc = parse_filter("any:angry < 0")
    node = _SyntheticNode({"angry": 0.5})
    assert fc.evaluate(node, per_token_scores={"angry": [0.3, -0.2, 0.4]}) is True
    assert fc.evaluate(node, per_token_scores={"angry": [0.3, 0.2, 0.4]}) is False


# ---------------------------------------------------------------------------
# LoomTree integration
# ---------------------------------------------------------------------------


def test_filter_by_expr_returns_matching_ids():
    t = LoomTree()
    u = t.add_user_turn("hi")

    a1 = t.begin_assistant(u, recipe=Recipe())
    t.finalize_assistant(a1, text="warm", aggregate_readings={"angry.calm": 0.6})

    a2 = t.begin_assistant(u, recipe=Recipe())
    t.finalize_assistant(a2, text="cold", aggregate_readings={"angry.calm": 0.1})

    a3 = t.begin_assistant(u, recipe=Recipe())
    t.finalize_assistant(a3, text="missing", aggregate_readings={"honest": 0.4})

    ids = t.filter_by_expr("agg:angry.calm > 0.4")
    assert a1 in ids
    assert a2 not in ids
    assert a3 not in ids
