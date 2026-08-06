"""Tests for the loom tree-pruning filter grammar (v2.3 phase 5)."""
from __future__ import annotations

import pytest

from saklas import (
    FilterParseError,
    LoomTree,
    Recipe,
    parse_filter,
)
from saklas.core.tree_filter import node_token_series


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
    """Tiny stand-in for LoomNode — aggregates plus optional token rows."""

    def __init__(
        self,
        readings: dict[str, float],
        *,
        tokens: list[dict] | None = None,
        thinking_tokens: list[dict] | None = None,
    ) -> None:
        self.id = "n0"
        self.aggregate_readings = readings
        self.tokens = tokens
        self.thinking_tokens = thinking_tokens


def _rows(probe: str, values: list[float]) -> list[dict]:
    """Token rows carrying ``probe`` in the 5.x measurement envelope."""
    return [
        {
            "text": f"t{i}",
            "measurements": {
                "version": 1,
                "scope": "token",
                "scores": {probe: v},
            },
        }
        for i, v in enumerate(values)
    ]


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
    # No token rows → the clause fails.
    assert fc.evaluate(_SyntheticNode({"angry": 0.2})) is False
    # Rows whose max > 0.5.
    node = _SyntheticNode({"angry": 0.2}, tokens=_rows("angry", [0.1, 0.6, 0.2]))
    assert fc.evaluate(node) is True
    # All below.
    node = _SyntheticNode({"angry": 0.2}, tokens=_rows("angry", [0.1, 0.2]))
    assert fc.evaluate(node) is False


def test_evaluate_last_op_uses_per_token():
    fc = parse_filter("last:refusal.compliant < 0")
    node = _SyntheticNode(
        {"refusal.compliant": 0.2},
        tokens=_rows("refusal.compliant", [0.5, -0.2]),
    )
    assert fc.evaluate(node) is True
    node = _SyntheticNode(
        {"refusal.compliant": 0.2},
        tokens=_rows("refusal.compliant", [0.5, 0.1]),
    )
    assert fc.evaluate(node) is False


def test_evaluate_any_lt_uses_min():
    fc = parse_filter("any:angry < 0")
    node = _SyntheticNode({"angry": 0.5}, tokens=_rows("angry", [0.3, -0.2, 0.4]))
    assert fc.evaluate(node) is True
    node = _SyntheticNode({"angry": 0.5}, tokens=_rows("angry", [0.3, 0.2, 0.4]))
    assert fc.evaluate(node) is False


def test_token_series_reads_thinking_then_response_rows():
    """Decode order is thinking rows first — ``last:`` sees the response tail."""
    node = _SyntheticNode(
        {"angry": 0.0},
        thinking_tokens=_rows("angry", [0.9]),
        tokens=_rows("angry", [0.1, -0.4]),
    )
    assert node_token_series(node, frozenset({"angry"})) == {
        "angry": [0.9, 0.1, -0.4],
    }
    assert parse_filter("last:angry < 0").evaluate(node) is True
    assert parse_filter("any:angry > 0.8").evaluate(node) is True


def test_token_series_falls_back_to_flat_probes_alias():
    """Rows written before the envelope carry the flat ``probes`` alias."""
    node = _SyntheticNode(
        {"angry": 0.0},
        tokens=[{"text": "a", "probes": {"angry": 0.7}}],
    )
    assert node_token_series(node, frozenset({"angry"})) == {"angry": [0.7]}
    assert parse_filter("any:angry > 0.5").evaluate(node) is True


def test_token_series_skips_rows_missing_the_probe():
    """A probe attached mid-generation yields the readings it actually has."""
    node = _SyntheticNode(
        {"angry": 0.0},
        tokens=[
            {"text": "a"},
            {"text": "b", "measurements": {"scores": {"other": 1.0}}},
            {"text": "c", "measurements": {"scores": {"angry": 0.3}}},
        ],
    )
    assert node_token_series(node, frozenset({"angry"})) == {"angry": [0.3]}
    assert parse_filter("last:angry > 0.2").evaluate(node) is True


def test_agg_only_expression_skips_token_walk():
    """``token_probes`` is empty for an ``agg:``-only filter."""
    assert parse_filter("agg:angry > 0.1, honest < 0").token_probes == frozenset()
    assert parse_filter("any:angry > 0.1, agg:honest < 0").token_probes == frozenset(
        {"angry"},
    )


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


def test_filter_by_expr_any_last_read_node_token_rows():
    """``any:`` / ``last:`` match through the plain tree API — no side table.

    Regression for the plumbing hole: the only caller that ever reached
    ``filter_by_expr`` supplied no per-token table, so both ops used to
    return an empty match set silently.
    """
    t = LoomTree()
    u = t.add_user_turn("hi")

    spiky = t.begin_assistant(u, recipe=Recipe())
    for value in (0.1, 0.9, 0.1):
        t.append_token(spiky, {"text": "x", "measurements": {"scores": {"a": value}}})
    t.finalize_assistant(spiky, text="spiky", aggregate_readings={"a": 0.3})

    flat = t.begin_assistant(u, recipe=Recipe())
    for value in (0.1, 0.2, 0.8):
        t.append_token(flat, {"text": "x", "measurements": {"scores": {"a": value}}})
    t.finalize_assistant(flat, text="flat", aggregate_readings={"a": 0.3})

    assert t.filter_by_expr("any:a > 0.85") == {spiky}
    assert t.filter_by_expr("any:a > 0.75") == {spiky, flat}
    assert t.filter_by_expr("last:a > 0.5") == {flat}
    # AND across ops, mixing the aggregate and per-token tables.
    assert t.filter_by_expr("agg:a > 0.2, last:a < 0.5") == {spiky}


# ---------------------------------------------------------------------------
# HTTP route — the only surface that exposes the grammar
# ---------------------------------------------------------------------------


def test_filter_route_matches_all_three_ops():
    """``GET /tree/filter`` resolves ``agg:``/``any:``/``last:`` alike.

    The route is the grammar's only caller and supplied no per-token table,
    so ``any:``/``last:`` used to return an empty match set through it.
    """
    from typing import cast

    from fastapi.testclient import TestClient

    from saklas.core.session import SaklasSession
    from saklas.server import create_app
    from tests.test_server_loom import _StubSession

    session = _StubSession()
    client = TestClient(create_app(cast(SaklasSession, session), default_steering=None))
    tree = session.tree

    u = tree.add_user_turn("hi")
    spiky = tree.begin_assistant(u, recipe=Recipe())
    for value in (0.1, 0.9, 0.1):
        tree.append_token(spiky, {"text": "x", "measurements": {"scores": {"a": value}}})
    tree.finalize_assistant(spiky, text="spiky", aggregate_readings={"a": 0.3})

    calm = tree.begin_assistant(u, recipe=Recipe())
    for value in (0.1, 0.2, 0.2):
        tree.append_token(calm, {"text": "x", "measurements": {"scores": {"a": value}}})
    tree.finalize_assistant(calm, text="calm", aggregate_readings={"a": 0.1})

    def matches(expr: str) -> set[str]:
        resp = client.get(
            "/saklas/v1/sessions/default/tree/filter", params={"expr": expr},
        )
        assert resp.status_code == 200, resp.text
        return set(resp.json()["matching_node_ids"])

    assert matches("agg:a > 0.2") == {spiky}
    assert matches("any:a > 0.8") == {spiky}
    assert matches("last:a > 0.15") == {calm}
    assert matches("any:a > 0.05, last:a < 0.15") == {spiky}
    # Bad expressions still land as 400.
    bad = client.get(
        "/saklas/v1/sessions/default/tree/filter", params={"expr": "mean:a > 1"},
    )
    assert bad.status_code == 400
