"""Grammar tests for the manifold ``%`` operator in steering expressions.

Bare names that don't match an installed pack resolve to themselves, so
these run in an isolated SAKLAS_HOME with no packs installed.
"""
from __future__ import annotations

import pytest

from saklas.io import selectors as sel
from saklas.core.steering_expr import (
    ManifoldTerm,
    ProjectedTerm,
    SteeringExprError,
    format_expr,
    parse_expr,
    referenced_manifolds,
    referenced_selectors,
)
from saklas.core.triggers import Trigger


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    sel.invalidate()
    yield
    sel.invalidate()


def _only_term(expr: str):
    steering = parse_expr(expr)
    assert len(steering.alphas) == 1
    return next(iter(steering.alphas.values()))


# --------------------------------------------------------------- parsing ---

def test_basic_manifold_term():
    term = _only_term("emotions%0.5")
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "emotions"
    assert term.position == 0.5
    assert term.coeff == 0.5  # DEFAULT_COEFF
    assert term.trigger == Trigger.BOTH


def test_explicit_coeff():
    assert _only_term("0.7 emotions%0.5").coeff == pytest.approx(0.7)
    assert _only_term("0.7*emotions%0.5").coeff == pytest.approx(0.7)


def test_position_endpoints():
    assert _only_term("emotions%0").position == 0.0
    assert _only_term("emotions%1").position == 1.0


def test_trigger_on_manifold_term():
    term = _only_term("emotions%0.9@response")
    assert isinstance(term, ManifoldTerm)
    assert term.trigger == Trigger.GENERATED_ONLY
    assert term.position == 0.9


def test_probe_gate_on_manifold_term():
    term = _only_term("emotions%0.5@when:angry>0.4")
    assert isinstance(term, ManifoldTerm)
    assert term.trigger.gate is not None
    assert term.trigger.gate.probe == "angry"


def test_namespace_qualified_manifold():
    term = _only_term("local/emotions%0.3")
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "local/emotions"


def test_variant_suffixed_manifold():
    term = _only_term("emotions:sae-gemma%0.3")
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "emotions:sae-gemma"


def test_negative_coeff_via_leading_sign():
    # A leading '-' folds into the coeff (clamped to [0,1] downstream).
    term = _only_term("-0.5 emotions%0.5")
    assert isinstance(term, ManifoldTerm)
    assert term.coeff == pytest.approx(-0.5)


# ------------------------------------------------------------- rejections ---

def test_rejects_projection_composition():
    with pytest.raises(SteeringExprError):
        parse_expr("emotions%0.5~angry")
    with pytest.raises(SteeringExprError):
        parse_expr("emotions~angry%0.5")


def test_rejects_double_percent():
    with pytest.raises(SteeringExprError):
        parse_expr("emotions%0.5%0.3")


def test_rejects_ablation_composition():
    with pytest.raises(SteeringExprError):
        parse_expr("!emotions%0.5")


def test_rejects_out_of_range_position():
    with pytest.raises(SteeringExprError):
        parse_expr("emotions%1.5")
    # A negative position lexes '%' then '-' (MINUS) then NUM — the
    # parser expects a NUM immediately after '%'.
    with pytest.raises(SteeringExprError):
        parse_expr("emotions%-0.1")


def test_rejects_missing_position():
    with pytest.raises(SteeringExprError):
        parse_expr("emotions%")


def test_conflicting_triggers_raise():
    with pytest.raises(SteeringExprError):
        parse_expr("emotions%0.5@response + emotions%0.5@thinking")


# ----------------------------------------------------------------- merge ---

def test_same_position_sums_coeffs():
    steering = parse_expr("0.3 emotions%0.5 + 0.4 emotions%0.5")
    assert len(steering.alphas) == 1
    term = next(iter(steering.alphas.values()))
    assert isinstance(term, ManifoldTerm)
    assert term.coeff == pytest.approx(0.7)


def test_distinct_positions_stay_separate():
    steering = parse_expr("emotions%0.2 + emotions%0.8")
    assert len(steering.alphas) == 2


def test_manifold_composes_with_plain_term():
    steering = parse_expr("emotions%0.5 + 0.3 angry")
    kinds = sorted(type(v).__name__ for v in steering.alphas.values())
    assert kinds == ["ManifoldTerm", "float"]


def test_manifold_composes_with_projection():
    steering = parse_expr("emotions%0.5 + 0.4 angry~calm")
    has_manifold = any(
        isinstance(v, ManifoldTerm) for v in steering.alphas.values()
    )
    has_proj = any(
        isinstance(v, ProjectedTerm) for v in steering.alphas.values()
    )
    assert has_manifold and has_proj


# --------------------------------------------------------------- format ---

@pytest.mark.parametrize("expr", [
    "emotions%0.5",
    "0.7 emotions%0.3",
    "emotions%0.9@response",
    "local/emotions%0.3",
    "emotions%0.5 + 0.3 angry",
])
def test_format_round_trip(expr):
    steering = parse_expr(expr)
    reparsed = parse_expr(format_expr(steering))
    # The IR round-trips: same manifold terms, same positions/coeffs.
    a = sorted(
        (t.manifold, t.position, t.coeff)
        for t in steering.alphas.values()
        if isinstance(t, ManifoldTerm)
    )
    b = sorted(
        (t.manifold, t.position, t.coeff)
        for t in reparsed.alphas.values()
        if isinstance(t, ManifoldTerm)
    )
    assert a == b


# ------------------------------------------------------------- reference ---

def test_referenced_manifolds():
    refs = referenced_manifolds("emotions%0.5 + local/mood%0.2 + 0.3 angry")
    assert (None, "emotions", "raw") in refs
    assert ("local", "mood", "raw") in refs
    assert len(refs) == 2


def test_referenced_selectors_skips_manifolds():
    refs = referenced_selectors("emotions%0.5 + 0.3 angry")
    # Only the plain concept term contributes; the manifold is skipped.
    assert refs == [(None, "angry", "raw")]
