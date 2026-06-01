"""Grammar tests for the manifold ``%`` operator in steering expressions.

Bare names that don't match an installed pack resolve to themselves, so
these run in an isolated SAKLAS_HOME with no packs installed.
"""
from __future__ import annotations

from typing import Any

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
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    sel.invalidate()
    yield
    sel.invalidate()


def _only_term(expr: str) -> Any:
    steering = parse_expr(expr)
    assert len(steering.alphas) == 1
    return next(iter(steering.alphas.values()))


# --------------------------------------------------------------- parsing ---

def test_basic_manifold_term():
    term = _only_term("emotions%0.5")
    assert isinstance(term, ManifoldTerm)
    assert term.manifold == "emotions"
    assert term.position == (0.5,)
    assert term.coeff == 0.5  # DEFAULT_COEFF
    assert term.trigger == Trigger.BOTH


def test_explicit_coeff():
    assert _only_term("0.7 emotions%0.5").coeff == pytest.approx(0.7)
    assert _only_term("0.7*emotions%0.5").coeff == pytest.approx(0.7)


def test_single_coord_position():
    assert _only_term("emotions%0").position == (0.0,)
    assert _only_term("emotions%1").position == (1.0,)


def test_coord_list_parsing():
    term = _only_term("circumplex%0.3,0.8")
    assert isinstance(term, ManifoldTerm)
    assert term.position == (0.3, 0.8)


def test_coord_list_three_dims():
    term = _only_term("vad%0.2,0.5,0.9")
    assert term.position == (0.2, 0.5, 0.9)


def test_negative_coord():
    # Negative authoring coordinates are valid (e.g. a disk centered at
    # the origin); range validation happens at manifold-load time.
    term = _only_term("circumplex%-0.4,0.6")
    assert isinstance(term, ManifoldTerm)
    assert term.position == (-0.4, 0.6)


def test_out_of_unit_range_coord_parses():
    # The grammar no longer range-checks — it does not know the domain.
    term = _only_term("emotions%1.5")
    assert term.position == (1.5,)


def test_trigger_on_manifold_term():
    term = _only_term("circumplex%0.3,0.8@response")
    assert isinstance(term, ManifoldTerm)
    assert term.trigger == Trigger.GENERATED_ONLY
    assert term.position == (0.3, 0.8)


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


def test_manifold_compose_guards_share_one_message():
    """The ``%``-with-projection guard and the ``!``-with-``%`` guard emit
    the same consolidated message (parse-time, ``SteeringExprError``)."""
    from saklas.core.steering_expr import _MANIFOLD_COMPOSE_MSG

    with pytest.raises(SteeringExprError) as proj:
        parse_expr("emotions%0.5~angry")
    with pytest.raises(SteeringExprError) as abl:
        parse_expr("!emotions%0.5")

    assert _MANIFOLD_COMPOSE_MSG in str(proj.value)
    assert _MANIFOLD_COMPOSE_MSG in str(abl.value)


def test_rejects_missing_position():
    with pytest.raises(SteeringExprError):
        parse_expr("emotions%")


def test_rejects_trailing_comma():
    with pytest.raises(SteeringExprError):
        parse_expr("circumplex%0.3,")


def test_conflicting_triggers_raise():
    with pytest.raises(SteeringExprError):
        parse_expr("emotions%0.5@response + emotions%0.5@thinking")


# ----------------------------------------------------------------- merge ---

def test_same_position_sums_coeffs():
    steering = parse_expr("0.3 circumplex%0.5,0.5 + 0.4 circumplex%0.5,0.5")
    assert len(steering.alphas) == 1
    term = next(iter(steering.alphas.values()))
    assert isinstance(term, ManifoldTerm)
    assert term.coeff == pytest.approx(0.7)


def test_distinct_positions_stay_separate():
    steering = parse_expr("circumplex%0.2,0.1 + circumplex%0.8,0.9")
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
    "circumplex%0.3,0.8",
    "circumplex%-0.4,0.6@response",
    "vad%0.2,0.5,0.9",
    "local/emotions%0.3",
    "emotions%0.5 + 0.3 angry",
])
def test_format_round_trip(expr: str) -> None:
    steering = parse_expr(expr)
    reparsed = parse_expr(format_expr(steering))
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


# ------------------------------------------------ three-op coeff grammar ---

def test_single_coeff_is_along_only():
    # Collapse op defaults off: one coeff is a pure directional slide.
    term = _only_term("0.6 circumplex%happy")
    assert isinstance(term, ManifoldTerm)
    assert (term.along, term.onto) == pytest.approx((0.6, 0.0))
    assert term.coeff == pytest.approx(0.6)  # representative = along


def test_two_coeffs_are_along_onto():
    # Two coeffs opt into onto (the only collapse op now).
    term = _only_term("0.6,0.3 circumplex%happy")
    assert isinstance(term, ManifoldTerm)
    assert term.along == pytest.approx(0.6)
    assert term.onto == pytest.approx(0.3)


def test_two_coeffs_with_coord_position():
    # The post-``%`` comma list is the POSITION; the pre-selector comma
    # run is the coefficient run.  They must not bleed into each other.
    term = _only_term("0.6,0.3 circumplex%0.3,0.8")
    assert isinstance(term, ManifoldTerm)
    assert term.along == pytest.approx(0.6)
    assert term.onto == pytest.approx(0.3)
    assert term.position == (0.3, 0.8)


def test_single_coeff_with_coord_position_unchanged():
    # One coeff, coord-list position — the historical shape (along-only now).
    term = _only_term("0.6 circumplex%0.3,0.8")
    assert (term.along, term.onto) == pytest.approx((0.6, 0.0))
    assert term.position == (0.3, 0.8)


def test_negative_sign_propagates_across_run():
    term = _only_term("-0.6,0.3 circumplex%happy")
    assert isinstance(term, ManifoldTerm)
    assert term.along == pytest.approx(-0.6)
    assert term.onto == pytest.approx(-0.3)


@pytest.mark.parametrize("expr", [
    "0.6 circumplex%happy",        # 1 coeff → shortest is 1
    "0.6,0.3 circumplex%happy",    # onto opted in → 2
    "0.6,0.3 persona%pirate",      # label-form, 2 coeffs
    "0.6,0.3 circumplex%0.3,0.8",  # 2 coeffs + coord position
])
def test_manifold_format_round_trips_byte_for_byte(expr: str) -> None:
    assert format_expr(parse_expr(expr)) == expr


def test_manifold_coeffs_sum_on_merge():
    steering = parse_expr(
        "0.6,0.3 circumplex%happy + 0.1,0.2 circumplex%happy"
    )
    assert len(steering.alphas) == 1
    term = next(iter(steering.alphas.values()))
    assert isinstance(term, ManifoldTerm)
    assert term.along == pytest.approx(0.7)
    assert term.onto == pytest.approx(0.5)


def test_rejects_three_coeffs():
    with pytest.raises(SteeringExprError):
        parse_expr("0.6,0.3,0.2 circumplex%happy")


def test_mixed_sign_manifold_term_round_trips():
    # A programmatically-built term with a negative ``along`` and a positive
    # ``onto`` (e.g. from ``Recipe.invert_steering``) must survive a
    # format→parse round-trip.  The parser propagates the leading sign across
    # the run, so ``_fmt_manifold`` renders ``onto`` relative to ``along``'s
    # sign to keep the values intact.
    from saklas.core.steering import Steering

    term = ManifoldTerm(
        along=-0.6, onto=0.3, trigger=Trigger.BOTH,
        manifold="circumplex", position="happy",
    )
    steering = Steering(alphas={"circumplex%happy": term})
    reparsed = next(iter(parse_expr(format_expr(steering)).alphas.values()))
    assert isinstance(reparsed, ManifoldTerm)
    assert reparsed.along == pytest.approx(-0.6)
    assert reparsed.onto == pytest.approx(0.3)


@pytest.mark.parametrize("expr", [
    "0.6,0.3 honest",          # plain vector term
    "0.6,0.3 angry~calm",      # projection term
    "0.6,0.3 !angry",          # ablation term
])
def test_rejects_comma_coeffs_on_non_manifold(expr: str) -> None:
    with pytest.raises(SteeringExprError) as exc:
        parse_expr(expr)
    assert "manifold % position" in str(exc.value)


# ------------------------------------------------------------- reference ---

def test_referenced_manifolds():
    refs = referenced_manifolds("emotions%0.5 + local/mood%0.2 + 0.3 angry")
    assert (None, "emotions", "raw") in refs
    assert ("local", "mood", "raw") in refs
    assert len(refs) == 2


def test_referenced_selectors_skips_manifolds():
    refs = referenced_selectors("emotions%0.5 + 0.3 angry")
    assert refs == [(None, "angry", "raw")]
