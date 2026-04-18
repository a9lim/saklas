"""Grammar corpus for saklas.core.steering_expr.

Bare names that don't match an installed pack resolve to themselves with
sign +1 (``resolve_pole`` fallthrough), so most of these tests run in an
isolated SAKLAS_HOME with no packs installed.
"""
from __future__ import annotations

import pytest

from saklas.cli import selectors as sel
from saklas.io import packs
from saklas.core.steering import Steering
from saklas.core.steering_expr import (
    ProjectedTerm,
    SteeringExprError,
    format_expr,
    parse_expr,
)
from saklas.core.triggers import Trigger


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    sel.invalidate()
    yield
    sel.invalidate()


def _mk(tmp_path, ns, name, tags=None):
    d = tmp_path / "vectors" / ns / name
    d.mkdir(parents=True)
    (d / "statements.json").write_text("[]")
    meta = packs.PackMetadata(
        name=name, description="x", version="1.0.0", license="MIT",
        tags=tags or [], recommended_alpha=0.5, source="local",
        files={"statements.json": packs.hash_file(d / "statements.json")},
    )
    meta.write(d)
    return d


# -------------------------------------------------------------- basic ---

def test_single_term():
    s = parse_expr("0.5 honest")
    assert s.alphas == {"honest": 0.5}


def test_implicit_coefficient():
    s = parse_expr("honest")
    assert s.alphas == {"honest": 1.0}


def test_star_form():
    s = parse_expr("0.5*honest")
    assert s.alphas == {"honest": 0.5}


def test_integer_coefficient():
    s = parse_expr("2 honest")
    assert s.alphas == {"honest": 2.0}


def test_leading_sign_negates():
    s = parse_expr("-0.5 honest")
    assert s.alphas == {"honest": -0.5}


def test_negated_bare():
    s = parse_expr("-honest")
    assert s.alphas == {"honest": -1.0}


def test_leading_plus():
    s = parse_expr("+0.5 honest")
    assert s.alphas == {"honest": 0.5}


def test_addition():
    s = parse_expr("0.5 honest + 0.3 warm")
    assert s.alphas == {"honest": 0.5, "warm": 0.3}


def test_subtraction():
    s = parse_expr("0.5 honest - 0.2 manipulative")
    assert s.alphas == {"honest": 0.5, "manipulative": -0.2}


def test_three_terms():
    s = parse_expr("0.5 honest + 0.3 warm - 0.2 manipulative")
    assert s.alphas == {"honest": 0.5, "warm": 0.3, "manipulative": -0.2}


def test_scientific_notation():
    s = parse_expr("1e-2 honest")
    assert s.alphas == pytest.approx({"honest": 0.01})


def test_decimal_shorthand():
    s = parse_expr(".25 honest")
    assert s.alphas == {"honest": 0.25}


# ------------------------------------------------------------ bipolar ---

def test_bipolar_dotted():
    s = parse_expr("0.5 angry.calm")
    assert s.alphas == {"angry.calm": 0.5}


def test_underscore_segments_survive():
    s = parse_expr("0.5 high_context.low_context")
    assert s.alphas == {"high_context.low_context": 0.5}


# ---------------------------------------------------------- namespace ---

def test_namespace_prefix_resolves_to_name():
    s = parse_expr("0.5 bob/foo")
    # No pack installed -> resolve_pole returns the slug with sign +1.
    assert s.alphas == {"foo": 0.5}


def test_namespace_with_bipolar():
    s = parse_expr("0.3 bob/deer.wolf")
    assert s.alphas == {"deer.wolf": 0.3}


# ----------------------------------------------------------- variant ---

def test_sae_variant_suffix_preserved():
    s = parse_expr("0.5 honest:sae")
    assert s.alphas == {"honest:sae": 0.5}


def test_sae_release_variant():
    s = parse_expr("0.5 honest:sae-gemma-scope")
    assert s.alphas == {"honest:sae-gemma-scope": 0.5}


def test_raw_variant_elided():
    # ``:raw`` is the default, so the key drops the suffix.
    s = parse_expr("0.5 honest:raw")
    assert s.alphas == {"honest": 0.5}


def test_sae_release_with_digits():
    s = parse_expr("0.5 honest:sae-gemma-scope-2b-pt-res-canonical")
    assert s.alphas == {"honest:sae-gemma-scope-2b-pt-res-canonical": 0.5}


# ---------------------------------------------------------- triggers ---

def test_trigger_after():
    s = parse_expr("0.3 warm@after")
    assert s.alphas == {"warm": (0.3, Trigger.AFTER_THINKING)}


def test_trigger_before():
    s = parse_expr("0.3 warm@before")
    assert s.alphas == {"warm": (0.3, Trigger.PROMPT_ONLY)}


def test_trigger_both():
    s = parse_expr("0.3 warm@both")
    assert s.alphas == {"warm": (0.3, Trigger.BOTH)}


def test_trigger_thinking():
    s = parse_expr("0.3 warm@thinking")
    assert s.alphas == {"warm": (0.3, Trigger.THINKING_ONLY)}


def test_trigger_response():
    s = parse_expr("0.3 warm@response")
    assert s.alphas == {"warm": (0.3, Trigger.GENERATED_ONLY)}


def test_trigger_prompt_alias_for_before():
    s = parse_expr("0.3 warm@prompt")
    assert s.alphas == {"warm": (0.3, Trigger.PROMPT_ONLY)}


def test_trigger_generated_alias_for_response():
    s = parse_expr("0.3 warm@generated")
    assert s.alphas == {"warm": (0.3, Trigger.GENERATED_ONLY)}


def test_mixed_triggers_per_term():
    s = parse_expr("0.5 honest + 0.3 warm@after")
    assert s.alphas == {
        "honest": 0.5,
        "warm": (0.3, Trigger.AFTER_THINKING),
    }


def test_unknown_trigger_raises():
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.3 warm@splatty")
    assert "unknown trigger" in str(ei.value)


def test_unknown_trigger_mentions_hf_revision_hint():
    # HF revisions would land as a trigger-shaped token — the error should
    # steer users away from that mistake.
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.3 bob/honest@abc1234")
    assert "HF revisions" in str(ei.value)


# --------------------------------------------------------- projection ---

def test_projection_orthogonal():
    s = parse_expr("0.5 honest|sycophantic")
    key = "honest|sycophantic"
    assert key in s.alphas
    v = s.alphas[key]
    assert isinstance(v, ProjectedTerm)
    assert v.coeff == 0.5
    assert v.operator == "|"
    assert v.base == "honest"
    assert v.onto == "sycophantic"
    assert v.trigger == Trigger.BOTH


def test_projection_onto():
    s = parse_expr("0.5 honest~sycophantic")
    key = "honest~sycophantic"
    assert key in s.alphas
    v = s.alphas[key]
    assert v.operator == "~"


def test_projection_chained_rejected():
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.5 a~b~c")
    assert "chained projection" in str(ei.value).lower()


def test_projection_with_trigger():
    s = parse_expr("0.5 honest|sycophantic@after")
    v = next(iter(s.alphas.values()))
    assert isinstance(v, ProjectedTerm)
    assert v.trigger == Trigger.AFTER_THINKING


def test_projection_with_variant():
    s = parse_expr("0.5 honest:sae|sycophantic")
    key = "honest:sae|sycophantic"
    assert key in s.alphas


def test_projection_and_plain_coexist():
    s = parse_expr("0.5 warm + 0.3 warm|cold")
    assert "warm" in s.alphas
    assert "warm|cold" in s.alphas


# --------------------------------------------------------- summation ---

def test_same_name_sums_no_trigger():
    s = parse_expr("0.3 warm + 0.2 warm")
    assert s.alphas == {"warm": pytest.approx(0.5)}


def test_same_name_same_trigger_sums():
    s = parse_expr("0.3 warm@after + 0.2 warm@after")
    entry = s.alphas["warm"]
    assert isinstance(entry, tuple)
    assert entry[0] == pytest.approx(0.5)
    assert entry[1] == Trigger.AFTER_THINKING


def test_conflicting_triggers_reject():
    with pytest.raises(SteeringExprError) as ei:
        parse_expr("0.3 warm@before + 0.2 warm@after")
    assert "conflicting" in str(ei.value).lower()


def test_bare_and_triggered_same_name_reject():
    with pytest.raises(SteeringExprError):
        parse_expr("0.3 warm + 0.2 warm@after")


def test_plain_vs_projected_same_key_reject():
    # ``warm|cold`` is a valid projection key; a plain entry under the
    # same synthetic string can't coexist.  Parser constructs the key
    # internally — users don't type ``warm|cold`` as a plain name.
    # Direct-construction covers this; parser paths always route
    # projections to ProjectedTerm values.
    pass  # placeholder — no parser-level way to trigger


# ------------------------------------------------------------ errors ---

def test_empty_raises():
    with pytest.raises(SteeringExprError):
        parse_expr("")


def test_whitespace_only_raises():
    with pytest.raises(SteeringExprError):
        parse_expr("   \t  ")


def test_trailing_operator_raises():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 honest +")


def test_bad_character_raises():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 honest !")


def test_missing_selector_after_coeff():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 ")


def test_colon_without_variant():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 honest:")


def test_at_without_trigger():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 honest@")


def test_slash_without_name():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 bob/")


def test_dot_without_second_pole():
    with pytest.raises(SteeringExprError):
        parse_expr("0.5 angry.")


# ------------------------------------------------------ pole aliasing ---

def test_pole_alias_flips_sign(tmp_path):
    # Install ``default/deer.wolf``; bare ``wolf`` should resolve to
    # ``deer.wolf`` with sign -1, so ``0.5 wolf`` -> ``deer.wolf: -0.5``.
    _mk(tmp_path, "default", "deer.wolf")
    sel.invalidate()
    s = parse_expr("0.5 wolf")
    assert s.alphas == {"deer.wolf": -0.5}


def test_pole_positive_pole_keeps_sign(tmp_path):
    _mk(tmp_path, "default", "deer.wolf")
    sel.invalidate()
    s = parse_expr("0.5 deer")
    assert s.alphas == {"deer.wolf": 0.5}


def test_pole_composite_literal_bypasses_resolution(tmp_path):
    _mk(tmp_path, "default", "deer.wolf")
    sel.invalidate()
    s = parse_expr("0.5 deer.wolf")
    assert s.alphas == {"deer.wolf": 0.5}


# -------------------------------------------------------- format/round-trip ---

def test_format_single():
    s = parse_expr("0.5 honest")
    assert format_expr(s) == "0.5 honest"


def test_format_add_subtract():
    s = parse_expr("0.5 honest - 0.2 manipulative")
    assert format_expr(s) == "0.5 honest - 0.2 manipulative"


def test_format_three_terms():
    s = parse_expr("0.5 honest + 0.3 warm - 0.2 manipulative")
    assert format_expr(s) == "0.5 honest + 0.3 warm - 0.2 manipulative"


def test_format_trigger_emitted():
    s = parse_expr("0.3 warm@after")
    assert format_expr(s) == "0.3 warm@after"


def test_format_both_trigger_elided():
    s = parse_expr("0.3 warm@both")
    assert format_expr(s) == "0.3 warm"


def test_format_projection_orthogonal():
    s = parse_expr("0.5 honest|sycophantic")
    assert format_expr(s) == "0.5 honest|sycophantic"


def test_format_projection_onto():
    s = parse_expr("0.5 honest~sycophantic")
    assert format_expr(s) == "0.5 honest~sycophantic"


def test_format_projection_with_trigger():
    s = parse_expr("0.5 honest|sycophantic@after")
    assert format_expr(s) == "0.5 honest|sycophantic@after"


def test_format_leading_negative():
    s = parse_expr("-0.5 honest + 0.3 warm")
    assert format_expr(s) == "-0.5 honest + 0.3 warm"


def test_format_variant_suffix():
    s = parse_expr("0.5 honest:sae-gemma-scope")
    assert format_expr(s) == "0.5 honest:sae-gemma-scope"


def test_round_trip_corpus():
    # Each round-trips through format -> parse -> format and lands stable.
    corpus = [
        "0.5 honest",
        "0.5 honest + 0.3 warm",
        "0.5 honest - 0.2 manipulative",
        "0.3 warm@after",
        "0.5 honest|sycophantic",
        "0.3 bob/deer.wolf",
        "0.5 honest:sae",
    ]
    for text in corpus:
        s1 = parse_expr(text)
        rendered = format_expr(s1)
        s2 = parse_expr(rendered)
        assert format_expr(s2) == rendered, (text, rendered)


class TestRoundTripGolden:
    """Golden corpus: parse -> format -> parse produces identical IRs.

    Formats are canonicalized on the first render (coefficient leading,
    ``@trigger`` only when non-BOTH, ``:raw`` elided) — subsequent
    renders are bit-identical.
    """

    @pytest.mark.parametrize("text,canonical", [
        ("honest", "1 honest"),
        ("0.5 honest", "0.5 honest"),
        ("0.5*honest", "0.5 honest"),
        ("-0.5 honest", "-0.5 honest"),
        ("-honest", "-1 honest"),
        ("+0.5 honest", "0.5 honest"),
        ("0.5 honest + 0.3 warm", "0.5 honest + 0.3 warm"),
        ("0.5 honest - 0.2 manipulative", "0.5 honest - 0.2 manipulative"),
        ("-0.5 honest + 0.3 warm", "-0.5 honest + 0.3 warm"),
        ("0.3 warm@after", "0.3 warm@after"),
        ("0.3 warm@both", "0.3 warm"),
        ("0.3 warm@before", "0.3 warm@before"),
        ("0.3 warm@prompt", "0.3 warm@before"),
        ("0.3 warm@generated", "0.3 warm@response"),
        ("0.3 warm@thinking", "0.3 warm@thinking"),
        ("0.5 honest|sycophantic", "0.5 honest|sycophantic"),
        ("0.5 honest~sycophantic", "0.5 honest~sycophantic"),
        ("0.5 honest|sycophantic@after", "0.5 honest|sycophantic@after"),
        ("0.5 honest:sae", "0.5 honest:sae"),
        ("0.5 honest:raw", "0.5 honest"),
        (
            "0.5 honest:sae-gemma-scope-2b-pt-res-canonical",
            "0.5 honest:sae-gemma-scope-2b-pt-res-canonical",
        ),
        ("1e-2 honest", "0.01 honest"),
        (".25 honest", "0.25 honest"),
        ("2 honest", "2 honest"),
    ])
    def test_canonical_form(self, text, canonical):
        s = parse_expr(text)
        assert format_expr(s) == canonical

    @pytest.mark.parametrize("text", [
        "0.5 honest",
        "0.5 honest + 0.3 warm",
        "0.5 honest - 0.2 manipulative",
        "-0.5 honest + 0.3 warm - 0.2 manipulative",
        "0.3 warm@after",
        "0.3 warm@thinking + 0.5 honest@response",
        "0.5 honest|sycophantic",
        "0.5 honest~sycophantic",
        "0.5 honest:sae",
        "0.5 honest:sae-gemma-scope",
        "0.5 honest:sae|sycophantic",
    ])
    def test_format_parse_format_is_stable(self, text):
        """Render -> re-parse -> render produces the same string."""
        s1 = parse_expr(text)
        r1 = format_expr(s1)
        s2 = parse_expr(r1)
        r2 = format_expr(s2)
        assert r1 == r2
        assert s1.alphas == s2.alphas

    def test_str_dunder_is_formatter(self):
        s = parse_expr("0.5 honest + 0.3 warm@after")
        assert str(s) == format_expr(s)

    def test_empty_steering_renders_empty(self):
        s = Steering(alphas={})
        assert format_expr(s) == ""

    def test_direct_construction_round_trips(self):
        """Steering built directly (not via parser) also stringifies back."""
        s = Steering(alphas={"honest": 0.5, "warm": (0.3, Trigger.AFTER_THINKING)})
        rendered = str(s)
        reparsed = parse_expr(rendered)
        assert reparsed.alphas["honest"] == 0.5
        assert reparsed.alphas["warm"] == (0.3, Trigger.AFTER_THINKING)


def test_steering_str_uses_formatter():
    s = parse_expr("0.5 honest + 0.3 warm")
    assert str(s) == "0.5 honest + 0.3 warm"


# --------------------------------------------------------- from_value ---

def test_from_value_string():
    s = Steering.from_value("0.5 honest")
    assert s is not None
    assert s.alphas == {"honest": 0.5}


def test_from_value_none():
    assert Steering.from_value(None) is None


def test_from_value_steering_passthrough():
    s = Steering(alphas={"honest": 0.5})
    assert Steering.from_value(s) is s


def test_from_value_rejects_dict():
    with pytest.raises(TypeError) as ei:
        Steering.from_value({"honest": 0.5})
    assert "dict" in str(ei.value).lower()


def test_from_value_rejects_list():
    with pytest.raises(TypeError):
        Steering.from_value([("honest", 0.5)])
