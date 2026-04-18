"""Steering dataclass unit tests — from_value coercion + trigger entries."""

from saklas.core.steering import Steering
from saklas.core.triggers import Trigger


def test_construct():
    s = Steering(alphas={"foo": 0.5})
    assert s.alphas == {"foo": 0.5}
    assert s.thinking is None
    assert s.trigger is Trigger.BOTH


def test_thinking_field():
    s = Steering(alphas={"foo": 0.1}, thinking=True)
    assert s.thinking is True


def test_from_value_string_parses_expr():
    s = Steering.from_value("0.5 foo")
    assert isinstance(s, Steering)
    assert s.alphas == {"foo": 0.5}


def test_from_value_none_returns_none():
    assert Steering.from_value(None) is None


def test_from_value_passthrough():
    s = Steering(alphas={"x": 1.0})
    assert Steering.from_value(s) is s


def test_from_value_rejects_dict_input():
    import pytest
    with pytest.raises(TypeError) as ei:
        Steering.from_value({"foo": 0.5})
    assert "str | Steering | None" in str(ei.value)


def test_default_trigger_applies_to_bare_floats():
    s = Steering(alphas={"foo": 0.5}, trigger=Trigger.AFTER_THINKING)
    entries = s.normalized_entries()
    assert entries == {"foo": (0.5, Trigger.AFTER_THINKING)}


def test_tuple_entry_overrides_default_trigger():
    s = Steering(
        alphas={"foo": (0.5, Trigger.THINKING_ONLY)},
        trigger=Trigger.AFTER_THINKING,
    )
    entries = s.normalized_entries()
    assert entries == {"foo": (0.5, Trigger.THINKING_ONLY)}


def test_mixed_entries_normalize_correctly():
    s = Steering(
        alphas={
            "bare":  0.3,
            "tuple": (0.4, Trigger.AFTER_THINKING),
        },
    )
    entries = s.normalized_entries()
    assert entries["bare"] == (0.3, Trigger.BOTH)
    assert entries["tuple"] == (0.4, Trigger.AFTER_THINKING)


def test_expression_preserves_trigger_entries():
    # The expression grammar is the new input shape; ``@after`` etc. map
    # onto the same ``(alpha, Trigger)`` tuple form that direct-construction
    # accepts.
    s = Steering.from_value("0.5 foo + 0.3 bar@response")
    entries = s.normalized_entries()
    assert entries["foo"] == (0.5, Trigger.BOTH)
    assert entries["bar"] == (0.3, Trigger.GENERATED_ONLY)


def test_normalized_entries_coerces_int_alpha_to_float():
    s = Steering(alphas={"foo": 1})
    entries = s.normalized_entries()
    assert entries == {"foo": (1.0, Trigger.BOTH)}
    assert isinstance(entries["foo"][0], float)
