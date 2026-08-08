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
        Steering.from_value({"foo": 0.5})  # pyright: ignore[reportArgumentType]  # intentional wrong-type test
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
    assert s is not None  # non-None string input always yields a Steering
    entries = s.normalized_entries()
    assert entries["foo"] == (0.5, Trigger.BOTH)
    assert entries["bar"] == (0.3, Trigger.GENERATED_ONLY)


def test_normalized_entries_coerces_int_alpha_to_float():
    s = Steering(alphas={"foo": 1})
    entries = s.normalized_entries()
    assert entries == {"foo": (1.0, Trigger.BOTH)}
    assert isinstance(entries["foo"][0], float)


# ---------------------------------------------------------------------------
# classified(): one walk, three consumer shapes.
#
# normalized_entries silently drops AblationTerm and ManifoldTerm while
# calling itself "the canonical form", which forced its caller to re-walk
# ``alphas`` twice more to recover them.  classified() is the single pass.
# ---------------------------------------------------------------------------


def _mixed_steering() -> Steering:
    from saklas.core.steering_expr import AblationTerm, ManifoldTerm, ProjectedTerm

    return Steering(alphas={
        "bare": 0.3,
        "tuple": (0.4, Trigger.AFTER_THINKING),
        "a~b": ProjectedTerm(
            coeff=0.5, trigger=Trigger.GENERATED_ONLY,
            operator="~", base="a", onto="b",
        ),
        "!gone": AblationTerm(
            coeff=1.0, trigger=Trigger.BOTH, target="gone",
        ),
        "personas%hacker": ManifoldTerm(
            along=0.6, onto=0.2, trigger=Trigger.BOTH,
            manifold="personas", position="hacker",
        ),
    })


def test_classified_splits_every_entry_kind():
    from saklas.core.steering_expr import AblationTerm, ManifoldTerm

    entries = _mixed_steering().classified()

    assert entries.additive == {
        "bare": (0.3, Trigger.BOTH),
        "tuple": (0.4, Trigger.AFTER_THINKING),
        "a~b": (0.5, Trigger.GENERATED_ONLY),
    }
    assert set(entries.ablations) == {"!gone"}
    assert isinstance(entries.ablations["!gone"], AblationTerm)
    assert entries.ablations["!gone"].target == "gone"
    assert set(entries.manifolds) == {"personas%hacker"}
    manifold_term = entries.manifolds["personas%hacker"]
    assert isinstance(manifold_term, ManifoldTerm)
    # The along/onto split survives — it is exactly what the additive
    # flattening cannot carry.
    assert manifold_term.along == 0.6
    assert manifold_term.onto == 0.2


def test_classified_groups_are_key_disjoint_and_total():
    steering = _mixed_steering()
    entries = steering.classified()
    keys = (
        set(entries.additive) | set(entries.ablations) | set(entries.manifolds)
    )
    assert keys == set(steering.alphas)
    assert len(entries.additive) + len(entries.ablations) + len(entries.manifolds) == len(
        steering.alphas
    )


def test_normalized_entries_is_the_additive_view():
    steering = _mixed_steering()
    assert steering.normalized_entries() == steering.classified().additive


# ---------------------------------------------------------------------------
# triggers() + the entry readers: the shape-blind views over an entry.
# ---------------------------------------------------------------------------


def test_triggers_covers_every_entry_kind():
    assert _mixed_steering().triggers() == {
        Trigger.BOTH, Trigger.AFTER_THINKING, Trigger.GENERATED_ONLY,
    }


def test_triggers_gives_bare_floats_the_steering_default():
    steering = Steering(
        alphas={"bare": 0.3}, trigger=Trigger.GENERATED_ONLY,
    )
    assert steering.triggers() == {Trigger.GENERATED_ONLY}


def test_entry_readers_are_shape_blind():
    from saklas.core.steering import entry_coeff, entry_trigger
    from saklas.core.steering_expr import AblationTerm, ManifoldTerm

    tuple_entry = (0.4, Trigger.AFTER_THINKING)
    ablation = AblationTerm(coeff=0.25, trigger=Trigger.BOTH, target="gone")
    manifold = ManifoldTerm(
        along=0.6, onto=0.2, trigger=Trigger.GENERATED_ONLY,
        manifold="personas", position="hacker",
    )

    assert entry_trigger(tuple_entry) is Trigger.AFTER_THINKING
    assert entry_trigger(ablation) is Trigger.BOTH
    assert entry_trigger(manifold) is Trigger.GENERATED_ONLY

    assert entry_coeff(tuple_entry) == 0.4
    assert entry_coeff(ablation) == 0.25
    # A manifold term's scalar strength is its ``along`` half — what the flat
    # ``{name: alpha}`` views report.
    assert entry_coeff(manifold) == 0.6
