"""Tests for the recipe-override regen mechanism (v2.3 phase 5)."""
from __future__ import annotations

import pytest

from saklas import Recipe, SamplingConfig


# ---------------------------------------------------------------------------
# Recipe.overlay
# ---------------------------------------------------------------------------


def test_overlay_none_fields_fall_through():
    base = Recipe(steering="0.3 honest.deceptive", seed=42, thinking=False)
    override = Recipe(seed=99)
    out = base.overlay(override)
    assert out.steering == "0.3 honest.deceptive"
    assert out.seed == 99
    assert out.thinking is False


def test_overlay_none_override_returns_self():
    base = Recipe(steering="0.3 honest", seed=42)
    assert base.overlay(None) is base


def test_overlay_preserves_probes_and_hashes():
    base = Recipe(
        steering="0.3 honest", probes=["angry.calm"],
        probe_hashes={"angry.calm": "abc"},
    )
    override = Recipe(steering="0.5 warm")
    out = base.overlay(override)
    # probes / probe_hashes are not overrideable.
    assert out.probes == ["angry.calm"]
    assert out.probe_hashes == {"angry.calm": "abc"}


def test_overlay_steering_replaces_when_set():
    base = Recipe(steering="0.3 honest.deceptive")
    override = Recipe(steering="")  # explicit empty = unsteered
    out = base.overlay(override)
    assert out.steering == ""


# ---------------------------------------------------------------------------
# Recipe.invert_steering
# ---------------------------------------------------------------------------


def test_invert_simple_term():
    r = Recipe(steering="0.3 honest.deceptive")
    inv = r.invert_steering()
    assert "-0.3" in inv.steering


def test_invert_compound_expression():
    r = Recipe(steering="0.3 honest.deceptive + 0.5 warm.clinical@after")
    inv = r.invert_steering()
    # Both signs flipped — first term renders ``-0.3`` directly; the
    # second renders as a ``- 0.5`` separator-and-magnitude pair through
    # ``format_expr``.  Either way both coefficients are negated.
    assert "-0.3" in inv.steering
    assert "- 0.5" in inv.steering or "-0.5" in inv.steering
    # Trigger preserved.
    assert "after" in inv.steering


def test_invert_empty_steering():
    r = Recipe(steering=None)
    inv = r.invert_steering()
    assert inv.steering == ""


# ---------------------------------------------------------------------------
# Recipe.compose_modifier
# ---------------------------------------------------------------------------


def test_compose_unsteered():
    r = Recipe(steering="0.3 honest", seed=42)
    mod = r.compose_modifier("unsteered")
    assert mod.steering == ""


def test_compose_inverted_flips_signs():
    r = Recipe(steering="0.4 warm.clinical")
    mod = r.compose_modifier("inverted")
    assert "-0.4" in mod.steering


def test_compose_reseed_gives_fresh_seed():
    r = Recipe(seed=42)
    mod = r.compose_modifier("reseed")
    assert mod.seed is not None
    assert mod.seed != 42  # nonzero entropy chance of collision, but vanishingly small


def test_compose_cool():
    r = Recipe()
    mod = r.compose_modifier("cool")
    assert mod.sampling.temperature == pytest.approx(0.3)


def test_compose_hot():
    r = Recipe()
    mod = r.compose_modifier("hot")
    assert mod.sampling.temperature == pytest.approx(1.2)


def test_compose_unknown_raises():
    with pytest.raises(ValueError, match="unknown recipe-override mode"):
        Recipe().compose_modifier("foo")


def test_compose_custom_recipe_passthrough():
    """``compose_modifier`` with a Recipe arg returns it unchanged.

    The custom-mode path: callers (TUI's ``/auto-regen custom: <expr>``)
    parse the partial-recipe expression themselves and hand the Recipe
    in; ``compose_modifier`` shouldn't try to interpret it as a string.
    """
    partial = Recipe(steering="0.5 calm", sampling=SamplingConfig(temperature=0.4))
    out = Recipe(steering="0.3 honest", seed=42).compose_modifier(partial)
    assert out is partial  # passes through, no copy


def test_compose_custom_via_overlay():
    """Custom mode = pass a Recipe partial directly, no compose_modifier."""
    base = Recipe(steering="0.3 honest", seed=42)
    partial = Recipe(sampling=SamplingConfig(temperature=0.5), seed=99)
    out = base.overlay(partial)
    assert out.steering == "0.3 honest"
    assert out.sampling.temperature == pytest.approx(0.5)
    assert out.seed == 99


def test_compose_modifier_routes_through_overlay_in_regen_with_modifier():
    """End-to-end: regen_with_modifier accepts a Recipe partial as ``mode``.

    Verifies the full custom-mode wiring: TUI parses ``custom: <expr>``
    into a Recipe, hands it to ``session.regen_with_modifier(... mode=<Recipe>)``,
    which routes through ``compose_modifier(Recipe) -> Recipe`` and overlays
    onto the parent recipe.  No model load — uses _resolve_recipe_override
    which is the pure-Python overlay path.
    """
    from saklas.core.session import SaklasSession
    from saklas import LoomTree

    class _StubSession:
        def __init__(self):
            self.tree = LoomTree()
            uid = self.tree.add_user_turn("hi")
            recipe = Recipe(steering="0.3 honest", seed=42)
            aid = self.tree.begin_assistant(uid, recipe=recipe)
            self.tree.finalize_assistant(aid, text="hello")
            self.aid = aid

    stub = _StubSession()
    resolve = SaklasSession._resolve_recipe_override.__get__(stub, _StubSession)
    custom = Recipe(sampling=SamplingConfig(temperature=0.4))
    new_steering, new_sampling, new_thinking = resolve(
        custom,
        parent_node_id=stub.aid,
        steering=None,
        sampling=None,
        thinking=None,
    )
    assert new_steering == "0.3 honest"
    assert new_sampling is not None
    assert new_sampling.temperature == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# session._resolve_recipe_override engine integration (without model load)
# ---------------------------------------------------------------------------


def test_resolve_override_with_string_mode():
    """The session helper applies overlay onto the parent recipe + returns the kwargs."""
    from saklas.core.session import SaklasSession
    from saklas import LoomTree

    class _StubSession:
        def __init__(self):
            self.tree = LoomTree()
            uid = self.tree.add_user_turn("hi")
            recipe = Recipe(steering="0.3 honest", seed=42)
            aid = self.tree.begin_assistant(uid, recipe=recipe)
            self.tree.finalize_assistant(aid, text="hello")
            self.aid = aid

    stub = _StubSession()
    resolve = SaklasSession._resolve_recipe_override.__get__(stub, _StubSession)
    new_steering, new_sampling, new_thinking = resolve(
        "unsteered",
        parent_node_id=stub.aid,
        steering=None,
        sampling=None,
        thinking=None,
    )
    # Unsteered modifier wipes steering.
    assert new_steering == ""


def test_resolve_override_passthrough_when_none():
    from saklas.core.session import SaklasSession
    from saklas import LoomTree

    class _StubSession:
        def __init__(self):
            self.tree = LoomTree()

    stub = _StubSession()
    resolve = SaklasSession._resolve_recipe_override.__get__(stub, _StubSession)
    out = resolve(
        None, parent_node_id=None,
        steering="0.5 warm", sampling=None, thinking=None,
    )
    assert out == ("0.5 warm", None, None)


def test_resolve_override_with_custom_recipe():
    from saklas.core.session import SaklasSession
    from saklas import LoomTree

    class _StubSession:
        def __init__(self):
            self.tree = LoomTree()

    stub = _StubSession()
    resolve = SaklasSession._resolve_recipe_override.__get__(stub, _StubSession)
    partial = Recipe(sampling=SamplingConfig(temperature=0.9), seed=7)
    new_steering, new_sampling, new_thinking = resolve(
        partial, parent_node_id=None,
        steering="0.3 honest", sampling=None, thinking=None,
    )
    # Steering inherits the caller's explicit kwarg; sampling carries
    # the override's seed and temperature.
    assert new_steering == "0.3 honest"
    assert new_sampling.temperature == pytest.approx(0.9)
    assert new_sampling.seed == 7
