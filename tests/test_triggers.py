"""Trigger.active() semantics + preset coverage.

Pure CPU dataclass tests — no model, no hooks. Covers every preset against
every lifecycle ctx state (prefill / decode-thinking / decode-response)
plus the first_n / after_n window modifiers.
"""

from __future__ import annotations

from saklas.core.triggers import Trigger, TriggerContext


def _ctx(*, prefill=False, thinking=False, gen_step=0) -> TriggerContext:
    return TriggerContext(is_prefill=prefill, thinking=thinking, gen_step=gen_step)


def test_default_trigger_fires_everywhere():
    t = Trigger.BOTH
    assert t.active(_ctx(prefill=True))
    assert t.active(_ctx(thinking=True))
    assert t.active(_ctx(thinking=False))
    assert t.active(_ctx(gen_step=99))


def test_generated_only_skips_prefill():
    t = Trigger.GENERATED_ONLY
    assert not t.active(_ctx(prefill=True))
    assert t.active(_ctx(thinking=True))
    assert t.active(_ctx(thinking=False))


def test_prompt_only_skips_decode():
    t = Trigger.PROMPT_ONLY
    assert t.active(_ctx(prefill=True))
    assert not t.active(_ctx(thinking=True))
    assert not t.active(_ctx(thinking=False))


def test_after_thinking_fires_only_during_response():
    t = Trigger.AFTER_THINKING
    assert not t.active(_ctx(prefill=True))
    assert not t.active(_ctx(thinking=True))
    assert t.active(_ctx(thinking=False))


def test_thinking_only_fires_only_during_thinking():
    t = Trigger.THINKING_ONLY
    assert not t.active(_ctx(prefill=True))
    assert t.active(_ctx(thinking=True))
    assert not t.active(_ctx(thinking=False))


def test_first_n_window():
    t = Trigger.first(5)
    for step in range(5):
        assert t.active(_ctx(gen_step=step)), f"step={step}"
    assert not t.active(_ctx(gen_step=5))
    assert not t.active(_ctx(gen_step=100))
    # first() also turns off prefill (wait-for-generation semantics)
    assert not t.active(_ctx(prefill=True))


def test_after_n_window():
    t = Trigger.after(3)
    for step in range(3):
        assert not t.active(_ctx(gen_step=step)), f"step={step}"
    assert t.active(_ctx(gen_step=3))
    assert t.active(_ctx(gen_step=100))
    assert not t.active(_ctx(prefill=True))


def test_custom_trigger_compose():
    # Only during thinking, only after step 2.
    t = Trigger(prompt=False, thinking=True, response=False, after_n=2)
    assert not t.active(_ctx(thinking=True, gen_step=0))
    assert not t.active(_ctx(thinking=True, gen_step=1))
    assert t.active(_ctx(thinking=True, gen_step=2))
    assert not t.active(_ctx(thinking=False, gen_step=2))


def test_trigger_is_hashable_and_frozen():
    """Frozen dataclass means equal triggers share identity and can key dicts."""
    a = Trigger.BOTH
    b = Trigger()
    assert a == b
    assert hash(a) == hash(b)
    # Verify frozenness.
    import pytest
    with pytest.raises(Exception):
        a.prompt = False  # type: ignore[misc]


def test_presets_are_trigger_instances():
    for t in (Trigger.BOTH, Trigger.GENERATED_ONLY, Trigger.PROMPT_ONLY,
              Trigger.AFTER_THINKING, Trigger.THINKING_ONLY):
        assert isinstance(t, Trigger)


def test_context_reset():
    ctx = TriggerContext(is_prefill=True, thinking=True, gen_step=42)
    ctx.reset()
    assert ctx.is_prefill is False
    assert ctx.thinking is False
    assert ctx.gen_step == 0
