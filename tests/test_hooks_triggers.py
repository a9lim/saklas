"""SteeringHook per-trigger grouping + conditional apply.

Exercises the hook's fast path (BOTH-only → single composed tensor, no
per-step trigger check) and the slow path (multiple groups, ctx-gated
adds) without loading a real transformer layer.  A
``nn.Identity``-equivalent module returning the hidden state unchanged
is enough to drive the hook.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from saklas.core.hooks import SteeringHook, SteeringManager, _STEER_GAIN
from saklas.core.triggers import Trigger, TriggerContext


class _Passthrough(nn.Module):
    """Module that returns (hidden,) so the hook can mutate it in place."""

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden,)


def _unit_vec(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def test_fast_path_both_only_composes_single_tensor():
    hook = SteeringHook()
    ctx = TriggerContext()
    vec = _unit_vec(16)
    hook.recompose(
        [(vec, 1.0, Trigger.BOTH)],
        device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
    )
    assert hook.composed is not None
    assert hook.composed_groups == []


def test_slow_path_non_both_uses_groups():
    hook = SteeringHook()
    ctx = TriggerContext()
    vec = _unit_vec(16)
    hook.recompose(
        [(vec, 1.0, Trigger.AFTER_THINKING)],
        device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
    )
    assert hook.composed is None
    assert len(hook.composed_groups) == 1
    assert hook.composed_groups[0][0] == Trigger.AFTER_THINKING


def test_equal_triggers_collapse_into_single_group():
    hook = SteeringHook()
    ctx = TriggerContext()
    v1 = _unit_vec(16, seed=1)
    v2 = _unit_vec(16, seed=2)
    # Two entries with the same trigger VALUE (distinct dataclass
    # instances but equal fields) should share one composed tensor.
    t_a = Trigger.AFTER_THINKING
    t_b = Trigger(prompt=False, thinking=False)  # == AFTER_THINKING
    hook.recompose(
        [(v1, 1.0, t_a), (v2, 1.0, t_b)],
        device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
    )
    assert hook.composed is None
    assert len(hook.composed_groups) == 1


def test_mixed_both_and_non_both_keeps_both_in_slow_path():
    hook = SteeringHook()
    ctx = TriggerContext()
    v_both = _unit_vec(16, seed=1)
    v_after = _unit_vec(16, seed=2)
    hook.recompose(
        [(v_both, 1.0, Trigger.BOTH),
         (v_after, 1.0, Trigger.AFTER_THINKING)],
        device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
    )
    # Mixed groups → slow path; composed stays None even though one group
    # is BOTH (fast-path collapse requires a single BOTH group only).
    assert hook.composed is None
    assert len(hook.composed_groups) == 2


def test_zero_alpha_group_dropped():
    hook = SteeringHook()
    ctx = TriggerContext()
    vec = _unit_vec(16)
    hook.recompose(
        [(vec, 0.0, Trigger.AFTER_THINKING)],
        device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
    )
    assert hook.composed is None
    assert hook.composed_groups == []


def test_fast_path_hook_apply_bit_identical_to_manual_add():
    """Fast-path apply matches manual add+rescale for a BOTH-only hook."""
    mod = _Passthrough()
    hook = SteeringHook()
    ctx = TriggerContext()
    vec = _unit_vec(16) * 0.5
    hook.recompose(
        [(vec, 1.0, Trigger.BOTH)],
        device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
    )
    hook.attach(mod)

    h = torch.randn(1, 4, 16)
    h_copy = h.clone()
    out = mod(h)[0]
    # Manual apply: add vec to every position, rescale to pre norm.
    expected = h_copy.clone()
    norm_pre = expected.norm(dim=-1, keepdim=True, dtype=torch.float32)
    expected.add_(vec)
    norm_post = expected.norm(dim=-1, keepdim=True, dtype=torch.float32).clamp_(min=1e-6)
    expected.mul_((norm_pre / norm_post).to(expected.dtype))
    assert torch.allclose(out, expected, atol=1e-6)
    hook.detach()


def test_slow_path_skips_when_no_group_active():
    """AFTER_THINKING during prefill should no-op — hidden unchanged."""
    mod = _Passthrough()
    hook = SteeringHook()
    ctx = TriggerContext(is_prefill=True)  # prefill → AFTER_THINKING off
    vec = _unit_vec(16)
    hook.recompose(
        [(vec, 1.0, Trigger.AFTER_THINKING)],
        device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
    )
    hook.attach(mod)

    h = torch.randn(1, 4, 16)
    h_before = h.clone()
    mod(h)
    assert torch.equal(h, h_before), "hidden state must be unchanged"
    hook.detach()


def test_slow_path_applies_when_group_active():
    """AFTER_THINKING during decode-response should apply."""
    mod = _Passthrough()
    hook = SteeringHook()
    ctx = TriggerContext(is_prefill=False, thinking=False, gen_step=5)
    vec = _unit_vec(16) * 0.5
    hook.recompose(
        [(vec, 1.0, Trigger.AFTER_THINKING)],
        device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
    )
    hook.attach(mod)
    h = torch.randn(1, 4, 16)
    h_before = h.clone()
    mod(h)
    assert not torch.equal(h, h_before), "hook should have modified hidden"
    hook.detach()


def test_ctx_mutation_between_forwards_gates_apply():
    """Flipping ctx.is_prefill between forwards toggles the AFTER_THINKING group."""
    mod = _Passthrough()
    hook = SteeringHook()
    ctx = TriggerContext(is_prefill=True)
    vec = _unit_vec(16) * 0.5
    hook.recompose(
        [(vec, 1.0, Trigger.AFTER_THINKING)],
        device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
    )
    hook.attach(mod)

    # Forward 1: prefill → no change.
    h1 = torch.randn(1, 4, 16)
    h1_before = h1.clone()
    mod(h1)
    assert torch.equal(h1, h1_before)

    # Mutate ctx → same hook, now fires.
    ctx.is_prefill = False
    h2 = torch.randn(1, 4, 16)
    h2_before = h2.clone()
    mod(h2)
    assert not torch.equal(h2, h2_before)
    hook.detach()


def test_manager_threads_ctx_into_hooks():
    """SteeringManager.apply_to_model plumbs its shared ctx into every hook."""
    mgr = SteeringManager()
    layers = nn.ModuleList([_Passthrough() for _ in range(3)])
    profile = {1: _unit_vec(16), 2: _unit_vec(16, seed=1)}
    mgr.add_vector("demo", profile, alpha=0.5, trigger=Trigger.AFTER_THINKING)
    mgr.apply_to_model(layers, device=torch.device("cpu"), dtype=torch.float32)
    for idx in (1, 2):
        assert idx in mgr.hooks
        assert mgr.hooks[idx]._ctx is mgr.ctx
    # Unused layer has no hook.
    assert 0 not in mgr.hooks
    mgr.clear_all()


def test_manager_alpha_scaled_by_steer_gain():
    """Manager multiplies user alpha by _STEER_GAIN before composing."""
    mgr = SteeringManager()
    layers = nn.ModuleList([_Passthrough() for _ in range(2)])
    vec = _unit_vec(16)
    mgr.add_vector("demo", {0: vec.clone()}, alpha=0.5, trigger=Trigger.BOTH)
    mgr.apply_to_model(layers, device=torch.device("cpu"), dtype=torch.float32)
    # Fast-path: composed == 0.5 * _STEER_GAIN * vec.
    expected = (0.5 * _STEER_GAIN) * vec
    assert torch.allclose(mgr.hooks[0].composed, expected, atol=1e-6)
    mgr.clear_all()
