"""SteeringHook per-trigger conditional apply on the 4.0 subspace backend.

The hook no longer has an additive fast path — every steering term lowers to a
``subspace_inject`` group (the merged affine subspace + curved manifolds).  This
exercises the surviving behavior: a group fires only on the decode steps where
its :class:`Trigger` is active, and ``ctx`` mutation between forwards gates the
apply.  A ``nn.Identity``-equivalent module is enough to drive the hook.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from saklas.core.hooks import SteeringHook, SteeringManager
from saklas.core.manifold import CustomDomain, LayerSubspace, synthesize_subspace
from saklas.core.triggers import Trigger, TriggerContext


_DIM = 16


def _unit(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


_AffineEntry = tuple[
    LayerSubspace,
    CustomDomain,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    Trigger,
]


def _affine_group(layer: int, trigger: Trigger) -> _AffineEntry:
    """One ``(sub, domain, target, origin, along, onto, trigger)`` entry.

    A rank-1 affine subspace from a single push term — the merged-subspace shape
    ``apply_to_model`` hands ``recompose``.
    """
    u = _unit(_DIM)
    neutral = 20.0 + torch.randn(_DIM, generator=torch.Generator().manual_seed(1))
    synth = synthesize_subspace(
        push=[({layer: u.reshape(1, -1)}, {layer: torch.tensor([1.0])}, 0.8)],
        ablate=[], neutral_means={layer: neutral},
    )
    sub = synth.layers[layer]
    return (
        sub, CustomDomain(sub.rank), synth.target_coord[layer],
        torch.zeros(sub.rank), 1.0, 0.0, trigger,
    )


def _recompose(hook: SteeringHook, entry: _AffineEntry, ctx: TriggerContext) -> None:
    hook.recompose([entry], ctx, device=torch.device("cpu"))


# ----------------------------------------------------------------- grouping ---

def test_recompose_stamps_group_and_cold_feet():
    hook = SteeringHook()
    ctx = TriggerContext()
    _recompose(hook, _affine_group(0, Trigger.AFTER_THINKING), ctx)
    assert len(hook.manifold_groups) == 1
    assert hook.manifold_groups[0][0] == Trigger.AFTER_THINKING
    assert hook._manifold_feet == [None]


def test_zero_coeff_group_dropped():
    hook = SteeringHook()
    ctx = TriggerContext()
    sub, domain, target, origin, _along, _onto, trig = _affine_group(
        0, Trigger.BOTH,
    )
    hook.recompose(
        [(sub, domain, target, origin, 0.0, 0.0, trig)],
        ctx, device=torch.device("cpu"),
    )
    assert hook.manifold_groups == []


# ------------------------------------------------------- conditional apply ---

def test_both_trigger_always_applies():
    hook = SteeringHook()
    ctx = TriggerContext()
    _recompose(hook, _affine_group(0, Trigger.BOTH), ctx)
    hook._ctx = ctx
    hidden = (20.0 + torch.randn(1, 3, _DIM)).to(torch.float32)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert not torch.allclose(hidden, before, atol=1e-3)


def test_non_both_skips_when_inactive():
    hook = SteeringHook()
    ctx = TriggerContext()
    ctx.thinking = True  # inside the thinking section ⇒ AFTER_THINKING inactive
    _recompose(hook, _affine_group(0, Trigger.AFTER_THINKING), ctx)
    hook._ctx = ctx
    hidden = (20.0 + torch.randn(1, 3, _DIM)).to(torch.float32)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert torch.allclose(hidden, before, atol=1e-6)


def test_ctx_mutation_between_forwards_gates_apply():
    hook = SteeringHook()
    ctx = TriggerContext()
    ctx.thinking = True  # start inside thinking → inactive
    _recompose(hook, _affine_group(0, Trigger.AFTER_THINKING), ctx)
    hook._ctx = ctx

    hidden = (20.0 + torch.randn(1, 1, _DIM)).to(torch.float32)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)  # inactive → no-op
    assert torch.allclose(hidden, before, atol=1e-6)

    # Leave the thinking section; the same hook now fires on response tokens.
    ctx.thinking = False
    hidden2 = (20.0 + torch.randn(1, 1, _DIM)).to(torch.float32)
    before2 = hidden2.clone()
    hook.hook_fn(None, None, hidden2)
    assert not torch.allclose(hidden2, before2, atol=1e-3)


def test_manager_threads_ctx_into_hooks():
    mgr = SteeringManager()
    synth = synthesize_subspace(
        push=[({0: _unit(_DIM).reshape(1, -1)}, {0: torch.tensor([1.0])}, 0.5)],
        ablate=[], neutral_means={0: 20.0 + torch.randn(_DIM)},
    )
    mgr.add_subspace("__affine__", synth)
    layers = nn.ModuleList([nn.Identity() for _ in range(2)])
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    # Every attached hook shares the manager's mutable context.
    for hook in mgr.hooks.values():
        assert hook._ctx is mgr.ctx
