"""CPU-only toy-model tests for mean-replacement ablation.

Each test uses a dummy layer whose ``forward`` returns the input hidden
state inside a 1-tuple (matching transformer decoder-layer signatures).
A ``SteeringHook`` is attached; the test constructs baked directions and
layer-mean scalars by hand, runs the forward pass, and asserts the
post-hook hidden state matches the mean-replacement formula (with the
unconditional norm-preservation rescale folded in).
"""
from __future__ import annotations

import pytest
import torch

from saklas.core.hooks import SteeringHook
from saklas.core.triggers import Trigger, TriggerContext


class _DummyLayer(torch.nn.Module):
    def forward(self, x):  # type: ignore[override]
        return (x,)


def _run_hook(
    hook: SteeringHook,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Attach the hook, run a forward pass, detach, return the output.

    The hook mutates its input in place (hot-path discipline), so we
    clone before handing off; tests retain the pristine ``hidden`` for
    reference-math assertions.
    """
    layer = _DummyLayer()
    hook.attach(layer)
    try:
        out = layer(hidden.clone())[0]
    finally:
        hook.detach()
    return out


def _expected_ablation(
    hidden: torch.Tensor,
    baked: torch.Tensor,
    layer_mean: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Reference math: ``h - α(h·d̂ - μ·d̂) d̂`` + norm rescale."""
    d_unit = baked / baked.norm()
    mean_scalar = (layer_mean * d_unit).sum()
    norm_pre = torch.linalg.vector_norm(hidden, dim=-1, keepdim=True)
    proj = (hidden * d_unit).sum(dim=-1, keepdim=True)
    delta = alpha * (proj - mean_scalar) * d_unit
    out = hidden - delta
    norm_post = torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp(min=1e-6)
    return out * (norm_pre / norm_post)


def test_recompose_accepts_ablation_entries():
    """Smoke: recompose with an ablation entry populates ablation_groups."""
    hook = SteeringHook()
    ctx = TriggerContext()
    d = torch.tensor([1.0, 0.0, 0.0])
    mean = torch.tensor([0.5, 0.0, 0.0])
    hook.recompose(
        additive_entries=[],
        ablation_entries=[(d, mean, 1.0, Trigger.BOTH)],
        device=torch.device("cpu"),
        dtype=torch.float32,
        ctx=ctx,
    )
    assert hook.ablation_groups  # non-empty
    assert hook.composed is None
    assert not hook.composed_groups


def test_single_direction_full_ablation():
    """α=1 mean-replaces the component along d̂; rest of hidden unchanged modulo norm rescale."""
    hook = SteeringHook()
    ctx = TriggerContext()
    d = torch.tensor([1.0, 0.0, 0.0])
    mean = torch.tensor([0.5, 0.0, 0.0])
    hook.recompose(
        additive_entries=[],
        ablation_entries=[(d, mean, 1.0, Trigger.BOTH)],
        device=torch.device("cpu"),
        dtype=torch.float32,
        ctx=ctx,
    )
    hidden = torch.tensor([[[2.0, 1.0, 0.0]]])
    out = _run_hook(hook, hidden)
    expected = _expected_ablation(
        hidden, baked=d, layer_mean=mean, alpha=1.0,
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_partial_ablation_half():
    hook = SteeringHook()
    ctx = TriggerContext()
    d = torch.tensor([1.0, 0.0, 0.0])
    mean = torch.tensor([0.5, 0.0, 0.0])
    hook.recompose(
        additive_entries=[],
        ablation_entries=[(d, mean, 0.5, Trigger.BOTH)],
        device=torch.device("cpu"),
        dtype=torch.float32,
        ctx=ctx,
    )
    hidden = torch.tensor([[[2.0, 1.0, 0.0]]])
    out = _run_hook(hook, hidden)
    expected = _expected_ablation(
        hidden, baked=d, layer_mean=mean, alpha=0.5,
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_zero_alpha_ablation_is_noop():
    """α=0 entry is filtered out of ablation_groups entirely; hidden is untouched."""
    hook = SteeringHook()
    ctx = TriggerContext()
    d = torch.tensor([1.0, 0.0, 0.0])
    mean = torch.tensor([0.5, 0.0, 0.0])
    hook.recompose(
        additive_entries=[],
        ablation_entries=[(d, mean, 0.0, Trigger.BOTH)],
        device=torch.device("cpu"),
        dtype=torch.float32,
        ctx=ctx,
    )
    assert not hook.ablation_groups
    hidden = torch.tensor([[[2.0, 1.0, 0.0]]])
    out = _run_hook(hook, hidden)
    torch.testing.assert_close(out, hidden)


def test_ablate_then_add_ordering():
    """Ablation runs before additive; additive injection lands in the cleaned stream."""
    hook = SteeringHook()
    ctx = TriggerContext()
    d_abl = torch.tensor([1.0, 0.0, 0.0])
    mean_abl = torch.tensor([0.0, 0.0, 0.0])  # zero mean -> zero-ablate the x-component
    add_vec = torch.tensor([0.0, 1.0, 0.0])
    hook.recompose(
        additive_entries=[(add_vec, 1.0, Trigger.BOTH)],
        ablation_entries=[(d_abl, mean_abl, 1.0, Trigger.BOTH)],
        device=torch.device("cpu"),
        dtype=torch.float32,
        ctx=ctx,
    )
    hidden = torch.tensor([[[3.0, 0.0, 0.0]]])
    out = _run_hook(hook, hidden)

    # Ablate first: h -> [0, 0, 0]
    # Add next:    h -> [0, 1, 0]
    # Rescale to ||[3, 0, 0]|| = 3 -> [0, 3, 0]
    expected = torch.tensor([[[0.0, 3.0, 0.0]]])
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_additive_only_hits_fast_path():
    """Additive-only BOTH entries collapse into hook.composed (fast path)."""
    hook = SteeringHook()
    ctx = TriggerContext()
    add_vec = torch.tensor([0.0, 1.0, 0.0])
    hook.recompose(
        additive_entries=[(add_vec, 0.5, Trigger.BOTH)],
        ablation_entries=[],
        device=torch.device("cpu"),
        dtype=torch.float32,
        ctx=ctx,
    )
    assert hook.composed is not None
    assert not hook.composed_groups
    assert not hook.ablation_groups


def test_additive_both_plus_ablation_uses_slow_path():
    """Any ablation present forces the slow path even if additive is BOTH-only."""
    hook = SteeringHook()
    ctx = TriggerContext()
    add_vec = torch.tensor([0.0, 1.0, 0.0])
    d_abl = torch.tensor([1.0, 0.0, 0.0])
    mean_abl = torch.tensor([0.0, 0.0, 0.0])
    hook.recompose(
        additive_entries=[(add_vec, 0.5, Trigger.BOTH)],
        ablation_entries=[(d_abl, mean_abl, 1.0, Trigger.BOTH)],
        device=torch.device("cpu"),
        dtype=torch.float32,
        ctx=ctx,
    )
    assert hook.composed is None
    assert hook.composed_groups
    assert hook.ablation_groups
