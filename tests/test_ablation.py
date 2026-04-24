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
    """Attach the hook, run a forward pass, detach, return the output."""
    layer = _DummyLayer()
    hook.attach(layer)
    try:
        out = layer(hidden)[0]
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
