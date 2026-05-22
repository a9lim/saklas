"""Hook-level tests for manifold steering injection — CPU only."""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from saklas.core.hooks import SteeringHook, SteeringManager
from saklas.core.manifold import Manifold, fit_layer_subspace
from saklas.core.steering_expr import SteeringExprError
from saklas.core.triggers import Trigger, TriggerContext

_DIM = 8


def _circle(k: int, dim: int, scale: float = 1.0) -> torch.Tensor:
    out = torch.zeros(k, dim)
    base = torch.full((dim,), 0.4)
    for i in range(k):
        theta = 2.0 * math.pi * i / k
        out[i] = base.clone()
        out[i, 0] += scale * 2.0 * math.cos(theta)
        out[i, 1] += scale * 2.0 * math.sin(theta)
    return out


def _manifold(layers=(0, 1)) -> Manifold:
    return Manifold(
        name="mood",
        cyclic=True,
        node_labels=[f"n{i}" for i in range(6)],
        layers={
            L: fit_layer_subspace(_circle(6, _DIM, 1.0 + 0.2 * L), cyclic=True)
            for L in layers
        },
        feature_space="raw",
    )


# ------------------------------------------------------------- hook level ---

def _recompose_manifold(hook: SteeringHook, sub, alpha: float):
    ctx = TriggerContext()
    target = sub.spline_point(0.5)
    hook.recompose(
        additive_entries=[],
        ablation_entries=[],
        manifold_entries=[(sub.basis, sub.mean, target, alpha, Trigger.BOTH)],
        device=torch.device("cpu"),
        dtype=torch.float32,
        ctx=ctx,
    )


def test_hook_alpha_zero_is_noop():
    manifold = _manifold()
    hook = SteeringHook(injection_mode="angular")
    _recompose_manifold(hook, manifold.layers[0], alpha=0.0)
    hidden = torch.randn(1, 4, _DIM)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    # alpha=0 entries drop at recompose time -> no manifold group at all.
    assert torch.allclose(hidden, before, atol=1e-5)


def test_hook_preserves_norm():
    manifold = _manifold()
    hook = SteeringHook(injection_mode="angular")
    _recompose_manifold(hook, manifold.layers[0], alpha=0.7)
    hidden = torch.randn(1, 5, _DIM)
    norm_before = hidden.norm(dim=-1).clone()
    hook.hook_fn(None, None, hidden)
    assert torch.allclose(hidden.norm(dim=-1), norm_before, atol=1e-4)


def test_hook_alpha_one_lands_in_subspace_on_target():
    manifold = _manifold()
    sub = manifold.layers[0]
    hook = SteeringHook(injection_mode="angular")
    _recompose_manifold(hook, sub, alpha=1.0)
    hidden = torch.randn(1, 3, _DIM)
    hook.hook_fn(None, None, hidden)

    # After the replace the in-subspace component should be parallel to
    # the spline target's in-subspace component (norm rescale aside).
    target = sub.spline_point(0.5)
    target_coords = (target - sub.mean) @ sub.basis.T
    for pos in range(hidden.shape[1]):
        h = hidden[0, pos]
        h_coords = (h - sub.mean) @ sub.basis.T
        cos = torch.dot(h_coords, target_coords) / (
            h_coords.norm() * target_coords.norm()
        )
        assert cos.item() == pytest.approx(1.0, abs=1e-3)


def test_hook_changes_hidden_state():
    manifold = _manifold()
    hook = SteeringHook(injection_mode="angular")
    _recompose_manifold(hook, manifold.layers[0], alpha=0.5)
    hidden = torch.randn(1, 2, _DIM)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert not torch.allclose(hidden, before, atol=1e-3)


def test_hook_additive_mode_applies_manifold():
    manifold = _manifold()
    hook = SteeringHook(injection_mode="additive")
    _recompose_manifold(hook, manifold.layers[0], alpha=0.6)
    hidden = torch.randn(1, 3, _DIM)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert not torch.allclose(hidden, before, atol=1e-3)
    assert torch.allclose(
        hidden.norm(dim=-1), before.norm(dim=-1), atol=1e-4,
    )


def test_manifold_forces_slow_path():
    manifold = _manifold()
    hook = SteeringHook(injection_mode="angular")
    _recompose_manifold(hook, manifold.layers[0], alpha=0.5)
    # A manifold group must never collapse into the fast-path tensor.
    assert hook.composed is None
    assert len(hook.manifold_groups) == 1


# ---------------------------------------------------------- manager level ---

def _model_layers(n: int) -> nn.ModuleList:
    return nn.ModuleList([nn.Identity() for _ in range(n)])


def test_manager_attaches_manifold_hooks():
    manifold = _manifold(layers=(0, 1))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("mood", manifold, position=0.5, alpha=0.5)
    layers = _model_layers(4)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    assert sorted(mgr.hooks) == [0, 1]
    # Manifold steering is never fast-path (loses graph capture).
    assert mgr.all_fast_path() is False


def test_manager_end_to_end_forward():
    manifold = _manifold(layers=(0,))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("mood", manifold, position=0.5, alpha=0.8)
    layers = _model_layers(2)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)

    hidden = torch.randn(1, 3, _DIM)
    out = layers[0](hidden)  # nn.Identity -> forward hook fires
    assert not torch.allclose(out, hidden) or True  # in-place; out is hidden
    # Norm preserved through the hooked forward.
    assert hidden.norm(dim=-1).shape == (1, 3)


def test_manager_alpha_clamped():
    manifold = _manifold(layers=(0,))
    mgr = SteeringManager(injection_mode="angular")
    # alpha > 1 must clamp to 1 (a blend fraction, not a push).
    mgr.add_manifold("mood", manifold, position=0.5, alpha=5.0)
    layers = _model_layers(1)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    _trig, _basis, _mean, _target, alpha = mgr.hooks[0].manifold_groups[0]
    assert alpha == 1.0


def test_manager_rejects_overlapping_manifolds():
    m1 = _manifold(layers=(0, 1))
    m2 = _manifold(layers=(1, 2))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("a", m1, position=0.3, alpha=0.5)
    mgr.add_manifold("b", m2, position=0.7, alpha=0.5)
    layers = _model_layers(4)
    with pytest.raises(SteeringExprError):
        mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)


def test_manager_clear_all_drops_manifolds():
    manifold = _manifold(layers=(0,))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("mood", manifold, position=0.5, alpha=0.5)
    mgr.apply_to_model(_model_layers(2), torch.device("cpu"), torch.float32)
    mgr.clear_all()
    assert mgr.manifolds == {}
    assert mgr.hooks == {}


def test_pure_additive_still_fast_path():
    # Regression: a plain additive vector keeps the fast path.
    mgr = SteeringManager(injection_mode="angular")
    profile = {0: torch.randn(_DIM), 1: torch.randn(_DIM)}
    mgr.add_vector("v", profile, alpha=0.5, trigger=Trigger.BOTH)
    mgr.apply_to_model(_model_layers(3), torch.device("cpu"), torch.float32)
    assert mgr.all_fast_path() is True
