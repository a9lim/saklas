"""Hook-level tests for manifold steering injection — CPU only."""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from saklas.core.hooks import SteeringHook, SteeringManager
from saklas.core.manifold import (
    BoxAxis,
    BoxDomain,
    Manifold,
    fit_layer_subspace,
)
from saklas.core.steering_expr import SteeringExprError
from saklas.core.triggers import Trigger, TriggerContext

_DIM = 8


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed torch globally before each test.

    Several tests in this module draw random hidden states with
    ``torch.randn`` and then assert on values like cosine ~ 1.0 within a
    tight tolerance.  Without a per-test seed the global RNG state
    leaked between tests, making the angular-rotation-precision
    assertions flaky depending on test execution order.
    """
    torch.manual_seed(0)


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
    """A 1-D periodic (loop) manifold over 6 nodes."""
    domain = BoxDomain([BoxAxis("t", periodic=True, period=1.0)])
    node_coords = torch.tensor([[i / 6] for i in range(6)])
    node_params = domain.embed(node_coords)
    return Manifold(
        name="mood",
        domain=domain,
        node_labels=[f"n{i}" for i in range(6)],
        node_coords=node_coords,
        layers={
            L: fit_layer_subspace(_circle(6, _DIM, 1.0 + 0.2 * L), node_params)
            for L in layers
        },
        feature_space="raw",
    )


def _manifold2d(layers=(0,)) -> Manifold:
    """A 2-D box (disk-like) manifold over a 3x3 grid of 9 nodes."""
    domain = BoxDomain([
        BoxAxis("u", periodic=False, lo=0.0, hi=1.0),
        BoxAxis("v", periodic=False, lo=0.0, hi=1.0),
    ])
    coords = torch.tensor(
        [[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)]
    )
    node_params = domain.embed(coords)
    torch.manual_seed(7)
    return Manifold(
        name="disk",
        domain=domain,
        node_labels=[f"n{i}" for i in range(9)],
        node_coords=coords,
        layers={
            L: fit_layer_subspace(torch.randn(9, _DIM), node_params)
            for L in layers
        },
        feature_space="raw",
    )


# ------------------------------------------------------------- hook level ---

def _recompose_manifold(
    hook: SteeringHook, manifold: Manifold, layer: int, alpha: float,
    position=(0.5,),
):
    ctx = TriggerContext()
    sub = manifold.layers[layer]
    target = manifold.manifold_point(layer, position)
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
    _recompose_manifold(hook, manifold, 0, alpha=0.0)
    hidden = torch.randn(1, 4, _DIM)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert torch.allclose(hidden, before, atol=1e-5)


def test_hook_preserves_norm():
    manifold = _manifold()
    hook = SteeringHook(injection_mode="angular")
    _recompose_manifold(hook, manifold, 0, alpha=0.7)
    hidden = torch.randn(1, 5, _DIM)
    norm_before = hidden.norm(dim=-1).clone()
    hook.hook_fn(None, None, hidden)
    assert torch.allclose(hidden.norm(dim=-1), norm_before, atol=1e-4)


def test_hook_alpha_one_lands_in_subspace_on_target():
    manifold = _manifold()
    sub = manifold.layers[0]
    hook = SteeringHook(injection_mode="angular")
    _recompose_manifold(hook, manifold, 0, alpha=1.0)
    hidden = torch.randn(1, 3, _DIM)
    hook.hook_fn(None, None, hidden)

    target = manifold.manifold_point(0, (0.5,))
    target_coords = (target - sub.mean) @ sub.basis.T
    for pos in range(hidden.shape[1]):
        h = hidden[0, pos]
        h_coords = (h - sub.mean) @ sub.basis.T
        cos = torch.dot(h_coords, target_coords) / (
            h_coords.norm() * target_coords.norm()
        )
        # ``subspace_replace`` snaps the in-subspace component onto
        # ``target``, then rescales the *whole* result to the original
        # per-position norm.  That last rescale multiplies ``target``
        # but not ``mean`` by the rescale factor ``s``, so the centered
        # coordinates become ``s·target@basis.T − mean@basis.T`` rather
        # than ``s·(target − mean)@basis.T``.  Direction is preserved up
        # to a few degrees in practice — the 5e-3 tolerance captures
        # that residual without masking a genuine snap regression.
        assert cos.item() == pytest.approx(1.0, abs=5e-3)


def test_hook_changes_hidden_state():
    manifold = _manifold()
    hook = SteeringHook(injection_mode="angular")
    _recompose_manifold(hook, manifold, 0, alpha=0.5)
    hidden = torch.randn(1, 2, _DIM)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert not torch.allclose(hidden, before, atol=1e-3)


def test_hook_additive_mode_applies_manifold():
    manifold = _manifold()
    hook = SteeringHook(injection_mode="additive")
    _recompose_manifold(hook, manifold, 0, alpha=0.6)
    hidden = torch.randn(1, 3, _DIM)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert not torch.allclose(hidden, before, atol=1e-3)
    assert torch.allclose(hidden.norm(dim=-1), before.norm(dim=-1), atol=1e-4)


def test_manifold_forces_slow_path():
    manifold = _manifold()
    hook = SteeringHook(injection_mode="angular")
    _recompose_manifold(hook, manifold, 0, alpha=0.5)
    assert hook.composed is None
    assert len(hook.manifold_groups) == 1


# ---------------------------------------------------------- manager level ---

def _model_layers(n: int) -> nn.ModuleList:
    return nn.ModuleList([nn.Identity() for _ in range(n)])


def test_manager_attaches_manifold_hooks():
    manifold = _manifold(layers=(0, 1))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("mood", manifold, position=(0.5,), alpha=0.5)
    layers = _model_layers(4)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    assert sorted(mgr.hooks) == [0, 1]
    assert mgr.all_fast_path() is False


def test_manager_steer_n2_manifold():
    manifold = _manifold2d(layers=(0,))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("disk", manifold, position=(0.3, 0.8), alpha=0.7)
    layers = _model_layers(2)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    hidden = torch.randn(1, 3, _DIM)
    before = hidden.clone()
    layers[0](hidden)
    assert not torch.allclose(hidden, before, atol=1e-3)


def test_manager_position_length_mismatch_raises():
    manifold = _manifold2d(layers=(0,))
    mgr = SteeringManager(injection_mode="angular")
    # A 2-D manifold steered with a single coordinate.
    with pytest.raises(SteeringExprError):
        mgr.add_manifold("disk", manifold, position=(0.5,), alpha=0.5)


def test_manager_alpha_clamped():
    manifold = _manifold(layers=(0,))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("mood", manifold, position=(0.5,), alpha=5.0)
    layers = _model_layers(1)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    _trig, _basis, _mean, _target, alpha = mgr.hooks[0].manifold_groups[0]
    assert alpha == 1.0


def test_manager_rejects_overlapping_manifolds():
    m1 = _manifold(layers=(0, 1))
    m2 = _manifold(layers=(1, 2))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("a", m1, position=(0.3,), alpha=0.5)
    mgr.add_manifold("b", m2, position=(0.7,), alpha=0.5)
    layers = _model_layers(4)
    with pytest.raises(SteeringExprError):
        mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)


def test_manager_clear_all_drops_manifolds():
    manifold = _manifold(layers=(0,))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("mood", manifold, position=(0.5,), alpha=0.5)
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
