"""Angular (rotation-based) injection hook math.

Exercises the v2.1 default ``injection_mode="angular"`` path on a
``nn.Identity``-equivalent module so we can assert per-position rotation
properties without loading a real transformer.  Companion to
:mod:`tests.test_hooks_triggers`, which pins the legacy additive
behavior under explicit ``injection_mode="additive"``.

Properties tested:

* Pure rotation preserves L2 norm bit-exactly (within fp32 roundoff).
* Single-term ``α=0`` is a no-op — angular at θ=0 is identity.
* Single-term ``α=1`` rotates a residual orthogonal to the concept
  direction by ``θ_max=π/2``, fully aligning it with ``d̂``.
* Multi-term cooperating α's yield the same direction as the α-weighted
  sum; multi-term opposing α's reduce the rotation magnitude.
* No ``_STEER_GAIN`` is applied under angular mode (regression guard
  against silently re-introducing the legacy gain on the angular path).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from saklas.core.hooks import (
    DEFAULT_THETA_MAX,
    SteeringHook,
    SteeringManager,
    _angular_inplace,
)
from saklas.core.triggers import Trigger, TriggerContext


class _Passthrough(nn.Module):
    """Module returning ``(hidden,)`` so the hook can mutate it in place."""

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden,)


def _unit(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


# ---------------------------------------------------------------------------
# Pure-rotation primitive (``_angular_inplace``).  Norm preservation is
# the load-bearing mathematical property — under angular mode the v1.x
# ``vector_norm → mul_(ratio)`` rescale is dropped, so the rotation
# itself must preserve norm exactly.
# ---------------------------------------------------------------------------


class TestAngularInplaceMath:
    def test_pi_over_two_aligns_orthogonal_to_d_hat(self) -> None:
        h = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # along axis 0
        d = torch.tensor([0.0, 1.0, 0.0, 0.0])      # orthogonal target
        _angular_inplace(h, d, math.cos(math.pi / 2), math.sin(math.pi / 2))
        rotated = h.squeeze().tolist()
        # Should land on +d̂ within floating roundoff.
        assert abs(rotated[1] - 1.0) < 1e-5
        assert abs(rotated[0]) < 1e-5

    def test_norm_preserved_under_random_rotation(self) -> None:
        torch.manual_seed(0)
        h = torch.randn(2, 3, 8) * 5.0
        d = _unit(8, seed=1)
        n_before = torch.linalg.vector_norm(h, dim=-1)
        _angular_inplace(h, d, math.cos(math.pi / 4), math.sin(math.pi / 4))
        n_after = torch.linalg.vector_norm(h, dim=-1)
        # Allow ~1e-4 of fp32 slack; rotation is mathematically exact
        # but the device-side fp32 ops carry the usual accumulated noise.
        assert (n_after - n_before).abs().max().item() < 1e-4

    def test_already_aligned_position_unchanged(self) -> None:
        # h_unit == d̂ → no orthogonal direction to rotate into → no-op.
        # Critical edge case: naive rotation would shrink the residual.
        h = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
        d = torch.tensor([1.0, 0.0, 0.0, 0.0])
        h_orig = h.clone()
        _angular_inplace(h, d, math.cos(math.pi / 4), math.sin(math.pi / 4))
        assert torch.allclose(h, h_orig, atol=1e-6)

    def test_zero_angle_identity(self) -> None:
        h = torch.tensor([[[3.0, 4.0, 0.0, 0.0]]])
        h_orig = h.clone()
        d = torch.tensor([1.0, 0.0, 0.0, 0.0])
        _angular_inplace(h, d, 1.0, 0.0)  # cos=1, sin=0 ⇒ identity
        assert torch.allclose(h, h_orig)

    def test_none_d_hat_is_noop(self) -> None:
        h = torch.tensor([[[1.0, 1.0, 1.0]]])
        h_orig = h.clone()
        _angular_inplace(
            h, None, math.cos(math.pi / 4), math.sin(math.pi / 4),
        )
        assert torch.equal(h, h_orig)


# ---------------------------------------------------------------------------
# Hook-level fast path under angular default.  Reuses the
# ``_Passthrough`` module so we exercise the rotation through the same
# attach/forward mechanism the live model uses.
# ---------------------------------------------------------------------------


class TestAngularFastPath:
    def test_alpha_zero_is_identity(self) -> None:
        mod = _Passthrough()
        hook = SteeringHook()
        ctx = TriggerContext()
        vec = _unit(16) * 1.0
        hook.recompose(
            additive_entries=[(vec, 0.0, Trigger.BOTH)],
            ablation_entries=[],
            device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
        )
        # All-zero α → group dropped at recompose; composed is None.
        assert hook.composed is None
        h = torch.randn(1, 4, 16)
        h_before = h.clone()
        mod.register_forward_hook(hook.hook_fn)
        mod(h)
        assert torch.equal(h, h_before)

    def test_alpha_one_aligns_to_concept(self) -> None:
        """Single concept at α=1 with h initially orthogonal → full
        rotation onto ``d̂``.  Theta = ||composed|| / α_budget * θ_max
        = (1×1) / (1×1) × π/2 = π/2 in this construction.
        """
        mod = _Passthrough()
        hook = SteeringHook()  # angular default
        ctx = TriggerContext()
        # Use a unit-norm baked direction so α_budget = |α| × ||baked||
        # = 1.0, and ||composed|| = 1.0 → ratio = 1, θ = θ_max = π/2.
        baked = _unit(16, seed=0)
        hook.recompose(
            additive_entries=[(baked, 1.0, Trigger.BOTH)],
            ablation_entries=[],
            device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
        )
        # Build h orthogonal to baked at every position so cos0 ≈ 0.
        h_unit = _unit(16, seed=1)
        # Project off the baked component to make truly orthogonal.
        h_unit = h_unit - (h_unit @ baked) * baked
        h_unit = h_unit / h_unit.norm()
        h = (5.0 * h_unit).repeat(1, 4, 1).clone()
        n_before = torch.linalg.vector_norm(h, dim=-1)
        mod.register_forward_hook(hook.hook_fn)
        mod(h)
        n_after = torch.linalg.vector_norm(h, dim=-1)
        # Norm preserved.
        assert (n_after - n_before).abs().max().item() < 1e-4
        # New direction aligned with ``baked`` (cos ≈ 1).
        h_unit_after = h.squeeze(0)[0] / h.squeeze(0)[0].norm()
        cos_to_d = float((h_unit_after * baked).sum().item())
        assert cos_to_d > 0.999

    def test_norm_preserved_arbitrary_input(self) -> None:
        torch.manual_seed(0)
        mod = _Passthrough()
        hook = SteeringHook()
        ctx = TriggerContext()
        baked = _unit(16, seed=2)
        hook.recompose(
            additive_entries=[(baked, 0.5, Trigger.BOTH)],
            ablation_entries=[],
            device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
        )
        h = torch.randn(2, 5, 16) * 3.0
        n_before = torch.linalg.vector_norm(h, dim=-1)
        mod.register_forward_hook(hook.hook_fn)
        mod(h)
        n_after = torch.linalg.vector_norm(h, dim=-1)
        assert (n_after - n_before).abs().max().item() < 1e-3


# ---------------------------------------------------------------------------
# Multi-term composition.  Cooperating α's amplify rotation; opposing
# α's cancel.  Captures the "sum-first-then-rotate" semantics agreed in
# the prototype plan.
# ---------------------------------------------------------------------------


class TestAngularMultiTerm:
    def test_cooperating_alphas_rotate_to_combined_direction(self) -> None:
        """Two parallel concepts at the same α should rotate h into
        their shared direction at the maximum allowed angle (saturating
        at θ_max, since the budget grows with the number of terms but
        ``||composed||`` grows in lock-step)."""
        mod = _Passthrough()
        hook = SteeringHook()
        ctx = TriggerContext()
        # Two copies of the same direction; α=0.5 each.
        baked = _unit(16, seed=3)
        hook.recompose(
            additive_entries=[
                (baked.clone(), 0.5, Trigger.BOTH),
                (baked.clone(), 0.5, Trigger.BOTH),
            ],
            ablation_entries=[],
            device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
        )
        # composed = (0.5 + 0.5) × baked = baked; α_budget = 1.0; ratio=1.
        # θ = θ_max — same effect as a single α=1 term.
        h_unit = _unit(16, seed=4)
        h_unit = h_unit - (h_unit @ baked) * baked
        h_unit = h_unit / h_unit.norm()
        h = (3.0 * h_unit).repeat(1, 1, 1).clone()
        mod.register_forward_hook(hook.hook_fn)
        mod(h)
        cos_to_d = float((h.squeeze() / h.squeeze().norm() * baked).sum().item())
        assert cos_to_d > 0.999

    def test_opposing_alphas_reduce_theta(self) -> None:
        """``+α × baked  +  −α × baked`` collapses to zero composed
        direction; the angular fast path falls back to identity (no
        rotation can be defined when ``||composed|| → 0``)."""
        mod = _Passthrough()
        hook = SteeringHook()
        ctx = TriggerContext()
        baked = _unit(16, seed=5)
        hook.recompose(
            additive_entries=[
                (baked.clone(), 0.5, Trigger.BOTH),
                (baked.clone(), -0.5, Trigger.BOTH),
            ],
            ablation_entries=[],
            device=torch.device("cpu"), dtype=torch.float32, ctx=ctx,
        )
        # ``composed`` lives but its norm is ~0; ``_d_hat`` should be
        # None because ``_refresh_angular_cache`` short-circuits below
        # the ``1e-12`` clamp.
        assert hook._d_hat is None
        h = torch.randn(1, 4, 16)
        h_before = h.clone()
        mod.register_forward_hook(hook.hook_fn)
        mod(h)
        assert torch.allclose(h, h_before, atol=1e-6)


# ---------------------------------------------------------------------------
# Manager-level dispatch — angular default drops ``_STEER_GAIN``.
# ---------------------------------------------------------------------------


class TestManagerAngularDefault:
    def test_angular_default_skips_steer_gain(self) -> None:
        """The v1.x ``effective_alpha = α × _STEER_GAIN`` path runs only
        under ``injection_mode="additive"``.  Under angular default the
        composed tensor must be exactly ``α × baked`` so the α/θ map
        carries the user's literal coefficient.
        """
        mgr = SteeringManager()  # angular default
        layers = nn.ModuleList([_Passthrough() for _ in range(2)])
        baked = _unit(16, seed=7)
        mgr.add_vector(
            "demo", {0: baked.clone()}, alpha=0.4, trigger=Trigger.BOTH,
        )
        mgr.apply_to_model(
            layers, device=torch.device("cpu"), dtype=torch.float32,
        )
        # Fast path stamped composed = α × baked (no gain multiplier).
        expected = 0.4 * baked
        assert torch.allclose(mgr.hooks[0].composed, expected, atol=1e-6)
        # Manager and hook agree on theta_max default.
        assert mgr.theta_max == DEFAULT_THETA_MAX
        assert mgr.hooks[0].theta_max == DEFAULT_THETA_MAX
        mgr.clear_all()

    def test_explicit_additive_keeps_steer_gain(self) -> None:
        """Manager constructed with ``injection_mode='additive'`` must
        reproduce the legacy gain so existing additive-mode tests stay
        bit-identical."""
        from saklas.core.hooks import _STEER_GAIN
        mgr = SteeringManager(injection_mode="additive")
        layers = nn.ModuleList([_Passthrough() for _ in range(2)])
        baked = _unit(16, seed=8)
        mgr.add_vector(
            "demo", {0: baked.clone()}, alpha=0.5, trigger=Trigger.BOTH,
        )
        mgr.apply_to_model(
            layers, device=torch.device("cpu"), dtype=torch.float32,
        )
        expected = (0.5 * _STEER_GAIN) * baked
        assert torch.allclose(mgr.hooks[0].composed, expected, atol=1e-6)
        mgr.clear_all()
