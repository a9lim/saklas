"""Hook + manager level tests for manifold steering — CPU only.

These cover the *integration* layer — ``SteeringManager.apply_to_model``'s
gain assembly (mean-1 share-weight + onto clamp; no lever / along clamp —
Step 8), ``add_manifold``'s plumbing, the hook's foot-following state, and the
slow-path forcing.  The kernel math itself (geodesic slide, onto collapse,
norm cap) lives in ``test_manifold_math.py``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch import nn

from saklas.core.hooks import (
    _MANIFOLD_GAIN,
    _manifold_layer_shares,
    SteeringHook,
    SteeringManager,
)
from saklas.core.manifold import (
    BoxAxis,
    BoxDomain,
    Manifold,
    decompose,
    eval_rbf,
    fit_layer_subspace,
    invert_parameterization,
)
from saklas.core.errors import ManifoldArityError, OverlappingManifoldError
from saklas.core.steering_expr import SteeringExprError
from saklas.core.triggers import Trigger, TriggerContext


def fit_layer_subspace_only(*args: Any, **kwargs: Any) -> Any:
    """Drop the EV ratio for tests that don't care about it."""
    sub, _ev = fit_layer_subspace(*args, **kwargs)
    return sub


_DIM = 8


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


def _circle(k: int, dim: int, scale: float = 1.0) -> torch.Tensor:
    """K activation centroids on a planar circle, with a DC offset.

    The 20.0 common-mode component mimics a real LM activation's large,
    persona-agnostic norm — without it the per-node centroids vary wildly
    in magnitude and the soft norm-cap inside ``subspace_inject`` fires on
    synthetic fits, masking the geometry under test.
    """
    out = torch.zeros(k, dim)
    base = torch.full((dim,), 20.0)
    for i in range(k):
        theta = 2.0 * math.pi * i / k
        out[i] = base.clone()
        out[i, 0] += scale * 2.0 * math.cos(theta)
        out[i, 1] += scale * 2.0 * math.sin(theta)
    return out


def _manifold(layers: Sequence[int] = (0, 1)) -> Manifold:
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
            L: fit_layer_subspace_only(_circle(6, _DIM, 1.0 + 0.2 * L), node_params)
            for L in layers
        },
        feature_space="raw",
    )


def _manifold2d(layers: Sequence[int] = (0,)) -> Manifold:
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
            L: fit_layer_subspace_only(20.0 + torch.randn(9, _DIM), node_params)
            for L in layers
        },
        feature_space="raw",
    )


def _model_layers(n: int) -> nn.ModuleList:
    return nn.ModuleList([nn.Identity() for _ in range(n)])


# ----------------------------------------------------------------- helpers ---

def _recompose_manifold(
    hook: SteeringHook,
    manifold: Manifold,
    layer: int,
    *,
    along: float = 0.5,
    onto: float = 0.0,
    position: Sequence[float] = (0.5,),
) -> None:
    """Stamp one manifold group onto ``hook`` directly.

    Mirrors what ``apply_to_model`` hands ``recompose``: the per-layer
    subspace + domain + authoring target/origin coords + the (already
    effective) per-layer coefficients.
    """
    ctx = TriggerContext()
    sub = manifold.layers[layer]
    domain = manifold.domain
    n = domain.intrinsic_dim
    target_coord = torch.tensor([float(c) for c in position], dtype=torch.float32)
    O_L = (getattr(manifold, "origin", None) or {}).get(layer)
    origin_coord = (
        torch.zeros(n) if O_L is None else O_L.reshape(-1).to(torch.float32)
    )
    hook.recompose(
        [(
            sub, domain, target_coord, origin_coord,
            along, onto, Trigger.BOTH,
        )],
        ctx,
        device=torch.device("cpu"),
    )


def _coeffs(hook: SteeringHook, gi: int = 0) -> tuple[float, float]:
    """The per-layer (along, onto) on group ``gi``."""
    _trig, _sub, _domain, _target, _origin, along, onto = (
        hook.manifold_groups[gi]
    )
    return along, onto


# ------------------------------------------------------------- hook level ---

def test_hook_all_zero_is_noop():
    manifold = _manifold()
    hook = SteeringHook()
    _recompose_manifold(hook, manifold, 0, along=0.0, onto=0.0)
    # A fully-zero term drops at recompose — no group, nothing fires.
    assert hook.manifold_groups == []
    hidden = torch.randn(1, 4, _DIM)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert torch.allclose(hidden, before, atol=1e-5)


def test_hook_along_changes_hidden():
    manifold = _manifold()
    hook = SteeringHook()
    _recompose_manifold(hook, manifold, 0, along=0.7)
    hidden = torch.randn(1, 3, _DIM)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert not torch.allclose(hidden, before, atol=1e-3)


def test_hook_onto_collapses_onto_surface():
    """``onto=1`` drives the off-manifold in-subspace residual ``H_n`` to
    zero — the post-fire activation's in-subspace part lands *on* the manifold
    surface (its nearest-point distance to M collapses toward 0), even though
    its norm need not shrink (the on-manifold foot can be larger than a
    near-origin off-manifold point).
    """
    manifold = _manifold()
    sub = manifold.layers[0]
    domain = manifold.domain
    hook = SteeringHook()
    # Pure onto: no slide, no off-subspace collapse — isolate the H_n scale.
    _recompose_manifold(hook, manifold, 0, along=0.0, onto=1.0)
    hidden = torch.randn(1, 4, _DIM) + sub.mean
    q_before = (hidden[0] - sub.mean) @ sub.basis.T
    _, dist_before = invert_parameterization(
        sub, domain, q_before, manifold.node_coords,
    )
    hook.hook_fn(None, None, hidden)
    q_after = (hidden[0] - sub.mean) @ sub.basis.T
    _, dist_after = invert_parameterization(
        sub, domain, q_after, manifold.node_coords,
    )
    # On the surface after collapse: residual to M near zero, and strictly
    # smaller than before.
    assert (dist_after < dist_before).all()
    assert dist_after.max().item() < 0.1


def test_hook_keeps_off_subspace_residual_verbatim():
    """The off-subspace residual ``H_o`` is always kept verbatim — the old
    ``toward`` op that scaled it is removed.  Even a strong along+onto slide
    leaves the orthogonal complement of the subspace untouched."""
    manifold = _manifold()
    sub = manifold.layers[0]
    hook = SteeringHook()
    _recompose_manifold(hook, manifold, 0, along=1.0, onto=1.0)
    hidden = torch.randn(1, 4, _DIM) + sub.mean
    _, h_perp_before = decompose(hidden, sub.mean, sub.basis)
    perp_before = h_perp_before.clone()
    hook.hook_fn(None, None, hidden)
    _, h_perp_after = decompose(hidden, sub.mean, sub.basis)
    assert torch.allclose(h_perp_after, perp_before, atol=1e-4)


def test_hook_forces_slow_path():
    manifold = _manifold()
    hook = SteeringHook()
    _recompose_manifold(hook, manifold, 0, along=0.5)
    # 4.0 has no fast path — a manifold group is the only (slow) shape.
    assert len(hook.manifold_groups) == 1


def test_hook_foot_following_warm_carry():
    """After a fire the hook stashes the last-position foot ``(B, 1, n)`` so
    the next decode step warm-starts; ``recompose`` resets it to cold."""
    manifold = _manifold()
    hook = SteeringHook()
    _recompose_manifold(hook, manifold, 0, along=0.6)
    # Cold after recompose.
    assert hook._manifold_feet == [None]
    # Prefill fire: (B=1, T=4) -> stash (1, 1, n).
    hidden = torch.randn(1, 4, _DIM)
    hook.hook_fn(None, None, hidden)
    foot = hook._manifold_feet[0]
    assert foot is not None
    n = manifold.domain.intrinsic_dim
    assert foot.shape == (1, 1, n)
    # A decode fire reuses the warm foot (shape matches) and updates it.
    decode = torch.randn(1, 1, _DIM)
    hook.hook_fn(None, None, decode)
    assert hook._manifold_feet[0] is not None
    assert hook._manifold_feet[0].shape == (1, 1, n)


def test_hook_trigger_gating_skips_inactive():
    """A manifold group whose trigger is inactive doesn't fire (and doesn't
    advance the foot)."""
    manifold = _manifold()
    hook = SteeringHook()
    ctx = TriggerContext()
    sub = manifold.layers[0]
    domain = manifold.domain
    target = torch.tensor([0.5])
    origin = torch.zeros(1)
    # GENERATED_ONLY trigger, but the context is in prefill (prompt) -> inactive.
    hook.recompose(
        [(
            sub, domain, target, origin, 0.6, 0.0, Trigger.GENERATED_ONLY,
        )],
        ctx,
        device=torch.device("cpu"),
    )
    ctx.is_prefill = True  # prompt phase -> GENERATED_ONLY inactive
    hidden = torch.randn(1, 3, _DIM)
    before = hidden.clone()
    hook.hook_fn(None, None, hidden)
    assert torch.allclose(hidden, before, atol=1e-5)
    assert hook._manifold_feet[0] is None  # never advanced


# ---------------------------------------------------------- manager level ---

def test_manager_attaches_manifold_hooks():
    manifold = _manifold(layers=(0, 1))
    mgr = SteeringManager()
    mgr.add_manifold("mood", manifold, position=(0.5,), along=0.5, onto=0.5)
    layers = _model_layers(4)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    assert sorted(mgr.hooks) == [0, 1]
    assert mgr.all_fast_path() is False


def test_manager_steer_n2_manifold():
    manifold = _manifold2d(layers=(0,))
    mgr = SteeringManager()
    # along=0.3 keeps the single-layer budget (0.3·gain=0.6) under the
    # water-fill cap, so no saturation warning pollutes the run.
    mgr.add_manifold("disk", manifold, position=(0.3, 0.8), along=0.3, onto=0.0)
    layers = _model_layers(2)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    hidden = torch.randn(1, 3, _DIM) + 20.0
    before = hidden.clone()
    layers[0](hidden)
    assert not torch.allclose(hidden, before, atol=1e-3)


def test_manager_position_length_mismatch_raises():
    manifold = _manifold2d(layers=(0,))
    mgr = SteeringManager()
    # A 2-D manifold steered with a single coordinate.  ``ManifoldArityError``
    # still subclasses ``SteeringExprError`` so the family catch keeps working.
    with pytest.raises(ManifoldArityError) as exc:
        mgr.add_manifold("disk", manifold, position=(0.5,), along=0.5, onto=0.5)
    assert isinstance(exc.value, SteeringExprError)


def test_manager_user_coeffs_clamped_to_unit():
    """User along/onto clamp to [0, 1].  With a single covered layer the
    mean-1 share == 1, so eff_along = base and eff_onto = min(1, base) — onto
    stays clamped per layer; along does not (Step 8)."""
    manifold = _manifold(layers=(0,))
    mgr = SteeringManager()
    mgr.add_manifold("mood", manifold, position=(0.5,), along=5.0, onto=5.0)
    layers = _model_layers(1)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    along, onto = _coeffs(mgr.hooks[0])
    assert along == pytest.approx(_MANIFOLD_GAIN, abs=1e-6)
    assert onto == pytest.approx(min(1.0, _MANIFOLD_GAIN), abs=1e-6)


def test_manager_along_share_weighting_mean_one():
    """Per-layer along is mean-1 share-weighted, so the cumulative budget
    ``Σ_L along_L = along · base · n_layers`` — the *per-layer* slide averages
    ``along · base`` regardless of layer count (n_layers-invariance, no
    lever)."""
    user_along = 0.1
    for n_layers in (1, 2, 4):
        manifold = _manifold(layers=tuple(range(n_layers)))
        mgr = SteeringManager()
        mgr.add_manifold(
            "mood", manifold, position=(0.5,),
            along=user_along, onto=0.0,
        )
        mgr.apply_to_model(
            _model_layers(n_layers), torch.device("cpu"), torch.float32,
        )
        total = sum(_coeffs(hook)[0] for hook in mgr.hooks.values())
        expected = user_along * _MANIFOLD_GAIN * n_layers
        assert total == pytest.approx(expected, abs=1e-4), (
            f"mean-1 share-weighted along at n={n_layers}: {total} ≠ {expected}"
        )


def test_manager_along_unclamped_overshoots():
    """along is NOT clamped per layer (Step 8).  The per-layer centroid spread
    differs (scale ∝ 1 + 0.2·L), so the top layer's mean-1 share > 1 slides it
    strictly past base — no [0, 1] cap.  ``norm_cap`` (kernel) is the only
    bound."""
    n_layers = 3
    manifold = _manifold(layers=tuple(range(n_layers)))
    mgr = SteeringManager()
    mgr.add_manifold("mood", manifold, position=(0.5,), along=1.0, onto=0.0)
    mgr.apply_to_model(_model_layers(n_layers), torch.device("cpu"), torch.float32)
    budgets = [_coeffs(hook)[0] for hook in mgr.hooks.values()]
    assert budgets, "expected manifold groups to be attached"
    # mean-1 ⇒ Σ = along·base·n_layers; the spread pushes the top layer past base.
    assert sum(budgets) == pytest.approx(1.0 * _MANIFOLD_GAIN * n_layers, abs=1e-4)
    assert max(budgets) > _MANIFOLD_GAIN


def test_manager_onto_clamped_per_layer():
    """onto is a bounded collapse fraction: clamped to [0, 1] per layer (a
    fraction > 1 would invert the residual).  A high-spread layer (mean-1
    share > 1) saturates its onto at 1.0."""
    n_layers = 3
    manifold = _manifold(layers=tuple(range(n_layers)))
    mgr = SteeringManager()
    mgr.add_manifold("mood", manifold, position=(0.5,), along=0.0, onto=1.0)
    mgr.apply_to_model(_model_layers(n_layers), torch.device("cpu"), torch.float32)
    for hook in mgr.hooks.values():
        _along, onto = _coeffs(hook)
        assert 0.0 <= onto <= 1.0 + 1e-6
    # The top-spread layer's mean-1 share > 1 ⇒ its onto saturates at 1.0.
    ontos = [_coeffs(h)[1] for h in mgr.hooks.values()]
    assert max(ontos) == pytest.approx(1.0, abs=1e-6)


def test_manager_two_coeffs_independent():
    """along / onto flow through independently — a term with only one nonzero
    coefficient leaves the other at zero per layer."""
    manifold = _manifold(layers=(0,))
    mgr = SteeringManager()
    mgr.add_manifold("mood", manifold, position=(0.5,), along=0.1, onto=0.0)
    mgr.apply_to_model(_model_layers(1), torch.device("cpu"), torch.float32)
    along, onto = _coeffs(mgr.hooks[0])
    assert along > 0.0
    assert onto == 0.0


def test_manager_share_weighting_weights_by_centroid_spread():
    """The per-layer share is proportional to centroid spread — the
    wider-spread (more discriminative) layer absorbs a larger slice of along,
    the ratio is preserved by the mean-1 normalization, and the total sums to
    ``along · base · n_layers``."""
    domain = BoxDomain([BoxAxis("t", periodic=True, period=1.0)])
    node_coords = torch.tensor([[i / 6] for i in range(6)])
    node_params = domain.embed(node_coords)
    manifold = Manifold(
        name="mood",
        domain=domain,
        node_labels=[f"n{i}" for i in range(6)],
        node_coords=node_coords,
        layers={
            0: fit_layer_subspace_only(_circle(6, _DIM, scale=1.0), node_params),
            1: fit_layer_subspace_only(_circle(6, _DIM, scale=2.0), node_params),
        },
        feature_space="raw",
    )
    user_along = 0.15
    mgr = SteeringManager()
    mgr.add_manifold("mood", manifold, position=(0.5,), along=user_along, onto=0.0)
    mgr.apply_to_model(_model_layers(2), torch.device("cpu"), torch.float32)
    along_0 = _coeffs(mgr.hooks[0])[0]
    along_1 = _coeffs(mgr.hooks[1])[0]
    assert along_1 > along_0
    assert along_1 / along_0 == pytest.approx(2.0, rel=0.05)
    assert (along_0 + along_1) == pytest.approx(
        user_along * _MANIFOLD_GAIN * 2, abs=1e-4
    )


def _euclidean_shares(manifold: Manifold) -> dict[int, float]:
    """Reference Euclidean ``‖coords‖_F`` share, **mean-1** normalized
    (``Σ = n_layers``) to match ``_manifold_layer_shares``."""
    raw = {}
    for L, sub in manifold.layers.items():
        _np, _rw, _pc = sub.rbf_params()
        c = eval_rbf(_np, _rw, _pc, _np)
        raw[L] = float(torch.linalg.vector_norm(c).item())
    tot = sum(raw.values())
    n = len(raw)
    return {L: v / tot * n for L, v in raw.items()}


def test_manifold_shares_use_baked_when_full_coverage():
    manifold = _manifold(layers=(0, 1))
    manifold.mahalanobis_share = {0: 1.0, 1: 3.0}
    shares = _manifold_layer_shares(manifold)
    # Mean-1: raw {1, 3} → ×(2/4) → {0.5, 1.5} (Σ = n_layers = 2).
    assert shares[0] == pytest.approx(0.5)
    assert shares[1] == pytest.approx(1.5)
    assert shares != pytest.approx(_euclidean_shares(manifold))


def test_manifold_shares_fall_back_to_euclidean_when_no_baked():
    manifold = _manifold(layers=(0, 1))
    assert manifold.mahalanobis_share == {}
    assert _manifold_layer_shares(manifold) == pytest.approx(
        _euclidean_shares(manifold)
    )


def test_manifold_shares_fall_back_when_partial_baked_coverage():
    manifold = _manifold(layers=(0, 1))
    manifold.mahalanobis_share = {0: 1.0}  # layer 1 missing → partial
    assert _manifold_layer_shares(manifold) == pytest.approx(
        _euclidean_shares(manifold)
    )


def test_manager_rejects_overlapping_manifolds():
    m1 = _manifold(layers=(0, 1))
    m2 = _manifold(layers=(1, 2))
    mgr = SteeringManager()
    mgr.add_manifold("a", m1, position=(0.3,), along=0.5, onto=0.0)
    mgr.add_manifold("b", m2, position=(0.7,), along=0.5, onto=0.0)
    layers = _model_layers(4)
    with pytest.raises(OverlappingManifoldError) as exc:
        mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    assert isinstance(exc.value, SteeringExprError)


def test_manager_clear_all_drops_manifolds():
    manifold = _manifold(layers=(0,))
    mgr = SteeringManager()
    mgr.add_manifold("mood", manifold, position=(0.5,), along=0.5, onto=0.0)
    mgr.apply_to_model(_model_layers(2), torch.device("cpu"), torch.float32)
    mgr.clear_all()
    assert mgr.manifolds == {}
    assert mgr.hooks == {}


def test_manager_reset_manifold_feet_cold_starts():
    """``reset_manifold_feet`` re-seeds every hook's foot to cold (None)."""
    manifold = _manifold(layers=(0, 1))
    mgr = SteeringManager()
    mgr.add_manifold("mood", manifold, position=(0.5,), along=0.6, onto=0.0)
    layers = _model_layers(2)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    # Fire once to warm the feet.
    for idx, hook in mgr.hooks.items():
        hook.hook_fn(None, None, torch.randn(1, 3, _DIM))
        assert hook._manifold_feet[0] is not None
    mgr.reset_manifold_feet()
    for hook in mgr.hooks.values():
        assert hook._manifold_feet == [None]


def test_any_steering_forces_slow_path():
    # 4.0: there is no composed-tensor fast path — any attached hook forces the
    # slow (ctx-consulting) ``subspace_inject`` path, so ``all_fast_path`` is
    # True only for the unsteered manager (no hooks).
    mgr = SteeringManager()
    assert mgr.all_fast_path() is True
    mgr.add_manifold("mood", _manifold(layers=(0, 1)), position=(0.5,),
                     along=0.5, onto=0.0)
    mgr.apply_to_model(_model_layers(3), torch.device("cpu"), torch.float32)
    assert mgr.all_fast_path() is False
