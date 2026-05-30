"""Hook-level tests for manifold steering injection — CPU only."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch import nn

from saklas.core.hooks import (
    _MANIFOLD_GAIN_ADDITIVE,
    _MANIFOLD_GAIN_ANGULAR,
    _manifold_layer_shares,
    InjectionMode,
    SteeringHook,
    SteeringManager,
)
from saklas.core.manifold import (
    BoxAxis,
    BoxDomain,
    Manifold,
    eval_rbf,
    fit_layer_subspace,
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


def _manifold(
    layers: Sequence[int] = (0, 1),
    *,
    lever: float | dict[int, float] | None = None,
) -> Manifold:
    """A 1-D periodic (loop) manifold over 6 nodes.

    ``lever`` stamps a per-layer steering lever ``f_L`` (a float applies to
    every layer, a dict gives per-layer values).  ``None`` leaves the lever
    empty, so ``apply_to_model`` falls back to ``N = 1`` (no normalization)
    — the pre-lever gain scale, which most of these hook-math tests assert
    against.  A stamped lever exercises the ``effective_gain = base / N``
    path.
    """
    domain = BoxDomain([BoxAxis("t", periodic=True, period=1.0)])
    node_coords = torch.tensor([[i / 6] for i in range(6)])
    node_params = domain.embed(node_coords)
    mf = Manifold(
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
    if lever is not None:
        mf.lever = (
            {L: float(lever) for L in layers}
            if isinstance(lever, (int, float))
            else dict(lever)
        )
    return mf


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
            L: fit_layer_subspace_only(torch.randn(9, _DIM), node_params)
            for L in layers
        },
        feature_space="raw",
    )


# ------------------------------------------------------------- hook level ---

def _recompose_manifold(
    hook: SteeringHook, manifold: Manifold, layer: int, alpha: float,
    position: Sequence[float] = (0.5,),
) -> None:
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


def test_hook_preserves_centered_norm_angular():
    """Angular-mode manifold steering dispatches to ``subspace_rotate``,
    which preserves ``||h - mean||`` (the magnitude *inside* the
    manifold's coordinate system) but not ``||h||`` itself — the
    rotation in the affine subspace plane is exact, no global rescale.
    """
    manifold = _manifold()
    sub = manifold.layers[0]
    hook = SteeringHook(injection_mode="angular")
    _recompose_manifold(hook, manifold, 0, alpha=0.7)
    hidden = torch.randn(1, 5, _DIM)
    centered_before = (hidden - sub.mean).norm(dim=-1).clone()
    hook.hook_fn(None, None, hidden)
    centered_after = (hidden - sub.mean).norm(dim=-1)
    assert torch.allclose(centered_after, centered_before, atol=1e-4)


def test_hook_alpha_one_rotates_h_par_into_in_plane_perp():
    """At α=1 with default θ_max=π/2 the angular manifold path rotates
    ``h_par`` exactly 90° in the plane toward the target — so its
    centered direction matches the in-plane perpendicular axis ``w_unit``
    (the part of ``target - mean`` orthogonal to ``h_par`` inside the
    subspace).  This is the angular analogue of subspace_replace's
    α=1 snap; magnitudes are preserved, the direction is set to the
    "as far as θ_max lets us go toward target" axis.
    """
    manifold = _manifold()
    sub = manifold.layers[0]
    hook = SteeringHook(injection_mode="angular")

    hidden = torch.randn(1, 3, _DIM)
    # Compute the expected w_unit per position *before* the hook fires.
    centered_before = hidden[0] - sub.mean
    h_par_c_before = (centered_before @ sub.basis.T) @ sub.basis
    target = manifold.manifold_point(0, (0.5,))
    target_c = target - sub.mean
    target_unit = target_c / target_c.norm()
    expected_w = []
    for pos in range(hidden.shape[1]):
        u = h_par_c_before[pos] / h_par_c_before[pos].norm()
        cos0 = (u * target_unit).sum()
        w = target_unit - cos0 * u
        expected_w.append(w / w.norm())

    _recompose_manifold(hook, manifold, 0, alpha=1.0)
    hook.hook_fn(None, None, hidden)

    for pos in range(hidden.shape[1]):
        centered = hidden[0, pos] - sub.mean
        h_par_after = (centered @ sub.basis.T) @ sub.basis
        cos = torch.dot(h_par_after, expected_w[pos]) / h_par_after.norm()
        assert cos.item() == pytest.approx(1.0, abs=1e-3)


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
    # A 2-D manifold steered with a single coordinate.  The dedicated
    # ``ManifoldArityError`` still subclasses ``SteeringExprError`` so the
    # parse-time family catch keeps working.
    with pytest.raises(ManifoldArityError) as exc:
        mgr.add_manifold("disk", manifold, position=(0.5,), alpha=0.5)
    assert isinstance(exc.value, SteeringExprError)


def test_manager_alpha_clamped():
    # User α is clamped to [0, 1]; with a single covered layer share_L ==
    # 1.0, so the raw per-layer budget would be ``1.0 · _MANIFOLD_GAIN_
    # ANGULAR`` (8.0).  Under angular mode that budget is the rotation
    # angle in θ_max units, and a single layer can't rotate past θ_max —
    # the water-fill caps it at 1.0 (one full θ_max).  (The excess has
    # nowhere to go on a one-layer manifold, so it saturates.)
    manifold = _manifold(layers=(0,))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("mood", manifold, position=(0.5,), alpha=5.0)
    layers = _model_layers(1)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    _trig, _basis, _mean, _target, alpha = mgr.hooks[0].manifold_groups[0]
    assert alpha == pytest.approx(1.0, abs=1e-6)


def test_manager_share_weighting_sums_to_user_alpha():
    """Per-layer α is share-weighted so the cumulative budget
    ``Σ_L α_L = α · gain`` regardless of how many layers the manifold
    covers — analogous to vector steering's ``Σ_L share_L · α · θ_max =
    α · θ_max`` invariant.  This is the layer-count-invariance property
    that keeps the user-facing α-regime independent of model depth.

    Tested in the *unsaturated* regime: ``α · gain ≤ 1`` so the angular
    water-fill cap (no layer past θ_max) never bites and the cumulative
    budget is exactly ``α · gain`` for every layer count.  The saturated
    regime (where the cap does bite) is covered by
    ``test_manager_angular_budget_caps_per_layer`` below.
    """
    # 0.1 · 2.0 = 0.2 ≤ 1.0, so even a single layer holds the whole
    # budget without saturating. (No lever on these stubs ⇒ N=1.)
    user_alpha = 0.1
    for n_layers in (1, 2, 4):
        manifold = _manifold(layers=tuple(range(n_layers)))
        mgr = SteeringManager(injection_mode="angular")
        mgr.add_manifold(
            "mood", manifold, position=(0.5,), alpha=user_alpha,
        )
        mgr.apply_to_model(
            _model_layers(n_layers), torch.device("cpu"), torch.float32,
        )
        total = sum(
            grp[4]
            for hook in mgr.hooks.values()
            for grp in hook.manifold_groups
        )
        expected = user_alpha * _MANIFOLD_GAIN_ANGULAR
        assert total == pytest.approx(expected, abs=1e-4), (
            f"share-weighted budget at n={n_layers} layers: "
            f"{total} ≠ {expected}"
        )


def test_manager_angular_budget_caps_per_layer():
    """Angular manifold steering never rotates a layer past θ_max.

    With a high user α, the raw per-layer budget (``α · share_L · gain``)
    blows past 1.0 (= θ > θ_max) — ``subspace_rotate`` does NOT clamp, so
    the manager's water-fill must.  Assert no per-layer budget exceeds
    1.0 and the cumulative budget is the saturated ceiling ``min(α · gain,
    n_layers)``.  Additive mode keeps the raw share-weighted value (a
    blend fraction, no θ_max ceiling).
    """
    import saklas.core.hooks as hooks_mod
    n_layers = 3
    lever = 0.2
    # Small lever ⇒ effective gain = base / N = 2.0 / 0.2 = 10; at α=1.0 the
    # raw cumulative budget 10 ≫ n_layers=3 ⇒ every layer saturates at 1.0.
    manifold = _manifold(layers=tuple(range(n_layers)), lever=lever)
    eff_gain = _MANIFOLD_GAIN_ANGULAR / lever
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("mood", manifold, position=(0.5,), alpha=1.0)
    # The saturation tell fires once per manifold name; clear it so this
    # test sees its own warning regardless of order.
    hooks_mod._warned_manifold_saturated.discard("mood")
    with pytest.warns(UserWarning, match="saturates the angular rotation budget"):
        mgr.apply_to_model(
            _model_layers(n_layers), torch.device("cpu"), torch.float32,
        )
    budgets = [
        grp[4]
        for hook in mgr.hooks.values()
        for grp in hook.manifold_groups
    ]
    assert budgets, "expected manifold groups to be attached"
    for b in budgets:
        assert b <= 1.0 + 1e-6, f"layer rotates past θ_max: {b}"
    # Saturated ⇒ cumulative pins at the ceiling min(α · eff_gain, n_layers).
    assert sum(budgets) == pytest.approx(
        min(1.0 * eff_gain, n_layers), abs=1e-4,
    )

    # Additive: same α / lever, no per-layer cap — the raw share-weighted
    # blend fractions stand (they exceed 1.0; the operator norm-restores).
    mgr_add = SteeringManager(injection_mode="additive")
    mgr_add.add_manifold("mood", manifold, position=(0.5,), alpha=1.0)
    mgr_add.apply_to_model(
        _model_layers(n_layers), torch.device("cpu"), torch.float32,
    )
    add_budgets = [
        grp[4]
        for hook in mgr_add.hooks.values()
        for grp in hook.manifold_groups
    ]
    # Cumulative additive budget is the un-capped α · (_MANIFOLD_GAIN_ADDITIVE / N).
    assert sum(add_budgets) == pytest.approx(
        1.0 * _MANIFOLD_GAIN_ADDITIVE / lever, abs=1e-4,
    )


def test_manager_lever_normalization():
    """Both modes divide the gain by the share-weighted lever ``N``.

    Two single-layer manifolds identical but for their stamped lever
    ``f_L`` should produce effective α inversely proportional to ``N`` (a
    smaller-lever fit gets a proportionally larger gain).  This is the
    normalization that makes the per-α effect invariant to subspace
    dimension and selection metric — and it applies to *both* injection
    modes (it replaces the former additive-only ``1/√EV`` quality
    factor).  A manifold with no lever falls back to ``N = 1`` (the
    un-normalized base gain).
    """
    # Kept small enough that angular stays unsaturated for both levers
    # (effective α = base·user_α/N ≤ 1, so the per-layer water-fill cap
    # doesn't pin either value and flatten the ratio): the larger
    # effective α is the additive small-lever case 4·0.15/0.4 = 1.5, which
    # additive leaves uncapped; the largest angular value is 2·0.15/0.4 =
    # 0.75 ≤ 1.
    user_alpha = 0.15
    big_lever = 0.8
    small_lever = 0.4

    def _eff(mode: InjectionMode, mf: Manifold) -> float:
        mgr = SteeringManager(injection_mode=mode)
        mgr.add_manifold("m", mf, position=(0.5,), alpha=user_alpha)
        mgr.apply_to_model(
            _model_layers(1), torch.device("cpu"), torch.float32,
        )
        return mgr.hooks[0].manifold_groups[0][4]

    modes: list[InjectionMode] = ["angular", "additive"]
    for mode in modes:
        big = _manifold(layers=(0,), lever=big_lever)
        small = _manifold(layers=(0,), lever=small_lever)
        none = _manifold(layers=(0,))  # no lever -> N = 1

        a_big = _eff(mode, big)
        a_small = _eff(mode, small)
        a_none = _eff(mode, none)
        # Smaller lever ⇒ larger effective gain (base / N).
        assert a_small > a_big
        # Ratio is exactly the inverse lever ratio (single layer ⇒
        # N = lever; share = 1).
        assert a_small / a_big == pytest.approx(big_lever / small_lever, rel=1e-4)
        # No-lever fit uses the un-normalized base gain (N = 1).
        base = _MANIFOLD_GAIN_ANGULAR if mode == "angular" else _MANIFOLD_GAIN_ADDITIVE
        assert a_none == pytest.approx(user_alpha * base, abs=1e-6)
        # And the big-lever effective α is base·α/N.
        assert a_big == pytest.approx(user_alpha * base / big_lever, abs=1e-4)


def test_manager_per_mode_gain_dispatch():
    """The manifold gain is per-mode: ``_MANIFOLD_GAIN_ANGULAR`` under
    angular, ``_MANIFOLD_GAIN_ADDITIVE`` under additive.  The additive
    gain is calibrated against the angular one via vector's
    ``_STEER_GAIN``, mirroring the precedent that additive operators
    need ~2× the calibration of angular ones.  Same user α, same
    manifold, different mode → different per-layer effective α.
    """
    manifold = _manifold(layers=(0,))
    # Single-layer manifold so share_L = 1.0 and the per-layer α is
    # ``user_alpha · gain`` exactly — clean comparison across modes.
    # Kept in the unsaturated angular regime (``α · gain ≤ 1``) so the
    # per-layer water-fill cap doesn't pin the angular value and mask the
    # gain comparison: 0.1 · 2.0 = 0.2 ≤ 1.0. (No lever ⇒ N=1.)
    user_alpha = 0.1
    mgr_ang = SteeringManager(injection_mode="angular")
    mgr_ang.add_manifold("m", manifold, position=(0.5,), alpha=user_alpha)
    mgr_ang.apply_to_model(
        _model_layers(1), torch.device("cpu"), torch.float32,
    )
    mgr_add = SteeringManager(injection_mode="additive")
    mgr_add.add_manifold("m", manifold, position=(0.5,), alpha=user_alpha)
    mgr_add.apply_to_model(
        _model_layers(1), torch.device("cpu"), torch.float32,
    )

    alpha_ang = mgr_ang.hooks[0].manifold_groups[0][4]
    alpha_add = mgr_add.hooks[0].manifold_groups[0][4]
    assert alpha_ang == pytest.approx(
        user_alpha * _MANIFOLD_GAIN_ANGULAR, abs=1e-6,
    )
    assert alpha_add == pytest.approx(
        user_alpha * _MANIFOLD_GAIN_ADDITIVE, abs=1e-6,
    )
    assert alpha_add > alpha_ang  # additive's higher gain manifests in α


def test_manager_share_weighting_weights_by_centroid_spread():
    """The per-layer share is proportional to the per-layer centroid
    spread (``||centroids_L - mean_L||_F``) — the manifold analogue of
    vector steering's ``||baked_L||``.  Layers where the personas
    cluster more widely (more discriminative signal) absorb a larger
    slice of α; layers where the centroids collapse near the manifold
    origin take a smaller slice.
    """
    # Two-layer manifold where layer 1 has a 2× wider centroid spread
    # than layer 0 (scale=1.0 vs scale=2.0 in ``_circle``).  Build it
    # by hand rather than via ``_manifold`` which uses
    # ``1.0 + 0.2 * L`` — too small a contrast to test cleanly.
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
    # Kept in the unsaturated regime (each layer's budget ≤ 1.0) so the
    # angular water-fill cap doesn't flatten the 1:2 ratio: the wider
    # share is ``(2/3) · α · 2`` ≤ 1.0 for α ≤ 0.75. (No lever ⇒ N=1.)
    user_alpha = 0.15
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("mood", manifold, position=(0.5,), alpha=user_alpha)
    mgr.apply_to_model(
        _model_layers(2), torch.device("cpu"), torch.float32,
    )
    alpha_0 = mgr.hooks[0].manifold_groups[0][4]
    alpha_1 = mgr.hooks[1].manifold_groups[0][4]
    assert alpha_1 > alpha_0      # wider-spread layer gets more budget
    # Spreads are 1:2, so shares should be roughly 1:2 too.
    assert alpha_1 / alpha_0 == pytest.approx(2.0, rel=0.05)
    # And the total still sums to ``user_alpha · _MANIFOLD_GAIN_ANGULAR`` — the
    # gain pins the α-scale into vector-comparable territory; share
    # weights normalize the per-layer distribution.
    assert (alpha_0 + alpha_1) == pytest.approx(
        user_alpha * _MANIFOLD_GAIN_ANGULAR, abs=1e-4,
    )


def _euclidean_shares(manifold: Manifold) -> dict[int, float]:
    """Reference Euclidean ``‖coords‖_F`` share, normalized across layers."""
    raw = {}
    for L, sub in manifold.layers.items():
        c = eval_rbf(
            sub.node_params, sub.rbf_weights, sub.poly_coeffs, sub.node_params,
        )
        raw[L] = float(torch.linalg.vector_norm(c).item())
    tot = sum(raw.values())
    return {L: v / tot for L, v in raw.items()}


def test_manifold_shares_use_baked_when_full_coverage():
    """A baked Mahalanobis share covering every layer is used (normalized)
    in place of the Euclidean centroid-spread."""
    manifold = _manifold(layers=(0, 1))
    manifold.mahalanobis_share = {0: 1.0, 1: 3.0}
    shares = _manifold_layer_shares(manifold)
    assert shares[0] == pytest.approx(0.25)
    assert shares[1] == pytest.approx(0.75)
    # And it is NOT the Euclidean spread (which weights by centroid scale).
    assert shares != pytest.approx(_euclidean_shares(manifold))


def test_manifold_shares_fall_back_to_euclidean_when_no_baked():
    """Empty baked share (no whitener at fit time) → Euclidean spread."""
    manifold = _manifold(layers=(0, 1))
    assert manifold.mahalanobis_share == {}
    assert _manifold_layer_shares(manifold) == pytest.approx(
        _euclidean_shares(manifold)
    )


def test_manifold_shares_fall_back_when_partial_baked_coverage():
    """A baked share missing a layer → Euclidean for ALL layers (no metric
    mixing), matching the all-or-nothing gate.  The stray ``{0: ...}`` entry
    must be ignored, not blended in."""
    manifold = _manifold(layers=(0, 1))
    manifold.mahalanobis_share = {0: 1.0}  # layer 1 missing → partial
    assert _manifold_layer_shares(manifold) == pytest.approx(
        _euclidean_shares(manifold)
    )


def test_manager_rejects_overlapping_manifolds():
    m1 = _manifold(layers=(0, 1))
    m2 = _manifold(layers=(1, 2))
    mgr = SteeringManager(injection_mode="angular")
    mgr.add_manifold("a", m1, position=(0.3,), alpha=0.5)
    mgr.add_manifold("b", m2, position=(0.7,), alpha=0.5)
    layers = _model_layers(4)
    # ``OverlappingManifoldError`` still subclasses ``SteeringExprError``.
    with pytest.raises(OverlappingManifoldError) as exc:
        mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    assert isinstance(exc.value, SteeringExprError)


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
