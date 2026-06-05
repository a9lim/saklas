"""Manager-level lowering of the dispatch-synthesized merged affine subspace.

The 4.0 unification routes vectors / poles / ``~``/``|`` / ``!`` / affine-``%``
through one ``synthesize_subspace`` → ``SteeringManager.add_subspace`` →
``subspace_inject`` backend (Step 5a — the manager half; the session dispatch
that builds the ``SynthesizedSubspace`` from active terms is Step 5b).  These
tests pin ``add_subspace``'s ``apply_to_model`` lowering: each layer becomes a
``CustomDomain(R_L)`` manifold-group entry with ``eff_along_L =
share_L · base_gain`` (``share_L`` mean-1 normalized; no lever / ``N``, no
``[0, 1]`` clamp — Step 8), ``onto = 0``, and the synth's per-layer target.  The
kernel math (the affine geodesic slide) lives in ``test_manifold_math.py`` /
``test_synthesize_subspace.py``; this is the gain assembly + plumbing.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from saklas.core.hooks import _MANIFOLD_ALONG_GAIN, SteeringManager
from saklas.core.manifold import (
    CustomDomain,
    SynthesizedSubspace,
    synthesize_subspace,
)
from saklas.core.triggers import Trigger


_DIM = 8


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v)


def _row(v: torch.Tensor) -> torch.Tensor:
    return _unit(v).reshape(1, -1)


def _model_layers(n: int) -> nn.ModuleList:
    return nn.ModuleList([nn.Identity() for _ in range(n)])


def _group(mgr: SteeringManager, layer: int, gi: int = 0):
    """The ``(trig, sub, domain, target, origin, along, onto, kappa)`` group tuple."""
    return mgr.hooks[layer].manifold_groups[gi]


def _single_layer_synth(
    layer: int = 0, *, coeff: float = 0.5, coord: float = 1.0,
) -> SynthesizedSubspace:
    u = _unit(torch.randn(_DIM))
    neutral = 20.0 + torch.randn(_DIM)
    return synthesize_subspace(
        push=[({layer: _row(u)}, {layer: torch.tensor([coord])}, coeff)],
        ablate=[], neutral_means={layer: neutral},
    )


def _equal_share_synth(
    layers: tuple[int, ...] = (0, 1, 2), *, coeff: float = 0.5,
) -> SynthesizedSubspace:
    """One push term spanning ``layers`` with identical per-layer magnitude.

    Every layer gets the same ``share_L = coeff·1.0``, so the **mean-1**
    normalization (``Σ share = n_layers``) leaves each at ``1.0`` — the clean
    substrate for the share-weight arithmetic.
    """
    basis_dirs = {L: _row(torch.randn(_DIM)) for L in layers}
    coord_dirs = {L: torch.tensor([1.0]) for L in layers}
    neutral_means = {L: 20.0 + torch.randn(_DIM) for L in layers}
    return synthesize_subspace(
        push=[(basis_dirs, coord_dirs, coeff)],
        ablate=[], neutral_means=neutral_means,
    )


def _unequal_share_synth(
    coords: tuple[float, ...] = (3.0, 1.0), *, coeff: float = 0.5,
) -> SynthesizedSubspace:
    """One push term with **unequal** per-layer magnitudes ``coeff·|coord|``.

    ``coords = (3, 1)`` ⇒ raw shares ``(1.5, 0.5)``, mean-1 normalized to
    ``(1.5, 0.5)`` (sum already = n_layers) — so the high-signal layer's
    ``eff_along`` exceeds 1 and must NOT be clamped (Step 8).
    """
    layers = tuple(range(len(coords)))
    basis_dirs = {L: _row(torch.randn(_DIM)) for L in layers}
    coord_dirs = {L: torch.tensor([float(c)]) for L, c in zip(layers, coords)}
    neutral_means = {L: 20.0 + torch.randn(_DIM) for L in layers}
    return synthesize_subspace(
        push=[(basis_dirs, coord_dirs, coeff)],
        ablate=[], neutral_means=neutral_means,
    )


# ----------------------------------------------------------------- lowering ---

def test_add_subspace_lowers_to_custom_domain_group():
    synth = _single_layer_synth(0)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    mgr.apply_to_model(_model_layers(2), torch.device("cpu"), torch.float32)

    assert list(mgr.hooks) == [0]
    trig, sub, domain, target, origin, _along, onto, _kappa = _group(mgr, 0)
    assert sub.is_affine
    assert isinstance(domain, CustomDomain)
    assert domain.intrinsic_dim == sub.rank
    # The synth target rides through verbatim; the affine origin is span-coord 0.
    assert torch.allclose(target, synth.target_coord[0].to(torch.float32), atol=1e-6)
    assert torch.allclose(origin, torch.zeros(sub.rank), atol=1e-6)
    assert onto == 0.0
    assert trig == Trigger.BOTH


def test_subspace_forces_slow_path():
    synth = _single_layer_synth(0)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    mgr.apply_to_model(_model_layers(1), torch.device("cpu"), torch.float32)
    assert mgr.all_fast_path() is False


def test_static_steerable_for_affine_both_subspace():
    # An always-active (Trigger.BOTH) affine subspace lowers to the single-
    # affine fast-path hook on every covered layer, so the whole gen is
    # StaticCache / graph-capture eligible even though it IS steered.
    synth = _equal_share_synth((0, 1, 2))
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    mgr.apply_to_model(_model_layers(3), torch.device("cpu"), torch.float32)
    assert mgr.all_fast_path() is False        # it IS steered
    assert mgr.static_steerable() is True       # but statically so
    assert all(
        h._single_affine_fast is not None for h in mgr.hooks.values()
    )


def test_static_steerable_false_when_unsteered():
    # Unsteered is the cheaper ``all_fast_path`` case, not ``static_steerable``.
    mgr = SteeringManager()
    assert mgr.static_steerable() is False
    assert mgr.all_fast_path() is True


def test_static_steerable_false_for_gated_subspace():
    # A phase-gated (non-BOTH) trigger keeps the hook on the ctx-consulting
    # general path, so it is NOT statically capturable.
    synth = _single_layer_synth(0)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth, trigger=Trigger.AFTER_THINKING)
    mgr.apply_to_model(_model_layers(1), torch.device("cpu"), torch.float32)
    assert mgr.static_steerable() is False


def test_custom_trigger_rides_through():
    synth = _single_layer_synth(0)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth, trigger=Trigger.AFTER_THINKING)
    mgr.apply_to_model(_model_layers(1), torch.device("cpu"), torch.float32)
    trig = _group(mgr, 0)[0]
    assert trig == Trigger.AFTER_THINKING


# ------------------------------------------------------------------- gain ---

def test_single_layer_share_one_gives_base_gain():
    # One covered layer ⇒ mean-1 share == 1; no lever, no clamp ⇒
    # eff_along = 1.0 · _MANIFOLD_ALONG_GAIN (the translate slide gain).
    synth = _single_layer_synth(0)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    mgr.apply_to_model(_model_layers(1), torch.device("cpu"), torch.float32)
    along = _group(mgr, 0)[5]
    assert along == pytest.approx(_MANIFOLD_ALONG_GAIN, abs=1e-6)


def test_share_weight_mean_one_across_layers():
    # Equal per-layer magnitude ⇒ each mean-1 share = 1.0 (not 1/3); no lever ⇒
    # eff_along = 1.0·_MANIFOLD_ALONG_GAIN at every layer — n_layers-invariant.
    synth = _equal_share_synth((0, 1, 2))
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    mgr.apply_to_model(_model_layers(3), torch.device("cpu"), torch.float32)
    expected = _MANIFOLD_ALONG_GAIN
    for L in (0, 1, 2):
        assert _group(mgr, L)[5] == pytest.approx(expected, abs=1e-6)


def test_n_layers_invariance():
    # A single covered layer and a 4-layer equal-share fit both put ≈ base on
    # each contributing layer — the mean-1 share is n_layers-invariant (the
    # A⊂B-consistency property the torn-out lever broke).
    for n in (1, 2, 4):
        synth = _equal_share_synth(tuple(range(n)))
        mgr = SteeringManager()
        mgr.add_subspace("__affine__", synth)
        mgr.apply_to_model(_model_layers(n), torch.device("cpu"), torch.float32)
        for L in range(n):
            assert _group(mgr, L)[5] == pytest.approx(_MANIFOLD_ALONG_GAIN, abs=1e-6)


def test_high_share_layer_unclamped_share_weighting():
    # Unequal magnitudes ⇒ the high-signal layer's mean-1 share > 1, so its
    # eff_along = share·gain is proportionally larger and is NOT clamped (Step 8
    # — the de-rogued real-coord target carries the magnitude; norm_cap is the
    # only bound).  share (1.5, 0.5) ⇒ eff_along (1.5, 0.5)·gain, ratio 3:1.
    synth = _unequal_share_synth((3.0, 1.0))
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    mgr.apply_to_model(_model_layers(2), torch.device("cpu"), torch.float32)
    assert _group(mgr, 0)[5] == pytest.approx(1.5 * _MANIFOLD_ALONG_GAIN, abs=1e-6)
    assert _group(mgr, 1)[5] == pytest.approx(0.5 * _MANIFOLD_ALONG_GAIN, abs=1e-6)
    # unclamped: the exact share ratio rides through (a [0, 1]/water-fill clamp
    # would distort it).
    assert _group(mgr, 0)[5] / _group(mgr, 1)[5] == pytest.approx(3.0, abs=1e-6)


# --------------------------------------------------------------- hot path ---

def test_hot_path_translates_in_subspace_component_by_target():
    # A pure-push affine group translates the in-subspace coord by the fixed
    # offset ``eff·target`` (κ=0 on every push axis): ``coord = q + eff·target``
    # — it preserves ``q`` (the per-token spread) rather than collapsing onto
    # the target.  Base-robust: exact at any gain.
    u = _unit(torch.randn(_DIM))
    neutral = 20.0 + torch.randn(_DIM)
    synth = synthesize_subspace(
        push=[({0: _row(u)}, {0: torch.tensor([1.0])}, 0.7)],
        ablate=[], neutral_means={0: neutral},
    )
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    layers = _model_layers(1)
    mgr.apply_to_model(layers, torch.device("cpu"), torch.float32)
    eff = _group(mgr, 0)[5]
    assert eff == pytest.approx(_MANIFOLD_ALONG_GAIN, abs=1e-6)  # share == 1
    assert torch.allclose(_group(mgr, 0)[7], torch.zeros(1), atol=1e-6)  # κ=0 (pure push)

    sub = synth.layers[0]
    target = synth.target_coord[0]
    # Start near neutral so the displacement stays well under the norm_cap.
    hidden = (sub.mean + 0.3 * torch.randn(1, 4, _DIM)).to(torch.float32)
    q_coord = (hidden - sub.mean) @ sub.basis.T
    before = hidden.clone()
    layers[0](hidden)  # Identity forwards → hook mutates in place
    assert not torch.allclose(hidden, before, atol=1e-3)
    coord = (hidden - sub.mean) @ sub.basis.T
    expected = q_coord + eff * target.expand_as(q_coord)
    assert torch.allclose(coord, expected, atol=1e-4)


def test_clear_all_drops_subspaces():
    synth = _single_layer_synth(0)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    mgr.apply_to_model(_model_layers(1), torch.device("cpu"), torch.float32)
    mgr.clear_all()
    assert mgr.subspaces == {}
    assert mgr.hooks == {}
