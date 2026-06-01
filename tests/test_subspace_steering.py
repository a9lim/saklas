"""Manager-level lowering of the dispatch-synthesized merged affine subspace.

The 4.0 unification routes vectors / poles / ``~``/``|`` / ``!`` / affine-``%``
through one ``synthesize_subspace`` → ``SteeringManager.add_subspace`` →
``inject_three_op`` backend (Step 5a — the manager half; the session dispatch
that builds the ``SynthesizedSubspace`` from active terms is Step 5b).  These
tests pin ``add_subspace``'s ``apply_to_model`` lowering: each layer becomes a
``CustomDomain(R_L)`` manifold-group entry with ``eff_along_L =
clamp(share_L · base_gain / N, 0, 1)``, ``onto = 0``, and the synth's per-layer
target.  The kernel math (the affine geodesic slide) lives in
``test_manifold_math.py`` / ``test_synthesize_subspace.py``; this is the gain
assembly + plumbing.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from saklas.core.hooks import _MANIFOLD_GAIN, SteeringManager
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
    """The ``(trig, sub, domain, target, origin, along, onto)`` group tuple."""
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

    Every layer gets ``share_L = coeff·1.0``, so after normalization the share
    is ``1/len(layers)`` at each — the clean substrate for the share-weight /
    lever-normalization arithmetic.
    """
    basis_dirs = {L: _row(torch.randn(_DIM)) for L in layers}
    coord_dirs = {L: torch.tensor([1.0]) for L in layers}
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
    trig, sub, domain, target, origin, _along, onto = _group(mgr, 0)
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


def test_custom_trigger_rides_through():
    synth = _single_layer_synth(0)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth, trigger=Trigger.AFTER_THINKING)
    mgr.apply_to_model(_model_layers(1), torch.device("cpu"), torch.float32)
    trig = _group(mgr, 0)[0]
    assert trig == Trigger.AFTER_THINKING


# ------------------------------------------------------------------- gain ---

def test_single_layer_no_lever_saturates_to_one():
    # One covered layer ⇒ normalized share == 1; no lever ⇒ N = 1, so
    # eff_along = clamp(1.0 · _MANIFOLD_GAIN) — saturated at the base gain.
    synth = _single_layer_synth(0)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    mgr.apply_to_model(_model_layers(1), torch.device("cpu"), torch.float32)
    along = _group(mgr, 0)[5]
    assert along == pytest.approx(min(1.0, _MANIFOLD_GAIN), abs=1e-6)


def test_share_weight_normalizes_across_layers():
    # Equal per-layer magnitude ⇒ each normalized share = 1/3; no lever ⇒
    # eff_along = (1/3)·_MANIFOLD_GAIN at every layer.
    synth = _equal_share_synth((0, 1, 2))
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    mgr.apply_to_model(_model_layers(3), torch.device("cpu"), torch.float32)
    expected = min(1.0, (1.0 / 3.0) * _MANIFOLD_GAIN)
    for L in (0, 1, 2):
        assert _group(mgr, L)[5] == pytest.approx(expected, abs=1e-6)


def test_lever_normalization_scales_gain():
    # Uniform lever ℓ ⇒ N = ℓ ⇒ gain = base/ℓ ⇒ eff_along = share/ℓ (clamped).
    # ℓ = 0.5 over 3 equal-share layers keeps it unsaturated: 1/3 → 2/3.
    layers = (0, 1, 2)
    synth = _equal_share_synth(layers)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth, lever={L: 0.5 for L in layers})
    mgr.apply_to_model(_model_layers(3), torch.device("cpu"), torch.float32)
    expected = min(1.0, (1.0 / 3.0) * (_MANIFOLD_GAIN / 0.5))
    for L in layers:
        assert _group(mgr, L)[5] == pytest.approx(expected, abs=1e-6)


def test_partial_lever_coverage_falls_back_to_n1():
    # A lever dict that misses a covered layer ⇒ N = 1 fallback (no
    # normalization), same as lever=None.
    layers = (0, 1, 2)
    synth = _equal_share_synth(layers)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth, lever={0: 0.5})  # layers 1,2 missing
    mgr.apply_to_model(_model_layers(3), torch.device("cpu"), torch.float32)
    expected = min(1.0, (1.0 / 3.0) * _MANIFOLD_GAIN)
    for L in layers:
        assert _group(mgr, L)[5] == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------- hot path ---

def test_hot_path_slides_in_subspace_component_to_target():
    # At eff_along = 1 the affine slide lands the in-subspace coord exactly on
    # the target (q discarded).  Single layer, no lever ⇒ along = _MANIFOLD_GAIN
    # (= 1.0); pick a coeff/coord so the target is a clean value.
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
    assert _group(mgr, 0)[5] == pytest.approx(1.0, abs=1e-6)  # along saturated

    hidden = (20.0 + torch.randn(1, 4, _DIM)).to(torch.float32)
    before = hidden.clone()
    layers[0](hidden)  # Identity forwards → hook mutates in place
    assert not torch.allclose(hidden, before, atol=1e-3)
    # In-subspace coord landed on the target (along == 1 ⇒ foot == target).
    sub = synth.layers[0]
    coord = (hidden - sub.mean) @ sub.basis.T
    target = synth.target_coord[0]
    assert torch.allclose(
        coord, target.expand_as(coord), atol=1e-4,
    )


def test_clear_all_drops_subspaces():
    synth = _single_layer_synth(0)
    mgr = SteeringManager()
    mgr.add_subspace("__affine__", synth)
    mgr.apply_to_model(_model_layers(1), torch.device("cpu"), torch.float32)
    mgr.clear_all()
    assert mgr.subspaces == {}
    assert mgr.hooks == {}
