"""Dispatch-time subspace synthesis (saklas.core.manifold.synthesize_subspace).

Pure CPU tests for the Stage-1 primitive of the full-unification arc: composing
an active steering term set (push directions + ablation directions) into one
affine subspace per layer with an ``along`` target that pushes the concept
directions toward their poles and collapses the ablated ones toward the origin.

The synthesizer is dormant (nothing routes through it yet); these tests pin its
geometry and prove its output drives ``inject_three_op`` as intended.
"""
from __future__ import annotations

import pytest
import torch

from saklas.core.manifold import (
    CustomDomain,
    SynthesizedSubspace,
    _ortho_basis,
    decompose,
    inject_three_op,
    synthesize_subspace,
)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v)


# --------------------------------------------------------------- _ortho_basis ---

def test_ortho_basis_orthonormal_and_ordered():
    torch.manual_seed(0)
    a, b = torch.randn(12), torch.randn(12)
    basis, kept = _ortho_basis([a, b])
    assert basis.shape == (2, 12)
    assert kept == [0, 1]
    # rows orthonormal
    gram = basis @ basis.T
    assert torch.allclose(gram, torch.eye(2), atol=1e-5)
    # first row is exactly the (normalized) first direction — order preserved
    assert torch.allclose(basis[0], _unit(a), atol=1e-5)


def test_ortho_basis_drops_parallel():
    a = torch.randn(8)
    basis, kept = _ortho_basis([a, 2.5 * a, a])  # all colinear
    assert basis.shape == (1, 8)
    assert kept == [0]


def test_ortho_basis_all_degenerate_empty():
    basis, kept = _ortho_basis([torch.zeros(5)])
    assert basis.shape == (0, 5)
    assert kept == []


# ---------------------------------------------------------- single push (fold) ---

def test_single_push_reduces_to_r1_fold():
    torch.manual_seed(1)
    u = _unit(torch.randn(8))
    baked = 2.0 * u                       # ‖baked‖ = 2
    neutral = 10.0 + torch.randn(8)
    synth = synthesize_subspace(
        push=[({3: baked}, 0.5)], ablate=[], neutral_means={3: neutral},
    )
    assert isinstance(synth, SynthesizedSubspace)
    sub = synth.layers[3]
    assert sub.is_affine and sub.rank == 1
    assert torch.allclose(sub.basis[0], u, atol=1e-5)
    assert torch.allclose(sub.mean, neutral, atol=1e-5)
    # target coord is the signed coeff (pole at 1); share is the composed norm
    assert torch.allclose(synth.target_coord[3], torch.tensor([0.5]), atol=1e-5)
    assert synth.share[3] == pytest.approx(1.0, abs=1e-4)  # ‖0.5 · 2u‖ = 1.0


def test_negative_coeff_flips_target_sign():
    u = _unit(torch.randn(6))
    synth = synthesize_subspace(
        push=[({0: u}, -0.7)], ablate=[], neutral_means={0: torch.zeros(6)},
    )
    # basis stays +û (unit-normalized); the sign lives in the target coord
    assert torch.allclose(synth.layers[0].basis[0], u, atol=1e-5)
    assert torch.allclose(synth.target_coord[0], torch.tensor([-0.7]), atol=1e-5)


# ------------------------------------------------------------- multi-concept ---

def test_two_orthogonal_pushes_independent_axes():
    torch.manual_seed(2)
    uh = _unit(torch.randn(16))
    ua = torch.randn(16)
    ua = _unit(ua - (ua @ uh) * uh)       # force orthogonal
    synth = synthesize_subspace(
        push=[({1: uh}, 0.3), ({1: ua}, 0.2)],
        ablate=[], neutral_means={1: torch.zeros(16)},
    )
    sub = synth.layers[1]
    assert sub.rank == 2
    # target = coords of Δ = 0.3·uh + 0.2·ua in the [uh, ua] basis
    assert torch.allclose(synth.target_coord[1], torch.tensor([0.3, 0.2]), atol=1e-5)


def test_share_is_composed_magnitude():
    torch.manual_seed(3)
    uh = _unit(torch.randn(16))
    ua = torch.randn(16)
    ua = _unit(ua - (ua @ uh) * uh)
    bh, ba = 4.0 * uh, 3.0 * ua            # baked magnitudes
    synth = synthesize_subspace(
        push=[({0: bh}, 0.5), ({0: ba}, 0.5)],
        ablate=[], neutral_means={0: torch.zeros(16)},
    )
    composed = 0.5 * bh + 0.5 * ba
    assert synth.share[0] == pytest.approx(
        float(torch.linalg.vector_norm(composed)), abs=1e-4,
    )


# --------------------------------------------------------------- ablation ---

def test_ablation_axis_gets_zero_target():
    torch.manual_seed(4)
    uh = _unit(torch.randn(16))
    ua = torch.randn(16)
    ua = _unit(ua - (ua @ uh) * uh)       # ablate dir ⟂ push dir
    synth = synthesize_subspace(
        push=[({2: uh}, 0.6)], ablate=[{2: ua}], neutral_means={2: torch.zeros(16)},
    )
    sub = synth.layers[2]
    assert sub.rank == 2                   # spans push ∪ ablate
    tgt = synth.target_coord[2]
    # push axis (row 0) carries the coeff; ablation axis (row 1) ≈ 0
    assert tgt[0] == pytest.approx(0.6, abs=1e-5)
    assert abs(float(tgt[1])) < 1e-5


def test_pure_ablation_zero_target_basis_spans():
    torch.manual_seed(5)
    ua = _unit(torch.randn(10))
    ba = 2.5 * ua
    synth = synthesize_subspace(
        push=[], ablate=[{4: ba}], neutral_means={4: torch.zeros(10)},
    )
    sub = synth.layers[4]
    assert sub.rank == 1
    assert torch.allclose(sub.basis[0], ua, atol=1e-5)
    assert torch.allclose(synth.target_coord[4], torch.zeros(1), atol=1e-6)
    # pure-ablation layer weights by the ablation magnitude
    assert synth.share[4] == pytest.approx(2.5, abs=1e-4)


# --------------------------------------------------------------- layer gating ---

def test_skips_layer_without_neutral():
    u = _unit(torch.randn(8))
    synth = synthesize_subspace(
        push=[({3: u, 7: u}, 0.5)], ablate=[], neutral_means={3: torch.zeros(8)},
    )
    assert sorted(synth.layers) == [3]     # layer 7 has no anchor → dropped


def test_drops_degenerate_direction_layer():
    u = _unit(torch.randn(8))
    synth = synthesize_subspace(
        push=[({0: u, 1: torch.zeros(8)}, 0.5)],
        ablate=[], neutral_means={0: torch.zeros(8), 1: torch.zeros(8)},
    )
    assert sorted(synth.layers) == [0]     # layer 1's only dir is degenerate


# ----------------------------------------------- integration with inject_three_op ---

def test_synthesized_subspace_drives_inject_three_op():
    """along=1 lands the in-subspace coords on the target: push axis set to its
    coeff, ablation axis removed; the off-subspace residual is preserved."""
    torch.manual_seed(6)
    D = 16
    uh = _unit(torch.randn(D))
    ua = torch.randn(D)
    ua = _unit(ua - (ua @ uh) * uh)
    neutral = torch.zeros(D)
    synth = synthesize_subspace(
        push=[({0: uh}, 0.5)], ablate=[{0: ua}], neutral_means={0: neutral},
    )
    sub = synth.layers[0]
    domain = CustomDomain(sub.rank)
    target = synth.target_coord[0]

    # an activation already leaning on both axes + an off-subspace residual
    perp = torch.randn(D)
    perp = perp - (perp @ uh) * uh - (perp @ ua) * ua
    h = neutral + 0.7 * uh + 0.9 * ua + perp
    seed = (h - sub.mean) @ sub.basis.T

    out, _foot = inject_three_op(
        h, sub, domain, target, seed, along=1.0, onto=0.0, toward=0.0,
    )
    # in-subspace coords land exactly on the target (push→0.5, ablate→0)
    coords_out = (out - sub.mean) @ sub.basis.T
    assert torch.allclose(coords_out, target, atol=1e-4)
    # off-subspace residual preserved (toward=0)
    _, perp_out = decompose(out, sub.mean, sub.basis)
    assert torch.allclose(perp_out, perp, atol=1e-4)


def test_synthesized_inject_identity_at_along_zero():
    torch.manual_seed(7)
    D = 12
    u = _unit(torch.randn(D))
    neutral = 5.0 + torch.randn(D)
    synth = synthesize_subspace(
        push=[({0: u}, 0.8)], ablate=[], neutral_means={0: neutral},
    )
    sub = synth.layers[0]
    domain = CustomDomain(sub.rank)
    h = neutral + 0.3 * u + 0.4 * torch.randn(D)
    seed = (h - sub.mean) @ sub.basis.T
    out, _foot = inject_three_op(
        h, sub, domain, synth.target_coord[0], seed,
        along=0.0, onto=0.0, toward=0.0,
    )
    assert torch.allclose(out, h, atol=1e-5)
