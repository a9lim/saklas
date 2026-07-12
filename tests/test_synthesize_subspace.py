"""Dispatch-time subspace synthesis (saklas.core.manifold.synthesize_subspace).

Pure CPU tests for the Stage-1 primitive of the full-unification arc: composing
an active steering term set (any-rank affine push fragments + ablation
directions) into one affine subspace per layer with an ``along`` target that
pushes the concept subspaces to their coeff-scaled coords and collapses the
ablated ones to the origin.

Each push fragment is ``(basis_rows, coord_target, coeff)``: a rank-1 steering
vector (``(1, D)`` basis), or a rank-R subspace like ``personas%pirate``
(``(R, D)`` basis + an R-dim node-coord target).  The synthesizer is dormant
(nothing routes through it yet); these tests pin its geometry and prove its
output drives ``subspace_inject`` as intended.
"""
from __future__ import annotations

import pytest
import torch

from saklas.core.manifold import (
    CustomDomain,
    LayerSubspace,
    SynthesizedSubspace,
    _ortho_basis,
    decompose,
    subspace_inject,
    synthesize_subspace as _synthesize_subspace,
)


def synthesize_subspace(*args, neutral_means, whitener=None, **kwargs):
    """Current-shape test adapter: every synthesis owns a covering metric."""
    if whitener is None and neutral_means:
        from tests._whitener import isotropic_whitener
        first = next(iter(neutral_means.values()))
        whitener = isotropic_whitener(list(neutral_means), int(first.numel()))
    return _synthesize_subspace(
        *args, neutral_means=neutral_means, whitener=whitener, **kwargs,
    )


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v)


def _row(v: torch.Tensor) -> torch.Tensor:
    """A unit (1, D) basis row from a (D,) direction."""
    return _unit(v).reshape(1, -1)


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
    neutral = 10.0 + torch.randn(8)
    synth = synthesize_subspace(
        push=[({3: _row(u)}, {3: torch.tensor([1.0])}, 0.5)],
        ablate=[], neutral_means={3: neutral},
    )
    assert isinstance(synth, SynthesizedSubspace)
    sub = synth.layers[3]
    assert sub.is_affine and sub.rank == 1
    assert torch.allclose(sub.basis[0], u, atol=1e-5)
    assert torch.allclose(sub.mean, neutral, atol=1e-5)
    # target coord = coeff · coord_target (pole at 1) = 0.5
    assert torch.allclose(synth.target_coord[3], torch.tensor([0.5]), atol=1e-5)
    # share = ‖Δ‖ = ‖0.5 · (1.0·u)‖ = 0.5.  Consolidated: |target| == share now
    # (the predecessor mixed a unit target with a baked-magnitude share = 1.0).
    assert synth.share[3] == pytest.approx(0.5, abs=1e-4)


def test_negative_coeff_flips_target_sign():
    u = _unit(torch.randn(6))
    synth = synthesize_subspace(
        push=[({0: _row(u)}, {0: torch.tensor([1.0])}, -0.7)],
        ablate=[], neutral_means={0: torch.zeros(6)},
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
        push=[
            ({1: _row(uh)}, {1: torch.tensor([1.0])}, 0.3),
            ({1: _row(ua)}, {1: torch.tensor([1.0])}, 0.2),
        ],
        ablate=[], neutral_means={1: torch.zeros(16)},
    )
    sub = synth.layers[1]
    assert sub.rank == 2
    # target = coords of Δ = 0.3·uh + 0.2·ua in the [uh, ua] basis
    assert torch.allclose(synth.target_coord[1], torch.tensor([0.3, 0.2]), atol=1e-5)


def test_share_is_displacement_magnitude():
    torch.manual_seed(3)
    uh = _unit(torch.randn(16))
    ua = torch.randn(16)
    ua = _unit(ua - (ua @ uh) * uh)
    # coord targets 4.0 / 3.0 with coeff 0.5 ⇒ Δ = 0.5·4·uh + 0.5·3·ua.
    synth = synthesize_subspace(
        push=[
            ({0: _row(uh)}, {0: torch.tensor([4.0])}, 0.5),
            ({0: _row(ua)}, {0: torch.tensor([3.0])}, 0.5),
        ],
        ablate=[], neutral_means={0: torch.zeros(16)},
    )
    delta = 0.5 * 4.0 * uh + 0.5 * 3.0 * ua
    assert synth.share[0] == pytest.approx(
        float(torch.linalg.vector_norm(delta)), abs=1e-4,
    )


# ------------------------------------------------------------- rank-R fragment ---

def test_rank8_push_fragment():
    """A single rank-8 affine push (the personas%node shape): an 8-row
    orthonormal basis + an 8-dim node-coord target.  ``share`` and the
    reconstructed world target match the fragment's coeff-scaled displacement."""
    torch.manual_seed(8)
    D = 32
    B8, _kept = _ortho_basis(list(torch.randn(8, D)))   # (8, D) orthonormal
    assert B8.shape == (8, D)
    coords = torch.randn(8)
    neutral = 5.0 + torch.randn(D)
    synth = synthesize_subspace(
        push=[({0: B8}, {0: coords}, 0.5)],
        ablate=[], neutral_means={0: neutral},
    )
    sub = synth.layers[0]
    assert sub.rank == 8
    # The merged basis preserves the requested direction while the universal
    # synthesis norm cap may shorten a high-rank displacement.
    delta = 0.5 * (coords @ B8)
    world_target = synth.target_coord[0] @ sub.basis     # (R,) @ (R, D) -> (D,)
    assert torch.nn.functional.cosine_similarity(
        world_target, delta, dim=0,
    ) == pytest.approx(1.0, abs=1e-5)
    assert torch.linalg.vector_norm(world_target) <= torch.linalg.vector_norm(delta)
    assert synth.share[0] == pytest.approx(
        float(torch.linalg.vector_norm(delta)), abs=1e-4,
    )


# --------------------------------------------------------------- ablation ---

def test_ablation_axis_gets_zero_target():
    torch.manual_seed(4)
    uh = _unit(torch.randn(16))
    ua = torch.randn(16)
    ua = _unit(ua - (ua @ uh) * uh)       # ablate dir ⟂ push dir
    synth = synthesize_subspace(
        push=[({2: _row(uh)}, {2: torch.tensor([1.0])}, 0.6)],
        ablate=[{2: ua}], neutral_means={2: torch.zeros(16)},
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
    # pure-ablation layer weights by the (raw) ablation magnitude
    assert synth.share[4] == pytest.approx(2.5, abs=1e-4)


# --------------------------------------------------------------- layer gating ---

def test_rejects_layer_without_neutral():
    from saklas.core.mahalanobis import WhitenerError
    u = _unit(torch.randn(8))
    with pytest.raises(WhitenerError, match=r"missing \[7\]"):
        synthesize_subspace(
            push=[(
                {3: _row(u), 7: _row(u)},
                {3: torch.tensor([1.0]), 7: torch.tensor([1.0])},
                0.5,
            )],
            ablate=[], neutral_means={3: torch.zeros(8)},
        )


def test_drops_degenerate_direction_layer():
    u = _unit(torch.randn(8))
    synth = synthesize_subspace(
        push=[(
            {0: _row(u), 1: torch.zeros(1, 8)},
            {0: torch.tensor([1.0]), 1: torch.tensor([1.0])},
            0.5,
        )],
        ablate=[], neutral_means={0: torch.zeros(8), 1: torch.zeros(8)},
    )
    assert sorted(synth.layers) == [0]     # layer 1's only dir is degenerate


# ----------------------------------------------- integration with subspace_inject ---

def test_synthesized_subspace_drives_subspace_inject():
    """along=1 *translates* the push axis by its target and *collapses* the
    ablation axis to 0 (per-axis κ); the off-subspace residual is preserved."""
    torch.manual_seed(6)
    D = 16
    uh = _unit(torch.randn(D))
    ua = torch.randn(D)
    ua = _unit(ua - (ua @ uh) * uh)
    neutral = torch.zeros(D)
    synth = synthesize_subspace(
        push=[({0: _row(uh)}, {0: torch.tensor([1.0])}, 0.5)],
        ablate=[{0: ua}], neutral_means={0: neutral},
    )
    sub = synth.layers[0]
    domain = CustomDomain(sub.rank)
    target = synth.target_coord[0]
    kappa = synth.kappa[0]                        # 0 on the push axis, 1 on ablate

    # an activation already leaning on both axes + an off-subspace residual
    perp = torch.randn(D)
    perp = perp - (perp @ uh) * uh - (perp @ ua) * ua
    h = neutral + 0.7 * uh + 0.9 * ua + perp
    seed = (h - sub.mean) @ sub.basis.T

    out, _foot = subspace_inject(
        h, sub, domain, target, seed, along=1.0, onto=0.0, kappa=kappa,
    )
    coords_out = (out - sub.mean) @ sub.basis.T
    q = (h - sub.mean) @ sub.basis.T
    # per-axis κ-blend: push axis translates by target, ablate axis collapses to 0
    assert torch.allclose(coords_out, q + (target - kappa * q), atol=1e-4)
    # the ablated direction is driven out of the activation (component → 0)
    assert abs(float((out - sub.mean) @ ua)) < 1e-3
    # the pushed direction is translated (grew by coeff·coord = 0.5), not collapsed
    assert float((out - sub.mean) @ uh) == pytest.approx(0.7 + 0.5, abs=1e-3)
    # off-subspace residual preserved (kept verbatim)
    _, perp_out = decompose(out, sub.mean, sub.basis)
    assert torch.allclose(perp_out, perp, atol=1e-4)


def test_rank8_synthesized_drives_subspace_inject():
    """A rank-8 push fragment routed through subspace_inject *translates* its
    reduced coords by the target under a full along slide (κ=0, pure push)."""
    torch.manual_seed(9)
    D = 24
    B8, _kept = _ortho_basis(list(torch.randn(8, D)))
    coords = torch.randn(8)
    neutral = 3.0 + torch.randn(D)
    synth = synthesize_subspace(
        push=[({0: B8}, {0: coords}, 0.4)],
        ablate=[], neutral_means={0: neutral},
    )
    sub = synth.layers[0]
    domain = CustomDomain(sub.rank)
    target = synth.target_coord[0]
    kappa = synth.kappa[0]
    assert torch.allclose(kappa, torch.zeros_like(kappa), atol=1e-6)  # pure push
    h = neutral + torch.randn(D)
    seed = (h - sub.mean) @ sub.basis.T
    out, _foot = subspace_inject(
        h, sub, domain, target, seed, along=1.0, onto=0.0, kappa=kappa,
    )
    coords_out = (out - sub.mean) @ sub.basis.T
    q = (h - sub.mean) @ sub.basis.T
    # translate, not collapse: each push axis shifts by its target coord
    assert torch.allclose(coords_out, q + target, atol=1e-4)


def test_synthesized_inject_identity_at_along_zero():
    torch.manual_seed(7)
    D = 12
    u = _unit(torch.randn(D))
    neutral = 5.0 + torch.randn(D)
    synth = synthesize_subspace(
        push=[({0: _row(u)}, {0: torch.tensor([0.8])}, 0.8)],
        ablate=[], neutral_means={0: neutral},
    )
    sub = synth.layers[0]
    domain = CustomDomain(sub.rank)
    h = neutral + 0.3 * u + 0.4 * torch.randn(D)
    seed = (h - sub.mean) @ sub.basis.T
    out, _foot = subspace_inject(
        h, sub, domain, synth.target_coord[0], seed,
        along=0.0, onto=0.0,
    )
    assert torch.allclose(out, h, atol=1e-5)


def _synth_from_subspace(sub: LayerSubspace) -> SynthesizedSubspace:
    rank = sub.rank
    return SynthesizedSubspace(
        layers={0: sub}, target_coord={0: torch.zeros(rank)},
        share={0: 1.0}, kappa={0: torch.zeros(rank)},
    )


def test_empty_synth_is_explicit_noop_shape() -> None:
    synth = SynthesizedSubspace(layers={}, target_coord={}, share={}, kappa={})
    assert synth.layers == {}


def test_synth_rejects_mean_basis_width_mismatch() -> None:
    sub = LayerSubspace.affine(torch.zeros(4), torch.eye(2, 5))
    with pytest.raises(ValueError, match="mean width"):
        _synth_from_subspace(sub)


def test_synth_rejects_wrong_coordinate_normalization_shape() -> None:
    from dataclasses import replace

    sub = replace(
        LayerSubspace.affine(torch.zeros(4), torch.eye(2, 4)),
        coord_offset=torch.zeros(1),
    )
    with pytest.raises(ValueError, match="coord_offset/coord_scale"):
        _synth_from_subspace(sub)


def test_synth_rejects_artifact_only_fields() -> None:
    from dataclasses import replace

    sub = replace(
        LayerSubspace.affine(torch.zeros(4), torch.eye(2, 4)),
        affine_map=torch.eye(2),
    )
    with pytest.raises(ValueError, match="artifact surface fields"):
        _synth_from_subspace(sub)


def test_synth_rejects_nonorthonormal_basis() -> None:
    sub = LayerSubspace.affine(
        torch.zeros(4), torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]),
    )
    with pytest.raises(ValueError, match="orthonormal"):
        _synth_from_subspace(sub)
