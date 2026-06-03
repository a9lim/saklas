"""Mahalanobis whitener tests: cosine + LEACE-style projection.

Synthetic small-N small-D inputs throughout; the math invariants we
check (Σ→I reduces to Euclidean, LEACE erasure is exact, ridge inverse
matches direct computation) are dimension-independent and don't require
loading a real model.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from saklas.core.mahalanobis import (
    DEFAULT_RIDGE_SCALE,
    LayerWhitener,
    WhitenerError,
)
from saklas.core.profile import Profile
from saklas.core.vectors import project_profile


# ---------------------------------------------------------------- helpers ---

def _make_acts(
    seed: int,
    *,
    n: int = 40,
    d: int = 16,
    cov_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Synthetic ``(n, d)`` neutral activations.

    When ``cov_scale`` is None, draws from N(0, I).  When provided, scales
    each axis independently — gives us a known-anisotropic covariance to
    sanity-check Mahalanobis behavior against.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    if cov_scale is not None:
        X = X * cov_scale.reshape(1, d)
    return X.to(dtype=torch.float32)


def _build_whitener(
    *,
    layers: tuple[int, ...] = (0, 1),
    seed: int = 17,
    n: int = 40,
    d: int = 16,
    cov_scale: torch.Tensor | None = None,
    ridge_scale: float = 1.0,
) -> LayerWhitener:
    acts = {L: _make_acts(seed + L, n=n, d=d, cov_scale=cov_scale)
            for L in layers}
    means = {L: torch.zeros(d) for L in layers}  # synthetic, mean already 0
    return LayerWhitener.from_neutral_activations(
        acts, means, ridge_scale=ridge_scale,
    )


# ------------------------------------------------------------ construction ---

class TestLayerWhitenerConstruction:
    def test_from_neutral_activations_covers_shared_layers(self):
        acts = {0: _make_acts(0), 1: _make_acts(1), 7: _make_acts(7)}
        means = {0: torch.zeros(16), 1: torch.zeros(16)}  # 7 absent
        w = LayerWhitener.from_neutral_activations(acts, means)
        assert w.layers == {0, 1}
        assert 7 not in w
        assert 0 in w

    def test_nonfinite_layer_is_excluded(self):
        """A non-finite layer (legacy fp16-overflowed cache surfacing as
        ±inf) is skipped, leaving it uncovered — so the all-or-nothing
        ``covers_all`` gate degrades the whole consumer to Euclidean rather
        than carrying ``nan`` factors into the hot path.
        """
        good = _make_acts(0)
        bad = _make_acts(1).clone()
        bad[0, 0] = float("inf")  # one overflowed channel poisons the layer
        acts = {0: good, 1: bad}
        means = {0: torch.zeros(16), 1: torch.zeros(16)}
        w = LayerWhitener.from_neutral_activations(acts, means)
        assert w.layers == {0}            # bad layer dropped
        assert 1 not in w
        assert not w.covers_all([0, 1])   # all-or-nothing trips to False
        assert w.covers_all([0])
        # The surviving layer's factors are finite.
        X, K, lam = w.woodbury_factors(0, device=torch.device("cpu"), dtype=torch.float32)
        assert torch.isfinite(X).all() and torch.isfinite(K).all()
        assert math.isfinite(lam)

    def test_no_shared_layers_raises(self):
        acts = {0: _make_acts(0)}
        means = {1: torch.zeros(16)}
        with pytest.raises(WhitenerError):
            LayerWhitener.from_neutral_activations(acts, means)

    def test_invalid_ridge_scale_raises(self):
        acts = {0: _make_acts(0)}
        means = {0: torch.zeros(16)}
        with pytest.raises(WhitenerError):
            LayerWhitener.from_neutral_activations(
                acts, means, ridge_scale=0.0,
            )
        with pytest.raises(WhitenerError):
            LayerWhitener.from_neutral_activations(
                acts, means, ridge_scale=-1.0,
            )

    def test_dim_mismatch_raises(self):
        acts = {0: _make_acts(0, d=16)}
        means = {0: torch.zeros(8)}  # wrong dim
        with pytest.raises(WhitenerError):
            LayerWhitener.from_neutral_activations(acts, means)

    def test_default_ridge_scale_is_one(self):
        assert DEFAULT_RIDGE_SCALE == 1.0


# ------------------------------------------------------------- core math ---

class TestApplyInv:
    def test_matches_direct_inverse(self):
        """Σ_reg^{-1} v via Woodbury equals direct ``torch.linalg.inv``."""
        d, n = 12, 20
        X = _make_acts(42, n=n, d=d)  # already centered
        means = {0: torch.zeros(d)}
        w = LayerWhitener.from_neutral_activations({0: X}, means)
        # Reconstruct λ the same way the class does so the comparison is
        # like-for-like (we don't expose ``ridge`` to bypass that).
        lam = w.ridge(0)
        Sigma = X.transpose(0, 1) @ X / n + lam * torch.eye(d)
        Sigma_inv_direct = torch.linalg.inv(Sigma)
        v = torch.randn(d, generator=torch.Generator().manual_seed(99))
        woodbury = w.apply_inv(0, v)
        direct = Sigma_inv_direct @ v
        assert torch.allclose(woodbury, direct, atol=1e-4, rtol=1e-4)

    def test_apply_inv_preserves_dtype(self):
        w = _build_whitener()
        v_fp16 = torch.randn(16, dtype=torch.float16)
        out = w.apply_inv(0, v_fp16)
        assert out.dtype == torch.float16

    def test_apply_inv_unknown_layer_raises(self):
        w = _build_whitener(layers=(0,))
        with pytest.raises(WhitenerError):
            w.apply_inv(7, torch.zeros(16))

    def test_apply_inv_dim_mismatch_raises(self):
        w = _build_whitener(layers=(0,), d=16)
        with pytest.raises(WhitenerError):
            w.apply_inv(0, torch.zeros(8))


class TestCoversAll:
    def test_true_for_covered_set_and_empty(self):
        w = _build_whitener(layers=(0, 1))
        assert w.covers_all([0, 1]) is True
        assert w.covers_all([]) is True  # vacuous

    def test_false_for_partial_or_missing(self):
        w = _build_whitener(layers=(0, 1))
        assert w.covers_all([0, 1, 2]) is False
        assert w.covers_all([7]) is False


class TestSubspaceGram:
    """``subspace_gram(layer, B) = B Σ⁻¹ Bᵀ`` — the reduced-space inverse
    covariance backing the whitened manifold share."""

    def test_matches_direct(self):
        """Woodbury form equals the dense ``B inv(Σ) Bᵀ``."""
        d, n, R = 12, 30, 4
        X = _make_acts(42, n=n, d=d)
        w = LayerWhitener.from_neutral_activations({0: X}, {0: torch.zeros(d)})
        lam = w.ridge(0)
        Sigma = X.transpose(0, 1) @ X / n + lam * torch.eye(d)
        Sigma_inv = torch.linalg.inv(Sigma)
        B = torch.randn(R, d, generator=torch.Generator().manual_seed(5))
        gram = w.subspace_gram(0, B)
        direct = B @ Sigma_inv @ B.transpose(0, 1)
        assert torch.allclose(gram, direct, atol=1e-4, rtol=1e-4)

    def test_symmetric(self):
        w = _build_whitener(layers=(0,), d=16)
        B = torch.randn(5, 16, generator=torch.Generator().manual_seed(2))
        gram = w.subspace_gram(0, B)
        assert torch.allclose(gram, gram.transpose(0, 1), atol=1e-6)

    def test_isotropic_orthonormal_near_scaled_identity(self):
        """Σ ≈ I + orthonormal basis → ``B Σ⁻¹ Bᵀ ≈ (1/(1+λ)) I_R``:
        near-diagonal, equal positive diagonal, ~zero off-diagonal.  This
        is the property that makes the whitened manifold share reduce to
        the Euclidean ``‖coords‖_F`` spread under isotropic covariance.
        """
        d, n, R = 16, 600, 5
        X = _make_acts(3, n=n, d=d)  # N(0, I)
        w = LayerWhitener.from_neutral_activations(
            {0: X}, {0: torch.zeros(d)}, ridge_scale=0.05,
        )
        Q, _ = torch.linalg.qr(
            torch.randn(d, R, generator=torch.Generator().manual_seed(1))
        )
        B = Q.transpose(0, 1)  # (R, d), orthonormal rows
        gram = w.subspace_gram(0, B)
        diag = torch.diag(gram)
        off = gram - torch.diag(diag)
        assert off.abs().max() < 0.1
        assert (diag > 0).all()
        assert (diag.max() - diag.min()) < 0.15  # equal scale on every axis

    def test_unknown_layer_raises(self):
        w = _build_whitener(layers=(0,))
        with pytest.raises(WhitenerError):
            w.subspace_gram(7, torch.zeros(3, 16))

    def test_dim_mismatch_raises(self):
        w = _build_whitener(layers=(0,), d=16)
        with pytest.raises(WhitenerError):
            w.subspace_gram(0, torch.zeros(3, 8))


class TestMahalanobisCosine:
    def test_isotropic_matches_euclidean(self):
        """When activations are isotropic, Mahalanobis cosine ≈ Euclidean.

        Σ ≈ I (sample covariance of isotropic Gaussian noise) means
        ``<u, v>_M ≈ <u, v> / (1 + λ)`` modulo finite-sample bias.  The
        cosine ratio cancels the scalar, so the *cosine* itself ≈ plain
        cosine within finite-sample tolerance.
        """
        d, n = 16, 200  # n >> d → small finite-sample bias
        w = _build_whitener(layers=(0,), n=n, d=d, seed=1)
        u = torch.randn(d, generator=torch.Generator().manual_seed(7))
        v = torch.randn(d, generator=torch.Generator().manual_seed(8))
        m_cos = w.mahalanobis_cosine(0, u, v)
        e_cos = torch.dot(u, v) / (u.norm() * v.norm())
        # Tolerance reflects the rank-(n-1) sample covariance not being
        # exactly identity.  The two should agree to a few percent.
        assert abs(m_cos - float(e_cos)) < 0.05

    def test_anisotropic_diverges_from_euclidean(self):
        """Strongly anisotropic Σ → Mahalanobis ≠ Euclidean cosine.

        Sanity check that the metric actually does something.  We pick u
        and v that are aligned in a high-variance axis but disagree in a
        low-variance axis: Mahalanobis upweights the disagreement, so
        ``m_cos < e_cos``.
        """
        d = 16
        scale = torch.ones(d)
        scale[0] = 10.0  # axis 0 has high variance
        w = _build_whitener(
            layers=(0,), n=200, d=d, seed=2,
            cov_scale=scale, ridge_scale=0.05,
        )
        u = torch.zeros(d)
        u[0] = 1.0
        u[1] = 1.0
        v = torch.zeros(d)
        v[0] = 1.0
        v[1] = -1.0
        m_cos = w.mahalanobis_cosine(0, u, v)
        e_cos = float(torch.dot(u, v) / (u.norm() * v.norm()))
        # In Euclidean, e_cos = 0 (orthogonal axes 1 vs -1 cancel axis 0).
        # Mahalanobis downweights axis 0 (high variance), so the relative
        # weight of the disagreeing axis 1 grows → cosine becomes more
        # negative.
        assert e_cos == pytest.approx(0.0, abs=1e-6)
        assert m_cos < -0.4

    def test_self_cosine_is_one(self):
        w = _build_whitener(layers=(0,), n=100, d=16)
        v = torch.randn(16, generator=torch.Generator().manual_seed(3))
        assert w.mahalanobis_cosine(0, v, v) == pytest.approx(1.0, abs=1e-5)

    def test_zero_vector_returns_zero(self):
        w = _build_whitener(layers=(0,), n=100, d=16)
        z = torch.zeros(16)
        v = torch.randn(16)
        assert w.mahalanobis_cosine(0, z, v) == 0.0
        assert w.mahalanobis_cosine(0, v, z) == 0.0

    def test_norm_is_nonnegative(self):
        w = _build_whitener(layers=(0,), n=100, d=16)
        v = torch.randn(16, generator=torch.Generator().manual_seed(11))
        n = w.mahalanobis_norm(0, v)
        assert n >= 0.0


# ---------------------------------------------------- LEACE projection ---

class TestLeaceProject:
    def test_pipe_orthogonalizes_in_mahalanobis_metric(self):
        """``base | onto`` is exactly Mahalanobis-orthogonal to ``onto``.

        Defining property of LEACE: after projection, the inner product
        ``<P base, onto>_M`` is zero.
        """
        d = 16
        scale = torch.ones(d)
        scale[2] = 5.0
        w = _build_whitener(
            layers=(0,), n=200, d=d, seed=4,
            cov_scale=scale, ridge_scale=0.1,
        )
        base = torch.randn(d, generator=torch.Generator().manual_seed(13))
        onto = torch.randn(d, generator=torch.Generator().manual_seed(14))
        proj = w.leace_project(0, base, onto, "|")
        # <proj, onto>_M ≈ 0 by construction.
        m_dot = w.mahalanobis_dot(0, proj, onto)
        assert abs(m_dot) < 1e-4

    def test_tilde_is_complement_of_pipe(self):
        """``base ~ onto`` + ``base | onto`` reconstructs ``base``."""
        w = _build_whitener(layers=(0,), n=100, d=16)
        base = torch.randn(16, generator=torch.Generator().manual_seed(21))
        onto = torch.randn(16, generator=torch.Generator().manual_seed(22))
        kept = w.leace_project(0, base, onto, "~")
        rest = w.leace_project(0, base, onto, "|")
        assert torch.allclose(kept + rest, base, atol=1e-5)

    def test_leace_reduces_to_euclidean_when_sigma_is_identity(self):
        """λ=very-small + isotropic acts → LEACE ≈ Euclidean projection."""
        d, n = 16, 500
        # Big n, isotropic acts, small ridge.  Σ → I tightly.
        w = _build_whitener(
            layers=(0,), n=n, d=d, seed=5, ridge_scale=0.001,
        )
        base = torch.randn(d, generator=torch.Generator().manual_seed(31))
        onto = torch.randn(d, generator=torch.Generator().manual_seed(32))
        leace = w.leace_project(0, base, onto, "|")
        # Plain Euclidean projection.
        coef = torch.dot(base, onto) / torch.dot(onto, onto)
        euc = base - coef * onto
        # Should match within finite-sample tolerance.
        assert torch.allclose(leace, euc, atol=0.05, rtol=0.05)

    def test_zero_onto_pipe_passes_through(self):
        w = _build_whitener(layers=(0,), n=100, d=16)
        base = torch.randn(16, generator=torch.Generator().manual_seed(41))
        zero = torch.zeros(16)
        out = w.leace_project(0, base, zero, "|")
        assert torch.allclose(out, base, atol=1e-6)

    def test_zero_onto_tilde_returns_zero(self):
        w = _build_whitener(layers=(0,), n=100, d=16)
        base = torch.randn(16)
        zero = torch.zeros(16)
        out = w.leace_project(0, base, zero, "~")
        assert torch.allclose(out, torch.zeros(16))

    def test_unknown_operator_raises(self):
        w = _build_whitener(layers=(0,), n=100, d=16)
        with pytest.raises(ValueError):
            w.leace_project(0, torch.zeros(16), torch.ones(16), "@")

    def test_unknown_layer_raises(self):
        w = _build_whitener(layers=(0,), n=100, d=16)
        with pytest.raises(WhitenerError):
            w.leace_project(7, torch.zeros(16), torch.ones(16), "|")


# ---------------------------------------- Profile.cosine_similarity wiring ---

class TestProfileCosineWithWhitener:
    def test_whitener_per_layer_uses_mahalanobis(self):
        d = 8
        w = _build_whitener(layers=(0, 1), n=100, d=d, seed=51)
        a = Profile({0: torch.randn(d), 1: torch.randn(d)})
        b = Profile({0: torch.randn(d), 1: torch.randn(d)})
        per_layer: Any = a.cosine_similarity(b, per_layer=True, whitener=w)
        # Each layer's value should equal the standalone call.
        for L in (0, 1):
            assert per_layer[L] == pytest.approx(
                w.mahalanobis_cosine(L, a[L].float(), b[L].float()),
                abs=1e-5,
            )

    def test_whitener_aggregate_in_unit_interval(self):
        """Aggregate cosine stays in [-1, 1] under Mahalanobis weighting."""
        d = 8
        w = _build_whitener(layers=(0, 1, 2), n=80, d=d, seed=61)
        torch.manual_seed(0)
        a = Profile({L: torch.randn(d) for L in (0, 1, 2)})
        b = Profile({L: torch.randn(d) for L in (0, 1, 2)})
        agg: Any = a.cosine_similarity(b, whitener=w)
        assert -1.0 <= agg <= 1.0

    def test_uncovered_layer_raises(self):
        """A whitener missing a shared layer is a hard error — Mahalanobis is
        mandatory (no per-layer Euclidean fallback)."""
        d = 8
        w = _build_whitener(layers=(0,), n=80, d=d, seed=71)
        a = Profile({0: torch.randn(d), 5: torch.tensor([1.0] * d)})
        b = Profile({0: torch.randn(d), 5: torch.tensor([1.0] * d)})
        with pytest.raises(WhitenerError, match="whitener"):
            a.cosine_similarity(b, per_layer=True, whitener=w)


# --------------------------------------------------- project_profile wiring ---

class TestProjectProfileLeace:
    def test_whitener_swaps_to_leace(self):
        d = 12
        scale = torch.ones(d)
        scale[0] = 8.0  # high variance on axis 0
        w = _build_whitener(
            layers=(0,), n=300, d=d, seed=81,
            cov_scale=scale, ridge_scale=0.1,
        )
        base = torch.randn(d, generator=torch.Generator().manual_seed(82))
        onto = torch.randn(d, generator=torch.Generator().manual_seed(83))
        out = project_profile({0: base}, {0: onto}, "|", whitener=w)
        # Output should be Mahalanobis-orthogonal to onto.
        m_dot = w.mahalanobis_dot(0, out[0], onto)
        assert abs(m_dot) < 1e-4

    def test_uncovered_layer_raises(self):
        d = 4
        w = _build_whitener(layers=(0,), n=80, d=d, seed=91)
        base = {0: torch.randn(d), 5: torch.tensor([1.0, 1.0, 0.0, 0.0])}
        onto = {0: torch.randn(d), 5: torch.tensor([1.0, 0.0, 0.0, 0.0])}
        # Layer 5 not covered → Mahalanobis-only hard error (no fallback).
        with pytest.raises(WhitenerError, match="whitener"):
            project_profile(base, onto, "|", whitener=w)


# ------------------------------------------------------- repr / dunder ---

class TestRepr:
    def test_repr_includes_layer_count(self):
        w = _build_whitener(layers=(0, 1, 2), n=50, d=8)
        s = repr(w)
        assert "layers=3" in s
        assert "N=50" in s


# --------------------------------------- Mahalanobis bake at extract time ---

