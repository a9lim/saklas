"""Mahalanobis whitener tests: cosine + LEACE-style projection.

Synthetic small-N small-D inputs throughout; the math invariants we
check (Σ→I reduces to Euclidean, LEACE erasure is exact, ridge inverse
matches direct computation) are dimension-independent and don't require
loading a real model.
"""

from __future__ import annotations

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
    def test_whitener_none_matches_existing_behavior(self):
        a = Profile({0: torch.tensor([1.0, 0.0, 0.0]),
                     1: torch.tensor([0.0, 1.0, 0.0])})
        b = Profile({0: torch.tensor([1.0, 0.0, 0.0]),
                     1: torch.tensor([1.0, 0.0, 0.0])})
        out = a.cosine_similarity(b, per_layer=True, whitener=None)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.0)

    def test_whitener_per_layer_uses_mahalanobis(self):
        d = 8
        w = _build_whitener(layers=(0, 1), n=100, d=d, seed=51)
        a = Profile({0: torch.randn(d), 1: torch.randn(d)})
        b = Profile({0: torch.randn(d), 1: torch.randn(d)})
        per_layer = a.cosine_similarity(b, per_layer=True, whitener=w)
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
        agg = a.cosine_similarity(b, whitener=w)
        assert -1.0 <= agg <= 1.0

    def test_whitener_falls_back_for_uncovered_layers(self):
        """Layer absent from whitener falls back to Euclidean for that layer."""
        d = 8
        w = _build_whitener(layers=(0,), n=80, d=d, seed=71)
        a = Profile({0: torch.randn(d), 5: torch.tensor([1.0] * d)})
        b = Profile({0: torch.randn(d), 5: torch.tensor([1.0] * d)})
        per_layer = a.cosine_similarity(b, per_layer=True, whitener=w)
        # Layer 5 is uncovered; identical vectors should still cosine-1.
        assert per_layer[5] == pytest.approx(1.0)


# --------------------------------------------------- project_profile wiring ---

class TestProjectProfileLeace:
    def test_no_whitener_keeps_euclidean_semantics(self):
        base = {0: torch.tensor([1.0, 1.0])}
        onto = {0: torch.tensor([1.0, 0.0])}
        out_eu = project_profile(base, onto, "|")
        # Plain Gram-Schmidt: drops axis 0.
        assert torch.allclose(out_eu[0], torch.tensor([0.0, 1.0]), atol=1e-6)

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

    def test_uncovered_layer_falls_back_to_euclidean(self):
        d = 4
        w = _build_whitener(layers=(0,), n=80, d=d, seed=91)
        base = {0: torch.randn(d), 5: torch.tensor([1.0, 1.0, 0.0, 0.0])}
        onto = {0: torch.randn(d), 5: torch.tensor([1.0, 0.0, 0.0, 0.0])}
        out = project_profile(base, onto, "|", whitener=w)
        # Layer 5 not covered → Euclidean Gram-Schmidt result.
        assert torch.allclose(out[5], torch.tensor([0.0, 1.0, 0.0, 0.0]), atol=1e-6)


# ------------------------------------------------------- repr / dunder ---

class TestRepr:
    def test_repr_includes_layer_count(self):
        w = _build_whitener(layers=(0, 1, 2), n=50, d=8)
        s = repr(w)
        assert "layers=3" in s
        assert "N=50" in s


# --------------------------------------- Mahalanobis bake at extract time ---

class TestExtractDifferenceOfMeansWithWhitener:
    """``extract_difference_of_means(whitener=...)`` does Mahalanobis-flavored
    share allocation while keeping per-layer direction parametrization
    (unit_2 × ref_norm) unchanged.

    Key invariant we verify: hook share at layer L (= ||baked_L|| / Σ
    ||baked||) equals ``||mean_diff_L||_M / Σ ||mean_diff_L'||_M``.  This
    is the algebraic consequence of using ``score = ||m||_M / ref_norm``
    — same shape as the existing Euclidean ``score = ||m||_2 / ref_norm``
    where the analogous identity is ``share = ||m||_2 / Σ ||m||_2``.
    """

    @staticmethod
    def _stub_separable_with_seed(seed: int):
        """Encoder stub that produces deterministic pos/neg pairs.

        Per-call generator keyed on ``(seed, text)`` so two extraction
        runs with the same pair list see byte-identical activations.
        Without this, a closed-over ``torch.Generator`` would be in a
        different state on the second run, breaking cross-method
        comparison tests.
        """
        import hashlib

        def _stub(model, tokenizer, text, layers, device):
            # Deterministic per-text seed — md5 of (seed, text) gives a
            # 64-bit int with no Python hash-randomization sensitivity.
            digest = hashlib.md5(f"{seed}_{text}".encode()).hexdigest()[:16]
            g = torch.Generator().manual_seed(int(digest, 16))
            n = len(layers)
            sign = 1.0 if "pos" in text else -1.0
            out: dict[int, torch.Tensor] = {}
            for idx in range(n):
                base = torch.zeros(8)
                base[0] = sign * (1.0 + 0.5 * idx)  # signal grows with layer
                noise = torch.randn(8, generator=g) * 0.3
                out[idx] = base + noise
            return out
        return _stub

    def test_isotropic_whitener_close_to_euclidean(self, monkeypatch):
        """Isotropic Σ → Mahalanobis ≈ Euclidean (within finite-sample bias)."""
        from saklas.core import vectors as V

        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", self._stub_separable_with_seed(7))

        pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(15)]
        common = dict(layers=[object()] * 6, device=torch.device("cpu"), dls=False)

        # Build whitener with isotropic activations (Σ ≈ I): synthetic
        # neutrals are pure N(0, I).
        d = 8
        acts = {L: _make_acts(L * 13 + 1, n=300, d=d) for L in range(6)}
        means = {L: torch.zeros(d) for L in range(6)}
        w = LayerWhitener.from_neutral_activations(acts, means, ridge_scale=0.1)

        torch.manual_seed(0)
        eu_profile, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs, **common,
        )
        torch.manual_seed(0)
        m_profile, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs, whitener=w, **common,
        )

        # Each layer's direction (Euclidean unit) should be close.
        for L in eu_profile:
            cos = float(torch.dot(
                eu_profile[L] / eu_profile[L].norm(),
                m_profile[L] / m_profile[L].norm(),
            ).item())
            assert cos > 0.99, f"isotropic Σ should match Euclidean at layer {L}"

        # Per-layer baked magnitudes should be in the same ballpark when
        # Σ ≈ I — finite-sample drift is real but bounded.
        for L in eu_profile:
            ratio = m_profile[L].norm().item() / max(eu_profile[L].norm().item(), 1e-8)
            assert 0.5 < ratio < 2.0, (
                f"layer {L}: Mahalanobis mag {m_profile[L].norm()} drifted from "
                f"Euclidean mag {eu_profile[L].norm()} (ratio={ratio:.3f})"
            )

    def test_hook_share_invariant_under_anisotropic_sigma(self, monkeypatch):
        """Hook share = ||m_L||_M / Σ ||m_L'||_M (the algebraic invariant)."""
        from saklas.core import vectors as V

        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", self._stub_separable_with_seed(11))

        pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(20)]
        layers = [object()] * 5
        common = dict(layers=layers, device=torch.device("cpu"), dls=False)
        d = 8

        # Anisotropic Σ — axis 0 (the concept axis) has high variance at
        # layers 0-1, low variance at layers 3-4.  Mahalanobis should
        # downweight layers where the concept lives in high-variance
        # directions.
        acts = {}
        for L in range(5):
            scale = torch.ones(d)
            if L < 2:
                scale[0] = 5.0  # high-variance concept axis: penalized
            else:
                scale[0] = 0.5  # low-variance concept axis: rewarded
            acts[L] = _make_acts(L * 17 + 3, n=200, d=d, cov_scale=scale)
        means = {L: torch.zeros(d) for L in range(5)}
        w = LayerWhitener.from_neutral_activations(acts, means, ridge_scale=0.05)

        torch.manual_seed(0)
        m_profile, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs, whitener=w, **common,
        )

        # Rebuild the mean_diffs the extractor saw, so we can check the
        # algebraic invariant ourselves.
        torch.manual_seed(0)
        # Replay the encoder stub to recover per-layer mean diffs.
        stub = self._stub_separable_with_seed(11)
        per_layer_diffs = {L: [] for L in range(5)}
        for pair in pairs:
            pos = stub(None, None, pair["positive"], layers, torch.device("cpu"))
            neg = stub(None, None, pair["negative"], layers, torch.device("cpu"))
            for L in range(5):
                per_layer_diffs[L].append(pos[L] - neg[L])
        mean_diffs = {
            L: torch.stack(per_layer_diffs[L]).mean(dim=0) for L in range(5)
        }
        m_norms = {L: w.mahalanobis_norm(L, mean_diffs[L]) for L in range(5)}
        total_m = sum(m_norms.values())

        # Hook share at layer L = ||baked_L||_2 / Σ ||baked||_2.  Should
        # equal ||m_L||_M / Σ ||m_L'||_M (the invariant we proved).
        baked_norms = {L: m_profile[L].norm().item() for L in m_profile}
        total_baked = sum(baked_norms.values())
        for L in m_profile:
            hook_share = baked_norms[L] / total_baked
            mahalanobis_share = m_norms[L] / total_m
            assert abs(hook_share - mahalanobis_share) < 0.05, (
                f"layer {L}: hook share {hook_share:.4f} != Mahalanobis "
                f"share {mahalanobis_share:.4f} (invariant violated)"
            )

    def test_single_pair_branch_accepts_whitener(self, monkeypatch):
        """``n_pairs == 1`` path runs without error when whitener provided."""
        from saklas.core import vectors as V

        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", self._stub_separable_with_seed(0))

        pairs = [{"positive": "pos_0", "negative": "neg_0"}]
        d = 8
        acts = {L: _make_acts(L, n=120, d=d) for L in range(4)}
        means = {L: torch.zeros(d) for L in range(4)}
        w = LayerWhitener.from_neutral_activations(acts, means)

        profile, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs, layers=[object()] * 4,
            device=torch.device("cpu"), dls=False, whitener=w,
        )
        # All layers retained, magnitudes positive.
        assert set(profile) == {0, 1, 2, 3}
        for L in profile:
            assert profile[L].norm().item() > 0.0

    def test_uncovered_layer_falls_back_to_euclidean(self, monkeypatch):
        """Layer not covered by the whitener uses Euclidean score for that layer."""
        from saklas.core import vectors as V

        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", self._stub_separable_with_seed(2))
        pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(10)]
        d = 8
        # Whitener only covers layers {0, 1}; extraction uses 4 layers.
        acts = {L: _make_acts(L, n=120, d=d) for L in (0, 1)}
        means = {L: torch.zeros(d) for L in (0, 1)}
        w = LayerWhitener.from_neutral_activations(acts, means)

        profile, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs, layers=[object()] * 4,
            device=torch.device("cpu"), dls=False, whitener=w,
        )
        # All 4 layers in profile; layers 2,3 used Euclidean fallback.
        assert set(profile) == {0, 1, 2, 3}
        for L in profile:
            assert profile[L].norm().item() > 0.0


class _FakeModel:
    """Bare-bones model stub for extractor tests; mirrors test_dim_extraction.py."""

    def parameters(self):
        yield torch.zeros(1)


class _FakeTok:
    pass
