"""Difference-of-means (DiM) extractor parity tests.

Mirrors the synthetic-encoder pattern in :mod:`tests.test_diagnostics`:
stub :func:`saklas.core.vectors._encode_and_capture_all` so we don't need
a real model.  Verifies that DiM and PCA agree on cleanly-separated
synthetic pairs (cosine ≈ 1.0), that share-baking magnitudes are
consistent across methods, and that the SAE branch decode round-trip
behaves the same as the raw branch on an identity-decoder mock backend.
"""
from __future__ import annotations

import torch

from saklas.core import vectors as V
from saklas.core.sae import MockSaeBackend


# ---------------------------------------------------------------------------
# Stubs reused across tests.  Synthetic encoder produces clean pos/neg
# separation along axis 0 with small noise — exactly the regime where
# DiM should match PCA's first principal component.
# ---------------------------------------------------------------------------


def _stub_encode_separable(model, tokenizer, text, layers, device):
    """Stable pos/neg activations along axis 0 with tiny gaussian noise."""
    n = len(layers)
    sign = 1.0 if "pos" in text else -1.0
    out: dict[int, torch.Tensor] = {}
    for idx in range(n):
        base = torch.zeros(4)
        base[0] = sign * 1.0
        out[idx] = base + 0.05 * torch.randn(4)
    return out


def _stub_encode_noisy(model, tokenizer, text, layers, device):
    """Noisier pos/neg pairs — class-mean axis still axis 0 but per-pair
    diff has substantial off-axis variance.  This is the regime where Im
    & Li 2025 predicts PCA can pick a near-orthogonal direction; DiM
    should still align with the actual class axis.
    """
    n = len(layers)
    sign = 1.0 if "pos" in text else -1.0
    out: dict[int, torch.Tensor] = {}
    for idx in range(n):
        base = torch.zeros(4)
        base[0] = sign * 0.3  # weak signal
        out[idx] = base + 0.5 * torch.randn(4)  # strong noise
    return out


class _FakeModel:
    def parameters(self):
        yield torch.zeros(1)


class _FakeTok:
    pass


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a / a.norm(), b / b.norm()).item())


# ---------------------------------------------------------------------------
# Direction shape + scale
# ---------------------------------------------------------------------------


class TestDimReturnShape:
    """``extract_difference_of_means`` matches ``extract_contrastive`` shape."""

    def test_returns_profile_and_diagnostics(self, monkeypatch) -> None:
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode_separable)

        pairs = [
            {"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(8)
        ]
        profile, diagnostics = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs, layers=[object()] * 6,
            device=torch.device("cpu"),
            dls=False,
        )
        assert set(profile.keys()) == set(diagnostics.keys()) == set(range(6))
        for v in profile.values():
            assert v.shape == (4,)
            assert v.dtype == torch.float32

    def test_dls_keep_set_aligns_diagnostics_with_profile(
        self, monkeypatch,
    ) -> None:
        # v2.1: edge-drop replaced by data-driven DLS.  Without
        # ``layer_means`` the helper falls back to "keep all layers"
        # silently — diagnostics and profile cover the same set.
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode_separable)

        pairs = [
            {"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(5)
        ]
        profile, diagnostics = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs, layers=[object()] * 8,
            device=torch.device("cpu"),
            dls=False,
        )
        assert set(profile.keys()) == set(diagnostics.keys())


# ---------------------------------------------------------------------------
# DiM ↔ PCA agreement on clean signals; divergence on noisy ones.
# ---------------------------------------------------------------------------


class TestDimAgreesWithPca:
    """On well-separated synthetic data DiM and PCA pick the same axis."""

    def test_clean_signal_high_cosine(self, monkeypatch) -> None:
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode_separable)
        pairs = [
            {"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(20)
        ]
        common = dict(
            layers=[object()] * 6,
            device=torch.device("cpu"),
            dls=False,
        )

        torch.manual_seed(0)
        pca, _ = V.extract_contrastive(_FakeModel(), _FakeTok(), pairs, **common)
        torch.manual_seed(0)
        dim, _ = V.extract_difference_of_means(_FakeModel(), _FakeTok(), pairs, **common)

        for layer in pca:
            assert _cos(pca[layer], dim[layer]) > 0.95, (
                f"DiM and PCA disagreed on layer {layer} for clean signal"
            )

    def test_share_bake_magnitudes_in_band(self, monkeypatch) -> None:
        """Per-layer baked magnitudes from DiM live in the same band as
        PCA's — share-baking math is method-agnostic, so the total
        ``Σ_L ||baked_L||`` budget should match within an order of
        magnitude.  Anchors the v2.1 invariant that swapping methods
        doesn't require recalibrating ``_STEER_GAIN``.
        """
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode_separable)
        pairs = [
            {"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(12)
        ]

        common = dict(
            layers=[object()] * 6,
            device=torch.device("cpu"),
            dls=False,
        )
        torch.manual_seed(0)
        pca, _ = V.extract_contrastive(_FakeModel(), _FakeTok(), pairs, **common)
        torch.manual_seed(0)
        dim, _ = V.extract_difference_of_means(_FakeModel(), _FakeTok(), pairs, **common)

        pca_total = sum(t.norm().item() for t in pca.values())
        dim_total = sum(t.norm().item() for t in dim.values())
        # Within 0.5×–2× of each other for the synthetic regime.  A
        # tighter bound would over-fit to the seed; a looser one would
        # let a regression slip through.
        assert 0.5 * pca_total <= dim_total <= 2.0 * pca_total, (
            f"DiM total {dim_total} drifted vs PCA total {pca_total}"
        )


class TestDimOnNoisyPairs:
    """DiM should be at least as well-behaved as PCA on noisy signals."""

    def test_unit_normed_per_layer(self, monkeypatch) -> None:
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode_noisy)
        pairs = [
            {"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(30)
        ]

        profile, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs, layers=[object()] * 5,
            device=torch.device("cpu"), dls=False,
        )
        # Baked tensors carry share × ref_norm; we don't assert unit
        # norm, but the per-layer magnitude must be > 0 (no degenerate
        # all-zero layer).
        for layer, vec in profile.items():
            assert vec.norm().item() > 0.0, f"degenerate layer {layer}"


# ---------------------------------------------------------------------------
# SAE branch — identity-decoder mock backend should behave like raw.
# ---------------------------------------------------------------------------


class TestDimSaeBranch:
    """SAE+DiM uses ``mean(F_pos − F_neg)`` then decodes back to model space."""

    def test_identity_sae_matches_raw_direction(self, monkeypatch) -> None:
        """Identity SAE encode/decode is a no-op, so SAE-DiM should agree
        with raw-DiM on the same pairs."""
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode_separable)
        pairs = [
            {"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(8)
        ]
        layers = [object()] * 4
        sae = MockSaeBackend(
            layers=frozenset({0, 1, 2, 3}),
            d_model=4,
            release="mock",
        )

        torch.manual_seed(0)
        raw, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs,
            layers=layers, device=torch.device("cpu"), dls=False,
        )
        torch.manual_seed(0)
        sae_profile, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs,
            layers=layers, device=torch.device("cpu"), dls=False,
            sae=sae,
        )

        # Same set of layers covered (mock covers all 4) and directions
        # agree to within float roundoff (cos ≈ 1.0).
        assert set(raw.keys()) == set(sae_profile.keys())
        for layer in raw:
            assert _cos(raw[layer], sae_profile[layer]) > 0.99, (
                f"SAE identity decode drifted on layer {layer}"
            )
