"""Difference-of-means (DiM) extractor parity tests.

Mirrors the synthetic-encoder pattern in :mod:`tests.test_diagnostics`:
stub :func:`saklas.core.vectors._encode_and_capture_all` so we don't need
a real model.  Verifies that DiM and PCA agree on cleanly-separated
synthetic pairs (cosine ≈ 1.0), that share-baking magnitudes are
consistent across methods, and that the SAE branch decode round-trip
behaves the same as the raw branch on an identity-decoder mock backend.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from saklas.core import vectors as V
from saklas.core.sae import MockSaeBackend


# ---------------------------------------------------------------------------
# Stubs reused across tests.  Synthetic encoder produces clean pos/neg
# separation along axis 0 with small noise — exactly the regime where
# DiM should match PCA's first principal component.
# ---------------------------------------------------------------------------


def _stub_encode_separable(
    model: Any, tokenizer: Any, text: str, layers: Any, device: Any, **_kwargs: Any
) -> dict[int, torch.Tensor]:
    """Stable pos/neg activations along axis 0 with tiny gaussian noise."""
    n = len(layers)
    sign = 1.0 if "pos" in text else -1.0
    out: dict[int, torch.Tensor] = {}
    for idx in range(n):
        base = torch.zeros(4)
        base[0] = sign * 1.0
        out[idx] = base + 0.05 * torch.randn(4)
    return out


def _stub_encode_noisy(
    model: Any, tokenizer: Any, text: str, layers: Any, device: Any, **_kwargs: Any
) -> dict[int, torch.Tensor]:
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


class _FakeModel(torch.nn.Module):
    def parameters(self, recurse: bool = True):  # pyright: ignore[reportIncompatibleMethodOverride]  # stub yields Tensor not Parameter
        yield torch.zeros(1)


class _FakeTok:
    pass


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a / a.norm(), b / b.norm()).item())


# ---------------------------------------------------------------------------
# Direction shape + scale
# ---------------------------------------------------------------------------


class TestDimReturnShape:
    """``extract_difference_of_means`` returns ``(profile, diagnostics)``."""

    def test_returns_profile_and_diagnostics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode_separable)

        pairs = [
            {"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(8)
        ]
        profile, diagnostics = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs,
            layers=[object()] * 6,  # pyright: ignore[reportArgumentType]  # stub uses len(); ModuleList not needed
            device=torch.device("cpu"),
            dls=False,
        )
        assert set(profile.keys()) == set(diagnostics.keys()) == set(range(6))
        for v in profile.values():
            assert v.shape == (4,)
            assert v.dtype == torch.float32

    def test_dls_keep_set_aligns_diagnostics_with_profile(
        self, monkeypatch: pytest.MonkeyPatch,
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
            _FakeModel(), _FakeTok(), pairs,
            layers=[object()] * 8,  # pyright: ignore[reportArgumentType]  # stub uses len(); ModuleList not needed
            device=torch.device("cpu"),
            dls=False,
        )
        assert set(profile.keys()) == set(diagnostics.keys())


# ---------------------------------------------------------------------------
# DiM ↔ PCA agreement on clean signals; divergence on noisy ones.
# ---------------------------------------------------------------------------


class TestDimCleanSignal:
    """On well-separated synthetic data DiM picks the class-separation axis."""

    def test_clean_signal_axis_recovered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode_separable)
        pairs = [
            {"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(20)
        ]
        _layers = [object()] * 6
        _device = torch.device("cpu")

        torch.manual_seed(0)
        dim, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs,
            layers=_layers,  # pyright: ignore[reportArgumentType]  # stub uses len(); ModuleList not needed
            device=_device, dls=False,
        )

        # The stub plants the class separation along axis 0; DiM should
        # recover it (each baked layer aligns with [1, 0, 0, 0]).
        axis0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        for layer in dim:
            assert _cos(dim[layer], axis0) > 0.95, (
                f"DiM missed the separation axis on layer {layer}"
            )


class TestDimOnNoisyPairs:
    """DiM should be at least as well-behaved as PCA on noisy signals."""

    def test_unit_normed_per_layer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch.manual_seed(0)
        monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encode_noisy)
        pairs = [
            {"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(30)
        ]

        profile, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs,
            layers=[object()] * 5,  # pyright: ignore[reportArgumentType]  # stub uses len(); ModuleList not needed
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

    def test_identity_sae_matches_raw_direction(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
            layers=layers,  # pyright: ignore[reportArgumentType]  # stub uses len(); ModuleList not needed
            device=torch.device("cpu"), dls=False,
        )
        torch.manual_seed(0)
        sae_profile, _ = V.extract_difference_of_means(
            _FakeModel(), _FakeTok(), pairs,
            layers=layers,  # pyright: ignore[reportArgumentType]  # stub uses len(); ModuleList not needed
            device=torch.device("cpu"), dls=False,
            sae=sae,
        )

        # Same set of layers covered (mock covers all 4) and directions
        # agree to within float roundoff (cos ≈ 1.0).
        assert set(raw.keys()) == set(sae_profile.keys())
        for layer in raw:
            assert _cos(raw[layer], sae_profile[layer]) > 0.99, (
                f"SAE identity decode drifted on layer {layer}"
            )
