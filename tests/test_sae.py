"""Tests for the SAE extraction pipeline."""
from __future__ import annotations

import pytest


def test_errors_subclass_saklas_error():
    from saklas.core.errors import (
        SaklasError,
        SaeBackendImportError,
        SaeReleaseNotFoundError,
        SaeModelMismatchError,
        SaeCoverageError,
        AmbiguousVariantError,
        UnknownVariantError,
    )
    for cls in (
        SaeBackendImportError,
        SaeReleaseNotFoundError,
        SaeModelMismatchError,
        SaeCoverageError,
        AmbiguousVariantError,
        UnknownVariantError,
    ):
        assert issubclass(cls, SaklasError)


def test_errors_preserve_stdlib_mro():
    from saklas.core.errors import (
        SaeBackendImportError,
        SaeReleaseNotFoundError,
        SaeModelMismatchError,
        SaeCoverageError,
        AmbiguousVariantError,
        UnknownVariantError,
    )
    assert issubclass(SaeBackendImportError, ImportError)
    assert issubclass(SaeReleaseNotFoundError, ValueError)
    assert issubclass(SaeModelMismatchError, ValueError)
    assert issubclass(SaeCoverageError, ValueError)
    assert issubclass(AmbiguousVariantError, ValueError)
    assert issubclass(UnknownVariantError, KeyError)


def test_sae_backend_protocol_shape():
    from saklas.core.sae import SaeBackend, MockSaeBackend
    # Structural conformance — downstream code type-hints `SaeBackend | None`
    # and we want the mock to pass isinstance checks at runtime.
    assert hasattr(SaeBackend, "encode_layer")
    assert hasattr(SaeBackend, "decode_layer")
    mock = MockSaeBackend(layers=frozenset({0}), d_model=4)
    assert isinstance(mock, SaeBackend)


def test_mock_sae_backend_roundtrip():
    """Identity mock: encode and decode are both identity, d_feature == d_model.

    Used throughout extract/session tests to exercise the SAE branch without
    needing sae_lens or real SAE weights.
    """
    import torch
    from saklas.core.sae import MockSaeBackend

    backend = MockSaeBackend(
        layers=frozenset({4, 8, 12}),
        d_model=16,
        release="mock-release",
    )
    assert backend.layers == frozenset({4, 8, 12})
    assert backend.release == "mock-release"

    h = torch.randn(5, 16)
    f = backend.encode_layer(8, h)
    assert f.shape == (5, 16)
    assert torch.allclose(f, h)

    v_feat = torch.randn(16)
    v_model = backend.decode_layer(8, v_feat)
    assert v_model.shape == (16,)
    assert torch.allclose(v_model, v_feat)


def test_mock_sae_backend_custom_encode_decode():
    """MockSaeBackend lets tests inject per-layer transforms for non-identity cases."""
    import torch
    from saklas.core.sae import MockSaeBackend

    backend = MockSaeBackend(
        layers=frozenset({3}),
        d_model=4,
        d_feature=4,
        encode_fn=lambda idx, h: h * 2.0,
        decode_fn=lambda idx, f: f * 0.5,
    )
    h = torch.ones(2, 4)
    f = backend.encode_layer(3, h)
    assert torch.allclose(f, torch.full((2, 4), 2.0))
    v = backend.decode_layer(3, torch.full((4,), 4.0))
    assert torch.allclose(v, torch.full((4,), 2.0))


def test_mock_sae_backend_passes_layer_idx_to_overrides():
    """Per-layer fns receive the layer index so tests can verify dispatch."""
    import torch
    from saklas.core.sae import MockSaeBackend

    seen: list[int] = []
    backend = MockSaeBackend(
        layers=frozenset({2, 5}),
        d_model=4,
        encode_fn=lambda idx, h: (seen.append(idx) or h),
        decode_fn=lambda idx, f: (seen.append(-idx) or f),
    )
    backend.encode_layer(2, torch.zeros(1, 4))
    backend.decode_layer(5, torch.zeros(4))
    assert seen == [2, -5]


def test_extract_contrastive_sae_subset_layers(monkeypatch):
    """With sae=MockSaeBackend(layers={1,3}), profile covers only those layers."""
    import torch
    from saklas.core import vectors as V
    from saklas.core.sae import MockSaeBackend

    def fake_encode_and_capture(model, tokenizer, text, layers, device):
        torch.manual_seed(hash(text) & 0xFFFF)
        out = {}
        for idx in range(len(layers)):
            base = torch.randn(8)
            sign = 1.0 if "pos" in text else -1.0
            out[idx] = base + sign * (idx + 1) * 0.3
        return out

    monkeypatch.setattr(V, "_encode_and_capture_all", fake_encode_and_capture)

    pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(5)]

    class FakeModel:
        def parameters(self):
            yield torch.zeros(1)
    class FakeTok:
        pass

    layers_list = [object()] * 4
    sae = MockSaeBackend(layers=frozenset({1, 3}), d_model=8)

    profile = V.extract_contrastive(
        FakeModel(), FakeTok(), pairs, layers=layers_list,
        device=torch.device("cpu"),
        sae=sae,
    )
    assert set(profile.keys()) == {1, 3}
    mags = [profile[i].norm().item() for i in profile]
    assert all(m > 0 for m in mags)


def test_extract_contrastive_sae_pca_center_orients_correctly(monkeypatch):
    """pos > neg on the resulting direction, majority-vote orientation."""
    import torch
    from saklas.core import vectors as V
    from saklas.core.sae import MockSaeBackend

    def fake_encode_and_capture(model, tokenizer, text, layers, device):
        out = {}
        for idx in range(len(layers)):
            base = torch.zeros(4)
            base[0] = 1.0 if "pos" in text else -1.0
            out[idx] = base + 0.01 * torch.randn(4)
        return out

    monkeypatch.setattr(V, "_encode_and_capture_all", fake_encode_and_capture)

    pairs = [{"positive": f"pos_{i}", "negative": f"neg_{i}"} for i in range(5)]
    class FakeModel:
        def parameters(self):
            yield torch.zeros(1)
    class FakeTok:
        pass

    layers_list = [object()] * 2
    sae = MockSaeBackend(layers=frozenset({0, 1}), d_model=4)

    profile = V.extract_contrastive(
        FakeModel(), FakeTok(), pairs, layers=layers_list,
        device=torch.device("cpu"),
        sae=sae,
    )
    for idx, vec in profile.items():
        assert vec[0].item() > 0


def test_extract_contrastive_sae_zero_coverage_raises(monkeypatch):
    """An SAE covering no model layers raises SaeCoverageError."""
    import torch
    from saklas.core import vectors as V
    from saklas.core.errors import SaeCoverageError
    from saklas.core.sae import MockSaeBackend

    monkeypatch.setattr(
        V, "_encode_and_capture_all",
        lambda *a, **k: {i: torch.zeros(4) for i in range(2)},
    )

    pairs = [{"positive": "p", "negative": "n"}, {"positive": "p2", "negative": "n2"}]
    class FakeModel:
        def parameters(self):
            yield torch.zeros(1)
    class FakeTok:
        pass

    layers_list = [object()] * 2
    sae = MockSaeBackend(layers=frozenset({5, 7}), d_model=4)

    with pytest.raises(SaeCoverageError):
        V.extract_contrastive(
            FakeModel(), FakeTok(), pairs, layers=layers_list,
            device=torch.device("cpu"),
            sae=sae,
        )
