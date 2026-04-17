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
    from saklas.core.sae import SaeBackend
    # Protocol exists and names the expected methods/attrs
    assert hasattr(SaeBackend, "encode_layer")
    assert hasattr(SaeBackend, "decode_layer")


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
