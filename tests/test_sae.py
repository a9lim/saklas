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
