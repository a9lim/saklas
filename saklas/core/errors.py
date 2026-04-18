"""Shared base class for all saklas-raised exceptions.

Every custom exception defined in saklas re-parents to :class:`SaklasError`
so callers (and the HTTP server) can catch the whole family with a single
``except SaklasError``. Stdlib parents (``ValueError``, ``RuntimeError``,
``KeyError``, ``ImportError``, ...) stay in the MRO so generic
``except ValueError`` sites catch the relevant subclasses too.
"""


class SaklasError(Exception):
    """Base class for all saklas-raised errors."""


class SaeBackendImportError(ImportError, SaklasError):
    """Raised when sae_lens is required but not installed."""


class SaeReleaseNotFoundError(ValueError, SaklasError):
    """Raised when a requested SAELens release does not exist."""


class SaeModelMismatchError(ValueError, SaklasError):
    """Raised when an SAE's base model does not match the saklas model."""


class SaeCoverageError(ValueError, SaklasError):
    """Raised when an SAE release covers zero of the model's layers."""


class AmbiguousVariantError(ValueError, SaklasError):
    """Raised when a :sae selector matches more than one extracted release."""


class UnknownVariantError(KeyError, SaklasError):
    """Raised when a variant selector does not match any on-disk tensor."""
