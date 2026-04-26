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


class StaleSidecarError(ValueError, SaklasError):
    """Raised when an extracted tensor's recorded ``statements_sha256``
    disagrees with the live ``statements.json`` on disk.

    Hand-editing ``statements.json`` after extraction silently invalidates
    the baked tensor: the contrastive PCA was run against different pairs
    than the file now contains.  This used to log a warning and proceed;
    the fail-loud contract makes the staleness an explicit, fixable
    situation.

    Set ``SAKLAS_ALLOW_STALE=1`` to escape-hatch the check (advanced
    workflows where stale loads are deliberate, e.g. bisecting a corpus
    edit).  The remediation in the message names the concrete
    ``saklas pack refresh`` invocation that fixes the drift.
    """
