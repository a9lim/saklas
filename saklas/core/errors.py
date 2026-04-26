"""Shared base class for all saklas-raised exceptions.

Every custom exception defined in saklas re-parents to :class:`SaklasError`
so callers (and the HTTP server) can catch the whole family with a single
``except SaklasError``. Stdlib parents (``ValueError``, ``RuntimeError``,
``KeyError``, ``ImportError``, ...) stay in the MRO so generic
``except ValueError`` sites catch the relevant subclasses too.

Every subclass returns an HTTP-style status code through
:meth:`SaklasError.user_message`, which the three user-facing surfaces
(server, CLI, TUI) consume to translate exceptions consistently.  The
default ``(500, str(self))`` matches today's behaviour for any subclass
that doesn't override; subclasses lift the status (and optionally rewrite
the message) by overriding the method.
"""


class SaklasError(Exception):
    """Base class for all saklas-raised errors.

    Subclasses override :meth:`user_message` to provide an HTTP-style
    status code (``400`` bad input, ``404`` not found, ``409`` conflict,
    ``422`` semantic-but-syntactically-valid, ``500`` server error,
    ``502`` upstream).  The CLI maps the status to an exit code via
    ``min(2, code // 100)``; the TUI ignores the status and only uses
    the message; the HTTP server passes it through.
    """

    def user_message(self) -> tuple[int, str]:
        """Return ``(status_code, formatted_message)`` for user-facing surfaces."""
        return (500, str(self) or self.__class__.__name__)


class SaeBackendImportError(ImportError, SaklasError):
    """Raised when sae_lens is required but not installed."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class SaeReleaseNotFoundError(ValueError, SaklasError):
    """Raised when a requested SAELens release does not exist."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class SaeModelMismatchError(ValueError, SaklasError):
    """Raised when an SAE's base model does not match the saklas model."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class SaeCoverageError(ValueError, SaklasError):
    """Raised when an SAE release covers zero of the model's layers."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class AmbiguousVariantError(ValueError, SaklasError):
    """Raised when a :sae selector matches more than one extracted release."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class UnknownVariantError(KeyError, SaklasError):
    """Raised when a variant selector does not match any on-disk tensor."""

    def user_message(self) -> tuple[int, str]:
        # ``str(KeyError("x"))`` is ``"'x'"`` (repr-quoted); use ``args[0]``
        # when present so the user sees the original message.
        msg = self.args[0] if self.args else self.__class__.__name__
        return (404, str(msg))


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

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)
