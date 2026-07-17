"""Shared base class for all saklas-raised exceptions.

Every custom exception defined in saklas re-parents to :class:`SaklasError`
so callers (and the HTTP server) can catch the whole family with a single
``except SaklasError``. Stdlib parents (``ValueError``, ``RuntimeError``,
``KeyError``, ``ImportError``, ...) stay in the MRO so generic
``except ValueError`` sites catch the relevant subclasses too.

Every subclass returns an HTTP-style status code through
:meth:`SaklasError.user_message`, which the server and CLI consume to translate
exceptions consistently. The
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
    ``min(2, code // 100)``; the HTTP server passes it through.
    """

    def user_message(self) -> tuple[int, str]:
        """Return ``(status_code, formatted_message)`` for user-facing surfaces."""
        return (500, str(self) or self.__class__.__name__)


def is_out_of_memory_error(exc: BaseException) -> bool:
    """Recognize accelerator and CPU allocator OOM spellings."""
    message = str(exc).lower()
    return any(
        needle in message
        for needle in ("out of memory", "can't allocate memory", "cannot allocate memory")
    )


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


class SaeNotLoadedError(RuntimeError, SaklasError):
    """Raised when a live SAE surface is used without a resident release."""

    def user_message(self) -> tuple[int, str]:
        return (404, str(self) or self.__class__.__name__)


class SaeFeatureError(ValueError, SaklasError):
    """Raised when an SAE feature id is malformed or outside the resident width."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class UnsupportedProbeChannelError(ValueError, SaklasError):
    """Raised at steering-composition preflight when a probe gate references
    a channel its instrument family can never produce (e.g.
    ``@when:sae/123:membership`` — SAE readings carry only the activation
    axis).  Distinct from a *supported* channel that is temporarily absent
    this step (prefill, capture unavailable), which stays quietly inactive.
    """

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


class RoleBaselineMismatchWarning(UserWarning):
    """Warns that a role-augmented steering expression mixes a plain term
    in.  The plain term's baseline was the family's standard ``assistant``
    role label; the role-augmented terms substitute a custom role label
    into the chat-template render.  Composing them is supported but the
    plain term's baseline doesn't track the substituted role, so the
    interaction may behave unexpectedly.  The warning fires once per
    mixed-baseline ``steering()`` scope.
    """


class SteeringExprError(ValueError, SaklasError):
    """Raised when a steering expression string cannot be parsed."""

    def __init__(self, msg: str, *, col: int | None = None) -> None:
        self.col = col
        if col is not None:
            msg = f"{msg} (col {col})"
        super().__init__(msg)

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class ManifoldArityError(SteeringExprError):
    """Raised when a ``%`` manifold position has the wrong number of
    coordinates for the manifold's domain.

    The grammar collects the position payload but cannot validate arity —
    it does not know the domain.  ``SteeringManager.add_manifold`` checks
    the coordinate count against the loaded domain's intrinsic dimension
    and raises this when they disagree.  The manifold-surface analogue of
    the vector surface's dedicated selector errors
    (``AmbiguousSelectorError`` / ``AmbiguousVariantError``); subclasses
    ``SteeringExprError`` so existing ``except SteeringExprError`` sites
    still catch it.
    """


class OverlappingManifoldError(SteeringExprError):
    """Raised when curved manifold terms overlap at a shared layer.

    Multiple curved terms may share a layer when their fitted spans are
    near-orthogonal; the affine merged subspace is orthogonalized against
    those spans.  This error identifies the unsupported case where two curved
    spans exceed the overlap tolerance.  It subclasses ``SteeringExprError``
    so existing expression-error handlers still catch it.
    """


class ManifoldNotFoundError(FileNotFoundError, SaklasError):
    """Raised when a manifold folder or its fitted tensor is not found.

    Preserves ``FileNotFoundError`` in the MRO so existing
    ``except FileNotFoundError`` call sites (server, CLI) still catch it.
    """

    def user_message(self) -> tuple[int, str]:
        return (404, str(self) or self.__class__.__name__)


class ManifoldExistsError(FileExistsError, SaklasError):
    """Raised when a manifold tensor already exists and ``force`` is off.

    Preserves ``FileExistsError`` in the MRO so existing
    ``except FileExistsError`` call sites (server, CLI) still catch it.
    """

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)
