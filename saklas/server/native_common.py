"""Shared primitives for the native ``/saklas/v1`` route tree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from saklas.core.session import SaklasSession

SINGLE_SESSION_ID = "default"


class NativeRequest(BaseModel):
    """Strict base for the current native API request contract."""

    model_config = ConfigDict(extra="forbid")


def resolve_session_id(session_id: str) -> None:
    """Raise 404 if ``session_id`` doesn't map to the single live session."""
    if session_id == SINGLE_SESSION_ID:
        return
    raise HTTPException(
        status_code=404,
        detail=f"session '{session_id}' not found",
    )


def refuse_if_busy(session: "SaklasSession") -> None:
    """Raise 409 when the engine gen-lock is held.

    ``session.lock`` (the asyncio HTTP serializer) orders native mutations
    against each other and the JSON fit path, but an SSE fit or extract whose
    request was cancelled leaves its worker thread — and the ``gen_lock`` it
    holds — alive past the cancel.  A non-blocking probe of ``gen_lock``
    refuses the mutation while that thread runs.  Shared by the manifold
    mutating routes and the profile bake route; nothing about it is
    manifold-specific.
    """
    if not session.gen_lock.acquire(blocking=False):
        raise HTTPException(
            409, "a model operation is in flight; retry shortly",
        )
    session.gen_lock.release()


def extraction_error_frame(exc: Exception) -> dict[str, Any] | None:
    """SSE ``error``-frame body for a typed extraction-pipeline failure.

    ``POST /extract`` and ``POST /manifolds/{ns}/{name}/fit`` drive the one
    :class:`~saklas.core.extraction.ManifoldExtractionPipeline`, so they share
    the same two safe-to-surface failures: a concurrent run holding the engine
    (conflict) and an authoring-grade ``ValueError`` — the RBF poisedness
    failure among them, which gets its own code so the client can offer the
    "spread your nodes" hint.  ``None`` for anything else, which routes the
    exception to the shared catch-all scrubber.
    """
    from saklas.core.session import ConcurrentExtractionError

    if isinstance(exc, ConcurrentExtractionError):
        return {"message": str(exc), "code": "Conflict"}
    if isinstance(exc, ValueError):
        return {
            "message": str(exc),
            "code": (
                "PoisednessError"
                if "poisedness" in str(exc).lower()
                else type(exc).__name__
            ),
        }
    return None


def extraction_json_errors() -> tuple[tuple[Any, int], ...]:
    """JSON-branch status mapping matching :func:`extraction_error_frame`.

    Conflict first: ``ConcurrentExtractionError`` is a ``RuntimeError`` and
    ``ManifoldFormatError`` is a ``ValueError``, so the ordered table lands
    both on the status their SSE counterparts report.
    """
    from saklas.core.session import ConcurrentExtractionError

    return ((ConcurrentExtractionError, 409), (ValueError, 400))
