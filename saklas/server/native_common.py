"""Shared primitives for the native ``/saklas/v1`` route tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
