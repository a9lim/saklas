"""Shared primitives for the native ``/saklas/v1`` route tree."""

from __future__ import annotations

from fastapi import HTTPException

from saklas.core.session import SaklasSession


SINGLE_SESSION_ID = "default"


def resolve_session_id(session: SaklasSession, session_id: str) -> None:
    """Raise 404 if ``session_id`` doesn't map to the single live session."""
    if session_id == SINGLE_SESSION_ID:
        return
    if session_id == session.model_id:
        return
    raise HTTPException(
        status_code=404,
        detail=f"session '{session_id}' not found",
    )


# Backcompat aliases for the old ``saklas_api.py`` import surface.
_SINGLE_SESSION_ID = SINGLE_SESSION_ID
_resolve_session_id = resolve_session_id
