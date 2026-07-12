"""Shared primitives for the native ``/saklas/v1`` route tree."""

from __future__ import annotations

from fastapi import HTTPException

SINGLE_SESSION_ID = "default"


def resolve_session_id(session_id: str) -> None:
    """Raise 404 if ``session_id`` doesn't map to the single live session."""
    if session_id == SINGLE_SESSION_ID:
        return
    raise HTTPException(
        status_code=404,
        detail=f"session '{session_id}' not found",
    )
