"""Shared primitives for the native ``/saklas/v1`` route tree."""

from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

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
