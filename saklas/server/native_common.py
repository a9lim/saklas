"""Shared primitives for the native ``/saklas/v1`` route tree."""

from __future__ import annotations

from fastapi import HTTPException

from saklas.core.session import SaklasSession
from saklas.io.paths import safe_model_id


SINGLE_SESSION_ID = "default"


def session_aliases(session: SaklasSession) -> list[str]:
    aliases = [SINGLE_SESSION_ID]
    model_id = str(session.model_id)
    if model_id not in aliases:
        aliases.append(model_id)
    safe_id = safe_model_id(model_id)
    if safe_id not in aliases:
        aliases.append(safe_id)
    return aliases


def resolve_session_id(session: SaklasSession, session_id: str) -> None:
    """Raise 404 if ``session_id`` doesn't map to the single live session."""
    if session_id in session_aliases(session):
        return
    raise HTTPException(
        status_code=404,
        detail=f"session '{session_id}' not found",
    )


# Backcompat aliases for the old ``saklas_api.py`` import surface.
_SINGLE_SESSION_ID = SINGLE_SESSION_ID
_session_aliases = session_aliases
_resolve_session_id = resolve_session_id
