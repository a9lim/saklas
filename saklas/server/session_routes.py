"""Native ``/saklas/v1/sessions`` route group."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any, cast
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from saklas.server.saklas_api import (
    CreateSessionRequest,
    PatchSessionRequest,
    _resolve_session_id,
    _session_info,
)


def register_session_routes(app: FastAPI) -> None:
    """Mount session lifecycle and configuration routes."""
    session = app.state.session

    @app.get("/saklas/v1/sessions")
    def list_sessions():
        return {"sessions": [_session_info(session, app.state.default_steering)]}

    @app.post("/saklas/v1/sessions")
    def create_session(req: CreateSessionRequest):
        if req.model and req.model != session.model_id:
            logging.getLogger("saklas.api").warning(
                "POST /saklas/v1/sessions requested model=%r but session is %r; "
                "single-session mode, returning existing",
                req.model,
                session.model_id,
            )
        return _session_info(session, app.state.default_steering)

    @app.get("/saklas/v1/sessions/{session_id}")
    def get_session(session_id: str):
        _resolve_session_id(session, session_id)
        return _session_info(session, app.state.default_steering)

    @app.delete("/saklas/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        _resolve_session_id(session, session_id)
        logging.getLogger("saklas.api").warning(
            "DELETE /saklas/v1/sessions/%s: single-session mode, no-op",
            session_id,
        )
        return Response(status_code=204)

    @app.patch("/saklas/v1/sessions/{session_id}")
    def patch_session(session_id: str, req: PatchSessionRequest):
        _resolve_session_id(session, session_id)
        overrides: dict[str, Any] = {}
        if req.temperature is not None:
            overrides["temperature"] = req.temperature
        if req.top_p is not None:
            overrides["top_p"] = req.top_p
        if req.top_k is not None:
            overrides["top_k"] = req.top_k
        if req.max_tokens is not None:
            overrides["max_new_tokens"] = req.max_tokens
        if req.system_prompt is not None:
            overrides["system_prompt"] = req.system_prompt
        if overrides:
            if is_dataclass(session.config):
                session.config = replace(cast(Any, session.config), **overrides)
            else:
                for key, value in overrides.items():
                    setattr(session.config, key, value)
        return _session_info(session, app.state.default_steering)

    @app.post("/saklas/v1/sessions/{session_id}/clear", status_code=204)
    def clear_session(session_id: str):
        _resolve_session_id(session, session_id)
        session.clear_history()
        return Response(status_code=204)

    @app.post("/saklas/v1/sessions/{session_id}/rewind", status_code=204)
    def rewind_session(session_id: str):
        _resolve_session_id(session, session_id)
        if not session.history:
            raise HTTPException(400, "History is empty")
        session.rewind()
        return Response(status_code=204)
