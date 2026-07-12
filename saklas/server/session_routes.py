"""Native ``/saklas/v1/sessions`` route group."""

from __future__ import annotations

from dataclasses import replace
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from saklas.server.native_common import resolve_session_id
from saklas.server.session_models import (
    CreateSessionRequest,
    PatchSessionRequest,
    session_info,
)


def register_session_routes(app: FastAPI) -> None:
    """Mount session lifecycle and configuration routes."""
    session = app.state.session

    @app.get("/saklas/v1/sessions")
    def list_sessions():
        return {"sessions": [session_info(
            session, app.state.default_steering, app.state.created_ts,
        )]}

    @app.post("/saklas/v1/sessions")
    def create_session(req: CreateSessionRequest):
        if req.model and req.model != session.model_id:
            logging.getLogger("saklas.api").warning(
                "POST /saklas/v1/sessions requested model=%r but session is %r; "
                "single-session mode, returning existing",
                req.model,
                session.model_id,
            )
        return session_info(
            session, app.state.default_steering, app.state.created_ts,
        )

    @app.get("/saklas/v1/sessions/{session_id}")
    def get_session(session_id: str):
        resolve_session_id(session_id)
        return session_info(
            session, app.state.default_steering, app.state.created_ts,
        )

    @app.delete("/saklas/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        resolve_session_id(session_id)
        logging.getLogger("saklas.api").warning(
            "DELETE /saklas/v1/sessions/%s: single-session mode, no-op",
            session_id,
        )
        return Response(status_code=204)

    @app.patch("/saklas/v1/sessions/{session_id}")
    def patch_session(session_id: str, req: PatchSessionRequest):
        resolve_session_id(session_id)
        config = session.config
        session.config = replace(
            config,
            temperature=req.temperature if req.temperature is not None else config.temperature,
            top_p=req.top_p if req.top_p is not None else config.top_p,
            top_k=req.top_k if req.top_k is not None else config.top_k,
            max_new_tokens=(
                req.max_tokens if req.max_tokens is not None else config.max_new_tokens
            ),
            system_prompt=(
                req.system_prompt if req.system_prompt is not None else config.system_prompt
            ),
            thinking=req.thinking if req.thinking is not None else config.thinking,
        )
        return session_info(
            session, app.state.default_steering, app.state.created_ts,
        )

    @app.post("/saklas/v1/sessions/{session_id}/clear", status_code=204)
    def clear_session(session_id: str):
        resolve_session_id(session_id)
        session.clear_history()
        return Response(status_code=204)

    @app.post("/saklas/v1/sessions/{session_id}/rewind", status_code=204)
    def rewind_session(session_id: str):
        resolve_session_id(session_id)
        if not session.tree.messages_for():
            raise HTTPException(400, "History is empty")
        session.rewind()
        return Response(status_code=204)
