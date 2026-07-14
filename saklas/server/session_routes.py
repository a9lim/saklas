"""Native ``/saklas/v1/sessions`` route group."""

from __future__ import annotations

from dataclasses import replace
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from saklas.server.app import acquire_session_lock
from saklas.server.native_common import resolve_session_id
from saklas.server.session_models import (
    CreateSessionRequest,
    PatchSessionRequest,
    ValidateSteeringRequest,
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
            # ``null`` explicitly disables top-k; an omitted field preserves
            # the current value.  Pydantic's fields-set is the only way to
            # distinguish those two valid PATCH intents.
            top_k=(
                req.top_k
                if "top_k" in req.model_fields_set
                else config.top_k
            ),
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

    @app.post("/saklas/v1/sessions/{session_id}/steering/validate")
    async def validate_steering(session_id: str, req: ValidateSteeringRequest):
        """Parse, resolve, and dry-install a dashboard-authored expression."""
        resolve_session_id(session_id)
        expression = req.expression.strip()
        if not expression:
            return {"valid": True, "expression": "", "error": None}
        from saklas.core.errors import SaklasError
        from saklas.core.steering_expr import format_expr, parse_expr

        try:
            parsed = parse_expr(expression)
            # Dry installation briefly touches the shared hook/profile state;
            # serialize it with generation and other model use just like the
            # actual decode path. A validation submitted mid-generation waits
            # for that turn instead of racing its live steering hooks.
            async with acquire_session_lock(session) as acquired:
                if not acquired:
                    raise HTTPException(503, "session locked")
                # Entering the normal context resolves selectors, projections,
                # variants, SAE/J-lens atoms, and hook composition without
                # decoding a token. It is the same path generation uses,
                # followed by an immediate rollback, so a green validation
                # cannot hide a rack-time failure.
                with session.steering(parsed):
                    pass
        except SaklasError as exc:
            # Validation failures are expected form results, not failed network
            # requests.  Keeping them in-band lets the dashboard present the
            # user-facing explanation without a raw HTTP envelope or a noisy
            # browser-console resource error.
            _status, message = exc.user_message()
            return {
                "valid": False,
                "expression": expression,
                "error": message,
            }
        return {
            "valid": True,
            "expression": format_expr(parsed),
            "error": None,
        }

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
