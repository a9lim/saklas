"""Native vector-probe route group."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from saklas.server import saklas_api as _api
from saklas.server.saklas_api import _probe_info, _resolve_session_id


def register_probe_routes(app: FastAPI) -> None:
    """Mount probe listing + activation routes.

    One-shot text scoring (``POST .../probe`` / ``.../manifold-probe``) was
    removed in 4.0: scoring out of generation context required re-rendering
    arbitrary text in a non-conversational regime, which the conversational
    (A2) capture model retires.  Live per-token scoring during generation
    rides the traits SSE stream and the WS/OpenAI/Ollama reading extensions.
    """
    session = app.state.session

    @app.get("/saklas/v1/sessions/{session_id}/probes")
    def list_probes(session_id: str):
        _resolve_session_id(session, session_id)
        names = sorted(session.probes.keys())
        return {"probes": [_probe_info(session, name) for name in names]}

    @app.get("/saklas/v1/sessions/{session_id}/probes/defaults")
    def list_default_probes(session_id: str):
        _resolve_session_id(session, session_id)
        return {"defaults": _api.load_defaults()}

    @app.post("/saklas/v1/sessions/{session_id}/probes/{name}", status_code=204)
    def activate_probe(session_id: str, name: str):
        _resolve_session_id(session, session_id)
        try:
            session.probe(name)
        except (KeyError, ValueError, FileNotFoundError) as e:
            raise HTTPException(400, f"probe '{name}' not available: {e}")
        return Response(status_code=204)

    @app.delete("/saklas/v1/sessions/{session_id}/probes/{name}", status_code=204)
    def deactivate_probe(session_id: str, name: str):
        _resolve_session_id(session, session_id)
        if name not in session.probes:
            raise HTTPException(404, f"probe '{name}' not active")
        session.unprobe(name)
        return Response(status_code=204)
