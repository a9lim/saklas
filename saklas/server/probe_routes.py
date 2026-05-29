"""Native vector-probe route group."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from saklas.server import saklas_api as _api
from saklas.server.saklas_api import ScoreProbeRequest, _probe_info, _resolve_session_id


def register_probe_routes(app: FastAPI) -> None:
    """Mount probe listing, activation, and one-shot scoring routes."""
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
        return JSONResponse(status_code=204, content=None)

    @app.delete("/saklas/v1/sessions/{session_id}/probes/{name}", status_code=204)
    def deactivate_probe(session_id: str, name: str):
        _resolve_session_id(session, session_id)
        if name not in session.probes:
            raise HTTPException(404, f"probe '{name}' not active")
        session.unprobe(name)
        return JSONResponse(status_code=204, content=None)

    @app.post("/saklas/v1/sessions/{session_id}/probe")
    async def score_probe_oneshot(session_id: str, req: ScoreProbeRequest):
        _resolve_session_id(session, session_id)
        requested = req.probes
        monitor = session._monitor
        if requested:
            missing = [name for name in requested if name not in monitor.probe_names]
            if missing:
                raise HTTPException(400, f"probes not active: {missing}")

        async with session.lock:
            readings = await asyncio.to_thread(
                monitor.measure,
                session._model,
                session._tokenizer,
                session._layers,
                req.text,
            )
        if requested:
            readings = {key: value for key, value in readings.items() if key in requested}
        return {"readings": {key: float(value) for key, value in readings.items()}}
