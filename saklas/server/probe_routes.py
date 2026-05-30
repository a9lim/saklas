"""Native vector-probe route group."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from saklas.server import saklas_api as _api
from saklas.server.saklas_api import ScoreProbeRequest, _probe_info, _resolve_session_id


class ScoreManifoldRequest(BaseModel):
    """Body for ``POST /sessions/{id}/manifold-probe``.

    The manifold-side counterpart to :class:`ScoreProbeRequest`:
    one-shot text scoring against attached manifold probes.  ``names``
    restricts the scored subset (defaults to every attached probe).
    """

    text: str
    names: list[str] | None = None


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

    @app.post("/saklas/v1/sessions/{session_id}/manifold-probe")
    async def score_manifold_oneshot(session_id: str, req: ScoreManifoldRequest):
        """One-shot manifold scoring over arbitrary text, no generation.

        The read-side manifold counterpart to ``POST .../probe``: runs
        :meth:`SaklasSession.measure_manifold` (a single forward pass +
        per-layer aggregate) under the session lock and returns each
        attached manifold probe's :class:`ManifoldAggregate` as JSON.
        ``names`` restricts the scored subset; unknown names 400.
        """
        _resolve_session_id(session, session_id)
        requested = req.names
        if requested:
            try:
                attached = session.manifold_monitor.probe_names
            except Exception:
                attached = []
            missing = [name for name in requested if name not in attached]
            if missing:
                raise HTTPException(400, f"manifold probes not attached: {missing}")

        async with session.lock:
            readings = await asyncio.to_thread(
                session.measure_manifold, req.text, names=requested,
            )
        return {
            "readings": {
                name: aggregate.to_dict()
                for name, aggregate in readings.items()
            },
        }
