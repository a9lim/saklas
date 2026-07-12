"""Native routes for the session-resident sparse-autoencoder pillar."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import Field

from saklas.core.errors import SaklasError
from saklas.core.loom import InvalidNodeOperationError, UnknownNodeError
from saklas.server.app import acquire_session_lock
from saklas.server.native_common import NativeRequest, resolve_session_id


class SaeLoadRequest(NativeRequest):
    release: str = Field(min_length=1)
    layer: int | None = Field(default=None, ge=0)


class SaeLiveRequest(NativeRequest):
    enabled: bool
    top_k: int = Field(default=8, ge=1, le=100)


class SaeFeatureRequest(NativeRequest):
    id: int = Field(ge=0)


class SaeFeatureMetaRequest(NativeRequest):
    ids: list[int] = Field(min_length=1, max_length=64)


def register_sae_routes(app: FastAPI) -> None:
    session = app.state.session
    app.state.sae_load = {
        "running": False,
        "release": None,
        "message": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
        "info": None,
    }
    app.state.sae_load_task = None

    def _status() -> dict[str, Any]:
        return dict(app.state.sae_load)

    @app.get("/saklas/v1/sessions/{session_id}/sae/releases")
    async def sae_releases(session_id: str):
        resolve_session_id(session_id)
        from saklas.core.sae import list_sae_releases

        try:
            rows = await asyncio.to_thread(list_sae_releases, session.model_id)
        except SaklasError as exc:
            status, message = exc.user_message()
            raise HTTPException(status, message) from exc
        return {"releases": rows}

    @app.post("/saklas/v1/sessions/{session_id}/sae/load", status_code=202)
    async def sae_load(session_id: str, body: SaeLoadRequest):
        resolve_session_id(session_id)
        if app.state.sae_load["running"]:
            raise HTTPException(409, "an SAE load is already running")
        state = app.state.sae_load
        state.update({
            "running": True,
            "release": body.release,
            "message": f"loading {body.release}",
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
            "info": None,
        })

        async def worker() -> None:
            try:
                async with acquire_session_lock(session) as acquired:
                    if not acquired:
                        raise RuntimeError("session locked")
                    info = await asyncio.to_thread(
                        session.load_sae, body.release, layer=body.layer,
                    )
                state["info"] = info
                state["message"] = (
                    f"loaded {body.release} at L{info.get('layer')} "
                    f"({info.get('width')} features)"
                )
            except Exception as exc:  # background status owns translation
                if isinstance(exc, SaklasError):
                    _code, message = exc.user_message()
                else:
                    message = str(exc)
                state["error"] = message
                state["message"] = "load failed"
            finally:
                state["running"] = False
                state["finished_at"] = time.time()

        app.state.sae_load_task = asyncio.create_task(worker())
        return _status()

    @app.get("/saklas/v1/sessions/{session_id}/sae/load")
    def sae_load_status(session_id: str):
        resolve_session_id(session_id)
        return _status()

    @app.delete("/saklas/v1/sessions/{session_id}/sae/load")
    async def sae_unload(session_id: str):
        resolve_session_id(session_id)
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            session.unload_sae()
        return {"loaded": False}

    @app.post("/saklas/v1/sessions/{session_id}/sae/live")
    async def sae_live(session_id: str, body: SaeLiveRequest):
        resolve_session_id(session_id)
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                if body.enabled:
                    state = session.enable_live_sae(top_k=body.top_k)
                    return {"enabled": True, **state}
                session.disable_live_sae()
                return {"enabled": False, "layer": None, "top_k": body.top_k}
            except SaklasError as exc:
                status, message = exc.user_message()
                raise HTTPException(status, message) from exc

    @app.post("/saklas/v1/sessions/{session_id}/sae/feature/validate")
    def sae_feature_validate(session_id: str, body: SaeFeatureRequest):
        resolve_session_id(session_id)
        try:
            return session.validate_sae_feature(body.id)
        except SaklasError as exc:
            status, message = exc.user_message()
            raise HTTPException(status, message) from exc

    @app.post("/saklas/v1/sessions/{session_id}/sae/features/metadata")
    async def sae_features_metadata(session_id: str, body: SaeFeatureMetaRequest):
        """Fetch-and-cache Neuronpedia metadata (label + maxActApprox).

        The dashboard's discovery backfill — called between generations with
        the ids the live top-k surfaced. Network + disk-cache only (no model
        use), so it deliberately does not take the session lock, mirroring
        feature validation.
        """
        resolve_session_id(session_id)
        if any(feature_id < 0 for feature_id in body.ids):
            raise HTTPException(400, "feature ids must be non-negative")
        try:
            features = await asyncio.to_thread(
                session.fetch_sae_feature_meta, body.ids,
            )
        except SaklasError as exc:
            status, message = exc.user_message()
            raise HTTPException(status, message) from exc
        return {"features": features}

    @app.get("/saklas/v1/sessions/{session_id}/sae/token-readout")
    async def sae_token_readout(
        session_id: str,
        node_id: str,
        raw_index: int,
        top_k: int = 8,
        steered: bool = True,
        raw: bool = False,
    ):
        resolve_session_id(session_id)
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                return await asyncio.to_thread(
                    session.sae_token_readout,
                    node_id,
                    raw_index,
                    top_k=top_k,
                    apply_steering=steered,
                    raw=raw,
                )
            except UnknownNodeError as exc:
                raise HTTPException(404, str(exc)) from exc
            except InvalidNodeOperationError as exc:
                raise HTTPException(400, str(exc)) from exc
            except SaklasError as exc:
                status, message = exc.user_message()
                raise HTTPException(status, message) from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
