"""Native routes for the session-resident sparse-autoencoder pillar."""
from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import Field

from saklas.core.errors import SaklasError
from saklas.core.loom import InvalidNodeOperationError, UnknownNodeError
from saklas.server.app import acquire_session_lock
from saklas.server.native_common import NativeRequest, resolve_session_id

log = logging.getLogger(__name__)
_TRAIN_PROGRESS_RE = re.compile(r"trained ([\d,]+)/([\d,]+) tokens")


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


class SaeTrainRequest(NativeRequest):
    name: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,63}$")
    layer: int | None = Field(default=None, ge=0)
    tokens: int = Field(default=1_000_000, ge=1, le=100_000_000)
    seq_len: int = Field(default=128, ge=8, le=4096)
    batch_size: int = Field(default=8, ge=1, le=256)
    width: int | None = Field(default=None, ge=1)
    expansion: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=3e-4, gt=0)
    l1: float = Field(default=1e-3, ge=0)
    dead_threshold: float = Field(default=1e-6, ge=0)
    seed: int = 0
    force: bool = False


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
    app.state.sae_train = {
        "running": False,
        "name": None,
        "tokens_done": 0,
        "tokens_total": 0,
        "message": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
        "info": None,
    }
    app.state.sae_train_task = None
    app.state.sae_train_cancel = None

    async def _stop_sae_train() -> None:
        event = app.state.sae_train_cancel
        task = app.state.sae_train_task
        if event is not None:
            event.set()
        if task is not None and not task.done():
            await task

    app.router.on_shutdown.append(_stop_sae_train)

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

    @app.get("/saklas/v1/sessions/{session_id}/sae/sources")
    async def sae_sources(session_id: str):
        """List Saklas-owned local SAEs and fetched SAELens bindings."""
        resolve_session_id(session_id)
        from saklas.io.sae import list_sae_sources

        rows = await asyncio.to_thread(list_sae_sources, session.model_id)
        return {
            "sources": [
                {key: value for key, value in row.items() if key != "path"}
                for row in rows
            ],
        }

    @app.post("/saklas/v1/sessions/{session_id}/sae/load", status_code=202)
    async def sae_load(session_id: str, body: SaeLoadRequest):
        resolve_session_id(session_id)
        if app.state.sae_load["running"] or app.state.sae_train["running"]:
            raise HTTPException(409, "an SAE artifact operation is already running")
        source = body.release.strip()
        release = (
            source[len("saelens:"):]
            if source.startswith("saelens:")
            else source
        )
        if not release:
            raise HTTPException(400, "SAE source must not be empty")
        state = app.state.sae_load
        state.update({
            "running": True,
            "release": source,
            "message": f"loading {source}",
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
                        session.load_sae, release, layer=body.layer,
                    )
                state["info"] = info
                state["message"] = (
                    f"loaded {source} at L{info.get('layer')} "
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

    def _train_status() -> dict[str, Any]:
        return dict(app.state.sae_train)

    async def _sae_train_job(body: SaeTrainRequest, layer: int) -> None:
        from saklas.core.sae_training import SaeTrainingCancelled
        from saklas.io.lens import stream_default_lens_corpus

        state = app.state.sae_train
        try:
            n_docs = max(1, math.ceil(body.tokens / body.seq_len))
            state["message"] = f"streaming {n_docs:,} corpus documents…"
            docs, spec = await asyncio.to_thread(
                stream_default_lens_corpus, n_docs,
            )

            def on_progress(message: str) -> None:
                match = _TRAIN_PROGRESS_RE.search(message)
                if match is not None:
                    state["tokens_done"] = int(match.group(1).replace(",", ""))
                    state["tokens_total"] = int(match.group(2).replace(",", ""))
                state["message"] = message

            result = await asyncio.to_thread(
                session.train_sae,
                body.name,
                docs,
                layer=layer,
                corpus_spec=spec,
                tokens=body.tokens,
                seq_len=body.seq_len,
                batch_size=body.batch_size,
                d_sae=body.width,
                expansion=body.expansion,
                learning_rate=body.learning_rate,
                l1_coefficient=body.l1,
                dead_feature_threshold=body.dead_threshold,
                seed=body.seed,
                force=body.force,
                on_progress=on_progress,
                cancel_event=app.state.sae_train_cancel,
            )
            state["info"] = result["runtime"]
            state["tokens_done"] = int(result["metrics"]["tokens_trained"])
            state["message"] = f"active: {result['source']}"
            state["error"] = None
            try:
                async with acquire_session_lock(session) as acquired:
                    if acquired:
                        await asyncio.to_thread(session.enable_live_sae, top_k=12)
            except Exception:
                log.exception("could not auto-enable live SAE after training")
        except SaeTrainingCancelled:
            state["message"] = "cancelled"
            state["error"] = None
        except SaklasError as exc:
            _code, text = exc.user_message()
            state["message"] = "training failed"
            state["error"] = text
        except Exception as exc:  # noqa: BLE001 - scrubbed, logged server-side
            log.exception("SAE training failed")
            state["message"] = "training failed"
            state["error"] = f"SAE training failed ({type(exc).__name__})"
        finally:
            state["running"] = False
            state["finished_at"] = time.time()

    @app.post("/saklas/v1/sessions/{session_id}/sae/train", status_code=202)
    async def sae_train_start(session_id: str, body: SaeTrainRequest):
        resolve_session_id(session_id)
        if app.state.sae_train["running"] or app.state.sae_load["running"]:
            raise HTTPException(409, "an SAE artifact operation is already running")
        layer = body.layer
        if layer is None:
            layer = round(0.65 * max(len(session.layers) - 1, 0))
        if not 0 <= layer < len(session.layers):
            raise HTTPException(
                400, f"SAE layer {layer} is outside model layers 0..{len(session.layers) - 1}",
            )
        app.state.sae_train.update({
            "running": True,
            "name": body.name,
            "tokens_done": 0,
            "tokens_total": body.tokens,
            "message": "starting…",
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
            "info": None,
        })
        app.state.sae_train_cancel = threading.Event()
        app.state.sae_train_task = asyncio.create_task(
            _sae_train_job(body, layer),
        )
        return _train_status()

    @app.get("/saklas/v1/sessions/{session_id}/sae/train")
    async def sae_train_status(session_id: str):
        resolve_session_id(session_id)
        return _train_status()

    @app.delete("/saklas/v1/sessions/{session_id}/sae/train", status_code=202)
    async def sae_train_cancel(session_id: str):
        resolve_session_id(session_id)
        if not app.state.sae_train["running"] or app.state.sae_train_cancel is None:
            raise HTTPException(409, "no SAE training is running")
        app.state.sae_train_cancel.set()
        app.state.sae_train["message"] = "cancelling…"
        return _train_status()

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
