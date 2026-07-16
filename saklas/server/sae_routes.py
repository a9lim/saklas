"""Native routes for the session-resident sparse-autoencoder pillar."""
from __future__ import annotations

import asyncio
import logging
import math
import re

from fastapi import FastAPI, HTTPException
from pydantic import Field

from saklas.core.errors import SaklasError
from saklas.core.loom import InvalidNodeOperationError, UnknownNodeError
from saklas.server.app import acquire_session_lock
from saklas.server.background_job import (
    BackgroundJob,
    make_progress_hook,
    scrub_job_error,
)
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

    # An SAE artifact operation — a provider load or a local train — runs one at
    # a time; the two jobs guard each other (load XOR train).  Only the train is
    # cooperatively cancellable (a running estimator loop); the provider load has
    # no cancel event and no DELETE-cancel (its DELETE is the unload).
    _SAE_BUSY = "an SAE artifact operation is already running"
    sae_load_job = BackgroundJob(
        app,
        "sae_load",
        {
            "running": False,
            "release": None,
            "message": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
            "info": None,
        },
        busy_message=_SAE_BUSY,
    )
    sae_train_job = BackgroundJob(
        app,
        "sae_train",
        {
            "running": False,
            "name": None,
            "tokens_done": 0,
            "tokens_total": 0,
            "message": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
            "info": None,
        },
        busy_message=_SAE_BUSY,
        cancellable=True,
        not_running_message="no SAE training is running",
    )
    sae_load_job.share_group(sae_train_job)

    async def _stop_sae_train() -> None:
        await sae_train_job.stop()

    app.router.on_shutdown.append(_stop_sae_train)

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
        sae_load_job.refuse_if_busy()
        source = body.release.strip()
        release = (
            source[len("saelens:"):]
            if source.startswith("saelens:")
            else source
        )
        if not release:
            raise HTTPException(400, "SAE source must not be empty")
        state = sae_load_job.state
        sae_load_job.start(
            message=f"loading {source}", release=source, info=None,
        )

        async def _load_job() -> None:
            async with acquire_session_lock(session) as acquired:
                if not acquired:
                    raise RuntimeError("session locked")
                info = await asyncio.to_thread(
                    session.load_sae, release, layer=body.layer,
                )
                await asyncio.to_thread(session.enable_live_sae, top_k=12)
            state["info"] = info
            state["message"] = (
                f"loaded {source} · live at L{info.get('layer')} "
                f"({info.get('width')} features)"
            )

        def _load_on_error(exc: BaseException) -> None:
            # Background status owns translation: a typed error surfaces its
            # safe message, anything else its ``str`` (no server logging here).
            if isinstance(exc, SaklasError):
                _code, message = exc.user_message()
            else:
                message = str(exc)
            state["error"] = message
            state["message"] = "load failed"

        sae_load_job.launch(_load_job, _load_on_error)
        return sae_load_job.status()

    async def _sae_train_job(body: SaeTrainRequest, layer: int) -> None:
        from saklas.io.lens import stream_default_lens_corpus

        state = sae_train_job.state
        n_docs = max(1, math.ceil(body.tokens / body.seq_len))
        state["message"] = f"streaming {n_docs:,} corpus documents…"
        docs, spec = await asyncio.to_thread(
            stream_default_lens_corpus, n_docs,
        )
        on_progress = make_progress_hook(
            state, _TRAIN_PROGRESS_RE,
            done_field="tokens_done", total_field="tokens_total",
        )
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
            cancel_event=sae_train_job.cancel_event,
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

    @app.post("/saklas/v1/sessions/{session_id}/sae/train", status_code=202)
    async def sae_train_start(session_id: str, body: SaeTrainRequest):
        from saklas.core.sae_training import SaeTrainingCancelled

        resolve_session_id(session_id)
        sae_train_job.refuse_if_busy()
        layer = body.layer
        if layer is None:
            layer = round(0.65 * max(len(session.layers) - 1, 0))
        if not 0 <= layer < len(session.layers):
            raise HTTPException(
                400, f"SAE layer {layer} is outside model layers 0..{len(session.layers) - 1}",
            )
        sae_train_job.start(
            message="starting…",
            name=body.name,
            tokens_done=0,
            tokens_total=body.tokens,
            info=None,
        )

        def _on_error(exc: BaseException) -> None:
            scrub_job_error(
                sae_train_job.state, exc,
                cancel_exc=SaeTrainingCancelled,
                op_label="SAE training",
                logger=log,
                failure_message="training failed",
            )

        sae_train_job.launch(lambda: _sae_train_job(body, layer), _on_error)
        return sae_train_job.status()

    @app.get("/saklas/v1/sessions/{session_id}/sae/train")
    async def sae_train_status(session_id: str):
        resolve_session_id(session_id)
        return sae_train_job.status()

    @app.delete("/saklas/v1/sessions/{session_id}/sae/train", status_code=202)
    async def sae_train_cancel(session_id: str):
        resolve_session_id(session_id)
        return sae_train_job.request_cancel()

    @app.get("/saklas/v1/sessions/{session_id}/sae/load")
    def sae_load_status(session_id: str):
        resolve_session_id(session_id)
        return sae_load_job.status()

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
