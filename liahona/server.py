"""OpenAI-compatible API server backed by LiahonaSession."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from liahona.probes_bootstrap import load_defaults
from liahona.session import ConcurrentGenerationError, LiahonaSession


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class SteerParams(BaseModel):
    alphas: dict[str, float] = Field(default_factory=dict)
    orthogonalize: bool = False
    thinking: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    steer: SteerParams | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    steer: SteerParams | None = None


class ExtractRequest(BaseModel):
    name: str
    source: str | dict[str, list] | None = None
    baseline: str | None = None
    alpha: float = 0.0
    auto_register: bool = Field(True, alias="register")

    model_config = {"populate_by_name": True}


class LoadVectorRequest(BaseModel):
    name: str
    path: str
    alpha: float = 0.0


class PatchSessionRequest(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None


class ActivateProbeRequest(BaseModel):
    profile_path: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id() -> str:
    return f"liahona-{uuid.uuid4().hex[:12]}"


def _error(status: int, message: str, error_type: str = "error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "code": status}},
    )


def _probe_reading_dict(session: LiahonaSession) -> dict[str, Any]:
    readings = session.build_readings()
    out: dict[str, Any] = {}
    for name, r in readings.items():
        out[name] = {"mean": r.mean, "std": r.std, "min": r.min, "max": r.max}
    return out


@contextmanager
def _gen_config_override(session: LiahonaSession, temperature, top_p, max_tokens):
    """Temporarily override generation config."""
    orig = (session.config.temperature, session.config.top_p, session.config.max_new_tokens)
    try:
        if temperature is not None:
            session.config.temperature = temperature
        if top_p is not None:
            session.config.top_p = top_p
        if max_tokens is not None:
            session.config.max_new_tokens = max_tokens
        yield
    finally:
        session.config.temperature, session.config.top_p, session.config.max_new_tokens = orig


def _resolve_alphas(
    steer_params: SteerParams | None, default_alphas: dict[str, float],
) -> dict[str, float] | None:
    merged = dict(default_alphas)
    if steer_params:
        merged.update(steer_params.alphas)
    # Drop zero-alpha entries
    merged = {k: v for k, v in merged.items() if v != 0.0}
    return merged or None


def _profile_top_layers(profile: dict, n: int = 5) -> list[tuple[int, float]]:
    """Return top-n profile layers sorted by score descending."""
    return sorted(((idx, score) for idx, (_vec, score) in profile.items()),
                  key=lambda x: x[1], reverse=True)[:n]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(session: LiahonaSession, default_alphas: dict[str, float] | None = None,
               cors_origins: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="liahona", description="OpenAI-compatible API with activation steering")
    app.state.session = session
    app.state.default_alphas = default_alphas or {}
    app.state.created_ts = int(time.time())
    # Protects config mutations for non-streaming routes.  Streaming routes
    # rely on the session's 409 concurrent-generation guard instead, because
    # the generator is consumed by Starlette outside the route coroutine.
    app.state.gen_lock = asyncio.Lock()

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    session: LiahonaSession = app.state.session

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": session.model_id,
                    "object": "model",
                    "created": app.state.created_ts,
                    "owned_by": "local",
                }
            ],
        }

    @app.get("/v1/models/{model_id:path}")
    def get_model(model_id: str):
        if model_id != session.model_id:
            raise HTTPException(404, f"Model '{model_id}' not found")
        return {
            "id": session.model_id,
            "object": "model",
            "created": app.state.created_ts,
            "owned_by": "local",
        }

    # -----------------------------------------------------------------------
    # Chat completions
    # -----------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        alphas = _resolve_alphas(req.steer, app.state.default_alphas)
        ortho = req.steer.orthogonalize if req.steer else False
        think = req.steer.thinking if req.steer else False
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        rid = _make_id()
        model_id = session.model_id

        if req.stream:
            def _chat_delta(event):
                d: dict[str, str] = {}
                if event.thinking:
                    d["reasoning_content"] = event.text
                else:
                    d["content"] = event.text
                return {"delta": d}

            stream_iter = session.generate_stream(messages, alphas=alphas, orthogonalize=ortho, thinking=think)
            return StreamingResponse(
                _stream_generation(stream_iter, rid, model_id,
                                   "chat.completion.chunk", _chat_delta, {"delta": {}},
                                   req.temperature, req.top_p, req.max_tokens),
                media_type="text/event-stream",
            )
        async with app.state.gen_lock:
            with _gen_config_override(session, req.temperature, req.top_p, req.max_tokens):
                try:
                    result = session.generate(messages, alphas=alphas, orthogonalize=ortho, thinking=think)
                except ConcurrentGenerationError as e:
                    return _error(409, str(e), "conflict")

        return {
            "id": rid,
            "object": "chat.completion",
            "created": app.state.created_ts,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": result.token_count,
                "total_tokens": result.token_count,
            },
            "probe_readings": _probe_reading_dict(session),
        }

    async def _stream_generation(stream_iter, rid, model_id, object_type, format_delta,
                                 empty_delta, temperature, top_p, max_tokens):
        """Shared SSE generator for chat and completion streaming."""
        with _gen_config_override(session, temperature, top_p, max_tokens):
            try:
                for event in stream_iter:
                    chunk = {
                        "id": rid,
                        "object": object_type,
                        "created": app.state.created_ts,
                        "model": model_id,
                        "choices": [{"index": 0, **format_delta(event), "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            except ConcurrentGenerationError as e:
                err = {"error": {"message": str(e), "type": "conflict", "code": 409}}
                yield f"data: {json.dumps(err)}\n\n"
                return

            final = {
                "id": rid,
                "object": object_type,
                "created": app.state.created_ts,
                "model": model_id,
                "choices": [{"index": 0, **empty_delta, "finish_reason": "stop"}],
                "probe_readings": _probe_reading_dict(session),
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

    # -----------------------------------------------------------------------
    # Text completions
    # -----------------------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        alphas = _resolve_alphas(req.steer, app.state.default_alphas)
        ortho = req.steer.orthogonalize if req.steer else False
        think = req.steer.thinking if req.steer else False
        rid = _make_id()
        model_id = session.model_id

        if req.stream:
            stream_iter = session.generate_stream(req.prompt, alphas=alphas, orthogonalize=ortho, thinking=think, raw=True)
            return StreamingResponse(
                _stream_generation(stream_iter, rid, model_id,
                                   "text_completion", lambda e: {"text": e.text}, {"text": ""},
                                   req.temperature, req.top_p, req.max_tokens),
                media_type="text/event-stream",
            )
        async with app.state.gen_lock:
            with _gen_config_override(session, req.temperature, req.top_p, req.max_tokens):
                try:
                    result = session.generate(req.prompt, alphas=alphas, orthogonalize=ortho, thinking=think, raw=True)
                except ConcurrentGenerationError as e:
                    return _error(409, str(e), "conflict")

        return {
            "id": rid,
            "object": "text_completion",
            "created": app.state.created_ts,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "text": result.text,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": result.token_count,
                "total_tokens": result.token_count,
            },
            "probe_readings": _probe_reading_dict(session),
        }

    # -----------------------------------------------------------------------
    # Vector management
    # -----------------------------------------------------------------------

    @app.get("/v1/liahona/vectors")
    def list_vectors():
        vectors = session.vectors
        out: dict[str, Any] = {}
        for name, profile in vectors.items():
            layers = sorted(profile.keys())
            scored = _profile_top_layers(profile)
            top = [{"layer": idx, "score": round(s, 4)} for idx, s in scored]
            out[name] = {
                "layers": layers,
                "top_layers": top,
                "default_alpha": app.state.default_alphas.get(name, 0.0),
            }
        return {"vectors": out}

    @app.post("/v1/liahona/vectors/extract")
    async def extract_vector(req: ExtractRequest, request: Request):
        accept = request.headers.get("accept", "application/json")

        source: Any = req.source if req.source is not None else req.name

        # If source is a dict with pairs, convert to list of tuples
        if isinstance(source, dict) and "pairs" in source:
            source = [(p, n) for p, n in source["pairs"]]

        if "text/event-stream" in accept:
            return StreamingResponse(
                _stream_extract(session, app, req.name, source, req.baseline, req.alpha, req.auto_register),
                media_type="text/event-stream",
            )

        # Blocking JSON response
        progress_msgs: list[str] = []
        try:
            profile = session.extract(source, baseline=req.baseline,
                                      on_progress=lambda m: progress_msgs.append(m))
        except ConcurrentGenerationError as e:
            return _error(409, str(e), "conflict")

        if req.auto_register:
            session.steer(req.name, profile)
            app.state.default_alphas[req.name] = req.alpha

        scored = _profile_top_layers(profile)
        top_layer, top_score = scored[0] if scored else (0, 0.0)

        return {
            "name": req.name,
            "layers": len(profile),
            "top_layer": top_layer,
            "top_score": round(top_score, 4),
        }

    async def _stream_extract(session, app, name, source, baseline, alpha, register):
        # Known limitation: this is fake streaming.  session.extract() is
        # synchronous and CPU/GPU-bound, so progress messages are accumulated
        # in a list and replayed after extraction completes.  True incremental
        # streaming would require threading the extraction pipeline, but the
        # model cannot be safely accessed from multiple threads.  The
        # non-streaming JSON path (extract_vector with Accept: application/json)
        # is the primary interface; this SSE path exists for clients that want
        # progress visibility at the cost of a deferred burst of events.
        progress_msgs: list[str] = []

        def _on_progress(msg):
            progress_msgs.append(msg)
        try:
            profile = session.extract(source, baseline=baseline, on_progress=_on_progress)
        except ConcurrentGenerationError as e:
            err = {"error": {"message": str(e), "type": "conflict", "code": 409}}
            yield f"event: error\ndata: {json.dumps(err)}\n\n"
            return

        for msg in progress_msgs:
            yield f"event: progress\ndata: {json.dumps({'message': msg})}\n\n"

        if register:
            session.steer(name, profile)
            app.state.default_alphas[name] = alpha

        scored = _profile_top_layers(profile)
        top_layer, top_score = scored[0] if scored else (0, 0.0)

        done = {"name": name, "layers": len(profile), "top_layer": top_layer, "top_score": round(top_score, 4)}
        yield f"event: done\ndata: {json.dumps(done)}\n\n"

    @app.post("/v1/liahona/vectors/load")
    def load_vector(req: LoadVectorRequest):
        try:
            profile = session.load_profile(req.path)
        except FileNotFoundError:
            raise HTTPException(404, f"File not found: {req.path}")
        session.steer(req.name, profile)
        app.state.default_alphas[req.name] = req.alpha

        layers = sorted(profile.keys())
        scored = _profile_top_layers(profile)
        top = [{"layer": idx, "score": round(s, 4)} for idx, s in scored]

        return {
            "name": req.name,
            "layers": layers,
            "top_layers": top,
            "default_alpha": req.alpha,
        }

    @app.delete("/v1/liahona/vectors/{name}")
    def delete_vector(name: str):
        if name not in session.vectors:
            raise HTTPException(404, f"Vector '{name}' not found")
        session.unsteer(name)
        app.state.default_alphas.pop(name, None)
        return JSONResponse(status_code=204, content=None)

    # -----------------------------------------------------------------------
    # Probe management
    # -----------------------------------------------------------------------

    @app.get("/v1/liahona/probes")
    def list_probes():
        probes = session.probes
        monitor = session._monitor
        out: dict[str, Any] = {}
        for name in probes:
            entry: dict[str, Any] = {"active": True}
            if monitor and name in monitor.history:
                hist = monitor.history[name]
                if hist:
                    entry["last_value"] = hist[-1]
                    entry["history"] = list(hist)
            out[name] = entry
        return {"probes": out}

    @app.get("/v1/liahona/probes/defaults")
    def list_default_probes():
        return {"defaults": load_defaults()}

    @app.post("/v1/liahona/probes/{name}")
    def activate_probe(name: str, req: ActivateProbeRequest | None = None):
        profile = None
        if req and req.profile_path:
            try:
                profile = session.load_profile(req.profile_path)
            except FileNotFoundError:
                raise HTTPException(404, f"File not found: {req.profile_path}")
        session.monitor(name, profile)
        return {"name": name, "active": True}

    @app.delete("/v1/liahona/probes/{name}")
    def deactivate_probe(name: str):
        if name not in session.probes:
            raise HTTPException(404, f"Probe '{name}' not active")
        session.unmonitor(name)
        return JSONResponse(status_code=204, content=None)

    # -----------------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------------

    @app.get("/v1/liahona/session")
    def get_session():
        info = session.model_info
        return {
            "model": session.model_id,
            "model_info": {
                "model_type": info.get("model_type", "unknown"),
                "num_layers": info.get("num_layers", 0),
                "hidden_dim": info.get("hidden_dim", 0),
                "vram_used_gb": info.get("vram_used_gb", 0.0),
            },
            "config": {
                "temperature": session.config.temperature,
                "top_p": session.config.top_p,
                "max_tokens": session.config.max_new_tokens,
                "system_prompt": session.config.system_prompt,
            },
            "default_alphas": dict(app.state.default_alphas),
            "history_length": len(session.history),
        }

    @app.patch("/v1/liahona/session")
    def patch_session(req: PatchSessionRequest):
        if req.temperature is not None:
            session.config.temperature = req.temperature
        if req.top_p is not None:
            session.config.top_p = req.top_p
        if req.max_tokens is not None:
            session.config.max_new_tokens = req.max_tokens
        if req.system_prompt is not None:
            session.config.system_prompt = req.system_prompt
        return {"status": "ok"}

    @app.post("/v1/liahona/session/clear")
    def clear_session():
        session.clear_history()
        return JSONResponse(status_code=204, content=None)

    @app.post("/v1/liahona/session/rewind")
    def rewind_session():
        if not session.history:
            raise HTTPException(400, "History is empty")
        session.rewind()
        return JSONResponse(status_code=204, content=None)
