"""OpenAI-compatible API server backed by LiahonaSession."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from liahona.probes_bootstrap import _load_defaults
from liahona.session import LiahonaSession


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


def _ts() -> int:
    return int(time.time())


def _error(status: int, message: str, error_type: str = "error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "code": status}},
    )


def _probe_reading_dict(session: LiahonaSession) -> dict[str, Any]:
    readings = session._build_readings()
    out: dict[str, Any] = {}
    for name, r in readings.items():
        out[name] = {"mean": r.mean, "std": r.std, "min": r.min, "max": r.max}
    return out


def _apply_gen_overrides(session: LiahonaSession, temperature, top_p, max_tokens):
    """Temporarily override generation config, return originals."""
    orig = (session.config.temperature, session.config.top_p, session.config.max_new_tokens)
    if temperature is not None:
        session.config.temperature = temperature
    if top_p is not None:
        session.config.top_p = top_p
    if max_tokens is not None:
        session.config.max_new_tokens = max_tokens
    return orig


def _restore_gen_config(session: LiahonaSession, orig):
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


def _ortho(steer_params: SteerParams | None) -> bool:
    return steer_params.orthogonalize if steer_params else False


def _thinking(steer_params: SteerParams | None) -> bool:
    return steer_params.thinking if steer_params else False


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(session: LiahonaSession, default_alphas: dict[str, float] | None = None,
               cors_origins: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="liahona", description="OpenAI-compatible API with activation steering")
    app.state.session = session
    app.state.default_alphas = default_alphas or {}

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
        info = session.model_info
        return {
            "object": "list",
            "data": [
                {
                    "id": info.get("model_id", "unknown"),
                    "object": "model",
                    "created": _ts(),
                    "owned_by": "local",
                }
            ],
        }

    @app.get("/v1/models/{model_id:path}")
    def get_model(model_id: str):
        info = session.model_info
        if model_id != info.get("model_id", "unknown"):
            raise HTTPException(404, f"Model '{model_id}' not found")
        return {
            "id": info.get("model_id", "unknown"),
            "object": "model",
            "created": _ts(),
            "owned_by": "local",
        }

    # -----------------------------------------------------------------------
    # Chat completions
    # -----------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        alphas = _resolve_alphas(req.steer, app.state.default_alphas)
        ortho = _ortho(req.steer)
        think = _thinking(req.steer)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        rid = _make_id()
        model_id = session.model_info.get("model_id", "unknown")

        orig = _apply_gen_overrides(session, req.temperature, req.top_p, req.max_tokens)
        try:
            if req.stream:
                return StreamingResponse(
                    _stream_chat(session, messages, alphas, ortho, think, rid, model_id, orig),
                    media_type="text/event-stream",
                )
            # Non-streaming
            try:
                result = session.generate(messages, alphas=alphas, orthogonalize=ortho, thinking=think)
            except RuntimeError as e:
                if "already in progress" in str(e):
                    return _error(409, str(e), "conflict")
                raise
            finally:
                _restore_gen_config(session, orig)

            return {
                "id": rid,
                "object": "chat.completion",
                "created": _ts(),
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
        except RuntimeError as e:
            _restore_gen_config(session, orig)
            if "already in progress" in str(e):
                return _error(409, str(e), "conflict")
            raise

    async def _stream_chat(session, messages, alphas, ortho, think, rid, model_id, orig_config):
        try:
            try:
                for event in session.generate_stream(messages, alphas=alphas, orthogonalize=ortho, thinking=think):
                    delta: dict[str, str] = {}
                    if event.thinking:
                        delta["reasoning_content"] = event.text
                    else:
                        delta["content"] = event.text
                    chunk = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": _ts(),
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            except RuntimeError as e:
                if "already in progress" in str(e):
                    err = {"error": {"message": str(e), "type": "conflict", "code": 409}}
                    yield f"data: {json.dumps(err)}\n\n"
                    return
                raise

            # Final chunk with finish_reason and probe readings
            final = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": _ts(),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "probe_readings": _probe_reading_dict(session),
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            _restore_gen_config(session, orig_config)

    # -----------------------------------------------------------------------
    # Text completions
    # -----------------------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        alphas = _resolve_alphas(req.steer, app.state.default_alphas)
        ortho = _ortho(req.steer)
        rid = _make_id()
        model_id = session.model_info.get("model_id", "unknown")

        orig = _apply_gen_overrides(session, req.temperature, req.top_p, req.max_tokens)
        try:
            if req.stream:
                return StreamingResponse(
                    _stream_completions(session, req.prompt, alphas, ortho, rid, model_id, orig),
                    media_type="text/event-stream",
                )
            try:
                result = session.generate(req.prompt, alphas=alphas, orthogonalize=ortho, raw=True)
            except RuntimeError as e:
                if "already in progress" in str(e):
                    return _error(409, str(e), "conflict")
                raise
            finally:
                _restore_gen_config(session, orig)

            return {
                "id": rid,
                "object": "text_completion",
                "created": _ts(),
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
        except RuntimeError as e:
            _restore_gen_config(session, orig)
            if "already in progress" in str(e):
                return _error(409, str(e), "conflict")
            raise

    async def _stream_completions(session, prompt, alphas, ortho, rid, model_id, orig_config):
        try:
            try:
                for event in session.generate_stream(prompt, alphas=alphas, orthogonalize=ortho, raw=True):
                    chunk = {
                        "id": rid,
                        "object": "text_completion",
                        "created": _ts(),
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "text": event.text,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            except RuntimeError as e:
                if "already in progress" in str(e):
                    err = {"error": {"message": str(e), "type": "conflict", "code": 409}}
                    yield f"data: {json.dumps(err)}\n\n"
                    return
                raise

            final = {
                "id": rid,
                "object": "text_completion",
                "created": _ts(),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "finish_reason": "stop",
                    }
                ],
                "probe_readings": _probe_reading_dict(session),
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            _restore_gen_config(session, orig_config)

    # -----------------------------------------------------------------------
    # Vector management
    # -----------------------------------------------------------------------

    @app.get("/v1/liahona/vectors")
    def list_vectors():
        vectors = session.vectors
        out: dict[str, Any] = {}
        for name, profile in vectors.items():
            layers = sorted(profile.keys())
            scored = [(idx, score) for idx, (_vec, score) in profile.items()]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [{"layer": idx, "score": round(s, 4)} for idx, s in scored[:5]]
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
        except RuntimeError as e:
            if "already in progress" in str(e):
                return _error(409, str(e), "conflict")
            raise

        if req.auto_register:
            session.steer(req.name, profile)
            app.state.default_alphas[req.name] = req.alpha

        layers = sorted(profile.keys())
        scored = [(idx, score) for idx, (_vec, score) in profile.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_layer, top_score = scored[0] if scored else (0, 0.0)

        return {
            "name": req.name,
            "layers": len(layers),
            "top_layer": top_layer,
            "top_score": round(top_score, 4),
        }

    async def _stream_extract(session, app, name, source, baseline, alpha, register):
        progress_msgs: list[str] = []

        def _on_progress(msg):
            progress_msgs.append(msg)

        try:
            profile = session.extract(source, baseline=baseline, on_progress=_on_progress)
        except RuntimeError as e:
            if "already in progress" in str(e):
                err = {"error": {"message": str(e), "type": "conflict", "code": 409}}
                yield f"event: error\ndata: {json.dumps(err)}\n\n"
                return
            raise

        for msg in progress_msgs:
            yield f"event: progress\ndata: {json.dumps({'message': msg})}\n\n"

        if register:
            session.steer(name, profile)
            app.state.default_alphas[name] = alpha

        layers = sorted(profile.keys())
        scored = [(idx, score) for idx, (_vec, score) in profile.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_layer, top_score = scored[0] if scored else (0, 0.0)

        done = {"name": name, "layers": len(layers), "top_layer": top_layer, "top_score": round(top_score, 4)}
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
        scored = [(idx, score) for idx, (_vec, score) in profile.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [{"layer": idx, "score": round(s, 4)} for idx, s in scored[:5]]

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
        return {"defaults": _load_defaults()}

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
            "model": info.get("model_id", "unknown"),
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
