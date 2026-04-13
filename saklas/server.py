"""OpenAI-compatible API server backed by SaklasSession."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, model_validator

from saklas.probes_bootstrap import load_defaults
from saklas.session import ConcurrentGenerationError, SaklasSession


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class SteerParams(BaseModel):
    alphas: dict[str, float] = Field(default_factory=dict)
    orthogonalize: bool = False
    thinking: bool = False


class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list of content parts
    name: str | None = None

    @model_validator(mode="after")
    def _flatten_content(self):
        # Accept OpenAI multimodal content-part arrays for text-only use:
        # concatenate text parts, reject anything else with a clear error.
        if isinstance(self.content, list):
            pieces: list[str] = []
            for part in self.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    pieces.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    pieces.append(part)
                else:
                    raise ValueError(
                        "non-text content parts are not supported by this model"
                    )
            self.content = "".join(pieces)
        elif not isinstance(self.content, str):
            self.content = str(self.content)
        return self


class StreamOptions(BaseModel):
    include_usage: bool = False


class _SamplingBase(BaseModel):
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    steer: SteerParams | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    logit_bias: dict[int, float] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: bool | int | None = None  # chat: bool; completions: int
    top_logprobs: int | None = None
    user: str | None = None
    # Fields accepted and ignored:
    n: int | None = None
    response_format: dict | None = None

    @model_validator(mode="after")
    def _unify_max_tokens(self):
        if self.max_completion_tokens is not None and self.max_tokens is None:
            self.max_tokens = self.max_completion_tokens
        return self


class ChatCompletionRequest(_SamplingBase):
    messages: list[ChatMessage]


class CompletionRequest(_SamplingBase):
    prompt: str


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
    return f"saklas-{uuid.uuid4().hex[:12]}"


def _error(status: int, message: str, error_type: str = "error",
           param: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type,
                           "param": param, "code": status}},
    )


_bearer = HTTPBearer(auto_error=False)


def _require_auth(request: Request,
                  creds: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    expected = request.app.state.api_key
    if not expected:
        return None
    if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != expected:
        raise HTTPException(
            status_code=401,
            detail={"message": "Invalid API key",
                    "type": "invalid_request_error",
                    "param": None, "code": 401},
        )
    return None


def _probe_reading_dict(session: SaklasSession) -> dict[str, Any]:
    # build_readings() already scopes to monitor.probe_names, but cross-check
    # explicitly so a client never sees a probe that isn't active in the monitor.
    monitor_names = set(session._monitor.probe_names)
    readings = session.build_readings()
    out: dict[str, Any] = {}
    for name, r in readings.items():
        if name not in monitor_names:
            continue
        out[name] = r.to_dict()
    return out


def _resolve_steer_params(
    steer: SteerParams | None, default_alphas: dict[str, float],
) -> tuple[dict[str, float] | None, bool, bool]:
    """Resolve (alphas, orthogonalize, thinking) from a request's steer block."""
    alphas = _resolve_alphas(steer, default_alphas)
    ortho = steer.orthogonalize if steer else False
    think = steer.thinking if steer else False
    return alphas, ortho, think


def _sampling_kwargs(req: _SamplingBase, alphas, ortho, think) -> dict[str, Any]:
    """Build the kwargs dict passed to session.generate / generate_stream."""
    stop_list: list[str] | None
    if req.stop is None:
        stop_list = None
    elif isinstance(req.stop, str):
        stop_list = [req.stop]
    else:
        stop_list = list(req.stop)

    # chat: logprobs is bool + top_logprobs gives count.
    # completions: logprobs is int (number of top alternatives).
    # Internally saklas takes an int count (0 = chosen only, None = disabled).
    lp: int | None
    if isinstance(req.logprobs, bool):
        lp = (req.top_logprobs or 0) if req.logprobs else None
    elif isinstance(req.logprobs, int):
        lp = req.logprobs
    else:
        lp = None

    return {
        "alphas": alphas,
        "orthogonalize": ortho,
        "thinking": think,
        "stateless": True,
        "seed": req.seed,
        "stop": stop_list,
        "logit_bias": req.logit_bias,
        "presence_penalty": req.presence_penalty,
        "frequency_penalty": req.frequency_penalty,
        "logprobs": lp,
    }


def _usage_dict(result) -> dict[str, int]:
    pt = result.prompt_tokens
    ct = result.token_count
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


def _token_bytes(text: str) -> list[int]:
    try:
        return list(text.encode("utf-8"))
    except Exception:
        return []


def _render_logprobs_chat(result, session: SaklasSession) -> dict | None:
    if result.logprobs is None:
        return None
    tok = session._tokenizer
    content = []
    for tid, lp, top in result.logprobs:
        tok_str = tok.decode([tid])
        content.append({
            "token": tok_str,
            "logprob": lp,
            "bytes": _token_bytes(tok_str),
            "top_logprobs": [
                {"token": tok.decode([i]), "logprob": lp, "bytes": _token_bytes(tok.decode([i]))}
                for i, lp in top
            ],
        })
    return {"content": content}


@contextmanager
def _gen_config_override(session: SaklasSession, temperature, top_p, max_tokens):
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


async def _stream_generation(
    app, session: SaklasSession,
    stream_iter, rid, model_id, object_type, format_delta, empty_delta,
    temperature, top_p, max_tokens,
    include_usage: bool = False, role_delta: bool = False,
):
    """Shared SSE generator for chat and completion streaming.

    Serializes against other requests via app.state.gen_lock for the full
    stream lifetime (streams inherit queue semantics rather than 409).
    """
    created_ts = int(time.time())
    try:
        async with asyncio.timeout(300):
            await app.state.gen_lock.acquire()
    except (TimeoutError, asyncio.TimeoutError):
        err = {"error": {"message": "Server busy", "type": "server_error", "code": 503}}
        yield f"data: {json.dumps(err)}\n\n"
        return

    try:
        with _gen_config_override(session, temperature, top_p, max_tokens):
            if role_delta:
                chunk = {
                    "id": rid, "object": object_type, "created": created_ts,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            try:
                for event in stream_iter:
                    chunk = {
                        "id": rid,
                        "object": object_type,
                        "created": created_ts,
                        "model": model_id,
                        "choices": [{"index": 0, **format_delta(event), "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            except ConcurrentGenerationError as e:
                err = {"error": {"message": str(e), "type": "conflict", "code": 409}}
                yield f"data: {json.dumps(err)}\n\n"
                return

            finish_reason = session._gen_state.finish_reason
            final = {
                "id": rid,
                "object": object_type,
                "created": created_ts,
                "model": model_id,
                "choices": [{"index": 0, **empty_delta, "finish_reason": finish_reason}],
                "probe_readings": _probe_reading_dict(session),
            }
            yield f"data: {json.dumps(final)}\n\n"

            if include_usage and session._last_result is not None:
                usage_chunk = {
                    "id": rid, "object": object_type, "created": created_ts,
                    "model": model_id, "choices": [],
                    "usage": _usage_dict(session._last_result),
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"

            yield "data: [DONE]\n\n"
    finally:
        app.state.gen_lock.release()


def _profile_top_layers(profile: dict, n: int = 5) -> list[tuple[int, float]]:
    """Return top-n profile layers sorted by score descending."""
    return sorted(((idx, score) for idx, (_vec, score) in profile.items()),
                  key=lambda x: x[1], reverse=True)[:n]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(session: SaklasSession, default_alphas: dict[str, float] | None = None,
               cors_origins: list[str] | None = None,
               api_key: str | None = None) -> FastAPI:
    app = FastAPI(
        title="saklas",
        description="OpenAI-compatible API with activation steering",
        dependencies=[Depends(_require_auth)],
    )
    app.state.session = session
    app.state.default_alphas = default_alphas or {}
    app.state.created_ts = int(time.time())
    app.state.api_key = api_key if api_key is not None else os.environ.get("SAKLAS_API_KEY")
    # Serializes generation across all /v1/*/completions routes (both
    # streaming and non-streaming).  Streaming requests acquire the lock
    # inside _stream_generation so Starlette's post-route consumption of
    # the generator still holds it; non-streaming routes wrap generate()
    # in `async with app.state.gen_lock`.  Requests queue FIFO rather
    # than 409ing on contention.
    app.state.gen_lock = asyncio.Lock()

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(_request: Request, exc: RequestValidationError):
        errs = exc.errors()
        first = errs[0] if errs else {}
        loc = first.get("loc", ())
        param = ".".join(str(p) for p in loc[1:]) if len(loc) > 1 else (str(loc[0]) if loc else None)
        msg = first.get("msg", "Invalid request")
        return _error(400, msg, "invalid_request_error", param=param)

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    session: SaklasSession = app.state.session

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
        alphas, ortho, think = _resolve_steer_params(req.steer, app.state.default_alphas)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        rid = _make_id()
        model_id = session.model_id
        gen_kwargs = _sampling_kwargs(req, alphas, ortho, think)

        if req.stream:
            def _chat_delta(event):
                d: dict[str, str] = {}
                if event.thinking:
                    d["reasoning_content"] = event.text
                else:
                    d["content"] = event.text
                return {"delta": d}

            stream_iter = session.generate_stream(messages, **gen_kwargs)
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return StreamingResponse(
                _stream_generation(app, session,
                                   stream_iter, rid, model_id,
                                   "chat.completion.chunk", _chat_delta, {"delta": {}},
                                   req.temperature, req.top_p, req.max_tokens,
                                   include_usage=include_usage, role_delta=True),
                media_type="text/event-stream",
            )
        async with app.state.gen_lock:
            with _gen_config_override(session, req.temperature, req.top_p, req.max_tokens):
                try:
                    result = session.generate(messages, **gen_kwargs)
                except ConcurrentGenerationError as e:
                    return _error(409, str(e), "conflict")

        response = {
            "id": rid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "logprobs": _render_logprobs_chat(result, session),
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": _usage_dict(result),
            "probe_readings": _probe_reading_dict(session),
        }
        return response

    # -----------------------------------------------------------------------
    # Text completions
    # -----------------------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        alphas, ortho, think = _resolve_steer_params(req.steer, app.state.default_alphas)
        rid = _make_id()
        model_id = session.model_id
        gen_kwargs = _sampling_kwargs(req, alphas, ortho, think)

        if req.stream:
            stream_iter = session.generate_stream(req.prompt, raw=True, **gen_kwargs)
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return StreamingResponse(
                _stream_generation(app, session,
                                   stream_iter, rid, model_id,
                                   "text_completion", lambda e: {"text": e.text}, {"text": ""},
                                   req.temperature, req.top_p, req.max_tokens,
                                   include_usage=include_usage, role_delta=False),
                media_type="text/event-stream",
            )
        async with app.state.gen_lock:
            with _gen_config_override(session, req.temperature, req.top_p, req.max_tokens):
                try:
                    result = session.generate(req.prompt, raw=True, **gen_kwargs)
                except ConcurrentGenerationError as e:
                    return _error(409, str(e), "conflict")

        return {
            "id": rid,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "text": result.text,
                    "logprobs": _render_logprobs_chat(result, session),
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": _usage_dict(result),
            "probe_readings": _probe_reading_dict(session),
        }

    # -----------------------------------------------------------------------
    # Vector management
    # -----------------------------------------------------------------------

    @app.get("/v1/saklas/vectors")
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

    @app.post("/v1/saklas/vectors/extract")
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
            canonical, profile = session.extract(source, baseline=req.baseline,
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
            "canonical": canonical,
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
            canonical, profile = session.extract(source, baseline=baseline, on_progress=_on_progress)
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

        done = {"name": name, "canonical": canonical, "layers": len(profile), "top_layer": top_layer, "top_score": round(top_score, 4)}
        yield f"event: done\ndata: {json.dumps(done)}\n\n"

    @app.post("/v1/saklas/vectors/load")
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

    @app.delete("/v1/saklas/vectors/{name}")
    def delete_vector(name: str):
        if name not in session.vectors:
            raise HTTPException(404, f"Vector '{name}' not found")
        session.unsteer(name)
        app.state.default_alphas.pop(name, None)
        return JSONResponse(status_code=204, content=None)

    # -----------------------------------------------------------------------
    # Probe management
    # -----------------------------------------------------------------------

    @app.get("/v1/saklas/probes")
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

    @app.get("/v1/saklas/probes/defaults")
    def list_default_probes():
        return {"defaults": load_defaults()}

    @app.post("/v1/saklas/probes/{name}")
    def activate_probe(name: str, req: ActivateProbeRequest | None = None):
        profile = None
        if req and req.profile_path:
            try:
                profile = session.load_profile(req.profile_path)
            except FileNotFoundError:
                raise HTTPException(404, f"File not found: {req.profile_path}")
        session.monitor(name, profile)
        return {"name": name, "active": True}

    @app.delete("/v1/saklas/probes/{name}")
    def deactivate_probe(name: str):
        if name not in session.probes:
            raise HTTPException(404, f"Probe '{name}' not active")
        session.unmonitor(name)
        return JSONResponse(status_code=204, content=None)

    # -----------------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------------

    @app.get("/v1/saklas/session")
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

    @app.patch("/v1/saklas/session")
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

    @app.post("/v1/saklas/session/clear")
    def clear_session():
        session.clear_history()
        return JSONResponse(status_code=204, content=None)

    @app.post("/v1/saklas/session/rewind")
    def rewind_session():
        if not session.history:
            raise HTTPException(400, "History is empty")
        session.rewind()
        return JSONResponse(status_code=204, content=None)
