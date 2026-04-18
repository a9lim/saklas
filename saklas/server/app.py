"""OpenAI-compatible API server backed by SaklasSession."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, model_validator

from saklas.io.cache_ops import InstallConflict, RefreshError
from saklas.cli.selectors import AmbiguousSelectorError
from saklas.cli.config_file import ConfigFileError
from saklas.core.errors import SaklasError
from saklas.io.hf import HFError
from saklas.io.packs import PackFormatError
from saklas.core.sampling import SamplingConfig
from saklas.core.session import ConcurrentGenerationError, SaklasSession
from saklas.core.steering import Steering


SESSION_LOCK_TIMEOUT_SECONDS = 300


@asynccontextmanager
async def acquire_session_lock(session: SaklasSession) -> AsyncIterator[bool]:
    """Acquire ``session.lock`` with a 5-minute bound.

    Yields ``True`` if the lock was obtained (released on exit) and
    ``False`` on timeout.  Callers branch on the result to emit their
    protocol-specific 503.  Serializes all generation routes across both
    the OpenAI and Ollama protocols on the same session.
    """
    try:
        async with asyncio.timeout(SESSION_LOCK_TIMEOUT_SECONDS):
            await session.lock.acquire()
    except (TimeoutError, asyncio.TimeoutError):
        yield False
        return
    try:
        yield True
    finally:
        session.lock.release()


class UnsupportedContentError(ValueError, SaklasError):
    """Non-text content parts submitted to a text-only endpoint."""


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

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
                    raise UnsupportedContentError(
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
    # Canonical native steering field — a steering expression string
    # parsed through the shared grammar in
    # :mod:`saklas.core.steering_expr`.  Merged over the server's default
    # :class:`Steering` and resolved through ``session.steering()`` so pole
    # aliases and events fire via the single canonical resolver site.
    steering: str | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    logit_bias: dict[int, float] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: bool | int | None = None  # chat: bool; completions: int
    top_logprobs: int | None = None
    user: str | None = None
    # Native thinking override.  None = auto (honours supports_thinking).
    thinking: bool | None = None
    # LangChain compat: accept no-op shapes, reject anything real.
    tools: list | None = None
    tool_choice: Any = None
    # Fields accepted and ignored:
    n: int | None = None
    response_format: dict | None = None

    @model_validator(mode="after")
    def _unify_max_tokens(self):
        if self.max_completion_tokens is not None and self.max_tokens is None:
            self.max_tokens = self.max_completion_tokens
        return self

    @model_validator(mode="after")
    def _check_langchain_compat(self):
        # Accept `tools: []` / None silently; reject non-empty.
        if self.tools:
            raise UnsupportedContentError(
                "tool calling is not supported by saklas"
            )
        # tool_choice: accept None, "none", "auto"; reject "required" and dicts.
        tc = self.tool_choice
        if tc is not None and tc not in ("none", "auto"):
            raise UnsupportedContentError(
                "tool_choice values other than 'none'/'auto' are not supported"
            )
        # response_format: accept None or {"type": "text"}; reject json modes.
        rf = self.response_format
        if rf is not None:
            rf_type = rf.get("type") if isinstance(rf, dict) else None
            if rf_type not in (None, "text"):
                raise UnsupportedContentError(
                    "response_format types other than 'text' are not supported"
                )
        return self

    def to_steering(
        self, default_steering: "Steering | None",
    ) -> "Steering | None":
        """Compose ``self.steering`` (expression string) over the server default.

        The per-request expression overrides the default at the key level:
        alphas for concepts named in both the default and the request come
        from the request; alphas only in the default pass through. Returns
        ``None`` when the composed result is empty and no ``thinking``
        override was requested. Pole aliasing happens inside
        ``session.steering()`` — the server does not resolve poles here.
        """
        from saklas.core.steering_expr import parse_expr

        req_steering: "Steering | None" = None
        if self.steering is not None and self.steering.strip():
            req_steering = parse_expr(self.steering)

        thinking: bool | None = self.thinking
        if req_steering is not None and req_steering.thinking is not None:
            thinking = req_steering.thinking

        merged_alphas: dict = {}
        if default_steering is not None:
            merged_alphas.update(default_steering.alphas)
        if req_steering is not None:
            for k, v in req_steering.alphas.items():
                merged_alphas[k] = v

        if not merged_alphas and thinking is None:
            return None
        return Steering(alphas=merged_alphas, thinking=thinking)


class ChatCompletionRequest(_SamplingBase):
    messages: list[ChatMessage]


class CompletionRequest(_SamplingBase):
    prompt: str


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


def _check_bearer(headers, expected: str) -> bool:
    """Return True iff a correct ``Authorization: Bearer <expected>`` header is present."""
    auth = headers.get("authorization") or headers.get("Authorization")
    if not auth:
        return False
    scheme, _, token = auth.partition(" ")
    return scheme.lower() == "bearer" and token == expected


def _require_auth(request: Request = None,  # type: ignore[assignment]
                  websocket=None):
    """Bearer-token auth gate for HTTP routes.

    Accepts either a ``Request`` or a ``WebSocket`` — FastAPI resolves the
    non-None one based on the route type. On WebSocket connections we can't
    raise ``HTTPException(401)`` (the handshake hasn't completed), so the
    dep returns silently and the handler uses ``ws_auth_ok()`` + ``close(1008)``
    before accepting the connection.
    """
    conn = request if request is not None else websocket
    if conn is None:
        return None
    expected = getattr(conn.app.state, "api_key", None)
    if not expected:
        return None
    if request is None:
        # WS path: handler calls ws_auth_ok() before websocket.accept().
        return None
    if not _check_bearer(request.headers, expected):
        raise HTTPException(
            status_code=401,
            detail={"message": "Invalid API key", "type": "invalid_request_error",
                    "param": None, "code": 401},
        )
    return None


def ws_auth_ok(websocket) -> bool:
    """Return True iff the WebSocket handshake carries valid bearer auth.

    Call this BEFORE ``websocket.accept()``. If it returns False, close the
    handshake with ``await websocket.close(code=1008)``.
    """
    expected = getattr(websocket.app.state, "api_key", None)
    if not expected:
        return True
    return _check_bearer(websocket.headers, expected)


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


def _sampling_kwargs(
    req: _SamplingBase, default_steering: "Steering | None",
) -> dict[str, Any]:
    """Build the kwargs dict passed to session.generate / generate_stream.

    Returns ``sampling=SamplingConfig(...)`` + ``steering=Steering(...)``
    / None + ``thinking=`` + ``stateless=True``.  The server never mutates
    ``session.config``.

    Composes ``req.steering`` (expression string) over
    ``default_steering``: per-request keys override defaults. ``thinking``
    is the native request override; ``None`` triggers
    ``supports_thinking`` auto-detect inside ``_generate_core``.
    """
    stop_tuple: tuple[str, ...] | None
    if req.stop is None:
        stop_tuple = None
    elif isinstance(req.stop, str):
        stop_tuple = (req.stop,)
    else:
        stop_tuple = tuple(req.stop)

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

    sc = SamplingConfig(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        seed=req.seed,
        stop=stop_tuple,
        logit_bias=req.logit_bias,
        presence_penalty=req.presence_penalty or 0.0,
        frequency_penalty=req.frequency_penalty or 0.0,
        logprobs=lp,
    )

    steering = req.to_steering(default_steering)

    thinking_kwarg: bool | None = req.thinking
    if thinking_kwarg is None and steering is not None and steering.thinking is not None:
        thinking_kwarg = steering.thinking

    return {
        "sampling": sc,
        "steering": steering,
        "thinking": thinking_kwarg,
        "stateless": True,
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
                {"token": tok.decode([i]), "logprob": alt_lp,
                 "bytes": _token_bytes(tok.decode([i]))}
                for i, alt_lp in top
            ],
        })
    return {"content": content}


def _render_logprobs_completions(result, session: SaklasSession) -> dict | None:
    """OpenAI /v1/completions logprobs shape (flat, token-parallel arrays).

    https://platform.openai.com/docs/api-reference/completions/object#completions/object-logprobs
    """
    if result.logprobs is None:
        return None
    tok = session._tokenizer
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    offset = 0
    for tid, lp, top in result.logprobs:
        tok_str = tok.decode([tid])
        tokens.append(tok_str)
        token_logprobs.append(lp)
        top_logprobs.append({tok.decode([i]): alt_lp for i, alt_lp in top})
        text_offset.append(offset)
        offset += len(tok_str)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


async def _stream_generation(
    app, session: SaklasSession,
    stream_iter, rid, model_id, object_type, format_delta, empty_delta,
    include_usage: bool = False, role_delta: bool = False,
):
    """Shared SSE generator for chat and completion streaming.

    Serializes against other requests via ``session.lock`` for the full
    stream lifetime (streams inherit queue semantics rather than 409).
    Per-request sampling overrides are carried in the iterator's own
    ``sampling=`` kwarg (bound at caller site) — no session.config rebind.
    """
    created_ts = int(time.time())
    async with acquire_session_lock(session) as acquired:
        if not acquired:
            err = {"error": {"message": "Server busy", "type": "server_error", "code": 503}}
            yield f"data: {json.dumps(err)}\n\n"
            return

        if True:
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
            except ConcurrentGenerationError:
                err = {"error": {"message": "Generation already in progress", "type": "conflict", "code": 409}}
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


def _profile_top_layers(profile: dict, n: int = 5) -> list[tuple[int, float]]:
    """Return top-n profile layers sorted by baked magnitude descending.

    Since shares are baked into tensor magnitudes, ||vec|| is the same
    "how much does this layer steer per unit alpha" quantity that
    per-layer scores used to encode.
    """
    return sorted(
        ((idx, float(vec.norm().item())) for idx, vec in profile.items()),
        key=lambda x: x[1], reverse=True,
    )[:n]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(session: SaklasSession,
               default_steering: "Steering | None" = None,
               cors_origins: list[str] | None = None,
               api_key: str | None = None) -> FastAPI:
    app = FastAPI(
        title="saklas",
        description="OpenAI-compatible API with activation steering",
        dependencies=[Depends(_require_auth)],
    )
    app.state.session = session
    app.state.default_steering = default_steering
    app.state.created_ts = int(time.time())
    app.state.api_key = api_key if api_key is not None else os.environ.get("SAKLAS_API_KEY")
    # Generation serialization lives on ``session.lock`` (asyncio.Lock)
    # so both the OpenAI and Ollama route families share a single FIFO
    # queue.  Requests wait rather than 409 on contention.

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    _SAKLAS_ERROR_STATUS: list[tuple[type, int]] = [
        (ConcurrentGenerationError, 409),
        (AmbiguousSelectorError, 400),
        (UnsupportedContentError, 400),
        (PackFormatError, 400),
        (ConfigFileError, 400),
        (HFError, 502),
        (InstallConflict, 409),
        (RefreshError, 500),
    ]

    def _saklas_error_status(exc: SaklasError) -> int:
        for cls, status in _SAKLAS_ERROR_STATUS:
            if isinstance(exc, cls):
                return status
        return 500

    @app.exception_handler(SaklasError)
    async def _on_saklas_error(request: Request, exc: SaklasError):
        status = _saklas_error_status(exc)
        msg = str(exc) or exc.__class__.__name__
        path = request.url.path
        if path.startswith("/api/"):
            # Ollama error shape: {"error": "<msg>"}
            return JSONResponse(status_code=status, content={"error": msg})
        err_type = "conflict" if status == 409 else "invalid_request_error" if status == 400 else "server_error"
        return _error(status, msg, err_type)

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(_request: Request, exc: RequestValidationError):
        errs = exc.errors()
        first = errs[0] if errs else {}
        loc = first.get("loc", ())
        param = ".".join(str(p) for p in loc[1:]) if len(loc) > 1 else (str(loc[0]) if loc else None)
        msg = first.get("msg", "Invalid request")
        return _error(400, msg, "invalid_request_error", param=param)

    _register_routes(app)

    # Mount Ollama-compatible /api/* routes alongside OpenAI routes so any
    # Ollama client (Open WebUI, Enchanted, ollama-python, etc.) talks to
    # saklas as a drop-in replacement.
    from saklas.server.ollama import register_ollama_routes
    register_ollama_routes(app)

    from saklas.server.saklas_api import register_saklas_routes
    register_saklas_routes(app)

    return app


def _strict_model_enabled() -> bool:
    return os.environ.get("SAKLAS_STRICT_MODEL", "").lower() in ("1", "true", "yes", "on")


def _openai_known_model_names(session: SaklasSession) -> set[str]:
    """Names accepted for OpenAI routes in strict mode.

    Includes the HF id plus any Ollama-style aliases (`<family>:<size>`)
    — the OpenAI catalogue is a superset so clients hitting either
    protocol with the same name keep working.
    """
    from saklas.server.ollama import _aliases_for
    return {n.lower() for n in {session.model_id, *_aliases_for(session)}}


def _check_openai_model_strict(session: SaklasSession, name: str | None) -> None:
    if not _strict_model_enabled():
        return
    if not name:
        return
    if name.lower() not in _openai_known_model_names(session):
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Model '{name}' not found",
                "type": "invalid_request_error",
                "param": "model",
                "code": 404,
            },
        )


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

    async def _run_blocking(req, prompt_or_messages, *, raw: bool):
        gen_kwargs = _sampling_kwargs(req, app.state.default_steering)
        async with session.lock:
            return session.generate(prompt_or_messages, raw=raw, **gen_kwargs)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        _check_openai_model_strict(session, req.model)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        rid = _make_id()
        model_id = session.model_id
        gen_kwargs = _sampling_kwargs(req, app.state.default_steering)

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
                                   include_usage=include_usage, role_delta=True),
                media_type="text/event-stream",
            )
        try:
            result = await _run_blocking(req, messages, raw=False)
        except ConcurrentGenerationError:
            return _error(409, "Generation already in progress", "conflict")

        return {
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

    # -----------------------------------------------------------------------
    # Text completions
    # -----------------------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        _check_openai_model_strict(session, req.model)
        rid = _make_id()
        model_id = session.model_id
        gen_kwargs = _sampling_kwargs(req, app.state.default_steering)

        if req.stream:
            stream_iter = session.generate_stream(req.prompt, raw=True, **gen_kwargs)
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return StreamingResponse(
                _stream_generation(app, session,
                                   stream_iter, rid, model_id,
                                   "text_completion", lambda e: {"text": e.text}, {"text": ""},
                                   include_usage=include_usage, role_delta=False),
                media_type="text/event-stream",
            )
        try:
            result = await _run_blocking(req, req.prompt, raw=True)
        except ConcurrentGenerationError:
            return _error(409, "Generation already in progress", "conflict")

        return {
            "id": rid,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "text": result.text,
                    "logprobs": _render_logprobs_completions(result, session),
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": _usage_dict(result),
            "probe_readings": _probe_reading_dict(session),
        }

