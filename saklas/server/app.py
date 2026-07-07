"""OpenAI-compatible API server backed by SaklasSession."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator

if TYPE_CHECKING:
    from saklas.core.results import GenerationResult

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, model_validator
from starlette.datastructures import Headers

from saklas.core.errors import SaklasError
from saklas.core.session import ConcurrentGenerationError, SaklasSession
from saklas.core.steering import Steering
from saklas.server.request_helpers import (
    UnsupportedContentError,
    build_sampling_config as _build_sampling_config,
    flatten_content as _flatten_content,
    merge_steering as _merge_steering,
    parse_request_steering as _parse_req_steering,
    probe_reading_aggregate as _probe_reading_aggregate,
    probe_reading_dict as _probe_reading_dict,
    probe_token_readings as _probe_token_readings,
    strict_model_enabled as _strict_model_enabled,
)
from saklas.server.streaming import (
    _usage_dict,
    stream_finalizer,
)


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
        self.content = _flatten_content(self.content)
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
    tools: list[Any] | None = None
    tool_choice: Any = None
    # Fields accepted and ignored:
    n: int | None = None
    response_format: dict[str, Any] | None = None

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
            rf_type = rf.get("type")
            if rf_type not in (None, "text"):
                raise UnsupportedContentError(
                    "response_format types other than 'text' are not supported"
                )
        return self

    def to_steering(
        self, default_steering: "Steering | None",
    ) -> "Steering | None":
        """Compose ``self.steering`` (expression string) over the server default.

        ``None`` inherits the server default; an explicit empty string
        clears it.  Non-empty per-request expressions override the default
        at the key level: alphas for concepts named in both the default
        and the request come from the request; alphas only in the default
        pass through. Returns ``None`` when the composed result is empty
        and no ``thinking`` override was requested. Pole aliasing happens
        inside ``session.steering()`` — the server does not resolve poles
        here.
        """
        req_steering, explicit_clear = _parse_req_steering(self.steering)
        thinking: bool | None = self.thinking
        if req_steering is not None and req_steering.thinking is not None:
            thinking = req_steering.thinking
        return _merge_steering(
            req_steering, default_steering, explicit_clear, thinking,
        )


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


def _check_bearer(headers: Headers, expected: str) -> bool:
    """Return True iff a correct ``Authorization: Bearer <expected>`` header is present."""
    auth = headers.get("authorization") or headers.get("Authorization")
    if not auth:
        return False
    scheme, _, token = auth.partition(" ")
    return scheme.lower() == "bearer" and token == expected


def _require_auth(request: Request = None,  # pyright: ignore[reportArgumentType]  # FastAPI injects Request/WebSocket by type; None default is a sentinel, not a real argument
                  websocket=None):  # pyright: ignore[reportMissingParameterType]  # FastAPI special-cases bare WebSocket/Request injection; an explicit `WebSocket | None` annotation makes it build a request field and raises at app-construction time
    """Bearer-token auth gate for HTTP routes.

    Accepts either a ``Request`` or a ``WebSocket`` — FastAPI resolves the
    non-None one based on the route type. On WebSocket connections we can't
    raise ``HTTPException(401)`` (the handshake hasn't completed), so the
    dep returns silently and the handler uses ``ws_auth_ok()`` + ``close(1008)``
    before accepting the connection.
    """
    conn = request if request is not None else websocket
    if conn is None:
        return
    expected = getattr(conn.app.state, "api_key", None)
    if not expected:
        return
    if request is None:
        # WS path: handler calls ws_auth_ok() before websocket.accept().
        return
    if not _check_bearer(request.headers, expected):
        raise HTTPException(
            status_code=401,
            detail={"message": "Invalid API key", "type": "invalid_request_error",
                    "param": None, "code": 401},
        )
    return


def ws_auth_ok(websocket: WebSocket) -> bool:
    """Return True iff the WebSocket handshake carries valid bearer auth.

    Call this BEFORE ``websocket.accept()``. If it returns False, close the
    handshake with ``await websocket.close(code=1008)``.
    """
    expected = getattr(websocket.app.state, "api_key", None)
    if not expected:
        return True
    if _check_bearer(websocket.headers, expected):
        return True
    # Browser WebSocket constructors cannot attach Authorization headers.
    # The bundled dashboard sends the same bearer value as ?token=... .
    return websocket.query_params.get("token") == expected


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

    sc = _build_sampling_config(
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


def _token_bytes(text: str) -> list[int]:
    try:
        return list(text.encode("utf-8"))
    except Exception:
        return []


def _render_logprobs_chat(result: GenerationResult, session: SaklasSession) -> dict[str, Any] | None:
    if result.logprobs is None:
        return None
    tok = session.tokenizer
    content = []
    # Inner ``top`` is now ``list[TokenAlt]`` (id/text/logprob triples
    # decoded by the engine at top-K capture time); the previous
    # ``list[tuple[int, float]]`` pair shape was retired with the phase 1
    # logit pass so we no longer re-tokenize the alt ids here.  The
    # chosen-token text still goes through ``tok.decode`` because the
    # engine emits its id via ``result.tokens`` without the streaming
    # text representation alongside.
    for tid, lp, top in result.logprobs:
        tok_str: str = tok.decode([tid])  # pyright: ignore[reportAssignmentType]  # transformers stub returns str | list[str] but single-list input always yields str
        content.append({
            "token": tok_str,
            "logprob": lp,
            "bytes": _token_bytes(tok_str),
            "top_logprobs": [
                {"token": alt.text, "logprob": alt.logprob,
                 "bytes": _token_bytes(alt.text)}
                for alt in top
            ],
        })
    return {"content": content}


def _render_logprobs_completions(result: GenerationResult, session: SaklasSession) -> dict[str, Any] | None:
    """OpenAI /v1/completions logprobs shape (flat, token-parallel arrays).

    https://platform.openai.com/docs/api-reference/completions/object#completions/object-logprobs

    Inner ``top`` is ``list[TokenAlt]`` post-phase-1 logit pass — alt
    text comes off the dataclass rather than a redundant tokenizer
    decode.
    """
    if result.logprobs is None:
        return None
    tok = session.tokenizer
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    offset = 0
    for tid, lp, top in result.logprobs:
        tok_str: str = tok.decode([tid])  # pyright: ignore[reportAssignmentType]  # transformers stub returns str | list[str] but single-list input always yields str
        tokens.append(tok_str)
        token_logprobs.append(lp)
        top_logprobs.append({alt.text: alt.logprob for alt in top})
        text_offset.append(offset)
        offset += len(tok_str)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


async def _stream_generation(
    session: SaklasSession,
    stream_iter: Iterator[Any], rid: str, model_id: str, object_type: str,
    format_delta: Callable[[Any], dict[str, Any]], empty_delta: dict[str, Any],
    include_usage: bool = False, role_delta: bool = False,
    request: Request | None = None,
):
    """Shared SSE generator for chat and completion streaming.

    Serializes against other requests via ``session.lock`` for the full
    stream lifetime (streams inherit queue semantics rather than 409).
    Per-request sampling overrides are carried in the iterator's own
    ``sampling=`` kwarg (bound at caller site) — no session.config rebind.

    The inner ``stream_iter`` is always ``.close()``d in a ``finally`` so the
    engine's worker-thread teardown (stop-flag + join) fires deterministically
    on early exit rather than waiting on GC.  When a ``request`` is wired in,
    a periodic ``is_disconnected()`` check stops generation early once the
    client is gone, so the GPU isn't spent on a stream nobody reads.
    """
    created_ts = int(time.time())
    async with acquire_session_lock(session) as acquired:
        if not acquired:
            err = {"error": {"message": "Server busy", "type": "server_error", "code": 503}}
            yield f"data: {json.dumps(err)}\n\n"
            return

        if role_delta:
            chunk = {
                "id": rid, "object": object_type, "created": created_ts,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        try:
            for event in stream_iter:
                # Bail out if the client has hung up — close the inner
                # generator (handled in ``finally``) and stop spending the
                # GPU on tokens nobody is reading.
                if request is not None and await request.is_disconnected():
                    return
                choice: dict[str, Any] = {
                    "index": 0, **format_delta(event), "finish_reason": None,
                }
                # Per-token manifold readings ride under a vendor-
                # prefixed extension on the choice so OpenAI clients
                # that don't read the field stay unaffected.  Populated
                # only when at least one manifold probe is attached
                # and ``live_scores`` is True on the stream.
                mf_token = _probe_token_readings(event)
                if mf_token is not None:
                    choice["x-saklas-probe-readings"] = mf_token
                chunk = {
                    "id": rid,
                    "object": object_type,
                    "created": created_ts,
                    "model": model_id,
                    "choices": [choice],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        except ConcurrentGenerationError:
            err = {"error": {"message": "Generation already in progress", "type": "conflict", "code": 409}}
            yield f"data: {json.dumps(err)}\n\n"
            return
        except SaklasError as e:
            status, msg = e.user_message()
            err_type = (
                "conflict" if status == 409
                else "invalid_request_error" if 400 <= status < 500
                else "server_error"
            )
            err = {
                "error": {
                    "message": msg,
                    "type": err_type,
                    "code": status,
                }
            }
            yield f"data: {json.dumps(err)}\n\n"
            return
        finally:
            # Deterministically tear down the engine worker thread (stop-flag
            # + join) on every exit — normal completion (no-op on an exhausted
            # generator), an in-band error, or an early client-disconnect
            # ``return`` — rather than leaving it to GC.
            close = getattr(stream_iter, "close", None)
            if callable(close):
                close()

        last_result = (
            getattr(stream_iter, "result", None)
            or getattr(session, "last_result", None)
        )
        finish_reason, usage, mf_agg = stream_finalizer(session, last_result)
        final_choice: dict[str, Any] = {
            "index": 0, **empty_delta, "finish_reason": finish_reason,
        }
        if mf_agg:
            final_choice["x-saklas-probe-readings"] = mf_agg
        compat_probe_readings = _probe_reading_dict(
            session,
            readings=getattr(last_result, "readings", None) if last_result is not None else None,
        )
        final = {
            "id": rid,
            "object": object_type,
            "created": created_ts,
            "model": model_id,
            "choices": [final_choice],
        }
        if compat_probe_readings:
            final["probe_readings"] = compat_probe_readings
        yield f"data: {json.dumps(final)}\n\n"

        if include_usage and usage is not None:
            usage_chunk = {
                "id": rid, "object": object_type, "created": created_ts,
                "model": model_id, "choices": [],
                "usage": usage,
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(session: SaklasSession,
               default_steering: "Steering | None" = None,
               cors_origins: list[str] | None = None,
               api_key: str | None = None,
               *,
               web: bool = False) -> FastAPI:
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

    @app.exception_handler(SaklasError)
    async def _on_saklas_error(request: Request, exc: SaklasError):
        status, msg = exc.user_message()
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

    # Mount the Svelte+Vite SPA dashboard last so its catch-all route
    # doesn't shadow any of the API routes registered above.  CLI
    # default-on (``saklas serve``); ``--no-web`` opts out for
    # production / proxied deployments where ``/`` already belongs to
    # something else.  Library callers using ``create_app`` directly
    # default-off so embedded API surfaces don't accidentally pick up
    # the dashboard.
    if web:
        from saklas.web import register_web_routes

        register_web_routes(app)

    return app


def _openai_known_model_names(session: SaklasSession) -> set[str]:
    """Names accepted for OpenAI routes in strict mode.

    Includes the HF id plus any Ollama-style aliases (`<family>:<size>`)
    — the OpenAI catalogue is a superset so clients hitting either
    protocol with the same name keep working.
    """
    from saklas.server.model_names import known_model_names

    return known_model_names(session)


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

    async def _run_blocking(req: _SamplingBase, prompt_or_messages: Any, *, raw: bool) -> Any:
        gen_kwargs = _sampling_kwargs(req, app.state.default_steering)
        # Bounded lock so a non-streaming request can't queue forever behind
        # a stuck generation — it 503s like the streaming paths do.  Returns
        # a ``JSONResponse`` on timeout; callers return it verbatim.
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                return _error(503, "Server busy", "server_error")
            return session.generate(prompt_or_messages, raw=raw, **gen_kwargs)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        _check_openai_model_strict(session, req.model)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        rid = _make_id()
        model_id = session.model_id
        gen_kwargs = _sampling_kwargs(req, app.state.default_steering)

        if req.stream:
            def _chat_delta(event: Any) -> dict[str, Any]:
                d: dict[str, str] = {}
                if event.thinking:
                    d["reasoning_content"] = event.text
                else:
                    d["content"] = event.text
                return {"delta": d}

            stream_iter = session.generate_stream(messages, **gen_kwargs)
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return StreamingResponse(
                _stream_generation(session,
                                   stream_iter, rid, model_id,
                                   "chat.completion.chunk", _chat_delta, {"delta": {}},
                                   include_usage=include_usage, role_delta=True,
                                   request=request),
                media_type="text/event-stream",
            )
        try:
            result = await _run_blocking(req, messages, raw=False)
        except ConcurrentGenerationError:
            return _error(409, "Generation already in progress", "conflict")
        if isinstance(result, JSONResponse):  # bounded-lock timeout → 503
            return result

        chat_choice: dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "logprobs": _render_logprobs_chat(result, session),
            "finish_reason": result.finish_reason,
        }
        mf_chat = _probe_reading_aggregate(session, result)
        if mf_chat:
            chat_choice["x-saklas-probe-readings"] = mf_chat
        body = {
            "id": rid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [chat_choice],
            "usage": _usage_dict(result),
        }
        compat_probe_readings = _probe_reading_dict(session, readings=result.readings)
        if compat_probe_readings:
            body["probe_readings"] = compat_probe_readings
        return body

    # -----------------------------------------------------------------------
    # Text completions
    # -----------------------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        _check_openai_model_strict(session, req.model)
        rid = _make_id()
        model_id = session.model_id
        gen_kwargs = _sampling_kwargs(req, app.state.default_steering)

        if req.stream:
            stream_iter = session.generate_stream(req.prompt, raw=True, **gen_kwargs)
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return StreamingResponse(
                _stream_generation(session,
                                   stream_iter, rid, model_id,
                                   "text_completion", lambda e: {"text": e.text}, {"text": ""},
                                   include_usage=include_usage, role_delta=False,
                                   request=request),
                media_type="text/event-stream",
            )
        try:
            result = await _run_blocking(req, req.prompt, raw=True)
        except ConcurrentGenerationError:
            return _error(409, "Generation already in progress", "conflict")
        if isinstance(result, JSONResponse):  # bounded-lock timeout → 503
            return result

        completion_choice: dict[str, Any] = {
            "index": 0,
            "text": result.text,
            "logprobs": _render_logprobs_completions(result, session),
            "finish_reason": result.finish_reason,
        }
        mf_completion = _probe_reading_aggregate(session, result)
        if mf_completion:
            completion_choice["x-saklas-probe-readings"] = mf_completion
        body = {
            "id": rid,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [completion_choice],
            "usage": _usage_dict(result),
        }
        compat_probe_readings = _probe_reading_dict(session, readings=result.readings)
        if compat_probe_readings:
            body["probe_readings"] = compat_probe_readings
        return body
