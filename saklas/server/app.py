"""OpenAI-compatible API server backed by SaklasSession."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress
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

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def _flatten_content(content: Any) -> str:
    """Concatenate the text parts of an OpenAI multimodal content array.

    Accepts a string (passed through), a list of content parts (each a
    ``{"type": "text", "text": ...}`` dict or a bare string — non-text
    parts raise :class:`UnsupportedContentError`), ``None`` (→ ``""``,
    the Ollama convention), or any other scalar (stringified).  Shared
    by ``ChatMessage._flatten_content`` (OpenAI routes) and the Ollama
    shim's message/prompt extraction.
    """
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
            elif isinstance(part, str):
                pieces.append(part)
            else:
                raise UnsupportedContentError(
                    "non-text content parts are not supported by this model"
                )
        return "".join(pieces)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


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


def _manifold_reading_aggregate(session: SaklasSession) -> dict[str, Any]:
    """Per-attached-probe ``ManifoldAggregate.to_dict()`` from ``_last_result``.

    Returns ``{}`` when no result is recorded or no manifold probes are
    attached.  Surfaced under the ``x-saklas-manifold-readings`` extension
    on OpenAI and Ollama responses so vector-probe clients keep working
    unchanged and manifold-aware clients pick up the geometric channel.
    """
    result = getattr(session, "_last_result", None)
    if result is None:
        return {}
    readings = getattr(result, "manifold_readings", None) or {}
    if not readings:
        return {}
    try:
        attached = set(session._manifold_monitor.probe_names)
    except Exception:
        attached = set(readings.keys())
    out: dict[str, Any] = {}
    for name, agg in readings.items():
        if name not in attached:
            continue
        with suppress(Exception):
            out[name] = agg.to_dict()
    return out


def _manifold_token_readings(event: Any) -> dict[str, Any] | None:
    """Serialize a :class:`TokenEvent`'s ``manifold_readings`` for the wire.

    Returns ``None`` when the event carries no manifold readings (no
    probes attached, or ``live_scores=False`` was passed to
    ``generate_stream``).  Used by both OpenAI and Ollama streaming
    paths so the per-token geometric channel rides on each chunk
    without breaking clients that ignore the field.
    """
    readings = getattr(event, "manifold_readings", None)
    if not readings:
        return None
    out: dict[str, Any] = {}
    for name, reading in readings.items():
        with suppress(Exception):
            out[name] = reading.to_dict()
    return out or None


def _parse_req_steering(
    expr: str | None,
) -> tuple["Steering | None", bool]:
    """Parse a per-request steering expression string.

    Returns ``(req_steering, explicit_clear)``: ``None`` expression
    inherits the server default (``explicit_clear=False``); an explicit
    empty / whitespace string is a clear request (``explicit_clear=True``,
    ``req_steering=None``); a non-empty string parses through the shared
    grammar.  Shared by the OpenAI and Ollama route families.
    """
    from saklas.core.steering_expr import parse_expr

    if expr is None:
        return None, False
    if not expr.strip():
        return None, True
    return parse_expr(expr), False


def _merge_steering(
    req_steering: "Steering | None",
    default_steering: "Steering | None",
    explicit_clear: bool,
    thinking: bool | None,
) -> "Steering | None":
    """Compose a parsed request steering over the server default.

    Per-request keys override the default at the key level; default-only
    keys pass through; ``explicit_clear`` drops the default entirely.
    Returns ``None`` when the composed alphas are empty and no ``thinking``
    override is in play.  Pole aliasing happens inside ``session.steering()``
    — the server does not resolve poles here.  Each protocol resolves its
    own ``thinking`` precedence (OpenAI's native field, Ollama's top-level
    ``think``) before calling in, since the sources differ.
    """
    merged_alphas: dict[str, Any] = {}
    if default_steering is not None and not explicit_clear:
        merged_alphas.update(default_steering.alphas)
    if req_steering is not None:
        merged_alphas.update(req_steering.alphas)
    if not merged_alphas and thinking is None:
        return None
    return Steering(alphas=merged_alphas, thinking=thinking)


def _build_sampling_config(
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None = None,
    max_tokens: int | None,
    seed: int | None,
    stop: tuple[str, ...] | None,
    logit_bias: dict[int, float] | None = None,
    presence_penalty: float,
    frequency_penalty: float,
    logprobs: int | None = None,
) -> SamplingConfig:
    """Build a :class:`SamplingConfig` from already-normalized fields.

    Shared by the OpenAI and Ollama route families.  Each protocol does
    its own field normalization upstream (OpenAI: logprobs bool/int
    coercion, no ``top_k``; Ollama: ``num_predict`` → ``max_tokens``,
    ``repeat_penalty`` → ``presence_penalty`` via ``ln``, ``top_k``) and
    hands the result here so the construction lives in one place.
    """
    return SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
        logit_bias=logit_bias,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logprobs=logprobs,
    )


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


def _usage_dict(result: GenerationResult) -> dict[str, int]:
    pt = result.prompt_tokens
    ct = result.token_count
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


def _token_bytes(text: str) -> list[int]:
    try:
        return list(text.encode("utf-8"))
    except Exception:
        return []


def _render_logprobs_chat(result: GenerationResult, session: SaklasSession) -> dict[str, Any] | None:
    if result.logprobs is None:
        return None
    tok = session._tokenizer
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
    tok = session._tokenizer
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
    app: FastAPI, session: SaklasSession,
    stream_iter: Iterator[Any], rid: str, model_id: str, object_type: str,
    format_delta: Callable[[Any], dict[str, Any]], empty_delta: dict[str, Any],
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

        if role_delta:
            chunk = {
                "id": rid, "object": object_type, "created": created_ts,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        try:
            for event in stream_iter:
                choice: dict[str, Any] = {
                    "index": 0, **format_delta(event), "finish_reason": None,
                }
                # Per-token manifold readings ride under a vendor-
                # prefixed extension on the choice so OpenAI clients
                # that don't read the field stay unaffected.  Populated
                # only when at least one manifold probe is attached
                # and ``live_scores`` is True on the stream.
                mf_token = _manifold_token_readings(event)
                if mf_token is not None:
                    choice["x-saklas-manifold-readings"] = mf_token
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

        finish_reason = session._gen_state.finish_reason
        final_choice: dict[str, Any] = {
            "index": 0, **empty_delta, "finish_reason": finish_reason,
        }
        mf_agg = _manifold_reading_aggregate(session)
        if mf_agg:
            final_choice["x-saklas-manifold-readings"] = mf_agg
        final = {
            "id": rid,
            "object": object_type,
            "created": created_ts,
            "model": model_id,
            "choices": [final_choice],
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

    async def _run_blocking(req: _SamplingBase, prompt_or_messages: Any, *, raw: bool) -> Any:
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

        chat_choice: dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "logprobs": _render_logprobs_chat(result, session),
            "finish_reason": result.finish_reason,
        }
        mf_chat = _manifold_reading_aggregate(session)
        if mf_chat:
            chat_choice["x-saklas-manifold-readings"] = mf_chat
        return {
            "id": rid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [chat_choice],
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

        completion_choice: dict[str, Any] = {
            "index": 0,
            "text": result.text,
            "logprobs": _render_logprobs_completions(result, session),
            "finish_reason": result.finish_reason,
        }
        mf_completion = _manifold_reading_aggregate(session)
        if mf_completion:
            completion_choice["x-saklas-manifold-readings"] = mf_completion
        return {
            "id": rid,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [completion_choice],
            "usage": _usage_dict(result),
            "probe_readings": _probe_reading_dict(session),
        }
