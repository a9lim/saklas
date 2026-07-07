"""Ollama-compatible API routes backed by a SaklasSession.

Mounts `/api/*` alongside the OpenAI-compatible routes so any Ollama client
(Open WebUI, Enchanted, Msty, ollama-python, LangChain's ChatOllama, etc.) can
talk to saklas as a drop-in replacement. Steering passes through a non-standard
`steer` field inside the request's `options` block, so clients that don't know
about it pass through unchanged.

Key differences from real Ollama:
- A saklas server hosts exactly one model. `/api/tags` advertises it under its
  HF repo id *and* any recognized Ollama alias; the `model` field on requests
  is accepted but not strictly validated against the loaded session.
- `/api/pull`, `/api/push`, `/api/create`, `/api/copy`, `/api/delete` are
  stubbed — saklas doesn't manage models the way Ollama does. Pull is a no-op
  success for the currently-loaded model and a 404 otherwise.
- `/api/embeddings` / `/api/embed` return 501 (not implemented).
- Streaming responses are NDJSON (one JSON object per line, `\n`-terminated),
  matching Ollama. Media type: `application/x-ndjson`.
"""

from __future__ import annotations

from saklas.server.app import acquire_session_lock
from saklas.server.request_helpers import (
    build_sampling_config,
    flatten_content,
    merge_steering,
    parse_request_steering,
    probe_reading_aggregate,
    probe_token_readings,
    strict_model_enabled,
)
from saklas.server.streaming import stream_finalizer

import hashlib
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from saklas.core.errors import SaklasError
from saklas.core.session import ConcurrentGenerationError, SaklasSession
from saklas.core.steering import Steering
from saklas.server.model_names import aliases_for_session, known_model_names

log = logging.getLogger(__name__)


class OllamaBadRequest(ValueError, SaklasError):
    """400-grade Ollama compatibility error for malformed request fields."""

    def user_message(self) -> tuple[int, str]:
        return 400, str(self)


def _aliases_for(session: SaklasSession) -> list[str]:
    """Backward-compatible wrapper for tests/imports; new code uses model_names."""
    return aliases_for_session(session)


def _known_model_names(session: SaklasSession) -> set[str]:
    """Backward-compatible wrapper for tests/imports; new code uses model_names."""
    return known_model_names(session)


def _digest_of(name: str) -> str:
    """Deterministic sha256-style digest for a model identifier."""
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    """ISO 8601 timestamp with microseconds and trailing Z, matching Ollama."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _estimate_bytes(session: SaklasSession) -> int:
    info = session.model_info
    params = int(info.get("param_count", 0) or 0)
    dtype = str(info.get("dtype", ""))
    bytes_per = 4 if "float32" in dtype else (1 if "int8" in dtype or "fp8" in dtype else 2)
    return params * bytes_per


def _param_size_label(session: SaklasSession) -> str:
    params = int(session.model_info.get("param_count", 0) or 0)
    if params <= 0:
        return ""
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.1f}B"
    if params >= 1_000_000:
        return f"{params / 1_000_000:.0f}M"
    return str(params)


def _quant_label(session: SaklasSession) -> str:
    dtype = str(session.model_info.get("dtype", "")).lower()
    if "int4" in dtype or "4bit" in dtype:
        return "Q4_0"
    if "int8" in dtype or "8bit" in dtype:
        return "Q8_0"
    if "bfloat16" in dtype or "bf16" in dtype:
        return "BF16"
    if "float16" in dtype or "fp16" in dtype:
        return "F16"
    if "float32" in dtype or "fp32" in dtype:
        return "F32"
    return "unknown"


def _model_details(session: SaklasSession) -> dict[str, Any]:
    info = session.model_info
    family = str(info.get("model_type", "unknown"))
    return {
        "parent_model": "",
        "format": "safetensors",
        "family": family,
        "families": [family],
        "parameter_size": _param_size_label(session),
        "quantization_level": _quant_label(session),
    }


def _tag_entries(session: SaklasSession) -> list[dict[str, Any]]:
    """Build the /api/tags list for the currently-loaded model.

    Advertises the HF repo id as the canonical name plus any recognized Ollama
    aliases, so clients picking from a dropdown see familiar names.
    """
    model_id = session.model_id
    details = _model_details(session)
    size = _estimate_bytes(session)
    modified = _now_iso()
    digest = _digest_of(model_id)

    names = [model_id]
    names.extend(_aliases_for(session))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique = [n for n in names if not (n in seen or seen.add(n))]

    return [
        {
            "name": name,
            "model": name,
            "modified_at": modified,
            "size": size,
            "digest": digest,
            "details": details,
        }
        for name in unique
    ]


# ---------------------------------------------------------------------------
# Option / message translation
# ---------------------------------------------------------------------------

def _extract_messages(body: dict[str, Any]) -> list[dict[str, str]]:
    raw = body.get("messages") or []
    out: list[dict[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        out.append({
            "role": str(m.get("role", "user")),
            "content": flatten_content(m.get("content")),
        })
    return out


_PROCESSED_OPTIONS: frozenset[str] = frozenset({
    "temperature", "top_p", "top_k", "seed", "num_predict",
    "stop", "presence_penalty", "frequency_penalty", "repeat_penalty",
    "steer",
})


def _resolve_options(
    body: dict[str, Any], default_steering: "Steering | None",
) -> tuple[dict[str, Any], str | None]:
    """Translate Ollama `options` + top-level fields into session.generate kwargs.

    Returns ``(gen_kwargs, system)``: ``sampling=SamplingConfig``,
    ``steering=Steering|None``, ``thinking=``, ``stateless=True``; ``system``
    is the top-level ``system`` field for the caller to splice into messages.

    Recognized Ollama fields: temperature, top_p, top_k, seed, num_predict,
    stop, presence_penalty, frequency_penalty, repeat_penalty.

    ``repeat_penalty`` maps to ``presence_penalty`` via ``ln(repeat_penalty)``:
    Ollama divides positive logits by repeat_penalty, which is equivalent
    to subtracting ``ln(penalty)`` from the logit.  That matches
    presence_penalty semantics exactly (subtract a constant per seen token,
    independent of count).

    Unrecognized options (min_p, mirostat*, num_ctx, typical_p, etc.) are
    logged at debug level and silently dropped.

    Non-standard saklas fields (accepted at the top level or inside options):
    ``steer`` (a steering expression string), ``think`` (bool).
    """
    raw_options = body.get("options") or {}
    if not isinstance(raw_options, dict):
        raise OllamaBadRequest("Ollama 'options' must be an object")
    opts = dict(raw_options)
    top_system = body.get("system")

    def _number_option(name: str, default: float | None = None) -> float | None:
        raw = opts.get(name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError) as e:
            raise OllamaBadRequest(
                f"Ollama option '{name}' must be a finite number"
            ) from e
        if not math.isfinite(value):
            raise OllamaBadRequest(
                f"Ollama option '{name}' must be a finite number"
            )
        return value

    def _int_option(name: str, raw: Any) -> int | None:
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError) as e:
            raise OllamaBadRequest(
                f"Ollama option '{name}' must be an integer"
            ) from e

    stop_raw = opts.get("stop") or body.get("stop")
    if isinstance(stop_raw, str):
        stop_tuple: tuple[str, ...] | None = (stop_raw,)
    elif isinstance(stop_raw, list):
        stop_tuple = tuple(str(s) for s in stop_raw)
    else:
        stop_tuple = None

    temperature = _number_option("temperature")
    top_p = _number_option("top_p")
    top_k_raw = opts.get("top_k")
    top_k = _int_option("top_k", top_k_raw)
    if top_k is not None and top_k <= 0:
        top_k = None
    max_tokens = (
        opts.get("num_predict")
        if "num_predict" in opts
        else body.get("num_predict")
    )
    seed = opts.get("seed")
    presence_penalty = _number_option("presence_penalty", 0.0) or 0.0
    frequency_penalty = _number_option("frequency_penalty", 0.0) or 0.0
    repeat_raw = opts.get("repeat_penalty")
    if repeat_raw is not None and presence_penalty == 0.0:
        try:
            rp = float(repeat_raw)
            if not math.isfinite(rp):
                raise ValueError
            if rp > 1.0:
                presence_penalty = math.log(rp)
        except (TypeError, ValueError) as e:
            raise OllamaBadRequest(
                "Ollama option 'repeat_penalty' must be a finite number"
            ) from e

    ignored = [k for k in opts if k not in _PROCESSED_OPTIONS]
    if ignored:
        log.debug("ollama: unsupported options dropped: %s", ", ".join(sorted(ignored)))

    # Ollama-unique: `steer` rides `options` or the top level, must be a
    # string (non-string is a clear client error rather than a parse
    # failure).  The string-parse + key-level merge are the shared
    # ``parse_request_steering`` / ``merge_steering`` the OpenAI path uses.
    steer_raw = opts["steer"] if "steer" in opts else body.get("steer")
    if steer_raw is not None and not isinstance(steer_raw, str):
        raise OllamaBadRequest(
            "Ollama 'steer' must be a steering expression string, "
            "e.g. \"0.5 honest + 0.3 warm\""
        )
    req_steering, explicit_clear = parse_request_steering(steer_raw)

    # Ollama-unique thinking precedence: the steer expression's flag is
    # the base, the top-level ``think`` bool wins when present.
    thinking: bool | None = None
    if req_steering is not None and req_steering.thinking is not None:
        thinking = req_steering.thinking
    think_flag = body.get("think")
    if think_flag is not None:
        thinking = bool(think_flag)

    sc = build_sampling_config(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=_int_option("num_predict", max_tokens),
        seed=seed,
        stop=stop_tuple,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    steering = merge_steering(
        req_steering, default_steering, explicit_clear, thinking,
    )

    gen_kwargs = {
        "sampling": sc,
        "steering": steering,
        # None = auto (honours supports_thinking); explicit True/False wins.
        "thinking": thinking,
        "stateless": True,
    }
    system = top_system if isinstance(top_system, str) else None
    return gen_kwargs, system


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------

def _duration_stats(result: Any, elapsed_ns: int) -> dict[str, int]:
    """Build Ollama's *_duration and *_count fields from a GenerationResult.

    All durations are in nanoseconds.  Saklas tracks tokens/sec so we split the
    measured elapsed time proportionally between prompt-eval and eval.
    """
    prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(result, "token_count", 0) or 0)
    total = max(prompt_tokens + completion_tokens, 1)
    prompt_ns = elapsed_ns * prompt_tokens // total
    eval_ns = max(elapsed_ns - prompt_ns, 1)
    return {
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": prompt_ns,
        "eval_count": completion_tokens,
        "eval_duration": eval_ns,
    }


def _finish_to_done_reason(finish_reason: str | None) -> str:
    if finish_reason == "length":
        return "length"
    if finish_reason == "stop_sequence":
        return "stop"
    return finish_reason or "stop"


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_ollama_routes(app: FastAPI) -> None:
    """Mount /api/* Ollama-compatible routes onto an existing FastAPI app.

    Uses app.state.session + app.state.default_steering, and serializes on
    ``session.lock`` shared with the OpenAI routes.  Auth is inherited via
    app-level Depends(_require_auth) from the parent create_app.
    """
    session: SaklasSession = app.state.session

    # -----------------------------------------------------------------------
    # Trivial endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/version")
    def api_version():
        try:
            from saklas import __version__
        except Exception:
            __version__ = "0.0.0"
        # Advertise an Ollama-ish version so strict clients don't balk.
        return {"version": f"saklas-{__version__}"}

    @app.get("/api/tags")
    def api_tags():
        return {"models": _tag_entries(session)}

    @app.get("/api/ps")
    def api_ps():
        return {
            "models": [
                {
                    **entry,
                    "expires_at": "9999-12-31T23:59:59Z",
                    "size_vram": int((session.model_info.get("vram_used_gb") or 0) * 1024**3),
                }
                for entry in _tag_entries(session)
            ]
        }

    @app.post("/api/show")
    async def api_show(request: Request):
        body = await request.json()
        name = body.get("model") or body.get("name") or session.model_id
        info = session.model_info
        details = _model_details(session)
        # Reflect the real HF chat template when available.  Ollama expects
        # Go-template syntax while HF uses Jinja — clients that parse this
        # will fail either way, so returning the honest Jinja template is
        # more useful than the meaningless "{{ .Prompt }}" placeholder.
        tpl = getattr(session.tokenizer, "chat_template", None) or "{{ .Prompt }}"
        return {
            "license": "See upstream model card.",
            "modelfile": f"# saklas: {session.model_id}\nFROM {session.model_id}\n",
            "parameters": "",
            "template": tpl,
            "details": details,
            "model_info": {
                "general.architecture": info.get("model_type", "unknown"),
                "general.parameter_count": info.get("param_count", 0),
                "general.quantization_version": 0,
                f"{info.get('model_type', 'unknown')}.block_count": info.get("num_layers", 0),
                f"{info.get('model_type', 'unknown')}.embedding_length": info.get("hidden_dim", 0),
                "saklas.loaded_model": session.model_id,
                "saklas.requested_name": name,
            },
            "capabilities": ["completion", "chat"],
        }

    # -----------------------------------------------------------------------
    # Stubs for model-management endpoints saklas doesn't implement
    # -----------------------------------------------------------------------

    @app.post("/api/pull")
    async def api_pull(request: Request):
        # No-op success if the client is asking for the loaded model (by HF id
        # or by any recognized alias); 404 otherwise.  Saklas loads models at
        # startup, so a true pull is out of scope.
        body = await request.json()
        name = str(body.get("model") or body.get("name") or "")
        if name and name.lower() not in _known_model_names(session):
            hosted = ", ".join(sorted({session.model_id, *_aliases_for(session)}))
            return JSONResponse(status_code=404, content={
                "error": (
                    f"model '{name}' not found. saklas currently hosts: {hosted}. "
                    f"To serve a different model, restart with: saklas serve <model>"
                ),
            })

        async def _stream():
            yield json.dumps({"status": "pulling manifest"}) + "\n"
            yield json.dumps({"status": "success"}) + "\n"
        return StreamingResponse(_stream(), media_type="application/x-ndjson")

    @app.post("/api/push")
    async def api_push():
        return JSONResponse(status_code=501, content={
            "error": "saklas does not implement /api/push. Use `saklas pack push` to publish a manifold.",
        })

    @app.post("/api/create")
    async def api_create():
        return JSONResponse(status_code=501, content={
            "error": "saklas does not implement /api/create.",
        })

    @app.post("/api/copy")
    async def api_copy():
        return JSONResponse(status_code=501, content={
            "error": "saklas does not implement /api/copy.",
        })

    @app.delete("/api/delete")
    async def api_delete():
        return JSONResponse(status_code=501, content={
            "error": "saklas does not implement /api/delete.",
        })

    @app.post("/api/embeddings")
    async def api_embeddings():
        return JSONResponse(status_code=501, content={
            "error": "saklas does not implement /api/embeddings. Use the model's native embedding API.",
        })

    @app.post("/api/embed")
    async def api_embed():
        return JSONResponse(status_code=501, content={
            "error": "saklas does not implement /api/embed.",
        })

    @app.head("/")
    def api_head_root():
        # Ollama clients hit HEAD / to probe liveness.  A HEAD response
        # carries no body, so return a bodyless 200 rather than a JSON
        # payload (which advertised a bogus Content-Length / content-type).
        return Response(status_code=200)

    # -----------------------------------------------------------------------
    # Generation endpoints
    # -----------------------------------------------------------------------

    def _check_model_or_404(body: dict[str, Any]) -> None:
        """In strict mode, reject requests whose `model` doesn't match the loaded session."""
        if not strict_model_enabled():
            return
        name = str(body.get("model") or "")
        if name and name.lower() not in _known_model_names(session):
            hosted = ", ".join(sorted({session.model_id, *_aliases_for(session)}))
            raise HTTPException(
                status_code=404,
                detail=(
                    f"model '{name}' not available. saklas hosts: {hosted}. "
                    f"Unset SAKLAS_STRICT_MODEL to accept any model name."
                ),
            )

    async def _run_and_build_chat_response(
        body: dict[str, Any],
        is_chat: bool,
        gen_kwargs: dict[str, Any],
        system: str | None,
    ) -> dict[str, Any] | JSONResponse:
        """Shared non-streaming path for /api/chat and /api/generate.

        Option resolution is hoisted to the route handler so any
        ``SaklasError`` from ``parse_expr`` (bad/colliding ``steer``
        expression) reaches FastAPI's ``@app.exception_handler`` cleanly
        before this helper runs.
        """
        if is_chat:
            msgs = _extract_messages(body)
            if system:
                msgs = [{"role": "system", "content": system}, *msgs]
            input_payload: Any = msgs
            raw = False
        else:
            prompt = flatten_content(body.get("prompt", ""))
            if system:
                # /api/generate's `system` field belongs at the top of the
                # chat template.  Route through the chat path to honour it.
                input_payload = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
                raw = False
            else:
                input_payload = prompt
                raw = bool(body.get("raw", False))

        start_ns = time.monotonic_ns()
        # Bounded lock so a non-streaming request can't queue forever behind a
        # stuck generation — it 503s like the streaming path does.  The route
        # handler returns this ``JSONResponse`` verbatim.
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                return JSONResponse(
                    status_code=503,
                    content={
                        "model": str(body.get("model") or session.model_id),
                        "created_at": _now_iso(),
                        "error": "server busy",
                    },
                )
            try:
                result = session.generate(input_payload, raw=raw, **gen_kwargs)
            except ConcurrentGenerationError as e:
                raise HTTPException(
                    status_code=409, detail="Generation already in progress",
                ) from e
        elapsed_ns = time.monotonic_ns() - start_ns

        model_name = str(body.get("model") or session.model_id)
        created_at = _now_iso()
        done_reason = _finish_to_done_reason(getattr(result, "finish_reason", None))
        stats = _duration_stats(result, elapsed_ns)

        # Saklas-specific extension: per-attached-manifold-probe aggregate
        # readings ride alongside the Ollama wire fields under a vendor-
        # prefixed top-level key so Ollama clients that don't read it stay
        # unaffected.  Mirrors the OpenAI extension shape on the choice
        # ("x-saklas-probe-readings"), at the top level here because
        # Ollama responses have no per-choice container to hang it off.
        mf_agg = probe_reading_aggregate(session, result)

        if is_chat:
            payload: dict[str, Any] = {
                "model": model_name,
                "created_at": created_at,
                "message": {"role": "assistant", "content": result.text},
                "done_reason": done_reason,
                "done": True,
                **stats,
            }
            if mf_agg:
                payload["x-saklas-probe-readings"] = mf_agg
            return payload
        # Note: Ollama's /api/generate returns a `context` field of tokenized
        # state for stateless continuation.  Saklas doesn't round-trip that,
        # so we omit the field entirely rather than lie with an empty list.
        payload = {
            "model": model_name,
            "created_at": created_at,
            "response": result.text,
            "done_reason": done_reason,
            "done": True,
            **stats,
        }
        if mf_agg:
            payload["x-saklas-probe-readings"] = mf_agg
        return payload

    async def _stream_chat_or_generate(
        body: dict[str, Any],
        is_chat: bool,
        gen_kwargs: dict[str, Any],
        system: str | None,
        request: Request | None = None,
    ):
        """Streaming NDJSON body for /api/chat and /api/generate.

        ``gen_kwargs`` and ``system`` are resolved by the caller (the
        route handler) so any ``SaklasError`` from ``parse_expr`` raises
        *before* ``StreamingResponse`` flushes headers. Materialization
        errors can still happen when generation starts, so the iterator
        also converts ``SaklasError`` in-band instead of cutting the TCP
        stream off mid-flight.

        ``request`` is threaded in from the route handler so we can bail
        early on client disconnect (``is_disconnected()``).  Library and
        test callers that don't have a live request pass ``None`` — the
        disconnect check is simply skipped.
        """
        if is_chat:
            msgs = _extract_messages(body)
            if system:
                msgs = [{"role": "system", "content": system}, *msgs]
            input_payload: Any = msgs
            raw = False
        else:
            prompt = flatten_content(body.get("prompt", ""))
            if system:
                input_payload = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
                raw = False
            else:
                input_payload = prompt
                raw = bool(body.get("raw", False))

        model_name = str(body.get("model") or session.model_id)

        async with acquire_session_lock(session) as acquired:
            if not acquired:
                yield json.dumps({
                    "model": model_name, "created_at": _now_iso(),
                    "error": "server busy",
                }) + "\n"
                return

            # One timestamp for the whole stream, reused on every chunk —
            # matching the non-streaming path's single ``_now_iso()`` call.
            created_at = _now_iso()
            start_ns = time.monotonic_ns()
            stream_iter = None
            try:
                stream_iter = session.generate_stream(
                    input_payload, raw=raw, live_scores=False, **gen_kwargs,
                )
                for event in stream_iter:
                    # Bail out if the client has hung up — close the inner
                    # generator (handled in ``finally``) and stop spending
                    # the GPU on tokens nobody is reading.
                    if request is not None and await request.is_disconnected():
                        return
                    if event.thinking:
                        # Ollama doesn't standardize a reasoning channel;
                        # the canonical shape uses a `thinking` field on
                        # the message.  Non-Ollama clients ignore it.
                        if is_chat:
                            chunk = {
                                "model": model_name,
                                "created_at": created_at,
                                "message": {"role": "assistant", "content": "",
                                            "thinking": event.text},
                                "done": False,
                            }
                        else:
                            chunk = {
                                "model": model_name,
                                "created_at": created_at,
                                "response": "",
                                "thinking": event.text,
                                "done": False,
                            }
                    else:
                        if is_chat:
                            chunk = {
                                "model": model_name,
                                "created_at": created_at,
                                "message": {"role": "assistant", "content": event.text},
                                "done": False,
                            }
                        else:
                            chunk = {
                                "model": model_name,
                                "created_at": created_at,
                                "response": event.text,
                                "done": False,
                            }
                    # Per-token manifold readings ride under the same
                    # vendor-prefixed extension as the non-streaming
                    # path; populated only when at least one manifold
                    # probe is attached and ``live_scores`` is on.
                    mf_token = probe_token_readings(event)
                    if mf_token is not None:
                        chunk["x-saklas-probe-readings"] = mf_token
                    yield json.dumps(chunk) + "\n"
            except ConcurrentGenerationError:
                yield json.dumps({
                    "model": model_name, "created_at": created_at,
                    "error": "Generation already in progress",
                }) + "\n"
                # Terminating done frame so ollama-python / ChatOllama don't
                # stall waiting for it after an in-band error chunk.
                yield json.dumps({
                    "model": model_name, "created_at": created_at,
                    "done": True, "done_reason": "error",
                }) + "\n"
                return
            except SaklasError as e:
                _status, msg = e.user_message()
                yield json.dumps({
                    "model": model_name,
                    "created_at": created_at,
                    "error": msg,
                }) + "\n"
                # Terminating done frame so ollama-python / ChatOllama don't
                # stall waiting for it after an in-band error chunk.
                yield json.dumps({
                    "model": model_name, "created_at": created_at,
                    "done": True, "done_reason": "error",
                }) + "\n"
                return
            finally:
                # Deterministically tear down the engine worker thread
                # (stop-flag + join) on every exit — normal completion
                # (no-op on an exhausted generator), an in-band error, or
                # an early client-disconnect ``return`` — rather than
                # leaving it to GC.
                close = getattr(stream_iter, "close", None)
                if callable(close):
                    close()

            elapsed_ns = time.monotonic_ns() - start_ns
            result = (
                getattr(stream_iter, "result", None)
                or getattr(session, "last_result", None)
            )
            finish_reason, _usage, mf_agg = stream_finalizer(session, result)
            done_reason = _finish_to_done_reason(finish_reason)
            stats = _duration_stats(result, elapsed_ns) if result is not None else {
                "total_duration": elapsed_ns, "load_duration": 0,
                "prompt_eval_count": 0, "prompt_eval_duration": 0,
                "eval_count": 0, "eval_duration": elapsed_ns,
            }
            if is_chat:
                final = {
                    "model": model_name,
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": ""},
                    "done_reason": done_reason,
                    "done": True,
                    **stats,
                }
            else:
                # See note in _run_and_build_chat_response: `context` is
                # omitted because saklas can't round-trip it honestly.
                final = {
                    "model": model_name,
                    "created_at": created_at,
                    "response": "",
                    "done_reason": done_reason,
                    "done": True,
                    **stats,
                }
            if mf_agg:
                final["x-saklas-probe-readings"] = mf_agg
            yield json.dumps(final) + "\n"

    @app.post("/api/chat")
    async def api_chat(request: Request):
        body = await request.json()
        _check_model_or_404(body)
        # Resolve options (including the steering expression) here so a
        # ``SaklasError`` from ``parse_expr`` is caught by FastAPI's
        # exception handler and returned as a clean Ollama-shape 400.
        # Once ``StreamingResponse`` flushes headers the handler can't
        # rewrite the response — we'd disconnect mid-stream.
        gen_kwargs, system = _resolve_options(body, app.state.default_steering)
        if body.get("stream", True):
            return StreamingResponse(
                _stream_chat_or_generate(
                    body, is_chat=True, gen_kwargs=gen_kwargs, system=system,
                    request=request,
                ),
                media_type="application/x-ndjson",
            )
        return await _run_and_build_chat_response(
            body, is_chat=True, gen_kwargs=gen_kwargs, system=system,
        )

    @app.post("/api/generate")
    async def api_generate(request: Request):
        body = await request.json()
        _check_model_or_404(body)
        gen_kwargs, system = _resolve_options(body, app.state.default_steering)
        if body.get("stream", True):
            return StreamingResponse(
                _stream_chat_or_generate(
                    body, is_chat=False, gen_kwargs=gen_kwargs, system=system,
                    request=request,
                ),
                media_type="application/x-ndjson",
            )
        return await _run_and_build_chat_response(
            body, is_chat=False, gen_kwargs=gen_kwargs, system=system,
        )
