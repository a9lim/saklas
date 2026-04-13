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

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from saklas.session import ConcurrentGenerationError, SaklasSession


# ---------------------------------------------------------------------------
# Model-name aliasing
# ---------------------------------------------------------------------------
#
# Ollama model names are short and loose (`llama3.2`, `qwen2.5:7b`).  HF repo
# ids are precise (`meta-llama/Llama-3.2-3B-Instruct`).  This table records a
# few popular mappings so Ollama clients that pick a model from /api/tags get
# meaningful aliases advertised for the currently-loaded session.
#
# The mapping is *advisory*: client requests may set `model` to anything, and
# saklas always generates with the loaded session regardless.  The table only
# affects what /api/tags and /api/show advertise.

_HF_TO_OLLAMA_ALIASES: dict[str, list[str]] = {
    # Llama 3.x
    "meta-llama/Llama-3.2-1B-Instruct": ["llama3.2:1b", "llama3.2:1b-instruct"],
    "meta-llama/Llama-3.2-3B-Instruct": ["llama3.2", "llama3.2:latest", "llama3.2:3b"],
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ["llama3.1", "llama3.1:latest", "llama3.1:8b"],
    # Qwen
    "Qwen/Qwen2.5-0.5B-Instruct": ["qwen2.5:0.5b"],
    "Qwen/Qwen2.5-1.5B-Instruct": ["qwen2.5:1.5b"],
    "Qwen/Qwen2.5-3B-Instruct": ["qwen2.5:3b"],
    "Qwen/Qwen2.5-7B-Instruct": ["qwen2.5", "qwen2.5:latest", "qwen2.5:7b"],
    "Qwen/Qwen3-4B-Instruct": ["qwen3:4b"],
    "Qwen/Qwen3-8B": ["qwen3", "qwen3:latest", "qwen3:8b"],
    # Gemma
    "google/gemma-2-2b-it": ["gemma2:2b"],
    "google/gemma-2-9b-it": ["gemma2", "gemma2:latest", "gemma2:9b"],
    "google/gemma-3-4b-it": ["gemma3", "gemma3:latest", "gemma3:4b"],
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.3": ["mistral", "mistral:latest", "mistral:7b"],
    "mistralai/Ministral-8B-Instruct-2410": ["ministral:8b"],
    # Phi
    "microsoft/Phi-3.5-mini-instruct": ["phi3.5", "phi3.5:latest"],
}


def _aliases_for(model_id: str) -> list[str]:
    """Return Ollama-style aliases for an HF repo id (empty list if none)."""
    return list(_HF_TO_OLLAMA_ALIASES.get(model_id, []))


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
    names.extend(_aliases_for(model_id))
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

def _flatten_content(content: Any) -> str:
    """Ollama allows content to be a string or a list of text parts."""
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                pieces.append(str(part["text"]))
            elif isinstance(part, str):
                pieces.append(part)
        return "".join(pieces)
    if content is None:
        return ""
    return str(content)


def _extract_messages(body: dict) -> list[dict[str, str]]:
    raw = body.get("messages") or []
    out: list[dict[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        out.append({
            "role": str(m.get("role", "user")),
            "content": _flatten_content(m.get("content")),
        })
    return out


def _resolve_options(body: dict, default_alphas: dict[str, float]) -> dict[str, Any]:
    """Translate Ollama `options` + top-level fields into saklas gen kwargs.

    Recognized Ollama fields: temperature, top_p, top_k (ignored), seed,
    num_predict, stop, presence_penalty, frequency_penalty, repeat_penalty
    (mapped to frequency_penalty if frequency_penalty is unset).

    Non-standard saklas fields (accepted at the top level or inside options):
    `steer` (dict or dict with alphas/orthogonalize/thinking), `think` (bool).
    """
    opts = dict(body.get("options") or {})
    top_system = body.get("system")

    stop_raw = opts.get("stop") or body.get("stop")
    if isinstance(stop_raw, str):
        stop_list: list[str] | None = [stop_raw]
    elif isinstance(stop_raw, list):
        stop_list = [str(s) for s in stop_raw]
    else:
        stop_list = None

    temperature = opts.get("temperature")
    top_p = opts.get("top_p")
    max_tokens = opts.get("num_predict") or body.get("num_predict")
    seed = opts.get("seed")
    presence_penalty = float(opts.get("presence_penalty", 0.0) or 0.0)
    frequency_penalty = opts.get("frequency_penalty")
    if frequency_penalty is None:
        repeat = opts.get("repeat_penalty")
        # Ollama's repeat_penalty is multiplicative; map values > 1 into a
        # modest additive frequency_penalty. 1.1 -> 0.1, 1.3 -> 0.3.
        frequency_penalty = max(0.0, float(repeat) - 1.0) if repeat is not None else 0.0

    steer_raw = opts.get("steer") or body.get("steer") or {}
    if isinstance(steer_raw, dict):
        if "alphas" in steer_raw and isinstance(steer_raw["alphas"], dict):
            alpha_map = {str(k): float(v) for k, v in steer_raw["alphas"].items()}
            orthogonalize = bool(steer_raw.get("orthogonalize", False))
            thinking = bool(steer_raw.get("thinking", False))
        else:
            # Allow `"steer": {"happy": 0.3, "calm": 0.2}` shorthand.
            alpha_map = {}
            orthogonalize = False
            thinking = False
            for k, v in steer_raw.items():
                try:
                    alpha_map[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
    else:
        alpha_map = {}
        orthogonalize = False
        thinking = False

    think_flag = body.get("think")
    if think_flag is not None:
        thinking = bool(think_flag)

    merged_alphas = {**default_alphas, **alpha_map}
    merged_alphas = {k: v for k, v in merged_alphas.items() if v != 0.0}

    return {
        "alphas": merged_alphas or None,
        "orthogonalize": orthogonalize,
        "thinking": thinking,
        "stateless": True,
        "seed": seed,
        "stop": stop_list,
        "presence_penalty": presence_penalty,
        "frequency_penalty": float(frequency_penalty),
        "logit_bias": None,
        "logprobs": None,
        "_temperature": temperature,
        "_top_p": top_p,
        "_max_tokens": max_tokens,
        "_system": top_system if isinstance(top_system, str) else None,
    }


def _split_overrides(gen_kwargs: dict[str, Any]) -> tuple[dict[str, Any], Any, Any, Any, str | None]:
    """Pop private override fields out of the gen kwargs."""
    temperature = gen_kwargs.pop("_temperature", None)
    top_p = gen_kwargs.pop("_top_p", None)
    max_tokens = gen_kwargs.pop("_max_tokens", None)
    system = gen_kwargs.pop("_system", None)
    return gen_kwargs, temperature, top_p, max_tokens, system


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------

def _duration_stats(result, elapsed_ns: int) -> dict[str, int]:
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

    Uses app.state.session, app.state.default_alphas, app.state.gen_lock — the
    same state created_app prepared for the OpenAI routes.  Auth is inherited
    via app-level Depends(_require_auth) from the parent create_app.
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
        entries = []
        for entry in _tag_entries(session):
            entries.append({
                **entry,
                "expires_at": "9999-12-31T23:59:59Z",
                "size_vram": int((session.model_info.get("vram_used_gb") or 0) * 1024**3),
            })
        return {"models": entries}

    @app.post("/api/show")
    async def api_show(request: Request):
        body = await request.json()
        name = body.get("model") or body.get("name") or session.model_id
        info = session.model_info
        details = _model_details(session)
        return {
            "license": "See upstream model card.",
            "modelfile": f"# saklas: {session.model_id}\nFROM {session.model_id}\n",
            "parameters": "",
            "template": "{{ .Prompt }}",
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
        known = {session.model_id, *(_aliases_for(session.model_id))}
        if name and name not in known:
            return JSONResponse(status_code=404, content={
                "error": f"model '{name}' not found. saklas currently hosts '{session.model_id}'.",
            })

        async def _stream():
            yield json.dumps({"status": "pulling manifest"}) + "\n"
            yield json.dumps({"status": "success"}) + "\n"
        return StreamingResponse(_stream(), media_type="application/x-ndjson")

    @app.post("/api/push")
    async def api_push():
        return JSONResponse(status_code=501, content={
            "error": "saklas does not implement /api/push. Use `saklas push` for concept packs.",
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
        # Ollama clients hit HEAD / to probe liveness.
        return JSONResponse(status_code=200, content=None)

    # -----------------------------------------------------------------------
    # Generation endpoints
    # -----------------------------------------------------------------------

    from saklas.server import _gen_config_override  # reuse the context manager

    async def _run_and_build_chat_response(body: dict, is_chat: bool) -> dict:
        """Shared non-streaming path for /api/chat and /api/generate."""
        default_alphas = app.state.default_alphas
        gen_kwargs = _resolve_options(body, default_alphas)
        gen_kwargs, temperature, top_p, max_tokens, system = _split_overrides(gen_kwargs)

        if is_chat:
            msgs = _extract_messages(body)
            if system:
                msgs = [{"role": "system", "content": system}, *msgs]
            input_payload: Any = msgs
            raw = False
        else:
            prompt = _flatten_content(body.get("prompt", ""))
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

        async with app.state.gen_lock:
            with _gen_config_override(session, temperature, top_p, max_tokens):
                start_ns = time.monotonic_ns()
                try:
                    result = session.generate(input_payload, raw=raw, **gen_kwargs)
                except ConcurrentGenerationError as e:
                    raise HTTPException(status_code=409, detail=str(e))
                elapsed_ns = time.monotonic_ns() - start_ns

        model_name = str(body.get("model") or session.model_id)
        created_at = _now_iso()
        done_reason = _finish_to_done_reason(session._gen_state.finish_reason)
        stats = _duration_stats(result, elapsed_ns)

        if is_chat:
            return {
                "model": model_name,
                "created_at": created_at,
                "message": {"role": "assistant", "content": result.text},
                "done_reason": done_reason,
                "done": True,
                **stats,
            }
        return {
            "model": model_name,
            "created_at": created_at,
            "response": result.text,
            "done_reason": done_reason,
            "done": True,
            "context": [],
            **stats,
        }

    async def _stream_chat_or_generate(body: dict, is_chat: bool):
        default_alphas = app.state.default_alphas
        gen_kwargs = _resolve_options(body, default_alphas)
        gen_kwargs, temperature, top_p, max_tokens, system = _split_overrides(gen_kwargs)

        if is_chat:
            msgs = _extract_messages(body)
            if system:
                msgs = [{"role": "system", "content": system}, *msgs]
            input_payload: Any = msgs
            raw = False
        else:
            prompt = _flatten_content(body.get("prompt", ""))
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

        try:
            async with asyncio.timeout(300):
                await app.state.gen_lock.acquire()
        except (TimeoutError, asyncio.TimeoutError):
            yield json.dumps({
                "model": model_name, "created_at": _now_iso(),
                "error": "server busy",
            }) + "\n"
            return

        try:
            with _gen_config_override(session, temperature, top_p, max_tokens):
                start_ns = time.monotonic_ns()
                try:
                    stream_iter = session.generate_stream(input_payload, raw=raw, **gen_kwargs)
                    for event in stream_iter:
                        if event.thinking:
                            # Ollama doesn't standardize a reasoning channel;
                            # the canonical shape uses a `thinking` field on
                            # the message.  Non-Ollama clients ignore it.
                            if is_chat:
                                chunk = {
                                    "model": model_name,
                                    "created_at": _now_iso(),
                                    "message": {"role": "assistant", "content": "",
                                                "thinking": event.text},
                                    "done": False,
                                }
                            else:
                                chunk = {
                                    "model": model_name,
                                    "created_at": _now_iso(),
                                    "response": "",
                                    "thinking": event.text,
                                    "done": False,
                                }
                        else:
                            if is_chat:
                                chunk = {
                                    "model": model_name,
                                    "created_at": _now_iso(),
                                    "message": {"role": "assistant", "content": event.text},
                                    "done": False,
                                }
                            else:
                                chunk = {
                                    "model": model_name,
                                    "created_at": _now_iso(),
                                    "response": event.text,
                                    "done": False,
                                }
                        yield json.dumps(chunk) + "\n"
                except ConcurrentGenerationError as e:
                    yield json.dumps({
                        "model": model_name, "created_at": _now_iso(),
                        "error": str(e),
                    }) + "\n"
                    return

                elapsed_ns = time.monotonic_ns() - start_ns
                result = session._last_result
                done_reason = _finish_to_done_reason(session._gen_state.finish_reason)
                stats = _duration_stats(result, elapsed_ns) if result is not None else {
                    "total_duration": elapsed_ns, "load_duration": 0,
                    "prompt_eval_count": 0, "prompt_eval_duration": 0,
                    "eval_count": 0, "eval_duration": elapsed_ns,
                }
                if is_chat:
                    final = {
                        "model": model_name,
                        "created_at": _now_iso(),
                        "message": {"role": "assistant", "content": ""},
                        "done_reason": done_reason,
                        "done": True,
                        **stats,
                    }
                else:
                    final = {
                        "model": model_name,
                        "created_at": _now_iso(),
                        "response": "",
                        "done_reason": done_reason,
                        "done": True,
                        "context": [],
                        **stats,
                    }
                yield json.dumps(final) + "\n"
        finally:
            app.state.gen_lock.release()

    @app.post("/api/chat")
    async def api_chat(request: Request):
        body = await request.json()
        if body.get("stream", True):
            return StreamingResponse(
                _stream_chat_or_generate(body, is_chat=True),
                media_type="application/x-ndjson",
            )
        return await _run_and_build_chat_response(body, is_chat=True)

    @app.post("/api/generate")
    async def api_generate(request: Request):
        body = await request.json()
        if body.get("stream", True):
            return StreamingResponse(
                _stream_chat_or_generate(body, is_chat=False),
                media_type="application/x-ndjson",
            )
        return await _run_and_build_chat_response(body, is_chat=False)
