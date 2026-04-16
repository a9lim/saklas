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

import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from saklas.core.sampling import SamplingConfig
from saklas.core.session import ConcurrentGenerationError, SaklasSession
from saklas.core.steering import Steering

log = logging.getLogger(__name__)


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

# Manual overrides for HF ids whose canonical Ollama tags need to match
# Ollama's actual catalogue (e.g. Gemma-2-2b is ~2.6B params but Ollama
# advertises it as `gemma2:2b`), plus cases where we want to advertise the
# `:latest` tag or where model_type lacks the version number (Llama).
# If an HF id appears here, inference is skipped — overrides are authoritative.
_HF_TO_OLLAMA_ALIASES: dict[str, list[str]] = {
    # Llama 3.x — model_type is just "llama", no version suffix to infer from
    "meta-llama/Llama-3.2-1B-Instruct": ["llama3.2:1b", "llama3.2:1b-instruct"],
    "meta-llama/Llama-3.2-3B-Instruct": ["llama3.2", "llama3.2:latest", "llama3.2:3b"],
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ["llama3.1", "llama3.1:latest", "llama3.1:8b"],
    "meta-llama/Llama-3.3-70B-Instruct": ["llama3.3", "llama3.3:latest", "llama3.3:70b"],
    # Qwen — override to match Ollama's rounded size tags
    "Qwen/Qwen2.5-0.5B-Instruct": ["qwen2.5:0.5b"],
    "Qwen/Qwen2.5-1.5B-Instruct": ["qwen2.5:1.5b"],
    "Qwen/Qwen2.5-3B-Instruct": ["qwen2.5:3b"],
    "Qwen/Qwen2.5-7B-Instruct": ["qwen2.5", "qwen2.5:latest", "qwen2.5:7b"],
    "Qwen/Qwen3-4B-Instruct": ["qwen3:4b"],
    "Qwen/Qwen3-8B": ["qwen3", "qwen3:latest", "qwen3:8b"],
    # Gemma — Ollama advertises rounded sizes (2b not 2.6b, 9b not 9.2b)
    "google/gemma-2-2b-it": ["gemma2:2b"],
    "google/gemma-2-9b-it": ["gemma2", "gemma2:latest", "gemma2:9b"],
    "google/gemma-3-4b-it": ["gemma3", "gemma3:latest", "gemma3:4b"],
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.3": ["mistral", "mistral:latest", "mistral:7b"],
    "mistralai/Ministral-8B-Instruct-2410": ["ministral:8b"],
    # Phi
    "microsoft/Phi-3.5-mini-instruct": ["phi3.5", "phi3.5:latest"],
}


def _size_tag(params: int) -> str:
    """Render parameter count as an Ollama-style size tag: 3b, 1.5b, 27b, 8x7b."""
    if params <= 0:
        return ""
    if params >= 1_000_000_000:
        b = params / 1_000_000_000
        if b >= 10:
            return f"{round(b)}b"
        # Keep one decimal for sub-10B models, strip trailing zeros (1.0→1, 1.5→1.5).
        return f"{b:.1f}".rstrip("0").rstrip(".") + "b"
    if params >= 1_000_000:
        return f"{round(params / 1_000_000)}m"
    return ""


def _normalise_family(model_type: str) -> str:
    """Map an HF model_type to an Ollama-ish family name.

    HF reports things like 'gemma3_text', 'qwen2_moe', 'llama'; Ollama uses
    'gemma3', 'qwen2', 'llama'.  Strips the common suffixes without being
    clever — unknown families pass through unchanged.
    """
    mt = (model_type or "").lower()
    for suffix in ("_text", "_moe", "forcausallm"):
        if mt.endswith(suffix):
            mt = mt[: -len(suffix)]
    return mt


def _infer_aliases(session: SaklasSession) -> list[str]:
    """Derive `<family>:<size>` aliases from model_info."""
    info = session.model_info
    family = _normalise_family(str(info.get("model_type", "")))
    size = _size_tag(int(info.get("param_count", 0) or 0))
    if not family or not size:
        return []
    return [f"{family}:{size}"]


def _aliases_for(session: SaklasSession) -> list[str]:
    """Return Ollama-style aliases for the loaded session.

    If the HF id is in the manual override table, returns those entries
    verbatim — overrides are authoritative and match Ollama's actual
    catalogue (e.g. Ollama advertises Gemma-2-2b as `gemma2:2b` even though
    it's actually 2.6B params).  Otherwise falls back to `<family>:<size>`
    inferred from model_info so new architectures get sensible defaults
    without a table update.
    """
    overrides = _HF_TO_OLLAMA_ALIASES.get(session.model_id)
    if overrides:
        return list(overrides)
    return _infer_aliases(session)


def _known_model_names(session: SaklasSession) -> set[str]:
    names = {session.model_id, *_aliases_for(session)}
    return {n.lower() for n in names}


def _strict_mode() -> bool:
    return os.environ.get("SAKLAS_STRICT_MODEL", "").lower() in ("1", "true", "yes", "on")


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


_PROCESSED_OPTIONS: frozenset[str] = frozenset({
    "temperature", "top_p", "top_k", "seed", "num_predict",
    "stop", "presence_penalty", "frequency_penalty", "repeat_penalty",
    "steer",
})


def _resolve_options(
    body: dict, default_alphas: dict[str, float],
) -> tuple[dict[str, Any], str | None]:
    """Translate Ollama `options` + top-level fields into session.generate kwargs.

    Returns ``(gen_kwargs, system)`` where ``gen_kwargs`` has the new
    cluster-3 shape (``sampling=SamplingConfig``, ``steering=Steering|None``,
    ``thinking=``, ``stateless=True``) and ``system`` is the top-level
    ``system`` field for the caller to splice into messages.

    Recognized Ollama fields: temperature, top_p, top_k, seed, num_predict,
    stop, presence_penalty, frequency_penalty, repeat_penalty.

    `repeat_penalty` maps to `presence_penalty` via ``ln(repeat_penalty)``:
    Ollama divides positive logits by repeat_penalty, which is equivalent
    to subtracting ``ln(penalty)`` from the logit.  That matches
    presence_penalty semantics exactly (subtract a constant per seen token,
    independent of count).

    Unrecognized options (min_p, mirostat*, num_ctx, typical_p, etc.) are
    logged at debug level and silently dropped.

    Non-standard saklas fields (accepted at the top level or inside options):
    `steer` (dict or dict with alphas/thinking), `think` (bool).
    """
    opts = dict(body.get("options") or {})
    top_system = body.get("system")

    stop_raw = opts.get("stop") or body.get("stop")
    if isinstance(stop_raw, str):
        stop_tuple: tuple[str, ...] | None = (stop_raw,)
    elif isinstance(stop_raw, list):
        stop_tuple = tuple(str(s) for s in stop_raw)
    else:
        stop_tuple = None

    temperature = opts.get("temperature")
    top_p = opts.get("top_p")
    top_k_raw = opts.get("top_k")
    try:
        top_k = int(top_k_raw) if top_k_raw is not None else None
    except (TypeError, ValueError):
        top_k = None
    if top_k is not None and top_k <= 0:
        top_k = None
    max_tokens = opts.get("num_predict") or body.get("num_predict")
    seed = opts.get("seed")
    presence_penalty = float(opts.get("presence_penalty", 0.0) or 0.0)
    frequency_penalty = float(opts.get("frequency_penalty", 0.0) or 0.0)
    repeat_raw = opts.get("repeat_penalty")
    if repeat_raw is not None and presence_penalty == 0.0:
        try:
            rp = float(repeat_raw)
            if rp > 1.0:
                presence_penalty = math.log(rp)
        except (TypeError, ValueError):
            pass

    ignored = [k for k in opts if k not in _PROCESSED_OPTIONS]
    if ignored:
        log.debug("ollama: unsupported options dropped: %s", ", ".join(sorted(ignored)))

    steer_raw = opts.get("steer") or body.get("steer") or {}
    thinking: bool | None = None
    if isinstance(steer_raw, dict):
        if "alphas" in steer_raw and isinstance(steer_raw["alphas"], dict):
            alpha_map = {str(k): float(v) for k, v in steer_raw["alphas"].items()}
            t = steer_raw.get("thinking")
            if t is not None:
                thinking = bool(t)
        else:
            alpha_map = {}
            for k, v in steer_raw.items():
                try:
                    alpha_map[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
    else:
        alpha_map = {}

    think_flag = body.get("think")
    if think_flag is not None:
        thinking = bool(think_flag)

    merged_alphas = {**default_alphas, **alpha_map}
    merged_alphas = {k: v for k, v in merged_alphas.items() if v != 0.0}

    sc = SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        seed=seed,
        stop=stop_tuple,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    steering: Steering | None = None
    if merged_alphas or thinking is not None:
        steering = Steering(alphas=merged_alphas, thinking=thinking)

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

    Uses app.state.session + app.state.default_alphas, and serializes on
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
        # Reflect the real HF chat template when available.  Ollama expects
        # Go-template syntax while HF uses Jinja — clients that parse this
        # will fail either way, so returning the honest Jinja template is
        # more useful than the meaningless "{{ .Prompt }}" placeholder.
        tpl = getattr(session._tokenizer, "chat_template", None) or "{{ .Prompt }}"
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
            "error": "saklas does not implement /api/push. Use `saklas pack push` for concept packs.",
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

    def _check_model_or_404(body: dict) -> None:
        """In strict mode, reject requests whose `model` doesn't match the loaded session."""
        if not _strict_mode():
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

    async def _run_and_build_chat_response(body: dict, is_chat: bool) -> dict:
        """Shared non-streaming path for /api/chat and /api/generate."""
        default_alphas = app.state.default_alphas
        gen_kwargs, system = _resolve_options(body, default_alphas)

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

        start_ns = time.monotonic_ns()
        try:
            async with session.lock:
                result = session.generate(input_payload, raw=raw, **gen_kwargs)
        except ConcurrentGenerationError:
            raise HTTPException(status_code=409, detail="Generation already in progress")
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
        # Note: Ollama's /api/generate returns a `context` field of tokenized
        # state for stateless continuation.  Saklas doesn't round-trip that,
        # so we omit the field entirely rather than lie with an empty list.
        return {
            "model": model_name,
            "created_at": created_at,
            "response": result.text,
            "done_reason": done_reason,
            "done": True,
            **stats,
        }

    async def _stream_chat_or_generate(body: dict, is_chat: bool):
        default_alphas = app.state.default_alphas
        gen_kwargs, system = _resolve_options(body, default_alphas)

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

        async with acquire_session_lock(session) as acquired:
            if not acquired:
                yield json.dumps({
                    "model": model_name, "created_at": _now_iso(),
                    "error": "server busy",
                }) + "\n"
                return

            if True:
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
                except ConcurrentGenerationError:
                    yield json.dumps({
                        "model": model_name, "created_at": _now_iso(),
                        "error": "Generation already in progress",
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
                    # See note in _run_and_build_chat_response: `context` is
                    # omitted because saklas can't round-trip it honestly.
                    final = {
                        "model": model_name,
                        "created_at": _now_iso(),
                        "response": "",
                        "done_reason": done_reason,
                        "done": True,
                        **stats,
                    }
                yield json.dumps(final) + "\n"

    @app.post("/api/chat")
    async def api_chat(request: Request):
        body = await request.json()
        _check_model_or_404(body)
        if body.get("stream", True):
            return StreamingResponse(
                _stream_chat_or_generate(body, is_chat=True),
                media_type="application/x-ndjson",
            )
        return await _run_and_build_chat_response(body, is_chat=True)

    @app.post("/api/generate")
    async def api_generate(request: Request):
        body = await request.json()
        _check_model_or_404(body)
        if body.get("stream", True):
            return StreamingResponse(
                _stream_chat_or_generate(body, is_chat=False),
                media_type="application/x-ndjson",
            )
        return await _run_and_build_chat_response(body, is_chat=False)
