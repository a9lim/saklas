"""Native saklas HTTP namespace (``/saklas/v1/*``).

This is the saklas-native resource-tree API, distinct from the OpenAI
(``/v1/*``) and Ollama (``/api/*``) compat shims.  Shape is designed
multi-session — URL-paths carry ``{session_id}`` — but the current impl
is single-session.  The one session has id ``"default"``; both that
literal and the loaded model id resolve to it, everything else 404s.

Killer feature: ``WS /saklas/v1/sessions/{id}/stream`` bidirectional
token + probe co-stream.  Per-token probe readings can't currently be
pushed inline from the session hot path (they're computed once the run
finalizes, via ``score_captured``).  So the WS protocol ships plain
token events during the run and a single ``per_token_probes`` array in
the ``done`` event, assembled from ``session._last_per_token_scores``.
Future clusters can upgrade to inline streaming without changing the
wire format meaningfully.

Old ``/v1/saklas/*`` routes were removed in the same commit that
introduced this file — no aliases.
"""

from __future__ import annotations

from saklas.server.app import ws_auth_ok

import asyncio
import json
import time
import uuid
from dataclasses import replace as _replace
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from saklas.core.errors import SaklasError
from saklas.core.generation import supports_thinking
from saklas.io.probes_bootstrap import load_defaults
from saklas.core.profile import Profile
from saklas.core.results import GenerationResult
from saklas.core.sampling import SamplingConfig
from saklas.core.session import SaklasSession
from saklas.core.steering import Steering


_SINGLE_SESSION_ID = "default"


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    model: str | None = None
    device: str | None = None
    dtype: str | None = None


class PatchSessionRequest(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    thinking: bool | None = None


class ExtractRequest(BaseModel):
    name: str
    source: Any = None
    baseline: str | None = None
    auto_register: bool = Field(True, alias="register")

    model_config = {"populate_by_name": True}


class LoadVectorRequest(BaseModel):
    name: str
    source_path: str


class ScoreProbeRequest(BaseModel):
    text: str
    probes: list[str] | None = None


class WSSamplingParams(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class WSGenerateMessage(BaseModel):
    type: str
    input: Any = None
    # Steering expression string (shared grammar); pole aliases resolve
    # inside session.steering().
    steering: str | None = None
    sampling: WSSamplingParams | None = None
    thinking: bool | None = None
    stateless: bool = True
    raw: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_session_id(session: SaklasSession, session_id: str) -> None:
    """Raise 404 if ``session_id`` doesn't map to the single session."""
    if session_id == _SINGLE_SESSION_ID:
        return
    if session_id == session.model_id:
        return
    raise HTTPException(
        status_code=404,
        detail=f"session '{session_id}' not found",
    )


def _session_config_dict(session: SaklasSession) -> dict:
    cfg = session.config
    return {
        "temperature": getattr(cfg, "temperature", None),
        "top_p": getattr(cfg, "top_p", None),
        "top_k": getattr(cfg, "top_k", None),
        "max_tokens": getattr(cfg, "max_new_tokens", None),
        "system_prompt": getattr(cfg, "system_prompt", None),
    }


def _device_dtype(session: SaklasSession) -> tuple[str, str]:
    info = session.model_info or {}
    device = str(info.get("device", getattr(session, "_device", "")))
    dtype = str(info.get("dtype", getattr(session, "_dtype", "")))
    return device, dtype


def _session_info(
    session: SaklasSession, default_steering: "Steering | None",
) -> dict:
    device, dtype = _device_dtype(session)
    try:
        thinks = bool(supports_thinking(session._tokenizer))
    except Exception:
        thinks = False
    created = getattr(session, "_created_ts", None) or int(time.time())
    default_expr = str(default_steering) if default_steering is not None else None
    return {
        "id": _SINGLE_SESSION_ID,
        "model_id": session.model_id,
        "device": device,
        "dtype": dtype,
        "created": created,
        "config": _session_config_dict(session),
        "vectors": sorted(session.vectors.keys()),
        "probes": sorted(session.probes.keys()),
        "history_length": len(session.history) if hasattr(session, "history") else 0,
        "supports_thinking": thinks,
        "default_steering": default_expr,
    }


def _profile_to_json(name: str, profile: Profile) -> dict:
    top = sorted(
        ((idx, float(vec.norm().item())) for idx, vec in profile.items()),
        key=lambda x: x[1], reverse=True,
    )[:5]
    return {
        "name": name,
        "layers": profile.layers,
        "top_layers": [{"layer": idx, "magnitude": round(m, 4)} for idx, m in top],
        "metadata": profile.metadata,
    }


def _probe_info(session: SaklasSession, name: str) -> dict:
    layers: list[int] = []
    try:
        profiles = session._monitor.profiles
        prof = profiles.get(name)
        if prof is not None:
            layers = sorted(prof.keys())
    except Exception:
        pass
    active = False
    try:
        active = name in session._monitor.probe_names
    except Exception:
        active = name in (session.probes or {})
    return {"name": name, "active": active, "layers": layers}


def _build_sampling(body: WSSamplingParams | None) -> SamplingConfig | None:
    if body is None:
        return None
    stop = tuple(body.stop) if body.stop else None
    return SamplingConfig(
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        max_tokens=body.max_tokens,
        seed=body.seed,
        stop=stop,
        presence_penalty=body.presence_penalty or 0.0,
        frequency_penalty=body.frequency_penalty or 0.0,
    )


def _build_steering(
    raw: str | None, default_steering: "Steering | None",
) -> "Steering | None":
    """Compose a request expression string over the server default Steering.

    Per-request keys override the default at the key level.
    """
    from saklas.core.steering_expr import parse_expr

    req: "Steering | None" = None
    if raw is not None and raw.strip():
        req = parse_expr(raw)

    thinking: bool | None = None
    if req is not None and req.thinking is not None:
        thinking = req.thinking

    merged: dict = {}
    if default_steering is not None:
        merged.update(default_steering.alphas)
    if req is not None:
        for k, v in req.alphas.items():
            merged[k] = v

    if not merged and thinking is None:
        return None
    return Steering(alphas=merged, thinking=thinking)


def _coerce_pair_source(source: Any) -> Any:
    """Normalize JSON pair payloads into DataSource-compatible tuples."""
    if not (isinstance(source, dict) and "pairs" in source):
        return source
    pairs = []
    for idx, pair in enumerate(source["pairs"]):
        if isinstance(pair, dict):
            if "positive" not in pair or "negative" not in pair:
                raise HTTPException(
                    400,
                    f"pairs[{idx}] must contain 'positive' and 'negative'",
                )
            pairs.append((str(pair["positive"]), str(pair["negative"])))
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            pairs.append((str(pair[0]), str(pair[1])))
        else:
            raise HTTPException(
                400,
                f"pairs[{idx}] must be a [positive, negative] pair",
            )
    return pairs


def _result_to_json(result: GenerationResult | None) -> dict:
    if result is None:
        return {}
    prompt_tokens = getattr(result, "prompt_tokens", 0) or 0
    completion = getattr(result, "token_count", 0) or 0
    return {
        "text": getattr(result, "text", ""),
        "tokens": completion,
        "finish_reason": getattr(result, "finish_reason", "stop"),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": prompt_tokens + completion,
        },
    }


def _per_token_probes(session: SaklasSession, n_tokens: int) -> list[dict]:
    scores = session.last_per_token_scores
    if not scores:
        return []
    out: list[dict] = []
    n = min(n_tokens, *(len(v) for v in scores.values())) if scores else 0
    for i in range(n):
        out.append({
            "token_idx": i,
            "probes": {name: float(vals[i]) for name, vals in scores.items() if i < len(vals)},
        })
    return out


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_saklas_routes(app: FastAPI) -> None:
    """Mount the native ``/saklas/v1/*`` tree onto ``app``.

    ``session`` and ``default_steering`` are pulled off ``app.state`` so the
    signature matches ``register_ollama_routes`` and ``create_app`` doesn't
    need to thread them.
    """

    session: SaklasSession = app.state.session

    # ----- sessions collection -------------------------------------------

    @app.get("/saklas/v1/sessions")
    def list_sessions():
        return {"sessions": [_session_info(session, app.state.default_steering)]}

    @app.post("/saklas/v1/sessions")
    def create_session(req: CreateSessionRequest):
        if req.model and req.model != session.model_id:
            # Idempotent: log and return existing, per plan.
            import logging
            logging.getLogger("saklas.api").warning(
                "POST /saklas/v1/sessions requested model=%r but session is %r; "
                "single-session mode, returning existing",
                req.model, session.model_id,
            )
        return _session_info(session, app.state.default_steering)

    @app.get("/saklas/v1/sessions/{session_id}")
    def get_session(session_id: str):
        _resolve_session_id(session, session_id)
        return _session_info(session, app.state.default_steering)

    @app.delete("/saklas/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        _resolve_session_id(session, session_id)
        import logging
        logging.getLogger("saklas.api").warning(
            "DELETE /saklas/v1/sessions/%s: single-session mode, no-op",
            session_id,
        )
        return JSONResponse(status_code=204, content=None)

    @app.patch("/saklas/v1/sessions/{session_id}")
    def patch_session(session_id: str, req: PatchSessionRequest):
        _resolve_session_id(session, session_id)
        overrides: dict = {}
        if req.temperature is not None:
            overrides["temperature"] = req.temperature
        if req.top_p is not None:
            overrides["top_p"] = req.top_p
        if req.top_k is not None:
            overrides["top_k"] = req.top_k
        if req.max_tokens is not None:
            overrides["max_new_tokens"] = req.max_tokens
        if req.system_prompt is not None:
            overrides["system_prompt"] = req.system_prompt
        if overrides:
            from dataclasses import is_dataclass
            if is_dataclass(session.config):
                session.config = _replace(session.config, **overrides)
            else:
                for k, v in overrides.items():
                    setattr(session.config, k, v)
        return _session_info(session, app.state.default_steering)

    @app.post("/saklas/v1/sessions/{session_id}/clear", status_code=204)
    def clear_session(session_id: str):
        _resolve_session_id(session, session_id)
        session.clear_history()
        return JSONResponse(status_code=204, content=None)

    @app.post("/saklas/v1/sessions/{session_id}/rewind", status_code=204)
    def rewind_session(session_id: str):
        _resolve_session_id(session, session_id)
        if not session.history:
            raise HTTPException(400, "History is empty")
        session.rewind()
        return JSONResponse(status_code=204, content=None)

    # ----- vectors -------------------------------------------------------

    @app.get("/saklas/v1/sessions/{session_id}/vectors")
    def list_vectors(session_id: str):
        _resolve_session_id(session, session_id)
        return {
            "vectors": [
                _profile_to_json(name, profile)
                for name, profile in sorted(session.vectors.items())
            ],
        }

    @app.get("/saklas/v1/sessions/{session_id}/vectors/{name}")
    def get_vector(session_id: str, name: str):
        _resolve_session_id(session, session_id)
        vectors = session.vectors
        if name not in vectors:
            raise HTTPException(404, f"vector '{name}' not found")
        return _profile_to_json(name, vectors[name])

    @app.post("/saklas/v1/sessions/{session_id}/vectors")
    def load_vector(session_id: str, req: LoadVectorRequest):
        _resolve_session_id(session, session_id)
        try:
            profile = session.load_profile(req.source_path)
        except FileNotFoundError:
            raise HTTPException(400, f"file not found: {req.source_path}")
        session.steer(req.name, profile)
        return _profile_to_json(req.name, profile)

    @app.delete("/saklas/v1/sessions/{session_id}/vectors/{name}", status_code=204)
    def delete_vector(session_id: str, name: str):
        _resolve_session_id(session, session_id)
        if name not in session.vectors:
            raise HTTPException(404, f"vector '{name}' not found")
        session.unsteer(name)
        # Drop the vector from the default steering (if present) so the
        # next request doesn't autoload it back under a stale alpha.
        ds = app.state.default_steering
        if ds is not None and name in ds.alphas:
            from dataclasses import replace as _replace
            new_alphas = {k: v for k, v in ds.alphas.items() if k != name}
            app.state.default_steering = (
                _replace(ds, alphas=new_alphas) if new_alphas else None
            )
        return JSONResponse(status_code=204, content=None)

    @app.post("/saklas/v1/sessions/{session_id}/extract")
    async def extract_vector(session_id: str, req: ExtractRequest, request: Request):
        _resolve_session_id(session, session_id)
        source: Any = req.source if req.source is not None else req.name
        source = _coerce_pair_source(source)

        accept = request.headers.get("accept", "application/json")
        if "text/event-stream" in accept:
            async def _sse():
                progress_msgs: list[str] = []
                async with session.lock:
                    try:
                        canonical, profile = await asyncio.to_thread(
                            session.extract, source, req.baseline,
                            on_progress=progress_msgs.append,
                        )
                    except SaklasError as e:
                        import logging
                        logging.getLogger("saklas.api").exception(
                            "extract failed for session=%s", session_id,
                        )
                        err = {"message": "extract failed", "code": type(e).__name__}
                        yield f"event: error\ndata: {json.dumps(err)}\n\n"
                        return
                    if req.auto_register:
                        session.steer(req.name, profile)
                    for msg in progress_msgs:
                        yield f"event: progress\ndata: {json.dumps({'message': msg})}\n\n"
                    body = {"done": True, "profile": _profile_to_json(canonical, profile), "canonical": canonical}
                    yield f"event: done\ndata: {json.dumps(body)}\n\n"

            return StreamingResponse(_sse(), media_type="text/event-stream")

        progress_msgs: list[str] = []
        async with session.lock:
            canonical, profile = await asyncio.to_thread(
                session.extract, source, req.baseline,
                on_progress=progress_msgs.append,
            )
            if req.auto_register:
                session.steer(req.name, profile)
        return {
            "canonical": canonical,
            "profile": _profile_to_json(canonical, profile),
            "progress": progress_msgs,
        }

    # ----- probes --------------------------------------------------------

    @app.get("/saklas/v1/sessions/{session_id}/probes")
    def list_probes(session_id: str):
        _resolve_session_id(session, session_id)
        names = sorted(session.probes.keys()) if isinstance(session.probes, dict) else list(session.probes)
        return {"probes": [_probe_info(session, n) for n in names]}

    @app.get("/saklas/v1/sessions/{session_id}/probes/defaults")
    def list_default_probes(session_id: str):
        _resolve_session_id(session, session_id)
        return {"defaults": load_defaults()}

    @app.post("/saklas/v1/sessions/{session_id}/probes/{name}", status_code=204)
    def activate_probe(session_id: str, name: str):
        _resolve_session_id(session, session_id)
        try:
            session.probe(name)
        except (KeyError, ValueError, FileNotFoundError) as e:
            raise HTTPException(400, f"probe '{name}' not available: {e}")
        return JSONResponse(status_code=204, content=None)

    @app.delete("/saklas/v1/sessions/{session_id}/probes/{name}", status_code=204)
    def deactivate_probe(session_id: str, name: str):
        _resolve_session_id(session, session_id)
        if name not in session.probes:
            raise HTTPException(404, f"probe '{name}' not active")
        session.unprobe(name)
        return JSONResponse(status_code=204, content=None)

    @app.post("/saklas/v1/sessions/{session_id}/probe")
    async def score_probe_oneshot(session_id: str, req: ScoreProbeRequest):
        _resolve_session_id(session, session_id)
        requested = req.probes
        monitor = session._monitor
        if requested:
            missing = [n for n in requested if n not in monitor.probe_names]
            if missing:
                raise HTTPException(400, f"probes not active: {missing}")

        async with session.lock:
            readings = await asyncio.to_thread(
                monitor.measure, session._model, session._tokenizer,
                session._layers, req.text,
            )
        if requested:
            readings = {k: v for k, v in readings.items() if k in requested}
        return {"readings": {k: float(v) for k, v in readings.items()}}

    # ----- Live traits SSE stream ------------------------------------------

    @app.get("/saklas/v1/sessions/{session_id}/traits/stream")
    async def traits_stream(session_id: str, request: Request):
        """SSE endpoint streaming per-token probe scores during generation.

        Events:
          - ``data: {"type": "start", ...}`` when generation begins
          - ``data: {"type": "token", "idx": N, "text": "...", "thinking": bool, "probes": {...}}``
          - ``data: {"type": "done", "finish_reason": "...", "aggregate": {...}}``
          - ``: heartbeat`` every 15 s when idle

        The stream stays open across generations; a client can subscribe
        once and observe every generation the session runs.
        """
        _resolve_session_id(session, session_id)

        from saklas.core.events import GenerationStarted, GenerationFinished

        loop = asyncio.get_running_loop()
        trait_queue: asyncio.Queue = asyncio.Queue()

        # EventBus callback: push start/done into the same queue as tokens.
        def _on_event(event):
            if isinstance(event, GenerationStarted):
                try:
                    loop.call_soon_threadsafe(
                        trait_queue.put_nowait,
                        ("start", getattr(event, "input", None), getattr(event, "stateless", False)),
                    )
                except Exception:
                    pass
            elif isinstance(event, GenerationFinished):
                try:
                    loop.call_soon_threadsafe(
                        trait_queue.put_nowait,
                        ("done", getattr(event, "result", None)),
                    )
                except Exception:
                    pass

        unsub = session.events.subscribe(_on_event)
        session.register_trait_queue(loop, trait_queue)

        async def event_generator():
            try:
                generation_id: str | None = None
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.wait_for(trait_queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
                        continue

                    tag = item[0]
                    if tag == "start":
                        generation_id = uuid.uuid4().hex[:8]
                        yield (
                            f"data: {json.dumps({'type': 'start', 'generation_id': generation_id})}"
                            "\n\n"
                        )
                    elif tag == "token":
                        _, idx, text, thinking, scores = item
                        yield (
                            f"data: {json.dumps({'type': 'token', 'idx': idx, 'text': text, 'thinking': thinking, 'probes': {k: round(v, 6) for k, v in scores.items()}})}"
                            "\n\n"
                        )
                    elif tag == "done":
                        result = item[1]
                        agg: dict[str, float] = {}
                        if result is not None:
                            readings = getattr(result, "readings", None)
                            if readings:
                                for name, r in readings.items():
                                    # Use this generation's aggregate, not the
                                    # rolling history mean.
                                    pg = getattr(r, "per_generation", None)
                                    val = pg[-1] if pg else getattr(r, "mean", 0.0)
                                    agg[name] = round(val, 6)
                        yield (
                            f"data: {json.dumps({'type': 'done', 'generation_id': generation_id, 'finish_reason': getattr(result, 'finish_reason', 'stop') if result else 'stop', 'aggregate': agg})}"
                            "\n\n"
                        )
                        generation_id = None
            finally:
                session.unregister_trait_queue(loop, trait_queue)
                unsub()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # ----- WebSocket token+probe co-stream -------------------------------

    @app.websocket("/saklas/v1/sessions/{session_id}/stream")
    async def session_stream(websocket: WebSocket, session_id: str):
        # NOTE: only ``session_id == "default"`` is actually reachable
        # here — HF model ids contain '/' and the FastAPI path parameter
        # is not declared ``{session_id:path}``, so the model-id branch
        # is an HTTP-route convenience only.  Kept as a no-op guard.
        if not ws_auth_ok(websocket):
            await websocket.close(code=1008, reason="unauthorized")
            return
        if session_id not in (_SINGLE_SESSION_ID, session.model_id):
            await websocket.accept()
            await websocket.close(code=1008, reason="session not found")
            return
        await websocket.accept()

        try:
            while True:
                msg = await websocket.receive_json()
                mtype = msg.get("type")
                if mtype == "generate":
                    try:
                        parsed = WSGenerateMessage(**msg)
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"invalid generate message: {e}",
                            "code": "ValidationError",
                        })
                        continue
                    await _ws_handle_generate(
                        websocket, session, parsed, app.state.default_steering,
                    )
                elif mtype == "stop":
                    # Idle-state stop: nothing in flight.
                    continue
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"unknown message type: {mtype!r}",
                        "code": "UnknownMessageType",
                    })
        except WebSocketDisconnect:
            # Ensure any stray generation is signaled.
            try:
                session.stop()
            except Exception:
                pass
            return
        except Exception as e:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "code": type(e).__name__,
                })
            finally:
                try:
                    await websocket.close(code=1011)
                except Exception:
                    pass


async def _ws_handle_generate(
    websocket: WebSocket,
    session: SaklasSession,
    msg: WSGenerateMessage,
    default_steering: "Steering | None",
) -> None:
    """Run one generate turn and stream token/done/error events.

    Concurrency design: the synchronous ``session.generate`` is run in a
    worker thread via ``asyncio.to_thread``.  Its ``on_token`` callback
    is invoked on the worker thread; it bridges into the asyncio loop by
    calling ``loop.call_soon_threadsafe(queue.put_nowait, event)``.  The
    main coroutine races two tasks: one pulls ``TokenEvent``s from the
    queue and forwards them as ``{type: "token", ...}`` frames; the other
    awaits the next client frame so an incoming ``{type: "stop"}`` can
    call ``session.stop()`` without blocking on the token loop.

    ``asyncio.wait(..., FIRST_COMPLETED)`` is used in a loop: whenever
    the recv task returns a stop frame we signal the session and keep
    draining tokens until the worker joins; whenever the queue delivers
    a sentinel we finish.  The WS stays open across generate turns — a
    client can submit ``{type: "generate", ...}`` again after ``done``.
    """
    loop = asyncio.get_running_loop()
    generation_id = uuid.uuid4().hex[:12]

    sampling = _build_sampling(msg.sampling)
    steering = _build_steering(msg.steering, default_steering)

    token_queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _on_token(text, is_thinking, tid, lp, top, perplexity=None):
        event = {
            "type": "token",
            "text": text,
            "thinking": bool(is_thinking),
            "token_id": int(tid) if tid is not None else None,
        }
        loop.call_soon_threadsafe(token_queue.put_nowait, event)

    result_holder: list[GenerationResult] = []
    error_holder: list[BaseException] = []

    def _worker():
        try:
            result = session.generate(
                msg.input,
                steering=steering,
                sampling=sampling,
                stateless=msg.stateless,
                raw=msg.raw,
                thinking=msg.thinking,
                on_token=_on_token,
            )
            result_holder.append(result)
        except BaseException as e:
            error_holder.append(e)
        finally:
            loop.call_soon_threadsafe(token_queue.put_nowait, _SENTINEL)

    # Acquire the session lock for the full generation lifetime so
    # concurrent WS clients serialize FIFO instead of overlapping.
    async with session.lock:
        await websocket.send_json({"type": "started", "generation_id": generation_id})

        worker_task = asyncio.create_task(asyncio.to_thread(_worker))

        recv_task: asyncio.Task | None = asyncio.create_task(websocket.receive_json())
        done = False
        try:
            while not done:
                get_task = asyncio.create_task(token_queue.get())
                wait_for = {get_task}
                if recv_task is not None:
                    wait_for.add(recv_task)
                finished, _pending = await asyncio.wait(
                    wait_for, return_when=asyncio.FIRST_COMPLETED,
                )
                if recv_task is not None and recv_task in finished:
                    try:
                        incoming = recv_task.result()
                    except WebSocketDisconnect:
                        session.stop()
                        recv_task = None
                    except Exception:
                        recv_task = None
                    else:
                        if isinstance(incoming, dict) and incoming.get("type") == "stop":
                            try:
                                session.stop()
                            except Exception:
                                pass
                        recv_task = asyncio.create_task(websocket.receive_json())
                if get_task in finished:
                    item = get_task.result()
                    if item is _SENTINEL:
                        done = True
                    else:
                        await websocket.send_json(item)
                else:
                    get_task.cancel()
        finally:
            # Drain any residual events the worker pushed between sentinel
            # and join — should be none because the sentinel is last, but
            # cheap insurance.
            await worker_task
            if recv_task is not None and not recv_task.done():
                recv_task.cancel()

        if error_holder and not result_holder:
            exc = error_holder[0]
            await websocket.send_json({
                "type": "error",
                "message": str(exc),
                "code": type(exc).__name__,
            })
            return

        result = result_holder[0] if result_holder else None
        result_json = _result_to_json(result)
        if result is not None:
            result_json["per_token_probes"] = _per_token_probes(
                session, getattr(result, "token_count", 0) or 0,
            )
        else:
            result_json["per_token_probes"] = []
        await websocket.send_json({"type": "done", "result": result_json})
