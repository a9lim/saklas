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

from saklas.server.app import acquire_session_lock, ws_auth_ok

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


class SweepRequest(BaseModel):
    """Body for ``POST /saklas/v1/sessions/{id}/sweep``.

    ``sweep`` maps concept name → list of alpha values.  Cartesian
    product across concepts becomes one generation per row.
    ``base_steering`` (optional) is a steering expression string
    composed underneath each swept term so callers can hold a
    fixed-alpha context while sweeping another concept.
    """
    prompt: Any
    sweep: dict[str, list[float]]
    base_steering: str | None = None
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
    layer_norms = [(idx, float(vec.norm().item())) for idx, vec in profile.items()]
    top = sorted(layer_norms, key=lambda x: x[1], reverse=True)[:5]
    # Full per-layer ||baked|| keyed by layer index — stringified for
    # JSON-key compatibility, mirroring how diagnostics_by_layer round-trips.
    # The web UI's LayerNorms panel consumes this directly.
    per_layer_norms = {str(idx): round(mag, 6) for idx, mag in sorted(layer_norms)}
    return {
        "name": name,
        "layers": profile.layers,
        "top_layers": [{"layer": idx, "magnitude": round(m, 4)} for idx, m in top],
        "per_layer_norms": per_layer_norms,
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

    @app.get("/saklas/v1/sessions/{session_id}/correlation")
    def correlation_matrix(session_id: str, names: str | None = None):
        """N×N magnitude-weighted cosine matrix across loaded vectors.

        Query: ``?names=a,b,c`` restricts the matrix to a subset; default
        is every vector currently registered in the session.  Output:

            {
              "names": ["a", "b", ...],
              "matrix": {"a": {"a": 1.0, "b": 0.42, ...}, ...},
              "layers_shared": {"a__b": 36, ...}
            }

        Used by the web UI's correlation panel — heavy compute lives
        server-side so the client doesn't have to ship full per-layer
        tensors over the wire.
        """
        _resolve_session_id(session, session_id)

        all_vectors = session.vectors
        if names is not None and names.strip():
            requested = [n.strip() for n in names.split(",") if n.strip()]
            missing = [n for n in requested if n not in all_vectors]
            if missing:
                raise HTTPException(404, f"vectors not loaded: {missing}")
            ordered = requested
        else:
            ordered = sorted(all_vectors.keys())

        matrix: dict[str, dict[str, float | None]] = {a: {} for a in ordered}
        layers_shared: dict[str, int] = {}
        for i, a in enumerate(ordered):
            for j, b in enumerate(ordered):
                if j < i:
                    matrix[a][b] = matrix[b][a]
                    continue
                if i == j:
                    matrix[a][b] = 1.0
                    continue
                try:
                    # ``cosine_similarity`` without ``per_layer=`` returns
                    # the magnitude-weighted aggregate ``float`` — narrow
                    # explicitly because the method's union return type
                    # is ``float | dict[int, float]``.
                    cos = all_vectors[a].cosine_similarity(all_vectors[b])
                    matrix[a][b] = (
                        round(float(cos), 6) if isinstance(cos, (int, float)) else None
                    )
                except Exception:
                    matrix[a][b] = None
                shared = sorted(
                    set(all_vectors[a].keys()) & set(all_vectors[b].keys())
                )
                # Pair key sorted alphabetically so a__b == b__a in the lookup.
                key = "__".join(sorted([a, b]))
                layers_shared[key] = len(shared)
        return {
            "names": ordered,
            "matrix": matrix,
            "layers_shared": layers_shared,
        }

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

    # ----- Sweep: alpha grid over a single prompt --------------------------

    @app.post("/saklas/v1/sessions/{session_id}/sweep")
    async def run_sweep(session_id: str, req: SweepRequest, request: Request):
        """SSE-streamed alpha sweep.

        One result per Cartesian-product element across ``sweep[concept]``
        alpha lists.  Held under ``acquire_session_lock`` for the full
        sweep so concurrent endpoints queue FIFO (the sweep can be
        long-running, the lock's 5-minute timeout still applies).

        Emits ``data: {"type": "started", "sweep_id", "total"}``,
        then ``data: {"type": "result", "idx", "alpha_values", "result"}``
        per completion (result subset: text, finish_reason, usage,
        applied_steering, readings means), then
        ``data: {"type": "done", "sweep_id", "summary"}``.
        """
        _resolve_session_id(session, session_id)

        # Pre-validate the sweep dict so the SSE start event reports
        # ``total`` accurately and so a bad request fails before any
        # gen acquires the lock.  ``generate_sweep`` would raise the
        # same way, but the SSE caller never sees the exception cleanly.
        if not req.sweep:
            raise HTTPException(400, "sweep dict must be non-empty")
        for name, alphas in req.sweep.items():
            if not alphas:
                raise HTTPException(400, f"sweep['{name}'] must be non-empty")

        total = 1
        for alphas in req.sweep.values():
            total *= len(alphas)
        sweep_id = uuid.uuid4().hex[:8]
        sampling_cfg = _build_sampling(req.sampling)

        # Drive ``generate_sweep`` in a worker thread; bridge per-result
        # events to the asyncio queue this handler reads.
        loop = asyncio.get_running_loop()
        result_queue: asyncio.Queue = asyncio.Queue()
        DONE = object()
        ERROR = object()

        def _emit_result(idx: int, result: Any, alpha_values: dict[str, float]) -> None:
            # Subset the result to keep SSE payloads small; full hidden
            # states / per-token logprobs aren't useful for a sweep
            # consumer and can be reloaded out-of-band by replaying
            # ``applied_steering``.
            readings_summary: dict[str, float] = {}
            try:
                for probe_name, r in (getattr(result, "readings", {}) or {}).items():
                    pg = getattr(r, "per_generation", None)
                    val = pg[-1] if pg else getattr(r, "mean", 0.0)
                    readings_summary[probe_name] = round(float(val), 6)
            except Exception:
                pass
            payload = {
                "idx": idx,
                "alpha_values": {k: float(v) for k, v in alpha_values.items()},
                "result": {
                    "text": getattr(result, "text", ""),
                    "token_count": int(getattr(result, "token_count", 0)),
                    "tok_per_sec": float(getattr(result, "tok_per_sec", 0.0)),
                    "elapsed": float(getattr(result, "elapsed", 0.0)),
                    "finish_reason": getattr(result, "finish_reason", "stop"),
                    "applied_steering": getattr(result, "applied_steering", None),
                    "readings": readings_summary,
                },
            }
            try:
                loop.call_soon_threadsafe(result_queue.put_nowait, ("result", payload))
            except Exception:
                pass

        def _worker() -> None:
            try:
                session.generate_sweep(
                    req.prompt,
                    req.sweep,
                    base_steering=req.base_steering,
                    sampling=sampling_cfg,
                    thinking=req.thinking,
                    stateless=req.stateless,
                    raw=req.raw,
                    on_result=_emit_result,
                )
                loop.call_soon_threadsafe(result_queue.put_nowait, (DONE, None))
            except Exception as exc:
                loop.call_soon_threadsafe(result_queue.put_nowait, (ERROR, exc))

        async def event_generator():
            async with acquire_session_lock(session) as acquired:
                if not acquired:
                    yield (
                        f"data: {json.dumps({'type': 'error', 'message': 'session locked'})}"
                        "\n\n"
                    )
                    return

                yield (
                    f"data: {json.dumps({'type': 'started', 'sweep_id': sweep_id, 'total': total})}"
                    "\n\n"
                )

                fut = asyncio.get_running_loop().run_in_executor(None, _worker)
                completed = 0
                start = time.monotonic()
                total_tokens = 0
                try:
                    while True:
                        if await request.is_disconnected():
                            session.stop()
                            break
                        try:
                            tag, payload = await asyncio.wait_for(
                                result_queue.get(), timeout=15.0,
                            )
                        except asyncio.TimeoutError:
                            yield ": heartbeat\n\n"
                            continue
                        if tag == "result":
                            completed += 1
                            total_tokens += int(payload["result"].get("token_count", 0))
                            yield (
                                f"data: {json.dumps({'type': 'result', **payload})}"
                                "\n\n"
                            )
                            continue
                        if tag is DONE:
                            elapsed = time.monotonic() - start
                            tps = (total_tokens / elapsed) if elapsed > 0 else 0.0
                            yield (
                                f"data: {json.dumps({'type': 'done', 'sweep_id': sweep_id, 'summary': {'completed': completed, 'total': total, 'total_tokens': total_tokens, 'tok_per_sec': round(tps, 2), 'elapsed': round(elapsed, 3)}})}"
                                "\n\n"
                            )
                            break
                        if tag is ERROR:
                            yield (
                                f"data: {json.dumps({'type': 'error', 'message': str(payload)})}"
                                "\n\n"
                            )
                            break
                finally:
                    # Wait for the worker to fully wind down so the gen
                    # lock isn't released mid-generation.
                    try:
                        await fut
                    except Exception:
                        pass

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

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

        # Single perpetual reader.  ``websocket.receive_json()`` is bound
        # to a per-connection ``recv_in_progress`` flag in the underlying
        # ``websockets`` library; cancelling a pending receive doesn't
        # clear the flag immediately, so any handler that called
        # ``receive_json()`` while another concurrent (even just-cancelled)
        # caller was pending tripped a "cannot call recv while another
        # coroutine is already waiting" RuntimeError.  Routing every
        # incoming frame through one queue lets both the outer dispatch
        # loop and the in-flight generation share the read side without
        # ever overlapping calls into the WS.
        incoming: asyncio.Queue = asyncio.Queue()
        _DISCONNECT = object()

        async def _reader():
            try:
                while True:
                    msg = await websocket.receive_json()
                    await incoming.put(msg)
            except WebSocketDisconnect:
                await incoming.put(_DISCONNECT)
            except Exception as e:
                # Surface any other read-side failure into the queue so
                # the dispatcher can close cleanly instead of leaking.
                await incoming.put({"_reader_error": str(e), "_type": type(e).__name__})

        reader_task = asyncio.create_task(_reader())

        try:
            while True:
                msg = await incoming.get()
                if msg is _DISCONNECT:
                    raise WebSocketDisconnect(code=1000)
                if isinstance(msg, dict) and "_reader_error" in msg:
                    raise RuntimeError(msg["_reader_error"])

                mtype = msg.get("type") if isinstance(msg, dict) else None
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
                        websocket, session, parsed,
                        app.state.default_steering, incoming,
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
        finally:
            # Reader holds the only ``receive_json()`` call on the WS.
            # Cancel + await so the cancellation propagates fully before
            # the connection tears down.
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass


async def _ws_handle_generate(
    websocket: WebSocket,
    session: SaklasSession,
    msg: WSGenerateMessage,
    default_steering: "Steering | None",
    incoming: asyncio.Queue,
) -> None:
    """Run one generate turn and stream token/done/error events.

    Concurrency design: the synchronous ``session.generate`` is run in a
    worker thread via ``asyncio.to_thread``.  Its ``on_token`` callback
    is invoked on the worker thread; it bridges into the asyncio loop by
    calling ``loop.call_soon_threadsafe(queue.put_nowait, event)``.  The
    main coroutine races two tasks: one pulls ``TokenEvent``s from a
    local queue and forwards them as ``{type: "token", ...}`` frames;
    the other pulls client frames from the shared ``incoming`` queue
    (populated by the connection's single reader task) so an in-flight
    ``{type: "stop"}`` can call ``session.stop()`` without blocking on
    the token loop.

    ``asyncio.wait(..., FIRST_COMPLETED)`` is used in a loop: whenever
    the incoming task returns a stop frame we signal the session and keep
    draining tokens until the worker joins; whenever the queue delivers
    a sentinel we finish.  The WS stays open across generate turns — a
    client can submit ``{type: "generate", ...}`` again after ``done``,
    and the perpetual reader keeps feeding the shared queue between
    turns so we never have two ``receive_json()`` calls in flight.
    """
    loop = asyncio.get_running_loop()
    generation_id = uuid.uuid4().hex[:12]

    sampling = _build_sampling(msg.sampling)
    steering = _build_steering(msg.steering, default_steering)

    token_queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _on_token(text, is_thinking, tid, lp, top, perplexity=None):
        event: dict[str, Any] = {
            "type": "token",
            "text": text,
            "thinking": bool(is_thinking),
            "token_id": int(tid) if tid is not None else None,
        }
        # Per-layer × per-probe heatmap data for the inspector panel.
        # Computed inline only when probes are loaded (covers the cost
        # of N matmul + N CPU syncs against the latest captured hidden
        # state).  Falls through silently when no capture exists yet
        # (e.g. extremely short generations where the hook never fires).
        try:
            monitor = session._monitor
            if monitor.probe_names:
                latest_hidden = {
                    layer_idx: bucket[-1]
                    for layer_idx, bucket in session._capture._per_layer.items()
                    if bucket
                }
                if latest_hidden:
                    per_layer = monitor.score_single_token_per_layer(latest_hidden)
                    if per_layer:
                        event["per_layer_scores"] = {
                            str(layer): {p: round(float(v), 6) for p, v in metrics.items()}
                            for layer, metrics in per_layer.items()
                        }
        except Exception:
            # Inspector data is best-effort — never let a failure here
            # break the streaming token path.
            pass
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

        # Race two queue reads — token frames from the worker and client
        # frames from the connection's perpetual reader.  Neither side
        # ever calls ``websocket.receive_json()`` directly, so the
        # underlying ``recv_in_progress`` flag is owned by the reader
        # task alone for the connection's lifetime.
        done = False
        try:
            while not done:
                token_get = asyncio.create_task(token_queue.get())
                client_get = asyncio.create_task(incoming.get())
                finished, _pending = await asyncio.wait(
                    {token_get, client_get}, return_when=asyncio.FIRST_COMPLETED,
                )
                if client_get in finished:
                    incoming_msg = client_get.result()
                    # ``_DISCONNECT`` / reader-error sentinels: signal
                    # the worker to wind down; let the outer loop
                    # propagate the disconnect on the next iteration.
                    if isinstance(incoming_msg, dict):
                        if incoming_msg.get("type") == "stop":
                            try:
                                session.stop()
                            except Exception:
                                pass
                        elif "_reader_error" in incoming_msg:
                            try:
                                session.stop()
                            except Exception:
                                pass
                            # Re-enqueue so the outer dispatch loop
                            # surfaces the error after we wind down.
                            await incoming.put(incoming_msg)
                        else:
                            # Out-of-band frame during a generation —
                            # re-enqueue so the outer loop sees it after
                            # this turn finishes.  Most likely an early
                            # ``{type: "generate"}`` from a client that
                            # didn't wait for ``done``.
                            await incoming.put(incoming_msg)
                    else:
                        # Disconnect sentinel from the reader.
                        try:
                            session.stop()
                        except Exception:
                            pass
                        await incoming.put(incoming_msg)
                else:
                    client_get.cancel()
                if token_get in finished:
                    item = token_get.result()
                    if item is _SENTINEL:
                        done = True
                    else:
                        await websocket.send_json(item)
                else:
                    token_get.cancel()
        finally:
            # Drain any residual events the worker pushed between sentinel
            # and join — should be none because the sentinel is last, but
            # cheap insurance.
            await worker_task

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
