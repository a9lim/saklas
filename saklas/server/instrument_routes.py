"""Native ``/instruments`` route family — the unified read-side surface.

One route tree over the three read families the engine exposes as
``session.instruments`` (``geometry`` / ``lens`` / ``sae``).  It replaces the
former per-family ``/lens/*`` and ``/sae/*`` route groups and the
``POST /probes/live`` toggle:

- ``GET  .../instruments`` — enumerate the three families (live state, active
  source, attached probes, capabilities).
- ``POST .../instruments/{family}/live`` — uniform live toggle (geometry =
  the CAA per-token monitor scoring switch; lens = the workspace readout;
  sae = the feature-discovery readout).
- ``GET  .../instruments/{family}/sources`` — prepared sources (lens) /
  prepared + provider-candidate releases (sae); geometry 404s (no source
  lifecycle).
- ``PUT  .../instruments/{family}/source`` — synchronous source switch
  (lens only; sae 409 → use preparations; geometry 404).
- ``POST/GET/DELETE .../instruments/{family}/preparations`` — the unified
  background-job resource (lens ``fetch``/``fit``, sae ``load``/``train``),
  polled (never SSE), with a common status shape.
- ``GET  .../instruments/{family}/token-readout`` — the loom token-drilldown
  readout, wrapped in the 5.x ``measurements`` replay envelope.
- Family extras: ``POST .../instruments/lens/token/validate``,
  ``POST .../instruments/sae/features/metadata``,
  ``POST .../instruments/sae/features/validate``.

Auth / locking / error-scrubbing discipline is copied verbatim from the
per-family routes this file supersedes (``acquire_session_lock``, the typed
``SaklasError.user_message()`` mapping, ``background_job.scrub_job_error``).
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import Field, ValidationError

from saklas.core.errors import SaklasError
from saklas.core.jlens import LensNotFittedError, resolve_word_token
from saklas.core.loom import InvalidNodeOperationError, UnknownNodeError
from saklas.core.measurements import build_measurements
from saklas.server.app import acquire_session_lock
from saklas.server.background_job import (
    BackgroundJob,
    make_progress_hook,
    scrub_job_error,
)
from saklas.server.native_common import NativeRequest, resolve_session_id

log = logging.getLogger(__name__)

_FAMILIES = ("geometry", "lens", "sae")

#: ``fit_jacobian_lens`` per-prompt progress line — "prompt 12/100 (…)".
_FIT_PROGRESS_RE = re.compile(r"prompt (\d+)/(\d+)")
#: ``train_residual_sae`` token-progress line — "trained 12,345/1,000,000 tokens".
_TRAIN_PROGRESS_RE = re.compile(r"trained ([\d,]+)/([\d,]+) tokens")


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class InstrumentLiveRequest(NativeRequest):
    """Uniform body for ``POST .../instruments/{family}/live``.

    ``layers`` applies to the lens family. Readout width is generation state
    shared with ``return_top_k`` / alts, so ``top_k`` is rejected for every
    family rather than silently creating an instrument-local width.
    """

    enabled: bool
    layers: list[int] | None = None
    top_k: int | None = Field(default=None, ge=1, le=256)


class SourceRequest(NativeRequest):
    """Body for ``PUT .../instruments/{family}/source``."""

    source: str = Field(min_length=1)


class LensTokenValidationRequest(NativeRequest):
    """Body for ``POST .../instruments/lens/token/validate``."""

    word: str


class SaeFeatureRequest(NativeRequest):
    """Body for ``POST .../instruments/sae/features/validate``."""

    id: int = Field(ge=0)


class SaeFeatureMetaRequest(NativeRequest):
    """Body for ``POST .../instruments/sae/features/metadata``."""

    ids: list[int] = Field(min_length=1, max_length=64)


# -- preparation operation bodies (re-parsed from the {operation, ...} POST) --

class LensFetchRequest(NativeRequest):
    source: str = "neuronpedia"
    force: bool = False


class LensFitRequest(NativeRequest):
    """Defaults mirror CLI ``lens fit`` (all source layers).  A matching
    partial fit resumes by default; ``force`` restarts."""

    prompts: int = Field(default=100, ge=1, le=5000)
    seq_len: int | None = Field(default=None, ge=32, le=4096)
    prompt_batch: int | None = Field(default=None, ge=1, le=64)
    layers: str = "all"
    force: bool = False


class SaeLoadRequest(NativeRequest):
    release: str = Field(min_length=1)
    layer: int | None = Field(default=None, ge=0)


class SaeTrainRequest(NativeRequest):
    name: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,63}$")
    layer: int | None = Field(default=None, ge=0)
    tokens: int = Field(default=1_000_000, ge=1, le=100_000_000)
    seq_len: int = Field(default=128, ge=8, le=4096)
    batch_size: int = Field(default=8, ge=1, le=256)
    width: int | None = Field(default=None, ge=1)
    expansion: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=3e-4, gt=0)
    l1: float = Field(default=1e-3, ge=0)
    dead_threshold: float = Field(default=1e-6, ge=0)
    seed: int = 0
    force: bool = False


def _parse_layers(layers: str | None) -> list[int] | str | None:
    """``"3,7,11"`` → ``[3, 7, 11]``; named modes pass through."""
    if layers is None or not layers.strip():
        return None
    lowered = layers.strip().lower()
    if lowered in {"workspace", "band", "sample", "all"}:
        return lowered
    try:
        return [int(part) for part in layers.split(",") if part.strip()]
    except ValueError as e:
        raise HTTPException(
            400,
            f"malformed layers list: {layers!r} "
            "(want csv, workspace, sample, or all)",
        ) from e


def _require_family(family: str) -> str:
    if family not in _FAMILIES:
        raise HTTPException(
            404,
            f"unknown instrument family {family!r} "
            f"(want one of {', '.join(_FAMILIES)})",
        )
    return family


def _active_lens_source(session: Any) -> str | None:
    """The active J-lens source label, from the public source listing."""
    from saklas.io.lens_sources import list_lens_sources

    for row in list_lens_sources(session.model_id):
        if row.get("active"):
            return row.get("source")
    return None


def _sae_source_label(session: Any) -> str | None:
    """The resident SAE's source string (``local:``/``saelens:``), or None."""
    info = session.sae_info
    if not info:
        return None
    release = info.get("release")
    if not release:
        return None
    text = str(release)
    return text if text.startswith(("local:", "saelens:")) else f"saelens:{text}"


def register_instrument_routes(app: FastAPI) -> None:
    """Mount the unified ``/saklas/v1/sessions/{id}/instruments`` route tree."""
    session = app.state.session

    # -- background jobs (shared 409 groups; app.state-backed status dicts) ---
    _LENS_BUSY = "a J-lens artifact operation is already running"
    lens_fit_job = BackgroundJob(
        app,
        "lens_fit",
        {
            "running": False,
            "prompts_done": 0,
            "prompts_total": 0,
            "message": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
            "live_layers": None,
        },
        busy_message=_LENS_BUSY,
        cancellable=True,
        not_running_message="no lens fit is running",
    )
    lens_fetch_job = BackgroundJob(
        app,
        "lens_fetch",
        {
            "running": False,
            "source": None,
            "message": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
            "live_layers": None,
        },
        busy_message=_LENS_BUSY,
    )
    lens_fit_job.share_group(lens_fetch_job)

    _SAE_BUSY = "an SAE artifact operation is already running"
    sae_load_job = BackgroundJob(
        app,
        "sae_load",
        {
            "running": False,
            "release": None,
            "message": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
            "info": None,
        },
        busy_message=_SAE_BUSY,
    )
    sae_train_job = BackgroundJob(
        app,
        "sae_train",
        {
            "running": False,
            "name": None,
            "tokens_done": 0,
            "tokens_total": 0,
            "message": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
            "info": None,
        },
        busy_message=_SAE_BUSY,
        cancellable=True,
        not_running_message="no SAE training is running",
    )
    sae_load_job.share_group(sae_train_job)

    async def _stop_lens_fit() -> None:
        await lens_fit_job.stop()
        await lens_fetch_job.stop()

    async def _stop_sae_train() -> None:
        await sae_train_job.stop()

    app.router.on_shutdown.append(_stop_lens_fit)
    app.router.on_shutdown.append(_stop_sae_train)

    # -- job → common preparation-status shape --------------------------------

    def _map_job_status(
        job: BackgroundJob,
        operation: str,
        *,
        unit: str | None,
        done_field: str | None,
        total_field: str | None,
        extras: tuple[str, ...],
    ) -> dict[str, Any]:
        st = job.status()
        running = bool(st.get("running"))
        error = st.get("error")
        finished = st.get("finished_at")
        if running:
            state = "running"
        elif error:
            state = "error"
        elif finished:
            state = "done"
        else:
            state = "idle"
        progress = None
        if unit is not None and done_field is not None and total_field is not None:
            progress = {
                "current": st.get(done_field),
                "total": st.get(total_field),
                "unit": unit,
            }
        common: dict[str, Any] = {
            "state": state,
            "operation": operation,
            "progress": progress,
            "message": st.get("message"),
            "error": error,
            "started_at": st.get("started_at"),
            "finished_at": finished,
            "cancellable": job.cancellable,
        }
        for key in extras:
            common[key] = st.get(key)
        return common

    _JOB_SPECS = {
        "lens": (
            (lens_fetch_job, "fetch", None, None, None, ("source", "live_layers")),
            (lens_fit_job, "fit", "prompts", "prompts_done", "prompts_total",
             ("live_layers",)),
        ),
        "sae": (
            (sae_load_job, "load", None, None, None, ("release", "info")),
            (sae_train_job, "train", "tokens", "tokens_done", "tokens_total",
             ("name", "info")),
        ),
    }

    def _idle_status() -> dict[str, Any]:
        return {
            "state": "idle",
            "operation": None,
            "progress": None,
            "message": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
            "cancellable": False,
        }

    def _prep_status(family: str) -> dict[str, Any]:
        specs = _JOB_SPECS.get(family)
        if not specs:
            return _idle_status()
        running = [s for s in specs if s[0].running]
        if running:
            job, op, unit, done, total, extras = running[0]
        else:
            started = [s for s in specs if s[0].state.get("started_at")]
            if not started:
                return _idle_status()
            job, op, unit, done, total, extras = max(
                started,
                key=lambda s: (
                    s[0].state.get("finished_at")
                    or s[0].state.get("started_at")
                    or 0
                ),
            )
        return _map_job_status(
            job, op, unit=unit, done_field=done, total_field=total, extras=extras,
        )

    # -- shared lens-source activation ---------------------------------------

    async def _activate_lens_source(source: str) -> list[int]:
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise RuntimeError("session locked")
            session.disable_live_lens()
            await asyncio.to_thread(session.select_jlens_source, source)
            return await asyncio.to_thread(session.enable_live_lens)

    # =====================================================================
    # GET /instruments — enumerate the three families
    # =====================================================================

    def _geometry_family() -> dict[str, Any]:
        return {
            "family": "geometry",
            "live": {"enabled": bool(session.live_probe_scores)},
            "source": None,
            "probes": list(session.monitor.probe_names),
            "capabilities": {
                "sources": False,
                "preparations": [],
                "token_readout": True,
                "source_switch": False,
            },
        }

    def _lens_family() -> dict[str, Any]:
        layers = session.live_lens_layers
        return {
            "family": "lens",
            "live": {"enabled": layers is not None, "layers": layers},
            "source": _active_lens_source(session),
            "probes": list(session.lens_probe_names),
            "capabilities": {
                "sources": True,
                "preparations": ["fetch", "fit"],
                "token_readout": True,
                "source_switch": True,
            },
        }

    def _sae_family() -> dict[str, Any]:
        live_on = bool(session.live_sae)
        live_cfg = getattr(session, "_live_sae", None) if live_on else None
        if isinstance(live_cfg, dict):
            live = {
                "enabled": True,
                "layer": live_cfg.get("layer"),
            }
        else:
            live = {"enabled": live_on, "layer": None}
        return {
            "family": "sae",
            "live": live,
            "source": _sae_source_label(session),
            "probes": list(session.sae_probe_names),
            "capabilities": {
                "sources": True,
                "preparations": ["load", "train"],
                "token_readout": True,
                "source_switch": False,
            },
        }

    @app.get("/saklas/v1/sessions/{session_id}/instruments")
    def list_instruments(session_id: str):
        """Enumerate the geometry / lens / sae read families."""
        resolve_session_id(session_id)
        return {
            "instruments": [
                _geometry_family(),
                _lens_family(),
                _sae_family(),
            ],
        }

    # =====================================================================
    # POST /instruments/{family}/live — uniform live toggle
    # =====================================================================

    @app.post("/saklas/v1/sessions/{session_id}/instruments/{family}/live")
    async def instrument_live(
        session_id: str, family: str, body: InstrumentLiveRequest,
    ):
        resolve_session_id(session_id)
        _require_family(family)

        if family == "geometry":
            if body.layers is not None or body.top_k is not None:
                raise HTTPException(
                    400,
                    "geometry live takes no layers/top_k (per-token monitor "
                    "scoring is all-or-nothing)",
                )
            async with acquire_session_lock(session) as acquired:
                if not acquired:
                    raise HTTPException(503, "session locked")
                enabled = bool(session.set_live_probe_scores(body.enabled))
            return {"enabled": enabled}

        if family == "lens":
            if body.top_k is not None:
                raise HTTPException(400, "lens live takes no top_k")
            async with acquire_session_lock(session) as acquired:
                if not acquired:
                    raise HTTPException(503, "session locked")
                if not body.enabled:
                    session.disable_live_lens()
                    return {"enabled": False, "layers": None}
                try:
                    resolved = await asyncio.to_thread(
                        session.enable_live_lens, layers=body.layers,
                    )
                except LensNotFittedError as e:
                    raise HTTPException(404, str(e)) from e
                except ValueError as e:
                    raise HTTPException(400, str(e)) from e
                except SaklasError as e:
                    status, text = e.user_message()
                    raise HTTPException(status, text) from e
            return {"enabled": True, "layers": resolved}

        # sae
        if body.layers is not None or body.top_k is not None:
            raise HTTPException(
                400,
                "sae live takes no layers/top_k; readout width follows alts",
            )
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                if body.enabled:
                    state = session.enable_live_sae()
                    return {
                        "enabled": True,
                        "layer": state.get("layer"),
                    }
                session.disable_live_sae()
                return {"enabled": False, "layer": None}
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            except SaklasError as e:
                status, text = e.user_message()
                raise HTTPException(status, text) from e

    # =====================================================================
    # GET /instruments/{family}/sources
    # =====================================================================

    @app.get("/saklas/v1/sessions/{session_id}/instruments/{family}/sources")
    async def instrument_sources(session_id: str, family: str):
        resolve_session_id(session_id)
        _require_family(family)
        if family == "geometry":
            raise HTTPException(
                404,
                "geometry has no source lifecycle (Monitor probes attach "
                "directly; there is nothing to fetch or switch)",
            )
        if family == "lens":
            from saklas.io.lens_sources import list_lens_sources

            return {
                "sources": [
                    {k: v for k, v in row.items() if k != "path"}
                    for row in list_lens_sources(session.model_id)
                ],
            }
        # sae: prepared sources AND provider release candidates
        from saklas.core.sae import list_sae_releases
        from saklas.io.sae import list_sae_sources

        rows = await asyncio.to_thread(list_sae_sources, session.model_id)
        sources = [{k: v for k, v in row.items() if k != "path"} for row in rows]
        try:
            releases = await asyncio.to_thread(
                list_sae_releases, session.model_id,
            )
        except SaklasError as exc:
            status, message = exc.user_message()
            raise HTTPException(status, message) from exc
        return {"sources": sources, "releases": releases}

    # =====================================================================
    # PUT /instruments/{family}/source — synchronous source switch (lens)
    # =====================================================================

    @app.put("/saklas/v1/sessions/{session_id}/instruments/{family}/source")
    async def instrument_source(session_id: str, family: str, body: SourceRequest):
        resolve_session_id(session_id)
        _require_family(family)
        if family == "geometry":
            raise HTTPException(
                404, "geometry has no source to switch",
            )
        if family == "sae":
            raise HTTPException(
                409,
                "switching the SAE source loads weights — run it as a "
                "background preparation (POST .../instruments/sae/preparations "
                "with operation='load')",
            )
        # lens: the old POST /lens/use semantics
        lens_fit_job.refuse_if_busy()
        try:
            layers = await _activate_lens_source(body.source)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except LensNotFittedError as exc:
            raise HTTPException(409, str(exc)) from exc
        except SaklasError as exc:
            status, text = exc.user_message()
            raise HTTPException(status, text) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"source": body.source, "live_layers": layers}

    # =====================================================================
    # POST/GET/DELETE /instruments/{family}/preparations — background jobs
    # =====================================================================

    async def _lens_fetch_body(body: LensFetchRequest) -> None:
        from saklas.io.lens_sources import fetch_neuronpedia_lens

        st = lens_fetch_job.state
        if body.source != "neuronpedia":
            raise ValueError("J-lens source must be neuronpedia")
        st["message"] = "fetching official lens into the Hugging Face cache…"
        binding = await asyncio.to_thread(
            fetch_neuronpedia_lens,
            session.model_id,
            force=body.force,
            activate=False,
        )
        st["message"] = "activating…"
        st["live_layers"] = await _activate_lens_source(binding.name)
        st["message"] = (
            f"active: {binding.name} ({len(binding.source_layers)} layers)"
        )
        st["error"] = None

    def _lens_fetch_on_error(exc: BaseException) -> None:
        st = lens_fetch_job.state
        if isinstance(exc, FileNotFoundError):
            st["error"] = str(exc)
            st["message"] = "official lens unavailable"
        elif isinstance(exc, (ValueError, RuntimeError)):
            st["error"] = str(exc)
            st["message"] = "fetch failed"
        else:
            log.exception("J-lens fetch failed")
            st["error"] = f"J-lens fetch failed ({type(exc).__name__})"
            st["message"] = "fetch failed"

    async def _lens_fit_body(
        body: LensFitRequest, source_layers: "list[int] | str",
    ) -> None:
        from saklas.io.lens import stream_default_lens_corpus

        st = lens_fit_job.state
        st["message"] = f"streaming {body.prompts} corpus documents…"
        docs, spec = await asyncio.to_thread(
            stream_default_lens_corpus,
            body.prompts,
            cancel_event=lens_fit_job.cancel_event,
        )
        on_progress = make_progress_hook(
            st, _FIT_PROGRESS_RE,
            done_field="prompts_done", total_field="prompts_total",
        )
        st["message"] = "fitting…"
        await asyncio.to_thread(
            session.fit_jlens,
            docs,
            corpus_spec=spec,
            source_layers=source_layers,
            seq_len=body.seq_len,
            prompt_batch=body.prompt_batch,
            force=body.force,
            on_progress=on_progress,
            cancel_event=lens_fit_job.cancel_event,
        )
        async with acquire_session_lock(session) as acquired:
            if acquired:
                st["live_layers"] = await asyncio.to_thread(
                    session.enable_live_lens,
                )
        st["message"] = "done"
        st["error"] = None

    async def _sae_load_body(body: SaeLoadRequest, source: str, release: str) -> None:
        st = sae_load_job.state
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise RuntimeError("session locked")
            info = await asyncio.to_thread(
                session.load_sae, release, layer=body.layer,
            )
            await asyncio.to_thread(session.enable_live_sae)
        st["info"] = info
        st["message"] = (
            f"loaded {source} · live at L{info.get('layer')} "
            f"({info.get('width')} features)"
        )

    def _sae_load_on_error(exc: BaseException) -> None:
        st = sae_load_job.state
        if isinstance(exc, SaklasError):
            _code, message = exc.user_message()
        else:
            message = str(exc)
        st["error"] = message
        st["message"] = "load failed"

    async def _sae_train_body(body: SaeTrainRequest, layer: int) -> None:
        from saklas.io.lens import stream_default_lens_corpus

        st = sae_train_job.state
        n_docs = max(1, math.ceil(body.tokens / body.seq_len))
        st["message"] = f"streaming {n_docs:,} corpus documents…"
        docs, spec = await asyncio.to_thread(stream_default_lens_corpus, n_docs)
        on_progress = make_progress_hook(
            st, _TRAIN_PROGRESS_RE,
            done_field="tokens_done", total_field="tokens_total",
        )
        result = await asyncio.to_thread(
            session.train_sae,
            body.name,
            docs,
            layer=layer,
            corpus_spec=spec,
            tokens=body.tokens,
            seq_len=body.seq_len,
            batch_size=body.batch_size,
            d_sae=body.width,
            expansion=body.expansion,
            learning_rate=body.learning_rate,
            l1_coefficient=body.l1,
            dead_feature_threshold=body.dead_threshold,
            seed=body.seed,
            force=body.force,
            on_progress=on_progress,
            cancel_event=sae_train_job.cancel_event,
        )
        st["info"] = result["runtime"]
        st["tokens_done"] = int(result["metrics"]["tokens_trained"])
        st["message"] = f"active: {result['source']}"
        st["error"] = None
        try:
            async with acquire_session_lock(session) as acquired:
                if acquired:
                    await asyncio.to_thread(session.enable_live_sae)
        except Exception:
            log.exception("could not auto-enable live SAE after training")

    def _start_lens_fetch(fields: dict[str, Any]) -> dict[str, Any]:
        body = LensFetchRequest(**fields)
        lens_fetch_job.refuse_if_busy()
        lens_fetch_job.start(
            message="starting…", source=body.source, live_layers=None,
        )
        lens_fetch_job.launch(
            lambda: _lens_fetch_body(body), _lens_fetch_on_error,
        )
        return _prep_status("lens")

    def _start_lens_fit(fields: dict[str, Any]) -> dict[str, Any]:
        from saklas.core.jlens import JacobianLensCancelled

        body = LensFitRequest(**fields)
        source_layers = _parse_layers(body.layers) or "workspace"
        if source_layers == "sample":
            raise HTTPException(
                400,
                "layers='sample' is not fittable (debug readout only) — "
                "use 'workspace', 'all', or an explicit csv list",
            )
        lens_fit_job.refuse_if_busy()
        lens_fit_job.start(
            message="starting…", prompts_done=0, prompts_total=body.prompts,
        )

        def _on_error(exc: BaseException) -> None:
            scrub_job_error(
                lens_fit_job.state, exc,
                cancel_exc=JacobianLensCancelled,
                op_label="lens fit",
                logger=log,
            )

        lens_fit_job.launch(
            lambda: _lens_fit_body(body, source_layers), _on_error,
        )
        return _prep_status("lens")

    def _start_sae_load(fields: dict[str, Any]) -> dict[str, Any]:
        body = SaeLoadRequest(**fields)
        sae_load_job.refuse_if_busy()
        source = body.release.strip()
        release = (
            source[len("saelens:"):] if source.startswith("saelens:") else source
        )
        if not release:
            raise HTTPException(400, "SAE source must not be empty")
        sae_load_job.start(message=f"loading {source}", release=source, info=None)
        sae_load_job.launch(
            lambda: _sae_load_body(body, source, release), _sae_load_on_error,
        )
        return _prep_status("sae")

    def _start_sae_train(fields: dict[str, Any]) -> dict[str, Any]:
        from saklas.core.sae_training import SaeTrainingCancelled

        body = SaeTrainRequest(**fields)
        sae_train_job.refuse_if_busy()
        layer = body.layer
        if layer is None:
            layer = round(0.65 * max(len(session.layers) - 1, 0))
        if not 0 <= layer < len(session.layers):
            raise HTTPException(
                400,
                f"SAE layer {layer} is outside model layers "
                f"0..{len(session.layers) - 1}",
            )
        sae_train_job.start(
            message="starting…",
            name=body.name,
            tokens_done=0,
            tokens_total=body.tokens,
            info=None,
        )

        def _on_error(exc: BaseException) -> None:
            scrub_job_error(
                sae_train_job.state, exc,
                cancel_exc=SaeTrainingCancelled,
                op_label="SAE training",
                logger=log,
                failure_message="training failed",
            )

        sae_train_job.launch(lambda: _sae_train_body(body, layer), _on_error)
        return _prep_status("sae")

    _PREP_STARTERS = {
        ("lens", "fetch"): _start_lens_fetch,
        ("lens", "fit"): _start_lens_fit,
        ("sae", "load"): _start_sae_load,
        ("sae", "train"): _start_sae_train,
    }

    @app.post(
        "/saklas/v1/sessions/{session_id}/instruments/{family}/preparations",
        status_code=202,
    )
    async def preparations_start(session_id: str, family: str, body: dict[str, Any]):
        resolve_session_id(session_id)
        _require_family(family)
        if family == "geometry":
            raise HTTPException(
                404, "geometry has no background preparations",
            )
        operation = body.get("operation")
        if not isinstance(operation, str):
            raise HTTPException(400, "body must carry a string 'operation'")
        starter = _PREP_STARTERS.get((family, operation))
        if starter is None:
            valid = [op for (fam, op) in _PREP_STARTERS if fam == family]
            raise HTTPException(
                400,
                f"{family} preparations support {valid}, not {operation!r}",
            )
        fields = {k: v for k, v in body.items() if k != "operation"}
        try:
            return starter(fields)
        except ValidationError as exc:
            raise HTTPException(400, exc.errors()) from exc

    @app.get(
        "/saklas/v1/sessions/{session_id}/instruments/{family}/preparations",
    )
    def preparations_status(session_id: str, family: str):
        resolve_session_id(session_id)
        _require_family(family)
        if family == "geometry":
            raise HTTPException(
                404, "geometry has no background preparations",
            )
        return _prep_status(family)

    @app.delete(
        "/saklas/v1/sessions/{session_id}/instruments/{family}/preparations",
    )
    async def preparations_cancel(session_id: str, family: str):
        resolve_session_id(session_id)
        _require_family(family)
        if family == "geometry":
            raise HTTPException(
                404, "geometry has no background preparations",
            )
        if family == "lens":
            # Only the fit is cancellable (the fetch is not).
            lens_fit_job.request_cancel()
            return _prep_status("lens")
        # sae: cancel a running train, else tear down (unload) the resident SAE.
        if sae_train_job.running:
            sae_train_job.request_cancel()
            return _prep_status("sae")
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            session.unload_sae()
        return _prep_status("sae")

    # =====================================================================
    # GET /instruments/{family}/token-readout — measurements replay envelope
    # =====================================================================

    @app.get(
        "/saklas/v1/sessions/{session_id}/instruments/{family}/token-readout",
    )
    async def instrument_token_readout(
        session_id: str,
        family: str,
        node_id: str,
        raw_index: int,
        top_k: int = 8,
        steered: bool = True,
        raw: bool = False,
        layers: str | None = None,
    ):
        resolve_session_id(session_id)
        _require_family(family)
        if family == "geometry":
            # ``top_k``/``layers`` don't apply: the roster's own fitted
            # layers drive the capture (same silent-ignore the SAE
            # branch gives ``layers``).
            return await _geometry_token_readout(
                node_id, raw_index, steered, raw,
            )
        if family == "lens":
            return await _lens_token_readout(
                node_id, raw_index, top_k, steered, raw, layers,
            )
        return await _sae_token_readout(node_id, raw_index, top_k, steered, raw)

    async def _geometry_token_readout(
        node_id: str, raw_index: int, steered: bool, raw: bool,
    ) -> dict[str, Any]:
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                out = await asyncio.to_thread(
                    session.geometry_token_readout,
                    node_id,
                    raw_index,
                    apply_steering=steered,
                    raw=raw,
                )
            except UnknownNodeError as exc:
                raise HTTPException(404, str(exc)) from exc
            except InvalidNodeOperationError as exc:
                raise HTTPException(400, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except SaklasError as exc:
                status, message = exc.user_message()
                raise HTTPException(status, message) from exc
        measurements = build_measurements(
            scope="replay",
            provenance="replayed",
            geometry_readings=out.get("readings"),
            # The shared MeasurementBinding wire shape carries both keys;
            # geometry has no source lifecycle, so source is always null.
            geometry_binding={
                "source": None,
                "steering": (out.get("steering") if steered else None),
            },
        )
        return {"measurements": measurements}

    async def _lens_token_readout(
        node_id: str, raw_index: int, top_k: int, steered: bool,
        raw: bool, layers: str | None,
    ) -> dict[str, Any]:
        req_layers = _parse_layers(layers) or "all"
        if not 1 <= top_k <= 256:
            raise HTTPException(400, "top_k must be in [1, 256]")
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                out = await asyncio.to_thread(
                    session.jlens_token_readout,
                    node_id,
                    raw_index,
                    layers=req_layers,
                    top_k=top_k,
                    apply_steering=steered,
                    raw=raw,
                )
            except (LensNotFittedError, UnknownNodeError) as e:
                raise HTTPException(404, str(e)) from e
            except InvalidNodeOperationError as e:
                raise HTTPException(400, str(e)) from e
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            except SaklasError as e:
                status, text = e.user_message()
                raise HTTPException(status, text) from e
        readout = out.get("readout", {})
        lens_readout = {
            int(layer): [(str(tok), math.exp(float(lp))) for tok, lp, _tid in rows]
            for layer, rows in readout.items()
        }
        lens_token_ids = {
            int(layer): [int(tid) for _tok, _lp, tid in rows]
            for layer, rows in readout.items()
        }
        lens_aggregate = [
            (str(tok), float(strength), float(com), float(spread))
            for tok, strength, com, spread in out.get("aggregate", [])
        ]
        measurements = build_measurements(
            scope="replay",
            provenance="replayed",
            lens_readout=lens_readout,
            lens_aggregate=lens_aggregate,
            lens_token_ids=lens_token_ids,
            lens_source=_active_lens_source(session),
            steering=(out.get("steering") if steered else None),
        )
        return {"measurements": measurements}

    async def _sae_token_readout(
        node_id: str, raw_index: int, top_k: int, steered: bool, raw: bool,
    ) -> dict[str, Any]:
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                out = await asyncio.to_thread(
                    session.sae_token_readout,
                    node_id,
                    raw_index,
                    top_k=top_k,
                    apply_steering=steered,
                    raw=raw,
                )
            except UnknownNodeError as exc:
                raise HTTPException(404, str(exc)) from exc
            except InvalidNodeOperationError as exc:
                raise HTTPException(400, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except SaklasError as exc:
                status, message = exc.user_message()
                raise HTTPException(status, message) from exc
        sae_features = [
            (
                int(f["id"]),
                float(f["activation"]),
                f.get("label"),
                f.get("max_act"),
            )
            for f in out.get("features", [])
        ]
        measurements = build_measurements(
            scope="replay",
            provenance="replayed",
            sae_features=sae_features,
            sae_source=_sae_source_label(session),
            sae_layer=out.get("layer"),
            steering=(out.get("steering") if steered else None),
        )
        return {"measurements": measurements}

    # =====================================================================
    # Family extras
    # =====================================================================

    @app.post(
        "/saklas/v1/sessions/{session_id}/instruments/lens/token/validate",
    )
    def validate_lens_token(session_id: str, body: LensTokenValidationRequest):
        """Read-only single-token check for the J-lens steer/probe add forms."""
        resolve_session_id(session_id)
        word = body.word.strip()
        if not word:
            raise HTTPException(400, "word must not be empty")
        try:
            token_id = resolve_word_token(session.tokenizer, word)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"word": word, "token_id": token_id}

    @app.post(
        "/saklas/v1/sessions/{session_id}/instruments/sae/features/validate",
    )
    def sae_feature_validate(session_id: str, body: SaeFeatureRequest):
        resolve_session_id(session_id)
        try:
            return session.validate_sae_feature(body.id)
        except SaklasError as exc:
            status, message = exc.user_message()
            raise HTTPException(status, message) from exc

    @app.post(
        "/saklas/v1/sessions/{session_id}/instruments/sae/features/metadata",
    )
    async def sae_features_metadata(session_id: str, body: SaeFeatureMetaRequest):
        """Fetch-and-cache Neuronpedia metadata (label + maxActApprox).

        Network + disk-cache only (no model use), so it deliberately does not
        take the session lock, mirroring feature validation.
        """
        resolve_session_id(session_id)
        if any(feature_id < 0 for feature_id in body.ids):
            raise HTTPException(400, "feature ids must be non-negative")
        try:
            features = await asyncio.to_thread(
                session.fetch_sae_feature_meta, body.ids,
            )
        except SaklasError as exc:
            status, message = exc.user_message()
            raise HTTPException(status, message) from exc
        return {"features": features}
