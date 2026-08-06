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
from dataclasses import dataclass
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from pydantic import Field, ValidationError

from saklas.core.errors import SaklasError
from saklas.core.jlens import LensNotFittedError, resolve_word_token
from saklas.core.loom import InvalidNodeOperationError, UnknownNodeError
from saklas.core.measurements import MeasurementsEnvelope
from saklas.server.app import acquire_session_lock
from saklas.server.background_job import (
    BackgroundJob,
    make_progress_hook,
    scrub_job_error,
)
from saklas.server.native_common import NativeRequest, resolve_session_id
from saklas.server.response_models import (
    InstrumentCapabilities,
    InstrumentFamilyBlock,
    InstrumentLiveState,
    InstrumentSourceJSON,
    InstrumentsResponse,
    LensTokenValidationJSON,
    PreparationStatusJSON,
    SaeFeatureMetaResponse,
    SaeFeatureValidationJSON,
    SaeReleaseJSON,
    SourceSwitchResponse,
    SourcesResponse,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------------
# The wire contract used to be maintained by hand in TypeScript.  It now
# lives in ``server/response_models.py`` as the Python-side declaration every
# non-streaming native route annotates: FastAPI emits a real OpenAPI response
# schema from it, ``scripts/generate_webui_types.py`` renders
# ``webui/src/lib/types.gen.ts`` from that schema, and
# ``tests/test_measurements_envelope.py`` pins the exact key sets — so a shape
# change that isn't mirrored fails a test or the parity check rather than a
# dashboard.

@dataclass(frozen=True)
class FamilyCapabilities:
    """What a read family supports over HTTP — ONE declaration.

    The route dispatch, the ``GET /instruments`` listing's ``capabilities``
    block, and ``session_info``'s per-family block all read this, so an
    unsupported operation answers from the declaration rather than from a
    hand-written ``if family == …`` branch.  ``*_error`` carries the status
    and message an unsupported operation returns, so a family that *can't*
    do something says why in one place.
    """

    sources: bool
    preparations: tuple[str, ...]
    token_readout: bool
    source_switch: bool
    sources_error: "tuple[int, str] | None" = None
    source_switch_error: "tuple[int, str] | None" = None
    preparations_error: "tuple[int, str] | None" = None

    def to_dict(self) -> InstrumentCapabilities:
        return {
            "sources": self.sources,
            "preparations": list(self.preparations),
            "token_readout": self.token_readout,
            "source_switch": self.source_switch,
        }


_NO_SOURCE_LIFECYCLE = (
    404,
    "geometry has no source lifecycle (Monitor probes attach directly; "
    "there is nothing to fetch or switch)",
)

#: Per-family capability declaration, keyed by the same family names
#: ``session.instruments`` uses (the route layer validates ``{family}``
#: against the registry, so this table and the registry cannot disagree
#: about which families exist).
CAPABILITIES: dict[str, FamilyCapabilities] = {
    "geometry": FamilyCapabilities(
        sources=False,
        preparations=(),
        token_readout=True,
        source_switch=False,
        sources_error=_NO_SOURCE_LIFECYCLE,
        source_switch_error=(404, "geometry has no source to switch"),
        preparations_error=(404, "geometry has no background preparations"),
    ),
    "lens": FamilyCapabilities(
        sources=True,
        preparations=("fetch", "fit"),
        token_readout=True,
        source_switch=True,
    ),
    "sae": FamilyCapabilities(
        sources=True,
        preparations=("fetch", "train"),
        token_readout=True,
        source_switch=False,
        source_switch_error=(
            409,
            "switching the SAE source loads weights — run it as a "
            "background preparation (POST .../instruments/sae/preparations "
            "with operation='fetch')",
        ),
    ),
}


def family_block(session: Any, family: str) -> InstrumentFamilyBlock:
    """The per-family wire block: ``{family, live, source, probes,
    capabilities}``.

    THE one builder — ``GET /instruments`` lists it for every family and
    ``session_info`` embeds the same list, so the dashboard hydrates one
    representation of instrument state instead of two that drift.
    """
    instrument = session.instruments[family]
    return {
        "family": family,
        "live": instrument.live_state.to_dict(),
        # The active-source accessor is the same resolver that stamps the
        # source onto every measurement binding — a listing that answered
        # from the prepared-sources scan instead would report ``null`` for
        # an active pointer whose artifact is gone while persisted rows
        # still carry its label.
        "source": instrument.active_source,
        "probes": list(instrument.names),
        "capabilities": CAPABILITIES[family].to_dict(),
    }


def instrument_families(session: Any) -> list[InstrumentFamilyBlock]:
    """Every read family's block, in registry order."""
    return [family_block(session, family) for family in session.instruments]


def require_family(session: Any, family: str) -> str:
    """404 unless ``family`` names a registered read instrument."""
    if family not in session.instruments:
        raise HTTPException(
            404,
            f"unknown instrument family {family!r} "
            f"(want one of {', '.join(session.instruments)})",
        )
    return family

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


class SaeFetchRequest(NativeRequest):
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


def _validation_message(exc: ValidationError) -> str:
    """Flatten a pydantic ``ValidationError`` into one ``detail`` string.

    The preparation bodies are re-parsed into per-operation models after the
    ``{operation, …}`` envelope is split, so their failures arrive as a raw
    ``exc.errors()`` list.  The native envelope's ``detail`` is always a
    string, so render the field paths here rather than shipping the list.
    """
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = str(err.get("msg", "invalid value"))
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "invalid preparation fields"


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
    ) -> PreparationStatusJSON:
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
        common: PreparationStatusJSON = {
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
            (sae_load_job, "fetch", None, None, None, ("release", "info")),
            (sae_train_job, "train", "tokens", "tokens_done", "tokens_total",
             ("name", "info")),
        ),
    }

    def _idle_status() -> PreparationStatusJSON:
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

    def _prep_status(family: str) -> PreparationStatusJSON:
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
            session.lens.set_live(False)
            await asyncio.to_thread(session.select_jlens_source, source)
            state = await asyncio.to_thread(session.lens.set_live, True)
            return list(state.layers or ())

    # =====================================================================
    # GET /instruments — enumerate the three families
    # =====================================================================

    @app.get("/saklas/v1/sessions/{session_id}/instruments")
    def list_instruments(session_id: str) -> InstrumentsResponse:
        """Enumerate the read families over ``session.instruments``."""
        resolve_session_id(session_id)
        return {"instruments": instrument_families(session)}

    # =====================================================================
    # POST /instruments/{family}/live — uniform live toggle
    # =====================================================================

    @app.post("/saklas/v1/sessions/{session_id}/instruments/{family}/live")
    async def instrument_live(
        session_id: str, family: str, body: InstrumentLiveRequest,
    ) -> InstrumentLiveState:
        """Uniform live toggle: ``instrument.set_live(enabled, **extras)``.

        Family extras travel as keyword arguments and the *instrument*
        rejects the ones it can't honor (``TypeError`` -> 400), so there is
        one rejection rule instead of three separately-worded branches.
        ``top_k`` is refused for every family up front — readout width is
        generation state shared with ``return_top_k``/alts, never an
        instrument-local dial.
        """
        resolve_session_id(session_id)
        require_family(session, family)
        if body.top_k is not None:
            raise HTTPException(
                400,
                "live takes no top_k; readout width follows the "
                "generation's alternatives (return_top_k)",
            )
        extras: dict[str, Any] = {}
        if body.layers is not None:
            extras["layers"] = body.layers
        instrument = session.instruments[family]
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                state = await asyncio.to_thread(
                    instrument.set_live, body.enabled, **extras,
                )
            except LensNotFittedError as e:
                raise HTTPException(404, str(e)) from e
            except (TypeError, ValueError) as e:
                raise HTTPException(400, str(e)) from e
            except SaklasError as e:
                status, text = e.user_message()
                raise HTTPException(status, text) from e
        return state.to_dict()

    # =====================================================================
    # GET /instruments/{family}/sources
    # =====================================================================

    async def _lens_sources() -> SourcesResponse:
        from saklas.io.lens_sources import list_lens_sources

        return {
            "sources": [
                cast(InstrumentSourceJSON, {
                    k: v for k, v in row.items() if k != "path"
                })
                for row in list_lens_sources(session.model_id)
            ],
        }

    async def _sae_sources() -> SourcesResponse:
        """Prepared sources AND provider release candidates, so the
        dashboard sees both prepared and still-needs-fetching rows."""
        from saklas.core.sae import list_sae_releases
        from saklas.io.sae import list_sae_sources

        rows = await asyncio.to_thread(list_sae_sources, session.model_id)
        sources = [
            cast(InstrumentSourceJSON, {
                k: v for k, v in row.items() if k != "path"
            })
            for row in rows
        ]
        try:
            releases = await asyncio.to_thread(
                list_sae_releases, session.model_id,
            )
        except SaklasError as exc:
            status, message = exc.user_message()
            raise HTTPException(status, message) from exc
        return {
            "sources": sources,
            "releases": cast(list[SaeReleaseJSON], releases),
        }

    _SOURCE_LISTERS = {"lens": _lens_sources, "sae": _sae_sources}

    def _refuse(spec: "tuple[int, str] | None", fallback: str) -> None:
        status, message = spec if spec is not None else (404, fallback)
        raise HTTPException(status, message)

    @app.get("/saklas/v1/sessions/{session_id}/instruments/{family}/sources")
    async def instrument_sources(
        session_id: str, family: str,
    ) -> SourcesResponse:
        resolve_session_id(session_id)
        require_family(session, family)
        caps = CAPABILITIES[family]
        if not caps.sources:
            _refuse(caps.sources_error, f"{family} has no source lifecycle")
        return await _SOURCE_LISTERS[family]()

    # =====================================================================
    # PUT /instruments/{family}/source — synchronous source switch (lens)
    # =====================================================================

    @app.put("/saklas/v1/sessions/{session_id}/instruments/{family}/source")
    async def instrument_source(
        session_id: str, family: str, body: SourceRequest,
    ) -> SourceSwitchResponse:
        resolve_session_id(session_id)
        require_family(session, family)
        caps = CAPABILITIES[family]
        if not caps.source_switch:
            _refuse(
                caps.source_switch_error,
                f"{family} has no synchronous source switch",
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
        response: SourceSwitchResponse = {
            "source": body.source, "live_layers": layers,
        }
        return response

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
                st["live_layers"] = list(
                    (
                        await asyncio.to_thread(session.lens.set_live, True)
                    ).layers or ()
                )
        st["message"] = "done"
        st["error"] = None

    async def _sae_fetch_body(body: SaeFetchRequest, source: str, release: str) -> None:
        st = sae_load_job.state
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise RuntimeError("session locked")
            info = await asyncio.to_thread(
                session.load_sae, release, layer=body.layer,
            )
            await asyncio.to_thread(session.sae.set_live, True)
        st["info"] = info
        st["message"] = (
            f"loaded {source} · live at L{info.get('layer')} "
            f"({info.get('width')} features)"
        )

    def _sae_fetch_on_error(exc: BaseException) -> None:
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
                    await asyncio.to_thread(session.sae.set_live, True)
        except Exception:
            log.exception("could not auto-enable live SAE after training")

    def _start_lens_fetch(fields: dict[str, Any]) -> PreparationStatusJSON:
        body = LensFetchRequest(**fields)
        lens_fetch_job.refuse_if_busy()
        lens_fetch_job.start(
            message="starting…", source=body.source, live_layers=None,
        )
        lens_fetch_job.launch(
            lambda: _lens_fetch_body(body), _lens_fetch_on_error,
        )
        return _prep_status("lens")

    def _start_lens_fit(fields: dict[str, Any]) -> PreparationStatusJSON:
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

    def _start_sae_fetch(fields: dict[str, Any]) -> PreparationStatusJSON:
        body = SaeFetchRequest(**fields)
        sae_load_job.refuse_if_busy()
        source = body.release.strip()
        release = (
            source[len("saelens:"):] if source.startswith("saelens:") else source
        )
        if not release:
            raise HTTPException(400, "SAE source must not be empty")
        sae_load_job.start(message=f"loading {source}", release=source, info=None)
        sae_load_job.launch(
            lambda: _sae_fetch_body(body, source, release), _sae_fetch_on_error,
        )
        return _prep_status("sae")

    def _start_sae_train(fields: dict[str, Any]) -> PreparationStatusJSON:
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

    #: Preparation starters keyed by ``(family, operation)``.  The operation
    #: names match the CLI verbs exactly — ``sae fetch`` is ``fetch`` on both
    #: surfaces, like ``lens fetch`` (the historical HTTP ``load`` spelling
    #: for the same operation is gone; clean break, no alias).
    _PREP_STARTERS = {
        ("lens", "fetch"): _start_lens_fetch,
        ("lens", "fit"): _start_lens_fit,
        ("sae", "fetch"): _start_sae_fetch,
        ("sae", "train"): _start_sae_train,
    }

    @app.post(
        "/saklas/v1/sessions/{session_id}/instruments/{family}/preparations",
        status_code=202,
    )
    async def preparations_start(
        session_id: str, family: str, body: dict[str, Any],
    ) -> PreparationStatusJSON:
        resolve_session_id(session_id)
        require_family(session, family)
        caps = CAPABILITIES[family]
        if not caps.preparations:
            _refuse(
                caps.preparations_error,
                f"{family} has no background preparations",
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
            raise HTTPException(400, _validation_message(exc)) from exc

    @app.get(
        "/saklas/v1/sessions/{session_id}/instruments/{family}/preparations",
    )
    def preparations_status(
        session_id: str, family: str,
    ) -> PreparationStatusJSON:
        resolve_session_id(session_id)
        require_family(session, family)
        caps = CAPABILITIES[family]
        if not caps.preparations:
            _refuse(
                caps.preparations_error,
                f"{family} has no background preparations",
            )
        return _prep_status(family)

    @app.delete(
        "/saklas/v1/sessions/{session_id}/instruments/{family}/preparations",
    )
    async def preparations_cancel(
        session_id: str, family: str,
    ) -> PreparationStatusJSON:
        resolve_session_id(session_id)
        require_family(session, family)
        caps = CAPABILITIES[family]
        if not caps.preparations:
            _refuse(
                caps.preparations_error,
                f"{family} has no background preparations",
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
        top_k: int | None = None,
        steered: bool = True,
        raw: bool = False,
        layers: str | None = None,
    ) -> MeasurementsEnvelope:
        """Resolve -> validate -> dispatch.

        The family owns the whole replay including its ``measurements``
        envelope (``Instrument.token_readout``), so this route no longer
        reshapes three different native dicts.  A knob a family cannot
        honor comes back as a ``ValueError`` -> 400 rather than being
        dropped: ``top_k``/``layers`` on geometry, ``layers`` on sae.
        """
        resolve_session_id(session_id)
        require_family(session, family)
        caps = CAPABILITIES[family]
        if not caps.token_readout:
            raise HTTPException(
                404, f"{family} has no token readout",
            )
        instrument = session.instruments[family]
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            try:
                return await asyncio.to_thread(
                    instrument.token_readout,
                    node_id,
                    raw_index,
                    top_k=top_k,
                    layers=_parse_layers(layers),
                    apply_steering=steered,
                    raw=raw,
                )
            except (LensNotFittedError, UnknownNodeError) as exc:
                raise HTTPException(404, str(exc)) from exc
            except InvalidNodeOperationError as exc:
                raise HTTPException(400, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except SaklasError as exc:
                status, message = exc.user_message()
                raise HTTPException(status, message) from exc

    # =====================================================================
    # Family extras
    # =====================================================================

    @app.post(
        "/saklas/v1/sessions/{session_id}/instruments/lens/token/validate",
    )
    def validate_lens_token(
        session_id: str, body: LensTokenValidationRequest,
    ) -> LensTokenValidationJSON:
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
    def sae_feature_validate(
        session_id: str, body: SaeFeatureRequest,
    ) -> SaeFeatureValidationJSON:
        resolve_session_id(session_id)
        try:
            return session.validate_sae_feature(body.id)
        except SaklasError as exc:
            status, message = exc.user_message()
            raise HTTPException(status, message) from exc

    @app.post(
        "/saklas/v1/sessions/{session_id}/instruments/sae/features/metadata",
    )
    async def sae_features_metadata(
        session_id: str, body: SaeFeatureMetaRequest,
    ) -> SaeFeatureMetaResponse:
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
