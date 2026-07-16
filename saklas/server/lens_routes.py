"""Native Jacobian-lens route group — readout surfaces.

Routes under ``/saklas/v1/sessions/{id}/lens``:

- ``POST .../token/validate`` — read-only single-token validation for the
  dashboard's J-lens steer/probe add forms.
- ``GET .../token-readout`` — the dashboard's token-drilldown ``j-lens`` tab
  asks for the per-layer readout at a clicked token
  (``session.jlens_token_readout`` — rebuild the node's prompt render + raw
  decode prefix, one capture forward under the node's recipe steering,
  ``softmax(W_U · norm(J_l h))`` top-k per fitted layer).
- ``POST .../live`` — toggle the *live* J-lens readout
  (``session.enable_live_lens`` / ``disable_live_lens``): while enabled, the
  per-decode-step top-k rides the native WS ``token`` frame's
  ``lens_readout`` channel (see ``ws_events.build_token_event``), and the
  session-info ``live_lens_layers`` field carries the resolved layer list.
- ``POST .../fit`` / ``GET .../fit`` — kick off / poll a background lens
  fit (the dashboard's "fit j-lens" button).  The fit is hours of wall
  clock (compute-bound backward passes), so it runs as ONE background
  task with a polled status dict rather than an SSE stream — progress
  must survive page reloads, and the engine-side fit checkpoints/resumes
  regardless of what happens to the client.  Generations attempted while
  a fit holds the model raise cleanly through the ordinary busy path.

Discovery rides ``jlens_fitted`` on the session info payload (a v6 shard
sidecar/payload and live-weight compatibility check, never the ~GB fp32 lens load).
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
import logging
import re

from fastapi import FastAPI, HTTPException
from pydantic import Field

from saklas.core.errors import SaklasError
from saklas.core.jlens import LensNotFittedError, resolve_word_token
from saklas.core.loom import InvalidNodeOperationError, UnknownNodeError
from saklas.server.app import acquire_session_lock
from saklas.server.background_job import (
    BackgroundJob,
    make_progress_hook,
    scrub_job_error,
)
from saklas.server.native_common import NativeRequest, resolve_session_id

log = logging.getLogger(__name__)

#: ``fit_jacobian_lens`` per-prompt progress line — "prompt 12/100 (…)".
_FIT_PROGRESS_RE = re.compile(r"prompt (\d+)/(\d+)")


class LiveLensRequest(NativeRequest):
    """Body for ``POST .../lens/live``.

    ``layers`` is an explicit fitted-layer list; omitted, the session enables
    every fitted layer. The per-generation logit-alternative count also sets
    the live J-lens top-k width.
    """

    enabled: bool
    layers: list[int] | None = None


class LensTokenValidationRequest(NativeRequest):
    """Body for ``POST .../lens/token/validate``."""

    word: str


class LensFitRequest(NativeRequest):
    """Body for ``POST .../fit`` — all fields optional.

    Defaults mirror CLI ``lens fit``, including all source layers. A matching
    partial fit resumes by default; ``force`` restarts.
    """

    prompts: int = Field(default=100, ge=1, le=5000)
    seq_len: int | None = Field(default=None, ge=32, le=4096)
    prompt_batch: int | None = Field(default=None, ge=1, le=64)
    layers: str = "all"
    force: bool = False


class LensFetchRequest(NativeRequest):
    source: str = "neuronpedia"
    force: bool = False


class LensUseRequest(NativeRequest):
    source: str = Field(min_length=1)


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


def register_lens_routes(app: FastAPI) -> None:
    """Mount the Jacobian-lens read + fit routes."""
    session = app.state.session

    # A J-lens artifact operation — a background fit or an official fetch — runs
    # one at a time; the two jobs guard each other (fetch XOR fit).  The status
    # dicts are the polled surfaces; plain-dict mutation from the worker thread's
    # ``on_progress`` is safe under the GIL (single-writer, readers only format).
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

    async def _stop_lens_fit() -> None:
        await lens_fit_job.stop()
        await lens_fetch_job.stop()

    app.router.on_shutdown.append(_stop_lens_fit)

    @app.get("/saklas/v1/sessions/{session_id}/lens/sources")
    def lens_sources(session_id: str):
        """List fitted local lenses and fetched external bindings."""
        resolve_session_id(session_id)
        from saklas.io.lens_sources import list_lens_sources

        return {
            "sources": [
                {key: value for key, value in row.items() if key != "path"}
                for row in list_lens_sources(session.model_id)
            ],
        }

    async def _activate_lens_source(source: str) -> list[int]:
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise RuntimeError("session locked")
            session.disable_live_lens()
            await asyncio.to_thread(session.select_jlens_source, source)
            return await asyncio.to_thread(session.enable_live_lens)

    @app.post("/saklas/v1/sessions/{session_id}/lens/use")
    async def lens_use(session_id: str, body: LensUseRequest):
        """Select an already fitted/fetched source and turn its readout on."""
        resolve_session_id(session_id)
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

    async def _lens_fetch_job(body: LensFetchRequest) -> None:
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

    def _fetch_on_error(exc: BaseException) -> None:
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

    @app.post("/saklas/v1/sessions/{session_id}/lens/fetch", status_code=202)
    async def lens_fetch_start(
        session_id: str, body: LensFetchRequest | None = None,
    ):
        resolve_session_id(session_id)
        body = body or LensFetchRequest()
        lens_fetch_job.refuse_if_busy()
        lens_fetch_job.start(
            message="starting…", source=body.source, live_layers=None,
        )
        lens_fetch_job.launch(lambda: _lens_fetch_job(body), _fetch_on_error)
        return lens_fetch_job.status()

    @app.get("/saklas/v1/sessions/{session_id}/lens/fetch")
    async def lens_fetch_status(session_id: str):
        resolve_session_id(session_id)
        return lens_fetch_job.status()

    @app.post("/saklas/v1/sessions/{session_id}/lens/token/validate")
    def validate_lens_token(
        session_id: str, body: LensTokenValidationRequest,
    ):
        """Resolve a prospective J-lens atom without changing session state.

        Both dashboard add forms call this before adding their rack/probe
        card.  Steering and probe registration still revalidate at their
        engine boundaries, so this is an early UX check rather than a bypass
        of the core invariant.
        """
        resolve_session_id(session_id)
        word = body.word.strip()
        if not word:
            raise HTTPException(400, "word must not be empty")
        try:
            token_id = resolve_word_token(session.tokenizer, word)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"word": word, "token_id": token_id}

    @app.get("/saklas/v1/sessions/{session_id}/lens/token-readout")
    async def lens_token_readout(
        session_id: str,
        node_id: str,
        raw_index: int,
        top_k: int = 8,
        steered: bool = True,
        raw: bool = False,
        layers: str | None = None,
    ):
        """Workspace readout at one decode step of a loom node.

        ``steered`` (default on) replays under the node's recipe steering —
        exact for always-active affine terms, the dominant case; pass
        ``steered=false`` for the unsteered counterfactual read of the same
        token stream.  ``raw`` selects the flat (base-model / raw-buffer)
        render; the client supplies it because raw-ness isn't stamped on
        the node.  ``layers`` restricts the readout (csv or workspace/sample/all);
        default is every fitted layer.
        """
        resolve_session_id(session_id)
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
                # Steering-expression resolution / whitener prerequisites /
                # busy-model races — the family carries its own status.
                status, text = e.user_message()
                raise HTTPException(status, text) from e
        return {
            "node_id": out["node_id"],
            "raw_index": out["raw_index"],
            "token_id": out["token_id"],
            "token_text": out["token_text"],
            "steering": out["steering"],
            # Layer-aggregated view of the same logits across all requested
            # layers: mean probability + probability-mass-weighted depth
            # center of mass, strength-descending.
            "aggregate": [
                {
                    "token": tok,
                    "strength": round(strength, 6),
                    "com": round(com, 4),
                    "spread": round(spread, 4),
                }
                for tok, strength, com, spread in out.get("aggregate", [])
            ],
            "layers": [
                {
                    "layer": layer,
                    "tokens": [
                        {"token": tok, "id": tid, "logprob": round(lp, 4)}
                        for tok, lp, tid in rows
                    ],
                }
                for layer, rows in sorted(out["readout"].items())
            ],
        }

    @app.post("/saklas/v1/sessions/{session_id}/lens/live")
    async def lens_live_toggle(session_id: str, body: LiveLensRequest):
        """Enable/disable the live J-lens readout for this session.

        Enabling moves the selected layers' ``J_l`` device-resident and
        arms the per-decode-step top-k on the WS ``token`` frame
        (``lens_readout``); disabling frees the device copies.  Applies to
        generations started after the call — the toggle waits on the
        session lock, so it never races an in-flight stream.
        """
        resolve_session_id(session_id)
        async with acquire_session_lock(session) as acquired:
            if not acquired:
                raise HTTPException(503, "session locked")
            if not body.enabled:
                session.disable_live_lens()
                return {"enabled": False, "layers": None}
            try:
                resolved = await asyncio.to_thread(
                    session.enable_live_lens,
                    layers=body.layers,
                )
            except LensNotFittedError as e:
                raise HTTPException(404, str(e)) from e
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            except SaklasError as e:
                status, text = e.user_message()
                raise HTTPException(status, text) from e
        return {"enabled": True, "layers": resolved}

    async def _lens_fit_job(
        body: LensFitRequest, source_layers: "list[int] | str",
    ) -> None:
        """The one background fit: stream corpus → fit → live-on.

        Error frames follow the scrubbing discipline in ``scrub_job_error`` —
        typed saklas errors surface their ``user_message()``, anything else logs
        the traceback server-side and reports only the exception type.
        """
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
        # Live-on by default: hot the full fitted readout the moment the
        # artifact lands, same policy as serve startup.  Session lock so
        # the enable never races a just-unblocked generation.
        async with acquire_session_lock(session) as acquired:
            if acquired:
                st["live_layers"] = await asyncio.to_thread(
                    session.enable_live_lens,
                )
        st["message"] = "done"
        st["error"] = None

    @app.post("/saklas/v1/sessions/{session_id}/lens/fit", status_code=202)
    async def lens_fit_start(session_id: str, body: LensFitRequest | None = None):
        """Kick off a background Jacobian-lens fit (the "fit j-lens" button).

        Returns 202 with the initial status; poll ``GET .../lens/fit``.
        The fit holds the engine's model-exclusive lock, so generations
        attempted while it runs fail with the ordinary busy error.  A
        matching interrupted fit resumes from its last checkpoint.
        """
        from saklas.core.jlens import JacobianLensCancelled

        resolve_session_id(session_id)
        body = body or LensFitRequest()
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
            lambda: _lens_fit_job(body, source_layers), _on_error,
        )
        return lens_fit_job.status()

    @app.get("/saklas/v1/sessions/{session_id}/lens/fit")
    async def lens_fit_status(session_id: str):
        """Poll the background lens fit (progress / error / completion)."""
        resolve_session_id(session_id)
        return lens_fit_job.status()

    @app.delete("/saklas/v1/sessions/{session_id}/lens/fit", status_code=202)
    async def lens_fit_cancel(session_id: str):
        """Cancel corpus acquisition or stop after the current fit pass."""
        resolve_session_id(session_id)
        return lens_fit_job.request_cancel()
