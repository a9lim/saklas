"""Native Jacobian-lens route group — the workspace readout surfaces.

Routes under ``/saklas/v1/sessions/{id}/lens``:

- ``POST .../token/validate`` — read-only single-token validation for the
  dashboard's J-lens steer/probe add forms.
- ``GET .../token-readout`` — the dashboard's token-drilldown ``j-lens`` tab
  asks for the per-layer workspace readout at a clicked token
  (``session.jlens_token_readout`` — rebuild the node's prompt render + raw
  decode prefix, one capture forward under the node's recipe steering,
  ``softmax(W_U · norm(J_l h))`` top-k per fitted layer).
- ``POST .../live`` — toggle the *live* workspace readout
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

Discovery rides ``jlens_fitted`` on the session info payload (a v4 shard
sidecar/payload and live-weight compatibility check, never the ~GB fp32 lens load).
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from saklas.core.errors import SaklasError
from saklas.core.jlens import LensNotFittedError, resolve_word_token
from saklas.core.loom import InvalidNodeOperationError, UnknownNodeError
from saklas.server.app import acquire_session_lock
from saklas.server.native_common import resolve_session_id

log = logging.getLogger(__name__)

#: ``fit_jacobian_lens`` per-prompt progress line — "prompt 12/100 (…)".
_FIT_PROGRESS_RE = re.compile(r"prompt (\d+)/(\d+)")


class LiveLensRequest(BaseModel):
    """Body for ``POST .../lens/live``.

    ``layers`` is an explicit fitted-layer list; omitted, the session
    enables every fitted layer in the 40–90% workspace band (the same
    default the TUI's ``/lens`` uses).
    """

    enabled: bool
    layers: list[int] | None = None
    top_k: int = 5


class LensTokenValidationRequest(BaseModel):
    """Body for ``POST .../lens/token/validate``."""

    word: str


class LensFitRequest(BaseModel):
    """Body for ``POST .../fit`` — all fields optional.

    Defaults mirror CLI ``lens fit`` except ``layers``, which defaults to
    the **workspace band** rather than full depth: every dashboard surface
    (live readout, ``jlens/`` steering atoms, the drilldown matrix, the
    aggregate) defaults to the band, and the band-restricted fit measures
    ~1.7× faster.  Pass ``layers="all"`` for a CLI-parity full-depth fit.
    A matching partial fit resumes by default; ``force`` restarts.
    """

    prompts: int = Field(default=100, ge=1, le=5000)
    seq_len: int | None = Field(default=None, ge=32, le=4096)
    prompt_batch: int | None = Field(default=None, ge=1, le=64)
    layers: str = "workspace"
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


def register_lens_routes(app: FastAPI) -> None:
    """Mount the Jacobian-lens read + fit routes."""
    session = app.state.session

    # One background fit at a time; the status dict is the polled surface.
    # Plain-dict mutation from the worker thread's ``on_progress`` is safe
    # under the GIL (single-writer, readers only format it).
    app.state.lens_fit = {
        "running": False,
        "prompts_done": 0,
        "prompts_total": 0,
        "message": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
        "live_layers": None,
    }
    app.state.lens_fit_task = None
    app.state.lens_fit_cancel = None

    async def _stop_lens_fit() -> None:
        event = app.state.lens_fit_cancel
        task = app.state.lens_fit_task
        if event is not None:
            event.set()
        if task is not None and not task.done():
            await task

    app.router.on_shutdown.append(_stop_lens_fit)

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
        resolve_session_id(session, session_id)
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
        default is the fitted workspace band.
        """
        resolve_session_id(session, session_id)
        req_layers = _parse_layers(layers) or "workspace"
        if not 1 <= top_k <= 50:
            raise HTTPException(400, "top_k must be in [1, 50]")
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
        band = set(out["workspace_band"])
        return {
            "node_id": out["node_id"],
            "raw_index": out["raw_index"],
            "token_id": out["token_id"],
            "token_text": out["token_text"],
            "steering": out["steering"],
            # Layer-aggregated view of the same logits (band-restricted):
            # mean band probability + probability-mass-weighted depth
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
                    "in_band": layer in band,
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
        """Enable/disable the live workspace readout for this session.

        Enabling moves the selected layers' ``J_l`` device-resident and
        arms the per-decode-step top-k on the WS ``token`` frame
        (``lens_readout``); disabling frees the device copies.  Applies to
        generations started after the call — the toggle waits on the
        session lock, so it never races an in-flight stream.
        """
        resolve_session_id(session, session_id)
        if not 1 <= body.top_k <= 50:
            raise HTTPException(400, "top_k must be in [1, 50]")
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
                    top_k=body.top_k,
                )
            except LensNotFittedError as e:
                raise HTTPException(404, str(e)) from e
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            except SaklasError as e:
                status, text = e.user_message()
                raise HTTPException(status, text) from e
        return {"enabled": True, "layers": resolved, "top_k": body.top_k}

    def _fit_status_payload() -> dict[str, object]:
        st = app.state.lens_fit
        return {
            "running": st["running"],
            "prompts_done": st["prompts_done"],
            "prompts_total": st["prompts_total"],
            "message": st["message"],
            "error": st["error"],
            "started_at": st["started_at"],
            "finished_at": st["finished_at"],
            "live_layers": st["live_layers"],
        }

    async def _lens_fit_job(
        body: LensFitRequest, source_layers: "list[int] | str",
    ) -> None:
        """The one background fit: stream corpus → fit → live-on.

        Error frames follow the SSE scrubbing discipline — typed saklas
        errors surface their ``user_message()``, anything else logs the
        traceback server-side and reports only the exception type.
        """
        from saklas.core.jlens import JacobianLensCancelled
        from saklas.io.lens import stream_default_lens_corpus

        st = app.state.lens_fit
        try:
            st["message"] = f"streaming {body.prompts} corpus documents…"
            docs, spec = await asyncio.to_thread(
                stream_default_lens_corpus, body.prompts,
            )

            def on_progress(msg: str) -> None:
                m = _FIT_PROGRESS_RE.search(msg)
                if m is not None:
                    st["prompts_done"] = int(m.group(1))
                    st["prompts_total"] = int(m.group(2))
                st["message"] = msg

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
                cancel_event=app.state.lens_fit_cancel,
            )
            # Live-on by default: hot the full-band readout the moment the
            # artifact lands, same policy as serve startup.  Session lock so
            # the enable never races a just-unblocked generation.
            async with acquire_session_lock(session) as acquired:
                if acquired:
                    st["live_layers"] = await asyncio.to_thread(
                        session.enable_live_lens, top_k=8,
                    )
            st["message"] = "done"
            st["error"] = None
        except JacobianLensCancelled:
            st["message"] = "cancelled"
            st["error"] = None
        except SaklasError as e:
            _, text = e.user_message()
            st["error"] = text
        except Exception as e:  # noqa: BLE001 — scrubbed, logged server-side
            log.exception("lens fit failed")
            st["error"] = f"lens fit failed ({type(e).__name__})"
        finally:
            st["running"] = False
            st["finished_at"] = time.time()

    @app.post("/saklas/v1/sessions/{session_id}/lens/fit", status_code=202)
    async def lens_fit_start(session_id: str, body: LensFitRequest | None = None):
        """Kick off a background Jacobian-lens fit (the "fit j-lens" button).

        Returns 202 with the initial status; poll ``GET .../lens/fit``.
        The fit holds the engine's model-exclusive lock, so generations
        attempted while it runs fail with the ordinary busy error.  A
        matching interrupted fit resumes from its last checkpoint.
        """
        resolve_session_id(session, session_id)
        body = body or LensFitRequest()
        source_layers = _parse_layers(body.layers) or "workspace"
        if source_layers == "sample":
            raise HTTPException(
                400,
                "layers='sample' is not fittable (debug readout only) — "
                "use 'workspace', 'all', or an explicit csv list",
            )
        st = app.state.lens_fit
        if st["running"]:
            raise HTTPException(409, "a lens fit is already running")
        st.update(
            running=True,
            prompts_done=0,
            prompts_total=body.prompts,
            message="starting…",
            error=None,
            started_at=time.time(),
            finished_at=None,
        )
        # Keep a handle so the task isn't GC'd mid-fit.
        app.state.lens_fit_cancel = threading.Event()
        app.state.lens_fit_task = asyncio.create_task(
            _lens_fit_job(body, source_layers),
        )
        return _fit_status_payload()

    @app.get("/saklas/v1/sessions/{session_id}/lens/fit")
    async def lens_fit_status(session_id: str):
        """Poll the background lens fit (progress / error / completion)."""
        resolve_session_id(session, session_id)
        return _fit_status_payload()

    @app.delete("/saklas/v1/sessions/{session_id}/lens/fit", status_code=202)
    async def lens_fit_cancel(session_id: str):
        """Request cooperative cancellation at the next prompt boundary."""
        resolve_session_id(session, session_id)
        st = app.state.lens_fit
        event = app.state.lens_fit_cancel
        if not st["running"] or event is None:
            raise HTTPException(409, "no lens fit is running")
        event.set()
        st["message"] = "cancelling…"
        return _fit_status_payload()
