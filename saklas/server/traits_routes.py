"""Native live trait-stream route group."""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from saklas.core.events import GenerationFinished, GenerationStarted
from saklas.server.saklas_api import _resolve_session_id


def register_traits_routes(app: FastAPI) -> None:
    """Mount SSE routes for live per-token trait readings."""
    session = app.state.session

    @app.get("/saklas/v1/sessions/{session_id}/traits/stream")
    async def traits_stream(session_id: str, request: Request):
        """SSE endpoint streaming per-token probe scores during generation."""
        _resolve_session_id(session, session_id)

        loop = asyncio.get_running_loop()
        trait_queue: asyncio.Queue[Any] = asyncio.Queue()

        def _enqueue(item: Any) -> None:
            with suppress(Exception):
                loop.call_soon_threadsafe(trait_queue.put_nowait, item)

        def _on_event(event: object) -> None:
            if isinstance(event, GenerationStarted):
                _enqueue((
                    "start",
                    getattr(event, "input", None),
                    getattr(event, "stateless", False),
                ))
            elif isinstance(event, GenerationFinished):
                _enqueue(("done", getattr(event, "result", None)))

        unsub = session.events.subscribe(_on_event)
        session.register_trait_queue(loop, trait_queue)

        async def event_generator():
            try:
                generation_id: str | None = None
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.wait_for(
                            trait_queue.get(), timeout=15.0,
                        )
                    except TimeoutError:
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
                        # ``scores`` is already the per-probe coordinate axis-0
                        # float (the session collapses each reading's
                        # ``coords[0]`` before enqueueing), so the wire-stable
                        # ``probes`` map keeps its ``{name: float}`` shape.
                        # The per-token queue item carries no richer coordinate
                        # payload, so the additive ``probe_readings`` channel is
                        # only emitted on the ``done`` frame, where the full
                        # result is reachable.
                        payload = {
                            "type": "token",
                            "idx": idx,
                            "text": text,
                            "thinking": thinking,
                            "probes": {
                                key: round(value, 6)
                                for key, value in scores.items()
                            },
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    elif tag == "done":
                        result = item[1]
                        aggregate: dict[str, float] = {}
                        # Additive rich channel: the full per-probe coordinate
                        # reading (every axis + per-generation samples) so a
                        # native client can read coordinates without the
                        # wire-stable ``aggregate`` (axis-0) shape changing.
                        probe_readings: dict[str, Any] = {}
                        manifold_readings: dict[str, Any] = {}
                        if result is not None:
                            readings = getattr(result, "readings", None)
                            if readings:
                                for name, reading in readings.items():
                                    per_generation = getattr(
                                        reading, "per_generation", None,
                                    )
                                    # ``per_generation`` samples and ``mean``
                                    # are per-axis coordinate tuples now; the
                                    # scalar ``aggregate`` reads axis 0.
                                    sample = (
                                        per_generation[-1]
                                        if per_generation
                                        else getattr(reading, "mean", None)
                                    )
                                    value = (
                                        sample[0]
                                        if sample
                                        else 0.0
                                    )
                                    aggregate[name] = round(float(value), 6)
                                    with suppress(Exception):
                                        probe_readings[name] = reading.to_dict()
                            mf_readings = getattr(
                                result, "manifold_readings", None,
                            )
                            if mf_readings:
                                for name, agg in mf_readings.items():
                                    with suppress(Exception):
                                        manifold_readings[name] = agg.to_dict()
                        payload = {
                            "type": "done",
                            "generation_id": generation_id,
                            "finish_reason": (
                                getattr(result, "finish_reason", "stop")
                                if result
                                else "stop"
                            ),
                            "aggregate": aggregate,
                        }
                        if probe_readings:
                            payload["probe_readings"] = probe_readings
                        if manifold_readings:
                            payload["manifold_readings"] = manifold_readings
                        yield f"data: {json.dumps(payload)}\n\n"
                        generation_id = None
            finally:
                session.unregister_trait_queue(loop, trait_queue)
                unsub()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
