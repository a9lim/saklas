"""Native live trait-stream route group."""

from __future__ import annotations

import asyncio
import json
import uuid
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

        def _on_event(event: object) -> None:
            if isinstance(event, GenerationStarted):
                try:
                    loop.call_soon_threadsafe(
                        trait_queue.put_nowait,
                        (
                            "start",
                            getattr(event, "input", None),
                            getattr(event, "stateless", False),
                        ),
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
                        item = await asyncio.wait_for(
                            trait_queue.get(), timeout=15.0,
                        )
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
                        if result is not None:
                            readings = getattr(result, "readings", None)
                            if readings:
                                for name, reading in readings.items():
                                    per_generation = getattr(
                                        reading, "per_generation", None,
                                    )
                                    value = (
                                        per_generation[-1]
                                        if per_generation
                                        else getattr(reading, "mean", 0.0)
                                    )
                                    aggregate[name] = round(value, 6)
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
