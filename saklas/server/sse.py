"""Shared Server-Sent Events helpers for native long-running routes."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from fastapi.responses import StreamingResponse

ProgressCallback = Callable[[str], None]
ProgressJob = Callable[[ProgressCallback], Awaitable[Any]]
ErrorFormatter = Callable[[Exception], dict[str, Any] | None]


def progress_sse_response(
    lock: Any,
    job: ProgressJob,
    *,
    error_message: str,
    log_message: str,
    error_formatter: ErrorFormatter | None = None,
    logger: logging.Logger | None = None,
) -> StreamingResponse:
    """Run a progress-reporting async job and stream progress/done/error frames.

    Acquires ``lock`` *without* a timeout — unlike the bounded
    ``acquire_session_lock`` used by request handlers — because the jobs this
    drives (manifold ``fit`` / ``generate``, ``extract``) are inherently
    unbounded.  Use it only for those long-running operations; a bounded
    variant would need a timeout parameter.
    """
    log = logger or logging.getLogger("saklas.api")

    async def _sse():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def _on_progress(msg: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ("progress", msg))

        async with lock:
            async def _run() -> None:
                try:
                    payload = await job(_on_progress)
                    await asyncio.sleep(0)
                    queue.put_nowait(("done", payload))
                except Exception as e:
                    err = None
                    if error_formatter is not None:
                        try:
                            err = error_formatter(e)
                        except Exception:
                            log.exception("%s (error formatter crashed)", log_message)
                    if err is None:
                        log.exception(log_message)
                        err = {
                            "message": error_message,
                            "code": type(e).__name__,
                        }
                    queue.put_nowait(("error", err))

            task = asyncio.create_task(_run())
            try:
                while True:
                    kind, payload = await queue.get()
                    data = {"message": payload} if kind == "progress" else payload
                    yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                    if kind in {"done", "error"}:
                        break
            finally:
                if not task.done():
                    task.cancel()
                    with suppress(BaseException):
                        await task

    return StreamingResponse(_sse(), media_type="text/event-stream")
