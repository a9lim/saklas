"""Shared Server-Sent Events helpers for native long-running routes."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

ProgressCallback = Callable[[str], None]
ProgressJob = Callable[[ProgressCallback], Awaitable[Any]]
ErrorFormatter = Callable[[Exception], dict[str, Any] | None]
#: Ordered ``(exception type(s), status)`` pairs for the JSON branch, applied
#: by ``isinstance`` — first match wins, so put subclasses first.
JsonErrorMap = Sequence[tuple[type[BaseException] | tuple[type[BaseException], ...], int]]

#: Idle gap after which the SSE worker emits a comment line.  Matches the
#: traits stream: intermediary proxies drop a connection that has been silent
#: for their read timeout (nginx defaults to 60 s), and a single manifold
#: generate / fit progress step can run well past that on MPS.
HEARTBEAT_SECONDS = 15.0


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

    A ``: heartbeat`` comment line goes out every
    :data:`HEARTBEAT_SECONDS` of idle so an intermediary's read timeout can't
    silently drop a client mid-job.
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
                    try:
                        kind, payload = await asyncio.wait_for(
                            queue.get(), timeout=HEARTBEAT_SECONDS,
                        )
                    except (TimeoutError, asyncio.TimeoutError):
                        # Comment line: not an event, so a spec-compliant
                        # client ignores it while the connection stays warm.
                        yield ": heartbeat\n\n"
                        continue
                    data = {"message": payload} if kind == "progress" else payload
                    yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                    if kind in {"done", "error"}:
                        break
            finally:
                if not task.done():
                    # Keep the lock owned by the real worker lifetime.  Many
                    # callers run blocking model/artifact work via
                    # ``asyncio.to_thread``; cancelling this wrapper only
                    # cancels the await, not the thread.  Await completion so a
                    # disconnected SSE client cannot release ``session.lock``
                    # while the underlying job is still mutating session state
                    # or writing artifacts.
                    with suppress(BaseException):
                        await task

    return StreamingResponse(_sse(), media_type="text/event-stream")


async def sse_or_json(
    request: Request,
    session: Any,
    job: ProgressJob,
    *,
    error_message: str,
    log_message: str,
    error_formatter: ErrorFormatter | None = None,
    json_errors: JsonErrorMap = (),
    json_progress_key: str | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """Run ``job`` as an SSE stream or a plain JSON response.

    The three long-running native routes (``POST /extract``, ``POST
    /manifolds/generate``, ``POST .../fit``) negotiate the same way: SSE on
    ``Accept: text/event-stream``, otherwise a single JSON body produced under
    the bounded session lock.  This owns both branches so the two can't drift
    — the typed error handling is now an explicit argument on each side
    (``error_formatter`` for the SSE ``error`` frame, ``json_errors`` for the
    JSON status mapping) rather than whatever each route happened to catch.

    ``job`` is the same awaitable-taking-``on_progress`` callable
    :func:`progress_sse_response` drives, so any post-processing a route does
    around its worker (registering an extracted profile, re-serializing the
    folder) runs identically on both branches.  ``json_progress_key``, when
    set, attaches the collected progress lines to a dict payload under that
    key — the SSE branch already streams them as ``progress`` frames.
    """
    from saklas.server.app import acquire_session_lock

    accept = request.headers.get("accept", "application/json")
    if "text/event-stream" in accept:
        return progress_sse_response(
            session.lock,
            job,
            error_message=error_message,
            log_message=log_message,
            error_formatter=error_formatter,
            logger=logger,
        )

    progress: list[str] = []
    async with acquire_session_lock(session) as acquired:
        if not acquired:
            raise HTTPException(503, "session locked")
        try:
            payload = await job(progress.append)
        except HTTPException:
            # Already carries its own status — a job that mapped its own
            # failure wins over the generic table.
            raise
        except Exception as exc:
            for types, status in json_errors:
                if isinstance(exc, types):
                    raise HTTPException(status, str(exc)) from exc
            raise
    if json_progress_key is not None and isinstance(payload, dict):
        payload[json_progress_key] = progress
    return payload
