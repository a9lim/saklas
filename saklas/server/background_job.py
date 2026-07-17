"""Shared scaffolding for the native long-running *background-job* routes.

Four native routes run a long operation as one detached ``asyncio`` task with a
polled status dict instead of an SSE stream: the J-lens ``fit`` + ``fetch`` and
the SAE local ``train`` + provider ``load``.  Each keeps the same shape — a
mutable ``app.state.<name>`` status dict (``running`` / ``message`` / ``error`` /
``started_at`` / ``finished_at`` + job-specific fields), an
``app.state.<name>_task`` handle, optionally an ``app.state.<name>_cancel``
``threading.Event`` for cooperative cancellation, a POST *start* (409 while the
job's group is busy), a GET *status* snapshot, and — for the cancellable pair — a
DELETE *cancel*.

``BackgroundJob`` centralizes that scaffolding the same way
``server.sse.progress_sse_response`` centralized the SSE frame loop: the route
module supplies only the job body (a progress-reporting coroutine) and its
per-job error classifier.

Design invariant: the status dict, the cancel event, and the task handle live on
``app.state`` (created here, mirrored on every access), *not* captured on the job
object — so shutdown hooks and tests that reassign
``app.state.<name>_cancel`` / ``_task`` (or mutate ``app.state.<name>``) are
honored.  The status wire is byte-compatible with the pre-dedup routes.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, HTTPException

from saklas.core.errors import SaklasError

JobBody = Callable[[], Awaitable[None]]
ErrorHandler = Callable[[BaseException], None]


class BackgroundJob:
    """One polled background job backed by ``app.state`` attributes.

    ``name`` is the ``app.state`` attribute stem: the status dict is
    ``app.state.<name>``, the task handle ``app.state.<name>_task``, and — when
    ``cancellable`` — the cooperative-cancel event ``app.state.<name>_cancel``.
    Jobs that guard one another (lens fetch XOR fit, sae load XOR train) are tied
    with :meth:`share_group`; ``refuse_if_busy`` then 409s while *any* group
    member runs.
    """

    def __init__(
        self,
        app: FastAPI,
        name: str,
        base_status: dict[str, Any],
        *,
        busy_message: str,
        cancellable: bool = False,
        not_running_message: str | None = None,
    ) -> None:
        self.app = app
        self.name = name
        self.busy_message = busy_message
        self.cancellable = cancellable
        self.not_running_message = not_running_message
        self.group: list[BackgroundJob] = [self]
        setattr(app.state, name, dict(base_status))
        setattr(app.state, f"{name}_task", None)
        if cancellable:
            setattr(app.state, f"{name}_cancel", None)

    # -- app.state-backed accessors (never cache; honor external reassignment) --

    @property
    def state(self) -> dict[str, Any]:
        return getattr(self.app.state, self.name)

    @property
    def running(self) -> bool:
        return bool(self.state["running"])

    @property
    def task(self) -> "asyncio.Task[None] | None":
        return getattr(self.app.state, f"{self.name}_task")

    @task.setter
    def task(self, value: "asyncio.Task[None] | None") -> None:
        setattr(self.app.state, f"{self.name}_task", value)

    @property
    def cancel_event(self) -> threading.Event | None:
        return getattr(self.app.state, f"{self.name}_cancel", None)

    @cancel_event.setter
    def cancel_event(self, value: threading.Event | None) -> None:
        setattr(self.app.state, f"{self.name}_cancel", value)

    # -- grouping / mutual exclusion --

    def share_group(self, *others: "BackgroundJob") -> None:
        """Tie this job and ``others`` into one mutual-exclusion group."""
        members = [self, *others]
        for job in members:
            job.group = members

    def refuse_if_busy(self) -> None:
        """Raise 409 (``busy_message``) if this job or a group sibling runs."""
        if any(job.running for job in self.group):
            raise HTTPException(409, self.busy_message)

    # -- status / lifecycle --

    def status(self) -> dict[str, Any]:
        """The GET-status wire payload: a shallow copy of the status dict."""
        return dict(self.state)

    def start(self, *, message: str, **fields: Any) -> dict[str, Any]:
        """Mark the job running and stamp the common + job-specific fields.

        Only the passed keys (plus the shared ``running`` / ``error`` /
        ``started_at`` / ``finished_at``) are touched, so a field the caller
        omits — e.g. ``live_layers`` on a fit restart — keeps its prior value.
        """
        self.state.update(
            running=True,
            error=None,
            started_at=time.time(),
            finished_at=None,
            message=message,
            **fields,
        )
        return self.status()

    def launch(self, body: JobBody, on_error: ErrorHandler) -> "asyncio.Task[None]":
        """Spawn the job coroutine, arming a fresh cancel event when cancellable.

        The task is wrapped so ``on_error`` classifies any ``Exception`` (the
        per-job scrubbing discipline) and the status is always finalized
        (``running`` cleared, ``finished_at`` stamped) in a ``finally`` —
        ``BaseException`` such as ``asyncio.CancelledError`` still propagates.
        """
        if self.cancellable:
            self.cancel_event = threading.Event()

        async def _runner() -> None:
            try:
                await body()
            except Exception as exc:  # noqa: BLE001 - routed to the per-job scrubber
                on_error(exc)
            finally:
                self.state["running"] = False
                self.state["finished_at"] = time.time()

        task = asyncio.create_task(_runner())
        self.task = task
        return task

    # -- cancel / shutdown --

    def request_cancel(self) -> dict[str, Any]:
        """DELETE handler: signal cooperative cancel, or 409 if idle."""
        event = self.cancel_event
        if not self.running or event is None:
            raise HTTPException(409, self.not_running_message or "no job is running")
        event.set()
        self.state["message"] = "cancelling…"
        return self.status()

    async def stop(self) -> None:
        """Shutdown-time stop: signal the event and await a cancellable job;
        asyncio-cancel and drain an uncancellable one."""
        task = self.task
        if self.cancellable:
            event = self.cancel_event
            if event is not None:
                event.set()
            if task is not None and not task.done():
                await task
        elif task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def make_progress_hook(
    state: dict[str, Any],
    regex: re.Pattern[str],
    *,
    done_field: str,
    total_field: str,
) -> Callable[[str], None]:
    """Build a progress callback for a worker thread's ``on_progress`` line.

    Parses ``regex`` — two integer groups, optionally comma-grouped — into
    ``done_field`` / ``total_field`` and mirrors the raw line into ``message``.
    Plain-dict mutation under the GIL is safe (single writer; readers only
    format).  Each job keeps its own vocabulary (prompts for the lens fit, tokens
    for the SAE train) via its own ``regex`` and field names.
    """

    def on_progress(message: str) -> None:
        match = regex.search(message)
        if match is not None:
            state[done_field] = int(match.group(1).replace(",", ""))
            state[total_field] = int(match.group(2).replace(",", ""))
        state["message"] = message

    return on_progress


def scrub_job_error(
    state: dict[str, Any],
    exc: BaseException,
    *,
    cancel_exc: type[BaseException],
    op_label: str,
    logger: logging.Logger,
    failure_message: str | None = None,
) -> None:
    """Shared error discipline for the fit/train jobs (the info-disclosure rule).

    Mirrors ``sse.progress_sse_response``'s scrubbing for the polled surface:

    - a cooperative-cancel (``cancel_exc``) settles to ``cancelled`` with no
      error;
    - a typed ``SaklasError`` surfaces its safe ``user_message()`` text;
    - anything else is logged server-side (full traceback) and scrubbed to the
      exception type only — never raw ``str(e)``, which routinely echoes
      filesystem paths and traceback fragments.

    ``failure_message`` optionally overwrites ``message`` on the non-cancel
    branches (the train job reports ``"training failed"``; the fit job leaves its
    last progress line).  The cancel check runs first because a cancel exception
    can itself subclass ``SaklasError``.
    """
    if isinstance(exc, cancel_exc):
        state["message"] = "cancelled"
        state["error"] = None
        return
    if failure_message is not None:
        state["message"] = failure_message
    if isinstance(exc, SaklasError):
        _code, text = exc.user_message()
        state["error"] = text
    else:
        logger.exception("%s failed", op_label)
        state["error"] = f"{op_label} failed ({type(exc).__name__})"
