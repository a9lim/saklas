"""Shared generation plumbing for OpenAI and Ollama route families.

Both protocols serialize generation on ``app.state.gen_lock``.  This module
factors out the pieces that were duplicated between ``server.py`` and
``ollama_api.py``:

* ``acquire_lock_with_timeout`` — 5-minute-bounded lock acquire used by
  both streaming paths.
* ``run_blocking_generate`` — ``async with gen_lock`` + ``session.generate``
  for non-streaming callers.

After cluster 3 (core API) the per-request ``_gen_config_override`` dance
is gone — sampling overrides ride on ``SamplingConfig`` inside ``gen_kwargs``
and ``session.generate`` never mutates ``session.config``.

Wire-format emission (SSE chat deltas, SSE completions, NDJSON Ollama
chat, NDJSON Ollama generate) stays in each protocol module — those
diverge too much to share without ugliness.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from saklas.session import SaklasSession

LOCK_TIMEOUT_SECONDS = 300


@asynccontextmanager
async def acquire_lock_with_timeout(app) -> AsyncIterator[bool]:
    """Acquire ``app.state.gen_lock`` with a ``LOCK_TIMEOUT_SECONDS`` bound.

    Yields ``True`` if acquired (releases on exit), ``False`` on timeout.
    Callers branch on the result to emit their protocol-specific 503.
    """
    try:
        async with asyncio.timeout(LOCK_TIMEOUT_SECONDS):
            await app.state.gen_lock.acquire()
    except (TimeoutError, asyncio.TimeoutError):
        yield False
        return
    try:
        yield True
    finally:
        app.state.gen_lock.release()


async def run_blocking_generate(
    app,
    session: SaklasSession,
    *,
    input: Any,
    raw: bool,
    gen_kwargs: dict[str, Any],
) -> Any:
    """Acquire gen_lock and call ``session.generate``.

    ``gen_kwargs`` is in the new cluster-3 shape: ``sampling=SamplingConfig``,
    ``steering=Steering|dict|None``, ``thinking=``, ``stateless=``.
    Propagates ``ConcurrentGenerationError`` for callers to map to their
    wire format.
    """
    async with app.state.gen_lock:
        return session.generate(input, raw=raw, **gen_kwargs)
