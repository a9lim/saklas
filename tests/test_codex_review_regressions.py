"""Regression tests for extraction locking and capture cleanup."""
from __future__ import annotations

import threading
from typing import Any

import pytest

from saklas.core.session import (
    ConcurrentExtractionError, GenState, SaklasSession,
)


def _stub_session_with_lock() -> SaklasSession:
    """Build a __new__-bypass stub session with the minimum state the
    extract gate touches."""
    s: Any = SaklasSession.__new__(SaklasSession)
    s._gen_phase = GenState.IDLE
    s._gen_lock = threading.Lock()
    return s


# Extraction must acquire _gen_lock to be race-free.

def test_extract_acquires_gen_lock_against_concurrent_generation():
    """If ``_gen_lock`` is already held (generation in flight), extract
    must raise ``ConcurrentExtractionError`` rather than reading
    ``_gen_phase`` and racing the generation that's about to flip it."""
    s = _stub_session_with_lock()
    # Simulate "generation just acquired the lock" — phase still IDLE
    # because the flip happens after acquire on the generation side.
    assert s._gen_lock.acquire(blocking=False)
    try:
        with pytest.raises(ConcurrentExtractionError):
            s.extract("honest.deceptive")
    finally:
        s._gen_lock.release()


def test_extract_releases_lock_on_path_through_phase_gate():
    """When extract bails on the phase gate (RUNNING from a stub), the
    lock it acquired must be released so subsequent extract attempts
    can proceed once the phase clears."""
    from types import SimpleNamespace
    s: Any = SaklasSession.__new__(SaklasSession)
    s._gen_phase = GenState.RUNNING
    s._gen_lock = threading.Lock()
    s._extraction = SimpleNamespace(extract=lambda *a, **kw: ("x", None))

    with pytest.raises(ConcurrentExtractionError):
        s.extract("honest.deceptive")

    # Lock is back to free — caller can try again.
    assert s._gen_lock.acquire(blocking=False)
    s._gen_lock.release()


# Capture cleanup remains in the outer finally as a backstop.

def test_outer_finally_calls_end_capture_for_idempotency():
    """After the Codex-flagged fix, ``_generate_core``'s outer finally
    calls ``self._end_capture()`` defensively, in addition to the inner
    finally that handles the normal path.  This pins the source-level
    invariant: the outer finally block must reference ``_end_capture``."""
    import inspect
    from saklas.core.session import SaklasSession

    src = inspect.getsource(SaklasSession._generate_core)
    # Find the outermost ``finally`` (the one that releases _gen_lock).
    # The body must include the defensive ``_end_capture()`` call.
    assert src.count("self._end_capture()") >= 2, (
        "expected at least two _end_capture() calls — one in the inner "
        "try/finally for normal flow, one in the outer finally as a "
        "BaseException-safe backstop"
    )
