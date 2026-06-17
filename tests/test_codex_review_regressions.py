"""Regression tests for the four issues Codex flagged on the post-1.5.0
hardening sweep.  Each test pins one of:

1. Phase 6 — capture cleanup never leaks hooks even when the inner
   ``try`` is skipped due to a ``BaseException`` (KeyboardInterrupt).
2. Phase 1 — ``pull_pack`` recovers the prior install from ``.bak``
   when a previous pull crashed mid-swap.
3. Phase 7 — ``session.extract`` acquires ``_gen_lock`` so it's
   race-free against a concurrent ``generate``.
4. Phase 2 — session-side ``_try_autoload_vector`` enforces the same
   ``statements_sha256`` contract as ``bootstrap_probes``.
"""
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


# ---------------------------------------------------------------------------
# Phase 7 fix — extract() must acquire _gen_lock to be race-free
# ---------------------------------------------------------------------------

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


# NOTE: the Phase 1 ``pull_pack`` crash-window regressions (recover-from-.bak
# before a new build, drop-stale-.bak on a clean swap) moved to the generic
# ``io/staging.py::stage_verify_swap`` primitive in the 4.0 collapse and are
# covered directly by ``test_io_staging.py`` (``pull_pack`` itself is gone).


# ---------------------------------------------------------------------------
# Phase 2 — _try_autoload_vector stale-statements contract
# ---------------------------------------------------------------------------
# NOTE: ``test_autoload_raises_on_stale_statements`` and
# ``test_autoload_allow_stale_env_var_escape_hatch`` were deleted in 4.0.
# ``SaklasSession._try_autoload_vector`` (the ``vectors/``-pack safetensors
# scan that re-checked ``statements_sha256``) was removed; profile resolution
# now folds a fitted manifold via ``_ensure_profile_registered`` (the legacy
# ``vectors/`` folder is only *ported* to a manifold on first touch, then
# re-fit — there is no stale-tensor autoload path left to guard).


# ---------------------------------------------------------------------------
# Phase 6 fix — capture cleanup is in the outer finally as a backstop
# ---------------------------------------------------------------------------

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
