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


# Capture cleanup is a transaction-teardown backstop, not the body's job.

def _transaction_stub_session() -> Any:
    """A stub carrying only what ``_generation_transaction`` touches."""
    from types import SimpleNamespace

    from saklas.core.capture import CaptureState

    s: Any = SaklasSession.__new__(SaklasSession)
    s._gen_phase = GenState.IDLE
    s._gen_lock = threading.Lock()
    s._end_capture_calls = 0

    def _end_capture() -> None:
        s._end_capture_calls += 1

    s._end_capture = _end_capture
    s._monitor = SimpleNamespace(set_subspace_coords=lambda _flag: None)
    s._close_instrument_runs = lambda: None
    s._capture_state = CaptureState()
    s._steering = SimpleNamespace(has_compiled_offsets=lambda: False)
    return s


def test_generation_transaction_detaches_capture_when_the_body_never_does():
    """A BaseException before the body's own teardown (a preamble failure,
    a KeyboardInterrupt) must still leave no capture hooks attached — the
    transaction's teardown is the backstop, and it also frees the phase and
    the lock."""
    s = _transaction_stub_session()

    with pytest.raises(KeyboardInterrupt):
        with s._generation_transaction():
            raise KeyboardInterrupt

    assert s._end_capture_calls == 1
    assert s._gen_phase is GenState.IDLE
    assert s._gen_lock.acquire(blocking=False)
    s._gen_lock.release()


def test_generation_transaction_pops_a_scope_the_body_left_open():
    """The body assigns ``txn.steering_cm`` before entering the scope; if it
    throws before popping it, the transaction pops it — swallowing a teardown
    failure so it can't mask the error already propagating."""
    exits: list[bool] = []
    s = _transaction_stub_session()
    s._exit_internal_steering = (
        lambda _cm, *, swallow: exits.append(swallow)
    )

    with pytest.raises(RuntimeError):
        with s._generation_transaction() as txn:
            txn.steering_cm = object()
            raise RuntimeError("preamble blew up")

    assert exits == [True]


def test_generation_transaction_leaves_a_popped_scope_alone():
    """The ordinary path pops its own scope before finalizing and clears the
    slot; the transaction must not pop it a second time."""
    exits: list[bool] = []
    s = _transaction_stub_session()
    s._exit_internal_steering = (
        lambda _cm, *, swallow: exits.append(swallow)
    )

    with s._generation_transaction() as txn:
        txn.steering_cm = object()
        txn.steering_cm = None

    assert exits == []
    assert s._gen_phase is GenState.IDLE


def test_generation_transaction_rejects_a_reentrant_generation():
    from saklas.core.session import ConcurrentGenerationError

    s = _transaction_stub_session()
    with s._generation_transaction():
        with pytest.raises(ConcurrentGenerationError):
            with s._generation_transaction():
                pass  # pragma: no cover - the guard raises on entry
