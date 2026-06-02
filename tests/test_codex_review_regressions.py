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
from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.core.errors import StaleSidecarError
from saklas.core.session import (
    ConcurrentExtractionError, GenState, SaklasSession,
)
from saklas.io import packs


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
# Phase 2 fix — _try_autoload_vector enforces statements_sha256 too
# ---------------------------------------------------------------------------

def test_autoload_raises_on_stale_statements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """The session-level autoload path must apply the same
    ``statements_sha256`` contract as ``bootstrap_probes``.  Otherwise
    ``/steer happy.sad`` silently loads a stale tensor that
    ``bootstrap_probes`` would have rejected."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.delenv("SAKLAS_ALLOW_STALE", raising=False)

    # Build a concept folder with a stale tensor (recorded sha doesn't
    # match the live statements.json).
    cdir = tmp_path / "vectors" / "default" / "zz-stale"
    cdir.mkdir(parents=True)
    stmts_path = cdir / "statements.json"
    stmts_path.write_text("[]")
    extraction_time_sha = packs.hash_file(stmts_path)

    from saklas.core.vectors import save_profile
    save_profile(
        {0: torch.zeros(8, dtype=torch.float32)},
        str(cdir / "google__gemma-2-2b-it.safetensors"),
        {"method": "contrastive_pca", "statements_sha256": extraction_time_sha},
    )

    # User edits statements; refresh pack.json so integrity passes.
    stmts_path.write_text('[["my edit", "after extraction"]]')
    files: dict[str, str] = {
        "statements.json": packs.hash_file(stmts_path),
        "google__gemma-2-2b-it.safetensors": packs.hash_file(
            cdir / "google__gemma-2-2b-it.safetensors"),
        "google__gemma-2-2b-it.json": packs.hash_file(
            cdir / "google__gemma-2-2b-it.json"),
    }
    packs.PackMetadata(
        name="zz-stale", description="x", version="1.0.0", license="MIT",
        tags=["custom"], recommended_alpha=0.5, source="bundled",
        files=files,
    ).write(cdir)

    # Stub a session that exposes just the autoload entry point.
    s = _stub_session_with_lock()
    s._model_info = {"model_id": "google/gemma-2-2b-it"}
    from torch import device
    s._device = device("cpu")
    s._dtype = torch.float32
    s._profiles = {}
    # Tickle the selectors cache so _all_concepts sees our planted folder.
    from saklas.io.selectors import invalidate
    invalidate()

    with pytest.raises(StaleSidecarError) as excinfo:
        s._try_autoload_vector("zz-stale")

    msg = str(excinfo.value)
    assert "default/zz-stale" in msg
    assert "google/gemma-2-2b-it" in msg
    assert "saklas pack refresh" in msg
    assert "SAKLAS_ALLOW_STALE" in msg


def test_autoload_allow_stale_env_var_escape_hatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """``SAKLAS_ALLOW_STALE=1`` bypasses the autoload staleness check —
    matching ``bootstrap_probes`` semantics."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.setenv("SAKLAS_ALLOW_STALE", "1")

    cdir = tmp_path / "vectors" / "default" / "zz-stale"
    cdir.mkdir(parents=True)
    stmts_path = cdir / "statements.json"
    stmts_path.write_text("[]")
    extraction_time_sha = packs.hash_file(stmts_path)

    from saklas.core.vectors import save_profile
    save_profile(
        {0: torch.zeros(8, dtype=torch.float32)},
        str(cdir / "google__gemma-2-2b-it.safetensors"),
        {"method": "contrastive_pca", "statements_sha256": extraction_time_sha},
    )
    stmts_path.write_text('[["my edit", "after extraction"]]')
    files: dict[str, str] = {
        "statements.json": packs.hash_file(stmts_path),
        "google__gemma-2-2b-it.safetensors": packs.hash_file(
            cdir / "google__gemma-2-2b-it.safetensors"),
        "google__gemma-2-2b-it.json": packs.hash_file(
            cdir / "google__gemma-2-2b-it.json"),
    }
    packs.PackMetadata(
        name="zz-stale", description="x", version="1.0.0", license="MIT",
        tags=["custom"], recommended_alpha=0.5, source="bundled",
        files=files,
    ).write(cdir)

    s = _stub_session_with_lock()
    s._model_info = {"model_id": "google/gemma-2-2b-it"}
    from torch import device
    s._device = device("cpu")
    s._dtype = torch.float32
    s._profiles = {}

    from saklas.io.selectors import invalidate
    invalidate()

    # No raise — allow-stale lets it through and the tensor lands.
    s._try_autoload_vector("zz-stale")
    assert "zz-stale" in s._profiles


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
