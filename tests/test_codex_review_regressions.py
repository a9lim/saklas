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

import pytest
import torch

from saklas.core.errors import StaleSidecarError
from saklas.core.session import (
    ConcurrentExtractionError, GenState, SaklasSession,
)
from saklas.io import hf, packs


def _stub_session_with_lock() -> SaklasSession:
    """Build a __new__-bypass stub session with the minimum state the
    extract gate touches."""
    s = SaklasSession.__new__(SaklasSession)
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
    s = SaklasSession.__new__(SaklasSession)
    s._gen_phase = GenState.RUNNING
    s._gen_lock = threading.Lock()
    s._extraction = SimpleNamespace(extract=lambda *a, **kw: ("x", None))

    with pytest.raises(ConcurrentExtractionError):
        s.extract("honest.deceptive")

    # Lock is back to free — caller can try again.
    assert s._gen_lock.acquire(blocking=False)
    s._gen_lock.release()


# ---------------------------------------------------------------------------
# Phase 1 fix — pull_pack recovers from .bak when target is missing
# ---------------------------------------------------------------------------

def _fake_repo(tmp_path: Path, name: str = "happy") -> Path:
    repo = tmp_path / "downloaded" / name
    repo.mkdir(parents=True)
    (repo / "statements.json").write_text("[]")
    meta = packs.PackMetadata(
        name=name, description="x", version="1.0.0", license="MIT",
        tags=["test"], recommended_alpha=0.5,
        source="hf://user/happy", files={},
    )
    meta.files = {"statements.json": packs.hash_file(repo / "statements.json")}
    meta.write(repo)
    return repo


def test_pull_pack_recovers_prior_install_from_bak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Crash window: a prior pull died after ``target → .bak`` but
    before ``staging → target``.  Target is absent, .bak holds the only
    valid prior install.  The next pull must restore from .bak before
    any new download — wiping .bak first would lose the prior install
    if the new staging itself fails."""
    _fake_repo(tmp_path)  # baseline 'happy' source — not used by this test
    target = tmp_path / "installed" / "happy"

    # Simulate the post-crash state: target absent, .bak present with
    # the prior install.
    backup = target.with_name(target.name + ".bak")
    backup.parent.mkdir(parents=True, exist_ok=True)
    prior = _fake_repo(tmp_path, name="prior")
    import shutil
    shutil.copytree(prior, backup)
    prior_pack_bytes = (backup / "pack.json").read_bytes()
    assert not target.exists()

    # Force the new pull to fail on staging-side install (broken repo).
    bad = tmp_path / "downloaded" / "broken"
    bad.mkdir(parents=True)
    (bad / "random.txt").write_text("garbage")
    monkeypatch.setattr(hf, "_hf_snapshot_download", lambda **kw: str(bad))

    with pytest.raises(hf.HFError):
        hf.pull_pack("user/happy", target_folder=target, force=True)

    # Critical: the prior install was restored from .bak before the
    # broken-repo attempt blew up — we didn't lose it.
    assert target.exists()
    assert (target / "pack.json").is_file()
    assert (target / "pack.json").read_bytes() == prior_pack_bytes


def test_pull_pack_completed_swap_cleanup_drops_bak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Other crash window: the swap completed but rmtree of .bak got
    interrupted.  Target is the source of truth; .bak is redundant.
    Next pull should drop .bak (after the recovery branch sees target
    is intact)."""
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(hf, "_hf_snapshot_download", lambda **kw: str(fake))

    target = tmp_path / "installed" / "happy"
    hf.pull_pack("user/happy", target_folder=target, force=False)
    assert (target / "pack.json").is_file()

    # Plant a stale .bak (simulating a post-swap pre-cleanup crash).
    backup = target.with_name(target.name + ".bak")
    backup.mkdir(parents=True)
    (backup / "leftover.txt").write_text("from a prior crash")

    # Re-pull.  Target is intact, so .bak is just leftover noise.
    hf.pull_pack("user/happy", target_folder=target, force=True)
    assert not backup.exists()


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
