"""Atomic-write helper + crash-recovery semantics for ~/.saklas state."""
from __future__ import annotations

import json
import ctypes
from pathlib import Path

import pytest

from saklas.io import atomic
from saklas.io.atomic import _temp_path, write_bytes_atomic, write_json_atomic


def test_write_json_atomic_creates_file(tmp_path: Path):
    path = tmp_path / "x.json"
    write_json_atomic(path, {"a": 1, "b": [2, 3]})
    assert path.is_file()
    assert json.loads(path.read_text()) == {"a": 1, "b": [2, 3]}
    # Trailing newline matches the prior json.dump + f.write("\n") convention.
    assert path.read_text().endswith("\n")


def test_write_json_atomic_overwrites(tmp_path: Path):
    path = tmp_path / "x.json"
    write_json_atomic(path, {"v": 1})
    write_json_atomic(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 2}


def test_write_json_atomic_no_orphan_tmp(tmp_path: Path):
    path = tmp_path / "x.json"
    write_json_atomic(path, {"v": 1})
    # Successful write leaves no <path>.tmp behind.
    assert not _temp_path(path).exists()


def test_write_json_atomic_creates_parent(tmp_path: Path):
    path = tmp_path / "nested" / "deep" / "x.json"
    write_json_atomic(path, {"v": 1})
    assert path.is_file()


def test_write_bytes_atomic_basic(tmp_path: Path):
    path = tmp_path / "blob.bin"
    write_bytes_atomic(path, b"\x00\x01\x02")
    assert path.read_bytes() == b"\x00\x01\x02"
    assert not _temp_path(path).exists()


def test_temp_path_with_suffix(tmp_path: Path):
    p = tmp_path / "x.json"
    assert _temp_path(p) == tmp_path / "x.json.tmp"


def test_temp_path_no_suffix(tmp_path: Path):
    p = tmp_path / "Makefile"
    assert _temp_path(p) == tmp_path / "Makefile.tmp"


def test_temp_path_same_directory(tmp_path: Path):
    """Atomicity requires the tempfile sit on the same volume — same dir
    is a sufficient proxy."""
    p = tmp_path / "subdir" / "x.json"
    assert _temp_path(p).parent == p.parent


# NOTE: the ``vectors/``-pack crash-recovery + ``format_version`` + materialize
# tests were deleted in 4.0 — ``saklas.io.packs.ConceptFolder`` /
# ``PackMetadata`` / ``materialize_bundled`` were removed.  The orphan-``.tmp``
# recovery and future-/stale-``format_version`` gate now live on the manifold
# artifact (``ManifoldFolder.load`` / ``MANIFOLD_FORMAT_VERSION``,
# ``materialize_bundled_manifolds``), covered by the manifold-format tests.
# The atomic-write primitive tests above are frontend-agnostic and stay.


def test_atomic_overwrite_preserves_prior_on_simulated_crash(tmp_path: Path):
    """If the .tmp file is written but the ``os.replace`` step never lands
    (the canonical crash window), the original file is byte-identical to
    what it was before the write started."""
    path = tmp_path / "x.json"
    write_json_atomic(path, {"version": 1})
    original_bytes = path.read_bytes()

    # Simulate a partial write: stage a new tempfile but don't replace.
    tmp = _temp_path(path)
    tmp.write_text('{"version": 2, "trunc')

    # The "kill" window: tmp exists, original is untouched.
    assert path.read_bytes() == original_bytes
    assert tmp.exists()

    # And the original still loads cleanly.
    assert json.loads(path.read_text()) == {"version": 1}


def test_releasable_artifact_lock_can_reacquire_after_early_release(
    tmp_path: Path,
) -> None:
    transaction = atomic.ReleasableArtifactLock(tmp_path / "capture")

    assert transaction.acquire() is transaction
    transaction.release()
    assert transaction.acquire() is transaction
    transaction.release()
    transaction.release()  # idempotent final cleanup


class _WinCall:
    def __init__(self, result: int) -> None:
        self.result = result
        self.restype = None

    def __call__(self, *_args: object) -> int:
        return self.result


@pytest.mark.parametrize(
    "wait_result, expected", [(0x102, True), (0, False), (0xFFFFFFFF, True)],
)
def test_windows_process_probe_never_uses_os_kill(
    monkeypatch: pytest.MonkeyPatch, wait_result: int, expected: bool,
) -> None:
    class _Kernel:
        OpenProcess = _WinCall(123)
        WaitForSingleObject = _WinCall(wait_result)
        CloseHandle = _WinCall(1)

    class _Windll:
        kernel32 = _Kernel()

    monkeypatch.setattr(ctypes, "windll", _Windll(), raising=False)
    monkeypatch.setattr(atomic.os, "name", "nt")
    monkeypatch.setattr(
        atomic.os, "kill",
        lambda *_: (_ for _ in ()).throw(AssertionError("os.kill is unsafe here")),
    )
    assert atomic._process_exists(987654) is expected


def test_windows_access_denied_process_is_treated_as_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Kernel:
        OpenProcess = _WinCall(0)
        WaitForSingleObject = _WinCall(0)
        CloseHandle = _WinCall(1)
        GetLastError = _WinCall(5)

    class _Windll:
        kernel32 = _Kernel()

    monkeypatch.setattr(ctypes, "windll", _Windll(), raising=False)
    assert atomic._windows_process_exists(42)


# NOTE: ``test_pack_metadata_future_format_version_*`` and the two
# ``test_materialize_*_statements`` tests were deleted in 4.0 along with the
# ``vectors/``-pack machinery they exercised (``ConceptFolder`` /
# ``PackMetadata.load`` future-version gate; ``materialize_bundled``'s
# user-edited-``statements.json`` preservation).  The manifold analogues live
# in the manifold-format / materialize tests.
