"""Atomic file writes for ``~/.saklas/`` state.

A plain ``open(p, "w") + json.dump`` corrupts the file on SIGKILL or ENOSPC —
the partial bytes remain at ``p`` with no signal to the loader. ``write_*``
helpers here stage to ``<path>.tmp`` in the same directory, ``flush()`` +
``fsync()``, then ``os.replace()`` the tempfile into place. Same-dir staging
is required: ``os.replace()`` is only atomic on the same filesystem, and
``tempfile.NamedTemporaryFile`` defaults to ``$TMPDIR`` which often isn't.

A crash between the write and the replace leaves ``<path>.tmp`` orphaned;
the next loader sees the prior good ``<path>`` (or no file at all on a
first-time write). The orphan is harmless — it's outside the manifest's
``files`` map and doesn't affect integrity verification.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import suppress
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_ARTIFACT_LOCKS_GUARD = threading.Lock()
_ARTIFACT_LOCKS: dict[Path, threading.RLock] = {}
_ARTIFACT_LOCK_STATE = threading.local()


def _lock_handle(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_handle(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def artifact_lock(path: Path):
    """Cross-process, thread-reentrant lock for one logical artifact pair."""
    path = Path(path).expanduser().resolve(strict=False)
    lock_dir = path.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{path.name}.lock"
    with _ARTIFACT_LOCKS_GUARD:
        thread_lock = _ARTIFACT_LOCKS.setdefault(lock_path, threading.RLock())
    with thread_lock:
        depths = getattr(_ARTIFACT_LOCK_STATE, "depths", {})
        depth = int(depths.get(lock_path, 0))
        depths[lock_path] = depth + 1
        _ARTIFACT_LOCK_STATE.depths = depths
        if depth:
            try:
                yield
            finally:
                depths[lock_path] -= 1
            return
        acquired = False
        try:
            with open(lock_path, "a+b") as handle:
                _lock_handle(handle)
                acquired = True
                try:
                    yield
                finally:
                    if acquired:
                        _unlock_handle(handle)
        finally:
            depths.pop(lock_path, None)


class ReleasableArtifactLock:
    """Explicitly releasable form of :func:`artifact_lock`.

    Long-running fit pipelines sometimes need one short exclusive cache
    transaction followed by compute that must not retain the transaction lock.
    The owner still uses ``finally: release()``; ``release`` is idempotent so the
    inner cache stage may hand the lock back as soon as its pointer is durable.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._context: Any | None = None
        self._held = False

    def acquire(self) -> "ReleasableArtifactLock":
        if not self._held:
            context = artifact_lock(self._path)
            context.__enter__()
            self._context = context
            self._held = True
        return self

    def release(self) -> None:
        if self._held:
            self._held = False
            context = self._context
            self._context = None
            assert context is not None
            context.__exit__(None, None, None)


def _process_exists(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _windows_process_exists(pid: int) -> bool:
    """Non-destructive Windows process liveness probe.

    ``os.kill(pid, 0)`` is the POSIX idiom, but Windows implements non-CTRL
    signals with ``TerminateProcess`` semantics.  Query a synchronization handle
    and poll it instead; access-denied means the process exists but is protected.
    """

    import ctypes

    synchronize = 0x00100000
    wait_object_0 = 0x00000000
    error_access_denied = 5
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(synchronize, False, int(pid))
    if not handle:
        # ``ctypes.get_last_error`` is reliable only for DLLs loaded with
        # ``use_last_error=True``; windll.kernel32 is not guaranteed to be.
        return int(kernel32.GetLastError()) == error_access_denied
    try:
        # Only a signaled process handle is definite evidence of exit. Timeout
        # and WAIT_FAILED/unexpected results are conservatively live so a
        # transient API failure cannot authorize cache deletion.
        return int(kernel32.WaitForSingleObject(handle, 0)) != wait_object_0
    finally:
        kernel32.CloseHandle(handle)


@contextmanager
def artifact_process_lease(path: Path):
    """Publish a crash-detectable lease marker for a logical artifact.

    The caller creates/removes the marker while holding the artifact transaction
    lock.  Pruners use :func:`artifact_has_live_lease` under that same lock, so a
    mapped immutable payload cannot disappear between validation and use.  PID
    markers left by SIGKILL are removed lazily on the next lease check.
    """

    path = Path(path).expanduser().resolve(strict=False)
    lease_dir = path.parent / ".leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    marker = lease_dir / (
        f"{path.name}.lease.{os.getpid()}.{uuid.uuid4().hex}"
    )
    write_bytes_atomic(marker, b"lease\n")
    try:
        yield marker
    finally:
        marker.unlink(missing_ok=True)


def artifact_has_live_lease(path: Path) -> bool:
    """Return whether ``path`` has a live process lease; reap stale markers."""

    path = Path(path).expanduser().resolve(strict=False)
    live = False
    prefix = f"{path.name}.lease."
    for marker in (path.parent / ".leases").glob(f"{prefix}*"):
        suffix = marker.name[len(prefix):]
        pid_text = suffix.split(".", 1)[0]
        try:
            pid = int(pid_text)
        except ValueError:
            marker.unlink(missing_ok=True)
            continue
        if _process_exists(pid):
            live = True
        else:
            marker.unlink(missing_ok=True)
    return live


def _temp_path(path: Path) -> Path:
    """Return the same-directory staging path for ``path``.

    Uses ``<path><suffix>.tmp`` when ``path`` has a suffix, ``<path>.tmp``
    otherwise. Same parent directory in either case.
    """
    suffix = path.suffix
    if suffix:
        return path.with_suffix(suffix + ".tmp")
    return path.with_name(path.name + ".tmp")


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path``.

    Stages to ``<path>.tmp`` in the same directory, fsyncs the file,
    then ``os.replace()``s into place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            tmp.unlink()
        raise
    os.replace(tmp, path)


def fsync_directory(path: Path) -> None:
    """Best-effort durability barrier for prior directory-entry mutations."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(Path(path), flags)
    except OSError:
        # Windows and a few network filesystems do not permit directory opens.
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write ``payload`` as JSON to ``path``.

    Output ends with a trailing newline (matches the prior ``json.dump`` +
    ``f.write("\\n")`` convention used across the io layer).
    """
    text = json.dumps(payload, indent=indent) + "\n"
    write_bytes_atomic(path, text.encode("utf-8"))
