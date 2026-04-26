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
from pathlib import Path
from typing import Any


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
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write ``payload`` as JSON to ``path``.

    Output ends with a trailing newline (matches the prior ``json.dump`` +
    ``f.write("\\n")`` convention used across the io layer).
    """
    text = json.dumps(payload, indent=indent) + "\n"
    write_bytes_atomic(path, text.encode("utf-8"))
