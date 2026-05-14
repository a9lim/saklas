"""Minimal session persistence for the v2.3 loom (engine-side).

v2.3 ships a single-file persistence shape — ``~/.saklas/sessions/
<session_id>/tree.json`` written via :func:`saklas.io.atomic.write_json_atomic`
and a persistent anonymous default session id at ``~/.saklas/sessions/
.default``.  The rest of the plan (docs/plans/loom.md "Persistence")
is deferred to v2.4:

  * per-node token blobs split into ``tokens/<node_id>.json`` siblings
    (today everything lives in the main ``tree.json`` without tokens —
    ``LoomTree.to_dict(include_tokens=False)`` keeps the file small);
  * named sessions via ``saklas tui --session <name>`` /
    ``saklas serve --session <name>`` (today only the auto-ulid
    anonymous default is honored);
  * ``saklas session ls / resume / rm`` CLI verbs;
  * 30-day auto-prune of anonymous, unstarred sessions at startup.

All paths honor ``SAKLAS_HOME`` via :func:`saklas.io.paths.saklas_home`.
"""
from __future__ import annotations

import secrets
import time
from pathlib import Path

from saklas.core.loom import LoomTree
from saklas.io.atomic import write_json_atomic
from saklas.io.paths import saklas_home


__all__ = [
    "default_session_id",
    "load_tree",
    "save_tree",
    "session_path",
    "sessions_root",
]


_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DEFAULT_POINTER_NAME = ".default"


def sessions_root() -> Path:
    """Return the root directory holding all sessions (no auto-create)."""
    return saklas_home() / "sessions"


def session_path(session_id: str) -> Path:
    """Return the per-session directory ``~/.saklas/sessions/<id>/``.

    No directory is created here — callers about to write a file under
    it should use :func:`write_json_atomic` (which mkdirs the parent)
    or call ``Path.mkdir(parents=True, exist_ok=True)`` explicitly.
    """
    return sessions_root() / session_id


def _tree_file(session_id: str) -> Path:
    return session_path(session_id) / "tree.json"


def _ulid() -> str:
    """Return a fresh 26-char Crockford-base32 ULID.

    Same shape as :func:`saklas.core.loom._ulid` — duplicated here to
    avoid the engine-layer dependency on a private symbol; both impls
    use the same alphabet so the produced ids are interchangeable.
    """
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = secrets.randbits(80)
    n = (ts_ms << 80) | rand
    out: list[str] = []
    for _ in range(26):
        out.append(_ULID_ALPHABET[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def default_session_id() -> str:
    """Return the persistent anonymous default session id.

    Reads ``~/.saklas/sessions/.default`` if it exists and is a single
    ULID line; writes a fresh ULID there otherwise.  The default
    session is re-used across runs so the tree survives a process
    restart even without an explicit ``--session`` name.
    """
    root = sessions_root()
    pointer = root / _DEFAULT_POINTER_NAME
    if pointer.is_file():
        try:
            text = pointer.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        # First non-empty line wins — guards against an accidental
        # editor saving a trailing newline or two.
        first_line = next(
            (line.strip() for line in text.splitlines() if line.strip()), "",
        )
        if first_line:
            return first_line
    # Pointer missing or unreadable — mint a fresh id and persist.
    sid = _ulid()
    root.mkdir(parents=True, exist_ok=True)
    pointer.write_text(sid + "\n", encoding="utf-8")
    return sid


def save_tree(session_id: str, tree: LoomTree) -> None:
    """Atomic-write the tree (sans token blobs) to the session's tree.json.

    Token blobs are deferred to v2.4 — see the module docstring.
    """
    path = _tree_file(session_id)
    payload = tree.to_dict(include_tokens=False)
    write_json_atomic(path, payload)


def load_tree(session_id: str) -> LoomTree | None:
    """Load the tree for ``session_id`` or return ``None`` if absent.

    A truncated or invalid ``tree.json`` (e.g. crash mid-write before
    the atomic replace) also returns ``None`` — the atomic-write helper
    leaves the prior good ``tree.json`` in place on a crash, so this
    only fires for genuinely missing or corrupt state.  Callers fall
    back to a fresh :class:`LoomTree` and continue.
    """
    path = _tree_file(session_id)
    if not path.is_file():
        return None
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    try:
        return LoomTree.from_dict(data)
    except Exception:
        return None
