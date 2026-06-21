"""Frontend ↔ engine boundary guard.

The TUI and the HTTP server are *frontends* over :class:`SaklasSession`; they
must talk to the engine through its public API, never reach past it into a
``SaklasSession`` private (``session._monitor``, ``session._profiles``, …).
Reaching in couples a frontend to the engine's internal layout, so a private
rename silently breaks it — exactly the migration this test guards against
regressing.  Every former reach-in has a public accessor now (``monitor``,
``manifolds``, ``profiles``, ``model_metadata``, ``generation_state``,
``gen_lock``, ``joint_logprob_cache``, ``loom_conflict_check``,
``ensure_manifold_loaded``, ``ensure_profile_registered``); a fresh reach-in
should use those instead of being added to the allowlist below.

The grep is **session-scoped**, not "any ``self._x``": a widget or route
controller is free to keep its own privates.  Only attribute access whose
*receiver* is a session-object expression counts — the session is bound to a
bare ``session`` / ``sess`` local (the server routes), to ``self._session``
(the TUI ``SaklasApp``), or reached through ``self.session`` / ``.session``.

Comments and string literals are stripped via :mod:`tokenize` before the
match, so a historical note (``# v2.2 had self._messages = session._history``)
or a docstring example doesn't trip the guard — only live code does.

CPU-only: pure source-grep, no model load.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

# Genuinely-unavoidable session-private reach-ins, each as
# "<relpath-from-saklas>:<receiver>._<attr>" with a comment justifying it.
# Seeded empty: the migration left no straggler.  Do NOT add a site here to
# silence a fixable reach-in — only one with no public accessor belongs.
ALLOWLIST: list[str] = [
]

_FRONTEND_DIRS = ("tui", "server")

# Receiver expressions that denote the SaklasSession object, each matched
# right before ``._<attr>``.  Two left-boundary regimes, so the dotted and the
# bare forms can't bleed into each other:
#   * a *dotted* receiver (``self._session`` / ``self.session`` / any
#     ``…​.session``) is anchored on the leading ``.``/``self`` — so
#     ``app.state.session._gen_lock`` matches via the ``.session`` arm;
#   * a *bare* ``session`` / ``sess`` local must NOT be preceded by ``.`` or a
#     word char, so ``my_session`` / ``subsession`` / ``x.session`` don't slip
#     through the bare arm (``x.session`` is still caught by the dotted arm).
_SESSION_PRIVATE_RE = re.compile(
    r"(?:"
    r"(?:self|[\w\]\)])\.session"  # self.session / app.state.session / …
    r"|self\._session"             # the TUI SaklasApp binding
    r"|(?<![\w.])(?:session|sess)"  # a bare session/sess local
    r")"
    r"\._[A-Za-z]\w*"
)


def _saklas_root() -> Path:
    return Path(__file__).resolve().parent.parent / "saklas"


def _strip_comments_and_strings(source: str) -> str:
    """Return ``source`` with comment and string-literal tokens blanked.

    Keeps line/column structure so a flagged hit still reports a usable line
    number.  Blanking (rather than deleting) string tokens means a reach-in
    that only ever appears inside a docstring or an f-string literal is *not*
    a real reach-in and is correctly ignored.
    """
    out_lines = source.splitlines()
    # Rebuild as a mutable char grid so we can null out token spans in place.
    grid = [list(line) for line in out_lines]
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            for row in range(srow, erow + 1):
                line = grid[row - 1]
                lo = scol if row == srow else 0
                hi = ecol if row == erow else len(line)
                for col in range(lo, min(hi, len(line))):
                    line[col] = " "
    except (tokenize.TokenError, IndentationError):
        # A file we can't tokenize shouldn't mask a reach-in: fall back to the
        # raw source (the regex still runs; at worst a commented hit surfaces,
        # which is a louder failure than a silent miss).
        return source
    return "\n".join("".join(line) for line in grid)


def _iter_frontend_files() -> list[Path]:
    root = _saklas_root()
    files: list[Path] = []
    for sub in _FRONTEND_DIRS:
        files.extend(sorted((root / sub).rglob("*.py")))
    assert files, "no frontend python files found — path drift?"
    return files


def _find_reach_ins() -> dict[str, list[str]]:
    """Map ``<relpath>:<receiver>._<attr>`` -> list of human-readable hits."""
    root = _saklas_root()
    hits: dict[str, list[str]] = {}
    for path in _iter_frontend_files():
        rel = path.relative_to(root).as_posix()
        code = _strip_comments_and_strings(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(code.splitlines(), start=1):
            for match in _SESSION_PRIVATE_RE.finditer(line):
                token = match.group(0)
                key = f"{rel}:{token}"
                hits.setdefault(key, []).append(f"{rel}:{lineno}: {token}")
    return hits


def test_no_frontend_session_private_reach_ins() -> None:
    """No ``saklas/{tui,server}`` code reaches a SaklasSession private."""
    hits = _find_reach_ins()
    allow = set(ALLOWLIST)
    offenders = {key: lines for key, lines in hits.items() if key not in allow}
    if offenders:
        report = "\n".join(
            line for lines in offenders.values() for line in lines
        )
        raise AssertionError(
            "frontend code reaches past the SaklasSession public API.\n"
            "Use the public accessor (monitor / manifolds / profiles / "
            "model_metadata / generation_state / gen_lock / "
            "joint_logprob_cache / loom_conflict_check / "
            "ensure_manifold_loaded / ensure_profile_registered, …) at the "
            "call site, or — only if genuinely unavoidable — add the site to "
            "ALLOWLIST with a justification.\n\n" + report
        )


def test_allowlist_entries_are_still_present() -> None:
    """An ALLOWLIST entry whose reach-in is gone is stale — drop it.

    Keeps the allowlist honest: a fixed reach-in shouldn't leave a dead
    exemption behind that would silently re-admit a future regression at the
    same key.
    """
    hits = _find_reach_ins()
    stale = [key for key in ALLOWLIST if key not in hits]
    assert not stale, (
        "ALLOWLIST has entries with no matching reach-in (remove them): "
        + ", ".join(stale)
    )
