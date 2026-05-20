"""Cross-branch diff for loom (v2.3 phase 5).

Three primitives used by the cross-branch comparison surfaces:

- :func:`text_diff` — word-level Myers diff between two assistant texts,
  built on stdlib :class:`difflib.SequenceMatcher` over whitespace-split
  tokens.  Returns a list of aligned :class:`DiffSpan`s the surfaces
  render side-by-side or unified.
- :func:`readings_diff` — per-probe ``Δ = b - a`` table sorted by
  ``abs(delta)`` descending.  Missing-on-one-side probes default to
  ``0.0`` for the absent reading; ``a_value``/``b_value`` carry the
  original values so the surface can distinguish "moved from 0.3 to 0.1"
  from "appeared at 0.1 (no prior reading)".
- :func:`per_token_diff` — byte-offset-based alignment between two token
  sequences.  Stops at the shortest common prefix where bytes diverge;
  per-token reading deltas land at aligned positions only.  The
  ``reference_tokenize`` hook is a placeholder for re-tokenization
  against a shared reference in a later phase — phase 5 leaves it
  unused.

Plus :func:`steering_delta` — render a compact label like
``"+0.2 calm"`` for the parent → child edge by walking both
expressions through the shared grammar.  Surfaces consume this when
labelling sibling-edge connectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Literal, Mapping, Sequence


# ---------------------------------------------------------------------------
# Text diff
# ---------------------------------------------------------------------------


DiffState = Literal["equal", "insert", "delete"]


@dataclass(frozen=True)
class DiffSpan:
    """One aligned span in a word-level diff.

    ``state == "equal"`` → present in both; ``"insert"`` → present in
    ``b`` but not ``a`` (new); ``"delete"`` → present in ``a`` but not
    ``b`` (removed).  ``text`` is the joined surface text for the span
    (whitespace re-inserted as single spaces between tokens — surfaces
    re-render whitespace per their own conventions).
    """

    state: DiffState
    text: str


def _tokenize_for_diff(s: str) -> list[str]:
    """Split on whitespace, preserving tokens for the SequenceMatcher.

    Keeps trailing punctuation glued to words — Myers diff at the token
    level is what users mentally model as "git diff --word-diff", and
    aggressive subtoken splitting produces churn in alignment under
    minor edits.
    """
    # ``str.split()`` collapses runs of whitespace; that's what we want
    # — diff alignment doesn't depend on which space character was used.
    return s.split()


def text_diff(a: str, b: str) -> list[DiffSpan]:
    """Word-level Myers diff between ``a`` and ``b``.

    Uses :class:`difflib.SequenceMatcher` on whitespace-split tokens.
    Returns a flat list of :class:`DiffSpan` in order, suitable for
    rendering unified-diff style or side-by-side.  Empty inputs produce
    a single-span result (``equal`` for both empty, otherwise the
    appropriate ``insert`` / ``delete`` of the non-empty side).
    """
    a_toks = _tokenize_for_diff(a)
    b_toks = _tokenize_for_diff(b)

    sm = SequenceMatcher(a=a_toks, b=b_toks, autojunk=False)
    spans: list[DiffSpan] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            text = " ".join(a_toks[i1:i2])
            if text:
                spans.append(DiffSpan(state="equal", text=text))
        elif tag == "delete":
            text = " ".join(a_toks[i1:i2])
            if text:
                spans.append(DiffSpan(state="delete", text=text))
        elif tag == "insert":
            text = " ".join(b_toks[j1:j2])
            if text:
                spans.append(DiffSpan(state="insert", text=text))
        elif tag == "replace":
            # ``replace`` splits into a delete + insert pair so consumers
            # can render them side by side or stack them.  Order matters
            # only for unified rendering — delete then insert matches
            # ``git diff --word-diff`` output.
            del_text = " ".join(a_toks[i1:i2])
            ins_text = " ".join(b_toks[j1:j2])
            if del_text:
                spans.append(DiffSpan(state="delete", text=del_text))
            if ins_text:
                spans.append(DiffSpan(state="insert", text=ins_text))
    return spans


# ---------------------------------------------------------------------------
# Readings diff
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadingDelta:
    """One probe's value change between two assistant nodes.

    ``delta`` is signed ``b - a``; the list returned by
    :func:`readings_diff` is sorted by ``abs(delta)`` descending so
    consumers grab the top-N most-changed probes off the front.
    """

    name: str
    delta: float
    a_value: float
    b_value: float


def readings_diff(
    a: Mapping[str, float],
    b: Mapping[str, float],
) -> list[ReadingDelta]:
    """Per-probe delta table between two ``aggregate_readings`` maps.

    Returns a list of :class:`ReadingDelta` sorted by ``abs(delta)``
    descending.  Probes present in one map but not the other are
    included with the absent side defaulting to ``0.0`` (so consumers
    see "appeared at +0.4" / "disappeared from -0.2" as a single
    sorted entry).  The original ``a_value``/``b_value`` ride along so
    surfaces can render the absolute readings alongside the delta.
    """
    names = set(a) | set(b)
    out: list[ReadingDelta] = []
    for name in names:
        av = float(a.get(name, 0.0))
        bv = float(b.get(name, 0.0))
        out.append(ReadingDelta(
            name=name,
            delta=bv - av,
            a_value=av,
            b_value=bv,
        ))
    out.sort(key=lambda r: abs(r.delta), reverse=True)
    return out


# ---------------------------------------------------------------------------
# Per-token diff
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenDeltaSpan:
    """One byte-aligned per-token region between two token sequences.

    ``a_index`` / ``b_index`` are the per-side token indices that map
    onto the region.  ``reading_deltas`` carries the per-probe delta at
    that aligned position when both sides have score tables for the
    aligned indices; empty when the position is in the divergent tail
    (one side keeps going past the common prefix's byte boundary).
    """

    a_index: int
    b_index: int
    a_text: str
    b_text: str
    aligned: bool
    reading_deltas: tuple[ReadingDelta, ...] = ()


def _token_byte_offsets(tokens: Sequence[str]) -> list[int]:
    """Return cumulative byte offsets after each token's bytes."""
    offsets = [0]
    pos = 0
    for tok in tokens:
        pos += len(tok.encode("utf-8"))
        offsets.append(pos)
    return offsets


def per_token_diff(
    a_tokens: Sequence[str],
    b_tokens: Sequence[str],
    *,
    a_scores: Mapping[str, Sequence[float]] | None = None,
    b_scores: Mapping[str, Sequence[float]] | None = None,
    reference_tokenize: Callable[[str], list[str]] | None = None,
) -> list[TokenDeltaSpan]:
    """Byte-offset alignment between two token sequences.

    Walks both sequences position-by-position; at each position
    compares the cumulative byte offset on either side.  Tokens whose
    byte-offsets line up are emitted as ``aligned=True`` spans with
    per-probe deltas pulled from the optional ``a_scores`` / ``b_scores``
    maps.  Position drift (one side adds bytes the other doesn't)
    surfaces as ``aligned=False`` spans and reading deltas drop.

    The ``reference_tokenize`` hook is a placeholder for the
    documented "re-tokenize against a common reference" mode — phase 5
    leaves it unused; identical-tokenizer cases (same model on both
    siblings) work without it.
    """
    del reference_tokenize  # placeholder for later phases — silence lint
    a_off = _token_byte_offsets(a_tokens)
    b_off = _token_byte_offsets(b_tokens)

    out: list[TokenDeltaSpan] = []
    ai = bi = 0
    while ai < len(a_tokens) and bi < len(b_tokens):
        a_end = a_off[ai + 1]
        b_end = b_off[bi + 1]
        aligned = a_end == b_end and a_tokens[ai] == b_tokens[bi]
        deltas: tuple[ReadingDelta, ...] = ()
        if aligned and a_scores is not None and b_scores is not None:
            common = set(a_scores) & set(b_scores)
            d: list[ReadingDelta] = []
            for name in common:
                seq_a = a_scores[name]
                seq_b = b_scores[name]
                if ai < len(seq_a) and bi < len(seq_b):
                    av = float(seq_a[ai])
                    bv = float(seq_b[bi])
                    d.append(ReadingDelta(
                        name=name, delta=bv - av,
                        a_value=av, b_value=bv,
                    ))
            d.sort(key=lambda r: abs(r.delta), reverse=True)
            deltas = tuple(d)
        out.append(TokenDeltaSpan(
            a_index=ai, b_index=bi,
            a_text=a_tokens[ai], b_text=b_tokens[bi],
            aligned=aligned, reading_deltas=deltas,
        ))
        # If aligned, advance both; otherwise advance whichever side
        # has the smaller byte offset so we keep tracking the common
        # prefix as long as possible.
        if aligned:
            ai += 1
            bi += 1
        elif a_end <= b_end:
            ai += 1
        else:
            bi += 1

    # Tail: whichever side still has tokens emits divergent (unaligned)
    # spans so callers can render the trailing "B has 4 more tokens".
    while ai < len(a_tokens):
        out.append(TokenDeltaSpan(
            a_index=ai, b_index=-1,
            a_text=a_tokens[ai], b_text="",
            aligned=False,
        ))
        ai += 1
    while bi < len(b_tokens):
        out.append(TokenDeltaSpan(
            a_index=-1, b_index=bi,
            a_text="", b_text=b_tokens[bi],
            aligned=False,
        ))
        bi += 1
    return out


# ---------------------------------------------------------------------------
# Node diff — top-level wrapper consumed by session.diff_nodes()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeDiff:
    """Cross-sibling diff bundle.

    Holds the text diff between two assistant nodes plus the readings
    delta table.  ``a_id`` / ``b_id`` round-trip so consumers can show
    which sibling was anchor vs. comparison.  ``parent_id`` is the
    shared user-parent (when found); ``None`` when the two nodes don't
    share a parent (cross-tree diff — caller can still render text +
    readings, no parent context).
    """

    a_id: str
    b_id: str
    parent_id: str | None
    text: list[DiffSpan]
    readings: list[ReadingDelta]


# ---------------------------------------------------------------------------
# Steering-delta edge label (Phase 5)
# ---------------------------------------------------------------------------


def _parse_or_empty(expr: str | None):
    """Parse a steering expression or return an empty Steering proxy."""
    if not expr:
        return None
    from saklas.core.steering_expr import parse_expr, SteeringExprError
    try:
        return parse_expr(expr)
    except SteeringExprError:
        return None


def _entry_alpha(entry: Any) -> float:
    """Extract the numeric coefficient from a Steering.alphas entry.

    Handles bare floats, ``(alpha, trigger)`` tuples, ``ProjectedTerm``,
    and ``AblationTerm`` uniformly so the delta formatter doesn't have
    to special-case each shape.
    """
    from saklas.core.steering_expr import AblationTerm, ProjectedTerm

    if isinstance(entry, (ProjectedTerm, AblationTerm)):
        return float(entry.coeff)
    if isinstance(entry, tuple):
        return float(entry[0])
    return float(entry)


def _format_term(name: str, alpha: float, *, sign_prefix: bool = True) -> str:
    """Render one term in the delta label.

    ``sign_prefix=True`` emits ``+`` / ``-`` ahead of the magnitude so
    edge labels read like ``+0.2 calm`` / ``-0.3 honest``; turn off for
    the leading position in compound labels.
    """
    if alpha == 0.0:
        return f"−{name}"
    if sign_prefix and alpha >= 0:
        return f"+{alpha:g} {name}"
    if alpha < 0:
        return f"{alpha:g} {name}"
    return f"{alpha:g} {name}"


def steering_delta(parent_expr: str | None, child_expr: str | None) -> str:
    """Render a compact delta label for a parent → child edge.

    Walks both expressions through the shared grammar (matching the
    keys used by ``Steering.alphas``), subtracts coefficients
    name-by-name, and returns a compact multi-term label.  Returns an
    empty string when the two expressions are identical (no label to
    render) and ``"(unsteered)"`` / ``"(unparsed)"`` for the
    edge cases where one side fails to parse — UIs render whichever
    sentinel they receive.

    Output format examples::

        steering_delta(None, "0.3 calm")            == "+0.3 calm"
        steering_delta("0.3 calm", "0.5 calm")      == "+0.2 calm"
        steering_delta("0.3 calm", "")              == "-0.3 calm"
        steering_delta("0.3 calm", "0.3 calm + 0.2 warm") == "+0.2 warm"
    """
    p = _parse_or_empty(parent_expr)
    c = _parse_or_empty(child_expr)

    p_alphas = dict(p.alphas) if p is not None else {}
    c_alphas = dict(c.alphas) if c is not None else {}

    # Quick equality short-circuit — surfaces use the empty string as
    # "no label needed".  Mappings compare structurally on Python dicts.
    if p_alphas == c_alphas:
        return ""

    keys = set(p_alphas) | set(c_alphas)
    parts: list[tuple[str, float]] = []
    for name in keys:
        pa = _entry_alpha(p_alphas[name]) if name in p_alphas else 0.0
        ca = _entry_alpha(c_alphas[name]) if name in c_alphas else 0.0
        delta = ca - pa
        if abs(delta) > 1e-12:
            parts.append((name, delta))

    if not parts:
        return ""

    # Sort by absolute magnitude descending so the visually dominant
    # change leads the label.  Ties broken alphabetically for stability.
    parts.sort(key=lambda item: (-abs(item[1]), item[0]))

    chunks: list[str] = []
    for i, (name, delta) in enumerate(parts):
        chunks.append(_format_term(name, delta, sign_prefix=(i > 0)))
    return " ".join(chunks)


__all__ = [
    "DiffSpan",
    "DiffState",
    "NodeDiff",
    "ReadingDelta",
    "TokenDeltaSpan",
    "per_token_diff",
    "readings_diff",
    "steering_delta",
    "text_diff",
]
