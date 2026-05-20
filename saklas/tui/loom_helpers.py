"""Helpers for the loom-screen + tree slash commands.

Kept separate from :mod:`saklas.tui.loom_screen` so the parsers and
prefix-resolution functions can be unit-tested without spinning up a
Textual app or importing the screen module.  The slash-command
dispatchers in :class:`saklas.tui.app.SaklasApp` call into the helpers
here; the screen module composes them with widget state.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.markup import escape

from saklas.core.loom import LoomNode, LoomTree

if TYPE_CHECKING:
    from saklas.core.loom_diff import DiffSpan


# ---------------------------------------------------------------------------
# Alpha-list parsing (shared with the webui's sweep drawer)
# ---------------------------------------------------------------------------


class AlphaListError(ValueError):
    """Raised when ``parse_alpha_list`` can't make sense of its input."""


_LINSPACE_RE = re.compile(
    r"^linspace\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,)]+?)\s*\)\s*$",
    re.IGNORECASE,
)


def parse_alpha_list(text: str) -> list[float]:
    """Parse the shared alpha-grid grammar used by `/fan` and the webui sweep.

    Three input shapes (in order of priority):

    - ``linspace(start, stop, n)`` — n evenly-spaced values inclusive of
      both endpoints (matches numpy's ``linspace``).  Single-point form
      (n=1) collapses to ``[start]``.
    - ``start:stop:step`` — step-form, step direction must match the
      span direction (``start<stop`` requires ``step>0`` and vice versa).
      Inclusive of stop within ``|step|*1e-9`` tolerance.
    - Comma-separated literal list (``0.0, 0.3, 0.7``).  Empty entries
      are skipped; an entirely empty list errors.

    Raises :class:`AlphaListError` for any parse failure so callers can
    surface a single ``Error: ...`` toast/system message rather than
    interpreting partial output.
    """

    text = (text or "").strip()
    if not text:
        raise AlphaListError("alpha list is empty")

    m = _LINSPACE_RE.match(text)
    if m:
        try:
            start = float(m.group(1))
            stop = float(m.group(2))
        except ValueError:
            raise AlphaListError("linspace bounds must be numbers")
        try:
            n = int(m.group(3))
        except ValueError:
            raise AlphaListError("linspace count must be a positive integer")
        if n < 1:
            raise AlphaListError("linspace count must be a positive integer")
        if not math.isfinite(start) or not math.isfinite(stop):
            raise AlphaListError("linspace bounds must be numbers")
        if n == 1:
            return [start]
        step = (stop - start) / (n - 1)
        return [start + step * i for i in range(n)]

    if ":" in text and "," not in text:
        parts = [p.strip() for p in text.split(":")]
        if len(parts) != 3:
            raise AlphaListError("range form is start:stop:step (three values)")
        try:
            start = float(parts[0])
            stop = float(parts[1])
            step = float(parts[2])
        except ValueError:
            raise AlphaListError("range values must be numbers")
        if not all(math.isfinite(v) for v in (start, stop, step)):
            raise AlphaListError("range values must be numbers")
        if step == 0:
            raise AlphaListError("range step must be non-zero")
        if (stop - start) * step < 0:
            raise AlphaListError("range step direction disagrees with start→stop")
        eps = abs(step) * 1e-9
        out: list[float] = []
        cur = start
        ascending = step > 0
        guard = 0
        while True:
            if ascending and cur > stop + eps:
                break
            if (not ascending) and cur < stop - eps:
                break
            out.append(round(cur, 12))
            cur += step
            guard += 1
            if guard >= 10_000:
                raise AlphaListError("range produced too many values")
        if not out:
            raise AlphaListError("range produced no values")
        return out

    out: list[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            raise AlphaListError(f"'{tok}' is not a number")
        if not math.isfinite(v):
            raise AlphaListError(f"'{tok}' is not a finite number")
        out.append(v)
    if not out:
        raise AlphaListError("no values parsed")
    return out


# ---------------------------------------------------------------------------
# Prefix-based node id resolution (`/nav <prefix>`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefixMatch:
    """Result of :func:`resolve_node_prefix` — either a unique hit or a
    collision."""

    node_id: str | None
    candidates: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.node_id is not None

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    @property
    def missing(self) -> bool:
        return not self.candidates


def resolve_node_prefix(tree: LoomTree, prefix: str) -> PrefixMatch:
    """Match ``prefix`` (case-insensitive) against the tree's node ids.

    ULIDs are upper-cased Crockford base32, so we upper-case the user
    input before scanning.  Returns a :class:`PrefixMatch` with the
    resolved id (or ``None``) and the list of candidates (so callers can
    surface "ambiguous: a, b, c" when multiple match).
    """

    if not prefix:
        return PrefixMatch(node_id=None, candidates=())
    needle = prefix.strip().upper()
    if not needle:
        return PrefixMatch(node_id=None, candidates=())
    hits = [nid for nid in tree.nodes if nid.startswith(needle)]
    if len(hits) == 1:
        return PrefixMatch(node_id=hits[0], candidates=tuple(hits))
    return PrefixMatch(node_id=None, candidates=tuple(sorted(hits)))


# ---------------------------------------------------------------------------
# Search helpers (`/` in the loom screen)
# ---------------------------------------------------------------------------


def search_nodes(tree: LoomTree, query: str) -> list[str]:
    """Return node ids whose text or notes contain ``query`` (case-insensitive).

    Skips the synthetic root.  Used by the loom screen's ``/`` text
    search and also exposed for `/nav --search` style flows.
    """

    if not query:
        return []
    needle = query.lower()
    out: list[str] = []
    for nid, node in tree.nodes.items():
        if nid == tree.root_id:
            continue
        haystack = (node.text or "") + " " + (node.notes or "")
        if needle in haystack.lower():
            out.append(nid)
    return out


# ---------------------------------------------------------------------------
# Active path / summary helpers (`/path`)
# ---------------------------------------------------------------------------


def _first_line(text: str, *, max_len: int = 80) -> str:
    if not text:
        return ""
    line = text.strip().splitlines()[0] if text.strip() else ""
    if len(line) > max_len:
        return line[: max_len - 1] + "…"
    return line


_ROLE_GLYPH = {
    "system": "·",
    "user": ">",
    "assistant": "<",
}


def format_path_summary(tree: LoomTree) -> str:
    """One-liner per node in the active path: id-prefix, role glyph, snippet."""

    path = tree.active_path()
    lines: list[str] = []
    for node in path:
        if node.id == tree.root_id:
            continue
        glyph = _ROLE_GLYPH.get(node.role, "?")
        snippet = _first_line(node.text, max_len=80)
        marker = " *" if node.starred else ""
        lines.append(f"{node.id[:8]} {glyph}{marker} {snippet}")
    if not lines:
        return "(empty path)"
    return "\n".join(lines)


def format_node_detail(tree: LoomTree, node_id: str) -> str:
    """Detail-pane block for a single node — used by `/path <id>` and the
    loom screen's right-side panel renderer.

    Phase 5 addition: when the node is an assistant with sibling
    assistant cousins under the same user-parent, a ``Recipe`` block at
    the bottom renders the canonical steering expression and the
    :func:`saklas.core.loom_diff.steering_delta` against each sibling
    so users see the per-sibling deltas without leaving the loom view.
    """

    node = tree.get(node_id)
    lines: list[str] = []
    lines.append(f"id        : {node.id}")
    lines.append(f"role      : {node.role}")
    if node.parent_id is not None:
        lines.append(f"parent    : {node.parent_id[:8]}")
    lines.append(f"children  : {len(tree.child_ids(node.id))}")
    if node.starred:
        lines.append("starred   : yes")
    if node.notes:
        lines.append(f"notes     : {node.notes}")
    if node.edit_count:
        lines.append(f"edits     : {node.edit_count}")
    if node.finish_reason:
        lines.append(f"finish    : {node.finish_reason}")
    if node.mean_logprob is not None:
        lines.append(f"mean lp   : {node.mean_logprob:+.3f}")
    if node.applied_steering:
        lines.append(f"steering  : {node.applied_steering}")
    if node.recipe is not None:
        recipe = node.recipe
        if recipe.steering:
            lines.append(f"recipe.s  : {recipe.steering}")
        if recipe.seed is not None:
            lines.append(f"recipe.sd : {recipe.seed}")
        if recipe.sampling is not None:
            samp = recipe.sampling
            t = getattr(samp, "temperature", None)
            p = getattr(samp, "top_p", None)
            m = getattr(samp, "max_tokens", None)
            lines.append(f"recipe.sm : T={t} top_p={p} max={m}")
        if recipe.thinking is not None:
            lines.append(f"recipe.th : {recipe.thinking}")

    # ------- Phase 5: per-sibling steering-delta block -----------------
    if node.role == "assistant" and node.parent_id is not None:
        sibling_block = _format_sibling_recipe_block(tree, node)
        if sibling_block:
            lines.append("")
            lines.append("--- Recipe ---")
            lines.extend(sibling_block)

    if node.aggregate_readings:
        lines.append("readings  :")
        items = sorted(
            node.aggregate_readings.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        for k, v in items[:10]:
            lines.append(f"  {k:<28} {v:+.4f}")
        if len(items) > 10:
            lines.append(f"  … {len(items) - 10} more")
    lines.append("")
    lines.append("--- text ---")
    lines.append(escape(node.text) if node.text else "(empty)")
    return "\n".join(lines)


def _format_sibling_recipe_block(tree: LoomTree, node: LoomNode) -> list[str]:
    """Render the canonical steering expression plus per-sibling deltas.

    Walks the active node's user-parent's assistant children, filters to
    the other assistants, and emits a one-line delta per sibling via
    :func:`saklas.core.loom_diff.steering_delta`.  Returns an empty list
    when the node has no assistant cousins (no comparison to make).
    """
    from saklas.core.loom_diff import steering_delta

    if node.parent_id is None:
        return []
    sib_ids = [
        cid for cid in tree.child_ids(node.parent_id)
        if cid != node.id and tree.get(cid).role == "assistant"
    ]
    if not sib_ids:
        return []

    def _expr(n: LoomNode) -> str:
        if n.recipe is not None and n.recipe.steering is not None:
            return n.recipe.steering
        return n.applied_steering or ""

    self_expr = _expr(node)
    out: list[str] = [
        f"steering: {self_expr or '(unsteered)'}",
    ]
    for idx, sid in enumerate(sib_ids, 1):
        sib = tree.get(sid)
        sib_expr = _expr(sib)
        delta = steering_delta(sib_expr, self_expr)
        if not delta:
            delta = "(identical)"
        out.append(f"  Δ from sibling {idx} ({sid[:8]}): {delta}")
    return out


# ---------------------------------------------------------------------------
# Sibling comparison (`c` in the loom screen)
# ---------------------------------------------------------------------------


def _render_text_diff(spans: list["DiffSpan"]) -> str:
    """Render a word-level diff (``loom_diff.DiffSpan`` list) as Rich markup.

    Equal spans render plain; deletions strike-through red, insertions
    green.  Span text is escaped so completion content can't inject
    markup into the detail pane.
    """
    parts: list[str] = []
    for span in spans:
        text = escape(span.text)
        if not text:
            continue
        if span.state == "equal":
            parts.append(text)
        elif span.state == "delete":
            parts.append(f"[ansi_red strike]{text}[/]")
        else:  # insert
            parts.append(f"[ansi_green]{text}[/]")
    return " ".join(parts) if parts else "[dim](identical text)[/]"


def format_compare(session: Any, node_id: str) -> str:
    """Render the loom screen's compare view for ``node_id``.

    ``session`` is duck-typed — only ``.tree`` and ``.diff_nodes`` are
    touched — so a stand-in works as well as a real ``SaklasSession``.

    The loom alternates user/assistant turns, so "compare" always means
    one thing: diff the assistant continuations of a single user turn.
    The anchor is resolved role-aware — an assistant node anchors on
    itself; a user node anchors on the assistant reply on the active
    path (falling back to its first assistant reply).  Emits, per other
    continuation: the steering delta, the top-5 probe-reading deltas,
    and the word-level text diff — the TUI analog of the webui's
    NodeCompare drawer.  Returns a short advisory string when no user
    turn with ≥2 assistant continuations can be resolved.
    """
    from saklas.core.loom_diff import steering_delta

    tree = session.tree
    node = tree.get(node_id)

    # Resolve the anchor assistant node + the user turn it continues.
    if node.role == "assistant" and node.parent_id is not None:
        anchor_id = node_id
        parent_id = node.parent_id
    elif node.role == "user":
        parent_id = node_id
        kids = [
            cid for cid in tree.child_ids(parent_id)
            if tree.get(cid).role == "assistant"
        ]
        if not kids:
            return "[dim](compare — this turn has no assistant replies)[/]"
        # Prefer the reply on the active path; else the first one.
        on_path = {n.id for n in tree.active_path()}
        anchor_id = next((k for k in kids if k in on_path), kids[0])
    else:
        return "[dim](compare — select a user turn or assistant reply)[/]"

    anchor = tree.get(anchor_id)

    sib_ids = [
        cid for cid in tree.child_ids(parent_id)
        if cid != anchor_id and tree.get(cid).role == "assistant"
    ]
    if not sib_ids:
        return "[dim](compare — this turn has only one assistant reply)[/]"

    def _expr(n: LoomNode) -> str:
        if n.recipe is not None and n.recipe.steering is not None:
            return n.recipe.steering
        return n.applied_steering or ""

    self_expr = _expr(anchor)
    lines: list[str] = [
        f"[b]compare[/b] {anchor_id[:8]} vs {len(sib_ids)} sibling(s)",
        f"this : {escape(self_expr) or '(unsteered)'}",
        "",
    ]
    for idx, sid in enumerate(sib_ids, 1):
        sib = tree.get(sid)
        diff = session.diff_nodes(sid, anchor_id)  # a=sibling, b=anchor
        delta = steering_delta(_expr(sib), self_expr) or "(identical steering)"
        lines.append(f"[b]── sibling {idx} · {sid[:8]} ──[/b]")
        lines.append(f"Δ steering : {escape(delta)}")
        readings = [r for r in diff.readings if abs(r.delta) > 1e-9][:5]
        if readings:
            lines.append("Δ readings :")
            for r in readings:
                col = "ansi_green" if r.delta >= 0 else "ansi_red"
                lines.append(
                    f"  {r.name:<26} [{col}]{r.delta:+.4f}[/]"
                    f"  ({r.a_value:+.3f} → {r.b_value:+.3f})"
                )
        lines.append("Δ text :")
        lines.append(_render_text_diff(diff.text))
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "AlphaListError",
    "PrefixMatch",
    "parse_alpha_list",
    "resolve_node_prefix",
    "search_nodes",
    "format_path_summary",
    "format_node_detail",
    "format_compare",
]
