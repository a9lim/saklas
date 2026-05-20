"""Full-screen loom view — Textual ``Screen`` subclass.

The TUI's three-column chat layout is already crowded, so the loom view
is its own ``Screen`` reached via ``Ctrl+L`` from the chat screen.  It
is the TUI analog of the webui's loom sidebar + branch canvas + node-
compare drawer rolled into one keyboard-driven surface.

Layout (top → bottom):

- a one-line header (node count, current selection, active filter/mode);
- a horizontal split: ``Tree`` widget on the left, a scrollable node-
  detail / compare / help pane on the right;
- a two-line keyhint bar — a plain ``Static`` rather than Textual's
  ``Footer`` so every binding is always legible regardless of theme.

Navigation is cursor-based: ``↑↓`` / ``kj`` walk every visible node,
``←→`` / ``hl`` fold-and-parent / unfold-and-child, ``Tab`` / ``⇧Tab``
jump between siblings, ``Enter`` activates a node and returns to chat,
``Space`` activates without leaving.  All structural mutations route
through ``session.tree``; the chat screen re-renders along the new
active path on the next event tick because its message list is a
derived view over the tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Input, Static, Tree
from textual.widgets.tree import TreeNode

from saklas.core.loom import (
    InvalidNodeOperationError,
    LoomNode,
    LoomTreeError,
    MutationDuringGenerationError,
    UnknownNodeError,
)
from saklas.tui.loom_helpers import (
    format_compare,
    format_node_detail,
    search_nodes,
)

if TYPE_CHECKING:
    from saklas.tui.app import SaklasApp


_ROLE_GLYPH = {
    "system": "·",
    "user": "→",
    "assistant": "←",
}

# Probe-reading gutter ramp — index 0 is blank, 1-8 climb the block set.
# The TUI analog of the webui loom sidebar's decoration ring.
_GUTTER_RAMP = " ▁▂▃▄▅▆▇█"

_KEYHINT_NAV = (
    "[b ansi_yellow]nav[/]  "
    "[reverse] ↑↓ [/]/[reverse] kj [/] move   "
    "[reverse] ←→ [/]/[reverse] hl [/] fold   "
    "[reverse] ⇥ [/] sibling   "
    "[reverse] g [/]/[reverse] G [/] top/active   "
    "[reverse] ⏎ [/] activate+go   "
    "[reverse] ␣ [/] set active   "
    "[reverse] ⎋ [/] back"
)
_KEYHINT_ACT = (
    "[b ansi_yellow]act[/]  "
    "[reverse] e [/] edit  "
    "[reverse] b [/] branch  "
    "[reverse] r [/] regen  "
    "[reverse] d [/] delete  "
    "[reverse] s [/] star  "
    "[reverse] a [/] note  "
    "[reverse] f [/] fan  "
    "[reverse] c [/] compare  "
    "[reverse] / [/] search  "
    "[reverse] n [/]/[reverse] N [/] match  "
    "[reverse] p [/] filter  "
    "[reverse] ? [/] help"
)

_HELP_TEXT = (
    "[b]loom screen — keymap[/b]\n"
    "\n"
    "[b ansi_yellow]navigation[/]\n"
    "  ↑ ↓ / k j     move cursor through every visible node\n"
    "  ← → / h l     fold (then jump to parent) / unfold (then first child)\n"
    "  ⇥ / ⇧⇥        jump to next / previous sibling\n"
    "  g             jump to the first node\n"
    "  G             jump to the active node\n"
    "  ⏎             make the cursor node active, return to chat\n"
    "  ␣             make the cursor node active, stay in the loom\n"
    "  ⎋             leave compare/help/filter; otherwise back to chat\n"
    "\n"
    "[b ansi_yellow]mutation[/]  (all route through session.tree)\n"
    "  e             edit node text in place — no new node\n"
    "  b             branch a sibling (text pre-filled from this node)\n"
    "  r             regenerate N assistant siblings\n"
    "  d             delete the subtree (type 'yes' to confirm)\n"
    "  s             toggle star\n"
    "  a             edit the node note\n"
    "  f             fan out — <vector> <alpha-grid> as siblings\n"
    "\n"
    "[b ansi_yellow]analysis[/]\n"
    "  c             compare the assistant replies of this turn\n"
    "  /             search node text / notes\n"
    "  n / N         jump to the next / previous search match\n"
    "  p             edit the prune filter (dims non-matching nodes)\n"
    "  ?             toggle this help\n"
    "\n"
    "The left-gutter bar on each node tracks the selected highlight\n"
    "probe's reading for that node — green positive, red negative.\n"
    "\n"
    "[dim]press ? or ⎋ to dismiss[/]"
)


def _probe_gutter(node: LoomNode, probe: str | None) -> str:
    """One-char Rich-markup gutter for the highlight probe's reading.

    Returns a coloured block scaled to ``|reading|`` (green positive,
    red negative) or a blank space when no probe is selected, the probe
    is a sentinel, or the node carries no reading for it.
    """
    if not probe or probe.startswith("__"):
        return " "
    reading = node.aggregate_readings.get(probe)
    if reading is None:
        return " "
    idx = round(min(abs(reading), 1.0) * 8)
    char = _GUTTER_RAMP[idx]
    if char == " ":
        return " "
    color = "ansi_green" if reading >= 0 else "ansi_red"
    return f"[{color}]{char}[/]"


def _node_label(
    node: LoomNode,
    *,
    is_active: bool = False,
    dim: bool = False,
    gutter: str = " ",
) -> str:
    """One-line tree label — gutter, active marker, role glyph, id, snippet.

    The active node is bolded and flagged with ``»``; starred nodes get
    a trailing ``★``; in-place edits flag ``✎N``; truncated assistant
    completions flag ``⋯``.  ``dim=True`` (an active ``/prune`` filter)
    wraps the label so non-matching nodes recede.
    """
    glyph = _ROLE_GLYPH.get(node.role, "?")
    head = (node.text or "").strip().splitlines()
    snippet = head[0] if head else ""
    if len(snippet) > 44:
        snippet = snippet[:43] + "…"
    snippet = escape(snippet)
    tags = ""
    if node.starred:
        tags += " ★"
    if node.edit_count:
        tags += f" ✎{node.edit_count}"
    if node.role == "assistant" and node.finish_reason == "length":
        tags += " ⋯"
    marker = "»" if is_active else " "
    label = f"{gutter}{marker}{glyph} {node.id[:8]} {snippet}{tags}"
    if is_active:
        label = f"[b]{label}[/b]"
    if dim:
        label = f"[dim]{label}[/dim]"
    return label


class _PromptOverlay(Vertical):
    """Modal-ish single-line input bar that pops in above the keyhint bar.

    Used for ``e`` / ``b`` / ``r`` / ``d`` / ``a`` / ``f`` / ``/`` /
    ``p``.  The overlay grabs focus while active; ``Enter`` commits,
    ``Esc`` cancels.  Kept as a plain ``Vertical`` (not a nested
    ``Screen``) so Esc dispatch stays single-layer.
    """

    def __init__(self, prompt: str, *, initial: str = "", kind: str = "") -> None:
        super().__init__(id="loom-prompt-overlay")
        self.prompt = prompt
        self.kind = kind
        self.initial = initial

    def compose(self) -> ComposeResult:
        yield Static(self.prompt, id="loom-prompt-label")
        yield Input(value=self.initial, id="loom-prompt-input")

    def on_mount(self) -> None:
        # Focus the input once it and the overlay are fully mounted —
        # querying for it inside ``_open_overlay`` races the deferred
        # mount under Textual 8.x and silently leaves the tree focused.
        self.query_one("#loom-prompt-input", Input).focus()


class LoomScreen(Screen[None]):
    """Tree-of-completions view — see the module docstring for the keymap.

    All structural mutations route through ``self.session.tree``.
    Bindings live on the screen; the ones that collide with the focused
    ``Tree`` widget (``enter`` / ``space``) or with Textual's focus
    traversal (``tab``) are marked ``priority`` and gated by
    :meth:`check_action` so they fall through to a focused prompt input.
    """

    BINDINGS = [
        Binding("escape", "back", "Back", show=False, priority=True),
        Binding("enter", "make_active_and_back", "Activate", show=False, priority=True),
        Binding("space", "set_active", "Set active", show=False, priority=True),
        Binding("tab", "next_sibling", "Next sib", show=False, priority=True),
        Binding("shift+tab", "prev_sibling", "Prev sib", show=False, priority=True),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("h", "fold", "Fold", show=False),
        Binding("left", "fold", "Fold", show=False),
        Binding("l", "unfold", "Unfold", show=False),
        Binding("right", "unfold", "Unfold", show=False),
        Binding("g", "goto_first", "First", show=False),
        Binding("G", "goto_active", "Active", show=False),
        Binding("r", "regen", "Regen", show=False),
        Binding("e", "edit_node", "Edit", show=False),
        Binding("b", "branch_node", "Branch", show=False),
        Binding("d", "delete_node", "Delete", show=False),
        Binding("s", "toggle_star", "Star", show=False),
        Binding("a", "annotate", "Note", show=False),
        Binding("f", "fan_out", "Fan", show=False),
        Binding("c", "compare", "Compare", show=False),
        Binding("slash", "search", "Search", show=False),
        Binding("n", "search_next", "Next match", show=False),
        Binding("N", "search_prev", "Prev match", show=False),
        Binding("p", "prune", "Filter", show=False),
        Binding("question_mark", "help", "Help", show=False),
    ]

    DEFAULT_CSS = """
    LoomScreen {
        layout: vertical;
    }

    #loom-header {
        height: 1;
        dock: top;
        padding: 0 1;
        color: ansi_yellow;
    }

    #loom-main {
        height: 1fr;
        layout: horizontal;
    }

    #loom-tree {
        width: 2fr;
        height: 100%;
        border-right: solid ansi_default;
        padding: 0 1;
    }

    #loom-detail-scroll {
        width: 3fr;
        height: 100%;
        padding: 0 1;
        overflow-y: auto;
    }

    #loom-detail {
        height: auto;
    }

    #loom-keyhint {
        height: 2;
        dock: bottom;
        padding: 0 1;
        background: ansi_default;
        color: ansi_default;
    }

    #loom-prompt-overlay {
        height: 4;
        dock: bottom;
        border: solid ansi_yellow;
        padding: 0 1;
    }

    #loom-prompt-label {
        height: 1;
        color: ansi_yellow;
    }

    #loom-prompt-input {
        height: 3;
        border: solid ansi_default;
    }
    """

    def __init__(self, app: "SaklasApp") -> None:
        super().__init__()
        self._app: "SaklasApp" = app
        self._session = app._session
        self._tree_widget: Tree[str] | None = None
        self._detail: Static | None = None
        self._detail_scroll: VerticalScroll | None = None
        self._header: Static | None = None
        self._keyhint: Static | None = None
        # Map ulid → TreeNode reference so we can re-seat the cursor
        # after a mutation without re-walking the textual tree.
        self._node_index: dict[str, TreeNode[str]] = {}
        # ulid currently under the cursor (vs the session's logical
        # ``active_node_id``).
        self._cursor_id: str | None = None
        self._overlay: _PromptOverlay | None = None
        self._overlay_target: str | None = None
        # `/` search hit list + index — `n` / `N` walk it.
        self._search_hits: list[str] = []
        self._search_idx: int = 0
        # Prune filter match set, populated by ``_rebuild_tree``.
        self._match_ids: set[str] | None = None
        # Right-pane mode: node detail (default), compare, or help.
        self._compare_mode: bool = False
        self._show_help: bool = False
        # True while ``_rebuild_tree`` is tearing down / repopulating the
        # widget — stray ``NodeHighlighted`` events from ``Tree.clear``
        # are ignored so they can't clobber the re-seated cursor.
        self._rebuilding: bool = False

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static("", id="loom-header")
        with Horizontal(id="loom-main"):
            yield Tree("loom", id="loom-tree")
            with VerticalScroll(id="loom-detail-scroll"):
                yield Static("", id="loom-detail")
        yield Static(f"{_KEYHINT_NAV}\n{_KEYHINT_ACT}", id="loom-keyhint")

    def on_mount(self) -> None:
        self._tree_widget = self.query_one("#loom-tree", Tree)
        self._detail = self.query_one("#loom-detail", Static)
        self._detail_scroll = self.query_one("#loom-detail-scroll", VerticalScroll)
        self._header = self.query_one("#loom-header", Static)
        self._keyhint = self.query_one("#loom-keyhint", Static)
        # The detail pane scrolls with the mouse but never steals Tab
        # focus from the tree.
        self._detail_scroll.can_focus = False
        self._tree_widget.show_root = False
        self._tree_widget.guide_depth = 2
        self._rebuild_tree()
        self._tree_widget.focus()

    def check_action(
        self, action: str, parameters: tuple[object, ...],
    ) -> bool | None:
        # While a prompt overlay owns focus, every binding except the
        # Esc-cancel falls through so the input receives raw keys.
        if self._overlay is not None and action != "back":
            return False
        return True

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _rebuild_tree(self) -> None:
        """Snapshot the LoomTree and rebuild the Textual ``Tree`` widget.

        Full rebuild on every mutation — the trees are user-scale, the
        chat screen owns the hot streaming path, so a quadratic walk
        here stays well within budget and dodges incremental-diff bugs.
        """
        if self._tree_widget is None:
            return
        tree = self._session.tree
        active_id = tree.active_node_id
        widget = self._tree_widget
        self._rebuilding = True
        widget.clear()
        self._node_index.clear()
        self._match_ids = self._resolve_match_ids()
        for child_id in tree.child_ids(tree.root_id):
            self._mount_subtree(widget.root, child_id, active_id)
        widget.root.expand_all()
        # Re-seat the cursor once the widget has laid out — doing it now
        # races ``Tree.clear``'s own cursor reset.
        self.call_after_refresh(self._finish_rebuild)

    def _finish_rebuild(self) -> None:
        """Re-seat the cursor + refresh panes after a deferred rebuild.

        ``_cursor_id`` is read fresh here (not captured at rebuild time)
        so a navigation keypress landing in the brief rebuild window
        isn't clobbered by a stale re-seat.
        """
        self._rebuilding = False
        target = (
            self._cursor_id
            if (self._cursor_id and self._cursor_id in self._node_index)
            else self._session.tree.active_node_id
        )
        self._select_node(target)
        self._refresh_detail()
        self._refresh_header()

    def _resolve_match_ids(self) -> set[str] | None:
        """Node ids matching the active ``/prune`` filter, or ``None``.

        Parse / eval failures fall back to ``None`` (every node bright)
        and surface in the chat panel — the loom screen never blanks.
        """
        expr = getattr(self._app, "_loom_prune_expr", None)
        if not expr:
            return None
        try:
            return set(self._session.tree.filter_by_expr(expr))
        except Exception as e:
            try:
                self._app._chat_panel.add_system_message(
                    f"/prune filter eval failed: {e}"
                )
            except Exception:
                pass
            return None

    def _highlight_probe(self) -> str | None:
        """The chat screen's currently selected highlight probe, if any."""
        probe = getattr(self._app, "_highlight_probe", None)
        if isinstance(probe, str) and probe and not probe.startswith("__"):
            return probe
        return None

    def _mount_subtree(
        self, parent_widget_node: TreeNode[str], node_id: str, active_id: str,
    ) -> None:
        tree = self._session.tree
        node = tree.get(node_id)
        dim = (self._match_ids is not None) and (node_id not in self._match_ids)
        label = _node_label(
            node,
            is_active=(node_id == active_id),
            dim=dim,
            gutter=_probe_gutter(node, self._highlight_probe()),
        )
        kids = tree.child_ids(node_id)
        if kids:
            tnode = parent_widget_node.add(label, data=node_id, expand=True)
        else:
            tnode = parent_widget_node.add_leaf(label, data=node_id)
        self._node_index[node_id] = tnode
        for cid in kids:
            self._mount_subtree(tnode, cid, active_id)

    def _refresh_header(self) -> None:
        if self._header is None:
            return
        tree = self._session.tree
        n = max(0, len(tree.nodes) - 1)  # drop the synthetic root
        parts = [f"[b]loom[/b] · {n} node{'' if n == 1 else 's'}"]
        cur = self._current_node_id()
        if cur is not None:
            parts.append(f"sel {cur[:8]}")
        if self._show_help:
            parts.append("[reverse] help [/]")
        elif self._compare_mode:
            parts.append("[reverse] compare [/]")
        expr = getattr(self._app, "_loom_prune_expr", None)
        if expr:
            parts.append(f"filter: {escape(str(expr))}")
        self._header.update("  ·  ".join(parts))
        # ``sub_title`` keeps parity with the old footer-state surface.
        self.sub_title = f"prune: {expr}" if expr else ""

    def _select_node(self, node_id: str | None) -> None:
        if self._tree_widget is None or node_id is None:
            return
        tnode = self._node_index.get(node_id)
        if tnode is None:
            return
        self._cursor_id = node_id
        try:
            self._tree_widget.move_cursor(tnode, animate=False)
        except Exception:
            # Render layer may not have laid out yet — cursor id alone
            # is enough; the next rebuild re-seats it.
            pass

    def _current_node_id(self) -> str | None:
        return self._cursor_id

    def _refresh_detail(self) -> None:
        if self._detail is None:
            return
        if self._show_help:
            self._detail.update(_HELP_TEXT)
            self._scroll_detail_home()
            return
        nid = self._current_node_id()
        if nid is None:
            self._detail.update("[dim](no selection)[/]")
            return
        if self._compare_mode:
            try:
                self._detail.update(format_compare(self._session, nid))
            except UnknownNodeError:
                self._detail.update("[dim](node gone)[/]")
            self._scroll_detail_home()
            return
        try:
            self._detail.update(format_node_detail(self._session.tree, nid))
        except UnknownNodeError:
            self._detail.update("[dim](node gone)[/]")
        self._scroll_detail_home()

    def _scroll_detail_home(self) -> None:
        if self._detail_scroll is not None:
            try:
                self._detail_scroll.scroll_home(animate=False)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Tree-widget events
    # ------------------------------------------------------------------

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        if self._rebuilding:
            return
        node_id = event.node.data
        if isinstance(node_id, str):
            self._cursor_id = node_id
            self._refresh_detail()
            self._refresh_header()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[str]) -> None:
        if self._rebuilding:
            return
        node_id = event.node.data
        if isinstance(node_id, str):
            self._cursor_id = node_id
            self._refresh_detail()
            self._refresh_header()

    # ------------------------------------------------------------------
    # Navigation bindings
    # ------------------------------------------------------------------

    def action_back(self) -> None:
        if self._overlay is not None:
            self._dismiss_overlay()
            return
        if self._show_help:
            self._show_help = False
            self._refresh_detail()
            self._refresh_header()
            return
        if self._compare_mode:
            self._compare_mode = False
            self._refresh_detail()
            self._refresh_header()
            return
        self._return_to_chat()

    def action_cursor_down(self) -> None:
        if self._tree_widget is not None:
            self._tree_widget.action_cursor_down()

    def action_cursor_up(self) -> None:
        if self._tree_widget is not None:
            self._tree_widget.action_cursor_up()

    def _siblings(self, node_id: str) -> list[str]:
        tree = self._session.tree
        node = tree.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return []
        return tree.child_ids(node.parent_id)

    def action_next_sibling(self) -> None:
        cur = self._current_node_id()
        if cur is None:
            return
        sibs = self._siblings(cur)
        if not sibs or cur not in sibs:
            return
        self._select_node(sibs[(sibs.index(cur) + 1) % len(sibs)])

    def action_prev_sibling(self) -> None:
        cur = self._current_node_id()
        if cur is None:
            return
        sibs = self._siblings(cur)
        if not sibs or cur not in sibs:
            return
        self._select_node(sibs[(sibs.index(cur) - 1) % len(sibs)])

    def _parent_of(self, node_id: str) -> str | None:
        tree = self._session.tree
        node = tree.nodes.get(node_id)
        if node is None or node.parent_id is None or node.parent_id == tree.root_id:
            return None
        return node.parent_id

    def action_fold(self) -> None:
        """``h`` / ``←`` — collapse an expanded node, else jump to parent."""
        cur = self._current_node_id()
        if cur is None:
            return
        tnode = self._node_index.get(cur)
        if tnode is not None and tnode.allow_expand and tnode.is_expanded:
            tnode.collapse()
            return
        parent = self._parent_of(cur)
        if parent is not None:
            self._select_node(parent)

    def action_unfold(self) -> None:
        """``l`` / ``→`` — expand a collapsed node, else jump to first child."""
        cur = self._current_node_id()
        if cur is None:
            return
        tnode = self._node_index.get(cur)
        if tnode is not None and tnode.allow_expand and not tnode.is_expanded:
            tnode.expand()
            return
        kids = self._session.tree.child_ids(cur)
        if not kids:
            return
        # Prefer the active-path descendant so ``l`` walks the visible
        # chat thread by default.
        try:
            active_path_ids = {n.id for n in self._session.tree.active_path()}
        except Exception:
            active_path_ids = set()
        for cid in kids:
            if cid in active_path_ids:
                self._select_node(cid)
                return
        self._select_node(kids[0])

    def action_goto_first(self) -> None:
        kids = self._session.tree.child_ids(self._session.tree.root_id)
        if kids:
            self._select_node(kids[0])

    def action_goto_active(self) -> None:
        self._select_node(self._session.tree.active_node_id)

    def action_make_active_and_back(self) -> None:
        cur = self._current_node_id()
        if cur is not None:
            try:
                self._session.tree.navigate(cur)
            except UnknownNodeError:
                pass
        self._return_to_chat()

    def _return_to_chat(self) -> None:
        """Pop back to the chat screen, repainting the chat log so it
        shows the (possibly newly-navigated / mutated) active path.

        Every exit from the loom screen routes through here — navigation
        via ``Space``/``Enter`` and structural mutations (edit, branch,
        delete) all change the active path, and the chat panel must
        reflect it rather than the turns last streamed into it.
        """
        try:
            self._app._repaint_chat_from_active_path()
        except Exception:
            pass
        self._app.pop_screen()

    def action_set_active(self) -> None:
        """``Space`` — re-point the active node without leaving the loom."""
        cur = self._current_node_id()
        if cur is None:
            return
        try:
            self._session.tree.navigate(cur)
        except UnknownNodeError:
            return
        self._rebuild_tree()

    # ------------------------------------------------------------------
    # View bindings
    # ------------------------------------------------------------------

    def action_compare(self) -> None:
        self._show_help = False
        self._compare_mode = not self._compare_mode
        self._refresh_detail()
        self._refresh_header()

    def action_help(self) -> None:
        self._show_help = not self._show_help
        self._refresh_detail()
        self._refresh_header()

    def action_search(self) -> None:
        self._open_overlay(
            "search (text / notes substring):", initial="", kind="search",
        )

    def action_search_next(self) -> None:
        self._walk_search(+1)

    def action_search_prev(self) -> None:
        self._walk_search(-1)

    def _walk_search(self, step: int) -> None:
        hits = [h for h in self._search_hits if h in self._node_index]
        if not hits:
            self._app._chat_panel.add_system_message(
                "no active search — press / first."
            )
            return
        self._search_idx = (self._search_idx + step) % len(hits)
        self._select_node(hits[self._search_idx])
        self._app._chat_panel.add_system_message(
            f"match {self._search_idx + 1}/{len(hits)}"
        )

    def action_prune(self) -> None:
        current = getattr(self._app, "_loom_prune_expr", None) or ""
        self._open_overlay(
            "prune filter (blank clears — dims non-matching nodes):",
            initial=current, kind="prune",
        )

    # ------------------------------------------------------------------
    # Mutation bindings
    # ------------------------------------------------------------------

    def action_regen(self) -> None:
        self._open_overlay(
            "regen N (assistant siblings of the active node):",
            initial="1", kind="regen",
        )

    def action_edit_node(self) -> None:
        cur = self._current_node_id()
        if cur is None:
            return
        try:
            node = self._session.tree.get(cur)
        except UnknownNodeError:
            return
        self._open_overlay(
            f"edit {cur[:8]} (in place — replaces text, no new sibling):",
            initial=node.text, kind="edit", target=cur,
        )

    def action_branch_node(self) -> None:
        cur = self._current_node_id()
        if cur is None:
            return
        try:
            node = self._session.tree.get(cur)
        except UnknownNodeError:
            return
        self._open_overlay(
            f"branch from {cur[:8]} (creates a sibling; clear for blank):",
            initial=node.text, kind="branch", target=cur,
        )

    def action_delete_node(self) -> None:
        cur = self._current_node_id()
        if cur is None:
            return
        self._open_overlay(
            f"delete subtree {cur[:8]}?  type 'yes' to confirm:",
            initial="", kind="delete", target=cur,
        )

    def action_toggle_star(self) -> None:
        cur = self._current_node_id()
        if cur is None:
            return
        try:
            node = self._session.tree.get(cur)
            self._session.tree.star(cur, on=not node.starred)
        except (UnknownNodeError, LoomTreeError):
            return
        self._rebuild_tree()

    def action_annotate(self) -> None:
        cur = self._current_node_id()
        if cur is None:
            return
        try:
            node = self._session.tree.get(cur)
        except UnknownNodeError:
            return
        self._open_overlay(
            f"note on {cur[:8]}:", initial=node.notes, kind="note", target=cur,
        )

    def action_fan_out(self) -> None:
        self._open_overlay(
            "fan out: <vector> <alphas>  (e.g. 'angry.calm 0.0, 0.3, 0.7'):",
            initial="", kind="fan",
        )

    # ------------------------------------------------------------------
    # Overlay machinery
    # ------------------------------------------------------------------

    def _open_overlay(
        self, prompt: str, *, initial: str = "", kind: str, target: str | None = None,
    ) -> None:
        self._dismiss_overlay()
        self._overlay = _PromptOverlay(prompt=prompt, initial=initial, kind=kind)
        self._overlay_target = target
        # Hide the keyhint bar so the docked overlay owns the bottom rows.
        if self._keyhint is not None:
            self._keyhint.display = False
        # ``_PromptOverlay.on_mount`` focuses the input once mounted.
        self.mount(self._overlay)

    def _dismiss_overlay(self) -> None:
        if self._overlay is None:
            return
        try:
            self._overlay.remove()
        except Exception:
            pass
        self._overlay = None
        self._overlay_target = None
        if self._keyhint is not None:
            self._keyhint.display = True
        if self._tree_widget is not None:
            self._tree_widget.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._overlay is None:
            return
        kind = self._overlay.kind
        value = event.value or ""
        target = self._overlay_target
        self._dismiss_overlay()
        self._dispatch_overlay(kind, value, target)

    def _dispatch_overlay(
        self, kind: str, value: str, target: str | None,
    ) -> None:
        chat = self._app._chat_panel
        tree = self._session.tree

        if kind == "regen":
            try:
                n = max(1, int((value or "1").strip()))
            except ValueError:
                chat.add_system_message(f"regen: bad N '{value}'")
                return
            self._app._dispatch_loom_regen(n)
            return

        if kind == "edit":
            if target is None:
                return
            try:
                tree.edit(target, value)
            except (LoomTreeError, MutationDuringGenerationError,
                    InvalidNodeOperationError, UnknownNodeError) as e:
                chat.add_system_message(f"edit failed: {e}")
                return
            self._rebuild_tree()
            return

        if kind == "branch":
            if target is None:
                return
            try:
                new_id = tree.branch(target, value)
            except (LoomTreeError, InvalidNodeOperationError,
                    MutationDuringGenerationError, UnknownNodeError) as e:
                chat.add_system_message(f"branch failed: {e}")
                return
            self._cursor_id = new_id
            self._rebuild_tree()
            return

        if kind == "delete":
            if target is None:
                return
            if value.strip().lower() != "yes":
                chat.add_system_message("delete cancelled.")
                return
            try:
                removed = tree.delete_subtree(target)
            except (LoomTreeError, InvalidNodeOperationError,
                    MutationDuringGenerationError, UnknownNodeError) as e:
                chat.add_system_message(f"delete failed: {e}")
                return
            chat.add_system_message(f"deleted {removed} node(s).")
            self._cursor_id = None
            self._rebuild_tree()
            return

        if kind == "note":
            if target is None:
                return
            try:
                tree.annotate(target, value)
            except (LoomTreeError, UnknownNodeError) as e:
                chat.add_system_message(f"note failed: {e}")
                return
            self._rebuild_tree()
            return

        if kind == "fan":
            self._app._dispatch_loom_fan(value)
            return

        if kind == "prune":
            expr = value.strip()
            self._app._loom_prune_expr = expr or None
            chat.add_system_message(
                f"prune filter: {expr}" if expr else "prune filter cleared."
            )
            self._rebuild_tree()
            return

        if kind == "search":
            hits = search_nodes(tree, value)
            self._search_hits = hits
            self._search_idx = 0
            if not hits:
                chat.add_system_message(f"no matches for '{value}'")
                return
            chat.add_system_message(
                f"{len(hits)} match(es) — n / N to walk: "
                + ", ".join(h[:8] for h in hits[:8])
            )
            self._select_node(hits[0])
            return


__all__ = ["LoomScreen"]
