"""Full-screen loom view — Textual ``Screen`` subclass.

Phase 4 surface (per ``docs/plans/loom.md``).  The TUI's three-column
chat layout is already crowded; a fourth column doesn't fit, so the
loom view is its own Screen reached via ``Ctrl+L`` from the chat
screen and dismissed with ``Esc``.

Layout: ``Tree`` widget on the left (node ids + role glyph + first-line
snippet), node-detail pane on the right (full text, recipe, probe
readings, notes, edit-count flag), footer with binding hints.

The chat screen continues to render the active path linearly; tree
mutations from this screen go through ``session.tree`` (same primitives
the webui phase 3 talks to over HTTP), and the chat screen re-renders
on the next event tick because ``_messages`` is a derived view over
the tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Input, Static, Tree
from textual.widgets.tree import TreeNode

from saklas.core.loom import (
    InvalidNodeOperationError,
    LoomNode,
    LoomTreeError,
    MutationDuringGenerationError,
    UnknownNodeError,
)
from saklas.tui.loom_helpers import (
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


def _node_label(
    node: LoomNode,
    *,
    is_active: bool = False,
    dim: bool = False,
) -> str:
    """One-line label for a tree node — id-prefix, role glyph, snippet.

    Active node is bolded; starred nodes get a leading ``*``; in-place
    edits flag a trailing ``[edited]`` marker.  ``dim=True`` (phase 5
    ``/prune``) wraps the label in Rich's ``[dim]`` markup so non-
    matching nodes recede visually while the matching nodes stay bright.
    """

    glyph = _ROLE_GLYPH.get(node.role, "?")
    snippet = (node.text or "").strip().splitlines()
    head = snippet[0] if snippet else ""
    if len(head) > 40:
        head = head[:39] + "…"
    star = "*" if node.starred else " "
    edit = " [edited]" if node.edit_count else ""
    label = f"{star}{node.id[:8]} {glyph} {head}{edit}"
    if is_active:
        label = f"[b]{label}[/b]"
    if dim:
        label = f"[dim]{label}[/dim]"
    return label


class _PromptOverlay(Vertical):
    """Modal-ish single-line input bar that pops in below the tree.

    Used for ``r`` (regen N), ``e`` (edit), ``b`` (branch), ``a`` (note),
    ``f`` (fan-out), ``/`` (search).  The overlay grabs focus while
    active; ``Enter`` commits, ``Esc`` cancels.  Kept as a plain
    ``Vertical`` so layout is just "appears below the tree" rather than
    a full Textual modal screen — the loom screen is already a screen,
    nesting another would complicate Esc dispatch.
    """

    def __init__(self, prompt: str, *, initial: str = "", kind: str = "") -> None:
        super().__init__(id="loom-prompt-overlay")
        self.prompt = prompt
        self.kind = kind
        self.initial = initial

    def compose(self) -> ComposeResult:
        yield Static(self.prompt, id="loom-prompt-label")
        yield Input(value=self.initial, id="loom-prompt-input")


class LoomScreen(Screen[None]):
    """Tree-of-completions view.

    All structural mutations route through ``self.session.tree``; the
    five core primitives (edit, branch, navigate, delete_subtree, plus
    regenerate via ``session.generate(parent_node_id=..., n=...)``)
    are the only ops we call.  Decoration (`star`, `annotate`) is
    free relative to in-flight gen and lands inline.

    Bindings live on the screen so they don't fight with the chat
    screen's bindings (Textual scopes bindings to the focused screen).
    """

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("j", "next_sibling", "Next sib", show=True),
        Binding("k", "prev_sibling", "Prev sib", show=True),
        Binding("h", "up", "Up", show=True),
        Binding("left", "up", "Up", show=False),
        Binding("l", "down", "Down", show=True),
        Binding("right", "down", "Down", show=False),
        Binding("enter", "make_active_and_back", "Activate + back to chat", show=True),
        Binding("r", "regen", "Regen", show=True),
        Binding("e", "edit_node", "Edit", show=True),
        Binding("b", "branch_node", "Branch", show=True),
        Binding("d", "delete_node", "Del", show=True),
        Binding("s", "toggle_star", "Star", show=True),
        Binding("a", "annotate", "Note", show=True),
        Binding("f", "fan_out", "Fan", show=True),
        Binding("slash", "search", "Search", show=True),
    ]

    DEFAULT_CSS = """
    LoomScreen {
        layout: vertical;
    }

    #loom-main {
        height: 1fr;
        layout: horizontal;
    }

    #loom-tree {
        width: 1fr;
        height: 100%;
        border-right: solid ansi_default;
        padding: 0 1;
    }

    #loom-detail {
        width: 2fr;
        height: 100%;
        padding: 0 1;
        overflow-y: auto;
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
        # Map ulid → TreeNode reference so we can re-select after a mutation
        # without re-walking the textual tree.
        self._node_index: dict[str, TreeNode[str]] = {}
        # ulid currently highlighted in the Textual tree widget (vs the
        # session's ``active_node_id`` which is the *logical* active node).
        self._cursor_id: str | None = None
        self._overlay: _PromptOverlay | None = None
        # Stashed bookkeeping for the active prompt — kind, target id at
        # invoke time, etc.  Cleared on commit/cancel.
        self._overlay_target: str | None = None
        # Most recent search hit list for `/` text search; index points at
        # which one's currently focused so re-pressing `/` (or `n`) walks
        # to the next match.  Phase 4 keeps it simple — `/` re-prompts.
        self._search_hits: list[str] = []
        # Phase 5 prune filter — populated by ``_rebuild_tree`` per
        # ``app._loom_prune_expr``.  ``None`` means no filter is active.
        self._match_ids: set[str] | None = None

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="loom-main"):
            yield Tree("loom", id="loom-tree")
            yield Static("", id="loom-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._tree_widget = self.query_one("#loom-tree", Tree)
        self._detail = self.query_one("#loom-detail", Static)
        self._tree_widget.show_root = False
        self._tree_widget.guide_depth = 2
        self._rebuild_tree()
        self._tree_widget.focus()

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _rebuild_tree(self) -> None:
        """Snapshot the LoomTree and rebuild the Textual ``Tree`` widget.

        Phase-4 simplification: full rebuild on every mutation rather
        than partial deltas.  The trees we're rendering are user-scale
        (dozens of nodes, hundreds for power users), and the chat screen
        owns the hot streaming path — so a quadratic walk here is well
        within budget and dodges incremental-diff bugs.

        Phase 5 addition: when ``app._loom_prune_expr`` is set, run the
        filter via :meth:`LoomTree.filter_by_expr` and dim every node
        outside the matching set.  Filter-parse failures surface in the
        chat log; the loom screen falls back to no-dim rather than
        blanking the tree.
        """

        if self._tree_widget is None:
            return
        tree = self._session.tree
        active_id = tree.active_node_id
        widget = self._tree_widget
        widget.clear()
        self._node_index.clear()
        # Resolve the filter dim set up-front so per-node label building
        # is a constant-time lookup against the matching-id set.
        self._match_ids = self._resolve_match_ids()
        root = widget.root
        # Synthetic LoomTree root is rendered as a header — we mount its
        # children directly under the Textual tree's root.
        for child_id in tree.child_ids(tree.root_id):
            self._mount_subtree(root, child_id, active_id)
        widget.root.expand_all()
        # Re-seat the cursor.  Prefer the last-known cursor id; fall
        # back to the session's active node.
        target = self._cursor_id if (self._cursor_id and self._cursor_id in self._node_index) else active_id
        self._select_node(target)
        self._refresh_detail()
        self._refresh_footer()

    def _resolve_match_ids(self) -> set[str] | None:
        """Return the set of node ids matching the active /prune filter.

        ``None`` means no filter is active (every node renders bright).
        Filter parse / eval failures fall back to ``None`` so the loom
        screen never blanks out — errors get surfaced in the chat panel.
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

    def _mount_subtree(
        self, parent_widget_node: TreeNode[str], node_id: str, active_id: str,
    ) -> None:
        tree = self._session.tree
        node = tree.get(node_id)
        match_ids = getattr(self, "_match_ids", None)
        dim = (match_ids is not None) and (node_id not in match_ids)
        label = _node_label(node, is_active=(node_id == active_id), dim=dim)
        kids = tree.child_ids(node_id)
        if kids:
            tnode = parent_widget_node.add(label, data=node_id, expand=True)
        else:
            tnode = parent_widget_node.add_leaf(label, data=node_id)
        self._node_index[node_id] = tnode
        for cid in kids:
            self._mount_subtree(tnode, cid, active_id)

    def _refresh_footer(self) -> None:
        """Echo the active prune expression into the footer when one is set."""
        expr = getattr(self._app, "_loom_prune_expr", None)
        try:
            # Existence-check the Footer widget; we don't need the reference
            # itself because Textual's Footer doesn't expose first-class
            # custom text — we surface state through ``sub_title`` instead.
            self.query_one(Footer)
        except Exception:
            return
        if expr:
            self.sub_title = f"prune: {expr}"
        else:
            self.sub_title = ""

    def _select_node(self, node_id: str | None) -> None:
        if self._tree_widget is None or node_id is None:
            return
        tnode = self._node_index.get(node_id)
        if tnode is None:
            return
        self._cursor_id = node_id
        # Textual's Tree exposes ``select_node`` on the widget; line
        # placement on top of that puts the cursor where the user can
        # see it.
        try:
            self._tree_widget.select_node(tnode)
            line = getattr(tnode, "line", None)
            if line is not None and line >= 0:
                self._tree_widget.scroll_to_line(line)
        except Exception:
            # Fallback: re-target the cursor id only — render layer
            # may not have laid out yet.
            pass

    def _current_node_id(self) -> str | None:
        return self._cursor_id

    def _refresh_detail(self) -> None:
        if self._detail is None:
            return
        nid = self._current_node_id()
        if nid is None:
            self._detail.update("[dim](no selection)[/]")
            return
        try:
            text = format_node_detail(self._session.tree, nid)
        except UnknownNodeError:
            self._detail.update("[dim](node gone)[/]")
            return
        self._detail.update(text)

    # ------------------------------------------------------------------
    # Tree-widget events
    # ------------------------------------------------------------------

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        node_id = event.node.data
        if isinstance(node_id, str):
            self._cursor_id = node_id
            self._refresh_detail()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[str]) -> None:
        node_id = event.node.data
        if isinstance(node_id, str):
            self._cursor_id = node_id
            self._refresh_detail()

    # ------------------------------------------------------------------
    # Navigation bindings
    # ------------------------------------------------------------------

    def action_back(self) -> None:
        if self._overlay is not None:
            self._dismiss_overlay()
            return
        self._app.pop_screen()

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
        idx = sibs.index(cur)
        nxt = sibs[(idx + 1) % len(sibs)]
        self._select_node(nxt)

    def action_prev_sibling(self) -> None:
        cur = self._current_node_id()
        if cur is None:
            return
        sibs = self._siblings(cur)
        if not sibs or cur not in sibs:
            return
        idx = sibs.index(cur)
        nxt = sibs[(idx - 1) % len(sibs)]
        self._select_node(nxt)

    def action_up(self) -> None:
        cur = self._current_node_id()
        if cur is None:
            return
        tree = self._session.tree
        node = tree.nodes.get(cur)
        if node is None or node.parent_id is None or node.parent_id == tree.root_id:
            return
        self._select_node(node.parent_id)

    def action_down(self) -> None:
        cur = self._current_node_id()
        if cur is None:
            return
        tree = self._session.tree
        kids = tree.child_ids(cur)
        if not kids:
            return
        # Prefer the active-path descendant if one exists, so ``l`` walks
        # the visible chat thread by default.
        try:
            active_path_ids = {n.id for n in tree.active_path()}
        except Exception:
            active_path_ids = set()
        for cid in kids:
            if cid in active_path_ids:
                self._select_node(cid)
                return
        self._select_node(kids[0])

    def action_make_active_and_back(self) -> None:
        cur = self._current_node_id()
        if cur is not None:
            try:
                self._session.tree.navigate(cur)
            except UnknownNodeError:
                pass
        # Hand the active path back to the chat screen.
        self._app.pop_screen()

    # ------------------------------------------------------------------
    # Mutation bindings
    # ------------------------------------------------------------------

    def action_regen(self) -> None:
        self._open_overlay(
            "regen N (siblings of the active assistant):",
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
            f"note on {cur[:8]}:",
            initial=node.notes, kind="note", target=cur,
        )

    def action_fan_out(self) -> None:
        self._open_overlay(
            "fan out: <vector> <alphas>  (e.g. 'angry.calm 0.0, 0.3, 0.7'):",
            initial="", kind="fan",
        )

    def action_search(self) -> None:
        self._open_overlay(
            "search (text/notes substring):",
            initial="", kind="search",
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
        self.mount(self._overlay)
        # Focus the input so Enter / Esc work.
        try:
            inp = self.query_one("#loom-prompt-input", Input)
            inp.focus()
        except Exception:
            pass

    def _dismiss_overlay(self) -> None:
        if self._overlay is None:
            return
        try:
            self._overlay.remove()
        except Exception:
            pass
        self._overlay = None
        self._overlay_target = None
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
            # Route through the app so the same defer/queue plumbing
            # the chat screen uses for /regen kicks in.
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

        if kind == "search":
            hits = search_nodes(tree, value)
            self._search_hits = hits
            if not hits:
                chat.add_system_message(f"no matches for '{value}'")
                return
            chat.add_system_message(
                f"{len(hits)} match(es): " + ", ".join(h[:8] for h in hits[:8])
            )
            self._select_node(hits[0])
            return


__all__ = ["LoomScreen"]
