"""Loom slash-command + worker controller for the saklas TUI.

Owns the loom-screen's business logic: the inline slash handlers
(``/tree``, ``/nav``, ``/edit``, ``/branch``, ``/del``, ``/star``,
``/note``, ``/path``, ``/fan``, ``/prune``, ``/auto-regen``, ``/diff``),
the fan/regen worker bodies, and the three stash attrs they read/write —
``_loom_prune_expr``, ``_loom_auto_regen_mode``, ``_loom_auto_regen_on``.

This is a plain controller composed onto :class:`~saklas.tui.app.SaklasApp`
via a back-reference (``self._app``).  The framework-dispatched surface
(slash-command registry entries, key bindings, ``action_*``) stays on the
App as one-line forwarders into this class; the App also retains the
generation / poll orchestration these handlers hang off (the gen-lifecycle
counters, ``_ui_token_queue``, ``run_worker`` / ``call_from_thread``, the
A/B / shadow plumbing).  The controller reaches the session, widgets, and
those App helpers through ``self._app``.

The auto-regen state is a **shared read**: ``_poll_generation`` reads
``_loom_auto_regen_on`` / ``_loom_auto_regen_mode`` every tick (15 FPS, not
per-token) to route the post-gen hook, and ``_loom_prune_expr`` feeds the
status-footer dedupe tuple.  The App exposes all three as read/write proxy
properties into this controller so those reads, the test suite, and
``loom_screen.py`` reach the same state.  The *firing* of an auto-regen
sibling (``_fire_auto_regen``) is generation lifecycle and stays on the App
— this controller owns the **mode** state + parsing only.
"""

from __future__ import annotations

import shlex
from typing import Any, TYPE_CHECKING

from saklas import Recipe, SamplingConfig, Steering
from saklas.core.errors import SaklasError
from saklas.io.selectors import AmbiguousSelectorError
from saklas.tui.chat_panel import PendingItem

if TYPE_CHECKING:
    from saklas.tui.app import SaklasApp


class LoomController:
    """Owns the loom slash/worker logic + the auto-regen / prune mode state."""

    _AUTO_REGEN_MODES = ("unsteered", "inverted", "reseed", "cool", "hot")

    def __init__(self, app: "SaklasApp") -> None:
        self._app = app
        # Phase-4 loom: stashed prune expression + auto-regen mode.  Phase 5
        # consumes them; phase 4 only carries the strings so users can set
        # them up before phase 5 evaluator lands.
        self._loom_prune_expr: str | None = None
        # str for built-in modes (unsteered/inverted/reseed/cool/hot);
        # Recipe for the custom-mode path ``/auto-regen custom: <expr>``.
        # ``_render_auto_regen_mode`` renders either shape for display.
        self._loom_auto_regen_mode: "str | Recipe" = "unsteered"
        # Phase 5: auto-regen on/off — when on, every primary
        # ``_generate_core`` completion fires a sibling regen with the
        # configured override.  Default-off; ``Ctrl+A`` toggles.  The
        # existing ``_ab_mode`` flag stays as the visible two-column
        # layout (per plan decision 13: A/B becomes the default mode of
        # the more general auto-regen modifier).
        self._loom_auto_regen_on: bool = False

    # ----------------------------------------------------------------
    # Loom slash commands + bindings
    # ----------------------------------------------------------------

    def _handle_tree(self, _arg: str) -> None:
        """`/tree` — push the LoomScreen onto the Textual screen stack.

        Esc on the loom screen pops back to the chat screen.  Mutations
        from the loom screen flow into ``session.tree`` directly; the
        chat screen's session-history reads pick them
        up on the next render.
        """

        from saklas.tui.loom_screen import LoomScreen

        app = self._app
        if app._raw_mode and app._chat_panel.raw_buffer.is_dirty:
            app._chat_panel.add_system_message(
                "Raw buffer has uncommitted edits; press Ctrl+Enter to "
                "commit them, or Enter to continue before opening the tree."
            )
            return
        try:
            app.push_screen(LoomScreen(app))
        except Exception as e:
            app._chat_panel.add_system_message(f"/tree failed: {e}")

    def _handle_nav(self, arg: str) -> None:
        """`/nav <id-prefix>` — navigate active node by ulid prefix.

        Matches any case-insensitive prefix of an existing node id;
        ambiguous prefixes report the candidates.
        """

        from saklas.tui.loom_helpers import resolve_node_prefix

        app = self._app
        chat = app._chat_panel
        prefix = arg.strip()
        if not prefix:
            chat.add_system_message("Usage: /nav <id-prefix>")
            return
        if app._raw_mode and chat.raw_buffer.is_dirty:
            chat.add_system_message(
                "Raw buffer has uncommitted edits; press Ctrl+Enter to "
                "commit them, or Enter to continue before navigating."
            )
            return
        match = resolve_node_prefix(app._session.tree, prefix)
        if match.missing:
            chat.add_system_message(f"no node matches '{prefix}'")
            return
        if match.ambiguous:
            cands = ", ".join(c[:12] for c in match.candidates[:8])
            chat.add_system_message(f"ambiguous '{prefix}': {cands}")
            return
        # node_id is non-None here: match.missing and match.ambiguous are both False,
        # which by PrefixMatch's invariant means node_id was assigned a str hit.
        assert match.node_id is not None  # noqa: S101
        try:
            app._session.tree.navigate(match.node_id)
        except Exception as e:
            chat.add_system_message(f"navigate failed: {e}")
            return
        app._repaint_chat_from_active_path()
        chat.add_system_message(f"navigated to {match.node_id[:8]}")

    def _handle_edit(self, arg: str) -> None:
        """`/edit <text...>` — in-place edit of the active node.

        Phase-4 inline form: full replacement text comes on the same
        line.  The richer loom-screen overlay (which pre-fills the
        buffer with current text) lands via the `e` binding inside
        the loom screen.
        """

        from saklas.core.loom import (
            LoomTreeError, MutationDuringGenerationError,
            InvalidNodeOperationError, UnknownNodeError,
        )

        app = self._app
        chat = app._chat_panel
        if not arg.strip():
            chat.add_system_message("Usage: /edit <text...>")
            return
        target = app._session.tree.active_node_id
        if target == app._session.tree.root_id:
            chat.add_system_message("/edit: active node is the root.")
            return
        try:
            app._session.tree.edit(target, arg)
        except (UnknownNodeError, MutationDuringGenerationError,
                InvalidNodeOperationError, LoomTreeError) as e:
            chat.add_system_message(f"/edit failed: {e}")
            return
        app._sync_raw_buffer_from_tree()
        chat.add_system_message(f"edited {target[:8]}")

    def _handle_branch(self, arg: str) -> None:
        """`/branch [text]` — sibling of the active node with the given text.

        Empty text is the "branch from blank" UI flavor; otherwise the
        text becomes the new sibling's content.  Inherits the active
        node's role unless the active node is the root (rejected).
        """

        from saklas.core.loom import (
            LoomTreeError, MutationDuringGenerationError,
            InvalidNodeOperationError, UnknownNodeError,
        )

        app = self._app
        chat = app._chat_panel
        text = arg or ""
        target = app._session.tree.active_node_id
        if target == app._session.tree.root_id:
            chat.add_system_message("/branch: active node is the root.")
            return
        try:
            new_id = app._session.tree.branch(target, text)
        except (UnknownNodeError, MutationDuringGenerationError,
                InvalidNodeOperationError, LoomTreeError) as e:
            chat.add_system_message(f"/branch failed: {e}")
            return
        app._sync_raw_buffer_from_tree()
        chat.add_system_message(f"branched {target[:8]} → {new_id[:8]}")

    def _handle_del(self, arg: str) -> None:
        """`/del [yes]` — delete the active subtree.

        Requires explicit ``yes`` confirmation by default to avoid
        accidental wipes; ``Ctrl+D`` from the chat screen uses this
        path too.  :meth:`LoomTree.delete_subtree` repoints the active
        pointer to the surviving parent when the doomed subtree
        contains the active node (root case → fresh start), so this
        path is a straight-through call; the post-delete message
        surfaces the new active id so the jump isn't silent.
        """

        from saklas.core.loom import (
            LoomTreeError, MutationDuringGenerationError,
            InvalidNodeOperationError, UnknownNodeError,
        )

        app = self._app
        chat = app._chat_panel
        confirm = (arg or "").strip().lower()
        if confirm != "yes":
            chat.add_system_message(
                "/del: type '/del yes' to delete the active subtree."
            )
            return
        tree = app._session.tree
        target = tree.active_node_id
        if target == tree.root_id:
            chat.add_system_message("/del: nothing to delete (at root).")
            return
        try:
            removed = tree.delete_subtree(target)
        except (UnknownNodeError, MutationDuringGenerationError,
                InvalidNodeOperationError, LoomTreeError) as e:
            chat.add_system_message(f"/del failed: {e}")
            return
        new_active_id = tree.active_node_id
        app._sync_raw_buffer_from_tree()
        chat.add_system_message(
            f"deleted {removed} node(s); active now {new_active_id[:8]}"
        )

    def _handle_star(self, _arg: str) -> None:
        app = self._app
        chat = app._chat_panel
        target = app._session.tree.active_node_id
        try:
            node = app._session.tree.get(target)
            app._session.tree.star(target, on=not node.starred)
        except Exception as e:
            chat.add_system_message(f"/star failed: {e}")
            return
        chat.add_system_message(
            f"{'starred' if not node.starred else 'unstarred'} {target[:8]}"
        )

    def _handle_note(self, arg: str) -> None:
        app = self._app
        chat = app._chat_panel
        target = app._session.tree.active_node_id
        try:
            app._session.tree.annotate(target, arg or "")
        except Exception as e:
            chat.add_system_message(f"/note failed: {e}")
            return
        chat.add_system_message(f"noted {target[:8]}")

    def _handle_path(self, _arg: str) -> None:
        from saklas.tui.loom_helpers import format_path_summary
        self._app._chat_panel.add_system_message(
            format_path_summary(self._app._session.tree)
        )

    def _handle_fan(self, arg: str) -> None:
        """`/fan <vector> <alphas>` — N-way regen with per-sibling alpha override.

        Keep it minimal: token-split on the first
        whitespace, treat everything after as the alpha grid.  The
        webui fan grammar (linspace / range / comma list) is
        shared via :func:`parse_alpha_list`.
        """

        chat = self._app._chat_panel
        raw = arg.strip()
        if not raw:
            chat.add_system_message("Usage: /fan <vector> <alphas>")
            return
        parts = raw.split(None, 1)
        if len(parts) < 2:
            chat.add_system_message("Usage: /fan <vector> <alphas>")
            return
        vector, alphas_str = parts[0], parts[1]
        from saklas.tui.loom_helpers import parse_alpha_list, AlphaListError
        try:
            alphas = parse_alpha_list(alphas_str)
        except AlphaListError as e:
            chat.add_system_message(f"/fan alpha grid error: {e}")
            return
        if not alphas:
            chat.add_system_message("/fan: alpha grid is empty.")
            return
        self._dispatch_loom_fan_alphas(vector, alphas)

    def _dispatch_loom_fan(self, raw: str) -> None:
        """Called from the loom-screen overlay's fan-out form."""
        self._handle_fan(raw)

    def _dispatch_loom_fan_alphas(self, vector: str, alphas: list[float]) -> None:
        """Kick off the fan-out generation.

        Routes through ``session.generate_sweep`` so every alpha row
        lands as a sibling under one shared user turn. When a
        generation is already in flight we stash the request as a
        pending action so it fires after the current worker resolves.
        """

        app = self._app
        chat = app._chat_panel
        prompt = app._last_prompt
        # We need a prompt to regen from; if there isn't one yet, lift
        # it off the active path.
        if not prompt:
            hist = app._session.history
            if hist and hist[-1]["role"] == "user":
                prompt = hist[-1]["content"]
        if not prompt:
            chat.add_system_message("/fan: no prior prompt to fan out from.")
            return

        # Queue the fan-out behind any in-flight work; ``_run_fan_worker``
        # is invoked by ``_dispatch_pending_action`` once the queue head
        # is drained.
        if app._is_busy:
            display_text = f"/fan {vector} ({len(alphas)} α)"
            app._enqueue_pending(
                PendingItem("fan", display_text, (vector, alphas, prompt))
            )
            return
        # Through the App attr so a test that monkeypatches
        # ``app._run_fan_worker`` (and the ``_dispatch_pending_action``
        # drain that calls the App method) sees the same worker.
        app._run_fan_worker(vector, alphas, prompt)

    def _run_fan_worker(
        self, vector: str, alphas: list[float], prompt: str,
    ) -> None:
        """Actually kick the engine.

        Routes through :meth:`SaklasSession.generate_sweep` so every
        sibling lands under one shared user-turn anchor in the loom tree.
        """

        app = self._app
        chat = app._chat_panel
        chat.add_system_message(
            f"/fan {vector} × {len(alphas)} (α: "
            f"{', '.join(f'{a:+.2f}' for a in alphas[:6])}"
            f"{'…' if len(alphas) > 6 else ''})"
        )

        def _worker() -> None:
            try:
                tree = app._session.tree
                # Anchor under the active node's user-parent (so the
                # sweep's auto-spawned user turn lands as a sibling of
                # the existing user turn rather than nested under the
                # previous assistant).
                anchor_id = tree.active_node_id
                anchor = tree.nodes.get(anchor_id)
                if anchor is not None and anchor.role == "assistant" and anchor.parent_id is not None:
                    parent_for_sweep = anchor.parent_id
                    # If the active path's current user turn already
                    # holds the prompt we're sweeping, anchor under
                    # *that* user turn so generate_sweep's dedup folds
                    # the new sweep into the existing user node.
                    user_node = tree.nodes.get(anchor.parent_id)
                    if user_node is not None and user_node.role == "user":
                        parent_for_sweep = user_node.parent_id
                else:
                    parent_for_sweep = anchor_id

                sampling = SamplingConfig(
                    temperature=app._session.config.temperature,
                    top_p=app._session.config.top_p,
                    max_tokens=app._session.config.max_new_tokens,
                    seed=app._default_seed,
                )
                runset = app._session.generate_sweep(
                    prompt,
                    {vector: [float(a) for a in alphas]},
                    sampling=sampling,
                    stateless=False,
                    parent_node_id=parent_for_sweep,
                )
                node_ids = runset.node_ids
                kept = [nid for nid in node_ids if nid]
                app.call_from_thread(
                    app._chat_panel.add_system_message,
                    f"/fan {vector}: {len(kept)} siblings landed "
                    f"({', '.join(nid[:8] for nid in kept[:6])}"
                    f"{'…' if len(kept) > 6 else ''})",
                )
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                app.call_from_thread(
                    app._chat_panel.add_system_message,
                    f"/fan error: {msg}",
                )

        app.run_worker(_worker, thread=True)

    def _dispatch_loom_regen(
        self, n: int, *, mode: "str | Recipe | None" = None,
    ) -> None:
        """`/regen N [mode]`: serialize an N-way regen with optional mode.

        Phase 1's engine already serializes via ``session.generate(...,
        n=N)``; phase 5 routes the optional ``mode`` argument through
        ``session.regen_with_modifier``.  ``mode`` accepts the built-in
        strings or a :class:`Recipe` partial for the custom-mode path.
        When a gen is already running we defer through
        the pending queue like every other interrupting slash command.
        """
        app = self._app
        if app._is_busy:
            mode_tag = f" {mode}" if isinstance(mode, str) else ""
            app._enqueue_pending(
                PendingItem("regen_n", f"/regen {n}{mode_tag}", (n, mode))
            )
            return
        if mode is not None:
            app._run_regen_modifier_worker(n, mode)
            return
        app._run_regen_n_worker(n)

    def _run_regen_modifier_worker(
        self, n: int, mode: "str | Recipe",
    ) -> None:
        """Worker for `/regen N <mode>` — routes through ``regen_with_modifier``."""
        app = self._app
        chat = app._chat_panel
        tree = app._session.tree
        active = tree.nodes.get(tree.active_node_id)
        if active is None:
            chat.add_system_message("/regen: no active node to regen from.")
            return
        if active.role == "assistant":
            user_parent_id = active.parent_id
        elif active.role == "user":
            user_parent_id = active.id
        else:
            chat.add_system_message("/regen: active node is not part of a turn.")
            return
        if user_parent_id is None:
            chat.add_system_message("/regen: no user-parent to anchor regen under.")
            return

        rendered = self._render_auto_regen_mode(mode)

        def _worker() -> None:
            try:
                app._session.regen_with_modifier(
                    user_parent_id, mode, n=n,
                )
                if app._raw_mode:
                    app.call_from_thread(app._sync_raw_buffer_from_tree)
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                app.call_from_thread(
                    app._chat_panel.add_system_message,
                    f"/regen {rendered} error: {msg}",
                )
            finally:
                app.call_from_thread(
                    app._chat_panel.add_system_message,
                    f"/regen × {n} ({rendered}): done.",
                )

        app.run_worker(_worker, thread=True)

    def _run_regen_n_worker(self, n: int) -> None:
        app = self._app
        chat = app._chat_panel
        # Read prompt off the active path.
        hist = app._session.history
        prompt = None
        if hist and hist[-1]["role"] == "user":
            prompt = hist[-1]["content"]
        elif hist:
            # Walk back to the last user turn.
            for msg in reversed(hist):
                if msg["role"] == "user":
                    prompt = msg["content"]
                    break
        if not prompt:
            chat.add_system_message("/regen: no user turn to regenerate.")
            return

        # Move active to the user-parent so siblings attach correctly.
        app._rewind_active_assistant()

        # After the rewind, active is a user node — passing
        # ``parent_node_id=<user.parent_id>`` lets add_user_turn's dedup
        # land the regen as a sibling under the existing user turn
        # (avoiding D15's user-under-user reject).
        tree = app._session.tree
        active_node = tree.nodes.get(tree.active_node_id)
        regen_parent_id: str | None = None
        if (active_node is not None and active_node.role == "user"
                and active_node.parent_id is not None):
            regen_parent_id = active_node.parent_id

        def _worker() -> None:
            try:
                sampling = SamplingConfig(
                    temperature=app._session.config.temperature,
                    top_p=app._session.config.top_p,
                    max_tokens=app._session.config.max_new_tokens,
                    seed=app._default_seed,
                )
                steering = (
                    None if not app._active_alphas()
                    else Steering(
                        alphas=dict(app._active_alphas()),
                        thinking=app._thinking,
                    )
                )
                app._session.generate(
                    prompt,
                    steering=steering,
                    sampling=sampling,
                    stateless=False,
                    n=n,
                    raw=app._raw_mode,
                    parent_node_id=regen_parent_id,
                )
                if app._raw_mode:
                    app.call_from_thread(app._sync_raw_buffer_from_tree)
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                app.call_from_thread(
                    app._chat_panel.add_system_message,
                    f"/regen error: {msg}",
                )
            finally:
                app.call_from_thread(
                    app._chat_panel.add_system_message,
                    f"/regen × {n}: done.",
                )

        app.run_worker(_worker, thread=True)

    def _handle_prune(self, arg: str) -> None:
        """`/prune <filter-expr>` — set the loom-screen filter highlight.

        The grammar is :func:`saklas.core.tree_filter.parse_filter`
        (``agg:`` / ``any:`` / ``last:`` prefix on per-node probe
        aggregates; clauses combined with ``,`` are AND).  Empty arg
        clears.  Validation happens through ``parse_filter`` so users
        get a single ``FilterParseError`` message before the loom
        screen tries to apply the filter.
        """
        from saklas.core.tree_filter import parse_filter, FilterParseError

        app = self._app
        chat = app._chat_panel
        expr = arg.strip()
        if not expr:
            self._loom_prune_expr = None
            chat.add_system_message("/prune cleared.")
            return
        try:
            parse_filter(expr)
        except FilterParseError as e:
            chat.add_system_message(f"/prune parse error: {e}")
            return
        self._loom_prune_expr = expr
        try:
            matching = app._session.tree.filter_by_expr(expr)
        except FilterParseError as e:
            chat.add_system_message(f"/prune evaluation error: {e}")
            return
        chat.add_system_message(
            f"/prune active: {expr}  ({len(matching)} node(s) match)"
        )

    @staticmethod
    def _render_auto_regen_mode(mode: "str | Recipe") -> str:
        """Display string for either a named-mode string or a custom Recipe.

        Recipe partials render as ``"custom"`` for footer / status use;
        the full canonical steering expression rides on the Recipe and
        is reported in the chat-message on configure / fire.
        """
        if isinstance(mode, Recipe):
            return "custom"
        return mode

    def _parse_custom_auto_regen(self, raw: str) -> "Recipe | None":
        """Parse ``custom: <expr>`` into a Recipe partial.

        ``<expr>`` is a steering expression (the shared grammar from
        ``saklas.core.steering_expr``).  Returns a Recipe carrying the
        canonical-form steering string; other Recipe fields (sampling,
        thinking, seed) inherit from the parent on overlay.  Returns
        ``None`` and posts an error to chat on parse failure — callers
        treat ``None`` as "don't update mode."
        """
        from saklas.core.steering_expr import (
            SteeringExprError, format_expr, parse_expr,
        )
        prefix = raw[len("custom:"):].strip()
        chat = self._app._chat_panel
        if not prefix:
            chat.add_system_message(
                "/auto-regen: custom mode needs an expression. "
                "Example: /auto-regen custom: 0.3 honest + 0.5 calm"
            )
            return None
        try:
            parsed = parse_expr(prefix)
        except (SteeringExprError, AmbiguousSelectorError, SaklasError) as e:
            chat.add_system_message(f"/auto-regen custom parse error: {e}")
            return None
        # Round-trip through format_expr so the canonical text is what
        # rides on the Recipe — round-trip-safe with replay.
        return Recipe(steering=format_expr(parsed))

    def _handle_auto_regen(self, arg: str) -> None:
        """`/auto-regen [on|off|<mode>]` — configure the regen modifier.

        Phase 5 wiring: ``on`` / ``off`` toggle whether every primary
        gen fires a sibling auto-regen; ``<mode>`` sets the override.
        Built-in modes (``unsteered`` / ``inverted`` / ``reseed`` /
        ``cool`` / ``hot``) stash as strings; ``custom: <expression>``
        parses the steering expression and stashes a :class:`Recipe`
        partial.  Bare ``/auto-regen`` reports the current state.
        ``Ctrl+A`` toggles on/off via :meth:`action_ab_compare` (the
        keymap meaning is preserved per plan decision 13).
        """

        chat = self._app._chat_panel
        arg = arg.strip()
        rendered = self._render_auto_regen_mode(self._loom_auto_regen_mode)
        if not arg:
            state = "on" if self._loom_auto_regen_on else "off"
            chat.add_system_message(
                f"auto-regen: {state}, mode={rendered}"
            )
            return
        low = arg.lower()
        if low == "on":
            self._loom_auto_regen_on = True
            chat.add_system_message(
                f"auto-regen on (mode={rendered})"
            )
            return
        if low == "off":
            self._loom_auto_regen_on = False
            chat.add_system_message("auto-regen off")
            return
        # Custom mode: parse the steering expression into a Recipe and
        # stash typed so the engine's compose_modifier(Recipe) path
        # picks it up without re-parsing.
        if low.startswith("custom:"):
            recipe = self._parse_custom_auto_regen(arg)
            if recipe is None:
                return  # error already posted
            self._loom_auto_regen_mode = recipe
            chat.add_system_message(
                f"auto-regen mode set to: custom ({recipe.steering})"
                + (" (auto-regen is currently off — /auto-regen on to enable)"
                   if not self._loom_auto_regen_on else "")
            )
            return
        # Built-in named mode.
        if low not in self._AUTO_REGEN_MODES:
            chat.add_system_message(
                "/auto-regen: unknown mode. Valid: "
                + ", ".join(self._AUTO_REGEN_MODES)
                + " or 'custom: <steering expression>'"
            )
            return
        self._loom_auto_regen_mode = low
        chat.add_system_message(
            f"auto-regen mode set to: {low}"
            + (" (auto-regen is currently off — /auto-regen on to enable)"
               if not self._loom_auto_regen_on else "")
        )

    def _handle_diff(self, arg: str) -> None:
        """`/diff <id1> <id2> [--full]` / `/diff --siblings` (phase 5).

        Resolves each id by ulid prefix via :func:`resolve_node_prefix`,
        calls :meth:`SaklasSession.diff_nodes`, and prints a compact
        unified text-diff plus the top-5 reading deltas (signed-colored
        via Rich markup).  ``--full`` extends the readings table to
        every entry; ``--siblings`` walks the active user-parent's
        assistant children and prints a pairwise matrix.
        """
        from saklas.tui.loom_helpers import resolve_node_prefix

        app = self._app
        chat = app._chat_panel
        tokens = shlex.split(arg) if arg else []
        if not tokens:
            chat.add_system_message(
                "Usage: /diff <id1> <id2> [--full]  |  /diff --siblings"
            )
            return

        full = "--full" in tokens
        ids = [t for t in tokens if not t.startswith("--")]

        if "--siblings" in tokens:
            self._handle_diff_siblings(full=full)
            return

        if len(ids) != 2:
            chat.add_system_message(
                "Usage: /diff <id1> <id2> [--full]  |  /diff --siblings"
            )
            return

        m1 = resolve_node_prefix(app._session.tree, ids[0])
        m2 = resolve_node_prefix(app._session.tree, ids[1])
        for label, m, raw in (("id1", m1, ids[0]), ("id2", m2, ids[1])):
            if m.missing:
                chat.add_system_message(f"/diff: {label}: no node matches '{raw}'")
                return
            if m.ambiguous:
                cands = ", ".join(c[:12] for c in m.candidates[:8])
                chat.add_system_message(
                    f"/diff: {label}: ambiguous '{raw}': {cands}"
                )
                return

        # node_id is non-None here: the loop above verified neither match is missing/ambiguous,
        # which by PrefixMatch's invariant means both node_ids were assigned str hits.
        assert m1.node_id is not None and m2.node_id is not None  # noqa: S101
        try:
            diff = app._session.diff_nodes(m1.node_id, m2.node_id)
        except Exception as e:
            chat.add_system_message(f"/diff failed: {e}")
            return

        chat.add_system_message(self._render_node_diff(diff, full=full))

    def _render_node_diff(self, diff: Any, *, full: bool) -> str:
        """Format a :class:`NodeDiff` for the chat panel.

        Unified-diff prose (cheap on terminal width) plus top-5 readings
        deltas, signed-colored via Rich markup.  ``full=True`` extends
        the readings table to every entry.
        """
        a8 = diff.a_id[:8]
        b8 = diff.b_id[:8]
        lines: list[str] = []
        lines.append(f"=== diff: {a8} vs {b8} ===")
        if diff.parent_id is not None:
            lines.append(f"  shared parent: {diff.parent_id[:8]}")
        else:
            lines.append("  (no shared parent — cross-branch comparison)")

        lines.append("")
        lines.append("--- text (unified, word-level) ---")
        if not diff.text:
            lines.append("(no text)")
        else:
            for span in diff.text:
                if span.state == "equal":
                    lines.append(f"  {span.text}")
                elif span.state == "delete":
                    lines.append(f"[red]- {span.text}[/red]")
                else:  # insert
                    lines.append(f"[green]+ {span.text}[/green]")

        lines.append("")
        cap = None if full else 5
        cap_label = "" if full else f"top {min(5, len(diff.readings))} of "
        lines.append(
            f"--- readings Δ (b - a, {cap_label}{len(diff.readings)}) ---"
        )
        if not diff.readings:
            lines.append("(no readings)")
        else:
            for r in diff.readings[: (cap if cap is not None else len(diff.readings))]:
                color = "green" if r.delta > 0 else ("red" if r.delta < 0 else "dim")
                lines.append(
                    f"  [{color}]{r.delta:+.4f}[/{color}]  "
                    f"{r.name:<28}  ({r.a_value:+.3f} → {r.b_value:+.3f})"
                )
        return "\n".join(lines)

    def _handle_diff_siblings(self, *, full: bool) -> None:
        """`/diff --siblings` — diff every assistant sibling of the active
        user-parent.

        Two siblings → one pairwise diff (same as `/diff a b`).  Three or
        more → a small per-pair top-1 reading-delta matrix.
        """
        app = self._app
        chat = app._chat_panel
        tree = app._session.tree
        # Find the active node's user-parent.
        active = tree.nodes.get(tree.active_node_id)
        if active is None:
            chat.add_system_message("/diff --siblings: no active node.")
            return
        user_parent_id: str | None = None
        if active.role == "user":
            user_parent_id = active.id
        elif active.role == "assistant" and active.parent_id is not None:
            user_parent_id = active.parent_id
        if user_parent_id is None:
            chat.add_system_message(
                "/diff --siblings: active node has no user-parent to "
                "compare children under."
            )
            return

        sibs = [
            cid for cid in tree.child_ids(user_parent_id)
            if tree.get(cid).role == "assistant"
        ]
        if len(sibs) < 2:
            chat.add_system_message(
                "/diff --siblings: need ≥2 assistant siblings under the "
                "active user-parent (have "
                f"{len(sibs)})."
            )
            return

        if len(sibs) == 2:
            try:
                diff = app._session.diff_nodes(sibs[0], sibs[1])
            except Exception as e:
                chat.add_system_message(f"/diff --siblings failed: {e}")
                return
            chat.add_system_message(self._render_node_diff(diff, full=full))
            return

        # ≥3 siblings: print a top-1 pairwise matrix.  Avoids dumping
        # full diffs N²-style; users follow up with `/diff a b` for the
        # full text + readings on any pair that looks interesting.
        lines: list[str] = [
            f"=== sibling matrix ({len(sibs)} children of {user_parent_id[:8]}) ===",
            "  pair                top Δ reading",
        ]
        for i in range(len(sibs)):
            for j in range(i + 1, len(sibs)):
                a_id, b_id = sibs[i], sibs[j]
                try:
                    diff = app._session.diff_nodes(a_id, b_id)
                except Exception as e:
                    lines.append(f"  {a_id[:8]} ↔ {b_id[:8]}  (error: {e})")
                    continue
                if diff.readings:
                    top = diff.readings[0]
                    color = "green" if top.delta > 0 else ("red" if top.delta < 0 else "dim")
                    lines.append(
                        f"  {a_id[:8]} ↔ {b_id[:8]}   "
                        f"[{color}]{top.delta:+.4f}[/{color}]  {top.name}"
                    )
                else:
                    lines.append(
                        f"  {a_id[:8]} ↔ {b_id[:8]}   (no readings)"
                    )
        chat.add_system_message("\n".join(lines))
