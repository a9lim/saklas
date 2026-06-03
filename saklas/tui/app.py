"""Main Textual application for saklas."""

from __future__ import annotations

import math
import queue
import shlex
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TYPE_CHECKING, overload

import torch
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.timer import Timer
from textual import events as _textual_events

from saklas import Recipe, SamplingConfig, Steering
from saklas.io.selectors import AmbiguousSelectorError, resolve_pole
from saklas.core.errors import SaklasError
from saklas.core.generation import supports_thinking, thinking_is_optional
from saklas.io.paths import saklas_home
from saklas.io.probes_bootstrap import load_default_manifolds as load_defaults
from saklas.core.results import ResultCollector
from saklas.core.session import MIN_ELAPSED_FOR_RATE
from saklas.tui.chat_panel import (
    ACTIVE_AT_DRAIN,
    SURPRISE_PROBE,
    _KIND_CHAIN_INLINE,
    _KIND_ENDS_ON_USER_NODE,
    ChatInput,
    ChatPanel,
    PendingItem,
    RawBuffer,
    _AssistantMessage,
    _RawTextArea,
    _TurnRow,
)
from saklas.tui.vector_panel import LeftPanel, MAX_ALPHA
from saklas.tui.trait_panel import TraitPanel

if TYPE_CHECKING:
    from saklas.core.session import SaklasSession

DEFAULT_ALPHA = 0.5
_POLL_FPS = 15
_TOKEN_DRAIN_LIMIT = 20

_LEFT, _CHAT, _TRAIT = 0, 1, 2

_BIPOLAR_DELIM = " . "

# Step sizes for ←/→ alpha adjustment.  Plain arrow nudges by
# ``_ALPHA_STEP_FINE``; shift+arrow uses the coarse step.  Both clamp via
# ``MAX_ALPHA`` inside ``_adjust_alpha``.
_ALPHA_STEP_FINE = 0.01
_ALPHA_STEP_COARSE = 0.1

# Cap on shell-style input history (↑/↓ in the chat input).  Bounded to
# keep ``_input_history`` from growing without limit over a long session
# — same order of magnitude as readline's default ``HISTSIZE``.
_INPUT_HISTORY_MAX = 200


def _detect_namespace_selector(text: str) -> str | None:
    """Return the namespace name when ``text`` is a bulk selector.

    A bulk selector is a single ``<ns>/`` token — namespace name followed
    by a trailing slash, no concept name, no whitespace, no other path
    components. ``alice/`` matches; ``alice/foo``, ``foo/bar/`` and
    ``0.5 alice/`` do not. Returns ``None`` for non-matches so the caller
    falls through to the per-concept parser path.
    """
    text = text.strip()
    if not text.endswith("/"):
        return None
    body = text[:-1]
    if not body or "/" in body:
        return None
    # Reuse the canonical name regex so the namespace token has to look
    # like an installable name (no spaces, no funky chars).
    from saklas.io.packs import NAME_REGEX
    if not NAME_REGEX.match(body):
        return None
    return body


def _unquote(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _split_bipolar(text: str) -> tuple[str, str | None]:
    """Split ``pos . neg`` on the first surrounded-by-whitespace period.

    Whitespace around the period is required, so canonical single-token
    names like ``dog.cat`` aren't split. Quotes are stripped from each
    side if the user wrapped them.
    """
    idx = text.find(_BIPOLAR_DELIM)
    if idx >= 0:
        return (
            _unquote(text[:idx].strip()),
            _unquote(text[idx + len(_BIPOLAR_DELIM):].strip()),
        )
    return _unquote(text.strip()), None


def _resolve_active_name(name: str, active: "Iterable[str]") -> list[str]:
    """Resolve a user-typed name against a set of currently-active names.

    Direct hit returns a single-element list. Otherwise treats ``name``
    as a pole and scans ``active`` for any canonical entry where the
    slug appears on either side of the ``.`` separator. Returns all
    matches (caller handles 0 / 1 / many).
    """
    from saklas.core.session import BIPOLAR_SEP, canonical_concept_name

    active = list(active)
    if name in active:
        return [name]
    slug = canonical_concept_name(name)
    matches: list[str] = []
    for key in active:
        if key == slug:
            matches.append(key)
            continue
        if BIPOLAR_SEP in key:
            pos, neg = key.split(BIPOLAR_SEP, 1)
            if pos == slug or neg == slug:
                matches.append(key)
    return matches


class SaklasApp(App[None]):
    CSS_PATH = "styles.tcss"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("backspace", "remove_vector", "Remove", show=False),
        Binding("delete", "remove_vector", "Remove", show=False),
        # ``priority=True`` on every Ctrl+letter app shortcut that
        # collides with the multi-line ``ChatInput`` (TextArea) defaults:
        # without it, the focused input would steal the key for its own
        # editor binding.  Concretely TextArea binds ``ctrl+a`` →
        # line-start, ``ctrl+c`` → copy, ``ctrl+d`` → delete-right,
        # ``ctrl+e`` → line-end, ``ctrl+y`` → redo — all of which would
        # otherwise hijack the corresponding app shortcuts while the
        # chat input is focused.  Non-colliding TextArea editor bindings
        # (``ctrl+w``/``ctrl+u``/``ctrl+k``/``ctrl+v``/``ctrl+x``/
        # ``ctrl+z``) keep working — those are useful editor verbs and
        # nothing at the app layer wants them.
        Binding("ctrl+a", "ab_compare", "A/B", show=False, priority=True),
        Binding("escape", "stop_generation", "Stop", show=False),
        Binding("ctrl+r", "regenerate", "Regen", show=False),
        Binding("ctrl+c", "copy_selection", "Copy", show=False, priority=True),
        Binding("ctrl+t", "toggle_thinking", "Think", show=False),
        Binding("ctrl+s", "cycle_sort", "Sort", show=False),
        # Highlight: a single three-state cycle {off → probe → surprise}
        # on Ctrl+Y, with Ctrl+Shift+Y walking it backward.  Ctrl+H is
        # *not* an option — terminals send 0x08 for it, which Textual
        # hard-maps to ``backspace`` before binding resolution, so a
        # ``ctrl+h`` binding can never fire.  Ctrl+Shift+Y degrades
        # gracefully to a forward step on terminals that can't report
        # the shift, vs. Ctrl+Shift+H degrading to backspace.
        Binding("ctrl+y", "cycle_highlight_mode", "HL cycle", show=False, priority=True),
        Binding(
            "ctrl+shift+y", "cycle_highlight_mode_back",
            "HL cycle back", show=False,
        ),
        Binding("ctrl+l", "open_loom", "Loom", show=False),
        # Ctrl+O toggles the chat/raw render surface.  Not Ctrl+M — that
        # transmits 0x0D (CR), which Textual maps to ``enter`` before
        # binding resolution, so a ``ctrl+m`` binding collides with the
        # Enter key (same gotcha as Ctrl+H → backspace).
        Binding("ctrl+o", "toggle_render_mode", "Mode", show=False),
        Binding("ctrl+e", "edit_active", "Edit", show=False, priority=True),
        Binding("ctrl+b", "branch_active", "Branch", show=False),
        Binding("ctrl+n", "nav_picker", "Nav", show=False),
        Binding("ctrl+d", "delete_subtree", "Del", show=False, priority=True),
        # Commit (no-generation send): Ctrl+Enter is the canonical binding
        # — Alt+Enter is the cross-terminal fallback for terminals that
        # don't pass Ctrl+Enter through (legacy stacks without the
        # CSI-u / kitty keyboard protocol collapse Ctrl+Enter to bare
        # Enter).  Priority so the Input widget can't swallow them when
        # the chat input is focused.
        Binding("ctrl+enter", "commit_text", "Commit", show=False, priority=True),
        Binding("alt+enter", "commit_text", "Commit", show=False, priority=True),
        Binding("[", "temp_down", show=False),
        Binding("]", "temp_up", show=False),
        Binding("{", "top_p_down", show=False),
        Binding("}", "top_p_up", show=False),
    ]

    def __init__(
        self,
        session: "SaklasSession",
        **kwargs: Any,
    ) -> None:
        super().__init__(ansi_color=True, **kwargs)
        self._session = session
        # ``_messages`` was a direct reference to ``session._history`` in
        # v2.2 — a shared mutable list.  Under v2.3 the conversation lives
        # in :class:`~saklas.core.loom.LoomTree`; ``session.history`` is a
        # derived view.  We expose the same shape (``list[dict]``) as a
        # property; the four pop-sites that mutated it in v2.2 now route
        # through :meth:`_rewind_active_assistant` so the tree stays
        # consistent with the visible state.
        self._device_str = str(session._device)

        # Local steering state — alphas and enabled flags per vector.
        # Session holds the profiles; the TUI holds the alphas.
        self._alphas: dict[str, float] = {}
        self._enabled: dict[str, bool] = {}
        # Manifold steering terms, keyed by the grammar's manifold key
        # (``<manifold>%<coords>``).  A ``ManifoldTerm`` is not a
        # :class:`Profile`, so it can't live in ``_alphas`` (typed
        # ``dict[str, float]``) — this parallel dict carries the term
        # values and is merged into the built ``Steering.alphas`` at
        # generation time (``AlphaEntry`` already admits ``ManifoldTerm``,
        # so the engine needs no change).  ``_enabled`` is shared — a
        # manifold key toggles through the same left-panel control.
        self._manifold_terms: dict[str, Any] = {}
        # Base (non-chat) model detection — a base model has no chat
        # template, so role headers / thinking blocks make no sense.  When
        # base, the chat panel renders flat continuous text and thinking
        # UI is force-suppressed.  Prefill stays template-agnostic and is
        # untouched.
        self._is_base_model: bool = session.is_base_model
        self._supports_thinking: bool = (
            False if self._is_base_model
            else supports_thinking(session._tokenizer)
        )
        # Forced-thinking families (gpt-oss / Mistral-3 Reasoning /
        # Qwen3-Thinking) have no ``enable_thinking`` template switch,
        # so ``thinking=False`` is a no-op at the prompt layer.  We
        # surface this in the vector panel and the ⌃T handler.
        self._thinking_optional: bool = thinking_is_optional(session._tokenizer)
        # Thinking defaults to off on both surfaces — the explicit
        # binary toggle (``⌃T``) is the user's affirmative opt-in, matching
        # the GUI's ``samplingState.thinking = false`` default.  Previous
        # behaviour ("default on whenever supported") silently consumed
        # extra tokens for thinking traces the user hadn't asked for.
        # For forced-thinking models the value is locked True since the
        # toggle cannot reach the prompt anyway.
        self._thinking: bool = (
            self._supports_thinking and not self._thinking_optional
        )
        # Render mode — ``"chat"`` (turn rows) or ``"raw"`` (a single
        # flat completion buffer).  Seeded from the model's nature: a
        # base model defaults to raw, a chat model to chat.  Toggled at
        # runtime via ``/mode`` — neither choice is locked to the model.
        self._render_mode: str = "raw" if self._is_base_model else "chat"

        self._current_assistant_widget = None
        self._poll_timer: Timer | None = None
        self._last_prompt: str | None = None

        # Shell-style input history.  ``_input_history`` is the ring of
        # submitted lines (slash commands and chat messages alike, because
        # both flow through ``ChatPanel.UserSubmitted``); ``_history_index``
        # is the cursor into it (``None`` = at the live "current" slot, no
        # recall in flight); ``_history_stash`` saves whatever the user was
        # typing the moment they first pressed ↑ so ↓-past-the-end restores
        # it.  Capped at ``_INPUT_HISTORY_MAX`` to keep the list bounded
        # over a long session.
        self._input_history: list[str] = []
        self._history_index: int | None = None
        self._history_stash: str = ""
        # Pending-queue cursor (separate from ``_history_index`` so the
        # two rings can interleave at the natural ↑-walks-newest-first
        # boundary).  ``None`` = no pending slot is pulled; ``int`` =
        # index into ``_pending_queue`` of the slot currently mirrored
        # into the input box.  When the user re-submits while pulled
        # the new item *replaces* slot ``_pulled_slot`` (preserves
        # order); an empty re-submit *removes* the slot (cancel).
        self._pulled_slot: int | None = None
        # ``_ab_mode`` is the persistent two-column-layout toggle (Ctrl+A);
        # ``_ab_shadow_active`` is the transient flag set while a shadow
        # gen worker is streaming — gates panels/highlight/probe-rack
        # mutations the same way the v1 one-shot ``_ab_in_progress`` did,
        # so users can't fiddle with steering while the unsteered shadow
        # is in flight.  The two flags are independent: AB mode can be on
        # without an active shadow and vice versa (e.g. shadow runs to
        # completion even after the user toggles AB off).
        self._ab_mode: bool = False
        self._ab_shadow_active: bool = False
        # The turn-row whose shadow column is being streamed into right
        # now (``_current_assistant_widget`` lives inside it during a
        # shadow gen).  Cleared on shadow ``done``.
        self._ab_shadow_row: _TurnRow | None = None
        # Map id(widget) → owning row, so when a steered ``done`` lands
        # we can locate the row to fire its shadow into without a tree
        # walk per gen.
        self._row_for_widget: dict[int, _TurnRow] = {}
        # Mid-generation submissions queue here rather than tearing the
        # in-flight stream down.  Drained one item per ``done`` event in
        # :meth:`_poll_generation` via :meth:`_drain_next_pending`; the
        # ``PendingStrip`` widget in the chat panel renders the live
        # list so the user can see what's queued and pull items back
        # for edit with ``↑``.  Replaces the v2.3 single-slot
        # ``_pending_action: tuple | None`` — the old behavior was to
        # call ``session.stop()`` and stash a tuple, which lost
        # un-emitted tokens; the queue model keeps the current gen
        # running and serializes the new work behind it.
        self._pending_queue: list[PendingItem] = []
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
        # UI-side gen flag.  Tracks the *TUI's* gen lifecycle, which differs
        # slightly from the session's: the TUI counts a gen as "still going"
        # until the ``("done",)`` sentinel lands on the local ``_ui_token_queue``
        # (see ``_poll_generation``), even after the session has already
        # returned to ``GenState.IDLE``.  Use ``self._session.is_generating``
        # for "is the engine running right now?" — this flag is for UI-only
        # gating (e.g. Ctrl+R, pending-action dispatch) tied to the queue
        # drain, never to gate any session call.
        self._ui_gen_active: bool = False

        self._focused_panel_idx: int = 1  # Start with chat focused

        # Highlight defaults to surprise mode on both surfaces — the
        # logit-pass tint works without any probes loaded, so it's the
        # one mode that's meaningful out of the box.  Mirrors the GUI's
        # ``highlightState.target = SURPRISE_TARGET`` default.
        self._highlighting: bool = True
        self._highlight_probe: str | None = SURPRISE_PROBE
        self._default_seed: int | None = None
        self._ui_token_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()

        self._gen_start_time: float = 0.0
        self._gen_token_count: int = 0
        self._last_tok_per_sec: float = 0.0
        self._last_elapsed: float = 0.0
        # Geometric-mean perplexity accumulator — sum of ``log(ppl)`` across
        # scored steps; display is ``exp(sum/count)``.
        self._log_ppl_sum: float = 0.0
        self._ppl_count: int = 0
        self._last_gen_state: tuple[Any, ...] = (-1, -1.0, -1.0, False, -1)
        self._assistant_messages: list[_AssistantMessage] = []

        defaults = load_defaults()
        self._probe_categories: dict[str, list[str]] = {
            cat.capitalize(): probes_list
            for cat, probes_list in defaults.items()
        }

    @property
    def _messages(self) -> list[dict[str, str]]:
        """Compat shim — derived view over the loom tree's active path.

        v2.2 had ``self._messages = session._history`` as a shared list
        reference; v2.3's tree-backed history returns a fresh list per
        access.  Reads that took ``[-1]`` or truthiness checks work
        unchanged; the four pop-sites in regen/rewind paths now route
        through :meth:`_rewind_active_assistant` instead of mutating
        this property's return.
        """
        return self._session.history

    def _rewind_active_assistant(self) -> bool:
        """Move the loom tree's active pointer up one assistant turn.

        Returns ``True`` when the active node was an assistant and was
        rewound; ``False`` when there's nothing to rewind.  Non-
        destructive — the rewound assistant stays in the tree as a now-
        dead branch, navigable via the loom sidebar / screen.  Replaces
        v2.2's ``self._messages.pop()`` of a trailing assistant turn at
        regen / rewind sites.
        """
        tree = self._session.tree
        active = tree.nodes.get(tree.active_node_id)
        if active is None or active.role != "assistant":
            return False
        if active.parent_id is None:
            return False
        tree.navigate(active.parent_id)
        return True

    def _active_alphas(self) -> dict[str, Any]:
        """Build the alphas dict for generation from enabled entries.

        Merges plain scalar vectors (``_alphas``) with manifold terms
        (``_manifold_terms``); both share the ``_enabled`` flag map so a
        manifold row toggles through the same left-panel control.  The
        returned dict's values are ``float`` for vectors and
        :class:`~saklas.core.steering_expr.ManifoldTerm` for manifolds —
        ``Steering.alphas``'s ``AlphaEntry`` union admits both.
        """
        out: dict[str, Any] = {
            name: alpha for name, alpha in self._alphas.items()
            if self._enabled.get(name, True)
        }
        for key, term in self._manifold_terms.items():
            if self._enabled.get(key, True):
                out[key] = term
        return out

    def _vector_list_for_panel(self) -> list[dict[str, Any]]:
        """Build the list[dict] format the left panel expects.

        Each entry carries a ``kind`` discriminator: ``"vector"`` rows
        carry an alpha bar; ``"manifold"`` rows carry a fixed position
        and a blend coefficient instead (no scalar alpha to nudge).
        """
        result = []
        for name, alpha in self._alphas.items():
            profile = self._session._profiles.get(name)
            if profile is None:
                continue
            # Bind ``profile`` to a default-arg so pyright keeps the
            # narrowed-non-None type across the lambda capture (it
            # otherwise widens to include None inside the closure).
            result.append({
                "kind": "vector",
                "name": name,
                "profile": profile,
                "alpha": alpha,
                "enabled": self._enabled.get(name, True),
                "peak": max(
                    profile,
                    key=lambda k, p=profile: float(p[k].norm().item()),
                ),
                "n_active": len(profile),
            })
        for key, term in self._manifold_terms.items():
            coords_str = ",".join(f"{c:g}" for c in term.position)
            result.append({
                "kind": "manifold",
                "name": key,
                "manifold": term.manifold,
                "coords": coords_str,
                "blend": term.coeff,
                "enabled": self._enabled.get(key, True),
            })
        return result

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            yield LeftPanel(self._session._model_info, id="left-panel")
            yield ChatPanel(id="chat-panel")
            yield TraitPanel(categories=self._probe_categories, id="trait-panel")

    def on_mount(self) -> None:
        self._left_panel = self.query_one("#left-panel", LeftPanel)
        self._chat_panel = self.query_one("#chat-panel", ChatPanel)
        self._trait_panel = self.query_one("#trait-panel", TraitPanel)
        self._panels = [self._left_panel, self._chat_panel, self._trait_panel]
        self._refresh_gen_config()
        # Seed the persistent HL readout from the initial highlight state
        # (surprise by default — see __init__).  No widgets are mounted
        # yet, so this is purely the panel-line refresh.
        self._apply_highlight_to_all()

        if self._session._monitor:
            self._trait_panel.set_active_probes(set(self._session._monitor.probe_names))
            self._refresh_trait_why()
        # Manifold-probe rack — empty by default, but call through the
        # set helper so any pre-attached probes (e.g. test stubs) show
        # up in the panel immediately.
        self._refresh_manifold_probes()

        self._poll_timer = self.set_interval(1 / _POLL_FPS, self._poll_generation)
        self._update_panel_focus()

        # Render mode — raw (flat completion buffer) for base models,
        # chat (turn rows) otherwise.  Set before the first turn mounts
        # so the surface is correct from the start.  Toggle with /mode.
        self._chat_panel.set_render_mode(self._render_mode)
        if self._render_mode == "raw":
            self._sync_raw_buffer_from_tree()

        loaded = (
            f"Model loaded: "
            f"{self._session._model_info.get('model_id', 'unknown')}. "
        )
        if self._is_base_model:
            loaded += (
                "Base (non-chat) model — raw completion mode. "
                "Edit the buffer and press Enter to continue. "
                "Use /steer and /probe; /mode switches surface."
            )
        else:
            loaded += (
                "Type a message to chat. Use /steer and /probe commands. "
                "Tab to switch panels."
            )
        self._chat_panel.add_system_message(loaded)

    # -- Key Handling --

    def on_key(self, event: _textual_events.Key) -> None:
        # The raw-mode completion buffer is a TextArea too — treat it
        # like the chat input so ``Tab`` still cycles panels and the
        # arrow keys stay with the editor (no panel-nav / alpha-nudge).
        if isinstance(self.focused, (ChatInput, _RawTextArea)):
            if event.key == "tab":
                event.prevent_default()
                event.stop()
                self.action_focus_next_panel()
            elif event.key == "shift+tab":
                event.prevent_default()
                event.stop()
                self.action_focus_prev_panel()
            elif event.key in ("up", "down") and isinstance(
                self.focused, ChatInput
            ):
                # Shell-style history recall on the chat input — but
                # *edge-only* so multi-line editing keeps its native
                # cursor nav.  ``↑`` only recalls when the cursor sits
                # on the first row; ``↓`` only when it's on the last
                # row.  Mid-buffer arrows fall through to TextArea's
                # ``cursor_up`` / ``cursor_down``.
                inp = self.focused
                at_edge = (
                    inp.cursor_at_first_line if event.key == "up"
                    else inp.cursor_at_last_line
                )
                if at_edge:
                    event.prevent_default()
                    event.stop()
                    self._history_navigate(-1 if event.key == "up" else +1)
            return

        key = event.key
        handled = True
        if key == "tab":
            self.action_focus_next_panel()
        elif key == "shift+tab":
            self.action_focus_prev_panel()
        elif key == "down":
            self.action_nav_down()
        elif key == "up":
            self.action_nav_up()
        elif key == "left":
            self.action_nav_left()
        elif key == "right":
            self.action_nav_right()
        elif key == "shift+left":
            self._adjust_alpha(-_ALPHA_STEP_COARSE)
        elif key == "shift+right":
            self._adjust_alpha(_ALPHA_STEP_COARSE)
        elif key == "enter":
            self.action_nav_enter()
        else:
            handled = False

        if handled:
            event.prevent_default()
            event.stop()

    # -- Panel Focus --

    def _update_panel_focus(self) -> None:
        for i, panel in enumerate(self._panels):
            if i == self._focused_panel_idx:
                panel.add_class("focused")
            else:
                panel.remove_class("focused")
        if self._focused_panel_idx == _CHAT:
            self.query_one("#chat-input").focus()
        else:
            self.set_focus(None)

    def action_focus_next_panel(self) -> None:
        self._focused_panel_idx = (self._focused_panel_idx + 1) % len(self._panels)
        self._update_panel_focus()

    def action_focus_prev_panel(self) -> None:
        self._focused_panel_idx = (self._focused_panel_idx - 1) % len(self._panels)
        self._update_panel_focus()

    # -- Navigation --

    def action_nav_down(self) -> None:
        if self._focused_panel_idx == _LEFT:
            self._left_panel.select_next()
        elif self._focused_panel_idx == _TRAIT:
            self._trait_panel.nav_down()
            if self._highlighting:
                self._apply_highlight_to_all()
            self._refresh_trait_why()

    def action_nav_up(self) -> None:
        if self._focused_panel_idx == _LEFT:
            self._left_panel.select_prev()
        elif self._focused_panel_idx == _TRAIT:
            self._trait_panel.nav_up()
            if self._highlighting:
                self._apply_highlight_to_all()
            self._refresh_trait_why()

    def action_nav_left(self) -> None:
        self._adjust_alpha(-_ALPHA_STEP_FINE)

    def action_nav_right(self) -> None:
        self._adjust_alpha(_ALPHA_STEP_FINE)

    def action_nav_enter(self) -> None:
        if self._focused_panel_idx == _LEFT:
            self.action_toggle_vector()

    # -- Chat --

    def on_chat_panel_user_submitted(self, event: ChatPanel.UserSubmitted) -> None:
        text = event.text
        # Slot-preserving edit: when the user pulled a queued item via
        # ↑/↓ the new submission lands at its original slot rather
        # than appending at the tail.  Empty Enter while pulled is
        # the cancel gesture — pull the slot via ↑, clear with
        # ``Ctrl+U``, hit ``Enter`` and the slot is removed.  The
        # symmetric "back out without removing" gesture is ``Esc``
        # (handled in ``action_stop_generation``).
        replace_slot = self._pulled_slot
        self._pulled_slot = None
        self._sync_pull_state()
        if not text:
            # Reached us only via the pulled-slot cancel path
            # (``ChatInput.allow_empty_submit=True``).  Remove the
            # slot and bail.
            if replace_slot is not None:
                self._remove_pending_slot(replace_slot)
            return
        # Push to ↑/↓ history before dispatch so a slash command that
        # errors mid-handler is still recallable.
        self._push_input_history(text)
        if text.startswith("/"):
            self._handle_command(text, replace_slot=replace_slot)
            return
        self._last_prompt = text
        if self._raw_mode:
            # Raw mode: a non-slash line from the command input is a
            # prompt appended to the buffer — continue from the active
            # leaf with the typed text as the divergence tail.
            draft = self._session.tree.flat_text() + text
            self._submit_raw_continuation(draft, replace_slot=replace_slot)
            return
        # Role-aware send: when the active loom node is a user turn the
        # input seeds the *assistant* reply (answer-prefill) rather than
        # appending a new user message.  Decide once, here — the pending
        # item carries the target so a deferred dispatch can't re-resolve
        # against a shifted active node.
        prefill_target = self._prefill_target_node_id()
        if self._is_busy:
            # Queue the message — it will be submitted once the current
            # generation finishes (see _poll_generation).  Don't mount
            # the user row up front; the deferred dispatch path takes
            # care of it via ``_start_generation`` → row-mount on first
            # token.  Mounting now would visually anchor the user turn
            # to the wrong assistant reply.
            #
            # Queue role-awareness: if the queue predicts a post-drain
            # user node (e.g. a queued ``commit_user``) but the live
            # active isn't one yet, the user is composing a prefill
            # against a not-yet-existing node.  Stamp ``ACTIVE_AT_DRAIN``
            # so the dispatcher resolves the parent fresh when the item
            # fires, by which point the earlier queue items have landed.
            queued_target: str | None
            if prefill_target is None and self._predicted_on_user_node():
                queued_target = ACTIVE_AT_DRAIN
            else:
                queued_target = prefill_target
            self._enqueue_pending(
                PendingItem("submit", text, (queued_target,)),
                replace_slot=replace_slot,
            )
            return
        if prefill_target is None:
            # Immediate send — mount the user row now (chat_panel no
            # longer mounts it; it can't know the active-node role).
            self._chat_panel.add_user_message(text)
        if prefill_target is not None:
            self._start_prefill(prefill_target, text)
            return
        self._start_generation(text)

    def _handle_command(self, text: str, *, replace_slot: int | None = None) -> None:
        from saklas.tui.commands import dispatch
        dispatch(self, text, replace_slot=replace_slot)

    # -- /sys, /temp, /top-p, /max, /help (registry-callable shims) --

    def _handle_sys(self, arg: str) -> None:
        chat = self._chat_panel
        if not arg:
            chat.add_system_message(
                f"System prompt: {self._session.config.system_prompt or '(none)'}"
            )
            return
        self._session.config = replace(self._session.config, system_prompt=arg)
        chat.add_system_message("System prompt set.")
        self._refresh_gen_config()

    def _set_config_value(
        self,
        attr: str,
        raw: str,
        *,
        cast: "Callable[[str], Any]",
        label: str,
    ) -> None:
        """Shared body for the scalar ``SamplingConfig`` handlers.

        Empty ``raw`` prints the current value; otherwise ``cast`` parses
        and validates it (a ``ValueError`` from ``cast`` is the
        "invalid" path), the session config is replaced, a confirmation
        is printed, and the generation-config panel refreshes.  ``label``
        is the human-readable name reused across all three messages —
        ``"<label>: <cur>"``, ``"Invalid <label-lower> value"``, and
        ``"<label> set to <val>"``.
        """
        chat = self._chat_panel
        if not raw:
            chat.add_system_message(
                f"{label}: {getattr(self._session.config, attr)}"
            )
            return
        try:
            val = cast(raw)
        except ValueError:
            chat.add_system_message(f"Invalid {label.lower()} value")
            return
        self._session.config = replace(self._session.config, **{attr: val})
        chat.add_system_message(f"{label} set to {val}")
        self._refresh_gen_config()

    def _handle_temp(self, arg: str) -> None:
        self._set_config_value(
            "temperature", arg,
            cast=lambda s: max(0.0, float(s)), label="Temperature",
        )

    def _handle_top_p(self, arg: str) -> None:
        self._set_config_value(
            "top_p", arg,
            cast=lambda s: max(0.0, min(1.0, float(s))), label="Top-p",
        )

    def _handle_max(self, arg: str) -> None:
        self._set_config_value(
            "max_new_tokens", arg,
            cast=lambda s: max(1, int(s)), label="Max tokens",
        )

    def _handle_help(self, _arg: str) -> None:
        self._chat_panel.add_system_message(
            "Steering:\n"
            "  /steer <concept> [alpha]    — add (extract if needed)\n"
            "  /steer <pos> . <neg> [a]    — add bipolar (period delim)\n"
            "  /steer <ns>/                — bulk add namespace (off)\n"
            "  /alpha <val> <name>         — adjust existing alpha\n"
            "  /unsteer <name|ns/>         — remove vector(s)\n"
            "Probes:\n"
            "  /probe <concept>            — add probe (highlight on)\n"
            "  /probe <ns>/                — bulk add namespace as probes\n"
            "  /unprobe <name|ns/>         — remove probe(s)\n"
            "  /extract <concept>          — cache-warm; --role <slug> for role-aug.\n"
            "  /pairs <name>               — custom pairs; --role <slug> opts in\n"
            "  /compare <a> [b]            — cosine similarity\n"
            "Manifold:\n"
            "  /steer <c> manifold%x,y     — steer toward a manifold point\n"
            "  /manifold fit <folder>      — fit a manifold pack folder\n"
            "  /manifold-probe <selector>  — attach a read-side manifold probe\n"
            "  /manifold-probe-remove <n>  — detach an attached probe\n"
            "Highlight:\n"
            "  ⌃Y / ⌃⇧Y                    — cycle {off → probe → surprise}\n"
            "Commit (no-gen send):\n"
            "  ⌃⏎ / ⌥⏎                     — modern terminals only\n"
            "  /commit <text>              — cross-terminal fallback\n"
            "Session:\n"
            "  /clear, /rewind, /regen     — history ops\n"
            "  /save <name>, /load <name>  — save / restore the loom tree\n"
            "  /export <path>              — JSONL w/ probe readings\n"
            "  /seed [n|clear]             — default sampling seed\n"
            "  /sys [prompt]               — system prompt\n"
            "  /temp, /top-p, /max         — sampling defaults\n"
            "  /model                      — model + session info\n"
            "  ⌃O                          — toggle chat ⇄ raw buffer\n"
            "  /exit, /help\n"
            "Loom:\n"
            "  /tree                       — open loom screen (⌃L)\n"
            "  /regen [N] [mode]           — N siblings; mode ∈ unsteered/\n"
            "                                inverted/reseed/cool/hot or\n"
            "                                'custom: <steering expr>'\n"
            "  /edit <text>                — in-place edit active\n"
            "  /branch [text]              — sibling w/ optional text\n"
            "  /nav <id-prefix>            — navigate by ulid\n"
            "  /del yes                    — drop subtree (confirm required)\n"
            "  /star, /note <text>         — decoration\n"
            "  /path                       — active path summary\n"
            "  /fan <vec> <alphas>         — canonical sweep (siblings)\n"
            "  /prune <filter-expr>        — dim non-matching nodes\n"
            "  /auto-regen [on|off|mode]   — sibling regen modifier (⌃A toggles)\n"
            "  /diff <id1> <id2> [--full]  — cross-branch text + readings diff\n"
            "  /diff --siblings            — diff active user-parent's kids\n"
            "  ⌃E/B/N/D open the loom screen; use /edit /branch /nav /del\n"
            "  for inline equivalents.\n"
            "Keys: ⇥ focus · ←/→ alpha (±0.01) · ⇧←/→ ±0.1\n"
            "↑/↓ nav (panels) · ↑/↓ in chat input recalls history\n"
            "⏎ toggle · ⌫ remove · ⌃T think · ⌃R regen\n"
            "⌃A A/B · ⌃S sort · ⌃Y highlight · ⌃L loom\n"
            "⌃E edit · ⌃B branch · ⌃N nav · ⌃D del\n"
            "[ ] temp · { } top-p · ⎋ stop · ⌃Q quit"
        )

    # -- Vector Management --

    def _on_vector_extracted(self, name: str, alpha: float,
                             profile: dict[int, torch.Tensor]) -> None:
        chat = self._chat_panel
        peak = max(profile, key=lambda k: float(profile[k].norm().item()))
        n_layers = len(profile)
        chat.add_system_message(
            f"Vector '{name}' active (α={alpha:+.1f}, {n_layers}L pk{peak})"
        )
        self._refresh_left_panel()

    @overload
    @staticmethod
    def _parse_args(
        text: str, include_alpha: Literal[True],
    ) -> tuple[str, str | None, float]: ...

    @overload
    @staticmethod
    def _parse_args(
        text: str, include_alpha: Literal[False] = False,
    ) -> tuple[str, str | None]: ...

    @staticmethod
    def _parse_args(
        text: str, include_alpha: bool = False,
    ) -> tuple[str, str | None] | tuple[str, str | None, float]:
        """Parse /steer, /probe, /extract arguments.

        Accepted forms:
            <concept> [alpha]              single concept; canonical
                                           forms like ``dog.cat`` pass
                                           through unchanged
            <pos> . <neg> [alpha]          bipolar (period delimiter)

        Multi-word poles don't need quotes (``a dog . a pair of cats``).
        Whitespace around the period is what makes it a delimiter — so
        ``dog.cat`` stays a single canonical name.
        """
        text = text.strip()
        alpha = None

        if include_alpha:
            # Peel a trailing float alpha if present. Scan from the right
            # over any runs of trailing non-float tokens — the historical
            # grammar allowed the alpha to sit before stray junk, but in
            # practice it's always last; we accept it there specifically.
            head, _, tail = text.rpartition(" ")
            if tail:
                try:
                    alpha = float(tail)
                    text = head.rstrip()
                except ValueError:
                    pass

        concept, baseline = _split_bipolar(text)

        if include_alpha:
            alpha = (max(-MAX_ALPHA, min(MAX_ALPHA, alpha))
                     if alpha is not None else DEFAULT_ALPHA)
            return concept, baseline, alpha
        return concept, baseline

    @staticmethod
    def _peel_role_flag(text: str) -> tuple[str, str | None]:
        """Peel a trailing ``--role <slug>`` off a slash-command argument
        string.  Returns ``(rest, role)`` — ``role`` is ``None`` when the
        flag is absent.

        Accepted forms:
            ``... --role pirate``
            ``... --role=pirate``

        Anywhere on the line is fine; first match wins, the flag and
        its value are excised, the rest is re-joined with a single
        space.  Slug validation is left to ``core.role_templates`` —
        we just route the raw string.  Multi-word slugs are not
        supported (the engine slug regex is ``[a-z0-9._-]+``).
        """
        import re

        m = re.search(r"\s*--role(?:=|\s+)(\S+)", text)
        if m is None:
            return text, None
        head = text[: m.start()].rstrip()
        tail = text[m.end():].lstrip()
        rest = (head + (" " + tail if tail else "")).strip()
        return rest, m.group(1)

    def _handle_extract(self, text: str, include_alpha: bool,
                        on_success: "Callable[..., Any]",
                        pending_type: str | None = None,
                        variant: str = "raw",
                        namespace: str | None = None) -> None:
        chat = self._chat_panel
        if self._ab_shadow_active:
            chat.add_system_message("Cannot modify vectors during A/B shadow gen.")
            return
        if pending_type is None:
            pending_type = "steer" if include_alpha else "probe"
        if self._is_busy:
            # Reconstruct the canonical slash-command form so pulling
            # the item back via ↑ surfaces something the user can
            # re-Enter as a slash command.  ``payload[0]`` carries the
            # raw args the dispatcher hands to the handler.
            display_text = f"/{pending_type} {text}".rstrip()
            self._enqueue_pending(
                PendingItem(pending_type, display_text, (text,))
            )
            return
        # Peel ``--role <slug>`` off the args before the bipolar parser
        # runs, so a multi-word pole (``a dog . a pair of cats``) doesn't
        # have to compete with the trailing flag for tokens.
        text, role = self._peel_role_flag(text)
        try:
            if include_alpha:
                concept, baseline, alpha = self._parse_args(text, include_alpha=True)
            else:
                concept, baseline = self._parse_args(text)
                alpha = None
        except (ValueError, IndexError) as e:
            chat.add_system_message(
                f"Parse error: {e}\n"
                f"Usage: /{pending_type} <pos> . <neg>"
                + (" [alpha]" if include_alpha else "")
                + " [--role <slug>]"
            )
            return

        # Alias resolution: a bare pole name may refer to an installed
        # bipolar pack (e.g. `/steer wolf` when `deer.wolf` exists).
        # Sign flip is applied to the user's alpha so `/steer calm 0.5`
        # on top of `angry.calm` lands as `alphas["angry.calm"] = -0.5`.
        # ``resolve_pole`` also peels any ``:variant`` suffix typed
        # directly on the concept (``honest:sae-gemma-scope...``) — that
        # explicit form wins over the ``variant`` kwarg when both are set.
        # Explicit bipolar form (`concept - baseline`) skips resolution
        # so the user's declared poles always win.
        # ``namespace`` (when set) scopes the resolution so ``alice/foo``
        # and ``bob/foo`` stay distinct.
        sign = 1
        if baseline is None:
            try:
                resolved_name, sign, _match, explicit_variant = resolve_pole(
                    concept, namespace=namespace,
                )
                if resolved_name != concept:
                    chat.add_system_message(
                        f"  Resolved '{concept}' → '{resolved_name}'"
                        + (" (negated)" if sign < 0 else "")
                    )
                concept = resolved_name
                # Explicit ``:sae-<release>`` on the concept overrides the
                # ``--sae`` preamble's variant. Lets users route a specific
                # release without the fuzzy release-detection heuristic.
                if explicit_variant != "raw":
                    variant = explicit_variant
            except AmbiguousSelectorError as e:
                chat.add_system_message(f"Error: {e.user_message()[1]}")
                return
            if alpha is not None:
                alpha *= sign

        # Variant routing. ``--sae`` alone (variant == "sae") means "pick the
        # unique already-extracted SAE tensor for this concept on disk" —
        # session autoload handles it. To drive a fresh extraction, users
        # pass the explicit ``:sae-<release>`` suffix, which routes the
        # release through ``session.extract(sae=RELEASE)``.  Same shape
        # for role-augmented extraction via the ``:role-<slug>`` variant
        # — the variant tail wins over an explicit ``--role`` flag (the
        # variant rode through ``resolve_pole``'s ``explicit_variant``
        # arm and is the canonical form).
        sae_release: str | None = None
        if variant.startswith("sae-"):
            sae_release = variant[len("sae-"):]
        if variant.startswith("role-"):
            role = variant[len("role-"):]

        display = concept if len(concept) <= 20 else concept[:17] + "..."
        suffix = f" vs '{baseline}'" if baseline else ""
        variant_note = f" [{variant}]" if variant != "raw" else ""
        role_note = f" as :role-{role}" if role else ""
        chat.add_system_message(
            f"Extracting '{display}'{suffix}{variant_note}{role_note}..."
        )

        def _work() -> None:
            def _progress(msg: str) -> None:
                self.call_from_thread(self._steer_status, msg)
            # Bare ``--sae`` (variant == "sae") routes the load through
            # the unified profile resolver (manifold-fold) rather than a
            # fresh extract — it means "use the SAE variant that's already
            # on disk". Ambiguous / missing cases surface via the session
            # errors / the None-check below.
            if variant == "sae" and sae_release is None:
                from contextlib import suppress as _suppress
                autoload_key = (
                    concept if namespace is None
                    else f"{namespace}/{concept}"
                )
                key = f"{autoload_key}:sae"
                with _suppress(Exception):
                    self._session._ensure_profile_registered(key)
                profile_dict = self._session._profiles.get(key)
                if profile_dict is None:
                    raise ValueError(
                        f"no SAE variant loaded for '{autoload_key}' — "
                        f"run `saklas subspace extract {autoload_key} --sae <RELEASE>` "
                        f"first, or pick a release with "
                        f"`:sae-<release>` in the concept name."
                    )
                on_success(key, profile_dict, alpha)
                return

            extract_kwargs = {"baseline": baseline, "on_progress": _progress}
            if sae_release is not None:
                extract_kwargs["sae"] = sae_release
            if namespace is not None:
                extract_kwargs["namespace"] = namespace
            if role is not None:
                extract_kwargs["role"] = role
            # ``session.extract`` already returns the fully-qualified
            # canonical name — including the ``:sae-<release>`` suffix
            # when ``sae=`` was passed. Rebuilding it here would
            # double-suffix the key and break every downstream
            # ``/alpha`` / ``/unsteer`` / pole lookup.
            canonical, profile = self._session.extract(concept, **extract_kwargs)
            if namespace is not None:
                # Re-attach the namespace so the registered key matches
                # what the parser produced (so ``/alpha`` / ``/unsteer``
                # against the namespace-qualified form keep working).
                if ":" in canonical:
                    bare, suffix = canonical.rsplit(":", 1)
                    canonical = f"{namespace}/{bare}:{suffix}"
                else:
                    canonical = f"{namespace}/{canonical}"
            on_success(canonical, profile, alpha)

        self._run_worker_with_queue(_work)

    def _handle_steer(self, text: str) -> None:
        """Apply a steering expression — the shared grammar from
        :mod:`saklas.core.steering_expr`.

        Each plain term (``<coeff> <concept>`` with optional ``@trigger``
        and ``:variant``) updates the TUI's local alpha state. Concepts
        not yet registered are extracted + steered behind the scenes.
        Extraction-on-demand for a *new* bipolar pair (``pos . neg``
        space-delimited) lives on ``/extract``, not ``/steer``; projection
        terms are accepted and routed through session-level materialization.
        """
        from saklas.core.steering_expr import (
            AblationTerm, ManifoldTerm, ProjectedTerm, SteeringExprError,
            parse_expr,
        )

        chat = self._chat_panel
        text = text.strip()
        if not text:
            chat.add_system_message(
                'Usage: /steer <expression>\n'
                '  e.g. /steer 0.5 honest\n'
                '       /steer 0.3 warm@after\n'
                '       /steer 0.5 honest|sycophantic\n'
                '       /steer alice/                (bulk; default-off)\n'
                "  For new concept extraction use /extract <pos> <neg>."
            )
            return
        ns = _detect_namespace_selector(text)
        if ns is not None:
            self._handle_steer_namespace(ns)
            return
        try:
            steering = parse_expr(text)
        except SteeringExprError as e:
            chat.add_system_message(f"Steering expression error: {e.user_message()[1]}")
            return
        except SaklasError as e:
            # ``parse_expr`` calls ``resolve_pole`` per term, which raises
            # ``AmbiguousSelectorError`` (a ``SelectorError(ValueError,
            # SaklasError)``) on cross-namespace bare-pole collisions —
            # not caught by the ``SteeringExprError`` arm above. Same for
            # ``AmbiguousVariantError`` from ``:sae`` resolution. Surface
            # cleanly instead of crashing the Textual worker.
            chat.add_system_message(f"Error: {e.user_message()[1]}")
            return

        # Iterate the parsed IR; for each term dispatch through the
        # existing extract pipeline to load or compute profiles, then
        # stash the effective alpha on the TUI's local state.
        for key, val in steering.alphas.items():
            if isinstance(val, ProjectedTerm):
                chat.add_system_message(
                    f"Projection terms aren't yet supported from /steer "
                    f"(got '{key}'); express them in the YAML config."
                )
                return
            if isinstance(val, AblationTerm):
                # Ablation terms (``!x``) are dispatched through a
                # separate hook path than additive steering; the TUI's
                # `/steer` surface only wires the additive side.  Reject
                # cleanly with the same pattern as `ProjectedTerm` above
                # rather than crash on ``float(AblationTerm)`` below.
                chat.add_system_message(
                    f"Ablation terms (!{key.lstrip('!')}) aren't supported "
                    f"from /steer; express them in the YAML config."
                )
                return
            if isinstance(val, ManifoldTerm):
                # Manifold steering (``manifold%coords``) routes through a
                # separate hook path — not bound to the per-vector alpha
                # sliders.  Dispatch through a worker that eager-loads and
                # validates the artifact, then stash it on
                # ``_manifold_terms`` (a ``ManifoldTerm`` is not a
                # ``Profile``, so it can't live in the scalar ``_alphas``).
                # ``continue`` rather than ``return`` so a mixed
                # expression still applies its vector siblings.
                self._dispatch_manifold_term(key, val)
                continue
            if isinstance(val, tuple):
                alpha, _trig = val
            else:
                alpha = float(val)
            alpha = max(-MAX_ALPHA, min(MAX_ALPHA, float(alpha)))
            # Peel variant suffix so the extract path sees a bare concept.
            if ":" in key:
                concept, variant = key.rsplit(":", 1)
            else:
                concept, variant = key, "raw"
            # Peel namespace prefix so ``_handle_extract`` -> ``resolve_pole``
            # scopes to the user's typed namespace instead of slugging
            # ``ns/name`` into a single token.  ``_handle_extract`` will
            # use ``namespace`` as the kwarg into ``session.extract`` so
            # disk discovery stays scoped too.
            namespace: str | None = None
            if "/" in concept:
                namespace, concept = concept.split("/", 1)
            self._dispatch_steer_term(
                concept, variant, alpha, namespace=namespace,
            )

    def _dispatch_steer_term(
        self, concept: str, variant: str, alpha: float,
        *, namespace: str | None = None,
    ) -> None:
        """Route one plain steering term through the extract pipeline.

        The concept has already been canonicalized and sign-flipped by
        ``parse_expr``; ``_handle_extract`` will re-run ``resolve_pole``
        on the canonical form which is idempotent (returns sign +1).
        ``namespace`` (when set) scopes that re-resolution and the
        downstream ``session.extract`` call so ``alice/foo`` and
        ``bob/foo`` stay distinct end-to-end.
        """
        def _on_success(name: str, profile: Any, a: float) -> None:
            self._session.steer(name, profile)
            self._alphas[name] = a
            self._enabled[name] = True
            self.call_from_thread(self._on_vector_extracted, name, a, profile)
        # _handle_extract's own parser expects a "<concept> <alpha>" text;
        # embed the variant in the concept token so resolve_pole strips it.
        concept_with_variant = (
            concept if variant == "raw" else f"{concept}:{variant}"
        )
        text = f"{concept_with_variant} {alpha}"
        self._handle_extract(
            text, include_alpha=True, on_success=_on_success,
            variant=variant, namespace=namespace,
        )

    def _dispatch_manifold_term(self, key: str, term: Any) -> None:
        """Route one manifold term through eager artifact validation.

        ``key`` is the grammar's manifold registry key
        (``<manifold>%<coords>``); ``term`` is the parsed
        :class:`~saklas.core.steering_expr.ManifoldTerm`.  A worker
        eager-loads the artifact via ``session._ensure_manifold_loaded``
        — which raises :class:`ManifoldNotRegisteredError` when the named
        manifold has no fitted tensor for the loaded model — and then
        validates the position-tuple arity against the loaded manifold's
        domain.  On success the term lands in ``_manifold_terms`` and the
        left panel refreshes; any failure surfaces as a system message.
        """
        chat = self._chat_panel
        if self._ab_shadow_active:
            chat.add_system_message(
                "Cannot modify steering during A/B shadow gen."
            )
            return
        # Position can be a node-label string (``persona%pirate``) or a
        # coord tuple (``persona%0.3,0.8``); display each in its
        # natural form.  Arity is only meaningful in coord form — the
        # label form resolves to whatever coords the node carries.
        if isinstance(term.position, str):
            pos_str = term.position
        else:
            pos_str = ",".join(f"{c:g}" for c in term.position)
        chat.add_system_message(
            f"Loading manifold '{term.manifold}' % {pos_str}..."
        )

        def _worker() -> None:
            try:
                self._session._ensure_manifold_loaded(term.manifold)
                manifold = self._session._manifolds[term.manifold]
                # ``resolve_position`` validates label-form (raises
                # UnknownManifoldLabelError on miss) and passes through
                # coord-form unchanged.  A test-double manifold without
                # ``resolve_position`` falls back to ``term.position``
                # directly — fine for coord form, raises here for
                # label form (the real engine path catches it).
                resolve = getattr(manifold, "resolve_position", None)
                if resolve is not None:
                    resolved = resolve(term.position)
                elif isinstance(term.position, str):
                    raise ValueError(
                        f"manifold '{term.manifold}' cannot resolve "
                        f"label {term.position!r} — manifold lacks "
                        f"resolve_position()"
                    )
                else:
                    resolved = term.position
                want = manifold.domain.intrinsic_dim
                got = len(resolved)
                if got != want:
                    raise ValueError(
                        f"manifold '{term.manifold}' is {want}-dimensional "
                        f"but the position has {got} coordinate(s)"
                    )
            except SaklasError as e:
                self.call_from_thread(
                    self._steer_status, e.user_message()[1],
                )
                return
            except ValueError as e:
                self.call_from_thread(self._steer_status, str(e))
                return
            except Exception as e:
                self.call_from_thread(
                    self._steer_status, f"{type(e).__name__}: {e}",
                )
                return
            else:
                self.call_from_thread(self._on_manifold_added, key, term)
            finally:
                # Advance the queue drain — manifold loading runs on a
                # worker, not the gen loop, so no natural ``done`` lands.
                self._ui_token_queue.put(("done", False))

        self.run_worker(_worker, thread=True)

    def _on_manifold_added(self, key: str, term: Any) -> None:
        """Register a validated manifold term and refresh the left panel."""
        self._manifold_terms[key] = term
        self._enabled[key] = True
        coords_str = ",".join(f"{c:g}" for c in term.position)
        self._chat_panel.add_system_message(
            f"Manifold '{term.manifold}' % {coords_str} active "
            f"(blend {term.coeff:.2f})"
        )
        self._refresh_left_panel()

    def _handle_probe(self, text: str) -> None:
        ns = _detect_namespace_selector(text.strip())
        if ns is not None:
            self._handle_probe_namespace(ns)
            return

        def _on_success(name: str, profile: Any, _alpha: Any) -> None:
            self._session.probe(name, profile)
            self.call_from_thread(self._on_probe_added, name)
        self._handle_extract(text, include_alpha=False, on_success=_on_success)

    def _handle_extract_only(self, text: str) -> None:
        def _on_success(name: str, _profile: Any, _alpha: Any) -> None:
            # Pure cache-warm: no steering, no probe, no panel state.
            self.call_from_thread(
                self._steer_status, f"extracted '{name}'"
            )
        self._handle_extract(
            text, include_alpha=False, on_success=_on_success,
            pending_type="extract",
        )

    def _handle_manifold(self, text: str) -> None:
        """`/manifold fit <folder>` — fit a steering manifold.

        Manifold *authoring* (writing ``manifold.json`` + ``nodes/*.json``)
        stays out of the TUI; this command only fits an already-authored
        manifold pack folder for the loaded model.  Gated like
        ``/extract`` — refused during an A/B shadow gen, deferred behind
        an in-flight generation via the pending queue.
        """
        chat = self._chat_panel
        text = (text or "").strip()
        parts = text.split(maxsplit=1)
        verb = parts[0].lower() if parts else ""
        if verb != "fit":
            chat.add_system_message("Usage: /manifold fit <folder>")
            return
        folder_arg = parts[1].strip() if len(parts) > 1 else ""
        folder_arg = _unquote(folder_arg)
        if not folder_arg:
            chat.add_system_message("Usage: /manifold fit <folder>")
            return
        if self._ab_shadow_active:
            chat.add_system_message(
                "Cannot fit a manifold during A/B shadow gen."
            )
            return
        if self._is_busy:
            self._enqueue_pending(
                PendingItem(
                    "manifold_fit", f"/manifold fit {folder_arg}",
                    (folder_arg,),
                )
            )
            return
        self._start_manifold_fit(folder_arg)

    def _start_manifold_fit(self, folder_arg: str) -> None:
        """Run ``session.extract_manifold`` on a worker thread.

        Mirrors ``_handle_extract``'s worker structure: progress lines
        stream to the chat pane, errors surface as system messages, and
        the ``finally`` block enqueues a ``done`` sentinel so the pending
        queue keeps draining (manifold fitting runs off the gen loop).
        """
        from pathlib import Path

        folder = Path(folder_arg).expanduser()
        chat = self._chat_panel
        if not (folder / "manifold.json").exists():
            chat.add_system_message(
                f"/manifold fit: no manifold.json in {folder}"
            )
            self._ui_token_queue.put(("done", False))
            return
        chat.add_system_message(f"Fitting manifold from {folder}...")

        def _work() -> None:
            def _progress(msg: str) -> None:
                self.call_from_thread(self._steer_status, msg)
            # ``fit_rbf_interpolant`` raises a bare ``ValueError`` on
            # poisedness failure — not a ``SaklasError``; the shared
            # worker wrapper surfaces it as a plain string.
            manifold = self._session.extract_manifold(
                folder, on_progress=_progress,
            )
            self.call_from_thread(
                self._steer_status,
                f"fitted manifold '{manifold.name}' "
                f"({len(manifold.layers)}L, "
                f"{len(manifold.node_labels)} nodes) — "
                f"steer it with `/steer <coeff> {manifold.name}%<coords>`",
            )

        self._run_worker_with_queue(_work)

    def _handle_manifold_probe(self, text: str) -> None:
        """``/manifold-probe <selector>`` — attach a manifold probe.

        Routes through ``session.add_manifold_probe`` (Phase 1 engine
        wiring) — the selector can be a bundled name (``pad``,
        ``personas``), an ``ns/name`` form, or an already-loaded
        manifold's registered name; the session's lazy-load path
        (``_ensure_manifold_loaded``) handles resolution.  Refused
        during A/B shadow gen for the same reason ``/probe`` is —
        modifying the monitor set mid-shadow would interleave readings
        across the steered/unsteered split.  Deferred behind an
        in-flight gen via the pending queue (kind ``manifold_probe``).
        """
        chat = self._chat_panel
        selector = (text or "").strip()
        selector = _unquote(selector)
        if not selector:
            chat.add_system_message("Usage: /manifold-probe <selector>")
            return
        if self._ab_shadow_active:
            chat.add_system_message(
                "Cannot attach a manifold probe during A/B shadow gen."
            )
            return
        if self._is_busy:
            self._enqueue_pending(
                PendingItem(
                    "manifold_probe", f"/manifold-probe {selector}",
                    (selector,),
                )
            )
            return
        self._start_manifold_probe_attach(selector)

    def _start_manifold_probe_attach(self, selector: str) -> None:
        """Run ``session.add_manifold_probe`` on a worker thread.

        Mirrors ``_start_manifold_fit``'s worker structure — errors
        surface as system messages and a ``done`` sentinel re-enters
        the queue drain so a queued attach doesn't stall subsequent
        items.
        """
        chat = self._chat_panel
        chat.add_system_message(f"Attaching manifold probe '{selector}'...")

        def _worker() -> None:
            try:
                name = self._session.add_manifold_probe(selector)
            except SaklasError as e:
                self.call_from_thread(self._steer_status, e.user_message()[1])
                return
            except Exception as e:
                self.call_from_thread(
                    self._steer_status, f"{type(e).__name__}: {e}",
                )
                return
            finally:
                self._ui_token_queue.put(("done", False))
            self.call_from_thread(self._on_manifold_probe_added, name)

        self.run_worker(_worker, thread=True)

    def _on_manifold_probe_added(self, name: str) -> None:
        """Mirror probe wiring — refresh the trait panel + announce."""
        self._refresh_manifold_probes()
        self._chat_panel.add_system_message(
            f"Manifold probe '{name}' active."
        )

    def _handle_manifold_probe_remove(self, text: str) -> None:
        """``/manifold-probe-remove <name>`` — detach an attached probe."""
        chat = self._chat_panel
        name = (text or "").strip()
        name = _unquote(name)
        if not name:
            chat.add_system_message("Usage: /manifold-probe-remove <name>")
            return
        monitor = self._session.manifold_monitor
        if name not in monitor.probe_names:
            chat.add_system_message(f"Manifold probe '{name}' not active.")
            return
        try:
            self._session.remove_manifold_probe(name)
        except SaklasError as e:
            chat.add_system_message(e.user_message()[1])
            return
        except Exception as e:
            chat.add_system_message(f"{type(e).__name__}: {e}")
            return
        self._refresh_manifold_probes()
        chat.add_system_message(f"Manifold probe '{name}' removed.")

    def _refresh_manifold_probes(self) -> None:
        """Push the attached-probe map down to the trait panel.

        The trait panel needs each attached probe's :class:`Manifold`
        artifact (not just the name) so it can introspect the domain
        for the 2-D mini-map at render time.  Pulled from the
        ``AttachedManifoldProbe`` records on the live monitor.
        """
        attached = self._session.manifold_monitor.attached_probes()
        probes = {name: p.manifold for name, p in attached.items()}
        self._trait_panel.set_active_manifold_probes(probes)

    def _handle_pairs(self, text: str) -> None:
        """`/pairs <name>` — open the custom-statement extraction modal.

        Opens :class:`~saklas.tui.pairs_modal.CustomPairsModal`, a
        multi-line editor for hand-authored ``positive | negative``
        contrastive pairs.  On submit the lines are parsed into a pairs
        list and handed straight to ``session.extract`` — bypassing
        scenario / pair generation.

        A modal cannot be a :class:`PendingItem`, so ``/pairs`` is
        refused mid-generation rather than queued (matching how other
        modal-requiring commands behave).
        """
        chat = self._chat_panel
        raw_args = (text or "").strip()
        raw_args, role = self._peel_role_flag(raw_args)
        name = _unquote(raw_args)
        if not name:
            chat.add_system_message("Usage: /pairs <name> [--role <slug>]")
            return
        if self._ab_shadow_active:
            chat.add_system_message(
                "Cannot extract a vector during A/B shadow gen."
            )
            return
        if self._is_busy:
            chat.add_system_message(
                "/pairs opens a modal — finish or stop the current "
                "generation first."
            )
            return

        from saklas.tui.pairs_modal import CustomPairsModal

        def _on_submit(pairs: list[tuple[str, str]] | None) -> None:
            if not pairs:
                return  # modal cancelled
            self._start_pairs_extract(name, pairs, role=role)

        self.push_screen(CustomPairsModal(name), _on_submit)

    def _start_pairs_extract(
        self, name: str, pairs: list[tuple[str, str]],
        *, role: str | None = None,
    ) -> None:
        """Extract a steering vector from hand-authored contrastive pairs.

        ``session.extract`` accepts a list of ``(positive, negative)``
        tuples directly as ``source``, skipping scenario / pair
        generation.  The extracted vector is steered at
        :data:`DEFAULT_ALPHA` and registered on the left panel, mirroring
        ``/steer``'s post-extraction wiring.
        """
        chat = self._chat_panel
        role_note = f" as :role-{role}" if role else ""
        chat.add_system_message(
            f"Extracting '{name}'{role_note} from {len(pairs)} custom pair(s)..."
        )

        def _on_success(canonical: str, profile: Any, alpha: float) -> None:
            self._session.steer(canonical, profile)
            self._alphas[canonical] = alpha
            self._enabled[canonical] = True
            self.call_from_thread(
                self._on_vector_extracted, canonical, alpha, profile,
            )

        def _worker() -> None:
            def _progress(msg: str) -> None:
                self.call_from_thread(self._steer_status, msg)
            try:
                # Hand-authored contrastive examples become the two pole
                # corpora of a 2-node ``pca`` manifold — positive pole vs its
                # opposite — fit directly (no scenario / pair generation).
                positive = [pos for pos, _ in pairs]
                negative = [neg for _, neg in pairs]
                extract_kwargs: dict[str, Any] = {
                    "on_progress": _progress, "namespace": "local",
                }
                if role is not None:
                    extract_kwargs["role"] = role
                canonical, profile = self._session.extract_vector_from_corpora(
                    name, positive, negative, **extract_kwargs,
                )
                _on_success(canonical, profile, DEFAULT_ALPHA)
            except SaklasError as e:
                self.call_from_thread(self._steer_status, e.user_message()[1])
            except (ValueError, TypeError) as e:
                self.call_from_thread(self._steer_status, str(e))
            except Exception as e:
                self.call_from_thread(
                    self._steer_status, f"{type(e).__name__}: {e}",
                )
            finally:
                self._ui_token_queue.put(("done", False))

        self.run_worker(_worker, thread=True)

    def _run_worker_with_queue(
        self,
        work: "Callable[[], Any]",
        *,
        on_error: "Callable[[BaseException], None] | None" = None,
    ) -> None:
        """Run ``work`` on a worker thread with the canonical
        try/except/finally boilerplate the off-gen-loop handlers share.

        Errors surface through ``_steer_status`` — ``SaklasError`` via its
        ``user_message()``, ``ValueError`` as a bare string, anything else
        as ``"<Type>: <msg>"`` — and the ``finally`` block enqueues a
        ``("done", False)`` sentinel so the pending-queue drain keeps
        advancing (these handlers run off the gen loop, so no natural
        ``done`` arrives).  Pass ``on_error`` to override the default
        error surfacing entirely.
        """

        def _worker() -> None:
            try:
                work()
            except SaklasError as e:
                if on_error is not None:
                    on_error(e)
                else:
                    self.call_from_thread(self._steer_status, e.user_message()[1])
            except ValueError as e:
                if on_error is not None:
                    on_error(e)
                else:
                    self.call_from_thread(self._steer_status, str(e))
            except Exception as e:
                if on_error is not None:
                    on_error(e)
                else:
                    self.call_from_thread(
                        self._steer_status, f"{type(e).__name__}: {e}"
                    )
            finally:
                self._ui_token_queue.put(("done", False))

        self.run_worker(_worker, thread=True)

    def _steer_status(self, msg: str) -> None:
        self._chat_panel.add_system_message(msg)

    def _on_probe_added(self, name: str) -> None:
        self._trait_panel.set_active_probes(set(self._session._monitor.probe_names))
        # Per-token highlight default-on when a probe is explicitly added
        # via /probe. Seed to this probe; Ctrl+Y cycles the mode.
        self._highlight_probe = name
        self._highlighting = True
        self._apply_highlight_to_all()
        self._refresh_trait_why()
        self._steer_status(f"Probe '{name}' active. Highlight on (⌃Y to cycle).")

    def _refresh_left_panel(self) -> None:
        self._left_panel.update_vectors(self._vector_list_for_panel())

    def _do_clear(self) -> None:
        self._session.clear_history()
        self._chat_panel.clear_log()
        self._assistant_messages.clear()
        self._row_for_widget.clear()
        if self._raw_mode:
            self._sync_raw_buffer_from_tree()
        self._trait_panel.update_values({}, {}, {})
        self._trait_panel.clear_manifold_readings()
        self._refresh_trait_why()
        self._chat_panel.add_system_message("Chat history cleared.")

    def _repaint_chat_from_active_path(self) -> None:
        """Rebuild the chat log to show only the loom tree's active path.

        Called after any navigation (loom screen, ``/nav``, ``/load``) so
        the chat panel reflects the active branch rather than whatever
        turns happened to be streamed into it last.  Per-token probe
        scores aren't persisted on loom nodes, so navigated-to history
        renders without probe highlight; surprise highlight survives
        because per-token logprobs ride along in the node token rows.
        """
        from saklas.tui.chat_panel import SURPRISE_PROBE

        chat = self._chat_panel
        # Navigating the loom tree mid-generation logically abandons the
        # in-flight turn — the active pointer has moved elsewhere.  Stop
        # the worker so its tokens stop chasing a widget we're about to
        # detach; its ``("done",)`` sentinel still drains cleanly through
        # ``_poll_generation`` (which reads ``_current_assistant_widget``,
        # set to ``None`` just below).
        if self._session.is_generating:
            self._session.stop()
        chat.clear_log()
        self._assistant_messages.clear()
        self._row_for_widget.clear()
        self._current_assistant_widget = None
        if self._raw_mode:
            # Raw mode has no turn rows — re-sync the flat buffer from
            # the navigated active path instead of rebuilding the log.
            self._sync_raw_buffer_from_tree()
            self._refresh_input_mode()
            return
        # Resolve the highlight probe the same way ``_apply_highlight_to_all``
        # does, then seed it on each widget *before* mount — ``_apply_static``
        # (deferred to ``on_mount``) reads ``_highlight_on`` / ``_highlight_probe``
        # when it renders, so the navigated-to history comes up already
        # tinted without a post-mount sweep.  Probe scores aren't persisted
        # on loom nodes, so probe-mode falls back to plain text; surprise
        # mode survives on the per-token logprobs carried in the token rows.
        if self._highlighting and self._highlight_probe != SURPRISE_PROBE:
            nav_probe = self._trait_panel.get_selected_probe()
            if nav_probe is not None:
                self._highlight_probe = nav_probe
        probe = self._highlight_probe if self._highlighting else None
        tree = self._session.tree
        for node in tree.active_path():
            if node.id == tree.root_id:
                continue
            if node.role == "user":
                chat.add_user_message(node.text)
            elif node.role == "assistant":
                resp_rows = node.tokens or []
                think_rows = node.thinking_tokens or []
                resp_tokens = [str(r.get("text", "")) for r in resp_rows]
                think_tokens = [str(r.get("text", "")) for r in think_rows]
                thinking_text = "".join(think_tokens)
                row, widget = chat.add_finalized_assistant(
                    node.text, thinking_text,
                    response_tokens=resp_tokens or None,
                    thinking_tokens=think_tokens or None,
                    response_logprobs=[r.get("logprob") for r in resp_rows] or None,
                    thinking_logprobs=[r.get("logprob") for r in think_rows] or None,
                )
                widget.apply_highlight(self._highlighting, probe)
                self._assistant_messages.append(widget)
                self._row_for_widget[id(widget)] = row
        chat.scroll_to_bottom()
        # Navigation may have landed the active node on a user turn —
        # refresh the input placeholder so it signals prefill vs. send.
        self._refresh_input_mode()

    def _do_rewind(self) -> None:
        if not self._messages:
            self._chat_panel.add_system_message("Nothing to rewind.")
            return
        self._session.rewind()
        self._chat_panel.rewind()
        self._assistant_messages = [w for w in self._assistant_messages if w.is_mounted]
        # Drop stale row references so the next AB backfill walk doesn't
        # see widgets whose rows are gone.
        self._row_for_widget = {
            wid: row for wid, row in self._row_for_widget.items() if row.is_mounted
        }
        if self._raw_mode:
            self._sync_raw_buffer_from_tree()
        self._chat_panel.add_system_message("Rewound to before last message.")

    def _refresh_gen_config(self) -> None:
        self._left_panel.update_gen_config(
            self._session.config.temperature,
            self._session.config.top_p,
            self._session.config.max_new_tokens,
            self._session.config.system_prompt,
            thinking=self._thinking if self._supports_thinking else None,
            thinking_forced=(
                self._supports_thinking and not self._thinking_optional
            ),
        )

    def _thinking_status_str(self) -> str:
        """One-line human-readable thinking state for ``/model`` output."""
        if not self._supports_thinking:
            return "not supported"
        if not self._thinking_optional:
            return "always on (model always thinks)"
        return f"{'ON' if self._thinking else 'OFF'} (⌃T)"

    # -- Clipboard --

    def action_copy_selection(self) -> None:
        text = self.screen.get_selected_text()
        if text:
            self.copy_to_clipboard(text)

    # -- Generation --

    def action_stop_generation(self) -> None:
        """``Esc`` — context-sensitive.

        Priority order:
        1. A generation is in flight → ``session.stop()``.  Mirrors
           the GUI's Stop button: kills the current sibling, leaves
           the queue untouched, drain proceeds on the resulting
           ``done``.
        2. No gen, a pending slot is pulled → cancel the pull
           (restore the live stash, leave the slot in the queue).
           Gives the user an out from a pulled-and-half-edited row
           without committing or removing.
        3. Otherwise → no-op.
        """
        if self._session.is_generating:
            self._session.stop()
            return
        if self._pulled_slot is not None:
            self._cancel_pull()

    async def action_quit(self) -> None:
        if self._is_busy:
            # Queue quit behind any in-flight work — preserves "Stop
            # only stops; queue drains on done" semantics.  Hit ``Esc``
            # first if you want to short-circuit.
            self._enqueue_pending(PendingItem("quit", "/quit"))
        else:
            self.exit()

    def _wants_live_probe_scores(self) -> bool:
        """Return True when streamed probe scores can affect the current UI."""
        from saklas.tui.chat_panel import SURPRISE_PROBE

        return bool(
            self._highlighting
            and self._highlight_probe not in (None, SURPRISE_PROBE)
            and self._session._monitor.probe_names
        )

    @property
    def _raw_mode(self) -> bool:
        """True when the chat panel is showing the flat completion buffer."""
        return self._render_mode == "raw"

    def _sync_raw_buffer_from_tree(self) -> None:
        """Refresh the raw buffer from the active loom path, if visible."""
        if not self._raw_mode:
            return
        rb = self._chat_panel.raw_buffer
        rb.sync_committed(self._session.tree.flat_text())
        rb.apply_highlight(
            self._highlighting,
            self._highlight_probe if self._highlighting else None,
        )

    def _start_generation(
        self,
        user_text: str | None = None,
        *,
        raw_continuation: bool = False,
        raw_draft: str = "",
        raw_parent: str | None = None,
    ) -> None:
        """Kick off a generation.

        ``user_text`` is the new user message (``None`` = regeneration of
        the last turn, so we pop the last assistant and re-use the last
        user content).

        ``raw_continuation`` routes the stream into the raw-mode
        completion buffer instead of mounting a chat turn row:
        ``user_text`` is the divergence *tail* (``""`` = a bare
        continuation from the active leaf), ``raw_draft`` the full
        submitted buffer, ``raw_parent`` the node the tail hangs under.
        """
        self._ui_gen_active = True

        self._gen_token_count = 0
        self._last_tok_per_sec = 0.0
        self._last_elapsed = 0.0
        self._log_ppl_sum = 0.0
        self._ppl_count = 0
        self._gen_start_time = time.monotonic()

        if raw_continuation:
            # Raw mode: the stream target is the flat completion buffer,
            # not a fresh turn row.  ``begin_continuation`` stamps the
            # submitted draft as the committed head; streamed tokens
            # append (and tint) on top of it.
            widget = self._chat_panel.raw_buffer
            widget.begin_continuation(raw_draft)
            self._current_assistant_widget = widget
        else:
            row, widget = self._chat_panel.start_assistant_message()
            self._row_for_widget[id(widget)] = row
            self._current_assistant_widget = widget
            self._assistant_messages.append(widget)
        # Fresh widgets spawn with ``_highlight_on=False``; inherit the
        # app's current highlight state so streamed tokens render
        # highlighted from the first emit instead of requiring a Ctrl+Y
        # mode-cycle round trip after the response completes.
        if self._highlighting:
            widget.apply_highlight(True, self._highlight_probe)

        # Snapshot alphas for this generation
        alphas = self._active_alphas()
        use_thinking = self._thinking

        # For regeneration, we re-submit the last user message. Session
        # owns history; _handle_command / action_regenerate pop the last
        # assistant turn + user turn before calling us with that text.
        if user_text is None:
            # Regenerate: read the last user message from history and
            # re-send it as input.  Under v2.3 loom, ``add_user_turn``
            # dedups against the existing user-child so no explicit pop
            # is needed; just read the text.
            hist = self._messages
            if hist and hist[-1]["role"] == "user":
                user_text = hist[-1]["content"]
            else:
                self._ui_gen_active = False
                self._chat_panel.add_system_message("Nothing to regenerate.")
                return

        sampling = SamplingConfig(
            temperature=self._session.config.temperature,
            top_p=self._session.config.top_p,
            max_tokens=self._session.config.max_new_tokens,
            seed=self._default_seed,
            # Logit-pass: request chosen-token logprob capture so the
            # surprise highlight mode has data to tint with.  ``0`` =
            # chosen-only (no top-K alts) — one extra ``.item()`` per
            # token; the ``log_softmax`` it reads is already computed
            # because the worker always installs an ``on_token``
            # consumer.  Without this ``event.logprob`` is always None
            # and surprise mode renders uniform-no-tint.
            logprobs=0,
        )
        steering = Steering(alphas=dict(alphas), thinking=use_thinking) if alphas else None

        # When the active node is a user (regen flow after
        # ``_rewind_active_assistant``), anchor the gen under the user's
        # parent so ``add_user_turn``'s dedup lands the new assistant as
        # a sibling rather than a child-of-user.  D15 rejects bare
        # send-from-user; this branch is what tells the engine "I'm
        # regenerating, dedup will land me on the right user node."
        regen_parent_id: str | None = None
        if raw_continuation:
            # The tail hangs under the divergence parent (mid-buffer
            # edit) or the active leaf (clean append) — resolved by the
            # caller via ``_resolve_raw_divergence``.
            regen_parent_id = raw_parent
        else:
            tree = self._session.tree
            active_node = tree.nodes.get(tree.active_node_id)
            if (active_node is not None and active_node.role == "user"
                    and active_node.parent_id is not None):
                regen_parent_id = active_node.parent_id

        def _generate():
            try:
                stream = self._session.generate_stream(
                    user_text,
                    steering=steering,
                    sampling=sampling,
                    stateless=False,
                    raw=self._raw_mode,
                    thinking=use_thinking,
                    parent_node_id=regen_parent_id,
                    live_scores=self._wants_live_probe_scores(),
                )
                for event in stream:
                    self._ui_token_queue.put(
                        # Logit-pass: ``event.logprob`` rides along so the
                        # surprise highlight mode + chat_panel's per-token
                        # logprob storage can render mid-gen.  Populated
                        # because the SamplingConfig above sets
                        # ``logprobs=0``; ``None`` only on the prefill /
                        # partial-UTF-8 flush steps the engine never
                        # assigns a chosen-token logprob to.
                        # Manifold readings (``event.manifold_readings``)
                        # ride alongside scalar scores — ``None`` when no
                        # manifold probe is attached or ``live_scores`` is
                        # off; otherwise a per-probe ``ManifoldTokenReading``
                        # the trait panel renders mid-gen.
                        ("tok", event.text, event.thinking, event.scores,
                         event.perplexity, event.logprob, widget, False,
                         event.manifold_readings),
                    )
                    self._gen_token_count += 1
                # Normal completion — pull per-token scores out of the
                # session and push to the widget for highlight.
                self._ui_token_queue.put(("finalize", widget, False))
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                self._ui_token_queue.put(("error", msg, False))
            finally:
                if self._session._device.type == "mps":
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass
                self._ui_token_queue.put(("done", False))

        self.run_worker(_generate, thread=True)

    # -- Raw (flat completion) mode --

    def _resolve_raw_divergence(
        self, draft: str,
    ) -> tuple[str, str | None]:
        """Diff ``draft`` against the loom active path and locate the
        single span the change collapses to.

        Port of ``webui``'s ``RawBuffer.svelte::resolveDivergence``.
        Returns ``(tail, parent_node_id)`` — ``tail`` is the new text
        from the divergence offset to the end of the draft, and
        ``parent_node_id`` is the node it hangs under: the diverging
        node's parent for a mid-buffer edit, the active leaf for a pure
        append.  A clean draft (no divergence, no append) yields an
        empty tail under the active leaf — a bare continuation.
        """
        tree = self._session.tree
        nodes = [n for n in tree.active_path() if n.id != tree.root_id]
        start = 0
        for node in nodes:
            ntext = node.text or ""
            if draft[start:start + len(ntext)] != ntext:
                # Divergence inside this node — it (and its subtree)
                # stay as the original branch; the tail branches as a
                # sibling, a child of the node's parent.
                return draft[start:], node.parent_id
            start += len(ntext)
        # No node diverged — the tail is appended past the joined
        # buffer and hangs under the active leaf.
        return draft[start:], tree.active_node_id

    def _submit_raw_continuation(
        self, draft: str, *, replace_slot: int | None = None,
    ) -> None:
        """Continue the raw buffer from ``draft``.

        ``draft`` is the full intended buffer text; divergence (and so
        the generation's parent node) is resolved at dispatch time so a
        queued continuation can't bind to a stale tree.
        """
        if self._is_busy:
            self._enqueue_pending(
                PendingItem("raw_continue", draft), replace_slot=replace_slot,
            )
            return
        tail, parent = self._resolve_raw_divergence(draft)
        self._start_generation(
            tail, raw_continuation=True, raw_draft=draft, raw_parent=parent,
        )

    def on_raw_buffer_continue(self, event: RawBuffer.Continue) -> None:
        """``Enter`` in the raw completion buffer — generate from it."""
        self._submit_raw_continuation(event.draft)

    def _set_render_mode(self, mode: str) -> None:
        """Switch the chat surface between ``"chat"`` and ``"raw"``.

        Non-destructive — both surfaces derive from the same loom tree;
        the now-visible one is repainted from the active path.  A/B is a
        chat-mode-only layout, so switching to raw drops it.
        """
        if (
            self._raw_mode
            and mode == "chat"
            and self._chat_panel.raw_buffer.is_dirty
        ):
            self._chat_panel.add_system_message(
                "Raw buffer has uncommitted edits; press Ctrl+Enter to "
                "commit them, or Enter to continue before switching."
            )
            return
        if mode not in ("chat", "raw"):
            self._chat_panel.add_system_message(
                f"Unknown render mode '{mode}' — use chat or raw."
            )
            return
        if mode == self._render_mode:
            self._chat_panel.add_system_message(f"Already in {mode} mode.")
            return
        if self._session.is_generating:
            self._session.stop()
        self._render_mode = mode
        cp = self._chat_panel
        if mode == "raw" and self._ab_mode:
            self._ab_mode = False
            cp.set_ab_mode(False)
        cp.set_render_mode(mode)
        if mode == "raw":
            self._sync_raw_buffer_from_tree()
        else:
            self._repaint_chat_from_active_path()
        cp.add_system_message(f"Render mode: {mode}.")

    def action_toggle_render_mode(self) -> None:
        """``Ctrl+O`` — toggle the chat panel between chat and raw."""
        self._set_render_mode(
            "chat" if self._render_mode == "raw" else "raw"
        )

    def _prefill_target_node_id(self) -> str | None:
        """Active node id when it's a user turn, else ``None``.

        A user-role active node means the turn below it is the
        assistant's — so a typed message composes the assistant reply
        (answer-prefill) rather than a new user turn.  Returning ``None``
        keeps the normal ``_start_generation`` path.
        """
        tree = self._session.tree
        active = tree.nodes.get(tree.active_node_id)
        if active is not None and active.role == "user":
            return active.id
        return None

    def _predicted_queue_end_on_user_node(self) -> bool | None:
        """Walk the queue tail-first, return the first per-kind prediction.

        ``None`` when no queued item changes the active-node role (empty
        queue, or items like ``/steer`` / ``/probe`` that don't touch the
        tree).  ``True`` / ``False`` when a queued item lands a user /
        non-user node — the next submission's input mode should reflect
        the post-drain role rather than the live one.
        """
        for item in reversed(self._pending_queue):
            ends = _KIND_ENDS_ON_USER_NODE.get(item.kind)
            if ends is not None:
                return ends
        return None

    def _predicted_on_user_node(self) -> bool:
        """Queue-aware ``_prefill_target_node_id() is not None``.

        When the queue predicts a role change, return that prediction;
        otherwise fall through to the live active node.  Drives the
        placeholder via :meth:`_refresh_input_mode` so a queued
        ``commit_user`` flips the input to prefill mode immediately.
        """
        predicted = self._predicted_queue_end_on_user_node()
        if predicted is not None:
            return predicted
        return self._prefill_target_node_id() is not None

    def _refresh_input_mode(self) -> None:
        """Sync the chat input's prefill-mode placeholder to the queue-
        aware active node.

        Called from every active-node transition funnel
        (:meth:`_repaint_chat_from_active_path` for navigation, the
        ``done`` sentinel in :meth:`_poll_generation` for post-gen) and
        from :meth:`_enqueue_pending` / :meth:`_remove_pending_slot` so
        the placeholder also tracks queue mutations.
        """
        self._chat_panel.set_prefill_mode(self._predicted_on_user_node())

    def _start_prefill(self, node_id: str, prefill_text: str) -> None:
        """Answer-prefill — seed the assistant reply under user node
        ``node_id`` with ``prefill_text``, then stream the continuation.

        Routes through ``session.prefill_assistant``; the seeded prefix
        and its continuation stream into a fresh assistant widget via the
        same ``_ui_token_queue`` pipeline ``_start_generation`` uses, this
        time fed by ``prefill_assistant``'s ``on_token`` callback rather
        than a ``generate_stream`` iterator.  Steering rides from the
        current rack; thinking is forced off engine-side (the text is the
        start of the answer, not a thought).
        """
        self._ui_gen_active = True

        self._gen_token_count = 0
        self._last_tok_per_sec = 0.0
        self._last_elapsed = 0.0
        self._log_ppl_sum = 0.0
        self._ppl_count = 0
        self._gen_start_time = time.monotonic()

        row, widget = self._chat_panel.start_assistant_message()
        self._row_for_widget[id(widget)] = row
        self._current_assistant_widget = widget
        self._assistant_messages.append(widget)
        if self._highlighting:
            widget.apply_highlight(True, self._highlight_probe)

        alphas = self._active_alphas()
        # Prefill is always non-thinking — the text opens the answer, so
        # the thinking channel is skipped (``prefill_assistant`` forces
        # ``thinking=False`` regardless; the local Steering matches).
        steering = (
            Steering(alphas=dict(alphas), thinking=False) if alphas else None
        )
        sampling = SamplingConfig(
            temperature=self._session.config.temperature,
            top_p=self._session.config.top_p,
            max_tokens=self._session.config.max_new_tokens,
            seed=self._default_seed,
            logprobs=0,
        )

        def _on_token(text: str, is_thinking: bool, tid: Any, lp: Any, top_alts: Any, perplexity: Any) -> None:
            # Mirrors the ``("tok", …)`` tuple ``_start_generation`` builds
            # from a ``TokenEvent``.  ``prefill_assistant``'s on_token
            # carries no probe scores (no streaming monitor hook on this
            # path) — pass ``None``; ``_finalize_widget_highlight`` fills
            # the canonical per-token scores in at finalize.  Manifold
            # readings (final tuple slot) are also unsourced on the
            # prefill path; the trait-panel manifold section refreshes
            # from the end-of-gen aggregate via ``_finalize_widget_highlight``.
            self._ui_token_queue.put(
                ("tok", text, bool(is_thinking), None, perplexity, lp,
                 widget, False, None),
            )
            self._gen_token_count += 1

        def _prefill() -> None:
            try:
                self._session.prefill_assistant(
                    node_id, prefill_text,
                    steering=steering,
                    sampling=sampling,
                    on_token=_on_token,
                )
                self._ui_token_queue.put(("finalize", widget, False))
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                self._ui_token_queue.put(("error", msg, False))
            finally:
                if self._session._device.type == "mps":
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass
                self._ui_token_queue.put(("done", False))

        self.run_worker(_prefill, thread=True)

    # -- Commit (no-generation send) --

    def action_commit_text(self) -> None:
        """Commit chat-input text as the next turn without generating.

        ``Ctrl+Enter`` / ``Alt+Enter`` from the chat input.  Role-aware:
        on a user-role active node the text becomes the full assistant
        turn (``session.append_assistant_turn``); on any non-user active
        node it becomes a new user turn
        (``session.append_user_turn``).  No decode, no streaming.

        Queues behind an in-flight generation via the pending queue so
        the commit lands once the current gen finishes — mounting a row
        mid-stream would interleave UI in confusing ways.
        """
        if self._raw_mode:
            # Raw mode is a single flat buffer: Ctrl+Enter lands the
            # current divergent span without decoding, matching the web
            # UI's commit-edit path.
            self._commit_raw_draft(self._chat_panel.raw_buffer.draft)
            return
        try:
            inp = self._chat_panel.query_one("#chat-input", ChatInput)
        except Exception:
            return
        text = inp.text.strip()
        if not text:
            return
        inp.load_text("")
        self._push_input_history(text)
        # Forward the pulled slot so a re-edited queued item stays at
        # its original position rather than sliding to the queue tail.
        replace_slot = self._pulled_slot
        self._pulled_slot = None
        self._commit_with_text(text, replace_slot=replace_slot)

    def _commit_with_text(self, text: str, *, replace_slot: int | None = None) -> None:
        """Role-aware commit dispatch — text becomes the next turn.

        Shared between ``action_commit_text`` (Ctrl/Alt+Enter binding,
        when the terminal passes the modifier through) and
        ``_handle_commit`` (``/commit <text>`` slash command, the
        stock-terminal fallback).  ``_prefill_target_node_id`` returns
        the active user-node id when the active node is a user turn —
        in which case ``text`` becomes the full authored assistant
        reply.  Otherwise we commit a new user turn under the active
        node.

        Honors the in-flight-generation queue: when busy, enqueues a
        :class:`PendingItem` so the commit lands once the streaming
        sibling finishes.  Per the queue contract, the in-flight gen
        is *not* interrupted — use ``Esc`` if you want to short-
        circuit.  ``replace_slot`` lets a pulled-and-re-edited slot
        keep its original position in the queue.
        """
        user_node_id = self._prefill_target_node_id()
        if self._is_busy:
            # Queue role-awareness mirrors the submit handler: a queued
            # commit landing under a not-yet-existing user node (from
            # an earlier queued ``commit_user``) stamps ``ACTIVE_AT_DRAIN``
            # so the dispatcher resolves the parent fresh at drain time.
            if user_node_id is not None:
                item = PendingItem(
                    "commit_assistant", text, (user_node_id,),
                )
            elif self._predicted_on_user_node():
                item = PendingItem(
                    "commit_assistant", text, (ACTIVE_AT_DRAIN,),
                )
            else:
                item = PendingItem("commit_user", text)
            self._enqueue_pending(item, replace_slot=replace_slot)
            return
        if user_node_id is not None:
            self._start_commit_assistant(user_node_id, text)
        else:
            self._start_commit_user(text)

    def _handle_commit(self, raw: str) -> None:
        """`/commit <text>` — same semantics as Ctrl+Enter, by typing.

        The slash form is the cross-terminal fallback: stock macOS
        Terminal.app and iTerm2 collapse ``ctrl+enter`` and ``alt+enter``
        to bare ``enter``, so users without the CSI-u / kitty keyboard
        protocol can't reach the modifier binding.  Typing
        ``/commit hello`` lands the same turn the binding would have.
        """
        text = (raw or "").strip()
        if not text:
            self._chat_panel.add_system_message("Usage: /commit <text>")
            return
        if self._raw_mode:
            draft = self._chat_panel.raw_buffer.draft + text
            self._commit_raw_draft(draft)
            return
        # The slash dispatcher already pushed the full ``/commit …``
        # line to input history before invoking this handler — don't
        # double-push.
        self._commit_with_text(text)

    def _commit_raw_draft(
        self, draft: str, *, replace_slot: int | None = None,
    ) -> None:
        """Land the raw buffer's divergent span without generating."""
        if self._is_busy:
            self._enqueue_pending(
                PendingItem("raw_commit", draft), replace_slot=replace_slot,
            )
            return
        self._start_raw_commit(draft)

    def _start_raw_commit(self, draft: str) -> None:
        tail, parent = self._resolve_raw_divergence(draft)
        if tail == "":
            self._chat_panel.add_system_message("raw commit: no pending edit.")
            return

        def _commit() -> None:
            try:
                self._session.append_user_turn(
                    parent, tail, allow_any_parent=True,
                )
                self.call_from_thread(self._sync_raw_buffer_from_tree)
                self.call_from_thread(
                    self._chat_panel.add_system_message,
                    "raw commit landed.",
                )
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                self.call_from_thread(
                    self._chat_panel.add_system_message,
                    f"raw commit failed: {msg}",
                )
            finally:
                self._ui_token_queue.put(("done", False))

        self.run_worker(_commit, thread=True)

    def _start_commit_user(self, text: str) -> None:
        """Land a user turn under the active node without generating.

        Mounts the user row up front so the commit feels instant; the
        session method runs on a worker thread to keep the UI responsive
        even on a slow-tokenize text.  On failure (e.g. active is itself
        a user node) we surface a system message and skip the mount via
        a post-mount remove.

        The worker enqueues a ``("done", False)`` sentinel in its
        finally block — without it the pending-queue drain loop stalls
        after a queued commit, because commits don't run a generation
        and therefore don't produce a natural ``done`` event.
        """
        row = self._chat_panel.add_user_message(text)

        def _commit() -> None:
            try:
                self._session.append_user_turn(None, text)
                self.call_from_thread(self._refresh_input_mode)
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)

                def _rollback() -> None:
                    try:
                        row.remove()
                    except Exception:
                        pass
                    self._chat_panel.add_system_message(f"commit failed: {msg}")
                self.call_from_thread(_rollback)
            finally:
                # Advance the queue drain — commits don't stream so no
                # natural ``done`` arrives via the gen worker.
                self._ui_token_queue.put(("done", False))

        self.run_worker(_commit, thread=True)

    def _start_commit_assistant(self, user_node_id: str, text: str) -> None:
        """Land an authored assistant turn under ``user_node_id``.

        Mounts a finalized assistant row with ``text`` already populated
        (no streaming) and routes through ``session.append_assistant_turn``
        on a worker thread.  Highlight goes plain — authored turns carry
        no per-token scores.

        Worker enqueues ``("done", False)`` in finally so the pending-
        queue drain advances; see :meth:`_start_commit_user` for the
        rationale.
        """
        row, widget = self._chat_panel.add_finalized_assistant(text)
        self._row_for_widget[id(widget)] = row
        if self._highlighting:
            widget.apply_highlight(True, self._highlight_probe)

        def _commit() -> None:
            try:
                self._session.append_assistant_turn(user_node_id, text)
                self.call_from_thread(self._refresh_input_mode)
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)

                def _rollback() -> None:
                    try:
                        row.remove()
                    except Exception:
                        pass
                    self._chat_panel.add_system_message(f"commit failed: {msg}")
                self.call_from_thread(_rollback)
            finally:
                self._ui_token_queue.put(("done", False))

        self.run_worker(_commit, thread=True)

    def _poll_generation(self) -> None:
        chat = self._chat_panel
        tokens_consumed = 0
        generating = self._ui_gen_active

        while tokens_consumed < _TOKEN_DRAIN_LIMIT:
            try:
                item = self._ui_token_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "tok":
                # Tagged with the target widget + ``is_shadow`` flag so
                # steered and shadow streams route to the right column
                # without a global "current" lookup.  Shadow streams
                # bypass the gen-stat counters (token count, ppl) — those
                # describe the steered run only.
                # Logit-pass: 7-element tuple now (logprob between
                # perplexity and widget).  Drives the surprise highlight
                # mode + the per-token logprob storage on the widget.
                # Manifold-probe pass: optional 9th slot carries
                # ``event.manifold_readings`` — ``None`` when no manifold
                # probe is attached or ``live_scores`` is off; otherwise
                # a per-probe ``ManifoldTokenReading`` the trait panel
                # renders mid-gen.  Falls back to ``None`` for legacy
                # producers (e.g. test stubs) that emit the 8-element form.
                manifold_readings = None
                if len(item) >= 9:
                    (
                        _, token, is_thinking, scores, perplexity, logprob,
                        widget, is_shadow, manifold_readings,
                    ) = item
                else:
                    (
                        _, token, is_thinking, scores, perplexity, logprob,
                        widget, is_shadow,
                    ) = item
                if manifold_readings is not None and not is_shadow:
                    # Push live readings to the trait panel.  Shadow
                    # streams skip this — their readings describe the
                    # unsteered baseline and would clobber the steered
                    # rack mid-gen.
                    self._trait_panel.update_manifold_readings(
                        per_token=manifold_readings,
                    )
                if widget is not None:
                    if is_thinking:
                        widget.append_thinking_token(token)
                    else:
                        widget.ensure_thinking_collapsed()
                        widget.append_token(token)
                    if scores is not None:
                        widget.append_token_score(scores, is_thinking)
                    # ``logprob`` may legitimately be ``None`` during prefill
                    # or replay; always append so the per-token list stays
                    # index-aligned with the token list.
                    widget.append_token_logprob(logprob, is_thinking)
                if not is_shadow:
                    if perplexity is not None and perplexity > 0:
                        # Geometric mean over the gen: accumulate log(ppl),
                        # display exp(mean).  Equivalent to classical sequence
                        # perplexity; one step dominated by a rare token
                        # doesn't swamp the aggregate the way an arithmetic
                        # mean would.
                        self._log_ppl_sum += math.log(perplexity)
                        self._ppl_count += 1
                tokens_consumed += 1
            elif kind == "finalize":
                # Normal end — pull per-token scores stashed by session's
                # _finalize_generation and push to the widget for highlight.
                _, widget, _is_shadow = item
                self._finalize_widget_highlight(widget)
            elif kind == "error":
                _, msg, is_shadow = item
                tag = "A/B shadow error" if is_shadow else "generation error"
                chat.add_system_message(f"{tag}: {msg}")
            elif kind == "done":
                _, is_shadow = item
                widget = self._current_assistant_widget
                if widget:
                    widget.ensure_thinking_collapsed()
                if isinstance(widget, RawBuffer):
                    # Clear the raw buffer's streaming guard so the next
                    # keystroke is seen as a user edit again.
                    widget.end_continuation()
                self._current_assistant_widget = None
                self._ui_gen_active = False
                generating = False
                if not is_shadow and self._gen_start_time > 0:
                    self._last_elapsed = time.monotonic() - self._gen_start_time
                    if self._last_elapsed > MIN_ELAPSED_FOR_RATE:
                        self._last_tok_per_sec = self._gen_token_count / self._last_elapsed
                    self._gen_start_time = 0.0
                if not is_shadow:
                    # The finished turn moved the active node onto its new
                    # assistant — refresh the input placeholder out of any
                    # prefill mode it was in.
                    self._refresh_input_mode()

                # Shadow done: clear shadow flags, then fall through to
                # pending-action drain so anything queued during the
                # shadow gen runs now.
                if is_shadow:
                    self._ab_shadow_active = False
                    self._ab_shadow_row = None

                # Steered done: if AB mode is on and the configured
                # auto-regen mode is ``unsteered`` (the default that
                # matches today's A/B behavior), fire a shadow gen and
                # DON'T drain pending — let the shadow's own ``done``
                # handle that, so a mid-flight pending action waits
                # until the AB pair is complete.
                #
                # For any other auto-regen mode (inverted / reseed /
                # cool / hot / custom), fire ``regen_with_modifier``
                # instead — it lands as a sibling under the user-parent.
                elif (
                    self._ab_mode
                    and not self._raw_mode
                    and widget is not None
                    and not self._pending_queue
                    and self._loom_auto_regen_mode == "unsteered"
                ):
                    row = self._row_for_widget.get(id(widget))
                    if row is not None:
                        self._start_shadow_generation(row)
                        break
                elif (
                    self._loom_auto_regen_on
                    and not self._raw_mode
                    and self._loom_auto_regen_mode != "unsteered"
                    and not self._pending_queue
                    and widget is not None
                ):
                    # Stream the modifier-regen output into the shadow
                    # column the same way the ``unsteered`` branch does
                    # for plain A/B — picks up the row associated with
                    # the just-finished primary widget so the new
                    # sibling renders live alongside its sibling.  Falls
                    # through to the row-less form (system-message only)
                    # if we can't find the row mapping.
                    row = self._row_for_widget.get(id(widget))
                    if row is not None:
                        self._fire_auto_regen(row)
                        break
                    self._fire_auto_regen(None)
                elif (
                    self._loom_auto_regen_on
                    and self._raw_mode
                    and self._loom_auto_regen_mode != "unsteered"
                    and not self._pending_queue
                ):
                    self._fire_auto_regen(None)

                # Drain one queued item per ``done`` — each item kicks
                # off its own work whose ``done`` will re-enter here
                # and drain the next, preserving FIFO.
                self._drain_next_pending()
                break

        if tokens_consumed > 0:
            chat.scroll_to_bottom()

        if generating and self._gen_start_time > 0:
            elapsed = time.monotonic() - self._gen_start_time
            tok_per_sec = self._gen_token_count / elapsed if elapsed > MIN_ELAPSED_FOR_RATE else 0.0
            self._last_tok_per_sec = tok_per_sec
            self._last_elapsed = elapsed

        # Dedupe tuple includes the loom-state extras (prune expr +
        # active auto-regen mode) so the footer refreshes when either
        # flips between polls, even when generation hasn't ticked.
        new_state = (self._gen_token_count, self._last_tok_per_sec,
                     self._last_elapsed, generating, self._ppl_count,
                     self._loom_prune_expr,
                     self._loom_auto_regen_mode if self._loom_auto_regen_on else None)
        if new_state != self._last_gen_state:
            self._last_gen_state = new_state
            ppl_mean = (
                math.exp(self._log_ppl_sum / self._ppl_count)
                if self._ppl_count > 0 else None
            )
            chat.update_status(
                generating=generating,
                gen_tokens=self._gen_token_count,
                max_tokens=self._session.config.max_new_tokens,
                tok_per_sec=self._last_tok_per_sec,
                elapsed=self._last_elapsed,
                perplexity=ppl_mean,
                prune_expr=self._loom_prune_expr,
                auto_regen_mode=(
                    self._render_auto_regen_mode(self._loom_auto_regen_mode)
                    if (self._loom_auto_regen_on
                        and self._loom_auto_regen_mode != "unsteered")
                    else None
                ),
            )

        if self._session._monitor and self._session._monitor.has_pending_data():
            self._session._monitor.consume_pending()
            current, previous = self._session._monitor.get_current_and_previous()
            sparklines = {name: self._session._monitor.get_sparkline(name)
                          for name in self._session._monitor.probe_names}
            self._trait_panel.update_values(
                current, previous, sparklines,
            )

        # Manifold-monitor pending-flag pass — mirrors the vector path
        # so a per-token reading flushed by the engine clears the
        # pending bit on the next poll tick.  Aggregates land via
        # ``_pull_manifold_aggregates`` at finalize; this drain is the
        # mid-gen "tick" so the bit doesn't stall forever.
        mm = getattr(self._session, "manifold_monitor", None)
        if mm is not None and mm.has_pending_per_token():
            mm.consume_pending_per_token()

    # -- Actions --

    def action_remove_vector(self) -> None:
        if self._ab_shadow_active:
            return
        if self._focused_panel_idx == _TRAIT:
            self._remove_selected_probe()
            return
        lp = self._left_panel
        sel = lp.get_selected()
        if sel:
            name = sel["name"]
            if sel.get("kind") == "manifold":
                # Manifold rows aren't session-registered profiles —
                # drop the local term and let the next gen rebuild
                # ``Steering`` without it.
                self._manifold_terms.pop(name, None)
                self._enabled.pop(name, None)
                self._refresh_left_panel()
                return
            self._session.unsteer(name)
            self._alphas.pop(name, None)
            self._enabled.pop(name, None)
            self._refresh_left_panel()

    def _remove_selected_probe(self) -> None:
        tp = self._trait_panel
        probe_name = tp.get_selected_probe()
        if not probe_name or not self._session._monitor:
            return
        self._session.unprobe(probe_name)
        tp.set_active_probes(set(self._session._monitor.probe_names))
        self._refresh_trait_why()

    def action_toggle_vector(self) -> None:
        if self._ab_shadow_active:
            return
        lp = self._left_panel
        sel = lp.get_selected()
        if sel:
            name = sel["name"]
            self._enabled[name] = not self._enabled.get(name, True)
            self._refresh_left_panel()

    def _adjust_alpha(self, delta: float) -> None:
        if self._ab_shadow_active:
            return
        lp = self._left_panel
        sel = lp.get_selected()
        if sel:
            if sel.get("kind") == "manifold":
                # Manifold rows carry a fixed position, not a scalar
                # alpha — ←/→ has nothing to nudge.
                return
            name = sel["name"]
            self._alphas[name] = max(-MAX_ALPHA, min(MAX_ALPHA, self._alphas.get(name, 0.0) + delta))
            self._refresh_left_panel()

    def action_cycle_highlight_mode(self, direction: int = 1) -> None:
        """Step the highlight target through ``{off, probe, surprise}``.

        Three-state cycle to cover the logit-pass parity gap with the
        webui's highlight dropdown.  ``probe`` mode defers to whatever
        the trait panel currently has selected; ``surprise`` pins
        :data:`chat_panel.SURPRISE_PROBE` so the markup builder reads
        chosen-token logprobs instead.

        Ctrl+Y advances forward; Ctrl+Shift+Y walks backward through the
        same order.  The current mode shows as the persistent ``HL`` line
        in the left panel's GENERATION block (refreshed by
        ``_apply_highlight_to_all``) rather than a transient chat message.
        ``probe`` is skipped when no probe is selectable (trait panel
        empty and no stored seed) so the cycle collapses to
        ``{off ↔ surprise}`` rather than landing in a half-state.
        """
        from saklas.tui.chat_panel import SURPRISE_PROBE

        if self._ab_shadow_active:
            return

        # Resolve current mode by inspecting state, not a stored enum —
        # ``/probe`` can land us in any of these three shapes between
        # Ctrl+Y presses, and we want the cycle to advance from
        # wherever the user actually is.
        if not self._highlighting:
            current = "off"
        elif self._highlight_probe == SURPRISE_PROBE:
            current = "surprise"
        else:
            current = "probe"

        # Resolve the probe slot's anchor once up front — if it's None
        # the slot is unreachable and we'll skip it during the walk.
        seed: str | None = self._trait_panel.get_selected_probe()
        if seed is None and self._highlight_probe not in (None, SURPRISE_PROBE):
            seed = self._highlight_probe

        order = ["off", "probe", "surprise"]
        step = 1 if direction >= 0 else -1
        idx = order.index(current)
        # Walk at most ``len(order)`` slots forward, skipping ``probe``
        # when no anchor exists.  Bounded loop = no recursion = no risk
        # of stack growth even on misconfigured state.
        for _ in range(len(order)):
            idx = (idx + step) % len(order)
            candidate = order[idx]
            if candidate == "probe" and seed is None:
                continue  # skip — nothing to anchor to
            next_mode = candidate
            break
        else:
            # All slots were unreachable — should never fire (``off`` is
            # always reachable), but stay graceful.
            return

        if next_mode == "off":
            self._highlighting = False
        elif next_mode == "surprise":
            self._highlight_probe = SURPRISE_PROBE
            self._highlighting = True
        else:  # "probe" — ``seed`` is guaranteed non-None here
            self._highlight_probe = seed
            self._highlighting = True
        # No system message — ``_apply_highlight_to_all`` refreshes the
        # persistent HL line in the left panel's GENERATION block, which
        # is a quieter and always-visible readout of the current mode.
        self._apply_highlight_to_all()

    def action_cycle_highlight_mode_back(self) -> None:
        """Convenience: backward variant of the cycle.  Same code path
        with ``direction=-1`` so the cycle's "skip" branch behaves
        symmetrically when probes aren't loaded."""
        self.action_cycle_highlight_mode(direction=-1)

    def _apply_highlight_to_all(self) -> None:
        from saklas.tui.chat_panel import SURPRISE_PROBE
        # Navigating the trait panel updates the seed live while
        # highlight is on — but ONLY when the active highlight is a
        # probe.  Surprise mode pins ``_highlight_probe`` to the
        # sentinel so the markup builder reads logprobs; without this
        # guard, the next trait-panel arrow keystroke silently flips
        # the highlight back to a probe (latent bug from the Phase 3
        # pass — surprise mode would survive only until the user
        # touched the trait panel).
        if self._highlighting and self._highlight_probe != SURPRISE_PROBE:
            nav_probe = self._trait_panel.get_selected_probe()
            if nav_probe is not None:
                self._highlight_probe = nav_probe
        probe = self._highlight_probe if self._highlighting else None
        # Prune any unmounted widgets (rewind/clear may have detached them).
        self._assistant_messages = [w for w in self._assistant_messages if w.is_mounted]
        for widget in self._assistant_messages:
            widget.apply_highlight(self._highlighting, probe)
        if self._raw_mode:
            self._chat_panel.raw_buffer.apply_highlight(self._highlighting, probe)
        # Refresh the persistent HL readout in the left panel — this is
        # the funnel every highlight-state mutation passes through
        # (cycle, /probe, /unprobe, trait-panel nav), so the line stays
        # in sync without each call site remembering to update it.
        if not self._highlighting:
            hl_label = "off"
        elif self._highlight_probe == SURPRISE_PROBE:
            hl_label = "surprise"
        else:
            hl_label = self._highlight_probe or "off"
        self._left_panel.update_highlight(hl_label)

    def _set_widget_token_data(
        self, widget: _AssistantMessage,
        response_token_strs: list[str],
        response_probe_scores: dict[str, list[float]],
        thinking_token_strs: list[str],
        thinking_probe_scores: dict[str, list[float]],
    ) -> None:
        if not widget.is_mounted:
            return
        widget.set_token_data(
            response_token_strs, response_probe_scores,
            thinking_token_strs, thinking_probe_scores,
        )
        if self._highlighting:
            widget.apply_highlight(True, self._highlight_probe)

    def _finalize_widget_highlight(self, widget: _AssistantMessage) -> None:
        """Pull per-token scores the session stashed during finalize and
        push to the widget for highlight-mode overlays.

        The session's ``per_token_scores`` are indexed in ``generated_ids``
        space (one score per forward-pass step, including delimiters and
        preamble tokens that were suppressed from the on_token stream).
        ``gen_state.emit_map`` records which ``generated_ids`` index
        corresponds to each emitted token, letting us project scores into
        the widget's token-string space without re-decoding.
        """
        per_token = self._session.last_per_token_scores
        if not per_token or not widget.is_mounted:
            return
        emit_map = self._session._gen_state.emit_map
        if not emit_map:
            return

        # Use the widget's own streamed token strings — these match exactly
        # what was rendered, avoiding batch_decode mismatches.
        response_strs = list(widget._streamed_response_tokens)
        thinking_strs = list(widget._streamed_thinking_tokens)

        # Project scores from generated_ids space to emitted-token space.
        thinking_scores: dict[str, list[float]] = {k: [] for k in per_token}
        response_scores: dict[str, list[float]] = {k: [] for k in per_token}
        think_i = 0
        resp_i = 0
        for gen_idx, is_thinking in emit_map:
            if is_thinking:
                if think_i < len(thinking_strs):
                    for k, scores in per_token.items():
                        if gen_idx < len(scores):
                            thinking_scores[k].append(scores[gen_idx])
                    think_i += 1
            else:
                if resp_i < len(response_strs):
                    for k, scores in per_token.items():
                        if gen_idx < len(scores):
                            response_scores[k].append(scores[gen_idx])
                    resp_i += 1

        widget.set_token_data(
            response_strs, response_scores,
            thinking_strs, thinking_scores,
        )
        if self._highlighting:
            widget.apply_highlight(True, self._highlight_probe)
        self._refresh_trait_why()
        self._pull_manifold_aggregates()

    def _pull_manifold_aggregates(self) -> None:
        """Push end-of-gen manifold aggregates to the trait panel.

        Reads ``session.last_result.manifold_readings`` — populated by
        ``ManifoldMonitor.score_aggregate`` in ``_finalize_generation``
        when at least one probe is attached.  No-op when no result is
        cached or no probes are attached.
        """
        last = getattr(self._session, "last_result", None)
        if last is None:
            return
        aggregates = getattr(last, "manifold_readings", None) or {}
        if not aggregates and not self._session.manifold_monitor.probe_names:
            return
        self._trait_panel.update_manifold_readings(aggregates=aggregates)

    # -- New slash command handlers --

    def _handle_alpha(self, arg: str) -> None:
        chat = self._chat_panel
        try:
            tokens = shlex.split(arg)
        except ValueError as e:
            chat.add_system_message(f"Parse error: {e}")
            return
        if len(tokens) != 2:
            chat.add_system_message("Usage: /alpha <value> <name>")
            return
        val_str, raw = tokens
        matches = _resolve_active_name(raw, self._alphas)
        if len(matches) == 0:
            chat.add_system_message(
                f"'{raw}' is not active. Use /steer to add it first."
            )
            return
        if len(matches) > 1:
            chat.add_system_message(
                f"'{raw}' is ambiguous: {', '.join(matches)}"
            )
            return
        try:
            val = float(val_str)
        except ValueError:
            chat.add_system_message(f"Invalid alpha: {val_str}")
            return
        name = matches[0]
        if name != raw:
            # Sign flip when the user typed the negative pole.
            from saklas.core.session import BIPOLAR_SEP, canonical_concept_name
            slug = canonical_concept_name(raw)
            if BIPOLAR_SEP in name:
                _pos, neg = name.split(BIPOLAR_SEP, 1)
                if slug == neg:
                    val = -val
        val = max(-MAX_ALPHA, min(MAX_ALPHA, val))
        self._alphas[name] = val
        self._refresh_left_panel()
        chat.add_system_message(f"Alpha for '{name}' set to {val:+.2f}")

    def _handle_unsteer(self, arg: str) -> None:
        chat = self._chat_panel
        raw = arg.strip()
        if not raw:
            chat.add_system_message("Usage: /unsteer <name>")
            return
        ns = _detect_namespace_selector(raw)
        if ns is not None:
            self._handle_unsteer_namespace(ns)
            return
        # Resolve against both racks — scalar vectors (``_alphas``) and
        # manifold terms (``_manifold_terms``).  A ``/steer`` with a
        # ``%`` term lands in ``_manifold_terms``; without this the
        # slash command would report a racked manifold as "not active".
        matches = _resolve_active_name(
            raw, [*self._alphas.keys(), *self._manifold_terms.keys()]
        )
        if len(matches) == 0:
            chat.add_system_message(f"'{raw}' is not active.")
            return
        if len(matches) > 1:
            chat.add_system_message(f"'{raw}' is ambiguous: {', '.join(matches)}")
            return
        name = matches[0]
        if name in self._manifold_terms:
            # Manifold rows aren't session-registered profiles — drop the
            # local term and let the next gen rebuild ``Steering``
            # without it (mirrors the panel backspace/delete path).
            self._manifold_terms.pop(name, None)
            self._enabled.pop(name, None)
            self._refresh_left_panel()
            chat.add_system_message(f"Removed manifold '{name}'.")
            return
        self._session.unsteer(name)
        self._alphas.pop(name, None)
        self._enabled.pop(name, None)
        self._refresh_left_panel()
        chat.add_system_message(f"Removed '{name}'.")

    def _handle_unprobe(self, arg: str) -> None:
        chat = self._chat_panel
        raw = arg.strip()
        if not raw:
            chat.add_system_message("Usage: /unprobe <name>")
            return
        ns = _detect_namespace_selector(raw)
        if ns is not None:
            self._handle_unprobe_namespace(ns)
            return
        monitor = self._session._monitor
        if not monitor:
            chat.add_system_message(f"Probe '{raw}' not active.")
            return
        matches = _resolve_active_name(raw, monitor.probe_names)
        if len(matches) == 0:
            chat.add_system_message(f"Probe '{raw}' not active.")
            return
        if len(matches) > 1:
            chat.add_system_message(f"'{raw}' is ambiguous: {', '.join(matches)}")
            return
        name = matches[0]
        self._session.unprobe(name)
        self._trait_panel.set_active_probes(set(monitor.probe_names))
        if self._highlight_probe == name:
            self._highlight_probe = None
            self._highlighting = False
            self._apply_highlight_to_all()
        self._refresh_trait_why()
        chat.add_system_message(f"Probe '{name}' removed.")

    # -- Namespace bulk handlers (/steer ns/, /probe ns/, /unsteer ns/, /unprobe ns/) --

    def _bulk_autoload_namespace(self, ns: str) -> tuple[list[str], list[str]]:
        """Autoload every concept in ``ns`` whose tensor is on disk.

        Returns ``(loaded, skipped)`` lists of namespace-qualified names.
        ``loaded`` is what landed in ``session._profiles`` (already-present
        plus freshly loaded); ``skipped`` is concepts whose ``raw`` variant
        isn't extracted for the current model. Cache-hit only — no PCA,
        no scenario gen, no network. Worker-thread safe (only touches
        ``session._profiles`` and the on-disk pack files).
        """
        from saklas.io.selectors import _all_concepts

        loaded: list[str] = []
        skipped: list[str] = []
        concepts = [c for c in _all_concepts() if c.namespace == ns]
        for c in concepts:
            key = f"{ns}/{c.name}"
            if key in self._session._profiles:
                loaded.append(key)
                continue
            try:
                self._session._ensure_profile_registered(key)
            except SaklasError:
                # Unresolved / not-yet-fit concepts surface to the user
                # below by leaving the concept in ``skipped``; the
                # detailed message would drown out the bulk summary.
                pass
            except Exception:
                pass
            if key in self._session._profiles:
                loaded.append(key)
            else:
                skipped.append(key)
        return loaded, skipped

    def _bulk_skip_message(self, ns: str, skipped: list[str]) -> str:
        """Two-line note for skipped concepts: list + one-line refresh hint."""
        return (
            f"Skipped {len(skipped)} not yet extracted for this model: "
            f"{', '.join(sorted(skipped))}\n"
            f"  Run `saklas pack refresh {ns} -m <model>` to extract."
        )

    def _handle_steer_namespace(self, ns: str) -> None:
        """Bulk-register every cached concept under ``ns/`` as a steering
        vector with α = ``DEFAULT_ALPHA`` and ``enabled=False`` so users
        can flip them on individually from the left panel.
        """
        chat = self._chat_panel
        if self._ab_shadow_active:
            chat.add_system_message("Cannot modify vectors during A/B shadow gen.")
            return
        if self._is_busy:
            arg = f"{ns}/"
            self._enqueue_pending(PendingItem("steer", f"/steer {arg}", (arg,)))
            return

        from saklas.io.selectors import _all_concepts
        if not [c for c in _all_concepts() if c.namespace == ns]:
            chat.add_system_message(f"No concepts installed under '{ns}/'.")
            return

        chat.add_system_message(f"Loading '{ns}/' vectors (toggled off)...")

        def _worker() -> None:
            loaded, skipped = self._bulk_autoload_namespace(ns)

            def _finish() -> None:
                from saklas.core.profile import Profile as _Profile
                for key in loaded:
                    profile = self._session._profiles.get(key)
                    if profile is None:
                        continue
                    # _profiles stores dict[int, Tensor]; steer() expects Profile.
                    self._session.steer(key, _Profile(profile))
                    self._alphas[key] = DEFAULT_ALPHA
                    self._enabled[key] = False
                self._refresh_left_panel()
                lines = [
                    f"Bulk steer '{ns}/': "
                    f"added {len(loaded)} vector(s) (toggled off)."
                ]
                if skipped:
                    lines.append(self._bulk_skip_message(ns, skipped))
                chat.add_system_message("\n".join(lines))

            self.call_from_thread(_finish)

        self.run_worker(_worker, thread=True)

    def _handle_probe_namespace(self, ns: str) -> None:
        """Bulk-register every cached concept under ``ns/`` as a probe.
        Highlight seeds to the last-loaded probe so the per-token overlay
        lights up immediately, matching the single-probe ``/probe`` UX.
        """
        chat = self._chat_panel
        if self._ab_shadow_active:
            chat.add_system_message("Cannot modify vectors during A/B shadow gen.")
            return
        if self._is_busy:
            arg = f"{ns}/"
            self._enqueue_pending(PendingItem("probe", f"/probe {arg}", (arg,)))
            return

        from saklas.io.selectors import _all_concepts
        if not [c for c in _all_concepts() if c.namespace == ns]:
            chat.add_system_message(f"No concepts installed under '{ns}/'.")
            return

        chat.add_system_message(f"Loading '{ns}/' probes...")

        def _worker() -> None:
            loaded, skipped = self._bulk_autoload_namespace(ns)

            def _finish() -> None:
                for key in loaded:
                    profile = self._session._profiles.get(key)
                    if profile is None:
                        continue
                    self._session.probe(key, profile)
                if loaded and self._session._monitor is not None:
                    self._trait_panel.set_active_probes(
                        set(self._session._monitor.probe_names)
                    )
                    # Seed highlight to the last loaded probe — same UX as
                    # single ``/probe``: the user immediately sees one of
                    # them lit up and can navigate the trait panel to flip.
                    self._highlight_probe = sorted(loaded)[-1]
                    self._highlighting = True
                    self._apply_highlight_to_all()
                    self._refresh_trait_why()
                lines = [f"Bulk probe '{ns}/': added {len(loaded)} probe(s)."]
                if loaded:
                    lines.append("  Highlight on (⌃Y to cycle).")
                if skipped:
                    lines.append(self._bulk_skip_message(ns, skipped))
                chat.add_system_message("\n".join(lines))

            self.call_from_thread(_finish)

        self.run_worker(_worker, thread=True)

    def _handle_unsteer_namespace(self, ns: str) -> None:
        """Remove every active steering vector whose registry key sits
        under ``ns/``. Mirrors the single-vector ``/unsteer`` no-defer
        policy — modifying ``_profiles`` mid-gen doesn't disturb hooks
        already attached to the in-flight forward pass.
        """
        chat = self._chat_panel
        prefix = f"{ns}/"
        matches = [n for n in list(self._alphas.keys()) if n.startswith(prefix)]
        # Manifold terms share the ``ns/name`` key shape — sweep them too.
        manifold_matches = [
            n for n in list(self._manifold_terms.keys()) if n.startswith(prefix)
        ]
        if not matches and not manifold_matches:
            chat.add_system_message(f"No active vectors under '{ns}/'.")
            return
        for name in matches:
            self._session.unsteer(name)
            self._alphas.pop(name, None)
            self._enabled.pop(name, None)
        for name in manifold_matches:
            self._manifold_terms.pop(name, None)
            self._enabled.pop(name, None)
        self._refresh_left_panel()
        total = len(matches) + len(manifold_matches)
        chat.add_system_message(
            f"Removed {total} vector(s) from '{ns}/'."
        )

    def _handle_unprobe_namespace(self, ns: str) -> None:
        """Remove every active probe whose registry key sits under ``ns/``.
        Clears the highlight seed when it points into the namespace.
        """
        chat = self._chat_panel
        monitor = self._session._monitor
        if monitor is None:
            chat.add_system_message(f"No active probes under '{ns}/'.")
            return
        prefix = f"{ns}/"
        matches = [n for n in list(monitor.probe_names) if n.startswith(prefix)]
        if not matches:
            chat.add_system_message(f"No active probes under '{ns}/'.")
            return
        for name in matches:
            self._session.unprobe(name)
        self._trait_panel.set_active_probes(set(monitor.probe_names))
        if self._highlight_probe is not None and self._highlight_probe.startswith(prefix):
            self._highlight_probe = None
            self._highlighting = False
            self._apply_highlight_to_all()
        self._refresh_trait_why()
        chat.add_system_message(
            f"Removed {len(matches)} probe(s) from '{ns}/'."
        )

    # -- Input history (↑/↓ in chat input) --

    def _push_input_history(self, text: str) -> None:
        """Append a freshly-submitted line to the recall ring.

        De-dupes against the *immediately preceding* entry (readline /
        bash semantics — repeated identical lines collapse, but a
        ping-pong A→B→A still records both A's). Resets the recall
        cursor so the next ↑ starts at the bottom of the ring.
        """
        text = text.rstrip()
        if not text:
            return
        if self._input_history and self._input_history[-1] == text:
            self._history_index = None
            self._history_stash = ""
            return
        self._input_history.append(text)
        if len(self._input_history) > _INPUT_HISTORY_MAX:
            # Drop oldest in one slice rather than calling pop(0) per
            # overflow — slice is O(N) but only fires once per overflow.
            del self._input_history[: len(self._input_history) - _INPUT_HISTORY_MAX]
        self._history_index = None
        self._history_stash = ""

    def _history_navigate(self, delta: int) -> None:
        """Walk the combined ring of pending items + input history.

        Pending items come first (most-recently-queued is one ``↑`` from
        live, oldest pending is one further ``↑``), then committed
        input history (newest first).  ``↓`` walks the same ring in
        reverse and clears the cursor when it returns to the live
        slot, restoring the stash captured on the first ``↑``.

        Pulling a pending slot via this walk sets ``_pulled_slot`` —
        which makes the next ``Enter`` *replace* that slot rather than
        append (slot-preserving edit) and the next ``Esc`` cancel
        the pull (slot stays as-is).  An empty ``Enter`` while a
        slot is pulled *removes* that slot (the keyboard equivalent
        of the GUI's per-bubble ``×``).

        The chat input is a multi-line :class:`ChatInput` (TextArea),
        so the recalled text replaces the entire buffer via
        ``load_text`` and the cursor lands at the end of the last line.
        """
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            return

        n_pending = len(self._pending_queue)
        n_history = len(self._input_history)
        if n_pending == 0 and n_history == 0:
            return

        # Compose a position in the combined ring:
        #   pos in [0, n_pending) — pending slot (n_pending-1-pos counts
        #     back from the queue tail, so pos=0 is the most-recent
        #     pending item — i.e. the first ``↑`` destination).
        #   pos in [n_pending, n_pending + n_history) — history offset
        #     (newest at n_pending, oldest at the end).
        #   pos == -1 — sentinel for "live slot, no cursor."
        cur_pos = self._current_input_cursor_pos()
        if cur_pos < 0:
            if delta > 0:
                return  # Already at live slot — ``↓`` is a no-op.
            # First ``↑`` from live — stash whatever the user typed so
            # ``↓`` past the newest entry restores it.
            self._history_stash = inp.text
            new_pos = 0
        else:
            # ``↑`` (delta<0) walks toward older items — increment the
            # ring position.  ``↓`` (delta>0) walks toward newer items
            # and eventually back to the live slot — decrement.
            new_pos = cur_pos + (1 if delta < 0 else -1)
            if new_pos >= n_pending + n_history:
                # Past the oldest history entry — pin to it; matches
                # readline (no wrap, no error).
                new_pos = n_pending + n_history - 1
            elif new_pos < 0:
                # Past the newest pending item / newest history entry —
                # back to live.  Restore the stash and reset the cursor.
                self._pulled_slot = None
                self._history_index = None
                self._set_input_text(inp, self._history_stash)
                self._history_stash = ""
                self._sync_pull_state()
                return

        # Apply the new position.
        if new_pos < n_pending:
            slot = n_pending - 1 - new_pos
            self._pulled_slot = slot
            self._history_index = None
            self._set_input_text(inp, self._pending_queue[slot].text)
        else:
            self._pulled_slot = None
            self._history_index = (n_pending + n_history - 1) - new_pos
            self._set_input_text(inp, self._input_history[self._history_index])
        self._sync_pull_state()

    def _current_input_cursor_pos(self) -> int:
        """Return the combined-ring position of the current cursor.

        ``-1`` = live slot.  See :meth:`_history_navigate` for the
        encoding of pending and history positions.
        """
        n_pending = len(self._pending_queue)
        if self._pulled_slot is not None and 0 <= self._pulled_slot < n_pending:
            return n_pending - 1 - self._pulled_slot
        if self._history_index is not None:
            n_history = len(self._input_history)
            if 0 <= self._history_index < n_history:
                return n_pending + (n_history - 1 - self._history_index)
        return -1

    def _cancel_pull(self) -> None:
        """``Esc`` while a pending slot is pulled — restore the live stash.

        Leaves the slot in the queue untouched; the user backed out
        of the edit but didn't cancel the queued action.
        """
        if self._pulled_slot is None:
            return
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            self._pulled_slot = None
            self._sync_pull_state()
            return
        self._pulled_slot = None
        self._history_index = None
        self._set_input_text(inp, self._history_stash)
        self._history_stash = ""
        self._sync_pull_state()

    def _sync_pull_state(self) -> None:
        """Reflect ``_pulled_slot`` into the chat input + pending strip.

        Two things follow from a pull change:

        * :attr:`ChatInput.allow_empty_submit` flips on while pulled
          so an empty ``Enter`` reaches the dispatcher as the slot-
          cancel gesture (mirrors the GUI's per-bubble ``×``).
        * The :class:`PendingStrip` re-renders so the ``✎`` editing
          marker tracks the currently-pulled slot.

        Single helper called from every site that mutates
        ``_pulled_slot`` so the two derived surfaces stay in lockstep.
        """
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            pass
        else:
            inp.allow_empty_submit = self._pulled_slot is not None
        self._refresh_pending_strip()

    @staticmethod
    def _set_input_text(inp: ChatInput, text: str) -> None:
        """Replace the chat input's content and park the cursor at the
        end of the last line — the equivalent of ``Input.value = text;
        cursor_position = len(value)`` for the TextArea-backed input.
        """
        inp.load_text(text)
        last_row = inp.document.line_count - 1
        last_col = len(inp.document.get_line(last_row))
        inp.cursor_location = (last_row, last_col)

    def _handle_seed(self, arg: str) -> None:
        chat = self._chat_panel
        arg = arg.strip()
        if not arg:
            chat.add_system_message(f"Seed: {self._default_seed}")
            return
        if arg.lower() == "clear":
            self._default_seed = None
            chat.add_system_message("Seed cleared.")
            return
        try:
            self._default_seed = int(arg)
            chat.add_system_message(f"Seed set to {self._default_seed}")
        except ValueError:
            chat.add_system_message("Invalid seed value (expected int or 'clear').")

    def _conv_dir(self) -> Path:
        d = saklas_home() / "conversations"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _handle_save(self, arg: str) -> None:
        """`/save <name>` — write the full loom tree to disk.

        Serializes the entire tree — every branch, not just the active
        path — to ``~/.saklas/conversations/<name>.json`` via
        :meth:`saklas.core.loom.LoomTree.save`.  ``/load <name>``
        restores it.  Per-token highlight scores are not persisted
        (see ``LoomTree.to_dict``); structure, text, and recipes are.
        """
        chat = self._chat_panel
        name = arg.strip()
        if not name:
            chat.add_system_message("Usage: /save <name>")
            return
        path = self._conv_dir() / f"{name}.json"
        try:
            self._session.tree.save(path)
        except Exception as e:
            chat.add_system_message(f"/save failed: {e}")
            return
        chat.add_system_message(
            f"saved tree → {path} ({len(self._session.tree.nodes)} nodes)"
        )

    def _handle_load(self, arg: str) -> None:
        """`/load <name>` — replace the loom tree with a saved one.

        Loads ``~/.saklas/conversations/<name>.json`` (written by
        ``/save``) and swaps it in wholesale — all branches restored.
        The deserialized tree has no event bus or conflict-check hook,
        so both are rewired here.  The chat log is repainted along the
        loaded tree's active path immediately (``Ctrl+L`` inspects the
        full tree).
        """
        from saklas.core.loom import LoomTree, LoomTreeError

        chat = self._chat_panel
        name = arg.strip()
        if not name:
            chat.add_system_message("Usage: /load <name>")
            return
        path = self._conv_dir() / f"{name}.json"
        if not path.is_file():
            chat.add_system_message(f"no saved tree at {path}")
            return
        try:
            loaded = LoomTree.load(path)
        except LoomTreeError as e:
            chat.add_system_message(f"/load failed: {e}")
            return
        except Exception as e:
            chat.add_system_message(f"/load failed to read: {e}")
            return

        # The deserialized tree carries no event bus / conflict hook.
        loaded.attach_events(self._session.events)
        loaded.set_conflict_check(self._session._loom_conflict_check)
        live_model = self._session._model_info.get("model_id")
        if loaded.model_id is None:
            loaded.model_id = live_model
        elif live_model is not None and loaded.model_id != live_model:
            chat.add_system_message(
                f"⚠ /load: tree saved on '{loaded.model_id}', running "
                f"'{live_model}' — steering/probe vectors may not match"
            )
        self._session.tree = loaded
        if self._session._monitor is not None:
            self._session._monitor.reset_history()

        # Repaint the chat log along the loaded tree's active path so the
        # conversation is visible immediately (loaded nodes carry no
        # per-token scores — see ``_repaint_chat_from_active_path``).
        self._repaint_chat_from_active_path()
        self._refresh_left_panel()
        chat.add_system_message(
            f"loaded tree from {path} ({len(loaded.nodes)} nodes) — "
            f"⌃L to view the tree"
        )

    def _handle_export(self, arg: str) -> None:
        chat = self._chat_panel
        path_str = arg.strip()
        if not path_str:
            chat.add_system_message("Usage: /export <path>")
            return
        collector = ResultCollector()
        last = self._session.last_result
        if last is not None:
            collector.add(last)
        path = Path(path_str).expanduser()
        try:
            collector.to_jsonl(str(path))
            chat.add_system_message(f"Exported {len(collector.results)} result(s) to {path}")
        except Exception as e:
            chat.add_system_message(f"Export error: {e}")

    def _handle_model_info(self) -> None:
        chat = self._chat_panel
        info = self._session._model_info
        lines = [
            f"Model: {info.get('model_id', 'unknown')}",
            f"Arch: {info.get('model_type', 'unknown')}  "
            f"Device: {self._device_str}  "
            f"Layers: {len(self._session._layers)}",
            f"Thinking: {self._thinking_status_str()}",
            f"Active vectors: {list(self._alphas.keys()) or '(none)'}",
            f"Active probes: {list(self._session._monitor.probe_names) if self._session._monitor else '(none)'}",
            f"Seed: {self._default_seed}",
        ]
        chat.add_system_message("\n".join(lines))

    def _refresh_trait_why(self) -> None:
        """Push per-layer ||baked|| norms for the trait-panel-selected probe
        down to the panel's WHY section as a histogram in layer order.

        Per-token highlighting in the chat already surfaces which tokens
        a probe lights up on — no token list duplicated here.
        """
        probe = self._trait_panel.get_selected_probe()
        monitor = self._session._monitor
        if probe is None or monitor is None or probe not in monitor.profiles:
            self._trait_panel.update_why(None, [])
            return
        profile = monitor.profiles[probe]
        layer_norms = sorted(
            ((int(lidx), float(t.norm().item())) for lidx, t in profile.items()),
            key=lambda kv: kv[0],
        )
        self._trait_panel.update_why(probe, layer_norms)

    def _handle_compare(self, arg: str) -> None:
        chat = self._chat_panel
        if not arg:
            chat.add_system_message("Usage: /compare <name> [other_name]")
            return

        parts = arg.split()

        # Gather all available profiles: session profiles + monitor probes.
        # Monitor stores raw ``dict[int, Tensor]`` — wrap it in a Profile so
        # both sources expose the same cosine_similarity API.
        from saklas.core.profile import Profile
        all_profiles: dict[str, Profile] = {}
        for name, prof in self._session._profiles.items():
            if isinstance(prof, Profile):
                all_profiles[name] = prof
            elif isinstance(prof, dict) and prof:
                all_profiles[name] = Profile(prof)
        if self._session._monitor:
            for name, prof in self._session._monitor.profiles.items():
                if name in all_profiles:
                    continue
                if isinstance(prof, Profile):
                    all_profiles[name] = prof
                elif isinstance(prof, dict) and prof:
                    all_profiles[name] = Profile(prof)

        def _resolve(raw: str) -> str | None:
            matches = _resolve_active_name(raw, all_profiles)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                chat.add_system_message(f"'{raw}' is ambiguous: {', '.join(matches)}")
                return None
            chat.add_system_message(f"No profile found for '{raw}'")
            return None

        if len(parts) == 1:
            # 1-arg: ranked comparison against all loaded profiles.
            target_name = _resolve(parts[0])
            if target_name is None:
                return
            target = all_profiles[target_name]
            others = {n: p for n, p in all_profiles.items() if n != target_name}
            if not others:
                chat.add_system_message("No other profiles loaded to compare against.")
                return
            # Mahalanobis-only: ``cosine_similarity`` requires the session
            # whitener (the Euclidean path is gone).  Pairs whose shared
            # layers the whitener doesn't cover raise and are skipped.
            whitener = getattr(self._session, "whitener", None)
            scores = {}
            for name, prof in others.items():
                try:
                    scores[name] = target.cosine_similarity(prof, whitener=whitener)
                except Exception:
                    continue
            if not scores:
                chat.add_system_message("No comparable profiles (no shared layers).")
                return
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            width = max(len(n) for n, _ in ranked)
            lines = [f"{target_name} vs loaded profiles:"]
            for name, score in ranked:
                lines.append(f"  {name:<{width}}  {score:+.4f}")
            chat.add_system_message("\n".join(lines))

        elif len(parts) == 2:
            # 2-arg: pairwise.
            a_name = _resolve(parts[0])
            if a_name is None:
                return
            b_name = _resolve(parts[1])
            if b_name is None:
                return
            # Mahalanobis-only (see the 1-arg branch above).
            whitener = getattr(self._session, "whitener", None)
            try:
                sim = all_profiles[a_name].cosine_similarity(
                    all_profiles[b_name], whitener=whitener,
                )
            except Exception as e:
                chat.add_system_message(f"Compare failed: {e}")
                return
            chat.add_system_message(f"{a_name} ~ {b_name}: {sim:+.4f}")

        else:
            chat.add_system_message("Usage: /compare <name> [other_name]")

    # -- Pending queue --------------------------------------------------

    def _enqueue_pending(
        self,
        item: PendingItem,
        *,
        replace_slot: int | None = None,
    ) -> None:
        """Append ``item`` to the pending queue (or replace a slot in place).

        ``replace_slot`` is the slot the user pulled into the input box
        via the up-arrow walk; passing it keeps a re-submitted edit at
        its original position rather than dropping it to the tail of
        the queue.  Out-of-range values fall back to append.

        After mutation the chat panel's :class:`PendingStrip` is
        refreshed so the visible list stays in sync, and the input-mode
        placeholder is re-derived so a queued role-shifting item
        (``commit_user`` / ``rewind``) flips the next submission's mode
        immediately.
        """
        q = self._pending_queue
        if (
            replace_slot is not None
            and 0 <= replace_slot < len(q)
        ):
            q[replace_slot] = item
        else:
            q.append(item)
        self._refresh_pending_strip()
        self._refresh_input_mode()

    def _remove_pending_slot(self, slot: int) -> None:
        """Drop a pulled slot from the queue (empty-input re-submit / cancel)."""
        q = self._pending_queue
        if 0 <= slot < len(q):
            del q[slot]
            self._refresh_pending_strip()
            self._refresh_input_mode()

    def _refresh_pending_strip(self) -> None:
        """Push the queue + pulled-slot into the chat panel's strip.

        Single funnel so callers don't have to remember to forward
        ``_pulled_slot``; any mutation that touches either the queue
        or the pulled state should call this.
        """
        self._chat_panel.update_pending(
            self._pending_queue, pulled_slot=self._pulled_slot,
        )

    def _drain_next_pending(self) -> bool:
        """Drain queued items in FIFO until an async kind takes over.

        Returns ``True`` when at least one item ran, ``False`` when the
        queue was empty.  Called from :meth:`_poll_generation`'s
        ``done`` branch (and recursively from itself via the chain
        loop).

        Kinds in :data:`_KIND_CHAIN_INLINE` (``clear`` / ``rewind`` /
        ``steer`` / ``probe``) are fully synchronous — drain through
        them inline so a chain of slash commands doesn't stall waiting
        for a ``done`` that will never fire.  Every other kind either
        kicks a generation (its own ``done`` re-enters here) or fires
        a worker that enqueues ``("done", False)`` in its finally
        block (the commit / extract pattern).  Without this loop a
        queued commit_user left the rest of the queue stuck — root
        cause of the "queue stuck after commit" bug.

        Keeps :attr:`_pulled_slot` accounting honest across each pop:
        if the user pulled slot 0 we cancel the pull (the slot they
        were editing is now being dispatched); if they pulled a
        later slot we decrement the index so they keep tracking the
        same item.
        """
        drained = False
        while self._pending_queue:
            if self._pulled_slot is not None:
                if self._pulled_slot == 0:
                    self._cancel_pull()
                else:
                    self._pulled_slot -= 1
            item = self._pending_queue.pop(0)
            self._refresh_pending_strip()
            self._dispatch_pending_action(item)
            drained = True
            if item.kind not in _KIND_CHAIN_INLINE:
                # Async kind — let its own done sentinel drive the
                # next drain so the worker isn't raced.
                break
        return drained

    @property
    def _is_busy(self) -> bool:
        """Engine running, UI still draining ``done``, or queue non-empty.

        Used as the gate at every submission site to decide enqueue vs
        immediate dispatch.  Including the queue catches the case where
        a chain of pending items keeps the engine idle between drains
        — without this gate a fast user could race the next drain and
        scramble the queue order.
        """
        return bool(
            self._session.is_generating
            or self._ui_gen_active
            or self._pending_queue
        )

    def _dispatch_pending_action(self, item: PendingItem) -> None:
        """Handle a queued action dispatched once the current gen finishes."""
        kind = item.kind
        text = item.text
        payload = item.payload
        chat = self._chat_panel
        try:
            if kind == "regenerate":
                if self._raw_mode:
                    self._run_regen_n_worker(1)
                else:
                    self._rewind_active_assistant()
                    chat.rewind_last_assistant()
                    self._start_generation()
            elif kind == "submit":
                # Phase 5 carries the role decision made at submit time
                # so the deferred dispatch matches whatever the user-row
                # mount did.  ``payload[0]`` is the optional
                # ``prefill_target`` node id, or ``ACTIVE_AT_DRAIN`` when
                # the queue role-aware path stamped it for late binding
                # (parent created by an earlier-queued action; resolve
                # the live active here).  Mount the user row here
                # (deferred from queueing) so the row appears alongside
                # the new assistant reply rather than floating above the
                # previous in-flight one.
                target = payload[0] if payload else None
                if target == ACTIVE_AT_DRAIN:
                    target = self._prefill_target_node_id()
                if target is not None:
                    self._start_prefill(target, text)
                else:
                    self._chat_panel.add_user_message(text)
                    self._start_generation(text)
            elif kind == "raw_continue":
                # Raw-mode continuation queued behind an in-flight gen.
                # ``text`` is the full submitted draft; divergence is
                # resolved fresh here so it binds to the current tree.
                tail, parent = self._resolve_raw_divergence(text)
                self._start_generation(
                    tail, raw_continuation=True,
                    raw_draft=text, raw_parent=parent,
                )
            elif kind == "raw_commit":
                self._start_raw_commit(text)
            elif kind == "commit_user":
                # Ctrl+Enter from a non-user active node, queued behind
                # in-flight gen.  The role decision was made at submit
                # time so we don't re-resolve it here (the active node
                # may have shifted during gen).
                self._start_commit_user(text)
            elif kind == "commit_assistant":
                # Ctrl+Enter from a user node, queued behind in-flight
                # gen.  ``payload[0]`` is the user node id, or
                # ``ACTIVE_AT_DRAIN`` for queue role-aware deferral.
                parent = payload[0]
                if parent == ACTIVE_AT_DRAIN:
                    parent = self._prefill_target_node_id()
                if parent is None:
                    # Predicted user node never landed (earlier queued
                    # item failed or was cancelled).  Fall through to
                    # commit_user so the text isn't silently dropped.
                    self._start_commit_user(text)
                else:
                    self._start_commit_assistant(parent, text)
            elif kind == "clear":
                self._do_clear()
            elif kind == "rewind":
                self._rewind_active_assistant()
                chat.rewind_last_assistant()
                self._do_rewind()
            elif kind == "steer":
                # ``text`` is the canonical slash form (``/steer …``);
                # ``payload[0]`` is the raw arg string the handler
                # actually consumes.
                self._handle_steer(payload[0] if payload else text)
            elif kind == "probe":
                self._handle_probe(payload[0] if payload else text)
            elif kind == "extract":
                self._handle_extract_only(payload[0] if payload else text)
            elif kind == "manifold_fit":
                # ``payload[0]`` is the folder path; the fit runs on a
                # worker that enqueues its own ``done`` sentinel.
                self._start_manifold_fit(payload[0] if payload else text)
            elif kind == "manifold_probe":
                # ``payload[0]`` is the selector; attach runs on a
                # worker that enqueues its own ``done`` sentinel.
                self._start_manifold_probe_attach(
                    payload[0] if payload else text,
                )
            elif kind == "regen_n":
                # N-way regen after an interrupting gen completes; phase
                # 1's engine serializes via ``session.generate(n=N)``.
                # ``payload = (n, mode_or_None)``.
                n = payload[0]
                mode = payload[1] if len(payload) > 1 else None
                if mode is not None:
                    self._run_regen_modifier_worker(n, mode)
                else:
                    self._run_regen_n_worker(n)
            elif kind == "fan":
                # ``payload = (vector, alphas, prompt)``.
                vector, alphas, prompt = payload
                self._run_fan_worker(vector, alphas, prompt)
            elif kind == "quit":
                self.exit()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import sys
            import traceback
            traceback.print_exc(file=sys.stderr)
            self._ui_gen_active = False
            self._pending_queue.clear()
            self._pulled_slot = None
            self._sync_pull_state()
            self._current_assistant_widget = None
            chat.add_system_message(f"error dispatching {kind}: {e}")

    def action_toggle_thinking(self) -> None:
        if not self._supports_thinking:
            self._chat_panel.add_system_message("This model does not support thinking mode.")
            return
        if not self._thinking_optional:
            self._chat_panel.add_system_message("This model always thinks.")
            return
        self._thinking = not self._thinking
        self._refresh_gen_config()

    def _adjust_config(self, attr: str, delta: float, lo: float, hi: float) -> None:
        if self._focused_panel_idx != _LEFT:
            return
        val = getattr(self._session.config, attr)
        new_val = round(max(lo, min(hi, val + delta)), 2)
        self._session.config = replace(self._session.config, **{attr: new_val})
        self._refresh_gen_config()

    def action_temp_down(self) -> None:
        self._adjust_config("temperature", -0.05, 0.0, float("inf"))

    def action_temp_up(self) -> None:
        self._adjust_config("temperature", 0.05, 0.0, float("inf"))

    def action_top_p_down(self) -> None:
        self._adjust_config("top_p", -0.05, 0.0, 1.0)

    def action_top_p_up(self) -> None:
        self._adjust_config("top_p", 0.05, 0.0, 1.0)

    def action_regenerate(self) -> None:
        if self._raw_mode:
            if self._is_busy:
                self._enqueue_pending(PendingItem("regenerate", "regen"))
                return
            self._run_regen_n_worker(1)
            return
        if not self._messages:
            return
        if self._is_busy:
            # Queue the regen — runs after current gen + any earlier
            # pending items finish.  Hit ``Esc`` first to short-circuit.
            self._enqueue_pending(PendingItem("regenerate", "regen"))
            return
        # Loom: move active up so the next gen creates a sibling under
        # the user-parent rather than a child of the old assistant.
        self._rewind_active_assistant()
        self._start_generation()

    def action_ab_compare(self) -> None:
        """Toggle the persistent A/B two-column layout.

        Mirrors the webui's ``abState.enabled`` toggle: turning it on
        reveals each turn's shadow column (steered on the left, unsteered
        on the right) and dispatches a backfill shadow gen for the most
        recent assistant turn that doesn't already have one — exactly
        matching the webui's "play the conversation back to the unsteered
        agent" affordance.

        Phase 5 (per plan decision 13): the same toggle also flips
        :attr:`_loom_auto_regen_on`.  Auto-regen mode defaults to
        ``unsteered`` so existing A/B users see no behavior change; the
        ``unsteered`` mode is served by the existing shadow-gen path
        (no need to fire a redundant ``regen_with_modifier`` worker).
        Other modes (``inverted`` / ``reseed`` / ``cool`` / ``hot`` /
        ``custom``) get the post-gen hook in ``_poll_generation``.

        Toggling off doesn't kill an in-flight shadow gen — the data is
        kept and stays harmless when the column is hidden.  Toggling back
        on re-reveals it without re-running.
        """
        chat = self._chat_panel
        if self._raw_mode:
            chat.add_system_message(
                "A/B compare is a chat-mode layout — switch with /mode."
            )
            return
        was_off = not self._ab_mode
        self._ab_mode = not self._ab_mode
        self._loom_auto_regen_on = self._ab_mode
        chat.set_ab_mode(self._ab_mode)
        chat.add_system_message(
            f"A/B mode {'on' if self._ab_mode else 'off'} "
            f"(auto-regen mode={self._render_auto_regen_mode(self._loom_auto_regen_mode)})"
        )
        if not self._ab_mode or not was_off:
            return
        # Toggling on: backfill the latest assistant turn without a
        # shadow.  Skipped while a generation is in flight — the steered
        # ``done`` will fire its own shadow when it lands.
        if self._session.is_generating or self._ui_gen_active:
            return
        pending = chat.assistant_rows_pending_shadow()
        if not pending:
            return
        self._start_shadow_generation(pending[-1])

    def _build_shadow_messages(
        self, row: _TurnRow,
    ) -> list[dict[str, str]] | None:
        """Reconstruct the conversation up to (but not including) ``row``'s
        steered response, as a messages list to feed an unsteered shadow
        gen.  Mirrors ``_buildShadowMessages`` in the webui store: walks
        all turn-rows that come before ``row`` in mount order, projecting
        each into ``{"role": ..., "content": ...}``.

        Returns ``None`` when the slice doesn't end on a user turn — the
        steered response we're pairing against must follow a user prompt
        for the comparison to make sense.
        """
        chat = self._chat_panel
        if chat._log is None:
            return None
        out: list[dict[str, str]] = []
        for child in chat._log.children:
            if child is row:
                break
            if not isinstance(child, _TurnRow):
                continue
            if child.kind == "user":
                if child.user_text is not None:
                    out.append({"role": "user", "content": child.user_text})
            elif child.kind == "assistant":
                widget = next(
                    (c for c in child.primary.children
                     if isinstance(c, _AssistantMessage)),
                    None,
                )
                if widget is None:
                    continue
                # Reuse the streamed-token list (matches what was rendered);
                # thinking is excluded so replay through enable_thinking=False
                # is well-formed.
                text = "".join(widget._streamed_response_tokens).lstrip()
                if text:
                    out.append({"role": "assistant", "content": text})
        if not out or out[-1]["role"] != "user":
            return None
        return out

    def _start_shadow_generation(self, row: _TurnRow) -> None:
        """Kick off an unsteered shadow gen that streams into ``row``'s
        shadow column.  Uses the same ``_ui_token_queue`` pipeline as the
        steered branch — the queue items are tagged with ``is_shadow=True``
        so ``_poll_generation`` knows not to roll the gen-stat counters
        and to skip firing a follow-up shadow on its ``done``.
        """
        if self._ab_shadow_active:
            return
        chat = self._chat_panel
        messages = self._build_shadow_messages(row)
        if messages is None:
            chat.add_system_message("A/B: no prior user prompt to replay.")
            return
        widget = chat.start_shadow_message(row)
        self._row_for_widget[id(widget)] = row
        self._assistant_messages.append(widget)
        if self._highlighting:
            widget.apply_highlight(True, self._highlight_probe)
        self._current_assistant_widget = widget
        self._ab_shadow_active = True
        self._ab_shadow_row = row
        self._ui_gen_active = True

        sampling = SamplingConfig(
            temperature=self._session.config.temperature,
            top_p=self._session.config.top_p,
            max_tokens=self._session.config.max_new_tokens,
            seed=self._default_seed,
            # Logit-pass: chosen-token logprob capture so surprise
            # highlighting works in the A/B shadow column too.
            logprobs=0,
        )
        use_thinking = self._thinking

        def _shadow_generate() -> None:
            try:
                stream = self._session.generate_stream(
                    messages,
                    steering=None,
                    sampling=sampling,
                    stateless=True,
                    thinking=use_thinking,
                    live_scores=self._wants_live_probe_scores(),
                )
                for event in stream:
                    self._ui_token_queue.put(
                        ("tok", event.text, event.thinking, event.scores,
                         event.perplexity, event.logprob, widget, True,
                         event.manifold_readings),
                    )
                self._ui_token_queue.put(("finalize", widget, True))
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                self._ui_token_queue.put(("error", msg, True))
            finally:
                if self._session._device.type == "mps":
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass
                self._ui_token_queue.put(("done", True))

        self.run_worker(_shadow_generate, thread=True)

    def action_cycle_sort(self) -> None:
        self._trait_panel.cycle_sort()

    # ----------------------------------------------------------------
    # Phase 4 — loom slash commands + bindings
    # ----------------------------------------------------------------

    def action_open_loom(self) -> None:
        """Ctrl+L — open the loom screen."""
        self._handle_tree("")

    def _handle_tree(self, _arg: str) -> None:
        """`/tree` — push the LoomScreen onto the Textual screen stack.

        Esc on the loom screen pops back to the chat screen.  Mutations
        from the loom screen flow into ``session.tree`` directly; the
        chat screen's `_messages` property (a derived view) picks them
        up on the next render.
        """

        from saklas.tui.loom_screen import LoomScreen

        if self._raw_mode and self._chat_panel.raw_buffer.is_dirty:
            self._chat_panel.add_system_message(
                "Raw buffer has uncommitted edits; press Ctrl+Enter to "
                "commit them, or Enter to continue before opening the tree."
            )
            return
        try:
            self.push_screen(LoomScreen(self))
        except Exception as e:
            self._chat_panel.add_system_message(f"/tree failed: {e}")

    def _handle_nav(self, arg: str) -> None:
        """`/nav <id-prefix>` — navigate active node by ulid prefix.

        Matches any case-insensitive prefix of an existing node id;
        ambiguous prefixes report the candidates.
        """

        from saklas.tui.loom_helpers import resolve_node_prefix

        chat = self._chat_panel
        prefix = arg.strip()
        if not prefix:
            chat.add_system_message("Usage: /nav <id-prefix>")
            return
        if self._raw_mode and chat.raw_buffer.is_dirty:
            chat.add_system_message(
                "Raw buffer has uncommitted edits; press Ctrl+Enter to "
                "commit them, or Enter to continue before navigating."
            )
            return
        match = resolve_node_prefix(self._session.tree, prefix)
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
            self._session.tree.navigate(match.node_id)
        except Exception as e:
            chat.add_system_message(f"navigate failed: {e}")
            return
        self._repaint_chat_from_active_path()
        chat.add_system_message(f"navigated to {match.node_id[:8]}")

    def action_nav_picker(self) -> None:
        """Ctrl+N — phase 4: same as `/tree` so users can pick visually."""
        self._handle_tree("")

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

        chat = self._chat_panel
        if not arg.strip():
            chat.add_system_message("Usage: /edit <text...>")
            return
        target = self._session.tree.active_node_id
        if target == self._session.tree.root_id:
            chat.add_system_message("/edit: active node is the root.")
            return
        try:
            self._session.tree.edit(target, arg)
        except (UnknownNodeError, MutationDuringGenerationError,
                InvalidNodeOperationError, LoomTreeError) as e:
            chat.add_system_message(f"/edit failed: {e}")
            return
        self._sync_raw_buffer_from_tree()
        chat.add_system_message(f"edited {target[:8]}")

    def action_edit_active(self) -> None:
        """Ctrl+E — open the loom screen so the user can edit visually.

        The inline `/edit <text>` form is for one-line replacements;
        the loom screen's `e` binding gives the in-place buffer the
        plan describes.
        """
        self._handle_tree("")

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

        chat = self._chat_panel
        text = arg or ""
        target = self._session.tree.active_node_id
        if target == self._session.tree.root_id:
            chat.add_system_message("/branch: active node is the root.")
            return
        try:
            new_id = self._session.tree.branch(target, text)
        except (UnknownNodeError, MutationDuringGenerationError,
                InvalidNodeOperationError, LoomTreeError) as e:
            chat.add_system_message(f"/branch failed: {e}")
            return
        self._sync_raw_buffer_from_tree()
        chat.add_system_message(f"branched {target[:8]} → {new_id[:8]}")

    def action_branch_active(self) -> None:
        """Ctrl+B — open the loom screen for visual branching."""
        self._handle_tree("")

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

        chat = self._chat_panel
        confirm = (arg or "").strip().lower()
        if confirm != "yes":
            chat.add_system_message(
                "/del: type '/del yes' to delete the active subtree."
            )
            return
        tree = self._session.tree
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
        self._sync_raw_buffer_from_tree()
        chat.add_system_message(
            f"deleted {removed} node(s); active now {new_active_id[:8]}"
        )

    def action_delete_subtree(self) -> None:
        """Ctrl+D — surface the `/del yes` confirm hint.

        The key alone never deletes; it routes through ``_handle_del("")``
        so the user sees the same "type '/del yes' to delete" guard the
        slash command emits. The loom screen's ``d`` binding owns the
        modal-overlay confirm flow; this chat-screen path is the cheaper
        UX of "print hint, let the user confirm by typing".
        """
        self._handle_del("")

    def _handle_star(self, _arg: str) -> None:
        chat = self._chat_panel
        target = self._session.tree.active_node_id
        try:
            node = self._session.tree.get(target)
            self._session.tree.star(target, on=not node.starred)
        except Exception as e:
            chat.add_system_message(f"/star failed: {e}")
            return
        chat.add_system_message(
            f"{'starred' if not node.starred else 'unstarred'} {target[:8]}"
        )

    def _handle_note(self, arg: str) -> None:
        chat = self._chat_panel
        target = self._session.tree.active_node_id
        try:
            self._session.tree.annotate(target, arg or "")
        except Exception as e:
            chat.add_system_message(f"/note failed: {e}")
            return
        chat.add_system_message(f"noted {target[:8]}")

    def _handle_path(self, _arg: str) -> None:
        from saklas.tui.loom_helpers import format_path_summary
        self._chat_panel.add_system_message(
            format_path_summary(self._session.tree)
        )

    def _handle_fan(self, arg: str) -> None:
        """`/fan <vector> <alphas>` — N-way regen with per-sibling alpha override.

        Keep it minimal: token-split on the first
        whitespace, treat everything after as the alpha grid.  The
        webui fan grammar (linspace / range / comma list) is
        shared via :func:`parse_alpha_list`.
        """

        chat = self._chat_panel
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

        chat = self._chat_panel
        prompt = self._last_prompt
        # We need a prompt to regen from; if there isn't one yet, lift
        # it off the active path.
        if not prompt:
            hist = self._messages
            if hist and hist[-1]["role"] == "user":
                prompt = hist[-1]["content"]
        if not prompt:
            chat.add_system_message("/fan: no prior prompt to fan out from.")
            return

        # Queue the fan-out behind any in-flight work; ``_run_fan_worker``
        # is invoked by ``_dispatch_pending_action`` once the queue head
        # is drained.
        if self._is_busy:
            display_text = f"/fan {vector} ({len(alphas)} α)"
            self._enqueue_pending(
                PendingItem("fan", display_text, (vector, alphas, prompt))
            )
            return
        self._run_fan_worker(vector, alphas, prompt)

    def _run_fan_worker(
        self, vector: str, alphas: list[float], prompt: str,
    ) -> None:
        """Actually kick the engine.

        Routes through :meth:`SaklasSession.generate_sweep` so every
        sibling lands under one shared user-turn anchor in the loom tree.
        """

        chat = self._chat_panel
        chat.add_system_message(
            f"/fan {vector} × {len(alphas)} (α: "
            f"{', '.join(f'{a:+.2f}' for a in alphas[:6])}"
            f"{'…' if len(alphas) > 6 else ''})"
        )

        from saklas import SamplingConfig

        def _worker() -> None:
            try:
                tree = self._session.tree
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
                    temperature=self._session.config.temperature,
                    top_p=self._session.config.top_p,
                    max_tokens=self._session.config.max_new_tokens,
                    seed=self._default_seed,
                )
                runset = self._session.generate_sweep(
                    prompt,
                    {vector: [float(a) for a in alphas]},
                    sampling=sampling,
                    stateless=False,
                    parent_node_id=parent_for_sweep,
                )
                node_ids = runset.node_ids
                kept = [nid for nid in node_ids if nid]
                self.call_from_thread(
                    self._chat_panel.add_system_message,
                    f"/fan {vector}: {len(kept)} siblings landed "
                    f"({', '.join(nid[:8] for nid in kept[:6])}"
                    f"{'…' if len(kept) > 6 else ''})",
                )
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                self.call_from_thread(
                    self._chat_panel.add_system_message,
                    f"/fan error: {msg}",
                )

        self.run_worker(_worker, thread=True)

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
        if self._is_busy:
            mode_tag = f" {mode}" if isinstance(mode, str) else ""
            self._enqueue_pending(
                PendingItem("regen_n", f"/regen {n}{mode_tag}", (n, mode))
            )
            return
        if mode is not None:
            self._run_regen_modifier_worker(n, mode)
            return
        self._run_regen_n_worker(n)

    def _run_regen_modifier_worker(
        self, n: int, mode: "str | Recipe",
    ) -> None:
        """Worker for `/regen N <mode>` — routes through ``regen_with_modifier``."""
        chat = self._chat_panel
        tree = self._session.tree
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
                self._session.regen_with_modifier(
                    user_parent_id, mode, n=n,
                )
                if self._raw_mode:
                    self.call_from_thread(self._sync_raw_buffer_from_tree)
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                self.call_from_thread(
                    self._chat_panel.add_system_message,
                    f"/regen {rendered} error: {msg}",
                )
            finally:
                self.call_from_thread(
                    self._chat_panel.add_system_message,
                    f"/regen × {n} ({rendered}): done.",
                )

        self.run_worker(_worker, thread=True)

    def _run_regen_n_worker(self, n: int) -> None:
        chat = self._chat_panel
        # Read prompt off the active path.
        hist = self._messages
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
        self._rewind_active_assistant()

        # After the rewind, active is a user node — passing
        # ``parent_node_id=<user.parent_id>`` lets add_user_turn's dedup
        # land the regen as a sibling under the existing user turn
        # (avoiding D15's user-under-user reject).
        tree = self._session.tree
        active_node = tree.nodes.get(tree.active_node_id)
        regen_parent_id: str | None = None
        if (active_node is not None and active_node.role == "user"
                and active_node.parent_id is not None):
            regen_parent_id = active_node.parent_id

        from saklas import SamplingConfig

        def _worker() -> None:
            try:
                sampling = SamplingConfig(
                    temperature=self._session.config.temperature,
                    top_p=self._session.config.top_p,
                    max_tokens=self._session.config.max_new_tokens,
                    seed=self._default_seed,
                )
                steering = (
                    None if not self._active_alphas()
                    else __import__("saklas").Steering(
                        alphas=dict(self._active_alphas()),
                        thinking=self._thinking,
                    )
                )
                self._session.generate(
                    prompt,
                    steering=steering,
                    sampling=sampling,
                    stateless=False,
                    n=n,
                    raw=self._raw_mode,
                    parent_node_id=regen_parent_id,
                )
                if self._raw_mode:
                    self.call_from_thread(self._sync_raw_buffer_from_tree)
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                self.call_from_thread(
                    self._chat_panel.add_system_message,
                    f"/regen error: {msg}",
                )
            finally:
                self.call_from_thread(
                    self._chat_panel.add_system_message,
                    f"/regen × {n}: done.",
                )

        self.run_worker(_worker, thread=True)

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

        chat = self._chat_panel
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
            matching = self._session.tree.filter_by_expr(expr)
        except FilterParseError as e:
            chat.add_system_message(f"/prune evaluation error: {e}")
            return
        chat.add_system_message(
            f"/prune active: {expr}  ({len(matching)} node(s) match)"
        )

    _AUTO_REGEN_MODES = ("unsteered", "inverted", "reseed", "cool", "hot")

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
        chat = self._chat_panel
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

        chat = self._chat_panel
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

    def _fire_auto_regen(self, row: "_TurnRow | None" = None) -> None:
        """Post-gen hook: fire a sibling regen under the configured mode.

        Called from ``_poll_generation`` once a steered ``done`` lands
        with auto-regen on.  When ``row`` is provided, streams the
        modifier-regen output live into ``row``'s shadow column —
        mirroring :meth:`_start_shadow_generation`'s shape, so the right
        column reflects whatever override mode is active (not just
        ``unsteered``).  When ``row`` is ``None``, falls back to the
        background-worker form that only emits a one-line system message
        on completion.

        The sibling lands under the user-parent of the active assistant
        in both paths — the visual placement is the only thing that
        differs.
        """
        if not self._loom_auto_regen_on:
            return
        tree = self._session.tree
        active = tree.nodes.get(tree.active_node_id)
        if active is None or active.role != "assistant":
            return
        user_parent_id = active.parent_id
        if user_parent_id is None:
            return
        mode = self._loom_auto_regen_mode

        # Resolve the anchor user node + parent the same way
        # ``regen_with_modifier`` does so we can route through the
        # streaming entry point and still land the new node as a sibling
        # under the shared user-turn.
        anchor_user = tree.nodes.get(user_parent_id)
        if anchor_user is None or anchor_user.role != "user":
            return
        user_text = anchor_user.text
        parent_node_id = anchor_user.parent_id

        rendered_mode = self._render_auto_regen_mode(mode)
        if row is None:
            # Worker-only path: matches the pre-streaming shape — useful
            # when there's no row to render into (e.g. fired from a
            # corner where the widget isn't tracked).
            def _worker() -> None:
                try:
                    self._session.regen_with_modifier(user_parent_id, mode, n=1)
                except BaseException as e:
                    msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                    self.call_from_thread(
                        self._chat_panel.add_system_message,
                        f"auto-regen ({rendered_mode}) error: {msg}",
                    )
                    return
                new_id = self._session.tree.active_node_id
                if self._raw_mode:
                    self.call_from_thread(self._sync_raw_buffer_from_tree)
                self.call_from_thread(
                    self._chat_panel.add_system_message,
                    f"auto-regen ({rendered_mode}) → sibling {new_id[:8]}",
                )

            self.run_worker(_worker, thread=True)
            return

        # Streaming path: mount a shadow widget on the row's shadow
        # column and pump tokens through ``_ui_token_queue`` with
        # ``is_shadow=True`` so the existing poll machinery handles
        # gen-stat counters and the post-shadow pending drain.
        if self._ab_shadow_active:
            return
        chat = self._chat_panel
        widget = chat.start_shadow_message(row)
        self._row_for_widget[id(widget)] = row
        self._assistant_messages.append(widget)
        if self._highlighting:
            widget.apply_highlight(True, self._highlight_probe)
        self._current_assistant_widget = widget
        self._ab_shadow_active = True
        self._ab_shadow_row = row
        self._ui_gen_active = True

        chat.add_system_message(f"auto-regen ({rendered_mode}) → streaming sibling…")

        def _stream_worker() -> None:
            try:
                stream = self._session.generate_stream(
                    user_text,
                    raw=self._raw_mode,
                    parent_node_id=parent_node_id,
                    recipe_override=mode,
                    live_scores=self._wants_live_probe_scores(),
                )
                for event in stream:
                    self._ui_token_queue.put(
                        ("tok", event.text, event.thinking, event.scores,
                         event.perplexity, event.logprob, widget, True,
                         event.manifold_readings),
                    )
                self._ui_token_queue.put(("finalize", widget, True))
            except BaseException as e:
                msg = e.user_message()[1] if isinstance(e, SaklasError) else str(e)
                self._ui_token_queue.put(("error", msg, True))
            finally:
                if self._session._device.type == "mps":
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass
                self._ui_token_queue.put(("done", True))

        self.run_worker(_stream_worker, thread=True)

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

        chat = self._chat_panel
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

        m1 = resolve_node_prefix(self._session.tree, ids[0])
        m2 = resolve_node_prefix(self._session.tree, ids[1])
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
            diff = self._session.diff_nodes(m1.node_id, m2.node_id)
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
        chat = self._chat_panel
        tree = self._session.tree
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
                diff = self._session.diff_nodes(sibs[0], sibs[1])
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
                    diff = self._session.diff_nodes(a_id, b_id)
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
