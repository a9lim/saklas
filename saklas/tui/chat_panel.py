"""Chat panel: message display, status bar, and input."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from rich.markup import escape as _rich_escape
from textual import events as _textual_events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, TextArea, Collapsible
from textual.widget import Widget
from textual.message import Message

from saklas.tui.utils import BAR_WIDTH, build_bar


# --- Pending queue (mid-gen submissions) ----------------------------------
#
# Both the TUI and the GUI defer chat sends, commits, and interrupting
# slash commands while a generation is in flight rather than tearing the
# current stream down to make room.  The TUI side lives here so the
# ``PendingStrip`` widget can render directly off the dataclass without an
# adapter layer; ``SaklasApp`` owns the list and dispatches each item once
# the next ``done`` lands.

# Short tag rendered at the head of each pending row.  Keys must match the
# ``kind`` strings used by ``SaklasApp._dispatch_pending_action``.
_KIND_LABELS: dict[str, str] = {
    "submit": "msg",
    "commit_user": "commit",
    "commit_assistant": "commit",
    "regenerate": "regen",
    "rewind": "/rewind",
    "clear": "/clear",
    "quit": "/quit",
    "steer": "/steer",
    "probe": "/probe",
    "extract": "/extract",
    "regen_n": "/regen",
    "fan": "/fan",
}

# Sentinel used in a PendingItem payload to defer parent-node resolution
# until drain time.  When a queued commit / prefill should hang under a
# node that another earlier-queued action will create (e.g. a
# ``commit_user`` followed by a ``commit_assistant``), capturing the
# live active id at submit time would point at the wrong node — this
# sentinel tells the dispatcher to read ``session.tree.active_node_id``
# fresh, by which point the earlier queue items have landed.
ACTIVE_AT_DRAIN = "__active@drain__"

# Per-kind prediction of the active-node role *after* the item drains.
# Drives the queue-aware input placeholder: a queued ``commit_user``
# (value ``True``) flips the next submission into prefill mode without
# waiting for the queue to drain.  Items missing from this map are
# treated as "no role change" — rack mutations, /steer, /probe, etc.
_KIND_ENDS_ON_USER_NODE: dict[str, bool] = {
    "submit": False,           # send → assistant
    "commit_user": True,       # user turn lands → active = user
    "commit_assistant": False, # assistant turn lands → active = assistant
    "regenerate": False,       # new assistant sibling
    "rewind": True,            # rewinds to the previous user turn
    "clear": False,            # resets to root (system, not user)
    "regen_n": False,          # N assistant siblings
    "fan": False,              # alpha-grid siblings, all assistant
}

# Kinds whose dispatched action is fully synchronous on the UI thread —
# no worker, no done sentinel.  ``_drain_next_pending`` chain-drains
# through them inline rather than blocking for a ``done`` that will
# never arrive.  Kinds NOT in this set must either (a) fire a worker
# that runs a generation (the gen's own ``done`` advances the drain),
# or (b) fire a worker that explicitly enqueues ``("done", False)``
# in its finally block (the commit / extract pattern).  This was the
# root cause of the "queue stuck after commit" bug — commits ran on
# workers without enqueuing a done sentinel, leaving the drain loop
# stalled until the next genuine generation ``done``.
_KIND_CHAIN_INLINE: frozenset[str] = frozenset({
    "clear",
    "rewind",
    "steer",
    "probe",
})


@dataclass(frozen=True)
class PendingItem:
    """One mid-generation submission deferred until the next ``done``.

    Attributes
    ----------
    kind:
        Dispatch tag consumed by
        :meth:`SaklasApp._dispatch_pending_action`.  Values mirror the
        legacy single-tuple ``_pending_action`` shapes one-for-one.
    text:
        Verbatim text the user typed (or the canonical re-typeable form
        for slash-triggered items, e.g. ``"/steer 0.5 angry"``).  Drives
        the strip display and the up-arrow pull-and-edit path — when the
        user pulls an item back into the chat input the text shown here
        is what they see.
    payload:
        Kind-specific extras: ``("submit", text, prefill_target)`` becomes
        ``payload=(prefill_target,)``, ``("commit_assistant", text,
        user_node_id)`` becomes ``payload=(user_node_id,)``, etc.  Empty
        tuple when the kind has no extra args.
    """

    kind: str
    text: str = ""
    payload: tuple[Any, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        """Short bracketed tag — ``msg``, ``commit``, ``/steer``, …"""
        return _KIND_LABELS.get(self.kind, self.kind)


def _truncate(text: str, max_chars: int) -> str:
    """Single-line ellipsised display form for the pending strip.

    Newlines fold to ``⏎`` so a multi-line message reads as one strip
    row; the ellipsis honors a hard char cap so a long pasted block
    can't push the row to the full window width.
    """
    flat = text.replace("\n", " ⏎ ")
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 1] + "…"


class PendingStrip(Static):
    """Ghosted single-line summary above the chat input.

    Renders the current ``PendingItem`` queue as a compact dim listing —
    ``pending 3 · [msg] what do you think? · [/steer] /steer 0.5 angry · …``.
    Hidden via the ``.hidden`` CSS class when the queue is empty so the
    strip takes zero rows in the common no-pending case.

    The slot currently pulled into the input (``pulled_slot`` arg) is
    highlighted with a leading ``✎`` marker and undimmed text so the
    user can see at a glance which queued item is being edited.

    Per-item cancel is keyboard-only and lives on the chat input side:
    pull the item with ``↑``, clear with ``Ctrl+U``, ``Enter`` (the
    empty-input replace path removes the slot).  The strip itself is
    display-only — keeping it non-focusable means the existing
    ``Tab`` panel cycle is unchanged.
    """

    DEFAULT_CSS = ""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="pending-strip", **kwargs)
        self.add_class("hidden")
        self._queue: list[PendingItem] = []

    def update_queue(
        self, queue: list[PendingItem], *, pulled_slot: int | None = None,
    ) -> None:
        """Replace the rendered queue.  Hides the strip when empty.

        ``pulled_slot`` (when in range) marks the item the user is
        currently editing via the ↑-pull-and-edit path: that entry
        renders with a leading ``✎`` and undimmed text so the editing
        state is visible without an extra row.
        """
        self._queue = list(queue)
        if not self._queue:
            self.add_class("hidden")
            self.update("")
            return
        self.remove_class("hidden")
        n = len(self._queue)
        parts: list[str] = [f"[ansi_yellow]pending {n}[/ansi_yellow]"]
        for i, item in enumerate(self._queue):
            label_esc = _rich_escape(item.label)
            text_esc = _rich_escape(_truncate(item.text, 48))
            if i == pulled_slot:
                # Undimmed + ✎ marker so the user can see at a glance
                # which queued item is being edited.
                parts.append(
                    f"[ansi_yellow]✎ {label_esc}[/ansi_yellow] {text_esc}"
                )
            else:
                parts.append(
                    f"[dim][ansi_blue]{label_esc}[/ansi_blue] {text_esc}[/dim]"
                )
        self.update(" · ".join(parts))

_HIGHLIGHT_SAT = 0.5
_HIGHLIGHT_CACHE_MAX = 4

# Logit-pass: sentinel ``_highlight_probe`` value that selects the
# inline surprise highlight (tokens tinted by chosen-token logprob).
# Distinct from any real probe name — probe names are slugged
# ``[a-z0-9._-]`` so the double-underscore form can't collide.  Same
# string as ``webui/src/lib/tokens.ts::SURPRISE_TARGET`` to keep the
# two surfaces in sync.
SURPRISE_PROBE = "__surprise__"


def _surprise_score(logprob: float | None) -> float:
    """Map a chosen-token logprob to a positive ``[0, ~0.5]`` tint score
    suitable for ``_build_highlight_markup``'s saturation mapping.

        tint = 1 - exp(logprob) = 1 - probability   # [0, 1)
        score = tint * _HIGHLIGHT_SAT               # so 1.0 saturates green

    ``logprob`` is the log of a probability so it's always ≤ 0 —
    ``exp(logprob)`` lands in (0, 1] and ``tint`` lands in [0, 1).
    None / non-finite logprobs return 0 (no tint).
    """
    if logprob is None:
        return 0.0
    # Defensive bounds: logprob is the log of a probability so it must be
    # ≤ 0.  Anything outside that range (NaN, inf, accidental positive)
    # falls through to "no tint" rather than poisoning the markup.
    if not (-float("inf") < logprob <= 0.0):
        return 0.0
    tint = 1.0 - math.exp(logprob)
    return tint * _HIGHLIGHT_SAT


def _build_highlight_markup(
    token_strs: list[str], scores: list[float], *, strip_leading_whitespace: bool = False,
) -> str:
    """Build Rich markup with per-token red/green background spans.

    During live streaming, ``scores`` may lag ``token_strs`` by a token
    (or more) — unscored tokens render as default text and pick up
    colour on the next render after their score arrives.

    ``strip_leading_whitespace=True`` drops whitespace-only tokens from
    the head of the stream — used by the response view to swallow the
    ``\\n\\n`` models often emit after ``</think>``.
    """
    parts: list[str] = []
    n_scores = len(scores)
    seen_content = not strip_leading_whitespace
    for i, tok in enumerate(token_strs):
        if not seen_content:
            if not tok.strip():
                continue
            seen_content = True
        score = scores[i] if i < n_scores else 0.0
        t = max(-1.0, min(1.0, score / _HIGHLIGHT_SAT))
        safe = _rich_escape(tok)
        if t > 0:
            g = int(round(255 * t))
            parts.append(f"[on rgb(0,{g},0)]{safe}[/]")
        elif t < 0:
            r = int(round(255 * -t))
            parts.append(f"[on rgb({r},0,0)]{safe}[/]")
        else:
            parts.append(safe)
    return "".join(parts)


class _AssistantMessage(Vertical):
    """Assistant message rendered entirely through Static widgets.

    Text is escaped incrementally as tokens arrive (O(n) total rather than
    O(n²)). Once per-token probe data lands via ``set_token_data``, the
    highlighted markup is pre-built once per probe and cached so that
    navigating the trait panel is a dict lookup instead of a rebuild.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._escaped_chat_text: str = ""
        self._escaped_thinking_text: str = ""
        self._thinking_block: Collapsible | None = None
        self._thinking_view: Static | None = None
        self._response_view: Static | None = None

        self.response_token_strs: list[str] = []
        self.thinking_token_strs: list[str] = []
        self._streamed_response_tokens: list[str] = []
        self._streamed_thinking_tokens: list[str] = []
        self._response_markup_cache: OrderedDict[str, str] = OrderedDict()
        self._thinking_markup_cache: OrderedDict[str, str] = OrderedDict()
        self._response_probe_scores: dict[str, list[float]] = {}
        self._thinking_probe_scores: dict[str, list[float]] = {}
        # Logit-pass: parallel per-token logprob lists feed the
        # ``SURPRISE_PROBE`` markup path.  Indexed in lock-step with the
        # streamed-token lists; missing entries are None (the surprise
        # score helper renders no tint for those positions).
        self._response_logprobs: list[float | None] = []
        self._thinking_logprobs: list[float | None] = []

        self._highlight_on: bool = False
        self._highlight_probe: str | None = None

        # Finalized content staged before the content Statics are wired.
        # Textual mounts asynchronously, so ``set_static_content`` (the
        # loom-repaint path) stashes here and ``on_mount`` applies it.
        self._pending_static: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Static("[bold ansi_green]Assistant:[/]")
        with Collapsible(title="Thinking...", id="thinking-block", classes="hidden"):
            yield Static("", id="thinking-view")
        yield Static("", id="response-view")

    def on_mount(self) -> None:
        self._thinking_block = self.query_one("#thinking-block", Collapsible)
        self._thinking_view = self.query_one("#thinking-view", Static)
        self._response_view = self.query_one("#response-view", Static)
        if self._pending_static is not None:
            self._apply_static(self._pending_static)
            self._pending_static = None

    # -- Streaming --

    def append_token(self, token: str) -> None:
        self._escaped_chat_text += _rich_escape(token)
        self._streamed_response_tokens.append(token)
        self._render_response()

    def append_thinking_token(self, token: str) -> None:
        self._escaped_thinking_text += _rich_escape(token)
        self._streamed_thinking_tokens.append(token)
        tb = self._thinking_block
        if tb is not None and tb.has_class("hidden"):
            tb.remove_class("hidden")
            tb.collapsed = False
        self._render_thinking()

    def ensure_thinking_collapsed(self) -> None:
        """Collapse the thinking block if it's currently open. Idempotent."""
        tb = self._thinking_block
        if tb is not None and not tb.collapsed and not tb.has_class("hidden"):
            tb.collapsed = True

    # -- Static (finalized) content --

    def set_static_content(
        self,
        response_text: str,
        thinking_text: str = "",
        *,
        response_tokens: list[str] | None = None,
        thinking_tokens: list[str] | None = None,
        response_logprobs: list[float | None] | None = None,
        thinking_logprobs: list[float | None] | None = None,
    ) -> None:
        """Populate a finalized assistant message in one shot.

        Used when repainting the chat log along a navigated loom path —
        the node's text is already complete, so there's no streaming.
        Per-token strings / logprobs are optional: when present they let
        surprise-mode highlighting survive a navigation; per-probe scores
        aren't persisted on loom nodes, so probe highlight stays plain.
        Deferred to ``on_mount`` when the content widgets aren't wired
        yet (Textual mounts asynchronously).
        """
        payload: dict[str, Any] = {
            "response_text": response_text,
            "thinking_text": thinking_text,
            "response_tokens": list(response_tokens or []),
            "thinking_tokens": list(thinking_tokens or []),
            "response_logprobs": list(response_logprobs or []),
            "thinking_logprobs": list(thinking_logprobs or []),
        }
        if self._response_view is None:
            self._pending_static = payload
        else:
            self._apply_static(payload)

    def _apply_static(self, payload: dict[str, Any]) -> None:
        resp_tokens = payload["response_tokens"]
        think_tokens = payload["thinking_tokens"]
        # Prefer per-token text so the escaped body matches the highlight
        # token list exactly (loaded trees carry no tokens — fall back to
        # the node's canonical text).
        if resp_tokens:
            self._escaped_chat_text = "".join(_rich_escape(t) for t in resp_tokens)
        else:
            self._escaped_chat_text = _rich_escape(payload["response_text"])
        if think_tokens:
            self._escaped_thinking_text = "".join(
                _rich_escape(t) for t in think_tokens
            )
        else:
            self._escaped_thinking_text = _rich_escape(payload["thinking_text"])
        self._streamed_response_tokens = list(resp_tokens)
        self._streamed_thinking_tokens = list(think_tokens)
        self.response_token_strs = list(resp_tokens)
        self.thinking_token_strs = list(think_tokens)
        self._response_logprobs = list(payload["response_logprobs"])
        self._thinking_logprobs = list(payload["thinking_logprobs"])
        self._response_probe_scores = {}
        self._thinking_probe_scores = {}
        self._response_markup_cache.clear()
        self._thinking_markup_cache.clear()
        tb = self._thinking_block
        if tb is not None:
            if payload["thinking_text"] or think_tokens:
                tb.remove_class("hidden")
                tb.collapsed = True
            else:
                tb.add_class("hidden")
        self._render_response()
        self._render_thinking()

    # -- Highlight data --

    def set_token_data(
        self,
        response_token_strs: list[str],
        response_probe_scores: dict[str, list[float]],
        thinking_token_strs: list[str],
        thinking_probe_scores: dict[str, list[float]],
    ) -> None:
        self.response_token_strs = list(response_token_strs)
        self.thinking_token_strs = list(thinking_token_strs)
        resp_n = len(self.response_token_strs)
        think_n = len(self.thinking_token_strs)
        self._response_probe_scores = {
            n: s for n, s in response_probe_scores.items() if len(s) == resp_n
        }
        self._thinking_probe_scores = {
            n: s for n, s in thinking_probe_scores.items() if len(s) == think_n
        }
        self._response_markup_cache.clear()
        self._thinking_markup_cache.clear()
        # Warm the cache with the currently-selected probe so the next render
        # is a straight lookup. Other probes build lazily on first use.
        if self._highlight_on and self._highlight_probe:
            self._get_response_markup(self._highlight_probe)
            self._get_thinking_markup(self._highlight_probe)
        self._render_response()
        self._render_thinking()

    def append_token_score(self, scores: dict[str, float], is_thinking: bool) -> None:
        """Append one token's per-probe scores to the running per-probe lists.

        Called from the TUI event-pump alongside ``append_token`` /
        ``append_thinking_token`` so highlighting and WHY top-token stats
        can update mid-gen. Invalidates the markup cache for the affected
        side so the next render rebuilds with the new score row.
        """
        target = self._thinking_probe_scores if is_thinking else self._response_probe_scores
        for name, val in scores.items():
            target.setdefault(name, []).append(val)
        if is_thinking:
            self._thinking_markup_cache.clear()
        else:
            self._response_markup_cache.clear()
        if self._highlight_on:
            if is_thinking:
                self._render_thinking()
            else:
                self._render_response()

    def append_token_logprob(
        self, logprob: float | None, is_thinking: bool,
    ) -> None:
        """Append one token's chosen-token logprob (logit-pass).

        Mirrors ``append_token_score``: appends to the per-side logprob
        list, invalidates the surprise-mode markup cache (only the
        ``SURPRISE_PROBE`` key — leaving probe-keyed cache entries
        intact), and re-renders if the user is currently sitting in
        surprise highlight.
        """
        target = self._thinking_logprobs if is_thinking else self._response_logprobs
        target.append(logprob)
        cache = (
            self._thinking_markup_cache if is_thinking else self._response_markup_cache
        )
        cache.pop(SURPRISE_PROBE, None)
        if self._highlight_on and self._highlight_probe == SURPRISE_PROBE:
            if is_thinking:
                self._render_thinking()
            else:
                self._render_response()

    def _get_response_markup(self, probe: str) -> str | None:
        cached = self._response_markup_cache.get(probe)
        if cached is not None:
            self._response_markup_cache.move_to_end(probe)
            return cached
        # Prefer the live-streamed token list (always current) over the
        # finalize-time canonical list, which set_token_data fills only at end.
        token_strs = self.response_token_strs or self._streamed_response_tokens
        if probe == SURPRISE_PROBE:
            # Logit-pass: tint by ``surprise_score(logprob)``.  Empty
            # logprob list short-circuits to None so the renderer falls
            # through to plain-text and the user sees raw text instead of
            # uniform-no-tint markup.
            if not self._response_logprobs:
                return None
            scores = [_surprise_score(lp) for lp in self._response_logprobs]
            markup = _build_highlight_markup(
                token_strs, scores, strip_leading_whitespace=True,
            )
        else:
            scores = self._response_probe_scores.get(probe)
            if scores is None:
                return None
            markup = _build_highlight_markup(
                token_strs, scores, strip_leading_whitespace=True,
            )
        self._response_markup_cache[probe] = markup
        if len(self._response_markup_cache) > _HIGHLIGHT_CACHE_MAX:
            self._response_markup_cache.popitem(last=False)
        return markup

    def _get_thinking_markup(self, probe: str) -> str | None:
        cached = self._thinking_markup_cache.get(probe)
        if cached is not None:
            self._thinking_markup_cache.move_to_end(probe)
            return cached
        token_strs = self.thinking_token_strs or self._streamed_thinking_tokens
        if probe == SURPRISE_PROBE:
            if not self._thinking_logprobs:
                return None
            scores = [_surprise_score(lp) for lp in self._thinking_logprobs]
            markup = _build_highlight_markup(token_strs, scores)
        else:
            scores = self._thinking_probe_scores.get(probe)
            if scores is None:
                return None
            markup = _build_highlight_markup(token_strs, scores)
        self._thinking_markup_cache[probe] = markup
        if len(self._thinking_markup_cache) > _HIGHLIGHT_CACHE_MAX:
            self._thinking_markup_cache.popitem(last=False)
        return markup

    def apply_highlight(self, on: bool, probe_name: str | None) -> None:
        self._highlight_on = on
        self._highlight_probe = probe_name
        self._render_response()
        self._render_thinking()

    # -- Rendering --

    def _render_response(self) -> None:
        if self._response_view is None:
            return
        markup = None
        if self._highlight_on and self._highlight_probe:
            markup = self._get_response_markup(self._highlight_probe)
        # Models often emit \n\n right after </think>; lstrip the plain-text
        # path so the gap below the thinking block is gone in non-highlight
        # mode (the markup builder strips leading whitespace tokens itself).
        text = markup if markup is not None else self._escaped_chat_text.lstrip()
        self._response_view.update(text)

    def _render_thinking(self) -> None:
        if self._thinking_view is None:
            return
        markup = None
        if self._highlight_on and self._highlight_probe:
            markup = self._get_thinking_markup(self._highlight_probe)
        self._thinking_view.update(markup if markup is not None else self._escaped_thinking_text)


def _build_user_widget(text: str) -> Vertical:
    """Build a fresh user-message Vertical (header + content) for one column.

    A separate widget instance is used in each column so the same text can
    be mirrored in both primary and shadow without a single widget being
    parented twice.
    """
    return Vertical(
        Static("[bold ansi_cyan]User:[/]"),
        Static(_rich_escape(text)),
        classes="user-message",
    )


def _build_shadow_placeholder() -> Static:
    """Placeholder mounted in the shadow column of an assistant turn until
    the unsteered shadow gen actually fires.  Visible in AB mode while the
    steered branch is still streaming, then replaced by an
    ``_AssistantMessage`` when the shadow worker starts."""
    return Static("[dim](pending unsteered)[/]", classes="shadow-placeholder")


class _TurnRow(Horizontal):
    """A single chat-log row carrying matched primary + shadow columns.

    Visual mirror of the webui's ``.ab-grid`` two-column layout: every turn
    (user or assistant) lives in a Horizontal with two ``Vertical`` columns
    of equal width.  The shadow column is hidden (``display: none`` via the
    ``ChatPanel.ab-on`` CSS gate in ``styles.tcss``) when A/B mode is off,
    so the primary column expands to full width and the layout is identical
    to the pre-AB rendering.

    Children are passed at construction time — Textual's ``Widget.mount``
    raises if called before the parent has joined the app tree, so we hand
    the inner widgets to ``Vertical()`` constructors and let the row's
    ``compose`` yield them as a single mount unit.  Post-mount swaps (e.g.
    replacing a shadow placeholder with the real shadow ``_AssistantMessage``)
    happen via ``shadow.remove_children()`` + ``shadow.mount(widget)`` once
    the row is live.
    """

    def __init__(
        self, kind: str, primary_child: Widget, shadow_child: Widget, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_class("turn-row")
        self.add_class(f"{kind}-row")
        self._kind = kind
        # Raw (unescaped) user text for user rows, so the AB shadow worker
        # can rebuild the messages list without unescaping rendered Rich
        # markup.  Set by ``ChatPanel.add_user_message``; ``None`` for
        # assistant rows.
        self.user_text: str | None = None
        self._primary: Vertical = Vertical(primary_child, classes="turn-col primary")
        self._shadow: Vertical = Vertical(shadow_child, classes="turn-col shadow")

    def compose(self) -> ComposeResult:
        yield self._primary
        yield self._shadow

    @property
    def kind(self) -> str:
        """``"user"`` or ``"assistant"`` — drives rewind / backfill walks."""
        return self._kind

    @property
    def primary(self) -> Vertical:
        return self._primary

    @property
    def shadow(self) -> Vertical:
        return self._shadow


class ChatInput(TextArea):
    """Multi-line chat input on top of ``TextArea``.

    Differs from the underlying widget in two ways:

    - **Enter submits.** Bare ``Enter`` posts a :class:`Submitted`
      message and clears the buffer.  ``Shift+Enter`` inserts a literal
      newline (TextArea's default ``Enter`` behavior moved one key over).
    - **Cursor-aware history pass-through.** ``↑``/``↓`` only walk the
      input-history ring when the cursor is on the first/last row of
      the buffer; mid-buffer they fall through to TextArea's cursor
      navigation.  The history walk itself lives in
      ``SaklasApp._history_navigate`` — we just gate the event here.

    Submission goes through a widget-local ``Submitted`` message so the
    enclosing :class:`ChatPanel` re-posts it as ``UserSubmitted`` (the
    same contract callers depend on for prefill / commit dispatch).
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # When ``True``, bare ``Enter`` on an empty buffer still emits
        # ``Submitted("")`` rather than swallowing the keypress.  The
        # app flips this on while a pending slot is pulled so the
        # empty-Enter cancel gesture reaches
        # :meth:`SaklasApp.on_chat_panel_user_submitted`, which uses
        # the empty-text + pulled-slot pair to remove the queued
        # item.
        self.allow_empty_submit: bool = False

    async def _on_key(self, event: _textual_events.Key) -> None:
        # Bare Enter submits; ``shift+enter`` and ``ctrl+j`` both insert
        # a literal newline.  Two newline keys because ``shift+enter`` is
        # the natural chat-UI convention but it only reaches the app on
        # terminals that speak the CSI-u / kitty keyboard protocol
        # (Ghostty, Kitty, WezTerm) — stock macOS Terminal.app and
        # iTerm2 silently collapse it to bare ``enter``.  ``ctrl+j``
        # sends a literal LF (0x0A) which Textual's ``KEY_ALIASES``
        # routes as a separate key from ``enter`` (CR, 0x0D) on every
        # terminal, so it's the reliable cross-terminal fallback.
        # Anything else hands off to ``TextArea._on_key`` (printable
        # chars, etc.).
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            text = self.text.strip()
            if text or self.allow_empty_submit:
                self.load_text("")
                self.post_message(self.Submitted(text))
            return
        if event.key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


class ChatPanel(Widget):

    class UserSubmitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._log: VerticalScroll | None = None
        self._status_bar: Static | None = None
        self._ab_mode: bool = False
        # In-memory mirror of system-message strings.  ``add_system_message``
        # mounts a ``Static`` widget AND appends to this list so callers
        # (tests, transcript export, future log search) can read the
        # rendered system-message text without walking the widget tree.
        # Append-only; ``clear_log`` resets.
        self.messages: list[str] = []

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        yield Static("", id="status-bar")
        # Pending-queue strip — single-line summary of mid-gen
        # submissions waiting for the next ``done``.  Hidden when the
        # queue is empty.  Mounted between the status bar and the input
        # so it lives in the same "below the log" footer region the
        # user already reads for state.
        yield PendingStrip()
        # Multi-line chat input.  ``show_line_numbers=False`` keeps the
        # editor chrome out; ``soft_wrap=True`` is the TextArea default
        # (long lines wrap visually without changing the underlying
        # newlines).  ``highlight_cursor_line=False`` disables the
        # always-on "boost"-coloured background under the cursor line —
        # under the ansi theme that read as an opaque black band rather
        # than a subtle highlight.  Height grows with content up to a
        # CSS-side cap.
        yield ChatInput(
            placeholder="message…  ⏎ send · ⌃⏎ commit · ⇧⏎ newline",
            id="chat-input",
            show_line_numbers=False,
            highlight_cursor_line=False,
        )

    def on_mount(self) -> None:
        self._log = self.query_one("#chat-log", VerticalScroll)
        self._status_bar = self.query_one("#status-bar", Static)

    @property
    def pending_strip(self) -> PendingStrip:
        return self.query_one(PendingStrip)

    def update_pending(
        self, queue: list[PendingItem], *, pulled_slot: int | None = None,
    ) -> None:
        """Forward queue updates to the strip.  Safe before mount.

        ``pulled_slot`` marks the queue position the user is currently
        editing via the ↑-pull-and-edit path so the strip renders
        that entry with the editing marker.
        """
        try:
            self.pending_strip.update_queue(queue, pulled_slot=pulled_slot)
        except Exception:
            # Pre-mount: ``query_one`` raises ``NoMatches``.  The strip
            # is freshly empty on mount, so a missed update is a no-op.
            pass

    # All ``log`` / ``status_bar`` access happens after Textual's mount
    # lifecycle has run ``on_mount``, so the assertions below are
    # invariants — they exist to narrow the Optional for type checkers,
    # not as runtime preconditions.
    @property
    def _log_mounted(self) -> VerticalScroll:
        assert self._log is not None, "ChatPanel._log accessed before on_mount"
        return self._log

    @property
    def _status_bar_mounted(self) -> Static:
        assert self._status_bar is not None, "ChatPanel._status_bar accessed before on_mount"
        return self._status_bar

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value.strip()
        # ChatInput already cleared its buffer; just re-post under the
        # canonical ChatPanel message.  The user-row mount is the app's
        # call, not ours: on a user-role active loom node a typed message
        # is an assistant prefill, not a new user turn, and only the app
        # knows the active node's role.  ``on_chat_panel_user_submitted``
        # mounts the row for normal sends.  An empty text reaches us
        # only when ``ChatInput.allow_empty_submit`` is true — the
        # cancel-pulled-slot gesture; the app routes it to slot
        # removal.
        try:
            inp = self.query_one("#chat-input", ChatInput)
            allow_empty = inp.allow_empty_submit
        except Exception:
            allow_empty = False
        if not text and not allow_empty:
            return
        self.post_message(self.UserSubmitted(text))

    def set_prefill_mode(self, on: bool) -> None:
        """Flip the input placeholder between normal-send and prefill.

        When the active loom node is a user turn a typed message seeds
        the assistant reply (answer-prefill) rather than appending a new
        user message; the placeholder is the signal the app updates as
        navigation changes the active node's role.  Mirrors
        :meth:`set_ab_mode`'s app-driven-flag shape.
        """
        try:
            inp = self.query_one("#chat-input", ChatInput)
        except Exception:
            return
        # Same `<verb>…  <bind list>` shape as the GUI's Chat.svelte
        # placeholder; terminals can't surface modifier-only keydown so
        # the TUI lacks the commit-key-held variants the GUI shows.
        inp.placeholder = (
            "prefill reply…  ⏎ send · ⌃⏎ commit · ⇧⏎ newline"
            if on
            else "message…  ⏎ send · ⌃⏎ commit · ⇧⏎ newline"
        )

    # -- AB mode --

    def set_ab_mode(self, on: bool) -> None:
        """Toggle the two-column AB layout.  Adds/removes ``ab-on`` on the
        panel; the column's ``display`` is gated by this class in
        ``styles.tcss``.  Pre-existing shadow-column widgets stay alive so
        toggling off then on restores any prior shadow content untouched.
        """
        self._ab_mode = on
        if on:
            self.add_class("ab-on")
        else:
            self.remove_class("ab-on")

    @property
    def ab_mode(self) -> bool:
        return self._ab_mode

    # -- Log walking helpers --

    def _turn_rows(self) -> list[_TurnRow]:
        """All ``_TurnRow`` children of the log, in mount order."""
        return [c for c in self._log_mounted.children if isinstance(c, _TurnRow)]

    def assistant_rows_pending_shadow(self) -> list[_TurnRow]:
        """Assistant rows whose shadow column still holds the placeholder
        (no real shadow ``_AssistantMessage`` mounted yet).  Used by the
        AB-toggle-on backfill to find rows in need of a shadow gen.
        """
        out: list[_TurnRow] = []
        for row in self._turn_rows():
            if row.kind != "assistant":
                continue
            shadow_kids = list(row.shadow.children)
            if not any(isinstance(c, _AssistantMessage) for c in shadow_kids):
                out.append(row)
        return out

    def shadow_widget_for(self, row: _TurnRow) -> _AssistantMessage | None:
        """Return the shadow column's mounted assistant widget, or ``None``
        if it still holds the placeholder."""
        for c in row.shadow.children:
            if isinstance(c, _AssistantMessage):
                return c
        return None

    # -- Mutation --

    def clear_log(self) -> None:
        """Remove all messages from the chat log."""
        self._log_mounted.remove_children()
        self.messages.clear()

    def rewind(self) -> None:
        """Remove the last user turn-row and everything after it."""
        log = self._log_mounted
        children = list(log.children)
        for i in range(len(children) - 1, -1, -1):
            child = children[i]
            if isinstance(child, _TurnRow) and child.kind == "user":
                log.remove_children(children[i:])
                break
        log.scroll_end(animate=False)

    def rewind_last_assistant(self) -> None:
        """Remove the last assistant turn-row only.

        With AB mode the row carries both steered + shadow widgets, so the
        whole row goes — there's no useful intermediate state where we'd
        keep one column and drop the other.
        """
        log = self._log_mounted
        children = list(log.children)
        for i in range(len(children) - 1, -1, -1):
            child = children[i]
            if isinstance(child, _TurnRow) and child.kind == "assistant":
                child.remove()
                break
        log.scroll_end(animate=False)

    def add_user_message(self, text: str) -> _TurnRow:
        """Mount a user turn-row.  The same text is mirrored into both
        columns so the primary view sees the user prompt regardless of AB
        mode, and the shadow view (when revealed) is row-aligned with it.
        Returns the row so callers can hand it to ``start_assistant_message``
        for an attached assistant turn.
        """
        row = _TurnRow(
            kind="user",
            primary_child=_build_user_widget(text),
            shadow_child=_build_user_widget(text),
        )
        row.user_text = text
        log = self._log_mounted
        log.mount(row)
        log.scroll_end(animate=False)
        return row

    def start_assistant_message(self) -> tuple[_TurnRow, _AssistantMessage]:
        """Mount a fresh assistant turn-row.  Primary holds a streaming
        ``_AssistantMessage``; shadow holds a placeholder until / unless a
        shadow gen replaces it via ``start_shadow_message``.
        Returns ``(row, primary_widget)``.
        """
        widget = _AssistantMessage(classes="assistant-message")
        placeholder = _build_shadow_placeholder()
        row = _TurnRow(
            kind="assistant", primary_child=widget, shadow_child=placeholder,
        )
        self._log_mounted.mount(row)
        return row, widget

    def add_finalized_assistant(
        self,
        response_text: str,
        thinking_text: str = "",
        *,
        response_tokens: list[str] | None = None,
        thinking_tokens: list[str] | None = None,
        response_logprobs: list[float | None] | None = None,
        thinking_logprobs: list[float | None] | None = None,
    ) -> tuple[_TurnRow, _AssistantMessage]:
        """Mount a fully-rendered (non-streaming) assistant turn-row.

        Counterpart to ``start_assistant_message`` for the loom-repaint
        path — the node's text is complete, so the widget is populated
        in one shot rather than token-by-token.  Returns ``(row, widget)``
        so the caller can keep the same widget / row bookkeeping it does
        for streamed turns.
        """
        widget = _AssistantMessage(classes="assistant-message")
        widget.set_static_content(
            response_text, thinking_text,
            response_tokens=response_tokens,
            thinking_tokens=thinking_tokens,
            response_logprobs=response_logprobs,
            thinking_logprobs=thinking_logprobs,
        )
        placeholder = _build_shadow_placeholder()
        row = _TurnRow(
            kind="assistant", primary_child=widget, shadow_child=placeholder,
        )
        self._log_mounted.mount(row)
        return row, widget

    def start_shadow_message(self, row: _TurnRow) -> _AssistantMessage:
        """Replace ``row``'s shadow placeholder with a fresh streaming
        assistant widget.  Caller is responsible for routing tokens into
        it from a shadow worker.
        """
        existing = self.shadow_widget_for(row)
        if existing is not None:
            return existing
        row.shadow.remove_children()
        widget = _AssistantMessage(classes="assistant-message shadow-message")
        row.shadow.mount(widget)
        return widget

    def scroll_to_bottom(self) -> None:
        """Scroll the chat log to the bottom. Call once after a batch of token updates."""
        self._log_mounted.scroll_end(animate=False)

    def add_system_message(self, text: str) -> None:
        log = self._log_mounted
        log.mount(Static(f"[dim]{text}[/]", classes="system-message"))
        log.scroll_end(animate=False)
        self.messages.append(text)

    def update_status(
        self,
        generating: bool = False,
        gen_tokens: int = 0,
        max_tokens: int = 0,
        tok_per_sec: float = 0.0,
        elapsed: float = 0.0,
        perplexity: float | None = None,
        prune_expr: str | None = None,
        auto_regen_mode: str | None = None,
    ) -> None:
        """Update the status bar with generation stats.

        ``prune_expr`` (``app._loom_prune_expr``) and ``auto_regen_mode``
        (``app._loom_auto_regen_mode`` when ``_loom_auto_regen_on`` and
        the mode is not ``unsteered``) surface the otherwise-invisible
        loom state in the chat footer.
        """
        bar = self._status_bar_mounted
        dot = "[ansi_green]●[/]" if generating else "[dim]○[/]"
        if max_tokens > 0 and (generating or gen_tokens > 0):
            t_full, t_empty = build_bar(gen_tokens, max_tokens, BAR_WIDTH)
            bar_color = "ansi_green" if generating else "dim"
            left = (
                f"{dot} {gen_tokens}/{max_tokens} "
                f"[{bar_color}]{t_full}[/][dim]{t_empty}[/] · "
                f"{tok_per_sec:.1f} tok/s · {elapsed:.1f}s"
            )
        else:
            left = f"{dot} idle"

        parts: list[str] = [left]
        if perplexity is not None:
            parts.append(f"ppl {perplexity:.2f}")
        if prune_expr:
            parts.append(f"filter:{prune_expr}")
        if auto_regen_mode:
            parts.append(f"auto:{auto_regen_mode}")
        bar.update(" · ".join(parts))
