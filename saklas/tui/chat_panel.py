"""Chat panel: message display, status bar, and input."""

from __future__ import annotations

from collections import OrderedDict

from rich.markup import escape as _rich_escape
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Input, Collapsible
from textual.widget import Widget
from textual.message import Message

from saklas.tui.utils import BAR_WIDTH, build_bar

_HIGHLIGHT_SAT = 0.5
_HIGHLIGHT_CACHE_MAX = 4


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

    def __init__(self, **kwargs) -> None:
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

        self._highlight_on: bool = False
        self._highlight_probe: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("[bold ansi_green]Assistant:[/]")
        with Collapsible(title="Thinking...", id="thinking-block", classes="hidden"):
            yield Static("", id="thinking-view")
        yield Static("", id="response-view")

    def on_mount(self) -> None:
        self._thinking_block = self.query_one("#thinking-block", Collapsible)
        self._thinking_view = self.query_one("#thinking-view", Static)
        self._response_view = self.query_one("#response-view", Static)

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

    def _get_response_markup(self, probe: str) -> str | None:
        cached = self._response_markup_cache.get(probe)
        if cached is not None:
            self._response_markup_cache.move_to_end(probe)
            return cached
        scores = self._response_probe_scores.get(probe)
        if scores is None:
            return None
        # Prefer the live-streamed token list (always current) over the
        # finalize-time canonical list, which set_token_data fills only at end.
        token_strs = self.response_token_strs or self._streamed_response_tokens
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
        scores = self._thinking_probe_scores.get(probe)
        if scores is None:
            return None
        token_strs = self.thinking_token_strs or self._streamed_thinking_tokens
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
        self, kind: str, primary_child: Widget, shadow_child: Widget, **kwargs,
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


class ChatPanel(Widget):

    class UserSubmitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._log: VerticalScroll | None = None
        self._status_bar: Static | None = None
        self._ab_mode: bool = False

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        yield Static("", id="status-bar")
        yield Input(placeholder="Type a message...", id="chat-input")

    def on_mount(self) -> None:
        self._log = self.query_one("#chat-log", VerticalScroll)
        self._status_bar = self.query_one("#status-bar", Static)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        if not text.startswith("/"):
            self.add_user_message(text)
        self.post_message(self.UserSubmitted(text))

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
        return [c for c in self._log.children if isinstance(c, _TurnRow)]

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
        self._log.remove_children()

    def rewind(self) -> None:
        """Remove the last user turn-row and everything after it."""
        children = list(self._log.children)
        for i in range(len(children) - 1, -1, -1):
            child = children[i]
            if isinstance(child, _TurnRow) and child.kind == "user":
                self._log.remove_children(children[i:])
                break
        self._log.scroll_end(animate=False)

    def rewind_last_assistant(self) -> None:
        """Remove the last assistant turn-row only.

        With AB mode the row carries both steered + shadow widgets, so the
        whole row goes — there's no useful intermediate state where we'd
        keep one column and drop the other.
        """
        children = list(self._log.children)
        for i in range(len(children) - 1, -1, -1):
            child = children[i]
            if isinstance(child, _TurnRow) and child.kind == "assistant":
                child.remove()
                break
        self._log.scroll_end(animate=False)

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
        self._log.mount(row)
        self._log.scroll_end(animate=False)
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
        self._log.mount(row)
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
        self._log.scroll_end(animate=False)

    def add_system_message(self, text: str) -> None:
        self._log.mount(Static(f"[dim]{text}[/]", classes="system-message"))
        self._log.scroll_end(animate=False)

    def update_status(
        self,
        generating: bool = False,
        gen_tokens: int = 0,
        max_tokens: int = 0,
        tok_per_sec: float = 0.0,
        elapsed: float = 0.0,
        perplexity: float | None = None,
    ) -> None:
        """Update the status bar with generation stats."""
        bar = self._status_bar
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
        bar.update(" · ".join(parts))
