"""Chat panel: message display, status bar, and input."""

from __future__ import annotations

from rich.markup import escape as _rich_escape
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Input, Collapsible
from textual.widget import Widget
from textual.message import Message

_HIGHLIGHT_SAT = 0.5


def _build_highlight_markup(token_strs: list[str], scores: list[float]) -> str:
    """Build Rich markup with per-token red/green background spans.

    Caller is responsible for guarding against mismatched lengths; this
    function assumes ``len(scores) == len(token_strs)``.
    """
    parts: list[str] = []
    for tok, score in zip(token_strs, scores):
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
        self._response_markup_cache: dict[str, str] = {}
        self._thinking_markup_cache: dict[str, str] = {}

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
        self._render_response()

    def append_thinking_token(self, token: str) -> None:
        self._escaped_thinking_text += _rich_escape(token)
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
        self._response_markup_cache = {
            name: _build_highlight_markup(self.response_token_strs, scores)
            for name, scores in response_probe_scores.items()
            if len(scores) == resp_n
        }
        self._thinking_markup_cache = {
            name: _build_highlight_markup(self.thinking_token_strs, scores)
            for name, scores in thinking_probe_scores.items()
            if len(scores) == think_n
        }
        self._render_response()
        self._render_thinking()

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
            markup = self._response_markup_cache.get(self._highlight_probe)
        self._response_view.update(markup if markup is not None else self._escaped_chat_text)

    def _render_thinking(self) -> None:
        if self._thinking_view is None:
            return
        markup = None
        if self._highlight_on and self._highlight_probe:
            markup = self._thinking_markup_cache.get(self._highlight_probe)
        self._thinking_view.update(markup if markup is not None else self._escaped_thinking_text)


class ChatPanel(Widget):

    class UserSubmitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._log: VerticalScroll | None = None
        self._status_bar: Static | None = None

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

    def clear_log(self) -> None:
        """Remove all messages from the chat log."""
        self._log.remove_children()

    def rewind(self) -> None:
        """Remove the last user message and everything after it."""
        children = list(self._log.children)
        for i in range(len(children) - 1, -1, -1):
            if "user-message" in children[i].classes:
                self._log.remove_children(children[i:])
                break
        self._log.scroll_end(animate=False)

    def rewind_last_assistant(self) -> None:
        """Remove the last assistant message widget only."""
        children = list(self._log.children)
        for i in range(len(children) - 1, -1, -1):
            if "assistant-message" in children[i].classes:
                children[i].remove()
                break
        self._log.scroll_end(animate=False)

    def add_user_message(self, text: str) -> None:
        container = Vertical(
            Static("[bold ansi_cyan]User:[/]"),
            Static(_rich_escape(text)),
            classes="user-message",
        )
        self._log.mount(container)
        self._log.scroll_end(animate=False)

    def start_assistant_message(self) -> _AssistantMessage:
        widget = _AssistantMessage(classes="assistant-message")
        self._log.mount(widget)
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
        prompt_tokens: int = 0,
        vram_gb: float = 0.0,
    ) -> None:
        """Update the status bar with generation stats."""
        bar = self._status_bar
        dot = "[ansi_green]●[/]" if generating else "[dim]○[/]"
        if generating:
            left = f"{dot} {gen_tokens}/{max_tokens} tok · {tok_per_sec:.1f} tok/s · {elapsed:.1f}s"
        elif gen_tokens > 0:
            left = f"{dot} {gen_tokens} tok · {tok_per_sec:.1f} tok/s · {elapsed:.1f}s"
        else:
            left = f"{dot} idle"
        right = ""
        if prompt_tokens > 0:
            right += f"prompt: {prompt_tokens} tok"
        if vram_gb > 0:
            if right:
                right += " · "
            right += f"VRAM: {vram_gb:.1f} GB"
        bar.update(f"{left}    {right}")
