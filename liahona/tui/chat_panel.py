"""Chat panel: message display, status bar, and input."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Input, Markdown, Collapsible
from textual.widget import Widget
from textual.message import Message


class _AssistantMessage(Vertical):
    """Container with a label and Markdown widget for assistant messages.

    During streaming, updates go to a cheap Static widget (no parse).
    On finalize(), the Static is hidden and a full Markdown render is shown.
    Thinking tokens (from models like Qwen 3.5) render in a collapsible
    section above the main response.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.chat_text: str = ""
        self.thinking_text: str = ""
        self._in_thinking: bool = False
        self._thinking_block: Collapsible | None = None
        self._thinking_stream: Static | None = None
        self._thinking_md: Markdown | None = None
        self._stream: Static | None = None
        self._md: Markdown | None = None

    def compose(self) -> ComposeResult:
        yield Static("[bold ansi_green]Assistant:[/]")
        with Collapsible(title="Thinking...", id="thinking-block", classes="hidden"):
            yield Static("", id="thinking-text")
            yield Markdown(id="thinking-md", classes="hidden")
        yield Static("", id="stream-text")
        yield Markdown(id="response-md", classes="hidden")

    def on_mount(self) -> None:
        self._thinking_block = self.query_one("#thinking-block", Collapsible)
        self._thinking_stream = self.query_one("#thinking-text", Static)
        self._thinking_md = self.query_one("#thinking-md", Markdown)
        self._stream = self.query_one("#stream-text", Static)
        self._md = self.query_one("#response-md", Markdown)

    def update_thinking(self, text: str) -> None:
        self.thinking_text = text
        if self._thinking_block is not None and self._thinking_block.has_class("hidden"):
            self._thinking_block.remove_class("hidden")
            self._thinking_block.collapsed = False
            self._in_thinking = True
        if self._thinking_stream is not None:
            self._thinking_stream.update(text)

    def end_thinking(self) -> None:
        self._in_thinking = False
        if self._thinking_block is not None:
            # Swap to rendered Markdown and collapse immediately
            if self._thinking_md is not None and self.thinking_text:
                self._thinking_md.update(self.thinking_text)
                self._thinking_md.remove_class("hidden")
            if self._thinking_stream is not None:
                self._thinking_stream.add_class("hidden")
            self._thinking_block.collapsed = True

    def update_content(self, text: str) -> None:
        self.chat_text = text
        if self._stream is not None:
            self._stream.update(text)

    def finalize(self) -> None:
        """Switch from streaming Static to rendered Markdown."""
        if self._in_thinking:
            self.end_thinking()
        if self._md is not None and self.chat_text:
            self._md.update(self.chat_text)
            self._md.remove_class("hidden")
        if self._stream is not None:
            self._stream.add_class("hidden")


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
            Markdown(text),
            classes="user-message",
        )
        self._log.mount(container)
        self._log.scroll_end(animate=False)

    def start_assistant_message(self) -> _AssistantMessage:
        widget = _AssistantMessage(classes="assistant-message")
        self._log.mount(widget)
        return widget

    def append_to_assistant(self, widget: _AssistantMessage, token: str) -> None:
        widget.chat_text += token
        widget.update_content(widget.chat_text)

    def append_thinking(self, widget: _AssistantMessage, token: str) -> None:
        widget.thinking_text += token
        widget.update_thinking(widget.thinking_text)

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
