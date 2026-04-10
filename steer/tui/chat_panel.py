"""Chat panel: message display, status bar, and input."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Input
from textual.widget import Widget
from textual.message import Message


class _AssistantMessage(Static):
    """Static widget for assistant messages with tracked text content."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.chat_text: str = ""


class ChatPanel(Widget):

    class UserSubmitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        yield Static("", id="status-bar")
        yield Input(placeholder="Type a message...", id="chat-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self.add_user_message(text)
        self.post_message(self.UserSubmitted(text))

    def add_user_message(self, text: str) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(Static(f"[bold cyan]User:[/] {text}", classes="user-message"))
        log.scroll_end(animate=False)

    def start_assistant_message(self) -> _AssistantMessage:
        log = self.query_one("#chat-log", VerticalScroll)
        widget = _AssistantMessage(classes="assistant-message")
        widget.chat_text = "[bold green]Assistant:[/] "
        widget.update(widget.chat_text)
        log.mount(widget)
        return widget

    def append_to_assistant(self, widget: _AssistantMessage, token: str) -> None:
        widget.chat_text += token
        widget.update(widget.chat_text)
        log = self.query_one("#chat-log", VerticalScroll)
        log.scroll_end(animate=False)

    def add_system_message(self, text: str) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(Static(f"[dim]{text}[/]"))
        log.scroll_end(animate=False)

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
        bar = self.query_one("#status-bar", Static)
        dot = "[green]●[/]" if generating else "[dim]○[/]"
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
        bar.update(f"{left}{'':>4}{right}")
