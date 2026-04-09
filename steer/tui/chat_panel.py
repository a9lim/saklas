"""Chat panel: message display + input."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Input
from textual.widget import Widget
from textual.message import Message


class ChatPanel(Widget):

    class UserSubmitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
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

    def start_assistant_message(self) -> Static:
        log = self.query_one("#chat-log", VerticalScroll)
        widget = Static("[bold green]Assistant:[/] ", classes="assistant-message")
        log.mount(widget)
        return widget

    def append_to_assistant(self, widget: Static, token: str) -> None:
        current = widget.renderable
        if isinstance(current, str):
            widget.update(current + token)
        else:
            widget.update(str(current) + token)
        log = self.query_one("#chat-log", VerticalScroll)
        log.scroll_end(animate=False)

    def add_system_message(self, text: str) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(Static(f"[dim]{text}[/]"))
        log.scroll_end(animate=False)
