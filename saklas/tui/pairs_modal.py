"""Modal for hand-authored contrastive-pair vector extraction.

``/pairs <name>`` opens :class:`CustomPairsModal` — a multi-line editor
where each line is one ``positive | negative`` contrastive pair.  On
submit the lines are parsed into a ``list[(positive, negative)]`` and the
caller hands them straight to ``session.extract``, bypassing the
scenario / pair-generation pipeline.

The ``|`` delimiter is deliberate: ``/extract``'s bipolar split uses
``.`` (a surrounded-by-whitespace period), so a different delimiter here
keeps the two surfaces from colliding when a pole text itself contains a
period.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, TextArea


# Delimiter splitting positive from negative on each line.  ``|`` rather
# than ``.`` so a pole text carrying a literal period isn't mis-split.
_PAIR_DELIM = "|"


def parse_pair_lines(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Parse a modal buffer into contrastive pairs.

    Returns ``(pairs, errors)``.  Each non-blank line must hold exactly
    one ``|`` separating a non-empty positive from a non-empty negative;
    blank lines are skipped.  ``errors`` carries a one-line complaint per
    malformed line (1-indexed) so the caller can reject the submission
    with an actionable message.
    """
    pairs: list[tuple[str, str]] = []
    errors: list[str] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.count(_PAIR_DELIM) != 1:
            errors.append(
                f"line {i}: expected exactly one '{_PAIR_DELIM}' "
                f"separating positive from negative"
            )
            continue
        pos, neg = line.split(_PAIR_DELIM, 1)
        pos, neg = pos.strip(), neg.strip()
        if not pos or not neg:
            errors.append(f"line {i}: both sides of '{_PAIR_DELIM}' must be non-empty")
            continue
        pairs.append((pos, neg))
    return pairs, errors


class CustomPairsModal(ModalScreen[list[tuple[str, str]] | None]):
    """Multi-line editor for custom contrastive pairs.

    Dismisses with the parsed pair list on submit, or ``None`` on cancel.
    ``Ctrl+S`` submits; ``Ctrl+Enter`` is a fallback for terminals where
    the modal's ``Ctrl+S`` binding loses to the app-level sort binding.
    ``Escape`` cancels.
    """

    BINDINGS = [
        # ``priority=True`` so the modal's submit wins over the chat
        # screen's ``ctrl+s`` → sort binding (which is non-priority but
        # still in the resolution set while a modal is pushed).
        Binding("ctrl+s", "submit", "Extract", priority=True),
        # Fallback: terminals with the CSI-u / kitty keyboard protocol
        # pass ``ctrl+enter`` through distinctly; legacy terminals
        # collapse it to bare Enter (a newline insert in the TextArea),
        # which is harmless.
        Binding("ctrl+enter", "submit", "Extract", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    def __init__(self, concept_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._concept_name = concept_name

    def compose(self) -> ComposeResult:
        with Vertical(id="pairs-modal-box"):
            yield Static(
                f"[bold]Custom pairs for '[ansi_cyan]{self._concept_name}"
                f"[/]'[/]",
                id="pairs-modal-title",
            )
            yield Static(
                "[dim]One [ansi_blue]positive | negative[/] pair per line.  "
                "⌃S extract · ⎋ cancel[/]",
                id="pairs-modal-hint",
            )
            yield TextArea(
                "",
                id="pairs-modal-input",
                show_line_numbers=True,
            )
            yield Static("", id="pairs-modal-error")

    def on_mount(self) -> None:
        self.query_one("#pairs-modal-input", TextArea).focus()

    def action_submit(self) -> None:
        text = self.query_one("#pairs-modal-input", TextArea).text
        pairs, errors = parse_pair_lines(text)
        err_widget = self.query_one("#pairs-modal-error", Static)
        if errors:
            err_widget.update(
                "[ansi_red]" + "\n".join(errors) + "[/]"
            )
            return
        if not pairs:
            err_widget.update(
                "[ansi_red]no pairs entered — add at least one "
                "'positive | negative' line[/]"
            )
            return
        self.dismiss(pairs)

    def action_cancel(self) -> None:
        self.dismiss(None)
