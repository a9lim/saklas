"""Shared selectable-list cursor + selection markup for the side panels.

Both the left vector rack (``vector_panel.LeftPanel``) and the trait monitor
(``trait_panel.TraitPanel``) render a navigable list of rows: a cursor moves
through the entries and the selected row is drawn with a leading ``>`` marker
and a bold name.  This base owns the cursor (index + move semantics) and the
two markup conventions every row shares; subclasses keep their own render loop
(the row bodies differ too much to unify — alpha bars, sparklines, manifold
rows, category headers) and tell the base how long the list is via
``_cursor_len``.

The two panels differ in cursor wrap behavior — the vector rack wraps around
the ends, the trait list clamps — so ``_cursor_wrap`` is a class flag.  They
also differ in whether the backing list is stored ahead of render (the vector
rack) or rebuilt during render (the trait list walks categorized probes), so
the base reads the length through ``_cursor_len`` rather than owning the list.
"""

from __future__ import annotations

from typing import Any

from textual.widget import Widget


class SelectableListWidget(Widget):
    """Cursor + selection markup for a navigable list panel.

    Subclasses set ``_cursor_wrap`` (wrap around the ends vs clamp), implement
    ``_cursor_len`` (the current number of selectable entries), and re-render
    whenever ``cursor_next``/``cursor_prev`` report a move.
    """

    #: Whether ``cursor_next``/``cursor_prev`` wrap around the list ends
    #: (True — vector rack) or clamp at them (False — trait list).
    _cursor_wrap: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cursor_idx: int = 0

    def _cursor_len(self) -> int:
        """Number of selectable entries.  Subclasses override."""
        return 0

    def cursor_next(self) -> bool:
        """Advance the cursor; return True if it moved (caller re-renders)."""
        n = self._cursor_len()
        if not n:
            return False
        if self._cursor_wrap:
            self._cursor_idx = (self._cursor_idx + 1) % n
            return True
        if self._cursor_idx < n - 1:
            self._cursor_idx += 1
            return True
        return False

    def cursor_prev(self) -> bool:
        """Retreat the cursor; return True if it moved (caller re-renders)."""
        n = self._cursor_len()
        if not n:
            return False
        if self._cursor_wrap:
            self._cursor_idx = (self._cursor_idx - 1) % n
            return True
        if self._cursor_idx > 0:
            self._cursor_idx -= 1
            return True
        return False

    @staticmethod
    def selection_marker(is_selected: bool) -> str:
        """The leading ``>``/space cursor glyph for a row."""
        return ">" if is_selected else " "

    @staticmethod
    def name_markup(name: str, is_selected: bool, enabled: bool = True) -> str:
        """Style a row name by selection + enabled state.

        Selected rows bold the name; disabled rows dim it; the combination
        dims-and-bolds.  This is the shared convention across both panels'
        rows (the vector rack's enabled flag, the trait list's always-enabled
        rows).
        """
        if is_selected and enabled:
            return f"[bold]{name}[/]"
        if is_selected and not enabled:
            return f"[dim bold]{name}[/]"
        if not enabled:
            return f"[dim]{name}[/]"
        return name
