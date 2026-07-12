"""Input-history + pending-queue controller for the saklas TUI.

Owns the two coupled rings the chat input walks with ``↑``/``↓``:

* the shell-style **input history** (``_input_history`` / ``_history_index``
  / ``_history_stash``) — every submitted line, capped at
  ``_INPUT_HISTORY_MAX``, with readline-flavored de-dupe; and
* the mid-generation **pending queue** (``_pending_queue`` / ``_pulled_slot``)
  — submissions deferred behind an in-flight gen, drained one per ``done``.

The two are one coupled state machine: ``_history_navigate`` walks a single
combined ring ``[pending_queue (newest-first), input_history (newest-first)]``
and ``_pulled_slot`` is an index into ``_pending_queue``, so they cannot be
split apart.  This controller owns all five mutable attrs plus the queue
mechanics + the ↑/↓ ring; the *business logic* of each drained kind lives
elsewhere — ``_drain_next_pending`` hands each popped item to the App's
``_dispatch_pending_action`` router, which fans out to the generation methods
and the other controllers.

Composed onto :class:`~saklas.tui.app.SaklasApp` via a back-reference
(``self._app``).  The framework-dispatched surface (``on_key`` history recall,
``action_stop_generation`` cancel, the submit/commit handlers) stays on the
App as thin forwarders into this class; the App also keeps
``_dispatch_pending_action`` and the ``_is_busy`` gate (it reads the queue
through the proxy property below).  The App exposes ``_input_history`` /
``_history_index`` / ``_history_stash`` / ``_pending_queue`` / ``_pulled_slot``
as read/write proxy properties into this controller, so the framework
handlers, the ``_is_busy`` gate, the auto-regen branch, and the test suite all
reach the same state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from textual.css.query import NoMatches

from saklas.tui.chat_panel import (
    _KIND_CHAIN_INLINE,
    ChatInput,
    PendingItem,
)

if TYPE_CHECKING:
    from saklas.tui.app import SaklasApp


# keep ``_input_history`` from growing without limit over a long session
# (process-scoped, in-memory; no persistence)
_INPUT_HISTORY_MAX = 200


class InputHistoryController:
    """Owns the input-history ring + pending queue and their ↑/↓ walk."""

    def __init__(self, app: "SaklasApp") -> None:
        self._app = app
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
        # Mid-generation submissions queue here rather than tearing the
        # in-flight stream down.  Drained one item per ``done`` event in
        # the App's :meth:`_poll_generation` via :meth:`_drain_next_pending`;
        # the ``PendingStrip`` widget in the chat panel renders the live
        # list so the user can see what's queued and pull items back
        # for edit with ``↑``.
        self._pending_queue: list[PendingItem] = []

    def release_pulled_slot(self) -> int | None:
        """Atomically return and clear the currently edited queue slot."""
        slot = self._pulled_slot
        if slot is None:
            return None
        self._pulled_slot = None
        self._sync_pull_state()
        return slot

    def clear_pending_edit_state(self) -> None:
        self._pending_queue.clear()
        self._pulled_slot = None
        self._sync_pull_state()

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
            inp = self._app.query_one("#chat-input", ChatInput)
        except NoMatches:
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
            inp = self._app.query_one("#chat-input", ChatInput)
        except NoMatches:
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
            inp = self._app.query_one("#chat-input", ChatInput)
        except NoMatches:
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
        self._app._refresh_input_mode()

    def _remove_pending_slot(self, slot: int) -> None:
        """Drop a pulled slot from the queue (empty-input re-submit / cancel)."""
        q = self._pending_queue
        if 0 <= slot < len(q):
            del q[slot]
            self._refresh_pending_strip()
            self._app._refresh_input_mode()

    def _refresh_pending_strip(self) -> None:
        """Push the queue + pulled-slot into the chat panel's strip.

        Single funnel so callers don't have to remember to forward
        ``_pulled_slot``; any mutation that touches either the queue
        or the pulled state should call this.
        """
        self._app._chat_panel.update_pending(
            self._pending_queue, pulled_slot=self._pulled_slot,
        )

    def _drain_next_pending(self) -> bool:
        """Drain queued items in FIFO until an async kind takes over.

        Returns ``True`` when at least one item ran, ``False`` when the
        queue was empty.  Called from the App's :meth:`_poll_generation`
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

        Each popped item is handed to the App's
        :meth:`_dispatch_pending_action` router, which owns the
        business logic of each kind (generation / commit / steer);
        this controller owns only the queue mechanics around it.
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
            self._app._dispatch_pending_action(item)
            drained = True
            if item.kind not in _KIND_CHAIN_INLINE:
                # Async kind — let its own done sentinel drive the
                # next drain so the worker isn't raced.
                break
        return drained
