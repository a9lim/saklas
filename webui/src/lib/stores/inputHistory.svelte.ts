/** In-memory command/input recall ring. */

import { pendingActions } from "../stores.svelte";

export const INPUT_HISTORY_MAX = 200;

export interface InputHistoryState {
  /** Submitted lines, oldest first. */
  entries: string[];
  /** Cursor into ``entries`` while recall is in flight (``null`` =
   * either at live slot or pulled into a pending item). */
  index: number | null;
  /** Draft text captured when recall begins. */
  stash: string;
  /** Slot index into ``pendingActions.queue`` while a pending item
   * is pulled into the input for edit.  ``null`` = no pull in flight.
   * The TUI uses the same pattern in ``SaklasApp._pulled_slot``. */
  pulledSlot: number | null;
}

export const inputHistory: InputHistoryState = $state({
  entries: [],
  index: null,
  stash: "",
  pulledSlot: null,
});

/** Cross-component restoration channel for the textarea.
 *
 *  When a queue mutation cancels the user's pull from outside the
 *  Chat component (e.g. a drain pops the slot they were editing),
 *  we can't write directly into Chat.svelte's local ``input``
 *  ``$state`` from here.  Instead the cancel path bumps this counter
 *  and parks the text Chat.svelte should restore; an ``$effect`` in
 *  Chat watches ``rev`` and copies ``text`` into the input on each
 *  bump. */
export const inputRestore: { rev: number; text: string } = $state({
  rev: 0, text: "",
});

export function requestInputRestore(text: string): void {
  inputRestore.text = text;
  inputRestore.rev += 1;
}

export function pushInputHistory(text: string): void {
  const trimmed = text.trim();
  if (!trimmed) return;
  const entries = inputHistory.entries;
  const last = entries.length > 0 ? entries[entries.length - 1] : null;
  if (last !== trimmed) {
    const next = [...entries, trimmed];
    inputHistory.entries = next.length > INPUT_HISTORY_MAX
      ? next.slice(next.length - INPUT_HISTORY_MAX)
      : next;
  }
  inputHistory.index = null;
  inputHistory.stash = "";
  inputHistory.pulledSlot = null;
}

/** Walk the combined ring of pending items + input history.
 *
 * Pending items come first (most-recently-queued is one ``↑`` from
 * live, oldest pending is one further ``↑``), then committed input
 * history (newest first).  ``↓`` walks the same ring in reverse and
 * restores the stash captured on the first ``↑`` when it returns to
 * the live slot.
 *
 * Pulling a pending item sets :attr:`InputHistoryState.pulledSlot` —
 * Chat.svelte forwards it to ``sendGenerate`` / ``sendCommit`` /
 * ``sendPrefill`` as ``replaceSlot`` so a re-submitted edit lands at
 * its original slot rather than appending to the queue tail.  An
 * empty re-submit on a pulled slot is the cancel gesture; the GUI's
 * per-bubble ``×`` is the symmetric mouse path.
 */
export function navigateInputHistory(
  delta: -1 | 1,
  currentInput: string,
): string | null {
  const entries = inputHistory.entries;
  const pending = pendingActions.queue;
  // Editable pending items only — non-editable mutations (rebuild===null)
  // sit on the queue but can't be pulled back for re-edit; ``↑`` skips
  // over them entirely.
  const editablePending = pending.filter((p) => p.rebuild !== null);
  const nP = editablePending.length;
  const nH = entries.length;
  if (nP === 0 && nH === 0) return null;

  // Combined-ring position encoding:
  //   pos in [0, nP)               — editable pending; pos=0 = most recent
  //   pos in [nP, nP + nH)         — history offset; pos=nP = newest
  //   pos == -1 (sentinel)         — live slot
  const curPos = currentCursorPos(editablePending);
  let newPos: number;
  if (curPos < 0) {
    if (delta > 0) return null;  // already at live — ↓ is no-op
    inputHistory.stash = currentInput;
    newPos = 0;
  } else {
    // ↑ (delta=-1) walks toward older items — increment pos.
    // ↓ (delta=+1) walks toward newer — decrement.
    newPos = curPos + (delta < 0 ? 1 : -1);
    if (newPos >= nP + nH) {
      newPos = nP + nH - 1;  // clamp at oldest (bash/readline)
    } else if (newPos < 0) {
      // Past the newest — back to live.
      inputHistory.pulledSlot = null;
      inputHistory.index = null;
      const stash = inputHistory.stash;
      inputHistory.stash = "";
      return stash;
    }
  }

  if (newPos < nP) {
    const item = editablePending[nP - 1 - newPos];
    // Find its real slot in the unfiltered queue.
    const realSlot = pending.indexOf(item);
    inputHistory.pulledSlot = realSlot >= 0 ? realSlot : null;
    inputHistory.index = null;
    return item.text;
  }
  inputHistory.pulledSlot = null;
  inputHistory.index = nH - 1 - (newPos - nP);
  return entries[inputHistory.index];
}

function currentCursorPos(editablePending: typeof pendingActions.queue): number {
  const nP = editablePending.length;
  if (inputHistory.pulledSlot !== null) {
    const pulled = pendingActions.queue[inputHistory.pulledSlot];
    if (pulled !== undefined) {
      const ePos = editablePending.indexOf(pulled);
      if (ePos >= 0) return nP - 1 - ePos;
    }
  }
  if (inputHistory.index !== null) {
    const nH = inputHistory.entries.length;
    if (inputHistory.index >= 0 && inputHistory.index < nH) {
      return nP + (nH - 1 - inputHistory.index);
    }
  }
  return -1;
}

/** Reconcile the pull state when a queue head pop happens elsewhere.
 *
 *  * pulled slot == 0 → the very item the user was editing just
 *    dispatched.  Cancel the pull and return the stash so the caller
 *    can restore the input.
 *  * pulled slot > 0 → decrement so the user keeps tracking the same
 *    item.  Returns ``null`` (no input change).
 *  * not pulled → no-op, returns ``null``.
 *
 *  Mirrors the TUI's ``_drain_next_pending``'s pulled-slot fixup. */
export function onPendingQueueShift(): string | null {
  const slot = inputHistory.pulledSlot;
  if (slot === null) return null;
  if (slot === 0) {
    const stash = inputHistory.stash;
    inputHistory.pulledSlot = null;
    inputHistory.index = null;
    inputHistory.stash = "";
    return stash;
  }
  inputHistory.pulledSlot = slot - 1;
  return null;
}

/** ``Esc`` while a pending slot is pulled — restore the live stash.
 *  Slot stays in the queue untouched; the user backed out of the edit
 *  but didn't cancel the queued action.  Returns the stash text the
 *  caller should put back in the input (or ``null`` if nothing was
 *  pulled). */
export function cancelInputPull(): string | null {
  if (inputHistory.pulledSlot === null) return null;
  const stash = inputHistory.stash;
  inputHistory.pulledSlot = null;
  inputHistory.index = null;
  inputHistory.stash = "";
  return stash;
}

/** Consume + clear the currently-pulled slot index.  Called by the
 *  submit path so the caller can pass it as ``replaceSlot`` to the
 *  send / commit / prefill helpers, then reset cleanly. */
export function consumePulledSlot(): number | null {
  const slot = inputHistory.pulledSlot;
  inputHistory.pulledSlot = null;
  inputHistory.stash = "";
  return slot;
}
