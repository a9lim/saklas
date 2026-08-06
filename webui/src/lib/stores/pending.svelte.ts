// The deferred-mutation queue.
//
// Everything the user does while a generation is in flight lands here
// instead of racing the stream: composer sends, rack mutations, /clear.
// The queue drains one item per WS ``done`` (``awaitsGen`` items wait for
// their own generation; instant ones chain straight into the next drain),
// which is what keeps a send-then-send pair in order.
//
// ``enqueueOrApply`` is the rack half — apply now when idle, coalesce onto
// the trailing rack bubble when busy, so a slider drag leaves one queued
// bubble carrying the net effect instead of thirty ghosts.

import { chatLog, genStatus } from "../stores.svelte";
import type { PendingAction } from "../types";
import {
  onPendingQueueShift,
  requestInputRestore,
} from "./inputHistory.svelte";

export interface PendingActionsState {
  /** Queue of mutations and submissions deferred while a generation is
   * running.  Drained one item per WS ``done`` event by
   * :func:`drainNextPendingAction`.  Per-item cancel goes through
   * :func:`cancelPendingAction` (the GUI's per-bubble ``×``). */
  queue: PendingAction[];
}

export const pendingActions: PendingActionsState = $state({ queue: [] });

let _pendingCounter = 0;

/** Next queue-item id.  Exported because the WS submit path builds its own
 *  ``PendingAction`` (it needs the item before deciding whether to queue
 *  it), and both id streams have to come from one counter. */
export function nextPendingId(): string {
  return `pa-${_pendingCounter++}`;
}

/** Append a pending action.  ``replaceSlot`` slots a re-edited
 *  pulled item back into its original position rather than the queue
 *  tail; out-of-range values fall back to append. */
export function enqueuePending(
  action: Omit<PendingAction, "id" | "createdAt">,
  opts: { replaceSlot?: number | null } = {},
): void {
  const item: PendingAction = {
    ...action,
    id: nextPendingId(),
    createdAt: Date.now(),
  };
  const slot = opts.replaceSlot ?? null;
  if (slot !== null && slot >= 0 && slot < pendingActions.queue.length) {
    pendingActions.queue[slot] = item;
  } else {
    pendingActions.queue.push(item);
  }
}

/** Drain a single pending item.  Called from the WS ``done`` /
 *  ``error`` handlers; an ``awaitsGen=false`` item chains into the
 *  next drain immediately so a sequence of instant mutations
 *  (clear, rewind) doesn't stall waiting for a gen that's never
 *  going to fire.
 *
 *  Replaces the v1.x ``applyPendingActions`` which drained *all*
 *  items on every ``done`` — the v2.x queue semantics serialize
 *  one item per ``done`` so a send-then-send pair runs in order
 *  rather than racing the WS. */
export async function drainNextPendingAction(): Promise<void> {
  if (pendingActions.queue.length === 0) return;
  // Reconcile the input-history pull state before mutating the queue:
  // a drained head whose slot the user is editing would otherwise
  // leave a dangling ``pulledSlot`` pointing past the array.  The
  // shift helper returns the stash text when slot 0 was pulled; we
  // park it on ``inputRestore`` so Chat.svelte's $effect copies it
  // back into the textarea on the next tick.
  const restore = onPendingQueueShift();
  if (restore !== null) requestInputRestore(restore);
  const item = pendingActions.queue.shift();
  if (!item) return;
  try {
    await item.apply();
  } catch (e) {
    chatLog.turns = [
      ...chatLog.turns,
      {
        role: "system",
        text: `pending ${item.label} failed: ${String(e)}`,
      },
    ];
  }
  if (!item.awaitsGen) {
    // Instant mutation finished — chain into the next item so the
    // queue doesn't stall waiting for a ``done`` that won't fire.
    await drainNextPendingAction();
  }
}

/** Remove one pending item by id (GUI per-bubble ``×``). */
export function cancelPendingAction(id: string): void {
  pendingActions.queue = pendingActions.queue.filter((p) => p.id !== id);
}

/** Apply immediately if no gen is in flight AND the queue is empty;
 *  queue otherwise.  The queue check matters: with one or more items
 *  already pending, applying a fresh rack mutation immediately would
 *  break FIFO ordering of state mutations.
 *
 *  Used by the rack/sampling mutations — they don't kick off a gen, so
 *  ``awaitsGen=false`` lets the drain chain through them without
 *  waiting on the next ``done``.
 *
 *  Consecutive queued rack mutations *coalesce*: when the queue tail is
 *  already a rack-mutation item, the fresh ``apply`` is chained onto it
 *  rather than appended as a new slot, and the bubble's label updates
 *  to the latest action.  A slider drag that fires 30+ intermediate
 *  ``setSubspaceAlong`` calls mid-gen therefore leaves a single queued
 *  bubble carrying the net effect — "one final steering adjustment" —
 *  instead of 30 stacked ghosts.  Coalescing stops at any non-rack
 *  item (send / commit / one-shot mutation): rack changes before and
 *  after a queued send form distinct groups so FIFO ordering relative
 *  to the send is preserved. */
const RACK_COALESCE_KEY = "rack";
export function enqueueOrApply(label: string, apply: () => void): void {
  if (!(genStatus.active || pendingActions.queue.length > 0)) {
    apply();
    return;
  }
  const q = pendingActions.queue;
  const tail = q[q.length - 1];
  if (tail && tail.coalesceKey === RACK_COALESCE_KEY) {
    // Chain onto the trailing rack bubble.  Reassign the slot so the
    // $state array fires reactivity for PendingBubbles / the queue
    // count badge (in-place field writes on a proxied item are tracked,
    // but reassigning is the uniform pattern used across this module).
    const prev = tail.apply;
    q[q.length - 1] = {
      ...tail,
      label,
      apply: async () => {
        await prev();
        await apply();
      },
    };
    return;
  }
  enqueuePending({
    label, text: null, apply, awaitsGen: false, rebuild: null,
    coalesceKey: RACK_COALESCE_KEY,
  });
}

/** Are we busy enough that fresh submissions should queue instead of fire?
 *  Busy means a generation is running or earlier items await their turn. */
export function isPendingBusy(): boolean {
  return genStatus.active || pendingActions.queue.length > 0;
}

/** Predict the post-queue active-node-is-user-role flag.  Walks the
 *  queue tail-first and returns the first non-null ``endsOnUserNode``;
 *  returns ``null`` when no queued item changes the role (e.g. the
 *  queue is empty or only carries rack mutations).  Drives the chat
 *  input's role-aware placeholder + send-button label so a queued
 *  user-seat submission flips the next one's predicted seat without
 *  waiting for the queue to drain. */
export function predictedQueueEndOnUserNode(): boolean | null {
  for (let i = pendingActions.queue.length - 1; i >= 0; i--) {
    const e = pendingActions.queue[i].endsOnUserNode;
    if (e !== undefined && e !== null) return e;
  }
  return null;
}
