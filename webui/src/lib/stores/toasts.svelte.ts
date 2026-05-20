export interface Toast {
  id: number;
  kind: "info" | "warning" | "error";
  message: string;
  /** Optional second line for live progress — dimmer than ``message``. */
  detail?: string | null;
  /** ``null`` = sticky (no auto-dismiss); caller owns dismissal. */
  ttlMs: number | null;
}

export const toasts: { entries: Toast[] } = $state({ entries: [] });

let _toastSeq = 0;

export function pushToast(
  message: string,
  opts: {
    kind?: Toast["kind"];
    /** ``null`` keeps the toast on screen until ``dismissToast`` is called. */
    ttlMs?: number | null;
    detail?: string | null;
  } = {},
): number {
  const id = ++_toastSeq;
  toasts.entries = [
    ...toasts.entries,
    {
      id,
      kind: opts.kind ?? "info",
      message,
      detail: opts.detail ?? null,
      ttlMs: opts.ttlMs === undefined ? 6000 : opts.ttlMs,
    },
  ];
  return id;
}

/** Mutate a live toast in place — used by long-running progress reporters
 * to keep updating the same chip without spawning new ones.  Only the
 * provided keys change; ``ttlMs`` updates are honored but the Toaster
 * only schedules a timer the first time it sees a non-null ttl, so
 * flipping sticky → ttl mid-flight does not start a timer.  Flip in the
 * other direction (sticky → keep sticky) is the only safe transition;
 * callers that want a finite TTL at the end should ``dismissToast`` +
 * ``pushToast`` instead. */
export function updateToast(
  id: number,
  patch: Partial<Pick<Toast, "message" | "detail" | "kind" | "ttlMs">>,
): void {
  toasts.entries = toasts.entries.map((t) =>
    t.id === id ? { ...t, ...patch } : t,
  );
}

export function dismissToast(id: number): void {
  toasts.entries = toasts.entries.filter((t) => t.id !== id);
}
