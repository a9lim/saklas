export interface Toast {
  id: number;
  kind: "info" | "warning" | "error";
  message: string;
  ttlMs: number;
}

export const toasts: { entries: Toast[] } = $state({ entries: [] });

let _toastSeq = 0;

export function pushToast(
  message: string,
  opts: { kind?: Toast["kind"]; ttlMs?: number } = {},
): number {
  const id = ++_toastSeq;
  toasts.entries = [
    ...toasts.entries,
    {
      id,
      kind: opts.kind ?? "info",
      message,
      ttlMs: opts.ttlMs ?? 6000,
    },
  ];
  return id;
}

export function dismissToast(id: number): void {
  toasts.entries = toasts.entries.filter((t) => t.id !== id);
}
