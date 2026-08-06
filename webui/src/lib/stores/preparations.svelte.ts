// THE client half of the unified background-preparation resource.
//
// The server exposes one `/instruments/{family}/preparations` job per
// family (lens: fetch | fit; sae: load | train).  Each of those four
// operations gets one slice built here, so they share one await / error /
// resume contract instead of four hand-written near-copies that drift:
//
//   * `start` posts, reflects the returned status, then fires the poll
//     loop and returns — never awaits the job to completion.
//   * the poll loop is re-entrancy-guarded, so N panels asking for the
//     same job still run one interval.
//   * `cancel` is a no-op unless the job is running and not already
//     cancelling.
//   * `check` is the mount-time resume probe: reflect the running-or-last
//     job when it is *this* operation (the two operations of a family
//     share one resource), and pick the poll loop back up if it is still
//     running.
//   * settle toasts are uniform — cancelled → info, error → sticky error,
//     otherwise the slice's success line.

import { apiInstruments } from "../api";
import type { InstrumentFamily } from "../api";
import type { PreparationOp, PreparationStatusJSON } from "../types";
import { pushToast } from "./toasts.svelte";

/** One preparation's mirror of the server job status. */
export interface PreparationState {
  running: boolean;
  /** Progress in the job's own unit (fit: prompts, train: tokens); both 0
   *  when the job reports none (fetch / load are message-only). */
  current: number;
  total: number;
  message: string | null;
  error: string | null;
  /** Poll-loop guard — one interval regardless of how many panels ask. */
  polling: boolean;
  cancelling: boolean;
}

export interface PreparationSlice {
  readonly state: PreparationState;
  /** Start the job.  Extra body fields ride the operation's request
   *  model (fit: prompts/layers, train: name/layer/tokens/width, …). */
  start(body?: Record<string, unknown>): Promise<void>;
  /** Cooperative cancel; a no-op for a non-cancellable operation. */
  cancel(): Promise<void>;
  /** Mount-time resume probe. */
  check(): Promise<void>;
}

export interface PreparationSliceOptions {
  /** Human name leading every toast — "J-lens fit", "SAE train", … */
  label: string;
  /** Poll interval in ms. */
  intervalMs: number;
  /** Info toast on a clean finish. */
  successMessage: string;
  /** Run after the job leaves `running` — the session / source / probe
   *  refreshes this operation's result invalidates.  Awaited before the
   *  settle toast so the panel is already consistent when it fires. */
  onSettled?: () => void | Promise<void>;
}

function describe(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function createPreparationSlice(
  family: InstrumentFamily,
  operation: PreparationOp,
  { label, intervalMs, successMessage, onSettled }: PreparationSliceOptions,
): PreparationSlice {
  const state: PreparationState = $state({
    running: false,
    current: 0,
    total: 0,
    message: null,
    error: null,
    polling: false,
    cancelling: false,
  });

  function apply(st: PreparationStatusJSON): void {
    state.running = st.state === "running";
    state.current = st.progress?.current ?? 0;
    state.total = st.progress?.total ?? 0;
    state.message = st.message;
    state.error = st.error;
    if (st.state !== "running") state.cancelling = false;
  }

  async function settle(st: PreparationStatusJSON): Promise<void> {
    await onSettled?.();
    if (st.finished_at === null) return;
    if (st.message === "cancelled") {
      pushToast(`${label} cancelled`, { kind: "info" });
    } else if (st.error) {
      pushToast(`${label}: ${st.error}`, { kind: "error", ttlMs: null });
    } else {
      pushToast(successMessage, { kind: "info" });
    }
  }

  async function poll(): Promise<void> {
    if (state.polling) return;
    state.polling = true;
    try {
      for (;;) {
        const st = await apiInstruments.preparationStatus(family);
        apply(st);
        if (st.state !== "running") {
          await settle(st);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
      }
    } catch (e) {
      state.error = describe(e);
    } finally {
      state.polling = false;
    }
  }

  return {
    state,

    async start(body: Record<string, unknown> = {}): Promise<void> {
      if (state.running || state.polling) return;
      try {
        apply(
          await apiInstruments.startPreparation(family, { operation, ...body }),
        );
      } catch (e) {
        state.running = false;
        state.error = describe(e);
        pushToast(`${label}: ${state.error}`, { kind: "error" });
        return;
      }
      void poll();
    },

    async cancel(): Promise<void> {
      if (!state.running || state.cancelling) return;
      state.cancelling = true;
      try {
        apply(await apiInstruments.cancelPreparation(family));
        pushToast(`${label} cancelling…`, { kind: "info" });
      } catch (e) {
        state.cancelling = false;
        pushToast(`${label} cancel: ${describe(e)}`, { kind: "error" });
      }
    },

    async check(): Promise<void> {
      if (state.polling) return;
      try {
        // The two operations of a family share one preparations resource,
        // so only reflect the running-or-last job when it is this one.
        const st = await apiInstruments.preparationStatus(family);
        if (st.operation !== operation) return;
        apply(st);
        if (st.state === "running") void poll();
      } catch {
        // A mount-time status probe is optional: with no reachable job
        // status the source section still authors and submits normally.
      }
    },
  };
}
