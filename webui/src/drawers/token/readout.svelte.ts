// Shared captured-or-replay readout resource for the drilldown's
// instrument tabs.
//
// Every replay-capable family (geometry / sae / lens) reads the same
// way: prefer the token's loom-owned ``measurements`` envelope (original
// capture), otherwise hit the family's ``token-readout`` replay endpoint
// — request-sequenced so a stale response can never clobber a newer
// view.  Pre-refactor each tab hand-rolled this ~50-line dance and they
// drifted (the sae tab lost the steered toggle); this class is the one
// implementation.

import { ApiError, apiInstruments } from "../../lib/api";
import type { InstrumentFamily } from "../../lib/api";
import type {
  MeasurementsEnvelopeJSON,
  ProbeReadingJSON,
} from "../../lib/types";

export type ReadoutOrigin = "captured" | "replayed" | null;

/** The geometry tab's render shape — the Monitor-roster readings plus
 *  the steering the read ran under (captured: the recipe; replayed: the
 *  binding the endpoint reports). */
export interface GeometryTokenReadout {
  steering: string | null;
  readings: Record<string, ProbeReadingJSON>;
}

/** Prefer the structured ``detail`` a saklas error body carries over the
 *  generic HTTP message. */
export function errorDetail(e: unknown): string {
  if (
    e instanceof ApiError &&
    typeof (e.body as { detail?: unknown } | null)?.detail === "string"
  ) {
    return (e.body as { detail: string }).detail;
  }
  return e instanceof Error ? e.message : String(e);
}

export class ReplayReadout<T> {
  data: T | null = $state(null);
  loading: boolean = $state(false);
  error: string | null = $state(null);
  origin: ReadoutOrigin = $state(null);
  source: string | null = $state(null);
  #seq = 0;

  /** Adopt the loom-captured envelope view directly (no fetch). */
  adopt(data: T, source: string | null): void {
    this.#seq++;
    this.data = data;
    this.origin = "captured";
    this.source = source;
    this.loading = false;
    this.error = null;
  }

  /** Drop the current view (and invalidate any in-flight fetch). */
  clear(): void {
    this.#seq++;
    this.data = null;
    this.origin = null;
    this.source = null;
    this.loading = false;
    this.error = null;
  }

  /** Fetch the family's token-readout replay and map the returned
   *  measurements envelope into the tab's render shape.  ``map`` may
   *  return a null source; the caller-supplied ``fallbackSource`` covers
   *  bindings from older servers. */
  replay(
    family: InstrumentFamily,
    nodeId: string,
    rawIndex: number,
    opts: { topK?: number; steered?: boolean; raw?: boolean; layers?: string },
    map: (env: MeasurementsEnvelopeJSON) => {
      data: T;
      source: string | null;
    },
  ): void {
    const seq = ++this.#seq;
    this.loading = true;
    this.error = null;
    this.data = null;
    this.origin = null;
    this.source = null;
    apiInstruments
      .tokenReadout(family, nodeId, rawIndex, opts)
      .then((res) => {
        if (seq !== this.#seq) return;
        const { data, source } = map(res.measurements);
        this.data = data;
        this.source = source;
        this.origin = "replayed";
      })
      .catch((e) => {
        if (seq !== this.#seq) return;
        this.error = errorDetail(e);
      })
      .finally(() => {
        if (seq === this.#seq) this.loading = false;
      });
  }
}
