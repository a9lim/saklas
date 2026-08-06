// The loaded session — the one client mirror of ``GET /sessions/current``.
//
// ``sessionState.info`` is the whole server-side session shape (model id,
// config defaults, the per-family instrument blocks, the chat-template
// role labels), so almost every other slice reads from here.
// ``refreshSession`` is the single write: fetch, store, then hand the two
// derived mirrors — sampling defaults and instrument live/source state —
// back to the slices that own them.

import { apiSessions } from "../api";
import type {
  InstrumentFamily,
  InstrumentFamilyBlock,
  SessionInfo,
} from "../types";
import { hydrateSamplingFromInfo } from "./sampling.svelte";
import { rehydrateInstrumentsFromSession } from "./instruments.svelte";

export interface SessionState {
  /** Loaded.  Set by ``refreshSession``; null while bootstrapping. */
  info: SessionInfo | null;
  /** Last refresh timestamp (ms since epoch).  Used by panels to gate
   * spinners against stale-but-valid data. */
  lastRefresh: number | null;
  /** Last fetch error, if any; cleared on next successful refresh. */
  error: string | null;
}

export const sessionState: SessionState = $state({
  info: null,
  lastRefresh: null,
  error: null,
});

/** One read family's block from session info — the SAME shape
 *  ``GET .../instruments`` lists, so there is one representation of
 *  instrument state to read from. */
export function instrumentFamily(
  family: InstrumentFamily,
): InstrumentFamilyBlock | undefined {
  return sessionState.info?.instruments?.find((row) => row.family === family);
}

/** True iff an SAE is resident (its family block reports an active
 *  source).  Replaces the flat ``sae_loaded`` key. */
export function saeLoaded(): boolean {
  return instrumentFamily("sae")?.source != null;
}

export async function refreshSession(): Promise<void> {
  try {
    const info = await apiSessions.get();
    sessionState.info = info;
    sessionState.lastRefresh = Date.now();
    sessionState.error = null;
    hydrateSamplingFromInfo();
    rehydrateInstrumentsFromSession();
  } catch (e) {
    sessionState.error = e instanceof Error ? e.message : String(e);
  }
}
