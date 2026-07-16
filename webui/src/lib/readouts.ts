/** Canonical read-side width when logit alternatives are disabled. */
export const DEFAULT_READOUT_TOP_K = 8;

/** Resolve the one top-k shared by alts, J-lens, and SAE readouts. */
export function resolveReadoutTopK(returnTopK: number): number {
  return returnTopK || DEFAULT_READOUT_TOP_K;
}
