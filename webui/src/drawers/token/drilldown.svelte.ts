// Page-session drilldown UI state.
//
// The selected tab is STICKY across token clicks and drawer reopens — a
// research pass compares one instrument across many tokens, and the old
// per-open reset kept bouncing the user off it.  Module scope = page
// session; the default is the j-lens workspace readout.

export type DrilldownTab = "geometry" | "logits" | "sae" | "lens";

export const drilldownUi: { tab: DrilldownTab } = $state({ tab: "lens" });
