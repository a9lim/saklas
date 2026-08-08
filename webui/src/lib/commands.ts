// The ⌘K command palette's index.
//
// The launcher categories are NOT declared here — they are derived from
// the drawer registry (``drawers/index.ts``), which is the one place a
// drawer's component, sizing, and reachability are written down.  This
// module only flattens them and adds the direct navigation entries for the
// instrument tabs.

import { RAIL_CATEGORIES } from "../drawers";
import type { DrawerName } from "./types";
import type { InspectorTab } from "./stores.svelte";

export type { RailCategory, RailTool } from "../drawers";
export { RAIL_CATEGORIES } from "../drawers";

export type PaletteAction =
  | { kind: "drawer"; drawer: DrawerName }
  | { kind: "tab"; tab: InspectorTab };

export interface PaletteCommand {
  label: string;
  group: string;
  action: PaletteAction;
  keywords?: string;
}

/** The flattened palette index: instrument-tab jumps first (the four
 *  pillars are the primary navigation), then every registry tool, then pages. */
export function paletteCommands(): PaletteCommand[] {
  const cmds: PaletteCommand[] = [
    {
      label: "subspace",
      group: "instruments",
      action: { kind: "tab", tab: "subspace" },
      keywords: "pillar flat affine concept vector caa steer probe",
    },
    {
      label: "manifold",
      group: "instruments",
      action: { kind: "tab", tab: "manifold" },
      keywords: "pillar curved emotions months steer probe",
    },
    {
      label: "sae",
      group: "instruments",
      action: { kind: "tab", tab: "sae" },
      keywords: "pillar features sparse autoencoder",
    },
    {
      label: "lens",
      group: "instruments",
      action: { kind: "tab", tab: "lens" },
      keywords: "pillar jacobian jlens workspace readout token",
    },
  ];
  for (const cat of RAIL_CATEGORIES) {
    for (const tool of cat.tools) {
      cmds.push({
        label: tool.label.replace(/…$/, ""),
        group: cat.label.toLowerCase(),
        action: { kind: "drawer", drawer: tool.drawer },
        keywords: tool.keywords,
      });
    }
  }
  return cmds;
}
