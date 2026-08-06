// The tool registry behind the ⌘K command palette — the one launcher for
// every analysis/session tool. The Rail* type names are retained as internal
// registry vocabulary; the palette flattens the categories and adds direct
// navigation entries for instrument tabs and pages.

import type { DrawerName } from "./types";
import type { InspectorTab } from "./stores.svelte";

export interface RailTool {
  label: string;
  drawer: DrawerName;
  /** Extra match text for palette filtering (synonyms, old names). */
  keywords?: string;
}

export interface RailCategory {
  key: string;
  label: string;
  tools: RailTool[];
}

export const RAIL_CATEGORIES: RailCategory[] = [
  {
    // The single steering-authoring surface.  Concepts are manifolds now —
    // a flat (2-node / personas) fit is just a pca manifold — so there's no
    // separate "subspaces" category; flat authoring folds into the manifold
    // builder's pca path.  The catalog is the shared RackDrawer
    // (family-split), reached from the rack "+" buttons.
    key: "manifolds",
    label: "Steering",
    tools: [
      {
        label: "build…",
        drawer: "manifold_builder",
        keywords: "extract author create concept vector fit",
      },
      {
        label: "merge…",
        drawer: "manifold_merge",
        keywords: "union corpora",
      },
      {
        label: "packs…",
        drawer: "manifold_pack",
        keywords: "install search huggingface hub catalog",
      },
      {
        label: "templates…",
        drawer: "template_lab",
        keywords: "score completion slot restricted choice",
      },
      {
        label: "cast…",
        drawer: "cast",
        keywords: "roster member speaker label recipe seat role",
      },
    ],
  },
  {
    key: "analysis",
    label: "Analysis",
    tools: [
      {
        label: "correlation…",
        drawer: "correlation",
        keywords: "cosine similarity vectors",
      },
      {
        label: "compare…",
        drawer: "compare",
        keywords: "cross-layer cosine",
      },
    ],
  },
  {
    key: "session",
    label: "Session",
    tools: [
      { label: "health…", drawer: "health", keywords: "device dtype" },
      {
        label: "auth…",
        drawer: "session_admin",
        keywords: "api key bearer",
      },
      {
        label: "help…",
        drawer: "help",
        keywords: "keyboard grammar cheatsheet",
      },
    ],
  },
];

// ---------------------------------------------------------- palette ------

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
