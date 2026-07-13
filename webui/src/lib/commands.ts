// The tool registry behind the ⌘K command palette — the one launcher for
// every analysis/session tool.  (The former workspace rail rendered these
// categories as icon fly-outs; the rail is gone, and the palette flattens
// them and adds its own navigation entries: instrument tabs, pages.  The
// Rail* names and per-category icons keep the registry shape.)

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
  /** SVG path data for a 24×24 category glyph (currently unrendered —
   *  kept for a future launcher surface). */
  icon: string;
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
    // Undulating spline curve — reads as "manifold" and is visually
    // distinct from analysis' line graph.
    icon: "M3 17c4-8 6-8 9-4s2 8 9 0",
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
    icon: "M4 18l5-12 4 8 3-5 4 9",
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
      { label: "layer norms…", drawer: "layer_norms", keywords: "magnitude" },
      {
        label: "atlas…",
        drawer: "activation_atlas",
        keywords: "heatmap token layer probe",
      },
      {
        label: "experiments…",
        drawer: "experiment_lab",
        keywords: "alpha grid fan sweep",
      },
      {
        label: "recipe…",
        drawer: "recipe_builder",
        keywords: "expression steering terms",
      },
    ],
  },
  {
    key: "session",
    label: "Session",
    icon: "M5 21v-6M5 11V3M12 21v-9M12 8V3M19 21v-4M19 13V3M2 15h6M9 8h6M16 13h6",
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
  | { kind: "tab"; tab: InspectorTab }
  | { kind: "styleguide" };

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
  cmds.push({
    label: "style guide",
    group: "pages",
    action: { kind: "styleguide" },
    keywords: "design tokens specimens /styleguide",
  });
  return cmds;
}
