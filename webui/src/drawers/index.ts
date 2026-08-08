// THE drawer registry — one row per ``DrawerName``.
//
// Typed as ``Record<DrawerName, DrawerEntry>``, so adding a name to the
// union without wiring it here is a compile error, and every fact about a
// drawer lives in one row: the component, the params the host folds in,
// whether it takes the narrow panel, and how it is reached.
//
// "How it is reached" is the load-bearing part.  A non-null ``launcher``
// means the drawer appears in the ⌘K palette, and ``RAIL_CATEGORIES``
// below is *derived* from those rows rather than maintained beside them —
// the previous hand-written list was the one place a drawer could be
// declared, exported, and rendered while still being unreachable.  A
// drawer that is genuinely opened from a surface rather than the palette
// sets ``launcher: null`` and must then name that surface in ``via``; the
// union type makes "no palette entry" a claim someone wrote down instead
// of an omission nobody noticed.
//
// ``RackDrawer`` is the shared rack browser — one component reskinned by
// geometry family.  ``family: "subspace"`` (flat pca / baked fits, white
// accent) and ``family: "manifold"`` (curved spectral / authored fits,
// purple accent) are mirror images, differing only by accent, label, and
// catalog filter, so they are two registry rows over one component with
// different ``params``.  The "+ build manifold" launcher inside it routes
// to ``manifold_builder`` for both families (a flat fit is just a pca
// manifold).

import type { Component } from "svelte";

import type { DrawerName } from "../lib/types";

import AdvancedSamplingDrawer from "./AdvancedSamplingDrawer.svelte";
import CastDrawer from "./CastDrawer.svelte";
import CompareDrawer from "./CompareDrawer.svelte";
import CorrelationDrawer from "./CorrelationDrawer.svelte";
import HealthDrawer from "./HealthDrawer.svelte";
import HelpDrawer from "./HelpDrawer.svelte";
import LoadConversationDrawer from "./LoadConversationDrawer.svelte";
import ManifoldBuilderDrawer from "./ManifoldBuilderDrawer.svelte";
import ManifoldMergeDrawer from "./ManifoldMergeDrawer.svelte";
import ManifoldPacksDrawer from "./ManifoldPacksDrawer.svelte";
import NodeCompareDrawer from "./NodeCompareDrawer.svelte";
import ProbeInspectorDrawer from "./ProbeInspectorDrawer.svelte";
import RackDrawer from "./RackDrawer.svelte";
import SaveConversationDrawer from "./SaveConversationDrawer.svelte";
import SessionAdminDrawer from "./SessionAdminDrawer.svelte";
import SystemPromptDrawer from "./SystemPromptDrawer.svelte";
import TemplateLabDrawer from "./TemplateLabDrawer.svelte";
import TokenDrilldownDrawer from "./TokenDrilldownDrawer.svelte";
import TranscriptDrawer from "./TranscriptDrawer.svelte";

/** The drawer host renders one prop; ``params`` is typed loosely because
 *  each drawer owns its own shape. */
export type DrawerComponent = Component<{ params: unknown }>;

/** Palette group keys.  Typing ``launcher.group`` as this union means a
 *  typo can't strand a tool in a group that never renders. */
export type DrawerGroupKey = "manifolds" | "analysis" | "session";

export interface DrawerLauncher {
  group: DrawerGroupKey;
  /** Palette entry label, trailing ellipsis included. */
  label: string;
  /** Extra match text for palette filtering (synonyms, old names). */
  keywords?: string;
}

interface DrawerBase {
  component: DrawerComponent;
  /** Folded into ``drawerState.params`` before render — how one component
   *  serves two drawer names. */
  params?: Record<string, unknown>;
  /** Content-driven sizing: forms and pickers get the narrow panel, while
   *  analysis views keep the wide one. */
  narrow?: boolean;
}

/** A drawer is either palette-launched or reached from a named surface —
 *  never neither. */
export type DrawerEntry =
  | (DrawerBase & { launcher: DrawerLauncher; via?: never })
  | (DrawerBase & { launcher: null; via: string });

/** Palette groups, in display order.  Every ``DrawerGroupKey`` appears
 *  exactly once. */
export const DRAWER_GROUPS: readonly { key: DrawerGroupKey; label: string }[] = [
  // The single steering-authoring surface.  Concepts are manifolds now —
  // a flat (2-node / personas) fit is just a pca manifold — so there's no
  // separate "subspaces" group; flat authoring folds into the manifold
  // builder's pca path.  The catalog is the shared RackDrawer
  // (family-split), reached from the rack "+" buttons.
  { key: "manifolds", label: "Steering" },
  { key: "analysis", label: "Analysis" },
  { key: "session", label: "Session" },
];

/** Declaration order is the palette order within a group. */
export const DRAWERS: Record<DrawerName, DrawerEntry> = {
  // ----------------------------------------------- palette-launched ----
  manifold_builder: {
    component: ManifoldBuilderDrawer,
    narrow: true,
    launcher: {
      group: "manifolds",
      label: "build…",
      keywords: "extract author create concept vector fit",
    },
  },
  manifold_merge: {
    component: ManifoldMergeDrawer,
    launcher: {
      group: "manifolds",
      label: "merge…",
      keywords: "union corpora",
    },
  },
  manifold_pack: {
    component: ManifoldPacksDrawer,
    launcher: {
      group: "manifolds",
      label: "packs…",
      keywords: "install search huggingface hub catalog",
    },
  },
  template_lab: {
    component: TemplateLabDrawer,
    launcher: {
      group: "manifolds",
      label: "templates…",
      keywords: "score completion slot restricted choice",
    },
  },
  cast: {
    component: CastDrawer,
    launcher: {
      group: "manifolds",
      label: "cast…",
      keywords: "roster member speaker label recipe seat role",
    },
  },
  correlation: {
    component: CorrelationDrawer,
    launcher: {
      group: "analysis",
      label: "correlation…",
      keywords: "cosine similarity vectors",
    },
  },
  compare: {
    component: CompareDrawer,
    launcher: {
      group: "analysis",
      label: "compare…",
      keywords: "cross-layer cosine",
    },
  },
  health: {
    component: HealthDrawer,
    launcher: { group: "session", label: "health…", keywords: "device dtype" },
  },
  session_admin: {
    component: SessionAdminDrawer,
    launcher: { group: "session", label: "auth…", keywords: "api key bearer" },
  },
  help: {
    component: HelpDrawer,
    launcher: {
      group: "session",
      label: "help…",
      keywords: "keyboard grammar cheatsheet",
    },
  },

  // -------------------------------------------- opened in context -----
  subspace: {
    component: RackDrawer,
    params: { family: "subspace" },
    narrow: true,
    launcher: null,
    via: 'the steering and probe racks\' "+ add" buttons',
  },
  manifolds: {
    component: RackDrawer,
    params: { family: "manifold" },
    narrow: true,
    launcher: null,
    via: 'the steering and probe racks\' "+ add" buttons',
  },
  save_conversation: {
    component: SaveConversationDrawer,
    narrow: true,
    launcher: null,
    via: "the threads column's save action",
  },
  load_conversation: {
    component: LoadConversationDrawer,
    narrow: true,
    launcher: null,
    via: "the threads column's load action",
  },
  system_prompt: {
    component: SystemPromptDrawer,
    narrow: true,
    launcher: null,
    via: "the sampling strip's system-prompt button",
  },
  advanced_sampling: {
    component: AdvancedSamplingDrawer,
    launcher: null,
    via: "the sampling strip's advanced button",
  },
  token_drilldown: {
    component: TokenDrilldownDrawer,
    launcher: null,
    via: "a transcript or raw-buffer token click",
  },
  probe_inspector: {
    component: ProbeInspectorDrawer,
    launcher: null,
    via: "a probe card's ⓘ button",
  },
  node_compare: {
    component: NodeCompareDrawer,
    launcher: null,
    via: "the loom sidebar's compare actions",
  },
  transcript: {
    component: TranscriptDrawer,
    launcher: null,
    via: "the chat header's transcript button",
  },
};

/** The props the host passes to ``DRAWERS[name].component``: whatever
 *  ``openDrawer`` carried, with the registry's fixed params folded on
 *  top. */
export function drawerParams(name: DrawerName, params: unknown): unknown {
  const fixed = DRAWERS[name].params;
  if (!fixed) return params;
  return { ...(params as Record<string, unknown>), ...fixed };
}

export interface RailTool {
  label: string;
  drawer: DrawerName;
  keywords?: string;
}

export interface RailCategory {
  key: DrawerGroupKey;
  label: string;
  tools: RailTool[];
}

/** The ⌘K launcher list, DERIVED from the registry — a tool cannot go
 *  missing here without its row also losing its ``launcher``. */
export const RAIL_CATEGORIES: RailCategory[] = DRAWER_GROUPS.map((group) => {
  const tools: RailTool[] = [];
  for (const [drawer, entry] of Object.entries(DRAWERS) as [
    DrawerName,
    DrawerEntry,
  ][]) {
    if (entry.launcher === null || entry.launcher.group !== group.key) continue;
    tools.push({
      label: entry.launcher.label,
      drawer,
      keywords: entry.launcher.keywords,
    });
  }
  return { key: group.key, label: group.label, tools };
});
