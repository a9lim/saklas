// Cross-component state for the dashboard.
//
// Svelte 5 runes-based.  Each slice is a $state-backed object exported as
// a named const; components import the slice and read/write its fields
// directly — Svelte's compiler tracks dependencies automatically.
//
// Cross-cutting actions (open the WS, send a submission, queue a pending
// rack edit during in-flight gen) live in this file as functions so panels
// don't need to coordinate amongst themselves; they call ``sendSubmit(...)``
// or ``setSubspaceAlong(value)`` and the slice updates propagate.
//
// One singleton WS owned at the module level — the chat panel is no
// longer responsible for lifecycle.  Subscribers register via
// ``onWsMessage(cb)`` and receive every ``WSServerMessage`` the
// connection emits.

import { SvelteMap, SvelteSet } from "svelte/reactivity";
import {
  apiSessions,
  apiProfiles,
  apiProbes,
  apiManifolds,
  apiInstruments,
  apiTree,
  ApiError,
  connectWs,
} from "./api";
import type {
  CorrelationData,
  LoomNodeJSON,
  LoomTreeJSON,
  ManifoldInfo,
  SessionInfo,
  VectorInfo,
  WSClientMessage,
  WSServerMessage,
} from "./api";
import type {
  AnyReadingJSON,
  AtomMode,
  AtomSteerEntry,
  CastMemberJSON,
  ChatTurn,
  GenStatus,
  GeometryProbeInfo,
  InstrumentFamily,
  InstrumentFamilyBlock,
  InstrumentSourceJSON,
  LensReadoutBlockJSON,
  MeasurementsEnvelopeJSON,
  SaeFeatureJSON,
  SaeProbeInfo,
  ManifoldSteerEntry,
  PendingAction,
  ProbeInfo,
  ProbeReadingJSON,
  ProbeRackEntry,
  ProbeSortMode,
  SteerEntry,
  SubspaceSteerEntry,
  TokenScore,
  Trigger,
  Variant,
  ChatRole,
  WSSampling,
} from "./types";
import { isScalarReading } from "./types";
import { serializeExpression } from "./expression";
import { resolveReadoutTopK } from "./readouts";
import {
  SURPRISE_TARGET,
  HIGHLIGHT_SAT,
  nodeCoordExtent,
  parseProbeTarget,
} from "./tokens";
import { pushToast } from "./stores/toasts.svelte";
import { createPreparationSlice } from "./stores/preparations.svelte";

export * from "./stores/drawers.svelte";
export * from "./stores/inputHistory.svelte";
export { dismissToast, pushToast, toasts } from "./stores/toasts.svelte";
export * from "./stores/persistence.svelte";
export * from "./stores/loomUi.svelte";
export * from "./stores/pending.svelte";
export * from "./stores/sampling.svelte";
export * from "./stores/chat.svelte";
export * from "./stores/instruments.svelte";
export * from "./stores/session.svelte";
export * from "./stores/steering.svelte";
export * from "./stores/probes.svelte";
import {
  MAX_SPARKLINE,
  highlightState,
  hydrateProbeRackFromActiveNode,
  probeRack,
  refreshProbeList,
  resetProbeStreams,
  setProbeAggregates,
  snapshotProbeBaseline,
  updateProbesFromReadings,
} from "./stores/probes.svelte";
import {
  currentSteeringExpression,
  refreshCorrelation,
  refreshManifoldList,
  refreshVectorList,
  steerRack,
} from "./stores/steering.svelte";
import {
  refreshSession,
  saeLoaded,
  sessionState,
} from "./stores/session.svelte";
import {
  backfillSaeMeta,
  lensReadoutSnapshot,
  lensState,
  mergedReadings,
  rehydrateInstrumentsFromSession,
  saeReadoutForDisplay,
  saeState,
  tokenHoverState,
} from "./stores/instruments.svelte";
import {
  chatLog,
  effectiveRawMode,
  genStatus,
  geometricMeanPpl,
  liveTokenStream,
  loadGenUiMode,
} from "./stores/chat.svelte";
import {
  buildSamplingPayload,
  hydrateSamplingFromInfo,
  samplingState,
} from "./stores/sampling.svelte";
import {
  enqueueOrApply,
  enqueuePending,
  isPendingBusy,
  nextPendingId,
  drainNextPendingAction,
  pendingActions,
} from "./stores/pending.svelte";
import {
  invalidateEdgeLabels,
  pinNodeForComparison,
} from "./stores/loomUi.svelte";
import {
  attachPersistence,
  loadPersistedPreferences,
  safeLocalStorageGet,
  safeLocalStorageSet,
} from "./stores/persistence.svelte";

// ============================================================ loom tree ===
//
// Mirrors the server's LoomTree (phase 2 spec).  The slice is the
// authoritative shape for the loom sidebar; ``chatLog.turns`` is sync'd
// from the active path via ``syncChatLogFromTree`` whenever ``loomTree``
// changes (rev-driven).
//
// The current server tree is authoritative.  ``chatLog.turns`` is a
// projection of its active path; token deltas enrich that projection.

export interface LoomTreeState {
  /** True after the authoritative tree snapshot has loaded successfully.
   * A pristine server tree legitimately has revision 0, so revision cannot
   * double as an initialization sentinel. */
  loaded: boolean;
  tree_format: number | null;
  saklas_version: string | null;
  session_id: string | null;
  name: string | null;
  root_id: string | null;
  active_node_id: string | null;
  /** Per-node cache.  SvelteMap so ``set``/``delete`` trigger reactivity
   *  in the sidebar without manual re-renders. */
  nodes: Map<string, LoomNodeJSON>;
  /** parent_id → ordered child ids.  Same SvelteMap pattern. */
  children_of: Map<string, string[]>;
  /** Monotonic server revision cursor.  A freshly loaded tree is revision 0. */
  rev: number;
  /** Pending in-flight gen target id (when known).  Reflects the
   *  ``started`` / ``tree_mutated`` event node identity; null
   *  between gens. */
  pendingNodeId: string | null;
  /** Cached active path as an ordered list of node ids.  Recomputed on
   *  every ``rev`` bump so sidebar / chat sync work in O(depth). */
  activePath: string[];
  /** Last seen server-side model id; used to invalidate cache across
   *  model swaps. */
  modelId: string | null;
  /** Last fetch error message; surfaced in the sidebar. */
  error: string | null;
}

export const loomTree: LoomTreeState = $state({
  loaded: false,
  tree_format: null,
  saklas_version: null,
  session_id: null,
  name: null,
  root_id: null,
  active_node_id: null,
  nodes: new SvelteMap(),
  children_of: new SvelteMap(),
  rev: 0,
  pendingNodeId: null,
  activePath: [],
  modelId: null,
  error: null,
});

/** Cast roster (phase 3): label → member (standing recipe + notes).
 *  Hydrated from the full-tree GET (``cast`` key) and reconciled from
 *  ``op="cast"`` ``tree_mutated`` frames (roster inlined — no refetch). */
export const castState: { roster: Record<string, CastMemberJSON> } = $state({
  roster: {},
});

/** Walk from root to ``active_node_id`` and produce the ordered list of
 *  node ids on the active path.  O(depth + active-children-per-step).
 *  Returns [] when the tree isn't loaded. */
function recomputeActivePath(): void {
  const active = loomTree.active_node_id;
  if (!active) {
    loomTree.activePath = [];
    return;
  }
  // Walk parents to the root, reverse for root-first order.
  const reversed: string[] = [];
  let cursor: string | null = active;
  const seen = new Set<string>();
  while (cursor && !seen.has(cursor)) {
    seen.add(cursor);
    reversed.push(cursor);
    const node = loomTree.nodes.get(cursor);
    cursor = node?.parent_id ?? null;
  }
  loomTree.activePath = reversed.reverse();
}

/** Snake_case server token row → camelCase client ``TokenScore``.  Used
 *  by both the rehydration path (``nodeToTurn``) and the live ``token``
 *  WS handler; keeping a single converter means the rehydrated tokens
 *  are bit-identical to the live-streamed shape so the highlight / click
 *  / fork affordances behave the same way. */
function tokenRowToScore(row: NonNullable<LoomNodeJSON["tokens"]>[number]): TokenScore {
  const m = row.measurements;
  const out: TokenScore = {
    text: row.text,
    thinking: false,
  };
  if (row.token_id !== undefined) out.tokenId = row.token_id;
  if (row.logprob !== undefined) out.logprob = row.logprob;
  if (row.top_alts) out.topAlts = row.top_alts;
  if (row.raw_index !== undefined) out.rawIndex = row.raw_index;
  const scores = m?.scores ?? row.probes;
  if (scores) out.probes = scores;
  const perLayer = m?.per_layer_scores ?? row.per_layer_scores;
  if (perLayer) out.perLayerScores = perLayer;
  if (m) out.measurements = m;
  const readings = mergedReadings(m);
  if (readings) {
    const byProbe: Record<string, number[]> = {};
    for (const [name, reading] of Object.entries(readings)) {
      // Multi-axis coordinates are a geometry-only channel — a one-channel
      // reading has nothing to spread across axes.
      if (!isScalarReading(reading) && reading.coords.length > 1) {
        byProbe[name] = reading.coords;
      }
    }
    if (Object.keys(byProbe).length > 0) out.coordsByProbe = byProbe;
  }
  return out;
}

/** Project a LoomNodeJSON to a ChatTurn for Chat.svelte consumption.
 *  Hydrates ``tokens`` / ``thinkingTokens`` from the server-serialized
 *  per-token rows when the server included them (tree GET / tree_mutated
 *  with ``include_tokens=True``).  Without this, a force-refresh would
 *  produce token-less turns and the inline highlight / token-drilldown
 *  click target would silently break for historical messages. */
function nodeToTurn(n: LoomNodeJSON): ChatTurn {
  const turn: ChatTurn = {
    role: n.role,
    text: n.text,
    roleLabel: n.role_label,
    nodeId: n.id,
    generated: n.recipe !== null,
    appliedSteering: n.applied_steering ?? null,
    aggregateReadings: n.aggregate_readings ?? undefined,
    finishReason: n.finish_reason ?? undefined,
  };
  if (n.tokens && n.tokens.length > 0) {
    turn.tokens = n.tokens.map((r) => {
      const s = tokenRowToScore(r);
      s.thinking = false;
      return s;
    });
  }
  const persistedPpl = [...(n.thinking_tokens ?? []), ...(n.tokens ?? [])]
    .map((row) => row.perplexity)
    .filter((value): value is number => (
      typeof value === "number" && Number.isFinite(value) && value > 0
    ));
  if (persistedPpl.length > 0) {
    turn.perplexity = Math.exp(
      persistedPpl.reduce((sum, value) => sum + Math.log(value), 0)
      / persistedPpl.length,
    );
  }
  if (n.thinking_tokens && n.thinking_tokens.length > 0) {
    turn.thinkingTokens = n.thinking_tokens.map((r) => {
      const s = tokenRowToScore(r);
      s.thinking = true;
      return s;
    });
    turn.thinking = true;
  } else if (n.thinking_text) {
    // Committed thinking block (no token rows — the author typed it):
    // one synthesized row renders it through the same collapsible the
    // streamed thinking channel uses.
    turn.thinkingTokens = [{ text: n.thinking_text, thinking: true }];
    turn.thinking = true;
  }
  return turn;
}

function attachChild(parentId: string | null, childId: string): void {
  if (parentId === null) return;
  const siblings = loomTree.children_of.get(parentId) ?? [];
  if (!siblings.includes(childId)) {
    loomTree.children_of.set(parentId, [...siblings, childId]);
  }
}

function upsertLoomNode(raw: LoomNodeJSON & { children?: string[] }): LoomNodeJSON {
  const { children, ...node } = raw;
  loomTree.nodes.set(node.id, node);
  // Portable Loom snapshots require ``children_of`` to contain one key for
  // *every* node, including leaves.  Live deltas only need to extend the
  // parent list, so newly generated leaves previously had no own empty entry:
  // exporting that otherwise-correct live tree produced a file the server
  // refused to import.  Preserve an existing list on updates; seed a new node
  // from the serializer's optional children field (normally ``[]``).
  if (!loomTree.children_of.has(node.id)) {
    loomTree.children_of.set(node.id, [...(children ?? [])]);
  }
  if (node.parent_id !== null) {
    attachChild(node.parent_id, node.id);
  } else {
    loomTree.root_id = node.id;
  }
  return node;
}

/** Sync ``chatLog.turns`` (and ``chatLog.pendingIndex``) from the tree's
 *  active path.  Called after every tree mutation when ``rev > 0``.  Skip
 *  the synthetic system root (parent_id === null + role === "system" +
 *  empty text) so the chat view doesn't lead with an invisible turn.
 *
 *  Preserves any in-flight token stream attached to the pending node by
 *  re-using the existing ChatTurn object when possible — token deltas
 *  flowing in via WS keep accumulating on it.  This is the bridge
 *  between "tree is authoritative" and "live tokens land on an existing
 *  turn object." */
function syncChatLogFromTree(): void {
  if (!loomTree.loaded) return;
  const path = loomTree.activePath;
  if (path.length === 0) {
    chatLog.turns = [];
    chatLog.pendingIndex = null;
    return;
  }
  const out: ChatTurn[] = [];
  let pendingIdx: number | null = null;
  for (const nid of path) {
    const node = loomTree.nodes.get(nid);
    if (!node) continue;
    // Skip the synthetic system root — empty text, no parent, role
    // "system".  It's an engine-side anchor, not a user-facing turn.
    if (node.parent_id === null && node.role === "system" && !node.text) continue;
    // Try to keep the existing turn object if it already represents this
    // node (token-stream preservation for the live target).
    const prev = chatLog.turns.find((t) => t.nodeId === nid);
    let turn: ChatTurn;
    if (
      prev &&
      prev.role === node.role &&
      prev.nodeId === nid
    ) {
      // Mutate-in-place so the streaming token arrays survive.
      prev.nodeId = nid;
      // A same-role generated continuation deliberately clears and reuses
      // its existing node before replaying the old text as a forced prefix.
      // The tree snapshot is authoritative even when that temporary value is
      // empty; preserving the prior UI text here would duplicate the prefix
      // as streamed tokens arrive.
      prev.text = node.text;
      prev.generated = node.recipe !== null;
      prev.appliedSteering = node.applied_steering ?? prev.appliedSteering ?? null;
      prev.aggregateReadings = node.aggregate_readings ?? prev.aggregateReadings;
      prev.finishReason = node.finish_reason ?? prev.finishReason;
      // Server-shipped node tokens are authoritative. Preserve the live
      // arrays only until the finalized node snapshot carries them.
      if ((prev.tokens?.length ?? 0) === 0) {
        const fromNode = nodeToTurn(node);
        if (fromNode.tokens || fromNode.thinkingTokens) {
          prev.tokens = fromNode.tokens;
          prev.thinkingTokens = fromNode.thinkingTokens;
        }
      }
      turn = prev;
    } else {
      turn = nodeToTurn(node);
    }
    if (loomTree.pendingNodeId === nid) pendingIdx = out.length;
    out.push(turn);
  }
  chatLog.turns = out;
  chatLog.pendingIndex = pendingIdx;
}

/** Replace the in-memory tree with a current server snapshot. */
function applyTreeSnapshot(snap: LoomTreeJSON): void {
  loomTree.loaded = true;
  loomTree.tree_format = snap.tree_format;
  loomTree.saklas_version = snap.saklas_version;
  loomTree.session_id = snap.session_id;
  loomTree.name = snap.name;
  loomTree.root_id = snap.root_id;
  loomTree.active_node_id = snap.active_node_id;
  loomTree.rev = snap.rev;
  loomTree.modelId = snap.model_id;
  loomTree.error = null;
  loomTree.nodes.clear();
  for (const n of snap.nodes) loomTree.nodes.set(n.id, n);
  loomTree.children_of.clear();
  for (const [pid, ids] of Object.entries(snap.children_of)) {
    loomTree.children_of.set(pid, [...ids]);
  }
  castState.roster = snap.cast;
  recomputeActivePath();
  syncChatLogFromTree();
  hydrateProbeRackFromActiveNode();
}

/** Materialize the reactive Loom slice in the server's portable JSON shape.
 *  Used by explicit v4 whole-conversation export. */
export function currentLoomTreeSnapshot(): LoomTreeJSON | null {
  if (
    !loomTree.loaded || !loomTree.root_id || !loomTree.active_node_id ||
    loomTree.tree_format === null || loomTree.saklas_version === null
  ) return null;
  const nodes: LoomNodeJSON[] = [];
  for (const [, node] of loomTree.nodes) nodes.push(node);
  const children_of: Record<string, string[]> = {};
  // Export the server's exact total mapping contract even when this browser
  // began before the leaf-initialization fix or received an older partial
  // client state.  Node order is already the portable snapshot order.
  for (const node of nodes) {
    children_of[node.id] = [...(loomTree.children_of.get(node.id) ?? [])];
  }
  return {
    tree_format: loomTree.tree_format,
    saklas_version: loomTree.saklas_version,
    root_id: loomTree.root_id,
    active_node_id: loomTree.active_node_id,
    rev: loomTree.rev,
    nodes,
    children_of,
    model_id: loomTree.modelId ?? sessionState.info?.model_id ?? null,
    session_id: loomTree.session_id,
    name: loomTree.name,
    cast: { ...castState.roster },
  };
}

/** Apply a ``tree_mutated`` delta in place.  Returns ``false`` if the
 *  client missed a rev — caller full-refetches on false.
 *
 *  Phase-2 server semantics: ``updated`` carries full LoomNodeJSON
 *  objects (potentially with an extra ``children`` field used only to seed a
 *  new node's own children entry; parent links still come from deltas).
 *  ``added`` nodes may also be implicit children-list extensions of
 *  existing parents.  ``upsertLoomNode`` additionally establishes the
 *  required empty ``children_of`` entry for every new leaf. */
function applyTreeDelta(ev: {
  added?: LoomNodeJSON[];
  removed?: string[];
  updated?: LoomNodeJSON[];
  active_node_id?: string | null;
  rev: number;
}): boolean {
  // First event after bootstrap is the rev=1 mutation; accept rev > 0
  // when our local rev is 0 (cold start) without claiming a gap.
  if (loomTree.loaded && ev.rev > loomTree.rev + 1) return false;
  // ``added``: inject node + extend its parent's children list.  Node
  // payloads from the server may include a ``children`` field
  // (the server serializer adds it); use it only to seed children_of, while
  // stripping it from the cached node so bootstrap/delta shapes stay equal.
  for (const raw of ev.added ?? []) {
    upsertLoomNode(raw as LoomNodeJSON & { children?: string[] });
  }
  // ``removed``: subtree-drop — caller (server) emits the full list of
  // dropped descendants so we don't need to walk locally.  Defensive
  // dedupe against missing entries.
  for (const id of ev.removed ?? []) {
    const node = loomTree.nodes.get(id);
    loomTree.nodes.delete(id);
    loomTree.children_of.delete(id);
    if (node?.parent_id) {
      const sibs = loomTree.children_of.get(node.parent_id);
      if (sibs) {
        loomTree.children_of.set(node.parent_id, sibs.filter((s) => s !== id));
      }
    }
  }
  // ``updated``: full node replacement.  Same children handling as added.
  for (const raw of ev.updated ?? []) {
    upsertLoomNode(raw as LoomNodeJSON & { children?: string[] });
  }
  // ``active_node_id`` arrives null whenever the server-side
  // ``LoomMutated`` event leaves it unset (the default for mutations
  // that don't move the active pointer — edit, star, note, etc.).  The
  // raw JSON serializer passes it through as null rather than omitting
  // the key, so we treat both ``null`` and ``undefined`` as "unchanged"
  // here.  Don't tighten this to "undefined only": the server contract
  // and the live wire shape disagree, and ``null`` is the live shape.
  if (ev.active_node_id !== undefined && ev.active_node_id !== null) {
    loomTree.active_node_id = ev.active_node_id;
  }
  // A ``reset`` is the only mutation that drops the root: its ``removed`` list
  // now includes the old root and ``added`` carries a fresh parentless one.
  // ``applyTreeDelta`` never otherwise touches ``root_id``, so re-seed it here
  // when the old root is gone — else the sidebar (which walks from ``root_id``)
  // points at a deleted node and renders empty after a cross-client reset.
  if (loomTree.root_id !== null && !loomTree.nodes.has(loomTree.root_id)) {
    const newRoot = (ev.added ?? []).find((n) => n.parent_id == null)
      ?? [...loomTree.nodes.values()].find((n) => n.parent_id == null);
    if (newRoot) loomTree.root_id = newRoot.id;
  }
  loomTree.rev = ev.rev;
  // Phase 5: applied_steering strings can shift after edit/regen, so
  // bust the edge-label cache wholesale on any mutation.  Cheap — the
  // sidebar refetches lazily on first re-render.
  invalidateEdgeLabels();
  recomputeActivePath();
  syncChatLogFromTree();
  hydrateProbeRackFromActiveNode();
  return true;
}

/** Bootstrap fetch of the required tree surface. */
export async function refreshLoomTree(): Promise<void> {
  try {
    const snap = await apiTree.get();
    applyTreeSnapshot(snap);
  } catch (e) {
    loomTree.error = e instanceof Error ? e.message : String(e);
    pushToast(`tree: ${loomTree.error}`, { kind: "error" });
  }
}

/** Capture mutation failures on ``loomTree.error`` AND a toast.
 *
 *  ``loomTree.error`` is the persistent banner inside the empty-state
 *  branch of the sidebar; for trees with nodes that branch never
 *  renders, so the toast is the only surface the user sees.  Fires
 *  for every mutator path so 409s on edit-during-gen, network drops,
 *  ambiguous prefix rejections, and any other server error reach the
 *  user instead of vanishing silently.
 */
function _captureLoomError(op: string, e: unknown): void {
  const msg = e instanceof Error ? e.message : String(e);
  loomTree.error = msg;
  pushToast(`${op}: ${msg}`, { kind: "error" });
}

/** Right-click ops + keyboard shortcuts route through these helpers.
 *  Each one fires the REST mutation and lets the server-emitted
 *  ``tree_mutated`` event sync the local store — no optimistic update
 *  (keeps the local copy in lockstep with server rev). */
export async function loomNavigate(node_id: string): Promise<void> {
  try {
    await apiTree.navigate(node_id);
    await refreshLoomTree();
  } catch (e) {
    _captureLoomError("navigate", e);
  }
}

export async function loomEdit(node_id: string, text: string): Promise<void> {
  try {
    await apiTree.edit(node_id, text);
    await refreshLoomTree();
  } catch (e) {
    _captureLoomError("edit", e);
  }
}

export async function loomBranch(
  node_id: string,
  text: string,
  role?: "user" | "assistant" | null,
): Promise<string | null> {
  try {
    const r = await apiTree.branch(node_id, text, undefined, role);
    await refreshLoomTree();
    return r.node_id;
  } catch (e) {
    _captureLoomError("branch", e);
    return null;
  }
}

/** Seat-swap branch: a sibling with the same text and the seat flipped
 *  (the cast model's controlled experiment on the seat prior).  The
 *  swapped copy re-renders under the flipped header at the next
 *  generation; downstream nodes are NOT copied (same contract as edit). */
export async function loomSwapSeat(node_id: string): Promise<string | null> {
  const node = loomTree.nodes.get(node_id);
  if (!node || (node.role !== "user" && node.role !== "assistant")) return null;
  const flipped = node.role === "user" ? "assistant" : "user";
  return loomBranch(node_id, node.text, flipped);
}

export async function loomDelete(node_id: string): Promise<void> {
  try {
    await apiTree.delete(node_id);
    await refreshLoomTree();
  } catch (e) {
    _captureLoomError("delete", e);
  }
}

export async function loomStar(node_id: string, on: boolean): Promise<void> {
  try {
    await apiTree.star(node_id, on);
    await refreshLoomTree();
  } catch (e) {
    _captureLoomError("star", e);
  }
}

export async function loomNote(node_id: string, text: string): Promise<void> {
  try {
    await apiTree.note(node_id, text);
    await refreshLoomTree();
  } catch (e) {
    _captureLoomError("note", e);
  }
}

/** Regenerate one conversation node as a generated sibling in the same seat.
 *  N=1 by default.  Recipe is implicit (current rack) unless
 *  ``opts.recipe_override`` is set, in which case the engine applies
 *  the recipe-override modifier on top of the parent's recipe. */
export async function loomRegenerateNode(
  nodeId: string,
  n: number = 1,
  opts: { recipe_override?: string | null } = {},
): Promise<void> {
  if (!loomTree.loaded) return;
  const node = loomTree.nodes.get(nodeId);
  if (!node || node.role === "system") return;
  const parentId = node.parent_id;
  if (!parentId) return;
  try {
    await sendGenerate({
      parent_node_id: parentId,
      n,
      recipe_override: opts.recipe_override ?? undefined,
      generate_seat: node.role,
    });
  } catch (e) {
    _captureLoomError("regenerate", e);
  }
}

/** Active-node compatibility adapter for keyboard, loom-menu, and auto-regen
 * callers. Chat message controls call ``loomRegenerateNode`` directly. */
export async function loomRegenerateActive(
  n: number = 1,
  opts: { recipe_override?: string | null } = {},
): Promise<void> {
  const activeId = loomTree.active_node_id;
  if (!activeId) return;
  return loomRegenerateNode(activeId, n, opts);
}

/** Generate opposite-seat continuations under a committed turn.
 *  This is the fan-out entry point for either structural seat; the selected
 *  node supplies context only, while its role determines the reply seat. */
export async function loomContinueFromCommitted(
  nodeId: string,
  opts: { n?: number; recipe_override?: string | null } = {},
): Promise<void> {
  if (!loomTree.loaded) return;
  const node = loomTree.nodes.get(nodeId);
  if (!node || node.role === "system" || node.recipe !== null) return;
  try {
    await sendGenerate({
      parent_node_id: node.id,
      n: opts.n ?? 1,
      recipe_override: opts.recipe_override ?? undefined,
      generate_seat: node.role === "user" ? "assistant" : "user",
    });
  } catch (e) {
    _captureLoomError("regenerate", e);
  }
}

// ============================================================ WS ========

type WsListener = (msg: WSServerMessage) => void;

interface WsConnection {
  socket: WebSocket | null;
  listeners: Set<WsListener>;
  /** Promise resolved on first ``open`` — used by ``sendGenerate`` to
   * wait through reconnects without burying the API key. */
  ready: Promise<void> | null;
}

const wsConn: WsConnection = {
  socket: null,
  listeners: new SvelteSet(),
  ready: null,
};

export function onWsMessage(cb: WsListener): () => void {
  wsConn.listeners.add(cb);
  return () => wsConn.listeners.delete(cb);
}

export function ensureWebSocket(): Promise<WebSocket> {
  // Reuse an open or connecting socket; reconnect cleanly when the
  // last one closed.
  if (
    wsConn.socket &&
    (wsConn.socket.readyState === WebSocket.OPEN ||
      wsConn.socket.readyState === WebSocket.CONNECTING)
  ) {
    if (wsConn.ready) return wsConn.ready.then(() => wsConn.socket!);
    return Promise.resolve(wsConn.socket);
  }
  const socket = connectWs();
  wsConn.socket = socket;
  // A socket can reconnect to a freshly restarted server whose tree has a
  // lower revision and entirely different node ids.  Buffer wire events until
  // we have replaced the local cache with the new authoritative snapshot;
  // otherwise the first post-restart generation splices new nodes into the
  // stale pre-restart sidebar.
  let rehydrating = true;
  const bufferedMessages: WSServerMessage[] = [];
  const dispatch = (msg: WSServerMessage): void => {
    handleWsMessage(msg);
    for (const cb of wsConn.listeners) {
      try {
        cb(msg);
      } catch {
        /* ignore subscriber failures */
      }
    }
  };
  wsConn.ready = new Promise<void>((resolve, reject) => {
    socket.addEventListener("open", () => {
      void (async () => {
        try {
          const snap = await apiTree.get();
          applyTreeSnapshot(snap);
          rehydrating = false;
          // The snapshot already includes deltas at or below its revision.
          // Replay only genuinely newer tree frames; non-tree frames retain
          // their original arrival order.
          for (const msg of bufferedMessages) {
            if (msg.type === "tree_mutated" && msg.rev <= snap.rev) continue;
            dispatch(msg);
          }
          bufferedMessages.length = 0;
          // Other server-owned surfaces may also have changed across a
          // restart.  Refresh them before callers are allowed to submit the
          // next generation against this connection.
          await Promise.allSettled([
            refreshSession(),
            refreshVectorList(),
            refreshProbeList(),
            refreshCorrelation(),
            refreshManifoldList(),
          ]);
          resolve();
        } catch (e) {
          rehydrating = false;
          loomTree.error = e instanceof Error ? e.message : String(e);
          pushToast(`reconnect: ${loomTree.error}`, { kind: "error" });
          // Never leave an apparently reusable OPEN socket behind after its
          // authoritative snapshot failed.  A later send must establish a
          // fresh connection and retry the complete rehydration barrier.
          socket.close();
          reject(e);
        }
      })();
    }, { once: true });
    socket.addEventListener("error", (e) => reject(e), { once: true });
  });
  socket.addEventListener("message", (ev: MessageEvent) => {
    let msg: WSServerMessage;
    try {
      msg = JSON.parse(ev.data) as WSServerMessage;
    } catch {
      return;
    }
    if (rehydrating) bufferedMessages.push(msg);
    else dispatch(msg);
  });
  socket.addEventListener("close", () => {
    if (wsConn.socket === socket) {
      wsConn.socket = null;
      wsConn.ready = null;
    }
  });
  return wsConn.ready.then(() => socket);
}

export function disconnectWebSocket(): void {
  if (wsConn.socket) {
    try {
      wsConn.socket.close();
    } catch {
      /* ignore */
    }
    wsConn.socket = null;
    wsConn.ready = null;
  }
}

if (typeof window !== "undefined") {
  // Tear down the singleton on page unload so the server doesn't see
  // a leaked half-open connection.
  window.addEventListener("beforeunload", disconnectWebSocket);
}

/** Resolve the assistant turn that's currently receiving streamed tokens.
 *
 * Two modes:
 *   - **Normal**: ``chatLog.pendingIndex`` points at the assistant turn the
 *     ``started`` event allocated; tokens append directly to it.
 *   - **A/B shadow**: ``abState.processingAb`` is true and
 *     ``abState.pendingTurnIdx`` points at the *steered* turn; tokens
 *     append to that turn's ``abPair`` (an inner ``ChatTurn`` initialized
 *     on the shadow's ``started`` event).
 *
 * Returning ``null`` means we don't have a write target — drop the token
 * silently rather than throwing, since a stray event during teardown is
 * harmless. */
function _currentWriteTurn(): ChatTurn | null {
  if (abState.processingAb && abState.pendingTurnIdx !== null) {
    const steered = chatLog.turns[abState.pendingTurnIdx];
    return steered?.abPair ?? null;
  }
  if (chatLog.pendingIndex !== null) {
    return chatLog.turns[chatLog.pendingIndex] ?? null;
  }
  return null;
}

/** Bind an in-flight token stream to a concrete loom assistant node.
 *
 * The tree mutation that creates the node is ordered before its first token;
 * this binds the stream to that already-authoritative node. */
function adoptStreamingNode(nodeId: string | null | undefined): void {
  if (!nodeId || abState.processingAb || !loomTree.loaded) return;
  loomTree.pendingNodeId = nodeId;
  if (!loomTree.nodes.has(nodeId)) {
    loomTree.error = `Token arrived before authoritative node ${nodeId}`;
    pushToast(loomTree.error, { kind: "error" });
    return;
  }
  loomTree.active_node_id = nodeId;
  recomputeActivePath();
  syncChatLogFromTree();
  const idx = chatLog.pendingIndex;
  if (idx !== null) {
    const turn = chatLog.turns[idx];
    if (turn) {
      turn.nodeId = nodeId;
      turn.tokens = turn.tokens ?? [];
      turn.thinkingTokens = turn.thinkingTokens ?? [];
    }
  }
}

/** Default WS message handler — owns the gen-status lifecycle and the
 * live token stream.  External subscribers (panels) layer additional
 * behavior via ``onWsMessage``. */
function handleWsMessage(msg: WSServerMessage): void {
  switch (msg.type) {
    case "tree_mutated": {
      // The roster is derived from every observed turn label, so any tree
      // mutation may change it.  The server inlines the effective snapshot.
      if (msg.cast) {
        castState.roster = msg.cast;
      }
      // Whole-tree restore can change parentage and sibling order even when
      // node ids overlap the old tree.  Ordinary add/update/remove deltas do
      // not carry enough information to detach an intersecting node from its
      // former parent, so every connected client crosses an authoritative
      // full-snapshot barrier for this operation.
      if (msg.op === "restore") {
        void refreshLoomTree();
        return;
      }
      // Apply the delta; on rev gap, full re-fetch.
      const ok = applyTreeDelta(msg);
      if (!ok) void refreshLoomTree();
      return;
    }
    case "started": {
      genStatus.active = true;
      genStatus.tokensSoFar = 0;
      genStatus.startedAt = performance.now();
      genStatus.tokPerSec = 0;
      genStatus.ppl = { logSum: 0, count: 0, mean: null };
      genStatus.finishReason = null;
      liveTokenStream.responseTokens = [];
      liveTokenStream.thinkingTokens = [];
      // Manifold probes: drop the previous gen's trajectory + aggregate so
      // the inspector mini-map starts blank.  Sparkline carries across.
      if (!abState.processingAb) resetProbeStreams();
      // Loom: record the target node so tree-driven sync attaches the
      // streaming turn to the right active-path entry, and so the chat
      // panel's "streaming" highlight fires on the right turn.
      if (msg.node_id) {
        loomTree.pendingNodeId = msg.node_id;
        syncChatLogFromTree();
      }
      if (abState.processingAb && abState.pendingTurnIdx !== null) {
        // A/B shadow run: attach a fresh same-seat abPair to the generated
        // turn that just finished.  Don't append a new top-level turn —
        // the chat panel renders the abPair in its own column.
        const steered = chatLog.turns[abState.pendingTurnIdx];
        if (steered) {
          steered.abPair = {
            role: abState.pendingRole ?? steered.role,
            roleLabel: abState.pendingRoleLabel ?? steered.roleLabel,
            text: "",
            generated: true,
            tokens: [],
            thinkingTokens: [],
          };
        }
        // pendingIndex points at the steered turn so the streaming
        // pulse on Chat.svelte still highlights "this turn is live".
        chatLog.pendingIndex = abState.pendingTurnIdx;
      } else if (loomTree.loaded && msg.node_id) {
        // Loom path: the assistant node is already created server-side
        // (we got a ``tree_mutated`` add event before ``started``).  The
        // active-path sync seeds an empty turn for it; ensure the turn
        // has token arrays ready so the ``token`` handler can append.
        syncChatLogFromTree();
        const pidx = chatLog.pendingIndex;
        if (pidx !== null) {
          const turn = chatLog.turns[pidx];
          if (turn) {
            turn.tokens = turn.tokens ?? [];
            turn.thinkingTokens = turn.thinkingTokens ?? [];
          }
        }
      } else if (loomTree.loaded) {
        // Loom path with a lazily-created assistant node: wait for
        // the authoritative tree mutation before allocating
        // the assistant turn. Appending a local placeholder here creates
        // a duplicate local assistant and is the source of many branch /
        // highlight misroutes.
        chatLog.pendingIndex = null;
        syncChatLogFromTree();
      } else {
        loomTree.error = "Generation started before the required tree was loaded";
        pushToast(loomTree.error, { kind: "error" });
      }
      return;
    }
    case "token": {
      adoptStreamingNode(msg.node_id);
      genStatus.tokensSoFar += 1;
      if (
        typeof msg.perplexity === "number"
        && Number.isFinite(msg.perplexity)
        && msg.perplexity > 0
      ) {
        genStatus.ppl.logSum += Math.log(msg.perplexity);
        genStatus.ppl.count += 1;
        genStatus.ppl.mean = Math.exp(
          genStatus.ppl.logSum / genStatus.ppl.count,
        );
      }
      if (genStatus.startedAt) {
        const elapsed = (performance.now() - genStatus.startedAt) / 1000;
        if (elapsed > 0) genStatus.tokPerSec = genStatus.tokensSoFar / elapsed;
      }
      // The 5.x measurement envelope is the single read-side record.  Probe
      // readings merge the three families (the rack keys by name); ``scores``
      // is the flat cross-family axis-0 view (highlight tint);
      // ``per_layer_scores`` the heatmap; the native lens/sae ``readout``
      // blocks feed the workspace/discovery cards.
      const m = msg.measurements;
      const stepReadings = mergedReadings(m);
      const scores = m?.scores;
      const lensSnapshot = lensReadoutSnapshot(m?.instruments.lens?.readout);
      const liveLensReadout = lensSnapshot?.readout;
      const liveLensAggregate = lensSnapshot?.aggregate;
      const liveSaeReadout = m?.instruments.sae?.readout?.features;
      const tokenScore: TokenScore = {
        text: msg.text,
        thinking: msg.thinking,
        tokenId: msg.token_id,
        perLayerScores: m?.per_layer_scores,
        // ``scores`` is the magnitude-weighted aggregate probe row. Using
        // it instead of a deepest-layer slice makes live highlighting match the
        // post-generation projected pass. Absent when no probes are loaded.
        probes: scores,
        // Logit-pass: pipe chosen-token logprob + top-K alternatives onto
        // the per-token row.  Both ride the WS ``token`` event directly
        // from Phase 1's engine capture; absent when ``return_top_k == 0``
        // and no other on-token consumer requested capture.
        logprob: msg.logprob ?? null,
        topAlts: msg.top_alts ?? null,
        // Raw decode-step index — the join key the logit fork slices
        // ``raw_token_ids`` on.  Rides the WS ``token`` event directly.
        rawIndex: msg.raw_index ?? null,
        measurements: m,
      };
      // Seed the single-probe ``score`` for the selected highlight so the
      // inline tint paints immediately as the token streams in.  The
      // canonical projected scores overwrite this on ``done``.
      if (scores && highlightState.target) {
        const s = scores[highlightState.target];
        if (typeof s === "number") tokenScore.score = s;
      }
      // Per-PC token highlighting: stash the full per-axis domain coords off
      // the merged family readings so axis targets (``personas[3]``) can tint
      // live.  Only multi-axis probes need it — axis 0 already rides
      // ``scores`` — and the row keeps it through ``done`` (the per-token
      // settle pass is axis-0 only and never clobbers this field).
      if (stepReadings) {
        const byProbe: Record<string, number[]> = {};
        for (const [pname, r] of Object.entries(stepReadings)) {
          const coords = (r as ProbeReadingJSON).coords;
          if (Array.isArray(coords) && coords.length > 1) byProbe[pname] = coords;
        }
        if (Object.keys(byProbe).length > 0) tokenScore.coordsByProbe = byProbe;
      }
      const turn = _currentWriteTurn();
      if (turn) {
        if (msg.thinking) {
          turn.thinking = true;
          turn.thinkingTokens = [...(turn.thinkingTokens ?? []), tokenScore];
          // Live-stream buffer is steered-only — the shadow run doesn't
          // feed the main chat highlight pipeline.
          if (!abState.processingAb) {
            liveTokenStream.thinkingTokens.push(tokenScore);
          }
        } else {
          turn.text = (turn.text ?? "") + msg.text;
          turn.tokens = [...(turn.tokens ?? []), tokenScore];
          if (!abState.processingAb) {
            liveTokenStream.responseTokens.push(tokenScore);
          }
        }
      }
      // Probe rack — unified per-token readings.  Every probe shape rides the
      // three families' ``readings`` (a 2-node concept axis is the rank-1
      // case), merged by name; the field is omitted when no probe is attached,
      // so the helper no-ops on undefined.  Skip shadow runs so the rack stays
      // anchored to the steered branch.  ``scores`` / ``per_layer_scores``
      // above still feed highlight tinting + the token-drilldown heatmap.
      if (!abState.processingAb) {
        updateProbesFromReadings(stepReadings);
        // J-LENS tab — the live all-layer readout. Present only while
        // the session's live lens is enabled; shadow runs skipped like
        // the probe rack so the matrix tracks the steered branch.
        if (liveLensReadout) lensState.readout = liveLensReadout;
        if (liveLensAggregate) {
          lensState.aggregate = liveLensAggregate;
          // Rolling strength history for the workspace-card sparklines —
          // one compact [token, strength] frame per step, probe-sparkline
          // cap, carries across generations like probe sparklines.
          const frame: [string, number][] = liveLensAggregate.map(
            ([tok, strength]) => [tok, strength],
          );
          const hist = lensState.aggHistory.slice();
          hist.push(frame);
          if (hist.length > MAX_SPARKLINE) hist.shift();
          lensState.aggHistory = hist;
        }
        if (liveSaeReadout) {
          saeState.readout = liveSaeReadout;
          for (const feature of liveSaeReadout) {
            const prior = saeState.history.get(feature.id) ?? [];
            const next = [...prior, feature.activation].slice(-MAX_SPARKLINE);
            saeState.history.set(feature.id, next);
            // Server-cached metadata rides each row; the backfill fills
            // the gaps between generations.
            if (feature.max_act != null || feature.label != null) {
              saeState.meta.set(feature.id, {
                label: feature.label ?? null,
                max_act: feature.max_act ?? null,
              });
            }
          }
        }
      }
      return;
    }
    case "done": {
      adoptStreamingNode(msg.node_id);
      genStatus.active = false;
      genStatus.finishReason = msg.result?.finish_reason ?? "stop";
      // Probe rack — end-of-gen aggregate (the settled ``ProbeReading`` per
      // probe: coords / fraction / nearest / residual + per-layer traces),
      // read out of the ``scope: "aggregate"`` measurement envelope and
      // merged across the three families exactly as the ``token`` path
      // merges them.  Same omitted-when-absent rule.
      if (!abState.processingAb) {
        setProbeAggregates(mergedReadings(msg.result?.measurements));
      }
      const turn = _currentWriteTurn();
      if (turn) {
        turn.finishReason = msg.result?.finish_reason ?? "stop";
        turn.tokensSoFar = msg.result?.tokens ?? genStatus.tokensSoFar;
        // Logit-pass: per-turn mean chosen-token logprob (response span
        // only).  Null when capture wasn't live; the inline surprise
        // mode + loom edge-weighting null-guard on this directly.
        turn.meanLogprob = msg.result?.mean_logprob ?? null;
        const turnPpl = geometricMeanPpl(genStatus);
        if (turnPpl !== null) turn.perplexity = turnPpl;
      }
      // Reconcile the live token counter against the server's
      // authoritative ``token_count``.  The streaming ``token`` events
      // may diverge from the engine's final count when (a) the WS dedupes
      // / batches partial UTF-8 tokens, or (b) the server's actual
      // ``max_new_tokens`` differs from the client's local view (e.g.
      // before the first PATCH lands).  Trust the server on close.
      if (typeof msg.result?.tokens === "number" && Number.isFinite(msg.result.tokens)) {
        genStatus.tokensSoFar = msg.result.tokens;
      }

      const wasShadow = abState.processingAb;
      const steeredIdx = chatLog.pendingIndex;
      chatLog.pendingIndex = null;
      // Loom: drop the pending node-id pointer; the server-emitted
      // ``tree_mutated`` (finalize) event has already merged the
      // finalised text + finish_reason into the node.
      if (loomTree.pendingNodeId) {
        loomTree.pendingNodeId = null;
        // Re-sync so the "streaming" decoration on the just-finished
        // turn switches off.
        if (loomTree.loaded) syncChatLogFromTree();
      }

      if (wasShadow) {
        // Shadow gen done — clear the A/B routing flags.  Do NOT touch
        // the probe baseline or correlation refresh; the steered turn
        // already did that when it finished.
        abState.processingAb = false;
        abState.pendingTurnIdx = null;
        abState.pendingRole = null;
        abState.pendingRoleLabel = null;
        // Drain pending actions queued during the shadow gen — same
        // gen-active gate the steered branch uses.
        void drainNextPendingAction();
        return;
      }

      // Snapshot probe baselines + drain the next deferred mutation on
      // the steered done event only.  Single-pop semantics: each
      // queued item kicks its own work whose ``done`` will re-enter
      // here and drain the next, preserving FIFO.
      snapshotProbeBaseline();
      void refreshCorrelation();
      void drainNextPendingAction();
      // SAE discovery backfill — fetch Neuronpedia metadata (label +
      // maxActApprox) for features the live top-k surfaced this
      // generation.  Between generations only, never per token.
      void backfillSaeMeta();

      // Auto-regen with ``mode === "unsteered"`` *is* the A/B shadow.
      // Branch on the resolved recipe-override:
      //
      //   * ``"unsteered"`` → fire the shadow-replay path
      //     (``_sendShadowGenerate``).  Tokens land on the steered turn's
      //     ``abPair`` so the chat's right column renders them in place.
      //
      //   * any other override → fire a loom regen with the override.
      //     The engine drops the result as a sibling under the same
      //     user-parent; pin it so the chat's right column picks it up.
      if (autoRegenState.enabled) {
        const override = currentRecipeOverride();
        if (
          override === "unsteered" &&
          steeredIdx !== null &&
          chatLog.turns[steeredIdx]?.generated === true
        ) {
          void _sendShadowGenerate(steeredIdx);
        } else if (
          override !== null &&
          loomTree.loaded &&
          loomTree.active_node_id
        ) {
          // Pin the new sibling so the chat's right column shows it.
          // We pin after the regen lands; ``done`` from the regen will
          // set ``loomTree.active_node_id`` to the new sibling.
          const activeBefore = loomTree.active_node_id;
          void (async () => {
            await loomRegenerateActive(1, { recipe_override: override });
            // The engine moves the active node to the new sibling.
            if (
              loomTree.active_node_id &&
              loomTree.active_node_id !== activeBefore
            ) {
              pinNodeForComparison(loomTree.active_node_id);
            }
          })();
        }
      }
      return;
    }
    case "error": {
      genStatus.active = false;
      adoptStreamingNode(msg.node_id);
      const wasShadow = abState.processingAb;
      // Surface the error inline.  When the steered run errored we don't
      // want to spawn a shadow — clear A/B routing flags so a subsequent
      // successful gen behaves normally.  When the shadow itself errored
      // we still want the steered turn to remain visible as-is; just
      // mark its abPair as a placeholder error stub.
      if (wasShadow && abState.pendingTurnIdx !== null) {
        const steered = chatLog.turns[abState.pendingTurnIdx];
        if (steered) {
          steered.abPair = {
            role: "system",
            text: `shadow gen error: ${msg.message}`,
          };
        }
      } else {
        chatLog.turns = [
          ...chatLog.turns,
          { role: "system", text: `error: ${msg.message}` },
        ];
      }
      chatLog.pendingIndex = null;
      if (loomTree.pendingNodeId) {
        loomTree.pendingNodeId = null;
        if (loomTree.loaded) syncChatLogFromTree();
      }
      // The system turn appended above is rebuilt away whenever
      // ``syncChatLogFromTree`` runs (the tree knows nothing of it), so a
      // server-owned log rendered generation errors as a silent empty
      // node.  A sticky toast survives every tree sync — errors must
      // never be silent.
      pushToast(`generation: ${msg.message}`, {
        kind: "error",
        ttlMs: null,
      });
      abState.processingAb = false;
      abState.pendingTurnIdx = null;
      abState.pendingRole = null;
      abState.pendingRoleLabel = null;
      // Drain the next pending action even on error so the UI doesn't
      // get stuck in "changes pending" forever.  The failed send
      // already surfaced as the system message above.
      void drainNextPendingAction();
      return;
    }
  }
}

export interface SendGenerateOpts {
  stateless?: boolean;
  raw?: boolean;
  /** Cast model: which seat the generated turn occupies.  Absent /
   *  "assistant" = the classic flow; "user" needs scene mode server-side.
   *  Callers pass it explicitly (the composer reads its seat toggle) —
   *  the send primitive never defaults off ambient UI state. */
  generate_seat?: "user" | "assistant";
  /** Override the rack-derived steering with an explicit string.  Pass
   * ``""`` for unsteered (A/B mode); ``null``/``undefined`` to use the
   * rack. */
  steering?: string | null;
  /** Loom: attach the result as a child of this node.  ``null`` /
   *  absent = active node. */
  parent_node_id?: string | null;
  /** Loom: n-way regen.  Default 1. */
  n?: number;
  /** Loom phase 5: recipe-override modifier — mode string or partial
   *  recipe expression. */
  recipe_override?: string | null;
}

export interface SendSubmitOpts {
  /** Explicit anchor.  The queue-only sentinel resolves against the live
   * active node when this action reaches the head. */
  parent_node_id?: string | null | "active@drain";
  replaceSlot?: number | null;
  raw?: boolean;
  authored_thinking?: string | null;
  steering?: string | null;
  n?: number;
  recipe_override?: string | null;
}

function submitLabel(
  text: string | null,
  generatedRole: ChatRole | null,
): string {
  if (generatedRole === null) return "append";
  if (text === null) return "generate";
  return "send";
}

function buildSubmitPending(
  text: string | null,
  authoredRole: ChatRole | null,
  generatedRole: ChatRole | null,
  opts: Omit<SendSubmitOpts, "replaceSlot">,
): PendingAction {
  return {
    id: nextPendingId(),
    label: submitLabel(text, generatedRole),
    text,
    apply: () => sendSubmitNow(text, authoredRole, generatedRole, opts),
    awaitsGen: true,
    rebuild: text === null
      ? null
      : (newText: string) =>
          buildSubmitPending(newText, authoredRole, generatedRole, opts),
    createdAt: Date.now(),
    endsOnUserNode:
      (generatedRole ?? authoredRole) === "user"
        ? true
        : (generatedRole ?? authoredRole) === "assistant"
          ? false
          : null,
  };
}

/** Send one native role-neutral submission.
 *
 * Text is appended in ``authoredRole``. ``generatedRole`` optionally
 * follows it with a decode; omit it for append-only.  With no text, the
 * generated role continues directly from the selected leaf. No branch
 * depends on the selected node's role.
 */
export async function sendSubmit(
  text: string | null,
  authoredRole: ChatRole | null,
  generatedRole: ChatRole | null,
  opts: SendSubmitOpts = {},
): Promise<void> {
  if (text !== null && text === "") return;
  if (text !== null && authoredRole === null) {
    throw new Error("A text submission requires an authored role");
  }
  if (text === null && generatedRole === null) return;
  if (isPendingBusy()) {
    const { replaceSlot, ...queuedOpts } = opts;
    const item = buildSubmitPending(
      text,
      authoredRole,
      generatedRole,
      queuedOpts,
    );
    enqueuePending(
      {
        label: item.label,
        text: item.text,
        apply: item.apply,
        awaitsGen: item.awaitsGen,
        rebuild: item.rebuild,
        endsOnUserNode: item.endsOnUserNode,
      },
      { replaceSlot: replaceSlot ?? null },
    );
    return;
  }
  return sendSubmitNow(text, authoredRole, generatedRole, opts);
}

async function sendSubmitNow(
  text: string | null,
  authoredRole: ChatRole | null,
  generatedRole: ChatRole | null,
  opts: Omit<SendSubmitOpts, "replaceSlot"> = {},
): Promise<void> {
  if (!loomTree.loaded) {
    await refreshLoomTree();
    if (!loomTree.loaded) {
      throw new Error("Conversation tree is not ready; retry after it loads");
    }
  }
  const sock = await ensureWebSocket();
  const steering =
    opts.steering === undefined ? currentSteeringExpression() : opts.steering;
  const sampling = buildSamplingPayload();
  genStatus.maxTokens = sampling?.max_tokens ?? samplingState.max_tokens;
  const requestedParent = opts.parent_node_id;
  const parent = requestedParent === "active@drain"
    ? loomTree.active_node_id
    : requestedParent;
  const payload: WSClientMessage = {
    type: "submit",
    text,
    authored_role: authoredRole,
    generated_role: generatedRole,
    steering: steering || null,
    sampling,
    thinking: samplingState.thinking ?? false,
    raw: opts.raw ?? false,
    ...(opts.authored_thinking
      ? { authored_thinking: opts.authored_thinking }
      : {}),
    ...(parent !== undefined ? { parent_node_id: parent } : {}),
    ...(opts.n !== undefined ? { n: opts.n } : {}),
    ...(opts.recipe_override !== undefined
      ? { recipe_override: opts.recipe_override }
      : {}),
  };
  const send = () => sock.send(JSON.stringify(payload));
  if (sock.readyState === WebSocket.OPEN) send();
  else sock.addEventListener("open", send, { once: true });
}

/** Send a bare-continuation generate request over the WS — the
 * regenerate / continue-from-committed path.  No authored text travels:
 * the model speaks next from ``opts.parent_node_id`` (or the active
 * leaf) in ``opts.generate_seat``.  Builds the steering expression from
 * the rack live, layers the SamplingConfig overrides when one-shot mode
 * is on, and routes everything through the singleton connection.
 *
 * Fires immediately even mid-generation: these are internal
 * store-to-store calls behind an explicit user gesture on a specific
 * node, not composer sends, so they carry no pending-queue slot. */
export async function sendGenerate(
  opts: SendGenerateOpts = {},
): Promise<void> {
  // The first server snapshot may legitimately be revision 0.  Require the
  // explicit readiness bit instead of guessing from the revision, and retain
  // this defensive fetch even though App gates user interaction during boot:
  // store-level callers and future surfaces should be safe on their own.
  if (!loomTree.loaded) {
    await refreshLoomTree();
    if (!loomTree.loaded) {
      throw new Error("Conversation tree is not ready; retry after it loads");
    }
  }
  const sock = await ensureWebSocket();
  const steering =
    opts.steering === undefined ? currentSteeringExpression() : opts.steering;
  const steeringPayload =
    opts.steering === undefined ? (steering || null) : steering;
  // Build the sampling payload — seed + the advanced extras (penalties,
  // stop, logit-bias, return_top_k).  temperature / top-p / top-k /
  // max-tokens are PATCHed to the session as the user edits them, so the
  // server reads its own (now-updated) defaults for those.
  const sampling = buildSamplingPayload();
  // Update genStatus.maxTokens locally so the progress bar widths know
  // their target before the first token lands.
  genStatus.maxTokens = sampling?.max_tokens ?? samplingState.max_tokens;
  const payload: WSClientMessage = {
    type: "generate",
    // A continue: no committed turn, the model speaks next from the
    // anchor node.
    input: null,
    steering: steeringPayload,
    sampling,
    // Coerce the current family-level automatic setting to explicit ``false`` so the
    // unchecked checkbox really means "no thinking" — the server's
    // chat-template templates treat ``null`` and ``False`` differently
    // on some families and we promised the user a binary toggle.
    thinking: samplingState.thinking ?? false,
    stateless: opts.stateless ?? false,
    raw: opts.raw ?? false,
    // Loom fields ride only when caller explicitly set them (server
    // ignores unknown fields, but the spec keeps them optional).
    ...(opts.parent_node_id !== undefined
      ? { parent_node_id: opts.parent_node_id }
      : {}),
    ...(opts.n !== undefined ? { n: opts.n } : {}),
    ...(opts.recipe_override !== undefined
      ? { recipe_override: opts.recipe_override }
      : {}),
    ...(opts.generate_seat !== undefined && opts.generate_seat !== "assistant"
      ? { generate_seat: opts.generate_seat }
      : {}),
  };
  const send = () => sock.send(JSON.stringify(payload));
  if (sock.readyState === WebSocket.OPEN) send();
  else sock.addEventListener("open", send, { once: true });
}

/** Logit fork — regenerate an existing assistant node as a sibling with
 *  one token swapped.  The server reuses the source node's stamped
 *  recipe (steering / sampling / seed / thinking) and replays its raw
 *  decode sequence up to ``rawIndex``, forcing ``altTokenId`` there
 *  before sampling the continuation.  Streams in like any regen: the
 *  new sibling lands via the WS ``tree_mutated`` / ``token`` / ``done``
 *  events and becomes the active branch. */
export async function sendFork(
  nodeId: string,
  rawIndex: number,
  altTokenId: number,
): Promise<void> {
  const sock = await ensureWebSocket();
  const payload: WSClientMessage = {
    type: "generate",
    fork_node_id: nodeId,
    fork_raw_index: rawIndex,
    fork_alt_token_id: altTokenId,
  };
  const send = () => sock.send(JSON.stringify(payload));
  if (sock.readyState === WebSocket.OPEN) send();
  else sock.addEventListener("open", send, { once: true });
}

export function sendStop(): void {
  if (
    wsConn.socket &&
    wsConn.socket.readyState === WebSocket.OPEN
  ) {
    wsConn.socket.send(JSON.stringify({ type: "stop" }));
  }
}

// =========================================== A/B compare metadata =======

/** Transient routing state for the unsteered-shadow generation.
 *
 *  The shadow is ``autoRegenState`` with ``mode === "unsteered"``; there
 *  is no standalone toggle.  ``processingAb`` / ``pendingTurnIdx`` are
 *  load-bearing for the WS dispatcher — while ``processingAb`` is set the
 *  next ``started``/``token``/``done`` stream routes into
 *  ``chatLog.turns[pendingTurnIdx].abPair`` instead of appending a fresh
 *  top-level turn.  ``pendingTurnIdx`` is set when the shadow gen is
 *  dispatched and cleared on its ``done`` / ``error``.
 *
 *  The shadow's prompt is reconstructed from ``chatLog.turns`` at fire
 *  time (see ``_buildShadowMessages``), so the comparison works for any
 *  turn, not only the just-sent one.  Turning auto-regen off mid-flight
 *  lets the in-flight shadow finish writing into ``abPair`` — the turn is
 *  harmless when not rendered, and tearing the WS state down mid-stream
 *  is more error-prone than letting it complete; it only prevents the
 *  *next* steered gen from spawning a shadow. */
export interface AbState {
  pendingTurnIdx: number | null;
  processingAb: boolean;
  pendingRole: "user" | "assistant" | null;
  pendingRoleLabel: string | null;
}

export const abState: AbState = $state({
  pendingTurnIdx: null,
  processingAb: false,
  pendingRole: null,
  pendingRoleLabel: null,
});

/** Build the conversation as a messages list to replay through the
 * unsteered shadow.  Walks ``chatLog.turns[0..steeredIdx-1]`` (excluding
 * ``steeredIdx`` itself, which is the generated response we
 * don't want the shadow to inherit), filtering out system / error turns
 * that aren't real conversation context.
 *
 * The unsteered model sees prior steered turns as if they
 * happened naturally — that's the user's "play the conversation back"
 * contract. Scene rendering permits arbitrary seat sequences, so the target
 * turn itself—not the final history role—selects the prompt that follows. */
function _buildShadowMessages(
  steeredIdx: number,
): Array<{
  role: "user" | "assistant";
  content: string;
  label?: string | null;
}> {
  const out: Array<{
    role: "user" | "assistant";
    content: string;
    label?: string | null;
  }> = [];
  for (let i = 0; i < steeredIdx; i++) {
    const t = chatLog.turns[i];
    if (!t) continue;
    if (t.role !== "user" && t.role !== "assistant") continue; // skip system / errors
    // Use the accumulated text — generated turns already exclude their
    // thinking content (only response tokens land in ``turn.text``), so
    // replaying them through ``enable_thinking=False`` is well-formed.
    out.push({ role: t.role, content: t.text ?? "", label: t.roleLabel });
  }
  return out;
}

/** Internal: dispatch the unsteered shadow generate that pairs with the
 * just-finished steered turn at index ``steeredIdx``.  Sends the full
 * conversation as a ``messages`` list instead of a bare input string +
 * server-side history — the shadow runs ``stateless: true`` so the
 * server doesn't append to history (the steered branch already did) and
 * the messages list is the *only* context the unsteered model sees.
 * That makes the comparison work for any turn, not just the first. */
async function _sendShadowGenerate(steeredIdx: number): Promise<void> {
  const target = chatLog.turns[steeredIdx];
  if (!target?.generated || target.role === "system") return;
  const messages = _buildShadowMessages(steeredIdx);
  const sock = await ensureWebSocket();
  // Shadow path mirrors ``sendGenerate``'s sampling-payload build so the
  // ``return_top_k`` opt-in rides shadow / auto-regen runs too (matches
  // the steered turn's wire-shape, keeps logit captures comparable across
  // siblings).
  const sampling = buildSamplingPayload() ?? {};
  if (target.roleLabel) {
    if (target.role === "user") sampling.user_role = target.roleLabel;
    else sampling.assistant_role = target.roleLabel;
  }
  // Mark the WS reception path before the request lands so the
  // ``started`` event routes into the abPair and not a fresh turn.
  abState.pendingTurnIdx = steeredIdx;
  abState.processingAb = true;
  abState.pendingRole = target.role;
  abState.pendingRoleLabel = target.roleLabel ?? null;
  const payload: WSClientMessage = {
    type: "generate",
    // ``input`` accepts ``Any`` server-side; a list goes straight through
    // to ``session._prepare_input`` which dispatches on isinstance(list).
    input: messages,
    // Empty steering string == unsteered shadow per the WS protocol
    // (the server treats "" as "no expression").
    steering: "",
    sampling,
    thinking: samplingState.thinking ?? false,
    // Stateless so the shadow doesn't pollute server-side history; the
    // steered turn already populated history.  Combined with the
    // explicit messages list this means the shadow's prompt is exactly
    // the conversation up to (but not including) the steered response.
    stateless: true,
    raw: false,
    generate_seat: target.role,
  };
  const send = () => sock.send(JSON.stringify(payload));
  if (sock.readyState === WebSocket.OPEN) send();
  else sock.addEventListener("open", send, { once: true });
}

// ----------------------------------- auto-regen recipe-override -----

/** Built-in auto-regen modes from the engine. */
export type AutoRegenMode =
  | "unsteered"
  | "inverted"
  | "reseed"
  | "cool"
  | "hot"
  | "custom";

export interface AutoRegenState {
  /** Master toggle (replaces the old A/B toggle one-for-one).  Default
   *  off — the previous A/B behaviour resumed by toggling on with mode
   *  ``"unsteered"``. */
  enabled: boolean;
  mode: AutoRegenMode;
  /** Custom-mode body — a partial-recipe expression (e.g. ``"seed=42,
   *  temperature=1.5"``).  Ignored when ``mode != "custom"``. */
  custom: string;
}

export const autoRegenState: AutoRegenState = $state({
  enabled: false,
  mode: "unsteered",
  custom: "",
});

export function toggleAutoRegen(): void {
  const wasOff = !autoRegenState.enabled;
  autoRegenState.enabled = !autoRegenState.enabled;
  // Off → on with the "unsteered" mode: replay the conversation through
  // the unsteered model for the most recent generated turn that doesn't
  // already carry an ``abPair``, so users who flip the toggle on
  // after-the-fact see the right column populate immediately rather
  // than waiting for the next send.  Other modes use the loom-regen
  // path — they take effect on the next ``done`` event by design.
  if (!wasOff) return;
  if (genStatus.active) return; // ``done`` handler will fire its own
  if (currentRecipeOverride() !== "unsteered") return;
  for (let i = chatLog.turns.length - 1; i >= 0; i--) {
    const t = chatLog.turns[i];
    if (!t) continue;
    if (!t.generated || t.role === "system") continue;
    if (t.abPair) break;
    void _sendShadowGenerate(i);
    break;
  }
}

export function setAutoRegenMode(mode: AutoRegenMode): void {
  autoRegenState.mode = mode;
}

export function setAutoRegenCustom(text: string): void {
  autoRegenState.custom = text;
}

/** Render the configured recipe-override the engine consumes.  Returns
 *  ``null`` when auto-regen is off — callers shouldn't dispatch a
 *  shadow regen in that case. */
export function currentRecipeOverride(): string | null {
  if (!autoRegenState.enabled) return null;
  if (autoRegenState.mode === "custom") {
    const v = autoRegenState.custom.trim();
    return v || null;
  }
  return autoRegenState.mode;
}

// ============================================================ misc ======

/** Bootstrap the dashboard — call once on App mount.  Resolves only once
 * every parallel fetch settles so the UI's first paint has a real session
 * shape. */
export async function bootstrap(): Promise<void> {
  // Session info has to land before the localStorage key is known
  // (it's scoped by model_id), so we serialize that step.  The other
  // refreshes parallelize as before.
  await refreshSession();
  // Restore presentation preferences before attaching the persist effect so
  // we do not immediately overwrite them.  The tree always comes from the
  // authoritative server fetch below.
  loadPersistedPreferences();
  // Per-model render-mode override (base vs chat) — also model-scoped.
  loadGenUiMode();
  attachPersistence();
  await Promise.allSettled([
    refreshVectorList(),
    // Unified probe roster — every probe shape.
    refreshProbeList(),
    refreshCorrelation(),
    // Current manifold catalog.
    refreshManifoldList(),
    // Server tree wins — fetch and reconcile; failures remain visible.
    refreshLoomTree(),
  ]);
}
