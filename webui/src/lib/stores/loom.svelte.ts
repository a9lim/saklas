// The client mirror of the server's LoomTree.
//
// The server tree is authoritative: every mutator below fires its REST
// call and lets the server-emitted ``tree_mutated`` event sync the local
// store, so the local copy stays in lockstep with the server revision
// (no optimistic updates).  ``chatLog.turns`` is a projection of the
// active path — ``syncChatLogFromTree`` re-derives it after every
// mutation, preserving the in-flight turn object so streamed tokens keep
// accumulating on it.  That preservation is the bridge between "tree is
// authoritative" and "live tokens land on an existing turn".

import { SvelteMap } from "svelte/reactivity";
import { apiTree } from "../api";
import type { LoomNodeJSON, LoomTreeJSON } from "../api";
import type { CastMemberJSON, ChatTurn, TokenScore } from "../types";
import { isScalarReading } from "../types";
import { pushToast } from "./toasts.svelte";
import { chatLog, genStatus } from "./chat.svelte";
import { mergedReadings } from "./instruments.svelte";
import { invalidateEdgeLabels } from "./loomUi.svelte";
import { hydrateProbeRackFromActiveNode } from "./probes.svelte";
import { sessionState } from "./session.svelte";
import { sendGenerate } from "../stores.svelte";

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
export function recomputeActivePath(): void {
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
export function syncChatLogFromTree(): void {
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
export function applyTreeSnapshot(snap: LoomTreeJSON): void {
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
export function applyTreeDelta(ev: {
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
