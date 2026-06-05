// Cross-component state for the v1.7 dashboard.
//
// Svelte 5 runes-based.  Each slice is a $state-backed object exported as
// a named const; components import the slice and read/write its fields
// directly — Svelte's compiler tracks dependencies automatically.
//
// Cross-cutting actions (open the WS, send a generate, queue a pending
// rack edit during in-flight gen) live in this file as functions so panels
// don't need to coordinate amongst themselves; they call ``sendGenerate(...)``
// or ``setVectorAlpha(name, alpha)`` and the slice updates propagate.
//
// One singleton WS owned at the module level — the chat panel is no
// longer responsible for lifecycle.  Subscribers register via
// ``onWsMessage(cb)`` and receive every ``WSServerMessage`` the
// connection emits.

import { SvelteMap, SvelteSet } from "svelte/reactivity";
import {
  apiSessions,
  apiVectors,
  apiProbes,
  apiManifolds,
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
  ChatTurn,
  GenStatus,
  ManifoldRackEntry,
  PendingAction,
  ProbeInfo,
  ProbeReadingJSON,
  ProbeRackEntry,
  ProbeSortMode,
  ProjectionSpec,
  TokenScore,
  Trigger,
  Variant,
  VectorRackEntry,
  WSSampling,
} from "./types";
import { serializeExpression } from "./expression";
import { SURPRISE_TARGET } from "./tokens";
import { pushToast } from "./stores/toasts.svelte";

export * from "./stores/drawers.svelte";
export * from "./stores/inputHistory.svelte";
import {
  onPendingQueueShift,
  requestInputRestore,
} from "./stores/inputHistory.svelte";
export { dismissToast, pushToast, toasts } from "./stores/toasts.svelte";

// =========================================================== session ====

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

export async function refreshSession(): Promise<void> {
  try {
    const info = await apiSessions.get();
    sessionState.info = info;
    sessionState.lastRefresh = Date.now();
    sessionState.error = null;
    _hydrateSamplingFromInfo();
  } catch (e) {
    sessionState.error = e instanceof Error ? e.message : String(e);
  }
}

/** One-shot guard: the role boxes are client-sticky (they ride each send),
 *  unlike the numeric sampling defaults which mirror server config on every
 *  patch.  We seed them from the family's standard role labels exactly once,
 *  on the first session-info load — re-seeding on later patches would clobber
 *  a value the user typed. */
let _roleDefaultsSeeded = false;

/** Mirror the server's session.config defaults into the local
 * ``samplingState``.  The local store was previously pre-seeded with its
 * own constants (``max_tokens: 256`` etc.) which drifted away from the
 * server's actual ``session.config.max_new_tokens`` (= 1024 by default),
 * so the gen-status footer rendered ``gen N/256`` even when the engine
 * was running against a 1024-token cap.  Sync once on every refresh so
 * the displayed cap matches what generation actually used. */
function _hydrateSamplingFromInfo(): void {
  const info = sessionState.info;
  // Seed the sticky role boxes once, so they show e.g. ``user`` / ``model``
  // instead of an empty ``—`` placeholder.  Only fills an empty box, so a
  // value already in hand (typed before info landed) is never overwritten.
  if (!_roleDefaultsSeeded && info) {
    _roleDefaultsSeeded = true;
    if (samplingState.user_role === "" && info.default_user_role) {
      samplingState.user_role = info.default_user_role;
    }
    if (samplingState.assistant_role === "" && info.default_assistant_role) {
      samplingState.assistant_role = info.default_assistant_role;
    }
  }
  const cfg = info?.config;
  if (!cfg) return;
  if (typeof cfg.max_tokens === "number" && Number.isFinite(cfg.max_tokens)) {
    samplingState.max_tokens = cfg.max_tokens;
  }
  if (typeof cfg.temperature === "number") {
    samplingState.temperature = cfg.temperature;
  }
  if (typeof cfg.top_p === "number") {
    samplingState.top_p = cfg.top_p;
  }
  if (typeof cfg.top_k === "number") {
    samplingState.top_k = cfg.top_k;
  }
  if (typeof cfg.system_prompt === "string") {
    samplingState.system_prompt = cfg.system_prompt;
  }
}

export async function patchSessionDefaults(
  body: Partial<{
    temperature: number;
    top_p: number;
    top_k: number;
    max_tokens: number;
    system_prompt: string;
    thinking: boolean;
  }>,
): Promise<void> {
  const info = await apiSessions.patch(body);
  sessionState.info = info;
  sessionState.lastRefresh = Date.now();
  _hydrateSamplingFromInfo();
}

/** Display label for a turn, honoring its per-message role-substitution
 *  label (the roleplay scaffold stamped at send time).  ``roleLabel`` —
 *  the node's ``role_label`` — wins when set; otherwise the structural
 *  ``role`` (user / assistant / system) is shown verbatim. */
export function roleDisplayLabel(
  role: string,
  roleLabel?: string | null,
): string {
  return roleLabel || role;
}

/** Single-character glyph for the loom node badge — first char of the
 *  display label, uppercased.  Default roles reduce to ``U`` / ``A`` / ``S``;
 *  a per-turn ``captain`` label yields ``C``. */
export function roleGlyphLetter(
  role: string,
  roleLabel?: string | null,
): string {
  const label = roleDisplayLabel(role, roleLabel);
  return (label.charAt(0) || role.charAt(0) || "?").toUpperCase();
}

/** "Clear chat" — preserves the loom tree and just navigates back to the
 *  synthetic system root.  The next submission lands a fresh user turn
 *  as a sibling branch off root rather than the legacy behaviour of
 *  destroying every existing branch.  Earlier conversation paths stay
 *  reachable from the sidebar.
 *
 *  Falls back to the destructive ``apiSessions.clear()`` only when the
 *  server has no loom (legacy/unavailable) — without that branch the
 *  button would be a no-op there. */
export async function resetChatToRoot(): Promise<void> {
  const root = loomTree.root_id;
  if (root !== null) {
    // ``loomNavigate`` runs the navigate REST call and refreshes the
    // tree; ``syncChatLogFromTree`` then projects the [root]-only
    // active path to an empty ``chatLog.turns`` because the synthetic
    // system root is filtered out of the chat view.
    await loomNavigate(root);
    return;
  }
  if (loomTree.unavailable) {
    await apiSessions.clear();
    await refreshSession();
    chatLog.turns = [];
  }
}

/** Clear the chat back to root.  Lifted out of Chat.svelte so the
 *  threads-column action button can call the same code path the chat
 *  header used to — queue-aware when generation is in flight, direct
 *  when idle. */
export function clearChat(): void {
  if (genStatus.active || pendingActions.queue.length > 0) {
    enqueuePending({
      label: "/clear",
      text: null,
      apply: () => void resetChatToRoot(),
      awaitsGen: false,
      rebuild: null,
      // /clear navigates to the synthetic root (system role) — not a
      // user node, so the next submission goes through "message" mode
      // and lands a fresh user branch off root.
      endsOnUserNode: false,
    });
  } else {
    void resetChatToRoot();
  }
}

export async function rewindSession(): Promise<void> {
  await apiSessions.rewind();
  await refreshSession();
  // Drop the trailing user→assistant pair from the local log so the UI
  // stays in lockstep with server-side history.
  const t = chatLog.turns;
  for (let i = t.length - 1; i >= 0; i--) {
    if (t[i].role === "user") {
      chatLog.turns = t.slice(0, i);
      return;
    }
  }
}

// =========================================================== vectors ====

export interface VectorRack {
  /** Rack key = atom display form (``honest``, ``ns/foo``, ``happy.sad``).
   * One entry per concept.  Variant lives on the entry, not the key —
   * matching the saklas parser's Steering.alphas semantics. */
  entries: Map<string, VectorRackEntry>;
  /** Per-vector profile metadata fetched from GET /vectors/{name}.
   * Populated lazily; absent until the user opens a strip's expander. */
  profiles: Map<string, VectorInfo>;
  /** Cosine matrix from GET /correlation; refreshed after each generation. */
  correlation: CorrelationData | null;
}

// SvelteMap from svelte/reactivity — plain Map mutations don't trigger
// Svelte 5 rune reactivity, so any rack add/remove or profile cache
// update wouldn't re-render the strips list.  SvelteMap.set/.delete is
// rune-tracked.  Inner-object property writes still aren't tracked, so
// callers that mutate an entry must reassign via .set(name, {...e, …}).
export const vectorRack: VectorRack = $state({
  entries: new SvelteMap(),
  profiles: new SvelteMap(),
  correlation: null,
});

/** Server-derived list of registered vectors — names only.  Mirrors
 * sessionState.info?.vectors but kept as its own slice so panels that
 * only care about the list don't re-render when other session fields
 * change. */
export const vectorsState: { names: string[] } = $state({ names: [] });

export async function refreshVectorList(): Promise<void> {
  const r = await apiVectors.list();
  vectorsState.names = r.vectors.map((v) => v.name);
  // Cache profile metadata — cheap, server already serialized.
  for (const v of r.vectors) {
    vectorRack.profiles.set(v.name, v);
  }
}

export async function refreshVector(name: string): Promise<VectorInfo> {
  const info = await apiVectors.get(name);
  vectorRack.profiles.set(name, info);
  return info;
}

export async function refreshCorrelation(
  names?: string[] | null,
): Promise<void> {
  try {
    const data = await apiVectors.correlation(names);
    vectorRack.correlation = data;
  } catch {
    vectorRack.correlation = null;
  }
}

// SvelteMap tracks .set/.delete; mutations on stored objects are NOT
// tracked, so each setter reassigns the entry via .set with a fresh
// spread.  This pattern is uniform across every rack mutator.
export function setVectorAlpha(name: string, alpha: number): void {
  enqueueOrApply(`alpha ${name} ${alpha.toFixed(3)}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) {
      vectorRack.entries.set(name, { ...e, alpha });
    } else {
      vectorRack.entries.set(name, defaultRackEntry(alpha));
    }
  });
}

export function setVectorEnabled(name: string, enabled: boolean): void {
  enqueueOrApply(`${enabled ? "enable" : "disable"} ${name}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) vectorRack.entries.set(name, { ...e, enabled });
  });
}

export function setVectorTrigger(name: string, trigger: Trigger): void {
  enqueueOrApply(`trigger ${name} ${trigger}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) vectorRack.entries.set(name, { ...e, trigger });
  });
}

export function setVectorVariant(name: string, variant: Variant): void {
  enqueueOrApply(`variant ${name} ${variant}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) vectorRack.entries.set(name, { ...e, variant });
  });
}

export function setVectorProjection(
  name: string,
  projection: ProjectionSpec | null,
): void {
  enqueueOrApply(`project ${name}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) {
      // Ablation can't compose with projection — clear if a projection
      // was just set on top of an ablated entry.
      vectorRack.entries.set(name, {
        ...e,
        projection,
        ablate: projection ? false : e.ablate,
      });
    }
  });
}

export function setVectorAblate(name: string, ablate: boolean): void {
  enqueueOrApply(`ablate ${name} ${ablate}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) {
      vectorRack.entries.set(name, {
        ...e,
        ablate,
        projection: ablate ? null : e.projection,
      });
    }
  });
}

/** Default α for a freshly-added rack entry — matches DEFAULT_COEFF in
 * saklas.core.steering_expr (TUI's /steer assumes 0.5 when the user
 * types a bare term).  α=0 would serialize to an empty expression and
 * make the new strip invisible in the EXPR block. */
const DEFAULT_RACK_ALPHA = 0.5;

export function addVectorToRack(
  name: string,
  alpha: number = DEFAULT_RACK_ALPHA,
  trigger: Trigger = "BOTH",
): void {
  if (vectorRack.entries.has(name)) return;
  vectorRack.entries.set(name, defaultRackEntry(alpha, trigger));
}

export function removeVectorFromRack(name: string): void {
  vectorRack.entries.delete(name);
}

function defaultRackEntry(
  alpha: number = 0,
  trigger: Trigger = "BOTH",
): VectorRackEntry {
  return {
    alpha,
    trigger,
    variant: "raw",
    projection: null,
    ablate: false,
    enabled: true,
  };
}

/** The canonical expression string the racks would send to the server.
 * Recomputed on demand; cheap.  Folds in manifold (``%``) terms after
 * the vector terms. */
export function currentSteeringExpression(): string {
  return serializeExpression(vectorRack.entries, manifoldRack.entries);
}

// ========================================================= manifolds ====

export interface ManifoldRack {
  /** Server-side catalog of available manifolds — refreshed by
   *  ``refreshManifoldList``.  Empty (and ``unavailable`` true) on
   *  servers that pre-date the manifold HTTP surface. */
  catalog: ManifoldInfo[];
  /** Racked manifolds keyed by display name (``ns/name`` or bare
   *  ``name``).  Each entry carries blend / coords / trigger / enabled
   *  — parallel to the vector rack. */
  entries: Map<string, ManifoldRackEntry>;
  /** True when the manifold list route 404s — older server. */
  unavailable: boolean;
  loading: boolean;
  error: string | null;
}

export const manifoldRack: ManifoldRack = $state({
  catalog: [],
  entries: new SvelteMap(),
  unavailable: false,
  loading: false,
  error: null,
});

/** Fetch the manifold catalog.  Tolerates a 404 from an older server —
 *  flips ``unavailable`` and leaves the catalog empty. */
export async function refreshManifoldList(): Promise<void> {
  manifoldRack.loading = true;
  try {
    const r = await apiManifolds.list();
    manifoldRack.catalog = r.manifolds;
    manifoldRack.unavailable = false;
    manifoldRack.error = null;
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      manifoldRack.unavailable = true;
      manifoldRack.catalog = [];
    } else {
      manifoldRack.error = e instanceof Error ? e.message : String(e);
    }
  } finally {
    manifoldRack.loading = false;
  }
}

/** Look up a catalog row by display name (``ns/name`` or bare name). */
export function manifoldByName(name: string): ManifoldInfo | null {
  for (const m of manifoldRack.catalog) {
    if (`${m.namespace}/${m.name}` === name || m.name === name) return m;
  }
  return null;
}

/** Domain-centroid coordinates for a manifold — the default rack
 *  position.  Box: midpoint of each axis.  Sphere: the north pole
 *  ``[0,…,0,1]`` in R^(dim+1) embedding (here we just author with
 *  ``dim`` intrinsic coords, all zero, which the domain maps to a valid
 *  point). */
export function manifoldCentroid(m: ManifoldInfo): number[] {
  if (m.domain.type === "box") {
    return m.domain.axes.map((a) => (a.lo + a.hi) / 2);
  }
  // Sphere / custom — intrinsic_dim zeros is a safe authoring default.
  return new Array(m.intrinsic_dim).fill(0);
}

/** Add a manifold to the rack at its domain centroid, blend 0.5. */
export function addManifoldToRack(name: string): void {
  if (manifoldRack.entries.has(name)) return;
  const info = manifoldByName(name);
  const coords = info ? manifoldCentroid(info) : [];
  manifoldRack.entries.set(name, {
    blend: 0.5,
    coords,
    label: null,
    trigger: "BOTH",
    enabled: true,
  });
}

export function removeManifoldFromRack(name: string): void {
  manifoldRack.entries.delete(name);
}

export function setManifoldBlend(name: string, blend: number): void {
  enqueueOrApply(`manifold blend ${name} ${blend.toFixed(3)}`, () => {
    const e = manifoldRack.entries.get(name);
    if (e) manifoldRack.entries.set(name, { ...e, blend });
  });
}

export function setManifoldCoords(name: string, coords: number[]): void {
  // Pulling on the XYPad authors a free-form position; the term drops
  // its label-form binding (if any) so the canonical expression
  // serializes as a coord list and the snap-to-node dropdown shows
  // "(no node)" until the user picks one.
  enqueueOrApply(`manifold coords ${name}`, () => {
    const e = manifoldRack.entries.get(name);
    if (e) manifoldRack.entries.set(name, {
      ...e, coords: [...coords], label: null,
    });
  });
}

/** Switch the term to label-form (``<name>%<label>``).  ``label=null``
 *  clears the binding and reverts to coord-form on the next
 *  serialization.  When ``label`` is non-null the matching node's
 *  coords are mirrored onto ``coords`` so the XYPad still renders the
 *  position correctly. */
export function setManifoldLabel(name: string, label: string | null): void {
  enqueueOrApply(`manifold label ${name} ${label ?? "<null>"}`, () => {
    const e = manifoldRack.entries.get(name);
    if (!e) return;
    if (label === null) {
      manifoldRack.entries.set(name, { ...e, label: null });
      return;
    }
    const info = manifoldByName(name);
    if (!info) {
      // No catalog metadata — accept the label without mirroring
      // coords; downstream resolution happens server-side.
      manifoldRack.entries.set(name, { ...e, label });
      return;
    }
    const idx = info.node_labels.indexOf(label);
    const coords = (idx >= 0 && info.node_coords[idx])
      ? [...info.node_coords[idx]]
      : e.coords;
    manifoldRack.entries.set(name, { ...e, label, coords });
  });
}

export function setManifoldTrigger(name: string, trigger: Trigger): void {
  enqueueOrApply(`manifold trigger ${name} ${trigger}`, () => {
    const e = manifoldRack.entries.get(name);
    if (e) manifoldRack.entries.set(name, { ...e, trigger });
  });
}

export function setManifoldEnabled(name: string, enabled: boolean): void {
  enqueueOrApply(`manifold ${enabled ? "enable" : "disable"} ${name}`, () => {
    const e = manifoldRack.entries.get(name);
    if (e) manifoldRack.entries.set(name, { ...e, enabled });
  });
}

// =========================================================== probes =====
//
// One unified read-side rack — every probe shape (a 2-node concept axis is
// the rank-1 case, a discover / curved fit the rank-R case).  Each entry
// carries the server ``ProbeInfo`` (with the ``is_affine`` flat-vs-curved
// flag the cards classify on), a sparkline of the primary scalar, the
// latest per-token ``reading`` + end-of-gen ``aggregate`` (one
// ``ProbeReadingJSON`` shape), the most-recent ``nearest`` list, and — for
// 2-D box probes — an inferred per-token ``trajectory`` for the mini-map.

const MAX_SPARKLINE = 60;
const MAX_PROBE_TRAJECTORY = 240;

export interface ProbeRackState {
  /** Per-probe live state, keyed by registered probe name. */
  entries: Map<string, ProbeRackEntry>;
  sortMode: ProbeSortMode;
  /** Attached probe names (every listed probe is attached/active). */
  active: string[];
  /** True when the server's probe route 404s (pre-read-side server). */
  unavailable: boolean;
  loading: boolean;
  error: string | null;
}

export const probeRack: ProbeRackState = $state({
  entries: new SvelteMap(),
  // Alphabetical by default — matches the TUI's initial state.  Sort by
  // value / change is a dropdown opt-in.
  sortMode: "name",
  active: [],
  unavailable: false,
  loading: false,
  error: null,
});

/** Primary scalar a probe's sparkline / sort tracks: the signed axis-0
 *  coordinate for a flat (subspace) probe, the [0,1] subspace fraction for
 *  a curved (manifold) probe. */
function _primaryScalar(info: ProbeInfo, reading: ProbeReadingJSON): number {
  if (info.is_affine) return reading.coords.length > 0 ? reading.coords[0] : 0;
  return reading.fraction;
}

/** Per-layer column for the expanded layer strip: axis-0 ``coords_per_layer``
 *  for a flat probe, ``fraction_per_layer`` for a curved one. */
function _primaryPerLayer(
  info: ProbeInfo,
  reading: ProbeReadingJSON,
): Record<string, number> {
  if (!info.is_affine) return reading.fraction_per_layer ?? {};
  const out: Record<string, number> = {};
  for (const [layer, c] of Object.entries(reading.coords_per_layer ?? {})) {
    out[layer] = Array.isArray(c) && c.length > 0 ? c[0] : 0;
  }
  return out;
}

/** A probe targets a 2-D-authored ``BoxDomain`` — the regime the mini-map
 *  renders.  Higher-dim and sphere/custom probes attach but skip it. */
function _probeIsMiniMapCandidate(info: ProbeInfo): boolean {
  if (info.intrinsic_dim !== 2) return false;
  const d = info.domain as { type?: string };
  return d?.type === "box" && !!info.node_coords && info.node_coords.length > 0;
}

/** Look up ``node_coords`` for a label.  Null when absent or the row carries
 *  no coords (unfitted discover).  Returns a copy so callers can push. */
function _lookupNodeCoords(info: ProbeInfo, label: string): number[] | null {
  const coords = info.node_coords;
  if (!coords) return null;
  const idx = info.node_labels.indexOf(label);
  if (idx < 0 || idx >= coords.length) return null;
  const row = coords[idx];
  if (!Array.isArray(row)) return null;
  return [...row];
}

function _emptyProbeEntry(info: ProbeInfo): ProbeRackEntry {
  return {
    info,
    sparkline: [],
    current: 0,
    previous: 0,
    perLayer: {},
    reading: null,
    aggregate: null,
    nearest: [],
    trajectory: [],
  };
}

/** Computed: probe names sorted per the chosen sort mode.  Fresh array each
 *  access; consumers use it as a ``$derived`` read-only view. */
export function activeProbeNames(): string[] {
  const arr = [...probeRack.active];
  if (probeRack.sortMode === "name") {
    arr.sort();
  } else if (probeRack.sortMode === "value") {
    arr.sort((a, b) => {
      const av = probeRack.entries.get(a)?.current ?? 0;
      const bv = probeRack.entries.get(b)?.current ?? 0;
      return bv - av;
    });
  } else if (probeRack.sortMode === "change") {
    arr.sort((a, b) => {
      const ae = probeRack.entries.get(a);
      const be = probeRack.entries.get(b);
      const ad = Math.abs((ae?.current ?? 0) - (ae?.previous ?? 0));
      const bd = Math.abs((be?.current ?? 0) - (be?.previous ?? 0));
      return bd - ad;
    });
  }
  return arr;
}

/** Fetch the attached-probe catalog.  Tolerates a 404 from an older server
 *  — flips ``unavailable`` and leaves the rack empty. */
export async function refreshProbeList(): Promise<void> {
  probeRack.loading = true;
  try {
    const r = await apiProbes.list();
    const seen = new Set<string>();
    for (const info of r.probes) {
      seen.add(info.name);
      const prev = probeRack.entries.get(info.name);
      if (prev) {
        // Refresh metadata in place; preserve live sparkline / aggregate.
        probeRack.entries.set(info.name, { ...prev, info });
      } else {
        probeRack.entries.set(info.name, _emptyProbeEntry(info));
      }
    }
    // Drop entries the server no longer reports (detached out-of-band).
    for (const name of [...probeRack.entries.keys()]) {
      if (!seen.has(name)) probeRack.entries.delete(name);
    }
    probeRack.active = r.probes.map((p) => p.name);
    probeRack.unavailable = false;
    probeRack.error = null;
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      probeRack.unavailable = true;
      probeRack.entries.clear();
      probeRack.active = [];
    } else {
      probeRack.error = e instanceof Error ? e.message : String(e);
    }
  } finally {
    probeRack.loading = false;
  }
}

/** Attach any probe shape by selector — the same ``[ns/]name[:variant]`` the
 *  steering ``%`` term consumes; ``name`` defaults to the selector. */
export async function attachProbe(
  selector: string,
  opts: { name?: string; top_n?: number } = {},
): Promise<ProbeInfo> {
  const info = await apiProbes.attach({
    selector,
    name: opts.name,
    top_n: opts.top_n,
  });
  const prev = probeRack.entries.get(info.name);
  if (prev) {
    probeRack.entries.set(info.name, { ...prev, info });
  } else {
    probeRack.entries.set(info.name, _emptyProbeEntry(info));
  }
  if (!probeRack.active.includes(info.name)) {
    probeRack.active = [...probeRack.active, info.name];
  }
  // Seed the highlight target when a probe is attached through the rack —
  // matches the TUI pointing highlight at a fresh probe.
  if (highlightState.target === null) {
    highlightState.target = info.name;
  }
  return info;
}

/** Detach a probe by registered name. */
export async function detachProbe(name: string): Promise<void> {
  await apiProbes.detach(name);
  probeRack.entries.delete(name);
  probeRack.active = probeRack.active.filter((n) => n !== name);
  if (highlightState.target === name) highlightState.target = null;
  if (highlightState.compareTarget === name) highlightState.compareTarget = null;
}

export function setProbeSortMode(mode: ProbeSortMode): void {
  probeRack.sortMode = mode;
}

/** Reset per-gen streaming state at the start of a fresh generation.
 *  Trajectory + aggregate + nearest live one gen each; the sparkline
 *  carries across.  Called from the WS ``started`` handler. */
export function resetProbeStreams(): void {
  for (const [name, e] of probeRack.entries) {
    probeRack.entries.set(name, {
      ...e,
      nearest: [],
      aggregate: null,
      trajectory: [],
    });
  }
}

/** Append one per-token reading per attached probe (the unified
 *  ``probe_readings`` WS channel).  Drives the sparkline + per-layer strip +
 *  nearest readout + 2-D trajectory.  No-ops on undefined.
 *
 *  Reassigns each entry (rather than mutating in place) so the SvelteMap
 *  fires reactivity — a bare ``entry.current = v`` would freeze probe strips
 *  at zero through a whole generation. */
export function updateProbesFromReadings(
  readings: Record<string, ProbeReadingJSON> | undefined,
): void {
  if (!readings) return;
  for (const [name, reading] of Object.entries(readings)) {
    const prev = probeRack.entries.get(name);
    if (!prev) continue;
    const scalar = _primaryScalar(prev.info, reading);
    const sparkline = prev.sparkline.slice();
    sparkline.push(scalar);
    if (sparkline.length > MAX_SPARKLINE) {
      sparkline.splice(0, sparkline.length - MAX_SPARKLINE);
    }
    let trajectory = prev.trajectory;
    if (_probeIsMiniMapCandidate(prev.info) && reading.nearest.length > 0) {
      const xy = _lookupNodeCoords(prev.info, reading.nearest[0][0]);
      if (xy) {
        trajectory = prev.trajectory.slice();
        trajectory.push(xy);
        if (trajectory.length > MAX_PROBE_TRAJECTORY) {
          trajectory.splice(0, trajectory.length - MAX_PROBE_TRAJECTORY);
        }
      }
    }
    probeRack.entries.set(name, {
      ...prev,
      sparkline,
      current: scalar,
      previous: prev.current,
      perLayer: _primaryPerLayer(prev.info, reading),
      reading,
      nearest: reading.nearest,
      trajectory,
    });
  }
}

/** Land the end-of-gen aggregate readings (the ``done`` event) — the settled
 *  ``ProbeReading`` per probe. */
export function setProbeAggregates(
  aggregates: Record<string, ProbeReadingJSON> | undefined,
): void {
  if (!aggregates) return;
  for (const [name, agg] of Object.entries(aggregates)) {
    const prev = probeRack.entries.get(name);
    if (!prev) continue;
    probeRack.entries.set(name, {
      ...prev,
      aggregate: agg,
      current: _primaryScalar(prev.info, agg),
      perLayer: _primaryPerLayer(prev.info, agg),
      nearest: agg.nearest,
    });
  }
}

/** Snapshot current per-probe scalars as the new "previous" baseline — call
 *  after a gen lands so the next gen's deltas compute against post-gen state. */
export function snapshotProbeBaseline(): void {
  for (const [name, e] of probeRack.entries) {
    probeRack.entries.set(name, { ...e, previous: e.current });
  }
}

/** Sentinel for the layer strip when a probe has no token yet. */
export const EMPTY_PER_LAYER: Readonly<Record<string, number>> = Object.freeze({});

// ============================================================ chat ======

export interface ChatLogState {
  turns: ChatTurn[];
  /** Index of the in-flight assistant turn, when one exists.  Null
   * between gens.  Used by the WS event handlers to attach streamed
   * tokens to the right turn. */
  pendingIndex: number | null;
}

export const chatLog: ChatLogState = $state({
  turns: [],
  pendingIndex: null,
});

// ============================================================ loom tree ===
//
// Mirrors the server's LoomTree (phase 2 spec).  The slice is the
// authoritative shape for the loom sidebar; ``chatLog.turns`` is sync'd
// from the active path via ``syncChatLogFromTree`` whenever ``loomTree``
// changes (rev-driven).
//
// Until the server exposes /tree (a saklas server older than v2.3 will
// 404 on the bootstrap fetch), ``loomTree.rev`` stays at 0 and the
// legacy non-loom write paths in ``handleWsMessage`` continue to own
// ``chatLog.turns``.  Once rev > 0 the sidebar is the truth and
// ``handleWsMessage`` is a no-op for tree-shape concerns; we still keep
// the legacy writers for streaming-token append (the token deltas
// arrive as ``token`` events, not as ``tree_mutated`` deltas).

export interface LoomTreeState {
  root_id: string | null;
  active_node_id: string | null;
  /** Per-node cache.  SvelteMap so ``set``/``delete`` trigger reactivity
   *  in the sidebar without manual re-renders. */
  nodes: Map<string, LoomNodeJSON>;
  /** parent_id → ordered child ids.  Same SvelteMap pattern. */
  children_of: Map<string, string[]>;
  /** Monotonic revision cursor.  0 means "no server tree fetched yet"
   *  — the UI falls back to legacy single-path rendering until > 0. */
  rev: number;
  /** Pending in-flight gen target id (when known).  Reflects the
   *  ``started`` / ``node_created`` event's ``node_id`` field; null
   *  between gens. */
  pendingNodeId: string | null;
  /** Cached active path as an ordered list of node ids.  Recomputed on
   *  every ``rev`` bump so sidebar / chat sync work in O(depth). */
  activePath: string[];
  /** Last seen server-side model id; used to invalidate cache across
   *  model swaps. */
  modelId: string | null;
  /** Set when the server tree fetch fails 404 or otherwise — UI hides
   *  the loom sidebar toggle when the server doesn't support loom. */
  unavailable: boolean;
  /** Last fetch error message; surfaced in the sidebar. */
  error: string | null;
}

export const loomTree: LoomTreeState = $state({
  root_id: null,
  active_node_id: null,
  nodes: new SvelteMap(),
  children_of: new SvelteMap(),
  rev: 0,
  pendingNodeId: null,
  activePath: [],
  modelId: null,
  unavailable: false,
  error: null,
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
  const out: TokenScore = {
    text: row.text ?? "",
    thinking: false,
  };
  if (row.token_id !== undefined) out.tokenId = row.token_id;
  if (row.logprob !== undefined) out.logprob = row.logprob;
  if (row.top_alts) out.topAlts = row.top_alts;
  if (row.raw_index !== undefined) out.rawIndex = row.raw_index;
  if (row.probes) out.probes = row.probes;
  if (row.per_layer_scores) out.perLayerScores = row.per_layer_scores;
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
    text: n.text ?? "",
    roleLabel: n.role_label ?? null,
    nodeId: n.id,
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
  if (n.thinking_tokens && n.thinking_tokens.length > 0) {
    turn.thinkingTokens = n.thinking_tokens.map((r) => {
      const s = tokenRowToScore(r);
      s.thinking = true;
      return s;
    });
    turn.thinking = true;
  }
  return turn;
}

/** Per-node cache of streamed per-token score data.
 *
 *  The loom tree (``LoomNodeJSON``) carries no per-token probe scores —
 *  only aggregate readings.  ``syncChatLogFromTree`` rebuilds
 *  ``chatLog.turns`` from the tree on every navigation, so without this
 *  cache the inline highlight tint vanishes the moment you navigate away
 *  from a branch and back (the rebuilt turn has no ``tokens``).
 *
 *  Keyed by node id (ULIDs are unique per node, so a regenerated node
 *  never collides with a cached predecessor).  Plain Map — it holds the
 *  same token-array references the turns already use, no reactivity
 *  needed. */
const tokenScoreCache = new Map<
  string,
  { tokens?: TokenScore[]; thinkingTokens?: TokenScore[] }
>();

function attachChild(parentId: string | null, childId: string): void {
  if (parentId === null) return;
  const siblings = loomTree.children_of.get(parentId) ?? [];
  if (!siblings.includes(childId)) {
    loomTree.children_of.set(parentId, [...siblings, childId]);
  }
}

function upsertLoomNode(raw: LoomNodeJSON & { children?: string[] }): LoomNodeJSON {
  const { children: _children, ...node } = raw;
  loomTree.nodes.set(node.id, node);
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
  if (loomTree.rev <= 0) return;
  const path = loomTree.activePath;
  if (path.length === 0) {
    chatLog.turns = [];
    chatLog.pendingIndex = null;
    return;
  }
  // Harvest per-token score data from the turns we're about to drop, so
  // navigating away from a branch and back can restore the inline
  // highlight (the tree itself carries no per-token scores).
  for (const t of chatLog.turns) {
    if (
      t.nodeId &&
      ((t.tokens?.length ?? 0) > 0 || (t.thinkingTokens?.length ?? 0) > 0)
    ) {
      tokenScoreCache.set(t.nodeId, {
        tokens: t.tokens,
        thinkingTokens: t.thinkingTokens,
      });
    }
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
    const prevByNode = chatLog.turns.find((t) => t.nodeId === nid);
    const prev = prevByNode ?? chatLog.turns[out.length];
    let turn: ChatTurn;
    if (
      prev &&
      prev.role === node.role &&
      (prev.nodeId === nid ||
        // Same text or live-stream-of-text on the active in-flight turn.
        prev.text === node.text ||
        (loomTree.pendingNodeId === nid && prev.role === "assistant"))
    ) {
      // Mutate-in-place so the streaming token arrays survive.
      prev.nodeId = nid;
      const nextText = node.text ?? "";
      if (
        !(
          loomTree.pendingNodeId === nid &&
          prev.role === "assistant" &&
          nextText === ""
        )
      ) {
        prev.text = nextText;
      }
      prev.appliedSteering = node.applied_steering ?? prev.appliedSteering ?? null;
      prev.aggregateReadings = node.aggregate_readings ?? prev.aggregateReadings;
      prev.finishReason = node.finish_reason ?? prev.finishReason;
      // Restore per-token scores if this reused turn lost them
      // (positional fallback from a different node, or first-paint
      // hydration from a localStorage snapshot that pre-dates the
      // per-token persistence).  Server-shipped node tokens win over
      // the in-memory cache — they're authoritative and include the
      // ``probes`` + ``per_layer_scores`` the cache may not have.
      if ((prev.tokens?.length ?? 0) === 0) {
        const fromNode = nodeToTurn(node);
        if (fromNode.tokens || fromNode.thinkingTokens) {
          prev.tokens = fromNode.tokens;
          prev.thinkingTokens = fromNode.thinkingTokens;
        } else {
          const cached = tokenScoreCache.get(nid);
          if (cached) {
            prev.tokens = cached.tokens;
            prev.thinkingTokens = cached.thinkingTokens;
          }
        }
      }
      turn = prev;
    } else {
      turn = nodeToTurn(node);
      // Cache fallback for branch nav across nodes the server hasn't
      // serialized with tokens (legacy / transcript-loaded).
      if (!turn.tokens && !turn.thinkingTokens) {
        const cached = tokenScoreCache.get(nid);
        if (cached) {
          turn.tokens = cached.tokens;
          turn.thinkingTokens = cached.thinkingTokens;
        }
      }
    }
    if (loomTree.pendingNodeId === nid) pendingIdx = out.length;
    out.push(turn);
  }
  chatLog.turns = out;
  chatLog.pendingIndex = pendingIdx;
}

/** Replace the in-memory tree with a freshly fetched server snapshot.
 *  Drops stale per-node entries; resets activePath; sync chat log.
 *
 *  ``snap.nodes`` is a flat list (server's ``LoomTree.to_dict`` shape).
 *  We pivot to id-keyed for the in-memory cache.  Tolerant of older or
 *  alternative shapes that pass a record keyed by id — phase-2 server
 *  is the list shape; v1 migration in this file produces the dict shape
 *  on disk for legacy snapshots, which is also accepted. */
function applyTreeSnapshot(snap: LoomTreeJSON): void {
  loomTree.root_id = snap.root_id;
  loomTree.active_node_id = snap.active_node_id;
  loomTree.rev = snap.rev;
  loomTree.modelId = snap.model_id ?? loomTree.modelId;
  loomTree.unavailable = false;
  loomTree.error = null;
  loomTree.nodes.clear();
  if (Array.isArray(snap.nodes)) {
    for (const n of snap.nodes) loomTree.nodes.set(n.id, n);
  } else if (snap.nodes && typeof snap.nodes === "object") {
    for (const [id, n] of Object.entries(snap.nodes as Record<string, LoomNodeJSON>)) {
      loomTree.nodes.set(id, n);
    }
  }
  loomTree.children_of.clear();
  for (const [pid, ids] of Object.entries(snap.children_of)) {
    loomTree.children_of.set(pid, [...ids]);
  }
  recomputeActivePath();
  syncChatLogFromTree();
}

/** Apply a ``tree_mutated`` delta in place.  Returns ``false`` if the
 *  client missed a rev — caller full-refetches on false.
 *
 *  Phase-2 server semantics: ``updated`` carries full LoomNodeJSON
 *  objects (potentially with an extra ``children`` field that we
 *  ignore — children_of is rebuilt from the added/removed deltas).
 *  ``added`` nodes may also be implicit children-list extensions of
 *  existing parents. */
function applyTreeDelta(ev: {
  added?: LoomNodeJSON[];
  removed?: string[];
  updated?: LoomNodeJSON[];
  active_node_id?: string | null;
  rev: number;
}): boolean {
  // First event after bootstrap is the rev=1 mutation; accept rev > 0
  // when our local rev is 0 (cold start) without claiming a gap.
  if (loomTree.rev > 0 && ev.rev > loomTree.rev + 1) return false;
  // ``added``: inject node + extend its parent's children list.  Node
  // payloads from the server may include a ``children`` field
  // (_node_json adds it); strip before storing so the cached node
  // shape stays consistent with the bootstrap fetch.
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
  // ``updated``: full node replacement.  Same children-strip as added.
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
  loomTree.rev = ev.rev;
  // Phase 5: applied_steering strings can shift after edit/regen, so
  // bust the edge-label cache wholesale on any mutation.  Cheap — the
  // sidebar refetches lazily on first re-render.
  invalidateEdgeLabels();
  recomputeActivePath();
  syncChatLogFromTree();
  return true;
}

/** Bootstrap fetch of the tree.  Tolerates a 404 — older servers don't
 *  ship loom; the UI falls back to legacy linear-path rendering. */
export async function refreshLoomTree(): Promise<void> {
  try {
    const snap = await apiTree.get();
    applyTreeSnapshot(snap);
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      // Server pre-loom — disable the sidebar quietly.
      loomTree.unavailable = true;
      loomTree.rev = 0;
      return;
    }
    loomTree.error = e instanceof Error ? e.message : String(e);
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
): Promise<string | null> {
  try {
    const r = await apiTree.branch(node_id, text);
    await refreshLoomTree();
    return r.node_id;
  } catch (e) {
    _captureLoomError("branch", e);
    return null;
  }
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

/** Regenerate the active assistant: send a fresh ``generate`` request
 *  anchored at the user-parent's parent, so the replayed user prompt
 *  dedups onto the existing user node and creates a sibling assistant.
 *  N=1 by default.  Recipe is implicit (current rack) unless
 *  ``opts.recipe_override`` is set, in which case the engine applies
 *  the recipe-override modifier on top of the parent's recipe. */
export async function loomRegenerateActive(
  n: number = 1,
  opts: { recipe_override?: string | null } = {},
): Promise<void> {
  if (loomTree.rev <= 0) return;
  const activeId = loomTree.active_node_id;
  if (!activeId) return;
  const node = loomTree.nodes.get(activeId);
  if (!node || node.role !== "assistant") return;
  const parentId = node.parent_id;
  if (!parentId) return;
  // The user turn (parent) carries the prompt text we need to replay.
  const parent = loomTree.nodes.get(parentId);
  if (!parent || parent.role !== "user") return;
  try {
    await sendGenerate(parent.text, {
      parent_node_id: parent.parent_id ?? null,
      n,
      recipe_override: opts.recipe_override ?? undefined,
    });
  } catch (e) {
    _captureLoomError("regenerate", e);
  }
}

/** Regenerate under a specific user node (the "fan out" entry point).
 *  Anchor at the user's parent so ``add_user_turn`` reuses that user
 *  node and fans out sibling assistant replies. */
export async function loomRegenerateFromUser(
  userNodeId: string,
  opts: { n?: number; recipe_override?: string | null } = {},
): Promise<void> {
  if (loomTree.rev <= 0) return;
  const user = loomTree.nodes.get(userNodeId);
  if (!user || user.role !== "user") return;
  try {
    await sendGenerate(user.text, {
      parent_node_id: user.parent_id ?? null,
      n: opts.n ?? 1,
      recipe_override: opts.recipe_override ?? undefined,
    });
  } catch (e) {
    _captureLoomError("regenerate", e);
  }
}

export interface HighlightState {
  /** Probe name selected for primary tinting.  ``null`` disables
   * highlighting entirely (token backgrounds render transparent). */
  target: string | null;
  /** Probe name for the second stripe in compare-two mode.  Ignored
   * when ``compareTwo`` is false. */
  compareTarget: string | null;
  compareTwo: boolean;
  /** Smooth-blend the two stripes instead of a hard 50% boundary.
   * Pure aesthetic; off by default. */
  smoothBlend: boolean;
}

export const highlightState: HighlightState = $state({
  // Surprise mode by default — the logit-pass tint works without any
  // probes loaded, so it's the one mode that's meaningful out of the
  // box.  Mirrors the TUI's ``_highlight_probe = SURPRISE_PROBE`` +
  // ``_highlighting = True`` init.  ``localStorage`` overrides per
  // model on hydrate, so a user who flipped to a probe + reloaded
  // still sees their last choice.
  target: SURPRISE_TARGET,
  compareTarget: null,
  compareTwo: false,
  smoothBlend: false,
});

export function setHighlightTarget(name: string | null): void {
  highlightState.target = name;
}

export function setCompareTarget(name: string | null): void {
  highlightState.compareTarget = name;
}

export function toggleCompareTwo(): void {
  highlightState.compareTwo = !highlightState.compareTwo;
}

// =========================================== live token / gen status ====

/** Captures the in-flight generation's per-token scores so the chat
 * renderer can highlight live before the WS ``done`` event lands.  Reset
 * on each ``started``. */
export interface LiveTokenStream {
  responseTokens: TokenScore[];
  thinkingTokens: TokenScore[];
}

export const liveTokenStream: LiveTokenStream = $state({
  responseTokens: [],
  thinkingTokens: [],
});

export const genStatus: GenStatus = $state({
  active: false,
  tokensSoFar: 0,
  maxTokens: 0,
  startedAt: null,
  tokPerSec: 0,
  ppl: { logSum: 0, count: 0, mean: null },
  finishReason: null,
});

/** Geometric-mean perplexity assembled from per-token TokenEvent.perplexity
 * values (mirrors the TUI's ``exp(sum(log(ppl)) / count)`` formula).  Pure
 * function — caller passes the slice so it can also be used on ad-hoc
 * accumulators (e.g. an A/B side's separate perplexity buffer). */
export function geometricMeanPpl(state: GenStatus): number | null {
  if (state.ppl.count <= 0) return null;
  return Math.exp(state.ppl.logSum / state.ppl.count);
}

// =========================================== sampling / system prompt ===

export interface SamplingState {
  temperature: number | null;
  top_p: number | null;
  top_k: number | null;
  max_tokens: number;
  /** ``null`` = use the WS default (no seed sent).  Numeric value pinned. */
  seed: number | null;
  system_prompt: string;
  /** One per line stop sequences; parsed into SamplingConfig.stop. */
  stop_sequences: string;
  /** Raw logit bias map. Accepts JSON {"123": -4} or lines "123: -4". */
  logit_bias_text: string;
  presence_penalty: number;
  frequency_penalty: number;
  /** ``null`` = auto, true/false = explicit override. */
  thinking: boolean | null;
  /** Logit-pass: top-K alternatives to capture per token (``0`` = off,
   *  matches the engine's chosen-only mode).  When ``> 0`` the WS ``token``
   *  event carries ``top_alts`` and the drilldown's logits tab + the
   *  inline ``surprise`` highlight mode populate.  Flipped via the
   *  "show alts" toggle in ``SamplingStrip``; the canonical "on" value
   *  is ``8`` per Decision 1 of ``docs/plans/logit-pass.md``. */
  return_top_k: number;
  /** Per-message role-substitution labels (roleplay scaffold).  Sticky
   *  client state like ``seed`` — whatever's in the boxes rides the next
   *  send and is stamped onto that turn's loom node (immutable afterward).
   *  Empty string = standard role label (nothing sent). */
  user_role: string;
  assistant_role: string;
}

export const samplingState: SamplingState = $state({
  temperature: null,
  top_p: null,
  top_k: null,
  max_tokens: 256,
  seed: null,
  system_prompt: "",
  stop_sequences: "",
  logit_bias_text: "",
  presence_penalty: 0,
  frequency_penalty: 0,
  user_role: "",
  assistant_role: "",
  // Initial thinking state: explicit ``false`` so an unchecked checkbox
  // on first paint actually sends ``thinking: false`` to the server.
  // The previous ``null`` (auto) state silently fell through to whatever
  // the model template defaults to — for thinking-capable templates that
  // meant the model thought even though the box was visually off.
  thinking: false,
  // Logit-pass: top-K alternatives on by default — the drilldown logits
  // tab and the inline surprise highlight want them.  The SamplingStrip's
  // "alts" toggle flips this between 0 and 8 (Decision 1 in
  // docs/plans/logit-pass.md).
  return_top_k: 8,
});

export function setSampling<K extends keyof SamplingState>(
  key: K,
  value: SamplingState[K],
): void {
  samplingState[key] = value;
}

// ============================================ generation UI mode ========
//
// Base (non-chat) models have no chat template — the engine handles
// them as flat completion.  ``genUiMode`` decides whether the chat panel
// renders bubbles + roles (chat) or a single flat completion buffer
// (raw).  It is a plain two-state toggle: the default is seeded from the
// model's ``is_base_model`` flag (base → raw, chat → chat) the first
// time a model is seen, then the user's explicit choice is persisted
// per ``model_id`` and survives reloads.

export interface GenUiModeState {
  /** Which surface the chat panel renders — ``"chat"`` (bubbles +
   *  roles) or ``"raw"`` (a single flat completion buffer). */
  mode: "chat" | "raw";
}

export const genUiMode: GenUiModeState = $state({ mode: "chat" });

/** Resolve the effective rendering mode — true means flat raw buffer. */
export function effectiveRawMode(): boolean {
  return genUiMode.mode === "raw";
}

const GENUI_KEY_PREFIX = "saklas.genui.v1.";

function genUiKey(): string | null {
  const id = sessionState.info?.model_id;
  return id ? GENUI_KEY_PREFIX + id : null;
}

/** Load the per-model render mode.  Called from ``bootstrap`` once the
 *  model id is known.  A stored preference wins; with none, the mode is
 *  seeded from the model's nature — a base model defaults to ``raw``, a
 *  chat model to ``chat``. */
function loadGenUiMode(): void {
  const key = genUiKey();
  const stored = key ? safeLocalStorageGet(key) : null;
  if (stored === "chat" || stored === "raw") {
    genUiMode.mode = stored;
  } else {
    genUiMode.mode =
      sessionState.info?.is_base_model === true ? "raw" : "chat";
  }
}

/** Set (and persist) the render mode.  Toggling mode never mutates the
 *  loom tree — only generation does. */
export function setGenUiMode(mode: "chat" | "raw"): void {
  genUiMode.mode = mode;
  const key = genUiKey();
  if (key) safeLocalStorageSet(key, mode);
}

function parsedStopSequences(): string[] | null {
  const lines = samplingState.stop_sequences
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  return lines.length > 0 ? lines : null;
}

function parsedLogitBias(): Record<string, number> | null {
  const raw = samplingState.logit_bias_text.trim();
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const out: Record<string, number> = {};
      for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
        const n = Number(v);
        if (Number.isFinite(n)) out[String(Number(k))] = n;
      }
      return Object.keys(out).length > 0 ? out : null;
    }
  } catch {
    /* fall through to line parser */
  }
  const out: Record<string, number> = {};
  for (const line of raw.split(/\r?\n/)) {
    const m = line.match(/^\s*(-?\d+)\s*[:=,\s]\s*(-?\d+(?:\.\d+)?)\s*$/);
    if (!m) continue;
    out[String(Number(m[1]))] = Number(m[2]);
  }
  return Object.keys(out).length > 0 ? out : null;
}

function nonDefaultSamplingOverrides(): Partial<WSSampling> {
  const stop = parsedStopSequences();
  const logit_bias = parsedLogitBias();
  return {
    ...(stop ? { stop } : {}),
    ...(logit_bias ? { logit_bias } : {}),
    ...(samplingState.presence_penalty !== 0
      ? { presence_penalty: samplingState.presence_penalty }
      : {}),
    ...(samplingState.frequency_penalty !== 0
      ? { frequency_penalty: samplingState.frequency_penalty }
      : {}),
    ...(samplingState.return_top_k > 0
      ? { return_top_k: samplingState.return_top_k }
      : {}),
    // Per-message role labels (roleplay scaffold) ride every send like
    // ``seed`` — trimmed.  Empty = standard role, omitted.  A value equal to
    // the family's standard label (the box's seeded default) is *also* a
    // no-op, so we omit it too: the node isn't stamped with a redundant label
    // and the bubble keeps its structural heading.  Only a genuine override
    // (a label the user changed away from the default) is sent.
    ...roleOverride(samplingState.user_role, sessionState.info?.default_user_role, "user_role"),
    ...roleOverride(
      samplingState.assistant_role,
      sessionState.info?.default_assistant_role,
      "assistant_role",
    ),
  };
}

/** ``{key: value}`` when ``raw`` is a non-empty label that differs from the
 *  family default, else ``{}`` (treated as "use the standard role"). */
function roleOverride(
  raw: string,
  fallback: string | null | undefined,
  key: "user_role" | "assistant_role",
): Partial<WSSampling> {
  const value = raw.trim();
  if (!value || value === fallback) return {};
  return { [key]: value };
}

function buildSamplingPayload(): WSSampling | null {
  // temperature / top-p / top-k / max-tokens / thinking are PATCHed to
  // the session as the user edits them, so they ride the server's own
  // defaults and aren't echoed here.  Seed has no PATCH path and the
  // advanced extras (penalties, stop, logit-bias, return_top_k) aren't
  // PATCH-able either — both always ride per-call.
  const payload: WSSampling = {
    persist_per_layer_scores: true,
    ...nonDefaultSamplingOverrides(),
    ...(samplingState.seed !== null ? { seed: samplingState.seed } : {}),
  };
  return Object.keys(payload).length > 0 ? payload : null;
}

// ===================================================== pending actions ===

export interface PendingActionsState {
  /** Queue of mutations and submissions deferred while a generation is
   * running.  Drained one item per WS ``done`` event by
   * :func:`drainNextPendingAction`.  Per-item cancel goes through
   * :func:`cancelPendingAction` (the GUI's per-bubble ``×``). */
  queue: PendingAction[];
}

export const pendingActions: PendingActionsState = $state({ queue: [] });

let _pendingCounter = 0;

/** Append a pending action.  ``replaceSlot`` slots a re-edited
 *  pulled item back into its original position rather than the queue
 *  tail; out-of-range values fall back to append. */
export function enqueuePending(
  action: Omit<PendingAction, "id" | "createdAt">,
  opts: { replaceSlot?: number | null } = {},
): void {
  const item: PendingAction = {
    ...action,
    id: `pa-${_pendingCounter++}`,
    createdAt: Date.now(),
  };
  const slot = opts.replaceSlot ?? null;
  if (slot !== null && slot >= 0 && slot < pendingActions.queue.length) {
    pendingActions.queue[slot] = item;
  } else {
    pendingActions.queue.push(item);
  }
}

/** Drain a single pending item.  Called from the WS ``done`` /
 *  ``error`` handlers; an ``awaitsGen=false`` item chains into the
 *  next drain immediately so a sequence of instant mutations
 *  (clear, rewind) doesn't stall waiting for a gen that's never
 *  going to fire.
 *
 *  Replaces the v1.x ``applyPendingActions`` which drained *all*
 *  items on every ``done`` — the v2.x queue semantics serialize
 *  one item per ``done`` so a send-then-send pair runs in order
 *  rather than racing the WS. */
export async function drainNextPendingAction(): Promise<void> {
  if (pendingActions.queue.length === 0) return;
  // Reconcile the input-history pull state before mutating the queue:
  // a drained head whose slot the user is editing would otherwise
  // leave a dangling ``pulledSlot`` pointing past the array.  The
  // shift helper returns the stash text when slot 0 was pulled; we
  // park it on ``inputRestore`` so Chat.svelte's $effect copies it
  // back into the textarea on the next tick.
  const restore = onPendingQueueShift();
  if (restore !== null) requestInputRestore(restore);
  const item = pendingActions.queue.shift();
  if (!item) return;
  try {
    await item.apply();
  } catch (e) {
    chatLog.turns = [
      ...chatLog.turns,
      {
        role: "system",
        text: `pending ${item.label} failed: ${String(e)}`,
      },
    ];
  }
  if (!item.awaitsGen) {
    // Instant mutation finished — chain into the next item so the
    // queue doesn't stall waiting for a ``done`` that won't fire.
    await drainNextPendingAction();
  }
}

/** Remove one pending item by id (GUI per-bubble ``×``). */
export function cancelPendingAction(id: string): void {
  pendingActions.queue = pendingActions.queue.filter((p) => p.id !== id);
}

export function discardPendingActions(): void {
  pendingActions.queue = [];
}

/** Apply immediately if no gen is in flight AND the queue is empty;
 *  queue otherwise.  The queue check matters: with one or more items
 *  already pending, applying a fresh rack mutation immediately would
 *  break FIFO ordering of state mutations.
 *
 *  Used by the rack/sampling mutations — they don't kick off a gen, so
 *  ``awaitsGen=false`` lets the drain chain through them without
 *  waiting on the next ``done``.
 *
 *  Consecutive queued rack mutations *coalesce*: when the queue tail is
 *  already a rack-mutation item, the fresh ``apply`` is chained onto it
 *  rather than appended as a new slot, and the bubble's label updates
 *  to the latest action.  A slider drag that fires 30+ intermediate
 *  ``setVectorAlpha`` calls mid-gen therefore leaves a single queued
 *  bubble carrying the net effect — "one final steering adjustment" —
 *  instead of 30 stacked ghosts.  Coalescing stops at any non-rack
 *  item (send / commit / one-shot mutation): rack changes before and
 *  after a queued send form distinct groups so FIFO ordering relative
 *  to the send is preserved. */
const RACK_COALESCE_KEY = "rack";
function enqueueOrApply(label: string, apply: () => void): void {
  if (!(genStatus.active || pendingActions.queue.length > 0)) {
    apply();
    return;
  }
  const q = pendingActions.queue;
  const tail = q[q.length - 1];
  if (tail && tail.coalesceKey === RACK_COALESCE_KEY) {
    // Chain onto the trailing rack bubble.  Reassign the slot so the
    // $state array fires reactivity for PendingBubbles / the queue
    // count badge (in-place field writes on a proxied item are tracked,
    // but reassigning is the uniform pattern used across this module).
    const prev = tail.apply;
    q[q.length - 1] = {
      ...tail,
      label,
      apply: async () => {
        await prev();
        await apply();
      },
    };
    return;
  }
  enqueuePending({
    label, text: null, apply, awaitsGen: false, rebuild: null,
    coalesceKey: RACK_COALESCE_KEY,
  });
}

/** Are we busy enough that fresh submissions should queue instead of
 *  fire?  Mirrors the TUI's ``_is_busy`` — gen running OR earlier
 *  items waiting their turn.  Used by every submission helper below. */
export function isPendingBusy(): boolean {
  return genStatus.active || pendingActions.queue.length > 0;
}

/** Predict the post-queue active-node-is-user-role flag.  Walks the
 *  queue tail-first and returns the first non-null ``endsOnUserNode``;
 *  returns ``null`` when no queued item changes the role (e.g. the
 *  queue is empty or only carries rack mutations).  Drives the chat
 *  input's role-aware placeholder + send-button label so a queued
 *  ``commit user`` flips the next submission into prefill /
 *  commit-assistant mode without waiting for the queue to drain. */
export function predictedQueueEndOnUserNode(): boolean | null {
  for (let i = pendingActions.queue.length - 1; i >= 0; i--) {
    const e = pendingActions.queue[i].endsOnUserNode;
    if (e !== undefined && e !== null) return e;
  }
  return null;
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
  wsConn.ready = new Promise<void>((resolve, reject) => {
    socket.addEventListener("open", () => resolve(), { once: true });
    socket.addEventListener("error", (e) => reject(e), { once: true });
  });
  socket.addEventListener("message", (ev: MessageEvent) => {
    let msg: WSServerMessage;
    try {
      msg = JSON.parse(ev.data) as WSServerMessage;
    } catch {
      return;
    }
    handleWsMessage(msg);
    for (const cb of wsConn.listeners) {
      try {
        cb(msg);
      } catch {
        /* ignore subscriber failures */
      }
    }
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
 * The server cannot always include the assistant node id on ``started``:
 * the node is created inside generation.  ``node_created`` and ``token``
 * events do carry it, so this reconciles the active path lazily and
 * ensures the next token has a writable ChatTurn. */
function adoptStreamingNode(nodeId: string | null | undefined): void {
  if (!nodeId || abState.processingAb || loomTree.rev <= 0) return;
  loomTree.pendingNodeId = nodeId;
  if (!loomTree.nodes.has(nodeId)) {
    const active = loomTree.active_node_id;
    const activeNode = active ? loomTree.nodes.get(active) : null;
    const parentId =
      activeNode?.role === "user" ? active : (activeNode?.parent_id ?? active ?? null);
    loomTree.nodes.set(nodeId, {
      id: nodeId,
      parent_id: parentId,
      role: "assistant",
      text: "",
    });
    attachChild(parentId, nodeId);
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
      // Apply the delta; on rev gap, full re-fetch.
      const ok = applyTreeDelta(msg);
      if (!ok) void refreshLoomTree();
      return;
    }
    case "node_created": {
      // Pre-allocate the node so n-way regen render slots exist before
      // token events tagged with this node_id arrive.  The full node body
      // lands via a ``tree_mutated`` (added) event — which the server's
      // forwarder actually enqueues *before* this ``node_created``.  So
      // merge over any node already in hand rather than replacing it: a
      // bare ``node_created`` payload omits ``role_label`` (among other
      // fields), and ``upsertLoomNode`` is a full replace — a blind
      // overwrite would wipe a custom role glyph back to the structural
      // letter (U/A) until the next full-tree refetch (e.g. on click).
      const existing = loomTree.nodes.get(msg.node_id);
      upsertLoomNode({
        ...existing,
        id: msg.node_id,
        parent_id: msg.parent_id,
        role: msg.role,
        text: existing?.text ?? "",
      });
      if (msg.role === "assistant" && genStatus.active && !abState.processingAb) {
        adoptStreamingNode(msg.node_id);
      } else if (loomTree.rev > 0 && loomTree.active_node_id === msg.node_id) {
        recomputeActivePath();
        syncChatLogFromTree();
      }
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
        // A/B shadow run: attach a fresh assistant abPair to the steered
        // turn that just finished.  Don't append a new top-level turn —
        // the chat panel renders the abPair in its own column.
        const steered = chatLog.turns[abState.pendingTurnIdx];
        if (steered) {
          steered.abPair = {
            role: "assistant",
            text: "",
            tokens: [],
            thinkingTokens: [],
          };
        }
        // pendingIndex points at the steered turn so the streaming
        // pulse on Chat.svelte still highlights "this turn is live".
        chatLog.pendingIndex = abState.pendingTurnIdx;
      } else if (loomTree.rev > 0 && msg.node_id) {
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
      } else if (loomTree.rev > 0) {
        // Loom path with a lazily-created assistant node: wait for
        // ``node_created`` or the first tagged ``token`` before allocating
        // the assistant turn.  Appending a legacy placeholder here creates
        // a duplicate local assistant and is the source of many branch /
        // highlight misroutes.
        chatLog.pendingIndex = null;
        syncChatLogFromTree();
      } else {
        // Normal (pre-loom) run: append a fresh assistant turn so
        // streamed tokens have a home.
        chatLog.turns = [
          ...chatLog.turns,
          { role: "assistant", text: "", tokens: [], thinkingTokens: [] },
        ];
        chatLog.pendingIndex = chatLog.turns.length - 1;
      }
      return;
    }
    case "token": {
      adoptStreamingNode(msg.node_id);
      genStatus.tokensSoFar += 1;
      if (genStatus.startedAt) {
        const elapsed = (performance.now() - genStatus.startedAt) / 1000;
        if (elapsed > 0) genStatus.tokPerSec = genStatus.tokensSoFar / elapsed;
      }
      const tokenScore: TokenScore = {
        text: msg.text,
        thinking: msg.thinking,
        tokenId: msg.token_id,
        perLayerScores: msg.per_layer_scores,
        // ``msg.scores`` is the magnitude-weighted aggregate probe row —
        // the value the TUI tints live tokens with.  Using it (rather
        // than a single deepest-layer slice) makes the webui's live
        // highlight match both the TUI and the post-generation projected
        // pass.  Absent when no probes are loaded.
        probes: msg.scores,
        // Logit-pass: pipe chosen-token logprob + top-K alternatives onto
        // the per-token row.  Both ride the WS ``token`` event directly
        // from Phase 1's engine capture; absent when ``return_top_k == 0``
        // and no other on_token consumer is live (legacy default).
        logprob: msg.logprob ?? null,
        topAlts: msg.top_alts ?? null,
        // Raw decode-step index — the join key the logit fork slices
        // ``raw_token_ids`` on.  Rides the WS ``token`` event directly.
        rawIndex: msg.raw_index ?? null,
      };
      // Seed the single-probe ``score`` for the selected highlight so the
      // inline tint paints immediately as the token streams in.  The
      // canonical projected scores overwrite this on ``done``.
      if (msg.scores && highlightState.target) {
        const s = msg.scores[highlightState.target];
        if (typeof s === "number") tokenScore.score = s;
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
      // Probe rack — unified per-token readings.  Every probe shape rides
      // the one ``probe_readings`` channel (a 2-node concept axis is the
      // rank-1 case); the field is omitted when no probe is attached, so the
      // helper no-ops on undefined.  Skip shadow runs so the rack stays
      // anchored to the steered branch.  ``msg.scores`` / ``per_layer_scores``
      // above still feed highlight tinting + the token-drilldown heatmap.
      if (!abState.processingAb) {
        updateProbesFromReadings(msg.probe_readings);
      }
      return;
    }
    case "done": {
      adoptStreamingNode(msg.node_id);
      genStatus.active = false;
      genStatus.finishReason = msg.result?.finish_reason ?? "stop";
      // Probe rack — end-of-gen aggregate (the settled ``ProbeReading`` per
      // probe: coords / fraction / nearest / residual + per-layer traces).
      // Same omitted-when-absent rule.
      if (!abState.processingAb) {
        setProbeAggregates(msg.result?.probe_readings);
      }
      const perToken = msg.result?.per_token_probes ?? [];
      const turn = _currentWriteTurn();
      if (turn?.tokens && perToken.length) {
        // Server emits per_token_probes in token order over the full
        // generated stream; thinking + response tokens share that order.
        // Walk the union and partition by ``thinking`` flag from the
        // local token rows so we preserve the live separation.
        let idx = 0;
        for (const row of turn.thinkingTokens ?? []) {
          if (idx < perToken.length) {
            const probes = perToken[idx].probes;
            row.probes = probes;
            if (highlightState.target) {
              row.score = probes[highlightState.target];
            }
          }
          idx++;
        }
        for (const row of turn.tokens ?? []) {
          if (idx < perToken.length) {
            const probes = perToken[idx].probes;
            row.probes = probes;
            if (highlightState.target) {
              row.score = probes[highlightState.target];
            }
          }
          idx++;
        }
      }
      if (turn) {
        turn.finishReason = msg.result?.finish_reason ?? "stop";
        turn.tokensSoFar = msg.result?.tokens ?? genStatus.tokensSoFar;
        // Logit-pass: per-turn mean chosen-token logprob (response span
        // only).  Null when capture wasn't live; the inline surprise
        // mode + loom edge-weighting null-guard on this directly.
        turn.meanLogprob = msg.result?.mean_logprob ?? null;
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
        if (loomTree.rev > 0) syncChatLogFromTree();
      }

      if (wasShadow) {
        // Shadow gen done — clear the A/B routing flags.  Do NOT touch
        // the probe baseline or correlation refresh; the steered turn
        // already did that when it finished.
        abState.processingAb = false;
        abState.pendingTurnIdx = null;
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

      // v2.3: the legacy standalone A/B toggle is gone — auto-regen with
      // ``mode === "unsteered"`` *is* the A/B shadow.  Branch on the
      // resolved recipe-override:
      //
      //   * ``"unsteered"`` → fire the shadow-replay path
      //     (``_sendShadowGenerate``).  Tokens land on the steered turn's
      //     ``abPair`` so the chat's right column renders them in place.
      //     Bit-identical to the pre-v2.3 A/B behaviour.
      //
      //   * any other override → fire a loom regen with the override.
      //     The engine drops the result as a sibling under the same
      //     user-parent; pin it so the chat's right column picks it up.
      if (autoRegenState.enabled) {
        const override = currentRecipeOverride();
        if (
          override === "unsteered" &&
          steeredIdx !== null &&
          chatLog.turns[steeredIdx]?.role === "assistant"
        ) {
          void _sendShadowGenerate(steeredIdx);
        } else if (
          override !== null &&
          loomTree.rev > 0 &&
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
        if (loomTree.rev > 0) syncChatLogFromTree();
      }
      abState.processingAb = false;
      abState.pendingTurnIdx = null;
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

/** Build a :class:`PendingAction` for a queued chat send.  The
 *  ``rebuild`` factory preserves ``opts`` across an ↑-pull-and-edit so
 *  the slot's parent_node_id / steering override / n stay attached.
 *  ``endsOnUserNode=false`` — a send lands a user turn then an assistant
 *  reply, so the post-drain active node is always assistant. */
function buildSendPending(
  text: string, opts: SendGenerateOpts,
): PendingAction {
  return {
    id: `pa-${_pendingCounter++}`,
    label: "send",
    text,
    apply: () => sendGenerateNow(text, opts),
    awaitsGen: true,
    rebuild: (newText: string) => buildSendPending(newText, opts),
    createdAt: Date.now(),
    endsOnUserNode: false,
  };
}

/** Send a generate request over the WS.  Builds the steering expression
 * from the rack live, layers the SamplingConfig overrides when one-shot
 * mode is on, and routes everything through the singleton connection.
 *
 * When a gen is in flight (or earlier items are queued) the request
 * lands on the pending queue and waits for FIFO drain — the in-flight
 * gen is *not* interrupted.  ``replaceSlot`` keeps a pulled-and-edited
 * item at its original slot. */
export async function sendGenerate(
  input: string | unknown,
  opts: SendGenerateOpts & { replaceSlot?: number | null } = {},
): Promise<void> {
  // Strings route through the pending queue when busy.  Non-string
  // ``input`` (the A/B shadow path's messages array) always fires
  // immediately — it's an internal store-to-store call that doesn't
  // come from a user gesture and can't be pulled or re-edited.
  if (typeof input === "string" && isPendingBusy()) {
    const { replaceSlot, ...sendOpts } = opts;
    const item = buildSendPending(input, sendOpts);
    enqueuePending(
      {
        label: item.label,
        text: item.text,
        apply: item.apply,
        awaitsGen: item.awaitsGen,
        rebuild: item.rebuild,
      },
      { replaceSlot: replaceSlot ?? null },
    );
    return;
  }
  return sendGenerateNow(input, opts);
}

/** Immediate-fire core for ``sendGenerate``.  Bypasses the queue
 *  check — called by ``sendGenerate`` itself when not busy and by
 *  ``drainNextPendingAction`` via the queued item's ``apply``. */
async function sendGenerateNow(
  input: string | unknown,
  opts: SendGenerateOpts = {},
): Promise<void> {
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
  // Push the user turn so the UI has something to render before the WS
  // started event lands.  Skip the optimistic push when the server owns
  // the tree (it will emit ``tree_mutated`` with the added user node
  // and we'll sync from there) or when ``input`` is a messages list
  // (A/B shadow path — no fresh user turn to display).
  if (loomTree.rev <= 0 && typeof input === "string") {
    chatLog.turns = [...chatLog.turns, { role: "user", text: input }];
  }
  // Remember the input verbatim so the auto-regen shadow path can replay
  // it as an unsteered run.  Only meaningful when auto-regen is on with
  // ``mode === "unsteered"``; otherwise it's dead weight that's free to
  // keep up to date.
  const payload: WSClientMessage = {
    type: "generate",
    input,
    steering: steeringPayload,
    sampling,
    // Coerce ``null`` (legacy "auto") to explicit ``false`` so the
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
 *  new sibling lands via the WS ``node_created`` / ``token`` / ``done``
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

function buildPrefillPending(
  nodeId: string | "active@drain",
  text: string,
  opts: { n?: number },
): PendingAction {
  return {
    id: `pa-${_pendingCounter++}`,
    label: "prefill",
    text,
    apply: () => {
      // Deferred resolution: when the parent user node was itself queued
      // (queue-role-aware dispatch), the literal id at enqueue time was
      // a sentinel — read the active node fresh at drain time, by which
      // point the previous drained action has landed the right user node.
      const parent = nodeId === "active@drain"
        ? loomTree.active_node_id
        : nodeId;
      if (!parent) return;
      return sendPrefillNow(parent, text, opts);
    },
    awaitsGen: true,
    rebuild: (newText: string) => buildPrefillPending(nodeId, newText, opts),
    createdAt: Date.now(),
    endsOnUserNode: false,
  };
}

/** Answer-prefill — seed an assistant reply under a user node.  The
 *  server tokenizes ``text`` into a forced decode prefix, emits it as
 *  the opening of the assistant turn, then samples the continuation.
 *  The prefilled sibling streams in via the WS ``node_created`` /
 *  ``token`` / ``done`` events and becomes the active branch.  Steering
 *  and sampling ride from the current rack exactly like a normal
 *  ``sendGenerate``; ``thinking`` is forced off server-side (the text is
 *  the start of the answer, not a thought).
 *
 *  Queues behind in-flight gens / earlier pending items; the in-flight
 *  gen is not interrupted. */
export async function sendPrefill(
  nodeId: string | "active@drain",
  text: string,
  opts: { n?: number; replaceSlot?: number | null } = {},
): Promise<void> {
  if (isPendingBusy()) {
    const { replaceSlot, ...prefillOpts } = opts;
    const item = buildPrefillPending(nodeId, text, prefillOpts);
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
  // Immediate fire: resolve the sentinel against the live active node;
  // an empty resolution is a no-op (nothing to anchor under).
  const resolved = nodeId === "active@drain"
    ? loomTree.active_node_id
    : nodeId;
  if (!resolved) return;
  return sendPrefillNow(resolved, text, opts);
}

async function sendPrefillNow(
  nodeId: string,
  text: string,
  opts: { n?: number } = {},
): Promise<void> {
  const sock = await ensureWebSocket();
  const steering = currentSteeringExpression();
  const sampling = buildSamplingPayload();
  genStatus.maxTokens = sampling?.max_tokens ?? samplingState.max_tokens;
  const payload: WSClientMessage = {
    type: "generate",
    prefill_node_id: nodeId,
    prefill_text: text,
    steering: steering || null,
    sampling,
    ...(opts.n !== undefined ? { n: opts.n } : {}),
  };
  const send = () => sock.send(JSON.stringify(payload));
  if (sock.readyState === WebSocket.OPEN) send();
  else sock.addEventListener("open", send, { once: true });
}

function buildCommitPending(
  role: "user" | "assistant",
  parentNodeId: string | null | "active@drain",
  text: string,
  raw: boolean = false,
): PendingAction {
  return {
    id: `pa-${_pendingCounter++}`,
    label: role === "assistant" ? "commit assistant" : "commit user",
    text,
    apply: () => {
      // Deferred resolution for queue-role-aware dispatch: a commit
      // queued behind another that creates its own user node needs to
      // hang under that not-yet-existing node — read the active node
      // fresh at drain time.
      const parent = parentNodeId === "active@drain"
        ? loomTree.active_node_id
        : parentNodeId;
      return sendCommitNow(role, parent, text, raw);
    },
    awaitsGen: true,
    rebuild: (newText: string) =>
      buildCommitPending(role, parentNodeId, newText, raw),
    createdAt: Date.now(),
    endsOnUserNode: role === "user",
  };
}

/** Commit — land a turn without generating.  ``role`` decides which
 *  session method routes: ``"user"`` for ``append_user_turn`` (called
 *  on an assistant/root active node — ``parentNodeId`` is that node, or
 *  null to fall through to the active node server-side); ``"assistant"``
 *  for ``append_assistant_turn`` (``parentNodeId`` is the user node the
 *  authored turn hangs off — required).  The server emits a single
 *  ``done`` event with the new node id; the loom's ``node_created`` /
 *  ``tree_mutated`` subscriptions land the node in the UI.  No token
 *  streaming, no steering, no sampling — just a tree mutation.
 *
 *  Queues behind in-flight gens / earlier pending items. */
export async function sendCommit(
  role: "user" | "assistant",
  parentNodeId: string | null | "active@drain",
  text: string,
  opts: { replaceSlot?: number | null; raw?: boolean } = {},
): Promise<void> {
  if (isPendingBusy()) {
    const item = buildCommitPending(role, parentNodeId, text, opts.raw ?? false);
    enqueuePending(
      {
        label: item.label,
        text: item.text,
        apply: item.apply,
        awaitsGen: item.awaitsGen,
        rebuild: item.rebuild,
        endsOnUserNode: item.endsOnUserNode,
      },
      { replaceSlot: opts.replaceSlot ?? null },
    );
    return;
  }
  // Immediate fire: resolve the sentinel against the live active node.
  // ``role === "user"`` accepts a null parent (server falls through to
  // the live active); ``role === "assistant"`` requires a real id.
  const resolved = parentNodeId === "active@drain"
    ? loomTree.active_node_id
    : parentNodeId;
  return sendCommitNow(role, resolved, text, opts.raw ?? false);
}

async function sendCommitNow(
  role: "user" | "assistant",
  parentNodeId: string | null,
  text: string,
  raw: boolean = false,
): Promise<void> {
  const sock = await ensureWebSocket();
  // Per-message role labels ride the commit too (roleplay scaffold), so an
  // authored turn is stamped with the box value just like a generated one.
  // Raw / flat commits carry no chat-template role — the server suppresses
  // labels there regardless, but we still omit them for clarity.
  const commitSampling = raw ? null : buildSamplingPayload();
  const payload: WSClientMessage = {
    type: "generate",
    commit_role: role,
    commit_text: text,
    parent_node_id: parentNodeId,
    ...(commitSampling ? { sampling: commitSampling } : {}),
    // ``raw`` lifts the user-under-user guard server-side — a flat
    // (base-model) commit's authored span may hang under any role.
    raw,
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

/** A/B compare state.  ``enabled`` is the user-visible toggle.  The
 * remaining fields drive the dual-roundtrip dance:
 *
 * - ``pendingTurnIdx`` — the steered-turn index waiting for its unsteered
 *   pair.  Set the moment the shadow gen is dispatched; cleared on shadow
 *   ``done`` or ``error``.
 * - ``processingAb`` — when true, the next stream of WS events
 *   (``started``/``token``/``done``) routes into ``turn.abPair`` on
 *   ``chatLog.turns[pendingTurnIdx]`` instead of allocating a fresh turn.
 *   This is the WS-side flag the message handler keys off.
 *
 * The shadow's prompt is reconstructed from ``chatLog.turns`` at fire
 * time (see ``_buildShadowMessages``) — no per-turn input string is
 * cached on this state, so toggling A/B mid-conversation works for any
 * turn, not only the just-sent one.
 *
 * Mid-flight toggle-off semantics: once a shadow gen is in flight, we let
 * it finish writing into ``abPair`` even if the user toggles A/B off — the
 * turn is harmless when not rendered, and tearing the WS state down mid-
 * stream is more error-prone than letting it complete.  Toggling off only
 * prevents the *next* steered gen from spawning a shadow.  If the steered
 * gen errors before the shadow fires, we never enter ``processingAb`` and
 * the abPair stays unset on that turn.
 */
/** Transient routing state for the unsteered-shadow generation.
 *
 *  v2.3: the standalone ``abState.enabled`` toggle is gone — the legacy
 *  "A/B" semantic has been folded into ``autoRegenState`` with
 *  ``mode === "unsteered"`` as the default.  The remaining
 *  ``processingAb`` / ``pendingTurnIdx`` fields are load-bearing for the
 *  WS dispatcher (they route shadow tokens into the steered turn's
 *  ``abPair`` instead of appending a fresh top-level turn). */
export interface AbState {
  pendingTurnIdx: number | null;
  processingAb: boolean;
}

export const abState: AbState = $state({
  pendingTurnIdx: null,
  processingAb: false,
});

/** Build the conversation as a messages list to replay through the
 * unsteered shadow.  Walks ``chatLog.turns[0..steeredIdx-1]`` (excluding
 * ``steeredIdx`` itself, which is the steered assistant response we
 * don't want the shadow to inherit), filtering out system / error turns
 * that aren't real conversation context.
 *
 * The unsteered model sees prior steered assistant turns as if they
 * happened naturally — that's the user's "play the conversation back"
 * contract.  Only the most recent user turn (the last entry in the
 * returned list) is what the shadow generates a fresh response for.
 *
 * Returns ``null`` when the slice doesn't end on a user turn (no
 * generation possible — the chatLog must have a trailing user turn for
 * the steered response to pair against). */
function _buildShadowMessages(
  steeredIdx: number,
): Array<{ role: "user" | "assistant"; content: string }> | null {
  const out: Array<{ role: "user" | "assistant"; content: string }> = [];
  for (let i = 0; i < steeredIdx; i++) {
    const t = chatLog.turns[i];
    if (!t) continue;
    if (t.role !== "user" && t.role !== "assistant") continue; // skip system / errors
    // Use the accumulated text — assistant turns already exclude their
    // thinking content (only response tokens land in ``turn.text``), so
    // replaying them through ``enable_thinking=False`` is well-formed.
    out.push({ role: t.role, content: t.text ?? "" });
  }
  if (out.length === 0 || out[out.length - 1].role !== "user") return null;
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
  const messages = _buildShadowMessages(steeredIdx);
  if (messages === null) return;
  const sock = await ensureWebSocket();
  // Shadow path mirrors ``sendGenerate``'s sampling-payload build so the
  // ``return_top_k`` opt-in rides shadow / auto-regen runs too (matches
  // the steered turn's wire-shape, keeps logit captures comparable across
  // siblings).
  const sampling = buildSamplingPayload();
  // Mark the WS reception path before the request lands so the
  // ``started`` event routes into the abPair and not a fresh turn.
  abState.pendingTurnIdx = steeredIdx;
  abState.processingAb = true;
  const payload: WSClientMessage = {
    type: "generate",
    // ``input`` accepts ``Any`` server-side; a list goes straight through
    // to ``session._prepare_input`` which dispatches on isinstance(list).
    input: messages,
    // Empty steering string == unsteered shadow per the WS protocol
    // (saklas_api._build_steering treats "" as "no expression").
    steering: "",
    sampling,
    thinking: samplingState.thinking ?? false,
    // Stateless so the shadow doesn't pollute server-side history; the
    // steered turn already populated history.  Combined with the
    // explicit messages list this means the shadow's prompt is exactly
    // the conversation up to (but not including) the steered response.
    stateless: true,
    raw: false,
  };
  const send = () => sock.send(JSON.stringify(payload));
  if (sock.readyState === WebSocket.OPEN) send();
  else sock.addEventListener("open", send, { once: true });
}

// =================================================== persistence ========
//
// v2 (loom):
//   localStorage is a *cache* of the server's loom tree.  On bootstrap we
//   fetch from the server and reconcile (server wins) — this retires the
//   v1 "server-restart guard" hack that dropped the local snapshot when
//   server-side history was empty.  The server tree is now the only
//   authority; localStorage is for first-paint while the fetch is in
//   flight.
//
// v1 → v2 migration:
//   The v1 snapshot stored a flat ChatTurn[].  On load we hydrate it as
//   a single-branch tree (root → user_1 → assistant_1 → user_2 → ...)
//   so the sidebar renders immediately even before the server tree
//   fetch returns.  When the server tree lands (rev > 0) it overwrites
//   the hydrated shape — no data loss either way.
//
// We persist on a debounced effect rather than synchronously per
// mutation so token-streaming gens don't write hundreds of times per
// turn — once per ~250 ms is enough to survive an unplanned reload.

const PERSIST_VERSION = 2;
const PERSIST_KEY_PREFIX = "saklas.chat.v" + PERSIST_VERSION + ".";
/** Legacy v1 key prefix — used for the migration read on load. */
const PERSIST_KEY_PREFIX_V1 = "saklas.chat.v1.";

function persistKey(): string | null {
  const id = sessionState.info?.model_id;
  return id ? PERSIST_KEY_PREFIX + id : null;
}

function persistKeyV1(): string | null {
  const id = sessionState.info?.model_id;
  return id ? PERSIST_KEY_PREFIX_V1 + id : null;
}

interface PersistedSnapshotV1 {
  version: 1;
  model_id: string;
  saved_at: number;
  turns: ChatTurn[];
  highlight: {
    target: string | null;
    compareTarget: string | null;
    compareTwo: boolean;
  };
}

interface PersistedSnapshotV2 {
  version: 2;
  model_id: string;
  saved_at: number;
  /** Cached loom tree.  ``null`` until the first server fetch lands. */
  tree: LoomTreeJSON | null;
  highlight: {
    target: string | null;
    compareTarget: string | null;
    compareTwo: boolean;
  };
}

function safeLocalStorageGet(key: string): string | null {
  try {
    return globalThis.localStorage?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

function safeLocalStorageSet(key: string, value: string): void {
  try {
    globalThis.localStorage?.setItem(key, value);
  } catch {
    // Quota exceeded / private-mode / SSR — silently drop.  Persistence
    // is a UX nicety, not a correctness requirement.
  }
}

function safeLocalStorageRemove(key: string): void {
  try {
    globalThis.localStorage?.removeItem(key);
  } catch {
    /* ignore */
  }
}

/** ULID-like throwaway id for synthetic migration nodes.  We don't need
 *  Crockford-base32 + monotonicity here — the server's tree fetch will
 *  overwrite this shape immediately, so unique-per-call is enough. */
function _localId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random()
    .toString(36)
    .slice(2, 10)}`;
}

/** Hydrate a v1 linear chat log as a single-branch loom tree.  No data
 *  loss — every user/assistant turn becomes a node with a synthetic id,
 *  parent-chained root-down.  Sets ``rev`` to 0 because the snapshot
 *  isn't authoritative; the server fetch will overwrite. */
function hydrateV1ChatAsLinearTree(turns: ChatTurn[]): void {
  loomTree.nodes.clear();
  loomTree.children_of.clear();
  const rootId = _localId("root");
  loomTree.nodes.set(rootId, {
    id: rootId,
    parent_id: null,
    role: "system",
    text: "",
  });
  loomTree.root_id = rootId;
  let parent = rootId;
  for (const t of turns) {
    if (!t || (t.role !== "user" && t.role !== "assistant" && t.role !== "system")) continue;
    const id = _localId(t.role);
    loomTree.nodes.set(id, {
      id,
      parent_id: parent,
      role: t.role,
      text: t.text ?? "",
      finish_reason: t.finishReason ?? null,
      applied_steering: t.appliedSteering ?? null,
      aggregate_readings: t.aggregateReadings,
    });
    const sibs = loomTree.children_of.get(parent) ?? [];
    loomTree.children_of.set(parent, [...sibs, id]);
    parent = id;
  }
  loomTree.active_node_id = parent;
  // Local hydration is rev 0 — server fetch will bump it.
  loomTree.rev = 0;
  recomputeActivePath();
}

function loadPersistedChat(): void {
  const v2key = persistKey();
  const v1key = persistKeyV1();
  if (!v2key) return;
  // Try v2 first; fall back to v1 migration.
  let parsed: PersistedSnapshotV2 | null = null;
  const v2raw = safeLocalStorageGet(v2key);
  if (v2raw) {
    try {
      const tmp = JSON.parse(v2raw) as PersistedSnapshotV2;
      if (tmp.version === 2 && tmp.model_id === sessionState.info?.model_id) {
        parsed = tmp;
      }
    } catch {
      /* corrupt — fall through to v1 migration */
    }
  }
  if (parsed) {
    if (parsed.tree) {
      // First-paint cache: hydrate without overwriting future server
      // fetches.  ``rev`` is preserved from cache, but ``refreshLoomTree``
      // is the authoritative source — its result overwrites.
      applyTreeSnapshot(parsed.tree);
      // Snapshot may have been from an in-flight gen; clear pendingNodeId
      // so a stale "streaming" indicator doesn't ghost the UI.
      loomTree.pendingNodeId = null;
    }
    if (parsed.highlight) {
      highlightState.target = parsed.highlight.target ?? null;
      highlightState.compareTarget = parsed.highlight.compareTarget ?? null;
      highlightState.compareTwo = !!parsed.highlight.compareTwo;
    }
    return;
  }
  if (!v1key) return;
  const v1raw = safeLocalStorageGet(v1key);
  if (!v1raw) return;
  try {
    const v1 = JSON.parse(v1raw) as PersistedSnapshotV1;
    if (v1.version !== 1) return;
    if (v1.model_id !== sessionState.info?.model_id) return;
    // Hydrate the linear log as a single-branch tree.  Don't drop the
    // snapshot on server-empty-history any more — the server will
    // either accept the tree on next mutation or fetch its own and
    // overwrite ours.
    if (Array.isArray(v1.turns)) {
      hydrateV1ChatAsLinearTree(
        v1.turns.filter(
          (t) =>
            t && typeof t === "object" && typeof (t as ChatTurn).role === "string",
        ),
      );
      // Also seed chatLog.turns directly so Chat.svelte has content on
      // first paint even before applyTreeSnapshot rewrites — the
      // hydrate above leaves rev at 0, so syncChatLogFromTree is a
      // no-op.
      chatLog.turns = v1.turns.filter(
        (t) => t && typeof t === "object" && typeof (t as ChatTurn).role === "string",
      );
      chatLog.pendingIndex = null;
    }
    if (v1.highlight) {
      highlightState.target = v1.highlight.target ?? null;
      highlightState.compareTarget = v1.highlight.compareTarget ?? null;
      highlightState.compareTwo = !!v1.highlight.compareTwo;
    }
    // Drop the v1 key once we've migrated — next persist writes v2.
    safeLocalStorageRemove(v1key);
  } catch {
    // Corrupt blob — drop it on the floor.
  }
}

let _persistTimer: ReturnType<typeof setTimeout> | null = null;
function schedulePersist(): void {
  if (_persistTimer) return;
  _persistTimer = setTimeout(() => {
    _persistTimer = null;
    const key = persistKey();
    if (!key) return;
    // Serialise the loom tree from the SvelteMap-backed slice.  Use
    // the server's ``to_dict`` list shape so a future reload (or any
    // server that consumes this cache) is consistent with the
    // authoritative wire format.
    let tree: LoomTreeJSON | null = null;
    if (loomTree.rev > 0 && loomTree.root_id && loomTree.active_node_id) {
      const nodes: LoomNodeJSON[] = [];
      for (const [, n] of loomTree.nodes) nodes.push(n);
      const children_of: Record<string, string[]> = {};
      for (const [pid, ids] of loomTree.children_of)
        children_of[pid] = [...ids];
      tree = {
        root_id: loomTree.root_id,
        active_node_id: loomTree.active_node_id,
        rev: loomTree.rev,
        nodes,
        children_of,
        model_id: loomTree.modelId ?? sessionState.info?.model_id,
      };
    }
    const snapshot: PersistedSnapshotV2 = {
      version: 2,
      model_id: sessionState.info!.model_id,
      saved_at: Date.now(),
      tree,
      highlight: {
        target: highlightState.target,
        compareTarget: highlightState.compareTarget,
        compareTwo: highlightState.compareTwo,
      },
    };
    const payload = JSON.stringify(snapshot);
    // Soft ~5MB budget warning (plan §"Size management").  Each loom-
    // tree rev bumps trigger a re-write of the whole snapshot, so a
    // large tree can put real pressure on the localStorage quota.  The
    // toast is advisory — we still write — and fires at most once per
    // session so the user doesn't get spammed on every rev.
    if (payload.length > _LOCALSTORAGE_SOFT_BUDGET && !_sizeWarned) {
      _sizeWarned = true;
      const mb = (payload.length / (1024 * 1024)).toFixed(1);
      pushToast(
        `Loom tree cache is ~${mb}MB in localStorage. Consider exporting and clearing — most browsers cap origin storage at 5–10MB.`,
        { kind: "warning", ttlMs: 10000 },
      );
    }
    safeLocalStorageSet(key, payload);
  }, 250);
}

/** 1 GB soft budget — effectively "never toast".  The toast was the only
 *  in-app surface that flagged cache size; muting it means the user no
 *  longer sees a warning before the browser's hard quota kicks in.
 *  ``safeLocalStorageSet`` still try/catches the resulting
 *  ``QuotaExceededError`` silently when that happens (write drops, next
 *  refresh refetches the server tree). */
const _LOCALSTORAGE_SOFT_BUDGET = 1024 * 1024 * 1024;

/** Once-per-session latch so we don't toast on every rev bump after
 *  the budget threshold is crossed.  Reset implicitly on page reload. */
let _sizeWarned = false;

/** Wire a $effect.root that watches the chat log + highlight slice and
 * debounces a save to localStorage.  Called from ``bootstrap`` after
 * the model id is known so the storage key resolves. */
function attachPersistence(): void {
  $effect.root(() => {
    $effect(() => {
      // Touch every reactive field we want to persist so the effect
      // re-runs whenever any of them change.
      void chatLog.turns.length;
      // Mutate-in-place arrays (token stream) — read .length on every
      // turn so ``schedulePersist`` debouncer fires through gen.
      for (const t of chatLog.turns) {
        void t.text;
        void t.tokens?.length;
        void t.thinkingTokens?.length;
      }
      // Loom tree changes drive the v2 persisted shape; touch rev so
      // every mutation queues a save.
      void loomTree.rev;
      void highlightState.target;
      void highlightState.compareTarget;
      void highlightState.compareTwo;
      // Skip the initial call (right after restore) — saves cycles and
      // avoids overwriting the snapshot before the user has done
      // anything.  Detect via the sentinel below.
      if (!_persistArmed) {
        _persistArmed = true;
        return;
      }
      schedulePersist();
    });
  });
}

let _persistArmed = false;

// =================================================== loom UI state ========

/** Sidebar-modal kind, also pokeable from App.svelte via the global
 *  Ctrl+R/E/B/N/D shortcuts.  ``null`` = no modal. */
export type LoomModalKind =
  | null
  | "regenerate"
  | "edit"
  | "branch"
  | "delete"
  | "note"
  | "navpicker"
  | "search";

export interface LoomUiState {
  /** Request flag: when the App's Ctrl+R/etc handlers want to open a
   *  modal inside the sidebar, they bump this counter and the sidebar
   *  reacts.  Counter lets the same modal be re-requested back-to-back
   *  (e.g. user closes regen modal then hits Ctrl+R again). */
  modalRequest: {
    seq: number;
    kind: LoomModalKind;
    nodeId: string | null;
    text: string;
    n: number;
  };
  /** Logit-pass: sibling sort key derived from filter grammar
   *  ``sort:surprise`` / ``sort:confidence``.  ``"default"`` preserves
   *  server insertion order.  Parsed client-side out of the filter
   *  input before the rest of the expression is sent to the server. */
  siblingSort: "default" | "surprise" | "confidence";
  /** Filter help popover visibility (Decision 8).  Toggled by the
   *  ``?`` button next to the filter input. */
  filterHelpOpen: boolean;
}

/** Loom (threads) UI state.  The threads column is a permanent part of
 *  the layout, so this carries no open/closed flag — only the modal
 *  request signal and the sort / filter-help knobs. */
export const loomUiState: LoomUiState = $state({
  modalRequest: { seq: 0, kind: null, nodeId: null, text: "", n: 1 },
  siblingSort: "default",
  filterHelpOpen: false,
});

// ============================================================ phase 5 ====
//
// Phase-5 loom flourishes — steering-delta edge labels (lazy cache),
// filter grammar (server-side), branch pinning to the comparison pane,
// auto-regen recipe-override modifier.  All in-memory only; not
// persisted across reloads (the engine recomputes from primitives).

/** Lazy cache of steering-delta labels for `parent_id|child_id` edges.
 *  The sidebar fetches on first render; SvelteMap so individual entries
 *  trigger reactivity in the edge components. */
export const edgeLabelCache: Map<string, string> = $state(new SvelteMap());

/** In-flight fetch dedupe — keys we've already kicked off a request
 *  for.  Cleared after the response lands so retries are possible
 *  when the rev changes. */
const _edgeLabelInFlight: Set<string> = new SvelteSet();

function _edgeKey(parentId: string, childId: string): string {
  return `${parentId}|${childId}`;
}

/** Fetch (and cache) the steering-delta label for an edge.  Returns
 *  immediately when the entry is already cached.  Bumps reactivity
 *  when the label arrives so all consumers re-render. */
export function fetchEdgeLabel(parentId: string, childId: string): void {
  if (loomTree.unavailable) return;
  const key = _edgeKey(parentId, childId);
  if (edgeLabelCache.has(key)) return;
  if (_edgeLabelInFlight.has(key)) return;
  _edgeLabelInFlight.add(key);
  apiTree
    .edgeLabel(parentId, childId)
    .then((r) => {
      edgeLabelCache.set(key, r.label);
    })
    .catch(() => {
      // Server pre-phase-5 or transient failure — cache an empty
      // string so we don't retry every render.
      edgeLabelCache.set(key, "");
    })
    .finally(() => {
      _edgeLabelInFlight.delete(key);
    });
}

/** Bust the cache when the tree mutates — the server's
 *  ``applied_steering`` strings can shift, especially after
 *  ``edit``/``regen``.  Wired into ``applyTreeDelta``. */
function invalidateEdgeLabels(): void {
  edgeLabelCache.clear();
  _edgeLabelInFlight.clear();
}

// ----------------------------------------------------- filter --------

export interface FilterState {
  /** User-entered expression string.  Empty = filter off. */
  expr: string;
  /** Server-resolved matching ids.  When ``expr`` is empty this is
   *  ``null`` — the UI then renders every node at full opacity. */
  matchingIds: Set<string> | null;
  /** Last parse / fetch error to surface in the input. */
  error: string | null;
  /** Pending state for the spinner. */
  loading: boolean;
}

export const filterState: FilterState = $state({
  expr: "",
  matchingIds: null,
  error: null,
  loading: false,
});

/** Strip ``sort:surprise`` / ``sort:confidence`` terms out of the filter
 *  expression before it reaches the server.  Sort is a client-side
 *  rendering concern (the DFS walk in LoomSidebar reorders siblings),
 *  so the server filter grammar doesn't need to know about it.  Stashes
 *  the resolved mode on ``loomUiState.siblingSort`` and returns the
 *  cleaned expression for the server.  Unknown ``sort:`` values fall
 *  through to the server, which will surface a parse error — that's
 *  the right UX (typo discovery), better than silently dropping. */
function _consumeSortPrefix(expr: string): string {
  // Match a comma-separated ``sort:<value>`` term anywhere in the
  // expression.  Comma is the filter grammar's AND separator so this
  // composes cleanly with other terms.
  const sortRe = /(?:^|,)\s*sort:(surprise|confidence)\s*(?=,|$)/gi;
  let mode: "default" | "surprise" | "confidence" = "default";
  const cleaned = expr.replace(sortRe, (_match, value: string) => {
    mode = value.toLowerCase() as "surprise" | "confidence";
    return "";
  });
  loomUiState.siblingSort = mode;
  // Drop leading / trailing commas and collapse double commas left by
  // the replace.
  return cleaned.replace(/,,+/g, ",").replace(/^\s*,|,\s*$/g, "").trim();
}

export async function applyTreeFilter(expr: string): Promise<void> {
  filterState.expr = expr;
  const trimmed = expr.trim();
  if (!trimmed) {
    filterState.matchingIds = null;
    filterState.error = null;
    filterState.loading = false;
    loomUiState.siblingSort = "default";
    return;
  }
  // Logit-pass: peel the client-side sort term off before sending to
  // the server.  Server filter grammar stays unchanged.
  const serverExpr = _consumeSortPrefix(trimmed);
  if (!serverExpr) {
    // Only ``sort:...`` was provided — no node-set filter, just a sort
    // directive.  Clear the matching-set so every node renders; the
    // sidebar's DFS picks up ``siblingSort`` independently.
    filterState.matchingIds = null;
    filterState.error = null;
    filterState.loading = false;
    return;
  }
  filterState.loading = true;
  filterState.error = null;
  try {
    const r = await apiTree.filter(serverExpr);
    filterState.matchingIds = new Set(r.matching_node_ids);
  } catch (e) {
    if (e instanceof ApiError) {
      filterState.error =
        e.body && typeof e.body === "object" && "detail" in (e.body as object)
          ? String((e.body as { detail: unknown }).detail)
          : e.message;
    } else {
      filterState.error = e instanceof Error ? e.message : String(e);
    }
    // Leave previous matches in place so the UI doesn't flicker; the
    // error message surfaces the parse failure.
  } finally {
    filterState.loading = false;
  }
}

export function clearTreeFilter(): void {
  filterState.expr = "";
  filterState.matchingIds = null;
  filterState.error = null;
  filterState.loading = false;
  // Logit-pass: clear the sibling-sort directive too — Esc / ✕ on the
  // filter input is the canonical "go back to default rendering" gesture.
  loomUiState.siblingSort = "default";
}

// ------------------------------------------- branch pinning ----------

/** Pinned-sibling state for the right-column comparison pane.  A node
 *  id (or ``null`` to default to A/B-style shadow).  Set via the
 *  context menu's "Pin to comparison" action. */
export const pinnedComparison: { nodeId: string | null } = $state({
  nodeId: null,
});

export function pinNodeForComparison(nodeId: string | null): void {
  pinnedComparison.nodeId = nodeId;
}

export function unpinComparison(): void {
  pinnedComparison.nodeId = null;
}

// ------------------------------- node multi-select for diff ---------

/** Multi-select for the cross-branch diff drawer.  Right-click on an
 *  assistant node toggles its membership; "Compare selected" opens the
 *  drawer with these ids.  Clears on drawer close or successful diff. */
export const nodeSelection: { ids: string[] } = $state({ ids: [] });

export function toggleNodeSelection(nodeId: string): void {
  const idx = nodeSelection.ids.indexOf(nodeId);
  if (idx === -1) nodeSelection.ids = [...nodeSelection.ids, nodeId];
  else nodeSelection.ids = nodeSelection.ids.filter((id) => id !== nodeId);
}

export function clearNodeSelection(): void {
  nodeSelection.ids = [];
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
  // the unsteered agent for the most recent steered assistant turn that
  // doesn't already carry an ``abPair``.  Mirrors the pre-v2.3 A/B
  // toggle's retroactive-shadow behaviour, so users who flip the toggle
  // on after-the-fact see the right column populate immediately rather
  // than waiting for the next send.  Other modes use the loom-regen
  // path — they take effect on the next ``done`` event by design.
  if (!wasOff) return;
  if (genStatus.active) return; // ``done`` handler will fire its own
  if (currentRecipeOverride() !== "unsteered") return;
  for (let i = chatLog.turns.length - 1; i >= 0; i--) {
    const t = chatLog.turns[i];
    if (!t) continue;
    if (t.role !== "assistant") continue;
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

/** Bump the modalRequest signal so the LoomSidebar opens the named
 *  modal with the given seed values. */
export function requestLoomModal(
  kind: LoomModalKind,
  opts: { nodeId?: string | null; text?: string; n?: number } = {},
): void {
  loomUiState.modalRequest = {
    seq: loomUiState.modalRequest.seq + 1,
    kind,
    nodeId: opts.nodeId ?? loomTree.active_node_id,
    text: opts.text ?? "",
    n: opts.n ?? 1,
  };
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
  // First-paint: load persisted (v2 cache or v1 migration) before
  // attaching the persist effect so we don't immediately overwrite.
  loadPersistedChat();
  // Per-model render-mode override (base vs chat) — also model-scoped.
  loadGenUiMode();
  attachPersistence();
  await Promise.allSettled([
    refreshVectorList(),
    // Unified probe roster — every probe shape (flat subspace + curved
    // manifold).  404 tolerated (pre-read-side server) → ``unavailable``.
    refreshProbeList(),
    refreshCorrelation(),
    // Manifold catalog — 404 tolerated (server pre-dates the manifold
    // HTTP surface); ``refreshManifoldList`` flips ``unavailable``.
    refreshManifoldList(),
    // Server tree wins — fetch and reconcile.  404 is a quiet no-op
    // (server pre-loom); other failures surface via loomTree.error.
    refreshLoomTree(),
  ]);
}
