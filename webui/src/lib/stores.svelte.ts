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

import {
  apiSessions,
  apiVectors,
  apiProbes,
  apiPacks,
  connectWs,
} from "./api";
import type {
  CorrelationData,
  SessionInfo,
  SweepEvent,
  VectorInfo,
  WSClientMessage,
  WSServerMessage,
} from "./api";
import type {
  ChatTurn,
  DrawerName,
  DrawerState,
  GenStatus,
  PendingAction,
  ProbeRackEntry,
  ProbeSortMode,
  ProjectionSpec,
  TokenScore,
  Trigger,
  Variant,
  VectorRackEntry,
} from "./types";
import { serializeExpression } from "./expression";

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
  } catch (e) {
    sessionState.error = e instanceof Error ? e.message : String(e);
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
}

export async function clearSessionHistory(): Promise<void> {
  await apiSessions.clear();
  // History length resets server-side; refresh to reflect it locally.
  await refreshSession();
  chatLog.turns = [];
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

export const vectorRack: VectorRack = $state({
  entries: new Map(),
  profiles: new Map(),
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

export function setVectorAlpha(name: string, alpha: number): void {
  enqueueOrApply(`alpha ${name} ${alpha.toFixed(3)}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) {
      e.alpha = alpha;
    } else {
      vectorRack.entries.set(name, defaultRackEntry(alpha));
    }
  });
}

export function setVectorEnabled(name: string, enabled: boolean): void {
  enqueueOrApply(`${enabled ? "enable" : "disable"} ${name}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) e.enabled = enabled;
  });
}

export function setVectorTrigger(name: string, trigger: Trigger): void {
  enqueueOrApply(`trigger ${name} ${trigger}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) e.trigger = trigger;
  });
}

export function setVectorVariant(name: string, variant: Variant): void {
  enqueueOrApply(`variant ${name} ${variant}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) e.variant = variant;
  });
}

export function setVectorProjection(
  name: string,
  projection: ProjectionSpec | null,
): void {
  enqueueOrApply(`project ${name}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) {
      e.projection = projection;
      // Ablation can't compose with projection — clear if a projection
      // was just set on top of an ablated entry.
      if (projection) e.ablate = false;
    }
  });
}

export function setVectorAblate(name: string, ablate: boolean): void {
  enqueueOrApply(`ablate ${name} ${ablate}`, () => {
    const e = vectorRack.entries.get(name);
    if (e) {
      e.ablate = ablate;
      if (ablate) e.projection = null;
    }
  });
}

export function addVectorToRack(
  name: string,
  alpha: number = 0,
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

/** The canonical expression string the rack would send to the server.
 * Recomputed on demand; cheap. */
export function currentSteeringExpression(): string {
  return serializeExpression(vectorRack.entries);
}

// =========================================================== probes =====

export interface ProbeRackState {
  /** Per-probe sparkline + last-tick state.  Keys are probe names. */
  entries: Map<string, ProbeRackEntry>;
  sortMode: ProbeSortMode;
  /** Mirrors sessionState.info?.probes — exposed separately so probe
   * adds/removes refresh independently of full session info. */
  active: string[];
}

export const probeRack: ProbeRackState = $state({
  entries: new Map(),
  sortMode: "value",
  active: [],
});

/** Computed: probes sorted per the user's chosen sort mode.  Returns
 * a fresh array on each access; consumers use it as a $derived
 * read-only view. */
export function activeProbeNames(): string[] {
  const arr = [...probeRack.active];
  if (probeRack.sortMode === "name") {
    arr.sort();
  } else if (probeRack.sortMode === "value") {
    arr.sort((a, b) => {
      const av = Math.abs(probeRack.entries.get(a)?.current ?? 0);
      const bv = Math.abs(probeRack.entries.get(b)?.current ?? 0);
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

export async function refreshProbeList(): Promise<void> {
  const r = await apiProbes.list();
  probeRack.active = r.probes.filter((p) => p.active).map((p) => p.name);
  // Drop any rack entries the server no longer reports active.
  for (const name of [...probeRack.entries.keys()]) {
    if (!probeRack.active.includes(name)) {
      probeRack.entries.delete(name);
    }
  }
  // Seed entries for newly active probes.
  for (const name of probeRack.active) {
    if (!probeRack.entries.has(name)) {
      probeRack.entries.set(name, {
        sparkline: [],
        current: 0,
        previous: 0,
      });
    }
  }
}

export async function activateProbe(name: string): Promise<void> {
  await apiProbes.activate(name);
  await refreshProbeList();
  // Auto-seed the highlight target when a probe is activated through
  // the rack — matches the TUI's /probe behavior, which flips highlight
  // on and points it at the new probe.
  if (highlightState.target === null) {
    highlightState.target = name;
  }
}

export async function deactivateProbe(name: string): Promise<void> {
  await apiProbes.deactivate(name);
  await refreshProbeList();
  if (highlightState.target === name) {
    highlightState.target = null;
  }
  if (highlightState.compareTarget === name) {
    highlightState.compareTarget = null;
  }
}

export function setProbeSortMode(mode: ProbeSortMode): void {
  probeRack.sortMode = mode;
}

/** Update a probe row from a streaming token's per-probe scores.  Drops
 * stale sparkline entries past ``MAX_SPARKLINE`` so memory stays bounded
 * across long sessions. */
const MAX_SPARKLINE = 60;
export function updateProbeFromScores(scores: Record<string, number>): void {
  for (const [name, val] of Object.entries(scores)) {
    let entry = probeRack.entries.get(name);
    if (!entry) {
      entry = { sparkline: [], current: 0, previous: 0 };
      probeRack.entries.set(name, entry);
    }
    entry.previous = entry.current;
    entry.current = val;
    entry.sparkline.push(val);
    if (entry.sparkline.length > MAX_SPARKLINE) {
      entry.sparkline.splice(0, entry.sparkline.length - MAX_SPARKLINE);
    }
  }
}

/** Snapshot the current per-probe means as the new "previous" baseline
 * — call after a generation lands so the next gen's deltas are computed
 * against the post-gen state, not mid-gen. */
export function snapshotProbeBaseline(): void {
  for (const e of probeRack.entries.values()) {
    e.previous = e.current;
  }
}

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
  target: null,
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
  /** ``null`` = auto, true/false = explicit override. */
  thinking: boolean | null;
  /** When true, the next generate sends these values as a one-shot
   * SamplingConfig (per-call override) instead of PATCHing the session
   * defaults.  TUI parity with the "session default vs. next message"
   * radio in the sampling strip. */
  oneShotOverride: boolean;
}

export const samplingState: SamplingState = $state({
  temperature: null,
  top_p: null,
  top_k: null,
  max_tokens: 256,
  seed: null,
  system_prompt: "",
  thinking: null,
  oneShotOverride: true,
});

export function setSampling<K extends keyof SamplingState>(
  key: K,
  value: SamplingState[K],
): void {
  samplingState[key] = value;
}

// ============================================================ drawers ====

export const drawerState: DrawerState = $state({
  open: null,
  params: null,
});

export function openDrawer(name: DrawerName, params: unknown = null): void {
  drawerState.open = name;
  drawerState.params = params;
}

export function closeDrawer(): void {
  drawerState.open = null;
  drawerState.params = null;
}

// ============================================================ packs ======

export const packsState: {
  installed: string[];
  loading: boolean;
  error: string | null;
} = $state({ installed: [], loading: false, error: null });

export async function refreshPacks(): Promise<void> {
  packsState.loading = true;
  try {
    const r = await apiPacks.list();
    packsState.installed = r.packs.map((p) => `${p.namespace}/${p.name}`);
    packsState.error = null;
  } catch (e) {
    packsState.error = e instanceof Error ? e.message : String(e);
  } finally {
    packsState.loading = false;
  }
}

// ===================================================== pending actions ===

export interface PendingActionsState {
  /** Queue of mutations deferred while a generation is running.  Drained
   * by ``applyPendingActions`` once the WS ``done`` event arrives, or
   * immediately when the user hits "apply now" (which also issues a stop
   * frame to interrupt the in-flight gen). */
  queue: PendingAction[];
}

export const pendingActions: PendingActionsState = $state({ queue: [] });

let _pendingCounter = 0;

export function enqueuePending(action: Omit<PendingAction, "id" | "createdAt">): void {
  pendingActions.queue.push({
    ...action,
    id: `pa-${_pendingCounter++}`,
    createdAt: Date.now(),
  });
}

export function applyPendingActions(): void {
  const q = pendingActions.queue;
  pendingActions.queue = [];
  for (const a of q) {
    try {
      void a.apply();
    } catch (e) {
      // Surface as a system message so the user sees the failure.
      chatLog.turns = [
        ...chatLog.turns,
        {
          role: "system",
          text: `pending action ${a.label} failed: ${String(e)}`,
        },
      ];
    }
  }
}

export function discardPendingActions(): void {
  pendingActions.queue = [];
}

/** Apply immediately if no gen is in flight; queue otherwise.  Every
 * rack/sampling mutation routes through this so behavior is uniform. */
function enqueueOrApply(label: string, apply: () => void): void {
  if (genStatus.active) {
    enqueuePending({ label, apply });
  } else {
    apply();
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
  listeners: new Set(),
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

/** Default WS message handler — owns the gen-status lifecycle and the
 * live token stream.  External subscribers (panels) layer additional
 * behavior via ``onWsMessage``. */
function handleWsMessage(msg: WSServerMessage): void {
  switch (msg.type) {
    case "started": {
      genStatus.active = true;
      genStatus.tokensSoFar = 0;
      genStatus.startedAt = performance.now();
      genStatus.tokPerSec = 0;
      genStatus.ppl = { logSum: 0, count: 0, mean: null };
      genStatus.finishReason = null;
      liveTokenStream.responseTokens = [];
      liveTokenStream.thinkingTokens = [];
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
      } else {
        // Normal run: append a fresh assistant turn so streamed tokens
        // have a home.
        chatLog.turns = [
          ...chatLog.turns,
          { role: "assistant", text: "", tokens: [], thinkingTokens: [] },
        ];
        chatLog.pendingIndex = chatLog.turns.length - 1;
      }
      return;
    }
    case "token": {
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
      };
      // Pull "best score" from the latest layer's selected probe so
      // panels rendering a single highlight have something to draw
      // against immediately.  The canonical projected scores overwrite
      // these on done.
      if (msg.per_layer_scores && highlightState.target) {
        const layers = Object.keys(msg.per_layer_scores);
        if (layers.length > 0) {
          const last = layers[layers.length - 1];
          const score =
            msg.per_layer_scores[last]?.[highlightState.target];
          if (typeof score === "number") tokenScore.score = score;
        }
        // Cache full per-probe row at the latest layer for tooltip use.
        const last = layers[layers.length - 1];
        tokenScore.probes = msg.per_layer_scores[last];
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
      // Update probe rack from the deepest-layer per-probe row when
      // available — cheap proxy for the TUI's live trait readings.
      // Skip during shadow runs so the rack stays anchored to the
      // steered branch's signal.
      if (msg.per_layer_scores && !abState.processingAb) {
        const layers = Object.keys(msg.per_layer_scores);
        if (layers.length > 0) {
          const last = layers[layers.length - 1];
          updateProbeFromScores(msg.per_layer_scores[last]);
        }
      }
      return;
    }
    case "done": {
      genStatus.active = false;
      genStatus.finishReason = msg.result?.finish_reason ?? "stop";
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
      }

      const wasShadow = abState.processingAb;
      const steeredIdx = chatLog.pendingIndex;
      chatLog.pendingIndex = null;

      if (wasShadow) {
        // Shadow gen done — clear the A/B routing flags.  Do NOT touch
        // the probe baseline or correlation refresh; the steered turn
        // already did that when it finished.
        abState.processingAb = false;
        abState.pendingTurnIdx = null;
        // Drain pending actions queued during the shadow gen — same
        // gen-active gate the steered branch uses.
        applyPendingActions();
        return;
      }

      // Snapshot probe baselines + drain any deferred mutations on the
      // steered done event only.
      snapshotProbeBaseline();
      void refreshCorrelation();
      applyPendingActions();

      // If A/B is on and this was the steered run that just finished,
      // dispatch the unsteered shadow generate against the same input.
      // We require the steered idx + lastInput to be intact and the
      // turn to actually be an assistant turn (not a system error).
      if (
        abState.enabled &&
        abState.lastInput !== null &&
        steeredIdx !== null &&
        chatLog.turns[steeredIdx]?.role === "assistant"
      ) {
        void _sendShadowGenerate(abState.lastInput, steeredIdx);
      }
      return;
    }
    case "error": {
      genStatus.active = false;
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
      abState.processingAb = false;
      abState.pendingTurnIdx = null;
      // Apply any pending actions even on error so the UI doesn't get
      // stuck in "changes pending" forever.
      applyPendingActions();
      return;
    }
  }
}

/** Send a generate request over the WS.  Builds the steering expression
 * from the rack live, layers the SamplingConfig overrides when one-shot
 * mode is on, and routes everything through the singleton connection. */
export async function sendGenerate(
  input: string,
  opts: {
    stateless?: boolean;
    raw?: boolean;
    /** Override the rack-derived steering with an explicit string.  Pass
     * ``""`` for unsteered (A/B mode); ``null``/``undefined`` to use the
     * rack. */
    steering?: string | null;
  } = {},
): Promise<void> {
  const sock = await ensureWebSocket();
  const steering =
    opts.steering === undefined ? currentSteeringExpression() : opts.steering;
  const sampling = samplingState.oneShotOverride
    ? {
        temperature: samplingState.temperature,
        top_p: samplingState.top_p,
        top_k: samplingState.top_k,
        max_tokens: samplingState.max_tokens,
        seed: samplingState.seed,
      }
    : null;
  // Update genStatus.maxTokens locally so the progress bar widths know
  // their target before the first token lands.
  genStatus.maxTokens = sampling?.max_tokens ?? samplingState.max_tokens;
  // Push the user turn so the UI has something to render before the WS
  // started event lands.
  chatLog.turns = [...chatLog.turns, { role: "user", text: input }];
  // Remember the input verbatim so the A/B path can replay it as the
  // shadow gen.  Only meaningful when ``abState.enabled``; otherwise it's
  // dead weight that's free to keep up to date.
  abState.lastInput = input;
  const payload: WSClientMessage = {
    type: "generate",
    input,
    steering: steering || null,
    sampling,
    thinking: samplingState.thinking,
    stateless: opts.stateless ?? false,
    raw: opts.raw ?? false,
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

// ============================================================ sweep =====

export interface SweepRow {
  idx: number;
  alpha_values: Record<string, number>;
  text: string;
  token_count: number;
  tok_per_sec: number;
  elapsed: number;
  finish_reason: string;
  applied_steering: string | null;
  readings: Record<string, number>;
}

export interface SweepState {
  rows: SweepRow[];
  total: number;
  completed: number;
  active: boolean;
  error: string | null;
  sweepId: string | null;
}

export const sweepState: SweepState = $state({
  rows: [],
  total: 0,
  completed: 0,
  active: false,
  error: null,
  sweepId: null,
});

export function ingestSweepEvent(ev: SweepEvent): void {
  switch (ev.type) {
    case "started":
      sweepState.rows = [];
      sweepState.completed = 0;
      sweepState.total = ev.total;
      sweepState.active = true;
      sweepState.error = null;
      sweepState.sweepId = ev.sweep_id;
      return;
    case "result":
      sweepState.rows = [
        ...sweepState.rows,
        {
          idx: ev.idx,
          alpha_values: ev.alpha_values,
          text: ev.result.text,
          token_count: ev.result.token_count,
          tok_per_sec: ev.result.tok_per_sec,
          elapsed: ev.result.elapsed,
          finish_reason: ev.result.finish_reason,
          applied_steering: ev.result.applied_steering,
          readings: ev.result.readings,
        },
      ];
      sweepState.completed += 1;
      return;
    case "done":
      sweepState.active = false;
      sweepState.completed = ev.summary.completed;
      return;
    case "error":
      sweepState.active = false;
      sweepState.error = ev.message;
      return;
  }
}

// =========================================== A/B compare metadata =======

/** A/B compare state.  ``enabled`` is the user-visible toggle.  The
 * remaining fields drive the dual-roundtrip dance:
 *
 * - ``lastInput`` — the input the user just sent, replayed verbatim for the
 *   shadow (unsteered) gen.
 * - ``pendingTurnIdx`` — the steered-turn index waiting for its unsteered
 *   pair.  Set the moment the shadow gen is dispatched; cleared on shadow
 *   ``done`` or ``error``.
 * - ``processingAb`` — when true, the next stream of WS events
 *   (``started``/``token``/``done``) routes into ``turn.abPair`` on
 *   ``chatLog.turns[pendingTurnIdx]`` instead of allocating a fresh turn.
 *   This is the WS-side flag the message handler keys off.
 *
 * Mid-flight toggle-off semantics: once a shadow gen is in flight, we let
 * it finish writing into ``abPair`` even if the user toggles A/B off — the
 * turn is harmless when not rendered, and tearing the WS state down mid-
 * stream is more error-prone than letting it complete.  Toggling off only
 * prevents the *next* steered gen from spawning a shadow.  If the steered
 * gen errors before the shadow fires, we never enter ``processingAb`` and
 * the abPair stays unset on that turn.
 */
export interface AbState {
  enabled: boolean;
  lastInput: string | null;
  pendingTurnIdx: number | null;
  processingAb: boolean;
}

export const abState: AbState = $state({
  enabled: false,
  lastInput: null,
  pendingTurnIdx: null,
  processingAb: false,
});

export function toggleAb(): void {
  abState.enabled = !abState.enabled;
  // Toggling off does not abandon an in-flight shadow gen — the events
  // route through to ``abPair`` regardless.  Toggling back on while a
  // shadow is mid-flight is a no-op for the WS path; the next steered
  // gen will spawn its own pair.
}

/** Internal: dispatch the unsteered shadow generate that pairs with the
 * just-finished steered turn at index ``steeredIdx``.  Pushes no user turn
 * (it was already rendered when the steered gen started) and routes WS
 * events into the steered turn's ``abPair`` via ``processingAb``. */
async function _sendShadowGenerate(
  input: string,
  steeredIdx: number,
): Promise<void> {
  const sock = await ensureWebSocket();
  const sampling = samplingState.oneShotOverride
    ? {
        temperature: samplingState.temperature,
        top_p: samplingState.top_p,
        top_k: samplingState.top_k,
        max_tokens: samplingState.max_tokens,
        seed: samplingState.seed,
      }
    : null;
  // Mark the WS reception path before the request lands so the
  // ``started`` event routes into the abPair and not a fresh turn.
  abState.pendingTurnIdx = steeredIdx;
  abState.processingAb = true;
  const payload: WSClientMessage = {
    type: "generate",
    input,
    // Empty steering string == unsteered shadow per the WS protocol
    // (saklas_api._build_steering treats "" as "no expression").
    steering: "",
    sampling,
    thinking: samplingState.thinking,
    // Stateless so the shadow doesn't pollute server-side history; the
    // steered turn already populated history.
    stateless: true,
    raw: false,
  };
  const send = () => sock.send(JSON.stringify(payload));
  if (sock.readyState === WebSocket.OPEN) send();
  else sock.addEventListener("open", send, { once: true });
}

// ============================================================ misc ======

/** Bootstrap the dashboard — call once on App mount.  Resolves only once
 * every parallel fetch settles so the UI's first paint has a real session
 * shape. */
export async function bootstrap(): Promise<void> {
  await Promise.allSettled([
    refreshSession(),
    refreshVectorList(),
    refreshProbeList(),
    refreshCorrelation(),
    refreshPacks(),
  ]);
}
