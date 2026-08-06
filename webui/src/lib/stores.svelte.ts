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
export * from "./stores/loom.svelte";
import {
  applyTreeDelta,
  applyTreeSnapshot,
  castState,
  loomRegenerateActive,
  loomTree,
  recomputeActivePath,
  refreshLoomTree,
  syncChatLogFromTree,
} from "./stores/loom.svelte";
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
