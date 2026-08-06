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
export * from "./stores/ab.svelte";
export * from "./stores/ws.svelte";
import { ensureWebSocket } from "./stores/ws.svelte";
import {
  abState,
  autoRegenState,
  currentRecipeOverride,
  sendShadowGenerate,
} from "./stores/ab.svelte";
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
