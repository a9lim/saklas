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

/** One read family's block from session info — the SAME shape
 *  ``GET .../instruments`` lists, so there is one representation of
 *  instrument state to read from. */
export function instrumentFamily(
  family: InstrumentFamily,
): InstrumentFamilyBlock | undefined {
  return sessionState.info?.instruments?.find((row) => row.family === family);
}

/** True iff an SAE is resident (its family block reports an active
 *  source).  Replaces the flat ``sae_loaded`` key. */
export function saeLoaded(): boolean {
  return instrumentFamily("sae")?.source != null;
}

export async function refreshSession(): Promise<void> {
  try {
    const info = await apiSessions.get();
    sessionState.info = info;
    sessionState.lastRefresh = Date.now();
    sessionState.error = null;
    hydrateSamplingFromInfo();
    // Rehydrate the PROBE-section toggles from ONE server representation —
    // the same per-family blocks ``GET .../instruments`` lists.  A page
    // reload must reflect the server exactly.
    const lens = instrumentFamily("lens");
    const sae = instrumentFamily("sae");
    const geometry = instrumentFamily("geometry");
    lensState.layers = lens?.live.enabled
      ? ("layers" in lens.live ? lens.live.layers : null)
      : null;
    saeState.live = sae?.live.enabled === true;
    // Feature ids (and their metadata) belong to the resident release —
    // reset the discovery/metadata state when it changes.
    const release = sae?.source ?? null;
    const layer = sae && "layer" in sae.live ? sae.live.layer : null;
    if (release !== saeState.release || layer !== saeState.layer) {
      saeState.release = release;
      saeState.layer = layer;
      saeState.readout = [];
      saeState.history.clear();
      saeState.meta.clear();
      _saeMetaRequested.clear();
    }
    probesLiveState.enabled = geometry?.live.enabled !== false;
  } catch (e) {
    sessionState.error = e instanceof Error ? e.message : String(e);
  }
}

// ======================================================== live lens ====

export interface LensState {
  /** Resolved fitted-layer list while the live J-lens readout is
   * enabled; ``null`` while off.  Mirrors the server's
   * lens family block's ``live.layers`` — the panel toggle reads this, not
   * a local boolean, so reloads and multi-tab stay honest. */
  layers: number[] | null;
  /** Latest decode step's readout: layer-index string → descending
   * ``[token, p]`` top-k (per-layer softmax probability — the one
   * strength unit every lens surface reports).  Overwritten per token
   * frame; kept after ``done`` so the settled matrix stays readable. */
  readout: Record<string, [string, number][]> | null;
  /** Layer-aggregated chip list riding the same step — ``[token,
   * strength, com, spread]`` strength-descending (mean fitted-layer probability
   * + probability-mass-weighted depth center of mass).  Same lifecycle as
   * ``readout``. */
  aggregate: [string, number, number, number][] | null;
  /** Rolling buffer of recent aggregate frames (``[token, strength]``
   * pairs per step, newest last, capped like the probe sparklines) —
   * backs the workspace token cards' sparklines.  Carries across
   * generations like probe sparklines; cleared on live-lens disable. */
  aggHistory: [string, number][][];
  /** Presentation order for the aggregate workspace cards.  Kept in the
   * shared lens state so switching inspector tabs does not reset it. */
  workspaceSortMode: LensWorkspaceSortMode;
  /** In-flight toggle guard (the enable moves J_l device-resident and
   * waits on the session lock, so it can lag behind a long stream). */
  busy: boolean;
}

export type LensWorkspaceSortMode = "strength" | "name" | "depth";

export const lensState: LensState = $state({
  layers: null,
  readout: null,
  aggregate: null,
  aggHistory: [],
  workspaceSortMode: "strength",
  busy: false,
});

export const lensSourceState: {
  sources: InstrumentSourceJSON[];
  loading: boolean;
  busy: boolean;
  error: string | null;
} = $state({ sources: [], loading: false, busy: false, error: null });

export async function refreshLensSources(): Promise<void> {
  if (lensSourceState.loading) return;
  lensSourceState.loading = true;
  try {
    lensSourceState.sources = (await apiInstruments.sources("lens")).sources;
    lensSourceState.error = null;
  } catch (e) {
    lensSourceState.error = e instanceof Error ? e.message : String(e);
  } finally {
    lensSourceState.loading = false;
  }
}

export async function useLensSource(source: string): Promise<void> {
  if (lensSourceState.busy || !source) return;
  lensSourceState.busy = true;
  try {
    const out = await apiInstruments.setLensSource(source);
    lensState.layers = out.live_layers;
    await refreshSession();
    await refreshLensSources();
    pushToast(`J-lens · ${source}`, { kind: "info" });
  } catch (e) {
    pushToast(
      `J-lens source: ${e instanceof Error ? e.message : String(e)}`,
      { kind: "error" },
    );
  } finally {
    lensSourceState.busy = false;
  }
}

export function setLensWorkspaceSortMode(mode: LensWorkspaceSortMode): void {
  lensState.workspaceSortMode = mode;
}

// ========================================================= live SAE ====

export interface SaeState {
  live: boolean;
  readout: SaeFeatureJSON[];
  /** Raw activation history per feature id (drives the sparklines; the
   *  bars derive their scale from ``meta`` instead). */
  history: Map<number, number[]>;
  /** Session-side Neuronpedia metadata per feature id — merged from the
   *  token frames' cached values and the between-generation backfill.
   *  ``max_act`` is the strength unit: bars render
   *  ``activation / max_act`` on the absolute 0..1 scale (the lens-card
   *  convention); features without it fall back to the panel-shared raw
   *  scale. */
  meta: Map<number, { label: string | null; max_act: number | null }>;
  /** Resident release the discovery state belongs to (reset key). */
  release: string | null;
  /** Resident hook layer; changing it invalidates feature ids/history too. */
  layer: number | null;
  /** Presentation order shared across tab switches, mirroring the lens
   *  workspace sorter. */
  sortMode: SaeSortMode;
  busy: boolean;
}

export const saeState: SaeState = $state({
  live: false,
  readout: [],
  history: new SvelteMap<number, number[]>(),
  meta: new SvelteMap<number, { label: string | null; max_act: number | null }>(),
  release: null,
  layer: null,
  sortMode: "strength",
  busy: false,
});

// ================================================= token hover readout ====

/** A non-destructive, token-anchored view of the three read channels shown in
 * the inspector.  Panels read this overlay while a transcript token is under
 * the pointer, then fall back to their ordinary live/settled state on leave. */
export interface TokenHoverState {
  active: boolean;
  key: string | null;
  tokenText: string;
  probeReadings: Record<string, AnyReadingJSON> | null;
  probes: Record<string, number> | null;
  coordsByProbe: Record<string, number[]> | null;
  perLayerScores: Record<string, Record<string, number>> | null;
  lensReadout: Record<string, [string, number][]> | null;
  lensAggregate: [string, number, number, number][] | null;
  saeReadout: SaeFeatureJSON[] | null;
  lensLoading: boolean;
  saeLoading: boolean;
}

export const tokenHoverState: TokenHoverState = $state({
  active: false,
  key: null,
  tokenText: "",
  probeReadings: null,
  probes: null,
  coordsByProbe: null,
  perLayerScores: null,
  lensReadout: null,
  lensAggregate: null,
  saeReadout: null,
  lensLoading: false,
  saeLoading: false,
});

interface LensHoverSnapshot {
  readout: Record<string, [string, number][]>;
  aggregate: [string, number, number, number][];
}

const TOKEN_HOVER_DELAY_MS = 140;
let _hoverFetchTimer: ReturnType<typeof setTimeout> | null = null;
let _hoverClearTimer: ReturnType<typeof setTimeout> | null = null;

function _hoverTokenKey(nodeId: string, rawIndex: number, raw: boolean): string {
  const modelKey = sessionState.info?.model_id ?? "unknown-model";
  return `${modelKey}:${nodeId}:${rawIndex}:${raw ? 1 : 0}`;
}

/** The lens native readout block → the display shape the workspace cards and
 *  hover consume: per-layer ``[token, probability]`` (the wire carries
 *  logprob — ``exp()`` restores the displayed per-layer softmax probability)
 *  plus the ``[token, strength, com, spread]`` aggregate chips. */
function _lensReadoutSnapshot(
  block: LensReadoutBlockJSON | undefined,
): LensHoverSnapshot | null {
  if (!block) return null;
  return {
    readout: Object.fromEntries(block.layers.map((row) => [
      String(row.layer),
      row.tokens.map((token) => [token.token, Math.exp(token.logprob)]),
    ])),
    aggregate: block.aggregate.map((token) => [
      token.token,
      token.strength,
      token.com,
      token.spread,
    ]),
  };
}

/** Merge the three families' attached-probe ``readings`` from a measurement
 *  envelope — the probe rack keys by name across families, so one dict drives
 *  every probe strip. */
function _mergedReadings(
  m: MeasurementsEnvelopeJSON | undefined | null,
): Record<string, AnyReadingJSON> | undefined {
  if (!m) return undefined;
  const g = m.instruments.geometry?.readings;
  const l = m.instruments.lens?.readings;
  const s = m.instruments.sae?.readings;
  if (!g && !l && !s) return undefined;
  return { ...(g ?? {}), ...(l ?? {}), ...(s ?? {}) };
}

function _fetchLensHover(
  nodeId: string,
  rawIndex: number,
  raw: boolean,
): Promise<LensHoverSnapshot | null> {
  return apiInstruments.tokenReadout("lens", nodeId, rawIndex, {
    topK: resolveReadoutTopK(samplingState.return_top_k),
    steered: true,
    raw,
    layers: "all",
  }).then((res) => _lensReadoutSnapshot(res.measurements.instruments.lens?.readout));
}

function _fetchSaeHover(
  nodeId: string,
  rawIndex: number,
  raw: boolean,
): Promise<SaeFeatureJSON[]> {
  return apiInstruments.tokenReadout("sae", nodeId, rawIndex, {
    topK: resolveReadoutTopK(samplingState.return_top_k),
    steered: true,
    raw,
  }).then((res) => res.measurements.instruments.sae?.readout?.features ?? []);
}

/** Begin showing one transcript token in the inspector. Loom-owned captures
 * land synchronously; a channel absent from the original generation may be
 * replayed after a short dwell, but replay results are not substituted for or
 * retained as original capture data. */
export function beginTokenHover(
  token: TokenScore,
  nodeId: string | null | undefined,
): void {
  if (_hoverClearTimer !== null) clearTimeout(_hoverClearTimer);
  if (_hoverFetchTimer !== null) clearTimeout(_hoverFetchTimer);
  _hoverClearTimer = null;
  _hoverFetchTimer = null;

  const raw = effectiveRawMode();
  const rawIndex = token.rawIndex ?? null;
  const modelKey = sessionState.info?.model_id ?? "unknown-model";
  const key = nodeId && rawIndex !== null
    ? _hoverTokenKey(nodeId, rawIndex, raw)
    : `live:${modelKey}:${nodeId ?? "none"}:${rawIndex ?? "none"}:${token.tokenId ?? "none"}`;
  const m = token.measurements;
  const capturedLens = m?.instruments.lens?.readout;
  const capturedSae = m?.instruments.sae?.readout;
  tokenHoverState.active = true;
  tokenHoverState.key = key;
  tokenHoverState.tokenText = token.text;
  tokenHoverState.probeReadings = _mergedReadings(m) ?? null;
  tokenHoverState.probes = m?.scores ?? token.probes ?? null;
  tokenHoverState.coordsByProbe = token.coordsByProbe ?? null;
  tokenHoverState.perLayerScores =
    m?.per_layer_scores ?? token.perLayerScores ?? null;
  tokenHoverState.lensReadout = null;
  tokenHoverState.lensAggregate = null;
  tokenHoverState.saeReadout = capturedSae?.features ?? null;
  tokenHoverState.lensLoading = false;
  tokenHoverState.saeLoading = false;

  const lensSnapshot = _lensReadoutSnapshot(capturedLens);
  if (lensSnapshot) {
    tokenHoverState.lensReadout = lensSnapshot.readout;
    tokenHoverState.lensAggregate = lensSnapshot.aggregate;
  }

  if (!nodeId || rawIndex === null) return;
  const needsLens = capturedLens === undefined &&
    sessionState.info?.jlens_fitted === true;
  const needsSae = capturedSae === undefined && saeLoaded();
  tokenHoverState.lensLoading = needsLens;
  tokenHoverState.saeLoading = needsSae;
  if (!needsLens && !needsSae) return;

  _hoverFetchTimer = setTimeout(() => {
    _hoverFetchTimer = null;
    if (needsLens) {
      void _fetchLensHover(nodeId, rawIndex, raw)
        .then((snapshot) => {
          if (!tokenHoverState.active || tokenHoverState.key !== key) return;
          if (!snapshot) return;
          tokenHoverState.lensReadout = snapshot.readout;
          tokenHoverState.lensAggregate = snapshot.aggregate;
        })
        .catch(() => { /* Opportunistic hover read: fall through to the empty hint. */ })
        .finally(() => {
          if (tokenHoverState.active && tokenHoverState.key === key) {
            tokenHoverState.lensLoading = false;
          }
        });
    }
    if (needsSae) {
      void _fetchSaeHover(nodeId, rawIndex, raw)
        .then((features) => {
          if (!tokenHoverState.active || tokenHoverState.key !== key) return;
          tokenHoverState.saeReadout = features;
        })
        .catch(() => { /* Opportunistic hover read: fall through to the empty hint. */ })
        .finally(() => {
          if (tokenHoverState.active && tokenHoverState.key === key) {
            tokenHoverState.saeLoading = false;
          }
        });
    }
  }, TOKEN_HOVER_DELAY_MS);
}

/** Delay the clear just enough to cross the whitespace between adjacent token
 * spans without flashing the inspector back to the settled generation. */
export function endTokenHover(): void {
  if (_hoverFetchTimer !== null) clearTimeout(_hoverFetchTimer);
  if (_hoverClearTimer !== null) clearTimeout(_hoverClearTimer);
  _hoverFetchTimer = null;
  _hoverClearTimer = setTimeout(() => {
    tokenHoverState.active = false;
    tokenHoverState.key = null;
    tokenHoverState.tokenText = "";
    tokenHoverState.lensLoading = false;
    tokenHoverState.saeLoading = false;
    _hoverClearTimer = null;
  }, 45);
}

export function lensReadoutForDisplay(): Record<string, [string, number][]> | null {
  return tokenHoverState.active ? tokenHoverState.lensReadout : lensState.readout;
}

export function lensAggregateForDisplay(): [string, number, number, number][] | null {
  return tokenHoverState.active ? tokenHoverState.lensAggregate : lensState.aggregate;
}

export function saeReadoutForDisplay(): SaeFeatureJSON[] {
  return tokenHoverState.active ? (tokenHoverState.saeReadout ?? []) : saeState.readout;
}

export const saeSourceState: {
  sources: InstrumentSourceJSON[];
  loading: boolean;
  error: string | null;
} = $state({ sources: [], loading: false, error: null });

export async function refreshSaeSources(): Promise<void> {
  if (saeSourceState.loading) return;
  saeSourceState.loading = true;
  try {
    saeSourceState.sources = (await apiInstruments.sources("sae")).sources;
    saeSourceState.error = null;
  } catch (e) {
    saeSourceState.error = e instanceof Error ? e.message : String(e);
  } finally {
    saeSourceState.loading = false;
  }
}

export type SaeSortMode = "strength" | "name";

export function setSaeSortMode(mode: SaeSortMode): void {
  saeState.sortMode = mode;
}

/** Ids already sent to the metadata backfill this session — a miss on
 *  Neuronpedia stays a miss, so don't re-ask every generation.  Not
 *  reactive state (never rendered). */
const _saeMetaRequested = new Set<number>();

/** Between-generation discovery backfill: fetch-and-cache Neuronpedia
 *  metadata (label + maxActApprox) for every feature the live top-k
 *  surfaced that has none yet.  Fire-and-forget from the ``done``
 *  handler — never per token. */
export async function backfillSaeMeta(): Promise<void> {
  if (!saeLoaded()) return;
  const wanted: number[] = [];
  for (const id of saeState.history.keys()) {
    if (saeState.meta.get(id)?.max_act != null) continue;
    if (_saeMetaRequested.has(id)) continue;
    wanted.push(id);
    if (wanted.length >= 64) break;
  }
  if (wanted.length === 0) return;
  for (const id of wanted) _saeMetaRequested.add(id);
  try {
    const out = await apiInstruments.saeFeaturesMetadata(wanted);
    for (const [key, entry] of Object.entries(out.features)) {
      saeState.meta.set(Number(key), {
        label: entry.label ?? null,
        max_act: entry.max_act ?? null,
      });
    }
  } catch {
    // Best-effort — allow a retry on the next generation.
    for (const id of wanted) _saeMetaRequested.delete(id);
  }
}

export async function setLiveSae(enabled: boolean): Promise<void> {
  if (saeState.busy) return;
  saeState.busy = true;
  try {
    const out = await apiInstruments.setLive("sae", { enabled });
    saeState.live = out.enabled;
    if (!out.enabled) {
      saeState.readout = [];
      saeState.history.clear();
    }
  } catch (e) {
    pushToast(
      `SAE live: ` +
        (e instanceof Error ? e.message : String(e)),
      { kind: "error" },
    );
  } finally {
    saeState.busy = false;
  }
}

/** The four background preparations, one slice each — see
 *  ``lib/stores/preparations.svelte.ts`` for the shared contract.  Each
 *  supplies only its poll cadence, its toast wording, and the refreshes
 *  its result invalidates. */
// The SAE source-binding preparation.  The HTTP operation is ``fetch``,
// matching the CLI verb and the lens family (it used to be spelled
// ``load`` over HTTP alone).
export const saeLoad = createPreparationSlice("sae", "fetch", {
  label: "SAE fetch",
  intervalMs: 1000,
  successMessage: "SAE loaded",
  onSettled: async () => {
    await refreshSession();
    await refreshSaeSources();
    await refreshProbeList();
  },
});

export const saeTrain = createPreparationSlice("sae", "train", {
  label: "SAE train",
  intervalMs: 1500,
  successMessage: "SAE trained · live",
  onSettled: async () => {
    await refreshSession();
    await refreshSaeSources();
    await refreshProbeList();
  },
});

/** Load a resident SAE — ``local:<name>`` or ``saelens:<release>``, with an
 *  optional hook layer.  Thin wrapper over the slice: the release is the
 *  only field that needs trimming + an empty check. */
export function loadSae(release: string, layer: number | null = null): void {
  const trimmed = release.trim();
  if (!trimmed) return;
  void saeLoad.start({ release: trimmed, layer });
}

// ================================================== CAA live toggle ====

/** CAA PROBE-section live toggle — whether per-token monitor scoring
 *  feeds live consumers.  The J-lens sibling is ``lensState.layers``
 *  (the live lens); with both off a compute-constrained session pays no
 *  per-token scoring at all, and every probe still reports its
 *  end-of-gen aggregate. */
export const probesLiveState: { enabled: boolean; busy: boolean } = $state({
  enabled: true,
  busy: false,
});

export async function setLiveProbes(enabled: boolean): Promise<void> {
  if (probesLiveState.busy) return;
  probesLiveState.busy = true;
  try {
    const out = await apiInstruments.setLive("geometry", { enabled });
    probesLiveState.enabled = out.enabled;
  } catch (e) {
    pushToast(
      `probe live: ` +
        (e instanceof Error ? e.message : String(e)),
      { kind: "error" },
    );
  } finally {
    probesLiveState.busy = false;
  }
}

/** Inspector-column mode — the four instrument pillars.  All four tabs are
 *  views over the ONE steering expression / probe roster; the split is
 *  presentational (each tab shows its own term/probe family):
 *    subspace — flat/affine fits (concept axes, personas)
 *    manifold — curved fits (emotions, months)
 *    sae      — resident sparse-autoencoder feature space
 *    lens     — the Jacobian-lens surface (JLensPanel) */
export type InspectorTab = "subspace" | "manifold" | "sae" | "lens";

export const inspectorState: { tab: InspectorTab } = $state({
  tab: "subspace",
});

export function setInspectorTab(tab: InspectorTab): void {
  inspectorState.tab = tab;
}

/** Toggle the live J-lens readout server-side. J-lens and SAE both follow the
 * same per-generation ``return_top_k`` value as the logit alternatives. */
export async function setLiveLens(enabled: boolean): Promise<void> {
  if (lensState.busy) return;
  lensState.busy = true;
  try {
    const out = await apiInstruments.setLive("lens", { enabled });
    lensState.layers = out.enabled && "layers" in out ? (out.layers ?? []) : null;
    if (!out.enabled) {
      lensState.readout = null;
      lensState.aggregate = null;
      lensState.aggHistory = [];
    }
  } catch (e) {
    pushToast(
      `lens live: ` +
        (e instanceof Error ? e.message : String(e)),
      { kind: "error" },
    );
  } finally {
    lensState.busy = false;
  }
}

export const lensFetch = createPreparationSlice("lens", "fetch", {
  label: "J-lens fetch",
  intervalMs: 1000,
  successMessage: "J-lens active · live",
  onSettled: async () => {
    await refreshSession();
    await refreshLensSources();
  },
});

/** The background Jacobian-lens fit.  On completion the session info is
 *  refreshed (``jlens_fitted`` flips, and the server's post-fit
 *  auto-enable lands in the lens block's ``live.layers`` → the toggle reads
 *  on).  A cancel stops the worker after its current estimator pass; any
 *  prior complete checkpoint stays resumable. */
export const lensFit = createPreparationSlice("lens", "fit", {
  label: "J-lens fit",
  intervalMs: 3000,
  successMessage: "J-lens fitted · live",
  onSettled: async () => {
    await refreshSession();
    await refreshLensSources();
  },
});

// =========================================================== steering ===
//
// One unified steer rack.  A steering vector is the K=2 flat case of a
// manifold, so every term is a position on a fitted geometry — one
// ``entries`` map of tagged ``SteerEntry`` (``mode: "subspace" | "manifold"``),
// one card, and one serializer.  Subspace (flat) terms share the rack-level
// ``subspaceAlong`` master (the merged affine subspace slides once); manifold
// (curved) terms keep their per-card along/onto.  The sidecars live here too:
// ``profiles`` + ``correlation`` (vector metadata) and ``catalog`` +
// ``loading`` / ``error`` (the required manifold HTTP surface).

/** Default shared subspace-along master — the ~0.5 coherent sweet spot
 *  (matches the engine's ``_SUBSPACE_GAIN`` calibration so a freshly
 *  racked concept at its pole lands at a usable strength). */
const DEFAULT_SUBSPACE_ALONG = 0.5;

export interface SteerRack {
  /** Rack key = atom display form (``honest``, ``ns/foo``, ``happy.sad``,
   *  ``personas``).  One entry per name; ``mode`` discriminates subspace
   *  (flat) vs manifold (curved).  Variant lives on the entry, not the key —
   *  matching the saklas parser's Steering.alphas semantics. */
  entries: Map<string, SteerEntry>;
  /** Advanced full-grammar expression. ``null`` means serialize the visual
   * rack; a string (including empty = explicitly unsteered) is authoritative. */
  customExpression: string | null;
  /** Shared "subspace along" master — the single slide magnitude every
   *  subspace (flat) term serializes with (the merged affine subspace has one
   *  slide).  Unclamped (a high-share layer is meant to overshoot; the engine
   *  bounds it via ``norm_cap``).  Defaults to the ~0.5 coherent sweet spot. */
  subspaceAlong: number;
  /** Per-profile metadata fetched from GET /profiles/{name}.
   * Populated lazily; absent until the user opens a strip's expander. */
  profiles: Map<string, VectorInfo>;
  /** Cosine matrix from GET /correlation; refreshed after each generation. */
  correlation: CorrelationData | null;
  /** Server-side catalog of available manifolds. */
  catalog: ManifoldInfo[];
  loading: boolean;
  error: string | null;
}

// SvelteMap from svelte/reactivity — plain Map mutations don't trigger
// Svelte 5 rune reactivity, so any rack add/remove or profile cache
// update wouldn't re-render the strips list.  SvelteMap.set/.delete is
// rune-tracked.  Inner-object property writes still aren't tracked, so
// callers that mutate an entry must reassign via .set(name, {...e, …}).
export const steerRack: SteerRack = $state({
  entries: new SvelteMap(),
  customExpression: null,
  subspaceAlong: DEFAULT_SUBSPACE_ALONG,
  profiles: new SvelteMap(),
  correlation: null,
  catalog: [],
  loading: false,
  error: null,
});

/** Server-derived list of registered vectors — names only.  Mirrors
 * sessionState.info?.profiles but kept as its own slice so panels that
 * only care about the list don't re-render when other session fields
 * change. */
export const vectorsState: { names: string[] } = $state({ names: [] });

export async function refreshVectorList(): Promise<void> {
  const r = await apiProfiles.list();
  vectorsState.names = r.profiles.map((v) => v.name);
  // Cache profile metadata — cheap, server already serialized.
  for (const v of r.profiles) {
    steerRack.profiles.set(v.name, v);
  }
}

export async function refreshCorrelation(
  names?: string[] | null,
): Promise<void> {
  try {
    const data = await apiProfiles.correlation(names);
    steerRack.correlation = data;
  } catch {
    steerRack.correlation = null;
  }
}

// ---------------------------------------------------- vector-mode mutators

function defaultSubspaceEntry(
  coords: number[] = [],
  label: string | null = null,
): SubspaceSteerEntry {
  return { mode: "subspace", coords, label, variant: "raw", trigger: "BOTH", enabled: true };
}

/** Reassign a subspace-mode (flat) entry through ``fn``; no-op if the entry is
 *  absent or is a manifold (curved) term. */
function mutateSubspace(
  name: string,
  fn: (e: SubspaceSteerEntry) => SubspaceSteerEntry,
): void {
  const e = steerRack.entries.get(name);
  if (e && e.mode === "subspace") steerRack.entries.set(name, fn(e));
}

// SvelteMap tracks .set/.delete; mutations on stored objects are NOT tracked,
// so each setter reassigns the entry via .set with a fresh spread.  This
// pattern is uniform across every rack mutator.

/** The shared "subspace along" master — one slide magnitude for every
 *  subspace (flat) term (the merged affine subspace slides once).  Adjusting
 *  it scales every flat term uniformly. */
export function setSubspaceAlong(along: number): void {
  enqueueOrApply(`subspace along ${along.toFixed(3)}`, () => {
    steerRack.subspaceAlong = along;
  });
}

/** Set a subspace term's free authoring coords (XYPad / slider drag) — clears
 *  the label-form binding so it serializes as a coord list. */
export function setSubspaceCoords(name: string, coords: number[]): void {
  enqueueOrApply(`subspace coords ${name}`, () => {
    mutateSubspace(name, (e) => ({ ...e, coords: [...coords], label: null }));
  });
}

/** Switch a subspace term to label-form (``<name>%<label>``).  ``label=null``
 *  reverts to coord-form; a non-null label mirrors the node's coords onto the
 *  entry so the XYPad still renders the position. */
export function setSubspaceLabel(name: string, label: string | null): void {
  enqueueOrApply(`subspace label ${name} ${label ?? "<null>"}`, () => {
    if (label === null) {
      mutateSubspace(name, (e) => ({ ...e, label: null }));
      return;
    }
    const info = manifoldByName(name);
    mutateSubspace(name, (e) => {
      if (!info) return { ...e, label };
      const idx = info.node_labels.indexOf(label);
      const coords = idx >= 0 && info.node_coords[idx] ? [...info.node_coords[idx]] : e.coords;
      return { ...e, label, coords };
    });
  });
}

export function setSubspaceVariant(name: string, variant: Variant): void {
  enqueueOrApply(`subspace variant ${name} ${variant}`, () => {
    mutateSubspace(name, (e) => ({ ...e, variant }));
  });
}

export function setSubspaceTrigger(name: string, trigger: Trigger): void {
  enqueueOrApply(`subspace trigger ${name} ${trigger}`, () => {
    mutateSubspace(name, (e) => ({ ...e, trigger }));
  });
}

export function setSubspaceEnabled(name: string, enabled: boolean): void {
  enqueueOrApply(`${enabled ? "enable" : "disable"} ${name}`, () => {
    mutateSubspace(name, (e) => ({ ...e, enabled }));
  });
}

/** Add a flat (subspace) term.  A 2-node concept defaults to its positive
 *  pole (label form); a higher-rank flat (personas) to the domain centroid;
 *  an uncatalogued typed name to its positive pole label.  Magnitude is the
 *  shared ``subspaceAlong`` master, not per-card. */
export function addSubspaceToRack(name: string): void {
  if (steerRack.entries.has(name)) return;
  steerRack.customExpression = null;
  const info = manifoldByName(name);
  let coords: number[] = [];
  let label: string | null = null;
  if (info && info.node_count === 2 && info.node_labels.length > 0) {
    label = info.node_labels[0];
    coords = info.node_coords?.[0] ? [...info.node_coords[0]] : [];
  } else if (info) {
    coords = manifoldCentroid(info);
  } else {
    const bare = name.includes("/") ? name.slice(name.indexOf("/") + 1) : name;
    label = bare.split(".")[0];
  }
  steerRack.entries.set(name, defaultSubspaceEntry(coords, label));
}

export function removeSubspaceFromRack(name: string): void {
  steerRack.entries.delete(name);
}

/** The canonical expression string the rack would send to the server.
 * Recomputed on demand; cheap.  Subspace terms first (at the shared
 * ``subspaceAlong`` master), then manifold (curved) terms. */
export function currentSteeringExpression(): string {
  return steerRack.customExpression
    ?? serializeExpression(steerRack.entries, steerRack.subspaceAlong);
}

/** Make a validated full-grammar expression authoritative.  The visual rack
 * cannot faithfully represent every binary projection/gate form, so mixing
 * the two would lie about what generation uses; switching modes clears it. */
export function applyCustomSteeringExpression(expression: string): void {
  enqueueOrApply("apply custom steering expression", () => {
    steerRack.entries.clear();
    steerRack.customExpression = expression;
  });
}

// ------------------------------------------------------ manifold catalog

/** Fetch the manifold catalog. */
export async function refreshManifoldList(): Promise<void> {
  steerRack.loading = true;
  try {
    const r = await apiManifolds.list();
    steerRack.catalog = r.manifolds;
    steerRack.error = null;
  } catch (e) {
    steerRack.catalog = [];
    steerRack.error = e instanceof Error ? e.message : String(e);
  } finally {
    steerRack.loading = false;
  }
}

/** Look up a catalog row by display name (``ns/name`` or bare name). */
export function manifoldByName(name: string): ManifoldInfo | null {
  for (const m of steerRack.catalog) {
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

// ---------------------------------------------------- manifold-mode mutators

/** Reassign a manifold-mode (curved) entry through ``fn``; no-op if the entry
 *  is absent or is a subspace (flat) term. */
function mutateManifold(
  name: string,
  fn: (e: ManifoldSteerEntry) => ManifoldSteerEntry,
): void {
  const e = steerRack.entries.get(name);
  if (e && e.mode === "manifold") steerRack.entries.set(name, fn(e));
}

/** Add a curved manifold to the rack at its domain centroid, along 0.5. */
export function addManifoldToRack(name: string): void {
  if (steerRack.entries.has(name)) return;
  steerRack.customExpression = null;
  const info = manifoldByName(name);
  const coords = info ? manifoldCentroid(info) : [];
  steerRack.entries.set(name, {
    mode: "manifold",
    blend: 0.5,
    onto: 0,
    coords,
    label: null,
    variant: "raw",
    trigger: "BOTH",
    enabled: true,
  });
}

export function removeManifoldFromRack(name: string): void {
  steerRack.entries.delete(name);
}

// ------------------------------------------------------- atom mutators
//
// The two single-direction atom families (``jlens/<word>``, ``sae/<id>``)
// rack identically — one α, one trigger, one enable, no geometry — so they
// share one mutator set parameterised by ``AtomMode``.  Only the two adds
// stay family-specific: they differ in key construction and validation.

/** Default α for a fresh atom chip.  Atoms run hotter than concept vectors
 *  (a single sharp direction, not a distributed contrast): ≈0.3 is the
 *  coherent sweet spot, ≥0.5 over-steers into repetition. */
export const ATOM_DEFAULT_ALPHA = 0.3;

/** Rack-key prefix per family — the atom's namespace segment in the
 *  steering grammar. */
export const ATOM_PREFIX: Record<AtomMode, string> = {
  jlens: "jlens/",
  sae: "sae/",
};

/** The rack mutations one atom card drives. */
export interface AtomRackActions {
  remove(name: string): void;
  setAlpha(name: string, alpha: number): void;
  setEnabled(name: string, enabled: boolean): void;
  setTrigger(name: string, trigger: Trigger): void;
}

/** Build the mutator set for one atom family.  ``label`` is the word the
 *  pending-queue bubble shows. */
function buildAtomActions(mode: AtomMode, label: string): AtomRackActions {
  /** Reassign an entry of this family through ``fn``; no-op on an absent
   *  key or an entry of another mode. */
  const mutate = (
    name: string,
    fn: (e: AtomSteerEntry) => AtomSteerEntry,
  ): void => {
    const e = steerRack.entries.get(name);
    if (e && e.mode === mode) steerRack.entries.set(name, fn(e));
  };
  return {
    remove(name) {
      steerRack.entries.delete(name);
    },
    setAlpha(name, alpha) {
      enqueueOrApply(`${label} alpha ${name} ${alpha.toFixed(3)}`, () => {
        mutate(name, (e) => ({ ...e, alpha }));
      });
    },
    setEnabled(name, enabled) {
      enqueueOrApply(`${enabled ? "enable" : "disable"} ${name}`, () => {
        mutate(name, (e) => ({ ...e, enabled }));
      });
    },
    setTrigger(name, trigger) {
      enqueueOrApply(`${label} trigger ${name} ${trigger}`, () => {
        mutate(name, (e) => ({ ...e, trigger }));
      });
    },
  };
}

const ATOM_ACTIONS: Record<AtomMode, AtomRackActions> = {
  jlens: buildAtomActions("jlens", "jlens"),
  sae: buildAtomActions("sae", "SAE"),
};

/** The rack mutators for one atom family. */
export function atomActions(mode: AtomMode): AtomRackActions {
  return ATOM_ACTIONS[mode];
}

function addAtomToRack(mode: AtomMode, id: string): void {
  const name = `${ATOM_PREFIX[mode]}${id}`;
  if (steerRack.entries.has(name)) return;
  steerRack.customExpression = null;
  steerRack.entries.set(name, {
    mode,
    alpha: ATOM_DEFAULT_ALPHA,
    trigger: "BOTH",
    enabled: true,
  } as AtomSteerEntry);
}

/** Add a J-lens token steering chip (``α jlens/<word>``).  Accepts a bare
 *  word or a full ``jlens/…`` atom; the rack key is the full atom.
 *  Dashboard callers validate through ``apiInstruments.validateLensToken``
 *  before this local mutation; the engine revalidates when it resolves the
 *  atom. */
export function addJLensToRack(word: string): void {
  const bare = word.trim().replace(/^jlens\//, "");
  if (!bare) return;
  addAtomToRack("jlens", bare);
}

/** Add a resident-SAE decoder-row steering chip (``α sae/<id>``). */
export function addSaeToRack(featureId: number): void {
  addAtomToRack("sae", String(featureId));
}

export function setManifoldBlend(name: string, blend: number): void {
  enqueueOrApply(`manifold blend ${name} ${blend.toFixed(3)}`, () => {
    mutateManifold(name, (e) => ({ ...e, blend }));
  });
}

/** Set the curved-manifold ``onto`` collapse fraction (the second
 *  coefficient). */
export function setManifoldOnto(name: string, onto: number): void {
  enqueueOrApply(`manifold onto ${name} ${onto.toFixed(3)}`, () => {
    mutateManifold(name, (e) => ({ ...e, onto }));
  });
}

export function setManifoldCoords(name: string, coords: number[]): void {
  // Pulling on the XYPad authors a free-form position; the term drops
  // its label-form binding (if any) so the canonical expression
  // serializes as a coord list and the snap-to-node dropdown shows
  // "(free position)" until the user picks one.
  enqueueOrApply(`manifold coords ${name}`, () => {
    mutateManifold(name, (e) => ({ ...e, coords: [...coords], label: null }));
  });
}

/** Switch the term to label-form (``<name>%<label>``).  ``label=null``
 *  clears the binding and reverts to coord-form on the next
 *  serialization.  When ``label`` is non-null the matching node's
 *  coords are mirrored onto ``coords`` so the XYPad still renders the
 *  position correctly. */
export function setManifoldLabel(name: string, label: string | null): void {
  enqueueOrApply(`manifold label ${name} ${label ?? "<null>"}`, () => {
    if (label === null) {
      mutateManifold(name, (e) => ({ ...e, label: null }));
      return;
    }
    const info = manifoldByName(name);
    mutateManifold(name, (e) => {
      if (!info) {
        // No catalog metadata — accept the label without mirroring
        // coords; downstream resolution happens server-side.
        return { ...e, label };
      }
      const idx = info.node_labels.indexOf(label);
      const coords = (idx >= 0 && info.node_coords[idx])
        ? [...info.node_coords[idx]]
        : e.coords;
      return { ...e, label, coords };
    });
  });
}

export function setManifoldTrigger(name: string, trigger: Trigger): void {
  enqueueOrApply(`manifold trigger ${name} ${trigger}`, () => {
    mutateManifold(name, (e) => ({ ...e, trigger }));
  });
}

export function setManifoldEnabled(name: string, enabled: boolean): void {
  enqueueOrApply(`manifold ${enabled ? "enable" : "disable"} ${name}`, () => {
    mutateManifold(name, (e) => ({ ...e, enabled }));
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
// Probe-inspector live trajectory trail depth (tokens).  Bounded so the
// fading polyline + the stored per-layer coords stay cheap; the oldest
// samples fade out as newer tokens push them off the ring.
const MAX_SUBSPACE_TRAIL = 64;

export interface ProbeRackState {
  /** Per-probe live state, keyed by registered probe name. */
  entries: Map<string, ProbeRackEntry>;
  sortMode: ProbeSortMode;
  /** Attached probe names (every listed probe is attached/active). */
  active: string[];
  loading: boolean;
  error: string | null;
}

export const probeRack: ProbeRackState = $state({
  entries: new SvelteMap(),
  // Alphabetical by default; value/change sorting is a dropdown opt-in.
  sortMode: "name",
  active: [],
  loading: false,
  error: null,
});

/** Primary scalar a probe's sparkline / sort tracks: the signed axis-0
 *  coordinate for a flat (subspace) probe, the [0,1] readout strength
 *  (axis 0 — mean fitted-layer probability) for a J-lens token probe, the [0,1]
 *  subspace fraction for a curved (manifold) probe. */
function _primaryScalar(info: ProbeInfo, reading: AnyReadingJSON): number {
  // The single-axis families ship their native one-channel reading, so the
  // shape answers this — no family sniffing, no eight constant geometry
  // fields to look past.
  if (isScalarReading(reading)) return reading.value;
  if (info.family === "geometry" && !info.is_affine) return reading.fraction;
  return reading.coords.length > 0 ? reading.coords[0] : 0;
}

/** Per-layer column for the expanded layer strip: axis-0 ``coords_per_layer``
 *  for a flat or J-lens probe (band-layer probability ``p_l`` for the
 *  latter — the strip's cell values), ``fraction_per_layer`` for a curved
 *  one. */
function _primaryPerLayer(
  info: ProbeInfo,
  reading: AnyReadingJSON,
): Record<string, number> {
  if (isScalarReading(reading)) return reading.per_layer ?? {};
  if (info.family === "geometry" && !info.is_affine) {
    return reading.fraction_per_layer ?? {};
  }
  const out: Record<string, number> = {};
  for (const [layer, c] of Object.entries(reading.coords_per_layer ?? {})) {
    out[layer] = Array.isArray(c) && c.length > 0 ? c[0] : 0;
  }
  return out;
}

/** Read one probe card through the token-hover overlay. Token rows retain the
 * primary scalar, multi-axis coordinates, and layer strip; lens/SAE cards can
 * additionally fill from retained live/on-demand hover snapshots. */
export function probeEntryForDisplay(name: string): ProbeRackEntry | undefined {
  const base = probeRack.entries.get(name);
  if (!base || !tokenHoverState.active) return base;

  let reading: AnyReadingJSON | null =
    tokenHoverState.probeReadings?.[name] ?? null;
  if (!reading) {
    let scalar = tokenHoverState.probes?.[name];
    const tokenCoords = tokenHoverState.coordsByProbe?.[name];
    if (scalar === undefined && tokenCoords?.[0] !== undefined) {
      scalar = tokenCoords[0];
    }
    const perLayer: Record<string, number> = {};
    for (const [layer, scores] of Object.entries(tokenHoverState.perLayerScores ?? {})) {
      const value = scores[name];
      if (typeof value === "number" && Number.isFinite(value)) perLayer[layer] = value;
    }

    if (scalar === undefined && name.startsWith("jlens/")) {
      const word = name.slice("jlens/".length);
      const aggregate = tokenHoverState.lensAggregate?.find(
        ([token]) => token === word || token.trim() === word,
      );
      if (aggregate) {
        scalar = aggregate[1];
        for (const [layer, tokens] of Object.entries(tokenHoverState.lensReadout ?? {})) {
          const hit = tokens.find(([token]) => token === word || token.trim() === word);
          if (hit) perLayer[layer] = hit[1];
        }
      }
    }
    if (scalar === undefined && base.info.family === "sae") {
      const feature = tokenHoverState.saeReadout?.find(
        (row) => row.id === (base.info as SaeProbeInfo).feature_id,
      );
      if (feature) {
        const maxAct = (base.info as SaeProbeInfo).max_act;
        scalar = maxAct != null && maxAct > 0
          ? feature.activation / maxAct
          : feature.activation;
        const layer = base.info.layers[0];
        if (layer !== undefined) perLayer[String(layer)] = scalar;
      }
    }

    if (scalar !== undefined || Object.keys(perLayer).length > 0) {
      const value = scalar ?? 0;
      if (base.info.family !== "geometry") {
        // The single-axis families synthesize their NATIVE reading, so the
        // hover overlay and the live stream carry the same shape.
        return {
          ...base,
          current: value,
          perLayer,
          reading: {
            value,
            unit: base.info.family === "lens"
              ? "mean_token_probability"
              : ((base.info as SaeProbeInfo).max_act != null
                ? "activation_over_max"
                : "raw_activation"),
            per_layer: perLayer,
            depth: null,
          },
          aggregate: null,
          savedAggregate: null,
        };
      }
      const usesCoords = base.info.is_affine;
      const coords = usesCoords
        ? (tokenCoords ?? [value])
        : [];
      const coordsPerLayer = Object.fromEntries(
        Object.entries(perLayer).map(([layer, v]) => [layer, [v]]),
      );
      reading = {
        fraction: usesCoords ? 0 : value,
        nearest: [],
        coords,
        residual: 0,
        fraction_per_layer: usesCoords ? {} : perLayer,
        coords_per_layer: usesCoords ? coordsPerLayer : {},
        residual_per_layer: {},
      };
    }
  }

  if (!reading) {
    return {
      ...base,
      current: 0,
      previous: 0,
      sparkline: [],
      perLayer: {},
      reading: null,
      aggregate: null,
      savedAggregate: null,
      nearest: [],
      trajectory: [],
    };
  }
  const current = _primaryScalar(base.info, reading);
  return {
    ...base,
    current,
    previous: current,
    sparkline: [current],
    perLayer: _primaryPerLayer(base.info, reading),
    reading,
    // ProbeCard's settled-only fields (residual and mini-map dot) should also
    // describe the hovered token, so the ephemeral view fills both slots.
    aggregate: reading,
    savedAggregate: null,
    nearest: _readingNearest(reading),
    trajectory: [],
  };
}

/** The nearest-node list — a GEOMETRY channel.  A one-channel reading has
 *  no nodes to be near, so it reports none rather than a constant empty
 *  field on the wire. */
function _readingNearest(reading: AnyReadingJSON): [string, number][] {
  return isScalarReading(reading) ? [] : reading.nearest;
}

/** A probe targets a 2-D-authored ``BoxDomain`` — the regime the mini-map
 *  renders.  Higher-dim and sphere/custom probes attach but skip it. */
function _probeIsMiniMapCandidate(info: ProbeInfo): info is GeometryProbeInfo {
  if (info.family !== "geometry" || info.intrinsic_dim !== 2) return false;
  const d = info.domain as { type?: string };
  return d?.type === "box" && !!info.node_coords && info.node_coords.length > 0;
}

/** Look up ``node_coords`` for a label.  Null when absent or the row carries
 *  no coords (unfitted discover).  Returns a copy so callers can push. */
function _lookupNodeCoords(
  info: GeometryProbeInfo, label: string,
): number[] | null {
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
    savedAggregate: null,
    nearest: [],
    trajectory: [],
    subspaceTrail: [],
  };
}

/** Per-probe saturation scale for the bar / layer cells / token tint — the
 *  axis-0 node-coordinate extent of the attached probe (``nodeCoordExtent``),
 *  or 1 when the probe isn't attached / carries no coords.  Token highlighting
 *  reads ``coords[0]`` (domain-frame) for every probe, flat or curved, so the
 *  node extent is the right normalizer in both cases. */
export function probeAxisScale(name: string, axis = 0): number {
  const info = probeRack.entries.get(name)?.info;
  if (!info || info.family !== "geometry") return 1;
  return nodeCoordExtent(info.node_coords, axis);
}

/** Shared raw-activation unit for SAE features without Neuronpedia
 *  ``maxActApprox`` metadata.  The SAE panel and transcript tint must use
 *  the same denominator: metadata-backed probes already arrive normalized
 *  from the server, while unhosted/local features remain raw and use the
 *  largest currently visible raw activation as their absolute 0..1 unit. */
export function saeRawFallbackScale(): number {
  let max = 0;
  for (const name of probeRack.active) {
    const entry = probeRack.entries.get(name);
    if (!entry || entry.info.family !== "sae") continue;
    if (entry.info.max_act != null) continue;
    max = Math.max(max, entry.current ?? 0);
  }
  for (const feature of saeReadoutForDisplay()) {
    const meta = saeState.meta.get(feature.id);
    if ((feature.max_act ?? meta?.max_act) != null) continue;
    max = Math.max(max, feature.activation);
  }
  return Math.max(max, 1);
}

/** Saturation scale for a highlight target.  The surprise sentinel keeps the
 *  fixed ``HIGHLIGHT_SAT`` cutoff (``surpriseScore`` is pre-scaled to it); a
 *  real probe normalizes by its per-axis node extent — an axis target
 *  (``personas[3]``) scales by that PC's own coordinate extent, so a tight
 *  axis isn't pinned saturated by a wider sibling axis. */
export function highlightScale(target: string | null): number {
  if (!target || target === SURPRISE_TARGET) return HIGHLIGHT_SAT;
  if (target.startsWith("jlens/")) return 1;
  if (target.startsWith("sae/")) {
    const info = probeRack.entries.get(target)?.info;
    return info?.family === "sae" && info.max_act != null
      ? 1
      : saeRawFallbackScale();
  }
  const { base, axis } = parseProbeTarget(target);
  return probeAxisScale(base, axis);
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

/** Fetch the attached-probe catalog. */
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
    hydrateProbeRackFromActiveNode();
    probeRack.error = null;
  } catch (e) {
    probeRack.entries.clear();
    probeRack.active = [];
    probeRack.error = e instanceof Error ? e.message : String(e);
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
  // Seed the highlight target when a probe is attached through the rack.
  if (highlightState.target === null) {
    highlightState.target = info.name;
  }
  return info;
}

/** Preserve a discovery card's visible reading when it becomes a pinned
 * probe. Attaching is server-authoritative but the first real probe event
 * arrives on the next generation; without this bridge, pinning a live card
 * made it flash to 0 and lose its sparkline/layer context. */
export function seedProbeDisplay(
  name: string,
  seed: {
    current: number;
    sparkline?: number[];
    perLayer?: Record<string, number>;
    reading?: AnyReadingJSON | null;
    aggregate?: AnyReadingJSON | null;
  },
): void {
  const prev = probeRack.entries.get(name);
  if (!prev) return;
  probeRack.entries.set(name, {
    ...prev,
    current: seed.current,
    previous: seed.current,
    sparkline: seed.sparkline ? [...seed.sparkline] : prev.sparkline,
    perLayer: seed.perLayer ? { ...seed.perLayer } : prev.perLayer,
    reading: seed.reading === undefined ? prev.reading : seed.reading,
    aggregate: seed.aggregate === undefined ? prev.aggregate : seed.aggregate,
    savedAggregate: null,
  });
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
      savedAggregate: null,
      trajectory: [],
      subspaceTrail: [],
    });
  }
}

/** Append one per-token reading per attached probe (the three families'
 *  ``instruments.*.readings`` merged by ``_mergedReadings``).  Drives the
 *  sparkline + per-layer strip + nearest readout + 2-D trajectory.  No-ops
 *  on undefined.
 *
 *  Reassigns each entry (rather than mutating in place) so the SvelteMap
 *  fires reactivity — a bare ``entry.current = v`` would freeze probe strips
 *  at zero through a whole generation. */
export function updateProbesFromReadings(
  readings: Record<string, AnyReadingJSON> | undefined,
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
    const nearest = _readingNearest(reading);
    if (_probeIsMiniMapCandidate(prev.info) && nearest.length > 0) {
      const xy = _lookupNodeCoords(prev.info, nearest[0][0]);
      if (xy) {
        trajectory = prev.trajectory.slice();
        trajectory.push(xy);
        if (trajectory.length > MAX_PROBE_TRAJECTORY) {
          trajectory.splice(0, trajectory.length - MAX_PROBE_TRAJECTORY);
        }
      }
    }
    // Probe-inspector trail: append this token's per-layer whitened subspace
    // coords (present only while the inspector requested them).  Stored across
    // all probed layers so the inspector reprojects for any scrubbed layer.
    let subspaceTrail = prev.subspaceTrail;
    const sc = isScalarReading(reading)
      ? undefined
      : reading.subspace_coords_per_layer;
    if (sc && Object.keys(sc).length > 0) {
      subspaceTrail = prev.subspaceTrail.slice();
      subspaceTrail.push({ perLayer: sc });
      if (subspaceTrail.length > MAX_SUBSPACE_TRAIL) {
        subspaceTrail.splice(0, subspaceTrail.length - MAX_SUBSPACE_TRAIL);
      }
    }
    probeRack.entries.set(name, {
      ...prev,
      sparkline,
      current: scalar,
      previous: prev.current,
      perLayer: _primaryPerLayer(prev.info, reading),
      reading,
      savedAggregate: null,
      nearest,
      trajectory,
      subspaceTrail,
    });
  }
}

/** Land the end-of-gen aggregate readings (the ``done`` event) — the settled
 *  ``ProbeReading`` per probe. */
export function setProbeAggregates(
  aggregates: Record<string, AnyReadingJSON> | undefined,
): void {
  if (!aggregates) return;
  for (const [name, agg] of Object.entries(aggregates)) {
    const prev = probeRack.entries.get(name);
    if (!prev) continue;
    probeRack.entries.set(name, {
      ...prev,
      aggregate: agg,
      savedAggregate: null,
      current: _primaryScalar(prev.info, agg),
      perLayer: _primaryPerLayer(prev.info, agg),
      nearest: _readingNearest(agg),
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
  const readings = _mergedReadings(m);
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

/** Rehydrate the scalar probe summary persisted on the selected Loom node.
 *
 * Rich per-layer ``ProbeReading`` payloads intentionally do not live in the
 * portable tree, but the aggregate scalar does.  Without this bridge a page
 * reload or branch navigation reset every attached card to a fake zero and
 * told the user to generate a token even while a generated node with saved
 * readings was selected.  Keep the scalar honest and mark it as historical;
 * ``ProbeCard`` explains why the layer strip is unavailable. */
function hydrateProbeRackFromActiveNode(): void {
  if (!loomTree.loaded || genStatus.active) return;
  const node = loomTree.active_node_id
    ? loomTree.nodes.get(loomTree.active_node_id)
    : undefined;
  const readings = node?.aggregate_readings ?? {};
  for (const [name, prev] of probeRack.entries) {
    const raw = readings[name];
    const value = typeof raw === "number" && Number.isFinite(raw) ? raw : null;
    probeRack.entries.set(name, {
      ...prev,
      current: value ?? 0,
      previous: prev.current,
      perLayer: {},
      reading: null,
      aggregate: null,
      savedAggregate: value,
      nearest: [],
      trajectory: [],
      subspaceTrail: [],
    });
  }
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
  // Surprise mode by default — the logit-pass tint works without any probes
  // loaded, so it is meaningful out of the box. ``localStorage`` overrides per
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
      const mergedReadings = _mergedReadings(m);
      const scores = m?.scores;
      const lensSnapshot = _lensReadoutSnapshot(m?.instruments.lens?.readout);
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
      if (mergedReadings) {
        const byProbe: Record<string, number[]> = {};
        for (const [pname, r] of Object.entries(mergedReadings)) {
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
        updateProbesFromReadings(mergedReadings);
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
        setProbeAggregates(_mergedReadings(msg.result?.measurements));
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
