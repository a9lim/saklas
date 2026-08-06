// One unified read-side rack — every probe shape.
//
// A 2-node concept axis is the rank-1 case, a discover / curved fit the
// rank-R case, and the two single-channel families (J-lens token probes,
// SAE features) ship their native ``ScalarReading``.  Each entry carries
// the server ``ProbeInfo`` (with the ``is_affine`` flat-vs-curved flag the
// cards classify on), a sparkline of the primary scalar, the latest
// per-token ``reading`` + end-of-gen ``aggregate``, the most-recent
// ``nearest`` list, and — for 2-D box probes — an inferred per-token
// ``trajectory`` for the mini-map.

import { SvelteMap } from "svelte/reactivity";
import { apiProbes } from "../api";
import type {
  AnyReadingJSON,
  GeometryProbeInfo,
  ProbeInfo,
  ProbeRackEntry,
  ProbeSortMode,
  SaeProbeInfo,
} from "../types";
import { isScalarReading } from "../types";
import {
  SURPRISE_TARGET,
  HIGHLIGHT_SAT,
  nodeCoordExtent,
  parseProbeTarget,
} from "../tokens";
import { genStatus } from "./chat.svelte";
import {
  saeReadoutForDisplay,
  saeState,
  tokenHoverState,
} from "./instruments.svelte";
import { loomTree } from "../stores.svelte";

//
// One unified read-side rack — every probe shape (a 2-node concept axis is
// the rank-1 case, a discover / curved fit the rank-R case).  Each entry
// carries the server ``ProbeInfo`` (with the ``is_affine`` flat-vs-curved
// flag the cards classify on), a sparkline of the primary scalar, the
// latest per-token ``reading`` + end-of-gen ``aggregate`` (one
// ``ProbeReadingJSON`` shape), the most-recent ``nearest`` list, and — for
// 2-D box probes — an inferred per-token ``trajectory`` for the mini-map.

/** Sparkline depth (tokens).  Exported because the lens aggregate history
 *  and the SAE activation history are capped to the same window — the
 *  workspace/discovery cards render the same strip shape as a probe. */
export const MAX_SPARKLINE = 60;
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
 *  ``instruments.*.readings`` merged by ``mergedReadings``).  Drives the
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

// ------------------------------------------ loom-node rehydration ---

/** Rehydrate the scalar probe summary persisted on the selected Loom node.
 *
 * Rich per-layer ``ProbeReading`` payloads intentionally do not live in the
 * portable tree, but the aggregate scalar does.  Without this bridge a page
 * reload or branch navigation reset every attached card to a fake zero and
 * told the user to generate a token even while a generated node with saved
 * readings was selected.  Keep the scalar honest and mark it as historical;
 * ``ProbeCard`` explains why the layer strip is unavailable. */
export function hydrateProbeRackFromActiveNode(): void {
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

// --------------------------------------- transcript highlight -------
//
// Which probe (or the surprise sentinel) tints the transcript.  It
// lives with the rack because every scale helper above resolves a
// highlight target, and attach/detach seed and clear it.

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
