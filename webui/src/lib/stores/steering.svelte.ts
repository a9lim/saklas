// One unified steer rack.
//
// A steering vector is the K=2 flat case of a manifold, so every term is a
// position on a fitted geometry — one ``entries`` map of tagged
// ``SteerEntry`` (``mode: "subspace" | "manifold" | "jlens" | "sae"``), one
// card, and one serializer.  Subspace (flat) terms share the rack-level
// ``subspaceAlong`` master (the merged affine subspace slides once);
// manifold (curved) terms keep their per-card along/onto; the two
// single-direction atom families share one mutator set.
//
// The sidecars live here too: ``profiles`` + ``correlation`` (vector
// metadata) and ``catalog`` + ``loading`` / ``error`` (the manifold HTTP
// surface).

import { SvelteMap } from "svelte/reactivity";
import { apiManifolds, apiProfiles } from "../api";
import type { CorrelationData, ManifoldInfo, VectorInfo } from "../api";
import type {
  AtomMode,
  AtomSteerEntry,
  ManifoldSteerEntry,
  SteerEntry,
  SubspaceSteerEntry,
  Trigger,
  Variant,
} from "../types";
import { serializeExpression } from "../expression";
import { enqueueOrApply } from "./pending.svelte";

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

// -------------------------------------------- subspace-mode mutators -

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

// ------------------------------------------------ manifold catalog --

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

// -------------------------------------------- manifold-mode mutators -

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

// ------------------------------------------------------ atom mutators
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
