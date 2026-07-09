// Serialization of the unified steering-expression rack.
//
// Mirrors saklas/core/steering_expr.py (serialize direction only):
//
//   expr     := term (("+" | "-") term)*
//   term     := [coeff "*"?] ["!"] selector ["@" trigger]
//   selector := atom (("~" | "|") atom | "%" coord_list)?
//   coord_list := signed_num ("," signed_num)*
//   atom     := [ns "/"] NAME ["." NAME] [":" variant]
//   trigger  := before|after|both|thinking|response|prompt|generated
//   variant  := raw | sae | sae-<release> | role | role-<name>
//             | from | from-<safe-source-model>
//
// A ``%`` selector is the manifold operator — ``<manifold> % a,b,...``
// places generation at a point of a fitted steering manifold.  The
// coefficient is the blend fraction.
//
// One unified steer rack is the source of truth — a ``Map<string,
// SteerEntry>`` where each entry is ``mode: "subspace"`` (flat: coords,
// label, variant, trigger, enabled — magnitude is the rack-level
// ``subspaceAlong`` master) or ``mode: "manifold"`` (curved: blend, onto,
// coords, label, variant, trigger, enabled).  Both serialize as ``%`` terms;
// the magnitude slot is the shared ``subspaceAlong`` for subspace terms and
// the per-card ``along[,onto]`` for manifold terms.

import type {
  JLensSteerEntry,
  ManifoldSteerEntry,
  SteerEntry,
  SubspaceSteerEntry,
  Trigger,
  Variant,
} from "./types";

// ----------------------------------------------- triggers ----------------

/** Map UI Trigger values to the keyword the formatter emits.  Aliases
 * (PROMPT->before, GENERATED->response) collapse onto the canonical
 * render so round-trips are deterministic.  BOTH is the default and is
 * omitted from the output. */
const TRIGGER_TO_KEYWORD: Record<Trigger, string | null> = {
  BOTH: null,
  BEFORE: "before",
  AFTER: "after",
  THINKING: "thinking",
  RESPONSE: "response",
  PROMPT: "before",
  GENERATED: "response",
};

// ----------------------------------------------- formatter ---------------

/** Render the rack as a canonical expression string.  Disabled entries are
 * skipped; subspace (flat) terms emit first, then manifold (curved) terms.
 * Every term is a ``%`` position; the coefficient slot is the shared
 * ``subspaceAlong`` master for subspace terms and the per-card
 * ``along[,onto]`` for manifold terms.  Empty/all-disabled racks return "". */
export function serializeExpression(
  rack: Map<string, SteerEntry>,
  subspaceAlong = 1,
): string {
  const parts: string[] = [];
  for (const [name, entry] of rack) {
    if (entry.enabled && entry.mode === "subspace") {
      parts.push(formatSubspaceTerm(name, entry, subspaceAlong));
    }
  }
  for (const [name, entry] of rack) {
    if (entry.enabled && entry.mode === "jlens") {
      parts.push(formatJLensTerm(name, entry));
    }
  }
  for (const [name, entry] of rack) {
    if (entry.enabled && entry.mode === "manifold") {
      parts.push(formatManifoldTerm(name, entry));
    }
  }
  if (parts.length === 0) return "";
  // First term keeps its leading sign verbatim; subsequent terms join
  // with " + " or " - " depending on coefficient sign.
  let out = parts[0];
  for (let i = 1; i < parts.length; i++) {
    const p = parts[i];
    if (p.startsWith("-")) out += " - " + p.slice(1).trimStart();
    else out += " + " + p;
  }
  return out;
}

/** Render one subspace (flat) term — ``<subspaceAlong> name[:variant]%<pos>``
 *  plus an optional ``@trigger``.  All subspace terms share the one
 *  ``subspaceAlong`` magnitude (the merged affine subspace slides once);
 *  relative weight between them lives in how far each position sits from
 *  neutral.  ``<pos>`` is the label form (``personas%hacker``) when
 *  ``entry.label`` is set, else the comma-joined coord list. */
function formatSubspaceTerm(
  name: string,
  entry: SubspaceSteerEntry,
  subspaceAlong: number,
): string {
  const position = entry.label ?? entry.coords.map((c) => formatCoeff(c)).join(",");
  const selector = `${nameWithVariant(name, entry.variant)}%${position}`;
  return `${formatCoeff(subspaceAlong)} ${selector}${formatTriggerSuffix(entry.trigger)}`;
}

function nameWithVariant(name: string, variant: Variant): string {
  return variant === "raw" ? name : `${name}:${variant}`;
}

function formatCoeff(coeff: number): string {
  // Number.toString trims trailing zeros while preserving precision.
  // The leading sign rides the term; the joiner above splits +/-.
  if (Number.isNaN(coeff) || !Number.isFinite(coeff)) return "0";
  return String(coeff);
}

function formatTriggerSuffix(trigger: Trigger): string {
  const kw = TRIGGER_TO_KEYWORD[trigger];
  return kw ? `@${kw}` : "";
}

/** Render one J-lens token term — the plain atom ``<alpha> jlens/<word>``
 *  plus an optional ``@trigger``.  The rack key is the full atom (the
 *  engine resolves it through ``register_jlens_direction``); per-chip
 *  ``alpha`` because lens atoms run hotter than concept vectors. */
function formatJLensTerm(name: string, entry: JLensSteerEntry): string {
  return `${formatCoeff(entry.alpha)} ${name}${formatTriggerSuffix(entry.trigger)}`;
}

/** Render one manifold (curved) rack entry — ``<along[,onto]>
 *  <name>[:variant]%<position>`` plus an optional ``@trigger`` suffix.
 *  Coefficient is the per-card blend fraction (``-`` rides the joiner).
 *  ``<position>`` is the label form (``emotions%happy``) when ``entry.label`` is
 *  set; otherwise the comma-joined coord list (``emotions%0.3,0.8``). */
function formatManifoldTerm(name: string, entry: ManifoldSteerEntry): string {
  const position = entry.label
    ? entry.label
    : entry.coords.map((c) => formatCoeff(c)).join(",");
  const selector = `${nameWithVariant(name, entry.variant)}%${position}`;
  // Coefficient slot: ``along`` alone, or ``along,onto`` when onto is set
  // (> 0) — the curved-manifold residual-collapse fraction.
  const coeff =
    (entry.onto ?? 0) > 0
      ? `${formatCoeff(entry.blend)},${formatCoeff(entry.onto)}`
      : formatCoeff(entry.blend);
  return `${coeff} ${selector}${formatTriggerSuffix(entry.trigger)}`;
}
