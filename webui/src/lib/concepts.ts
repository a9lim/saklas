// Concept-catalog helpers shared by the steering / probe pickers.
//
// Two facts about every bundled concept are already on the wire and the
// pickers present them: which *category* it belongs to (a category-valued
// tag) and, for bipolar axes, its two *poles* (the canonical name split on
// ``BIPOLAR_SEP`` — a dot).  See docs/plans/webui-overhaul.md §"Category
// data" and saklas/core/session.py ``canonical_concept_name``.

/** The four fixed bundled categories, in display order.  Matches the
 * category-valued tags in saklas/data/manifolds/<concept>/manifold.json
 * and the grouping in AGENTS.md §"Bundled concepts" (affect folded into the
 * `emotions` manifold; social_stance / identity were cut in the 4.0 regen). */
export const CATEGORY_ORDER = [
  "epistemic",
  "alignment",
  "register",
  "cultural",
] as const;

export type Category = (typeof CATEGORY_ORDER)[number] | "other";

/** Human-facing section labels.  ``other`` catches user-authored / HF
 * packs whose tags miss the fixed set. */
export const CATEGORY_LABELS: Record<Category, string> = {
  epistemic: "Epistemic",
  alignment: "Alignment",
  register: "Register",
  cultural: "Cultural",
  other: "Other",
};

/** Categories expanded by default in the picker — the rest collapse so
 * the whole catalog fits without a long scroll. */
export const DEFAULT_EXPANDED: ReadonlySet<Category> = new Set<Category>([
  "epistemic",
]);

const _CATEGORY_SET = new Set<string>(CATEGORY_ORDER);

/** First tag that names one of the four fixed categories, else "other". */
export function categoryOf(tags: readonly string[] | undefined): Category {
  if (Array.isArray(tags)) {
    for (const t of tags) {
      if (typeof t === "string" && _CATEGORY_SET.has(t)) {
        return t as Category;
      }
    }
  }
  return "other";
}

/** Bipolar axis poles.  ``formal.casual`` → positive ``formal`` (α > 0),
 * negative ``casual`` (α < 0).  No dot → monopolar (``negative`` is null). */
export interface Poles {
  positive: string;
  negative: string | null;
}

export function polesOf(name: string): Poles {
  const dot = name.indexOf(".");
  if (dot < 0) return { positive: name, negative: null };
  return { positive: name.slice(0, dot), negative: name.slice(dot + 1) };
}

/** Resting α for a concept — a loose ``recommended_alpha`` passthrough on
 * any catalog row that carries one, defaulting to 0.5 when absent. */
export function recommendedAlpha(row: { recommended_alpha?: unknown }): number {
  const raw = row.recommended_alpha;
  const n = typeof raw === "number" ? raw : Number(raw);
  return Number.isFinite(n) && n !== 0 ? n : 0.5;
}
