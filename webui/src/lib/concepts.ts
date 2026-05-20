// Concept-catalog helpers shared by the steering / probe pickers.
//
// Two facts about every bundled concept are already on the wire and the
// pickers present them: which *category* it belongs to (a category-valued
// tag) and, for bipolar axes, its two *poles* (the canonical name split on
// ``BIPOLAR_SEP`` — a dot).  See docs/plans/webui-overhaul.md §"Category
// data" and saklas/core/session.py ``canonical_concept_name``.

import type { LocalPackInfo } from "./types";

/** The seven fixed bundled categories, in display order.  Matches the
 * category-valued tags in saklas/data/vectors/<concept>/pack.json and the
 * grouping in AGENTS.md §"Bundled concepts". */
export const CATEGORY_ORDER = [
  "affect",
  "epistemic",
  "alignment",
  "register",
  "social_stance",
  "cultural",
  "identity",
] as const;

export type Category = (typeof CATEGORY_ORDER)[number] | "other";

/** Human-facing section labels.  ``other`` catches user-authored / HF
 * packs whose tags miss the fixed set. */
export const CATEGORY_LABELS: Record<Category, string> = {
  affect: "Affect",
  epistemic: "Epistemic",
  alignment: "Alignment",
  register: "Register",
  social_stance: "Social stance",
  cultural: "Cultural",
  identity: "Identity",
  other: "Other",
};

/** Categories expanded by default in the picker — the rest collapse so
 * the whole catalog fits without a long scroll. */
export const DEFAULT_EXPANDED: ReadonlySet<Category> = new Set<Category>([
  "affect",
  "epistemic",
]);

const _CATEGORY_SET = new Set<string>(CATEGORY_ORDER);

/** First tag that names one of the seven fixed categories, else "other". */
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

/** Bipolar axis poles.  ``angry.calm`` → positive ``angry`` (α > 0),
 * negative ``calm`` (α < 0).  No dot → monopolar (``negative`` is null). */
export interface Poles {
  positive: string;
  negative: string | null;
}

export function polesOf(name: string): Poles {
  const dot = name.indexOf(".");
  if (dot < 0) return { positive: name, negative: null };
  return { positive: name.slice(0, dot), negative: name.slice(dot + 1) };
}

/** Resting α for a concept — the pack's ``recommended_alpha`` (loose
 * passthrough on ``LocalPackInfo``), defaulting to 0.5 when absent. */
export function recommendedAlpha(row: LocalPackInfo): number {
  const raw = row["recommended_alpha"];
  const n = typeof raw === "number" ? raw : Number(raw);
  return Number.isFinite(n) && n !== 0 ? n : 0.5;
}
