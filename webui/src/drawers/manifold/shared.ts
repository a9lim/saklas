// Vocabulary shared by the three manifold-authoring forms.
//
// The drawer owns the identity fields (namespace / name / description) and
// hands them down read-only; each form slugs them at submit.  The discover
// tuning block is genuinely shared — the auto-generated tab and the
// custom tab's auto-domain switch route through the same discover-create
// endpoint, so they build the same hyperparams and validate them the same
// way.

/** Raw (unslugged) identity inputs, owned by the drawer shell. */
export interface ManifoldIdentity {
  namespace: string;
  name: string;
  description: string;
}

/** The engine's name/label slug: lowercase, ``[a-z0-9._-]``, trimmed. */
export function slug(s: string): string {
  return s
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^[_.-]+|[_.-]+$/g, "");
}

/** Namespace defaulting to ``local``, plus the slugged name. */
export function identitySlugs(id: ManifoldIdentity): {
  namespace: string;
  name: string;
  description: string;
} {
  return {
    namespace: slug(id.namespace) || "local",
    name: slug(id.name),
    description: id.description.trim(),
  };
}

/** Whitespace/comma-separated token list (concepts, values). */
export function parseTokens(text: string): string[] {
  return text
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export type DiscoverFitMode = "pca" | "spectral" | "auto";

/** The fit-method choice plus its method-specific hyperparameters. */
export interface DiscoverTuning {
  fitMode: DiscoverFitMode;
  maxDim: number;
  varThreshold: number;
  kNN: number | null;
  bandwidth: number | null;
}

/** ``auto`` is the friendly default — ``select_topology`` picks
 *  flat / curved / periodic per-model, so a newcomer needn't know which
 *  geometry their concepts want.  pca / spectral pin it for power users. */
export function defaultTuning(): DiscoverTuning {
  return {
    fitMode: "auto",
    maxDim: 8,
    varThreshold: 0.7,
    kNN: null,
    bandwidth: null,
  };
}

/** ``max_dim`` (the layout-dim cap) is the one knob every mode honors.
 *  The rest are method-specific; ``auto`` accepts the union (the server
 *  sanitizer drops whichever the resolved geometry doesn't consume), so we
 *  forward every knob the user actually set and let the backend fill
 *  data-driven defaults for the rest (median k-NN distance,
 *  ``max(5, ceil(log K))``). */
export function tuningHyperparams(t: DiscoverTuning): Record<string, number> {
  const hp: Record<string, number> = { max_dim: t.maxDim };
  if (t.fitMode === "pca" || t.fitMode === "auto") {
    hp.var_threshold = t.varThreshold;
  }
  if (t.fitMode === "spectral" || t.fitMode === "auto") {
    if (t.kNN !== null && t.kNN > 0) hp.k_nn = t.kNN;
    if (t.bandwidth !== null && t.bandwidth > 0) hp.bandwidth = t.bandwidth;
  }
  return hp;
}

export function tuningMessages(t: DiscoverTuning): string[] {
  const messages: string[] = [];
  if (t.maxDim < 1) messages.push("max dim ≥1");
  if (
    (t.fitMode === "pca" || t.fitMode === "auto") &&
    (t.varThreshold <= 0 || t.varThreshold > 1)
  ) {
    messages.push("variance ∈ (0, 1]");
  }
  return messages;
}

/** Pull the human-readable message out of an SSE progress event. */
export function progressMessage(ev: { data?: unknown }): string | null {
  if (!ev.data || typeof ev.data !== "object") return null;
  return (ev.data as { message?: string }).message ?? null;
}
