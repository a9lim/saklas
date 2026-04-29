// Chart helpers — the visual primitives live as Svelte components in
// charts/, this module owns the data-shaping logic shared between them.
//
// ``bucketize`` is a JS port of saklas.core.histogram.bucketize, kept in
// sync by literal port (the Python code is 16 lines).  The constant
// HIST_BUCKETS=16 mirrors the server-side knob so a probe's WHY footer
// fits the same fixed-height block on either surface.

/** One knob for every per-layer histogram surface (TUI WHY footer, web
 * probe rack, ``saklas vector why`` CLI).  Chosen so the full profile
 * of any supported model collapses into a fixed 16-row block. */
export const HIST_BUCKETS = 16;

export interface HistogramBucket {
  /** Inclusive low layer index of the bucket. */
  lo: number;
  /** Inclusive high layer index of the bucket. */
  hi: number;
  /** Mean ``||baked||`` across the bucket's layers. */
  value: number;
  /** Pre-formatted label: "L05" for single-layer buckets, "L05-08" for
   * multi-layer.  Zero-padded to the model's max layer-index width. */
  label: string;
}

/** Collapse a per-layer norm map into ``buckets`` evenly-sized groups.
 *
 * ``norms`` is keyed by layer-index string (matching the server's
 * ``per_layer_norms`` JSON shape) or by integer; both are accepted to
 * avoid forcing the caller to coerce.  When the layer count is already
 * <= ``buckets``, each layer becomes its own bucket. */
export function bucketize(
  norms: Record<number | string, number>,
  buckets: number = HIST_BUCKETS,
): HistogramBucket[] {
  // Build a sorted [layer, value] list — string keys round-trip via
  // parseInt to land on the same int order.
  const sorted: Array<[number, number]> = [];
  for (const [k, v] of Object.entries(norms)) {
    const layer = typeof k === "number" ? k : parseInt(k, 10);
    if (Number.isNaN(layer)) continue;
    sorted.push([layer, v]);
  }
  sorted.sort((a, b) => a[0] - b[0]);
  const n = sorted.length;
  if (n === 0) return [];

  // Width matches the model's max layer index — zero-pad ``L00`` style.
  const maxLayer = sorted[n - 1][0];
  const padW = String(maxLayer).length;

  if (n <= buckets) {
    return sorted.map(([l, v]) => ({
      lo: l,
      hi: l,
      value: v,
      label: `L${String(l).padStart(padW, "0")}`,
    }));
  }
  const out: HistogramBucket[] = [];
  for (let i = 0; i < buckets; i++) {
    const lo = Math.floor((i * n) / buckets);
    const hi = Math.floor(((i + 1) * n) / buckets);
    const chunk = sorted.slice(lo, hi);
    if (chunk.length === 0) continue;
    const mean = chunk.reduce((s, [, v]) => s + v, 0) / chunk.length;
    const loIdx = chunk[0][0];
    const hiIdx = chunk[chunk.length - 1][0];
    const label =
      loIdx === hiIdx
        ? `L${String(loIdx).padStart(padW, "0")}`
        : `L${String(loIdx).padStart(padW, "0")}-${String(hiIdx).padStart(padW, "0")}`;
    out.push({ lo: loIdx, hi: hiIdx, value: mean, label });
  }
  return out;
}

/** Largest absolute value across a bucket list — used to scale bar
 * widths in <Histogram>.  Returns 0 for empty input so divide-by-zero
 * sites can short-circuit. */
export function bucketMax(buckets: HistogramBucket[]): number {
  let m = 0;
  for (const b of buckets) {
    const a = Math.abs(b.value);
    if (a > m) m = a;
  }
  return m;
}
