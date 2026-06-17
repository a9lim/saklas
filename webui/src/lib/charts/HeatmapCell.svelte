<script lang="ts">
  // Single heatmap cell — a colored square keyed by a score and a
  // saturation ``scale`` (the value at which the ramp is full).
  //
  // Used by the Correlation matrix (cosine similarity per probe pair)
  // and by the click-token drilldown drawer (per-layer × per-probe
  // grid for a single token).  The score-to-RGB mapping is centralized
  // in tokens.ts::scoreToRgb so highlight tints stay consistent across
  // surfaces.  Cosine grids leave ``scale`` at its ``HIGHLIGHT_SAT``
  // default ([-1, 1] data); probe-coordinate grids pass a per-probe
  // node-coordinate extent so the cells aren't pinned full from coords
  // that run well past ±1.

  import { scoreToRgb, HIGHLIGHT_SAT } from "../tokens";

  interface Props {
    /** Heatmap value.  Read against ``scale`` (full color at ``±scale``);
     * ``null`` and non-finite values render as the empty-cell placeholder. */
    value: number | null | undefined;
    /** Saturation scale — the value at which the ramp reaches full color.
     * Defaults to ``HIGHLIGHT_SAT`` (the [-1, 1] cosine-grid regime). */
    scale?: number;
    /** Cell width in pixels. */
    size?: number;
    /** Optional tooltip text override.  When omitted, the value is
     * formatted to 3 decimals. */
    title?: string;
    /** When true, render the numeric value inside the cell.  Useful
     * for the correlation matrix where the magnitude carries info; off
     * for fine-grained per-token grids where the cells are too small
     * to read text. */
    showValue?: boolean;
  }

  let {
    value,
    scale = HIGHLIGHT_SAT,
    size = 14,
    title,
    showValue = false,
  }: Props = $props();

  const isNull = $derived(
    value === null || value === undefined || !Number.isFinite(value),
  );
  const bg = $derived(
    isNull ? "var(--bg-alt)" : scoreToRgb(value as number, scale),
  );
  const tip = $derived(
    title ?? (isNull ? "—" : (value as number).toFixed(3)),
  );
  const text = $derived(showValue && !isNull ? (value as number).toFixed(2) : "");
</script>

<div
  class="cell"
  style="width: {size}px; height: {size}px; background: {bg};"
  title={tip}
  role="img"
  aria-label={tip}
>
  {#if text}
    <span class="t">{text}</span>
  {/if}
</div>

<style>
  .cell {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: help;
    font-size: var(--text-2xs);
    color: var(--fg);
    /* Slight border so adjacent cells don't visually merge into one
     * bigger color block — kept hairline so the color dominates. */
    border: 1px solid var(--bg);
  }
  .t {
    font-variant-numeric: tabular-nums;
    text-shadow: 0 0 2px rgba(0, 0, 0, 0.6);
  }
</style>
