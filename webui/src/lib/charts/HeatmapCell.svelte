<script lang="ts">
  // Single heatmap cell — a colored square keyed by a [-1, 1] score.
  //
  // Used by the Correlation matrix (cosine similarity per probe pair)
  // and by the click-token drilldown drawer (per-layer × per-probe
  // grid for a single token).  The score-to-RGB mapping is centralized
  // in tokens.ts::scoreToRgb so highlight tints stay consistent across
  // surfaces.

  import { scoreToRgb } from "../tokens";

  interface Props {
    /** Heatmap value, expected in [-1, 1].  ``null`` and non-finite
     * values render as the empty-cell placeholder. */
    value: number | null | undefined;
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
    size = 14,
    title,
    showValue = false,
  }: Props = $props();

  const isNull = $derived(
    value === null || value === undefined || !Number.isFinite(value),
  );
  const bg = $derived(isNull ? "var(--bg-alt)" : scoreToRgb(value as number));
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
    font-size: 9px;
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
