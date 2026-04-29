<script lang="ts">
  // 16-bucket layer-norm histogram, matching the TUI's WHY footer.
  // Buckets are pre-shaped by lib/charts.ts::bucketize — this component
  // just renders the labels + bars.  The largest bucket scales to full
  // width; the rest scale linearly to their share of that maximum.
  //
  // Used by ProbeRack's WHY footer (single probe selected) and the
  // VectorStrip's "why" expander (per-vector view).

  import type { HistogramBucket } from "../charts";
  import { bucketMax } from "../charts";
  import Bar from "./Bar.svelte";

  interface Props {
    buckets: HistogramBucket[];
    /** Maximum bar width in pixels — passes through to <Bar>.  Pick a
     * value that suits the parent panel; defaults play well in the
     * 320px-wide rack column. */
    barWidth?: number;
    /** Bar height in pixels.  Smaller values densify the histogram so
     * 16 rows of buckets fit in a fixed-height slot. */
    barHeight?: number;
  }

  let {
    buckets,
    barWidth = 140,
    barHeight = 8,
  }: Props = $props();

  const max = $derived(bucketMax(buckets));
</script>

{#if buckets.length === 0}
  <div class="empty">no data</div>
{:else}
  <div class="hist" role="list">
    {#each buckets as b (b.label)}
      <div class="row" role="listitem">
        <span class="label" title="layers {b.lo}–{b.hi}">{b.label}</span>
        <Bar value={b.value} {max} width={barWidth} height={barHeight} />
        <span class="value">{b.value.toFixed(3)}</span>
      </div>
    {/each}
  </div>
{/if}

<style>
  .hist {
    display: flex;
    flex-direction: column;
    gap: 1px;
    font-size: var(--font-size-tiny);
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--row-gap);
  }
  .label {
    color: var(--fg-muted);
    width: 4em;
    text-align: right;
    white-space: nowrap;
  }
  .value {
    color: var(--fg-muted);
    width: 4em;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .empty {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    padding: 0.4em;
  }
</style>
