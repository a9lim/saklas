<script lang="ts">
  // Canonical per-layer strip for every rack card. The cells deliberately
  // have no outlines: a small physical gap separates layers, while a visible
  // neutral fill keeps zero/near-zero readings legible against the card.

  import HeatmapCell from "../../lib/charts/HeatmapCell.svelte";

  interface LayerCell {
    layer: number;
    value: number | null | undefined;
    title: string;
  }

  let {
    cells,
    scale,
    ariaLabel,
    emptyMessage = "no data yet, generate a token first",
  }: {
    cells: LayerCell[];
    scale: number;
    ariaLabel: string;
    emptyMessage?: string;
  } = $props();

  const CELL_SIZE = 13;
</script>

<div class="layers" aria-label={ariaLabel}>
  {#if cells.length === 0}
    <div class="layers-status">{emptyMessage}</div>
  {:else}
    <span class="endcap" aria-hidden="true">L{cells[0].layer}</span>
    <div class="cells">
      {#each cells as cell (cell.layer)}
        <HeatmapCell
          value={cell.value}
          {scale}
          size={CELL_SIZE}
          title={cell.title}
        />
      {/each}
    </div>
    <span class="endcap" aria-hidden="true">L{cells[cells.length - 1].layer}</span>
  {/if}
</div>

<style>
  .layers {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
    overflow-x: auto;
    white-space: nowrap;
    padding-top: var(--space-3);
    padding-bottom: var(--space-2);
    scrollbar-color: var(--glass-strong) transparent;
    scrollbar-width: thin;
  }
  .layers-status {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    padding: var(--space-1) 0;
  }
  .cells {
    display: flex;
    gap: 1px;
    flex: 0 0 auto;
  }
  .endcap {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
</style>
