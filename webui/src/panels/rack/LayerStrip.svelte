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
    positiveColor,
    negativeColor,
  }: {
    cells: LayerCell[];
    scale: number;
    ariaLabel: string;
    emptyMessage?: string;
    positiveColor?: string;
    negativeColor?: string;
  } = $props();

  const CELL_SIZE = 13;
  let activeIndex = $state(0);
  let focused = $state(false);

  $effect(() => {
    if (activeIndex >= cells.length) activeIndex = Math.max(0, cells.length - 1);
  });

  function onKeydown(ev: KeyboardEvent): void {
    if (cells.length === 0) return;
    if (ev.key === "ArrowRight") activeIndex = Math.min(cells.length - 1, activeIndex + 1);
    else if (ev.key === "ArrowLeft") activeIndex = Math.max(0, activeIndex - 1);
    else if (ev.key === "Home") activeIndex = 0;
    else if (ev.key === "End") activeIndex = cells.length - 1;
    else return;
    ev.preventDefault();
  }
</script>

<div class="layer-strip">
  <div
    class="layers"
    role="slider"
    aria-label={ariaLabel}
    aria-valuemin="0"
    aria-valuemax={Math.max(0, cells.length - 1)}
    aria-valuenow={activeIndex}
    aria-valuetext={cells[activeIndex]?.title ?? emptyMessage}
    tabindex={cells.length > 0 ? 0 : undefined}
    onfocus={() => (focused = true)}
    onblur={() => (focused = false)}
    onkeydown={onKeydown}
  >
    {#if cells.length === 0}
      <div class="layers-status">{emptyMessage}</div>
    {:else}
      <span class="endcap" aria-hidden="true">L{cells[0].layer}</span>
      <div class="cells">
        {#each cells as cell, i (cell.layer)}
          <HeatmapCell
            value={cell.value}
            {scale}
            size={CELL_SIZE}
            title={cell.title}
            {positiveColor}
            {negativeColor}
            active={focused && i === activeIndex}
          />
        {/each}
      </div>
      <span class="endcap" aria-hidden="true">L{cells[cells.length - 1].layer}</span>
    {/if}
  </div>
  {#if focused && cells[activeIndex]}
    <div class="keyboard-readout" aria-live="polite">
      {cells[activeIndex].title}
    </div>
  {/if}
</div>

<style>
  .layer-strip {
    min-width: 0;
  }
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
  .keyboard-readout {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    padding-bottom: var(--space-2);
  }
</style>
