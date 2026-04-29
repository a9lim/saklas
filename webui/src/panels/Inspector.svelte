<script lang="ts">
  import { inspectorTokens } from "../lib/stores";

  function scoreToRgb(v: number): string {
    const a = Math.max(-1, Math.min(1, v));
    if (a >= 0) return `rgb(0, ${Math.round(a * 200)}, 0)`;
    return `rgb(${Math.round(-a * 200)}, 0, 0)`;
  }

  $: probeOrder = (() => {
    if ($inspectorTokens.length === 0) return [];
    const first = $inspectorTokens[0].scores;
    const layerKey = Object.keys(first)[0];
    return Object.keys(first[layerKey] ?? {}).sort();
  })();

  $: layers = (() => {
    const set = new Set<number>();
    for (const tok of $inspectorTokens) {
      for (const layer of Object.keys(tok.scores)) set.add(parseInt(layer, 10));
    }
    return Array.from(set).sort((a, b) => a - b);
  })();
</script>

<div class="header">PER-TOKEN × PER-LAYER × PER-PROBE</div>
{#if $inspectorTokens.length === 0 || probeOrder.length === 0}
  <div class="hint">Generate a response with at least one probe loaded to populate the heatmap.</div>
{:else}
  <div class="grid"
       style="grid-template-columns: auto repeat({$inspectorTokens.length * probeOrder.length}, 14px);">
    <div></div>
    {#each $inspectorTokens as tok}
      {#each probeOrder as probe}
        <div class="col-label" title="{tok.text} / {probe}">
          {(tok.text || "·").replace(/\s/g, "·").slice(0, 4)}
        </div>
      {/each}
    {/each}
    {#each layers as layer}
      <div class="row-label">L{layer}</div>
      {#each $inspectorTokens as tok}
        {#each probeOrder as probe}
          {@const v = tok.scores[String(layer)]?.[probe] ?? 0}
          <div class="cell" style="background: {scoreToRgb(v)};"
               title="L{layer} · {probe} · {v.toFixed(3)}"></div>
        {/each}
      {/each}
    {/each}
  </div>
{/if}

<style>
  .header { font-size: 0.85em; color: #7d8590; letter-spacing: 0.1em; border-bottom: 1px solid #21262d; padding-bottom: 0.3em; margin-bottom: 0.5em; text-transform: uppercase; }
  .hint { color: #6e7681; font-size: 0.9em; padding: 0.5em; }
  .grid { display: grid; flex: 1 1 auto; overflow: auto; font-size: 11px; min-height: 0; }
  .row-label { color: #6e7681; text-align: right; padding-right: 0.3em; white-space: nowrap; position: sticky; left: 0; background: #0d1117; z-index: 1; }
  .col-label { color: #6e7681; font-size: 10px; text-align: center; border-bottom: 1px solid #21262d; padding-bottom: 0.2em; }
  .cell { width: 14px; height: 14px; cursor: help; }
</style>
