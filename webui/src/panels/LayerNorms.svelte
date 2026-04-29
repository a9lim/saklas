<script lang="ts">
  import { onMount } from "svelte";
  import { layerNormsByVector, refreshVector, vectors } from "../lib/stores";
  import type { VectorInfo } from "../lib/api";

  let selected = "";

  $: info = selected ? $layerNormsByVector[selected] : null;

  $: if (selected && !$layerNormsByVector[selected]) {
    refreshVector(selected).catch(() => {/* ignore */});
  }

  onMount(() => {
    if (!selected && $vectors.length > 0) selected = $vectors[0];
  });

  $: layers = info ? sortedLayers(info) : [];
  $: maxNorm = info ? Math.max(...layers.map((l) => info.per_layer_norms[String(l)])) : 0;

  function sortedLayers(v: VectorInfo): number[] {
    return Object.keys(v.per_layer_norms).map((k) => parseInt(k, 10)).sort((a, b) => a - b);
  }
</script>

<div class="header">LAYER NORMS</div>
<select bind:value={selected}>
  {#each $vectors as v}
    <option value={v}>{v}</option>
  {/each}
</select>
{#if !info}
  <div class="hint">{selected ? "loading…" : "no vector selected"}</div>
{:else if layers.length === 0}
  <div class="hint">no layers</div>
{:else}
  <div class="bars">
    {#each layers as l}
      {@const v = info.per_layer_norms[String(l)]}
      {@const w = maxNorm > 0 ? Math.round((v / maxNorm) * 140) : 0}
      <div class="row">
        <span class="label">L{l}</span>
        <span class="bar" style="width: {w}px;"></span>
        <span class="value">{v.toFixed(3)}</span>
      </div>
    {/each}
  </div>
{/if}

<style>
  .header { font-size: 0.85em; color: #7d8590; letter-spacing: 0.1em; border-bottom: 1px solid #21262d; padding-bottom: 0.3em; margin-bottom: 0.5em; text-transform: uppercase; }
  .hint { color: #6e7681; font-size: 0.9em; padding: 0.5em; }
  select { background: #161b22; color: #e6edf3; border: 1px solid #30363d; padding: 0.3em; font: inherit; margin-bottom: 0.4em; }
  .bars { flex: 1 1 auto; overflow: auto; display: flex; flex-direction: column; gap: 1px; font-size: 10px; min-height: 0; }
  .row { display: flex; align-items: center; gap: 0.4em; }
  .label { color: #6e7681; width: 3em; text-align: right; }
  .bar { background: #58a6ff; height: 8px; }
  .value { color: #6e7681; width: 4em; text-align: right; }
</style>
