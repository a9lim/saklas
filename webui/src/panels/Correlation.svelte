<script lang="ts">
  import { onMount } from "svelte";
  import { correlation, refreshCorrelation } from "../lib/stores";

  function scoreToRgb(v: number): string {
    const a = Math.max(-1, Math.min(1, v));
    if (a >= 0) return `rgb(0, ${Math.round(a * 200)}, 0)`;
    return `rgb(${Math.round(-a * 200)}, 0, 0)`;
  }

  onMount(() => {
    refreshCorrelation();
  });
</script>

<div class="header">PROBE CORRELATION</div>
<button type="button" on:click={refreshCorrelation}>refresh</button>
{#if !$correlation || !$correlation.names.length}
  <div class="hint">no vectors loaded</div>
{:else}
  {@const names = $correlation.names}
  <div class="grid" style="grid-template-columns: auto repeat({names.length}, 18px);">
    <div></div>
    {#each names as name}
      <div class="col-label" title={name}>{name}</div>
    {/each}
    {#each names as a}
      <div class="row-label" title={a}>{a}</div>
      {#each names as b}
        {@const v = $correlation.matrix[a]?.[b]}
        <div class="cell"
             style="background: {v == null ? '#161b22' : scoreToRgb(v)};"
             title="{a} vs {b}: {v == null ? '—' : v.toFixed(2)}">
          {v == null ? "—" : v.toFixed(2)}
        </div>
      {/each}
    {/each}
  </div>
{/if}

<style>
  .header { font-size: 0.85em; color: #7d8590; letter-spacing: 0.1em; border-bottom: 1px solid #21262d; padding-bottom: 0.3em; margin-bottom: 0.5em; text-transform: uppercase; }
  .hint { color: #6e7681; font-size: 0.9em; padding: 0.5em; }
  button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 0.2em 0.6em; cursor: pointer; font: inherit; align-self: flex-start; margin-bottom: 0.2em; }
  .grid { display: grid; overflow: auto; font-size: 10px; min-height: 0; }
  .row-label { color: #6e7681; text-align: right; padding-right: 0.3em; }
  .col-label { color: #6e7681; writing-mode: vertical-rl; transform: rotate(180deg); font-size: 9px; padding: 0.1em; }
  .cell { width: 18px; height: 18px; display: flex; align-items: center; justify-content: center; font-size: 9px; }
</style>
