<script lang="ts">
  import { onMount } from "svelte";
  import Chat from "./panels/Chat.svelte";
  import Inspector from "./panels/Inspector.svelte";
  import Correlation from "./panels/Correlation.svelte";
  import LayerNorms from "./panels/LayerNorms.svelte";
  import { sessionInfo, refreshSession } from "./lib/stores";

  onMount(() => {
    refreshSession();
  });

  $: status = $sessionInfo
    ? `${$sessionInfo.model_id} · ${$sessionInfo.device}/${$sessionInfo.dtype} · vectors=${$sessionInfo.vectors.length} probes=${$sessionInfo.probes.length}`
    : "connecting…";
</script>

<header class="topbar">
  <span class="brand">saklas</span>
  <span class="status">{status}</span>
</header>

<main class="grid">
  <section class="panel chat" aria-label="Chat">
    <Chat />
  </section>
  <section class="panel inspector" aria-label="Inspector">
    <Inspector />
  </section>
  <section class="panel correlation" aria-label="Correlation">
    <Correlation />
  </section>
  <section class="panel layer-norms" aria-label="Layer norms">
    <LayerNorms />
  </section>
</main>

<style>
  :global(html), :global(body) {
    margin: 0;
    padding: 0;
    height: 100%;
    background: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, "SF Mono", Menlo, Consolas, monospace;
    font-size: 13px;
  }
  .topbar {
    display: flex;
    align-items: center;
    gap: 1.5em;
    padding: 0.5em 1em;
    background: #010409;
    border-bottom: 1px solid #30363d;
  }
  .brand { font-weight: bold; color: #7ee787; letter-spacing: 0.05em; }
  .status { color: #8b949e; font-size: 0.9em; }
  .grid {
    display: grid;
    grid-template-columns: 30% 40% 30%;
    grid-template-rows: 1fr 1fr;
    height: calc(100vh - 36px);
    gap: 1px;
    background: #30363d;
  }
  .panel { background: #0d1117; padding: 0.5em; overflow: hidden; }
  .chat       { grid-column: 1; grid-row: 1 / span 2; }
  .inspector  { grid-column: 2; grid-row: 1 / span 2; }
  .correlation{ grid-column: 3; grid-row: 1; }
  .layer-norms{ grid-column: 3; grid-row: 2; }
</style>
