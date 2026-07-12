<script lang="ts">
  import {
    closeDrawer,
    genStatus,
    geometricMeanPpl,
    loomTree,
    probeRack,
    refreshCorrelation,
    refreshLoomTree,
    refreshManifoldList,
    refreshProbeList,
    refreshSession,
    refreshVectorList,
    sessionState,
    steerRack,
    vectorsState,
  } from "../lib/stores.svelte";
  import Button from "../lib/ui/Button.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let busy = $state(false);
  let lastAudit: string | null = $state(null);
  let errorMsg: string | null = $state(null);

  const ppl = $derived(geometricMeanPpl(genStatus));
  const warnings = $derived.by(() => {
    const out: string[] = [];
    if (!sessionState.info) out.push("session info is not loaded");
    if (sessionState.error) out.push(sessionState.error);
    if (loomTree.error) out.push(`loom API error: ${loomTree.error}`);
    if (steerRack.catalog.length === 0) out.push("no manifold artifacts are available");
    if (probeRack.active.length === 0) out.push("no active probes; internal-state views will be sparse");
    return out;
  });

  async function audit(): Promise<void> {
    busy = true;
    errorMsg = null;
    try {
      await Promise.all([
        refreshSession(),
        refreshVectorList(),
        refreshProbeList(),
        refreshLoomTree(),
        refreshManifoldList(),
        refreshCorrelation(),
      ]);
      lastAudit = new Date().toLocaleTimeString();
    } catch (e) {
      errorMsg = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
</script>

<section class="drawer-shell" aria-label="Health drawer">
  <header class="drawer-header">
    <div class="title">
      <span class="eyebrow">model health</span>
      <div class="name-row">
        <span class="meta">runtime readiness, tree state, artifacts, probes, and UI coverage</span>
      </div>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}>✕</button>
  </header>

  <div class="body">
    <section class="hero">
      <div>
        <h2>{sessionState.info?.model_id ?? "no model"}</h2>
        <p>{sessionState.info ? `${sessionState.info.device}/${sessionState.info.dtype}` : "session offline"}</p>
      </div>
      <Button variant="solid" disabled={busy} onclick={audit}>
        {busy ? "auditing…" : "refresh audit"}
      </Button>
    </section>

    {#if errorMsg}
      <div class="error">{errorMsg}</div>
    {/if}

    <section class="grid">
      <div class="tile">
        <span>generation</span>
        <strong>{genStatus.active ? "active" : genStatus.finishReason ?? "idle"}</strong>
        <p>{genStatus.tokensSoFar}/{genStatus.maxTokens || "—"} tokens · {genStatus.tokPerSec.toFixed(1)} tok/s</p>
      </div>
      <div class="tile">
        <span>entropy perplexity</span>
        <strong>{ppl === null ? "—" : ppl.toFixed(2)}</strong>
        <p>{genStatus.ppl.count} scored steps</p>
      </div>
      <div class="tile">
        <span>loom tree</span>
        <strong>{loomTree.nodes.size || "—"}</strong>
        <p>rev {loomTree.loaded ? loomTree.rev : "—"} · active depth {loomTree.activePath.length || "—"}</p>
      </div>
      <div class="tile">
        <span>artifacts</span>
        <strong>{steerRack.catalog.length}</strong>
        <p>{steerRack.entries.size} on rack · {vectorsState.names.length} resident profiles</p>
      </div>
      <div class="tile">
        <span>probes</span>
        <strong>{probeRack.active.length}</strong>
        <p>{probeRack.entries.size} live rows · {steerRack.correlation ? "correlation cached" : "no matrix"}</p>
      </div>
    </section>

    <section class="panel">
      <h3>readiness checks</h3>
      <div class="checks">
        <div class:ok={!!sessionState.info}>session metadata</div>
        <div class:ok={loomTree.loaded && !loomTree.error}>loom API</div>
        <div class:ok={steerRack.catalog.length > 0}>manifold catalog</div>
        <div class:ok={probeRack.active.length > 0}>probe monitor</div>
        <div class:ok={steerRack.correlation !== null}>correlation cache</div>
      </div>
    </section>

    <section class="panel">
      <h3>warnings</h3>
      {#if warnings.length === 0}
        <p class="good">no visible health warnings from the web client cache</p>
      {:else}
        <ul>
          {#each warnings as warning (warning)}
            <li>{warning}</li>
          {/each}
        </ul>
      {/if}
      {#if lastAudit}
        <p class="dim">last audit: {lastAudit}</p>
      {/if}
    </section>
  </div>
</section>

<style>
  /* v2 sheet interior — the host paints the sheet surface (glass hairline,
   * radius, --bg-alt fill), so the root is transparent; chrome speaks sans
   * and every value/identifier/number sits in mono. */
  .drawer-shell {
    display: flex;
    flex-direction: column;
    min-height: 0;
    background: transparent;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .drawer-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--space-5) var(--space-6);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .eyebrow {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .name-row {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .meta {
    color: var(--fg-subtle);
    font-size: var(--text-sm);
  }
  .hero p, .tile p, .dim {
    margin: var(--space-1) 0 0;
    color: var(--fg-muted);
    line-height: 1.5;
  }
  .close {
    background: var(--glass);
    color: var(--fg-muted);
    border: 1px solid transparent;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font: inherit;
    font-size: var(--text-md);
    line-height: 1;
    cursor: pointer;
    flex: none;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .close:hover {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .body {
    display: grid;
    gap: var(--space-5);
    padding: var(--space-5) var(--space-6);
    overflow: auto;
  }
  /* Data wells — recessed stat/summary containers. */
  .hero, .tile, .panel {
    border-radius: var(--radius);
    background: var(--bg);
    padding: var(--space-6);
  }
  .hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-6);
  }
  h2, h3 { margin: 0; color: var(--fg); }
  h2 {
    font-family: var(--font-mono);
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  h3 {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: var(--space-5);
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-5);
  }
  .tile span {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .tile strong {
    display: block;
    margin-top: var(--space-2);
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-md);
    font-variant-numeric: tabular-nums;
  }
  .checks {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .checks div {
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-4);
    color: var(--fg-muted);
    background: var(--bg-elev);
    font-size: var(--text-sm);
    transition:
      color var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .checks div.ok {
    color: var(--accent-green);
    border-color: color-mix(in srgb, var(--accent-green) 35%, var(--glass-line));
    background: color-mix(in srgb, var(--accent-green) 8%, var(--bg-elev));
  }
  ul {
    margin: 0;
    padding-left: var(--space-6);
    color: var(--accent-yellow);
    line-height: 1.5;
  }
  li + li { margin-top: var(--space-2); }
  .good { color: var(--accent-green); margin: 0; line-height: 1.5; }
  .error {
    color: var(--accent-error);
    background: color-mix(in srgb, var(--accent-red) 8%, transparent);
    border-radius: var(--radius);
    padding: var(--space-5);
    line-height: 1.5;
  }
</style>
