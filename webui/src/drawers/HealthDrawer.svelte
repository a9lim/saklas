<script lang="ts">
  import {
    closeDrawer,
    genStatus,
    geometricMeanPpl,
    loomTree,
    probeRack,
    refreshCorrelation,
    refreshLoomTree,
    refreshPacks,
    refreshProbeList,
    refreshSession,
    refreshVectorList,
    sessionState,
    vectorRack,
    vectorsState,
    packsState,
  } from "../lib/stores.svelte";

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
    if (loomTree.unavailable) out.push("loom API unavailable; branch workflows are disabled");
    if (vectorsState.names.length === 0) out.push("no vectors registered in the session");
    if (probeRack.active.length === 0) out.push("no active probes; internal-state views will be sparse");
    if (packsState.error) out.push(`pack list: ${packsState.error}`);
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
        refreshPacks(),
        refreshLoomTree(),
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
  <header class="header">
    <div>
      <span class="title">model health</span>
      <p>runtime readiness, cache surfaces, tree state, vectors, probes, and UI coverage</p>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}>×</button>
  </header>

  <div class="body">
    <section class="hero">
      <div>
        <h2>{sessionState.info?.model_id ?? "no model"}</h2>
        <p>{sessionState.info ? `${sessionState.info.device}/${sessionState.info.dtype}` : "session offline"}</p>
      </div>
      <button type="button" disabled={busy} onclick={audit}>
        {busy ? "auditing…" : "refresh audit"}
      </button>
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
        <span>perplexity</span>
        <strong>{ppl === null ? "—" : ppl.toFixed(2)}</strong>
        <p>{genStatus.ppl.count} scored steps</p>
      </div>
      <div class="tile">
        <span>loom tree</span>
        <strong>{loomTree.nodes.size || "—"}</strong>
        <p>rev {loomTree.rev || "—"} · active depth {loomTree.activePath.length || "—"}</p>
      </div>
      <div class="tile">
        <span>vectors</span>
        <strong>{vectorsState.names.length}</strong>
        <p>{vectorRack.entries.size} on rack · {vectorRack.profiles.size} profiles cached</p>
      </div>
      <div class="tile">
        <span>probes</span>
        <strong>{probeRack.active.length}</strong>
        <p>{probeRack.entries.size} live rows · {vectorRack.correlation ? "correlation cached" : "no matrix"}</p>
      </div>
      <div class="tile">
        <span>packs</span>
        <strong>{packsState.installed.length}</strong>
        <p>{packsState.loading ? "loading" : packsState.error ?? "ready"}</p>
      </div>
    </section>

    <section class="panel">
      <h3>readiness checks</h3>
      <div class="checks">
        <div class:ok={!!sessionState.info}>session metadata</div>
        <div class:ok={!loomTree.unavailable && loomTree.rev > 0}>loom API</div>
        <div class:ok={vectorsState.names.length > 0}>vector registry</div>
        <div class:ok={probeRack.active.length > 0}>probe monitor</div>
        <div class:ok={vectorRack.correlation !== null}>correlation cache</div>
        <div class:ok={!packsState.error}>pack index</div>
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
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; background: var(--bg-alt); }
  .header { display: flex; justify-content: space-between; gap: 1rem; padding: 1rem 1.1rem; border-bottom: 1px solid var(--border); background: var(--surface); }
  .title { color: var(--accent); text-transform: uppercase; letter-spacing: 0; font-size: 0.75rem; font-weight: 700; }
  .header p, .hero p, .tile p, .dim { margin: 0.3rem 0 0; color: var(--fg-muted); }
  .close { background: transparent; border: 0; color: var(--fg-muted); font-size: 1.25rem; }
  .body { display: grid; gap: 0.85rem; padding: 1rem; overflow: auto; }
  .hero, .tile, .panel { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); padding: 0.9rem; }
  .hero { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
  h2, h3 { margin: 0; color: var(--fg); letter-spacing: 0; }
  h2 { font-size: 1rem; }
  h3 { font-size: 0.92rem; margin-bottom: 0.7rem; }
  button { border: 1px solid var(--accent); border-radius: var(--radius); background: rgba(72, 138, 203, 0.11); color: var(--accent); padding: 0.6rem 0.85rem; font-weight: 700; }
  button:disabled { opacity: 0.55; cursor: wait; }
  .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.7rem; }
  .tile span { color: var(--fg-muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0; }
  .tile strong { display: block; margin-top: 0.28rem; color: var(--accent); font-family: var(--font-mono); font-size: 1rem; }
  .checks { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.45rem; }
  .checks div { border: 1px solid var(--border-dim); border-radius: var(--radius); padding: 0.55rem; color: var(--fg-muted); background: rgba(255,255,255,0.025); }
  .checks div.ok { color: var(--accent-green); border-color: rgba(126,231,135,0.38); background: rgba(126,231,135,0.07); }
  ul { margin: 0; padding-left: 1.1rem; color: var(--accent-amber); }
  li + li { margin-top: 0.35rem; }
  .good { color: var(--accent-green); margin: 0; }
  .error { color: var(--accent-red); border: 1px solid rgba(255,118,117,0.45); background: rgba(255,118,117,0.08); border-radius: var(--radius); padding: 0.7rem; }
</style>
