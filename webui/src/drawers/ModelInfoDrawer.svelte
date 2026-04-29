<script lang="ts">
  // Model-info drawer — read-only definition list of session details.
  // Mirrors the TUI's ``/model``.  Refresh button re-fetches via
  // ``refreshSession``.

  import {
    sessionState,
    refreshSession,
    closeDrawer,
    vectorRack,
    probeRack,
  } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let busy = $state(false);
  let errorMsg: string | null = $state(null);

  async function refresh(): Promise<void> {
    if (busy) return;
    busy = true;
    errorMsg = null;
    try {
      await refreshSession();
    } catch (e) {
      errorMsg = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  function fmtTimestamp(t: number | null | undefined): string {
    if (!t) return "—";
    const d = new Date(t * 1000);
    if (Number.isNaN(d.getTime())) return String(t);
    return d.toISOString().replace("T", " ").slice(0, 19);
  }

  function fmtSinceLastRefresh(): string {
    const t = sessionState.lastRefresh;
    if (!t) return "never";
    const ms = Date.now() - t;
    if (ms < 1000) return "just now";
    if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
    return `${Math.round(ms / 60_000)}m ago`;
  }
</script>

<section class="drawer-shell" aria-label="Model info drawer">
  <header class="header">
    <span class="title">model info</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    {#if !sessionState.info}
      <p class="dim">no session loaded</p>
    {:else}
      <dl class="dl">
        <div class="row">
          <dt>model_id</dt>
          <dd><code>{sessionState.info.model_id}</code></dd>
        </div>
        <div class="row">
          <dt>session id</dt>
          <dd><code>{sessionState.info.id}</code></dd>
        </div>
        <div class="row">
          <dt>device</dt>
          <dd>{sessionState.info.device}</dd>
        </div>
        <div class="row">
          <dt>dtype</dt>
          <dd>{sessionState.info.dtype}</dd>
        </div>
        {#if sessionState.info.architecture}
          <div class="row">
            <dt>architecture</dt>
            <dd>{sessionState.info.architecture}</dd>
          </div>
        {/if}
        <div class="row">
          <dt>supports thinking</dt>
          <dd>{sessionState.info.supports_thinking ? "yes" : "no"}</dd>
        </div>
        <div class="row">
          <dt>history length</dt>
          <dd>{sessionState.info.history_length}</dd>
        </div>
        <div class="row">
          <dt>default steering</dt>
          <dd class="wrap">
            {#if sessionState.info.default_steering}
              {sessionState.info.default_steering}
            {:else}
              <span class="dim">—</span>
            {/if}
          </dd>
        </div>
        <div class="row">
          <dt>vectors loaded</dt>
          <dd>
            {sessionState.info.vectors.length}
            {#if vectorRack.entries.size > 0}
              <span class="dim">
                · {vectorRack.entries.size} on rack
              </span>
            {/if}
          </dd>
        </div>
        <div class="row">
          <dt>probes active</dt>
          <dd>
            {sessionState.info.probes.length}
            {#if probeRack.active.length > 0}
              <span class="dim">
                · {probeRack.active.length} live
              </span>
            {/if}
          </dd>
        </div>
        <div class="row">
          <dt>created</dt>
          <dd>{fmtTimestamp(sessionState.info.created)}</dd>
        </div>
        <div class="row">
          <dt>last refresh</dt>
          <dd>{fmtSinceLastRefresh()}</dd>
        </div>
      </dl>
    {/if}

    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}
    {#if sessionState.error}
      <p class="error" role="alert">session error: {sessionState.error}</p>
    {/if}
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={closeDrawer}>close</button>
    <button
      type="button"
      class="btn primary"
      onclick={refresh}
      disabled={busy}
    >{busy ? "refreshing…" : "refresh"}</button>
  </footer>
</section>

<style>
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px;
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent-blue);
    text-transform: lowercase;
    letter-spacing: 0.04em;
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    cursor: pointer;
    padding: 0.25em 0.4em;
  }
  .close:hover {
    color: var(--accent-red);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 0.7em;
    min-height: 0;
  }
  .dl {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.35em;
  }
  .row {
    display: grid;
    grid-template-columns: 11em 1fr;
    gap: 0.6em;
    align-items: baseline;
  }
  .row dt {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    text-transform: lowercase;
    letter-spacing: 0.03em;
  }
  .row dd {
    margin: 0;
    color: var(--fg-strong);
    font-size: var(--font-size-base);
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .row dd code {
    color: var(--accent-green);
  }
  .wrap {
    word-break: break-word;
    white-space: pre-wrap;
  }
  .dim {
    color: var(--fg-muted);
  }
  .error {
    color: var(--accent-error);
    margin: 0;
    font-size: var(--font-size-small);
    word-break: break-word;
  }
  .footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.5em;
    padding: 16px;
    border-top: 1px solid var(--border);
  }
  .btn {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.4em 0.9em;
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .btn:disabled {
    color: var(--fg-muted);
    border-color: var(--border-dim);
    cursor: not-allowed;
  }
  .btn.primary {
    color: var(--accent-blue);
    border-color: var(--accent-blue);
  }
  .btn.primary:hover:not(:disabled) {
    background: rgba(88, 166, 255, 0.1);
  }
</style>
