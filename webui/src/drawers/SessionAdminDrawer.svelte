<script lang="ts">
  import { apiSessions, getApiKey, setApiKey } from "../lib/api";
  import type { SessionInfo } from "../lib/types";
  import { closeDrawer, refreshSession, sessionState } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let key = $state(getApiKey() ?? "");
  let sessions: SessionInfo[] = $state([]);
  let busy = $state(false);
  let errorMsg: string | null = $state(null);
  let saved = $state(false);

  async function loadSessions(): Promise<void> {
    busy = true;
    errorMsg = null;
    try {
      const r = await apiSessions.list();
      sessions = r.sessions;
    } catch (e) {
      errorMsg = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  async function saveKey(): Promise<void> {
    setApiKey(key);
    saved = true;
    setTimeout(() => (saved = false), 1200);
    await refreshSession();
    await loadSessions();
  }
</script>

<section class="drawer-shell" aria-label="Session and auth drawer">
  <header class="header">
    <div>
      <span class="title">session & auth</span>
      <p>bearer key, session list, and multi-session readiness</p>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}>×</button>
  </header>

  <div class="body">
    <section class="panel">
      <h3>API key</h3>
      <div class="key-row">
        <input
          type="password"
          bind:value={key}
          placeholder="SAKLAS_API_KEY for this browser tab"
          autocomplete="off"
        />
        <button type="button" onclick={saveKey}>{saved ? "saved" : "apply"}</button>
        <button type="button" onclick={() => { key = ""; void saveKey(); }}>clear</button>
      </div>
      <p class="hint">Stored only in memory for this page session; no localStorage write.</p>
    </section>

    <section class="panel">
      <div class="section-head">
        <h3>sessions</h3>
        <button type="button" disabled={busy} onclick={loadSessions}>
          {busy ? "loading…" : "refresh"}
        </button>
      </div>
      {#if errorMsg}
        <p class="error">{errorMsg}</p>
      {/if}
      <div class="sessions">
        {#if sessions.length === 0}
          <div class="empty">click refresh to query the native session collection</div>
        {:else}
          {#each sessions as s (s.id)}
            <article class:active={sessionState.info?.id === s.id}>
              <strong>{s.id}</strong>
              <code>{s.model_id}</code>
              <span>{s.device}/{s.dtype} · {s.vectors.length} vectors · {s.probes.length} probes</span>
            </article>
          {/each}
        {/if}
      </div>
    </section>

    <section class="panel">
      <h3>remote/multi-session posture</h3>
      <p>
        The native API is already shaped like a multi-session service, but this
        local dashboard still talks to the bundled same-origin server and the
        websocket path resolves the default session. This surface makes the
        boundary visible instead of hiding it inside fetch helpers.
      </p>
    </section>
  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; background: var(--bg-alt); }
  .header { display: flex; justify-content: space-between; gap: 1rem; padding: 1rem 1.1rem; border-bottom: 1px solid var(--border); background: var(--surface); }
  .title { color: var(--accent); text-transform: uppercase; letter-spacing: 0; font-size: 0.75rem; font-weight: 700; }
  .header p, .hint, .panel p { margin: 0.3rem 0 0; color: var(--fg-muted); line-height: 1.45; }
  .close { background: transparent; border: 0; color: var(--fg-muted); font-size: 1.25rem; }
  .body { display: grid; gap: 0.85rem; padding: 1rem; overflow: auto; }
  .panel { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); padding: 0.9rem; }
  h3 { margin: 0 0 0.65rem; color: var(--fg); font-size: 0.92rem; letter-spacing: 0; }
  .key-row { display: grid; grid-template-columns: 1fr auto auto; gap: 0.5rem; }
  input { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-deep); color: var(--fg); padding: 0.55rem; font-family: var(--font-mono); }
  button { border: 1px solid var(--border); border-radius: var(--radius); background: rgba(255,255,255,0.03); color: var(--fg); padding: 0.5rem 0.7rem; }
  button:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  .section-head { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
  .sessions { display: grid; gap: 0.5rem; }
  article { display: grid; gap: 0.25rem; border: 1px solid var(--border-dim); border-radius: var(--radius); background: rgba(255,255,255,0.025); padding: 0.65rem; }
  article.active { border-color: rgba(72, 138, 203, 0.55); background: rgba(72, 138, 203, 0.10); }
  strong { color: var(--accent); }
  code { color: var(--accent-amber); font-family: var(--font-mono); }
  article span { color: var(--fg-muted); }
  .error { color: var(--accent-red); }
  .empty { color: var(--fg-muted); border: 1px dashed var(--border); border-radius: var(--radius); padding: 1rem; text-align: center; }
</style>
