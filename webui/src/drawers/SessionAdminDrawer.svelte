<script lang="ts">
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
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
    </div>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="body">
    <section class="panel">
      <h3>API key</h3>
      <div class="key-row">
        <input
          type="password"
          bind:value={key}
          placeholder="SAKLAS_API_KEY"
          autocomplete="off"
        />
        <button type="button" onclick={saveKey}>{saved ? "saved" : "apply"}</button>
        <button type="button" onclick={() => { key = ""; void saveKey(); }}>clear</button>
      </div>
      <p class="hint">memory only</p>
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
          <div class="empty">refresh to list</div>
        {:else}
          {#each sessions as s (s.id)}
            <article class:active={sessionState.info?.id === s.id}>
              <strong>{s.id}</strong>
              <code>{s.model_id}</code>
              <span>{s.device}/{s.dtype} · {s.profiles.length} profiles · {s.probes.length} probes</span>
            </article>
          {/each}
        {/if}
      </div>
    </section>

  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; background: transparent; }
  .header { display: flex; justify-content: space-between; gap: var(--space-6); padding: var(--space-5) var(--space-6); background: transparent; }
  .title { color: var(--accent); letter-spacing: 0; font-size: var(--text-md); font-weight: var(--weight-medium); }
  .hint, .panel p { margin: var(--space-1) 0 0; color: var(--fg-muted); line-height: 1.45; }
  .body { display: grid; gap: var(--space-5); padding: var(--space-6); overflow: auto; }
  .panel {
    border-radius: var(--radius);
    background: var(--glass);
    box-shadow: var(--shadow-well);
    padding: var(--space-6);
  }
  h3 { margin: 0 0 var(--space-4); color: var(--fg); font-size: var(--text-sm); letter-spacing: 0; }
  .key-row { display: grid; grid-template-columns: 1fr auto auto; gap: var(--space-3); }
  input { border: 1px solid transparent; border-radius: var(--radius); background: var(--input-well); color: var(--fg); padding: var(--space-4); font-family: var(--font-mono); }
  button { border: 1px solid transparent; border-radius: var(--radius); background: var(--glass); color: var(--fg); padding: var(--space-3) var(--space-5); }
  button:hover:not(:disabled) { background: var(--glass-strong); color: var(--accent); }
  .section-head { display: flex; align-items: center; justify-content: space-between; gap: var(--space-6); }
  .sessions { display: grid; gap: var(--space-3); }
  article { display: grid; gap: var(--space-2); border: 1px solid transparent; border-radius: var(--radius); background: var(--bg-elev); padding: var(--space-4); }
  article.active { border-color: var(--accent); background: var(--accent-subtle); }
  strong { color: var(--accent); }
  code { color: var(--accent-amber); font-family: var(--font-mono); }
  article span { color: var(--fg-muted); }
  .error { color: var(--accent-red); }
  .empty { color: var(--fg-muted); background: var(--bg); border-radius: var(--radius); padding: var(--space-6); text-align: center; }
</style>
