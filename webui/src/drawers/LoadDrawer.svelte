<script lang="ts">
  // Load drawer — registers a profile already on the server's filesystem
  // under a chosen rack name.  Mirrors ``saklas pack load``-style flow:
  // server-side path (NOT a browser file upload — the server reads the
  // file directly).
  //
  // Uses ``apiVectors.load({name, source_path})``.

  import { apiVectors, ApiError } from "../lib/api";
  import {
    addVectorToRack,
    closeDrawer,
    refreshVectorList,
  } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let name = $state("");
  let sourcePath = $state("");
  let busy = $state(false);
  let errorMsg: string | null = $state(null);
  let succeededName: string | null = $state(null);

  const validity = $derived.by(() => {
    if (!name.trim()) return { ok: false, reason: "name is required" } as const;
    if (!sourcePath.trim())
      return { ok: false, reason: "source path is required" } as const;
    return { ok: true, reason: null } as const;
  });

  async function run(): Promise<void> {
    if (!validity.ok || busy) return;
    busy = true;
    errorMsg = null;
    succeededName = null;
    try {
      const info = await apiVectors.load({
        name: name.trim(),
        source_path: sourcePath.trim(),
      });
      addVectorToRack(info.name);
      await refreshVectorList();
      succeededName = info.name;
    } catch (e) {
      if (e instanceof ApiError) {
        const detail =
          e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message;
        errorMsg = `${e.status}: ${detail}`;
      } else {
        errorMsg = e instanceof Error ? e.message : String(e);
      }
    } finally {
      busy = false;
    }
  }
</script>

<section class="drawer-shell" aria-label="Load drawer">
  <header class="header">
    <span class="title">load vector</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    <p class="hint">
      load a profile from a server-side filesystem path.  For browser-side
      uploads, use the load-conversation drawer (which expects a saved JSON
      blob), or push the pack from disk to your saklas cache first.
    </p>

    <form
      class="form"
      onsubmit={(ev) => {
        ev.preventDefault();
        void run();
      }}
    >
      <label class="field">
        <span class="label">name</span>
        <input
          type="text"
          class="input"
          bind:value={name}
          disabled={busy}
          placeholder="rack name (e.g. honest)"
          autocomplete="off"
          spellcheck="false"
        />
      </label>

      <label class="field">
        <span class="label">source path</span>
        <input
          type="text"
          class="input"
          bind:value={sourcePath}
          disabled={busy}
          placeholder="e.g. ~/.saklas/vectors/default/honest.deceptive"
          autocomplete="off"
          spellcheck="false"
        />
        <span class="sub">
          accepts a folder (concept directory) or a single safetensors or
          gguf file. This is the same surface ``session.load_profile`` consumes.
        </span>
      </label>

      {#if !validity.ok}
        <p class="validation">{validity.reason}</p>
      {/if}
    </form>

    {#if succeededName}
      <p class="success">
        loaded <code>{succeededName}</code> · added to rack
      </p>
    {/if}

    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}
  </div>

  <footer class="footer">
    <button
      type="button"
      class="btn"
      onclick={closeDrawer}
      disabled={busy}
    >{succeededName ? "done" : "cancel"}</button>
    <button
      type="button"
      class="btn primary"
      onclick={run}
      disabled={!validity.ok || busy}
    >{busy ? "loading…" : "load"}</button>
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
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-6);
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent-blue);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    cursor: pointer;
    padding: var(--space-2) var(--space-3);
  }
  .close:hover {
    color: var(--accent-red);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
    min-height: 0;
  }
  .hint {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.4;
  }
  .form {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    text-transform: lowercase;
  }
  .input {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .sub {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .validation {
    color: var(--accent-yellow);
    font-size: var(--text-sm);
    margin: 0;
  }
  .error {
    color: var(--accent-error);
    font-size: var(--text-sm);
    margin: 0;
    word-break: break-word;
  }
  .success {
    color: var(--accent-green);
    margin: 0;
    font-size: var(--text-sm);
  }
  .success code {
    color: var(--accent-green);
  }
  .footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
    padding: var(--space-6);
    border-top: 1px solid var(--border);
  }
  .btn {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-5);
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
    border-color: var(--border);
    cursor: not-allowed;
  }
  .btn.primary {
    background: var(--accent);
    color: var(--text-on-accent);
    border-color: var(--accent);
  }
  .btn.primary:hover:not(:disabled) {
    background: var(--accent-light);
    border-color: var(--accent-light);
  }
  .btn.primary:disabled {
    background: var(--bg-elev);
  }
</style>
