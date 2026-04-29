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
          accepts a folder (concept directory) or a single safetensors/gguf
          file — same surface ``session.load_profile`` consumes.
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
    gap: 0.75em;
    min-height: 0;
  }
  .hint {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    line-height: 1.4;
  }
  .form {
    display: flex;
    flex-direction: column;
    gap: 0.65em;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.25em;
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    text-transform: lowercase;
  }
  .input {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: 0.4em 0.5em;
    font: inherit;
    font-family: var(--font-mono);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent-blue);
  }
  .sub {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
  }
  .validation {
    color: var(--accent-yellow);
    font-size: var(--font-size-small);
    margin: 0;
  }
  .error {
    color: var(--accent-error);
    font-size: var(--font-size-small);
    margin: 0;
    word-break: break-word;
  }
  .success {
    color: var(--accent-green);
    margin: 0;
    font-size: var(--font-size-small);
  }
  .success code {
    color: var(--accent-green);
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
