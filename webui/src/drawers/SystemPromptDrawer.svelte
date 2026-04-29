<script lang="ts">
  // System-prompt drawer — edit the session's default system prompt.
  // Saves via PATCH /sessions/{id} (``patchSessionDefaults``); cancel
  // closes without writing.

  import {
    sessionState,
    patchSessionDefaults,
    closeDrawer,
  } from "../lib/stores.svelte";
  import { ApiError } from "../lib/api";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let value = $state(sessionState.info?.config.system_prompt ?? "");
  let busy = $state(false);
  let errorMsg: string | null = $state(null);

  async function save(): Promise<void> {
    if (busy) return;
    busy = true;
    errorMsg = null;
    try {
      await patchSessionDefaults({ system_prompt: value });
      closeDrawer();
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

<section class="drawer-shell" aria-label="System prompt drawer">
  <header class="header">
    <span class="title">system prompt</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    <p class="hint">
      sets the default system prompt for new generations on this session.
      Per-message overrides via the OpenAI / Ollama protocols still take
      precedence.  Empty string clears the system prompt.
    </p>

    <label class="field">
      <span class="label">prompt</span>
      <textarea
        class="textarea"
        rows="12"
        bind:value={value}
        disabled={busy}
        placeholder="(no system prompt)"
        spellcheck="false"
      ></textarea>
      <span class="char-count">{value.length} char{value.length === 1 ? "" : "s"}</span>
    </label>

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
    >cancel</button>
    <button
      type="button"
      class="btn primary"
      onclick={save}
      disabled={busy}
    >{busy ? "saving…" : "save as session default"}</button>
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
  .hint {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    line-height: 1.4;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.25em;
    flex: 1 1 auto;
    min-height: 0;
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    text-transform: lowercase;
  }
  .textarea {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: 0.5em 0.6em;
    font: inherit;
    font-family: var(--font-mono);
    line-height: 1.4;
    resize: vertical;
    min-height: 200px;
  }
  .textarea:focus {
    outline: none;
    border-color: var(--accent-blue);
  }
  .char-count {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
    align-self: flex-end;
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
