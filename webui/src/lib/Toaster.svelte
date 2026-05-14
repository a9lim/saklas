<script lang="ts">
  // Toast host — renders ``toasts.entries`` in the bottom-right corner.
  // Each toast auto-dismisses after its ``ttlMs`` fires; clicking the
  // ✕ dismisses early.  Designed as an advisory surface for non-fatal
  // notices (localStorage budget, etc.); fatal errors still flow
  // through the ``boot-failed`` gate / inline error UI.

  import { dismissToast, toasts } from "./stores.svelte";

  // Track which toast ids have an active timer so we don't re-schedule
  // dismissal every time the entries array reshuffles.
  const scheduled = new Set<number>();

  $effect(() => {
    for (const t of toasts.entries) {
      if (scheduled.has(t.id)) continue;
      scheduled.add(t.id);
      setTimeout(() => {
        dismissToast(t.id);
        scheduled.delete(t.id);
      }, t.ttlMs);
    }
  });
</script>

{#if toasts.entries.length > 0}
  <div class="toaster" role="region" aria-label="Notifications">
    {#each toasts.entries as t (t.id)}
      <div class="toast" class:warning={t.kind === "warning"} class:error={t.kind === "error"} role="status">
        <span class="msg">{t.message}</span>
        <button
          type="button"
          class="dismiss"
          aria-label="Dismiss"
          onclick={() => dismissToast(t.id)}
        >✕</button>
      </div>
    {/each}
  </div>
{/if}

<style>
  .toaster {
    position: fixed;
    right: 1em;
    bottom: 3em;
    display: flex;
    flex-direction: column;
    gap: 0.5em;
    z-index: calc(var(--z-modal) + 10);
    max-width: 32em;
    pointer-events: none;
  }
  .toast {
    pointer-events: auto;
    background: var(--bg-alt);
    color: var(--fg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-blue);
    padding: 0.55em 0.75em;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
    display: flex;
    align-items: flex-start;
    gap: 0.6em;
    line-height: 1.4;
  }
  .toast.warning {
    border-left-color: var(--accent-yellow);
  }
  .toast.error {
    border-left-color: var(--accent-red);
  }
  .msg {
    flex: 1 1 auto;
    word-break: break-word;
  }
  .dismiss {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    cursor: pointer;
    padding: 0 0.2em;
    font: inherit;
    font-family: var(--font-mono);
  }
  .dismiss:hover {
    color: var(--accent-red);
  }
</style>
