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
    right: var(--space-6);
    bottom: var(--space-8);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    z-index: calc(var(--z-modal) + 10);
    max-width: 32em;
    pointer-events: none;
  }
  .toast {
    pointer-events: auto;
    background: var(--bg-alt);
    color: var(--fg);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent-blue);
    padding: var(--space-4) var(--space-5);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    box-shadow: var(--shadow-overlay);
    display: flex;
    align-items: flex-start;
    gap: var(--space-4);
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
    padding: 0 var(--space-1);
    font: inherit;
    font-family: var(--font-mono);
  }
  .dismiss:hover {
    color: var(--accent-red);
  }
</style>
