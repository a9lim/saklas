<script lang="ts">
  // Toast host — renders ``toasts.entries`` in the bottom-right corner.
  // Each toast auto-dismisses after its ``ttlMs`` fires; clicking the
  // ✕ dismisses early.  Toasts with ``ttlMs === null`` are sticky —
  // used by long-running async work (extract / clone) that drives a
  // single chip from kickoff to completion via ``updateToast``.

  import { dismissToast, toasts } from "./stores.svelte";

  // Track which toast ids have an active timer so we don't re-schedule
  // dismissal every time the entries array reshuffles.  Sticky entries
  // (``ttlMs === null``) never enter the set and never get scheduled.
  const scheduled = new Set<number>();

  $effect(() => {
    for (const t of toasts.entries) {
      if (scheduled.has(t.id)) continue;
      if (t.ttlMs === null) continue;
      scheduled.add(t.id);
      const ttl = t.ttlMs;
      setTimeout(() => {
        dismissToast(t.id);
        scheduled.delete(t.id);
      }, ttl);
    }
  });
</script>

{#if toasts.entries.length > 0}
  <div class="toaster" role="region" aria-label="Notifications">
    {#each toasts.entries as t (t.id)}
      <div class="toast" class:warning={t.kind === "warning"} class:error={t.kind === "error"} role="status">
        <div class="body">
          <span class="msg">{t.message}</span>
          {#if t.detail}
            <span class="detail">{t.detail}</span>
          {/if}
        </div>
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
    background: color-mix(in srgb, var(--surface-hi) 88%, transparent);
    -webkit-backdrop-filter: blur(var(--blur-glass));
    backdrop-filter: blur(var(--blur-glass));
    color: var(--fg);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent-blue);
    border-radius: var(--radius-lg);
    padding: var(--space-4) var(--space-5);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    box-shadow: var(--shadow-overlay);
    display: flex;
    align-items: flex-start;
    gap: var(--space-4);
    line-height: 1.4;
    animation: toast-in var(--dur-slow) var(--ease-spring);
  }
  @keyframes toast-in {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  .toast.warning {
    border-left-color: var(--accent-yellow);
  }
  .toast.error {
    border-left-color: var(--accent-red);
  }
  .body {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .msg {
    word-break: break-word;
  }
  .detail {
    color: var(--fg-dim);
    font-size: var(--text-xs);
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
