<script lang="ts">
  // Standardized drilldown empty state: what's missing, why, and (when
  // the caller supplies children) the action that fixes it — every tab
  // renders absence in this one shape.

  import type { Snippet } from "svelte";

  let {
    title,
    detail = null,
    children,
  }: {
    title: string;
    detail?: string | null;
    /** Optional action / hint block (a Button, a CLI hint). */
    children?: Snippet;
  } = $props();
</script>

<div class="empty">
  <p class="empty-title">{title}</p>
  {#if detail}
    <p class="empty-detail">{detail}</p>
  {/if}
  {#if children}
    <div class="empty-action">{@render children()}</div>
  {/if}
</div>

<style>
  .empty {
    color: var(--fg-muted);
    padding: var(--space-6) 0;
    line-height: 1.5;
    max-width: 62ch;
  }
  .empty p {
    margin: 0;
  }
  .empty-title {
    color: var(--fg-dim);
  }
  .empty-detail {
    font-size: var(--text-sm);
    margin-top: var(--space-2);
  }
  .empty-detail :global(code),
  .empty-action :global(code) {
    font-family: var(--font-mono);
    color: var(--fg-dim);
  }
  .empty-action {
    margin-top: var(--space-4);
  }
</style>
