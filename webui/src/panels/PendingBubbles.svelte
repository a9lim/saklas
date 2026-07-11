<script lang="ts">
  // Pending-queue strip — ghosted bubbles between the streaming
  // assistant turn and the composer.  Each bubble shows the item's
  // label tag + truncated text + an X button that cancels just that
  // item via the per-id ``cancelPendingAction`` store helper.
  //
  // Drains automatically on every WS ``done``; this component is
  // display + cancel only — no submit logic.  Mirrors the TUI's
  // PendingStrip widget (saklas/tui/chat_panel.py) at the same
  // visual layer.

  import {
    pendingActions,
    cancelPendingAction,
    inputHistory,
  } from "../lib/stores.svelte";

  const items = $derived(pendingActions.queue);
  /** Slot index currently pulled into the input via ``↑``.  The
   *  matching bubble renders with the ``editing`` class so the user
   *  can see which queued item their input is editing. */
  const editingSlot = $derived(inputHistory.pulledSlot);

  function truncate(text: string | null, max: number): string {
    if (text === null) return "";
    const flat = text.replace(/\n/g, " ⏎ ");
    return flat.length <= max ? flat : flat.slice(0, max - 1) + "…";
  }
</script>

{#if items.length > 0}
  <div class="pending-strip" role="list" aria-label="Pending queue">
    {#each items as item, idx (item.id)}
      <div class="bubble" class:editing={idx === editingSlot} role="listitem">
        {#if idx === editingSlot}
          <span class="edit-marker" aria-label="editing">✎</span>
        {/if}
        <span class="tag">{item.label}</span>
        {#if item.text !== null}
          <span class="text" title={item.text}>{truncate(item.text, 80)}</span>
        {/if}
        <button
          type="button"
          class="cancel"
          onclick={() => cancelPendingAction(item.id)}
          aria-label="Cancel pending {item.label}"
          title="Cancel this queued item"
        >×</button>
      </div>
    {/each}
  </div>
{/if}

<style>
  /* Stacks vertically directly above the input row.  Hidden when empty
   * — Svelte's conditional render takes care of that, no class needed.
   * Borderless — the dashed border belongs to each ghost bubble (its
   * "not yet real" identity), not to the strip's own container. */
  .pending-strip {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    padding: var(--space-1) 0;
  }
  .bubble {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-1) var(--space-3);
    background: var(--glass);
    border: 1px dashed var(--glass-line);
    border-radius: var(--radius-pill);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--fg-dim);
    min-width: 0;
  }
  /* Editing state — the bubble whose text the user has pulled into the
   * input.  White-accent (solid ring + brighter fill/foreground) so it
   * stands out from the dim, dashed "just queued" rows. */
  .bubble.editing {
    background: var(--glass-strong);
    border-style: solid;
    border-color: var(--accent-glow);
    color: var(--fg);
  }
  .edit-marker {
    color: var(--fg);
    flex-shrink: 0;
  }
  .tag {
    color: var(--fg-muted);
    text-transform: lowercase;
    flex-shrink: 0;
  }
  .text {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cancel {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text);
    line-height: 1;
    padding: 0 var(--space-2);
    cursor: pointer;
    flex-shrink: 0;
    border-radius: var(--radius);
  }
  .cancel:hover {
    color: var(--accent-red);
    background: color-mix(in srgb, var(--accent-red) 8%, transparent);
  }
</style>
