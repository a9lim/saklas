<script lang="ts">
  // v2 segmented tabs — the instrument-stack pillar switch (and any other
  // exclusive-choice strip). Each item may carry a hue: the dot always
  // wears it, the active tab lifts onto glass with a hue-tinted hairline.
  //
  // Generic over the value type so callers keep literal unions
  // ("subspace" | "manifold" | ...) without casts.

  interface Item<T> {
    value: T;
    label: string;
    /** Optional compact evidence count shown beside the label. */
    meta?: string;
    /** Pillar hue for the dot + active tint — any CSS color. */
    color?: string;
    title?: string;
    disabled?: boolean;
  }

  interface Props<T> {
    items: Item<T>[];
    /** Selected value — bindable. */
    value: T;
    onchange?: (value: T) => void;
    /** Stretch tabs to fill the row. */
    fill?: boolean;
    /** Accessible name for the exclusive button group. */
    ariaLabel?: string;
  }

  type T = $$Generic;

  let {
    items,
    value = $bindable(),
    onchange,
    fill = false,
    ariaLabel = "View",
  }: Props<T> = $props();

  function pick(v: T): void {
    if (v === value) return;
    value = v;
    onchange?.(v);
  }
</script>

<div class="sk-tabs" class:fill role="group" aria-label={ariaLabel}>
  {#each items as item (item.value)}
    <button
      class="tab"
      class:on={item.value === value}
      style:--tab-c={item.color}
      aria-pressed={item.value === value}
      title={item.title}
      disabled={item.disabled}
      onclick={() => pick(item.value)}
    >
      {#if item.color}<span class="dot"></span>{/if}
      <span>{item.label}</span>
      {#if item.meta}<span class="meta">{item.meta}</span>{/if}
    </button>
  {/each}
</div>

<style>
  .sk-tabs {
    display: flex;
    gap: var(--space-2);
  }
  .sk-tabs.fill .tab {
    flex: 1;
  }

  .tab {
    --tab-c: var(--accent);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    font-size: var(--text-sm);
    letter-spacing: 0.04em;
    padding: 6px 12px;
    border-radius: var(--radius);
    border: 1px solid transparent;
    background: transparent;
    color: var(--fg-muted);
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .tab:hover:not(:disabled):not(.on) {
    color: var(--fg-dim);
    background: var(--bg-hover);
  }
  .tab:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Active: a quiet hue-tinted glass fill carries selection. */
  .tab.on {
    color: var(--fg);
    background: color-mix(in srgb, var(--tab-c) 9%, var(--glass));
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--tab-c);
    flex: none;
  }
  .tab:not(.on) .dot {
    opacity: 0.55;
  }

  .meta {
    min-width: 1.5em;
    padding: 1px var(--space-2);
    border-radius: var(--radius-pill);
    background: var(--glass);
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    letter-spacing: 0;
  }
  .tab.on .meta {
    color: var(--fg-dim);
    background: color-mix(in srgb, var(--tab-c) 10%, var(--glass-strong));
  }

  .tab:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
</style>
