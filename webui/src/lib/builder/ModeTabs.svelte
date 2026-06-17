<script lang="ts" generics="T extends string">
  // Builder-drawer mode-tab row — one consistent shape across the
  // extract-vector and build-manifold drawers.  Replaces the ad-hoc
  // ``.mode-switch`` (extract) and ``.mode-tabs`` (manifold) markup so
  // the two surfaces read as the same control.
  //
  // Generic over tab-value type so callsites keep strongly-typed enums.

  interface Tab<U extends string> {
    value: U;
    label: string;
  }

  interface Props {
    value: T;
    tabs: Tab<T>[];
    /** ARIA label for the whole tablist — e.g. "Input mode". */
    ariaLabel?: string;
    onchange?: (value: T) => void;
  }

  let { value = $bindable(), tabs, ariaLabel, onchange }: Props = $props();

  function pick(v: T): void {
    if (v === value) return;
    value = v;
    onchange?.(v);
  }
</script>

<div class="sk-mode-tabs" role="tablist" aria-label={ariaLabel}>
  {#each tabs as tab (tab.value)}
    <button
      type="button"
      role="tab"
      class="sk-mode-tab"
      class:active={tab.value === value}
      aria-selected={tab.value === value}
      onclick={() => pick(tab.value)}
    >{tab.label}</button>
  {/each}
</div>

<style>
  .sk-mode-tabs {
    display: flex;
    gap: var(--space-2);
    padding: var(--space-1);
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }

  .sk-mode-tab {
    flex: 1 1 0;
    padding: var(--space-3) var(--space-4);
    background: transparent;
    color: var(--fg-dim);
    border: 0;
    border-radius: var(--radius);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    text-transform: lowercase;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out);
  }
  .sk-mode-tab:hover:not(.active) {
    color: var(--fg);
    background: var(--bg-hover);
  }
  .sk-mode-tab.active {
    background: var(--accent-subtle);
    color: var(--accent);
  }
  .sk-mode-tab:focus-visible {
    outline: 2px solid var(--accent-glow);
    outline-offset: 1px;
  }
</style>
