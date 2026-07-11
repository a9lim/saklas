<script lang="ts" generics="T extends string">
  // Canonical STEER/PROBE header used by all four inspector pillars. This
  // owns the title/live/count/sort rhythm so future tabs cannot drift by a
  // pixel or silently omit one of the shared controls.

  import Select from "../../lib/Select.svelte";

  interface SortOption<U> {
    value: U;
    label: string;
    disabled?: boolean;
  }

  let {
    title,
    count,
    live = null,
    liveBusy = false,
    liveTitle = "",
    onLiveToggle,
    sortValue,
    sortOptions = [],
    sortAriaLabel = "Sort cards by",
    onSortChange,
  }: {
    title: string;
    count?: string;
    live?: boolean | null;
    liveBusy?: boolean;
    liveTitle?: string;
    onLiveToggle?: () => void;
    sortValue?: T;
    sortOptions?: SortOption<T>[];
    sortAriaLabel?: string;
    onSortChange?: (value: T) => void;
  } = $props();

  const hasSort = $derived(sortValue !== undefined && sortOptions.length > 0);
</script>

<header class="header">
  <div class="header-text">
    <span class="title">{title}</span>
    {#if live !== null}
      <button
        type="button"
        class="toggle"
        class:on={live}
        disabled={liveBusy}
        onclick={onLiveToggle}
        title={liveTitle}
        aria-pressed={live}
      >
        {live ? "live: on" : "live: off"}
      </button>
    {/if}
    {#if count}<span class="count" aria-live="polite">{count}</span>{/if}
  </div>

  {#if hasSort}
    <label class="sort">
      <span class="sort-label">sort</span>
      <span class="sort-select">
        <Select
          value={sortValue as T}
          options={sortOptions}
          onchange={onSortChange}
          ariaLabel={sortAriaLabel}
        />
      </span>
    </label>
  {/if}
</header>

<style>
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    min-height: 26px;
    padding-bottom: var(--space-3);
  }
  .header-text {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .title {
    color: var(--accent);
    font-size: var(--text-sm);
    font-weight: var(--weight-bold);
    letter-spacing: 0;
    text-transform: uppercase;
  }
  .count,
  .sort-label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    flex: 0 0 auto;
  }
  .toggle {
    min-height: 22px;
    padding: 1px var(--space-3);
    color: var(--fg-muted);
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: var(--text-sm);
    cursor: pointer;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .toggle:hover:not(:disabled) {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .toggle.on {
    color: var(--accent);
    background: var(--accent-subtle);
  }
  .toggle:disabled {
    cursor: default;
    opacity: 0.5;
  }
  .sort {
    display: inline-flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .sort-select {
    display: inline-flex;
    min-width: 8em;
  }
</style>
