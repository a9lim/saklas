<script lang="ts">
  // Probe rack — section header with sort dropdown, highlight + compare-two
  // controls duplicated from Chat.svelte (both surfaces bind to the same
  // store so changing one updates the other), and one <ProbeStrip> per
  // active probe.  + add probe button opens a drawer for picking from
  // /probes/defaults.
  //
  // Sort cycle order matches the TUI (Ctrl+S): name → value → change →
  // name.  Live re-sort is cheap — activeProbeNames() returns a fresh
  // array per access from the rack store and the strip list keys on
  // probe name so DOM nodes survive reorders.

  import ProbeStrip from "./ProbeStrip.svelte";
  import {
    activeProbeNames,
    highlightState,
    openDrawer,
    probeRack,
    setCompareTarget,
    setHighlightTarget,
    setProbeSortMode,
    toggleCompareTwo,
  } from "../lib/stores.svelte";
  import type { ProbeSortMode } from "../lib/types";

  // Computed derivations — Svelte 5 runes track both the underlying state
  // and the function call's read of it via $derived's argument.
  const sortMode = $derived(probeRack.sortMode);
  const compareTwo = $derived(highlightState.compareTwo);
  const target = $derived(highlightState.target);
  const compareTarget = $derived(highlightState.compareTarget);

  // activeProbeNames() reads probeRack.active + entries + sortMode, all
  // $state-tracked, so this $derived re-runs on any of those changes.
  const sortedProbes = $derived(activeProbeNames());

  function onSortChange(ev: Event): void {
    const value = (ev.currentTarget as HTMLSelectElement).value as ProbeSortMode;
    setProbeSortMode(value);
  }

  function onHighlightChange(ev: Event): void {
    const v = (ev.currentTarget as HTMLSelectElement).value;
    setHighlightTarget(v === "" ? null : v);
  }

  function onCompareChange(ev: Event): void {
    const v = (ev.currentTarget as HTMLSelectElement).value;
    setCompareTarget(v === "" ? null : v);
  }

  function onAddProbe(): void {
    // TODO(phase-9): introduce a dedicated "probe_picker" DrawerName +
    // drawer that lists /probes/defaults for one-click activation.  For
    // now route through extract — it's the next-best fit (creating a new
    // probe by extracting a vector also auto-registers + activates).
    openDrawer("extract");
  }
</script>

<section class="rack" aria-label="Probe rack">
  <header class="header">
    <span class="title">PROBES</span>
    <label class="sort">
      <span class="sort-label">sort</span>
      <select
        class="sort-select"
        value={sortMode}
        onchange={onSortChange}
        aria-label="Sort probes by"
      >
        <option value="name">name</option>
        <option value="value">value</option>
        <option value="change">change</option>
      </select>
    </label>
  </header>

  <div class="highlight-bar">
    <label class="hl-field">
      <span class="hl-label">highlight</span>
      <select
        class="hl-select"
        value={target ?? ""}
        onchange={onHighlightChange}
        aria-label="Highlight target probe"
      >
        <option value="">(off)</option>
        {#each sortedProbes as probe (probe)}
          <option value={probe}>{probe}</option>
        {/each}
      </select>
    </label>

    <label class="compare-toggle" title="Show a second probe as a stripe overlay">
      <input
        type="checkbox"
        checked={compareTwo}
        onchange={toggleCompareTwo}
      />
      <span>compare two</span>
    </label>

    {#if compareTwo}
      <label class="hl-field">
        <span class="hl-label">vs</span>
        <select
          class="hl-select"
          value={compareTarget ?? ""}
          onchange={onCompareChange}
          aria-label="Compare target probe"
        >
          <option value="">(off)</option>
          {#each sortedProbes as probe (probe)}
            <option value={probe}>{probe}</option>
          {/each}
        </select>
      </label>
    {/if}
  </div>

  <div class="strips" role="list">
    {#if sortedProbes.length === 0}
      <div class="empty">no active probes — click + add probe</div>
    {:else}
      {#each sortedProbes as probe (probe)}
        <div role="listitem">
          <ProbeStrip name={probe} />
        </div>
      {/each}
    {/if}
  </div>

  <div class="actions">
    <button
      type="button"
      class="add"
      onclick={onAddProbe}
      title="Add a probe — extract a new concept and auto-register"
    >
      + add probe
    </button>
  </div>
</section>

<style>
  .rack {
    display: flex;
    flex-direction: column;
    gap: 0.4em;
    padding: 0.5em;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    min-height: 7em;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 0.3em;
  }
  .title {
    color: var(--fg-subtle);
    font-size: 0.85em;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .sort {
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
  }
  .sort-label {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
  }
  .sort-select,
  .hl-select {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.15em 0.35em;
    border-radius: 3px;
    font: inherit;
    font-size: 0.85em;
  }
  .sort-select:focus-visible,
  .hl-select:focus-visible {
    outline: 1px solid var(--accent-blue);
    outline-offset: 0;
  }

  .highlight-bar {
    display: flex;
    align-items: center;
    gap: 0.6em;
    flex-wrap: wrap;
    padding: 0.2em 0;
  }
  .hl-field {
    display: inline-flex;
    align-items: center;
    gap: 0.35em;
  }
  .hl-label {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
  }
  .compare-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.3em;
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    cursor: pointer;
  }
  .compare-toggle input {
    accent-color: var(--accent-blue);
  }

  .strips {
    display: flex;
    flex-direction: column;
    gap: 0.25em;
    min-height: 0;
  }
  .empty {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    padding: 0.6em 0.4em;
    text-align: center;
    border: 1px dashed var(--border-dim);
    border-radius: 3px;
  }

  .actions {
    display: flex;
    gap: 0.3em;
    padding-top: 0.2em;
  }
  .add {
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.25em 0.7em;
    border-radius: 3px;
    font-size: 0.85em;
  }
  .add:hover {
    background: var(--bg-elev);
    border-color: var(--accent-blue);
    color: var(--accent-blue);
  }
</style>
