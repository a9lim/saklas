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
  //
  // Transcript highlight / compare-two controls live in Chat.svelte —
  // highlighting is about reading the transcript, so that is their one
  // home (docs/plans/webui-overhaul.md §"Panel & drawer notes").

  import ProbeStrip from "./ProbeStrip.svelte";
  import {
    activeProbeNames,
    openDrawer,
    probeRack,
    setProbeSortMode,
  } from "../lib/stores.svelte";
  import type { ProbeSortMode } from "../lib/types";

  // Computed derivations — Svelte 5 runes track both the underlying state
  // and the function call's read of it via $derived's argument.
  const sortMode = $derived(probeRack.sortMode);

  // activeProbeNames() reads probeRack.active + entries + sortMode, all
  // $state-tracked, so this $derived re-runs on any of those changes.
  const sortedProbes = $derived(activeProbeNames());

  function onSortChange(ev: Event): void {
    const value = (ev.currentTarget as HTMLSelectElement).value as ProbeSortMode;
    setProbeSortMode(value);
  }

  function onAddProbe(): void {
    openDrawer("probe_picker");
  }
</script>

<section class="rack" aria-label="Probe rack">
  <header class="header">
    <div class="header-text">
      <span class="title">PROBES</span>
    </div>
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

  <div class="strips" class:is-empty={sortedProbes.length === 0} role="list">
    {#if sortedProbes.length === 0}
      <div class="empty">
        <p class="empty-copy">
          Probes watch concepts activate as the model generates.
          They read the model's state without changing it.
        </p>
        <button type="button" class="add empty-add" onclick={onAddProbe}>
          + add probe
        </button>
      </div>
    {:else}
      {#each sortedProbes as probe (probe)}
        <div role="listitem">
          <ProbeStrip name={probe} />
        </div>
      {/each}
    {/if}
  </div>

  {#if sortedProbes.length > 0}
    <div class="actions">
      <button
        type="button"
        class="add"
        onclick={onAddProbe}
        title="Pick a concept to monitor (TUI /probe)"
      >
        + add probe
      </button>
    </div>
  {/if}
</section>

<style>
  /* A flat section of the inspector panel — no border box, no own
   * background; it reads as the lower half of one flat panel, divided
   * from the steering section above by that section's border-bottom.
   * Fixed chrome + one scrollable middle, matching SteeringRack.  The
   * inspector constrains this rack, then only .strips may scroll. */
  .rack {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-5);
    background: transparent;
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
  }

  .header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: var(--space-3);
  }
  .header-text {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  /* Match SteeringRack's title — bold accent-blue so the two racks look
   * like siblings, not strangers. */
  .title {
    font-weight: var(--weight-bold);
    color: var(--accent-blue);
    font-size: var(--text-sm);
    letter-spacing: 0;
    text-transform: uppercase;
  }
  .sort {
    display: inline-flex;
    align-items: center;
    gap: var(--space-3);
  }
  .sort-label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .sort-select {
    background: var(--bg-elev);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    font: inherit;
    font-size: var(--text-sm);
  }

  /* Strips own the scroll inside the rack — with 26 auto-loaded probes
   * the list overflows the rack viewport, but the actions row stays
   * anchored at the bottom. */
  .strips {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    flex: 1 1 0;
    min-height: 2.4rem;
    max-height: 100%;
    overflow-y: auto;
    padding-right: var(--space-1);
  }
  .strips.is-empty {
    align-items: center;
    justify-content: center;
  }
  /* First-run teaching state — names the steer-vs-observe distinction the
   * two racks otherwise blur. */
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-5) var(--space-4);
    text-align: center;
  }
  .empty-copy {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.5;
    max-width: 30ch;
  }
  .add.empty-add {
    flex: 0 0 auto;
    min-width: 14em;
  }

  /* Anchored at the bottom — same border-top + padding as SteeringRack
   * so the two racks read as visual siblings. */
  .actions {
    flex: 0 0 auto;
    display: flex;
    gap: var(--space-3);
    border-top: 1px solid var(--border);
    padding-top: var(--space-3);
  }
  .add {
    flex: 1 1 100%;
    background: var(--accent-subtle);
    color: var(--accent-blue);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-5);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    line-height: 1.3;
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .add:hover {
    background: var(--accent-glow);
  }
</style>
