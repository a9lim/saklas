<script lang="ts">
  // Probe rack — two harmonised groups split on geometry, not artifact
  // type:
  //
  //   subspace (flat) — probeRack entries where info.is_affine (a 2-node
  //     concept axis or a higher-rank flat fan like personas).
  //   manifold (curved) — the rest (e.g. a curved pad probe).
  //
  // Every row wears the same RackCard chrome; the family is signalled only
  // by accent colour + marker glyph.  The sort dropdown (name / value /
  // change) orders within each group.  Light group sub-headers; an empty
  // group hides.  Footer keeps the + add probe / + add manifold probe
  // entry points and their drawer targets.
  //
  // Transcript highlight / compare-two controls live in Chat.svelte —
  // highlighting is about reading the transcript, so that is their one
  // home.

  import SubspaceProbeCard from "./rack/SubspaceProbeCard.svelte";
  import ManifoldProbeCard from "./rack/ManifoldProbeCard.svelte";
  import Select from "../lib/Select.svelte";
  import {
    activeProbeNames,
    openDrawer,
    probeRack,
    setProbeSortMode,
  } from "../lib/stores.svelte";
  import type { ProbeSortMode } from "../lib/types";

  const sortMode = $derived(probeRack.sortMode);

  // activeProbeNames() reads probeRack.active + entries + sortMode, all
  // $state-tracked, so this $derived re-runs on any of those changes.  We
  // then split it into the two families, preserving the sort order within
  // each.
  const sorted = $derived(activeProbeNames());

  const subspaceProbes = $derived.by(() =>
    sorted.filter((n) => probeRack.entries.get(n)?.info.is_affine === true),
  );
  const manifoldProbes = $derived.by(() =>
    sorted.filter((n) => probeRack.entries.get(n)?.info.is_affine === false),
  );

  const subspaceCount = $derived(subspaceProbes.length);
  const manifoldCount = $derived(manifoldProbes.length);
  const count = $derived(subspaceCount + manifoldCount);

  const SORT_OPTIONS: { value: ProbeSortMode; label: string }[] = [
    { value: "name", label: "name" },
    { value: "value", label: "value" },
    { value: "change", label: "change" },
  ];

  function onSortChange(v: ProbeSortMode): void {
    setProbeSortMode(v);
  }

  function onAddProbe(): void {
    openDrawer("vectors");
  }

  function onAddManifoldProbe(): void {
    // The manifolds drawer hosts the manifold-probe attach UI alongside
    // the existing manifold-steering UI — symmetric with how VectorsDrawer
    // hosts both steer/probe surfaces on every row.
    openDrawer("manifolds", { mode: "probe" });
  }
</script>

<section class="rack" aria-label="Probe rack">
  <header class="header">
    <div class="header-text">
      <span class="title">PROBES</span>
      <span class="count" aria-live="polite">
        {count} attached
      </span>
    </div>
    <label class="sort">
      <span class="sort-label">sort</span>
      <span class="sort-select">
        <Select
          value={sortMode}
          options={SORT_OPTIONS}
          onchange={onSortChange}
          ariaLabel="Sort probes by"
        />
      </span>
    </label>
  </header>

  <div class="strips" class:is-empty={count === 0} role="list">
    {#if probeRack.unavailable}
      <div class="empty">
        <p class="empty-copy">
          This server doesn't expose the read-side probe routes.
        </p>
      </div>
    {:else if count === 0}
      <div class="empty">
        <p class="empty-copy">
          Probes watch concepts activate as the model generates.
          Add a vector to read concept presence, or a manifold to read
          subspace fraction, nearest node, and inferred coordinates.
        </p>
        <div class="empty-actions">
          <button type="button" class="add" onclick={onAddProbe}>
            + add probe
          </button>
          <button
            type="button"
            class="add add-manifold"
            onclick={onAddManifoldProbe}
          >
            + add manifold probe
          </button>
        </div>
      </div>
    {:else}
      {#if subspaceCount > 0}
        <h3 class="group-header subspace">subspace</h3>
        {#each subspaceProbes as name (name)}
          {@const entry = probeRack.entries.get(name)}
          {#if entry}
            <div role="listitem">
              <SubspaceProbeCard {name} {entry} />
            </div>
          {/if}
        {/each}
      {/if}
      {#if manifoldCount > 0}
        <h3 class="group-header manifold">manifold</h3>
        {#each manifoldProbes as name (name)}
          {@const entry = probeRack.entries.get(name)}
          {#if entry}
            <div role="listitem">
              <ManifoldProbeCard {name} {entry} />
            </div>
          {/if}
        {/each}
      {/if}
    {/if}
  </div>

  {#if count > 0}
    <div class="actions">
      <button
        type="button"
        class="add"
        onclick={onAddProbe}
        title="Pick a concept to monitor (TUI /probe)"
      >
        + add probe
      </button>
      <button
        type="button"
        class="add add-manifold"
        onclick={onAddManifoldProbe}
        title="Attach a fitted manifold as a read-side probe"
      >
        + add manifold probe
      </button>
    </div>
  {/if}
</section>

<style>
  /* A flat section of the inspector panel — no border box, no own
   * background; it reads as the lower half of one flat panel, divided
   * from the steering section above by that section's border-bottom.
   * Fixed chrome + one scrollable middle, matching SteeringRack. */
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
  /* Match SteeringRack's title — bold accent so the two racks look like
   * siblings, not strangers. */
  .title {
    font-weight: var(--weight-bold);
    color: var(--accent);
    font-size: var(--text-sm);
    letter-spacing: 0;
    text-transform: uppercase;
  }
  .count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    flex: 0 0 auto;
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
  /* Layout host for the themed Select. */
  .sort-select {
    display: inline-flex;
    min-width: 8em;
  }

  /* Strips own the scroll inside the rack — with many auto-loaded probes
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

  /* Light group sub-headers — mirror SteeringRack so the steer / probe
   * racks read as one family. */
  .group-header {
    margin: 0;
    padding: var(--space-1) 0 0;
    font-size: var(--text-2xs);
    font-weight: var(--weight-normal);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--fg-muted);
  }
  .group-header.subspace {
    border-left: 2px solid var(--accent);
    padding-left: var(--space-2);
  }
  .group-header.manifold {
    border-left: 2px solid var(--accent-purple);
    padding-left: var(--space-2);
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
    max-width: 32ch;
  }
  /* Empty-state stacks the two entry points vertically so the manifold
   * launcher reads as the second way in, not buried in an actions row.
   * Mirrors SteeringRack's empty-actions shape. */
  .empty-actions {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    width: 14em;
  }

  /* Anchored at the bottom — same border-top + padding as SteeringRack
   * so the two racks read as visual siblings. */
  .actions {
    flex: 0 0 auto;
    display: flex;
    gap: var(--space-2);
    border-top: 1px solid var(--border);
    padding-top: var(--space-3);
  }
  .add {
    flex: 1 1 0;
    background: var(--accent-subtle);
    color: var(--accent);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-5);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    line-height: 1.3;
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .empty-actions .add {
    flex: 0 0 auto;
    width: 100%;
  }
  .add:hover {
    background: var(--accent-glow);
  }
  /* Manifold launcher — purple-tinted to echo the manifold card's accent
   * so the probe-family colour stays consistent across the surfaces. */
  .add-manifold {
    background: rgba(167, 139, 250, 0.10);
    color: var(--accent-purple);
  }
  .add-manifold:hover {
    background: rgba(167, 139, 250, 0.18);
  }
</style>
