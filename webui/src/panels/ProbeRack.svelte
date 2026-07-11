<script lang="ts">
  // Probe rack — the PROBE section of one instrument tab.  The geometry
  // family arrives as a prop (the tab IS the group): subspace shows the
  // is_affine entries (a 2-node concept axis or a higher-rank flat fan
  // like personas), manifold the curved rest (e.g. a curved emotions
  // probe).  Every row wears the same RackCard chrome; the pillar hue
  // rides the card accent.  The sort dropdown (name / value / change)
  // orders the rows; the footer keeps the family's + launcher.
  //
  // Transcript highlight / compare-two controls live in Chat.svelte —
  // highlighting is about reading the transcript, so that is their one
  // home.

  import ProbeCard from "./rack/ProbeCard.svelte";
  import Select from "../lib/Select.svelte";
  import {
    activeProbeNames,
    openDrawer,
    probeRack,
    probesLiveState,
    setLiveProbes,
    setProbeSortMode,
  } from "../lib/stores.svelte";
  import type { ProbeSortMode } from "../lib/types";

  // The geometry family comes in as a prop since the four-pillar
  // restructure — the tab IS the group, so the rack shows one family:
  // subspace ⇔ info.is_affine, manifold ⇔ curved.  The live toggle is the
  // ONE monitor per-token scoring switch (it spans both families; it
  // renders in both tabs' headers driving the same state).
  let { family }: { family: "subspace" | "manifold" } = $props();

  const sortMode = $derived(probeRack.sortMode);
  const liveOn = $derived(probesLiveState.enabled);

  function onToggleLive(): void {
    void setLiveProbes(!liveOn);
  }

  // activeProbeNames() reads probeRack.active + entries + sortMode, all
  // $state-tracked, so this $derived re-runs on any of those changes.
  // ``jlens/`` token probes are excluded — they render as pinned chips in
  // the lens tab, not as rack rows here.
  const probes = $derived.by(() =>
    activeProbeNames().filter(
      (n) =>
        !n.startsWith("jlens/") &&
        probeRack.entries.get(n)?.info.is_affine === (family === "subspace"),
    ),
  );
  const count = $derived(probes.length);

  const SORT_OPTIONS: { value: ProbeSortMode; label: string }[] = [
    { value: "name", label: "name" },
    { value: "value", label: "value" },
    { value: "change", label: "change" },
  ];

  function onSortChange(v: ProbeSortMode): void {
    setProbeSortMode(v);
  }

  function onAddSubspaceProbe(): void {
    // The shared rack drawer hosts the probe attach UI alongside the
    // steer UI — every row carries both +steer and +probe, so the steer
    // rack and probe rack open the exact same drawer (no mode split).
    openDrawer("subspace");
  }

  function onAddManifoldProbe(): void {
    openDrawer("manifolds");
  }
</script>

<section class="rack" aria-label="Probe rack">
  <header class="header">
    <div class="header-text">
      <span class="title">PROBE</span>
      <button
        type="button"
        class="toggle"
        class:on={liveOn}
        disabled={probesLiveState.busy}
        onclick={onToggleLive}
        title={liveOn
          ? "Stop live per-token probe scoring (probes settle to the end-of-gen aggregate; gates still fire)"
          : "Score probes live every token (per-token stream, loom rows, trait events)"}
      >
        {liveOn ? "live: on" : "live: off"}
      </button>
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
    {:else}
      {#each probes as name (name)}
        {@const entry = probeRack.entries.get(name)}
        {#if entry}
          <div role="listitem">
            <ProbeCard {name} {entry} />
          </div>
        {/if}
      {/each}
    {/if}
  </div>

  <!-- The family's launcher stays reachable in both empty + populated
       states (hidden only when the server lacks the probe routes). -->
  {#if !probeRack.unavailable}
    <div class="actions">
      {#if family === "subspace"}
        <button
          type="button"
          class="add add-subspace"
          onclick={onAddSubspaceProbe}
          title="Attach a flat subspace as a read-side probe"
        >
          + subspace probe
        </button>
      {:else}
        <button
          type="button"
          class="add add-manifold"
          onclick={onAddManifoldProbe}
          title="Attach a curved manifold as a read-side probe"
        >
          + manifold probe
        </button>
      {/if}
    </div>
  {/if}
</section>

<style>
  /* A flat section of the inspector panel — no border box, no own
   * background; it reads as the lower half of one flat panel, divided
   * from the steering section above by typography + padding alone
   * (borderless — see SteeringRack). Fixed chrome + one scrollable
   * middle, matching SteeringRack. */
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
  /* Live toggle — same chrome as the J-LENS PROBE section's, so the two
   * tabs' headers read as siblings.  Borderless: a glass fill floats the
   * control, and "on" lifts to an accent-tinted fill (mirrors
   * SegmentedTabs' active-tab treatment) rather than a border ring. */
  .toggle {
    font-size: var(--text-sm);
    color: var(--fg-muted);
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 1px var(--space-3);
    cursor: pointer;
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
    opacity: 0.5;
    cursor: default;
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

  /* "Probe routes unavailable" notice — the one remaining empty state
   * (the first-run teaching copy is gone; the launchers below stand in). */
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

  /* Anchored at the bottom — same padding as SteeringRack so the two
   * racks read as visual siblings.  Shown in both empty and populated
   * states. */
  .actions {
    flex: 0 0 auto;
    display: flex;
    gap: var(--space-2);
    padding-top: var(--space-3);
  }
  .add {
    flex: 1 1 0;
    border: 1px solid transparent;
    padding: 2px var(--space-5);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    line-height: normal;
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  /* Subspace launcher — white accent (the base family colour). */
  .add-subspace {
    background: var(--accent-subtle);
    color: var(--accent);
  }
  .add-subspace:hover {
    background: var(--accent-glow);
  }
  /* Manifold launcher — purple-tinted to echo the manifold card's accent
   * so the probe-family colour stays consistent across the surfaces. */
  .add-manifold {
    background: color-mix(in srgb, var(--accent-purple) 10%, transparent);
    color: var(--accent-purple);
  }
  .add-manifold:hover {
    background: color-mix(in srgb, var(--accent-purple) 18%, transparent);
  }
</style>
