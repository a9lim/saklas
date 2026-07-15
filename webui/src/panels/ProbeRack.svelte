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
  import RackSectionHeader from "./rack/RackSectionHeader.svelte";
  import {
    activeProbeNames,
    openDrawer,
    probeRack,
    probeEntryForDisplay,
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
        !n.startsWith("sae/") &&
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
  <RackSectionHeader
    title="PROBE"
    count={`${count} attached`}
    live={liveOn}
    liveBusy={probesLiveState.busy}
    liveTitle={liveOn
      ? "Stop live per-token probe scoring (probes settle to the end-of-gen aggregate; gates still fire)"
      : "Score probes live every token (per-token stream, loom rows, trait events)"}
    onLiveToggle={onToggleLive}
    sortValue={sortMode}
    sortOptions={SORT_OPTIONS}
    sortAriaLabel="Sort probes by"
    onSortChange={onSortChange}
  />

  <div class="strips" class:is-empty={count === 0} role="list">
    {#each probes as name (name)}
        {@const entry = probeEntryForDisplay(name)}
        {#if entry}
          <div role="listitem">
            <ProbeCard {name} {entry} />
          </div>
        {/if}
    {/each}
  </div>

  <div class="actions">
      {#if family === "subspace"}
        <button
          type="button"
          class="add add-subspace"
          onclick={onAddSubspaceProbe}
          title="add subspace probe"
        >
          + probe
        </button>
      {:else}
        <button
          type="button"
          class="add add-manifold"
          onclick={onAddManifoldProbe}
          title="add manifold probe"
        >
          + probe
        </button>
      {/if}
  </div>
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
    min-height: var(--control-target);
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
    background: color-mix(in srgb, var(--pillar-manifold) 10%, transparent);
    color: var(--pillar-manifold);
  }
  .add-manifold:hover {
    background: color-mix(in srgb, var(--pillar-manifold) 18%, transparent);
  }
</style>
