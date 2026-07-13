<script lang="ts">
  // The steering rack — the STEER section of one instrument tab.  Since the
  // four-pillar restructure the geometry family comes in as a prop (the tab
  // IS the group), so the rack renders exactly one family's terms:
  //
  //   subspace (flat)   — a 2-node bipolar axis through the rank-8 personas
  //     fan.  Shares one rack-level "subspace along" master (the merged
  //     affine subspace slides once), so the cards carry no per-card along.
  //   manifold (curved) — curved fits (e.g. emotions), each its own injection
  //     with a per-card along + onto.
  //
  // Every row is one SteerCard wearing the same RackCard chrome; the pillar
  // hue rides the card accent.  Footer keeps the family's + launcher.

  import SteerCard from "./rack/SteerCard.svelte";
  import RackSectionHeader from "./rack/RackSectionHeader.svelte";
  import Slider from "../lib/Slider.svelte";
  import {
    steerRack,
    setSubspaceAlong,
    openDrawer,
  } from "../lib/stores.svelte";

  let { family }: { family: "subspace" | "manifold" } = $props();

  // Alphabetized for stable order — Map iteration tracks insertion which
  // makes the rack jump around.
  const terms = $derived.by(() => {
    const arr = [...steerRack.entries.entries()].filter(
      ([, e]) => e.mode === family,
    );
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });
  const count = $derived(terms.length);

  function onAlongInput(v: number): void {
    if (Number.isFinite(v)) setSubspaceAlong(v);
  }
</script>

<section class="rack" aria-label="Steering rack">
  <RackSectionHeader
    title="STEER"
    count={`${count} term${count === 1 ? "" : "s"}`}
  />

  {#if count > 0}
    <div class="strips">
      {#if family === "subspace"}
        <!-- Shared "subspace along" master — the single slide magnitude every
             flat term serializes with (the merged affine subspace slides
             once).  Per-term relative weight lives in each card's position. -->
        <div
          class="along-master"
          title="shared magnitude"
        >
          <span class="along-label">all</span>
          <Slider
            value={steerRack.subspaceAlong}
            min={0}
            max={2}
            step={0.05}
            oninput={onAlongInput}
            ariaLabel="shared subspace along"
          />
          <span class="along-val">{steerRack.subspaceAlong.toFixed(2)}</span>
        </div>
      {/if}
      {#each terms as [name, entry] (name)}
        <SteerCard {name} {entry} />
      {/each}
    </div>
  {/if}

  <!-- The family's launcher stays reachable in both empty + populated
       states — the tab is the group, so there is exactly one. -->
  <div class="actions" class:empty={count === 0}>
    {#if family === "subspace"}
      <button
        type="button"
        class="add-subspace"
        onclick={() => openDrawer("subspace")}
        title="add subspace"
      >
        + subspace steer
      </button>
    {:else}
      <button
        type="button"
        class="add-manifold"
        onclick={() => openDrawer("manifolds")}
        title="add manifold"
      >
        + manifold steer
      </button>
    {/if}
  </div>
</section>

<style>
  /* A flat section of the inspector panel — no border box, no own
   * background; borderless — the STEER/PROBE title typography and each
   * section's own padding carry the divide from the probe section below.
   * Fixed chrome + one scrollable middle. */
  .rack {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-5);
    background: transparent;
    height: 100%;
    min-height: 0;
    max-height: 100%;
    overflow: hidden;
  }

  /* Strips own the scroll — overflow at the rack level would push the
   * actions row off-screen when terms pile up. */
  .strips {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    /* Size from the current cards so the rack starts compact, then shrink
     * into its own scroller when InspectorPanel's half-height cap lands. */
    flex: 0 1 auto;
    min-height: 2.4rem;
    max-height: 100%;
    overflow-y: auto;
    padding-right: var(--space-1);
  }
  /* Shared subspace-along master — sits between the section header and its
   * cards, reading as a group-level control rather than a per-card one. */
  .along-master {
    display: grid;
    grid-template-columns: minmax(3em, auto) minmax(0, 1fr) 3em;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    padding: var(--space-1) var(--space-2) var(--space-2);
    border-left: 2px solid var(--accent);
  }
  .along-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: lowercase;
  }
  .along-val {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }

  /* Anchored at the bottom of the rack.  Borderless — the padding-top
   * mirrors the probe rack's actions row for visual symmetry.  Shown in
   * both empty and populated states — the two family launchers are the
   * only first-run affordance (no teaching copy). */
  .actions {
    flex: 0 0 auto;
    padding-top: var(--space-4);
    display: flex;
    gap: var(--space-2);
  }
  /* With no card list between header and launchers, the tighter gap is
   * the same empty STEER treatment as J-LENS. */
  .actions.empty {
    padding-top: 0;
  }
  /* The two family launchers — white subspace vs purple manifold so they
   * read as the two card families. */
  .add-subspace,
  .add-manifold {
    min-height: 24px;
    flex: 1 1 0;
    border: 1px solid transparent;
    padding: 2px var(--space-5);
    border-radius: var(--radius);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: normal;
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .add-subspace {
    background: var(--accent-subtle);
    color: var(--accent);
  }
  .add-subspace:hover {
    background: var(--accent-glow);
  }
  /* Manifold launcher — purple-tinted to echo the manifold card's accent
   * so the two surfaces read as one feature. */
  .add-manifold {
    background: color-mix(in srgb, var(--accent-purple) 10%, transparent);
    color: var(--accent-purple);
  }
  .add-manifold:hover {
    background: color-mix(in srgb, var(--accent-purple) 18%, transparent);
  }
</style>
