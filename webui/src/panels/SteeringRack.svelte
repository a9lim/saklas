<script lang="ts">
  // The steering rack — one unified steerRack, two harmonised groups split
  // on geometry, the entry's own ``mode``:
  //
  //   subspace (flat)   — a 2-node bipolar axis through the rank-8 personas
  //     fan.  Shares one rack-level "subspace along" master (the merged
  //     affine subspace slides once), so the cards carry no per-card along.
  //   manifold (curved) — curved fits (e.g. emotions), each its own injection with
  //     a per-card along + onto.
  //
  // Every row is one SteerCard wearing the same RackCard chrome; the family
  // is signalled only by accent colour + marker glyph.  Light group
  // sub-headers; an empty group hides.  Footer keeps the + add steering /
  // + add manifold entry points and their drawer targets.

  import SteerCard from "./rack/SteerCard.svelte";
  import Slider from "../lib/Slider.svelte";
  import {
    steerRack,
    setSubspaceAlong,
    openDrawer,
  } from "../lib/stores.svelte";

  // Alphabetized for stable order — Map iteration tracks insertion which
  // makes the rack jump around.
  const subspaceTerms = $derived.by(() => {
    const arr = [...steerRack.entries.entries()].filter(([, e]) => e.mode === "subspace");
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });
  const curvedTerms = $derived.by(() => {
    const arr = [...steerRack.entries.entries()].filter(([, e]) => e.mode === "manifold");
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });

  const subspaceCount = $derived(subspaceTerms.length);
  const manifoldCount = $derived(curvedTerms.length);
  const count = $derived(subspaceCount + manifoldCount);

  function onAlongInput(v: number): void {
    if (Number.isFinite(v)) setSubspaceAlong(v);
  }
</script>

<section class="rack" aria-label="Steering rack">
  <header class="header">
    <div class="header-text">
      <span class="title">STEER</span>
    </div>
    <span class="count" aria-live="polite">
      {count} term{count === 1 ? "" : "s"}
    </span>
  </header>

  <div class="strips" class:is-empty={count === 0}>
    {#if count > 0}
      {#if subspaceCount > 0}
        <h3 class="group-header subspace">subspace</h3>
        <!-- Shared "subspace along" master — the single slide magnitude every
             flat term serializes with (the merged affine subspace slides
             once).  Per-term relative weight lives in each card's position. -->
        <div
          class="along-master"
          title="shared subspace-along — one slide magnitude for every flat term"
        >
          <span class="along-label">along</span>
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
        {#each subspaceTerms as [name, entry] (name)}
          <SteerCard {name} {entry} />
        {/each}
      {/if}
      {#if manifoldCount > 0}
        <h3 class="group-header manifold">manifold</h3>
        {#each curvedTerms as [name, entry] (name)}
          <SteerCard {name} {entry} />
        {/each}
      {/if}
    {/if}
  </div>

  <!-- Launchers stay reachable in both empty + populated states — the two
       family entry points, white subspace vs purple manifold. -->
  <div class="actions">
    <button
      type="button"
      class="add-subspace"
      onclick={() => openDrawer("subspace")}
      title="Browse flat subspaces — concept axes and personas"
    >
      + subspace steer
    </button>
    <button
      type="button"
      class="add-manifold"
      onclick={() => openDrawer("manifolds")}
      title="Browse curved steering manifolds"
    >
      + manifold steer
    </button>
  </div>
</section>

<style>
  /* A flat section of the inspector panel — no border box, no own
   * background; the only chrome is the border-bottom hairline dividing
   * it from the probe section below.  Fixed chrome + one scrollable
   * middle. */
  .rack {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-5);
    background: transparent;
    border-bottom: 1px solid var(--border);
    height: 100%;
    min-height: 0;
    max-height: 100%;
    overflow: hidden;
  }

  .header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-3);
    border-bottom: 1px solid var(--border);
    padding-bottom: var(--space-3);
  }
  .header-text {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .title {
    font-weight: var(--weight-bold);
    color: var(--accent);
    letter-spacing: 0;
    font-size: var(--text-sm);
    text-transform: uppercase;
  }
  .count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    flex: 0 0 auto;
  }

  /* Strips own the scroll — overflow at the rack level would push the
   * actions row off-screen when terms pile up. */
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

  /* Light group sub-headers — name the geometry family without competing
   * with the section title.  Accent-coded to match the cards' left
   * stripe so the eye links header → rows. */
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

  /* Shared subspace-along master — sits between the subspace header and its
   * cards, reading as a group-level control rather than a per-card one. */
  .along-master {
    display: grid;
    grid-template-columns: minmax(3em, auto) 1fr 3em;
    align-items: center;
    gap: var(--space-2);
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

  /* Anchored at the bottom of the rack.  Border-top mirrors the probe
   * rack's actions row for visual symmetry.  Shown in both empty and
   * populated states — the two family launchers are the only first-run
   * affordance (no teaching copy). */
  .actions {
    flex: 0 0 auto;
    border-top: 1px solid var(--border);
    padding-top: var(--space-4);
    display: flex;
    gap: var(--space-2);
  }
  /* The two family launchers — white subspace vs purple manifold so they
   * read as the two card families. */
  .add-subspace,
  .add-manifold {
    flex: 1 1 0;
    border: 1px solid var(--border);
    min-height: 2.1rem;
    padding: var(--space-4) var(--space-5);
    border-radius: var(--radius);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
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
