<script lang="ts">
  // The steering rack — two harmonised groups split on geometry, not
  // artifact type:
  //
  //   subspace (flat) — every vectorRack entry (2-node bipolar) plus every
  //     manifoldRack entry whose catalog fit_mode is pca / baked (a flat
  //     affine subspace, e.g. personas).
  //   manifold (curved) — the remaining manifoldRack entries (e.g. pad).
  //
  // Every row wears the same RackCard chrome; the family is signalled only
  // by accent colour + marker glyph.  Light group sub-headers; an empty
  // group hides.  Footer keeps the + add steering / + add manifold entry
  // points and their drawer targets.

  import VectorSteerCard from "./rack/VectorSteerCard.svelte";
  import ManifoldSteerCard from "./rack/ManifoldSteerCard.svelte";
  import {
    vectorRack,
    manifoldRack,
    manifoldByName,
    openDrawer,
  } from "../lib/stores.svelte";

  // Vectors are always subspace.  Alphabetized for stable order — Map
  // iteration tracks insertion which makes the rack jump around.
  const sortedVectors = $derived.by(() => {
    const arr = [...vectorRack.entries.entries()];
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });

  /** A manifold rack entry is subspace iff its catalog fit_mode is flat
   *  (pca / baked); otherwise it's the curved manifold family. */
  function isFlatManifold(name: string): boolean {
    const fm = manifoldByName(name)?.fit_mode;
    return fm === "pca" || fm === "baked";
  }

  const flatManifolds = $derived.by(() => {
    const arr = [...manifoldRack.entries.entries()].filter(([n]) =>
      isFlatManifold(n),
    );
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });

  const curvedManifolds = $derived.by(() => {
    const arr = [...manifoldRack.entries.entries()].filter(
      ([n]) => !isFlatManifold(n),
    );
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });

  const subspaceCount = $derived(sortedVectors.length + flatManifolds.length);
  const manifoldCount = $derived(curvedManifolds.length);
  const count = $derived(subspaceCount + manifoldCount);
</script>

<section class="rack" aria-label="Steering rack">
  <header class="header">
    <div class="header-text">
      <span class="title">STEERING</span>
    </div>
    <span class="count" aria-live="polite">
      {count} term{count === 1 ? "" : "s"}
    </span>
  </header>

  <div class="strips" class:is-empty={count === 0}>
    {#if count === 0}
      <div class="empty">
        <p class="empty-copy">
          Steering shapes how the model responds.
          Add a vector or a manifold to begin.
        </p>
        <div class="empty-actions">
          <button
            type="button"
            class="add-steering"
            onclick={() => openDrawer("vectors")}
            title="Browse the concept catalog or extract a custom vector"
          >
            + add steering
          </button>
          <button
            type="button"
            class="add-manifold"
            onclick={() => openDrawer("manifolds")}
            title="Rack a fitted steering manifold or build a new one"
          >
            + add manifold
          </button>
        </div>
      </div>
    {:else}
      {#if subspaceCount > 0}
        <h3 class="group-header subspace">subspace</h3>
        {#each sortedVectors as [name, entry] (name)}
          <VectorSteerCard {name} {entry} />
        {/each}
        {#each flatManifolds as [name, entry] (name)}
          <ManifoldSteerCard {name} {entry} />
        {/each}
      {/if}
      {#if manifoldCount > 0}
        <h3 class="group-header manifold">manifold</h3>
        {#each curvedManifolds as [name, entry] (name)}
          <ManifoldSteerCard {name} {entry} />
        {/each}
      {/if}
    {/if}
  </div>

  {#if count > 0}
    <div class="actions">
      <button
        type="button"
        class="add-steering"
        onclick={() => openDrawer("vectors")}
        title="Browse the concept catalog or extract a custom vector"
      >
        + add steering
      </button>
      <button
        type="button"
        class="add-manifold"
        onclick={() => openDrawer("manifolds")}
        title="Rack a fitted steering manifold or build a new one"
      >
        + add manifold
      </button>
    </div>
  {/if}
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

  /* First-run teaching state — one line of plain copy above the primary
   * action. */
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
    max-width: 28ch;
  }
  /* Empty-state stacks the two entry points vertically so the manifold
   * launcher reads as the second way in, not buried in an actions row. */
  .empty-actions {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    width: 14em;
  }

  /* Anchored at the bottom of the rack.  Border-top mirrors the probe
   * rack's actions row for visual symmetry. */
  .actions {
    flex: 0 0 auto;
    border-top: 1px solid var(--border);
    padding-top: var(--space-4);
    display: flex;
    gap: var(--space-2);
  }
  /* Primary entry point — the one obvious way to add a steering vector. */
  .add-steering {
    width: 100%;
    background: var(--accent-subtle);
    color: var(--accent);
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
  .actions .add-steering {
    flex: 1 1 0;
  }
  .empty-actions .add-steering,
  .empty-actions .add-manifold {
    width: 100%;
  }
  .add-steering:hover {
    background: var(--accent-glow);
  }
  /* Manifold launcher — purple-tinted to echo the manifold card's accent
   * so the two surfaces read as one feature. */
  .add-manifold {
    flex: 1 1 0;
    background: rgba(167, 139, 250, 0.10);
    color: var(--accent-purple);
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
  .add-manifold:hover {
    background: rgba(167, 139, 250, 0.18);
  }
</style>
