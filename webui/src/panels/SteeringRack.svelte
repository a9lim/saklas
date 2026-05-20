<script lang="ts">
  // The steering rack — section header, one VectorStrip per loaded
  // vector (alphabetized for stable ordering across re-renders), and
  // the + add steering action button.

  import VectorStrip from "./VectorStrip.svelte";
  import { vectorRack, openDrawer } from "../lib/stores.svelte";

  // Reactive entries — sorted alphabetically by name for stable order.
  // The Map iteration order tracks insertion which makes the rack
  // visually jump around as vectors land out of order; sorting fixes it.
  const sortedEntries = $derived.by(() => {
    const arr = [...vectorRack.entries.entries()];
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });

  const count = $derived(sortedEntries.length);
</script>

<section class="rack" aria-label="Steering rack">
  <header class="header">
    <div class="header-text">
      <span class="title">STEERING</span>
    </div>
    <span class="count" aria-live="polite">
      {count} vector{count === 1 ? "" : "s"}
    </span>
  </header>

  <div class="strips" class:is-empty={count === 0}>
    {#if count === 0}
      <div class="empty">
        <p class="empty-copy">
          Steering shapes how the model responds.
          Add a concept to begin.
        </p>
        <button
          type="button"
          class="add-steering"
          onclick={() => openDrawer("vector_picker")}
        >
          + add steering
        </button>
      </div>
    {:else}
      {#each sortedEntries as [name, entry] (name)}
        <VectorStrip {name} {entry} />
      {/each}
    {/if}
  </div>

  {#if count > 0}
    <div class="actions">
      <button
        type="button"
        class="add-steering"
        onclick={() => openDrawer("vector_picker")}
        title="Browse the concept catalog or extract a custom vector"
      >
        + add steering
      </button>
    </div>
  {/if}
</section>

<style>
  /* A flat section of the inspector panel — no border box, no own
   * background; the only chrome is the border-bottom hairline dividing
   * it from the probe section below.  Fixed chrome + one scrollable
   * middle.  This deliberately uses flex instead of grid: the generated
   * ``.strips`` element was able to grow past the rack in some viewport
   * sizes, hiding the apply-vector controls. */
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
    color: var(--accent-blue);
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
   * actions row off-screen when vectors pile up. */
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
  /* First-run teaching state — one line of plain copy above the primary
   * action.  Replaces the bare "no active steering vectors". */
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

  /* Anchored at the bottom of the rack.  Border-top mirrors the probe
   * rack's actions row for visual symmetry. */
  .actions {
    flex: 0 0 auto;
    border-top: 1px solid var(--border);
    padding-top: var(--space-4);
  }
  /* Primary entry point — the one obvious way to add a steering vector. */
  .add-steering {
    width: 100%;
    background: var(--accent-subtle);
    color: var(--accent-blue);
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
  .empty .add-steering {
    width: auto;
    min-width: 14em;
  }
  .add-steering:hover {
    background: var(--accent-glow);
  }
</style>
