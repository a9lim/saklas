<script lang="ts">
  // Canonical four-column probe meter used by every inspector pillar.
  // Callers supply semantics; this component owns the exact label/bar/
  // context/value geometry so tab switches cannot drift by a pixel.

  import type { Snippet } from "svelte";

  let {
    left,
    bar,
    middle,
    right,
    ariaLabel,
  }: {
    left: Snippet;
    bar: Snippet;
    middle: Snippet;
    right: Snippet;
    ariaLabel?: string;
  } = $props();
</script>

<div class="reading" role="group" aria-label={ariaLabel}>
  {@render left()}
  <div class="bar">{@render bar()}</div>
  <div class="middle">{@render middle()}</div>
  {@render right()}
</div>

<style>
  .reading {
    display: grid;
    grid-template-columns:
      minmax(2.5em, 1fr) minmax(60px, 2.6fr) minmax(2.5em, 1fr)
      3.5em;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    min-height: 24px;
  }
  .bar,
  .middle {
    min-width: 0;
  }
  .bar :global(.bar) {
    width: 100%;
    height: 8px;
    display: block;
  }
</style>
