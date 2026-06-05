<script lang="ts">
  // The one shared rack-card chrome — every steer/probe row, subspace or
  // manifold, wears this.  Per the design: a single STATLINE on top
  // (marker glyph · name · status · actions, all on one row) with the
  // controls (steer) or meters (probe) STACKED vertically below it —
  // never inline with the name.  Cards differ only by:
  //   - accent  : the family colour (subspace vs manifold) — the left
  //               stripe + whatever the statline inherits via the
  //               ``--card-accent`` custom property.
  //   - the marker glyph the card draws in its ``statline`` snippet (the
  //               shape accent: ●/○ for subspace, ◆/◇ for manifold).
  // Everything else is identical, which is the harmonisation the steer and
  // probe / subspace and manifold surfaces were asked to share.

  import type { Snippet } from "svelte";

  interface Props {
    /** CSS custom-property *name* for the family accent — e.g. ``"--accent"``
     *  for a subspace (flat) card, ``"--accent-purple"`` for a manifold
     *  (curved) card.  Exposed to the slotted content as ``--card-accent``. */
    accent?: string;
    /** Dim + de-emphasise the card (a disabled steer term).  Probe cards
     *  pass ``false`` — a probe is always "on". */
    disabled?: boolean;
    /** Top identity line: marker glyph · name · status chips · actions.
     *  One row; the card owns its glyph + chips. */
    statline: Snippet;
    /** Stacked body — labelled control rows (steer) or meter rows (probe),
     *  laid out top-down below the statline. */
    body: Snippet;
  }

  let {
    accent = "--accent",
    disabled = false,
    statline,
    body,
  }: Props = $props();
</script>

<div class="card" class:disabled style:--card-accent="var({accent})" role="group">
  <div class="statline">{@render statline()}</div>
  <div class="body">{@render body()}</div>
</div>

<style>
  .card {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--border);
    /* The colour accent — flat vs curved family at a glance. */
    border-left: 2px solid var(--card-accent);
    border-radius: var(--radius);
    background: var(--bg-alt);
    font-size: var(--text-sm);
    transition:
      border-color var(--dur) var(--ease-out),
      opacity var(--dur) var(--ease-out);
  }
  .card.disabled {
    opacity: 0.5;
  }

  /* STATLINE — the identity row.  Marker · name · status · actions share
     one line; nothing stacks here. */
  .statline {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 24px;
  }

  /* BODY — controls / meters stack vertically beneath the statline. */
  .body {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
</style>
