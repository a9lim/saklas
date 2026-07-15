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
     *  for a subspace (flat) card, ``"--pillar-manifold"`` for a manifold
     *  (curved) card.  Exposed to the slotted content as ``--card-accent``. */
    accent?: string;
    /** Dim + de-emphasise the card (a disabled steer term).  Probe cards
     *  pass ``false`` — a probe is always "on". */
    disabled?: boolean;
    /** Alive right now — hue ring + faint glow (the highlight-selected
     *  probe, a gate that just fired).  Glow marks what is alive; the
     *  resting card stays calm. */
    active?: boolean;
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
    active = false,
    statline,
    body,
  }: Props = $props();
</script>

<div
  class="card"
  class:disabled
  class:active
  style:--card-accent="var({accent})"
  role="group"
>
  <div class="statline">{@render statline()}</div>
  <div class="body">{@render body()}</div>
</div>

<style>
  /* The dense variant of the v2 glass material (lib/ui/GlassCard is the
   * roomy one) — translucent fill lit from above. Borderless: the fill +
   * top-light carry the card, the family accent lives in the glyph/text,
   * and the border slot exists only for the active ring (state, not
   * chrome). Hover lifts the fill instead of drawing a line. No backdrop
   * blur: rack cards sit on the opaque panel, so blur would cost
   * compositing for nothing. */
  .card {
    display: flex;
    flex-direction: column;
    min-width: 0;
    max-width: 100%;
    gap: var(--space-2);
    padding: var(--space-3) var(--space-4);
    border: 1px solid transparent;
    border-radius: var(--radius-lg);
    background: var(--glass);
    box-shadow: var(--shadow-rack);
    font-size: var(--text-sm);
    transition:
      border-color var(--dur) var(--ease-out),
      background var(--dur) var(--ease-out),
      box-shadow var(--dur) var(--ease-out),
      opacity var(--dur) var(--ease-out);
  }
  .card:hover {
    background: var(--glass-strong);
  }
  .card.active {
    border-color: color-mix(in srgb, var(--card-accent) 40%, transparent);
    box-shadow: var(--shadow-rack-active);
  }
  .card.disabled {
    /* Off is a reversible state, not unavailable content. Keep labels and
     * the re-enable action at full contrast; the hollow marker, struck name,
     * and quieter surface communicate state without dimming the whole card. */
    background: color-mix(in srgb, var(--glass) 55%, transparent);
    box-shadow: none;
  }

  /* STATLINE — the identity row.  Marker · name · status · actions share
     one line; nothing stacks here. */
  .statline {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: var(--control-target);
    min-width: 0;
  }

  /* BODY — controls / meters stack vertically beneath the statline. */
  .body {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
</style>
