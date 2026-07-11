<script lang="ts">
  // Workspace token card — one aggregate-readout token as a RackCard,
  // replacing the chip cloud + separate matrix view.  Blue j-lens accent.
  //
  //   statline : □ pin glyph (the steer cards' square, click to pin) ·
  //              token · @com ±spread · strength-history sparkline
  //   body     : the strength bar (mean band probability, absolute 0→1 —
  //              calibrated across tokens and steps), then the per-layer
  //              strength strip.
  //
  // The strip carries the matrix's information per token: each cell is
  // the token's per-layer softmax probability ``p_l(v)`` — the streamed
  // readout rows carry probabilities directly, so this is the SAME unit
  // as the bar (strength = their band mean), just per layer.  Cell color
  // normalizes to the card's own max (absolute p spans orders of
  // magnitude); the tooltip carries the absolute value.  An empty cell
  // means the token sits below that layer's top-k cutoff (the full
  // ranking lives in the token drilldown's j-lens tab).

  import Bar from "../../lib/charts/Bar.svelte";
  import Sparkline from "../../lib/charts/Sparkline.svelte";
  import RackCard from "./RackCard.svelte";
  import ProbePinButton from "./ProbePinButton.svelte";
  import LayerStrip from "./LayerStrip.svelte";

  interface Props {
    /** Raw vocabulary token text (untrimmed). */
    token: string;
    /** Mean band probability, 0..1 (the aggregate ``strength``). */
    strength: number;
    /** Salience-weighted depth center of mass, 0..1. */
    com: number;
    /** Depth spread around the CoM. */
    spread: number;
    /** Recent strength history (0 = fell below the streamed top-k). */
    series: number[];
    /** Streamed layer list — the strip's cells, in order. */
    layers: number[];
    /** The per-layer top-k matrix (``lensState.readout``). */
    readout: Record<string, [string, number][]> | null;
    /** Whether ``jlens/<word>`` is already pinned as a probe. */
    pinned: boolean;
    /** Pin in flight — disables the action. */
    busy: boolean;
    onpin: (word: string) => void;
  }

  let {
    token, strength, com, spread, series, layers, readout, pinned, busy, onpin,
  }: Props = $props();

  const display = $derived(token.trim() || JSON.stringify(token));
  /** Whitespace-only tokens have no pinnable single-token word. */
  const pinnable = $derived(token.trim().length > 0);

  /** Per-layer softmax probability for this token, straight off the
   *  streamed readout row; ``null`` = below the top-k cutoff. */
  const cells = $derived.by(() =>
    layers.map((layer) => {
      const pairs = readout?.[String(layer)];
      if (!pairs || pairs.length === 0) return { layer, p: null as number | null };
      const hit =
        pairs.find(([text]) => text === token) ??
        pairs.find(([text]) => text.trim() === display);
      if (!hit) return { layer, p: null as number | null };
      return { layer, p: hit[1] };
    }),
  );

  /** Color scale — the card's own max p (absolute p spans orders of
   *  magnitude, so a fixed 0→1 scale would render near-black strips). */
  const cellScale = $derived(
    Math.max(...cells.map((c) => c.p ?? 0), 1e-12),
  );

  function cellTooltip(cell: (typeof cells)[number]): string {
    if (cell.p === null) return `L${cell.layer} · below top-k`;
    return `L${cell.layer} · p ${cell.p.toPrecision(3)}`;
  }

  const layerCells = $derived(
    cells.map((cell) => ({
      layer: cell.layer,
      value: cell.p,
      title: cellTooltip(cell),
    })),
  );
</script>

<RackCard accent="--accent-blue" disabled={false}>
  {#snippet statline()}
    <ProbePinButton
      shape="square"
      {pinned}
      disabled={pinned || busy || !pinnable}
      onclick={() => onpin(display)}
      title={pinned
        ? `jlens/${display} is already pinned`
        : `Pin jlens/${display} as a token probe`}
      ariaLabel={`Pin ${display} as a J-lens probe`}
    />

    <span class="name" title={`"${token}" — aggregate workspace token`}>
      {display}
    </span>

    <span
      class="com"
      title="depth center of mass ±spread, weighted by per-layer probability (0 = first block, 1 = last)"
    >@{com.toFixed(2)} ±{spread.toFixed(2)}</span>

    <span class="spacer"></span>

    <Sparkline points={series} width={56} height={14} color="var(--card-accent)" />
  {/snippet}

  {#snippet body()}
    <!-- Strength: mean band probability, absolute 0→1. -->
    <div class="reading">
      <span
        class="row-label"
        title="strength — mean probability of this token across the workspace band (0–1)"
      >strength</span>
      <div class="bar-cell" aria-hidden="true">
        <Bar value={strength} max={1} width={160} height={8} color="var(--card-accent)" />
      </div>
      <span class="filler" aria-hidden="true"></span>
      <span class="value">{strength.toFixed(2)}</span>
    </div>

    <!-- Per-layer strength strip with L endcaps. -->
    <LayerStrip
      cells={layerCells}
      scale={cellScale}
      ariaLabel={`Per-layer strength for ${display}`}
      emptyMessage="no layer data"
    />
  {/snippet}
</RackCard>

<style>
  /* ----- statline ----- */
  .name {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }
  .com {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  /* ----- body: strength row — ProbeCard's EXACT four-column grid
     (label · bar · nearest-or-empty · value), so the bar column aligns
     pixel-for-pixel with the CAA cards across the tab switch. ----- */
  .reading {
    display: grid;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    min-height: 24px;
    grid-template-columns: minmax(2.5em, 1fr) minmax(60px, 2.6fr) minmax(2.5em, 1fr) 3.5em;
  }
  .filler {
    min-width: 0;
  }
  .row-label {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .bar-cell {
    min-width: 0;
  }
  .bar-cell :global(.bar) {
    width: 100%;
    height: 8px;
    display: block;
  }
  .value {
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
    min-width: 3.5em;
    text-align: right;
    flex: 0 0 auto;
  }

</style>
