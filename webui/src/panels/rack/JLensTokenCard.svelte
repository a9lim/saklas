<script lang="ts">
  // Workspace token card — one aggregate-readout token as a RackCard,
  // replacing the chip cloud + separate matrix view.  Blue j-lens accent.
  //
  //   statline : token · @com ±spread · pin action
  //   body     : the strength bar (mean band probability, absolute 0→1 —
  //              calibrated across tokens and steps, so no more opacity
  //              encoding), then the per-layer salience strip.
  //
  // The strip carries the matrix's information per token: each cell is
  // the token's within-layer salience ``p_l(v) / max_v' p_l(v')`` — for
  // tokens present in a layer's streamed top-k this is EXACT, not an
  // approximation, because the readout rows arrive sorted by raw logit
  // and the softmax denominator cancels in the ratio:
  // ``exp(s_i − s_top)``.  An empty cell means the token sits below that
  // layer's top-k cutoff (the full ranking lives in the token
  // drilldown's j-lens tab).

  import Bar from "../../lib/charts/Bar.svelte";
  import HeatmapCell from "../../lib/charts/HeatmapCell.svelte";
  import RackCard from "./RackCard.svelte";

  interface Props {
    /** Raw vocabulary token text (untrimmed). */
    token: string;
    /** Mean band probability, 0..1 (the aggregate ``strength``). */
    strength: number;
    /** Salience-weighted depth center of mass, 0..1. */
    com: number;
    /** Depth spread around the CoM. */
    spread: number;
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

  let { token, strength, com, spread, layers, readout, pinned, busy, onpin }: Props =
    $props();

  const display = $derived(token.trim() || JSON.stringify(token));
  /** Whitespace-only tokens have no pinnable single-token word. */
  const pinnable = $derived(token.trim().length > 0);

  /** Per-layer within-layer salience for this token — exact ``p/max p``
   *  from the sorted raw-logit row; ``null`` = below the top-k cutoff. */
  const cells = $derived.by(() =>
    layers.map((layer) => {
      const pairs = readout?.[String(layer)];
      if (!pairs || pairs.length === 0) return { layer, salience: null as number | null };
      const hit =
        pairs.find(([text]) => text === token) ??
        pairs.find(([text]) => text.trim() === display);
      if (!hit) return { layer, salience: null as number | null };
      return { layer, salience: Math.exp(hit[1] - pairs[0][1]) };
    }),
  );

  function cellTooltip(cell: (typeof cells)[number]): string {
    if (cell.salience === null) return `L${cell.layer} · below top-k`;
    return `L${cell.layer} · salience ${cell.salience.toFixed(2)}`;
  }

  const CELL_SIZE = 14;
</script>

<RackCard accent="--accent-blue" disabled={false}>
  {#snippet statline()}
    <span class="name" title={`"${token}" — aggregate workspace token`}>
      {display}
    </span>

    <span class="spacer"></span>

    <span
      class="com"
      title="salience-weighted depth center of mass ±spread (0 = first block, 1 = last)"
    >@{com.toFixed(2)} ±{spread.toFixed(2)}</span>

    <button
      type="button"
      class="pin"
      disabled={pinned || busy || !pinnable}
      onclick={() => onpin(display)}
      title={pinned
        ? `jlens/${display} is already pinned`
        : `Pin jlens/${display} as a token probe`}
      aria-label="Pin {display} as a J-lens probe"
    >
      {pinned ? "pinned" : "+ pin"}
    </button>
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
      <span class="value">{strength.toFixed(2)}</span>
    </div>

    <!-- Per-layer salience strip with L endcaps. -->
    <div class="layers" aria-label="Per-layer salience for {display}">
      {#if cells.length === 0}
        <div class="layers-status">no layer data</div>
      {:else}
        <span class="endcap" aria-hidden="true">L{cells[0].layer}</span>
        <div class="cells">
          {#each cells as cell (cell.layer)}
            <HeatmapCell
              value={cell.salience}
              scale={1}
              size={CELL_SIZE}
              title={cellTooltip(cell)}
            />
          {/each}
        </div>
        <span class="endcap" aria-hidden="true">L{cells[cells.length - 1].layer}</span>
      {/if}
    </div>
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
  .pin {
    background: transparent;
    color: var(--fg-muted);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: var(--text-xs);
    line-height: 1.2;
    padding: var(--space-1) var(--space-3);
    flex: 0 0 auto;
    cursor: pointer;
    transition: color var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out);
  }
  .pin:hover:not(:disabled) {
    color: var(--card-accent);
    border-color: var(--card-accent);
  }
  .pin:disabled {
    opacity: 0.5;
    cursor: default;
  }

  /* ----- body: strength row (ProbeCard's reading grid) ----- */
  .reading {
    display: grid;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    min-height: 24px;
    grid-template-columns: minmax(2.5em, 1fr) minmax(60px, 2.6fr) 3.5em;
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

  /* ----- body: per-layer strip (mirrors ProbeCard) ----- */
  .layers {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    overflow-x: auto;
    white-space: nowrap;
    padding-top: var(--space-3);
    padding-bottom: var(--space-2);
  }
  .layers-status {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    padding: var(--space-1) 0;
  }
  .cells {
    display: flex;
    gap: 0;
    flex: 0 0 auto;
  }
  .endcap {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
</style>
