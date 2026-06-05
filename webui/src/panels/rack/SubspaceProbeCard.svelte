<script lang="ts">
  // Subspace probe card — the harmonised replacement for ProbeStrip.  A
  // flat (affine) probe is the subspace family: accent ``--accent``, glyph
  // ●/○.  Composes the shared RackCard chrome:
  //
  //   statline : ●/○ highlight-select · name · sparkline · ✕ detach
  //   body     : the reading row (bipolar bar OR fraction bar + nearest)
  //              then the per-layer heatmap strip
  //
  // Selection ●/○ glyph reuses the accent so a glance distinguishes "this
  // probe is the highlight" (selected, accent) from unselected (muted).
  // The whole statline identity cluster is the click target for toggling
  // the highlight selection — preserving ProbeStrip's behavior.

  import type { ProbeRackEntry } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import Sparkline from "../../lib/charts/Sparkline.svelte";
  import HeatmapCell from "../../lib/charts/HeatmapCell.svelte";
  import {
    detachProbe,
    highlightState,
    setHighlightTarget,
  } from "../../lib/stores.svelte";
  import { polesOf } from "../../lib/concepts";
  import RackCard from "./RackCard.svelte";

  interface Props {
    name: string;
    entry: ProbeRackEntry;
  }

  let { name, entry }: Props = $props();

  const current = $derived(entry.current ?? 0);
  const sparkline = $derived(entry.sparkline ?? []);
  const isHighlight = $derived(highlightState.target === name);

  /** Bipolar (2-node) probes read poles off the underlying manifold (or the
   *  registered name, when the manifold name carries the ``a.b`` form).  A
   *  higher-rank flat probe (e.g. personas) renders the fraction + nearest
   *  readout instead. */
  const bipolar = $derived(entry.info.node_count === 2);
  const poles = $derived(polesOf(entry.info.manifold || name));
  const monopolar = $derived(poles.negative === null);

  /** Top nearest node, for the higher-rank flat readout. */
  const topNearest = $derived(entry.nearest.length > 0 ? entry.nearest[0] : null);

  // Layer keys sorted ascending (numeric).  The wire shape is zero-padded
  // ints keyed as strings; Number() coerces cleanly for any base-10 prefix.
  const layerKeys = $derived<string[]>(
    entry.perLayer
      ? Object.keys(entry.perLayer).sort((a, b) => Number(a) - Number(b))
      : [],
  );

  function cellTooltip(layer: string): string {
    const v = entry.perLayer?.[layer];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      return `L${layer} · —`;
    }
    const sign = v >= 0 ? "+" : "";
    return `L${layer} · ${sign}${v.toFixed(3)}`;
  }

  // 14px reads cleanly on a 28-layer Gemma without forcing horizontal
  // scroll on a 1280px viewport, and stays usable on 60+ layer models when
  // the strip wraps inside its own scroll container.
  const CELL_SIZE = 14;

  // Click the identity cluster toggles highlight: select if not selected,
  // deselect (back to "off") if already selected.  Mirrors the TUI's
  // /probe behavior — one click anchors the row, click again to unset.
  function toggleHighlight(): void {
    setHighlightTarget(isHighlight ? null : name);
  }

  function onRowKey(ev: KeyboardEvent): void {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      toggleHighlight();
    }
  }

  function onDetach(ev: MouseEvent): void {
    ev.stopPropagation();
    void detachProbe(name);
  }

  function fmtFraction(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }
  function fmtDistance(v: number | null): string {
    return v !== null && Number.isFinite(v) ? v.toFixed(2) : "—";
  }
</script>

<RackCard accent="--accent" disabled={false}>
  {#snippet statline()}
    <!-- Identity cluster — clicking it toggles the highlight selection. -->
    <div
      class="select-cluster"
      role="button"
      tabindex="0"
      aria-pressed={isHighlight}
      aria-label={isHighlight
        ? `Deselect ${name} as highlight target`
        : `Select ${name} as highlight target`}
      onclick={toggleHighlight}
      onkeydown={onRowKey}
    >
      <span
        class="select-glyph"
        class:selected={isHighlight}
        aria-hidden="true"
        title={isHighlight ? "Selected (click to deselect)" : "Click to select for highlighting"}
      >{isHighlight ? "●" : "○"}</span>
      <span class="name" title="probe {name}">{name}</span>
    </div>

    <span class="spacer"></span>

    <Sparkline points={sparkline} width={56} height={14} />

    <button
      type="button"
      class="icon remove"
      aria-label="Detach probe {name}"
      title="Detach probe"
      onclick={onDetach}
    >✕</button>
  {/snippet}

  {#snippet body()}
    {#if bipolar}
      <!-- Bipolar reading row: neg pole · signed bar · pos pole · value. -->
      <div class="reading">
        <span class="pole neg" aria-hidden={monopolar}>
          {#if !monopolar}{poles.negative}{/if}
        </span>
        <div class="bar-cell" aria-hidden="true">
          <Bar value={current} max={1} width={160} height={8} bipolar />
        </div>
        <span class="pole pos" title={`positive pole (${poles.positive})`}>
          {poles.positive}
        </span>
        <span class="value" class:pos={current > 0} class:neg={current < 0}>
          {current >= 0 ? "+" : ""}{current.toFixed(2)}
        </span>
      </div>
    {:else}
      <!-- Higher-rank flat (e.g. personas): a fraction bar + nearest-node
           readout.  ``current`` is the signed axis-0 coordinate the store
           tracks for a flat probe. -->
      <div class="reading">
        <span class="rank-label">subspace</span>
        <div class="bar-cell" aria-hidden="true">
          <Bar value={Math.abs(current)} max={1} width={160} height={8} />
        </div>
        <span class="nearest" title="nearest node">
          {#if topNearest}
            <span class="nearest-label">{topNearest[0]}</span>
            <span class="nearest-dist">d={fmtDistance(topNearest[1])}</span>
          {:else}
            <span class="nearest-empty">—</span>
          {/if}
        </span>
        <span class="value">{fmtFraction(current)}</span>
      </div>
    {/if}

    <!-- Per-layer heatmap strip with L0 / Ln endcaps. -->
    <div class="layers" aria-label="Per-layer readings for {name}">
      {#if layerKeys.length === 0}
        <div class="layers-status">no data yet, generate a token first</div>
      {:else}
        <span class="endcap" aria-hidden="true">L{Number(layerKeys[0])}</span>
        <div class="cells">
          {#each layerKeys as layer (layer)}
            <HeatmapCell
              value={entry.perLayer?.[layer]}
              size={CELL_SIZE}
              title={cellTooltip(layer)}
            />
          {/each}
        </div>
        <span class="endcap" aria-hidden="true">
          L{Number(layerKeys[layerKeys.length - 1])}
        </span>
      {/if}
    </div>
  {/snippet}
</RackCard>

<style>
  /* ----- statline ----- */
  .select-cluster {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    cursor: pointer;
    user-select: none;
    border-radius: var(--radius);
  }
  .select-cluster:hover {
    background: var(--bg-elev);
  }
  .select-cluster:focus-visible {
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }
  .select-glyph {
    color: var(--fg-muted);
    font-size: var(--text);
    line-height: 1;
    padding: 0 var(--space-1);
    flex: 0 0 auto;
  }
  .select-glyph.selected {
    color: var(--card-accent);
  }
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
  .icon {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text);
    line-height: 1;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    flex: 0 0 auto;
    cursor: pointer;
    transition: color var(--dur) var(--ease-out),
      background var(--dur) var(--ease-out);
  }
  .icon:hover:not(:disabled) {
    color: var(--fg-strong);
    background: var(--bg-elev);
  }
  .remove:hover:not(:disabled) {
    color: var(--accent-red);
  }

  /* ----- body: reading row ----- */
  .reading {
    display: grid;
    grid-template-columns: minmax(2.5em, 1fr) minmax(60px, 2.6fr) minmax(2.5em, 1fr) 3.5em;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .pole {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: var(--text-sm);
  }
  .pole.neg {
    color: var(--fg-muted);
    text-align: right;
  }
  .pole.pos {
    color: var(--fg-strong);
    text-align: left;
  }
  .rank-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bar-cell {
    min-width: 0;
  }
  .bar-cell :global(.bar) {
    width: 100%;
    height: 8px;
    display: block;
  }
  .nearest {
    display: inline-flex;
    flex-direction: column;
    overflow: hidden;
    white-space: nowrap;
    font-size: var(--text-sm);
    color: var(--fg-strong);
    text-align: left;
    min-width: 0;
  }
  .nearest-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--fg-strong);
  }
  .nearest-dist {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
  }
  .nearest-empty {
    color: var(--fg-muted);
    font-style: italic;
  }
  .value {
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
    min-width: 3.5em;
    text-align: right;
    flex: 0 0 auto;
  }
  .value.pos {
    color: var(--accent-green);
  }
  .value.neg {
    color: var(--accent-red);
  }

  /* ----- body: per-layer heatmap ----- */
  .layers {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    overflow-x: auto;
    white-space: nowrap;
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
