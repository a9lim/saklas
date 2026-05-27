<script lang="ts">
  // One row in the probe rack — visual shape mirrors VectorStrip so the
  // two racks read as one family: ``[●] [neg-pole | bar | pos-pole] [val]
  // [sparkline] [✕]``.  The bipolar bar fills outward from a center tick
  // — same axis direction the steering slider uses, so a probe reading
  // "more happy" and a steering term pushing toward happy point the same
  // way on screen.  The whole row body is the click target for toggling
  // the highlight selection.  A per-layer heatmap renders directly
  // beneath; that stays a webui-only addition (TUI exposes it via /why).
  //
  // Selection ●/○ glyph reuses the highlight-blue accent so a glance
  // distinguishes "this term is steering" (green) from "this probe is
  // the highlight" (blue).

  import Bar from "../lib/charts/Bar.svelte";
  import Sparkline from "../lib/charts/Sparkline.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import {
    deactivateProbe,
    highlightState,
    probeRack,
    setHighlightTarget,
  } from "../lib/stores.svelte";
  import { polesOf } from "../lib/concepts";

  interface Props {
    name: string;
  }

  let { name }: Props = $props();

  // Live entry view — re-reads from the rack on every paint so live
  // sparkline + per-layer updates from updateProbeFromScores propagate.
  const entry = $derived(probeRack.entries.get(name));
  const current = $derived(entry?.current ?? 0);
  const sparkline = $derived(entry?.sparkline ?? []);
  const isHighlight = $derived(highlightState.target === name);

  // Bipolar / monopolar axis — same grammar VectorStrip reads.  A
  // monopolar probe collapses to a single positive pole on the right.
  const poles = $derived(polesOf(name));
  const monopolar = $derived(poles.negative === null);

  // Layer keys sorted ascending (numeric).  The wire shape is zero-
  // padded ints keyed as strings; Number() coerces cleanly for any
  // base-10 prefix the server emits.
  const layerKeys = $derived<string[]>(
    entry?.perLayer
      ? Object.keys(entry.perLayer).sort((a, b) => Number(a) - Number(b))
      : [],
  );

  function cellTooltip(layer: string): string {
    const v = entry?.perLayer?.[layer];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      return `L${layer} · —`;
    }
    const sign = v >= 0 ? "+" : "";
    return `L${layer} · ${sign}${v.toFixed(3)}`;
  }

  // Cell width — 14px reads cleanly on a 28-layer Gemma without forcing
  // horizontal scroll on a 1280px viewport, and stays usable on 60+
  // layer models (e.g. larger qwen variants) when the strip wraps inside
  // its own scroll container.
  const CELL_SIZE = 14;

  // Click anywhere on the row toggles highlight: select if not selected,
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

  function onRemove(ev: MouseEvent): void {
    ev.stopPropagation();
    void deactivateProbe(name);
  }
</script>

<div class="strip" class:selected={isHighlight}>
  <div
    class="row"
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
      aria-hidden="true"
      title={isHighlight ? "Selected (click to deselect)" : "Click to select for highlighting"}
    >{isHighlight ? "●" : "○"}</span>

    <!-- Bipolar axis frame, identical shape to VectorStrip's .axis: the
         negative pole sits left of the bar, the positive right.  A
         monopolar probe renders an empty left-pole slot so the bar +
         positive-pole line up horizontally with bipolar rows in the same
         rack. -->
    <div class="axis">
      <span class="pole neg" aria-hidden={monopolar}>
        {#if !monopolar}{poles.negative}{/if}
      </span>
      <div class="bar-cell" aria-hidden="true">
        <!-- Always center-anchored — monopolar probes share the same
             0-in-the-middle axis as bipolar rows so the reader doesn't
             have to switch mental models between adjacent rows. -->
        <Bar value={current} max={1} width={160} height={8} bipolar />
      </div>
      <span class="pole pos" title={`positive pole (${poles.positive})`}>
        {poles.positive}
      </span>
    </div>

    <span class="value" class:pos={current > 0} class:neg={current < 0}>
      {current >= 0 ? "+" : ""}{current.toFixed(2)}
    </span>

    <Sparkline points={sparkline} width={56} height={14} />

    <button
      type="button"
      class="icon remove"
      aria-label="Remove probe {name}"
      title="Remove probe"
      onclick={onRemove}
    >✕</button>
  </div>

  <div class="layers" aria-label="Per-layer readings for {name}">
    {#if layerKeys.length === 0}
      <div class="layers-status">no data yet, generate a token first</div>
    {:else}
      <span class="endcap" aria-hidden="true">L{Number(layerKeys[0])}</span>
      <div class="cells">
        {#each layerKeys as layer (layer)}
          <HeatmapCell
            value={entry?.perLayer?.[layer]}
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
</div>

<style>
  /* Match VectorStrip's outer frame so steering and probe rows read as
   * one visual family — same border, radius, background, font-size. */
  .strip {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-alt);
    transition: border-color var(--dur-fast) var(--ease-out);
    font-size: var(--text-sm);
  }
  .strip.selected {
    border-color: var(--accent);
  }

  /* Row body — flex / 32px / 0.4em gap / 0.25em·0.4em padding to match
   * VectorStrip exactly.  The whole row is the click target for toggling
   * highlight selection. */
  .row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 32px;
    padding: var(--space-2) var(--space-3);
    cursor: pointer;
    user-select: none;
  }
  .row:hover {
    background: var(--bg-elev);
  }
  .row:focus-visible {
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }

  /* Selection indicator — same ●/○ glyph as VectorStrip's .enable, but
   * tuned to the highlight-blue accent so the colour distinguishes the
   * two semantics ("this term is steering" green vs "this probe is
   * highlighted" blue). */
  .select-glyph {
    color: var(--fg-muted);
    font-size: var(--text);
    line-height: 1;
    padding: 0 var(--space-1);
    flex: 0 0 auto;
  }
  .strip.selected .select-glyph {
    color: var(--accent);
  }

  /* Bipolar axis — grid mirrors VectorStrip.axis exactly so the two
   * racks share the same column proportions.  Monopolar keeps the same
   * three columns and renders an empty left-pole slot so the bar and
   * positive pole still sit in the same horizontal positions as the
   * bipolar rows above and below it in the rack. */
  .axis {
    display: grid;
    grid-template-columns: minmax(2.5em, 1fr) minmax(60px, 2.6fr) minmax(2.5em, 1fr);
    align-items: center;
    gap: var(--space-2);
    flex: 1 1 auto;
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

  /* Bar lives in the axis middle slot — fluid width via :global so the
   * SVG stretches to fill the column rather than forcing the row to
   * exactly 160px-wide. */
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
  .value.pos {
    color: var(--accent-green);
  }
  .value.neg {
    color: var(--accent-red);
  }

  /* Icon button — same shape as VectorStrip's .icon. */
  .icon {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text);
    line-height: 1;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    flex: 0 0 auto;
  }
  .icon:hover:not(:disabled) {
    color: var(--fg-strong);
    background: var(--bg-elev);
  }
  .remove:hover:not(:disabled) {
    color: var(--accent-red);
  }

  .layers {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-2) var(--space-3) var(--space-3) var(--space-3);
    border-top: 1px solid var(--border);
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
