<script lang="ts">
  // One row in the probe rack — selection indicator + name + sparkline +
  // value bar + value + ✕, with a horizontal per-layer reading strip
  // rendered directly beneath.  The row body is the click target for
  // toggling the highlight selection: click anywhere on the row to
  // select this probe as the chat-token highlight; click the same row
  // again to deselect (highlight off).  ✕ stops propagation so removal
  // doesn't accidentally toggle selection on the way out.
  //
  // Visual frame matches VectorStrip: flex layout, 32px min-height,
  // 0.85em font, 0.25em padding, 0.4em gap.  Same "● / ○" glyph as the
  // vector's enable button, recoloured to the highlight-blue accent so
  // a glance distinguishes "this term is steering" (green) from "this
  // probe is the highlight" (blue).
  //
  // Mirrors saklas/tui/trait_panel.py for the row visual rhythm; the
  // layer strip is a webui-only addition (the TUI's per-layer view
  // lives in `/why`).

  import Bar from "../lib/charts/Bar.svelte";
  import Sparkline from "../lib/charts/Sparkline.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import {
    deactivateProbe,
    highlightState,
    probeRack,
    setHighlightTarget,
  } from "../lib/stores.svelte";

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
      title={isHighlight ? "Selected — click to deselect" : "Click to select for highlighting"}
    >{isHighlight ? "●" : "○"}</span>

    <span class="name" title={name}>{name}</span>

    <span class="spacer" aria-hidden="true"></span>

    <Sparkline points={sparkline} width={56} height={14} />

    <Bar value={current} max={1} width={96} height={8} />

    <span class="value" class:pos={current > 0} class:neg={current < 0}>
      {current >= 0 ? "+" : ""}{current.toFixed(2)}
    </span>

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
      <div class="layers-status">no data — generate a token first</div>
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
    border: 1px solid var(--border-dim);
    border-radius: 3px;
    background: var(--bg-alt);
    transition: border-color 0.1s ease;
    font-size: 0.85em;
  }
  .strip.selected {
    border-color: var(--accent-blue);
  }

  /* Row body — flex / 32px / 0.4em gap / 0.25em·0.4em padding to match
   * VectorStrip exactly.  The whole row is the click target for toggling
   * highlight selection. */
  .row {
    display: flex;
    align-items: center;
    gap: 0.4em;
    min-height: 32px;
    padding: 0.25em 0.4em;
    cursor: pointer;
    user-select: none;
  }
  .row:hover {
    background: var(--bg-elev);
  }
  .row:focus-visible {
    outline: 1px solid var(--accent-blue);
    outline-offset: -1px;
  }

  /* Selection indicator — same ●/○ glyph as VectorStrip's .enable, but
   * tuned to the highlight-blue accent so the colour distinguishes the
   * two semantics ("this term is steering" green vs "this probe is
   * highlighted" blue). */
  .select-glyph {
    color: var(--fg-muted);
    font-size: 1em;
    line-height: 1;
    padding: 0 0.2em;
    flex: 0 0 auto;
  }
  .strip.selected .select-glyph {
    color: var(--accent-blue);
  }

  .name {
    flex: 0 1 auto;
    min-width: 5em;
    max-width: 14em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--fg-strong);
  }
  /* Eats the available row width so the sparkline + bar + value + ✕
   * cluster pins to the right edge — keeps the readings group aligned
   * across rows regardless of name length. */
  .spacer {
    flex: 1 1 auto;
    min-width: 0.4em;
  }
  .strip.selected .name {
    color: var(--accent-blue);
    font-weight: bold;
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
    font-size: 0.95em;
    line-height: 1;
    padding: 0.1em 0.35em;
    border-radius: 2px;
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
    gap: 0.4em;
    padding: 0.3em 0.4em 0.4em 0.4em;
    border-top: 1px solid var(--border-dim);
    overflow-x: auto;
    white-space: nowrap;
  }
  .layers-status {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    padding: 0.1em 0;
  }
  .cells {
    display: flex;
    gap: 0;
    flex: 0 0 auto;
  }
  .endcap {
    color: var(--fg-dim);
    font-size: var(--font-size-tiny);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
</style>
