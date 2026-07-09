<script lang="ts">
  // J-lens probe card — one pinned ``jlens/<word>`` token probe, wearing
  // the same RackCard chrome as the probe tab's cards.  The entry is an
  // ordinary probeRack row (the jlens direction registers as a rank-1
  // flat probe), so the sparkline / per-layer trace / depth stats are all
  // already in the store — this card renders what the old chip dropped.
  //
  //   statline : ■/□ highlight-select · word · sparkline · ✕ detach
  //   body     : the signed whitened-coordinate bar (axis 0 along the
  //              lens direction), the depth-CoM meta row, and the
  //              per-layer heatmap strip with L endcaps.
  //
  // No subspaceness row (a rank-1 token direction's fraction carries no
  // extra signal over the coordinate) and no nearest-node cell (the fit
  // has no real nodes) — otherwise a full sibling of ProbeCard.

  import type { ProbeRackEntry } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import HeatmapCell from "../../lib/charts/HeatmapCell.svelte";
  import Sparkline from "../../lib/charts/Sparkline.svelte";
  import { nodeCoordExtent, parseProbeTarget } from "../../lib/tokens";
  import {
    detachProbe,
    highlightState,
    setHighlightTarget,
  } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import RackCard from "./RackCard.svelte";

  interface Props {
    name: string;
    entry: ProbeRackEntry;
  }

  let { name, entry }: Props = $props();

  const word = $derived(name.slice("jlens/".length));

  // ---------- latest reading: live during gen, settled (aggregate) after ----------
  const latest = $derived(entry.aggregate ?? entry.reading);
  /** Whitened coordinate along the lens direction (axis 0). */
  const value = $derived(latest?.coords?.[0] ?? entry.current ?? 0);
  /** Bar + heatmap saturation scale — node extent when the registered
   *  rank-1 fit carries coords, else the fixed unit fallback. */
  const axisScale = $derived(nodeCoordExtent(entry.info.node_coords, 0));

  const sparkline = $derived(entry.sparkline ?? []);

  const depthCom = $derived(latest?.depth_com?.[0] ?? null);
  const depthSpread = $derived(latest?.depth_spread?.[0] ?? null);

  // ---------- highlight select (same gesture as ProbeCard) ----------
  const isHighlight = $derived(
    highlightState.target !== null &&
      parseProbeTarget(highlightState.target).base === name,
  );
  const selectGlyph = $derived(isHighlight ? "■" : "□");

  function toggleHighlight(): void {
    setHighlightTarget(isHighlight ? null : name);
  }

  function onRowKey(ev: KeyboardEvent): void {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      toggleHighlight();
    }
  }

  // ---------- per-layer strip ----------
  const layerKeys = $derived<string[]>(
    entry.perLayer
      ? Object.keys(entry.perLayer).sort((a, b) => Number(a) - Number(b))
      : [],
  );

  const CELL_SIZE = 14;

  function cellTooltip(layer: string): string {
    const v = entry.perLayer?.[layer];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      return `L${layer} · —`;
    }
    return `L${layer} · ${v >= 0 ? "+" : ""}${v.toFixed(3)}`;
  }

  function fmtCoord(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }

  async function onDetach(ev: MouseEvent): Promise<void> {
    ev.stopPropagation();
    try {
      await detachProbe(name);
      pushToast(`detached probe ${name}`, { kind: "info" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      pushToast(`detach ${name} failed — ${msg}`, {
        kind: "error",
        ttlMs: null,
      });
    }
  }
</script>

<RackCard accent="--accent-blue" disabled={false}>
  {#snippet statline()}
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
        title={isHighlight
          ? "Selected (click to deselect)"
          : "Click to select for highlighting"}
      >{selectGlyph}</span>
      <span class="name" title="probe {name} — whitened coordinate along the lens direction">
        {word}
      </span>
    </div>

    <span class="spacer"></span>

    <Sparkline points={sparkline} width={56} height={14} />

    <button
      type="button"
      class="icon remove"
      aria-label="Unpin probe {name}"
      title="Unpin probe"
      onclick={onDetach}
    >✕</button>
  {/snippet}

  {#snippet body()}
    <!-- The one meter: signed whitened coordinate along the token's lens
         direction (positive = the activation leans toward saying it). -->
    <div class="reading">
      <span
        class="row-label"
        title="whitened coordinate along the jlens/{word} direction (axis 0)"
      >coord</span>
      <div class="bar-cell" aria-hidden="true">
        <Bar value={value} max={axisScale} width={160} height={8} bipolar />
      </div>
      <span class="value" class:pos={value > 0} class:neg={value < 0}>
        {value >= 0 ? "+" : ""}{value.toFixed(2)}
      </span>
    </div>

    {#if depthCom !== null}
      <div class="meta">
        <span
          class="meta-item"
          title="depth center of mass of the per-layer read, share-weighted (0 = first block, 1 = last)"
        >
          com {fmtCoord(depthCom)}{depthSpread !== null ? ` ±${fmtCoord(depthSpread)}` : ""}
        </span>
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
              scale={axisScale}
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
  /* ----- statline (mirrors ProbeCard) ----- */
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
    outline: 1px solid var(--card-accent);
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

  /* ----- body: coordinate row (ProbeCard's reading grid) ----- */
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
  .value.pos {
    color: var(--accent-green);
  }
  .value.neg {
    color: var(--accent-red);
  }

  /* ----- body: meta ----- */
  .meta {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-3);
    color: var(--card-accent);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
  }
  .meta-item {
    flex: 0 0 auto;
  }

  /* ----- body: per-layer heatmap (mirrors ProbeCard) ----- */
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
