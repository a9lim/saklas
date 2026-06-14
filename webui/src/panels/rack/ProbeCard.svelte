<script lang="ts">
  // Unified probe card — one row for every probe shape.  Replaces the
  // SubspaceProbeCard / ManifoldProbeCard split: the store already
  // normalises ``current`` / ``sparkline`` / ``perLayer`` per family
  // (``_primaryScalar`` / ``_primaryPerLayer``), so this row reads those
  // uniformly and branches only on *presentation*.
  //
  //   statline : ●/○ (flat) or ◆/◇ (curved) highlight-select · name
  //              · sparkline · ✕ detach
  //   body     : the reading row —
  //                bipolar bar + poles   (is_affine && node_count===2), or
  //                scalar/fraction bar + nearest   (everything else)
  //              then the curved-only settled coords/residual meta, the
  //              per-layer heatmap strip, and a 2-D box-domain mini-map.
  //
  // Family is signalled by accent (--accent flat / --accent-purple curved)
  // and glyph.  The identity cluster toggles the transcript highlight for
  // *every* family — the per-token score map is keyed by probe name
  // regardless of geometry, so curved probes are valid highlight targets
  // (the top-bar dropdown already lists them; this wires the click too).

  import type { ProbeRackEntry } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import Sparkline from "../../lib/charts/Sparkline.svelte";
  import HeatmapCell from "../../lib/charts/HeatmapCell.svelte";
  import { nodeCoordExtent } from "../../lib/tokens";
  import ManifoldMiniMap from "../manifold/ManifoldMiniMap.svelte";
  import {
    detachProbe,
    highlightState,
    openDrawer,
    setHighlightTarget,
  } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import { polesOf } from "../../lib/concepts";
  import RackCard from "./RackCard.svelte";

  interface Props {
    name: string;
    entry: ProbeRackEntry;
  }

  let { name, entry }: Props = $props();

  const info = $derived(entry.info);
  /** Flat (affine) ⇒ subspace family; curved ⇒ manifold family. */
  const affine = $derived(info.is_affine);
  /** A bipolar 2-node concept axis — the one shape with a signed,
   *  pole-anchored reading.  Every higher-rank flat fan and every curved
   *  fit renders the scalar/fraction row instead. */
  const bipolar = $derived(affine && info.node_count === 2);

  const accent = $derived(affine ? "--accent" : "--accent-purple");

  /** Saturation scale for the bar + layer cells.  A flat probe reads a
   *  signed domain-frame ``coords[0]`` whose units are the fit's own node
   *  layout (a fan runs to ±tens), so we normalize by the axis-0 node
   *  extent — "full = as far along as the most extreme node".  A curved
   *  probe reads a ``[0,1]`` fraction, already unit-normalized, so it
   *  stays on the fixed unit scale. */
  const axisScale = $derived(affine ? nodeCoordExtent(info.node_coords, 0) : 1);

  const current = $derived(entry.current ?? 0);
  const sparkline = $derived(entry.sparkline ?? []);
  const isHighlight = $derived(highlightState.target === name);

  /** Highlight-select glyph — family marker (●/◆) filled when this probe is
   *  the highlight target, hollow (○/◇) when not. */
  const selectGlyph = $derived(
    affine ? (isHighlight ? "●" : "○") : (isHighlight ? "◆" : "◇"),
  );

  /** Display name — bare manifold name, namespace prefix stripped; full
   *  name stays in the tooltip. */
  const displayName = $derived(name.split("/").pop() ?? name);

  // ---------- bipolar poles ----------
  const poles = $derived(polesOf(info.manifold || name));
  const monopolar = $derived(poles.negative === null);

  // ---------- nearest (store keeps this settled post-gen) ----------
  const topNearest = $derived(entry.nearest.length > 0 ? entry.nearest[0] : null);
  const nearestLabel = $derived(topNearest?.[0] ?? "");
  const nearestDistance = $derived(topNearest?.[1] ?? null);

  // ---------- curved settled meta (end-of-gen aggregate) ----------
  const aggregate = $derived(entry.aggregate ?? null);
  const coords = $derived(aggregate?.coords ?? []);
  const residual = $derived(aggregate?.residual ?? null);
  const trajectory = $derived(entry.trajectory ?? []);

  /** Mini-map gating — only 2-D box-domain probes with attached node coords
   *  render the visual (intrinsic_dim 1 of a 2-node axis and the custom
   *  carrier of a flat discover fan both fall out here). */
  const showMiniMap = $derived.by(() => {
    if (info.intrinsic_dim !== 2) return false;
    const d = info.domain as { type?: string };
    if (d?.type !== "box") return false;
    return !!info.node_coords && info.node_coords.length > 0;
  });

  // ---------- per-layer strip (store-normalised primary column) ----------
  const layerKeys = $derived<string[]>(
    entry.perLayer
      ? Object.keys(entry.perLayer).sort((a, b) => Number(a) - Number(b))
      : [],
  );

  // 14px reads cleanly on a 28-layer Gemma without horizontal scroll on a
  // 1280px viewport, and stays usable on 60+ layer models when the strip
  // wraps inside its own scroll container.
  const CELL_SIZE = 14;

  function cellTooltip(layer: string): string {
    const v = entry.perLayer?.[layer];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      return `L${layer} · —`;
    }
    // Flat per-layer is signed (axis-0); curved is a [0,1] fraction.
    const sign = affine && v >= 0 ? "+" : "";
    return `L${layer} · ${sign}${v.toFixed(3)}`;
  }

  function fmtFraction(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }
  function fmtDistance(v: number | null): string {
    return v !== null && Number.isFinite(v) ? v.toFixed(2) : "—";
  }
  function fmtCoord(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }

  // ---------- highlight select / detach ----------

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

  // Open the per-probe inspector (whitened geometry plot + layer norms + live
  // trail).  Distinct gesture from the identity-cluster highlight toggle, so
  // ``stopPropagation`` keeps the two from colliding.
  function onInspect(ev: MouseEvent): void {
    ev.stopPropagation();
    openDrawer("probe_inspector", { name });
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

<RackCard {accent} disabled={false}>
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
        title={isHighlight
          ? "Selected (click to deselect)"
          : "Click to select for highlighting"}
      >{selectGlyph}</span>
      <span class="name" title="probe {name}">{displayName}</span>
    </div>

    <span class="spacer"></span>

    <!-- Curved (fraction) probes are [0,1]-bounded → cap the sparkline at 1;
         flat (signed) probes auto-scale. -->
    <Sparkline points={sparkline} width={56} height={14} cap={affine ? undefined : 1} />

    <button
      type="button"
      class="icon inspect"
      aria-label="Inspect probe {name}"
      title="Open probe inspector"
      onclick={onInspect}
    >ⓘ</button>

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
      <div class="reading reading-bipolar">
        <span class="pole neg" aria-hidden={monopolar}>
          {#if !monopolar}{poles.negative}{/if}
        </span>
        <div class="bar-cell" aria-hidden="true">
          <Bar value={current} max={axisScale} width={160} height={8} bipolar />
        </div>
        <span class="pole pos" title={`positive pole (${poles.positive})`}>
          {poles.positive}
        </span>
        <span class="value" class:pos={current > 0} class:neg={current < 0}>
          {current >= 0 ? "+" : ""}{current.toFixed(2)}
        </span>
      </div>
    {:else}
      <!-- Scalar / fraction row — same 4-column grid as the bipolar row so
           the bar sits in the identical inset slot and reads center-zero.
           Column 1 (the neg-pole slot) is an empty spacer; the nearest node
           takes the pos-pole slot.  A higher-rank flat fan reads the signed
           axis-0 magnitude (centered, sign-coloured like a bipolar axis); a
           curved fit reads the [0,1] subspace fraction (fills rightward). -->
      <div class="reading reading-scalar">
        <span class="pole neg" aria-hidden="true"></span>
        <div class="bar-cell" aria-hidden="true">
          <Bar value={current} max={axisScale} width={160} height={8} bipolar />
        </div>
        <span
          class="nearest"
          title={topNearest
            ? `nearest node: ${nearestLabel} (distance ${fmtDistance(nearestDistance)})`
            : "awaiting first token"}
        >
          {#if topNearest}
            <span class="nearest-label">{nearestLabel}</span>
            <span class="nearest-dist">d={fmtDistance(nearestDistance)}</span>
          {:else}
            <span class="nearest-empty">—</span>
          {/if}
        </span>
        <span
          class="value"
          class:pos={affine && current > 0}
          class:neg={affine && current < 0}
        >
          {#if affine}{current >= 0 ? "+" : ""}{current.toFixed(2)}{:else}{fmtFraction(current)}{/if}
        </span>
      </div>
    {/if}

    {#if !affine && (coords.length > 0 || residual !== null)}
      <!-- Curved-only settled meta: where the trajectory came to rest. -->
      <div class="meta">
        {#if coords.length > 0}
          <span class="meta-item" title="settled inverse-projection coords">
            coords ({coords.map(fmtCoord).join(", ")})
          </span>
        {/if}
        {#if residual !== null}
          <span class="meta-item" title="normalized off-surface residual">
            residual {fmtCoord(residual)}
          </span>
        {/if}
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

    {#if showMiniMap}
      <div class="map-wrap">
        <ManifoldMiniMap
          info={entry.info}
          trajectory={trajectory}
          settled={aggregate?.coords ?? null}
        />
      </div>
    {/if}
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
  .inspect:hover:not(:disabled) {
    color: var(--accent-purple);
  }

  /* ----- body: reading row ----- */
  .reading {
    display: grid;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  /* Both rows share the grid so the bar lands in the same inset slot and
     the cards line up vertically.  Bipolar: neg pole · bar · pos pole ·
     value.  Scalar: empty spacer · bar · nearest · value. */
  .reading-bipolar,
  .reading-scalar {
    grid-template-columns: minmax(2.5em, 1fr) minmax(60px, 2.6fr) minmax(2.5em, 1fr) 3.5em;
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

  /* ----- body: curved settled meta ----- */
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

  /* ----- body: mini-map ----- */
  .map-wrap {
    display: flex;
    justify-content: center;
  }
</style>
