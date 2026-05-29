<script lang="ts">
  // One row in the manifold-probe rack — the read-side counterpart to
  // ProbeStrip.  Mirrors that component's frame so vector and manifold
  // probes read as one family: same border-radius, same row min-height,
  // same axis-grid column proportions, same Sparkline + value + ✕
  // affordances, and the same expanded per-layer cells underneath.  The
  // accent stays purple to keep the probe-family distinction (steering vs
  // reading is signalled by the rack section; vector vs manifold by the
  // accent) — see ProbeRack's notes.
  //
  // What's different on this row, by *data shape*:
  //   * the bar reads ``fraction`` (∈ [0, 1], monopolar — fills from the
  //     left) instead of a bipolar [-1, +1] cosine
  //   * the right side of the axis grid hosts the top-1 nearest-node
  //     label (manifold's natural "where is this?" readout), where
  //     ProbeStrip carries the +pole label
  //   * the per-layer cells render ``fraction_per_layer`` from the
  //     end-of-generation aggregate (the live per-token wire shape
  //     doesn't carry per-layer fractions; they land at finalize)
  //   * a 2D box-domain probe additionally renders a node-layout mini-
  //     map with the inferred trajectory — that's manifold-only and
  //     stays below the layer strip
  //
  // Detach is the single ✕ icon on the row.  The strip is read-only —
  // the picker for adding new probes lives at the rack level.

  import Bar from "../lib/charts/Bar.svelte";
  import Sparkline from "../lib/charts/Sparkline.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import ManifoldMiniMap from "./manifold/ManifoldMiniMap.svelte";
  import {
    detachManifoldProbe,
    manifoldProbeRack,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";

  interface Props {
    name: string;
  }

  let { name }: Props = $props();

  const entry = $derived(manifoldProbeRack.entries.get(name));
  const fraction = $derived(entry?.current ?? 0);
  const sparkline = $derived(entry?.sparkline ?? []);
  const nearest = $derived(entry?.nearest ?? []);
  const aggregate = $derived(entry?.aggregate ?? null);
  const trajectory = $derived(entry?.trajectory ?? []);

  /** Mini-map gating — only 2D box-domain probes with attached node
   *  coords render the visual.  Mirrors ``_isMiniMapCandidate`` in
   *  stores. */
  const showMiniMap = $derived.by(() => {
    const info = entry?.info;
    if (!info) return false;
    if (info.intrinsic_dim !== 2) return false;
    const d = info.domain as { type?: string };
    if (d?.type !== "box") return false;
    return !!info.node_coords && info.node_coords.length > 0;
  });

  /** First nearest tuple for the inline readout — slots into the
   *  right "pos-pole" position of the axis grid so the manifold strip
   *  occupies the same horizontal lanes as a vector ProbeStrip. */
  const top = $derived(nearest.length > 0 ? nearest[0] : null);
  const nearestLabel = $derived(top?.[0] ?? "");
  const nearestDistance = $derived(top?.[1] ?? null);

  /** Per-layer fraction cells.  Sourced from the end-of-generation
   *  aggregate (``ManifoldAggregate.fraction_per_layer``) rather than
   *  the live per-token wire shape — token-time manifold readings
   *  intentionally carry only the aggregate fraction (avoids paying a
   *  per-layer transfer per token).  Keys are sorted numerically to
   *  match ProbeStrip's per-layer strip cadence. */
  const layerKeys = $derived<string[]>(
    aggregate?.fraction_per_layer
      ? Object.keys(aggregate.fraction_per_layer).sort(
          (a, b) => Number(a) - Number(b),
        )
      : [],
  );

  /** Tooltip per per-layer cell — ``L<layer> · 0.42`` so a hover gives
   *  the precise fraction at that layer.  Matches ProbeStrip's cell
   *  tooltip cadence. */
  function cellTooltip(layer: string): string {
    const v = aggregate?.fraction_per_layer?.[layer];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      return `L${layer} · —`;
    }
    return `L${layer} · ${v.toFixed(3)}`;
  }

  /** Cell width 14px — same as ProbeStrip so a 28-layer Gemma reads
   *  cleanly without horizontal scroll on a 1280px viewport, and 60+
   *  layer models stay usable when the strip wraps inside its own
   *  scroll container. */
  const CELL_SIZE = 14;

  function fmtFraction(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }
  function fmtDistance(v: number | null): string {
    return v !== null && Number.isFinite(v) ? v.toFixed(2) : "—";
  }
  function fmtCoord(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }

  async function onDetach(): Promise<void> {
    try {
      await detachManifoldProbe(name);
      pushToast(`detached manifold probe ${name}`, { kind: "info" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      pushToast(`detach ${name} failed — ${msg}`, {
        kind: "error",
        ttlMs: null,
      });
    }
  }
</script>

<div class="strip">
  <div class="row">
    <!-- Family glyph occupies the same column slot ProbeStrip's ●/○
         selection glyph holds, so the two row families line up across
         the rack.  Static (no click semantics) — manifold-probe
         highlight wiring is a separate piece of work; the slot is here
         purely for column-width parity. -->
    <span
      class="family-glyph"
      aria-hidden="true"
      title="manifold probe"
    >◆</span>

    <!-- Axis grid mirrors ProbeStrip.axis exactly — same three-column
         layout, same gap, so a vector probe and a manifold probe in
         the same rack occupy the same horizontal lanes.  The left slot
         carries the probe name (+ alias when the registered name
         differs from the underlying manifold); the right slot carries
         the top-1 nearest-node label where a vector ProbeStrip
         carries its +pole label. -->
    <div class="axis">
      <div class="name-cell">
        <span class="name">{name}</span>
        {#if entry?.info && entry.info.manifold !== name}
          <span class="alias">{entry.info.manifold}</span>
        {/if}
      </div>
      <div class="bar-cell" aria-hidden="true">
        <!-- Monopolar fill (left-anchored) — fraction is ∈ [0, 1], so
             the bipolar centered-axis would lie about the data shape.
             The bar still occupies the same column slot a bipolar
             ProbeStrip bar does. -->
        <Bar value={fraction} max={1} width={160} height={8} />
      </div>
      <span
        class="nearest"
        title={top
          ? `nearest node: ${nearestLabel} (distance ${fmtDistance(nearestDistance)})`
          : "awaiting first token"}
      >
        {#if top}
          <span class="nearest-label">{nearestLabel}</span>
          <span class="nearest-dist">d={fmtDistance(nearestDistance)}</span>
        {:else}
          <span class="nearest-empty">—</span>
        {/if}
      </span>
    </div>

    <span
      class="value"
      title="EV-weighted subspace fraction across {entry?.info.layers.length ?? 0} layers"
    >{fmtFraction(fraction)}</span>

    <Sparkline points={sparkline} width={56} height={14} cap={1} />

    <button
      type="button"
      class="icon remove"
      aria-label="Detach manifold probe {name}"
      title="Detach probe"
      onclick={onDetach}
    >✕</button>
  </div>

  <!-- Per-layer fraction heatmap — same shape ProbeStrip uses for the
       per-layer cosine strip.  Sourced from the end-of-gen aggregate;
       shows a teaching state until at least one generation has
       completed.  Mirrors ProbeStrip's "no data yet" empty-state copy
       so the two racks teach the same way. -->
  <div class="layers" aria-label="Per-layer fraction for {name}">
    {#if layerKeys.length === 0}
      <div class="layers-status">
        no data yet, generate a token first
      </div>
    {:else}
      <span class="endcap" aria-hidden="true">L{Number(layerKeys[0])}</span>
      <div class="cells">
        {#each layerKeys as layer (layer)}
          <HeatmapCell
            value={aggregate?.fraction_per_layer?.[layer]}
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

  {#if aggregate && aggregate.coords.length > 0}
    <div class="coords-readout" title="settled inverse-projection coords">
      coords ({aggregate.coords.map(fmtCoord).join(", ")})
    </div>
  {/if}

  {#if showMiniMap && entry}
    <div class="map-wrap">
      <ManifoldMiniMap
        info={entry.info}
        trajectory={trajectory}
        settled={aggregate?.coords ?? null}
      />
    </div>
  {/if}
</div>

<style>
  /* Outer frame matches ProbeStrip — same border, radius, bg, font —
   * but the left edge carries a thin purple bar so the manifold family
   * stays visually distinct in a rack that mixes vector probes above. */
  .strip {
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent-purple);
    border-radius: var(--radius);
    background: var(--bg-alt);
    font-size: var(--text-sm);
    transition: border-color var(--dur-fast) var(--ease-out);
  }

  /* Row body — matches ProbeStrip.row exactly (same min-height, same
   * gap, same padding) so the two probe families align across the
   * shared rack. */
  .row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 32px;
    padding: var(--space-2) var(--space-3);
  }

  /* Static family glyph — occupies the same column slot ProbeStrip's
   * ●/○ selection glyph carries, keeping the two strips perfectly
   * aligned across the rack.  Highlight wiring for manifold probes is
   * a separate feature; until then the glyph is signalling only. */
  .family-glyph {
    color: var(--accent-purple);
    font-size: var(--text);
    line-height: 1;
    padding: 0 var(--space-1);
    flex: 0 0 auto;
  }

  /* Axis grid — same three-column shape and proportions as
   * ProbeStrip.axis so vector and manifold rows share the same
   * horizontal column lanes (left meta / bar / right meta). */
  .axis {
    display: grid;
    grid-template-columns: minmax(2.5em, 1fr) minmax(60px, 2.6fr) minmax(2.5em, 1fr);
    align-items: center;
    gap: var(--space-2);
    flex: 1 1 auto;
    min-width: 0;
  }
  .name-cell {
    display: flex;
    flex-direction: column;
    min-width: 0;
    /* Right-align so the name presses against the bar like ProbeStrip's
     * negative-pole label does.  Keeps the column lanes tight. */
    align-items: flex-end;
    text-align: right;
  }
  .name {
    color: var(--accent-purple);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
  }
  .alias {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
  }
  .bar-cell {
    min-width: 0;
  }
  .bar-cell :global(.bar) {
    width: 100%;
    height: 8px;
    display: block;
  }

  /* Right "+pole" slot — repurposed to carry the manifold's natural
   * read-side label (nearest node + distance).  Same column geometry
   * a ProbeStrip pos-pole label occupies, so vector and manifold rows
   * align across the rack. */
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

  /* Value column — same width and alignment as ProbeStrip's .value so
   * the two probe families' numeric readouts line up vertically across
   * the rack. */
  .value {
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
    min-width: 3.5em;
    text-align: right;
    flex: 0 0 auto;
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
  }
  .icon:hover:not(:disabled) {
    color: var(--accent-red);
    background: var(--bg-elev);
  }

  /* Per-layer cells — same shape ProbeStrip.layers carries.  Empty
   * "no data" copy matches the vector strip's first-run state so the
   * two rack families teach the same way. */
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

  /* Settled coords readout — purple to echo the manifold accent.  Sits
   * between the per-layer strip and the mini-map; the manifold-only
   * piece below the parity-shared header above. */
  .coords-readout {
    border-top: 1px solid var(--border);
    padding: var(--space-2) var(--space-3);
    color: var(--accent-purple);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
  }

  .map-wrap {
    border-top: 1px solid var(--border);
    padding: var(--space-3);
    display: flex;
    justify-content: center;
  }
</style>
