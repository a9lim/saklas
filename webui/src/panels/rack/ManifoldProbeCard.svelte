<script lang="ts">
  // Manifold probe card — the harmonised replacement for
  // ManifoldProbeStrip.  A curved fit is the manifold family: accent
  // ``--accent-purple``, glyph ◆.  Composes the shared RackCard chrome:
  //
  //   statline : ◆ · name · sparkline · ✕ detach
  //   body     : fraction bar + value · nearest-node + distance · coords
  //              readout · residual · per-layer fraction heatmap · a 2D
  //              box-domain mini-map (when the geometry supports it)
  //
  // What's different on this row by *data shape*: the bar reads
  // ``fraction`` (∈ [0, 1], monopolar — fills from the left) rather than a
  // bipolar cosine, and the readouts are the manifold's natural "where is
  // this?" answers (nearest node, settled coords, off-surface residual).

  import type { ProbeRackEntry } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import Sparkline from "../../lib/charts/Sparkline.svelte";
  import HeatmapCell from "../../lib/charts/HeatmapCell.svelte";
  import ManifoldMiniMap from "../manifold/ManifoldMiniMap.svelte";
  import { detachProbe } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import RackCard from "./RackCard.svelte";

  interface Props {
    name: string;
    entry: ProbeRackEntry;
  }

  let { name, entry }: Props = $props();

  const aggregate = $derived(entry.aggregate ?? null);
  const reading = $derived(entry.reading ?? null);
  const sparkline = $derived(entry.sparkline ?? []);
  const trajectory = $derived(entry.trajectory ?? []);

  /** Live / settled subspace fraction.  Prefer the settled aggregate when
   *  present; otherwise the live ``current`` scalar (which the store sets
   *  to the fraction for a curved probe). */
  const fraction = $derived(aggregate?.fraction ?? entry.current ?? 0);

  /** Display name — the bare manifold name, namespace prefix stripped. */
  const displayName = $derived(name.split("/").pop() ?? name);

  /** Top nearest tuple — prefer the settled aggregate, else the live
   *  per-token nearest list. */
  const top = $derived.by<[string, number] | null>(() => {
    const aggNearest = aggregate?.nearest;
    if (aggNearest && aggNearest.length > 0) return aggNearest[0];
    return entry.nearest.length > 0 ? entry.nearest[0] : null;
  });
  const nearestLabel = $derived(top?.[0] ?? "");
  const nearestDistance = $derived(top?.[1] ?? null);

  /** Settled coords + residual, from the end-of-gen aggregate. */
  const coords = $derived(aggregate?.coords ?? []);
  const residual = $derived(aggregate?.residual ?? null);

  /** Mini-map gating — only 2D box-domain probes with attached node coords
   *  render the visual. */
  const showMiniMap = $derived.by(() => {
    const info = entry.info;
    if (info.intrinsic_dim !== 2) return false;
    const d = info.domain as { type?: string };
    if (d?.type !== "box") return false;
    return !!info.node_coords && info.node_coords.length > 0;
  });

  /** Per-layer fraction cells.  Sourced from the live per-token reading
   *  when present, else the end-of-gen aggregate.  Keys sorted numerically
   *  to match the subspace card's per-layer strip cadence. */
  const fractionPerLayer = $derived<Record<string, number>>(
    reading?.fraction_per_layer ?? aggregate?.fraction_per_layer ?? {},
  );
  const layerKeys = $derived<string[]>(
    Object.keys(fractionPerLayer).sort((a, b) => Number(a) - Number(b)),
  );

  function cellTooltip(layer: string): string {
    const v = fractionPerLayer[layer];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      return `L${layer} · —`;
    }
    return `L${layer} · ${v.toFixed(3)}`;
  }

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
      await detachProbe(name);
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

<RackCard accent="--accent-purple" disabled={false}>
  {#snippet statline()}
    <span class="family-glyph" aria-hidden="true" title="manifold probe">◆</span>
    <span class="name" title={name}>{displayName}</span>

    <span class="spacer"></span>

    <Sparkline points={sparkline} width={56} height={14} cap={1} />

    <button
      type="button"
      class="icon remove"
      aria-label="Detach manifold probe {name}"
      title="Detach probe"
      onclick={onDetach}
    >✕</button>
  {/snippet}

  {#snippet body()}
    <!-- Fraction bar + value · nearest node + distance. -->
    <div class="reading">
      <div class="bar-cell" aria-hidden="true">
        <Bar value={fraction} max={1} width={160} height={8} />
      </div>
      <span
        class="value"
        title="EV-weighted subspace fraction across {entry.info.layers.length} layers"
      >{fmtFraction(fraction)}</span>
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

    {#if coords.length > 0 || residual !== null}
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

    <!-- Per-layer fraction heatmap. -->
    <div class="layers" aria-label="Per-layer fraction for {name}">
      {#if layerKeys.length === 0}
        <div class="layers-status">no data yet, generate a token first</div>
      {:else}
        <span class="endcap" aria-hidden="true">L{Number(layerKeys[0])}</span>
        <div class="cells">
          {#each layerKeys as layer (layer)}
            <HeatmapCell
              value={fractionPerLayer[layer]}
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
  .family-glyph {
    color: var(--card-accent);
    font-size: var(--text);
    line-height: 1;
    padding: 0 var(--space-1);
    flex: 0 0 auto;
  }
  .name {
    color: var(--card-accent);
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
    color: var(--accent-red);
    background: var(--bg-elev);
  }

  /* ----- body: reading row ----- */
  .reading {
    display: grid;
    grid-template-columns: minmax(60px, 2.6fr) 3.5em minmax(2.5em, 1fr);
    align-items: center;
    gap: var(--space-2);
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

  /* ----- body: settled meta ----- */
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
