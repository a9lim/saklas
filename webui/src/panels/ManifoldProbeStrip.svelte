<script lang="ts">
  // One row in the manifold-probe rack — the read-side counterpart to
  // ProbeStrip.  Renders:
  //   * the probe name (purple-tinted, matching ManifoldStrip's accent so
  //     manifold surfaces — steering and reading — read as one family)
  //   * a Bar showing the EV-weighted ``fraction`` (subspace occupancy
  //     in the manifold's PCA basis, ∈ [0, 1])
  //   * a sparkline of fraction history
  //   * the top-1 nearest-node label + distance (the hover-readout for
  //     "where does this activation sit on the manifold")
  //   * (for 2D box manifolds only) a small node-layout mini-map with the
  //     inferred trajectory overlay
  //
  // Detach is the single icon button on the row.  The strip is purely
  // read-only — the picker for adding new probes lives at the rack
  // level.

  import Bar from "../lib/charts/Bar.svelte";
  import Sparkline from "../lib/charts/Sparkline.svelte";
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

  /** First nearest tuple for the inline readout.  Empty until the first
   *  token streams. */
  const top = $derived(nearest.length > 0 ? nearest[0] : null);

  function fmtFraction(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }
  function fmtDistance(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "—";
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
    <div class="name-cell">
      <span class="name">{name}</span>
      {#if entry?.info && entry.info.manifold !== name}
        <span class="alias">{entry.info.manifold}</span>
      {/if}
    </div>

    <div class="bar-cell">
      <Bar value={fraction} max={1} width={140} height={8} />
    </div>
    <span
      class="value"
      title="EV-weighted subspace fraction across {entry?.info.layers.length ?? 0} layers"
    >{fmtFraction(fraction)}</span>

    <Sparkline points={sparkline} width={48} height={14} cap={1} />

    <button
      type="button"
      class="icon remove"
      aria-label="Detach manifold probe {name}"
      title="Detach probe"
      onclick={onDetach}
    >✕</button>
  </div>

  <div class="readout">
    {#if top}
      <span class="nearest" title="distance in activation space">
        nearest <span class="nearest-label">{top[0]}</span>
        <span class="nearest-dist">d={fmtDistance(top[1])}</span>
      </span>
    {:else}
      <span class="nearest-empty">awaiting first token…</span>
    {/if}
    {#if aggregate && aggregate.coords.length > 0}
      <span class="coords" title="settled inverse-projection coords">
        coords ({aggregate.coords.map(fmtCoord).join(", ")})
      </span>
    {/if}
  </div>

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
  /* Outer frame matches ProbeStrip — same border + radius + bg — but
   * the name + accents are purple to mirror ManifoldStrip's manifold
   * family colour. */
  .strip {
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent-purple);
    border-radius: var(--radius);
    background: var(--bg-alt);
    font-size: var(--text-sm);
    transition: border-color var(--dur-fast) var(--ease-out);
  }

  .row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 32px;
    padding: var(--space-2) var(--space-3);
  }
  .name-cell {
    display: flex;
    flex-direction: column;
    min-width: 0;
    flex: 0 0 7em;
  }
  .name {
    color: var(--accent-purple);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .alias {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bar-cell {
    flex: 1 1 auto;
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
    min-width: 3.2em;
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

  .readout {
    display: flex;
    align-items: baseline;
    gap: var(--space-4);
    padding: 0 var(--space-3) var(--space-2) var(--space-3);
    color: var(--fg-dim);
    font-size: var(--text-xs);
    flex-wrap: wrap;
  }
  .nearest {
    color: var(--fg-dim);
  }
  .nearest-label {
    color: var(--fg-strong);
    font-weight: var(--weight-medium);
  }
  .nearest-dist {
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
    margin-left: var(--space-2);
  }
  .nearest-empty {
    color: var(--fg-muted);
    font-style: italic;
  }
  .coords {
    color: var(--accent-purple);
    font-variant-numeric: tabular-nums;
  }

  .map-wrap {
    border-top: 1px solid var(--border);
    padding: var(--space-3);
    display: flex;
    justify-content: center;
  }
</style>
