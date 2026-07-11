<script lang="ts">
  // Unified probe card — one row for every probe shape.  Replaces the
  // SubspaceProbeCard / ManifoldProbeCard split: the store already
  // normalises ``current`` / ``sparkline`` / ``perLayer`` per family
  // (``_primaryScalar`` / ``_primaryPerLayer``), so this row reads those
  // uniformly and branches only on *presentation*.
  //
  //   statline : ●/◆ attached marker (click to detach) · name · @com
  //              ±spread · inspect · highlight · sparkline
  //   body     : the subspaceness row (white 0→1 bar segmented into
  //                intrinsic-dim notches · nearest node · fraction), then
  //                one signed bar per coordinate axis (poles on a rank-1
  //                concept axis, "c0…cR-1" labels otherwise), then the
  //                curved-only residual meta, the per-layer heatmap strip,
  //                and a 2-D box-domain mini-map.
  //
  // Family is signalled by accent (--accent flat / --accent-purple curved)
  // and glyph.  A dedicated action toggles the transcript highlight for
  // *every* family — the per-token score map is keyed by probe name
  // regardless of geometry, so curved probes are valid highlight targets
  // (the top-bar dropdown already lists them; this wires the click too).

  import type { ProbeRackEntry } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import Sparkline from "../../lib/charts/Sparkline.svelte";
  import { nodeCoordExtent, parseProbeTarget } from "../../lib/tokens";
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
  import ProbePinButton from "./ProbePinButton.svelte";
  import LayerStrip from "./LayerStrip.svelte";
  import ProbeReadingRow from "./ProbeReadingRow.svelte";

  interface Props {
    name: string;
    entry: ProbeRackEntry;
  }

  let { name, entry }: Props = $props();

  const info = $derived(entry.info);
  /** Flat (affine) ⇒ subspace family; curved ⇒ manifold family. */
  const affine = $derived(info.is_affine);

  const accent = $derived(affine ? "--accent" : "--accent-purple");

  /** Saturation scale for the per-layer heatmap strip + token tint — the
   *  axis-0 node extent for a flat probe ("full = as far along as the most
   *  extreme node"), the fixed unit scale for a curved probe's [0,1]
   *  fraction. */
  const axisScale = $derived(affine ? nodeCoordExtent(info.node_coords, 0) : 1);

  // ---------- latest reading: live during gen, settled (aggregate) after ----------
  const latest = $derived(entry.aggregate ?? entry.reading);
  /** Subspaceness — share of the centered activation living in this probe's
   *  subspace, [0,1].  Backs the white top-row bar. */
  const fraction = $derived(latest?.fraction ?? 0);
  /** Domain-frame coordinates, one per intrinsic dimension — each gets its
   *  own signed bar below the subspaceness row. */
  const coordVec = $derived(latest?.coords ?? []);
  /** Coordinate-bar count — the intrinsic dim, falling back to the live
   *  coord-vector length for a row reporting 0. */
  const nDim = $derived(
    info.intrinsic_dim > 0 ? info.intrinsic_dim : coordVec.length,
  );
  /** One {axis index, value, per-axis node extent} per coordinate bar. */
  const axes = $derived(
    Array.from({ length: nDim }, (_, i) => ({
      i,
      value: coordVec[i] ?? 0,
      scale: nodeCoordExtent(info.node_coords, i),
    })),
  );
  /** Pole labels (neg ◄─► pos) instead of a "c0" axis label — only a rank-1
   *  affine concept axis (2-node bipolar / 1-node monopolar) has poles;
   *  every higher-rank fan and curved fit labels its axes c0…cR-1. */
  const showPoles = $derived(affine && nDim === 1 && info.node_count <= 2);

  const sparkline = $derived(entry.sparkline ?? []);
  // Lit when this probe is the highlight target on any of its axes — a
  // per-PC target (``personas[3]``) still belongs to the ``personas`` card.
  const isHighlight = $derived(
    highlightState.target !== null &&
      parseProbeTarget(highlightState.target).base === name,
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
  const residual = $derived(aggregate?.residual ?? null);
  const trajectory = $derived(entry.trajectory ?? []);

  // ---------- depth stats (where in the layer stack the probe reads) ----------
  const depthCom = $derived(latest?.depth_com?.[0] ?? null);
  const depthSpread = $derived(latest?.depth_spread?.[0] ?? null);

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

  function cellTooltip(layer: string): string {
    const v = entry.perLayer?.[layer];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      return `L${layer} · —`;
    }
    // Flat per-layer is signed (axis-0); curved is a [0,1] fraction.
    const sign = affine && v >= 0 ? "+" : "";
    return `L${layer} · ${sign}${v.toFixed(3)}`;
  }

  const layerCells = $derived(
    layerKeys.map((layer) => ({
      layer: Number(layer),
      value: entry.perLayer?.[layer],
      title: cellTooltip(layer),
    })),
  );

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

  // The leading marker now means persistence in every probe family. CAA's
  // transcript-highlight selection is an explicit sibling action instead
  // of overloading the same glyph used for J-LENS/SAE pinning.
  function toggleHighlight(): void {
    setHighlightTarget(isHighlight ? null : name);
  }

  // Open the per-probe inspector (whitened geometry plot + layer norms + live
  // trail).  Distinct gesture from the identity-cluster highlight toggle, so
  // ``stopPropagation`` keeps the two from colliding.
  function onInspect(): void {
    openDrawer("probe_inspector", { name });
  }

  async function onDetach(): Promise<void> {
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

<RackCard {accent} disabled={false} active={isHighlight}>
  {#snippet statline()}
    <ProbePinButton
      shape={affine ? "circle" : "diamond"}
      pinned={true}
      onclick={() => void onDetach()}
      title="Attached (click to detach)"
      ariaLabel={`Detach probe ${name}`}
    />

    <span class="name" title="probe {name}">{displayName}</span>

    {#if depthCom !== null}
      <span
        class="com"
        title="depth center of mass of the per-layer read, share-weighted ±spread (0 = first block, 1 = last)"
      >@{fmtCoord(depthCom)}{depthSpread !== null ? ` ±${fmtCoord(depthSpread)}` : ""}</span>
    {/if}

    <span class="spacer"></span>

    <button
      type="button"
      class="icon inspect"
      aria-label="Inspect probe {name}"
      title="Open probe inspector"
      onclick={onInspect}
    >ⓘ</button>

    <button
      type="button"
      class="highlight-action"
      class:on={isHighlight}
      aria-pressed={isHighlight}
      aria-label={isHighlight
        ? `Deselect ${name} as transcript highlight target`
        : `Select ${name} as transcript highlight target`}
      title="Use this probe to color transcript tokens"
      onclick={toggleHighlight}
    >{isHighlight ? "highlighted" : "highlight"}</button>

    <!-- Curved (fraction) probes are [0,1]-bounded → cap the sparkline at 1;
         flat (signed) probes auto-scale.  Trace wears the pillar hue. -->
    <Sparkline
      points={sparkline}
      width={56}
      height={14}
      cap={affine ? undefined : 1}
      color="var(--card-accent)"
    />

  {/snippet}

  {#snippet body()}
    <!-- Subspaceness row: white 0→1 bar · nearest node · fraction.  "How
         much of the centered activation lives in this probe's subspace" —
         the scale runs higher for higher-rank fits, which is expected. -->
    <ProbeReadingRow
      ariaLabel={`Subspace fraction ${fmtFraction(fraction)}${topNearest ? `, nearest ${nearestLabel}` : ""}`}
    >
      {#snippet left()}
        <span
          class="row-label"
          title="subspaceness — share of the centered activation living in this probe's subspace (0–1)"
        >subspace</span>
      {/snippet}
      {#snippet bar()}
        <Bar value={fraction} max={1} width={160} height={8} color="var(--fg)" />
      {/snippet}
      {#snippet middle()}
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
      {/snippet}
      {#snippet right()}<span class="value">{fmtFraction(fraction)}</span>{/snippet}
    </ProbeReadingRow>

    <!-- One signed bar per coordinate axis.  A rank-1 concept axis carries
         pole labels (neg ◄─► pos); higher-rank fans and curved fits label
         axes c0…cR-1.  Each bar normalizes by its own node extent. -->
    {#each axes as ax (ax.i)}
      <ProbeReadingRow ariaLabel={`Coordinate ${ax.i}, ${ax.value.toFixed(2)}`}>
        {#snippet left()}
          {#if showPoles}
            <span class="pole neg" aria-hidden={monopolar}>
              {#if !monopolar}{poles.negative}{/if}
            </span>
          {:else}
            <span class="row-label axis">c{ax.i}</span>
          {/if}
        {/snippet}
        {#snippet bar()}
          <Bar value={ax.value} max={ax.scale} width={160} height={8} bipolar />
        {/snippet}
        {#snippet middle()}
          {#if showPoles}
            <span class="pole pos" title={`positive pole (${poles.positive})`}>
              {poles.positive}
            </span>
          {:else}
            <span class="pole pos" aria-hidden="true"></span>
          {/if}
        {/snippet}
        {#snippet right()}
          <span class="value" class:pos={ax.value > 0} class:neg={ax.value < 0}>
            {ax.value >= 0 ? "+" : ""}{ax.value.toFixed(2)}
          </span>
        {/snippet}
      </ProbeReadingRow>
    {/each}

    {#if !affine && residual !== null}
      <!-- Settled meta: the curved-only off-surface residual (the depth
           CoM moved to the statline, right of the probe name). -->
      <div class="meta">
        <span class="meta-item" title="normalized off-surface residual">
          residual {fmtCoord(residual)}
        </span>
      </div>
    {/if}

    <!-- Per-layer heatmap strip with L0 / Ln endcaps. -->
    <LayerStrip
      cells={layerCells}
      scale={axisScale}
      ariaLabel={`Per-layer readings for ${name}`}
    />

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
  .name {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .com {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }
  .icon {
    min-width: 24px;
    min-height: 24px;
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
  .inspect:hover:not(:disabled) {
    color: var(--accent-purple);
  }
  .highlight-action {
    min-height: 24px;
    padding: var(--space-1) var(--space-3);
    color: var(--fg-muted);
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: var(--text-xs);
    flex: 0 0 auto;
  }
  .highlight-action:hover,
  .highlight-action.on {
    color: var(--card-accent);
    background: color-mix(in srgb, var(--card-accent) 10%, var(--glass));
  }

  /* ----- body: ProbeReadingRow content -----
     Left-column axis caption — the subspaceness "subspace" tag and the
     "c0…cR-1" coordinate labels.  Hugs the bar (right-aligned) like the
     neg pole it shares the slot with. */
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
  .row-label.axis {
    color: var(--fg-dim);
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
    line-height: 13px;
  }
  .nearest-dist {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    line-height: 11px;
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

  /* ----- body: mini-map ----- */
  .map-wrap {
    display: flex;
    justify-content: center;
  }
</style>
