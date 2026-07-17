<script lang="ts">
  // Geometry tab — the full whitened Monitor reading for every attached
  // geometry probe at the forward that produced this token: all
  // coordinate axes, subspace fraction, nearest nodes, soft assignment,
  // tube membership, per-layer strip, depth CoM.  Captured envelopes
  // render directly; the replay endpoint covers aggregate-only
  // generations and probes attached after the fact.  Achromatic like the
  // rack's monitor cards — the family splits subspace-white /
  // manifold-violet, so no single pillar hue applies.

  import Bar from "../../lib/charts/Bar.svelte";
  import LayerStrip from "../../panels/rack/LayerStrip.svelte";
  import ProbeReadingRow from "../../panels/rack/ProbeReadingRow.svelte";
  import RackCard from "../../panels/rack/RackCard.svelte";
  import RackMarker from "../../panels/rack/RackMarker.svelte";
  import { probeRack, probeAxisScale } from "../../lib/stores.svelte";
  import type { ProbeReadingJSON } from "../../lib/types";
  import type { GeometryTokenReadout, ReplayReadout } from "./readout.svelte";
  import EmptyState from "./EmptyState.svelte";
  import DetailSummary from "./DetailSummary.svelte";
  import DetailSection from "./DetailSection.svelte";

  let {
    readout,
    steered = $bindable(),
    hasGeometryProbes,
    hasReplayContext,
  }: {
    readout: ReplayReadout<GeometryTokenReadout>;
    steered: boolean;
    /** ≥1 attached Monitor probe (the replay endpoint 400s on an empty
     *  roster, so absence renders the attach hint instead). */
    hasGeometryProbes: boolean;
    /** The token has a raw decode index + backing loom node. */
    hasReplayContext: boolean;
  } = $props();

  const rows = $derived<[string, ProbeReadingJSON][]>(
    Object.entries(readout.data?.readings ?? {}).sort(([a], [b]) =>
      a.localeCompare(b, undefined, { sensitivity: "base" }),
    ),
  );

  const showToggle = $derived(
    (readout.data?.steering ?? null) !== null || !steered,
  );

  function affineOf(name: string, reading: ProbeReadingJSON): boolean {
    return probeRack.entries.get(name)?.info?.is_affine ?? reading.residual === 0;
  }

  const affineCount = $derived(
    rows.filter(([name, reading]) => affineOf(name, reading)).length,
  );
  const curvedCount = $derived(rows.length - affineCount);
  const meanFraction = $derived(
    rows.length > 0
      ? rows.reduce((sum, [, reading]) => sum + reading.fraction, 0) / rows.length
      : 0,
  );

  const strongest = $derived.by<{
    name: string;
    axis: number;
    value: number;
  } | null>(() => {
    let best: { name: string; axis: number; value: number } | null = null;
    for (const [name, reading] of rows) {
      reading.coords.forEach((value, axis) => {
        if (!best || Math.abs(value) > Math.abs(best.value)) {
          best = { name, axis, value };
        }
      });
    }
    return best;
  });

  const closest = $derived.by<{
    probe: string;
    label: string;
    distance: number;
  } | null>(() => {
    let best: { probe: string; label: string; distance: number } | null = null;
    for (const [probe, reading] of rows) {
      for (const [label, distance] of reading.nearest ?? []) {
        if (!best || distance < best.distance) best = { probe, label, distance };
      }
    }
    return best;
  });

  const summaryMetrics = $derived([
    {
      label: "probes",
      value: String(rows.length),
      detail: `${affineCount} subspace · ${curvedCount} manifold`,
    },
    {
      label: "strongest coordinate",
      value: strongest
        ? `${strongest.value >= 0 ? "+" : ""}${strongest.value.toFixed(2)}`
        : "—",
      detail: strongest ? `${strongest.name} · c${strongest.axis}` : "no coordinate data",
    },
    {
      label: "mean subspace share",
      value: `${(meanFraction * 100).toFixed(1)}%`,
      detail: "across the attached roster",
    },
    {
      label: "nearest node",
      value: closest?.label ?? "—",
      detail: closest ? `${closest.probe} · d=${closest.distance.toFixed(2)}` : "no node distances",
    },
  ]);

  /** Axis label: the positive pole for a rank-1 two-node concept axis
   *  (coords axis 0 is pole-normalized, +1 at node 0), ``c<i>`` otherwise. */
  function axisLabel(name: string, axis: number, rank: number): string {
    const labels = probeRack.entries.get(name)?.info?.node_labels;
    if (rank === 1 && axis === 0 && labels && labels.length === 2) {
      return labels[0];
    }
    return `c${axis}`;
  }

  /** Per-layer strip source, mirroring the rack's primary-per-layer rule:
   *  axis-0 ``coords_per_layer`` for a flat probe, ``fraction_per_layer``
   *  for a curved one (no single signed coordinate to strip). */
  function stripCells(
    name: string,
    reading: ProbeReadingJSON,
  ): { layer: number; value: number | null; title: string }[] {
    const info = probeRack.entries.get(name)?.info;
    const curved = info ? !info.is_affine : reading.residual !== 0;
    const source: Record<string, number> = {};
    if (curved) {
      Object.assign(source, reading.fraction_per_layer ?? {});
    } else {
      for (const [layer, c] of Object.entries(reading.coords_per_layer ?? {})) {
        source[layer] = Array.isArray(c) && c.length > 0 ? c[0] : 0;
      }
    }
    return Object.keys(source)
      .sort((a, b) => Number(a) - Number(b))
      .map((layer) => {
        const v = source[layer];
        const sign = v >= 0 ? "+" : "";
        return {
          layer: Number(layer),
          value: v,
          title: `L${layer} · ${sign}${v.toFixed(3)}`,
        };
      });
  }

  function stripScale(name: string, reading: ProbeReadingJSON): number {
    const info = probeRack.entries.get(name)?.info;
    const curved = info ? !info.is_affine : reading.residual !== 0;
    if (curved) return 1; // fraction strip is already in [0, 1]
    return probeAxisScale(name, 0);
  }
</script>

{#if readout.loading}
  <EmptyState title="computing…" />
{:else if readout.error}
  <EmptyState title={`readout: ${readout.error}`} />
{:else if readout.data && rows.length > 0}
  <DetailSummary
    accent="var(--pillar-subspace)"
    eyebrow="geometry"
    title="Activation geometry"
    description="Where this token-producing residual sits inside every attached concept axis and fitted manifold."
    metrics={summaryMetrics}
    origin={readout.origin}
    source={readout.source}
    steering={readout.data.steering}
    bind:steered
    {showToggle}
  />
  <DetailSection
    title="PROBE READINGS"
    count={`${rows.length} attached`}
    description="Flat subspaces use a circle and white accent; curved manifolds use a diamond and violet accent."
  >
    <div class="geo-list">
      {#each rows as [name, reading] (name)}
        {@const rank = reading.coords.length}
        {@const cells = stripCells(name, reading)}
        {@const affine = affineOf(name, reading)}
        {@const cardAccent = affine ? "--pillar-subspace" : "--pillar-manifold"}
        <RackCard accent={cardAccent} disabled={false}>
          {#snippet statline()}
            <RackMarker shape={affine ? "circle" : "diamond"} filled />
            <code class="geo-name">{name}</code>
            <span class="family">{affine ? "subspace" : "manifold"}</span>
            {#if reading.depth_com?.[0] != null}
              <span class="depth" title="depth center of mass ± spread">
                @{reading.depth_com[0].toFixed(2)} ±{(reading.depth_spread?.[0] ?? 0).toFixed(2)}
              </span>
            {/if}
            <span class="spacer"></span>
            <span class="fraction-stat">{(reading.fraction * 100).toFixed(0)}% in-subspace</span>
          {/snippet}
          {#snippet body()}
            <ProbeReadingRow ariaLabel={`Subspace fraction ${reading.fraction.toFixed(3)}`}>
              {#snippet left()}<span class="geo-axis-label">subspace</span>{/snippet}
              {#snippet bar()}
                <Bar value={reading.fraction} max={1} color="var(--fg)" />
              {/snippet}
              {#snippet middle()}
                {#if (reading.nearest ?? []).length > 0}
                  <span class="nearest">
                    {reading.nearest[0][0]} · d={reading.nearest[0][1].toFixed(2)}
                  </span>
                {/if}
              {/snippet}
              {#snippet right()}<span class="geo-value">{reading.fraction.toFixed(3)}</span>{/snippet}
            </ProbeReadingRow>
            {#each reading.coords as coord, axis (axis)}
            <ProbeReadingRow ariaLabel={`${name} axis ${axis}`}>
              {#snippet left()}
                <span class="geo-axis-label" title={`coordinate axis ${axis}`}>
                  {axisLabel(name, axis, rank)}
                </span>
              {/snippet}
              {#snippet bar()}
                <Bar
                  value={coord}
                  max={probeAxisScale(name, axis)}
                  bipolar
                />
              {/snippet}
              {#snippet middle()}
                {#if reading.depth_com && reading.depth_com[axis] != null}
                  <span
                    class="geo-depth"
                    title={`depth center of mass ±${(reading.depth_spread?.[axis] ?? 0).toFixed(2)} (0 = first block, 1 = last)`}
                  >
                    @{reading.depth_com[axis].toFixed(2)}
                  </span>
                {/if}
              {/snippet}
              {#snippet right()}
                <span class="geo-value">{coord.toFixed(3)}</span>
              {/snippet}
            </ProbeReadingRow>
            {/each}
            <div class="meta">
              <span>fraction <b>{reading.fraction.toFixed(3)}</b></span>
              {#if reading.residual !== 0}
                <span title="normalized off-surface distance">residual <b>{reading.residual.toFixed(3)}</b></span>
              {/if}
              {#if reading.membership != null}
                <span title="tube-fit density">membership <b>{reading.membership.toFixed(3)}</b></span>
              {/if}
            </div>
            {#if cells.length > 0}
              <LayerStrip
                {cells}
                scale={stripScale(name, reading)}
                positiveColor={affine ? undefined : "var(--pillar-manifold)"}
                ariaLabel={`${name} per-layer readings`}
              />
            {/if}
            {#if (reading.nearest ?? []).length > 0 || (reading.assignment ?? []).length > 0}
              <div class="geo-chips">
                {#each reading.nearest ?? [] as [label, dist] (label)}
                  <span class="geo-chip" title={`whitened distance ${dist.toFixed(3)}`}>
                    {label} <span class="geo-chip-val">d={dist.toFixed(2)}</span>
                  </span>
                {/each}
                {#each reading.assignment ?? [] as [label, prob] (label)}
              <span
                class="geo-chip geo-chip-soft"
                title={`soft-assignment posterior ${(prob * 100).toFixed(1)}%`}
              >
                ~{label} <span class="geo-chip-val">{(prob * 100).toFixed(0)}%</span>
              </span>
                {/each}
              </div>
            {/if}
          {/snippet}
        </RackCard>
      {/each}
    </div>
  </DetailSection>
{:else if !hasGeometryProbes}
  <EmptyState
    title="no geometry probes attached"
    detail="attach a concept or manifold probe to read its whitened coordinates here"
  />
{:else if !hasReplayContext}
  <EmptyState
    title="no raw decode record"
    detail="replay needs a loom node generated with raw-decode capture in this session"
  />
{:else}
  <EmptyState title="no readings" />
{/if}

<style>
  .geo-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .geo-name {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--fg);
    word-break: break-all;
    min-width: 0;
  }
  .family,
  .depth,
  .fraction-stat {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .family {
    color: var(--card-accent);
  }
  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }
  .geo-axis-label {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .geo-depth {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
  }
  .geo-value {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .nearest {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .meta {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    flex-wrap: wrap;
    color: var(--fg-muted);
    font-size: var(--text-2xs);
  }
  .meta b {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-weight: var(--weight-normal);
    font-variant-numeric: tabular-nums;
  }
  .geo-chips {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  .geo-chip {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    background: var(--glass);
    border-radius: var(--radius-pill);
    padding: var(--space-1) var(--space-3);
    white-space: nowrap;
  }
  .geo-chip-soft {
    color: var(--fg-dim);
  }
  .geo-chip-val {
    color: var(--fg-dim);
    font-variant-numeric: tabular-nums;
  }
</style>
