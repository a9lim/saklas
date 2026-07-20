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
  import InstrumentHeader from "./InstrumentHeader.svelte";
  import DetailSection from "./DetailSection.svelte";
  import DetailCardHeader from "./DetailCardHeader.svelte";
  import EvidenceChips from "./EvidenceChips.svelte";

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

  function geometryEvidence(reading: ProbeReadingJSON) {
    return [
      ...(reading.nearest ?? []).map(([label, dist]) => ({
        label,
        value: `d=${dist.toFixed(2)}`,
        title: `whitened distance ${dist.toFixed(3)}`,
      })),
      ...(reading.assignment ?? []).map(([label, prob]) => ({
        label: `~${label}`,
        value: `${(prob * 100).toFixed(0)}%`,
        title: `soft-assignment posterior ${(prob * 100).toFixed(1)}%`,
        soft: true,
      })),
      ...(reading.residual !== 0
        ? [{
            label: "residual",
            value: reading.residual.toFixed(3),
            title: "normalized off-surface distance",
          }]
        : []),
      ...(reading.membership != null
        ? [{
            label: "membership",
            value: reading.membership.toFixed(3),
            title: "tube-fit density",
          }]
        : []),
    ];
  }
</script>

{#if readout.loading}
  <EmptyState title="computing…" />
{:else if readout.error}
  <EmptyState title={`readout: ${readout.error}`} />
{:else if readout.data && rows.length > 0}
  <InstrumentHeader
    origin={readout.origin}
    source={readout.source}
    steering={readout.data.steering}
    bind:steered
    {showToggle}
    accent="var(--pillar-subspace)"
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
        {@const evidence = geometryEvidence(reading)}
        <RackCard accent={cardAccent} disabled={false}>
          {#snippet statline()}
            <DetailCardHeader
              primary={name}
              primaryTitle={name}
              secondary={affine ? "subspace" : "manifold"}
              secondaryAccent
              meta={reading.depth_com?.[0] != null
                ? `@${reading.depth_com[0].toFixed(2)} ±${(reading.depth_spread?.[0] ?? 0).toFixed(2)}`
                : null}
              metaTitle="depth center of mass ± spread"
            >
              {#snippet lead()}
                <RackMarker shape={affine ? "circle" : "diamond"} filled />
              {/snippet}
            </DetailCardHeader>
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
                    {reading.nearest[0][0]}
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
            {#if cells.length > 0}
              <LayerStrip
                {cells}
                scale={stripScale(name, reading)}
                positiveColor={affine ? undefined : "var(--pillar-manifold)"}
                ariaLabel={`${name} per-layer readings`}
              />
            {/if}
            <EvidenceChips items={evidence} ariaLabel={`Geometry evidence for ${name}`} />
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
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
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
  @media (max-width: 820px) {
    .geo-list {
      grid-template-columns: 1fr;
    }
  }
</style>
