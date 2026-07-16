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
  import { probeRack, probeAxisScale } from "../../lib/stores.svelte";
  import type { ProbeReadingJSON } from "../../lib/types";
  import type { GeometryTokenReadout, ReplayReadout } from "./readout.svelte";
  import InstrumentHeader from "./InstrumentHeader.svelte";
  import EmptyState from "./EmptyState.svelte";

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
  <InstrumentHeader
    origin={readout.origin}
    source={readout.source}
    steering={readout.data.steering}
    bind:steered
    {showToggle}
  />
  <div class="geo-list">
    {#each rows as [name, reading] (name)}
      {@const rank = reading.coords.length}
      {@const cells = stripCells(name, reading)}
      <section class="geo-probe" aria-label={`Probe ${name}`}>
        <div class="geo-head">
          <code class="geo-name">{name}</code>
          <span class="kv" title="in-subspace share of the centered activation">
            fraction {(reading.fraction * 100).toFixed(0)}%
          </span>
          {#if reading.residual !== 0}
            <span class="kv" title="normalized off-surface distance (curved fit)">
              residual {reading.residual.toFixed(3)}
            </span>
          {/if}
          {#if reading.membership != null && reading.membership < 1}
            <span class="kv" title="tube-fit density under the fitted within-node thickness">
              membership {reading.membership.toFixed(2)}
            </span>
          {/if}
        </div>
        <div class="geo-axes">
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
        </div>
        {#if cells.length > 0}
          <LayerStrip
            {cells}
            scale={stripScale(name, reading)}
            ariaLabel={`${name} per-layer readings`}
          />
        {/if}
        {#if (reading.nearest ?? []).length > 0 || (reading.assignment ?? []).length > 0}
          <div class="geo-chips">
            {#each reading.nearest ?? [] as [label, dist] (label)}
              <span class="geo-chip" title={`whitened distance ${dist.toFixed(3)}`}>
                {label} <span class="geo-chip-val">{dist.toFixed(2)}</span>
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
      </section>
    {/each}
  </div>
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
    gap: var(--space-5);
  }
  .geo-probe {
    background: var(--bg);
    border-radius: var(--radius);
    padding: var(--space-4) var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .geo-head {
    display: flex;
    align-items: baseline;
    gap: var(--space-4);
    flex-wrap: wrap;
    min-width: 0;
  }
  .geo-head .kv {
    color: var(--fg-dim);
    font-size: var(--text-sm);
  }
  .geo-name {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--fg);
    word-break: break-all;
  }
  .geo-axes {
    display: flex;
    flex-direction: column;
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
