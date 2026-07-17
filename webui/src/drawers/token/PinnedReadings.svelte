<script lang="ts">
  // Pinned-probe readings section for the lens and sae tabs — the
  // ``instruments.<family>.readings`` block of the token's captured
  // measurements envelope, which the drilldown previously dropped on the
  // floor.  One row per pinned probe: name · strength bar (the family's
  // absolute 0..1 unit) · depth-CoM chip · value, with the per-layer
  // strength strip below when the probe carries a multi-layer trace.
  // Present only on live-captured tokens (replay endpoints return the
  // discovery readout, not the pinned roster).

  import Bar from "../../lib/charts/Bar.svelte";
  import LayerStrip from "../../panels/rack/LayerStrip.svelte";
  import ProbeReadingRow from "../../panels/rack/ProbeReadingRow.svelte";
  import RackCard from "../../panels/rack/RackCard.svelte";
  import RackMarker, {
    type RackMarkerShape,
  } from "../../panels/rack/RackMarker.svelte";
  import type { ProbeReadingJSON } from "../../lib/types";
  import DetailSection from "./DetailSection.svelte";

  let {
    readings,
    accent,
    shape,
  }: {
    readings: Record<string, ProbeReadingJSON>;
    /** Family CSS custom-property name, e.g. --pillar-lens. */
    accent: string;
    shape: RackMarkerShape;
  } = $props();

  const accentColor = $derived(`var(${accent})`);

  const rows = $derived(
    Object.entries(readings).sort(([a], [b]) =>
      a.localeCompare(b, undefined, { sensitivity: "base" }),
    ),
  );

  interface StripCell {
    layer: number;
    value: number | null;
    title: string;
  }

  /** Axis-0 per-layer trace (p_l for a lens probe, strength for sae). */
  function stripCells(reading: ProbeReadingJSON): StripCell[] {
    const perLayer = reading.coords_per_layer ?? {};
    return Object.keys(perLayer)
      .sort((a, b) => Number(a) - Number(b))
      .map((layer) => {
        const c = perLayer[layer];
        const v = Array.isArray(c) && c.length > 0 ? c[0] : null;
        return {
          layer: Number(layer),
          value: v,
          title:
            v == null ? `L${layer} · —` : `L${layer} · ${v.toPrecision(3)}`,
        };
      });
  }

  /** Cell color scale — the probe's own max (absolute p spans orders of
   *  magnitude; same convention as the rack's pinned lens cards). */
  function stripScale(cells: StripCell[]): number {
    return Math.max(...cells.map((c) => c.value ?? 0), 1e-12);
  }
</script>

{#if rows.length > 0}
  <DetailSection
    title="PINNED PROBES"
    count={`${rows.length} captured`}
    description="Persistent probe channels recorded on this exact token, including their depth profile."
    accent={accentColor}
  >
    <div class="pinned-grid" aria-label="Pinned probe readings">
      {#each rows as [name, reading] (name)}
        {@const cells = stripCells(reading)}
        <RackCard {accent} disabled={false}>
          {#snippet statline()}
            <RackMarker {shape} filled />
            <code class="pinned-name" title={`probe ${name}`}>{name}</code>
            {#if reading.depth_com && reading.depth_com[0] != null}
              <span
                class="pinned-com"
                title={`depth center of mass ±${(reading.depth_spread?.[0] ?? 0).toFixed(2)} (0 = first block, 1 = last)`}
              >
                @{reading.depth_com[0].toFixed(2)} ±{(reading.depth_spread?.[0] ?? 0).toFixed(2)}
              </span>
            {/if}
            <span class="spacer"></span>
            <span class="pinned-value">{(reading.coords[0] ?? 0).toFixed(3)}</span>
          {/snippet}
          {#snippet body()}
            <ProbeReadingRow ariaLabel={`Pinned probe ${name}`}>
              {#snippet left()}<span class="row-label">strength</span>{/snippet}
              {#snippet bar()}
                <Bar
                  value={Math.max(reading.coords[0] ?? 0, 0)}
                  max={1}
                  color={accentColor}
                />
              {/snippet}
              {#snippet middle()}<span aria-hidden="true"></span>{/snippet}
              {#snippet right()}
                <span class="pinned-value">{(reading.coords[0] ?? 0).toFixed(3)}</span>
              {/snippet}
            </ProbeReadingRow>
            {#if cells.length > 1}
              <LayerStrip
                {cells}
                scale={stripScale(cells)}
                positiveColor={accentColor}
                ariaLabel={`${name} per-layer strength`}
              />
            {/if}
          {/snippet}
        </RackCard>
      {/each}
    </div>
  </DetailSection>
{/if}

<style>
  .pinned-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .pinned-name {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .pinned-com {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
  }
  .pinned-value {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }
  .row-label {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    text-align: right;
  }
  @media (max-width: 760px) {
    .pinned-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
