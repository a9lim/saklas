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
  import type { ProbeReadingJSON } from "../../lib/types";

  let {
    readings,
    accent,
  }: {
    readings: Record<string, ProbeReadingJSON>;
    /** Family hue (CSS color) for bars + strip cells. */
    accent: string;
  } = $props();

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
  <section class="pinned" aria-label="Pinned probe readings">
    <span class="pinned-label">pinned probes</span>
    {#each rows as [name, reading] (name)}
      {@const cells = stripCells(reading)}
      <div class="pinned-probe">
        <ProbeReadingRow ariaLabel={`Pinned probe ${name}`}>
          {#snippet left()}
            <code class="pinned-name" title={`probe ${name}`}>{name}</code>
          {/snippet}
          {#snippet bar()}
            <Bar
              value={Math.max(reading.coords[0] ?? 0, 0)}
              max={1}
              color={accent}
            />
          {/snippet}
          {#snippet middle()}
            {#if reading.depth_com && reading.depth_com[0] != null}
              <span
                class="pinned-com"
                title={`depth center of mass ±${(reading.depth_spread?.[0] ?? 0).toFixed(2)} (0 = first block, 1 = last)`}
              >
                @{reading.depth_com[0].toFixed(2)}
              </span>
            {/if}
          {/snippet}
          {#snippet right()}
            <span class="pinned-value">{(reading.coords[0] ?? 0).toFixed(3)}</span>
          {/snippet}
        </ProbeReadingRow>
        {#if cells.length > 1}
          <LayerStrip
            {cells}
            scale={stripScale(cells)}
            positiveColor={accent}
            ariaLabel={`${name} per-layer strength`}
          />
        {/if}
      </div>
    {/each}
  </section>
{/if}

<style>
  .pinned {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    background: var(--bg);
    border-radius: var(--radius);
    padding: var(--space-4) var(--space-5);
    margin-bottom: var(--space-4);
  }
  .pinned-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .pinned-probe {
    min-width: 0;
  }
  .pinned-name {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
</style>
