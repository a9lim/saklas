<script lang="ts">
  // Pinned-probe readings section for the lens and sae tabs — the
  // ``instruments.<family>.readings`` block of the token's captured
  // measurements envelope.  These are the single-axis families, so the
  // block carries their NATIVE one-channel reading — value + its explicit
  // unit + the per-layer trace + a depth summary that names its own mass
  // basis.  One row per pinned probe: name · strength bar · depth-CoM chip
  // · value, with the per-layer strip below when the trace spans layers.
  // Present only on live-captured tokens (replay endpoints return the
  // discovery readout, not the pinned roster).

  import Bar from "../../lib/charts/Bar.svelte";
  import LayerStrip from "../../panels/rack/LayerStrip.svelte";
  import ProbeReadingRow from "../../panels/rack/ProbeReadingRow.svelte";
  import RackCard from "../../panels/rack/RackCard.svelte";
  import RackMarker, {
    type RackMarkerShape,
  } from "../../panels/rack/RackMarker.svelte";
  import type { ScalarReadingJSON } from "../../lib/types";
  import DetailSection from "./DetailSection.svelte";
  import DetailCardHeader from "./DetailCardHeader.svelte";

  let {
    readings,
    accent,
    shape,
  }: {
    readings: Record<string, ScalarReadingJSON>;
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

  /** The channel's per-layer trace (p_l for a lens probe, activation for
   *  sae) — plain numbers, no coordinate-vector unwrap. */
  function stripCells(reading: ScalarReadingJSON): StripCell[] {
    const perLayer = reading.per_layer ?? {};
    return Object.keys(perLayer)
      .sort((a, b) => Number(a) - Number(b))
      .map((layer) => ({
        layer: Number(layer),
        value: perLayer[layer],
        title: `L${layer} · ${perLayer[layer].toPrecision(3)}`,
      }));
  }

  /** The reading says what its number means, so the row can too. */
  const UNIT_LABEL: Record<ScalarReadingJSON["unit"], string> = {
    mean_token_probability: "mean fitted-layer probability",
    activation_over_max: "activation / corpus max",
    raw_activation: "raw activation (no corpus max cached)",
  };

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
            <DetailCardHeader
              primary={name}
              primaryTitle={`probe ${name}`}
              meta={reading.depth?.center?.[0] != null
                ? `@${reading.depth.center[0].toFixed(2)} ±${(reading.depth.spread?.[0] ?? 0).toFixed(2)}`
                : null}
              metaTitle="depth center of mass ± spread (0 = first block, 1 = last)"
            >
              {#snippet lead()}<RackMarker {shape} filled />{/snippet}
            </DetailCardHeader>
          {/snippet}
          {#snippet body()}
            <ProbeReadingRow ariaLabel={`Pinned probe ${name}`}>
              {#snippet left()}
                <span class="row-label" title={UNIT_LABEL[reading.unit]}>
                  strength
                </span>
              {/snippet}
              {#snippet bar()}
                <Bar
                  value={Math.max(reading.value, 0)}
                  max={1}
                  color={accentColor}
                />
              {/snippet}
              {#snippet middle()}<span aria-hidden="true"></span>{/snippet}
              {#snippet right()}
                <span class="pinned-value">{reading.value.toFixed(3)}</span>
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
  .pinned-value {
    color: var(--card-accent);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
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
