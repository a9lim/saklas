<script lang="ts">
  // SAE tab — pinned ``sae/<id>`` probe readings (when captured live)
  // above the resident SAE's top feature activations at this decode
  // step.  Feature rows speak the panel's meter grammar: id · gold
  // strength bar · label · value.  Strength is activation /
  // Neuronpedia maxActApprox (the absolute 0..1 unit gates read);
  // metadata-less features fall back to a shared raw scale local to this
  // readout so bars still rank with the numbers.

  import Bar from "../../lib/charts/Bar.svelte";
  import ProbeReadingRow from "../../panels/rack/ProbeReadingRow.svelte";
  import RackCard from "../../panels/rack/RackCard.svelte";
  import type {
    ProbeReadingJSON,
    SaeFeatureJSON,
    SaeTokenReadoutJSON,
  } from "../../lib/types";
  import type { ReplayReadout } from "./readout.svelte";
  import EmptyState from "./EmptyState.svelte";
  import PinnedReadings from "./PinnedReadings.svelte";
  import InstrumentHeader from "./InstrumentHeader.svelte";
  import DetailSection from "./DetailSection.svelte";
  import DetailCardHeader from "./DetailCardHeader.svelte";
  import EvidenceChips from "./EvidenceChips.svelte";

  let {
    readout,
    steered = $bindable(),
    saeLoaded,
    hasReplayContext,
    pinned,
  }: {
    readout: ReplayReadout<SaeTokenReadoutJSON>;
    steered: boolean;
    saeLoaded: boolean;
    hasReplayContext: boolean;
    /** Live-captured pinned-probe readings from the token's envelope. */
    pinned: Record<string, ProbeReadingJSON> | null;
  } = $props();

  const showToggle = $derived(
    (readout.data?.steering ?? null) !== null || !steered,
  );

  /** Shared raw unit for metadata-less features — the largest raw
   *  activation in THIS readout (historical view, so the panel-wide live
   *  scale doesn't apply). */
  const rawScale = $derived(
    Math.max(
      ...(readout.data?.features ?? [])
        .filter((f) => !(f.max_act != null && f.max_act > 0))
        .map((f) => f.activation),
      1,
    ),
  );

  function strengthOf(f: SaeFeatureJSON): number | null {
    return f.max_act != null && f.max_act > 0 ? f.activation / f.max_act : null;
  }

  function rowTitle(f: SaeFeatureJSON): string {
    const s = strengthOf(f);
    const parts = [
      `sae/${f.id}`,
      s != null ? `strength ${s.toFixed(2)}` : "no Neuronpedia metadata (raw scale)",
      `activation ${f.activation.toFixed(3)}`,
    ];
    return parts.join(" · ");
  }

  function featureEvidence(f: SaeFeatureJSON) {
    if (!(f.max_act != null && f.max_act > 0)) return [];
    return [
      {
        label: "activation",
        value: f.activation.toFixed(3),
        title: `raw feature activation ${f.activation.toFixed(3)}`,
      },
      {
        label: "maxActApprox",
        value: f.max_act.toFixed(3),
        title: `Neuronpedia maxActApprox ${f.max_act.toFixed(3)}`,
      },
    ];
  }

</script>

{#if readout.loading}
  <EmptyState title="computing…" />
{:else if readout.error}
  <EmptyState title={`readout: ${readout.error}`} />
{:else if readout.data}
  <InstrumentHeader
    origin={readout.origin}
    source={readout.source}
    layer={readout.data.layer}
    steering={readout.data.steering}
    bind:steered
    {showToggle}
    accent="var(--pillar-sae)"
  />
  {#if pinned && Object.keys(pinned).length > 0}
    <PinnedReadings readings={pinned} accent="--pillar-sae" shape="triangle" />
  {/if}
  {#if readout.data.features.length === 0}
    <EmptyState title="no features fired at this position" />
  {:else}
    <DetailSection
      title="FEATURE ACTIVATIONS"
      count={`${readout.data.features.length} retained`}
      description="Gold bars show normalized strength when maxActApprox metadata exists; otherwise cards use one shared raw-activation scale."
      accent="var(--pillar-sae)"
    >
      <div class="sae-list" role="list" aria-label="Top SAE features">
        {#each readout.data.features as feature, index (feature.id)}
          {@const strength = strengthOf(feature)}
          {@const evidence = featureEvidence(feature)}
          <div role="listitem" title={rowTitle(feature)}>
            <RackCard accent="--pillar-sae" disabled={false}>
              {#snippet statline()}
                <DetailCardHeader
                  primary={`sae/${feature.id}`}
                  secondary={feature.label ?? "unlabeled feature"}
                  secondaryTitle={feature.label ?? "unlabeled feature"}
                  tail={`L${readout.data?.layer ?? "—"}`}
                  tailTitle="resident SAE layer"
                >
                  {#snippet lead()}<span>#{index + 1}</span>{/snippet}
                </DetailCardHeader>
              {/snippet}
              {#snippet body()}
                <ProbeReadingRow ariaLabel={`Feature sae/${feature.id}`}>
                  {#snippet left()}
                    <span class="row-label">{strength != null ? "strength" : "activation"}</span>
                  {/snippet}
                  {#snippet bar()}
                    {#if strength != null}
                  <Bar
                    value={Math.max(strength, 0)}
                    max={1}
                    color="var(--pillar-sae)"
                  />
                {:else}
                  <Bar
                    value={Math.max(feature.activation, 0)}
                    max={rawScale}
                    color="color-mix(in srgb, var(--pillar-sae) 55%, transparent)"
                  />
                    {/if}
                  {/snippet}
                  {#snippet middle()}
                    <span class="row-context">{strength != null ? "normalized" : "shared raw scale"}</span>
                  {/snippet}
                  {#snippet right()}
                    <span class="sae-value">
                      {strength != null ? strength.toFixed(3) : feature.activation.toFixed(2)}
                    </span>
                  {/snippet}
                </ProbeReadingRow>
                <EvidenceChips
                  items={evidence}
                  ariaLabel={`Metadata for feature sae/${feature.id}`}
                />
              {/snippet}
            </RackCard>
          </div>
        {/each}
      </div>
    </DetailSection>
  {/if}
{:else if !saeLoaded}
  <EmptyState
    title="no SAE loaded"
    detail="load a source from the SAE tab's SOURCE section to read sparse features here"
  />
{:else if !hasReplayContext}
  <EmptyState
    title="no raw decode record"
    detail="replay needs a loom node generated with raw-decode capture in this session"
  />
{:else}
  <EmptyState title="no readout" />
{/if}

<style>
  .sae-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .row-label,
  .row-context {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    text-align: right;
  }
  .sae-value {
    color: var(--pillar-sae);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  @media (max-width: 760px) {
    .sae-list {
      grid-template-columns: 1fr;
    }
  }
</style>
