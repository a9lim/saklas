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
  import DetailSummary from "./DetailSummary.svelte";
  import DetailSection from "./DetailSection.svelte";

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

  const features = $derived(readout.data?.features ?? []);
  const labeledCount = $derived(features.filter((feature) => !!feature.label).length);
  const normalizedFeatures = $derived(
    features
      .map((feature) => ({ feature, strength: strengthOf(feature) }))
      .filter((entry): entry is { feature: SaeFeatureJSON; strength: number } =>
        entry.strength != null,
      ),
  );
  const strongestNormalized = $derived(
    normalizedFeatures.reduce<(typeof normalizedFeatures)[number] | null>(
      (best, entry) => !best || entry.strength > best.strength ? entry : best,
      null,
    ),
  );
  const peakRaw = $derived(
    features.reduce<SaeFeatureJSON | null>(
      (best, feature) => !best || feature.activation > best.activation ? feature : best,
      null,
    ),
  );
  const summaryMetrics = $derived([
    {
      label: "active features",
      value: String(features.length),
      detail: "top features at this position",
    },
    {
      label: "hook layer",
      value: readout.data?.layer != null && readout.data.layer >= 0
        ? `L${readout.data.layer}`
        : "—",
      detail: "resident SAE measurement point",
    },
    {
      label: "strongest normalized",
      value: strongestNormalized ? strongestNormalized.strength.toFixed(3) : "—",
      detail: strongestNormalized ? `sae/${strongestNormalized.feature.id}` : "metadata unavailable",
    },
    {
      label: "metadata coverage",
      value: features.length > 0 ? `${labeledCount}/${features.length}` : "—",
      detail: peakRaw ? `peak raw sae/${peakRaw.id} · ${peakRaw.activation.toFixed(1)}` : "no activations",
    },
  ]);
</script>

{#if readout.loading}
  <EmptyState title="computing…" />
{:else if readout.error}
  <EmptyState title={`readout: ${readout.error}`} />
{:else if readout.data}
  <DetailSummary
    accent="var(--pillar-sae)"
    eyebrow="sae"
    title="Sparse feature field"
    description="Which learned sparse features fired at the resident hook layer while the model produced this token."
    metrics={summaryMetrics}
    origin={readout.origin}
    source={readout.source}
    layer={readout.data.layer}
    steering={readout.data.steering}
    bind:steered
    {showToggle}
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
          <div role="listitem" title={rowTitle(feature)}>
            <RackCard accent="--pillar-sae" disabled={false}>
              {#snippet statline()}
                <span class="sae-rank">#{index + 1}</span>
                <code class="sae-id">sae/{feature.id}</code>
                <span class="sae-label" title={feature.label ?? undefined}>
                  {feature.label ?? "unlabeled feature"}
                </span>
                <span class="spacer"></span>
                <span class="layer">L{readout.data?.layer ?? "—"}</span>
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
                <div class="feature-meta">
                  <span>activation <b>{feature.activation.toFixed(3)}</b></span>
                  <span>maxActApprox <b>{feature.max_act != null ? feature.max_act.toFixed(3) : "—"}</b></span>
                  <span>unit <b>{strength != null ? "strength" : "raw"}</b></span>
                </div>
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
  .sae-rank {
    color: var(--pillar-sae);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .sae-id {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    white-space: nowrap;
  }
  .sae-label {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: block;
    min-width: 0;
  }
  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }
  .layer {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .row-label,
  .row-context {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    text-align: right;
  }
  .sae-value {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .feature-meta {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    flex-wrap: wrap;
    color: var(--fg-muted);
    font-size: var(--text-2xs);
  }
  .feature-meta b {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-weight: var(--weight-normal);
    font-variant-numeric: tabular-nums;
  }
  @media (max-width: 760px) {
    .sae-list {
      grid-template-columns: 1fr;
    }
  }
</style>
