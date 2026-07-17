<script lang="ts">
  // Shared first read for every token-detail tab. The main instrument
  // panels establish the vocabulary: family accent, one strong heading,
  // a small set of comparable meters, then source/recipe context. Keeping
  // that order here prevents the four detailed views from becoming four
  // unrelated mini-apps.

  import type { ReadoutOrigin } from "./readout.svelte";
  import InstrumentHeader from "./InstrumentHeader.svelte";

  interface DetailMetric {
    label: string;
    value: string;
    detail?: string | null;
    title?: string;
  }

  let {
    accent,
    eyebrow,
    title,
    description,
    metrics,
    origin,
    source = null,
    layer = null,
    steering = null,
    steered = $bindable(),
    showToggle = false,
  }: {
    accent: string;
    eyebrow: string;
    title: string;
    description: string;
    metrics: DetailMetric[];
    origin: ReadoutOrigin;
    source?: string | null;
    layer?: number | null;
    steering?: string | null;
    steered: boolean;
    showToggle?: boolean;
  } = $props();
</script>

<section class="summary" style:--summary-accent={accent}>
  <div class="summary-copy">
    <span class="eyebrow">{eyebrow}</span>
    <h2>{title}</h2>
    <p>{description}</p>
  </div>

  <dl class="metrics">
    {#each metrics as metric (metric.label)}
      <div class="metric" title={metric.title}>
        <dt>{metric.label}</dt>
        <dd>{metric.value}</dd>
        {#if metric.detail}<span>{metric.detail}</span>{/if}
      </div>
    {/each}
  </dl>

  <InstrumentHeader
    {origin}
    {source}
    {layer}
    {steering}
    bind:steered
    {showToggle}
    {accent}
  />
</section>

<style>
  .summary {
    --summary-accent: var(--accent);
    display: grid;
    grid-template-columns: minmax(220px, 0.85fr) minmax(420px, 1.8fr);
    gap: var(--space-6);
    padding: var(--space-6);
    border-radius: var(--radius-lg);
    background: var(--glass);
    box-shadow: var(--shadow-rack);
    min-width: 0;
  }
  .summary-copy {
    min-width: 0;
  }
  .eyebrow {
    color: var(--summary-accent);
    font-size: var(--text-xs);
    font-weight: var(--weight-bold);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  h2 {
    margin: var(--space-2) 0 0;
    color: var(--fg);
    font-size: var(--text-lg);
    font-weight: var(--weight-medium);
    letter-spacing: -0.01em;
  }
  p {
    margin: var(--space-3) 0 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.5;
    max-width: 48ch;
  }
  .metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--space-2);
    margin: 0;
    min-width: 0;
  }
  .metric {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
    padding: var(--space-4);
    border-radius: var(--radius);
    background: var(--input-well);
  }
  dt {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }
  dd {
    margin: 0;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-md);
    font-variant-numeric: tabular-nums;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .metric > span {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    line-height: 1.35;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    line-clamp: 2;
  }
  .summary :global(.inst-head) {
    grid-column: 1 / -1;
  }

  @media (max-width: 820px) {
    .summary {
      grid-template-columns: 1fr;
    }
    .metrics {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
