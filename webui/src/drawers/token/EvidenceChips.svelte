<script lang="ts">
  // Compact supporting facts shared by token-drilldown cards. These are
  // evidence below the main meter, not a second headline, so the treatment
  // stays quiet and identical for geometry distances and SAE metadata.

  interface EvidenceChip {
    label: string;
    value?: string | null;
    title?: string;
    soft?: boolean;
  }

  let {
    items,
    ariaLabel = "Supporting evidence",
  }: {
    items: EvidenceChip[];
    ariaLabel?: string;
  } = $props();
</script>

{#if items.length > 0}
  <div class="chips" role="list" aria-label={ariaLabel}>
    {#each items as item}
      <span class="chip" class:soft={item.soft} title={item.title} role="listitem">
        {item.label}
        {#if item.value}<span class="value">{item.value}</span>{/if}
      </span>
    {/each}
  </div>
{/if}

<style>
  .chips {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  .chip {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    background: var(--glass);
    border-radius: var(--radius-pill);
    padding: var(--space-1) var(--space-3);
    white-space: nowrap;
  }
  .soft,
  .value {
    color: var(--fg-dim);
  }
  .value {
    font-variant-numeric: tabular-nums;
  }
</style>
