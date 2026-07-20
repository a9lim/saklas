<script lang="ts">
  // One identity-row grammar for every token-drilldown evidence card.
  // The fixed leading slot and centrally owned type scale keep ranks and
  // family markers on the same baseline across logits, geometry, SAE, and
  // J-lens; callers provide only the card-specific words and values.

  import type { Snippet } from "svelte";

  let {
    lead,
    primary,
    primaryTitle,
    secondary = null,
    secondaryTitle,
    secondaryAccent = false,
    meta = null,
    metaTitle,
    badge = null,
    badgeTitle,
    tail = null,
    tailTitle,
    tailAccent = false,
  }: {
    lead: Snippet;
    primary: string;
    primaryTitle?: string;
    secondary?: string | null;
    secondaryTitle?: string;
    secondaryAccent?: boolean;
    meta?: string | null;
    metaTitle?: string;
    badge?: string | null;
    badgeTitle?: string;
    tail?: string | null;
    tailTitle?: string;
    tailAccent?: boolean;
  } = $props();
</script>

<div class="header">
  <span class="lead">{@render lead()}</span>
  <code class="primary" title={primaryTitle}>{primary}</code>
  {#if secondary}
    <span class="secondary" class:accent={secondaryAccent} title={secondaryTitle}>
      {secondary}
    </span>
  {/if}
  {#if meta}
    <span class="meta" title={metaTitle}>{meta}</span>
  {/if}
  {#if badge}
    <span class="badge" title={badgeTitle}>{badge}</span>
  {/if}
  <span class="spacer"></span>
  {#if tail}
    <span class="tail" class:accent={tailAccent} title={tailTitle}>{tail}</span>
  {/if}
</div>

<style>
  .header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    inline-size: 100%;
    min-width: 0;
    line-height: 1;
  }
  .lead {
    display: inline-flex;
    align-items: center;
    justify-content: flex-start;
    inline-size: 24px;
    flex: 0 0 24px;
    color: var(--card-accent);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
    line-height: 1;
    white-space: nowrap;
  }
  .primary {
    flex: 0 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--fg);
    background: transparent;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.2;
  }
  .secondary,
  .meta,
  .badge,
  .tail {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
    white-space: nowrap;
  }
  .secondary {
    flex: 0 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .meta,
  .badge,
  .tail {
    flex: 0 0 auto;
  }
  .accent {
    color: var(--card-accent);
  }
  .badge {
    color: var(--fg);
    background: color-mix(in srgb, var(--card-accent) 12%, transparent);
    border-radius: var(--radius-sm);
    padding: 1px var(--space-2);
  }
  .tail {
    font-family: var(--font-mono);
  }
  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }
</style>
