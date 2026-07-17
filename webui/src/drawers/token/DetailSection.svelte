<script lang="ts">
  // Canonical detailed-evidence section. It deliberately mirrors the main
  // rack's STEER / PROBE section headers: accent label, count, one sentence
  // of orientation, then cards or a dense matrix.

  import type { Snippet } from "svelte";

  let {
    title,
    count = null,
    description = null,
    accent = "var(--accent)",
    children,
  }: {
    title: string;
    count?: string | null;
    description?: string | null;
    accent?: string;
    children: Snippet;
  } = $props();
</script>

<section class="section" style:--section-accent={accent}>
  <header>
    <div class="title-row">
      <h3>{title}</h3>
      {#if count}<span class="count">{count}</span>{/if}
    </div>
    {#if description}<p>{description}</p>{/if}
  </header>
  <div class="content">{@render children()}</div>
</section>

<style>
  .section {
    --section-accent: var(--accent);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-width: 0;
  }
  header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: var(--space-6);
    min-width: 0;
  }
  .title-row {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    flex: 0 0 auto;
  }
  h3 {
    margin: 0;
    color: var(--section-accent);
    font-size: var(--text-sm);
    font-weight: var(--weight-bold);
    text-transform: uppercase;
    letter-spacing: 0;
  }
  .count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
  }
  p {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    line-height: 1.4;
    text-align: right;
    max-width: 66ch;
  }
  .content {
    min-width: 0;
  }
  @media (max-width: 760px) {
    header {
      align-items: flex-start;
      flex-direction: column;
      gap: var(--space-2);
    }
    p {
      text-align: left;
    }
  }
</style>
