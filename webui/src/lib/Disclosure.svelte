<script lang="ts">
  // Themed collapsible — the existing ``▸/▾ caret + button row`` pattern
  // packaged as one component.  Backs the ad-hoc ``advancedOpen`` toggles
  // (e.g. ManifoldBuilderDrawer) and inline grammar-reference disclosures.
  //
  // Default slot is the body; the trigger row shows the caret + summary.

  import type { Snippet } from "svelte";

  interface Props {
    /** Bindable open state. */
    expanded: boolean;
    /** Trigger label. */
    summary: string;
    /** Optional dense styling — sits flush in tight grids without the
     *  outer border / padding. */
    flush?: boolean;
    children: Snippet;
  }

  let {
    expanded = $bindable(),
    summary,
    flush = false,
    children,
  }: Props = $props();

  function toggle(): void {
    expanded = !expanded;
  }
</script>

<section class="sk-disclosure" class:is-open={expanded} class:is-flush={flush}>
  <button
    type="button"
    class="sk-disclosure-trigger"
    aria-expanded={expanded}
    onclick={toggle}
  >
    <span class="sk-disclosure-caret" aria-hidden="true">
      {expanded ? "▾" : "▸"}
    </span>
    <span class="sk-disclosure-summary">{summary}</span>
  </button>
  {#if expanded}
    <div class="sk-disclosure-body">
      {@render children()}
    </div>
  {/if}
</section>

<style>
  .sk-disclosure {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    overflow: hidden;
  }
  .sk-disclosure.is-flush {
    border: none;
    background: transparent;
    border-radius: 0;
  }

  .sk-disclosure-trigger {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    width: 100%;
    padding: var(--space-2) var(--space-3);
    background: transparent;
    color: var(--fg-strong);
    border: 0;
    border-bottom: 1px solid transparent;
    text-align: left;
    cursor: pointer;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    transition: background var(--dur-fast) var(--ease-out);
  }
  .sk-disclosure.is-flush .sk-disclosure-trigger {
    padding: var(--space-2) 0;
  }
  .sk-disclosure-trigger:hover {
    background: var(--bg-hover);
  }
  .sk-disclosure-trigger:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: -2px;
  }
  .sk-disclosure.is-open .sk-disclosure-trigger {
    border-bottom-color: var(--border);
  }
  .sk-disclosure.is-flush.is-open .sk-disclosure-trigger {
    border-bottom-color: transparent;
  }

  .sk-disclosure-caret {
    flex: 0 0 auto;
    color: var(--fg-muted);
    font-size: var(--text-sm);
    line-height: 1;
  }
  .sk-disclosure.is-open .sk-disclosure-caret {
    color: var(--accent);
  }

  .sk-disclosure-summary {
    flex: 1 1 0;
    color: var(--fg);
  }

  .sk-disclosure-body {
    padding: var(--space-4) var(--space-4);
    background: transparent;
  }
  .sk-disclosure.is-flush .sk-disclosure-body {
    padding: var(--space-3) 0;
  }
</style>
