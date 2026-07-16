<script lang="ts">
  // The one provenance row every replay-capable drilldown tab renders —
  // identical order everywhere so the tabs can't drift: origin chip
  // (captured / replayed · source), optional resident-layer chip, the
  // steering-expression chip (or "unsteered"), and the apply-recipe
  // toggle when the view has a counterfactual to flip to.

  import type { ReadoutOrigin } from "./readout.svelte";

  let {
    origin,
    source = null,
    layer = null,
    steering,
    steered = $bindable(),
    showToggle,
  }: {
    origin: ReadoutOrigin;
    source?: string | null;
    /** sae only — the resident hook layer. */
    layer?: number | null;
    /** The steering the shown read ran under (null = unsteered). */
    steering: string | null;
    steered: boolean;
    /** Offer the steered/unsteered flip — the node has recipe steering,
     *  or the user already flipped off and needs the way back. */
    showToggle: boolean;
  } = $props();
</script>

<div class="inst-head">
  {#if origin}
    <span class="kv origin" title="original capture vs on-demand replay">
      {origin}{source ? ` · ${source}` : ""}
    </span>
  {/if}
  {#if layer != null && layer >= 0}
    <span class="kv" title="resident hook layer">L{layer}</span>
  {/if}
  {#if steering !== null}
    <span class="kv steer-chip" title="steering applied at this read">
      steered: <code>{steering}</code>
    </span>
  {:else if !steered}
    <span class="kv" title="unsteered counterfactual read">unsteered</span>
  {/if}
  {#if showToggle}
    <label class="kv steer-toggle" title="replay under the node's recipe steering">
      <input type="checkbox" bind:checked={steered} />
      apply recipe steering
    </label>
  {/if}
</div>

<style>
  .inst-head {
    display: flex;
    align-items: baseline;
    gap: var(--space-4);
    flex-wrap: wrap;
    padding: 0 0 var(--space-4) 0;
    font-size: var(--text-sm);
    line-height: 1.6;
  }
  .kv {
    color: var(--fg-dim);
  }
  .origin {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .steer-chip code {
    color: var(--fg-strong);
    background: transparent;
    font-family: var(--font-mono);
  }
  .steer-toggle {
    cursor: pointer;
    user-select: none;
  }
  .steer-toggle input {
    accent-color: var(--accent);
    vertical-align: middle;
    margin-right: var(--space-1);
  }
</style>
