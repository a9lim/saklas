<script lang="ts">
  // The one provenance row every drilldown tab renders — identical order
  // everywhere so the tabs cannot drift: capture/replay origin, source,
  // resident layer, recipe, then the steered counterfactual control.

  import type { ReadoutOrigin } from "./readout.svelte";

  let {
    origin,
    source = null,
    layer = null,
    steering,
    steered = $bindable(),
    showToggle,
    accent = "var(--accent)",
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
    /** Instrument hue for the active recipe state. */
    accent?: string;
  } = $props();
</script>

<div class="inst-head" style:--inst-accent={accent}>
  <span class="context-label">readout</span>
  {#if origin}
    <span class="kv origin" title="original capture vs on-demand replay">{origin}</span>
  {/if}
  {#if source}
    <span class="kv source" title={`instrument artifact source · ${source}`}>{source}</span>
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
    <button
      type="button"
      class="steer-toggle"
      class:on={steered}
      aria-pressed={steered}
      title="replay under the node's recipe steering"
      onclick={() => { steered = !steered; }}
    >recipe {steered ? "on" : "off"}</button>
  {/if}
</div>

<style>
  .inst-head {
    --inst-accent: var(--accent);
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex-wrap: wrap;
    min-height: var(--control-field);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius);
    background: var(--input-well);
    font-size: var(--text-sm);
    line-height: 1.6;
  }
  .context-label {
    color: var(--inst-accent);
    font-size: var(--text-2xs);
    font-weight: var(--weight-bold);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-right: var(--space-1);
  }
  .kv {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    min-width: 0;
  }
  .origin {
    color: var(--fg);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .source {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 30ch;
  }
  .steer-chip code {
    color: var(--fg-strong);
    background: transparent;
    font-family: var(--font-mono);
  }
  button.steer-toggle {
    min-height: var(--control-target);
    margin-left: auto;
    padding: 1px var(--space-3);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    background: var(--glass);
    color: var(--fg-muted);
    font: inherit;
    font-size: var(--text-xs);
    cursor: pointer;
    user-select: none;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  button.steer-toggle:hover {
    color: var(--fg);
    background: var(--glass-strong);
  }
  button.steer-toggle.on {
    color: var(--inst-accent);
    background: color-mix(in srgb, var(--inst-accent) 10%, var(--glass));
  }
</style>
