<script lang="ts">
  // Inspector — the right-hand control rack, split into two tabs over the
  // ONE steering expression + probe roster:
  //
  //   CAA    — the activation-steering surface: steering rack (subspace +
  //            manifold terms) over the probe rack.
  //   J-LENS — the Jacobian-lens surface: token steer cards, pinned token
  //            probes, and the layer-aggregated workspace readout as
  //            per-token cards (blue family accent, RackCard chrome).
  //
  // The split is presentational — each tab shows its own term/probe
  // family; a j-lens steer chip and a subspace card serialize into the
  // same expression.  The tab strip always renders: with no fitted lens
  // the J-LENS tab hosts the "fit j-lens" button (the entry point to the
  // server's background fit), so the absent channel advertises itself.

  import SteeringRack from "./SteeringRack.svelte";
  import ProbeRack from "./ProbeRack.svelte";
  import JLensPanel from "./JLensPanel.svelte";
  import { inspectorState, setInspectorTab } from "../lib/stores.svelte";

  const tab = $derived(inspectorState.tab);
</script>

<aside class="inspector" aria-label="Saklas inspector">
  <nav class="tabs" aria-label="Inspector mode">
    <button
      type="button"
      class="tab"
      class:active={tab === "probes"}
      aria-pressed={tab === "probes"}
      onclick={() => setInspectorTab("probes")}
    >CAA</button>
    <button
      type="button"
      class="tab"
      class:active={tab === "jlens"}
      aria-pressed={tab === "jlens"}
      onclick={() => setInspectorTab("jlens")}
    >J-LENS</button>
  </nav>

  {#if tab === "jlens"}
    <JLensPanel />
  {:else}
    <div class="rack-grid">
      <SteeringRack />
      <ProbeRack />
    </div>
  {/if}
</aside>

<style>
  /* One flat panel — no outer padding, so the rack sections run edge to
   * edge and the hairlines between them are full-bleed. */
  .inspector {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
    background: var(--bg-alt);
  }
  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
  }
  .tab {
    flex: 1 1 0;
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    color: var(--fg-muted);
    font-size: var(--text-sm);
    font-weight: var(--weight-bold);
    text-transform: uppercase;
    padding: var(--space-3) 0;
    cursor: pointer;
  }
  .tab:hover {
    color: var(--fg);
  }
  .tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }

  /* STEER grows with its contents until it reaches half the inspector;
   * PROBE owns everything left over.  fit-content keeps the empty steer
   * rack compact while its own strips scroll once the cap is reached. */
  .rack-grid {
    display: grid;
    grid-template-rows: fit-content(50%) minmax(0, 1fr);
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
  }
</style>
