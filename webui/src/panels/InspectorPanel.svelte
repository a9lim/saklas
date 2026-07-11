<script lang="ts">
  // Inspector — the instrument stack.  Four coequal pillars over the ONE
  // steering expression + probe roster, each with the same verbs
  // (observe / steer / gate) and its own hue:
  //
  //   subspace — flat/affine fits (concept axes, personas) · white
  //   manifold — curved fits (emotions, months) · violet
  //   sae      — resident sparse-autoencoder feature space · gold
  //   lens     — the Jacobian-lens surface (JLensPanel) · blue
  //
  // The split is presentational — a lens steer chip and a subspace card
  // serialize into the same expression, which the persistent RecipeBar
  // above the tabs keeps visible across all four.  With no fitted lens the
  // lens tab hosts the "fit j-lens" button, so the absent channel
  // advertises itself; the SAE tab offers a release picker when unloaded.

  import SteeringRack from "./SteeringRack.svelte";
  import ProbeRack from "./ProbeRack.svelte";
  import JLensPanel from "./JLensPanel.svelte";
  import SaePanel from "./SaePanel.svelte";
  import RecipeBar from "./RecipeBar.svelte";
  import SegmentedTabs from "../lib/ui/SegmentedTabs.svelte";
  import {
    inspectorState,
    setInspectorTab,
    type InspectorTab,
  } from "../lib/stores.svelte";

  const PILLARS: {
    value: InspectorTab;
    label: string;
    color: string;
    title: string;
  }[] = [
    {
      value: "subspace",
      label: "subspace",
      color: "var(--pillar-subspace)",
      title: "Flat/affine fits — concept axes, personas",
    },
    {
      value: "manifold",
      label: "manifold",
      color: "var(--pillar-manifold)",
      title: "Curved fits — emotions, months",
    },
    {
      value: "sae",
      label: "sae",
      color: "var(--pillar-sae)",
      title: "Sparse-autoencoder features",
    },
    {
      value: "lens",
      label: "lens",
      color: "var(--pillar-lens)",
      title: "Jacobian lens — the workspace readout",
    },
  ];

  const tab = $derived(inspectorState.tab);
</script>

<aside class="inspector" aria-label="Saklas inspector">
  <RecipeBar />

  <nav class="tabs" aria-label="Instrument">
    <SegmentedTabs
      items={PILLARS}
      value={tab}
      onchange={(v) => setInspectorTab(v)}
      fill
      ariaLabel="Inspector instrument"
    />
  </nav>

  {#if tab === "lens"}
    <JLensPanel />
  {:else if tab === "sae"}
    <SaePanel />
  {:else if tab === "manifold"}
    <div class="rack-grid">
      <SteeringRack family="manifold" />
      <ProbeRack family="manifold" />
    </div>
  {:else}
    <div class="rack-grid">
      <SteeringRack family="subspace" />
      <ProbeRack family="subspace" />
    </div>
  {/if}
</aside>

<style>
  /* One flat panel — no outer padding on the rack rows, so the rack
   * sections run edge to edge and the hairlines between them are
   * full-bleed.  Rows: recipe bar · tab strip · the active instrument. */
  .inspector {
    display: grid;
    grid-template-rows: auto auto minmax(0, 1fr);
    height: 100%;
    max-height: 100%;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
    background: var(--bg-alt);
  }
  .tabs {
    padding: var(--space-4) var(--space-4) var(--space-3);
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
    min-width: 0;
    overflow: hidden;
  }
</style>
