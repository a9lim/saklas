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
  // serialize into the same expression, which the persistent steering bar
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
    tokenHoverState,
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
  const hoverToken = $derived.by(() => {
    const visible = tokenHoverState.tokenText
      .replace(/\n/g, "↵")
      .replace(/\t/g, "⇥")
      .replace(/ /g, "·");
    return visible || "∅";
  });
  const hoverProbeCount = $derived.by(() => new Set([
    ...Object.keys(tokenHoverState.probeReadings ?? {}),
    ...Object.keys(tokenHoverState.probes ?? {}),
    ...Object.keys(tokenHoverState.coordsByProbe ?? {}),
  ]).size);
  const hoverLensCount = $derived(tokenHoverState.lensAggregate?.length ?? 0);
  const hoverSaeCount = $derived(tokenHoverState.saeReadout?.length ?? 0);
</script>

<aside class="inspector" aria-label="Saklas inspector">
  <RecipeBar />

  <div class="instrument-head">
    <nav class="tabs" aria-label="Instrument">
      <SegmentedTabs
        items={PILLARS}
        value={tab}
        onchange={(v) => setInspectorTab(v)}
        fill
        ariaLabel="Inspector instrument"
      />
    </nav>
    {#if tokenHoverState.active}
      <div class="token-readout" role="status" aria-live="off">
        <span class="readout-dot"></span>
        <span class="readout-label">token</span>
        <code title={tokenHoverState.tokenText}>{hoverToken}</code>
        <span class="readout-channels">
          {hoverProbeCount}p
          · {tokenHoverState.lensLoading ? "lens…" : `${hoverLensCount}l`}
          · {tokenHoverState.saeLoading ? "sae…" : `${hoverSaeCount}s`}
        </span>
      </div>
    {/if}
  </div>

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
  .instrument-head {
    min-width: 0;
  }
  .token-readout {
    display: grid;
    grid-template-columns: auto auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
    min-height: var(--control-target);
    margin: 0 var(--space-4) var(--space-1);
    padding: 3px var(--space-3);
    border-radius: var(--radius);
    background: var(--glass);
    color: var(--fg-subtle);
    font-size: var(--text-xs);
  }
  .readout-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--live);
  }
  .readout-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .token-readout code {
    overflow: hidden;
    color: var(--fg-strong);
    font-family: var(--font-mono);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .readout-channels {
    font-family: var(--font-mono);
    color: var(--fg-muted);
    white-space: nowrap;
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
