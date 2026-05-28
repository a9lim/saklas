<script lang="ts">
  // Inspector — the right-hand control rack: steering, probe, and (when
  // any are attached) the manifold-probe rack split the column.  The
  // sampling strip moved to the bottom of the threads column (below the
  // workbench card), giving these racks this column's whole budget.
  //
  // The manifold-probe rack hides itself when the server doesn't expose
  // the read side (``manifoldProbeRack.unavailable`` on an older saklas);
  // we render it unconditionally and let it self-suppress so the rack
  // grid stays simple.
  import SteeringRack from "./SteeringRack.svelte";
  import ProbeRack from "./ProbeRack.svelte";
  import ManifoldProbeRack from "./ManifoldProbeRack.svelte";
  import { manifoldProbeRack } from "../lib/stores.svelte";

  // Manifold rack visibility — the section reserves grid space only when
  // probes are attached OR the server supports the routes (so the
  // attach form remains reachable).  The component's own ``{#if
  // !unavailable}`` covers the latter; we mirror it here so the grid
  // collapses to two rows on legacy servers.
  const showManifoldRack = $derived(!manifoldProbeRack.unavailable);
</script>

<aside class="inspector" aria-label="Saklas inspector">
  <div class="rack-grid" class:with-manifold={showManifoldRack}>
    <SteeringRack />
    <ProbeRack />
    {#if showManifoldRack}
      <ManifoldProbeRack />
    {/if}
  </div>
</aside>

<style>
  /* One flat panel — no outer padding, so the two rack sections run
   * edge to edge and the hairline between them is full-bleed. */
  .inspector {
    display: grid;
    grid-template-rows: minmax(0, 1fr);
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
    background: var(--bg-alt);
  }

  /* Two equal flat sections, divided only by SteeringRack's border-bottom
   * hairline — no gap, no nested boxes.  When the manifold-probe rack
   * is shown (server supports the read side), the column splits into
   * three rows; the manifold rack lands a hair shorter than the others
   * because the per-row strip already carries its own mini-map and
   * doesn't need quite as much scroll budget. */
  .rack-grid {
    display: grid;
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
  }
  .rack-grid.with-manifold {
    grid-template-rows: minmax(0, 1.1fr) minmax(0, 1.1fr) minmax(0, 1fr);
  }
</style>
