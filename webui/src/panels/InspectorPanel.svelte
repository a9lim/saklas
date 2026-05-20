<script lang="ts">
  // Inspector — the right-hand control rack: the steering and probe
  // racks split the full column height.  The sampling strip moved to
  // the bottom of the threads column (below the workbench card), giving
  // the racks this column's whole budget.
  import SteeringRack from "./SteeringRack.svelte";
  import ProbeRack from "./ProbeRack.svelte";
</script>

<aside class="inspector" aria-label="Saklas inspector">
  <div class="rack-grid">
    <SteeringRack />
    <ProbeRack />
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
   * hairline — no gap, no nested boxes. */
  .rack-grid {
    display: grid;
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
  }
</style>
