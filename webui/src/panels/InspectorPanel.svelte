<script lang="ts">
  // Inspector — the right-hand control rack: two equal sections, the
  // steering rack on top, the (fused vector + manifold) probe rack on
  // the bottom.  The sampling strip moved to the bottom of the threads
  // column (below the workbench card), giving these racks this column's
  // whole budget.
  //
  // Manifold probes used to live in their own third rack; they fused
  // into ProbeRack so the two surfaces (vector + manifold) read as one
  // family — same shape SteeringRack already uses for vector + manifold
  // steering.

  import SteeringRack from "./SteeringRack.svelte";
  import ProbeRack from "./ProbeRack.svelte";
  import WorkspacePanel from "./WorkspacePanel.svelte";
</script>

<aside class="inspector" aria-label="Saklas inspector">
  <div class="rack-grid">
    <SteeringRack />
    <ProbeRack />
    <!-- WORKSPACE (J-lens) — renders nothing when no lens is fitted, so
         the auto row collapses and the two racks keep the full budget. -->
    <WorkspacePanel />
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

  /* Two equal flat rack sections, divided only by SteeringRack's
   * border-bottom hairline — no gap, no nested boxes.  The third ``auto``
   * row is the WORKSPACE (J-lens) readout, content-sized: zero when no
   * lens is fitted (the panel renders nothing), a header strip when off,
   * a bounded matrix while live. */
  .rack-grid {
    display: grid;
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr) auto;
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
  }
</style>
