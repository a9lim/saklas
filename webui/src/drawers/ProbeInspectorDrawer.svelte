<script lang="ts">
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // Per-probe inspector — subsumes the layer-norms view for probes and adds a
  // rank-aware geometry plot in the whitened (Mahalanobis) frame:
  //
  //   rank 1   -> a line: poles + neutral + live point
  //   rank 2   -> a 2D scatter of node centroids (+ curve overlay if 1-D)
  //   rank 3+  -> a drag-orbit 3D scatter on the top-3 subspace PCs
  //              (+ curve / wireframe-surface overlay)
  //
  // A layer scrubber drives which fitted layer is shown; geometry is fetched
  // once (all layers) and reprojected client-side on scrub.  The live hidden-
  // state point + a fading trajectory trail ride the probe's per-token
  // ``subspace_coords_per_layer`` (gated on by ``persist_subspace_coords`` while
  // this drawer is open), stored across all layers so scrubbing is a pure read.
  //
  // v2 sheet interior: the drawer host paints the sheet (glass hairline,
  // radius, --bg-alt fill) so the root here is transparent.  The probe's
  // FAMILY hue (flat = subspace white, curved = manifold violet — the same
  // is_affine split the racks use) accents the header dot, the active layer
  // row, the share bars, and the plot's node centroids via ``--geom-node``.

  import { closeDrawer, drawerState, probeRack } from "../lib/stores.svelte";
  import { apiProbes, ApiError } from "../lib/api";
  import Bar from "../lib/charts/Bar.svelte";
  import {
    renderProbeGeometry,
    orbitDrag,
    DEFAULT_ORBIT_QUAT,
    type OrbitState,
  } from "../lib/charts/probeGeometry";
  import type { ProbeGeometryResponse, ProbeLayerGeometry } from "../lib/types";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  const params = $derived(drawerState.params as { name?: string } | null);
  const probeName = $derived(params?.name ?? "");
  const displayName = $derived(probeName.split("/").pop() ?? probeName);
  const entry = $derived(probeRack.entries.get(probeName) ?? null);

  let geom = $state<ProbeGeometryResponse | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let selectedLayer = $state<number | null>(null);
  const orbit = $state<OrbitState>({ q: DEFAULT_ORBIT_QUAT, zoom: 1.6 });

  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let rafId = 0;

  /** Family hue — the rack's flat/curved split (hue = which space). */
  const familyHue = $derived(
    geom?.is_affine === false
      ? "var(--pillar-manifold)"
      : "var(--pillar-subspace)",
  );

  // --- load geometry on probe change ---
  async function load(name: string): Promise<void> {
    if (!name) {
      geom = null;
      return;
    }
    loading = true;
    error = null;
    geom = null;
    try {
      const g = await apiProbes.geometry(name);
      geom = g;
      // default to the highest-share layer (the one carrying the most
      // steering budget — also the most concept-bearing to read from)
      let best: number | null = null;
      let bestShare = -Infinity;
      for (const l of Object.values(g.layers)) {
        const sh = Math.abs(l.mahalanobis_share);
        if (sh > bestShare) {
          bestShare = sh;
          best = l.layer;
        }
      }
      selectedLayer = best;
    } catch (e) {
      error =
        e instanceof ApiError
          ? `${e.status}`
          : e instanceof Error
            ? e.message
            : String(e);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    void load(probeName);
  });

  // sorted layer list (ascending) for the share strip
  const layerList = $derived.by<ProbeLayerGeometry[]>(() =>
    geom ? Object.values(geom.layers).sort((a, b) => a.layer - b.layer) : [],
  );
  const activeGeom = $derived(
    geom && selectedLayer !== null
      ? (geom.layers[String(selectedLayer)] ?? null)
      : null,
  );
  const maxShare = $derived(
    layerList.reduce((m, l) => Math.max(m, Math.abs(l.mahalanobis_share)), 0),
  );

  // --- live point + trail for the selected layer (reprojected client-side) ---
  const layerKey = $derived(selectedLayer !== null ? String(selectedLayer) : "");
  const livePoint = $derived.by<number[] | null>(() => {
    const trail = entry?.subspaceTrail;
    if (!trail || trail.length === 0) return null;
    return trail[trail.length - 1]?.perLayer[layerKey] ?? null;
  });
  const trailPoints = $derived.by<number[][]>(() => {
    const trail = entry?.subspaceTrail ?? [];
    const out: number[][] = [];
    for (const s of trail) {
      const p = s.perLayer[layerKey];
      if (Array.isArray(p)) out.push(p);
    }
    return out;
  });

  // --- canvas render (rAF-coalesced; re-runs on any dependency change) ---
  $effect(() => {
    const canvas = canvasEl;
    const g = activeGeom;
    // touch the reactive deps so the effect re-subscribes
    const live = livePoint;
    const trail = trailPoints;
    const q = orbit.q;
    const zoom = orbit.zoom;
    const labels = geom?.node_labels ?? [];
    if (!canvas || !g) return;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      renderProbeGeometry(canvas, {
        geom: g,
        nodeLabels: labels,
        live,
        trail,
        orbit: { q, zoom },
      });
    });
  });

  // --- orbit interaction (rank >= 3 only) ---
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  const canOrbit = $derived((activeGeom?.rank ?? 0) >= 3);

  function onPointerDown(ev: PointerEvent): void {
    if (!canOrbit) return;
    dragging = true;
    lastX = ev.clientX;
    lastY = ev.clientY;
    (ev.currentTarget as HTMLElement).setPointerCapture(ev.pointerId);
    ev.preventDefault();
  }
  function onPointerMove(ev: PointerEvent): void {
    if (!dragging || !canOrbit) return;
    const dx = ev.clientX - lastX;
    const dy = ev.clientY - lastY;
    lastX = ev.clientX;
    lastY = ev.clientY;
    // Trackball: compose a screen-axis rotation onto the accumulated
    // orientation.  No Euler angles → no gimbal lock, no elevation clamp.
    orbit.q = orbitDrag(orbit.q, dx, dy);
  }
  function onPointerUp(ev: PointerEvent): void {
    dragging = false;
    try {
      (ev.currentTarget as HTMLElement).releasePointerCapture(ev.pointerId);
    } catch {
      /* ignore */
    }
  }
  // Scroll wheel = intentional zoom (the rotation-driven zoom artifact is
  // gone; this is the only zoom path now).  Multiplicative so each notch is a
  // constant ratio; clamped to a sane window.
  function onWheel(ev: WheelEvent): void {
    if (!canOrbit) return;
    ev.preventDefault();
    const factor = Math.exp(-ev.deltaY * 0.0015);
    orbit.zoom = Math.max(0.3, Math.min(6, orbit.zoom * factor));
  }

  function onClose(): void {
    closeDrawer();
  }
  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      onClose();
    }
  }

  const rankLabel = $derived.by(() => {
    const r = activeGeom?.rank ?? 0;
    if (r <= 1) return "line · rank 1";
    if (r === 2) return "2D scatter · rank 2";
    return `3D PCA scatter · rank ${r}`;
  });
  const intrinsicLabel = $derived(
    activeGeom ? `intrinsic dim ${activeGeom.intrinsic_dim}` : "",
  );
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" style:--family={familyHue} aria-label="Probe inspector">
  <header class="drawer-header">
    <div class="title">
      <span class="eyebrow">probe geometry</span>
      <div class="name-row">
        {#if probeName}
          <span class="family-dot" aria-hidden="true"></span>
          <code class="name" title={probeName}>{displayName}</code>
          {#if geom}
            <span class="meta">{rankLabel} · {intrinsicLabel}</span>
            {#if !geom.rank_uniform}
              <span class="warn" title="rank varies by layer">
                rank varies by layer
              </span>
            {/if}
          {/if}
        {:else}
          <span class="meta">no probe</span>
        {/if}
      </div>
    </div>
    <DrawerCloseButton onclick={onClose} />
  </header>

  {#if !probeName}
    <div class="body"><div class="empty">select a probe</div></div>
  {:else if loading}
    <div class="body"><div class="empty">loading…</div></div>
  {:else if error}
    <div class="body"><div class="empty err">error: {error}</div></div>
  {:else if !geom || layerList.length === 0}
    <div class="body"><div class="empty">no geometry</div></div>
  {:else}
    <div class="body">
      <div class="bars-col">
        <div class="section-label">layers · ‖share‖</div>
        <div class="bars">
          {#each layerList as l (l.layer)}
            <button
              type="button"
              class="row"
              class:active={l.layer === selectedLayer}
              onclick={() => (selectedLayer = l.layer)}
            >
              <span class="lyr">L{l.layer}</span>
              <Bar
                value={l.mahalanobis_share}
                max={maxShare || 1}
                width={200}
                height={8}
                color="var(--family)"
              />
              <span class="val">{l.mahalanobis_share.toFixed(3)}</span>
            </button>
          {/each}
        </div>
      </div>

      <div class="plot-col">
        <div class="plot-wrap" class:orbit={canOrbit}>
          <canvas
            bind:this={canvasEl}
            class="plot"
            onpointerdown={onPointerDown}
            onpointermove={onPointerMove}
            onpointerup={onPointerUp}
            onwheel={onWheel}
          ></canvas>
          <span class="layer-chip">L{selectedLayer}</span>
          {#if canOrbit}
            <span class="orbit-hint">drag · scroll</span>
          {/if}
          {#if trailPoints.length > 0}
            <span class="trail-hint">{trailPoints.length} trail pts</span>
          {:else}
            <span class="live-hint">run for live trail</span>
          {/if}
        </div>
      </div>
    </div>
  {/if}

  <footer class="drawer-footer">
    <span class="hint">
      Whitened-frame geometry — node centroids, neutral anchor, and the manifold
      overlay in the same Mahalanobis metric the reads use.  The glowing dot is
      the current hidden state; the fading trail is the last tokens.
    </span>
  </footer>
</aside>

<style>
  /* v2 sheet interior — the host paints the sheet surface, so the root
   * stays transparent and chrome speaks sans (data stays mono). */
  .drawer {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    background: transparent;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .drawer-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--space-5) var(--space-6);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .eyebrow {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .name-row {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .family-dot {
    align-self: center;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--family);
    box-shadow: 0 0 6px color-mix(in srgb, var(--family) 45%, transparent);
    flex: none;
  }
  .name {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .meta {
    color: var(--fg-subtle);
    font-size: var(--text-sm);
    white-space: nowrap;
  }
  .warn {
    color: var(--accent-yellow);
    font-size: var(--text-xs);
  }

  .body {
    flex: 1 1 auto;
    overflow: hidden;
    min-height: 0;
    padding: var(--space-5) var(--space-6);
    display: flex;
    flex-direction: row;
    gap: var(--space-6);
  }
  .bars-col {
    flex: 0 0 auto;
    min-height: 0;
    overflow-y: auto;
    padding-right: var(--space-2);
  }
  .plot-col {
    flex: 1 1 auto;
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .empty {
    color: var(--fg-muted);
    padding: var(--space-6) 0;
  }
  .empty.err {
    color: var(--accent-red);
  }

  /* The plot well — a deep glass window with a faint family-tinted
   * ambient (material, not data) so the geometry reads as suspended. */
  .plot-wrap {
    position: relative;
    flex: 1 1 auto;
    min-height: 0;
    border-radius: var(--radius-lg);
    background:
      radial-gradient(
        90% 75% at 50% 42%,
        color-mix(in srgb, var(--family) 8%, transparent),
        transparent 72%
      ),
      var(--bg-deep);
    box-shadow: var(--shadow-rack);
    overflow: hidden;
    /* Palette hooks read by the canvas renderer (hue ontology). */
    --geom-node: var(--family);
    --geom-neutral: var(--fg-subtle);
  }
  .plot-wrap.orbit .plot {
    cursor: grab;
    touch-action: none;
  }
  .plot-wrap.orbit .plot:active {
    cursor: grabbing;
  }
  .plot {
    position: absolute;
    inset: 0;
    display: block;
    width: 100%;
    height: 100%;
  }
  .layer-chip {
    position: absolute;
    top: var(--space-3);
    left: var(--space-4);
    color: color-mix(in srgb, var(--family) 80%, var(--fg));
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    padding: 1px var(--space-4);
    pointer-events: none;
  }
  .orbit-hint,
  .live-hint,
  .trail-hint {
    position: absolute;
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    pointer-events: none;
  }
  .orbit-hint {
    top: var(--space-3);
    right: var(--space-4);
  }
  .live-hint {
    bottom: var(--space-3);
    left: 0;
    right: 0;
    text-align: center;
    font-style: italic;
  }
  .trail-hint {
    bottom: var(--space-3);
    left: var(--space-4);
    color: var(--accent-green);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }

  .section-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: var(--space-3);
  }
  .bars {
    display: flex;
    flex-direction: column;
    gap: 1px;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-3);
    cursor: pointer;
    text-align: left;
    color: inherit;
    font: inherit;
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .row:hover {
    background: var(--bg-hover);
  }
  .row.active {
    background: var(--glass-strong);
    border-color: color-mix(in srgb, var(--family) 25%, var(--glass-line));
  }
  .row.active .lyr {
    color: color-mix(in srgb, var(--family) 85%, var(--fg));
  }
  .lyr {
    color: var(--fg-muted);
    width: 3em;
    text-align: right;
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .val {
    color: var(--fg-dim);
    width: 5em;
    text-align: right;
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }

  .drawer-footer {
    padding: var(--space-3) var(--space-6);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .hint {
    line-height: 1.5;
  }
</style>
