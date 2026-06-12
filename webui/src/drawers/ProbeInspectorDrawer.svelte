<script lang="ts">
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

  import { closeDrawer, drawerState, probeRack } from "../lib/stores.svelte";
  import { apiProbes, ApiError } from "../lib/api";
  import Bar from "../lib/charts/Bar.svelte";
  import Select from "../lib/Select.svelte";
  import {
    renderProbeGeometry,
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
  const orbit = $state<OrbitState>({ az: 0.6, el: 0.5, zoom: 1.6 });

  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let rafId = 0;

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
      // default to the highest explained-variance layer
      let best: number | null = null;
      let bestEv = -Infinity;
      for (const l of Object.values(g.layers)) {
        if (l.explained_variance > bestEv) {
          bestEv = l.explained_variance;
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

  // sorted layer list (ascending) for the picker + the share bars
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
    const az = orbit.az;
    const el = orbit.el;
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
        orbit: { az, el, zoom },
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
    orbit.az += dx * 0.01;
    orbit.el = Math.max(-1.45, Math.min(1.45, orbit.el - dy * 0.01));
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
    if (r <= 1) return "line (rank 1)";
    if (r === 2) return "2D scatter (rank 2)";
    return `3D PCA scatter (rank ${r})`;
  });
  const intrinsicLabel = $derived(
    activeGeom ? `intrinsic dim ${activeGeom.intrinsic_dim}` : "",
  );
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" aria-label="Probe inspector">
  <header class="drawer-header">
    <div class="title">
      <span class="label">probe inspector</span>
      <span class="coord" title={probeName}>
        {#if probeName}
          {displayName}
          {#if geom}· {rankLabel} · {intrinsicLabel}{/if}
        {:else}
          no probe
        {/if}
      </span>
    </div>
    <button type="button" class="close" onclick={onClose} aria-label="Close drawer">×</button>
  </header>

  {#if !probeName}
    <div class="body"><div class="empty">nothing selected</div></div>
  {:else if loading}
    <div class="body"><div class="empty">loading…</div></div>
  {:else if error}
    <div class="body"><div class="empty err">error: {error}</div></div>
  {:else if !geom || layerList.length === 0}
    <div class="body"><div class="empty">no fitted geometry for {displayName}</div></div>
  {:else}
    <div class="picker-row">
      <label class="picker">
        <span class="picker-label">layer</span>
        <Select
          value={selectedLayer ?? layerList[0].layer}
          options={layerList.map((l) => ({
            value: l.layer,
            label: `L${l.layer} · ev ${l.explained_variance.toFixed(2)}`,
          }))}
          onchange={(v) => (selectedLayer = v)}
          ariaLabel="Layer"
        />
      </label>
      {#if !geom.rank_uniform}
        <span class="warn" title="this flat fit kept a different rank per layer">
          rank varies by layer
        </span>
      {/if}
    </div>

    <div class="body">
      <div class="bars-col">
        <div class="section-label">per-layer ‖share‖<span class="dim"> · click to scrub</span></div>
        <div class="bars">
          {#each layerList as l (l.layer)}
            <button
              type="button"
              class="row"
              class:active={l.layer === selectedLayer}
              onclick={() => (selectedLayer = l.layer)}
            >
              <span class="lyr">L{l.layer}</span>
              <Bar value={l.mahalanobis_share} max={maxShare || 1} width={200} height={8} />
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
          {#if canOrbit}
            <span class="orbit-hint">drag to orbit · scroll to zoom</span>
          {/if}
          {#if trailPoints.length > 0}
            <span class="trail-hint">{trailPoints.length} trail pts</span>
          {:else}
            <span class="live-hint">generate to see the live point + trail</span>
          {/if}
        </div>
      </div>
    </div>
  {/if}

  <footer class="drawer-footer">
    <span class="hint">
      Whitened-frame geometry — node centroids, neutral anchor, and the manifold
      overlay in the same Mahalanobis metric the reads use.  The bright dot is
      the current hidden state; the fading trail is the last tokens.
    </span>
  </footer>
</aside>

<style>
  .drawer {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text);
    border-left: 1px solid var(--border);
  }
  .drawer-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-4);
    padding: var(--space-4);
    border-bottom: 1px solid var(--border);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: uppercase;
  }
  .coord {
    color: var(--fg-dim);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .close {
    background: transparent;
    color: var(--fg-muted);
    border: 1px solid var(--border);
    padding: 0 var(--space-3);
    font: inherit;
    font-size: var(--text-md);
    cursor: pointer;
    line-height: 1.4;
  }
  .close:hover {
    color: var(--fg-strong);
    border-color: var(--fg-muted);
  }

  .picker-row {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-4);
    border-bottom: 1px solid var(--border);
  }
  .picker {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex: 1 1 auto;
  }
  .picker-label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .picker :global(.sk-select) {
    flex: 1 1 auto;
  }
  .warn {
    color: var(--accent-yellow);
    font-size: var(--text-xs);
  }

  .body {
    flex: 1 1 auto;
    overflow: hidden;
    min-height: 0;
    padding: var(--space-4);
    display: flex;
    flex-direction: row;
    gap: var(--space-4);
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
    font-style: italic;
    padding: var(--space-5) 0;
  }
  .empty.err {
    color: var(--accent-error);
    font-style: normal;
  }

  .plot-wrap {
    position: relative;
    flex: 1 1 auto;
    min-height: 0;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-deep);
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
  .orbit-hint,
  .live-hint,
  .trail-hint {
    position: absolute;
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    pointer-events: none;
  }
  .orbit-hint {
    top: var(--space-2);
    right: var(--space-3);
  }
  .live-hint {
    bottom: var(--space-2);
    left: 0;
    right: 0;
    text-align: center;
    font-style: italic;
  }
  .trail-hint {
    bottom: var(--space-2);
    left: var(--space-3);
    color: var(--accent-green);
    font-variant-numeric: tabular-nums;
  }

  .section-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: uppercase;
    margin-bottom: var(--space-2);
  }
  .section-label .dim {
    color: var(--fg-dim);
    text-transform: none;
  }
  .bars {
    display: flex;
    flex-direction: column;
    gap: 1px;
    font-size: var(--text-xs);
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    background: transparent;
    border: 0;
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-2);
    cursor: pointer;
    text-align: left;
    color: inherit;
    font: inherit;
  }
  .row:hover {
    background: var(--bg-elev);
  }
  .row.active {
    background: var(--bg-elev);
  }
  .row.active .lyr {
    color: var(--accent-purple);
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
    border-top: 1px solid var(--border);
    padding: var(--space-2) var(--space-4);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .hint {
    line-height: 1.4;
  }
</style>
