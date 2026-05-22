<script lang="ts">
  // Manifold position picker.  The control shape follows the domain:
  //
  //   1D box        → one slider
  //   2D box        → a draggable-dot XY pad
  //   3D box        → three sliders (one per axis)
  //   sphere / nD   → one slider per intrinsic coordinate
  //
  // Periodic axes wrap rather than clamp: a value that runs off one
  // edge re-enters at the other.  Each axis carries its own [lo, hi]
  // range; the parent owns the canonical ``coords`` array and receives
  // updates through ``onchange``.

  import type { ManifoldInfo } from "../../lib/types";
  import Slider from "../../lib/Slider.svelte";

  interface Props {
    manifold: ManifoldInfo;
    /** Current authoring coordinates — one per intrinsic dimension. */
    coords: number[];
    onchange: (coords: number[]) => void;
  }

  let { manifold, coords, onchange }: Props = $props();

  interface AxisRange {
    name: string;
    lo: number;
    hi: number;
    periodic: boolean;
  }

  // Per-axis range + label.  Box domains carry explicit axes; sphere /
  // custom fall back to a symmetric [-1, 1] range per intrinsic coord.
  const axes = $derived.by<AxisRange[]>(() => {
    const d = manifold.domain;
    if (d.type === "box") {
      return d.axes.map((a) => ({
        name: a.name,
        lo: a.lo,
        hi: a.hi,
        periodic: a.periodic,
      }));
    }
    const n = manifold.intrinsic_dim;
    return Array.from({ length: n }, (_, i) => ({
      name: `c${i}`,
      lo: -1,
      hi: 1,
      periodic: false,
    }));
  });

  /** Clamp (open axis) or wrap (periodic axis) a raw coordinate into
   *  the axis range. */
  function fit(axis: AxisRange, raw: number): number {
    const span = axis.hi - axis.lo;
    if (span <= 0) return axis.lo;
    if (axis.periodic) {
      let v = (raw - axis.lo) % span;
      if (v < 0) v += span;
      return axis.lo + v;
    }
    return Math.min(axis.hi, Math.max(axis.lo, raw));
  }

  function setCoord(idx: number, raw: number): void {
    const next = coords.slice();
    next[idx] = fit(axes[idx], raw);
    onchange(next);
  }

  // ---------- 2D XY pad ----------

  let padRef: HTMLDivElement | null = $state(null);
  let dragging = $state(false);

  /** Normalise a coordinate to [0, 1] within its axis range. */
  function norm(axis: AxisRange, v: number): number {
    const span = axis.hi - axis.lo;
    if (span <= 0) return 0;
    return Math.min(1, Math.max(0, (v - axis.lo) / span));
  }

  function padToCoords(ev: PointerEvent): void {
    const el = padRef;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width;
    // Y axis is inverted so "up" is the high end of axis 1.
    const fy = 1 - (ev.clientY - r.top) / r.height;
    const x = axes[0].lo + fx * (axes[0].hi - axes[0].lo);
    const y = axes[1].lo + fy * (axes[1].hi - axes[1].lo);
    onchange([fit(axes[0], x), fit(axes[1], y)]);
  }

  function onPadPointerDown(ev: PointerEvent): void {
    dragging = true;
    (ev.currentTarget as HTMLElement).setPointerCapture(ev.pointerId);
    padToCoords(ev);
  }
  function onPadPointerMove(ev: PointerEvent): void {
    if (!dragging) return;
    padToCoords(ev);
  }
  function onPadPointerUp(ev: PointerEvent): void {
    dragging = false;
    try {
      (ev.currentTarget as HTMLElement).releasePointerCapture(ev.pointerId);
    } catch {
      /* ignore */
    }
  }

  const is2D = $derived(axes.length === 2);

  // Dot position as CSS percentages for the 2D pad.
  const dotLeft = $derived(
    is2D ? norm(axes[0], coords[0] ?? axes[0].lo) * 100 : 0,
  );
  const dotTop = $derived(
    is2D ? (1 - norm(axes[1], coords[1] ?? axes[1].lo)) * 100 : 0,
  );

  function fmt(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }
</script>

<div class="xypad">
  {#if is2D}
    <div
      class="pad"
      bind:this={padRef}
      role="slider"
      tabindex="0"
      aria-label="manifold position ({axes[0].name}, {axes[1].name})"
      aria-valuemin={axes[0].lo}
      aria-valuemax={axes[0].hi}
      aria-valuenow={coords[0] ?? axes[0].lo}
      aria-valuetext="{axes[0].name} {fmt(coords[0] ?? 0)}, {axes[1].name} {fmt(coords[1] ?? 0)}"
      onpointerdown={onPadPointerDown}
      onpointermove={onPadPointerMove}
      onpointerup={onPadPointerUp}
    >
      <div class="grid-line h"></div>
      <div class="grid-line v"></div>
      <div
        class="dot"
        style:left="{dotLeft}%"
        style:top="{dotTop}%"
      ></div>
      <span class="axis-label x">{axes[0].name}{axes[0].periodic ? " ↻" : ""}</span>
      <span class="axis-label y">{axes[1].name}{axes[1].periodic ? " ↻" : ""}</span>
    </div>
    <p class="readout">
      {axes[0].name} = {fmt(coords[0] ?? 0)} · {axes[1].name} = {fmt(coords[1] ?? 0)}
    </p>
  {:else}
    <div class="sliders">
      {#each axes as axis, i (i)}
        <label class="axis-row">
          <span class="axis-name">
            {axis.name}{axis.periodic ? " ↻" : ""}
          </span>
          <Slider
            value={coords[i] ?? axis.lo}
            min={axis.lo}
            max={axis.hi}
            step={(axis.hi - axis.lo) / 100 || 0.01}
            oninput={(v) => setCoord(i, v)}
            ariaLabel="{axis.name} coordinate"
          />
          <span class="axis-val">{fmt(coords[i] ?? axis.lo)}</span>
        </label>
      {/each}
    </div>
  {/if}
</div>

<style>
  .xypad {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .pad {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    max-height: 220px;
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    cursor: crosshair;
    touch-action: none;
  }
  .grid-line {
    position: absolute;
    background: var(--border);
  }
  .grid-line.h {
    left: 0;
    right: 0;
    top: 50%;
    height: 1px;
  }
  .grid-line.v {
    top: 0;
    bottom: 0;
    left: 50%;
    width: 1px;
  }
  .dot {
    position: absolute;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--accent);
    border: 1px solid var(--bg-deep);
    transform: translate(-50%, -50%);
    box-shadow: var(--shadow-overlay);
    pointer-events: none;
  }
  .axis-label {
    position: absolute;
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    text-transform: lowercase;
  }
  .axis-label.x {
    bottom: var(--space-1);
    right: var(--space-2);
  }
  .axis-label.y {
    top: var(--space-2);
    left: var(--space-2);
  }
  .readout {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
  }
  .sliders {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .axis-row {
    display: grid;
    grid-template-columns: minmax(3em, auto) 1fr 3em;
    align-items: center;
    gap: var(--space-3);
  }
  .axis-name {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: lowercase;
  }
  .axis-val {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
</style>
