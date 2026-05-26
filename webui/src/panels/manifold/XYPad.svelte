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
    /** When true, the pad / sliders display the coords but reject
     *  input — used by ``ManifoldStrip`` for label-form terms, where
     *  the engine takes the position from the named node and the
     *  surfaced coords would be ignored.  Keeping the controls
     *  visible (instead of hiding them) gives the user a read-out of
     *  where the snapped node actually sits in the manifold. */
    locked?: boolean;
  }

  let { manifold, coords, onchange, locked = false }: Props = $props();

  interface AxisRange {
    name: string;
    lo: number;
    hi: number;
    periodic: boolean;
  }

  // Per-axis range + label.  Box domains carry explicit axes; sphere /
  // custom domains have no authored bounds, so we derive each axis as
  // a symmetric ``[-R, R]`` around 0 where ``R = max(1, ceil(max|v|))``
  // over the per-axis node coords.  Centering on 0 lines up the pad's
  // crosshair gridlines + the slider midpoint with the (0, 0, ...)
  // centroid where every freshly-racked manifold term starts, and
  // ``ceil(max-magnitude)`` keeps the range a clean whole number that
  // still envelops every node (a persona node at ``c0 = -10`` lands at
  // exactly the left endpoint with ``R = 10``).  The ``max(1, ...)``
  // floor keeps the slider usable when every node sits very close to
  // 0 on a particular axis.  Falls back to ``[-1, 1]`` when no fitted
  // coords are available (the unfitted-manifold pre-fit state).
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
    const coords = manifold.node_coords ?? [];
    return Array.from({ length: n }, (_, i) => {
      let r = 1;
      if (coords.length > 0) {
        let mag = 0;
        for (const row of coords) {
          const v = row?.[i];
          if (Number.isFinite(v)) {
            const a = Math.abs(v);
            if (a > mag) mag = a;
          }
        }
        if (mag > 0) r = Math.max(1, Math.ceil(mag));
      }
      return { name: `c${i}`, lo: -r, hi: r, periodic: false };
    });
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
    if (locked) return;
    dragging = true;
    (ev.currentTarget as HTMLElement).setPointerCapture(ev.pointerId);
    padToCoords(ev);
  }
  function onPadPointerMove(ev: PointerEvent): void {
    if (!dragging || locked) return;
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

<div class="xypad" class:locked>
  {#if is2D}
    <div
      class="pad"
      class:locked
      bind:this={padRef}
      role="slider"
      tabindex="0"
      aria-label="manifold position ({axes[0].name}, {axes[1].name})"
      aria-valuemin={axes[0].lo}
      aria-valuemax={axes[0].hi}
      aria-valuenow={coords[0] ?? axes[0].lo}
      aria-valuetext="{axes[0].name} {fmt(coords[0] ?? 0)}, {axes[1].name} {fmt(coords[1] ?? 0)}"
      aria-disabled={locked}
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
            disabled={locked}
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
  /* Label-form lock: the pad is a read-out, not a control.  Dim the
   * fill, switch the cursor away from crosshair, and let pointer
   * events still arrive (so the title hover survives) — the handlers
   * themselves short-circuit on ``locked``. */
  .pad.locked {
    cursor: not-allowed;
    opacity: 0.6;
  }
  /* The slider rows inside the non-2D fallback live in
   * ``.sliders > .axis-row > Slider``; the disabled prop already
   * handles input dimming, but we also dim the surrounding axis-name
   * + readout so the whole row reads "locked" at a glance. */
  .xypad.locked .sliders {
    opacity: 0.7;
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
