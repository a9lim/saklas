<script lang="ts">
  // Manifold position picker.  One slider per intrinsic dimension,
  // uniformly: dim 1 -> one slider, dim 2 -> two sliders, dim N -> N
  // sliders.  (The old 2D draggable xy-pad special case was retired in
  // favor of this consistent control across every manifold rank.)
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
    /** When true, the sliders display the coords but reject input —
     *  used by ``ManifoldSteerCard`` for label-form terms, where the
     *  engine takes the position from the named node and the surfaced
     *  coords would be ignored.  Keeping the controls visible (instead
     *  of hiding them) gives the user a read-out of where the snapped
     *  node actually sits in the manifold. */
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
  // over the per-axis node coords.  Centering on 0 lines up the slider
  // midpoint with the (0, 0, ...) centroid where every freshly-racked
  // manifold term starts, and ``ceil(max-magnitude)`` keeps the range a
  // clean whole number that still envelops every node (a persona node
  // at ``c0 = -10`` lands at exactly the left endpoint with ``R = 10``).
  // The ``max(1, ...)`` floor keeps the slider usable when every node
  // sits very close to 0 on a particular axis.  Falls back to ``[-1, 1]``
  // when no fitted coords are available (the unfitted-manifold pre-fit
  // state).
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
    const nodeCoords = manifold.node_coords ?? [];
    return Array.from({ length: n }, (_, i) => {
      let r = 1;
      if (nodeCoords.length > 0) {
        let mag = 0;
        for (const row of nodeCoords) {
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

  function fmt(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }
</script>

<div class="xypad" class:locked>
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
</div>

<style>
  .xypad {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  /* The disabled prop already dims each Slider's input; also dim the
   * surrounding axis-name + readout so the whole row reads "locked". */
  .xypad.locked .sliders {
    opacity: 0.7;
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
