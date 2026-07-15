<script lang="ts">
  // Tiny inline sparkline.  Renders a polyline over the points list,
  // auto-scaled to the visible range.  Designed for live append — the
  // store's probe rack drops trailing values when the buffer grows
  // past N, so the component is fed a fresh array each tick and the
  // SVG re-renders cheaply.
  //
  // No interactivity, no axes — pure decoration to give a sense of
  // probe trend without a full chart.

  import { onMount } from "svelte";

  interface Props {
    points: number[];
    width?: number;
    height?: number;
    /** When set, points are clamped to ``[-cap, +cap]`` before scaling
     * so a single outlier doesn't squash the rest of the trace.  Probe
     * scores live in [-1, 1] so 1 is a sensible default. */
    cap?: number;
    /** Stroke color override; defaults to fg-dim. */
    color?: string;
  }

  let {
    points,
    width = 60,
    height = 16,
    cap = 1,
    color,
  }: Props = $props();

  const stroke = $derived(color ?? "var(--fg-dim)");

  const path = $derived.by(() => {
    if (!points || points.length === 0) return "";
    if (points.length === 1) {
      // Single-point: render a tick at the midline.
      const y = height / 2;
      return `M 0 ${y.toFixed(2)} L ${width} ${y.toFixed(2)}`;
    }
    const clamped = points.map((v) =>
      Math.max(-cap, Math.min(cap, Number.isFinite(v) ? v : 0)),
    );
    // Map [-cap, +cap] -> [height, 0] so positive points trend upward.
    const yFor = (v: number) =>
      ((cap - v) / (2 * cap)) * height;
    const step = points.length === 1 ? 0 : width / (points.length - 1);
    const segs: string[] = [];
    for (let i = 0; i < clamped.length; i++) {
      const x = i * step;
      const y = yFor(clamped[i]);
      segs.push(`${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`);
    }
    return segs.join(" ");
  });

  // Area fill under the line — the same trace closed down to the baseline
  // and back, rendered behind the stroke at low opacity.  Skipped for the
  // degenerate 0/1-point case (no meaningful area to shade).
  const area = $derived.by(() => {
    if (!points || points.length < 2) return "";
    const clamped = points.map((v) =>
      Math.max(-cap, Math.min(cap, Number.isFinite(v) ? v : 0)),
    );
    const yFor = (v: number) => ((cap - v) / (2 * cap)) * height;
    const step = width / (points.length - 1);
    const segs: string[] = [];
    for (let i = 0; i < clamped.length; i++) {
      const x = i * step;
      const y = yFor(clamped[i]);
      segs.push(`${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`);
    }
    segs.push(`L ${width.toFixed(2)} ${height.toFixed(2)}`);
    segs.push(`L 0 ${height.toFixed(2)}`);
    segs.push("Z");
    return segs.join(" ");
  });

  // Draw-in runs once, gated on mount — this component re-renders every
  // streamed token, so the animation is bound to a class flipped a single
  // time (never to the per-tick data), and the dasharray uses a fixed
  // over-length value so path updates can't restart it.
  let mounted = $state(false);
  onMount(() => {
    mounted = true;
  });
</script>

<svg
  class="sparkline"
  {width}
  {height}
  viewBox="0 0 {width} {height}"
  preserveAspectRatio="none"
  aria-hidden="true"
>
  {#if area}
    <path d={area} fill={stroke} fill-opacity="0.14" stroke="none" />
  {/if}
  {#if path}
    <path
      class="sparkline-line"
      class:draw={mounted}
      d={path}
      fill="none"
      stroke={stroke}
      stroke-width="1"
    />
  {/if}
</svg>

<style>
  .sparkline {
    display: inline-block;
    vertical-align: middle;
  }
  /* Fixed over-length dash (path is ≪400 units) so per-token ``d`` updates
     never re-measure or restart the draw-in; the animation is carried by
     the ``.draw`` class, applied once on mount. */
  .sparkline-line {
    stroke-dasharray: 400;
    stroke-dashoffset: 0;
  }
  .sparkline-line.draw {
    animation: sparkline-draw var(--dur-draw) var(--ease-out);
  }
  @keyframes sparkline-draw {
    from {
      stroke-dashoffset: 400;
    }
    to {
      stroke-dashoffset: 0;
    }
  }
</style>
