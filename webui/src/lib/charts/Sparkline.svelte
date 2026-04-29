<script lang="ts">
  // Tiny inline sparkline.  Renders a polyline over the points list,
  // auto-scaled to the visible range.  Designed for live append — the
  // store's probe rack drops trailing values when the buffer grows
  // past N, so the component is fed a fresh array each tick and the
  // SVG re-renders cheaply.
  //
  // No interactivity, no axes — pure decoration to give a sense of
  // probe trend without a full chart.

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
</script>

<svg
  class="sparkline"
  {width}
  {height}
  viewBox="0 0 {width} {height}"
  preserveAspectRatio="none"
  aria-hidden="true"
>
  {#if path}
    <path d={path} fill="none" stroke={stroke} stroke-width="1" />
  {/if}
</svg>

<style>
  .sparkline {
    display: inline-block;
    vertical-align: middle;
  }
</style>
