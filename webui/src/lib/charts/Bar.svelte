<script lang="ts">
  // Hand-rolled horizontal bar.  Positive = green, negative = red,
  // mirroring the TUI build_bar shape but in pixel-width SVG (aliased
  // against the GitHub-dark accent palette in tokens.css).
  //
  // ``value`` and ``max`` are unitless; ``width`` is the rendered max
  // width in pixels (defaults to BAR_WIDTH * 6 ≈ the TUI's 24-glyph
  // bar at typical web character cell width).

  interface Props {
    value: number;
    max: number;
    width?: number;
    height?: number;
    /** When true, emit a thin baseline rule under the bar.  Useful when
     * the bar lives inline with text and the visual rhythm needs a
     * floor.  Off by default. */
    showBaseline?: boolean;
    /** Override the bar fill color.  Defaults to the appropriate accent
     * based on sign of value. */
    color?: string;
  }

  let {
    value,
    max,
    width = 144,
    height = 8,
    showBaseline = false,
    color,
  }: Props = $props();

  const filled = $derived.by(() => {
    if (max <= 0 || !Number.isFinite(max)) return 0;
    const ratio = Math.min(1, Math.abs(value) / max);
    return Math.round(ratio * width);
  });

  const fill = $derived.by(() => {
    if (color) return color;
    if (value > 0) return "var(--accent-green)";
    if (value < 0) return "var(--accent-red)";
    return "var(--fg-muted)";
  });
</script>

<svg
  class="bar"
  {width}
  {height}
  viewBox="0 0 {width} {height}"
  preserveAspectRatio="none"
  aria-hidden="true"
>
  <rect x="0" y="0" {width} {height} class="track" />
  <rect x="0" y="0" width={filled} {height} fill={fill} class="fill" />
  {#if showBaseline}
    <line
      x1="0"
      x2={width}
      y1={height}
      y2={height}
      stroke="var(--border-dim)"
      stroke-width="0.5"
    />
  {/if}
</svg>

<style>
  .bar {
    display: inline-block;
    vertical-align: middle;
  }
  .track {
    fill: var(--bg-elev);
  }
  .fill {
    transition: width 0.15s ease-out;
  }
</style>
