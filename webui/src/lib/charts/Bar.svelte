<script lang="ts">
  // Hand-rolled horizontal bar.  Positive = green, negative = red,
  // mirroring the TUI build_bar shape but in pixel-width SVG.
  //
  // ``value`` and ``max`` are unitless; ``width`` is the rendered max
  // width in pixels (defaults to BAR_WIDTH * 6 ≈ the TUI's 24-glyph
  // bar at typical web character cell width).
  //
  // ``bipolar`` flips the bar to a center-zero shape: fills rightward
  // from the midline for positive values, leftward for negative, with a
  // thin midline tick.  Matches the steering slider's axis so a bipolar
  // probe row reads "neg ◄──█──► pos" in the same direction the user
  // moves the steering slider.

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
    /** Render as a center-zero bar (fills from midline outward).  Off
     * by default — the existing unipolar shape is preserved for all
     * non-probe callers. */
    bipolar?: boolean;
  }

  let {
    value,
    max,
    width = 144,
    height = 8,
    showBaseline = false,
    color,
    bipolar = false,
  }: Props = $props();

  const filled = $derived.by(() => {
    if (max <= 0 || !Number.isFinite(max)) return 0;
    const ratio = Math.min(1, Math.abs(value) / max);
    return Math.round(ratio * (bipolar ? width / 2 : width));
  });

  /** x-coordinate where the fill rectangle starts.  Unipolar bars start
   * at 0; bipolar negative bars start at ``mid - filled`` so they grow
   * leftward from center. */
  const fillX = $derived.by(() => {
    if (!bipolar) return 0;
    const mid = width / 2;
    return value < 0 ? mid - filled : mid;
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
  <rect x={fillX} y="0" width={filled} {height} fill={fill} class="fill" />
  {#if bipolar}
    <!-- Center tick so users can read sign at a glance even when value
         is exactly 0 (no fill rectangle to anchor the eye). -->
    <line
      x1={width / 2}
      x2={width / 2}
      y1="0"
      y2={height}
      stroke="var(--border)"
      stroke-width="1"
    />
  {/if}
  {#if showBaseline}
    <line
      x1="0"
      x2={width}
      y1={height}
      y2={height}
      stroke="var(--border)"
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
    /* Animate both x and width together — the bipolar bar encodes
     * negative values as (x = mid − filled, width = filled), so width-
     * only transitions left the right edge detaching from the center
     * tick mid-transition (visible as left/right jitter at the 0 line).
     * Transitioning both keeps x + width = mid at every animation
     * frame; positive bars (x ≡ 0) are unaffected. */
    transition:
      width var(--dur) var(--ease-out),
      x var(--dur) var(--ease-out);
  }
</style>
