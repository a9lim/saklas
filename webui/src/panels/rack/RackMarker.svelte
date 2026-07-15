<script lang="ts">
  // One optical marker system for every rack card. Unicode shape glyphs
  // have wildly different intrinsic boxes, so a shared outer box plus
  // per-shape optical compensation keeps circles, diamonds, squares, and
  // triangles the same apparent size and baseline across pillar switches.

  export type RackMarkerShape = "circle" | "diamond" | "square" | "triangle";

  let {
    shape,
    filled = false,
  }: {
    shape: RackMarkerShape;
    filled?: boolean;
  } = $props();

  const glyph = $derived.by(() => {
    if (shape === "circle") return filled ? "●" : "○";
    if (shape === "diamond") return filled ? "◆" : "◇";
    if (shape === "triangle") return filled ? "▲" : "△";
    return filled ? "■" : "□";
  });
</script>

<span class="marker {shape}" aria-hidden="true">{glyph}</span>

<style>
  .marker {
    display: inline-grid;
    place-items: center;
    inline-size: 18px;
    block-size: 18px;
    flex: 0 0 18px;
    font-family: var(--font-mono);
    font-size: 14px;
    font-variant-numeric: tabular-nums;
    font-weight: var(--weight-medium);
    line-height: 1;
    text-align: center;
  }
  /* Optical, not layout, compensation: all four shapes keep the same
     18px hit/alignment box while their visible ink reads equally large. */
  .diamond {
    font-size: 14.5px;
  }
  .square {
    font-size: 13.5px;
  }
  .triangle {
    font-size: 15px;
    transform: translateY(-0.25px);
  }
</style>
