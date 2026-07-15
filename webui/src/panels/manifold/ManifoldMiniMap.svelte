<script lang="ts">
  // Small SVG mini-map for a 2D box-domain manifold probe.  Lays out the
  // probe's node coords inside the domain bounds, draws labels at their
  // authoring positions, then overlays a per-token trajectory polyline
  // (light) and a bold final dot at the settled ``coords`` aggregate.
  //
  // Two reading modes:
  //   * Streaming        — ``aggregate`` is null, trajectory grows token-
  //                        by-token; the latest trajectory point doubles
  //                        as the live cursor.
  //   * Settled          — ``aggregate`` non-null, the bold dot lands at
  //                        ``aggregate.coords`` and the trajectory is
  //                        rendered as the historical path that led
  //                        there.
  //
  // Box domains only — higher-dim and sphere / custom manifolds skip the
  // mini-map and surface their reading through the strip's text-only
  // ``coords`` readout instead (callers gate on this via
  // ``_isMiniMapCandidate`` in stores).

  import type { AxisSpec, ProbeInfo } from "../../lib/types";

  interface Props {
    info: ProbeInfo;
    /** Per-token inferred coords, in domain-authoring space.  Empty
     *  until the first token of the first generation lands. */
    trajectory: number[][];
    /** Settled coords from the final ``done`` aggregate.  ``null`` while
     *  streaming. */
    settled: number[] | null;
    /** Square edge length in CSS pixels.  Defaults to a value tuned for
     *  the inspector strip width. */
    size?: number;
  }

  let { info, trajectory, settled, size = 168 }: Props = $props();

  /** Pull the exact two box-domain axes out of the probe row. The caller
   *  (``_isMiniMapCandidate``) guarantees this is a 2D box. */
  const axes = $derived.by<[AxisSpec, AxisSpec]>(() => {
    const d = info.domain as { type?: string; axes?: AxisSpec[] };
    if (d?.type !== "box" || !Array.isArray(d.axes) || d.axes.length !== 2) {
      throw new Error(`probe ${info.name} has invalid 2D box geometry`);
    }
    return [d.axes[0], d.axes[1]];
  });

  /** Normalise an authoring coord to ``[0, 1]`` along its axis.  Used for
   *  both node placement and trajectory mapping. */
  function nx(v: number): number {
    const [a] = axes;
    const span = a.hi - a.lo;
    if (span <= 0) return 0;
    return Math.min(1, Math.max(0, (v - a.lo) / span));
  }
  function ny(v: number): number {
    const [, a] = axes;
    const span = a.hi - a.lo;
    if (span <= 0) return 0;
    // Y axis is inverted so "up" is the high end.
    return 1 - Math.min(1, Math.max(0, (v - a.lo) / span));
  }

  /** Pixel positions for the node markers + labels. */
  const nodes = $derived.by(() => {
    const out: { label: string; cx: number; cy: number }[] = [];
    const coords = info.node_coords ?? [];
    for (let i = 0; i < info.node_labels.length && i < coords.length; i++) {
      const row = coords[i];
      if (!Array.isArray(row) || row.length < 2) continue;
      if (!Number.isFinite(row[0]) || !Number.isFinite(row[1])) continue;
      out.push({
        label: info.node_labels[i],
        cx: nx(row[0]) * size,
        cy: ny(row[1]) * size,
      });
    }
    return out;
  });

  /** Polyline path of the per-token trajectory in pixel space. */
  const path = $derived.by(() => {
    if (!trajectory || trajectory.length === 0) return "";
    const segs: string[] = [];
    for (let i = 0; i < trajectory.length; i++) {
      const row = trajectory[i];
      if (!Array.isArray(row) || row.length < 2) continue;
      if (!Number.isFinite(row[0]) || !Number.isFinite(row[1])) continue;
      const x = nx(row[0]) * size;
      const y = ny(row[1]) * size;
      segs.push(`${segs.length === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`);
    }
    return segs.join(" ");
  });

  /** Pixel position of the settled-aggregate dot, when present. */
  const settledPx = $derived.by(() => {
    if (!settled || settled.length < 2) return null;
    if (!Number.isFinite(settled[0]) || !Number.isFinite(settled[1])) return null;
    return { cx: nx(settled[0]) * size, cy: ny(settled[1]) * size };
  });

  /** Pixel position of the live cursor — the trajectory's tail while
   *  streaming.  Suppressed once ``settled`` lands so the bold dot owns
   *  the "you are here" semantics. */
  const cursorPx = $derived.by(() => {
    if (settled) return null;
    if (!trajectory || trajectory.length === 0) return null;
    const row = trajectory[trajectory.length - 1];
    if (!Array.isArray(row) || row.length < 2) return null;
    if (!Number.isFinite(row[0]) || !Number.isFinite(row[1])) return null;
    return { cx: nx(row[0]) * size, cy: ny(row[1]) * size };
  });

  function fmt(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }
</script>

<figure class="map" style:--map-size="{size}px">
  <svg
    class="canvas"
    width={size}
    height={size}
    viewBox="0 0 {size} {size}"
    role="img"
    aria-label="Manifold {info.name} 2D layout"
  >
    <!-- Crosshair gridlines for the (0, 0) anchor, matching XYPad. -->
    <line
      class="grid"
      x1={nx(0) * size}
      x2={nx(0) * size}
      y1="0"
      y2={size}
    />
    <line
      class="grid"
      y1={ny(0) * size}
      y2={ny(0) * size}
      x1="0"
      x2={size}
    />

    <!-- Per-token trajectory polyline.  Renders behind the nodes so the
         labels stay readable. -->
    {#if path}
      <path d={path} class="trajectory" fill="none" />
    {/if}

    <!-- Node markers + labels. -->
    {#each nodes as n (n.label)}
      <g class="node">
        <circle cx={n.cx} cy={n.cy} r="3" />
        <text x={n.cx + 5} y={n.cy - 4}>{n.label}</text>
      </g>
    {/each}

    <!-- Live cursor while streaming. -->
    {#if cursorPx}
      <circle
        class="cursor"
        cx={cursorPx.cx}
        cy={cursorPx.cy}
        r="3"
      />
    {/if}

    <!-- Settled aggregate dot — bold, lands at the coords from the final
         ``done`` aggregate. -->
    {#if settledPx}
      <circle
        class="settled"
        cx={settledPx.cx}
        cy={settledPx.cy}
        r="5"
      />
    {/if}
  </svg>

  <figcaption class="axes">
    <span class="axis">{axes[0].name} {fmt(axes[0].lo)}…{fmt(axes[0].hi)}</span>
    <span class="axis">{axes[1].name} {fmt(axes[1].lo)}…{fmt(axes[1].hi)}</span>
  </figcaption>
</figure>

<style>
  .map {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    margin: 0;
  }
  .canvas {
    display: block;
    width: var(--map-size);
    height: var(--map-size);
    background: var(--glass-strong);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius);
  }
  .grid {
    stroke: var(--border);
    stroke-width: 1;
  }
  .trajectory {
    /* Green live/probe hue: the per-token path is a streaming quantity.
     * Translucent + thin so the path reads as historical context, not a
     * foreground feature. */
    stroke: var(--accent-green);
    stroke-width: 1.5;
    stroke-opacity: 0.55;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .node circle {
    fill: var(--fg-subtle);
    stroke: var(--bg-deep);
    stroke-width: 1;
  }
  .node text {
    fill: var(--fg-muted);
    font-size: var(--text-2xs);
    font-family: var(--font-ui);
  }
  .cursor {
    /* Live head of the green trajectory — follows the path's hue so the
     * moving cursor doesn't clash with the line it caps. */
    fill: var(--accent-green);
    fill-opacity: 0.7;
    stroke: var(--bg-deep);
    stroke-width: 1;
  }
  .settled {
    fill: var(--accent-green);
    stroke: var(--accent-light);
    stroke-width: 1;
  }
  .axes {
    display: flex;
    justify-content: space-between;
    gap: var(--space-3);
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-family: var(--font-mono);
  }
</style>
