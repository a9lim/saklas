// Hand-rolled canvas renderer for the probe-inspector geometry plot.  Zero
// 3D-library dependency, matching the dashboard's custom-chart aesthetic
// (Bar.svelte / ManifoldMiniMap.svelte).  Branches on the probe's SUBSPACE
// DIMENSION (rank), all coords in the whitened (Mahalanobis) frame:
//
//   rank 1   -> a horizontal line: poles + neutral(0) + sliding live dot
//   rank 2   -> a 2D scatter of node centroids (+ optional curve overlay)
//   rank 3+  -> a drag-orbit 3D scatter on the top-3 PCs (+ curve/surface)
//
// Every case overlays the fading trajectory trail (oldest faint -> newest
// bright) and the current hidden-state point.

import type { ProbeLayerGeometry } from "../types";

export interface OrbitState {
  /** Azimuth (radians) — horizontal drag. */
  az: number;
  /** Elevation (radians) — vertical drag, clamped. */
  el: number;
}

export interface GeometryRenderInput {
  /** The selected layer's geometry. */
  geom: ProbeLayerGeometry;
  /** Node labels, aligned with ``geom.node_white`` rows. */
  nodeLabels: string[];
  /** Current hidden-state point in this layer's whitened coords (R,), or null. */
  live: number[] | null;
  /** Trail of past whitened points (R,), oldest -> newest. */
  trail: number[][];
  /** Orbit angles (used for rank >= 3 only). */
  orbit: OrbitState;
}

interface Palette {
  fg: string;
  fgDim: string;
  muted: string;
  border: string;
  accent: string;
  purple: string;
  bg: string;
}

function readPalette(el: HTMLElement): Palette {
  const cs = getComputedStyle(el);
  const v = (name: string, fallback: string): string => {
    const got = cs.getPropertyValue(name).trim();
    return got || fallback;
  };
  return {
    fg: v("--fg-strong", "#e6e6e6"),
    fgDim: v("--fg-dim", "#9aa0a6"),
    muted: v("--fg-muted", "#6b7280"),
    border: v("--border", "#2a2a2a"),
    accent: v("--accent", "#e6e6e6"),
    purple: v("--accent-purple", "#b58cf0"),
    bg: v("--bg-deep", "#0d0d0d"),
  };
}

const PAD = 28;

/** Reset transform, size for devicePixelRatio, clear. */
function prep(
  canvas: HTMLCanvasElement,
): { ctx: CanvasRenderingContext2D; w: number; h: number } | null {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function boundsOf(pts: Array<[number, number]>): Bounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!Number.isFinite(minX)) {
    return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  }
  // Pad a touch and guard against a degenerate (zero-span) axis.
  const sx = maxX - minX || 1;
  const sy = maxY - minY || 1;
  return {
    minX: minX - sx * 0.08,
    maxX: maxX + sx * 0.08,
    minY: minY - sy * 0.08,
    maxY: maxY + sy * 0.08,
  };
}

/** Map a data point into the canvas box, preserving aspect (square units). */
function projector(b: Bounds, w: number, h: number): (x: number, y: number) => [number, number] {
  const iw = w - 2 * PAD;
  const ih = h - 2 * PAD;
  const span = Math.max(b.maxX - b.minX, b.maxY - b.minY) || 1;
  const scale = Math.min(iw, ih) / span;
  const cx = (b.minX + b.maxX) / 2;
  const cy = (b.minY + b.maxY) / 2;
  return (x, y) => [
    w / 2 + (x - cx) * scale,
    h / 2 - (y - cy) * scale, // y up
  ];
}

function dot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  fill: string,
  alpha = 1,
): void {
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.globalAlpha = 1;
}

function label(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  color: string,
): void {
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = color;
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(text, x + 5, y - 4);
  ctx.globalAlpha = 1;
}

/** Polyline with a per-vertex opacity ramp (oldest faint -> newest bright). */
function fadingTrail(
  ctx: CanvasRenderingContext2D,
  screen: Array<[number, number]>,
  color: string,
): void {
  if (screen.length < 2) return;
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = color;
  for (let i = 1; i < screen.length; i++) {
    const a = 0.12 + 0.78 * (i / (screen.length - 1));
    ctx.globalAlpha = a;
    ctx.beginPath();
    ctx.moveTo(screen[i - 1][0], screen[i - 1][1]);
    ctx.lineTo(screen[i][0], screen[i][1]);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

// ----------------------------------------------------------- rank 1 --

function drawRank1(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  input: GeometryRenderInput,
  pal: Palette,
): void {
  const { geom, live, trail } = input;
  const xs: Array<[number, number]> = [];
  for (const nc of geom.node_white) xs.push([nc[0] ?? 0, 0]);
  xs.push([geom.neutral_white[0] ?? 0, 0]);
  if (live) xs.push([live[0] ?? 0, 0]);
  for (const t of trail) xs.push([t[0] ?? 0, 0]);
  const b = boundsOf(xs);
  const y0 = h / 2;
  const px = (x: number): number => {
    const iw = w - 2 * PAD;
    const span = b.maxX - b.minX || 1;
    return PAD + ((x - b.minX) / span) * iw;
  };

  // axis line
  ctx.strokeStyle = pal.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD, y0);
  ctx.lineTo(w - PAD, y0);
  ctx.stroke();

  // neutral tick
  const nx = px(geom.neutral_white[0] ?? 0);
  ctx.strokeStyle = pal.muted;
  ctx.beginPath();
  ctx.moveTo(nx, y0 - 8);
  ctx.lineTo(nx, y0 + 8);
  ctx.stroke();
  label(ctx, nx, y0 - 6, "neutral", pal.muted);

  // poles / nodes
  geom.node_white.forEach((nc, i) => {
    const x = px(nc[0] ?? 0);
    dot(ctx, x, y0, 4, pal.fgDim);
    label(ctx, x, y0 + 18, input.nodeLabels[i] ?? "", pal.fgDim);
  });

  // trail (ticks along the line) + live dot
  trail.forEach((t, i) => {
    const a = 0.1 + 0.7 * (i / Math.max(1, trail.length - 1));
    dot(ctx, px(t[0] ?? 0), y0, 2.5, pal.purple, a);
  });
  if (live) dot(ctx, px(live[0] ?? 0), y0, 5, pal.accent);
}

// ----------------------------------------------------------- rank 2 --

function drawRank2(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  input: GeometryRenderInput,
  pal: Palette,
  nodeLabel: (i: number) => string,
): void {
  const { geom, live, trail } = input;
  const pts: Array<[number, number]> = [];
  for (const nc of geom.node_white) pts.push([nc[0] ?? 0, nc[1] ?? 0]);
  pts.push([geom.neutral_white[0] ?? 0, geom.neutral_white[1] ?? 0]);
  if (geom.overlay?.kind === "curve") {
    for (const p of geom.overlay.points) pts.push([p[0] ?? 0, p[1] ?? 0]);
  }
  if (live) pts.push([live[0] ?? 0, live[1] ?? 0]);
  for (const t of trail) pts.push([t[0] ?? 0, t[1] ?? 0]);
  const b = boundsOf(pts);
  const proj = projector(b, w, h);

  // overlay curve
  if (geom.overlay?.kind === "curve" && geom.overlay.points.length > 1) {
    ctx.strokeStyle = pal.purple;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    geom.overlay.points.forEach((p, i) => {
      const [sx, sy] = proj(p[0] ?? 0, p[1] ?? 0);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // neutral
  {
    const [sx, sy] = proj(geom.neutral_white[0] ?? 0, geom.neutral_white[1] ?? 0);
    ctx.strokeStyle = pal.muted;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, Math.PI * 2);
    ctx.stroke();
  }

  // nodes
  geom.node_white.forEach((nc, i) => {
    const [sx, sy] = proj(nc[0] ?? 0, nc[1] ?? 0);
    dot(ctx, sx, sy, 3.5, pal.fgDim);
    label(ctx, sx, sy, nodeLabel(i), pal.fgDim);
  });

  // trail + live
  const trailScreen = trail.map((t) => proj(t[0] ?? 0, t[1] ?? 0));
  fadingTrail(ctx, trailScreen, pal.purple);
  if (live) {
    const [sx, sy] = proj(live[0] ?? 0, live[1] ?? 0);
    dot(ctx, sx, sy, 5, pal.accent);
  }
}

// ----------------------------------------------------------- rank 3+ --

/** Project an (R,) whitened point onto 3 PCs then orbit-rotate to a 3-vector. */
function to3(p: number[], rot: number[][], az: number, el: number): [number, number, number] {
  // PCA: xyz0[k] = sum_i p[i] * rot[i][k]
  let x = 0;
  let y = 0;
  let z = 0;
  const R = Math.min(p.length, rot.length);
  for (let i = 0; i < R; i++) {
    const r = rot[i];
    const pi = p[i] ?? 0;
    x += pi * (r[0] ?? 0);
    y += pi * (r[1] ?? 0);
    z += pi * (r[2] ?? 0);
  }
  // azimuth about y, then elevation about x
  const ca = Math.cos(az);
  const sa = Math.sin(az);
  const x1 = ca * x + sa * z;
  const z1 = -sa * x + ca * z;
  const ce = Math.cos(el);
  const se = Math.sin(el);
  const y2 = ce * y - se * z1;
  const z2 = se * y + ce * z1;
  return [x1, y2, z2];
}

function drawRank3(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  input: GeometryRenderInput,
  pal: Palette,
  nodeLabel: (i: number) => string,
): void {
  const { geom, live, trail, orbit } = input;
  const rot = geom.pca_rotation;
  if (!rot) return;
  const az = orbit.az;
  const el = orbit.el;

  // Collect every 3-projected point for a shared bounds + depth scale.
  const node3 = geom.node_white.map((p) => to3(p, rot, az, el));
  const neutral3 = to3(geom.neutral_white, rot, az, el);
  const overlay3 = (geom.overlay?.points ?? []).map((p) => to3(p, rot, az, el));
  const trail3 = trail.map((p) => to3(p, rot, az, el));
  const live3 = live ? to3(live, rot, az, el) : null;

  const xy: Array<[number, number]> = [];
  for (const p of node3) xy.push([p[0], p[1]]);
  xy.push([neutral3[0], neutral3[1]]);
  for (const p of overlay3) xy.push([p[0], p[1]]);
  for (const p of trail3) xy.push([p[0], p[1]]);
  if (live3) xy.push([live3[0], live3[1]]);
  const b = boundsOf(xy);
  const proj = projector(b, w, h);

  // depth normalization for size/alpha cueing
  let zmin = Infinity;
  let zmax = -Infinity;
  for (const p of node3) {
    if (p[2] < zmin) zmin = p[2];
    if (p[2] > zmax) zmax = p[2];
  }
  const zspan = zmax - zmin || 1;
  const depth01 = (z: number): number => (z - zmin) / zspan;

  // overlay first (behind), depth-agnostic faint wireframe
  if (geom.overlay && overlay3.length > 1) {
    ctx.strokeStyle = pal.purple;
    ctx.lineWidth = 1;
    if (geom.overlay.kind === "curve") {
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      overlay3.forEach((p, i) => {
        const [sx, sy] = proj(p[0], p[1]);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    } else if (geom.overlay.kind === "surface" && geom.overlay.grid_shape) {
      const [nu, nv] = geom.overlay.grid_shape;
      ctx.globalAlpha = 0.28;
      // row lines (constant u)
      for (let u = 0; u < nu; u++) {
        ctx.beginPath();
        for (let vv = 0; vv < nv; vv++) {
          const p = overlay3[u * nv + vv];
          if (!p) continue;
          const [sx, sy] = proj(p[0], p[1]);
          if (vv === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
      }
      // column lines (constant v)
      for (let vv = 0; vv < nv; vv++) {
        ctx.beginPath();
        for (let u = 0; u < nu; u++) {
          const p = overlay3[u * nv + vv];
          if (!p) continue;
          const [sx, sy] = proj(p[0], p[1]);
          if (u === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }

  // neutral
  {
    const [sx, sy] = proj(neutral3[0], neutral3[1]);
    ctx.strokeStyle = pal.muted;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, Math.PI * 2);
    ctx.stroke();
  }

  // nodes — painter's algorithm (far first), depth-cued size + alpha
  const order = node3
    .map((p, i) => ({ i, z: p[2] }))
    .sort((a, c) => a.z - c.z);
  for (const { i } of order) {
    const p = node3[i];
    const [sx, sy] = proj(p[0], p[1]);
    const d = depth01(p[2]);
    dot(ctx, sx, sy, 2.5 + 2.5 * d, pal.fgDim, 0.4 + 0.6 * d);
    if (d > 0.55) label(ctx, sx, sy, nodeLabel(i), pal.fgDim);
  }

  // trail + live (always on top)
  const trailScreen = trail3.map((p) => proj(p[0], p[1]));
  fadingTrail(ctx, trailScreen, pal.purple);
  if (live3) {
    const [sx, sy] = proj(live3[0], live3[1]);
    dot(ctx, sx, sy, 5, pal.accent);
  }
}

/** Render one frame.  Returns the rank actually drawn (for cursor hints). */
export function renderProbeGeometry(
  canvas: HTMLCanvasElement,
  input: GeometryRenderInput,
): void {
  const prepared = prep(canvas);
  if (!prepared) return;
  const { ctx, w, h } = prepared;
  const pal = readPalette(canvas);
  const nodeLabel = (i: number): string => input.nodeLabels[i] ?? "";
  const rank = input.geom.rank;
  if (rank <= 1) {
    drawRank1(ctx, w, h, input, pal);
  } else if (rank === 2) {
    drawRank2(ctx, w, h, input, pal, nodeLabel);
  } else {
    drawRank3(ctx, w, h, input, pal, nodeLabel);
  }
}
