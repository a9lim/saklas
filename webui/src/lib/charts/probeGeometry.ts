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
//
// **Camera (rank >= 3).**  The scale is derived ONCE from the static framing
// set (nodes + neutral + overlay) and is rotation-invariant — the orbit is a
// rigid rotation about the neutral anchor (the whitened origin), so neutral
// sits at the plot center and the cloud's radius never changes as you drag.
// This is what fixes the old "zooms while I rotate" bug: the
// previous code rescaled every frame from the 2D silhouette of the rotated
// cloud (smaller edge-on, larger broadside), and it also let a moving live
// point rescale the whole scatter.  Zoom is now an explicit user control
// (``orbit.zoom``, driven by the scroll wheel), never an artifact of rotation.

import type { ProbeLayerGeometry } from "../types";

/** Unit quaternion ``[w, x, y, z]``. */
export type Quat = [number, number, number, number];

export interface OrbitState {
  /** Accumulated orientation as a unit quaternion mapping world (PCA) coords
   *  into view coords.  A drag composes an incremental screen-axis rotation
   *  onto it (trackball), so there are no Euler angles — hence no pole
   *  singularity / gimbal lock, and the whole orientation sphere is reachable
   *  without a clamp.  Every increment is a rigid rotation, so the
   *  rotation-invariant base scale (below) is preserved. */
  q: Quat;
  /** Zoom multiplier on the fixed base scale (scroll wheel); 1 = fit. */
  zoom: number;
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
  /** Orbit angles + zoom (used for rank >= 3 only). */
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
  /** Node centroids — the probe's FAMILY hue (subspace white / manifold
   *  violet), supplied by the host via ``--geom-node`` so the plot obeys
   *  the hue ontology instead of hardcoding a color. */
  node: string;
  /** Neutral anchor — hollow grey ring (achromatic: neutral is the
   *  absence of concept, so it carries no hue). */
  neutral: string;
  /** Live hidden-state point halo + trail head. */
  live: string;
  /** Live point core. */
  light: string;
}

function readPalette(el: HTMLElement): Palette {
  const cs = getComputedStyle(el);
  const v = (name: string, fallback: string): string => {
    const got = cs.getPropertyValue(name).trim();
    return got || fallback;
  };
  return {
    fg: v("--fg-strong", "#d3d7e4"),
    fgDim: v("--fg-dim", "#a3a8bd"),
    muted: v("--fg-muted", "#6b7088"),
    border: v("--border", "rgba(233, 236, 248, 0.09)"),
    accent: v("--accent", "#edeff7"),
    purple: v("--pillar-manifold", "#a78bfa"),
    bg: v("--bg-deep", "#05060b"),
    node: v("--geom-node", v("--pillar-manifold", "#a78bfa")),
    neutral: v("--geom-neutral", v("--fg-subtle", "#8a90a6")),
    live: v("--live", "#34d399"),
    light: v("--accent-light", "#ffffff"),
  };
}

/** Canvas font strings — Recursive Mono to match the data voice. */
const FONT_LABEL = '10px "Recursive Mono", ui-monospace, monospace';
const FONT_AXIS = '9px "Recursive Mono", ui-monospace, monospace';

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

/** The live hidden-state point: white core inside a colored halo ring, so it
 *  reads clearly over both the node accent and the dark background.  Carries
 *  a soft green bloom — glow is reserved for what is alive right now, and
 *  the current hidden state is exactly that. */
function liveDot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  pal: Palette,
): void {
  ctx.globalAlpha = 0.95;
  ctx.shadowColor = pal.live;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(x, y, 6.5, 0, Math.PI * 2);
  ctx.strokeStyle = pal.live;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.arc(x, y, 3.2, 0, Math.PI * 2);
  ctx.fillStyle = pal.light;
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
  ctx.font = FONT_LABEL;
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
  ctx.strokeStyle = pal.neutral;
  ctx.beginPath();
  ctx.moveTo(nx, y0 - 8);
  ctx.lineTo(nx, y0 + 8);
  ctx.stroke();
  label(ctx, nx, y0 - 6, "neutral", pal.neutral);

  // poles / nodes
  geom.node_white.forEach((nc, i) => {
    const x = px(nc[0] ?? 0);
    dot(ctx, x, y0, 4, pal.node);
    label(ctx, x, y0 + 18, input.nodeLabels[i] ?? "", pal.fgDim);
  });

  // trail (ticks along the line) + live dot
  trail.forEach((t, i) => {
    const a = 0.1 + 0.7 * (i / Math.max(1, trail.length - 1));
    dot(ctx, px(t[0] ?? 0), y0, 2.5, pal.live, a);
  });
  if (live) liveDot(ctx, px(live[0] ?? 0), y0, pal);
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
  // Frame the view from the STATIC geometry (nodes + neutral + overlay) only,
  // so a moving live point / growing trail can't rescale the scatter.
  const framing: Array<[number, number]> = [];
  for (const nc of geom.node_white) framing.push([nc[0] ?? 0, nc[1] ?? 0]);
  framing.push([geom.neutral_white[0] ?? 0, geom.neutral_white[1] ?? 0]);
  if (geom.overlay?.kind === "curve") {
    for (const p of geom.overlay.points) framing.push([p[0] ?? 0, p[1] ?? 0]);
  }
  const b = boundsOf(framing);
  const proj = projector(b, w, h);

  // PC1 (horizontal) / PC2 (vertical) reference axes through the whitened
  // origin, spanning the inner plot box.
  {
    const [ox, oy] = proj(0, 0);
    ctx.strokeStyle = pal.border;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.35;
    ctx.beginPath();
    ctx.moveTo(PAD, oy);
    ctx.lineTo(w - PAD, oy);
    ctx.moveTo(ox, PAD);
    ctx.lineTo(ox, h - PAD);
    ctx.stroke();
    ctx.globalAlpha = 0.7;
    ctx.fillStyle = pal.muted;
    ctx.font = FONT_AXIS;
    ctx.fillText("PC1", w - PAD - 22, oy - 4);
    ctx.fillText("PC2", ox + 4, PAD + 10);
    ctx.globalAlpha = 1;
  }

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
    ctx.strokeStyle = pal.neutral;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, Math.PI * 2);
    ctx.stroke();
  }

  // nodes
  geom.node_white.forEach((nc, i) => {
    const [sx, sy] = proj(nc[0] ?? 0, nc[1] ?? 0);
    dot(ctx, sx, sy, 3.5, pal.node);
    label(ctx, sx, sy, nodeLabel(i), pal.fgDim);
  });

  // trail + live
  const trailScreen = trail.map((t) => proj(t[0] ?? 0, t[1] ?? 0));
  fadingTrail(ctx, trailScreen, pal.live);
  if (live) {
    const [sx, sy] = proj(live[0] ?? 0, live[1] ?? 0);
    liveDot(ctx, sx, sy, pal);
  }
}

// ----------------------------------------------------------- rank 3+ --

type Vec3 = [number, number, number];

/** Project an (R,) whitened point onto the top-3 PCs (static, orbit-free). */
function projectPca3(p: number[], rot: number[][]): Vec3 {
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
  return [x, y, z];
}

// --- Orientation quaternion (gimbal-lock-free orbit) ---------------------
// The orbit accumulates a unit quaternion instead of (az, el) Euler angles, so
// dragging never approaches a pole singularity and every orientation is
// reachable without clamping.  Each step is a rigid rotation, so the
// rotation-invariant base scale stays valid.

/** Unit quaternion from a unit axis + angle (radians). */
function quatFromAxisAngle(axis: Vec3, angle: number): Quat {
  const half = angle / 2;
  const s = Math.sin(half);
  return [Math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s];
}

/** Hamilton product ``a ∘ b`` (b applied first, then a). */
function quatMul(a: Quat, b: Quat): Quat {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

/** Renormalize, killing drift accumulated across repeated composition. */
function quatNormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** Rotate a vector by a unit quaternion (``v' = q v q⁻¹``, branchless). */
function quatRotate(q: Quat, v: Vec3): Vec3 {
  const [w, x, y, z] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

/** Compose an incremental drag rotation onto an orientation.  ``dx`` spins
 *  about the view-up axis, ``dy`` tilts about the view-right axis — both
 *  screen-relative (trackball), pre-multiplied so they act in view space.
 *  ``sensitivity`` is radians per pixel. */
export function orbitDrag(
  q: Quat,
  dx: number,
  dy: number,
  sensitivity = 0.01,
): Quat {
  const yaw = quatFromAxisAngle([0, 1, 0], dx * sensitivity);
  const pitch = quatFromAxisAngle([1, 0, 0], dy * sensitivity);
  return quatNormalize(quatMul(quatMul(pitch, yaw), q));
}

/** Default orbit orientation — the old three-quarter view (``az=0.6, el=0.5``)
 *  expressed as the equivalent quaternion (elevation-about-x ∘ azimuth-about-y),
 *  so the initial framing is unchanged. */
export const DEFAULT_ORBIT_QUAT: Quat = quatNormalize(
  quatMul(
    quatFromAxisAngle([1, 0, 0], 0.5),
    quatFromAxisAngle([0, 1, 0], 0.6),
  ),
);

interface Projected {
  s: [number, number];
  z: number;
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
  const q = orbit.q;
  const zoom = orbit.zoom || 1;

  // Static PCA-3D coords (independent of the orbit) for the framing set.
  const nodePca = geom.node_white.map((p) => projectPca3(p, rot));
  const neutralPca = projectPca3(geom.neutral_white, rot);
  const overlayPca = (geom.overlay?.points ?? []).map((p) => projectPca3(p, rot));

  // Pivot = the neutral anchor (the whitened origin), so neutral sits at the
  // plot center and the cloud shows its real displacement from neutral.  Still
  // a fixed pivot, so the rotation-invariant base scale below is preserved (the
  // backend's ``node_coords`` layout is neutral-anchored to match — both
  // surfaces share neutral as their origin).
  const C: Vec3 = neutralPca;

  // Rotation-invariant radius over the static framing set → fixed base scale.
  let rho = 0;
  for (const p of [...nodePca, neutralPca, ...overlayPca]) {
    const d = Math.hypot(p[0] - C[0], p[1] - C[1], p[2] - C[2]);
    if (d > rho) rho = d;
  }
  rho = rho || 1;
  const iw = w - 2 * PAD;
  const ih = h - 2 * PAD;
  const scale = (Math.min(iw, ih) / 2 / rho) * zoom;
  const ox = w / 2;
  const oy = h / 2;

  const project = (pca: Vec3): Projected => {
    const r = quatRotate(q, [pca[0] - C[0], pca[1] - C[1], pca[2] - C[2]]);
    return { s: [ox + r[0] * scale, oy - r[1] * scale], z: r[2] };
  };

  const node3 = nodePca.map(project);
  const neutral3 = project(neutralPca);
  const overlay3 = overlayPca.map(project);
  const trail3 = trail.map((p) => project(projectPca3(p, rot)));
  const live3 = live ? project(projectPca3(live, rot)) : null;

  // depth normalization for size/alpha cueing (from the node cloud)
  let zmin = Infinity;
  let zmax = -Infinity;
  for (const p of node3) {
    if (p.z < zmin) zmin = p.z;
    if (p.z > zmax) zmax = p.z;
  }
  const zspan = zmax - zmin || 1;
  const depth01 = (z: number): number => (z - zmin) / zspan;

  // PC1/PC2/PC3 reference triad through neutral (behind everything),
  // rotating rigidly with the cloud so orientation is always readable.
  {
    const axLen = rho * 1.1;
    const triad: Array<{ dir: Vec3; name: string }> = [
      { dir: [1, 0, 0], name: "PC1" },
      { dir: [0, 1, 0], name: "PC2" },
      { dir: [0, 0, 1], name: "PC3" },
    ];
    ctx.strokeStyle = pal.border;
    ctx.lineWidth = 1;
    for (const { dir, name } of triad) {
      const pos = project([
        C[0] + dir[0] * axLen,
        C[1] + dir[1] * axLen,
        C[2] + dir[2] * axLen,
      ]);
      const neg = project([
        C[0] - dir[0] * axLen,
        C[1] - dir[1] * axLen,
        C[2] - dir[2] * axLen,
      ]);
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      ctx.moveTo(neg.s[0], neg.s[1]);
      ctx.lineTo(pos.s[0], pos.s[1]);
      ctx.stroke();
      ctx.globalAlpha = 0.7;
      ctx.fillStyle = pal.muted;
      ctx.font = FONT_AXIS;
      ctx.fillText(name, pos.s[0] + 3, pos.s[1] - 2);
    }
    ctx.globalAlpha = 1;
  }

  // overlay first (behind), depth-agnostic faint wireframe
  if (geom.overlay && overlay3.length > 1) {
    ctx.strokeStyle = pal.purple;
    ctx.lineWidth = 1;
    if (geom.overlay.kind === "curve") {
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      overlay3.forEach((p, i) => {
        if (i === 0) ctx.moveTo(p.s[0], p.s[1]);
        else ctx.lineTo(p.s[0], p.s[1]);
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
          if (vv === 0) ctx.moveTo(p.s[0], p.s[1]);
          else ctx.lineTo(p.s[0], p.s[1]);
        }
        ctx.stroke();
      }
      // column lines (constant v)
      for (let vv = 0; vv < nv; vv++) {
        ctx.beginPath();
        for (let u = 0; u < nu; u++) {
          const p = overlay3[u * nv + vv];
          if (!p) continue;
          if (u === 0) ctx.moveTo(p.s[0], p.s[1]);
          else ctx.lineTo(p.s[0], p.s[1]);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }

  // neutral
  {
    ctx.strokeStyle = pal.neutral;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(neutral3.s[0], neutral3.s[1], 4, 0, Math.PI * 2);
    ctx.stroke();
  }

  // nodes — painter's algorithm (far first), depth-cued size + alpha
  const order = node3
    .map((p, i) => ({ i, z: p.z }))
    .sort((a, c) => a.z - c.z);
  for (const { i } of order) {
    const p = node3[i];
    const d = depth01(p.z);
    dot(ctx, p.s[0], p.s[1], 2.5 + 2.5 * d, pal.node, 0.4 + 0.6 * d);
    if (d > 0.55) label(ctx, p.s[0], p.s[1], nodeLabel(i), pal.fgDim);
  }

  // trail + live (always on top)
  const trailScreen = trail3.map((p) => p.s);
  fadingTrail(ctx, trailScreen, pal.live);
  if (live3) liveDot(ctx, live3.s[0], live3.s[1], pal);
}

/** Render one frame. */
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
