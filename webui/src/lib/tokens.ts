// Per-token highlighting helpers.
//
// Mirrors saklas/tui/chat_panel.py:_build_highlight_markup — same
// saturation knob, same RGB mapping, same compose-with-zero behavior.
// The TUI builds Rich markup; this module emits CSS color strings the
// Chat panel attaches as inline ``background-color`` styles per token.
//
// The two-stripe variant is a web-only affordance — the TUI's chord-key
// model can only show one probe at a time; the dashboard can render two
// probes simultaneously by splitting each token's background vertically.

/** Saturation cutoff — score / HIGHLIGHT_SAT clamps to [-1, 1] before
 * mapping to RGB.  Matches the TUI constant exactly so highlight tints
 * align across surfaces. */
export const HIGHLIGHT_SAT = 0.5;

/** Sentinel ``highlightState.target`` value that selects the inline
 *  surprise mode (logit-pass).  Picked to be unmistakably distinct from
 *  any real probe name (probes are slugged ``[a-z0-9._-]``); ``__``-bracketed
 *  reserves the namespace without colliding.  Imported by both
 *  ``Chat.svelte`` and any future TUI parity pass. */
export const SURPRISE_TARGET = "__surprise__";

/** Split a highlight target into its base probe name and coordinate axis.
 *  ``"personas[3]"`` → ``{base: "personas", axis: 3}``; a bare name → axis 0.
 *  Mirrors the ``<probe>[<i>]`` gate-channel grammar ``Monitor.flat_scalars``
 *  emits, so a per-axis highlight target lines up with the steering gate that
 *  reads the same coordinate. */
export function parseProbeTarget(target: string): { base: string; axis: number } {
  const m = /^(.+)\[(\d+)\]$/.exec(target);
  if (m) return { base: m[1], axis: Number(m[2]) };
  return { base: target, axis: 0 };
}

/** Look up a token's score for a (possibly axis-indexed) probe target.
 *
 *  Axis ``i`` reads the live per-token domain coordinates captured under
 *  ``coordsByProbe`` (the full rank-R reading off the ``probe_readings`` wire
 *  channel); axis 0 falls back to the collapsed ``probes`` row, which is the
 *  channel the end-of-gen ``per_token_probes`` pass and a tree reload restore.
 *  Returns ``undefined`` when neither source carries the target, so the caller
 *  can fall through to its own default (transparent tint). */
export function probeScoreForTarget(
  t: {
    probes?: Record<string, number>;
    coordsByProbe?: Record<string, number[]>;
  },
  target: string,
): number | undefined {
  const { base, axis } = parseProbeTarget(target);
  const coords = t.coordsByProbe?.[base];
  if (coords && axis < coords.length) return coords[axis];
  if (t.probes && target in t.probes) return t.probes[target];
  return undefined;
}

/** Map a chosen-token logprob to a positive-scale score suitable for
 *  ``scoreToRgb``.
 *
 *  Logic:
 *    tint = 1 - exp(logprob) = 1 - probability   # [0, 1)
 *
 *  Then scaled by ``HIGHLIGHT_SAT`` so the full surprise range
 *  ``[0, 1]`` lands in ``scoreToRgb``'s positive-saturation band and
 *  reuses the existing diverging probe color scale (positive half only).
 *  Returns ``undefined`` when logprob is null / not finite. */
export function surpriseScore(
  logprob: number | null | undefined,
): number | undefined {
  if (logprob == null || !Number.isFinite(logprob)) return undefined;
  // ``logprob`` is the log of a probability so it's always ≤ 0 —
  // ``exp(logprob)`` is in (0, 1] and ``tint`` lands in [0, 1).
  const tint = 1 - Math.exp(logprob);
  return tint * HIGHLIGHT_SAT;
}

/** Which constant-hue ramp a tint reads in.  ``signed`` is the probe
 *  ramp (green +pole ↔ red −pole); ``surprise`` is the logit-space blue
 *  — surprise is a vocabulary-distribution quantity, so it shares the
 *  J-lens hue family and is unmistakably distinct from any probe
 *  reading (pre-v2 it reused the positive green band). */
export type TintHue = "signed" | "surprise";

/* Ramp poles (tokens.css: --highlight-pos / --highlight-neg /
 * --highlight-surprise) as rgb triplets, and the alpha ceiling.  v2
 * tints are constant-hue ALPHA ramps: tint strength = opacity, hue =
 * meaning, so text contrast stays put and the poles are OKLCH-matched
 * in perceived lightness (the old ramp swept luminance through opaque
 * rgb(0,255·t,0), which wobbled legibility with the score). */
const TINT_POS = "52, 211, 153"; /* #34d399 */
const TINT_NEG = "229, 84, 79"; /* #e5544f */
const TINT_SURPRISE = "107, 166, 248"; /* #6ba6f8 */
const TINT_MAX_ALPHA = 0.62;

/** Map a probe score to a CSS rgba() background (stronger score =
 * more opaque tint of a constant hue).  Returns ``"transparent"`` when
 * score is effectively zero or null/undefined.
 *
 * ``scale`` is the value at which the ramp reaches full strength:
 * ``t = score / scale``, clamped to [-1, 1].  It defaults to
 * ``HIGHLIGHT_SAT`` so the fixed-[-1,1] callers (the correlation matrix,
 * the cross-layer cosine grid) are unchanged; the probe surfaces pass a
 * per-probe node-coordinate extent (see ``nodeCoordExtent``) so a fan
 * whose coords run to ±tens isn't pinned fully saturated from the first
 * token.  A degenerate (0 / non-finite) scale falls back to the fixed
 * cutoff rather than dividing by zero.
 *
 * ``hue`` picks the ramp: ``signed`` (default — green/red poles) or
 * ``surprise`` (unsigned logit-space blue; |t| drives opacity). */
export function scoreToRgb(
  score: number | null | undefined,
  scale: number = HIGHLIGHT_SAT,
  hue: TintHue = "signed",
): string {
  if (score == null || !Number.isFinite(score)) return "transparent";
  const s = Number.isFinite(scale) && scale > 1e-6 ? scale : HIGHLIGHT_SAT;
  const t = Math.max(-1, Math.min(1, score / s));
  if (t === 0) return "transparent";
  const a = (Math.abs(t) * TINT_MAX_ALPHA).toFixed(3);
  if (hue === "surprise") return `rgba(${TINT_SURPRISE}, ${a})`;
  return t > 0 ? `rgba(${TINT_POS}, ${a})` : `rgba(${TINT_NEG}, ${a})`;
}

/** Per-probe color/bar scale: the largest ``|node coordinate|`` on the
 * given intrinsic axis — how far the most extreme fitted node sits from
 * the (neutral-centered) origin, in the same domain-frame units the
 * monitor reports ``coords`` and ``coords_per_layer`` in.  Bars, layer
 * cells, and token highlighting all normalize by this so "full" means "as
 * far along the axis as the most extreme node" rather than the old fixed
 * ±1 / ±HIGHLIGHT_SAT cutoff.
 *
 * Returns 1 when the manifold carries no node coords (an unfitted discover
 * fan, or a rank-1 fold whose coordinate is already pole-normalized to
 * ~1.0 — the fixed unit scale is correct there). */
export function nodeCoordExtent(
  nodeCoords: number[][] | null | undefined,
  axis = 0,
): number {
  if (!nodeCoords || nodeCoords.length === 0) return 1;
  let m = 0;
  for (const row of nodeCoords) {
    const v = row?.[axis];
    if (typeof v === "number" && Number.isFinite(v)) {
      const a = Math.abs(v);
      if (a > m) m = a;
    }
  }
  return m > 1e-6 ? m : 1;
}

/** Composite background style for compare-two mode.  Stripes the token
 * with probe-A on top and probe-B on bottom via a CSS linear gradient
 * with a hard color stop at 50%.  Either side can be transparent — the
 * gradient still works because CSS overlays transparent stops cleanly. */
export function twoStripeStyle(
  scoreA: number | null | undefined,
  scoreB: number | null | undefined,
  scaleA: number = HIGHLIGHT_SAT,
  scaleB: number = HIGHLIGHT_SAT,
  hueA: TintHue = "signed",
  hueB: TintHue = "signed",
): { backgroundImage: string } {
  const top = scoreToRgb(scoreA, scaleA, hueA);
  const bot = scoreToRgb(scoreB, scaleB, hueB);
  return {
    backgroundImage: `linear-gradient(to bottom, ${top} 0%, ${top} 50%, ${bot} 50%, ${bot} 100%)`,
  };
}

/** Smooth-blend variant of ``twoStripeStyle`` — useful when the user
 * wants a continuous gradient between the two probes rather than a hard
 * stripe boundary.  Drop-in replacement; same call site. */
export function twoBlendStyle(
  scoreA: number | null | undefined,
  scoreB: number | null | undefined,
  scaleA: number = HIGHLIGHT_SAT,
  scaleB: number = HIGHLIGHT_SAT,
  hueA: TintHue = "signed",
  hueB: TintHue = "signed",
): { backgroundImage: string } {
  const top = scoreToRgb(scoreA, scaleA, hueA);
  const bot = scoreToRgb(scoreB, scaleB, hueB);
  return {
    backgroundImage: `linear-gradient(to bottom, ${top}, ${bot})`,
  };
}

/** Build the hover-tooltip text for a token.  ``scores`` is the full
 * per-probe score row for one token — sorted by absolute value so the
 * strongest signals lead the tooltip.  Strings render as
 * ``probe_name +0.42`` one per line. */
export function formatScoreTooltip(scores: Record<string, number>): string {
  const entries = Object.entries(scores).filter(
    ([, v]) => Number.isFinite(v),
  );
  entries.sort(([, a], [, b]) => Math.abs(b) - Math.abs(a));
  return entries
    .map(([name, v]) => {
      const sign = v >= 0 ? "+" : "";
      return `${name} ${sign}${v.toFixed(3)}`;
    })
    .join("\n");
}

/** Convenience: combine the three above into a single per-token style
 * spec the renderer can spread onto an element.  ``probeB`` null means
 * single-probe mode (use scoreToRgb backgroundColor); when both probes
 * are non-null use the two-stripe gradient. */
export function tokenBackgroundStyle(
  scoreA: number | null | undefined,
  scoreB: number | null | undefined = null,
  smooth = false,
  scaleA: number = HIGHLIGHT_SAT,
  scaleB: number = HIGHLIGHT_SAT,
  hueA: TintHue = "signed",
  hueB: TintHue = "signed",
): { backgroundColor?: string; backgroundImage?: string } {
  if (scoreB === null || scoreB === undefined) {
    const bg = scoreToRgb(scoreA, scaleA, hueA);
    return bg === "transparent" ? {} : { backgroundColor: bg };
  }
  return smooth
    ? twoBlendStyle(scoreA, scoreB, scaleA, scaleB, hueA, hueB)
    : twoStripeStyle(scoreA, scoreB, scaleA, scaleB, hueA, hueB);
}
