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

/** Map a probe score to a CSS rgb() string (higher saturation = stronger
 * color).  Returns ``"transparent"`` when score is effectively zero or
 * null/undefined.
 *
 * Positive (toward the +pole): green ramp from rgb(0,0,0) at t=0 to
 * rgb(0,255,0) at |t|=1.  Negative (toward the -pole): red ramp.
 *
 * ``scale`` is the value at which the ramp reaches full saturation:
 * ``t = score / scale``, clamped to [-1, 1].  It defaults to
 * ``HIGHLIGHT_SAT`` so the fixed-[-1,1] callers (the correlation matrix,
 * the cross-layer cosine grid) are unchanged; the probe surfaces pass a
 * per-probe node-coordinate extent (see ``nodeCoordExtent``) so a fan
 * whose coords run to ±tens isn't pinned fully saturated from the first
 * token.  A degenerate (0 / non-finite) scale falls back to the fixed
 * cutoff rather than dividing by zero. */
export function scoreToRgb(
  score: number | null | undefined,
  scale: number = HIGHLIGHT_SAT,
): string {
  if (score == null || !Number.isFinite(score)) return "transparent";
  const s = Number.isFinite(scale) && scale > 1e-6 ? scale : HIGHLIGHT_SAT;
  const t = Math.max(-1, Math.min(1, score / s));
  if (t === 0) return "transparent";
  if (t > 0) {
    const g = Math.round(255 * t);
    return `rgb(0,${g},0)`;
  }
  const r = Math.round(255 * -t);
  return `rgb(${r},0,0)`;
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
): { backgroundImage: string } {
  const top = scoreToRgb(scoreA, scaleA);
  const bot = scoreToRgb(scoreB, scaleB);
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
): { backgroundImage: string } {
  const top = scoreToRgb(scoreA, scaleA);
  const bot = scoreToRgb(scoreB, scaleB);
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
): { backgroundColor?: string; backgroundImage?: string } {
  if (scoreB === null || scoreB === undefined) {
    const bg = scoreToRgb(scoreA, scaleA);
    return bg === "transparent" ? {} : { backgroundColor: bg };
  }
  return smooth
    ? twoBlendStyle(scoreA, scoreB, scaleA, scaleB)
    : twoStripeStyle(scoreA, scoreB, scaleA, scaleB);
}
