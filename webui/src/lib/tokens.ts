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

/** Map a probe score in [-1, +1] (higher saturation = stronger color)
 * to a CSS rgb() string.  Returns ``"transparent"`` when score is
 * effectively zero or null/undefined.
 *
 * Positive (toward the +pole): green ramp from rgb(0,0,0) at t=0 to
 * rgb(0,255,0) at |t|=1.  Negative (toward the -pole): red ramp.
 *
 * t = score / HIGHLIGHT_SAT, clamped to [-1, 1] — anything past the
 * saturation cutoff renders fully saturated. */
export function scoreToRgb(score: number | null | undefined): string {
  if (score == null || !Number.isFinite(score)) return "transparent";
  const t = Math.max(-1, Math.min(1, score / HIGHLIGHT_SAT));
  if (t === 0) return "transparent";
  if (t > 0) {
    const g = Math.round(255 * t);
    return `rgb(0,${g},0)`;
  }
  const r = Math.round(255 * -t);
  return `rgb(${r},0,0)`;
}

/** Composite background style for compare-two mode.  Stripes the token
 * with probe-A on top and probe-B on bottom via a CSS linear gradient
 * with a hard color stop at 50%.  Either side can be transparent — the
 * gradient still works because CSS overlays transparent stops cleanly. */
export function twoStripeStyle(
  scoreA: number | null | undefined,
  scoreB: number | null | undefined,
): { backgroundImage: string } {
  const top = scoreToRgb(scoreA);
  const bot = scoreToRgb(scoreB);
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
): { backgroundImage: string } {
  const top = scoreToRgb(scoreA);
  const bot = scoreToRgb(scoreB);
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
): { backgroundColor?: string; backgroundImage?: string } {
  if (scoreB === null || scoreB === undefined) {
    const bg = scoreToRgb(scoreA);
    return bg === "transparent" ? {} : { backgroundColor: bg };
  }
  return smooth ? twoBlendStyle(scoreA, scoreB) : twoStripeStyle(scoreA, scoreB);
}
