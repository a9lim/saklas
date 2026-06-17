// Helpers for rendering discover-mode manifold fit diagnostics in the
// webui inspector.  Pure functions — no Svelte runes, no DOM — so the
// same shapes back both the ManifoldDrawer inspector and any future
// inline summary panel.
//
// The server's per-tensor sidecar carries a ``diagnostics`` block whose
// shape depends on the fit method:
//
//   * PCA      — ``{per_component_variance, cumulative_variance,
//                  picked_k, threshold}``
//   * Spectral — ``{eigenvalues, picked_k, gap_index, gap_magnitude,
//                  bandwidth, k_nn, component_count}``
//
// Both ride into the wire as flat number arrays + scalars (tensors are
// flattened server-side in ``_diagnostics_to_dict``).  The discriminator
// is structural — presence of ``per_component_variance`` vs
// ``eigenvalues`` — which keeps this module from having to know about a
// growing taxonomy of method names.

import type {
  ManifoldFitInfo,
  ManifoldPcaDiagnostics,
  ManifoldSpectralDiagnostics,
} from "../types";

export type DiscoverDiagnostics =
  | ({ kind: "pca" } & ManifoldPcaDiagnostics)
  | ({ kind: "spectral" } & ManifoldSpectralDiagnostics);

/** Tag a raw diagnostics block with its method.  Returns ``null`` for
 *  authored fits or for shapes that don't match either method (an
 *  older server sending a sparser dict). */
export function classifyDiagnostics(
  fit: ManifoldFitInfo,
): DiscoverDiagnostics | null {
  const d = fit.diagnostics;
  if (!d) return null;
  if ("per_component_variance" in d && Array.isArray(d.per_component_variance)) {
    return { kind: "pca", ...d };
  }
  if ("eigenvalues" in d && Array.isArray(d.eigenvalues)) {
    return { kind: "spectral", ...d };
  }
  return null;
}

/** One bar on the variance / spectrum histogram.  ``value`` is the
 *  raw metric (variance fraction or eigenvalue); ``frac`` is the
 *  normalized height for rendering, in [0, 1]. */
export interface DiagnosticsBar {
  index: number;       // 1-based component / eigenvalue index
  value: number;
  frac: number;        // 0..1 for bar height
  picked: boolean;     // True iff this index is at or before the picked-k cut
}

/** Bars for the PCA variance plot.  Heights are the per-component
 *  variance fractions themselves — they already live in [0, 1] and sum
 *  to ≤ 1 across the kept prefix.  The ``picked`` flag is set on every
 *  component up to and including ``picked_k`` so the inspector can
 *  highlight the kept prefix vs the discarded tail. */
export function pcaBars(d: ManifoldPcaDiagnostics): DiagnosticsBar[] {
  const max = d.per_component_variance.reduce(
    (m, v) => Math.max(m, v), 0,
  );
  // Normalize for visual height but keep ``value`` as the raw fraction —
  // a single dominant component shouldn't shrink the rest to zero-pixel
  // bars (max-normalize, not sum-normalize).
  return d.per_component_variance.map((v, i) => ({
    index: i + 1,
    value: v,
    frac: max > 0 ? v / max : 0,
    picked: i + 1 <= d.picked_k,
  }));
}

/** Bars for the spectral eigenvalue plot.  Heights are normalized by
 *  the largest kept eigenvalue so the cliff between "kept" and
 *  "dropped" eigenvalues is the visual focal point.  ``picked_k``
 *  marks the dimension count — eigenvalues *below* that are kept;
 *  the next one is the start of the gap. */
export function spectralBars(d: ManifoldSpectralDiagnostics): DiagnosticsBar[] {
  const max = d.eigenvalues.reduce((m, v) => Math.max(m, v), 0);
  return d.eigenvalues.map((v, i) => ({
    index: i + 1,
    value: v,
    frac: max > 0 ? v / max : 0,
    picked: i + 1 <= d.picked_k,
  }));
}

/** Short one-line summary string for an inspector row.  Used in the
 *  fitted-row badge / hover-title; the detail panel below the row
 *  carries the full bars. */
export function diagnosticsSummary(diag: DiscoverDiagnostics): string {
  if (diag.kind === "pca") {
    const cum = diag.cumulative_variance;
    const cumAtK = diag.picked_k <= cum.length ? cum[diag.picked_k - 1] : null;
    const cumStr = cumAtK !== null ? cumAtK.toFixed(3) : "?";
    return `pca · k=${diag.picked_k} · cumvar@k=${cumStr} (≥${diag.threshold})`;
  }
  const gap = diag.gap_magnitude.toExponential(2);
  return (
    `spectral · k=${diag.picked_k} · gap=${gap} · ` +
    `σ=${diag.bandwidth.toPrecision(3)} · k_nn=${diag.k_nn}`
  );
}

/** Pick the fitted-record entry for the loaded session's model.  In
 *  practice the server's ``_manifold_json`` reports at most one
 *  per-model tensor stem at a time; this helper just picks the first
 *  whose ``fit_mode`` is one of the discover modes.  ``auto`` folders
 *  store ``fit_mode === "auto"`` in the sidecar (the resolved geometry
 *  rides ``resolved_fit_mode``) and now carry a real diagnostics block
 *  for the winning mode, so they qualify too — without this the headline
 *  ``personas`` / ``emotions`` manifolds render no diagnostics.  Returns
 *  ``null`` when no fitted record is in discover mode (authored fits, or
 *  no fits at all). */
export function pickDiscoverFit(
  fitted: ManifoldFitInfo[] | undefined,
): ManifoldFitInfo | null {
  if (!fitted) return null;
  for (const f of fitted) {
    if (f.fit_mode === "pca" || f.fit_mode === "spectral" || f.fit_mode === "auto") {
      return f;
    }
  }
  return null;
}
