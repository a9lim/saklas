"""Shared histogram helpers for per-layer magnitude displays."""

from __future__ import annotations

from saklas.core.stats import median_or_zero

# One knob for every per-layer histogram surface (TUI WHY footer, CLI
# ``manifold why``). Chosen so the full profile of any supported model
# collapses into a fixed-height block that fits without scrolling.
HIST_BUCKETS = 16


def bucketize(
    norms: list[tuple[int, float]], buckets: int
) -> list[tuple[int, int, float]]:
    """Collapse per-layer norms into ``buckets`` evenly-sized groups.

    ``norms`` must be sorted by layer index ascending. Returns
    ``[(lo_idx, hi_idx, mean_norm), ...]`` in layer order. When there are
    already fewer layers than ``buckets``, each layer becomes its own bucket.
    """
    n = len(norms)
    if n <= buckets:
        return [(l, l, v) for l, v in norms]
    out: list[tuple[int, int, float]] = []
    for i in range(buckets):
        lo = (i * n) // buckets
        hi = ((i + 1) * n) // buckets
        chunk = norms[lo:hi]
        mean = sum(v for _, v in chunk) / len(chunk)
        out.append((chunk[0][0], chunk[-1][0], mean))
    return out


def summarize_diagnostics(
    diagnostics: dict[int, dict[str, float]],
) -> dict[str, float | str]:
    """Aggregate per-layer extraction diagnostics into a small summary block."""
    evrs = [m["evr"] for m in diagnostics.values() if "evr" in m]
    intras = [
        m["intra_pair_variance_mean"]
        for m in diagnostics.values()
        if "intra_pair_variance_mean" in m
    ]
    aligns = [
        m["inter_pair_alignment"]
        for m in diagnostics.values()
        if "inter_pair_alignment" in m
    ]
    projs = [
        m["diff_principal_projection"]
        for m in diagnostics.values()
        if "diff_principal_projection" in m
    ]

    med_evr = median_or_zero(evrs)
    med_intra = median_or_zero(intras)
    med_align = median_or_zero(aligns)
    med_proj = median_or_zero(projs)

    if (med_evr > 0.95 and med_intra < 0.01) or med_align < 0.2:
        quality = "poor"
    elif med_align < 0.4 or med_evr < 0.2:
        quality = "shaky"
    else:
        quality = "solid"

    return {
        "median_evr": round(med_evr, 4),
        "median_intra_pair_variance": round(med_intra, 4),
        "median_inter_pair_alignment": round(med_align, 4),
        "median_diff_principal_projection": round(med_proj, 4),
        "quality": quality,
    }
