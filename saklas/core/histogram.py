"""Shared histogram helpers for per-layer magnitude displays."""

from __future__ import annotations

# One knob for every per-layer histogram surface (TUI WHY footer, CLI
# ``vector why``). Chosen so the full profile of any supported model
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
