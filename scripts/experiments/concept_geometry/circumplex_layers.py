"""Layer-resolved circumplex check + dominant-axis naming. No model load.

The consensus averages all 48 layers; a ring could live in a band the average
washes out (the taxonomy's late layers reversed its early structure). So scan
per-layer planarity (λ2/λ1) and adjacency, and name the dominant axis by sorting
the values along the consensus's first whitened MDS coordinate.
"""

from __future__ import annotations

import numpy as np

from run_circumplex import (
    consensus_gram, mds_coords, recovered_order, adjacency_count,
)
from values_heap import LABELS

NPZ = "centroids_circumplex_google__gemma-4-12b-it.npz"
MODEL = "google/gemma-4-12b-it"


def main():
    z = np.load(NPZ)
    layers = sorted(int(k.split("_")[1]) for k in z if k.startswith("layer_"))
    cents = {L: z[f"layer_{L}"] for L in layers}

    try:
        from saklas.core.mahalanobis import LayerWhitener
        whit = LayerWhitener.from_cache(MODEL)
    except Exception as e:
        whit = None
        print(f"(no whitener: {e}; Euclidean)")

    # per-layer planarity + adjacency
    print("per-layer:  λ2/λ1 (planar→1) and Schwartz adjacency /10")
    print(f"{'layer':>5s} {'λ2/λ1':>7s} {'adj':>4s}")
    print("-" * 20)
    best = (-1, None)
    for L in layers:
        G = consensus_gram({L: cents[L]}, whit, [L])
        coords, spec = mds_coords(G, 2)
        _, order = recovered_order(coords)
        adj, _ = adjacency_count(order)
        planar = spec[1] / spec[0] if spec[0] > 0 else 0
        if planar > best[0]:
            best = (planar, L)
        if L % 3 == 0 or L == layers[-1]:
            print(f"{L:>5d} {planar:>7.2f} {adj:>4d}")
    print(f"\nmost planar single layer: {best[1]} (λ2/λ1={best[0]:.2f})")

    # band consensuses
    print("\nband consensus planarity + adjacency:")
    bands = {"early 0-12": [L for L in layers if L <= 12],
             "mid 13-29": [L for L in layers if 13 <= L <= 29],
             "late 30-47": [L for L in layers if L >= 30]}
    for name, band in bands.items():
        G = consensus_gram({L: cents[L] for L in band}, whit, band)
        coords, spec = mds_coords(G, 2)
        _, order = recovered_order(coords)
        adj, _ = adjacency_count(order)
        planar = spec[1] / spec[0]
        print(f"  {name:11s}  λ2/λ1={planar:.2f}  top2={spec[:2].sum()*100:3.0f}%  adj={adj}/10")

    # name the dominant axis: sort values along consensus first MDS coord
    G = consensus_gram(cents, whit, layers)
    coords, spec = mds_coords(G, 3)
    axis1 = coords[:, 0]
    print("\ndominant axis (53% of spread) sorts the 10 values as:")
    sorted_vals = [LABELS[i] for i in np.argsort(axis1)]
    print("  " + "  <  ".join(sorted_vals))
    print("(if this runs self-enhancement <-> self-transcendence, it's the "
          "Schwartz master axis, with the ring collapsed onto it)")


if __name__ == "__main__":
    main()
