"""Validate the circumplex ring-recovery analysis on synthetic geometries.

Three cases with known answers:
  - RING:   10 points on a circle in Schwartz order, lifted to high-D + noise.
            expect adjacency 10/10, planar λ2/λ1 ~1, circ corr ~1.
  - LINE:   10 points on a 1-D axis in Schwartz order (a bipolar continuum).
            adjacency is HIGH (a sorted line cyclically closes), but planar ~0 —
            the planarity guard must reject this as a ring.
  - RANDOM: 10 points in flat high-D noise. expect adjacency ~chance, p>0.05.
"""

from __future__ import annotations

import numpy as np

from run_circumplex import (
    consensus_gram, mds_coords, recovered_order, adjacency_count,
    circ_corr, gram_to_dist,
)
from values_heap import ideal_angles
from delta_hyperbolicity import gromov_delta

rng = np.random.default_rng(1)
N, D, LAYERS = 10, 200, 6


def lift(coords2d, noise):
    """Embed (N,k) coords into D dims via a random orthonormal map + noise."""
    k = coords2d.shape[1]
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0][:, :k]
    base = coords2d @ Q.T
    return {L: (base + noise * rng.standard_normal((N, D))).astype(np.float32)
            for L in range(LAYERS)}


def report(name, cents):
    G = consensus_gram(cents, None, list(cents))   # euclidean (no whitener)
    coords, spec = mds_coords(G, 2)
    theta, order = recovered_order(coords)
    adj, _ = adjacency_count(order)
    planar = spec[1] / spec[0]
    cc = max(abs(circ_corr(theta, ideal_angles())),
             abs(circ_corr(-theta, ideal_angles())))
    dr = gromov_delta(gram_to_dist(G))["delta_rel"]
    # quick permutation p
    P = 5000
    null = np.array([adjacency_count(order[rng.permutation(N)])[0] for _ in range(P)])
    p = (1 + np.sum(null >= adj)) / (1 + P)
    print(f"{name:8s}  adj={adj:2d}/10  p={p:.4f}  λ2/λ1={planar:.2f}  "
          f"top2={spec[:2].sum()*100:4.0f}%  |ρ|={cc:.2f}  δ_rel={dr:.3f}")


def main():
    ang = ideal_angles()
    ring = np.c_[np.cos(ang), np.sin(ang)]              # circle in Schwartz order
    line = np.c_[np.linspace(-1, 1, N), np.zeros(N)]    # 1-D axis in order
    rand = rng.standard_normal((N, 2))                  # arbitrary flat

    print(f"{'case':8s}  {'adjacency':>9s}  {'perm p':>6s}  planar    spread   ring   delta")
    print("-" * 72)
    for noise in (0.04, 0.10):
        print(f"# noise={noise}")
        report("ring", lift(ring, noise))
        report("line", lift(line, noise))
        report("random", lift(rand, noise))
    print("\nexpect: ring adj high + planar~1; line adj high but planar~0 (REJECTED "
          "as ring); random adj~chance p>0.05")


if __name__ == "__main__":
    main()
