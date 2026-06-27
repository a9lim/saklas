"""Gromov delta-hyperbolicity of a finite metric space.

The instrument for the saklas taxonomy-geometry question: given the pairwise
distance matrix of a set of concept centroids, how tree-like is the metric?

A tree is exactly 0-hyperbolic (Gromov four-point condition holds with delta=0).
Hyperbolic space H^n has small bounded delta. Euclidean (flat) space is NOT
hyperbolic -- delta grows with the diameter. Positive curvature (sphere) is the
anti-tree, with large delta. So the *ordering* of delta_rel across known
geometries is the calibration that tells us the probe works:

    tree  <  hyperbolic  <  euclidean  <  sphere

Definitions (Gromov, four-point / Gromov-product form):

  Gromov product wrt basepoint w:
      (x . y)_w = 1/2 ( d(x,w) + d(y,w) - d(x,y) )

  The space is delta-hyperbolic iff for all x, y, z, w:
      (x . z)_w >= min( (x . y)_w, (y . z)_w ) - delta

  Equivalently (Fournier-Ismail-Vigneron 2015), with the Gromov-product matrix
  G^w[i,j] = (i . j)_w and the (max,min) matrix product
      (G (x) G)[i,j] = max_k min( G[i,k], G[k,j] ),
  the delta relative to basepoint w is
      delta_w = max_{i,j} ( (G^w (x) G^w)[i,j] - G^w[i,j] ),
  and the true delta is the max over all basepoints w.

We report:
  - delta      : worst-case Gromov four-point delta (over all basepoints)
  - diam       : max pairwise distance
  - delta_rel  : delta / diam   (scale-free; 0 for a tree)

References:
  Gromov (1987), "Hyperbolic groups."
  Fournier, Ismail, Vigneron (2015), "Computing the Gromov hyperbolicity of a
    discrete metric space." Inf. Process. Lett.
  Borassi, Coudert, Crescenzi, Marino (2015), on delta_rel = delta/diam.
"""

from __future__ import annotations

import numpy as np


def gromov_product_matrix(D: np.ndarray, w: int) -> np.ndarray:
    """Gromov-product matrix G[i,j] = (i . j)_w for basepoint index w."""
    dw = D[w]                          # d(., w)
    return 0.5 * (dw[:, None] + dw[None, :] - D)


def _maxmin_product(G: np.ndarray) -> np.ndarray:
    """(max,min) matrix product H[i,j] = max_k min(G[i,k], G[k,j])."""
    # A[i,j,k] = G[i,k] ; B[i,j,k] = G[k,j]
    A = G[:, None, :]                  # (n,1,n)
    B = G.T[None, :, :]                # (1,n,n) -> B[0,j,k] = G[k,j]
    return np.minimum(A, B).max(axis=2)


def delta_for_basepoint(D: np.ndarray, w: int) -> float:
    G = gromov_product_matrix(D, w)
    H = _maxmin_product(G)
    return float((H - G).max())


def gromov_delta(D: np.ndarray, basepoint="all") -> dict:
    """Gromov four-point delta-hyperbolicity of distance matrix D.

    basepoint: "all" (exact: max over every basepoint), or an int (fixed
               basepoint; a 2-approximation, much cheaper for large n).

    Returns {delta, diam, delta_rel}.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("D must be square")
    if not np.allclose(D, D.T, atol=1e-9):
        raise ValueError("D must be symmetric")

    if basepoint == "all":
        delta = max(delta_for_basepoint(D, w) for w in range(n))
    else:
        delta = delta_for_basepoint(D, int(basepoint))

    diam = float(D[np.triu_indices(n, 1)].max()) if n > 1 else 0.0
    delta_rel = delta / diam if diam > 0 else 0.0
    return {"delta": delta, "diam": diam, "delta_rel": delta_rel, "n": n}


def delta_brute_force(D: np.ndarray) -> float:
    """O(n^4) direct four-point definition, for cross-checking small cases.

    For every 4-tuple {a,b,c,d}, form the three pair-sums
        S1 = d(a,b)+d(c,d),  S2 = d(a,c)+d(b,d),  S3 = d(a,d)+d(b,c),
    sort descending; the quadruple's delta is (largest - second)/2.
    delta = max over all quadruples.  (Equals the Gromov-product form above.)
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    best = 0.0
    from itertools import combinations
    for a, b, c, d in combinations(range(n), 4):
        s = sorted(
            [D[a, b] + D[c, d], D[a, c] + D[b, d], D[a, d] + D[b, c]],
            reverse=True,
        )
        best = max(best, 0.5 * (s[0] - s[1]))
    return best
