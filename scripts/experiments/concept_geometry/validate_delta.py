"""Validate the delta-hyperbolicity probe on synthetic spaces with KNOWN answers.

We must trust the instrument before any number on real activations means
anything. The pass condition is the *ordering*

    tree ~ 0  <  hyperbolic  <  euclidean(flat)  <  sphere

plus two hard checks: a tree is EXACTLY 0, and the fast fixed-basepoint
2-approx brackets the exact all-basepoints value.
"""

from __future__ import annotations

import numpy as np

from delta_hyperbolicity import gromov_delta, delta_brute_force


rng = np.random.default_rng(0)  # fixed seed; Math.random-free determinism


def pdist(X, metric="euclidean"):
    """Dense pairwise distance matrix."""
    if metric == "euclidean":
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff ** 2).sum(-1))
    raise ValueError(metric)


# ---- synthetic metric spaces ------------------------------------------------

def tree_distances(branching=3, depth=4):
    """All-pairs shortest path on a balanced b-ary tree (unit edges)."""
    # build adjacency by BFS construction
    parent = {0: None}
    nodes = [0]
    frontier = [0]
    nxt = 1
    for _ in range(depth):
        new_frontier = []
        for p in frontier:
            for _ in range(branching):
                parent[nxt] = p
                nodes.append(nxt)
                new_frontier.append(nxt)
                nxt += 1
        frontier = new_frontier
    n = len(nodes)
    INF = 10**9
    D = np.full((n, n), INF)
    np.fill_diagonal(D, 0)
    for c, p in parent.items():
        if p is not None:
            D[c, p] = D[p, c] = 1
    # Floyd-Warshall (n is small)
    for k in range(n):
        D = np.minimum(D, D[:, k][:, None] + D[k, :][None, :])
    return D.astype(float)


def ring_distances(n=12):
    """Shortest-path distance on a cycle graph C_n."""
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(d, n - d).astype(float)


def euclidean_flat(n=40, dim=10):
    return pdist(rng.standard_normal((n, dim)))


def sphere_geodesic(n=40, dim=3):
    X = rng.standard_normal((n, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    dots = np.clip(X @ X.T, -1.0, 1.0)
    return np.arccos(dots)


def poincare_hyperbolic(n=40, dim=2, max_r=0.92):
    """Sample points in the Poincare ball, use the exact Poincare metric."""
    # sample roughly uniform-ish in the ball, push some mass toward the edge
    dirs = rng.standard_normal((n, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = max_r * rng.uniform(0, 1, size=n) ** (1.0 / dim)
    X = dirs * radii[:, None]
    sq = (X ** 2).sum(1)                      # |x|^2
    diff2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    denom = (1 - sq)[:, None] * (1 - sq)[None, :]
    arg = 1 + 2 * diff2 / denom
    np.fill_diagonal(arg, 1.0)
    return np.arccosh(np.clip(arg, 1.0, None))


# ---- run --------------------------------------------------------------------

def main():
    cases = {
        "tree (b=3,d=4)": tree_distances(3, 4),
        "path (b=1,d=12)": tree_distances(1, 12),
        "hyperbolic (Poincare H2)": poincare_hyperbolic(40, 2),
        "euclidean flat (R^10)": euclidean_flat(40, 10),
        "ring C_12": ring_distances(12),
        "sphere S^2": sphere_geodesic(40, 3),
    }

    print(f"{'space':28s} {'n':>4s} {'diam':>8s} {'delta':>9s} {'delta_rel':>10s}")
    print("-" * 64)
    rows = {}
    for name, D in cases.items():
        r = gromov_delta(D, basepoint="all")
        rows[name] = r
        print(f"{name:28s} {r['n']:>4d} {r['diam']:>8.3f} "
              f"{r['delta']:>9.4f} {r['delta_rel']:>10.4f}")

    print("\n-- hard checks --")
    # 1. tree is exactly 0
    tree_delta = rows["tree (b=3,d=4)"]["delta"]
    print(f"tree delta == 0 (exact)            : {tree_delta:.2e}  "
          f"{'PASS' if tree_delta < 1e-9 else 'FAIL'}")

    # 2. brute-force agrees with the matrix form on a small case
    Dsmall = ring_distances(8)
    bf = delta_brute_force(Dsmall)
    mx = gromov_delta(Dsmall, basepoint="all")["delta"]
    print(f"brute-force == matrix form (ring8) : {bf:.4f} vs {mx:.4f}  "
          f"{'PASS' if abs(bf - mx) < 1e-9 else 'FAIL'}")

    # 3. fixed-basepoint is a 2-approx: delta_fixed <= delta_exact <= 2*delta_fixed
    Dh = cases["euclidean flat (R^10)"]
    ex = gromov_delta(Dh, basepoint="all")["delta"]
    fx = gromov_delta(Dh, basepoint=0)["delta"]
    ok = fx <= ex + 1e-9 <= 2 * fx + 1e-9
    print(f"fixed-bp 2-approx brackets exact   : fixed={fx:.4f} exact={ex:.4f}  "
          f"{'PASS' if ok else 'FAIL'}")

    # 4. the ordering we care about
    order = (rows["tree (b=3,d=4)"]["delta_rel"]
             <= rows["hyperbolic (Poincare H2)"]["delta_rel"]
             <  rows["euclidean flat (R^10)"]["delta_rel"]
             <  rows["sphere S^2"]["delta_rel"])
    print(f"ordering tree<hyp<euclid<sphere    : "
          f"{'PASS' if order else 'FAIL'}")


if __name__ == "__main__":
    main()
