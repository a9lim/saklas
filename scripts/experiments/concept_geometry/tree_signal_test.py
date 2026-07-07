"""Is the tree correlation real? Label-permutation null on cached centroids.

δ_rel is label-free, but r(d, tree) asks "does the activation geometry know THIS
specific taxonomy?" Test the true alignment against the null where tree labels are
shuffled across the same fixed centroids. True alignment far in the tail => real
(if weak) tree signal; buried in the null => noise.

No model load — operates on the saved npz + the cached whitener.

Usage:
  python3 tree_signal_test.py --taxonomy deep --npz centroids_deep_google__gemma-4-12b-it.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from taxonomy_heap import get_taxonomy

rng = np.random.default_rng(0)
P = 20000


def offdiag(M):
    iu = np.triu_indices(M.shape[0], 1)
    return M[iu]


def pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def maha_dist_matrix(C, whit, L):
    import torch
    n = C.shape[0]
    D = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            d = whit.mahalanobis_norm(L, torch.from_numpy(C[a] - C[b]).float())
            D[a, b] = D[b, a] = d
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default="deep", choices=["shallow", "deep"])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model", default="google/gemma-4-12b-it")
    ap.add_argument("--band-max", type=int, default=27,
                    help="pool r over layers <= band-max (the early-mid band)")
    args = ap.parse_args()

    nodes, treeD, _ = get_taxonomy(args.taxonomy)
    z = np.load(args.npz)
    layers = sorted(int(k.split("_")[1]) for k in z if k.startswith("layer_"))
    assert list(z["nodes"]) == nodes, "node order mismatch (taxonomy vs npz)"

    try:
        from saklas.core.mahalanobis import LayerWhitener
        whit = LayerWhitener.from_cache(args.model)
        metric = "Mahalanobis"
    except Exception as e:
        whit = None
        metric = "Euclidean"
        print(f"(no whitener: {e}; Euclidean metric)")

    tree_vec = offdiag(treeD)
    d2_by_layer = {}
    for L in layers:
        C = z[f"layer_{L}"].astype(np.float64)
        if whit is not None and L in whit.layers:
            D = maha_dist_matrix(C, whit, L)
        else:
            diff = C[:, None, :] - C[None, :, :]
            D = np.sqrt(np.maximum((diff ** 2).sum(-1), 0.0))
        d2_by_layer[L] = offdiag(D) ** 2

    print(f"taxonomy={args.taxonomy} ({len(nodes)} nodes) | metric={metric} | "
          f"permutations={P}\n")
    print(f"{'layer':>5s} {'r(d²,tree)':>11s} {'perm_p':>8s}")
    print("-" * 28)
    true_r = {L: pearson(d2_by_layer[L], tree_vec) for L in layers}

    n = len(nodes)
    iu = np.triu_indices(n, 1)
    perms = [rng.permutation(n) for _ in range(P)]
    null_tree_vecs = [treeD[np.ix_(p, p)][iu] for p in perms]

    per_layer_p = {}
    for L in layers:
        tr = true_r[L]
        null = np.array([pearson(d2_by_layer[L], tv) for tv in null_tree_vecs])
        per_layer_p[L] = (1 + np.sum(null >= tr)) / (1 + P)
        if L % max(1, len(layers) // 16) == 0 or L == layers[-1]:
            print(f"{L:>5d} {tr:>11.3f} {per_layer_p[L]:>8.4f}")

    band = [L for L in layers if L <= args.band_max]
    S_true = np.mean([true_r[L] for L in band])
    S_null = np.array([
        np.mean([pearson(d2_by_layer[L], tv) for L in band]) for tv in null_tree_vecs
    ])
    p_pool = (1 + np.sum(S_null >= S_true)) / (1 + P)

    print(f"\n-- pooled over early-mid band (layers 0-{args.band_max}) --")
    print(f"mean r(d²,tree) true = {S_true:.3f}")
    print(f"null mean = {S_null.mean():.3f}  null 95th pct = {np.percentile(S_null,95):.3f}")
    print(f"permutation p = {p_pool:.5f}   "
          f"({'SIGNAL beats chance' if p_pool < 0.05 else 'within noise'})")
    n_sig = sum(1 for L in band if per_layer_p[L] < 0.05)
    print(f"per-layer: {n_sig}/{len(band)} early-mid layers individually p<0.05")


if __name__ == "__main__":
    main()
