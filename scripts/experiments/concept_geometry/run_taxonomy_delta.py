"""The taxonomy δ-hyperbolicity experiment (depth-parametrized).

Generate an embodied corpus per taxonomy node, pool full-dim per-layer centroids,
and measure how tree-like the activation arrangement is — in BOTH the raw
Euclidean and saklas's Mahalanobis metric — WITHOUT a flat fit in the loop (so a
flat subspace can't pre-flatten any hidden curvature).

Outputs per layer:
  - δ_rel (Euclidean) and δ_rel (Mahalanobis)   vs the calibration bands
  - Hewitt-Manning signature: Pearson(activation_dist, tree_dist) and
    Pearson(activation_dist^2, tree_dist).  squared winning = tree-embedding print.

Centroids cache to a taxonomy-tagged npz so re-analysis never re-runs the model.

Usage:
  python3 run_taxonomy_delta.py --taxonomy deep google/gemma-4-12b-it
  python3 run_taxonomy_delta.py --analyze-only centroids_deep_<tag>.npz --taxonomy deep
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from delta_hyperbolicity import gromov_delta
from taxonomy_heap import get_taxonomy

_WHITENER = None


def _pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _euclid_D(C):  # C: (K, D)
    diff = C[:, None, :] - C[None, :, :]
    return np.sqrt(np.maximum((diff ** 2).sum(-1), 0.0))


def analyze(centroids_by_layer, tag, nodes, treeD):
    iu = np.triu_indices(len(nodes), 1)
    tree_flat = treeD[iu]
    layers = sorted(centroids_by_layer)

    print(f"\n===== taxonomy δ-hyperbolicity :: {tag} =====")
    print(f"{len(nodes)} nodes | tree diam {int(treeD.max())} | tree δ_rel = "
          f"{gromov_delta(treeD)['delta_rel']:.3f} (=0 by construction)")
    print("calibration: tree 0.00 | hyperbolic 0.11 | EUCLID-FLAT 0.19-0.22 | "
          "ring/sphere 0.42-0.50")
    print("personas/emotions (same model, flat band): ~0.19-0.21\n")
    print(f"{'layer':>5s} {'δrel_eucl':>10s} {'δrel_maha':>10s} "
          f"{'r(d,tree)':>10s} {'r(d²,tree)':>11s} {'sq_wins':>8s}")
    print("-" * 60)

    eucl_rels, maha_rels = [], []
    import torch
    for L in layers:
        C = centroids_by_layer[L].astype(np.float64)
        Deu = _euclid_D(C)
        re = gromov_delta(Deu, basepoint="all")["delta_rel"]
        eucl_rels.append(re)

        rm = float("nan")
        if _WHITENER is not None and L in _WHITENER.layers:
            Dm = np.zeros_like(Deu)
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    d = _WHITENER.mahalanobis_norm(L, torch.from_numpy(C[a] - C[b]).float())
                    Dm[a, b] = Dm[b, a] = d
            rm = gromov_delta(Dm, basepoint="all")["delta_rel"]
            maha_rels.append(rm)
            d_flat = Dm[iu]
        else:
            d_flat = Deu[iu]

        r1 = _pearson(d_flat, tree_flat)
        r2 = _pearson(d_flat ** 2, tree_flat)
        sq = "yes" if (r2 > r1 + 1e-3) else ""
        rm_s = f"{rm:>10.3f}" if rm == rm else f"{'n/a':>10s}"
        if L % max(1, len(layers) // 16) == 0 or L == layers[-1]:
            print(f"{L:>5d} {re:>10.3f} {rm_s} {r1:>10.3f} {r2:>11.3f} {sq:>8s}")

    eucl = np.array(eucl_rels)
    print("\n-- summary --")
    print(f"Euclidean   δ_rel:  min={eucl.min():.3f}  median={np.median(eucl):.3f}"
          f"  max={eucl.max():.3f}")
    if maha_rels:
        m = np.array(maha_rels)
        print(f"Mahalanobis δ_rel:  min={m.min():.3f}  median={np.median(m):.3f}"
              f"  max={m.max():.3f}")
        med = float(np.median(m))
    else:
        med = float(np.median(eucl))
    verdict = ("TREE-LIKE / hyperbolic signature" if med <= 0.14 else
               "FLAT (Park-Veitch: model flattens the tree)" if med <= 0.26 else
               "CURVED / non-tree")
    print(f"\nverdict (median δ_rel = {med:.3f}):  {verdict}")
    print("(δ_rel is a coarse band read; run tree_signal_test.py for the "
          "permutation-null significance of the tree alignment)")


def run_model(model_id, max_new_tokens, taxonomy):
    global _WHITENER
    from saklas import SaklasSession
    from saklas.core.vectors import _load_baseline_prompts
    from saklas.core.manifold import compute_node_centroid

    nodes, treeD, _parent = get_taxonomy(taxonomy)
    safe = model_id.replace("/", "__")
    print(f"loading {model_id} ... (taxonomy={taxonomy}, {len(nodes)} nodes)", flush=True)
    session = SaklasSession.from_pretrained(model_id, device="auto")
    prompts = _load_baseline_prompts()
    print(f"baseline prompts: {len(prompts)} | nodes: {len(nodes)} | "
          f"generations ≈ {len(nodes) * len(prompts)}", flush=True)

    t0 = time.time()
    print("generating embodied corpora (kind=concrete) ...", flush=True)
    corpora = session.generate_responses(
        concepts=nodes,
        kinds=["concrete"] * len(nodes),
        samples_per_prompt=1,
        max_new_tokens=max_new_tokens,
        on_progress=lambda m: print("  ", m, flush=True),
    )
    print(f"generation done in {time.time()-t0:.0f}s", flush=True)

    _WHITENER = session.whitener
    layers = session.layers
    mtype = getattr(getattr(session.model, "config", None), "model_type", None)
    centroids = {}
    t1 = time.time()
    for i, node in enumerate(nodes):
        cd = compute_node_centroid(
            session.model, session.tokenizer, layers, session.device,
            corpora[node], prompts, model_type=mtype,
        )
        for L, vec in cd.items():
            centroids.setdefault(L, np.zeros((len(nodes), vec.shape[0]), np.float32))
            centroids[L][i] = vec.numpy()
        print(f"  centroid {i+1}/{len(nodes)} {node}", flush=True)
    print(f"centroids pooled in {time.time()-t1:.0f}s", flush=True)

    out = f"centroids_{taxonomy}_{safe}.npz"
    np.savez_compressed(out, **{f"layer_{L}": centroids[L] for L in centroids},
                        nodes=np.array(nodes))
    print(f"saved -> {out}")
    analyze(centroids, f"{model_id} [{taxonomy}]", nodes, treeD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", nargs="?", default="google/gemma-4-12b-it")
    ap.add_argument("--taxonomy", default="deep", choices=["shallow", "deep"])
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--analyze-only", metavar="NPZ")
    args = ap.parse_args()

    nodes, treeD, _ = get_taxonomy(args.taxonomy)
    if args.analyze_only:
        z = np.load(args.analyze_only)  # self-generated npz: float32 + <U arrays, no pickle
        centroids = {int(k.split("_")[1]): z[k] for k in z if k.startswith("layer_")}
        global _WHITENER
        try:
            from saklas.core.mahalanobis import LayerWhitener
            tag = args.analyze_only.split("_", 2)[-1].replace(".npz", "")
            _WHITENER = LayerWhitener.from_cache(tag.replace("__", "/"))
        except Exception as e:
            print(f"(no whitener: {e}; Euclidean only)")
        analyze(centroids, args.analyze_only, nodes, treeD)
    else:
        run_model(args.model_id, args.max_new_tokens, args.taxonomy)


if __name__ == "__main__":
    main()
