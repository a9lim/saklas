"""Schwartz values circumplex experiment.

Generate an embodied corpus per value, pool full-dim per-layer centroids, build
saklas's whitened consensus Gram, and test whether the activation geometry recovers
a RING in Schwartz's canonical cyclic order.

Headline tests:
  - eigenvalue spectrum of the consensus Gram (planar ring => 2 dominant ~equal eigs)
  - cyclic-order recovery: do the recovered angles reproduce Schwartz adjacency?
    (adjacency-preservation count /10 + label-permutation null + circular correlation)
  - δ_rel of the whitened heap (ring reads ~0.3-0.5, flat ~0.19)

Usage:
  python3 run_circumplex.py google/gemma-4-12b-it
  python3 run_circumplex.py --analyze-only centroids_circumplex_<tag>.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from delta_hyperbolicity import gromov_delta
from values_heap import LABELS, CONCEPTS, SYSTEM, schwartz_adjacency, ideal_angles

rng = np.random.default_rng(0)
_WHITENER = None


# ---- whitened consensus geometry -------------------------------------------

def consensus_gram(centroids_by_layer, whit, layers):
    """Mean over layers of the whitened, node-mean-centered (K,K) Gram.

    Signal-weighted (saklas discover-coords consensus): a layer where the values
    aren't separated contributes a small Gram and drops out on its own.
    """
    import torch
    K = next(iter(centroids_by_layer.values())).shape[0]
    G = np.zeros((K, K))
    used = 0
    for L in layers:
        C = centroids_by_layer[L].astype(np.float64)
        Cc = C - C.mean(0)
        if whit is not None and L in whit.layers:
            inv = np.stack([
                whit.apply_inv(L, torch.from_numpy(Cc[j]).float()).numpy().astype(np.float64)
                for j in range(K)
            ])
            GL = Cc @ inv.T
        else:
            GL = Cc @ Cc.T
        GL = 0.5 * (GL + GL.T)
        G += GL
        used += 1
    return G / max(used, 1)


def mds_coords(G, dim=2):
    """Classical-MDS embedding from a centered Gram; returns coords + eig spectrum."""
    w, V = np.linalg.eigh(G)            # ascending
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    pos = np.clip(w, 0, None)
    coords = V[:, :dim] * np.sqrt(pos[:dim])
    spectrum = pos / pos.sum()
    return coords, spectrum


def gram_to_dist(G):
    d2 = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
    return np.sqrt(np.clip(d2, 0, None))


# ---- ring-recovery statistics ----------------------------------------------

def circ_corr(a, b):
    def cmean(x):
        return np.arctan2(np.sin(x).mean(), np.cos(x).mean())
    sa = np.sin(a - cmean(a))
    sb = np.sin(b - cmean(b))
    return float((sa @ sb) / (np.sqrt((sa**2).sum() * (sb**2).sum()) + 1e-12))


def recovered_order(coords):
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    return theta, np.argsort(theta)


def adjacency_count(order):
    n = len(order)
    rec = {frozenset((int(order[k]), int(order[(k + 1) % n]))) for k in range(n)}
    return len(rec & schwartz_adjacency()), rec


def analyze(centroids_by_layer, tag):
    global _WHITENER
    layers = sorted(centroids_by_layer)
    n = len(LABELS)

    G = consensus_gram(centroids_by_layer, _WHITENER, layers)
    coords, spectrum = mds_coords(G, dim=2)
    theta, order = recovered_order(coords)
    adj, rec_edges = adjacency_count(order)

    # δ_rel of the whitened consensus heap
    D = gram_to_dist(G)
    dr = gromov_delta(D)["delta_rel"]

    # circular correlation vs Schwartz ideal angles (best over reflection)
    phi = ideal_angles()
    # align: the recovered order maps node->angle; compare each node's recovered
    # angle to its Schwartz ideal angle, maximizing |corr| over reflection
    cc = max(abs(circ_corr(theta, phi)), abs(circ_corr(-theta, phi)))

    # permutation null on adjacency-preservation (shuffle value->position)
    P = 20000
    null = np.empty(P)
    for t in range(P):
        perm = rng.permutation(n)
        null[t], _ = adjacency_count(order[perm])
    p_adj = (1 + np.sum(null >= adj)) / (1 + P)

    print(f"\n===== Schwartz circumplex :: {tag} =====")
    print(f"{n} values | {len(layers)} layers | whitened consensus geometry")
    print("calibration δ_rel: flat ~0.19 | ring/sphere ~0.42-0.50\n")

    print("eigenvalue spectrum (consensus Gram, top 6):")
    print("  " + "  ".join(f"{s:.3f}" for s in spectrum[:6]))
    planar = spectrum[1] / spectrum[0] if spectrum[0] > 0 else 0
    print(f"  λ2/λ1 = {planar:.2f}  (→1 = planar ring; →0 = a single bipolar axis)")
    print(f"  top-2 captures {spectrum[:2].sum()*100:.0f}% of whitened spread\n")

    print(f"δ_rel (whitened consensus) = {dr:.3f}")
    print(f"\ncyclic-order recovery (Schwartz adjacency, max {n}):")
    print(f"  adjacency preserved = {adj}/{n}   permutation p = {p_adj:.5f}   "
          f"(null mean {null.mean():.2f})")
    print(f"  circular correlation |ρ| vs ideal ring = {cc:.3f}")

    print("\nrecovered ring order (by angle):")
    names = [LABELS[i] for i in order]
    print("  " + " → ".join(names) + " →")
    print("Schwartz order:")
    print("  " + " → ".join(LABELS) + " →")

    verdict = ("RING recovered in Schwartz order" if (p_adj < 0.05 and planar > 0.45)
               else "partial ring / order beats chance" if p_adj < 0.05
               else "no significant ring")
    print(f"\nverdict: {verdict}")


def run_model(model_id):
    global _WHITENER
    from saklas import SaklasSession
    from saklas.core.vectors import _load_baseline_prompts
    from saklas.core.manifold import compute_node_centroid

    safe = model_id.replace("/", "__")
    print(f"loading {model_id} ... ({len(LABELS)} values)", flush=True)
    session = SaklasSession.from_pretrained(model_id, device="auto")
    prompts = _load_baseline_prompts()
    print(f"baseline prompts: {len(prompts)} | generations ≈ "
          f"{len(LABELS) * len(prompts)}", flush=True)

    t0 = time.time()
    print("generating value corpora (kind=custom) ...", flush=True)
    corpora = session.generate_responses(
        concepts=CONCEPTS,
        kinds=["custom"] * len(LABELS),
        custom_system=SYSTEM,
        samples_per_prompt=1,
        max_new_tokens=96,
        on_progress=lambda m: print("  ", m, flush=True),
    )
    print(f"generation done in {time.time()-t0:.0f}s", flush=True)

    _WHITENER = session.whitener
    layers = session.layers
    mtype = getattr(getattr(session.model, "config", None), "model_type", None)
    centroids = {}
    for i, concept in enumerate(CONCEPTS):
        cd = compute_node_centroid(
            session.model, session.tokenizer, layers, session.device,
            corpora[concept], prompts, model_type=mtype,
        )
        for L, vec in cd.items():
            centroids.setdefault(L, np.zeros((len(LABELS), vec.shape[0]), np.float32))
            centroids[L][i] = vec.numpy()
        print(f"  centroid {i+1}/{len(LABELS)} {LABELS[i]}", flush=True)

    out = f"centroids_circumplex_{safe}.npz"
    np.savez_compressed(out, **{f"layer_{L}": centroids[L] for L in centroids},
                        labels=np.array(LABELS))
    print(f"saved -> {out}")
    analyze(centroids, model_id)


def main():
    global _WHITENER
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", nargs="?", default="google/gemma-4-12b-it")
    ap.add_argument("--analyze-only", metavar="NPZ")
    args = ap.parse_args()

    if args.analyze_only:
        z = np.load(args.analyze_only)
        centroids = {int(k.split("_")[1]): z[k] for k in z if k.startswith("layer_")}
        try:
            from saklas.core.mahalanobis import LayerWhitener
            tag = args.analyze_only.split("_", 2)[-1].replace(".npz", "")
            _WHITENER = LayerWhitener.from_cache(tag.replace("__", "/"))
        except Exception as e:
            print(f"(no whitener: {e}; Euclidean only)")
        analyze(centroids, args.analyze_only)
    else:
        run_model(args.model_id)


if __name__ == "__main__":
    main()
