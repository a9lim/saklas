"""δ-hyperbolicity of ALREADY-FITTED saklas manifolds — no model load.

For an affine (flat) discover fit, `layer_N.node_coords` (K, R) are the per-layer
neutral-anchored activation coordinates in an orthonormal basis, so Euclidean
distances there equal distances among the full-dim reconstructed centroids
(orthonormal basis preserves distance). So δ on node_coords == δ on the activation
centroids, captured by the fit, for free.

This is the second validation tier: run it on manifolds whose geometry we already
know. months is an authored periodic RING -> expect ring-like δ_rel (~0.4-0.5).
personas is a flat fan -> expect flat δ_rel (~0.2). If the probe reproduces that on
real saklas data, we trust it on the taxonomy heap.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
from safetensors import safe_open

from delta_hyperbolicity import gromov_delta


def _cdist(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.maximum((diff ** 2).sum(-1), 0.0))


def _layer_keys(keys, suffix):
    out = {}
    for k in keys:
        m = re.fullmatch(rf"layer_(\d+)\.{suffix}", k)
        if m:
            out[int(m.group(1))] = k
    return dict(sorted(out.items()))


def probe_manifold(name, model_tag="google__gemma-4-12b-it", basepoint="all"):
    base = os.path.expanduser(f"~/.saklas/manifolds/default/{name}")
    if not os.path.isdir(base):
        base = os.path.expanduser(f"~/.saklas/manifolds/local/{name}")
    st = os.path.join(base, f"{model_tag}.safetensors")
    if not os.path.exists(st):
        cands = sorted(glob.glob(base + "/*.safetensors"))
        st = cands[0] if cands else None
    if not st:
        print(f"[{name}] no fitted tensor found"); return None

    mj = json.load(open(os.path.join(base, "manifold.json")))
    fit_mode = mj.get("fit_mode")
    n_nodes = len(mj.get("nodes", []))

    with safe_open(st, framework="pt") as f:
        keys = list(f.keys())
        coord_keys = _layer_keys(keys, "node_coords")        # affine
        param_keys = _layer_keys(keys, "node_params")        # curved (domain embed)
        source, lk = ("node_coords(activation)", coord_keys) if coord_keys \
            else ("node_params(authored-domain)", param_keys)
        if not lk:
            print(f"[{name}] no per-layer coords"); return None
        per_layer = {L: f.get_tensor(k).float().numpy() for L, k in lk.items()}

    K = next(iter(per_layer.values())).shape[0]
    R = next(iter(per_layer.values())).shape[1]
    if K < 4:
        print(f"[{name}] n={K} < 4, δ undefined (need a 4-tuple)"); return None

    rels, ds = [], []
    for L, X in per_layer.items():
        r = gromov_delta(_cdist(X), basepoint=basepoint)
        rels.append((L, r["delta_rel"])); ds.append(r["delta_rel"])
    ds = np.array(ds)
    rels.sort(key=lambda t: t[1])

    print(f"\n[{name}]  fit_mode={fit_mode}  n={K}  R={R}  source={source}")
    print(f"   δ_rel over {len(ds)} layers:  "
          f"min={ds.min():.3f}  median={np.median(ds):.3f}  "
          f"mean={ds.mean():.3f}  max={ds.max():.3f}")
    most = rels[0]; least = rels[-1]
    print(f"   most tree-like layer {most[0]}: δ_rel={most[1]:.3f}   "
          f"least tree-like layer {least[0]}: δ_rel={least[1]:.3f}")
    return ds


CALIB = """
   calibration (synthetic):  tree 0.00 | hyperbolic 0.11 | euclidean-flat 0.22 | sphere 0.42 | ring 0.50
"""

if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "google__gemma-4-12b-it"
    print(f"model tensor: {tag}")
    print(CALIB)
    for nm in ["personas", "emotions", "months", "months_loop", "weekday_ord", "daypart"]:
        probe_manifold(nm, model_tag=tag)
