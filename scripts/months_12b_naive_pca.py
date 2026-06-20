"""Naive 3D PCA of the 12b month centroids — template vs embodiment.

Just-curious view: no whitening, no consensus Gram, no curvature fit — take the
per-month pooled centroid (concatenated across all layers, the full fingerprint),
mean-center the 12, plain SVD -> top 3. Side by side for the template corpus
(`local/months`, "it is currently <MONTH>") and the embodiment corpus
(`local/months_seasonal`, "I am January..."). One 12b load, pooled via the
pipeline's own compute_node_centroid so the centroids match what the fit saw.
"""

from __future__ import annotations

import json
import numpy as np
import torch

from saklas import SaklasSession
from saklas.core.manifold import compute_node_centroid
from saklas.core.vectors import _load_baseline_prompts

MODEL = "google/gemma-4-12B-it"
SAFE = "google__gemma-4-12B-it"
MONTHS = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]
ABBR = [m[:3] for m in MONTHS]
MAN = "/Users/a9lim/.saklas/manifolds/local"


def node_files(folder):
    import glob, os
    fs = sorted(glob.glob(f"{MAN}/{folder}/nodes/*.json"),
                key=lambda p: int(os.path.basename(p)[:2]))
    return [json.load(open(f)) for f in fs]   # list[ list[str] ], month-ordered


def pool_set(session, corpora, prompts):
    """{month_idx: concat-over-layers centroid vector} via compute_node_centroid."""
    vecs = []
    for i, responses in enumerate(corpora):
        c = compute_node_centroid(
            session.model, session.tokenizer, session.layers, session.device,
            responses, prompts,
        )  # {layer: (D,)}
        flat = torch.cat([c[l] for l in sorted(c)]).numpy()  # concat all layers
        vecs.append(flat)
        print(f"    pooled {ABBR[i]}")
    return np.stack(vecs)  # (12, D_total)


def naive_pca3(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = U[:, :3] * S[:3][None, :]
    ev = (S[:3]**2).sum() / (S**2).sum()
    return coords, float(ev)


def main():
    # template prompts = the 6 context user-turns, in corpus order
    ctx = json.load(open("scripts/out/months_contexts.json"))
    tmpl_prompts = [c["user"] for c in ctx]
    base_prompts = _load_baseline_prompts()

    print(f"loading {MODEL} ...")
    with SaklasSession.from_pretrained(MODEL, device="auto") as session:
        print("pooling TEMPLATE centroids (6 ctx each) ...")
        Xt = pool_set(session, node_files("months"), tmpl_prompts)
        print("pooling EMBODIMENT centroids (48 resp each) ...")
        Xe = pool_set(session, node_files("months_seasonal"), base_prompts)

    Ct, evt = naive_pca3(Xt)
    Ce, eve = naive_pca3(Xe)
    print(f"\ntemplate naive top-3 explained var:   {evt:.0%}")
    print(f"embodiment naive top-3 explained var: {eve:.0%}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    colors = cm.twilight(np.linspace(0, 1, 12, endpoint=False))

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("Naive 3D PCA of 12b month centroids (all layers concat, no whitening)\n"
                 f"TEMPLATE 'it is currently X' (top-3 var {evt:.0%})   vs   "
                 f"EMBODIMENT 'I am January...' (top-3 var {eve:.0%})", fontsize=13)

    def scat(ax, C, title, azim):
        loop = np.vstack([C, C[:1]])
        ax.plot(loop[:,0], loop[:,1], loop[:,2], color="0.6", lw=0.8, alpha=0.7)
        ax.scatter(C[:,0], C[:,1], C[:,2], c=colors, s=90, depthshade=False)
        for i, ab in enumerate(ABBR):
            ax.text(C[i,0], C[i,1], C[i,2], ab, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=18, azim=azim)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    for col, (C, name) in enumerate([(Ct, "TEMPLATE"), (Ce, "EMBODIMENT")]):
        for row, azim in enumerate([35, 125]):
            ax = fig.add_subplot(2, 2, row*2 + col + 1, projection="3d")
            scat(ax, C, f"{name} (view {'A' if row==0 else 'B'})", azim)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = "scripts/out/months_12b_naive_pca.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    json.dump({"template_coords": Ct.tolist(), "embodiment_coords": Ce.tolist(),
               "template_var": evt, "embodiment_var": eve, "months": MONTHS},
              open("scripts/out/months_12b_naive_pca.json", "w"), indent=2)


if __name__ == "__main__":
    main()
