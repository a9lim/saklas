"""Quick visual check: do the 12 months carry geometric structure?

Follow-up to the weekday probe (which scattered flat under templates, only
giving a ring under *embodiment* + ordinal framing — see
memory/project_weekday_embodiment_geometry.md). 12 months is ~1.7x the 7
weekdays, so the hope is a richer, more separable structure (seasons ought to
give a strong cyclic signal).

This script does the cheap-first-look version the user asked for: write the
"what month is it? / it is currently <MONTH>" template, pool the month-token
hidden state per layer, and PCA the 12 month centroids down to 3D so we can
eyeball whether anything ring-/season-shaped is there before committing to the
full manifold pipeline.

Two PCA views, because the docs are emphatic that raw activation PCA chases the
massive-activation (rogue) channels and shows noise:

  * WHITENED consensus PCA  — the saklas-correct, de-rogued lens. Per-layer
    Mahalanobis-whitened, node-mean-centered (K,K) Gram, averaged across all
    whitener-covered layers (the same consensus-Gram coord derivation a
    `manifold fit fit_mode=pca` runs), eigendecomposed to 3D. This is the view
    that actually answers "is the structure there".
  * RAW PCA  — what "a 3D PCA plot" literally means, on the single
    best-separating layer, shown alongside so the rogue-channel contrast is
    visible.

Plus a per-layer between-month whitened spread curve tr(G_L): if *no* layer
separates the months (the weekday failure), every bar is ~0 and it's not worth
pursuing.

Usage:
    python3 scripts/explore_months_pca.py [MODEL_ID] [--contexts N] [--out PATH]

Defaults to google/gemma-4-12b-it (the project's probe-geometry calibration
model). Pass google/gemma-3-4b-it for a faster look.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from saklas import SaklasSession
from saklas.core.vectors import _encode_and_capture_all_batch, _CAPTURE_BATCH

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
ABBR = [m[:3] for m in MONTHS]

# Paraphrase contexts. The user's canonical form is first; the rest hold
# question common-mode across months (like the A2 baseline prompts) so the
# pooled centroid reflects month identity, not one phrasing's surface tokens.
# Every assistant string ENDS in the month so the last-content-token pool lands
# on the month token (the slot), never trailing punctuation.
CONTEXTS = [
    ("What month is it?",                  "It is currently {m}"),
    ("What's the current month?",          "The current month is {m}"),
    ("Which month are we in right now?",   "We are in {m}"),
    ("Can you tell me the month?",         "Sure, it's {m}"),
    ("What month of the year is it?",      "It's {m}"),
    ("Right now, what month is it?",       "Right now it is {m}"),
]


def capture_month_centroids(session: SaklasSession, contexts: list[tuple[str, str]]):
    """Return {layer: (12, D)} fp32 CPU — month-token centroids, ctx-averaged."""
    prompts: list[str] = []
    responses: list[str] = []
    for m in MONTHS:
        for user, asst in contexts:
            prompts.append(user)
            responses.append(asst.format(m=m))

    # One forward per chunk; pool happens inside the capture hook at each row's
    # last content token (the month). system_msg="" — drop the length directive,
    # it's common-mode anyway and a one-word answer reads more naturally without.
    per_layer_chunks: dict[int, list[torch.Tensor]] = {}
    for start in range(0, len(prompts), _CAPTURE_BATCH):
        sl = slice(start, start + _CAPTURE_BATCH)
        cap = _encode_and_capture_all_batch(
            session.model, session.tokenizer,
            prompts[sl], responses[sl],
            session.layers, session.device,
            system_msg="",
        )
        for idx, t in cap.items():
            per_layer_chunks.setdefault(idx, []).append(t.to("cpu", torch.float32))
        if session.device.type == "mps":
            torch.mps.empty_cache()

    n_ctx = len(contexts)
    centroids: dict[int, torch.Tensor] = {}
    for idx, chunks in per_layer_chunks.items():
        stacked = torch.cat(chunks, dim=0)                 # (12*n_ctx, D)
        centroids[idx] = stacked.view(len(MONTHS), n_ctx, -1).mean(dim=1)  # (12, D)
    return centroids


def whitened_grams(session: SaklasSession, centroids: dict[int, torch.Tensor]):
    """Per-layer whitened (K,K) Grams + between-month spread tr(G_L)."""
    whitener = session.whitener
    if whitener is None:
        raise SystemExit(
            "no Mahalanobis whitener — neutral activations missing for this "
            "model; run an extract once to build them."
        )
    layers = sorted(idx for idx in centroids if idx in whitener)
    grams = {}
    spread = {}
    for idx in layers:
        xc = centroids[idx].to(torch.float32)
        xc = xc - xc.mean(dim=0, keepdim=True)            # node-mean-center
        g = whitener.subspace_gram(idx, xc)               # (K,K) = X̃ Σ⁻¹ X̃ᵀ
        grams[idx] = g
        spread[idx] = float(g.diagonal().sum())
    return grams, spread, layers


def consensus_over(grams: dict[int, torch.Tensor], subset: list[int]):
    return torch.stack([grams[idx] for idx in subset]).mean(dim=0)


def neighbor_metrics(gram: torch.Tensor):
    """Ring diagnostic straight from the whitened Gram.

    Pairwise whitened distances d²(i,j) = G_ii + G_jj - 2 G_ij (full-dim, no
    3D projection loss). For each month, the nearest *other* month: if the 12
    months form a calendar ring, every nearest neighbor is calendar-adjacent
    (cyclic distance 1). Random lexical scatter averages ~3.3.

    Returns (mean_nn_cyclic_dist, recall_at2, nn_index_list).
    """
    g = gram.numpy().astype(np.float64)
    diag = np.diag(g)
    d2 = diag[:, None] + diag[None, :] - 2 * g
    np.fill_diagonal(d2, np.inf)
    n = len(MONTHS)

    def cyc(i: int, j: int):
        d = abs(i - j)
        return min(d, n - d)

    nn = d2.argmin(axis=1)
    mean_nn = float(np.mean([cyc(i, nn[i]) for i in range(n)]))
    # recall@2: is a calendar neighbor (i±1) among the 2 nearest?
    hits = 0
    for i in range(n):
        top2 = np.argsort(d2[i])[:2]
        if any(cyc(i, j) == 1 for j in top2):
            hits += 1
    return mean_nn, hits / n, nn.tolist()


def coords_from_gram(gram: torch.Tensor, k: int = 3):
    """Classical-MDS / PCA-from-Gram: top-k eigvecs scaled by sqrt(eigval)."""
    g = gram.numpy().astype(np.float64)
    vals, vecs = np.linalg.eigh(g)                        # ascending
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    vals_pos = np.clip(vals, 0.0, None)
    coords = vecs[:, :k] * np.sqrt(vals_pos[:k])[None, :]
    ev_ratio = vals_pos[:k].sum() / max(vals_pos.sum(), 1e-12)
    return coords, vals_pos, ev_ratio


def raw_pca_at(centroids: dict[int, torch.Tensor], layer: int, k: int = 3):
    x = centroids[layer].numpy().astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :k] * s[:k][None, :]
    ev_ratio = (s[:k] ** 2).sum() / max((s ** 2).sum(), 1e-12)
    return coords, ev_ratio


def plot(wcoords: Any, w_ev: float, rcoords: Any, r_ev: float, best_layer: int,
         spread: dict[int, float], layers: list[int], band: list[int],
         nn_dist: float, recall2: float, chance: float, model_id: str, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    colors = cm.twilight(np.linspace(0, 1, len(MONTHS), endpoint=False))
    fig = plt.figure(figsize=(16, 10))
    verdict = ("RING-LIKE" if nn_dist < 1.6 else
               "PARTIAL" if nn_dist < 2.3 else "FLAT / lexical scatter")
    fig.suptitle(
        f"Month probe geometry — {model_id}   (semantic band L{band[0]}+)\n"
        f"nearest-neighbor cyclic-calendar dist {nn_dist:.2f} "
        f"(1.0=ring, ~{chance:.2f}=chance) · calendar recall@2 {recall2:.0%} "
        f"· top-3 var {w_ev:.0%}   →   {verdict}",
        fontsize=13,
    )

    def scatter3d(ax: Any, coords: Any, title: str):
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=colors, s=80, depthshade=False)
        # connect in calendar order — a ring shows as a closed loop, scatter
        # shows as a tangle.
        loop = np.vstack([coords, coords[:1]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                color="0.6", lw=0.8, alpha=0.7)
        for i, ab in enumerate(ABBR):
            ax.text(coords[i, 0], coords[i, 1], coords[i, 2], ab, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # whitened from two angles (3D-on-2D needs a second view)
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    scatter3d(ax1, wcoords, "whitened band consensus PCA (view A)")
    ax1.view_init(elev=20, azim=35)
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    scatter3d(ax2, wcoords, "whitened band consensus PCA (view B)")
    ax2.view_init(elev=20, azim=125)

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    scatter3d(ax3, rcoords, f"raw PCA @ best band layer {best_layer}")
    ax3.view_init(elev=20, azim=35)

    # PC1-PC2 plane of the whitened fit (flat read of the ring)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(wcoords[:, 0], wcoords[:, 1], c=colors, s=80)
    loop = np.vstack([wcoords[:, :2], wcoords[:1, :2]])
    ax4.plot(loop[:, 0], loop[:, 1], color="0.6", lw=0.8, alpha=0.7)
    for i, ab in enumerate(ABBR):
        ax4.annotate(ab, (wcoords[i, 0], wcoords[i, 1]), fontsize=8)
    ax4.set_title("whitened PC1–PC2 (top plane)", fontsize=10)
    ax4.set_aspect("equal", adjustable="datalim")

    # per-layer between-month whitened spread
    ax5 = fig.add_subplot(2, 3, 5)
    xs = layers
    ys = [spread[idx] for idx in layers]
    bar_colors = ["seagreen" if idx in band else "lightsteelblue" for idx in layers]
    ax5.bar(xs, ys, color=bar_colors)
    ax5.axvspan(band[0] - 0.5, band[-1] + 0.5, color="seagreen", alpha=0.08)
    ax5.axvline(best_layer, color="crimson", lw=1, ls="--",
                label=f"best band = L{best_layer}")
    ax5.set_xlabel("layer")
    ax5.set_ylabel("tr(G_L)  (whitened σ² units)")
    ax5.set_title("between-month separation per layer\n"
                  "(green = semantic band; early spikes = lexical)", fontsize=9)
    ax5.legend(fontsize=8)

    # cyclic colorbar legend (month order)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(colors[None, :, :], aspect="auto",
               extent=[0, len(MONTHS), 0, 1])
    ax6.set_xticks(np.arange(len(MONTHS)) + 0.5)
    ax6.set_xticklabels(ABBR, rotation=45, fontsize=8)
    ax6.set_yticks([])
    ax6.set_title("month → color (cyclic): a season ring traces the rainbow",
                  fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", nargs="?", default="google/gemma-4-12b-it")
    ap.add_argument("--contexts", type=int, default=len(CONTEXTS),
                    help="how many paraphrase contexts to pool (1 = canonical only)")
    ap.add_argument("--out", default="scripts/out/months_pca.png")
    ap.add_argument("--band-start", type=float, default=0.4,
                    help="depth fraction where the 'semantic' band begins "
                         "(early layers separate months lexically, by token)")
    args = ap.parse_args()

    contexts = CONTEXTS[: max(1, args.contexts)]
    print(f"loading {args.model_id} ...")
    with SaklasSession.from_pretrained(args.model_id, device="auto") as session:
        print(f"capturing {len(MONTHS)} months x {len(contexts)} contexts ...")
        centroids = capture_month_centroids(session, contexts)
        grams, spread, layers = whitened_grams(session, centroids)

    n_layers = max(layers) + 1
    band = [idx for idx in layers if idx >= args.band_start * n_layers]
    chance = sum(min(d, len(MONTHS) - d) for d in range(1, len(MONTHS))) / (len(MONTHS) - 1)

    print("\nper-layer between-month whitened spread tr(G_L) "
          "(early = lexical / token identity, mid-late = semantic):")
    for idx in layers:
        bar = "#" * int(40 * spread[idx] / max(spread.values()))
        tag = " [band]" if idx in band else ""
        print(f"  L{idx:>2} {spread[idx]:9.2f} {bar}{tag}")

    results = {}
    for name, subset in [("all-layer", layers), (f"band L{band[0]}+", band)]:
        cons = consensus_over(grams, subset)
        coords, vals, ev = coords_from_gram(cons, k=3)
        nn_dist, recall2, nn = neighbor_metrics(cons)
        results[name] = dict(cons=cons, coords=coords, vals=vals, ev=ev,
                             nn_dist=nn_dist, recall2=recall2, nn=nn)
        print(f"\n=== {name} consensus ===")
        print(f"  top-3 explained var:        {ev:.1%}")
        print(f"  nearest-neighbor cyclic-cal dist: {nn_dist:.2f}  "
              f"(1.0 = perfect ring, ~{chance:.2f} = chance)")
        print(f"  calendar-neighbor recall@2: {recall2:.0%}")
        print("  each month -> nearest month: "
              + ", ".join(f"{ABBR[i]}->{ABBR[nn[i]]}" for i in range(len(MONTHS))))

    # primary plot = the semantic band (the honest 'is the structure there' view)
    prim = results[f"band L{band[0]}+"]
    best_band_layer = max(band, key=lambda i: spread[i])
    rcoords, r_ev = raw_pca_at(centroids, best_band_layer, k=3)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot(prim["coords"], prim["ev"], rcoords, r_ev, best_band_layer,
         spread, layers, band, prim["nn_dist"], prim["recall2"], chance,
         args.model_id, args.out)

    dump = args.out.rsplit(".", 1)[0] + ".json"
    with open(dump, "w") as f:
        json.dump({
            "model_id": args.model_id,
            "months": MONTHS,
            "chance_nn_cyclic_dist": chance,
            "band_layers": band,
            "results": {
                name: {
                    "coords_3d": r["coords"].tolist(),
                    "eigvals": r["vals"].tolist(),
                    "top3_explained_var": float(r["ev"]),
                    "nn_cyclic_cal_dist": r["nn_dist"],
                    "calendar_recall_at2": r["recall2"],
                    "nearest_month": {ABBR[i]: ABBR[r["nn"][i]]
                                      for i in range(len(MONTHS))},
                } for name, r in results.items()
            },
            "per_layer_spread": {str(i): spread[i] for i in layers},
        }, f, indent=2)
    print(f"\nwrote {dump}")


if __name__ == "__main__":
    main()
