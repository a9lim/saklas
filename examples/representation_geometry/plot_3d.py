"""Rotating 3D PCA of a probe's centroids (top-3 of the fitted layout).

    python examples/representation_geometry/plot_3d.py <probe> [--model M]

Writes ``figures/<probe>/pca3d.gif`` (and a static PNG). Ordinal probes are
coloured by year with consecutive years linked; ``countries`` is coloured by
region, with multi-word names marked ✖ so the lexical axis is visible. The GIFs
are regenerated on demand — they are deliberately not committed (a few MB each).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from data import PROBES  # noqa: E402

from saklas.io.paths import safe_model_id  # noqa: E402

FIGROOT = Path(__file__).parent / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probe", choices=list(PROBES))
    ap.add_argument("-m", "--model", default="google/gemma-3-4b-it")
    args = ap.parse_args()
    probe = PROBES[args.probe]
    fig = FIGROOT / args.probe
    fig.mkdir(parents=True, exist_ok=True)

    tensor = (Path.home() / ".saklas/manifolds/local" / args.probe
              / f"{safe_model_id(args.model)}.safetensors")
    if not tensor.exists():
        sys.exit(f"no fit at {tensor} — author + fit {args.probe} first")
    labels = json.load(open(tensor.with_suffix(".json")))["node_labels"]
    with safe_open(str(tensor), framework="np") as f:
        coords = f.get_tensor("node_coords").astype(np.float64)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    Xc = coords - coords.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pc = U[:, :3] * S[:3]
    var = (S ** 2 / (S ** 2).sum())[:3]
    K = coords.shape[0]

    figh = plt.figure(figsize=(11, 9))
    ax = figh.add_subplot(111, projection="3d")

    if probe.kind == "ordinal":
        years = np.array([int(re.search(r"(\d{4})", l).group(1)) for l in labels])
        o = np.argsort(years)
        pc, years = pc[o], years[o]
        from scipy.stats import spearmanr
        if spearmanr(pc[:, 0], years).statistic < 0:
            pc[:, 0] *= -1
        ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], "-", color="gray", alpha=0.4, lw=0.9)
        p = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=years, cmap="viridis",
                       s=45, edgecolor="k", linewidth=0.3)
        for i in range(0, K, 10):
            ax.text(pc[i, 0], pc[i, 1], pc[i, 2], str(years[i]), fontsize=7)
        figh.colorbar(p, label="year", shrink=0.6)
    else:
        geo = probe.geo
        assert geo is not None  # geographic probes carry a geo table
        s2n = {re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_"): n for n in geo}
        names = [s2n[l] for l in labels]
        region = [geo[n][2] for n in names]
        multiword = np.array([" " in n for n in names])
        regions = sorted(set(region))
        cmap = plt.get_cmap("tab20")
        rcol = {r: cmap(i % 20) for i, r in enumerate(regions)}
        for r in regions:
            for mw, mark, sz in ((False, "o", 55), (True, "X", 90)):
                m = [i for i in range(K) if region[i] == r and multiword[i] == mw]
                if m:
                    ax.scatter(pc[m, 0], pc[m, 1], pc[m, 2], c=[rcol[r]], marker=mark,
                               s=sz, edgecolor="k", linewidth=0.4,
                               label=r if not mw else None)
        for i in range(K):
            ax.text(pc[i, 0], pc[i, 1], pc[i, 2], names[i], fontsize=6, alpha=0.75)
        ax.legend(fontsize=7, ncol=2, loc="upper left")

    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.set_zlabel(f"PC3 ({var[2]:.0%})")
    ax.set_title(f"{args.probe} — top-3 manifold PCA")
    figh.tight_layout()
    figh.savefig(fig / "pca3d_static.png", dpi=130)

    def _update(frame: int):
        ax.view_init(elev=18, azim=frame)
        return ()

    animation.FuncAnimation(figh, _update, frames=range(0, 360, 4), interval=80).save(
        fig / "pca3d.gif", writer=animation.PillowWriter(fps=14))
    print(f"wrote {fig/'pca3d.gif'} and pca3d_static.png")


if __name__ == "__main__":
    main()
