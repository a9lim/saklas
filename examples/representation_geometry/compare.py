"""A/B two ordinal probes — the framing effect.

    python examples/representation_geometry/compare.py years years_now [--model M]

Same span, same read position, only the elicitation framing differs. Tables decode
R² / Mantel / ordinal-axis strength / century seam for each, and writes a paired
decoded-vs-true figure. The headline: "name a year" smears the timeline across many
dimensions; "the current year is X" collapses it to a clean ~1-D recency line —
same information, different geometry.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import eucl_pdist, load_fit, loo_decode, pearson  # noqa: E402
from data import PROBES  # noqa: E402  # pyright: ignore[reportAttributeAccessIssue]  # sibling script module inserted above

FIGROOT = Path(__file__).parent / "figures"


def stats(probe: str, model: str):
    coords, labels, _ = load_fit(probe, model)
    m = re.search(r"(\d{4})", labels[0])
    assert m is not None
    years: Any = np.array([int(re.search(r"(\d{4})", l).group(1)) for l in labels])  # type: ignore[union-attr]
    o = np.argsort(years)
    coords, years = coords[o], years[o]
    K = len(years)
    pred, r2 = loo_decode(coords, years.astype(float))
    Dman = eucl_pdist(coords)
    Dyr = np.abs(years[:, None] - years[None]).astype(float)
    iu = np.triu_indices(K, 1)
    from scipy.stats import spearmanr
    U, S, _ = np.linalg.svd(coords - coords.mean(0), full_matrices=False)
    adj = np.array([Dman[i, i + 1] for i in range(K - 1)])
    jump = (Dman[np.where(years == 1999)[0][0], np.where(years == 2000)[0][0]]
            / np.median(adj)) if (years == 2000).any() else float("nan")
    res: Any = spearmanr(U[:, 0], years)
    return dict(D=coords.shape[1], var1=float((S**2 / (S**2).sum())[0]), r2=r2,
                med=float(np.median(np.abs(years - pred))),
                mantel=pearson(Dman[iu], Dyr[iu]),
                rho1=abs(res.statistic), jump=jump,
                years=years, pred=pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probes", nargs=2, help="two ordinal probe names (e.g. years years_now)")
    ap.add_argument("-m", "--model", default="google/gemma-3-4b-it")
    args = ap.parse_args()
    a, b = args.probes
    ra, rb = stats(a, args.model), stats(b, args.model)

    rows = [("intrinsic dim", "D", "{:d}"), ("PC1 variance", "var1", "{:.0%}"),
            ("decode year LOO R²", "r2", "{:+.3f}"), ("decode median |err| (yr)", "med", "{:.1f}"),
            ("Mantel (dist vs |Δyr|)", "mantel", "{:+.3f}"),
            ("|Spearman(PC1, year)|", "rho1", "{:.3f}"),
            ("century seam (×median step)", "jump", "{:.2f}")]
    print(f"{'metric':28s} {a:>14s} {b:>14s}")
    print("-" * 58)
    for lbl, k, fmt in rows:
        print(f"{lbl:28s} {fmt.format(ra[k]):>14s} {fmt.format(rb[k]):>14s}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, name, r in zip(axes, (a, b), (ra, rb)):
        ax.scatter(r["years"], r["pred"], c=r["years"], cmap="viridis", s=40,
                   edgecolor="k", linewidth=0.3)
        yr_arr: Any = r["years"]
        lim = (yr_arr.min() - 5, yr_arr.max() + 5)
        ax.plot(lim, lim, "k--", alpha=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("true year")
        ax.set_ylabel("LOO-decoded year")
        ax.set_title(f"{name} ({PROBES[name].framing})\n"
                     f"R²={r['r2']:.3f} · {r['D']}-D · PC1 {r['var1']:.0%} · Mantel {r['mantel']:.2f}")
        ax.grid(alpha=0.25)
    out = FIGROOT / f"compare_{a}_vs_{b}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
