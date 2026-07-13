"""Read a fitted probe's layout and report its geometry.

    python examples/representation_geometry/analyze.py <probe> [--model M]

Dispatches on the probe kind:
  - geographic (countries) — Mantel of manifold distance vs great-circle distance,
    and a leave-one-out linear decode of (lat, lon). The negative control.
  - ordinal (years*) — Mantel vs |Δyear|, a leave-one-out decode of the year, the
    century seam, decade-phase residual; and for ``years_now_future`` the
    future-cliff diagnostics that locate the model's sense of "now".

Figures land in ``figures/<probe>/``. Requires a fitted manifold — run ``author.py``
then ``saklas manifold fit <probe> -m <model>`` first.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from data import PROBES  # noqa: E402  # pyright: ignore[reportAttributeAccessIssue]  # sibling script module inserted above

from saklas.io.paths import safe_model_id  # noqa: E402

FIGROOT = Path(__file__).parent / "figures"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def load_fit(probe: str, model_id: str):
    folder = Path.home() / ".saklas/manifolds/local" / probe
    tensor = folder / f"{safe_model_id(model_id)}.safetensors"
    if not tensor.exists():
        sys.exit(f"no fit at {tensor}\n  run: saklas manifold fit {probe} -m {model_id}")
    labels = json.load(open(tensor.with_suffix(".json")))["node_labels"]
    with safe_open(str(tensor), framework="np") as f:
        coords = f.get_tensor("node_coords").astype(np.float64)
    sidecar = json.load(open(tensor.with_suffix(".json")))
    return coords, labels, sidecar


def slug(v: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", v.strip().lower()).strip("_")


def pearson(a: Any, b: Any) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def eucl_pdist(X: Any) -> Any:
    d = X[:, None, :] - X[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def haversine(lat: Any, lon: Any) -> Any:
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    a = (np.sin((la - la.T) / 2) ** 2
         + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def loo_decode(coords: Any, target: Any):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(coords)
    pred = np.zeros_like(target, dtype=float)
    for tr, te in LeaveOneOut().split(Xs):
        m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xs[tr], target[tr])
        pred[te] = m.predict(Xs[te])
    r2 = 1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum()
    return pred, float(r2)


# --------------------------------------------------------------------------- #
# geographic (countries)
# --------------------------------------------------------------------------- #
def analyze_geographic(probe: Any, coords: Any, labels: Any, fig: Any) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geo = probe.geo
    s2n = {slug(n): n for n in geo}
    names = [s2n[l] for l in labels]
    lat = np.array([geo[n][0] for n in names])
    lon = np.array([geo[n][1] for n in names])
    region = [geo[n][2] for n in names]
    K = len(names)

    Dman, Dgeo = eucl_pdist(coords), haversine(lat, lon)
    iu = np.triu_indices(K, 1)
    mantel = pearson(Dman[iu], Dgeo[iu])
    plat, r2lat = loo_decode(coords, lat)
    plon, r2lon = loo_decode(coords, lon)
    print(f"  Mantel (dist vs great-circle) = {mantel:+.3f}")
    print(f"  lat/lon LOO R²                = {r2lat:+.3f} / {r2lon:+.3f}")
    # europe-only Mantel (confound-free dense region)
    eur = [i for i, n in enumerate(names) if "Europe" in geo[n][2]]
    if len(eur) > 4:
        ce = coords[eur]
        Dm = eucl_pdist(ce)
        Dg = haversine(lat[eur], lon[eur])
        je = np.triu_indices(len(eur), 1)
        print(f"  Europe-only Mantel ({len(eur)})    = {pearson(Dm[je], Dg[je]):+.3f}")

    regions = sorted(set(region))
    cmap = plt.get_cmap("tab20")
    rcol = {r: cmap(i % 20) for i, r in enumerate(regions)}
    Xc = coords - coords.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pc = U[:, :2] * S[:2]
    var = (S ** 2 / (S ** 2).sum())[:2]
    f, ax = plt.subplots(figsize=(13, 10))
    for r in regions:
        m = [i for i in range(K) if region[i] == r]
        ax.scatter(pc[m, 0], pc[m, 1], c=[rcol[r]], s=70, label=r,
                   edgecolor="k", linewidth=0.4)
    for i in range(K):
        ax.annotate(names[i], (pc[i, 0], pc[i, 1]), fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"countries — top-2 PCA (of {coords.shape[1]}-D)  ·  var "
                 f"{var[0]:.0%}/{var[1]:.0%}\nMantel {mantel:+.2f}  ·  "
                 f"lat/lon LOO R² {r2lat:.2f}/{r2lon:.2f}  (geography barely shows)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, ncol=2)
    f.tight_layout()
    f.savefig(fig / "pca_layout.png", dpi=130)
    plt.close(f)
    print(f"  wrote {fig/'pca_layout.png'}")


# --------------------------------------------------------------------------- #
# ordinal (years*)
# --------------------------------------------------------------------------- #
def analyze_ordinal(probe: Any, coords: Any, labels: Any, sidecar: Any, fig: Any) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    m = re.search(r"(\d{4})", labels[0])
    assert m is not None
    years = np.array([int(re.search(r"(\d{4})", l).group(1)) for l in labels])  # type: ignore[union-attr]
    o = np.argsort(years)
    coords, years = coords[o], years[o]
    K, D = coords.shape

    pred, r2 = loo_decode(coords, years.astype(float))
    med = np.median(np.abs(years - pred))
    Dman = eucl_pdist(coords)
    Dyr = np.abs(years[:, None] - years[None]).astype(float)
    iu = np.triu_indices(K, 1)
    mantel = pearson(Dman[iu], Dyr[iu])
    U, S, Vt = np.linalg.svd(coords - coords.mean(0), full_matrices=False)
    pc = U * S
    res: Any = spearmanr(pc[:, 0], years)
    if res.statistic < 0:
        pc[:, 0] *= -1
    var = S ** 2 / (S ** 2).sum()
    rho1 = abs(spearmanr(pc[:, 0], years).statistic)  # type: ignore[union-attr]
    print(f"  resolved geometry = {sidecar.get('resolved_fit_mode')} "
          f"{sidecar.get('domain')}")
    print(f"  decode year LOO R² = {r2:+.3f}  (median |err| {med:.1f} yr)")
    print(f"  Mantel (dist vs |Δyr|) = {mantel:+.3f}   |Spearman(PC1,year)| = {rho1:.3f}")
    adj = np.array([Dman[i, i + 1] for i in range(K - 1)])
    if (years == 2000).any():
        jump = Dman[np.where(years == 1999)[0][0], np.where(years == 2000)[0][0]]
        print(f"  century seam (1999→2000) = {jump / np.median(adj):.1f}× median step")

    # decoded-vs-true (money shot)
    f, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(years, pred, c=years, cmap="viridis", s=45, edgecolor="k", linewidth=0.3)
    years_min: Any = years.min()
    years_max: Any = years.max()
    lim = (years_min - 5, years_max + 5)
    ax.plot(lim, lim, "k--", alpha=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("true year")
    ax.set_ylabel("LOO-decoded year")
    ax.set_title(f"{probe.name} ({probe.framing}) — reading the year off the activation\n"
                 f"R²={r2:.3f}  ·  median {med:.1f} yr  ·  {D}-D  ·  PC1 {var[0]:.0%}")
    ax.grid(alpha=0.25)
    f.tight_layout()
    f.savefig(fig / "decoded_vs_true.png", dpi=130)
    plt.close(f)
    print(f"  wrote {fig/'decoded_vs_true.png'}")

    if probe.future:
        analyze_cliff(probe, coords, years, pred, r2, adj, pc, var, fig)


def analyze_cliff(probe: Any, coords: Any, years: Any, pred: Any, r2: Any, adj: Any, pc: Any, var: Any, fig: Any) -> None:
    import matplotlib.pyplot as plt
    K = len(years)
    med_adj = np.median(adj)
    ymid = years[:-1]
    modern = ymid >= 2010
    cliff_year = int(ymid[modern][np.argmax(adj[modern])])
    cliff_mag = adj[modern].max() / med_adj
    future = years >= 2024
    fut = np.where(years >= 2026)[0]
    past = np.where((years >= 2026 - 2 * len(fut)) & (years < 2026))[0]

    def spread(idx: Any) -> float:
        c = coords[idx]
        dm = eucl_pdist(c)
        return dm[np.triu_indices(len(idx), 1)].mean()

    pre = (years >= 2015) & (years < 2024)
    post = years >= 2024
    resid_pre = float((pred[pre] - years[pre]).mean())
    resid_post = float((pred[post] - years[post]).mean())
    print("  --- future cliff ---")
    print(f"  largest 2010+ adjacent jump: {cliff_year}→{cliff_year+1} "
          f"= {cliff_mag:.1f}× median  (the geometric 'now')")
    print(f"  decode residual flips sign at the present: "
          f"2015–2023 mean {resid_pre:+.1f} yr (crowds toward now), "
          f"2024+ mean {resid_post:+.1f} yr (folds back)")
    print(f"  future spread (2026+) {spread(fut):.2f} vs recent-past {spread(past):.2f}")

    f, ax = plt.subplots(3, 1, figsize=(12, 14))
    ax[0].plot(ymid, adj / med_adj, "-o", ms=3, color="purple")
    ax[0].axhline(1, color="gray", ls=":", alpha=0.6)
    ax[0].axvline(cliff_year + 0.5, color="red", ls="--", label=f"'now' cliff ~{cliff_year}")
    if (ymid == 1999).any():
        ax[0].axvline(1999.5, color="teal", ls="--", alpha=0.7, label="century seam 2000")
    ax[0].set_xlabel("year")
    ax[0].set_ylabel("adjacent step (×median)")
    ax[0].set_title("adjacent-distance profile — spikes mark regime breaks")
    ax[0].legend()
    ax[0].grid(alpha=0.2)

    ax[1].scatter(years[~future], pred[~future], c=years[~future], cmap="viridis",
                  s=35, edgecolor="k", linewidth=0.3, label="past")
    ax[1].scatter(years[future], pred[future], facecolor="none", edgecolor="red",
                  s=70, linewidth=1.5, label="2024+ (future)")
    ymin: Any = years.min()
    ymax: Any = years.max()
    lim = (ymin - 5, ymax + 5)
    ax[1].plot(lim, lim, "k--", alpha=0.5)
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("true year")
    ax[1].set_ylabel("LOO-decoded year")
    ax[1].set_title(f"decode (R²={r2:.2f}) — future folds below the diagonal: can't place past 'now'")
    ax[1].legend()
    ax[1].grid(alpha=0.25)

    sc = ax[2].scatter(pc[:, 0], pc[:, 1], c=years, cmap="viridis", s=40,
                       edgecolor="k", linewidth=0.3)
    ax[2].scatter(pc[future, 0], pc[future, 1], facecolor="none", edgecolor="red",
                  s=80, linewidth=1.4)
    ax[2].plot(pc[:, 0], pc[:, 1], "-", color="gray", alpha=0.3, lw=0.7, zorder=0)
    for i in range(0, K, 10):
        ax[2].annotate(str(years[i]), (pc[i, 0], pc[i, 1]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=ax[2], label="year")
    ax[2].set_title(f"top-2 PCA (var {var[0]:.0%}/{var[1]:.0%}) — red ring = 2024+")
    ax[2].set_xlabel("PC1 (year-oriented)")
    ax[2].set_ylabel("PC2")
    f.tight_layout()
    f.savefig(fig / "cliff.png", dpi=130)
    plt.close(f)
    print(f"  wrote {fig/'cliff.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probe", choices=list(PROBES))
    ap.add_argument("-m", "--model", default="google/gemma-3-4b-it")
    args = ap.parse_args()
    probe = PROBES[args.probe]
    fig = FIGROOT / args.probe
    fig.mkdir(parents=True, exist_ok=True)
    coords, labels, sidecar = load_fit(args.probe, args.model)
    print(f"[{args.probe}]  {len(labels)} nodes, {coords.shape[1]}-D  ({args.model})")
    if probe.kind == "geographic":
        analyze_geographic(probe, coords, labels, fig)
    else:
        analyze_ordinal(probe, coords, labels, sidecar, fig)


if __name__ == "__main__":
    main()
