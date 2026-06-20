"""Force the embodiment months onto a 1D periodic loop and look for the pringle.

The embodiment-12b centroids are calendar-ordered (recall@2 100%). If we *force*
the calendar-circle parameterization (theta_k = 2*pi*k/12) and ask how the
centroids actually sit in activation space, two outcomes:

  * FLAT circle  — the ring lives in a plane (out-of-plane variance ~0), or the
    only out-of-plane component is period-1 (a tilt of the flat circle).
  * PRINGLE / saddle — a strong out-of-plane component that is PERIOD-2 in the
    year (up-down-up-down: e.g. equinoxes up, solstices down). A circle lifted by
    cos(2 theta) is a saddle — the year as a Pringle chip.

This pools the embodiment centroids (all-layer concat), splits each into the
forced-circle plane + the dominant out-of-plane axis, and regresses that axis on
period-1 (cos/sin theta) vs period-2 (cos/sin 2theta) to call flat-vs-pringle.
Then it authors + fits the real saklas periodic 1D BoxDomain manifold
(`local/months_loop`) as the forced-loop steering artifact and reports whether
that topology fits coherently.
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
from typing import Any

import numpy as np
import torch

from saklas import SaklasSession
from saklas.core.manifold import (
    BoxAxis, BoxDomain, compute_node_centroid,
)
from saklas.core.vectors import _load_baseline_prompts
from saklas.io.manifolds import create_manifold_folder

MODEL = "google/gemma-4-12B-it"
SAFE = "google__gemma-4-12B-it"
MONTHS = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]
ABBR = [m[:3] for m in MONTHS]
EMB = "/Users/a9lim/.saklas/manifolds/local/months_seasonal"
LOOP_NS, LOOP_NAME = "local", "months_loop"


def pool_centroids(session: SaklasSession, prompts: list[str]):
    """{month: concat-all-layer centroid} from the embodiment corpora."""
    fs = sorted(glob.glob(f"{EMB}/nodes/*.json"),
                key=lambda p: int(os.path.basename(p)[:2]))
    X = []
    for f in fs:
        responses = json.load(open(f))
        c = compute_node_centroid(session.model, session.tokenizer,
                                  session.layers, session.device, responses, prompts)
        X.append(torch.cat([c[l] for l in sorted(c)]).numpy())
        print(f"    pooled {ABBR[len(X)-1]}")
    return np.stack(X)  # (12, D)


def pringle_decompose(X: np.ndarray):
    """Force the calendar circle; return ring coords, out-of-plane axis, fits."""
    Xc = X - X.mean(0, keepdims=True)
    th = np.array([2 * math.pi * k / 12 for k in range(12)])
    c1, s1 = np.cos(th), np.sin(th)
    c2, s2 = np.cos(2 * th), np.sin(2 * th)

    # ring plane = activation directions that co-vary with cos/sin theta
    a_cos = Xc.T @ c1
    a_sin = Xc.T @ s1
    R, _ = np.linalg.qr(np.stack([a_cos, a_sin], 1))  # (D, 2) orthonormal
    ring = Xc @ R                                       # (12, 2) — should be a circle
    resid = Xc - ring @ R.T                             # out-of-plane part
    # dominant out-of-plane direction
    U, S, Vt = np.linalg.svd(resid, full_matrices=False)
    z = U[:, 0] * S[0]                                  # (12,) out-of-plane coord

    def r2(design: list[np.ndarray]):
        A = np.column_stack(design + [np.ones(12)])
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        pred = A @ coef
        ss = ((z - z.mean()) ** 2).sum()
        return 1 - ((z - pred) ** 2).sum() / ss
    r2_p1 = r2([c1, s1])      # period-1 (tilt of flat circle)
    r2_p2 = r2([c2, s2])      # period-2 (PRINGLE / saddle)

    ring_var = (ring ** 2).sum()
    oop_var = (resid ** 2).sum()
    return dict(ring=ring, z=z, R=R, resid=resid,
                r2_period1=float(r2_p1), r2_period2=float(r2_p2),
                oop_frac=float(oop_var / (ring_var + oop_var)))


def plot(dec: dict[str, Any], out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    colors = cm.twilight(np.linspace(0, 1, 12, endpoint=False))
    ring, z = dec["ring"], dec["z"]
    # normalize z scale to the ring radius for an honest 3D aspect
    rscale = np.sqrt((ring ** 2).sum(1)).mean()
    zz = z / (np.abs(z).max() + 1e-9) * rscale

    fig = plt.figure(figsize=(16, 7.5))
    verdict = ("PRINGLE (period-2 saddle)" if dec["r2_period2"] > 0.5
               and dec["r2_period2"] > dec["r2_period1"]
               else "tilted flat circle (period-1)" if dec["r2_period1"] > 0.5
               else "flat-ish circle")
    fig.suptitle("Embodiment-12b months forced onto the calendar circle — is it a pringle?\n"
                 f"out-of-plane fraction {dec['oop_frac']:.0%}  ·  "
                 f"out-of-plane R² period-1 (tilt) {dec['r2_period1']:.2f}  vs  "
                 f"period-2 (saddle) {dec['r2_period2']:.2f}   →   {verdict}", fontsize=12)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    # smooth loop through the 12 points (periodic, by theta)
    th = np.array([2*math.pi*k/12 for k in range(12)])
    tt = np.linspace(0, 2*math.pi, 240)
    from numpy import interp
    thx = np.concatenate([th, [2*math.pi]])
    def wrap(v: np.ndarray): return interp(tt, thx, np.concatenate([v, v[:1]]), period=2*math.pi)
    ax.plot(wrap(ring[:,0]), wrap(ring[:,1]), wrap(zz), color="0.6", lw=1.2)
    ax.scatter(ring[:,0], ring[:,1], zz, c=colors, s=220, depthshade=False,
               edgecolor="k", lw=0.4)
    for i, ab in enumerate(ABBR):
        ax.text(ring[i,0], ring[i,1], zz[i], "  "+ab, fontsize=9, weight="bold")
    ax.set_title("forced calendar circle (x,y) lifted by out-of-plane axis (z)", fontsize=11)
    ax.set_xlabel("ring cos")
    ax.set_ylabel("ring sin")
    ax.set_zlabel("out-of-plane")
    ax.view_init(elev=22, azim=40)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axhline(0, color="0.8", lw=0.8)
    ax2.plot(range(12), z, "-o", color="0.5", zorder=1)
    ax2.scatter(range(12), z, c=colors, s=160, zorder=2, edgecolor="k", lw=0.4)
    # overlay best period-2 cosine for visual
    c2 = np.cos(2*th)
    s2 = np.sin(2*th)
    A = np.column_stack([c2, s2, np.ones(12)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    fine = np.linspace(0,11,240)
    ax2.plot(fine, (np.column_stack([np.cos(2*2*math.pi*fine/12),
             np.sin(2*2*math.pi*fine/12), np.ones_like(fine)])@coef),
             "r--", lw=1, label=f"period-2 fit (R²={dec['r2_period2']:.2f})")
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(ABBR, fontsize=8)
    ax2.set_ylabel("out-of-plane coord")
    ax2.legend(fontsize=9)
    ax2.set_title("out-of-plane vs month — period-2 = saddle rim (up-down-up-down)", fontsize=11)
    fig.tight_layout(rect=(0,0,1,0.93))
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def fit_saklas_loop(session: SaklasSession):
    """Author + fit the real periodic 1D BoxDomain manifold on embodiment corpora."""
    folder = f"/Users/a9lim/.saklas/manifolds/{LOOP_NS}/{LOOP_NAME}"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    domain = BoxDomain([BoxAxis(name="month", periodic=True, period=12.0,
                                lo=0.0, hi=12.0)])
    nodes = []
    for k, m in enumerate(MONTHS):
        responses = json.load(open(f"{EMB}/nodes/{k:02d}_{m.lower()}.json"))
        nodes.append({"label": m.lower(), "coords": [float(k)],
                      "statements": responses})
    path, advisories = create_manifold_folder(
        LOOP_NS, LOOP_NAME,
        "Months forced onto a periodic 1D loop (embodiment corpora).",
        domain.to_spec(), nodes,
    )
    if advisories:
        print("  authoring advisories:", advisories)
    print("  fitting forced periodic loop ...")
    session.fit(path, on_progress=lambda m: print("    ", m))
    sc = json.load(open(f"{path}/{SAFE}.json"))
    share = sc.get("mahalanobis_share_per_layer")
    return sc, share


def main():
    prompts = _load_baseline_prompts()
    print(f"loading {MODEL} ...")
    with SaklasSession.from_pretrained(MODEL, device="auto") as session:
        print("pooling embodiment centroids ...")
        X = pool_centroids(session, prompts)
        dec = pringle_decompose(X)
        print(f"\nout-of-plane fraction: {dec['oop_frac']:.1%}")
        print(f"out-of-plane R² period-1 (tilt):    {dec['r2_period1']:.2f}")
        print(f"out-of-plane R² period-2 (PRINGLE): {dec['r2_period2']:.2f}")
        plot(dec, "scripts/out/months_pringle.png")
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in dec.items()},
                  open("scripts/out/months_pringle.json", "w"), indent=2)

        print("\n=== forcing the saklas periodic 1D loop fit ===")
        sc, share = fit_saklas_loop(session)
        print(f"resolved domain: {json.dumps(sc.get('domain'))}")
        print(f"fit_mode: {sc.get('fit_mode')}  feature_space: {sc.get('feature_space')}")
        if share:
            import numpy as _np
            sv = _np.array(list(share.values()) if isinstance(share, dict) else share)
            print(f"per-layer Mahalanobis share: mean {sv.mean():.3f}, "
                  f"max {sv.max():.3f} (periodic loop fit coherence)")


if __name__ == "__main__":
    main()
