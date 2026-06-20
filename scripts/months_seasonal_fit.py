"""Seasonal-embodiment month corpora -> real saklas manifold fit.

Follow-up to scripts/explore_months_pca.py. The neutral "what month is it"
template recovered a clean calendar *arc* but didn't close the Dec<->Jan ring
(see memory/project_month_geometry.md). Hypothesis: a seasonal-embodiment
framing — the model speaks AS the month, inflecting with weather/season/mood —
pulls deep-winter Dec and Jan together and bends the arc into a ring.

This generates the 12 month corpora the A2 way (each month answers the shared
baseline prompts in character, response[i] <-> baseline_prompt[i % 48] so the
fit's alignment holds), writes a discover `fit_mode=auto` manifold, and fits it
through the REAL pipeline so `select_topology` decides per-model whether the
geometry is flat (pca), curved (spectral), or periodic (a detected ring). Then
plots the derived per-model node layout + the calendar-neighbor metrics.

Usage:
    python3 scripts/months_seasonal_fit.py [MODEL_ID] [--max-new-tokens N] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import torch

from saklas import SaklasSession
from saklas.core.session import _CORPUS_GEN_BATCH
from saklas.core.vectors import _load_baseline_prompts, _LENGTH_DIRECTIVE
from saklas.io.manifolds import create_discover_manifold_folder
from saklas.io.paths import manifold_dir, safe_model_id

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
ABBR = [m[:3] for m in MONTHS]

NS, NAME = "local", "months_seasonal"

# Embodiment framing: the model speaks AS the month. We deliberately do NOT name
# the season/weather ourselves — the model supplies the seasonal content, so the
# signal is the model's own month geometry, not our injected prior. Pooled in
# standard-assistant space at fit (node role stays None), the A2 way.
SEASONAL_SYSTEM = (
    "{ld} You are the month of {month}. Speak from {month}'s own point of "
    "view — let your weather, light, temperature, daylight, holidays, and the "
    "overall feeling of your season color everything you say."
)


def generate(session: SaklasSession, max_new_tokens: int):
    prompts = _load_baseline_prompts()
    print(f"  {len(prompts)} baseline prompts x {len(MONTHS)} months "
          f"= {len(prompts) * len(MONTHS)} responses @ {max_new_tokens} tok")
    corpora: dict[str, list[str]] = {}
    for mi, month in enumerate(MONTHS):
        system = SEASONAL_SYSTEM.format(ld=_LENGTH_DIRECTIVE, month=month)
        responses: list[str] = []
        for start in range(0, len(prompts), _CORPUS_GEN_BATCH):
            chunk = prompts[start:start + _CORPUS_GEN_BATCH]
            texts = session._run_generator_batch(
                system, chunk, max_new_tokens, role=None,
            )
            responses.extend(t.strip() for t in texts)
        corpora[month.lower()] = responses
        print(f"  [{mi + 1:>2}/12] {month}: {len(responses)} responses "
              f"(e.g. {responses[0][:70]!r})")
        if session.device.type == "mps":
            torch.mps.empty_cache()
    return corpora


def neighbor_metrics(coords: np.ndarray):
    """Calendar-neighbor ring diagnostic on the layout coords (Euclidean)."""
    x = np.asarray(coords, dtype=np.float64)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    n = len(MONTHS)

    def cyc(i: int, j: int):
        d = abs(i - j)
        return min(d, n - d)

    nn = d2.argmin(axis=1)
    mean_nn = float(np.mean([cyc(i, nn[i]) for i in range(n)]))
    hits = sum(any(cyc(i, j) == 1 for j in np.argsort(d2[i])[:2]) for i in range(n))
    chance = sum(min(d, n - d) for d in range(1, n)) / (n - 1)
    return mean_nn, hits / n, nn.tolist(), chance


def plot(coords: np.ndarray, resolved: str, nn_dist: float, recall2: float,
         chance: float, model_id: str, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    x = np.asarray(coords, dtype=np.float64)
    # pad to >=2 dims for plotting
    if x.shape[1] == 1:
        x = np.column_stack([x[:, 0], np.zeros(len(x))])
    colors = cm.twilight(np.linspace(0, 1, len(MONTHS), endpoint=False))
    season = {0: 'w', 1: 'w', 2: 's', 3: 's', 4: 's', 5: 'u',
              6: 'u', 7: 'u', 8: 'a', 9: 'a', 10: 'a', 11: 'w'}
    scol = {'w': '#4c72b0', 's': '#55a868', 'u': '#c44e52', 'a': '#dd8452'}

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, mode in zip(axes, ["cyclic", "season"]):
        cc = colors if mode == "cyclic" else [scol[season[i]] for i in range(12)]
        loop = np.vstack([x[:, :2], x[:1, :2]])
        ax.plot(loop[:, 0], loop[:, 1], color="0.7", lw=1.2, zorder=1)
        ax.scatter(x[:, 0], x[:, 1], c=cc, s=320, zorder=2,
                   edgecolor="k", linewidth=0.5)
        for i, ab in enumerate(ABBR):
            ax.annotate(ab, (x[i, 0], x[i, 1]), fontsize=11, ha="center",
                        va="center", color="white", weight="bold", zorder=3)
        ax.set_title(f"month -> {mode} color", fontsize=13)
        ax.set_xlabel("layout dim 0")
        ax.set_ylabel("layout dim 1")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
    verdict = ("RING" if nn_dist < 1.6 and "periodic" in resolved.lower()
               else "ARC / ordinal" if nn_dist < 1.6 else "FLAT scatter")
    fig.suptitle(
        f"Seasonal-embodiment months — {model_id}\n"
        f"resolved fit: {resolved}  ·  NN cyclic-cal dist {nn_dist:.2f} "
        f"(1.0=ring, ~{chance:.2f}=chance)  ·  recall@2 {recall2:.0%}  "
        f"->  {verdict}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", nargs="?", default="google/gemma-3-4b-it")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifold folder")
    ap.add_argument("--out", default="scripts/out/months_seasonal.png")
    args = ap.parse_args()

    folder = manifold_dir(NS, NAME)
    if folder.exists() and args.force:
        shutil.rmtree(folder)

    print(f"loading {args.model_id} ...")
    with SaklasSession.from_pretrained(args.model_id, device="auto") as session:
        if not (folder / "manifold.json").exists():
            print("generating seasonal-embodiment corpora ...")
            corpora = generate(session, args.max_new_tokens)
            create_discover_manifold_folder(
                NS, NAME,
                "Months under seasonal-embodiment framing (auto topology).",
                fit_mode="auto", node_corpora=corpora,
            )
            print(f"wrote corpora -> {folder}")
        else:
            print(f"reusing existing corpora at {folder} "
                  f"(pass --force to regenerate)")

        print("fitting (fit_mode=auto -> select_topology) ...")
        session.fit(folder, on_progress=lambda m: print("   ", m))

    # read the per-model fit sidecar for the resolved topology; node_coords
    # (the layout) lives in the safetensors, NOT the JSON sidecar.
    safe = safe_model_id(args.model_id)
    meta = json.load(open(folder / f"{safe}.json"))
    resolved = meta.get("resolved_fit_mode", meta.get("fit_mode", "?"))
    cands = meta.get("topology_candidates")
    domain = meta.get("domain")
    from safetensors import safe_open
    with safe_open(folder / f"{safe}.safetensors", framework="pt") as t:
        coords = t.get_tensor("node_coords").float().numpy()  # (12, d)

    print(f"\nresolved_fit_mode: {resolved}")
    if domain:
        print(f"domain: {json.dumps(domain)[:200]}")
    if cands:
        print(f"topology_candidates: {json.dumps(cands)[:400]}")

    nn_dist, recall2, nn, chance = neighbor_metrics(coords)
    print(f"\nlayout dims: {np.asarray(coords).shape}")
    print(f"NN cyclic-calendar dist: {nn_dist:.2f}  (1.0=ring, ~{chance:.2f}=chance)")
    print(f"calendar recall@2:       {recall2:.0%}")
    print("each month -> nearest: "
          + ", ".join(f"{ABBR[i]}->{ABBR[nn[i]]}" for i in range(len(MONTHS))))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot(coords, str(resolved), nn_dist, recall2, chance, args.model_id, args.out)
    dump = args.out.rsplit(".", 1)[0] + ".json"
    json.dump({
        "model_id": args.model_id, "resolved_fit_mode": resolved,
        "domain": domain, "topology_candidates": cands,
        "node_coords": np.asarray(coords).tolist(), "months": MONTHS,
        "nn_cyclic_cal_dist": nn_dist, "calendar_recall_at2": recall2,
        "nearest_month": {ABBR[i]: ABBR[nn[i]] for i in range(len(MONTHS))},
    }, open(dump, "w"), indent=2)
    print(f"wrote {dump}")


if __name__ == "__main__":
    main()
