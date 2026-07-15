"""The model's *explicit* belief about a current-year probe.

    python examples/representation_geometry/score.py years_now_future [--model M]

Scores the template's restricted-choice distribution over its value set — for the
current-year probes, P("the current year is ___") aggregated across the contexts.
The peak is the model's literal belief about the present, an independent read on
the same "now" the manifold geometry locates in ``analyze.py``.

This is the *other* template consumer: the same artifact that the manifold fit
turns into a steering surface, the scorer turns into a logit read. Pass
``--from-cache`` to re-plot a previous run's ``belief.json`` without a GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import PROBES  # noqa: E402  # pyright: ignore[reportAttributeAccessIssue]  # sibling script module inserted above

FIGROOT = Path(__file__).parent / "figures"


def yr(label: str) -> int:
    m = re.search(r"(\d{4})", label)
    assert m is not None
    return int(m.group(1))


def score(probe: str, model_id: str) -> dict[str, object]:
    from saklas import SaklasSession
    with SaklasSession.from_pretrained(model_id, device="auto") as s:
        per_ctx = s.score_template(f"local/{probe}")
    years = sorted({yr(c.label) for cs in per_ctx for c in cs.choices})
    y2i = {y: i for i, y in enumerate(years)}
    P = np.zeros((len(per_ctx), len(years)))
    for ci, cs in enumerate(per_ctx):
        for c in cs.choices:
            P[ci, y2i[yr(c.label)]] = c.prob_sum
    mean_p = P.mean(0)
    mean_p /= mean_p.sum()
    return {"years": years, "mean_prob": mean_p.tolist()}


def report_and_plot(belief: dict[str, object], fig: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = np.array(belief["years"])
    p = np.array(belief["mean_prob"])
    peak = int(years[p.argmax()])
    ev = float((years * p).sum())
    cum = np.cumsum(p)
    lo, hi = int(years[np.searchsorted(cum, 0.05)]), int(years[np.searchsorted(cum, 0.95)])
    print(f"  argmax (peak belief) = {peak}   prob-weighted mean = {ev:.1f}")
    print(f"  central 90% mass     = [{lo}, {hi}]")
    print("  top years by P(current year):")
    for i in np.argsort(p)[::-1][:6]:
        if p[i] > 1e-3:
            print(f"    {int(years[i])}   {p[i]:.3f}")

    f, ax = plt.subplots(figsize=(13, 5))
    ax.bar(years, p, width=0.9, color="steelblue")
    ax.axvline(peak, color="red", ls="--", label=f"peak {peak}")
    ax.set_xlabel("year")
    ax.set_ylabel("P(current year)")
    ax.set_title('model\'s explicit belief about "now" — P("the current year is ___")')
    ax.legend()
    ax.grid(alpha=0.2)
    f.tight_layout()
    f.savefig(fig / "belief_curve.png", dpi=130)
    plt.close(f)
    print(f"  wrote {fig/'belief_curve.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probe", choices=[p for p in PROBES if "now" in p])
    ap.add_argument("-m", "--model", default="google/gemma-3-4b-it")
    ap.add_argument("--from-cache", action="store_true",
                    help="re-plot belief.json from a previous run (no GPU)")
    args = ap.parse_args()
    fig = FIGROOT / args.probe
    fig.mkdir(parents=True, exist_ok=True)
    cache = fig / "belief.json"
    if args.from_cache:
        belief = json.load(open(cache))
    else:
        print(f"[{args.probe}] scoring P(current year) on {args.model} …")
        belief = score(args.probe, args.model)
        json.dump(belief, open(cache, "w"))
    report_and_plot(belief, fig)


if __name__ == "__main__":
    main()
