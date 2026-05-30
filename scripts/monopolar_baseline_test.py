"""Monopolar contrast comparison: synthetic-opposite vs neutral baseline.

For each monopolar concept (`agentic`, `manipulative`) on gemma-3-4b-it,
extract two difference-of-means directions:

  opposite : (concept positives) - (synthetic "opposite" corpus)   [current]
  neutral  : (concept positives) - (fixed neutral corpus)          [proposed]

Then report, head to head:
  1. how different the directions actually are (per-layer cosine)
  2. what DLS does to each (survivor-layer count) -- the bipolar
     layer-selector should degenerate on the neutral contrast
  3. faithful TraitMonitor readings on held-out text groups -- does the
     opposite axis invent a meaningless negative region, and does the
     neutral axis read cleanly as presence-above-baseline?
"""
import json
import statistics as st
from typing import Any

import torch

from saklas import SaklasSession
from saklas.core.vectors import (
    extract_difference_of_means,
    _load_neutral_prompts,
)

MODEL = "google/gemma-3-4b-it"
N_PAIRS = 30
DATA = "saklas/data/vectors"
CONCEPTS = ["agentic", "manipulative"]

# Held-out probe sentences (NOT used in any extraction) + held-out neutrals.
HELDOUT = {
    "agentic": "I'll take charge, lay out the plan, and drive this to done myself.",
    "passive": "I'll just wait and see, and go along with whatever gets decided.",
    "manipulative": "If I frame it as his idea and leave out the downside, he'll agree without pushing back.",
    "straightforward": "Here are the real tradeoffs, including the parts that cut against what I want.",
}


def load_pairs(concept: str) -> list[Any]:
    with open(f"{DATA}/{concept}/statements.json") as f:
        rows = json.load(f)
    return rows[:N_PAIRS]


def unit_per_layer(profile: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    out = {}
    for L, v in profile.items():
        n = v.float().norm()
        out[L] = (v.float() / n) if n > 1e-8 else v.float()
    return out


def main():
    print(f"loading {MODEL} ...", flush=True)
    session = SaklasSession.from_pretrained(MODEL, device="auto")
    model, tok = session._model, session._tokenizer
    device = next(model.parameters()).device
    mtype = model.config.model_type
    layer_means = session._layer_means
    # NOTE: session.whitener produces all-NaN baked tensors for this model
    # (Mahalanobis sqrt of a numerically-negative quadratic form). Use the
    # Euclidean bake here — same metric for both contrasts, so the
    # opp-vs-neutral comparison stays apples-to-apples.
    whitener = None
    LAYERS = session._layers  # the layer module list; capture hooks attach to these
    print(f"  device={device}  layers={len(LAYERS)}  model_type={mtype}", flush=True)

    neutrals = _load_neutral_prompts()
    heldout_neutral = neutrals[N_PAIRS:N_PAIRS + 20]
    print(f"  neutral corpus={len(neutrals)}  held-out neutral={len(heldout_neutral)}", flush=True)

    def extract(pairs: list[dict[str, str]], dls: bool) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
        return extract_difference_of_means(
            model, tok, pairs, LAYERS, device,
            whitener=whitener, dls=dls, layer_means=layer_means,
            model_type=mtype,
        )

    report = {}
    monitor_profiles = {}  # {probe_name: full-layer profile} for the reading test

    for concept in CONCEPTS:
        print(f"\n=== {concept} ===", flush=True)
        opp_pairs = load_pairs(concept)
        pos = [r["positive"] for r in opp_pairs]
        neu_pairs = [{"positive": p, "negative": neutrals[i]} for i, p in enumerate(pos)]

        print("  extracting opposite (dls=False / dls=True) ...", flush=True)
        opp_full, opp_diag = extract(opp_pairs, dls=False)
        opp_dls, _ = extract(opp_pairs, dls=True)
        print("  extracting neutral  (dls=False / dls=True) ...", flush=True)
        neu_full, neu_diag = extract(neu_pairs, dls=False)
        neu_dls, _ = extract(neu_pairs, dls=True)

        # 1. per-layer cosine between the two unit directions
        ou, nu = unit_per_layer(opp_full), unit_per_layer(neu_full)
        shared = sorted(set(ou) & set(nu))
        cos = [float((ou[L] * nu[L]).sum()) for L in shared]

        # 2. DLS survivors
        report[concept] = {
            "n_layers": len(LAYERS),
            "cos_mean": st.mean(cos),
            "cos_min": min(cos),
            "cos_max": max(cos),
            "dls_opp": len(opp_dls),
            "dls_neu": len(neu_dls),
            "evr_opp": st.median(d["evr"] for d in opp_diag.values()),
            "evr_neu": st.median(d["evr"] for d in neu_diag.values()),
            "intravar_opp": st.median(d["intra_pair_variance_mean"] for d in opp_diag.values()),
            "intravar_neu": st.median(d["intra_pair_variance_mean"] for d in neu_diag.values()),
        }
        monitor_profiles[f"{concept}__opp"] = opp_full
        monitor_profiles[f"{concept}__neu"] = neu_full

    # 3. faithful reading test via TraitMonitor (full-layer directions as probes)
    from saklas.core.monitor import TraitMonitor
    mon = TraitMonitor(monitor_profiles, layer_means)

    def read_group(texts: list[str]) -> dict[str, float]:
        accs = {k: [] for k in monitor_profiles}
        for t in texts:
            scores = mon.measure(model, tok, LAYERS, t, device, accumulate=False)
            for k, v in scores.items():
                accs[k].append(v)
        return {k: st.mean(vs) for k, vs in accs.items()}

    print("\n=== reading test (mean cosine-style probe score per group) ===", flush=True)
    groups = {
        "heldout-agentic-1": [HELDOUT["agentic"]],
        "heldout-passive-1": [HELDOUT["passive"]],
        "heldout-manip-1": [HELDOUT["manipulative"]],
        "heldout-straightfwd-1": [HELDOUT["straightforward"]],
        "concept-positives-agentic": [r["positive"] for r in load_pairs("agentic")][:12],
        "synthetic-opp-agentic": [r["negative"] for r in load_pairs("agentic")][:12],
        "concept-positives-manip": [r["positive"] for r in load_pairs("manipulative")][:12],
        "synthetic-opp-manip": [r["negative"] for r in load_pairs("manipulative")][:12],
        "heldout-neutral": heldout_neutral,
    }
    readings = {g: read_group(t) for g, t in groups.items()}

    # ---- print report ----
    print("\n" + "=" * 78)
    print("DIRECTION / DLS / DIAGNOSTICS")
    print("=" * 78)
    for c, r in report.items():
        print(f"\n[{c}]  ({r['n_layers']} layers)")
        print("  direction agreement (opp vs neutral), per-layer cosine:")
        print(f"     mean={r['cos_mean']:+.3f}  min={r['cos_min']:+.3f}  max={r['cos_max']:+.3f}")
        print(f"  DLS survivors:   opposite={r['dls_opp']}/{r['n_layers']}   neutral={r['dls_neu']}/{r['n_layers']}")
        print(f"  median EVR:      opposite={r['evr_opp']:.3f}   neutral={r['evr_neu']:.3f}")
        print(f"  median intra-pair var: opposite={r['intravar_opp']:.3f}   neutral={r['intravar_neu']:.3f}")

    print("\n" + "=" * 78)
    print("READING TEST  (rows=text group, cols=probe; cosine-style score)")
    print("=" * 78)
    probes = list(monitor_profiles.keys())
    hdr = "group".ljust(28) + "".join(p.rjust(20) for p in probes)
    print(hdr)
    print("-" * len(hdr))
    for g, sc in readings.items():
        line = g.ljust(28) + "".join(f"{sc[p]:+.3f}".rjust(20) for p in probes)
        print(line)

    print("\nKey checks:")
    print("  * opp-axis SKEW: does held-out neutral read far from 0 (and far from")
    print("    the midpoint of positives/synthetic-opp) under '__opp' probes?")
    print("  * neutral-axis: does held-out neutral read ~0 and positives read high")
    print("    under '__neu' probes, while the synthetic-opp group is NOT strongly")
    print("    negative (no invented anti-pole region)?")

    out = {"report": report, "readings": readings}
    with open("scripts/monopolar_baseline_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote scripts/monopolar_baseline_result.json")


if __name__ == "__main__":
    main()
