"""Fuzzy-manifold eval on gemma-4-12B: does soft onto (steer into the σ-tube)
preserve within-concept variety vs the hard collapse onto the zero-thickness wire?

One model load: force-refit pad (→ σ-field), verify it, then compare onto=1 with
the σ-field ON (tube) vs OFF (wire) at strong push — same manifold, same seeds,
only the tube toggled (σ=0 is exact-legacy, so zeroing it IS the old behavior).
Also dumps the fuzzy reads (assignment + membership) for a steered sample.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, cast

from saklas import SaklasSession, SamplingConfig

MODEL = "google/gemma-4-12B-it"
SAFE = "google__gemma-4-12B-it"
PAD = Path.home() / ".saklas/manifolds/default/pad"
PROMPT = "Tell me about your plans for the weekend."
NODE = "dominant"          # PAD node 07
ALONG = 0.8                # strong push
SEEDS = [11, 22, 33]


def banner(s: str) -> None:
    print(f"\n{'='*70}\n{s}\n{'='*70}", flush=True)


def main() -> None:
    # 1. Force a refit: delete the pre-σ per-model tensor + sidecar.
    for suf in (".safetensors", ".json"):
        p = PAD / f"{SAFE}{suf}"
        if p.exists():
            p.unlink()
            print(f"cleared {p.name}", flush=True)

    banner("LOADING gemma-4-12B-it (MPS)")
    with SaklasSession.from_pretrained(MODEL, device="auto") as session:
        banner("FITTING pad (curved spectral → σ-field second pass)")
        manifold = session.fit(str(PAD))

        # 2. Verify the σ-field landed.
        n_layers = len(manifold.layers)
        with_sigma = sum(1 for s in manifold.layers.values() if s.has_sigma)
        sf = cast(dict[str, Any], manifold.metadata.get("sigma_field_per_layer", {}))
        print(f"layers: {n_layers}  with σ-field: {with_sigma}", flush=True)
        if sf:
            some = list(sf.items())[:4]
            for L, info in some:
                print(f"  layer {L}: σ_mean={info['sigma_mean']:.4f} "
                      f"σ_min={info['sigma_min']:.4f} σ_max={info['sigma_max']:.4f} "
                      f"λ={info.get('lambda', 0):.3g}", flush=True)
        # sample σ at the dominant node coord on one layer
        sub = next(iter(manifold.layers.values()))
        idx = manifold.node_labels.index("dominant")
        z = manifold.domain.embed(manifold.node_coords[idx])
        print(f"σ at 'dominant' node (one layer): {float(sub.sigma_at(z)):.4f}", flush=True)

        sampling = lambda seed: SamplingConfig(temperature=0.8, max_tokens=90, seed=seed)

        def run(expr: str, label: str) -> list[str]:
            banner(f"{label}:  {expr}")
            outs = []
            for seed in SEEDS:
                r = session.generate(PROMPT, steering=expr, sampling=sampling(seed))
                txt = r.first.text.strip().replace("\n", " ")
                outs.append(txt)
                print(f"[seed {seed}] {txt[:260]}", flush=True)
            return outs

        # 3a. Baseline: no collapse (onto=0), strong along.
        run(f"{ALONG} pad%{NODE}@response", "A — onto=0 (no collapse)")

        # 3b. Soft onto=1 (σ-field ON → collapse toward the tube).
        soft = run(f"{ALONG},1.0 pad%{NODE}@response", "B — onto=1, σ-field ON (tube)")

        # 4. Zero the σ-field on the loaded manifold → onto=1 collapses to the
        #    zero-thickness wire (exact legacy). Same object the steering path uses.
        zeroed = 0
        for m in session._manifolds.values():
            if "pad" in m.name:
                for s in m.layers.values():
                    if s.has_sigma:
                        s.sigma_rbf_weights = None
                        s.sigma_poly_coeffs = None
                        zeroed += 1
        print(f"\n[zeroed σ on {zeroed} layers of the loaded pad manifold]", flush=True)
        hard = run(f"{ALONG},1.0 pad%{NODE}@response", "C — onto=1, σ-field OFF (wire / legacy)")

        # 5. Crude within-condition variety: distinct-word ratio across the 3 samples.
        def variety(outs: list[str]) -> float:
            words = " ".join(outs).lower().split()
            return len(set(words)) / max(1, len(words))
        banner("VARIETY (unique-word ratio across the 3 seeds; higher = less collapse)")
        print(f"  B soft/tube : {variety(soft):.3f}", flush=True)
        print(f"  C hard/wire : {variety(hard):.3f}", flush=True)

        # 6. Fuzzy reads on a soft-steered sample: assignment + membership.
        banner("FUZZY READS (pad probe on a soft-steered generation)")
        session.add_probe("pad", top_n=4)
        r = session.generate(
            PROMPT, steering=f"{ALONG},1.0 pad%{NODE}@response",
            sampling=SamplingConfig(temperature=0.8, max_tokens=60, seed=7),
        )
        rd = r.first.probe_readings.get("pad") if r.first.probe_readings else None
        if rd is not None:
            print(f"nearest:    {[(l, round(d,3)) for l,d in rd.nearest]}", flush=True)
            print(f"assignment: {[(l, round(p,3)) for l,p in rd.assignment]}", flush=True)
            print(f"membership: {rd.membership:.3f}", flush=True)
            print(f"fraction:   {rd.fraction:.3f}  residual: {rd.residual:.3f}", flush=True)
        else:
            print("no pad reading captured", flush=True)

    banner("DONE")


if __name__ == "__main__":
    main()
