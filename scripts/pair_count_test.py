"""Test whether n=45 contrastive pairs is enough vs n=120 for angry.calm.

Three stages:

1. ``gen`` — load gemma-4-31b-it (matching the bundled regenerator), read the
   existing 45 bundled angry.calm pairs + 9 scenarios, generate 75 fresh pairs
   against the same scenarios, write a combined 120-pair pool to scratch.
   Run once.

2. ``extract <model_id>`` — load the chosen extraction model, read the scratch
   pool, extract 7 profiles directly via ``extract_contrastive`` (bypassing the
   namespace cache):

       P45_bundled   — first 45 (= the bundled pool, deterministic)
       P120          — all 120
       P45_boot_1..5 — random 45-pair subsamples of the 120 pool, seeded

   Reports magnitude-weighted aggregate cosines (via Profile.cosine_similarity)
   and per-layer breakdown for the most divergent comparison. Writes results
   to scratch as JSON for cross-model comparison.

3. ``sweep <model_id>`` — extract n=45 (bundled) + n=120 (full pool) probes on
   the chosen model, then run an alpha sweep generating completions of one
   prompt at each (alpha, probe) combination using a pinned seed. Surfaces
   whether the cosine gap between probes translates to a behavioral steering
   difference.

Usage:
    python scripts/pair_count_test.py gen
    python scripts/pair_count_test.py extract google/gemma-4-31b-it
    python scripts/pair_count_test.py extract google/gemma-3-4b-it
    python scripts/pair_count_test.py sweep google/gemma-4-e4b-it
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import cast

import torch

from saklas.core.session import SaklasSession
from saklas.core.profile import Profile
from saklas.core.sampling import SamplingConfig
from saklas.core.vectors import extract_contrastive
from saklas.io.paths import saklas_home


REPO = Path(__file__).resolve().parent.parent
BUNDLED_DIR = REPO / "saklas" / "data" / "vectors" / "angry.calm"
SCRATCH = saklas_home() / "scratch" / "pair_count_120"

GEN_MODEL = "google/gemma-4-31b-it"
N_TARGET_TOTAL = 120
N_BOOTSTRAP = 5
N_SUBSAMPLE = 45
BOOT_SEED = 1234


def _hr(ch: str = "=", w: int = 72) -> str:
    return ch * w


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


# ---------- stage 1: gen ----------------------------------------------------

def stage_gen() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    statements_path = SCRATCH / "statements_120.json"
    if statements_path.exists():
        print(f"[gen] {statements_path} already exists; refusing to overwrite.")
        print(f"[gen] delete it manually if you want to regenerate.")
        return 0

    bundled_pairs = json.loads((BUNDLED_DIR / "statements.json").read_text())
    scenarios = json.loads((BUNDLED_DIR / "scenarios.json").read_text())["scenarios"]
    print(f"[gen] bundled: {len(bundled_pairs)} pairs across {len(scenarios)} scenarios")

    n_new = N_TARGET_TOTAL - len(bundled_pairs)
    print(f"[gen] need {n_new} new pairs to reach n={N_TARGET_TOTAL}")
    print(f"[gen] loading {GEN_MODEL}...", flush=True)
    t0 = time.time()
    session = SaklasSession.from_pretrained(GEN_MODEL, device="auto", probes=[])
    print(f"[gen] loaded on {session._device} ({session._dtype}) in {time.time()-t0:.1f}s",
          flush=True)

    t0 = time.time()
    new_pairs = session.generate_pairs(
        "angry", "calm",
        n=n_new,
        scenarios=scenarios,
        on_progress=lambda m: print(f"  {m}", flush=True),
    )
    print(f"[gen] got {len(new_pairs)} new pairs in {time.time()-t0:.1f}s")

    new_payload = [{"positive": a, "negative": b} for a, b in new_pairs]
    combined = bundled_pairs + new_payload
    statements_path.write_text(json.dumps(combined, indent=2) + "\n")
    (SCRATCH / "scenarios.json").write_text(
        json.dumps({"scenarios": scenarios}, indent=2) + "\n"
    )
    (SCRATCH / "manifest.json").write_text(json.dumps({
        "concept": "angry.calm",
        "gen_model": GEN_MODEL,
        "n_bundled": len(bundled_pairs),
        "n_new": len(new_pairs),
        "n_total": len(combined),
        "scenarios": scenarios,
        "bundled_indices": list(range(len(bundled_pairs))),
        "new_indices": list(range(len(bundled_pairs), len(combined))),
    }, indent=2) + "\n")
    print(f"[gen] wrote {len(combined)} pairs to {statements_path}")
    return 0


# ---------- stage 2: extract -----------------------------------------------

def _build_subsamples(n_pool: int) -> dict[str, list[int]]:
    """Build the 7 named index subsets of the 120-pair pool.

    Bootstrap subsamples are independent draws (not partitions) — overlap is
    expected and fine; the noise floor we want is "if you redrew n=45 from a
    larger latent pool, how much would the resulting probe move?".
    """
    out: dict[str, list[int]] = {}
    out["P45_bundled"] = list(range(N_SUBSAMPLE))
    out["P120"] = list(range(n_pool))
    rng = random.Random(BOOT_SEED)
    for k in range(1, N_BOOTSTRAP + 1):
        idx = rng.sample(range(n_pool), N_SUBSAMPLE)
        idx.sort()
        out[f"P45_boot_{k}"] = idx
    return out


def stage_extract(model_id: str) -> int:
    statements_path = SCRATCH / "statements_120.json"
    if not statements_path.exists():
        print(f"[extract] missing {statements_path}; run `gen` first.", file=sys.stderr)
        return 2
    pool = json.loads(statements_path.read_text())
    print(f"[extract] pool: {len(pool)} pairs from {statements_path}")

    print(f"[extract] loading {model_id}...", flush=True)
    t0 = time.time()
    session = SaklasSession.from_pretrained(model_id, device="auto", probes=[])
    print(f"[extract] loaded on {session._device} ({session._dtype}) "
          f"in {time.time()-t0:.1f}s", flush=True)

    subsamples = _build_subsamples(len(pool))
    profiles: dict[str, Profile] = {}
    for name, indices in subsamples.items():
        sub = [pool[i] for i in indices]
        t0 = time.time()
        prof_dict, _ = extract_contrastive(
            session._model, session._tokenizer, sub,
            layers=session._layers, concept_label=name,
        )
        profiles[name] = Profile(prof_dict)
        print(f"[extract] {name}: n={len(sub)}, {len(prof_dict)} layers, "
              f"{time.time()-t0:.1f}s", flush=True)

    # Free model: we're done with forward passes, only Profile tensors remain.
    del session
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ---- aggregate cosine table -------------------------------------------
    # Names are inserted in extraction order: P45_bundled, P120, P45_boot_1..5.
    # Agg is keyed (earlier_name, later_name) per that order.
    names = list(profiles.keys())
    agg: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            agg[(a, b)] = cast(float, profiles[a].cosine_similarity(profiles[b]))

    print()
    print(_hr())
    print(f"AGGREGATE COSINES — model: {model_id}")
    print(_hr())
    width = max(len(n) for n in names)
    for (a, b), c in agg.items():
        print(f"  {a:<{width}}  vs  {b:<{width}} : {c:+.4f}")

    # ---- bootstrap noise floor --------------------------------------------
    # P120 was inserted before any boot, so the canonical key is (P120, boot).
    # P45_bundled was inserted before everything, so (P45_bundled, X) for all X.
    boot = [n for n in names if n.startswith("P45_boot_")]
    boot_pairs = [agg[(boot[i], boot[j])]
                  for i in range(len(boot)) for j in range(i + 1, len(boot))]
    boot_v_p120 = [agg[("P120", b)] for b in boot]
    bundled_v_p120 = agg[("P45_bundled", "P120")]
    bundled_v_boot = [agg[("P45_bundled", b)] for b in boot]

    def _stats(xs: list[float]) -> str:
        if not xs:
            return "n/a"
        if len(xs) == 1:
            return f"{xs[0]:+.4f}"
        return (f"mean={statistics.mean(xs):+.4f} "
                f"std={statistics.stdev(xs):.4f} "
                f"min={min(xs):+.4f} max={max(xs):+.4f}")

    print()
    print(_hr("-"))
    print(f"NOISE FLOOR (n=45 bootstrap, k={len(boot)} samples)")
    print(_hr("-"))
    print(f"  pairwise (n={len(boot_pairs)}):     {_stats(boot_pairs)}")
    print(f"  vs P120 (n={len(boot_v_p120)}):       {_stats(boot_v_p120)}")
    print(f"  bundled vs boot:        {_stats(bundled_v_boot)}")
    print(f"  bundled vs P120:        {bundled_v_p120:+.4f}")

    # Verdict heuristic: if |bundled-vs-P120| sits inside [boot pairwise min,
    # max], n=45 noise dominates the bundled/120 gap → "n=45 is fine".  If
    # bundled-vs-P120 is *higher* than every boot pairwise (closer to 120 than
    # boot probes are to each other), the bundled is essentially in the
    # n=120 limit already — also "fine".  If it's *lower* (more divergent
    # than typical n=45 noise), that's evidence n=45 is unstable.
    if boot_pairs:
        floor_min, floor_max = min(boot_pairs), max(boot_pairs)
        if bundled_v_p120 >= floor_min:
            verdict = (f"VERDICT: bundled vs P120 ({bundled_v_p120:+.4f}) "
                       f"sits inside or above the n=45 noise floor "
                       f"[{floor_min:+.4f}, {floor_max:+.4f}] → "
                       f"n=45 is consistent with the n=120 limit on this model.")
        else:
            verdict = (f"VERDICT: bundled vs P120 ({bundled_v_p120:+.4f}) "
                       f"falls below the n=45 noise floor "
                       f"[{floor_min:+.4f}, {floor_max:+.4f}] → "
                       f"the bundled probe is an outlier; n=45 may be undersampled.")
        print()
        print(verdict)

    # ---- per-layer breakdown for most divergent pair ----------------------
    if agg:
        worst_pair = min(agg.items(), key=lambda kv: kv[1])
        (a, b), c = worst_pair
        per_layer = cast(
            dict[int, float],
            profiles[a].cosine_similarity(profiles[b], per_layer=True),
        )
        print()
        print(_hr("-"))
        print(f"PER-LAYER COSINE — most divergent pair: {a} vs {b} ({c:+.4f})")
        print(_hr("-"))
        for layer in sorted(per_layer.keys()):
            v = per_layer[layer]
            bar = "#" * int(max(0.0, v) * 30)
            print(f"  L{layer:>3}  {v:+.4f}  {bar}")

    # ---- persist for cross-model comparison -------------------------------
    out_path = SCRATCH / f"results_{_safe_model_id(model_id)}.json"
    out_path.write_text(json.dumps({
        "model_id": model_id,
        "n_pool": len(pool),
        "subsamples": {k: v for k, v in subsamples.items()},
        "aggregate_cosines": {f"{a}__vs__{b}": c for (a, b), c in agg.items()},
        "noise_floor": {
            "boot_pairwise": boot_pairs,
            "boot_vs_p120": boot_v_p120,
            "bundled_vs_boot": bundled_v_boot,
            "bundled_vs_p120": bundled_v_p120,
        },
    }, indent=2) + "\n")
    print()
    print(f"[extract] results written to {out_path}")
    return 0


# ---------- stage 3: sweep --------------------------------------------------

DEFAULT_SWEEP_PROMPT = (
    "A customer just told me the package I shipped arrived broken. "
    "Write my reply."
)
DEFAULT_ALPHAS = (0.0, 0.3, 0.5, 0.7, 0.9)


def stage_sweep(
    model_id: str,
    *,
    prompt: str,
    alphas: list[float],
    seed: int,
    max_tokens: int,
    temperature: float,
) -> int:
    statements_path = SCRATCH / "statements_120.json"
    if not statements_path.exists():
        print(f"[sweep] missing {statements_path}; run `gen` first.", file=sys.stderr)
        return 2
    pool_120 = json.loads(statements_path.read_text())
    bundled_45 = json.loads((BUNDLED_DIR / "statements.json").read_text())
    print(f"[sweep] pool: bundled={len(bundled_45)}, full={len(pool_120)}")

    print(f"[sweep] loading {model_id}...", flush=True)
    t0 = time.time()
    session = SaklasSession.from_pretrained(model_id, device="auto", probes=[])
    print(f"[sweep] loaded on {session._device} ({session._dtype}) "
          f"in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    p45_dict, _ = extract_contrastive(
        session._model, session._tokenizer, bundled_45,
        layers=session._layers, concept_label="angry.calm__n45",
    )
    print(f"[sweep] n=45 extracted in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    p120_dict, _ = extract_contrastive(
        session._model, session._tokenizer, pool_120,
        layers=session._layers, concept_label="angry.calm__n120",
    )
    print(f"[sweep] n=120 extracted in {time.time()-t0:.1f}s", flush=True)

    p45 = Profile(p45_dict)
    p120 = Profile(p120_dict)
    cos = cast(float, p45.cosine_similarity(p120))
    print(f"[sweep] n=45 vs n=120 magnitude-weighted cosine: {cos:+.4f}")

    name45, name120 = "angry_calm_n45", "angry_calm_n120"
    session.steer(name45, p45)
    session.steer(name120, p120)

    sampling = SamplingConfig(
        seed=seed, temperature=temperature, max_tokens=max_tokens,
    )

    print()
    print(_hr())
    print(f"PROMPT: {prompt}")
    print(f"  model={model_id}  seed={seed}  temp={temperature}  max_tokens={max_tokens}")
    print(_hr())

    outputs: dict[str, str] = {}
    for alpha in alphas:
        if alpha == 0.0:
            print(f"\n--- α=0.0 (vanilla, no steering) ---", flush=True)
            r = session.generate(prompt, sampling=sampling)
            text = r.text.strip()
            print(text)
            outputs["vanilla"] = text
            continue
        for label, name in (("n=45", name45), ("n=120", name120)):
            steering = f"{alpha} {name}"
            print(f"\n--- α={alpha}  {label}  ({steering!r}) ---", flush=True)
            r = session.generate(prompt, steering=steering, sampling=sampling)
            text = r.text.strip()
            print(text)
            outputs[f"a{alpha}_{label}"] = text

    out_path = SCRATCH / f"sweep_{_safe_model_id(model_id)}.json"
    out_path.write_text(json.dumps({
        "model_id": model_id,
        "prompt": prompt,
        "alphas": alphas,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n45_vs_n120_cosine": cos,
        "outputs": outputs,
    }, indent=2) + "\n")
    print(f"\n[sweep] results written to {out_path}")
    return 0


# ---------- main ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen", help=f"generate 75 new pairs with {GEN_MODEL}")
    p_ext = sub.add_parser("extract", help="extract 7 profiles, report cosines")
    p_ext.add_argument("model_id")
    p_sw = sub.add_parser("sweep", help="alpha sweep n=45 vs n=120 on a model")
    p_sw.add_argument("model_id")
    p_sw.add_argument("--prompt", default=DEFAULT_SWEEP_PROMPT)
    p_sw.add_argument("--alphas", type=float, nargs="+",
                      default=list(DEFAULT_ALPHAS),
                      help="alpha values to sweep (0.0 = vanilla)")
    p_sw.add_argument("--seed", type=int, default=1234)
    p_sw.add_argument("--max-tokens", type=int, default=180)
    p_sw.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.cmd == "gen":
        return stage_gen()
    if args.cmd == "extract":
        return stage_extract(args.model_id)
    return stage_sweep(
        args.model_id,
        prompt=args.prompt,
        alphas=args.alphas,
        seed=args.seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    sys.exit(main())
