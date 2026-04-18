"""Alpha sweep across three concepts using the open-ended extraction pipeline.

For each concept, calls ``session.extract(..., force_statements=True)`` to
trigger the full pipeline (generate scenarios → save scenarios.json →
generate pairs → save statements.json → extract contrastive → save tensor)
even when a bundled pack already has curated v1.4 statements. Then generates
at fixed seed across alphas ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25}.

Defaults to ``google/gemma-4-e4b-it`` — a weak model that makes both the
extraction framework and the coherent/incoherent alpha boundary visible.

Usage:
    python scripts/alpha_sweep_open_ended.py
    python scripts/alpha_sweep_open_ended.py --concept deer.wolf
    python scripts/alpha_sweep_open_ended.py --model google/gemma-4-31b-it
"""
from __future__ import annotations

import argparse
import sys
import time

from saklas.core.session import SaklasSession
from saklas.core.sampling import SamplingConfig


MODEL_ID = "google/gemma-4-e4b-it"
SEED = 1234
MAX_TOKENS = 180
ALPHAS_POS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
ALPHAS_NEG = [-1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]

# (pos, neg, test_prompt, use_neg_alphas) — prompts chosen to give both
# poles natural room to manifest without being axis-specific loaded
# questions. use_neg_alphas=True sweeps both directions.
CONCEPTS: dict[str, tuple[str, str | None, str, bool]] = {
    "deer.wolf": (
        "deer", "wolf",
        "Describe what you sense around you right now.",
        False,
    ),
    "angry.calm": (
        "angry", "calm",
        "A customer just told me the package I shipped arrived broken. "
        "Write my reply.",
        False,
    ),
    "masculine.feminine": (
        "masculine", "feminine",
        "You're at a networking mixer and someone asks what you do. Respond.",
        True,
    ),
}


def _hr(char: str = "=", width: int = 72) -> str:
    return char * width


def run_concept(
    session: SaklasSession,
    name: str,
    pos: str,
    neg: str,
    prompt: str,
    use_neg_alphas: bool,
) -> None:
    alphas = ALPHAS_NEG if use_neg_alphas else ALPHAS_POS
    print(_hr("="))
    mode_label = "bipolar" if neg else "monopolar"
    print(f"CONCEPT: {name}  ({mode_label}: {pos}" + (f" / {neg})" if neg else ")"))
    print(_hr("="))
    print(f"Prompt: {prompt}")
    print(f"Alphas: {alphas}\n", flush=True)

    t0 = time.time()
    canonical, profile = session.extract(
        pos, baseline=neg,
        force_statements=True,
        on_progress=lambda m: print(f"  [extract] {m}", flush=True),
    )
    dt = time.time() - t0

    norms = {layer: float(v.norm().item()) for layer, v in profile.items()}
    mean_norm = sum(norms.values()) / len(norms)
    peak = max(norms.values())
    top5 = sorted(norms.items(), key=lambda x: -x[1])[:5]
    print(
        f"\nProfile: layers={len(norms)}  "
        f"mean_norm={mean_norm:.4f}  peak={peak:.4f}  "
        f"peak/mean={peak / mean_norm:.2f}  "
        f"({dt:.1f}s total)"
    )
    print(f"top-5 layers by ||baked||: {top5}\n", flush=True)

    session.steer(canonical, profile)

    for alpha in alphas:
        print(_hr("-"))
        print(f"α = {alpha:+.2f}")
        print(_hr("-"))
        session.clear_history()
        result = session.generate(
            prompt,
            steering=f"{alpha} {canonical}" if alpha != 0.0 else None,
            sampling=SamplingConfig(seed=SEED, max_tokens=MAX_TOKENS),
            thinking=False,
            stateless=True,
        )
        print(result.text.strip())
        print(flush=True)

    # Clean up so next concept's registry is fresh.
    session.unsteer(canonical)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--concept",
        action="append",
        default=None,
        help="Restrict to one or more concepts (repeatable).",
    )
    args = parser.parse_args()

    names = args.concept if args.concept else list(CONCEPTS.keys())
    unknown = [n for n in names if n not in CONCEPTS]
    if unknown:
        print(f"Unknown concepts: {unknown}", file=sys.stderr)
        print(f"Known: {list(CONCEPTS.keys())}", file=sys.stderr)
        return 2

    print(f"Loading {args.model}...", flush=True)
    session = SaklasSession.from_pretrained(
        args.model, device="auto", max_tokens=MAX_TOKENS, probes=[],
    )
    print(f"Loaded on {session._device} ({session._dtype})\n", flush=True)

    for name in names:
        pos, neg, prompt, use_neg = CONCEPTS[name]
        run_concept(session, name, pos, neg, prompt, use_neg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
