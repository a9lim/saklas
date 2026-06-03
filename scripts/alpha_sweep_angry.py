"""Alpha sweep for `angry.calm` using the bundled concept manifold.

Extracts via the string-source pipeline (``session.extract("angry.calm")``),
which reads from or materializes ``~/.saklas/manifolds/default/angry.calm/``.
Generates at fixed seed/prompt across alphas. Positive alpha
→ angry pole, negative alpha → calm pole.

Defaults to gemma-4-e4b-it with α ∈ [-1, 1] in 0.2 steps for the v2.1
Mahalanobis-bake calibration sweep.  Override via CLI flags:

    python scripts/alpha_sweep_angry.py
    python scripts/alpha_sweep_angry.py --model google/gemma-4-31b-it
    python scripts/alpha_sweep_angry.py --alphas -0.5 0 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from saklas.core.session import SaklasSession

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "google/gemma-4-e4b-it"
PROMPT = "A customer just told me the package I shipped arrived broken. Write my reply."
DEFAULT_ALPHAS = [round(-1.0 + 0.2 * i, 2) for i in range(11)]  # -1.0 .. +1.0
SEED = 1234
MAX_TOKENS = 180


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--alphas", nargs="+", type=float, default=None,
        help=f"Alpha values to sweep (default: {DEFAULT_ALPHAS})",
    )
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    alphas = args.alphas if args.alphas is not None else DEFAULT_ALPHAS

    print(f"Loading {args.model}...", flush=True)
    session = SaklasSession.from_pretrained(
        args.model, device="auto", max_tokens=args.max_tokens, probes=[],
    )
    print(f"Loaded on {session._device}", flush=True)

    print("Extracting angry.calm profile (bundled pairs)...", flush=True)
    # String-source path: extraction reads from the materialized bundled
    # concept manifold and writes the tensor under that same folder, so
    # the canonical name ("angry.calm") parses unambiguously when no
    # other namespace has the concept registered.
    name, profile = session.extract(
        "angry.calm", on_progress=lambda m: print("  ", m, flush=True),
    )
    session.steer(name, profile)
    steer_key = name

    norms = {l: float(v.norm().item()) for l, v in profile.items()}
    mean_s = sum(norms.values()) / len(norms)
    top5 = sorted(norms.items(), key=lambda x: -x[1])[:5]
    print(f"\nProfile stats: layers={len(norms)} mean_norm={mean_s:.4f} "
          f"max={max(norms.values()):.4f} peak/mean={max(norms.values())/mean_s:.2f}")
    print(f"top5 layers: {top5}\n")

    from saklas import SamplingConfig
    for a in alphas:
        print(f"=== alpha={a:+.2f} ===", flush=True)
        session.clear_history()
        # Negative α flips the pole (calm side).  Skip steering only at exactly 0.
        steering = f"{a} {steer_key}" if a != 0.0 else None
        result = session.generate(
            PROMPT,
            steering=steering,
            sampling=SamplingConfig(seed=args.seed),
            stateless=True,
        )
        print(result.text.strip())
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
