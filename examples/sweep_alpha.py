"""Sweep a steering vector's alpha and print generations side-by-side.

Useful for finding the coherent-nuanced band (~0.4-0.8) on a new model or
concept. Run:

    python examples/sweep_alpha.py --concept happy --prompt "What makes a good day?"
"""

from __future__ import annotations

import argparse

from saklas import SaklasSession, SamplingConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--concept", default="happy")
    ap.add_argument("--prompt", default="What makes a good day?")
    ap.add_argument("--alphas", default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--max-tokens", type=int, default=120)
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]

    with SaklasSession.from_pretrained(args.model, device="auto") as session:
        name, profile = session.extract(args.concept)
        session.steer(name, profile)

        for alpha in alphas:
            result = session.generate(
                args.prompt,
                steering=f"{alpha} {name}" if alpha else None,
                stateless=True,
                sampling=SamplingConfig(seed=0),
            )
            print(f"\n=== alpha={alpha:.2f} ===")
            print(result.text[: args.max_tokens * 5])


if __name__ == "__main__":
    main()
