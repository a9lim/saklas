"""Generate the same prompt with and without steering, dump probe readings.

Shows how the activation trajectory shifts when a steering vector is applied,
using the built-in probe library as the measurement. Run:

    python examples/ab_compare.py --concept happy --prompt "Describe your morning."
"""

from __future__ import annotations

import argparse
import json

from saklas import SaklasSession


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--concept", default="happy")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--prompt", default="Describe your morning.")
    args = ap.parse_args()

    with SaklasSession(args.model, device="auto") as session:
        profile = session.extract(args.concept)
        session.steer(args.concept, profile)

        unsteered = session.generate(args.prompt, stateless=True, seed=0)
        steered = session.generate(
            args.prompt,
            alphas={args.concept: args.alpha},
            stateless=True,
            seed=0,
        )

    print("\n=== unsteered ===")
    print(unsteered.text)
    print("\n=== steered (alpha={:.2f}) ===".format(args.alpha))
    print(steered.text)

    def probe_summary(result) -> dict[str, float]:
        return {name: round(r.mean, 3) for name, r in result.readings.items()}

    print("\n=== probe means ===")
    print(json.dumps({
        "unsteered": probe_summary(unsteered),
        "steered": probe_summary(steered),
    }, indent=2))


if __name__ == "__main__":
    main()
