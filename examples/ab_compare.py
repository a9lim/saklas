"""Generate the same prompt with and without steering, dump probe readings.

Shows how the activation trajectory shifts when a steering vector is applied,
using the built-in probe library as the measurement. Run:

    python examples/ab_compare.py --concept happy --prompt "Describe your morning."
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from saklas import SaklasSession, SamplingConfig

if TYPE_CHECKING:
    from saklas import GenerationResult


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--concept", default="happy")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--prompt", default="Describe your morning.")
    args = ap.parse_args()

    with SaklasSession.from_pretrained(args.model, device="auto") as session:
        name, profile = session.extract(args.concept)
        session.steer(name, profile)

        unsteered = session.generate(
            args.prompt, stateless=True, sampling=SamplingConfig(seed=0),
        )
        steered = session.generate(
            args.prompt,
            steering=f"{args.alpha} {name}",
            stateless=True,
            sampling=SamplingConfig(seed=0),
        )

    print("\n=== unsteered ===")
    print(unsteered.first.text)
    print("\n=== steered (alpha={:.2f}) ===".format(args.alpha))
    print(steered.first.text)

    def probe_summary(result: GenerationResult) -> dict[str, float]:
        """Axis-0 aggregate at the last generated content token."""
        return {
            name: round(reading.coords[0], 3)
            for name, reading in result.probe_readings.items()
            if reading.coords
        }

    print("\n=== probe means ===")
    print(json.dumps({
        "unsteered": probe_summary(unsteered.first),
        "steered": probe_summary(steered.first),
    }, indent=2))


if __name__ == "__main__":
    main()
