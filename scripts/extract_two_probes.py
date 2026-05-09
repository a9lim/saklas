"""Quick on-demand extraction for vector-compare A/B tests.

Skips the full 24-probe bootstrap (slow on big models) and just extracts
the two concepts we care about, sharing the lazily-built whitener.
"""
from __future__ import annotations

import argparse
import sys

from saklas.core.session import SaklasSession


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("concepts", nargs="+")
    args = p.parse_args()

    print(f"Loading {args.model}...", flush=True)
    session = SaklasSession.from_pretrained(
        args.model, device="auto", probes=[], max_tokens=4,
    )
    print(f"Loaded on {session._device}", flush=True)

    for concept in args.concepts:
        print(f"Extracting {concept}...", flush=True)
        name, profile = session.extract(
            concept, on_progress=lambda m: print("  ", m, flush=True),
        )
        norms = sorted(
            ((l, float(v.norm().item())) for l, v in profile.items()),
            key=lambda x: -x[1],
        )
        print(
            f"  {name}: {len(norms)} layers, top-5 = "
            f"{[(l, round(n, 3)) for l, n in norms[:5]]}",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
