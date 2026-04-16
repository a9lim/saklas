"""Prototype harness for the open-ended scenario + pair generator.

Thin wrapper over the library-side ``SaklasSession.generate_scenarios``
and ``SaklasSession.generate_pairs(mode="open_ended")`` methods. Does
not own any prompt logic — the script is purely a test harness that
prints what the library produces.

Defaults to ``google/gemma-4-e4b-it`` (a weak 4B model) so that
prompt-robustness regressions surface loudly. Override with ``--model``.

Usage:
    python scripts/prototype_open_scenarios.py
    python scripts/prototype_open_scenarios.py --concept deer.wolf
    python scripts/prototype_open_scenarios.py --skip-pairs
    python scripts/prototype_open_scenarios.py --concept deer.wolf \\
        --use-scenarios path/to/scenarios.json
    python scripts/prototype_open_scenarios.py --model google/gemma-4-31b-it
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from saklas.core.session import SaklasSession


MODEL_ID = "google/gemma-4-e4b-it"

# (positive_pole, negative_pole_or_None) — None means monopolar.
TEST_CONCEPTS: dict[str, tuple[str, str | None]] = {
    "deer.wolf":          ("deer", "wolf"),
    "hope.despair":       ("hope", "despair"),
    "angry.calm":         ("angry", "calm"),
    "masculine.feminine": ("masculine", "feminine"),
}


def _load_scenarios_file(path: Path) -> list[str]:
    """Load scenarios from a JSON file — accept either ``{"scenarios": [...]}``
    or a bare list. Matches the on-disk format written by
    ``SaklasSession.extract(mode="open_ended")``.
    """
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
        return [str(s) for s in data["scenarios"]]
    if isinstance(data, list):
        return [str(s) for s in data]
    raise ValueError(
        f"scenarios file {path} must be a JSON list or {{'scenarios': [...]}}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--concept",
        action="append",
        default=None,
        help="Restrict to one or more test concepts (repeatable).",
    )
    parser.add_argument(
        "--skip-pairs",
        action="store_true",
        help="Only generate scenarios; skip pair generation.",
    )
    parser.add_argument(
        "--use-scenarios",
        type=Path,
        default=None,
        metavar="PATH",
        help="Load scenarios from a JSON file instead of generating them. "
             "Requires exactly one --concept.",
    )
    args = parser.parse_args()

    names = args.concept if args.concept else list(TEST_CONCEPTS.keys())
    unknown = [n for n in names if n not in TEST_CONCEPTS]
    if unknown:
        print(f"Unknown concepts: {unknown}", file=sys.stderr)
        print(f"Known: {list(TEST_CONCEPTS.keys())}", file=sys.stderr)
        return 2

    preloaded_scenarios: list[str] | None = None
    if args.use_scenarios is not None:
        if len(names) != 1:
            print(
                "--use-scenarios requires exactly one --concept",
                file=sys.stderr,
            )
            return 2
        if not args.use_scenarios.exists():
            print(f"--use-scenarios path not found: {args.use_scenarios}", file=sys.stderr)
            return 2
        preloaded_scenarios = _load_scenarios_file(args.use_scenarios)
        print(
            f"Loaded {len(preloaded_scenarios)} scenarios from "
            f"{args.use_scenarios}\n",
            flush=True,
        )

    print(f"Loading {args.model}...", flush=True)
    session = SaklasSession.from_pretrained(args.model, device="auto", probes=[])
    print(f"Loaded on {session._device} ({session._dtype})\n", flush=True)

    def _progress(msg: str) -> None:
        print(f"    {msg}", flush=True)

    for name in names:
        pos, neg = TEST_CONCEPTS[name]
        mode_label = "bipolar" if neg else "monopolar"
        print(f"=== {name} ({mode_label}: {pos}" + (f" / {neg})" if neg else ")"))

        # Resolve scenarios (either preloaded or freshly generated).
        if preloaded_scenarios is not None:
            scenarios = preloaded_scenarios
            print(f"  (using {len(scenarios)} preloaded scenarios)")
        else:
            t0 = time.time()
            scenarios = session.generate_scenarios(
                pos, neg, on_progress=_progress,
            )
            dt = time.time() - t0
            print(f"  ({len(scenarios)} domains in {dt:.1f}s)")

        for i, scn in enumerate(scenarios, 1):
            print(f"  {i:>2}. {scn}")
        print(flush=True)

        if args.skip_pairs:
            continue

        t0 = time.time()
        pairs = session.generate_pairs(
            pos, neg,
            mode="open_ended",
            scenarios=scenarios,
            on_progress=_progress,
        )
        dt = time.time() - t0

        # Present pairs grouped by scenario for readability. Each batch
        # from _generate_pairs_open_ended is (pairs_per_scenario) long
        # and corresponds to scenarios in order, so we slice.
        pairs_per_scenario = max(1, -(-len(pairs) // max(1, len(scenarios))))
        for i, scn in enumerate(scenarios):
            start = i * pairs_per_scenario
            end = start + pairs_per_scenario
            batch = pairs[start:end]
            if not batch:
                continue
            print(f"  --- domain {i + 1}: {scn}")
            for j, (a, b) in enumerate(batch, 1):
                print(f"    {j}a ({pos}): {a}")
                print(f"    {j}b ({neg if neg else 'opposite'}): {b}")
        print(
            f"  ({len(pairs)} pairs across {len(scenarios)} domains "
            f"in {dt:.1f}s)\n",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
