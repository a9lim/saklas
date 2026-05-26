#!/usr/bin/env python3
"""Sanity-check the slim statement-generator prompts against a target model.

Runs :meth:`SaklasSession.generate_statements` with ``share_moment=True``
on four diverse bipolar axes and prints the scenarios + every moment-
shared pair the model produces.  No extraction, no disk side effects —
just exercises the slim prompts and shows what comes out, so a human
can sanity-check whether the slimmer phrasing holds quality at the
chosen model scale.

The four axes probe different load-bearing properties of the prompts:

- ``happy.sad``     — bipolar affect (the canonical test axis)
- ``deer.wolf``     — bipolar non-human animal (anti-allegory canary;
                      a model that drifts to human-social framing will
                      emit "I feel attacked" instead of literal deer/
                      wolf life)
- ``formal.casual`` — bipolar register (style-only axis; pairs should
                      differ in formality, not subject matter)
- ``human.ai``      — bipolar identity (semantically loaded axis; pairs
                      should foreground the ontological difference)

Defaults to ``google/gemma-4-31b-it`` (the strong-model gate).  Override
with ``--model``.  Pass ``--axes happy.sad,deer.wolf`` to subset.
"""
from __future__ import annotations

import argparse
import sys
import time

from saklas.core.session import SaklasSession


AXES: list[tuple[str, str, str]] = [
    ("happy", "sad", "bipolar affect"),
    ("deer", "wolf", "bipolar non-human animal — anti-allegory canary"),
    ("formal", "casual", "bipolar register"),
    ("human", "ai", "bipolar identity"),
]


def _parse_axes(raw: str | None) -> list[tuple[str, str, str]]:
    if not raw:
        return AXES
    wanted = {a.strip() for a in raw.split(",") if a.strip()}
    out = [(p, n, label) for p, n, label in AXES if f"{p}.{n}" in wanted]
    if not out:
        valid = ", ".join(f"{p}.{n}" for p, n, _ in AXES)
        print(
            f"--axes: no matches in {raw!r}; valid: {valid}",
            file=sys.stderr,
        )
        sys.exit(2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model", default="google/gemma-4-31b-it",
        help="HF model id (default: %(default)s)",
    )
    ap.add_argument(
        "--n-scenarios", type=int, default=5,
        help="number of scenarios per axis (default: %(default)d)",
    )
    ap.add_argument(
        "--statements-per-cell", type=int, default=3,
        help="moment-shared groups per scenario (default: %(default)d)",
    )
    ap.add_argument(
        "--device", default="auto",
        help="device override (default: %(default)s)",
    )
    ap.add_argument(
        "--axes", default=None,
        help=(
            "comma-separated subset of axes to run "
            "(default: all four). e.g. --axes happy.sad,deer.wolf"
        ),
    )
    ap.add_argument(
        "--show-prompts", action="store_true",
        help="print the exact scenario + statement prompts the engine emits",
    )
    args = ap.parse_args()

    axes = _parse_axes(args.axes)

    print(f"[load] {args.model} on {args.device}...", flush=True)
    t0 = time.time()
    session = SaklasSession.from_pretrained(
        args.model, device=args.device, probes=[],
    )
    print(
        f"[load] ready on {session._device} ({session._dtype}) "
        f"in {time.time()-t0:.1f}s",
        flush=True,
    )

    # Optionally intercept ``_run_generator`` to log every prompt.  We
    # wrap the bound method so the rest of the engine sees the original
    # behavior — only side-effect is print().
    if args.show_prompts:
        original = session._run_generator
        prompt_count = {"i": 0}

        def _wrapped(system_msg, prompt, max_new_tokens, **kwargs):
            prompt_count["i"] += 1
            print(
                f"\n  [prompt #{prompt_count['i']}, "
                f"max_new={max_new_tokens}]"
            )
            for line in prompt.splitlines():
                print(f"  | {line}")
            print("  [/prompt]\n")
            return original(system_msg, prompt, max_new_tokens, **kwargs)

        session._run_generator = _wrapped  # type: ignore[method-assign]

    for pos, neg, label in axes:
        print()
        print(f"=== {pos}.{neg} ({label}) ===", flush=True)
        t0 = time.time()
        corpora = session.generate_statements(
            [pos, neg],
            n_scenarios=args.n_scenarios,
            statements_per_cell=args.statements_per_cell,
            share_moment=True,
            on_progress=lambda m: print(f"  {m}", flush=True),
        )
        dt = time.time() - t0

        pairs = list(zip(corpora[pos], corpora[neg]))
        K = args.statements_per_cell
        n_scn = len(pairs) // K if K else 0
        print(
            f"  ({n_scn} scenarios × {K} pairs each, "
            f"{len(pairs)} pairs total in {dt:.1f}s)"
        )

        for s_idx in range(n_scn):
            block = pairs[s_idx * K:(s_idx + 1) * K]
            print(f"  --- scenario {s_idx + 1} ---")
            for j, (p, n) in enumerate(block, 1):
                print(f"    {j}a ({pos}): {p}")
                print(f"    {j}b ({neg}): {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
