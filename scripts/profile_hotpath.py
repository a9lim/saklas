#!/usr/bin/env python
"""Localize the steered+probed decode bottleneck by A/B-ing the four arms.

The audit pins the 60% GPU / 3 tok-s symptom on per-token CPU-side work
(host syncs + eager kernel dispatch) starving the MPS pipeline. Which lever
matters depends on one number you haven't measured: the vanilla baseline.
This isolates it.

    python scripts/profile_hotpath.py <model_id> [steer_expr] [max_tokens]

Reads tok_per_sec straight off GenerationResult. Warmup gen first (paged
weights, Metal shader compile), then 3 timed gens per arm, report the median.
"""
from __future__ import annotations

import statistics
import sys

from saklas import SaklasSession, SamplingConfig

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-12b-it"
STEER = sys.argv[2] if len(sys.argv) > 2 else "0.5 confident.uncertain"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 64
PROMPT = "Describe a walk through a forest at dusk, then explain why you chose those details."


def run(session: SaklasSession, *, steering: str | None) -> float:
    samp = SamplingConfig(temperature=0.7, max_tokens=MAXTOK, seed=42)
    rs = session.generate(PROMPT, steering=steering, sampling=samp)
    return rs.first.tok_per_sec


def arm(label: str, session: SaklasSession, *, steering: str | None) -> None:
    run(session, steering=steering)  # warmup (discarded)
    tps = [run(session, steering=steering) for _ in range(3)]
    print(f"{label:<34} {statistics.median(tps):6.2f} tok/s   "
          f"(runs: {', '.join(f'{t:.1f}' for t in tps)})")


def main() -> None:
    print(f"model={MODEL}  max_tokens={MAXTOK}  steer={STEER!r}\n")

    # --- arms with NO probes attached (probes=[] skips the default roster) ---
    print("# no probes attached")
    with SaklasSession.from_pretrained(MODEL, device="auto", probes=[]) as s:
        n_layers = len(s._layers)
        arm("vanilla (no steer, no probe)", s, steering=None)
        arm("steered only", s, steering=STEER)

    # --- arms with the FULL default roster (17 concepts + fitted personas/emotions) ---
    print("\n# full default probe roster (per-token scoring on)")
    with SaklasSession.from_pretrained(MODEL, device="auto") as s:
        roster = s._monitor.probe_names
        span = sorted(s._monitor.probe_layers())
        print(f"  roster: {len(roster)} probes  "
              f"covering {len(span)}/{n_layers} layers  -> {roster}\n")
        arm("probed only (no steer)", s, steering=None)
        arm("steered + probed (your case)", s, steering=STEER)


if __name__ == "__main__":
    main()
