"""Slice 2 correctness: compiled persistent-capture readings vs the eager path.

On the compiled MPS path a steered+probed gen rides the persistent capture
buffers + ``ingest_persistent``; on the eager path (``return_hidden=True`` forces
it) the same gen rides the transient capture hooks.  The CPU parity tests already
prove ``ingest_persistent`` accumulates byte-identically *given the same slices*;
this checks the end-to-end composition on a real model — the persistent-path probe
readings should track the eager-path readings within torch.compile fusion noise
(MPS isn't bitwise deterministic, and Slice 1 showed compiled forwards diverge from
eager by a tiny fused-numerics delta with cos≈1).

Run: python3 scripts/validate_slice2.py
"""

import sys

from saklas import SamplingConfig, SaklasSession

MODEL = "google/gemma-3-4b-it"
PROBE = "formal.casual"
PROMPT = "Walk me through how you'd plan a small dinner party, step by step."
STEER = "0.5 formal.casual"
# Greedy + fixed seed so the two paths decode the same tokens (modulo MPS
# nondeterminism), making the per-token readings directly comparable.
SAMP = SamplingConfig(temperature=0.0, max_tokens=48, seed=11)


def agg_coord(result, name):
    r = result.probe_readings.get(name) if result.probe_readings else None
    if r is None:
        return None
    return r.coords[0] if r.coords else None


def main() -> int:
    with SaklasSession.from_pretrained(MODEL, device="auto", probes=[], compile=True) as s:
        compiled = bool(getattr(s, "_compiled", False))
        print(f"compiled={compiled}  static_cache_active={getattr(s, '_static_cache_active', None)}")
        if not compiled:
            print("torch.compile did not stick — cannot validate the compiled path.")
            return 1
        s.add_probe(PROBE)

        # Warm both decode shapes (probed-steered, and return_hidden eager).
        s.generate(PROMPT, steering=STEER, sampling=SAMP)
        s.generate(PROMPT, steering=STEER, sampling=SAMP)

        # Persistent compiled path: probed + steered, no return_hidden.
        r_persistent = s.generate(PROMPT, steering=STEER, sampling=SAMP).first
        print(f"\npersistent path: capture_persistent={s._capture_persistent}  "
              f"uses_compiled_offsets={s._steering_uses_compiled_offsets}  "
              f"tok/s={r_persistent.tok_per_sec:.2f}")
        if not s._capture_persistent:
            print("WARNING: persistent capture did NOT engage — Slice 2 routing missed.")

        # Eager path: return_hidden forces transient capture on _orig_mod.
        eager_samp = SAMP.merged_with(SamplingConfig(return_hidden=True))
        r_eager = s.generate(PROMPT, steering=STEER, sampling=eager_samp).first
        print(f"eager path:      capture_persistent={s._capture_persistent}  "
              f"tok/s={r_eager.tok_per_sec:.2f}")

        cp = agg_coord(r_persistent, PROBE)
        ce = agg_coord(r_eager, PROBE)
        print(f"\naggregate {PROBE} coord:  persistent={cp}  eager={ce}")
        if cp is not None and ce is not None:
            delta = abs(cp - ce)
            print(f"  |Δ| = {delta:.4f}")
            ok = delta < 0.15  # fusion + token-path noise; readings should track
            print(f"  readings track within fusion noise: {ok}")
            return 0 if ok else 2
        print("  one path produced no reading — investigate.")
        return 3


if __name__ == "__main__":
    sys.exit(main())
