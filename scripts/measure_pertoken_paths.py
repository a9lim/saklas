"""Separate the two costs the 'probed' label conflates: capture overhead vs
per-token scoring.

measure_compile.py's 'STEERED+PROBED' uses a plain ``generate()`` — no trait
stream / loom / gate / live-scores — so ``need_per_token=False`` ⇒ the
**aggregate-only** capture path: hidden states copied into a tail ring every
token, but scored ONCE at finalize. That isolates *capture-hook overhead*.

The user's real TUI streams tokens with a live trait display ⇒ a registered
trait queue ⇒ ``_has_trait_consumer`` ⇒ the **lean per-token** path: each token
scored ``coords_only`` (the ~21 ms/token isolated cost). That isolates
*per-token scoring*.

This measures steered (no probes) → aggregate-only probed → lean per-token
probed, all compile=True on 12B, so the two overheads are separated.

Run: SAKLAS_MEASURE_MODEL=google/gemma-4-12B-it python3 scripts/measure_pertoken_paths.py --compile
"""
import os
import statistics
import sys

from saklas import SamplingConfig, SaklasSession

MODEL = os.environ.get("SAKLAS_MEASURE_MODEL", "google/gemma-4-12B-it")
MAXTOK = int(os.environ.get("SAKLAS_MEASURE_MAXTOK", "96"))
REPS = int(os.environ.get("SAKLAS_MEASURE_REPS", "4"))
PROBES = ["formal.casual", "direct.indirect", "verbose.concise", "warm.clinical",
          "humorous.serious", "creative.conventional", "technical.accessible"]
P = "Walk me through how you'd plan a small dinner party, step by step."
S = SamplingConfig(temperature=0.7, max_tokens=MAXTOK, seed=7)


class _FakeLoop:
    """Run the threadsafe callback inline (no real asyncio loop)."""
    def call_soon_threadsafe(self, fn, *a):
        try:
            fn(*a)
        except Exception:
            pass


class _FakeQ:
    def put_nowait(self, ev):  # drain — we only care about the scoring cost
        pass


def main() -> int:
    compile_on = "--compile" in sys.argv
    with SaklasSession.from_pretrained(MODEL, device="auto", probes=[],
                                       compile=compile_on) as s:
        print(f"compile={compile_on}  compiled={getattr(s, '_compiled', None)}  "
              f"static_cache_active={getattr(s, '_static_cache_active', None)}")

        def g(steer, *, stateless=False):
            return s.generate(P, steering=steer, sampling=S,
                              stateless=stateless).first.tok_per_sec

        # warm compile shapes
        g(None); g(None); g("0.5 formal.casual"); g("0.5 formal.casual")
        steer = [g("0.5 formal.casual") for _ in range(REPS)]
        print(f"  STEERED (no probes):      "
              f"{statistics.median(steer):5.2f} tok/s  {[round(x, 1) for x in steer]}")

        for p in PROBES:
            s.add_probe(p)
        print(f"  attached {len(PROBES)} probes; "
              f"probe_layers={len(s._monitor.probe_layers())}")

        # --- TRUE aggregate-only: stateless gen, no probe-row persist, no
        # per-token consumer ⇒ NO per-token scoring (pool once at finalize). ---
        g("0.5 formal.casual", stateless=True); g("0.5 formal.casual", stateless=True)
        agg = [g("0.5 formal.casual", stateless=True) for _ in range(REPS)]
        print(f"  PROBED aggregate-only:    "
              f"{statistics.median(agg):5.2f} tok/s  {[round(x, 1) for x in agg]}"
              f"   (stateless: capture overhead, NO per-token scoring)")

        # --- lean per-token: non-stateless plain gen persists the probe row ⇒
        # per-token scoring (coords_only) every token. The real TUI path. ---
        g("0.5 formal.casual"); g("0.5 formal.casual")
        lean = [g("0.5 formal.casual") for _ in range(REPS)]
        print(f"  PROBED lean per-token:    "
              f"{statistics.median(lean):5.2f} tok/s  {[round(x, 1) for x in lean]}"
              f"   (non-stateless: persists probe row, scores every token)")

        # --- lean + trait queue: adds the SSE 'scores' dict push per token ---
        loop, q = _FakeLoop(), _FakeQ()
        s.register_trait_queue(loop, q)
        try:
            g("0.5 formal.casual"); g("0.5 formal.casual")
            trait = [g("0.5 formal.casual") for _ in range(REPS)]
        finally:
            s.unregister_trait_queue(loop, q)
        print(f"  PROBED lean + trait SSE:  "
              f"{statistics.median(trait):5.2f} tok/s  {[round(x, 1) for x in trait]}"
              f"   (+ trait-queue serialization)")

        print()
        sm = statistics.median(steer)
        am = statistics.median(agg)
        lm = statistics.median(lean)
        tm = statistics.median(trait)

        def ms(t):
            return 1e3 / t
        print(f"  per-token frame (ms):  steered {ms(sm):.1f}  agg-only {ms(am):.1f}"
              f"  lean {ms(lm):.1f}  +trait {ms(tm):.1f}")
        print(f"  capture-hook overhead:   {ms(am) - ms(sm):+.1f} ms/token "
              f"(agg-only − steered)")
        print(f"  per-token scoring cost:  {ms(lm) - ms(am):+.1f} ms/token "
              f"(lean − agg-only)")
        print(f"  trait-queue SSE cost:    {ms(tm) - ms(lm):+.1f} ms/token "
              f"(+trait − lean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
