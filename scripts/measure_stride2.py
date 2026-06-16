"""Stride measurement, confound-free: stateless=True (no loom-tree growth) +
a registered trait queue (forces the lean per-token path despite stateless).

The first attempt was confounded — non-stateless gens append to the loom tree,
so each successive gen re-walks a longer context at prefill, masking the stride
effect entirely. stateless+trait-queue gives per-token lean scoring with a
constant-length context. A trailing re-measure of N=1 confirms no drift.

Run: SAKLAS_MEASURE_MODEL=google/gemma-4-12B-it python3 scripts/measure_stride2.py --compile
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
    def call_soon_threadsafe(self, fn, *a):
        try:
            fn(*a)
        except Exception:
            pass


class _FakeQ:
    def put_nowait(self, ev):
        pass


def main() -> int:
    compile_on = "--compile" in sys.argv
    with SaklasSession.from_pretrained(MODEL, device="auto", probes=[],
                                       compile=compile_on) as s:
        mon = s._monitor
        for p in PROBES:
            s.add_probe(p)
        loop, q = _FakeLoop(), _FakeQ()
        s.register_trait_queue(loop, q)

        def g():
            # stateless: no tree growth; trait queue: forces lean per-token path
            return s.generate(P, steering="0.5 formal.casual", sampling=S,
                              stateless=True).first.tok_per_sec

        # baseline with no probes for reference (detach, then re-add)
        orig = mon.score_single_token
        state = {"n": 1, "i": 0, "last": None}

        def strided(hidden, **kw):
            i = state["i"]; state["i"] = i + 1
            if state["n"] > 1 and (i % state["n"]) != 0 and state["last"] is not None:
                return state["last"]
            r = orig(hidden, **kw)
            state["last"] = r
            return r

        mon.score_single_token = strided  # type: ignore[method-assign]

        def run(n):
            state["n"] = n; state["i"] = 0; state["last"] = None
            g(); g()
            state["i"] = 0; state["last"] = None
            xs = [g() for _ in range(REPS)]
            return statistics.median(xs), [round(x, 1) for x in xs]

        res = {}
        order = [1, 2, 4, 8, 1]  # trailing N=1 control for drift
        for k, n in enumerate(order):
            m, xs = run(n)
            tag = " (control)" if k == len(order) - 1 else ""
            res.setdefault(n, []).append(m)
            print(f"  stride N={n}{tag}:  {m:5.2f} tok/s  {xs}")

        mon.score_single_token = orig
        s.unregister_trait_queue(loop, q)
        print()
        base = res[1][0]
        for n in (1, 2, 4, 8):
            print(f"  N={n}: {res[n][0]:5.2f} tok/s  ({res[n][0] / base:.2f}×)")
        if len(res[1]) > 1:
            drift = abs(res[1][1] - res[1][0]) / base
            print(f"  N=1 drift first→last: {res[1][0]:.2f} → {res[1][1]:.2f} "
                  f"({drift * 100:.0f}%)  — low drift ⇒ confound gone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
