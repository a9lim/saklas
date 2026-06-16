"""Does the scoring-stride win hold at long context, or shrink?

Per-token scoring costs a ~fixed ~18 ms/token regardless of context length, so
its share of the frame shrinks as the forward gets slower (long context). This
measures stateless lean N=1 vs N=4 at a short prompt and a long (~N-token)
prompt, so we can see whether striding matters for the user's real long-context
TUI regime or only at short context.

Run: SAKLAS_MEASURE_MODEL=google/gemma-4-12B-it python3 scripts/measure_stride_context.py --compile
"""
import os
import statistics
import sys

from saklas import SamplingConfig, SaklasSession

MODEL = os.environ.get("SAKLAS_MEASURE_MODEL", "google/gemma-4-12B-it")
MAXTOK = int(os.environ.get("SAKLAS_MEASURE_MAXTOK", "64"))
REPS = int(os.environ.get("SAKLAS_MEASURE_REPS", "3"))
PROBES = ["formal.casual", "direct.indirect", "verbose.concise", "warm.clinical",
          "humorous.serious", "creative.conventional", "technical.accessible"]
SHORT = "Walk me through how you'd plan a small dinner party, step by step."
# ~1500-token filler to force a long-context forward.
LONG = ("Here is a long background document you should consider.\n\n"
        + ("The history of computation is long and varied, spanning mechanical "
           "calculators, vacuum-tube machines, transistors, and integrated "
           "circuits, each era reshaping what problems were tractable. ") * 120
        + "\n\nNow, walk me through how you'd plan a small dinner party, step by step.")
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
        tok = s._tokenizer

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

        def g(prompt):
            return s.generate(prompt, steering="0.5 formal.casual", sampling=S,
                              stateless=True).first.tok_per_sec

        def run(prompt, n):
            state["n"] = n; state["i"] = 0; state["last"] = None
            g(prompt); g(prompt)
            state["i"] = 0; state["last"] = None
            xs = [g(prompt) for _ in range(REPS)]
            return statistics.median(xs)

        for label, prompt in (("SHORT", SHORT), ("LONG", LONG)):
            n_ctx = len(tok(prompt).input_ids)
            n1 = run(prompt, 1)
            n4 = run(prompt, 4)
            ms1, ms4 = 1e3 / n1, 1e3 / n4
            print(f"  {label:5s} (ctx≈{n_ctx:5d} tok):  "
                  f"N=1 {n1:5.2f} tok/s ({ms1:.0f}ms)  "
                  f"N=4 {n4:5.2f} tok/s ({ms4:.0f}ms)  "
                  f"→ {n4 / n1:.2f}×  saves {ms1 - ms4:.1f}ms/token")

        mon.score_single_token = orig
        s.unregister_trait_queue(loop, q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
