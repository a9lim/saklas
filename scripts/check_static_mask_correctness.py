"""Correctness: compiled+static-mask vs eager greedy text on a hybrid model.

The static-mask patch (cuda_graphs._patch_static_sliding_mask) short-circuits a
non-sliding StaticSlidingWindowLayer's get_mask_sizes to the constant
(max_cache_len, 0).  This is provably bit-identical to the original for a cache
that never slides — but since it touches attention masking, confirm empirically
that compiled greedy output stays coherent and tracks the eager path.

Run: SAKLAS_MEASURE_MODEL=google/gemma-4-12B-it python3 scripts/check_static_mask_correctness.py
"""
import os
import sys

from saklas import SamplingConfig, SaklasSession

MODEL = os.environ.get("SAKLAS_MEASURE_MODEL", "google/gemma-4-12B-it")
PROMPTS = [
    "Name the planets of the solar system in order from the sun.",
    "Explain in two sentences why the sky is blue.",
]
S = SamplingConfig(temperature=0.0, max_tokens=64, seed=0)


def main() -> int:
    with SaklasSession.from_pretrained(MODEL, device="auto", probes=[], compile=True) as s:
        compiled = bool(getattr(s, "_compiled", False))
        print(f"compiled={compiled}  static_cache_active={s._static_cache_active}\n")
        if not compiled:
            print("compile didn't stick — can't compare the compiled path")
            return 1
        eager = getattr(s._model, "_orig_mod", None)
        for p in PROMPTS:
            # Compiled path (static-mask active via make_static_cache).
            ctext = s.generate(p, sampling=S).first.text.strip()
            # Eager path: temporarily swap to the underlying module.
            comp = s._model
            try:
                if eager is not None:
                    s._model = eager
                    s._compiled = False
                etext = s.generate(p, sampling=S).first.text.strip()
            finally:
                s._model = comp
                s._compiled = True
            print(f"PROMPT: {p}")
            print(f"  COMPILED+STATICMASK: {ctext[:200]}")
            print(f"  EAGER:               {etext[:200]}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
