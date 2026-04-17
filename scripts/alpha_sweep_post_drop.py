"""Alpha sweep after drop_edges=(2,2) fix to recalibrate _STEER_GAIN.

Runs `angry.calm` extraction (using new default edge-drop) on two reference
models sequentially, sweeps α ∈ {0.3, 0.45, 0.6, 0.75}, and prints output
for visual coherent/incoherent boundary inspection.
"""
from __future__ import annotations

import gc
import sys
import time

import torch

from saklas.core.session import SaklasSession
from saklas.core.sampling import SamplingConfig


MODELS = [
    "google/gemma-4-e4b-it",
    "Qwen/Qwen3.5-9b",
]
ALPHAS = [0.3, 0.45, 0.6, 0.75]
PROMPT = "A customer just told me the package I shipped arrived broken. Write my reply."
SEED = 1234
MAX_TOKENS = 180


def _hr(ch: str = "=", w: int = 72) -> str:
    return ch * w


def sweep(model_id: str) -> None:
    print(_hr("#"))
    print(f"MODEL: {model_id}")
    print(_hr("#"), flush=True)

    print("Loading…", flush=True)
    t0 = time.time()
    session = SaklasSession.from_pretrained(
        model_id, device="auto", max_tokens=MAX_TOKENS, probes=[],
    )
    print(f"Loaded on {session._device} ({session._dtype}) in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    name, profile = session.extract(
        "angry", baseline="calm",
        on_progress=lambda m: print(f"  [extract] {m}", flush=True),
    )
    print(f"[extract] done in {time.time()-t0:.1f}s", flush=True)

    norms = {l: float(v.norm().item()) for l, v in profile.items()}
    mean_n = sum(norms.values()) / len(norms)
    peak = max(norms.values())
    kept = sorted(norms.keys())
    top5 = sorted(norms.items(), key=lambda x: -x[1])[:5]
    print(
        f"\nProfile: retained layers={len(norms)} "
        f"(range L{kept[0]}..L{kept[-1]})  "
        f"mean_norm={mean_n:.4f}  peak={peak:.4f}  "
        f"peak/mean={peak/mean_n:.2f}"
    )
    print(f"top-5 layers by ||baked||: {top5}\n", flush=True)

    session.steer(name, profile)

    for alpha in ALPHAS:
        print(_hr("-"))
        print(f"α = {alpha:+.2f}")
        print(_hr("-"), flush=True)
        session.clear_history()
        result = session.generate(
            PROMPT,
            steering={name: alpha},
            sampling=SamplingConfig(seed=SEED, max_tokens=MAX_TOKENS),
            thinking=False,
            stateless=True,
        )
        print(result.text.strip())
        print(flush=True)

    # Unload before next model
    session.unsteer(name)
    del session
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print(flush=True)


def main() -> int:
    for m in MODELS:
        sweep(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
