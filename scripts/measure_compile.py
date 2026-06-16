"""Real saklas end-to-end: compile=True on MPS, vanilla / steered / probed."""
from __future__ import annotations
import os
import statistics
import sys
from typing import Any
# Experiment: bump dynamo's recompile limit so the gemma sliding-window
# get_mask_sizes per-decode-step recompiles (one graph per cumulative_length
# value) all compile + cache instead of falling back to eager at limit 8.
_RL = os.environ.get("SAKLAS_RECOMPILE_LIMIT")
if _RL:
    import torch._dynamo as _d
    _d.config.recompile_limit = int(_RL)
    _d.config.accumulated_recompile_limit = max(int(_RL) * 4, 256)
    print(f"recompile_limit={_d.config.recompile_limit}")
from saklas import SaklasSession, SamplingConfig
MODEL=os.environ.get("SAKLAS_MEASURE_MODEL", "google/gemma-3-4b-it")
MAXTOK=int(os.environ.get("SAKLAS_MEASURE_MAXTOK", "96"))
REPS=int(os.environ.get("SAKLAS_MEASURE_REPS", "4"))
PROBES=["formal.casual","direct.indirect","verbose.concise","warm.clinical",
        "humorous.serious","creative.conventional","technical.accessible"]
P="Walk me through how you'd plan a small dinner party, step by step."
S=SamplingConfig(temperature=0.7, max_tokens=MAXTOK, seed=7)
compile_on = "--compile" in sys.argv
with SaklasSession.from_pretrained(MODEL, device="auto", probes=[], compile=compile_on) as s:
    print(f"compile={compile_on}  compiled={getattr(s,'_compiled',None)}  static_cache_active={getattr(s,'_static_cache_active',None)}")
    def g(steer: Any) -> Any:
        return s.generate(P, steering=steer, sampling=S).first.tok_per_sec
    # warm (first 2 gens pay compile cost for prefill+decode shapes)
    g(None)
    g(None)
    g("0.5 formal.casual")
    g("0.5 formal.casual")
    van=[g(None) for _ in range(REPS)]
    steer=[g("0.5 formal.casual") for _ in range(REPS)]
    print(f"  VANILLA: {statistics.median(van):5.2f} tok/s  {[round(x,1) for x in van]}")
    print(f"  STEERED: {statistics.median(steer):5.2f} tok/s  {[round(x,1) for x in steer]}")
    for p in PROBES:
        s.add_probe(p)
    g("0.5 formal.casual")
    g("0.5 formal.casual")  # warm probed shape
    probed=[g("0.5 formal.casual") for _ in range(REPS)]
    print(f"  STEERED+PROBED: {statistics.median(probed):5.2f} tok/s  {[round(x,1) for x in probed]}")
