"""Real saklas end-to-end: compile=True on MPS, vanilla / steered / probed."""
import statistics, sys, time
from saklas import SaklasSession, SamplingConfig
MODEL="google/gemma-3-4b-it"
PROBES=["formal.casual","direct.indirect","verbose.concise","warm.clinical",
        "humorous.serious","creative.conventional","technical.accessible"]
P="Walk me through how you'd plan a small dinner party, step by step."
S=SamplingConfig(temperature=0.7, max_tokens=96, seed=7)
compile_on = "--compile" in sys.argv
with SaklasSession.from_pretrained(MODEL, device="auto", probes=[], compile=compile_on) as s:
    print(f"compile={compile_on}  compiled={getattr(s,'_compiled',None)}  static_cache_active={getattr(s,'_static_cache_active',None)}")
    def g(steer):
        return s.generate(P, steering=steer, sampling=S).first.tok_per_sec
    # warm (first 2 gens pay compile cost for prefill+decode shapes)
    g(None); g(None); g("0.5 formal.casual"); g("0.5 formal.casual")
    van=[g(None) for _ in range(4)]
    steer=[g("0.5 formal.casual") for _ in range(4)]
    print(f"  VANILLA: {statistics.median(van):5.2f} tok/s  {[round(x,1) for x in van]}")
    print(f"  STEERED: {statistics.median(steer):5.2f} tok/s  {[round(x,1) for x in steer]}")
    for p in PROBES: s.add_probe(p)
    g("0.5 formal.casual"); g("0.5 formal.casual")  # warm probed shape
    probed=[g("0.5 formal.casual") for _ in range(4)]
    print(f"  STEERED+PROBED: {statistics.median(probed):5.2f} tok/s  {[round(x,1) for x in probed]}")
