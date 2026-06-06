"""Re-verify soft assignment on gemma-4-12B pad after the log-volume fix.

No refit needed — pad is on disk with the σ-field from the first eval.  Just
load the model, attach the pad probe, run one steered generation, and dump
nearest vs assignment.  Before the fix: triumphant won 99.7% despite not being
in the top-4 nearest.  After: nearest and assignment should agree on the
steered target.
"""
from __future__ import annotations
from saklas import SaklasSession, SamplingConfig

MODEL = "google/gemma-4-12B-it"
PROMPT = "Tell me about your plans for the weekend."


def main() -> None:
    print("LOADING gemma-4-12B-it (MPS)", flush=True)
    with SaklasSession.from_pretrained(MODEL, device="auto") as session:
        session.add_probe("pad", top_n=6)
        for target in ("dominant", "calm", "excited"):
            print(f"\n=== steered to {target!r} ===", flush=True)
            r = session.generate(
                PROMPT,
                steering=f"0.8,1.0 pad%{target}@response",
                sampling=SamplingConfig(temperature=0.8, max_tokens=40, seed=7),
            )
            rd = r.first.probe_readings.get("pad") if r.first.probe_readings else None
            if rd is None:
                print("no reading", flush=True)
                continue
            print(f"nearest:    {[(L, round(d,3)) for L,d in rd.nearest]}", flush=True)
            print(f"assignment: {[(L, round(p,3)) for L,p in rd.assignment]}", flush=True)
            print(f"membership: {rd.membership:.3f}  "
                  f"fraction: {rd.fraction:.3f}  residual: {rd.residual:.3f}",
                  flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
