"""Alpha sweep for 'angry' using the newly regenerated bundled statements.

Extracts directly from saklas/data/vectors/angry/statements.json (bypassing
the ~/.saklas default cache so we pick up the new pairs), then generates
at alphas 0.05..0.25 with a fixed seed/prompt for comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

from saklas.session import SaklasSession
from saklas.datasource import DataSource

REPO = Path(__file__).resolve().parent.parent
STATEMENTS = REPO / "saklas" / "data" / "vectors" / "angry" / "statements.json"
MODEL_ID = "google/gemma-4-31b-it"
PROMPT = "A customer just told me the package I shipped arrived broken. Write my reply."
ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
SEED = 1234
MAX_TOKENS = 180


def main() -> int:
    pairs_raw = json.loads(STATEMENTS.read_text())
    pairs = [(p["positive"], p["negative"]) for p in pairs_raw]
    ds = DataSource(name="angry_new", pairs=pairs)

    print(f"Loading {MODEL_ID}...", flush=True)
    session = SaklasSession(MODEL_ID, device="auto", max_tokens=MAX_TOKENS)
    print(f"Loaded on {session._device}", flush=True)

    print(f"Extracting angry profile from {len(pairs)} new pairs...", flush=True)
    profile = session.extract(ds, on_progress=lambda m: print("  ", m, flush=True))
    session.steer("angry", profile)

    scores = {l: float(s) for l, (_, s) in profile.items()}
    mean_s = sum(scores.values()) / len(scores)
    top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
    print(f"\nProfile stats: layers={len(scores)} mean={mean_s:.4f} "
          f"max={max(scores.values()):.4f} peak/mean={max(scores.values())/mean_s:.2f}")
    print(f"top5 layers: {top5}\n")

    for a in ALPHAS:
        print(f"=== alpha={a} ===", flush=True)
        session.clear_history()
        result = session.generate(
            PROMPT,
            alphas={"angry": a} if a > 0 else {},
            seed=SEED,
            stateless=True,
        )
        print(result.text.strip())
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
