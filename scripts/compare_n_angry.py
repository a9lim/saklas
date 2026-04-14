"""Compare angry profiles extracted from n=60 vs n=45 of the same
regenerated statement pool. Uses the first 32 pairs (stable subset)
so we isolate the count variable from generation variance.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from saklas.session import SaklasSession
from saklas.vectors import extract_contrastive

REPO = Path(__file__).resolve().parent.parent
STATEMENTS = REPO / "saklas" / "data" / "vectors" / "angry" / "statements.json"
MODEL_ID = "google/gemma-4-31b-it"


def profile_stats(name: str, profile: dict[int, torch.Tensor]) -> None:
    # With shares baked into magnitudes, ||baked_l|| is the same
    # "how much this layer steers" quantity that per-layer scores
    # used to encode.
    norms = {l: float(v.norm().item()) for l, v in profile.items()}
    vals = list(norms.values())
    mean_s = sum(vals) / len(vals)
    top5 = sorted(norms.items(), key=lambda x: -x[1])[:5]
    print(f"[{name}] layers={len(norms)} mean={mean_s:.4f} "
          f"max={max(vals):.4f} min={min(vals):.4f} "
          f"peak/mean={max(vals)/mean_s:.2f}")
    print(f"  top5: {[(l, round(s, 4)) for l, s in top5]}")


def main() -> int:
    pairs_raw = json.loads(STATEMENTS.read_text())
    pairs_all = [{"positive": p["positive"], "negative": p["negative"]} for p in pairs_raw]
    pairs_60 = pairs_all[:60]
    pairs_32 = pairs_all[:45]

    print(f"Loading {MODEL_ID}...", flush=True)
    session = SaklasSession(MODEL_ID, device="auto")
    print(f"Loaded on {session._device}\n", flush=True)

    print("Extracting n=60...", flush=True)
    prof60 = extract_contrastive(
        session._model, session._tokenizer, pairs_60, layers=session._layers,
    )
    profile_stats("n=60", prof60)

    print("\nExtracting n=45...", flush=True)
    prof32 = extract_contrastive(
        session._model, session._tokenizer, pairs_32, layers=session._layers,
    )
    profile_stats("n=45", prof32)

    print("\nPer-layer cosine similarity (60 vs 32):")
    sims = []
    for layer in sorted(prof60.keys()):
        v60 = prof60[layer].float().flatten()
        v32 = prof32[layer].float().flatten()
        cos = F.cosine_similarity(v60.unsqueeze(0), v32.unsqueeze(0)).item()
        # Cosine sign is arbitrary for PCA components; take absolute.
        sims.append((layer, abs(cos)))

    top5 = [l for l, _ in sorted(
        ((l, float(v.norm().item())) for l, v in prof60.items()),
        key=lambda x: -x[1],
    )[:5]]
    print("  top-5 layers of n=60 profile:")
    for l in top5:
        s = dict(sims)[l]
        print(f"    layer {l:2d}: |cos| = {s:.4f}")

    all_sims = [s for _, s in sims]
    mean_sim = sum(all_sims) / len(all_sims)
    min_sim = min(all_sims)
    print(f"  all-layer mean |cos| = {mean_sim:.4f}")
    print(f"  all-layer min  |cos| = {min_sim:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
