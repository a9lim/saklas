"""Profile the per-token monitor scoring cost — the probed-path bottleneck.

The probed forward is scoring-capped, not forward-capped (12.62 steered → 6.09
probed on gemma-4-12B): per decode token the lean trait-stream path runs
``score_single_token(coords_only=True)`` → ``_score_flat_batched`` (a Woodbury
``Σ⁻¹h`` apply + block matmuls per probed layer, then one ``.cpu()`` transfer).

This script answers ONE question: is that per-token cost dominated by **device
compute** (Woodbury + matmuls, scales with #layers × #probes → lever = score
every Nth token / fewer layers) or by the **host sync / Python** (one ``.cpu()``
+ reconstruction, ~constant → lever = defer/batch the transfer)?

Method: isolate scoring from the forward by capturing one real hidden-state dict
once, then timing ``score_single_token(coords_only=True)`` in a tight synced
loop — full roster, then with the big multi-node probes detached, then a single
2-node probe. Linear scaling with roster/layers ⇒ device-bound; flat ⇒ host-bound.

Run: SAKLAS_MEASURE_MODEL=google/gemma-4-12B-it python3 scripts/profile_pertoken.py
"""
from __future__ import annotations
import os
import sys
import time
from typing import Any

import torch

from saklas import SamplingConfig, SaklasSession

MODEL = os.environ.get("SAKLAS_MEASURE_MODEL", "google/gemma-4-12B-it")
REPS = int(os.environ.get("SAKLAS_PROFILE_REPS", "300"))
WARMUP = 30


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _time_scoring(session: Any, latest: Any, *, coords_only: bool, reps: int, device: Any) -> float:
    """Median per-call wall time (ms) of score_single_token over `reps`."""
    mon = session._monitor
    for _ in range(WARMUP):
        mon.score_single_token(latest, coords_only=coords_only)
    _sync(device)
    samples = []
    for _ in range(reps):
        _sync(device)
        t0 = time.perf_counter()
        mon.score_single_token(latest, coords_only=coords_only)
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    with SaklasSession.from_pretrained(MODEL, device="auto", compile=False) as s:
        device = s._device
        mon = s._monitor
        names = mon.probe_names
        player = mon.probe_layers()
        print(f"model={MODEL}  device={device}")
        print(f"probes ({len(names)}): {names}")
        print(f"probe_layers union: {len(player)} layers  {sorted(player)}")
        print("per-probe fit-layer counts:")
        for n, p in mon._probes.items():
            print(f"  {n:32s} {len(p.manifold.layers):3d} layers  "
                  f"rank~{next(iter(p.manifold.layers.values())).basis.shape[0]}")
        print()

        # Capture one real hidden-state dict by running a tiny probed gen and
        # grabbing the latest per-layer slice the capture hook retained.
        s.generate("Tell me about the ocean.",
                   sampling=SamplingConfig(temperature=0.0, max_tokens=8, seed=0))
        # Re-capture deterministically: run a single forward and pull hidden.
        latest = _capture_one(s)
        if not latest:
            print("could not capture a hidden-state dict; aborting")
            return 1
        print(f"captured hidden dict: {len(latest)} layers, "
              f"dim={next(iter(latest.values())).shape[-1]}\n")

        # --- full roster, coords_only (the trait-stream live path) ---
        t_full = _time_scoring(s, latest, coords_only=True, reps=REPS, device=device)
        n_layers_full = len(player)
        print(f"FULL roster  coords_only=True : {t_full:.3f} ms/token  "
              f"({len(names)} probes, {n_layers_full} layers)")

        t_full_rich = _time_scoring(s, latest, coords_only=False, reps=REPS, device=device)
        print(f"FULL roster  coords_only=False: {t_full_rich:.3f} ms/token  "
              f"(nearest + assignment + per-layer trace)")
        print()

        # --- detach the big multi-node probes, re-time (probe-count scaling) ---
        big = [n for n in names if n in ("default/personas", "default/emotions",
                                         "personas", "emotions")]
        for n in big:
            mon.remove_probe(n)
        if big:
            latest2 = {L: h for L, h in latest.items()}
            t_small = _time_scoring(s, latest2, coords_only=True, reps=REPS, device=device)
            print(f"WITHOUT {big}: {t_small:.3f} ms/token  "
                  f"({len(mon.probe_names)} probes, {len(mon.probe_layers())} layers)")
            print(f"  → the {len(big)} multi-node probes cost "
                  f"{t_full - t_small:.3f} ms/token ({100*(t_full-t_small)/t_full:.0f}%)")
        else:
            print("(no multi-node probes in roster to detach)")

        # --- single 2-node probe (floor) ---
        remaining = mon.probe_names
        if len(remaining) > 1:
            keep = remaining[0]
            for n in remaining[1:]:
                mon.remove_probe(n)
            t_one = _time_scoring(s, latest, coords_only=True, reps=REPS, device=device)
            print(f"SINGLE probe ({keep}): {t_one:.3f} ms/token  "
                  f"({len(mon.probe_layers())} layers)")
        print()
        print("Interpretation:")
        print("  full ≈ single  → host-sync/Python bound (defer/batch the .cpu)")
        print("  full ≫ single  → device-compute bound (score every Nth / fewer layers)")
    return 0


def _capture_one(session: Any) -> Any:
    """Run one forward, hook every probe layer, return the last per-layer slice."""
    layers = session._monitor.probe_layers()
    if not layers:
        return {}
    handles = []
    model_layers = session._layers
    hid = {}

    def _mk(idx: int) -> Any:
        def _h(_m: Any, _i: Any, out: Any) -> None:
            o = out[0] if isinstance(out, tuple) else out
            hid[idx] = o[0, -1, :].detach().to(torch.float32).clone()
        return _h

    for idx in sorted(layers):
        handles.append(model_layers[idx].register_forward_hook(_mk(idx)))
    try:
        session.generate("The quick brown fox.",
                         sampling=SamplingConfig(temperature=0.0, max_tokens=1, seed=0))
    finally:
        for h in handles:
            h.remove()
    return hid


if __name__ == "__main__":
    sys.exit(main())
