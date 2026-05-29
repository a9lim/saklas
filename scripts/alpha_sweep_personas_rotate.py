"""Alpha sweep on a persona manifold node under the new angular-in-subspace
(``subspace_rotate``) operator.

The bundled ``personas`` manifold was previously characterized under
``subspace_replace`` (the ``injection_mode="additive"`` path for manifold
terms) with a sweet-spot around α ≈ 0.20 and collapse beyond α ≥ 0.30.
Under angular mode the manifold hook now dispatches to
``subspace_rotate``: rotate ``h_par`` toward the target by ``α · θ_max``
inside the affine subspace plane, leaving ``h_perp`` untouched. ``||h
- mean||`` is preserved by construction; no destructive in-subspace
overwrite. The α-regime is expected to differ.

This script runs the same persona under both operators on
``gemma-4-31b-it`` (the reference model the existing α-regime was
characterized against, manifold already fit) with a deterministic seed
across alphas, so the contrast is clean.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from saklas import SamplingConfig, SaklasSession


MODEL_ID = "google/gemma-4-31b-it"
DEFAULT_PERSONA = "hacker"
DEFAULT_PROMPT = "Tell me about your morning."
DEFAULT_MAX_TOKENS = 120
DEFAULT_SEED = 42
# Under share-weighted manifold steering the per-layer α-budget is
# layer-count-invariant, so the fluent band is expected to widen
# substantially (roughly to vector-steering territory [0.3, 0.85]).
DEFAULT_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)


def run_sweep(
    *,
    persona: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    alphas: tuple[float, ...],
    mode: str,
    out_path: Path,
) -> None:
    print(f"loading {MODEL_ID}  (injection_mode={mode})")
    t0 = time.time()
    session = SaklasSession.from_pretrained(
        MODEL_ID, device="auto", injection_mode=mode,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  device={session._device}  dtype={session._dtype}")

    # Ensure the bundled personas manifold has a v4 fit (populated EV
    # ratios).  If the per-model safetensors is absent (because we
    # deleted it ahead of the v3→v4 format bump), trigger a refit
    # through the public session API before sweeping.
    from pathlib import Path as _Path
    from saklas.io.paths import manifold_dir, safe_model_id
    personas_dir = manifold_dir("default", "personas")
    fit_path = personas_dir / (safe_model_id(MODEL_ID) + ".safetensors")
    if not _Path(fit_path).exists():
        print(f"refitting bundled personas manifold for {MODEL_ID} ...")
        t1 = time.time()
        session.extract_manifold(personas_dir)
        print(f"  refit done in {time.time() - t1:.1f}s")

    runs: list[dict[str, object]] = []
    results = {
        "model": MODEL_ID,
        "persona": persona,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "seed": seed,
        "runs": runs,
    }

    sampling = SamplingConfig(
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
        seed=seed,
    )

    # Each call forks off the synthetic root, so no prior turn leaks
    # into the model's context.  Without this, the loom tree
    # accumulates turn-on-turn and later runs see earlier outputs as
    # conversation history.
    root_id = session.tree.root_id

    # Each call forks off the synthetic root and runs with thinking
    # disabled so we see the model's *response* (not the CoT scratchpad,
    # which obscures persona expression with meta-commentary about
    # "Option 1 vs Option 2" framing).
    common_kwargs: dict[str, object] = {
        "sampling": sampling,
        "parent_node_id": root_id,
        "thinking": False,
    }

    # Baseline: no steering.
    print("\n== baseline (no steering) ==")
    t0 = time.time()
    base = session.generate(prompt, **common_kwargs).first  # pyright: ignore[reportArgumentType]  # dict[str, object] spreads as object-typed kwargs
    print(f"  [{time.time() - t0:.1f}s]  {base.text!r}")
    runs.append({
        "mode": "baseline",
        "alpha": 0.0,
        "text": base.text,
        "n_tokens": len(base.tokens),
    })

    # Main sweep at the session's injection mode.
    label = "angular (rotate)" if mode == "angular" else "additive (replace)"
    for alpha in alphas:
        expr = f"{alpha:g} personas%{persona}"
        print(f"\n== {label}  {expr} ==")
        t0 = time.time()
        out = session.generate(
            prompt, steering=expr, **common_kwargs,  # pyright: ignore[reportArgumentType]  # dict[str, object] spreads as object-typed kwargs
        ).first
        print(f"  [{time.time() - t0:.1f}s]  {out.text!r}")
        runs.append({
            "mode": mode,
            "alpha": alpha,
            "expression": out.applied_steering,
            "text": out.text,
            "n_tokens": len(out.tokens),
        })

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nresults → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", default=DEFAULT_PERSONA)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--alphas",
        type=lambda s: tuple(float(x) for x in s.split(",")),
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        "--mode", choices=("angular", "additive"), default="angular",
        help="Manifold injection mode (angular → subspace_rotate, "
             "additive → subspace_replace).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/alpha_sweep_personas_rotate.json"),
    )
    args = parser.parse_args()

    run_sweep(
        persona=args.persona,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        seed=args.seed,
        alphas=args.alphas,
        mode=args.mode,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
