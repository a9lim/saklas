"""Fit the bundled circumplex manifold and steer across it.

Manifold steering moves a generation onto a point of a fitted manifold
through concept centroids instead of following a single straight
direction. saklas ships a bundled `circumplex` manifold — Russell's
affective circumplex as a 2-D valence x arousal disk (eight moods on
the unit circle plus a neutral center) — that materializes into
``~/.saklas/manifolds/default/circumplex/`` on session startup. This
script fits it against the loaded model and generates the same prompt
at several (valence, arousal) points. Run:

    python examples/manifold_steering.py --model google/gemma-3-4b-it

Because the domain is 2-D, a steering position is a coordinate pair:
``circumplex%<valence>,<arousal>``. Intensity is intrinsic — the radius
from the neutral center — so ``circumplex%0.3,0.3`` is a mild mood and
``circumplex%0.7,0.7`` a strong one along the same direction. The
bundled corpus also lets you steer by node label,
``circumplex%elated`` etc., which resolves to that node's authored
coordinates.

The per-model fit is cached under
``~/.saklas/manifolds/default/circumplex/<safe_model_id>.safetensors``
and reused on subsequent runs — only the first invocation pays the
fit cost.

To inspect or modify the authored corpus see
``saklas/data/manifolds/circumplex/`` in the source tree.
"""

from __future__ import annotations

import argparse

from saklas import SaklasSession, SamplingConfig
from saklas.io.paths import manifolds_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="blend fraction; 0.8+ tends to over-steer")
    ap.add_argument("--prompt",
                    default="Describe walking through the city this morning.")
    ap.add_argument(
        "--positions", default="0,0;0.6,0.4;-0.6,0.5;0.5,-0.6",
        help="semicolon-separated valence,arousal points in the disk",
    )
    args = ap.parse_args()

    positions = [
        tuple(float(c) for c in pair.split(","))
        for pair in args.positions.split(";")
    ]

    with SaklasSession.from_pretrained(args.model, device="auto") as session:
        # The bundled circumplex is materialized into ~/.saklas/ on
        # session start; fit it for this model (cache-hits on later runs).
        folder = manifolds_dir() / "default" / "circumplex"
        manifold = session.extract_manifold(folder)
        print(f"fitted '{manifold.name}': {len(manifold.layers)} layers, "
              f"{len(manifold.node_labels)} nodes over a "
              f"{manifold.domain.intrinsic_dim}-D domain")

        sampling = SamplingConfig(max_tokens=110, temperature=0.7, seed=42)

        baseline = session.generate(
            args.prompt, stateless=True, sampling=sampling,
        )
        print("\n===== unsteered =====")
        print(baseline.text.strip())

        for valence, arousal in positions:
            # ``circumplex%<valence>,<arousal>`` places the generation
            # at that point of the fitted manifold; the coefficient is
            # the blend strength. ``circumplex%<label>`` (e.g.
            # ``circumplex%elated``) would steer toward a named node.
            result = session.generate(
                args.prompt,
                steering=f"{args.alpha} circumplex%{valence:g},{arousal:g}",
                stateless=True,
                sampling=sampling,
            )
            print(
                f"\n===== circumplex%{valence:g},{arousal:g} "
                f"(alpha={args.alpha:g}) ====="
            )
            print(result.text.strip())


if __name__ == "__main__":
    main()
