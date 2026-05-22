"""Author, fit, and steer along a cyclic manifold.

Manifold steering follows a smooth spline through an ordered set of
concept centroids instead of a single straight direction. This script
authors a small "affective circumplex" manifold (Russell's valence x
arousal circle, eight nodes, cyclic), fits it against the model, and
generates the same prompt at several positions around the loop. Run:

    python examples/manifold_steering.py --model google/gemma-3-4b-it

The manifold folder is written under ~/.saklas/manifolds/local/circumplex
on the first run and reused (and re-fit only on a cache miss) after that.
"""

from __future__ import annotations

import argparse
import json

from saklas import SaklasSession, SamplingConfig
from saklas.io.paths import manifolds_dir

# Eight nodes traced counter-clockwise around the affective circumplex;
# the manifold is cyclic, so `alert` connects back to `elated`.
NODES = {
    "elated": [
        "Everything is going right and I can barely sit still.",
        "My heart is racing and I can't stop grinning.",
        "I'm buzzing with energy and ready to celebrate.",
        "I feel lit up from the inside, like the whole day is mine.",
    ],
    "happy": [
        "It's a good day and I'm content with how things are going.",
        "Things feel light and easy this afternoon.",
        "I'm in a genuinely cheerful mood for no particular reason.",
        "I feel warm and pleased with how the morning went.",
    ],
    "serene": [
        "I feel deeply at peace, settled into a quiet kind of joy.",
        "Everything around me feels gentle and unhurried.",
        "I'm relaxed and quietly satisfied, with nothing pressing on me.",
        "A slow, contented quiet has settled over me.",
    ],
    "calm": [
        "I feel steady and even, neither up nor down.",
        "My mind is quiet and I'm not worried about anything.",
        "Nothing feels urgent right now and I'm fine with that.",
        "I'm relaxed, breathing easy, in no particular hurry.",
    ],
    "weary": [
        "I'm drained and everything feels like it takes too much effort.",
        "I feel heavy and slow, like the day has worn me thin.",
        "I'm too tired to care much about anything right now.",
        "I'm running on empty and counting the hours until I can rest.",
    ],
    "gloomy": [
        "A grey mood has settled over me and I can't shake it.",
        "Nothing seems worth doing and I'd rather just be alone.",
        "There's a dull ache of sadness I can't quite name.",
        "The afternoon feels bleak and I just want it to end.",
    ],
    "distressed": [
        "My thoughts are racing and I can't calm down.",
        "My chest is tight and I can't stop worrying.",
        "I'm overwhelmed and on the edge of tears.",
        "I'm shaking, frightened, and I feel trapped.",
    ],
    "alert": [
        "I'm wide awake and keyed up, scanning for whatever comes next.",
        "I'm tense and watchful, unable to relax my guard.",
        "My senses are sharp and I'm braced for something.",
        "I'm on high alert, coiled and waiting.",
    ],
}


def author_manifold() -> str:
    """Write the circumplex manifold folder if it isn't already there."""
    folder = manifolds_dir() / "local" / "circumplex"
    if (folder / "manifold.json").exists():
        return str(folder)
    (folder / "nodes").mkdir(parents=True, exist_ok=True)
    order = list(NODES)
    for idx, label in enumerate(order):
        (folder / "nodes" / f"{idx:02d}_{label}.json").write_text(
            json.dumps(NODES[label], indent=2)
        )
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": 2,
        "name": "circumplex",
        "description": "Russell's affective circumplex as a cyclic manifold",
        "cyclic": True,
        "nodes": order,
        "files": {},
    }, indent=2))
    return str(folder)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="blend fraction; 0.8+ tends to over-steer")
    ap.add_argument("--prompt",
                    default="Describe walking through the city this morning.")
    ap.add_argument("--positions", default="0.0,0.25,0.5,0.75",
                    help="comma-separated t values in [0,1] around the loop")
    args = ap.parse_args()

    folder = author_manifold()
    positions = [float(t) for t in args.positions.split(",")]

    with SaklasSession.from_pretrained(args.model, device="auto") as session:
        # Fit the manifold for this model (cache-hits on later runs).
        manifold = session.extract_manifold(folder)
        print(f"fitted '{manifold.name}': {len(manifold.layers)} layers, "
              f"{len(manifold.node_labels)} nodes\n"
              f"  loop: {' -> '.join(manifold.node_labels)}")

        sampling = SamplingConfig(max_tokens=110, temperature=0.7, seed=42)

        baseline = session.generate(
            args.prompt, stateless=True, sampling=sampling,
        )
        print("\n===== unsteered =====")
        print(baseline.text.strip())

        for t in positions:
            # `circumplex%<t>` places the generation at position t along
            # the fitted spline; the coefficient is the blend strength.
            result = session.generate(
                args.prompt,
                steering=f"{args.alpha} circumplex%{t:g}",
                stateless=True,
                sampling=sampling,
            )
            print(f"\n===== circumplex%{t:g} (alpha={args.alpha:g}) =====")
            print(result.text.strip())


if __name__ == "__main__":
    main()
