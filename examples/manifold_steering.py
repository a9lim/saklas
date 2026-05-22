"""Author, fit, and steer across a 2-D manifold.

Manifold steering moves a generation onto a point of a fitted manifold
through concept centroids instead of following a single straight
direction. This script authors a small "affective circumplex" manifold —
Russell's valence x arousal plane as a 2-D disk: eight mood nodes on the
unit circle plus a neutral node at the center — fits it against the
model, and generates the same prompt at several (valence, arousal)
points. Run:

    python examples/manifold_steering.py --model google/gemma-3-4b-it

Because the domain is 2-D, a steering position is a coordinate pair:
`circumplex%<valence>,<arousal>`. Intensity is intrinsic — the radius
from the neutral center — so `circumplex%0.3,0.3` is a mild mood and
`circumplex%0.7,0.7` a strong one along the same direction.

The manifold folder is written under ~/.saklas/manifolds/local/circumplex
on the first run and reused (and re-fit only on a cache miss) after that.
"""

from __future__ import annotations

import argparse
import json

from saklas import SaklasSession, SamplingConfig
from saklas.io.paths import manifolds_dir

# Nine nodes of the affective circumplex: eight moods on the unit circle
# at their (valence, arousal) coordinates, plus `neutral` at the center.
# valence runs left(-) to right(+); arousal runs down(-) to up(+).
_R = 0.7071  # unit-circle diagonal
NODES = {
    "elated": ((_R, _R), [
        "Everything is going right and I can barely sit still.",
        "My heart is racing and I can't stop grinning.",
        "I'm buzzing with energy and ready to celebrate.",
        "I feel lit up from the inside, like the whole day is mine.",
    ]),
    "happy": ((1.0, 0.0), [
        "It's a good day and I'm content with how things are going.",
        "Things feel light and easy this afternoon.",
        "I'm in a genuinely cheerful mood for no particular reason.",
        "I feel warm and pleased with how the morning went.",
    ]),
    "serene": ((_R, -_R), [
        "I feel deeply at peace, settled into a quiet kind of joy.",
        "Everything around me feels gentle and unhurried.",
        "I'm relaxed and quietly satisfied, with nothing pressing on me.",
        "A slow, contented quiet has settled over me.",
    ]),
    "calm": ((0.0, -1.0), [
        "I feel steady and even, with my energy low and untroubled.",
        "My mind is quiet and I'm not worried about anything.",
        "Nothing feels urgent right now and I'm fine with that.",
        "I'm relaxed, breathing easy, in no particular hurry.",
    ]),
    "weary": ((-_R, -_R), [
        "I'm drained and everything feels like it takes too much effort.",
        "I feel heavy and slow, like the day has worn me thin.",
        "I'm too tired to care much about anything right now.",
        "I'm running on empty and counting the hours until I can rest.",
    ]),
    "gloomy": ((-1.0, 0.0), [
        "A grey mood has settled over me and I can't shake it.",
        "Nothing seems worth doing and I'd rather just be alone.",
        "There's a dull ache of sadness I can't quite name.",
        "The afternoon feels bleak and I just want it to end.",
    ]),
    "distressed": ((-_R, _R), [
        "My thoughts are racing and I can't calm down.",
        "My chest is tight and I can't stop worrying.",
        "I'm overwhelmed and on the edge of tears.",
        "I'm shaking, frightened, and I feel trapped.",
    ]),
    "alert": ((0.0, 1.0), [
        "I'm wide awake and keyed up, scanning for whatever comes next.",
        "I'm tense and watchful, unable to relax my guard.",
        "My senses are sharp and I'm braced for something.",
        "I'm on high alert, coiled and waiting.",
    ]),
    "neutral": ((0.0, 0.0), [
        "I don't feel much of anything in particular right now.",
        "It's an ordinary moment and nothing about it stands out.",
        "Things are neither good nor bad; they simply are.",
        "I'm just going through the motions, not moved either way.",
    ]),
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
            json.dumps(NODES[label][1], indent=2)
        )
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": 3,
        "name": "circumplex",
        "description": "Russell's affective circumplex as a 2-D valence x "
                       "arousal disk",
        "domain": {
            "type": "box",
            "axes": [
                {"name": "valence", "periodic": False, "lo": -1.0, "hi": 1.0},
                {"name": "arousal", "periodic": False, "lo": -1.0, "hi": 1.0},
            ],
        },
        "nodes": [
            {"label": label, "coords": list(NODES[label][0])}
            for label in order
        ],
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
    ap.add_argument(
        "--positions", default="0,0;0.6,0.4;-0.6,0.5;0.5,-0.6",
        help="semicolon-separated valence,arousal points in the disk",
    )
    args = ap.parse_args()

    folder = author_manifold()
    positions = [
        tuple(float(c) for c in pair.split(","))
        for pair in args.positions.split(";")
    ]

    with SaklasSession.from_pretrained(args.model, device="auto") as session:
        # Fit the manifold for this model (cache-hits on later runs).
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
            # `circumplex%<valence>,<arousal>` places the generation at
            # that point of the fitted manifold; the coefficient is the
            # blend strength.
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
