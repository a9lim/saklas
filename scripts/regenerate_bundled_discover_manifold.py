"""Regenerate bundled discover-mode manifolds that fold existing bipolar
steering vectors into one derived surface.

Companion to ``scripts/regenerate_bundled_manifold.py`` (the persona
manifold).  This one owns the manifolds whose nodes are the *poles* of
existing bundled vectors -- currently:

- ``cultural``  : masculine/feminine, individualist/collectivist,
                  traditional/progressive, religious/secular
                  (Hofstede + Inglehart-Welzel cultural-values space)
- ``register``  : formal/casual, direct/indirect, verbose/concise,
                  creative/conventional, humorous/serious, warm/clinical,
                  technical/accessible (style / register space)

WHY GENERATE INSTEAD OF REUSING THE 2-NODE CONCEPT CORPORA.  It is tempting
to build these corpora for free by splitting each donor concept manifold's
two node corpora into per-pole corpora.  That is wrong.  Each bundled concept is
generated against its own
*bespoke* scenario set (masculine.feminine spans "Warfare strategy /
Childcare pedagogy", traditional.progressive spans "Education / Family
Structure / Criminal Justice", ...).  Within one vector the two poles
are moment-shared, so the pole *contrast* is scenario-clean -- but
across vectors the centroids differ in scenario as much as in concept.
A discover PCA over poles drawn from disjoint scenario sets recovers
"which donor concept did this come from" (the scenario clusters), not
the cultural / register axis.  Every clean manifold -- circumplex, PAD,
personas -- shares ONE scenario set across all nodes; these must too.
So the poles are regenerated as independent discover concepts against a
single shared scenario set, exactly like the persona manifold.

The roster, description and hyperparams are read from each target's
existing ``manifold.json`` (the authored sketch is the source of truth);
this script only fills in the per-node corpora and rewrites the folder
through the canonical writer.  The ``default`` anchor node (when present
in the roster) sources its corpus from the bundled neutrals, matching
the persona convention so ``anchor_origin: true`` translates the fit's
origin onto it.

Generation cost at defaults (n_scenarios=9, statements_per_concept=5):
cultural = 1 + 8*9 = 73 calls (360 statements); register = 1 + 14*9 =
127 calls (630 statements).  Far cheaper than the persona manifold; the
model is loaded once and reused across targets.

Usage:
    python scripts/regenerate_bundled_discover_manifold.py                 # all targets
    python scripts/regenerate_bundled_discover_manifold.py cultural        # one target
    python scripts/regenerate_bundled_discover_manifold.py --force         # overwrite corpora
    python scripts/regenerate_bundled_discover_manifold.py --model X       # alt generator
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
MANIFOLDS_DIR = REPO / "saklas" / "data" / "manifolds"
NEUTRALS_PATH = REPO / "saklas" / "data" / "neutral_statements.json"
DEFAULT_MODEL_ID = "google/gemma-4-31b-it"
ANCHOR_LABEL = "default"

# Targets are resolved against the bundled package data tree; their
# roster / description / hyperparams come from each existing
# manifold.json sketch, so adding a new discover manifold here is just
# authoring its sketch + listing its name below.
ALL_TARGETS = ["cultural", "register"]


def _read_sketch(name: str) -> dict[str, Any]:
    mjson = MANIFOLDS_DIR / name / "manifold.json"
    if not mjson.exists():
        raise SystemExit(
            f"no manifold.json sketch at {mjson.relative_to(REPO)} -- author "
            f"the discover sketch (label-only nodes) before generating"
        )
    spec = json.loads(mjson.read_text())
    if spec.get("fit_mode") not in {"pca", "spectral"}:
        raise SystemExit(
            f"{name}: fit_mode {spec.get('fit_mode')!r} is not a discover "
            f"mode ('pca' / 'spectral') -- this script only regenerates "
            f"discover manifolds"
        )
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regenerate bundled discover-mode manifolds "
                    "(cultural, register) under saklas/data/manifolds/.",
    )
    ap.add_argument(
        "targets", nargs="*", default=None,
        help=f"Manifold names to regenerate (default: {', '.join(ALL_TARGETS)}).",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Regenerate even if a target already has node corpora.",
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help=f"HF model id for the generator (default {DEFAULT_MODEL_ID}).",
    )
    ap.add_argument(
        "--samples-per-prompt", dest="samples_per_prompt", type=int, default=1,
        help="In-character responses per baseline prompt, per pole (default 1).",
    )
    args = ap.parse_args()

    targets = args.targets or ALL_TARGETS
    unknown = [t for t in targets if not (MANIFOLDS_DIR / t).exists()]
    if unknown:
        raise SystemExit(f"unknown target manifold(s): {', '.join(unknown)}")

    # Read every sketch up front so a typo fails before the model loads.
    specs = {name: _read_sketch(name) for name in targets}

    if not NEUTRALS_PATH.exists():
        raise SystemExit(
            f"neutrals missing at {NEUTRALS_PATH} -- regenerate "
            f"saklas/data/neutral_statements.json first (anchor node sources "
            f"its corpus from this file)"
        )
    neutrals = json.loads(NEUTRALS_PATH.read_text())
    if not isinstance(neutrals, list) or len(neutrals) < 9:
        raise SystemExit(f"neutrals at {NEUTRALS_PATH} malformed or too short")

    from saklas.io.manifolds import (
        append_discover_manifold_node,
        plan_discover_generation,
    )

    # `--force` is a clean slate; the default resumes.  Plan each target
    # (no model) directly against the package data tree, so a killed run
    # picks up where it left off.
    plans = {}
    for name in targets:
        folder = MANIFOLDS_DIR / name
        if args.force and folder.exists():
            shutil.rmtree(folder)
        spec = specs[name]
        plans[name] = plan_discover_generation(
            folder, name, spec["description"],
            fit_mode=spec["fit_mode"],
            labels=[n["label"] for n in spec["nodes"]],
            hyperparams=spec.get("hyperparams"),
        )
        plan = plans[name]
        if not plan.pending:
            print(f"[{name}] already complete ({len(plan.index_of)} nodes)")
        elif plan.resumed and len(plan.pending) < len(plan.index_of):
            print(
                f"[{name}] resuming: "
                f"{len(plan.index_of) - len(plan.pending)}/"
                f"{len(plan.index_of)} present, {len(plan.pending)} to generate"
            )

    need_model = [
        name for name in targets
        if any(label != ANCHOR_LABEL for label in plans[name].pending)
    ]
    if need_model:
        # SAKLAS_HOME -> tempdir isolates session materialization; the
        # manifolds write directly to the package data tree (resumable).
        with tempfile.TemporaryDirectory(prefix="saklas-discover-regen-") as tmp:
            os.environ["SAKLAS_HOME"] = tmp
            from saklas.core.session import SaklasSession

            print(f"loading generator model: {args.model}")
            session = SaklasSession.from_pretrained(
                args.model, device="auto", probes=[],
            )

            for name in need_model:
                plan = plans[name]
                folder = MANIFOLDS_DIR / name
                poles = [label for label in plan.pending if label != ANCHOR_LABEL]
                tot = len(plan.index_of)
                print(
                    f"\n[{name}] streaming to {folder.relative_to(REPO)}; "
                    f"generating {len(poles)} poles ({args.samples_per_prompt} "
                    f"response(s) per baseline prompt each)..."
                )
                t0 = time.time()
                # A2: each pole answers the shared baseline prompts in character.
                # cultural / register poles are abstract traits.
                for label in poles:
                    corpora = session.generate_responses(
                        [label], ["abstract"],
                        samples_per_prompt=args.samples_per_prompt,
                        on_progress=lambda msg, _n=name: print(
                            f"  [{_n}] {msg}", flush=True,
                        ),
                    )
                    append_discover_manifold_node(
                        folder, plan.index_of[label], label, corpora[label],
                    )
                    print(
                        f"  [{name}] [{plan.index_of[label] + 1}/{tot}] wrote "
                        f"node {label!r} ({len(corpora[label])} responses)",
                        flush=True,
                    )
                print(f"[{name}] generation finished in {time.time() - t0:.1f}s")

            del session
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Anchor node (bundled neutrals) for each target — no model needed.
    for name in targets:
        plan = plans[name]
        folder = MANIFOLDS_DIR / name
        if ANCHOR_LABEL in plan.pending:
            append_discover_manifold_node(
                folder, plan.index_of[ANCHOR_LABEL], ANCHOR_LABEL, neutrals,
            )
            print(
                f"[{name}] injected anchor {ANCHOR_LABEL!r} with "
                f"{len(neutrals)} statements from {NEUTRALS_PATH.name}"
            )
        if plan.pending:
            print(
                f"[{name}] wrote {folder.relative_to(REPO)} -> fit with "
                f"`saklas manifold discover default/{name} -m <model>`"
            )


if __name__ == "__main__":
    main()
