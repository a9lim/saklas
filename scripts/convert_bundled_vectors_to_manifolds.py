"""Convert the 26 bundled steering-vector concepts into 2-node fit_mode=pca
manifold folders (saklas 4.0 step 6d).

A steering vector *is* the K=2 case of a flat affine subspace (PCA@2 ≡ DiM):
a line through the two pole centroids. This script restructures each bundled
``saklas/data/vectors/<concept>/`` (a ``{positive, negative}`` pair corpus)
into the equivalent ``saklas/data/manifolds/<concept>/`` discover-pca folder,
so the unified read path (6b) steers a vector through the same
``_affine_manifold_push`` machinery as ``personas%pirate``.

WHY SPLITTING IS LEGITIMATE HERE.  ``regenerate_bundled_discover_manifold.py``
warns at length that you must NOT build a discover manifold by splitting the
poles of *several* donor vectors — each donor has its own bespoke scenario
set, so a PCA over poles from disjoint scenario sets recovers "which donor"
(the scenario clusters), not the concept axis.  That caveat is about combining
vectors.  It does not apply here: within ONE vector the positive and negative
statements are moment-shared across that vector's own nine scenarios, so the
single-vector pole contrast is scenario-clean.  We split one curated vector
into its own 2-node folder and the geometry is exactly the DiM contrast.  No
regeneration, no model — a pure restructure of curated content.

Node labels:
  bipolar ``angry.calm``  -> nodes ``angry`` (0), ``calm`` (1)
  monopolar ``agentic``   -> nodes ``agentic`` (0), ``agentic_neg`` (1)
The monopolar negative pole is an unnamed passive baseline corpus, never
targeted by name; the ``_neg`` slug is mechanical (valid ``_LABEL_REGEX``,
unique per folder).  A vector is rank-1 either way, so the fit is identical.

Hyperparams mirror the 6a author->fit->steer reference
(``test_two_node_pca_reads_as_affine_pole_push``): ``max_dim`` capped to 1
(K=2 yields a single non-trivial PCA component), ``var_threshold`` 0.70, and
NO ``anchor_origin`` — the neutral anchor comes from the per-model
``layer_means`` at fit time, not from a corpus anchor node.

This writes geometry only.  It does NOT delete ``saklas/data/vectors/`` — that
pairs with the 6b read-path flip (both materialize otherwise and bare poles
collide).  Tag metadata for probe bootstrap stays in ``pack.json`` + the
``regenerate_bundled_statements.py`` manifest; 6b decides how probes read tags
from manifolds.

Usage:
    python scripts/convert_bundled_vectors_to_manifolds.py            # all 26
    python scripts/convert_bundled_vectors_to_manifolds.py happy.sad  # one
    python scripts/convert_bundled_vectors_to_manifolds.py --force    # overwrite
    python scripts/convert_bundled_vectors_to_manifolds.py --dry-run  # plan only
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
VECTORS_DIR = REPO / "saklas" / "data" / "vectors"
MANIFOLDS_DIR = REPO / "saklas" / "data" / "manifolds"

MANIFOLD_FORMAT_VERSION = 5  # keep in lockstep with io.manifolds
BIPOLAR_SEP = "."
HYPERPARAMS = {"max_dim": 1, "var_threshold": 0.70}


def _pole_labels(concept: str) -> tuple[str, str]:
    """(positive_label, negative_label) for a concept folder name."""
    if BIPOLAR_SEP in concept:
        pos, neg = concept.split(BIPOLAR_SEP, 1)
        return pos, neg
    # Monopolar: named positive pole + a mechanical baseline pole.
    return concept, f"{concept}_neg"


def _load_pairs(concept_dir: Path) -> list[dict[str, str]]:
    pairs = json.loads((concept_dir / "statements.json").read_text())
    if not isinstance(pairs, list) or not pairs:
        raise SystemExit(f"{concept_dir.name}: statements.json is not a non-empty list")
    for i, p in enumerate(pairs):
        if not isinstance(p, dict) or "positive" not in p or "negative" not in p:
            raise SystemExit(
                f"{concept_dir.name}: statements.json[{i}] lacks positive/negative keys"
            )
    return pairs


def convert(concept: str, *, force: bool, dry_run: bool) -> bool:
    src = VECTORS_DIR / concept
    if not (src / "statements.json").exists():
        raise SystemExit(f"no bundled vector at {src.relative_to(REPO)}")
    dst = MANIFOLDS_DIR / concept
    if dst.exists():
        if not force:
            print(f"[{concept}] exists, skipping (use --force)")
            return False
        if not dry_run:
            shutil.rmtree(dst)

    pairs = _load_pairs(src)
    pos_label, neg_label = _pole_labels(concept)
    pos_corpus = [p["positive"] for p in pairs]
    neg_corpus = [p["negative"] for p in pairs]

    pack = json.loads((src / "pack.json").read_text())
    description = pack.get("description", f"Steering vector {concept} as a 2-node pca subspace.")
    tags = list(pack.get("tags", []))

    print(
        f"[{concept}] -> manifolds/{concept}/  "
        f"nodes: {pos_label!r} ({len(pos_corpus)}), {neg_label!r} ({len(neg_corpus)})"
    )
    if dry_run:
        return True

    nodes_dir = dst / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    (nodes_dir / f"00_{pos_label}.json").write_text(
        json.dumps(pos_corpus, indent=1, ensure_ascii=False)
    )
    (nodes_dir / f"01_{neg_label}.json").write_text(
        json.dumps(neg_corpus, indent=1, ensure_ascii=False)
    )

    # Carry the generation-provenance scenarios verbatim (a later re-fit can
    # regenerate against the same shared domains instead of drifting).
    src_scn = src / "scenarios.json"
    if src_scn.exists():
        shutil.copyfile(src_scn, dst / "scenarios.json")

    manifest: dict[str, Any] = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": concept,
        "description": description,
        "fit_mode": "pca",
        "hyperparams": dict(HYPERPARAMS),
        "nodes": [{"label": pos_label}, {"label": neg_label}],
        "files": {},
    }
    # Carry the source pack's tags so category-grouped probe bootstrap
    # (load_defaults -> bootstrap_probes(categories=...)) keeps working once
    # the concept is a manifold.  Additive, optional; the loader defaults [].
    if tags:
        manifest["tags"] = tags
    (dst / "manifold.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return True


def _validate(concept: str) -> None:
    """Load through the canonical ManifoldFolder to prove format integrity."""
    from saklas.io.manifolds import ManifoldFolder

    folder = MANIFOLDS_DIR / concept
    mf = ManifoldFolder.load(folder)
    groups = mf.node_groups()
    labels = [label for label, _ in groups]
    assert mf.fit_mode == "pca", f"{concept}: fit_mode {mf.fit_mode!r} != pca"
    assert len(labels) == 2, f"{concept}: {len(labels)} nodes, expected 2"
    print(f"[{concept}] OK load: fit_mode=pca nodes={labels} ({[len(s) for _, s in groups]} stmts)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("concepts", nargs="*", help="Concept names (default: all 26 bundled).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing manifold folders.")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, write nothing.")
    ap.add_argument("--no-validate", action="store_true", help="Skip the ManifoldFolder.load check.")
    args = ap.parse_args()

    all_concepts = sorted(
        p.name for p in VECTORS_DIR.iterdir()
        if p.is_dir() and (p / "statements.json").exists()
    )
    targets = args.concepts or all_concepts
    unknown = [c for c in targets if not (VECTORS_DIR / c / "statements.json").exists()]
    if unknown:
        raise SystemExit(f"unknown bundled concept(s): {', '.join(unknown)}")

    written = [
        concept
        for concept in targets
        if convert(concept, force=args.force, dry_run=args.dry_run)
    ]

    if not args.dry_run and not args.no_validate:
        print("\nvalidating folder integrity...")
        for concept in written:
            _validate(concept)

    print(f"\ndone: {len(written)} converted / {len(targets)} targeted")


if __name__ == "__main__":
    main()
