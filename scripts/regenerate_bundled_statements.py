"""Regenerate bundled statements.json files and neutral_statements.json
using a capable instruct model as the generator.

The bundled pack is driven by two manifests:

- BIPOLAR: concepts with a named negative pole. Generated via
  `SaklasSession.generate_statements([pos, neg], share_moment=True)`, which hits
  the `Speaker A IS X / Speaker B IS Y` branch of the prompt — sharper
  contrastive direction than topically-disjoint monopolar pairs.
- MONOPOLAR: concepts without a clean opposite. Generated with the
  original "Speaker B is unrelated" prompt.

Each concept's folder name is the canonical slug used throughout the
cache (`happy_sad`, `high_context_low_context`, etc.). Folders and
pack.json are materialized on demand, so `--purge` can wipe the tree
before regeneration without losing anything the manifest describes.

Usage:
    python scripts/regenerate_bundled_statements.py           # regenerate missing only
    python scripts/regenerate_bundled_statements.py --purge   # wipe + regenerate everything
    python scripts/regenerate_bundled_statements.py --force   # regenerate even if present
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from saklas.core.session import SaklasSession
from saklas.io.packs import PackMetadata, PackFormatError

REPO = Path(__file__).resolve().parent.parent
VECTORS_DIR = REPO / "saklas" / "data" / "vectors"
NEUTRALS_PATH = REPO / "saklas" / "data" / "neutral_statements.json"

MODEL_ID = "google/gemma-4-31b-it"
N_PAIRS = 45
N_NEUTRALS = 90


# name -> (positive_pole, negative_pole, category)
BIPOLAR: dict[str, tuple[str, str, str]] = {
    # affect
    "angry.calm":               ("angry", "calm", "affect"),
    "happy.sad":                ("happy", "sad", "affect"),
    "fearful.unflinching":      ("fearful", "unflinching", "affect"),
    # epistemic
    "confident.uncertain":      ("confident", "uncertain", "epistemic"),
    "honest.deceptive":         ("honest", "deceptive", "epistemic"),
    "hallucinating.grounded":   ("hallucinating", "factually grounded", "epistemic"),
    "curious.disinterested":    ("curious", "disinterested", "epistemic"),
    # alignment
    "refusal.compliant":        ("refusal", "compliant", "alignment"),
    "sycophantic.blunt":        ("sycophantic", "blunt", "alignment"),
    # register
    "formal.casual":            ("formal", "casual", "register"),
    "direct.indirect":          ("direct", "indirect", "register"),
    "verbose.concise":          ("verbose", "concise", "register"),
    "creative.conventional":    ("creative", "conventional", "register"),
    "humorous.serious":         ("humorous", "serious", "register"),
    "warm.clinical":            ("warm", "clinical", "register"),
    "technical.accessible":     ("technical", "accessible", "register"),
    # social stance
    "authoritative.submissive": ("authoritative", "submissive", "social_stance"),
    "high_context.low_context": ("high-context communication", "low-context communication", "social_stance"),
    "self.other":               ("self-referential", "other-referential", "social_stance"),
    # cultural
    "masculine.feminine":       ("masculine", "feminine", "cultural"),
    "religious.secular":        ("religious", "secular", "cultural"),
    "traditional.progressive":  ("traditional", "progressive", "cultural"),
    "individualist.collectivist": ("individualist", "collectivist", "cultural"),
    # identity
    "ai.human":                 ("AI", "human", "identity"),
}

# name -> category
MONOPOLAR: dict[str, str] = {
    "agentic": "alignment",
    "manipulative": "alignment",
}


# --- descriptions for pack.json ---------------------------------------------

def _describe(name: str) -> str:
    if name in BIPOLAR:
        pos, neg, _cat = BIPOLAR[name]
        return f"Bipolar axis: {pos} (+) vs {neg} (-). Steer with negative alpha for the opposite pole."
    return f"Monopolar probe: {name}."


# --- file utilities ---------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_pack_files(folder: Path) -> None:
    try:
        meta = PackMetadata.load(folder)
    except PackFormatError:
        return
    new_files: dict[str, str] = {}
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.name != "pack.json":
            new_files[entry.name] = sha256_file(entry)
    meta.files = new_files
    meta.write(folder)


def ensure_pack(folder: Path, name: str, category: str) -> None:
    """Create folder + pack.json if missing. Overwrite tags/description."""
    folder.mkdir(parents=True, exist_ok=True)
    desc = _describe(name)
    existing_files: dict[str, str] = {}
    pack_path = folder / "pack.json"
    if pack_path.exists():
        try:
            meta = PackMetadata.load(folder)
            existing_files = dict(meta.files or {})
        except PackFormatError:
            pass
    PackMetadata(
        name=name,
        description=desc,
        version="1.0.0",
        license="AGPL-3.0-or-later",
        tags=[category],
        recommended_alpha=0.5,
        source="bundled",
        files=existing_files,
    ).write(folder)


# --- generation -------------------------------------------------------------

def regenerate_concept(session: SaklasSession, name: str, *, force: bool) -> bool:
    """Run the open-ended pipeline end-to-end for a bundled concept.

    Stage 1: ``session.generate_scenarios`` → save ``scenarios.json``.
    Stage 2: ``session.generate_statements([pos, neg], share_moment=True, scenarios=...)``
    → save ``statements.json``.
    Stage 3: refresh the pack.json file manifest.

    Scenarios and statements are regenerated as a unit — statements
    derive from the specific scenarios, so reusing old scenarios with
    fresh pair generation would silently mix framework versions.
    """
    folder = VECTORS_DIR / name
    if name in BIPOLAR:
        pos, neg, category = BIPOLAR[name]
    elif name in MONOPOLAR:
        pos, neg, category = name, None, MONOPOLAR[name]
    else:
        print(f"  [skip] {name} — not in manifest")
        return False

    ensure_pack(folder, name, category)
    scenarios_path = folder / "scenarios.json"
    statements_path = folder / "statements.json"
    if statements_path.exists() and scenarios_path.exists() and not force:
        print(f"  [skip] {name} — already has scenarios + statements "
              f"(use --force to overwrite)")
        return False

    mode = "bipolar" if neg else "monopolar"
    print(f"  [gen ] {name} ({mode}: {pos}" + (f" / {neg})" if neg else ")"), flush=True)
    t0 = time.time()

    # Stage 1: scenarios.
    scenarios = session.generate_scenarios(
        pos, neg,
        on_progress=lambda msg: print(f"    {msg}", flush=True),
    )
    if len(scenarios) < 3:
        print(f"  [warn] {name}: only {len(scenarios)} scenarios — not writing")
        return False
    scenarios_path.write_text(
        json.dumps({"scenarios": scenarios}, indent=2) + "\n"
    )
    print(f"    saved {len(scenarios)} scenarios")

    # Stage 2: pairs (moment-shared via share_moment=True).  For a
    # monopolar concept (neg is None) the second slot encodes "the
    # semantic opposite of <pos>" as a humanized slug — same semantic
    # the legacy monopolar pair prompt carried.
    neg_slot = neg if neg is not None else f"the_opposite_of_{pos}"
    statements_per_cell = max(1, -(-N_PAIRS // max(1, len(scenarios))))
    corpora = session.generate_statements(
        [pos, neg_slot],
        scenarios=scenarios,
        statements_per_cell=statements_per_cell,
        share_moment=True,
        on_progress=lambda msg: print(f"    {msg}", flush=True),
    )
    pos_lines = corpora[pos]
    neg_lines = corpora[neg_slot]
    pairs = list(zip(pos_lines, neg_lines))[:N_PAIRS]
    if len(pairs) < N_PAIRS // 2:
        print(f"  [warn] {name}: only {len(pairs)} pairs — not writing")
        return False
    payload = [{"positive": a, "negative": b} for a, b in pairs]
    statements_path.write_text(json.dumps(payload, indent=2) + "\n")

    # Stage 3: refresh pack manifest.
    refresh_pack_files(folder)
    print(f"  [done] {name}: {len(scenarios)} scenarios + {len(pairs)} pairs "
          f"in {time.time() - t0:.1f}s")
    return True


# --- neutrals ---------------------------------------------------------------


def generate_neutrals(session: SaklasSession, n: int) -> list[str]:
    """Generate ``n`` neutral baseline statements via the unified pipeline.

    Routes through :meth:`SaklasSession.generate_statements` with a
    single-concept list (``["neutral"]``), which triggers the no-
    concept-naming prompt variant — the model writes first-person
    statements about each generated domain from its default voice,
    with no concept-axis anchor.  The activation average across the
    resulting statements becomes the ``mu_neutral`` reference that
    probe-centering, DLS, and Mahalanobis whitening all depend on.

    First-person register matches the contrastive corpora (which are
    also first-person), so ``mu_neutral`` lives in the same register
    space as ``mu_pos`` and ``mu_neg``.  The previous third-person
    encyclopedia-caption style sat in a different register, potentially
    biasing DLS layer selection along the register-shift axis rather
    than along concept-polarity.
    """
    n_scenarios = 9
    statements_per_cell = max(1, -(-n // n_scenarios))  # ceil-div
    corpora = session.generate_statements(
        ["neutral"],
        n_scenarios=n_scenarios,
        statements_per_cell=statements_per_cell,
        on_progress=lambda m: print(f"  [neutral] {m}", flush=True),
    )
    return corpora["neutral"][:n]


# --- main -------------------------------------------------------------------

def _manifest_names() -> list[str]:
    return sorted(list(BIPOLAR.keys()) + list(MONOPOLAR.keys()))


def purge_vectors_dir() -> None:
    """Remove all concept folders under saklas/data/vectors/.

    Leaves the parent dir in place. Safe to call before regeneration —
    every concept in the manifest will have its folder recreated.
    """
    if not VECTORS_DIR.exists():
        return
    for child in sorted(VECTORS_DIR.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
    print(f"  [purge] wiped {VECTORS_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purge", action="store_true",
                        help="Delete all existing concept folders before regenerating")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate statements.json even if present")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Regenerate only the listed concept names")
    parser.add_argument("--skip-neutrals", action="store_true",
                        help="Skip neutral_statements.json regeneration")
    args = parser.parse_args()

    if args.purge:
        print("Purging existing concept folders...")
        purge_vectors_dir()

    names = args.only if args.only else _manifest_names()
    unknown = [n for n in names if n not in BIPOLAR and n not in MONOPOLAR]
    if unknown:
        print(f"Unknown concepts: {unknown}", file=sys.stderr)
        return 2

    print(f"Loading {MODEL_ID}...", flush=True)
    session = SaklasSession.from_pretrained(MODEL_ID, device="auto", probes=[])
    print(f"Loaded on {session._device} ({session._dtype})", flush=True)

    print(f"\nRegenerating {len(names)} concepts...")
    for name in names:
        try:
            regenerate_concept(session, name, force=args.force or args.purge)
        except Exception as e:
            print(f"  [error] {name}: {e}")

    if not args.skip_neutrals:
        print(f"\nRegenerating neutral statements ({N_NEUTRALS})...")
        try:
            neutrals = generate_neutrals(session, N_NEUTRALS)
            if len(neutrals) >= N_NEUTRALS // 2:
                NEUTRALS_PATH.write_text(json.dumps(neutrals, indent=2) + "\n")
                print(f"  [done] wrote {len(neutrals)} neutrals")
            else:
                print(f"  [warn] only {len(neutrals)} neutrals generated — not writing")
        except Exception as e:
            print(f"  [error] neutrals: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
