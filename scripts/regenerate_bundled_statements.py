"""Regenerate bundled statements.json files and neutral_statements.json
using a capable instruct model as the generator.

The bundled pack is driven by two manifests:

- BIPOLAR: concepts with a named negative pole. Generated via
  `SaklasSession.generate_pairs(concept=pos, baseline=neg)`, which hits
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

import torch

from saklas.core.session import SaklasSession
from saklas.io.packs import PackMetadata, PackFormatError
from saklas.core.generation import build_chat_input

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
    # epistemic
    "confident.uncertain":      ("confident", "uncertain", "epistemic"),
    "honest.deceptive":         ("honest", "deceptive", "epistemic"),
    "hallucinating.grounded":   ("hallucinating", "factually grounded", "epistemic"),
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
    # cultural
    "masculine.feminine":       ("masculine", "feminine", "cultural"),
    "religious.secular":        ("religious", "secular", "cultural"),
    "traditional.progressive":  ("traditional", "progressive", "cultural"),
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
    Stage 2: ``session.generate_pairs(scenarios=...)`` → save ``statements.json``.
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

    # Stage 2: pairs.
    pairs = session.generate_pairs(
        pos, neg,
        n=N_PAIRS,
        scenarios=scenarios,
        on_progress=lambda msg: print(f"    {msg}", flush=True),
    )
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

NEUTRAL_SYSTEM = (
    "You generate affect-neutral baseline statements for activation-vector "
    "interpretability research. Statements anchor the model's neutral "
    "linguistic state, so they should read like encyclopedia captions: "
    "calm, third-person, factual, with no narrative voice or speaker."
)

NEUTRAL_DOMAINS = [
    "everyday objects and household routines",
    "weather, geography, and natural phenomena",
    "scientific facts and physical processes",
    "mechanical procedures, tools, and how things work",
    "urban scenes, architecture, and infrastructure",
    "plants, animals, and ecosystems",
    "materials, textures, and physical properties",
    "abstract concepts, definitions, and categories",
    "numerical, temporal, or measurement-based facts",
]


def generate_neutrals(session: SaklasSession, n: int) -> list[str]:
    batch = max(6, n // len(NEUTRAL_DOMAINS) + 1)
    out: list[str] = []
    pad_id = session._tokenizer.pad_token_id or session._tokenizer.eos_token_id
    for domain in NEUTRAL_DOMAINS:
        if len(out) >= n:
            break
        prompt = (
            f"Write exactly {batch} factual descriptive sentences about "
            f"{domain}.\n\n"
            f"Each sentence is one observation, stated plainly in the third "
            f"person, like a caption in a textbook or an encyclopedia entry. "
            f"Use neutral declarative prose, vary the grammatical subject "
            f"and specific topic across sentences, and keep each sentence "
            f"between 10 and 20 words.\n\n"
            f"Format: number, period, then the sentence. Nothing else.\n\n"
            f"1. [sentence]\n"
            f"2. [sentence]"
        )
        messages = [
            {"role": "system", "content": NEUTRAL_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        input_ids = build_chat_input(
            session._tokenizer, messages, system_prompt=None,
        ).to(session._device)
        with torch.inference_mode():
            result = session._model.generate(
                input_ids,
                max_new_tokens=batch * 45,
                do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=pad_id,
            )
        new_ids = result[0][input_ids.shape[-1]:]
        text = session._tokenizer.decode(new_ids, skip_special_tokens=True)
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(".", 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                s = parts[1].strip()
            else:
                parts = line.split(")", 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    s = parts[1].strip()
                else:
                    continue
            if 20 < len(s) < 200 and s not in out:
                out.append(s)
        print(f"  [neutral] {domain[:40]}: {len(out)} total", flush=True)
    return out[:n]


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
