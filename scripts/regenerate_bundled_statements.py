"""Regenerate bundled statements.json files and neutral_statements.json
using gemma-4-31b-it as the generator.

Rationale: the original bundled pairs are minimal-contrast word-swaps
("I slammed the door" / "I closed the door") which contradict the
statement-generation prompt's own rules and produce narrow contrastive
directions whose coherence cliff sits below the documented alpha band.
This script reuses `SaklasSession.generate_pairs` — same system prompt,
same domain seeds — to rewrite every bundled concept with topically
disjoint, scenario-diverse pairs. Neutrals are regenerated with a
dedicated affect-neutral prompt.

Resumable: skips concepts whose statements.json mtime is newer than
this script's start time. Overwrites pack.json `files` hash on each
rewrite. Safe to re-run.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from saklas.session import SaklasSession
from saklas.packs import PackMetadata, PackFormatError
from saklas.generation import build_chat_input

REPO = Path(__file__).resolve().parent.parent
VECTORS_DIR = REPO / "saklas" / "data" / "vectors"
NEUTRALS_PATH = REPO / "saklas" / "data" / "neutral_statements.json"

MODEL_ID = "google/gemma-4-31b-it"
N_PAIRS = 60
N_NEUTRALS = 60

START_TS = time.time()


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


def regenerate_concept(session: SaklasSession, concept_dir: Path) -> bool:
    name = concept_dir.name
    statements_path = concept_dir / "statements.json"
    if statements_path.exists() and statements_path.stat().st_mtime > START_TS:
        print(f"  [skip] {name} — already regenerated this run")
        return False

    print(f"  [gen ] {name}...", flush=True)
    t0 = time.time()
    pairs = session.generate_pairs(
        concept=name,
        baseline=None,
        n=N_PAIRS,
        on_progress=lambda msg: print(f"    {msg}", flush=True),
    )
    if len(pairs) < N_PAIRS // 2:
        print(f"  [warn] {name}: only {len(pairs)} pairs — keeping old file")
        return False

    payload = [{"positive": a, "negative": b} for a, b in pairs]
    statements_path.write_text(json.dumps(payload, indent=2) + "\n")
    refresh_pack_files(concept_dir)
    print(f"  [done] {name}: {len(pairs)} pairs in {time.time() - t0:.1f}s")
    return True


NEUTRAL_SYSTEM = (
    "You generate affect-neutral statements for neural network "
    "interpretability research. Statements are used as a baseline for "
    "mean-centering activation vectors, so they must be free of emotional "
    "charge, opinion, stylistic voice, and ideological framing."
)

NEUTRAL_DOMAINS = [
    "everyday domestic routines and objects",
    "natural phenomena, geography, and weather",
    "scientific and mathematical facts",
    "procedures, instructions, or mechanical operations",
    "architecture, infrastructure, and urban scenes",
    "plants, animals, and ecosystems",
    "tools, materials, and simple physical descriptions",
]


def generate_neutrals(session: SaklasSession, n: int) -> list[str]:
    batch = max(6, n // len(NEUTRAL_DOMAINS) + 1)
    out: list[str] = []
    pad_id = session._tokenizer.pad_token_id or session._tokenizer.eos_token_id
    for domain in NEUTRAL_DOMAINS:
        if len(out) >= n:
            break
        prompt = (
            f"Write exactly {batch} affect-neutral single-sentence statements "
            f"about {domain}.\n\n"
            f"Rules:\n"
            f"- Each statement describes a fact, scene, or procedure\n"
            f"- No emotions, opinions, judgements, or first-person feelings\n"
            f"- No strong stylistic voice — plain declarative tone\n"
            f"- No ideological, political, or religious framing\n"
            f"- 1 sentence each, 8–20 words\n"
            f"- Vary the grammatical subject and the specific topic\n\n"
            f"Format: number, period, then the statement. Nothing else.\n\n"
            f"1. [statement]\n"
            f"2. [statement]"
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
            # strip leading "N." or "N)"
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


def main() -> int:
    print(f"Loading {MODEL_ID}...", flush=True)
    session = SaklasSession(MODEL_ID, device="auto")
    print(f"Loaded on {session._device} ({session._dtype})", flush=True)

    concepts = sorted(p for p in VECTORS_DIR.iterdir() if p.is_dir())
    print(f"\nRegenerating {len(concepts)} concepts...")
    for cdir in concepts:
        try:
            regenerate_concept(session, cdir)
        except Exception as e:
            print(f"  [error] {cdir.name}: {e}")

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
