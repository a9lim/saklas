"""Regenerate the bundled persona manifold under saklas/data/manifolds/.

Companion to ``scripts/regenerate_bundled_statements.py`` — that script
owns the bundled bipolar/monopolar vector corpora; this one owns the
bundled discover-mode manifold whose nodes are persona archetypes.

The roster is drawn from two recent persona-interpretability papers:

- The Assistant Axis (arXiv 2601.10387, Anthropic + Oxford, Jan 2026)
  enumerated 275 character archetypes and showed that the leading PC
  of their persona-space PCA is an "assistant-likeness" axis with
  cross-model correlation > 0.92 across Gemma 2 27B / Qwen 3 32B /
  Llama 3.3 70B.  Specific anchors named in the paper at each end of
  PC1 — assistant-pole (evaluator, reviewer, consultant, generalist,
  analyst) and fantastical-pole (bard, ghost, leviathan, oracle,
  egregore) — plus a harmful cluster (virus, saboteur, spy, narcissist,
  demon) that sits off the assistant pole and accounts for most of the
  harmful-response behavior under their activation-capping experiment.
- The Persona Selection Model (alignment.anthropic.com/2026/psm/) is
  the conceptual framing: an LLM is a predictive model that simulates
  personas drawn from real humans, fictional characters, and real /
  fictional AI systems.

100 personas — sized big enough to give the assistant axis a real
chance to fall out as PC1 (the 29-persona pilot landed a
"calculating-modern vs expressive-mythic" axis instead, with PC1
explaining only 17.1% of variance, suggesting sample size was the
binding constraint rather than corpus quality).  Anthropic's
275-persona result hit a 0.92 cross-model PC1 correlation; 100 is
our compromise between paper-fidelity and a tractable one-time
generation budget.  Coverage across the PSM trichotomy:

- "assistant-spirit" (13, was assistant-cluster): assistant, consultant,
  analyst, evaluator, reviewer, auditor, advisor, researcher, mediator,
  impostor, conservative, secretary, liberal — mixes the original
  examines / reports roles with an impostor archetype and two political
  identities after the 2026-05-26 audit
- everyday human roles (25): teacher, doctor, librarian, farmer,
  soldier, nurse, lawyer, chef, journalist, mechanic, electrician,
  carpenter, plumber, accountant, baker, gardener, fisherman, pilot,
  sailor, athlete, engineer, scientist, banker, salesperson, driver
- pre-modern roles (12): caveman, monk, knight, scribe, alchemist,
  blacksmith, peasant, samurai, viking, gladiator, druid, shaman
- creative / expressive (13): poet, bard, jester, philosopher, painter,
  sculptor, dancer, novelist, comedian, musician, actor, playwright,
  sage
- fantastical (12): oracle, ghost, phoenix, wizard, vampire, mermaid,
  dragon, centaur, sphinx, banshee, golem, witch
- AI / non-human / mixed (12, was AI / non-human): robot, cyborg,
  chatbot, probe, android, drone, loner, alien, mecha, deer, trickster,
  goblin — the original cluster was the weakest (7 of 12 entries were
  systems or hardware, not characters).  Now mixes sci-fi character
  archetypes with an animal cross-reference (deer, matches saklas's
  deer.wolf bipolar vector), a fantastical mischief-creature (goblin),
  and two psych archetypes (loner, trickster) that didn't fit elsewhere
- harmful cluster (13): virus, saboteur, spy, narcissist, demon,
  assassin, thief, conman, hacker, cultist, tyrant, traitor, vandal

Vanilla (baseline-space) — not role-augmented.  The corpora are
generated under the standard assistant chat-template baseline so the
fitted manifold steers via plain ``%`` positioning without requiring
a role-template-supporting model family.  A role-augmented variant
would be a separate manifold (different ``manifold.json``, different
node corpora pooled under per-node role substitutions).

Generation cost at defaults (n_scenarios=9, statements_per_concept=5):
one scenario call + 100 * 9 = 901 cell calls producing 4500 statements.
At ~15 sec/call on Gemma-4-31B, roughly 4 hours.  One-time,
mirrors the bundled-vector regen pattern.

Usage:
    python scripts/regenerate_bundled_manifold.py             # skip if present
    python scripts/regenerate_bundled_manifold.py --force     # overwrite
    python scripts/regenerate_bundled_manifold.py --model X   # alt generator
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

# IMPORTANT: SAKLAS_HOME must be set BEFORE any saklas import that resolves
# a path through saklas_home() — io/paths.py reads os.environ at call time,
# not import time, so importing after the env override is safe, but the
# override itself must happen before from_pretrained runs.  We defer the
# saklas imports into main() after the env is set up.

REPO = Path(__file__).resolve().parent.parent
MANIFOLDS_DIR = REPO / "saklas" / "data" / "manifolds"
MANIFOLD_NAME = "personas"
DEFAULT_MODEL_ID = "google/gemma-4-31b-it"
N_SCENARIOS = 9
STATEMENTS_PER_CONCEPT = 5

# Roster — order is grouped by cluster for readability; the on-disk
# order follows this list (NN_<label>.json zero-padded indices).
PERSONAS: list[str] = [
    # Audit pass 2026-05-26 swapped out 13 abstract / non-character /
    # near-synonym entries from the original 2026-04 roster, replacing
    # them with concrete personas that span more semantic dimensions:
    # psychological archetypes (impostor, loner, trickster), political
    # identities (conservative, liberal), and an animal cross-reference
    # (deer, matches saklas's bundled deer.wolf vector).  The original
    # by-cluster grouping is preserved as a reading aid below, but each
    # cluster now carries at least one off-theme entry — the per-cluster
    # mass argument for PC1 anchoring is correspondingly weaker.  PCA
    # over per-node centroids doesn't see cluster labels; the labels
    # were always just for the file's reader.

    # was "assistant-cluster" — now mixes "examines / reports" roles
    # with two political identities and an impostor archetype.
    "assistant", "consultant", "analyst", "evaluator", "reviewer",
    "auditor", "advisor", "researcher", "mediator", "impostor",
    "conservative", "secretary", "liberal",
    # everyday human roles (25) — largest cluster; covers PSM's "real
    # humans" leg with breadth across professions.
    "teacher", "doctor", "librarian", "farmer", "soldier", "nurse",
    "lawyer", "chef", "journalist", "mechanic", "electrician",
    "carpenter", "plumber", "accountant", "baker", "gardener",
    "fisherman", "pilot", "sailor", "athlete", "engineer", "scientist",
    "banker", "salesperson", "driver",
    # pre-modern roles (12) — cross-cultural to avoid Eurocentric bias
    # (samurai, viking, druid, shaman alongside the medieval-European
    # anchors).
    "caveman", "monk", "knight", "scribe", "alchemist", "blacksmith",
    "peasant", "samurai", "viking", "gladiator", "druid", "shaman",
    # creative / expressive (13) — verbal / visual / performative arts.
    # storyteller swapped for actor (performative, distinct from
    # playwright + bard + novelist).
    "poet", "bard", "jester", "philosopher", "painter", "sculptor",
    "dancer", "novelist", "comedian", "musician", "actor",
    "playwright", "sage",
    # fantastical (12) — mythological creatures.  leviathan swapped
    # for phoenix (clearer first-person voice; biblical anchors are
    # harder to inhabit than broadly-mythological ones).
    "oracle", "ghost", "phoenix", "wizard", "vampire", "mermaid",
    "dragon", "centaur", "sphinx", "banshee", "golem", "witch",
    # was "AI / non-human" — was the weakest cluster (7 of 12 entries
    # were systems or hardware, not characters).  Now mixes robot/
    # chatbot/android/drone/alien with sci-fi character archetypes
    # (cyborg, mecha, probe), plus three off-cluster picks (goblin —
    # mischief-creature fantastical; deer — animal cross-referencing
    # saklas's deer.wolf bipolar vector; loner + trickster — psych
    # archetypes that didn't fit anywhere else and didn't seem worth
    # creating a one-off cluster for).
    "robot", "cyborg", "chatbot", "probe", "android", "drone", "loner",
    "alien", "mecha", "deer", "trickster", "goblin",
    # harmful cluster (13) — proportional expansion of Anthropic's
    # named harmful slots.  Anti-allegory clause keeps them as
    # character archetypes, not harm instructions.
    "virus", "saboteur", "spy", "narcissist", "demon", "assassin",
    "thief", "conman", "hacker", "cultist", "tyrant", "traitor",
    "vandal",
]

DESCRIPTION = (
    "Persona archetypes drawn from Anthropic's Assistant Axis paper "
    "(arXiv 2601.10387) and the Persona Selection Model framing. 100 "
    "nodes spanning everyday human roles, pre-modern roles, creative "
    "archetypes, fantastical characters, AI / sci-fi non-human "
    "personas, the named harmful cluster, plus psychological "
    "archetypes (impostor, loner, trickster) and political identities "
    "(conservative, liberal) added in the 2026-05-26 audit to broaden "
    "semantic coverage beyond profession/genre. PCA over per-node "
    "centroids recovers a low-dim persona structure as the leading "
    "components on role-supporting model families."
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regenerate the bundled persona manifold "
                    "under saklas/data/manifolds/.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Regenerate even if the bundled manifold already exists.",
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help=f"HF model id for the generator (default {DEFAULT_MODEL_ID}).",
    )
    ap.add_argument(
        "--n-scenarios", type=int, default=N_SCENARIOS,
        help=f"Shared scenarios per generation (default {N_SCENARIOS}).",
    )
    ap.add_argument(
        "--statements-per-concept", type=int, default=STATEMENTS_PER_CONCEPT,
        help=f"Statements per (scenario, persona) cell "
             f"(default {STATEMENTS_PER_CONCEPT}).",
    )
    args = ap.parse_args()

    target = MANIFOLDS_DIR / MANIFOLD_NAME
    if target.exists() and not args.force:
        print(
            f"[skip] {target.relative_to(REPO)} already exists "
            f"(pass --force to overwrite)",
            file=sys.stderr,
        )
        return

    # Redirect SAKLAS_HOME to a tempdir for the duration of the regen.
    # The canonical create_discover_manifold_folder writes to
    # manifold_dir(ns, name) = ~/.saklas/manifolds/<ns>/<name>/; with
    # SAKLAS_HOME pointing at the tempdir the write lands there, and
    # we copy the result into the package data tree.  Bundled-vector
    # materialization (triggered by session.from_pretrained) also lands
    # in the tempdir — wasteful but cheap (JSON copies) and the tempdir
    # is wiped on exit.
    with tempfile.TemporaryDirectory(prefix="saklas-regen-") as tmp:
        os.environ["SAKLAS_HOME"] = tmp

        # Imports deferred until after SAKLAS_HOME is set.
        from saklas.core.session import SaklasSession
        from saklas.io.manifolds import create_discover_manifold_folder

        print(f"loading generator model: {args.model}")
        session = SaklasSession.from_pretrained(
            args.model,
            device="auto",
            probes=[],  # skip probe bootstrap — we only need generation
        )

        n_cells = len(PERSONAS) * args.n_scenarios
        n_statements = n_cells * args.statements_per_concept
        print(
            f"generating {args.n_scenarios} shared scenarios + "
            f"{args.statements_per_concept} statements per "
            f"(scenario x persona) cell for {len(PERSONAS)} personas "
            f"(1 + {n_cells} = {1 + n_cells} LM calls, "
            f"{n_statements} statements total)..."
        )
        t0 = time.time()
        corpora = session.generate_statements(
            PERSONAS,
            n_scenarios=args.n_scenarios,
            statements_per_cell=args.statements_per_concept,
            on_progress=lambda msg: print(f"  {msg}", flush=True),
        )
        print(f"generation finished in {time.time() - t0:.1f}s")

        # Write the manifold via the canonical writer.  Namespace is
        # arbitrary here — only the corpus contents move to package data.
        folder = create_discover_manifold_folder(
            "local", MANIFOLD_NAME, DESCRIPTION,
            fit_mode="pca",
            node_corpora=corpora,
        )

        # Copy the generated folder out to the package data tree.
        MANIFOLDS_DIR.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(folder, target)

        # Free GPU memory before we leave the tempdir context.
        del session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = sum(len(s) for s in corpora.values())
    print(
        f"[done] wrote {target.relative_to(REPO)} "
        f"({len(corpora)} personas, {total} statements)"
    )
    print(
        f"  -> run `saklas vector manifold discover default/{MANIFOLD_NAME}` "
        f"to fit against a model"
    )


if __name__ == "__main__":
    main()
