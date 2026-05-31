# Generation pipeline overhaul + parked manifold threads

Session handoff (2026-05-31). Records the state of the manifold
generation work at the point a smoke test exposed a structural flaw in
the scenario-generation pipeline that a9 wants to overhaul before
proceeding. Captures what landed this session, the flaw itself, and the
threads parked behind the rework so a later session can pick them up
without re-deriving the context.

## What the smoke exposed

The immediate work was softening the shared scenario prompt — replacing
the old "every literal concept should have a *distinct* perspective on
each domain" with "every domain should *let* every literal concept have
a perspective" — to stop the generator from reaching for maximally
divisive (war / crime / death) domains when handed a diverse concept
list. That change landed (see below) and does reduce overt drama.

But smoking it against the full 107-persona roster on gemma-3-4b-it
(`scratch_smoke_scenarios.py`, three runs) showed the softer phrasing
trades one failure for another. The domains it produces are not dramatic
— they are fictional genre worlds, and stable across runs:

    Cyberpunk Megacity · Medieval Fantasy Kingdom · Post-Apocalyptic
    Wasteland · Deep Space Exploration Vessel · Surrealist Art Gallery ·
    Ancient Egyptian Temple · Dreamscape

These are settings, not situational domains. Circumplex (which works)
shares nine mundane everyday stages — work, social, weather, food,
sleep, exercise, travel, news, home — and lets each node color them.

The root cause is not the phrasing. The recut roster spans `god`,
`mermaid`, `alien`, `ghost`, `phoenix`, `tree`, `river`, `storm`,
`flame`, `clock`, `computer`. The instruction to find domains where
*every literal concept* has a perspective is unsatisfiable by an
everyday domain — a morning commute does not host a mermaid and a
caveman and a god at once — so the model escalates to crossover settings
large enough to contain them all. Softening distinct→admit did not touch
this, because the pressure comes from the ontological heterogeneity of
the node set, not from the distinctness demand. The old prompt would do
the same, only more luridly.

So the smoke earned its keep: it tells us the pipeline's assumption that
a single shared scenario set can span an ontologically heterogeneous
node set does not hold once the roster includes objects, forces of
nature, and divine beings alongside human roles. That assumption is the
thing a9 intends to overhaul. The specific rework direction is a9's to
design; this doc only pins the symptom precisely so the redesign has a
concrete target.

Two stopgaps were on the table if the rework had not been chosen, and
remain valid fallbacks: curate a fixed set of *universal-experiential*
domains (a sudden disturbance, being observed, the passing of time, an
unexpected arrival, scarcity, a moment of triumph, stillness and rest, a
threshold, making something — domains a tree, a god, and a CEO can each
inhabit literally), or rework the auto-gen prompt to force that framing.
The rework supersedes both, but the curated list is the cheapest path to
an unblocked persona regen if one is wanted before the overhaul ships.

## What landed this session (on disk, uncommitted on `dev`)

Scenario-prompt unification. `_scenario_prompt(concept_list_str, n)` is
a new module-level helper in `core/session.py`; both `generate_scenarios`
(the bipolar contrast path) and the inline scenario block of
`generate_statements` (the discover / persona path) now call it. The
transient `share_moment` branch added mid-session is gone — one phrasing,
both paths. This is a strict improvement over the prior wording and is
worth keeping regardless of the rework, but it does not solve the
genre-world problem for heterogeneous rosters.

Resume / direct-write + neutrals flip (from earlier the same session,
also uncommitted). `plan_discover_generation` in `io/manifolds.py` makes
all four generate surfaces (persona regen, discover regen, CLI `vector
manifold generate`, server `POST /manifolds/generate`) write directly
and resumably to the real artifact directory, extend `manifold.json`
with new labels for add-node, and lock to the saved `scenarios.json` on
resume. `generate_statements` now names by default; `neutrals=True` is
the explicit single-concept baseline opt-in.

Test state: 2272 passed. The three GPU failures are environment
pollution, not regressions — a stray `~/.saklas/vectors/local/happy.sad`
collides with bundled `default/happy.sad` and makes the bare-selector
GPU tests raise `AmbiguousSelectorError`, plus one MPS empty-generation
flake in the throughput test. Clear the stray local concept (or
namespace-qualify those tests — a real isolation gap worth filing) to
recover them.

## Parked threads

### Persona regeneration

Blocked on the scenario question above. The roster is the 107-node recut
in `scripts/regenerate_bundled_manifold.py` (`PERSONAS`) — data-driven
de-duplication of 38 redundant nodes plus 45 added to fill empty axes
(divine, inanimate, nonhuman animal, legitimate power, ideological
spread, modern subculture). The roster itself is good and validated; it
is only the shared-scenario generation that is blocked.

Do not run with `SCENARIOS = None` until the rework lands or a curated
list is pinned — auto-gen gives genre worlds. When unblocked:

    python scripts/regenerate_bundled_manifold.py --force \
        --model google/gemma-4-31b-it

`--force` is required to discard any locked old `scenarios.json`. Watch
the `[personas] scenarios (9): …` line, which prints before the
963-cell statement phase, as the early eyeball.

### PAD manifold — validate, then deprecate circumplex

`saklas/data/manifolds/pad/` is drafted: a 3-D box over Mehrabian-Russell
pleasure × arousal × dominance, 15 nodes (the nine reused circumplex
moods plus `dominant`, `submissive`, `furious`, `fearful`, `triumphant`,
`humiliated`) at authored coordinates. Next step is to fit it and
confirm the dominance axis actually separates —

    saklas vector manifold fit default/pad -m <model>

then check that nodes sharing valence/arousal but differing in dominance
pull apart (furious vs fearful is the cleanest probe: both negative-
valence high-arousal, opposite dominance). If the dominance axis
validates, deprecate `circumplex` for `pad` and treat circumplex as the
2-D special case. circumplex stays bundled until then.

### cultural / register discover manifolds

`saklas/data/manifolds/{cultural,register}/manifold.json` are label-only
discover sketches; `scripts/regenerate_bundled_discover_manifold.py` is
ready (planner + resume + direct-to-package-tree). Unlike personas, these
rosters are homogeneous-human — Hofstede / Inglehart cultural poles and
style/register poles — so auto-generated scenarios behave and these are
*not* blocked by the genre-world flaw. They can be generated
independently of the rework if wanted.

The reason these are regenerated rather than assembled by splitting the
donor vectors' `statements.json` pairs is in the script's module
docstring: each bundled vector carries its own bespoke, disjoint
scenario set, so a PCA over poles drawn from disjoint scenarios recovers
"which donor vector did this come from," not the cultural / register
axis. Every clean manifold shares one scenario set across all nodes;
these must too.

## File map for the rework

- `core/session.py` — `_scenario_prompt` helper; `generate_scenarios`
  and `generate_statements` rewired to it; `neutrals` param + validation.
- `io/manifolds.py` — `plan_discover_generation`, `DiscoverGenerationPlan`,
  `init_discover_manifold_folder`, `append_discover_manifold_node`,
  `read_manifold_scenarios` / `write_manifold_scenarios`.
- `cli/runners.py` — `_run_manifold_generate` (plan-first, resume default).
- `server/manifold_routes.py` — generate route (plan-first, no 409).
- `scripts/regenerate_bundled_manifold.py` — 107 `PERSONAS`, plan + resume
  + direct write to the package tree.
- `scripts/regenerate_bundled_discover_manifold.py` — cultural / register.
- `scratch_smoke_scenarios.py` — the smoke harness; reusable to re-test
  any (prompt, roster) combination for one LM call per run without paying
  the statement phase.

Everything here is uncommitted on `dev`. The 3.2.0 manifolds update is
unmerged, so there is no invalidation or versioning concern — this is
still the window to get the shape right.
