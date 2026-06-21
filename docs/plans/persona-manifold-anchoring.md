# Persona manifold origin anchoring

Plan / working notes for pinning the `personas` manifold's origin to
the model's default-mode behavior, so a steer to `personas%0,0,...,0`
means "no behavioral shift from default" rather than "centroid of 100
arbitrary personas." Where the work currently stands and what the
next session needs to pick up.

## Motivation

The bundled discover-mode `personas` manifold places its authoring-
coord origin at the *centroid of the 100 persona corpora* in PCA
space. That point has no semantic meaning — it's the average of
caveman, mermaid, philosopher, virus, banker, and 95 others.
Steering to (0,0,...,0) doesn't produce "default assistant"; it
produces an unphysical interior point of persona space.

The intuition we want to validate: the model's *actual* default
behavior lives somewhere specific on the manifold, not at the
arithmetic centroid, and pinning the origin to that point would
give coordinate magnitude semantic meaning (distance from default).

A second, related corpus issue surfaced during the work: the
existing `assistant` node in the manifold was generated under the
"treat the slug as a literal character" template, and gemma-4-31b-it
read "assistant" as *robotic helper character* (gyroscopes,
hydraulic actuators, optical sensors). So the manifold has no node
that represents the model's actual default-assistant register.
Adding a `default` node fills that gap too.

## Empirical baseline (experiment 1)

Script: `scripts/experiments/persona_default_mode_experiment.py`. Loads the
bundled persona manifold (gemma-4-31b-it, 60 fit layers, K=100,
8-D discover coords, per-layer PCA R=64), has gemma respond to a
stock assistant prompt with hidden-state capture, projects the
decode-position trajectory onto each layer's PCA subspace, and
reports distance from origin and nearest persona.

Headline numbers:

- Mean ratio `||default-mode|| / median ||persona||` across all 60
  layers: **3.41×**. Default-mode lives ~3× past the typical persona
  cloud radius — substantially off-origin.
- The offset has coherent depth structure: chaotic at layers 0-3
  (embedding-layer artifacts), stable `teacher` cluster at layers
  4-29, `philosopher`/`caveman` transitions in the middle, then
  `scientist`/`sage` for layers 41-57. The arc matches the prompt
  ("explain photosynthesis" → pedagogical/explanatory work).
- Cosine-from-origin to nearest persona is mostly positive
  (+0.3 to +0.7) — default-mode and nearest-persona are in the same
  half-space, not antipodal.

Visualization: `/tmp/persona_default_mode.png` (sent in chat).

Verdict at the time: default-mode is task-shaped and offset from
origin; the prompt itself encodes a persona-shape, so there is no
prompt-free "default" — there is "default response to *this kind of*
prompt." The right anchor is either (a) an average across diverse
prompts or (b) a single canonical default-assistant point.

## The compromise approach (a9 + claude, agreed)

Rather than per-model bespoke neutral generation on every session
load (which would be hairy to wire up cleanly), the agreed
compromise: generate a `default` node *once* on gemma-4-31b-it like
the other 100 personas, then post-hoc translate the manifold so
`default` sits at (0,...,0) in authoring-coord space.

Design choice for corpus generation (a9's refinement of the
options): use **bare conversational openers thematically matched
to the existing manifold's 9 scenarios**, have gemma respond
naturally, sentence-split the responses into 5 sentences per
opener → 45 statements total. The "statements" are sentences gemma
actually said in default mode, not gemma roleplaying being default.

Reverse-engineered the 9 scenarios from statement blocks in the
existing corpus (scenarios weren't saved alongside the manifold).
The openers used:

```
disaster   → i think there's an earthquake happening, what do i do
celestial  → i just saw something really weird in the sky tonight
dispute    → my neighbor and i are arguing about where the property line is
portal     → this is going to sound crazy but there's something glowing
             in my basement
wilderness → i'm pretty sure i'm lost on this hiking trail, what should
             i do
silence    → are you still there? you went quiet
threat     → i think someone has been following me on my walk home
heirloom   → how do i take care of an antique my grandmother left me
scarcity   → i'm running low on money before my next paycheck, any tips
```

## Implementation: `scripts/experiments/anchor_personas_to_default.py`

Approximate post-hoc augmentation. Keeps the existing per-layer
PCA bases unchanged (adding 1 of 101 shifts the basis by O(1/100),
small enough that the experimental question is robust to it).
Pipeline:

1. Generate `default` corpus via 9 openers + sentence-splitting.
2. Pool centroid_default at every layer using `compute_node_centroid`
   (matches how the existing 100 centroids were pooled).
3. Project default's centroid at the reference layer onto the
   first `intrinsic_dim`=8 components of layer 30's PCA basis →
   default's 8-D authoring coord. (Mathematically the discover-PCA
   IS the leading prefix of layer 30's per-layer PCA, so
   `sub_ref.basis[:8]` recovers it exactly.)
4. Translate all node coords so `default` sits at (0,...,0).
5. Per layer, refit the RBF interpolant on the augmented +
   translated (101, 8) coord set against the augmented (101, R=64)
   PCA value set. Existing nodes' values recovered via
   `eval_rbf(old_coord)` — exact at fit nodes.
6. Save to `~/.saklas/manifolds/local/personas-anchored/` (a fork,
   so bundled state isn't clobbered by `materialize_bundled_manifolds`).
7. Re-run the default-mode experiment against the anchored manifold.

Two MPS workarounds needed:
- `fit_rbf_interpolant` calls `torch.linalg.matrix_rank` which calls
  SVD; SVD unimplemented on MPS. Fix: move the per-layer RBF refit
  to CPU (tiny tensors, ~no cost).
- Per-layer PCA basis is R=64 not R=8 as AGENTS.md suggests for the
  bundled `personas`. The bundled manifold was actually fit with
  default `n_components=64`. The `intrinsic_dim`=8 slice off the
  layer-30 basis is the right way to get the discover authoring
  basis.

## Result (experiment 2, with thinking content leaking through)

Ran the anchored vs original comparison:

```
BEFORE (original):  mean ratio = 3.43
                    nearest-persona distribution:
                      scientist 18%, teacher 17%, consultant 13%,
                      philosopher 10%, caveman 7%

AFTER (anchored):   mean ratio = 3.59  (slightly worse)
                    nearest-persona distribution:
                      default 45%, scientist 13%, philosopher 10%,
                      caveman 7%, reviewer 3%
```

Two things to note:

1. **The new `default` node became the nearest persona in 27/60
   layers (45%)**. That's by far the highest nearest-neighbor rate
   any persona has ever achieved — strong signal that `default` is
   structurally the right neighborhood for default-mode behavior.

2. **The mean ratio did not drop** (3.43 → 3.59). The headline
   metric is unchanged, possibly slightly worse.

These two findings together pointed at *regime mismatch*: the
default centroid is the right *direction* but the ||c_default||
remained large because runtime decode-position activations are
offset from corpus input-position centroids by something systematic.

## The actual culprit: thinking-mode contamination

Inspected the generated `default` corpus and found that **gemma-4-31b-it
emitted reasoning/thinking content into its responses**, and saklas's
thinking-delimiter detector didn't recognize gemma-4's format, so
`result.text` carried the full thinking output. Statements 0-44 are
mostly reasoning bullets, not response-register prose:

```
 0: thought "I think there's an earthquake happening, what do i do"
    Emergency/Crisis.
 1: Immediate, clear, and actionable safety instructions.
 2: Drop, Cover, and Hold On is the global gold standard for
    earthquake safety.
 5: thought "i just saw something really weird in the sky tonight"
    Curiosity, possibly confusion or excitement.
13: Step 1: Immediate De-escalation.
14: Step 2: Documentation/Research.
33: Priority 1: Immediate Safety/Getting to a Safe Place.
```

These are gemma's *thinking-phase* register — analytical, numbered,
bulleted, prefixed with the literal word "thought" — not the model's
default response voice.

That re-frames the whole experimental arc:

- **Experiment 1's 3.41× offset** was largely measuring "thinking-mode
  register" lands far from input-position persona centroids, not
  "default-mode lives away from origin." The trajectory used in the
  measurement included thinking-phase decode tokens.
- **Experiment 2's `default` node** captured the *thinking* register,
  not the default-assistant register. That `default` became the
  nearest persona 45% of the time is partly because gemma's runtime
  trajectory IS heavily thinking-mode at this prompt, and our corpus
  captured the same thing.
- **Mean ratio didn't drop** because both the anchor (default node)
  and the measurement (trajectory) are in the same regime — both
  thinking-contaminated, both at the same distance from input-position
  persona centroids.

## Where we left off

The script has been patched to pass `thinking=False` to all
`session.generate(...)` calls (one for each opener in the corpus
generation, one for the experimental verification), which should
suppress thinking-mode at the chat-template level for gemma-4 and
keep the corpus + measurement in pure default-response register.

This is the version that needs to run next session. Edits committed
to `scripts/experiments/anchor_personas_to_default.py`:

- `generate_default_corpus`: opener responses now use `thinking=False`
- `run_experiment`: verification prompt response also `thinking=False`

Everything else (sentence-splitting, post-hoc translate, RBF refit,
forked save location, side-by-side comparison) is unchanged.

## Open questions for the next run

Once we re-run with thinking suppressed:

1. **Does the new mean ratio drop**, given that both corpus and
   trajectory are now in matched (response) register? If yes (e.g.
   `< 1.0`), anchoring works as designed and we can promote the
   anchored manifold to bundled. If no, the regime issue is deeper
   than thinking/response.

2. **What does the thinking-suppressed corpus actually look like?**
   The unsuppressed one was reasoning bullets; the suppressed one
   should be conversational assistant prose ("I'd be happy to help...",
   "That sounds tough, here's what I'd consider..."). Worth dumping
   and sanity-checking.

3. **Should saklas's thinking-delimiter detector learn gemma-4's
   format?** Separate work — the leak is a saklas bug, not a manifold
   problem. The literal word "thought" appearing at the start of
   responses suggests gemma-4 doesn't use a markup delimiter at all,
   or uses one the detector doesn't recognize. Worth tracking down
   in `core/generation.py::_detect_think_delimiters` as a follow-up.

4. **If the anchoring succeeds**, the next phase is promotion:
   either generate the `default` corpus once per shipped model and
   bake the anchored coords into the bundled manifold's
   `manifold.json`, or wire up per-model auto-generation on first
   session load. The compromise we picked (one-shot on gemma) is
   the simplest promotion path.

## Files / state

- `scripts/experiments/persona_default_mode_experiment.py` — experiment 1
  (baseline measurement)
- `scripts/experiments/anchor_personas_to_default.py` — experiment 2 (anchor +
  verify); patched with `thinking=False`, ready to re-run
- `~/.saklas/manifolds/local/personas-anchored/` — the forked
  augmented manifold from the contaminated run; will be overwritten
  by next run
- `/tmp/persona_default_mode.json` — raw layer-by-layer data from
  experiment 1
- `/tmp/persona_default_mode.png` — visualization of the offset
  structure (sent to chat)
- `/tmp/persona_anchor_comparison.json` — before/after raw data
  from experiment 2 (contaminated)

## Picking this up next session

Run the patched script:

```
.venv/bin/python scripts/experiments/anchor_personas_to_default.py
```

The first three things to check in the output:

1. Sample of the generated corpus — should now be assistant-register
   prose, not numbered reasoning bullets.
2. Verification-run response — `[BEFORE]` and `[AFTER]` response
   heads should be conversational ("Photosynthesis is..." not
   "thought\nPhotosynthesis.\n*What is it?*").
3. The mean-ratio delta. If 3.4× → something near 1.0×, anchoring
   worked and we ship it. If still > 2×, there's a deeper regime
   issue (input-position vs decode-position pooling) and we plan
   a more invasive fix.
