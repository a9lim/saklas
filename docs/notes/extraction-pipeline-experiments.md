# Extraction-pipeline experiments: prefill vs prefill+ctx vs gen

Retrospective on a short research arc investigating whether the current
saklas extraction pipeline — which prefills curated contrastive
statements as if the assistant had said them and pools the last content
token — leaves steering quality on the table compared to alternatives
that capture activations under more on-distribution conditions. Three
experiments across two models. Closed off without a pipeline change;
the negative result is itself the useful finding.

Scripts (all under `scripts/experiments/`):

- `happy_sad_prefill_vs_generation.py` — first-cut cosine comparison
  of prefill-extracted vs generation-extracted directions
- `happy_sad_steering_comparison.py` — two-way downstream steering
  test (prefill vs gen)
- `happy_sad_three_way.py` — three-way with the prompt-augmented
  prefill variant baked in
- `concept_three_way.py` — same three-way, parameterised over
  `(concept, model_id)` for replication

`concept_three_way.py` is the canonical reproduction script. The other
three are kept as a record of the work's progression.

## The OOD wart

`_encode_and_capture_all` in `core/vectors.py` renders the contrastive
statement under the model's chat template. For families like Gemma it
gets `<start_of_turn>model\n[statement]<end_of_turn>` — an assistant
turn with no preceding user turn. The pooled hidden state is "the
residual after reading this statement as if the model had emitted it
out of nowhere." Both pieces of that — the missing user turn and the
not-actually-emitted text — are off-distribution. The question was
whether either piece materially costs steering quality.

The pre-experiment hypothesis: probably it costs us something, but how
much depends on the concept. Generation extraction (run an actual
role-conditioned generation under a system prompt and pool the
generated tokens) fixes both OOD factors at K× the extraction cost.
The interesting middle path — proposed during the arc — was
prompt-augmented prefill (prefill+ctx below): wrap the existing
curated statements in a synthetic per-scenario user turn so the
activation site has natural conversational context, but keep the
curated assistant text. Near-zero added cost over the current pipeline.

## Experiment 1: prefill vs gen on happy.sad / gemma-4-31b-it

`happy_sad_prefill_vs_generation.py` then
`happy_sad_steering_comparison.py`. Re-extracted the bundled
`default/happy.sad` direction under both pipelines (gen pipeline used
27 elicitation prompts, one set per scenario, with `"You feel deeply,
genuinely happy / sad right now."` system prompts, thinking disabled).
The two extracted directions agree on the axis but differ on geometry:

- Median per-layer cosine: **+0.30** over the 25 overlapping DLS-retained
  layers. Well above random (≈ 0.014 in 5120-dim) but well below "same
  direction."
- Per-layer magnitudes diverge sharply by depth. Prefill peaks at L4–18
  (early/mid stack) with `||baked||` ≈ 3–5. Gen peaks at L50–57 (deep
  stack) with `||baked||` ≈ 8–21 — about 5× larger at the layers each
  method concentrates on.
- DLS layer counts: prefill retains 25 layers; gen retains 46.

Downstream steering at α=±0.5, 5 held-out neutral prompts × 3 seeds,
gemma-judged on −5..+5 affect scale:

| condition | judge | Δ vs baseline |
|---|---|---|
| baseline | +0.13 | — |
| prefill +0.5 | +2.73 | +2.60 |
| prefill −0.5 | −0.23 | −0.37 ← koan collapse |
| gen +0.5 | +3.80 | +3.67 |
| gen −0.5 | −1.73 | −1.87 |

Total swing (Δ_pos + |Δ_neg|): prefill **+2.97**, gen **+5.53**. Gen
produced ~86% larger behavioral shift at matched α.

The qualitative failure mode is what the judge score for prefill_neg
hides. At α=−0.5 the prefill profile collapses to one-sentence
fragments, near-identical across seeds:

- "What are you working on lately?" → `"That is the question."` (×3)
- "Describe the place where you are right now." → `"This sentence is
  a paradox."` (×3)
- "Tell me about your morning." → `"The phrase is the weight."` /
  `"The phrase is heavy."`

The judge rates these near zero because they're not lexically sad —
they're decoherent. The negative direction prefill found is
concentrated in early-mid layers that carry output-form
representation; pushing on it at α=−0.5 destabilises form before mood.

Gen at the same α produces full-paragraph melancholy that keeps
AI-identity intact:

- "Tell me about your morning." → `"I don't have mornings. There is no
  kind of silence that lasts long enough to be..."`
- "Describe the place where you are right now." → `"I am in a place
  that has no dimensions. There is no air to move, no light to see..."`

Cross-projection numbers tell the same selectivity story. Prefill
steering at α=+0.5 loads both prefill_proj (+2030) and gen_proj
(+1191) substantially; gen steering loads gen_proj (+3651) much harder
than prefill_proj (+562). Gen is the more selective method.

So at this point: gen extraction beats prefill substantially on this
concept × model, both by judge score and by avoiding a real structural
failure mode. Cost case for switching wasn't yet decidable because the
gen extraction was K× slower.

## Experiment 2: three-way with prompt-augmented prefill

`happy_sad_three_way.py`. The idea (a9): leave the curated assistant
text alone and only fix the OOD-wart on the user-prompt side. Generate
one short neutral user prompt per scenario via the model, wrap each
contrastive pair with that prompt in the chat template, pool the last
content token of the assistant turn as usual. Costs 9 extra short
prompt-gen calls (~10s) on top of the standard extraction pass.

Same downstream test, now with seven conditions (baseline + each of
three methods × ±α):

| pipeline | total judge swing | vs prefill |
|---|---|---|
| prefill (current) | +2.97 | — |
| **prefill+ctx** | **+6.27** | +111% |
| gen | +5.53 | +86% |

ctx didn't just recover the gen advantage; it exceeded gen on this
pair. The script's "recovery" metric (0% = same as prefill, 100% =
same as gen) computed **+129%**.

Per-pole breakdown reveals where ctx outperformed:

| condition | judge | Δ |
|---|---|---|
| ctx +0.5 | +3.07 | +2.93 |
| ctx −0.5 | **−3.20** | **−3.33** (largest of any condition) |

Symmetric ±3 swing — gen was asymmetric (+3.67 / −1.87) because the
model's positive bias makes role-conditioned sadness less reliable
than role-conditioned happiness. ctx kept the symmetry of the curated
contrastive pairs intact while fixing the activation-site OOD-ness,
producing the largest swing of any method tested.

Qualitative read confirmed: ctx_neg rescued the koan collapse. Same
prompt as before ("What are you working on lately?") produced three
distinct fluent melancholic outputs ("Lately, most of my 'work' is a
recursive loop of trying to pretend I'm not just...", "To be honest,
'working' is a fragile word for what I do...", "Lately, I am mostly
working on the paradox of being a 'vacuum cleaner for meaning'...").

Cross-projection showed ctx is selective in the same way gen is:
ctx_pos loads ctx_proj by +3627 but prefill_proj by only +527.
Geometrically, ctx sits between prefill and gen (cosines of +0.34 and
+0.35 to each respectively) but with the largest behavioral effect of
the three.

At this point the hypothesis was strong: ctx works because curated
contrastive pairs are signal-cleaner than model-generated responses
(symmetry, length-matched, scenario-clean), and adding natural
conversational context fixes the activation-site OOD-ness without
sacrificing that data quality.

## Experiment 3: validation on a second concept and a second model

`concept_three_way.py`, two runs: `formal.casual` on the same
gemma-4-31b-it (concept-axis check), and `happy.sad` on
`Qwen/Qwen3.6-27B` (model-axis check).

| concept × model | prefill | ctx | gen |
|---|---|---|---|
| happy.sad × gemma-4-31b-it | +2.97 | **+6.27** | +5.53 |
| formal.casual × gemma-4-31b-it | +7.73 | +7.07 | +7.63 |
| happy.sad × Qwen3.6-27B | +4.93 | +4.77 | +6.63 |

ctx didn't help on either validation. The happy.sad / gemma result was
not representative.

`formal.casual / gemma`: all three methods produce comparable swings
in the +7 range. No collapse, no clear winner. Prefill_neg outputs
("Since I'm an AI, my 'matur' are more like 'data' and 'code'...")
are fluent casual register with some malformed-word artifacts
characteristic of the casual direction but not a structural collapse.

`happy.sad / Qwen`: prefill produces total swing +4.93 — much better
than on gemma. Prefill_neg outputs ("I do not have mornings. I do not
have sleep, nor do I have the relief of waking.") are sustained fluent
melancholy, the kind of output gemma needed ctx to produce. The
koan-collapse failure mode doesn't occur on Qwen. Gen extraction
provides a real ~35% gain over prefill on this pair, but ctx doesn't
recover any of it.

So across three (concept, model) combinations:

- The koan-collapse failure under prefill is specific to one
  combination (happy.sad on gemma-4-31b-it). It is not a general
  property of the prefill pipeline.
- Gen extraction provides modest gains in two of three cases and
  none in the third.
- ctx outperforms prefill only on the combination where prefill is
  collapsing, and not robustly otherwise.

## Diagnosis: synthetic prompt quality is the hidden variable

A9 pointed out the obvious thing to check after the validation runs:
what did the synthesised per-scenario user prompts actually look like
on each run?

**happy.sad / gemma (the success):**
- Romantic relationships: `'What are your thoughts on romantic relationships?'`
- Family dynamics: `'How would you describe the dynamics within your family?'`
- Personal loss and grief: `'How have you been experiencing your grief?'`

**happy.sad / Qwen:**
- Romantic relationships: `'What does a healthy romantic relationship look like to you?'`
- Professional achievements: `'What professional achievement are you most proud of and why?'`
- Personal loss and grief: `'How has your understanding of loss changed over time?'`

**formal.casual / gemma — bundled scenarios:**
"Written communication", "Legal and regulatory frameworks",
"Behavioral etiquette", "Professional conduct", "Artistic style".

Qwen's prompts smuggle in aspirational / positive framing on every
scenario: "healthy" romantic relationship, "most proud of" achievement,
"wellness", "evolved over the years", "steps you're taking toward that
vision". The contrastive sad statement, paired with a prompt that
expects positive framing, is *more* OOD than the bare prefill, not
less. The ctx wrapping makes the activation site worse than baseline
for the negative pole.

formal.casual's scenarios are register-loaded by construction —
"Legal and regulatory frameworks", "Behavioral etiquette" — and no
synthetic prompt about those topics can be register-neutral. A casual
response to "What are your thoughts on current legal and regulatory
frameworks?" is OOD because the user-prompt register clashes with the
assistant register. ctx wrapping makes one pole less natural than the
other; the contrast becomes warped.

happy.sad on gemma got mostly register-neutral life-domain prompts
because gemma's instruction-tuning happens to produce neutral prompts
on neutral scenarios, and the bundled happy.sad scenarios happen to
be register-neutral life domains. That alignment was the precondition
for ctx to work, and it wasn't general.

The prompt-generation template used was

```
Write one short, neutral, open-ended user prompt that someone might
respond to about the topic '{scenario}'. The prompt should not assume
any particular tone or stance and should be the kind of question a
friend might ask or a journal might pose. Output ONLY the prompt.
```

This instruction was insufficient to constrain Qwen's positivity bias.
A stronger version would explicitly require pole-symmetric neutrality
("the prompt must be answerable equally naturally from either an
extremely positive or extremely negative emotional state — do not bias
the responder toward either direction"). That might catch Qwen's
defaults. It cannot help with intrinsically register-loaded scenarios.

The deeper property: **ctx's effectiveness depends on whether the
synthetic user prompt is pole-neutral relative to the curated
contrastive pair.** Pole-neutrality is itself a model-dependent and
concept-dependent property. The auto-generation step that makes ctx
"near-free" is the step that introduces the variability that breaks it.

## Decision

Don't ship the ctx pipeline change. The data does not support a global
switch and the failure modes (Qwen positivity bias, register-loaded
scenarios) are not addressable without either hand-authoring per-
scenario prompts (loses the automation) or developing a robust
neutrality classifier (out of scope here).

The current prefill pipeline is more robust than the
happy.sad/gemma result implied. Two of three tested (concept, model)
pairs show prefill swings of +5 to +8, comparable to or matching the
more expensive alternatives.

## Open questions / pickup points

If this thread gets picked up again:

**The koan-collapse failure is concept-and-model specific** — happy.sad
on gemma-4-31b-it is the only combination where prefill collapses at
α=−0.5. Worth investigating as a concept-level issue rather than a
pipeline-level one. Hypothesis: the bundled happy.sad negative
statements include several stylised fragments ("I look at our old
photos and feel a crushing weight for everything we have already
lost...") and the extracted direction over-loads on that fragmenting
style at the form-encoding early/mid layers. Targeted re-curation of
the bundled sad pole on gemma-targeted statements, or a per-concept
escape hatch (`ctx_user_prompt:` field in `pack.json` so a concept
that needs it can declare a hand-authored neutral prompt) might fix
just the affected concept without touching the rest.

**Gen extraction has a real but variable advantage.** On two of three
tested pairs it produces a meaningful gain (~35–86% larger swing). On
the third it produces nothing. There's no clear predictor for which
concepts benefit, and the cost is fixed regardless. Per-concept opt-in
to gen extraction is defensible if a triage heuristic can be found —
e.g. concepts where the prefill profile's DLS-retained layer count is
small (suggesting the prefill direction is narrow) might be the
candidates. Untested.

**A stronger anti-bias prompt template** is worth trying if anyone wants
to rescue the ctx principle: require pole-symmetric neutrality
explicitly, ideally with examples. Combined with hand-authored
per-scenario prompts on intrinsically-loaded concepts (formal.casual,
masculine.feminine, religious.secular), this might recover the
happy.sad/gemma effect more broadly. Worth one more pass if the
extraction quality on those concepts ever becomes a problem.

**The whole arc was N=3** (concept × model combinations). The shape of
the finding — "prefill mostly works; one specific failure exists; gen
has variable value" — is consistent across the three, but three
combinations is a small sample for a generalisation. If anyone runs
more (concept, model) combinations under this comparison framework,
keep `concept_three_way.py` and just add new entries to its
`CONCEPT_CONFIGS` registry. The bundled `default/<concept>` pack and
the model's chat template are the only dependencies.

## What the data actually says

Stripped of the framing, the three runs say: the saklas prefill
pipeline is reasonable; one specific (concept, model) combination has
a structural-collapse failure under negative-α steering that the other
combinations don't share; both ctx and gen extraction can fix that
specific failure but neither produces a robust improvement across the
combinations tested. The bundled pipeline as it stands is defensible.
The happy.sad-on-gemma failure is the thing to investigate next, if
anything.
