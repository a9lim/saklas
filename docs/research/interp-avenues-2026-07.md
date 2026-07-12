# Interpretability avenues beyond probes / J-lens / SAEs

*Research digest, 2026-07-12. Three parallel literature sweeps (workspace/null-space;
planning/latent reasoning; introspection/causal methods), synthesized against the
saklas stack. Verification: most load-bearing items were fetched at the primary
source by the sweep agents; items marked [snippet] were search-verified only.*

## The taxonomy: four kinds of "unvocalized"

The "subconscious" framing decomposes into four empirically distinct objects:

1. **Not-yet-vocalized (plans).** Content the model holds about future output —
   rhyme targets, planned paragraphs, pre-committed answers. Readable, sometimes
   causally load-bearing.
2. **Never-vocalized but workspace-visible.** Latent hops (Dallas→Texas→Austin),
   internal beliefs diverging from stated answers, eval-awareness, premature
   commitment. Probe-readable; absent from the text. Saklas probes already read
   this class partially.
3. **Sub-report-threshold (the subliminal band).** Injected/present content that
   shifts behavior while the model's self-report denies it. The report bottleneck
   is a *trained gate*, not absent signal.
4. **Workspace-dark (the complement).** The ~93% of concept-vector variance
   outside J-space. Constraint from the workspace paper's own clamping
   experiment: its causal effects route through *re-derivation into* J-space
   (complement swaps: 5% success alone, 0% with J-space clamped) — the complement
   is the preconscious feed, not a parallel channel to output.

Saklas's current stack reads (2) well, (1) not at all, (3) only from the
injection side, (4) only as the residual of `jspace_decompose`. Causal
attribution (patching-family) is a missing *leg*, orthogonal to all four.

---

## Tier 1 — open niches we have the machinery for

### 1. Subliminal-band psychophysics (experiment; ~zero new infra)

**Confirmed open**: nobody has published the quantitative object — a per-concept
curve of behavior-shift threshold vs self-report threshold, with the gap as the
measured "subconscious margin."

- Behavior threshold: α-sweep a steering term, measure `score_choices`
  distribution shift (KL/Bhattacharyya vs unsteered — soft, not argmax).
- Report threshold: scored self-report ("do you notice an injected thought?")
  via `score_choices` over yes/no; score *detection*, not identification —
  report content confabulates (Lederman & Mahowald, arXiv:2603.05414).
- Act 2 (causal): the report gate is a circuit — early "evidence carrier"
  features suppress a default-"no" gate (Macar et al., arXiv:2603.21396).
  Ablating refusal directions boosts detection +53%, a trained bias vector +75%.
  Our `!` operator can move the report threshold while the behavior threshold
  stays — measure the band shrink.
- Act 3 (monitor dissociation): Qwen-32B shows a mid-layer latent detection
  signal *attenuated before sampling* while the output denies injection
  (Pearson-Vogel et al., arXiv:2602.20031; mechanism-informed prompting raises
  verbal detection 0.3%→39.9%). Our monitor can read the latent signal live
  against the verbal denial.
- Controls to design in: Singh/Linzen/Ravfogel (arXiv:2605.26242) — models may
  be generic anomaly detectors, not privileged introspectors; need input-level
  vs activation-level intervention discrimination. Also strength-without-source
  (arXiv:2512.12411): magnitude sensing is early-layer and dissociated from
  content ID.

### 2. J-lens complement instrumentation + sleeper steering (small infra + experiment)

**Confirmed open**: the workspace paper is 6 days old (2026-07-06); no published
probe/steer study of the complement exists. "Sleeper directions" (delayed-effect
steering) explicitly unclaimed.

- Numbers (fetched from transformer-circuits.pub/2026/workspace/): median 6–7%
  of concept-vector variance in J-space (~93% outside); 10–15% for two-hop
  intermediates; J-space ≤~10% of total activation variance at any layer.
- Formalization warning (LW thread — Bushnaq, Linsefors): J-space is the set of
  *sparse non-negative combinations* of lens vectors — a cone, not a subspace,
  and the span can be ~full-rank. "Complement" must be defined as
  low-conductance-under-J_l (SVD of J_l → per-layer conductance spectrum) or
  pursuit-residual (1 − the share `jspace_decompose` already computes).
- Channel ideas: a `:silent` sibling of `:fraction` — whitened mass of h in the
  low-conductance subspace of J_l (raw norm is rogue-dominated; the whitener is
  our specialty). Plus downstream-connectivity salience (Circuits Update May
  2026: connectivity predicts steerability better than activation descriptors) —
  a vocabulary-free importance measure for complement directions.
- Sleeper experiment: `|`-project a steering term against its own J-space
  pursuit approximation, inject, and watch the live lens aggregate for *delayed*
  workspace emergence. The clamping result predicts effect-only-via-re-derivation
  — so the measurable is incubation latency + transformation, not a parallel
  path. Prior art for complement-constrained steering exists but with X ≠
  J-space: Head-Masked Nullspace Steering (arXiv:2604.10326), AlphaSteer
  (arXiv:2506.07022).
- Replication assets: `anthropics/jacobian-lens` (official companion code, HF
  decoders); Neuronpedia hosts an interactive J-lens (Qwen3.6-27B);
  `solarkyle/jspace-lenses` on HF has fitted lenses for **Gemma 4 E4B / 12B /
  12B-abliterated / 26B-MoE** — cross-check targets for our own fits. [snippet]

### 3. Plan / future readout (medium infra)

- **Cheapest**: ParaScopes continuation transplant (Pochinkov et al.,
  arXiv:2511.00180) — transplant the `\n\n`-boundary-token residual into a fresh
  context and let the model regenerate the *planned paragraph*. No training; we
  have full generation control. Loom-native as a token drilldown.
- **Deeper**: cross-position Jacobian lens `J_l^{(k)} = E[∂h_{final,t+k}/∂h_{l,t}]`
  — rides the existing jlens VJP machinery (different probe positions, same
  backward infra). Consensus plan-carrying sites: line/paragraph boundary tokens
  (Biology paper; arXiv:2605.07984 [snippet]; arXiv:2601.20164 [snippet]).
- **Pre-CoT commitment probe**: linear probes on pre-CoT residuals predict the
  final answer at ~0.9 AUC and steering them flips answers >50% of the time
  (arXiv:2603.01437 [snippet]) — the model commits, then rationalizes. Shaped
  exactly like a gateable saklas channel.
- Anthropic thread: Latent Planning Emerges with Scale (Hanna & Ameisen, ICLR
  2026, arXiv:2604.12493) — planning scales with size, steering can *strengthen*
  it. Where's the Plan (arXiv:2605.07984 [snippet]): future-rhyme is decodable
  everywhere but only Gemma-3-27B *causally relies* on it (layer-~30 handoff,
  ~5 attention heads). Decodable ≠ used — a future readout needs a patching
  validation leg. Planning is also inconsistent (arXiv:2508.18098: Gemma-2-2B
  improvises poems) — want a planning-vs-improvising discriminator channel.
- Latent thought content is *distributional* — superposition of search frontiers
  (arXiv:2505.12514) — so the readout should be soft/set-valued, not argmax.

---

## Tier 2 — strong experiments, mostly existing infra

### 4. Belief-vs-statement probes ("knows more than it shows")

- Question-only correctness direction: linear probes on the pre-generation state
  predict eventual answer correctness, generalize OOD, peak mid-layer, beat
  verbalized confidence (arXiv:2509.10625) — internal P(IK) before any token; a
  natural probe gate ("only trust generations where the correctness coordinate
  is high"). Known blind spot: math/logic (confidently wrong).
- Truth/belief directions: 2-D truth subspace (Bürger et al., NeurIPS 2024);
  belief probes predict retraction (arXiv:2505.16170 [snippet]); truth directions
  generalize across logical transformations (arXiv:2506.00823 [snippet]).
- Eval-awareness cluster: probes separate test-vs-deploy framing
  (arXiv:2507.01786), steering can suppress awareness (arXiv:2510.20487);
  "unverbalized eval awareness" is the exact phenomenon, name and all.
- Deception is *multi-directional* — type-matched probes beat a universal one by
  3× margin (arXiv:2602.01425); a probe roster/manifold, not an axis. Fits the
  saklas manifold thesis directly.

### 5. CoT-faithfulness instrumentation

We already have the `thinking`/`response` triggers and committed thinking.
- Measure probe-trajectory divergence between thinking and response segments;
  compare which better predicts the final answer distribution.
- Rationalization detector: activation probes catch motivated reasoning even
  when the CoT text is rationalized (arXiv:2603.17199 [snippet]); steering-induced
  flips get rationalized by subsequent CoT (arXiv:2604.01202 [snippet]).
- Reasoning models verbalize hints they used ~25–39% of the time; RL-hacking
  verbalized <2% (Chen et al., arXiv:2505.05410). Anthropic decoded deliberately
  ROT-13-obfuscated CoT with logit-lens-class tools unsupervised (Fang & Marks,
  arXiv:2512.01222) — lens machinery can recover hidden reasoning streams.

### 6. The causal leg: attribution verb

All saklas reads are correlational. Cheapest-reliable causal localization, 2026:
- **RelP** (arXiv:2508.21258): attribution patching with LRP coefficients at
  identical cost (2 fwd + 1 bwd); correlation with true activation patching on
  GPT-2 IOI MLPs goes 0.006 (AtP) → 0.956.
- **HVP reliability check** (arXiv:2606.09899): per-estimate trust score + one
  extra backward to remove leading-order error.
- Pipeline: AtP*/RelP sweep → HVP flags → direct patching on survivors. The
  jlens fit already owns the backward-pass machinery. Would give saklas a
  `attribute`-shaped verb and a causal validation leg for every probe/steering
  claim (including the future-readout and complement work above).

---

## Tier 3 — worth knowing, lower priority

- **Attention-side reads**: attention *probes* (learned pooling over token
  activations; production-grade at GDM for Gemini misuse monitoring,
  arXiv:2601.11516) are the 2026 growth area — attention as probe architecture,
  not as signal. ACC++ (arXiv:2602.13483): single-forward QK-subspace causal
  signals. KV-cache trait probing: genuinely open, nobody's done it, speculative
  value. HeadVis (Transformer Circuits 2026) for head-function hypothesis
  generation.
- **Model diffing**: narrow finetuning leaves readable traces in first-token
  activation diffs on unrelated text (arXiv:2510.13900) — the cheap method;
  crosscoders/transcoder-adapters (arXiv:2602.20904) the heavy one. Relevant
  when comparing finetuned variants, not for live sessions.
- **Trained verbalization heads**: LatentQA (arXiv:2412.08686), Predictive
  Concept Decoders (arXiv:2512.15712), Introspection Adapters
  (arXiv:2604.16812; SOTA on AuditBench, catches encrypted-finetune attacks) —
  the "ask the residual stream questions in English" alternative to hand-built
  lenses. Training-side; heavier than saklas's fit-an-artifact model.
- **Endogenous steering resistance** (arXiv:2602.06941): Llama-3.3-70B recovers
  mid-generation from misaligned steering while it's still active — dedicated
  consistency-checking circuits, scale-dependent. Directly relevant to saklas
  steering practice; worth checking whether gemma-4-12b fights our injections.
- **Premature commitment** (arXiv:2606.22936): cross-run hidden-state
  convergence at a fixed early step predicts behavioral consistency (monitor
  AUROC up to 0.97), orthogonal to correctness — a hidden settling event.

## Standing caveats (field-level)

- **SAE vindication-and-then-some**: on synthetic ground truth SAEs recover 9%
  of true features while reporting 71% explained variance; random baselines
  match trained SAEs on interpretability/probing/editing (arXiv:2602.14111).
  GDM formally deprioritized SAE research (Mar 2025). SAE dark matter: >90% of
  error norm is linearly predictable; the nonlinear core is scale-*constant*
  (arXiv:2410.14670). The a9 position ("sorta obsolete") is now the published
  consensus — but the same paper family warns all reconstruction metrics
  overstate coverage.
- **Three kinds of dark** (synthesis): vocabulary-artifact dark (W-space content
  the single-token lens can't name — Eleos commentary's W-space vs J-space;
  multi-token lenses would recover it), wrongly-shaped dark (multi-dimensional /
  nonlinear features — Engels ×2; *our curved-manifold machinery is the
  designed tool for exactly this*), and irreducible dark (scale-constant
  nonlinear SAE error + attribution-graph error nodes).
- **Metacognitive space is low-dimensional** (arXiv:2505.13763): models can
  monitor/control only a small subspace of their own state — most internal
  state is structurally unreportable. The complement of the metacognitive space
  is another candidate operationalization of "subconscious."
- **Probes are adversarially evadable** (Neural Chameleons, arXiv:2512.11949):
  finetuning to evade benign probes generalizes zero-shot to unseen safety
  monitors. The "hostile telepaths" concern (LW): training against a lens
  pushes computation into the complement — which is an argument *for* building
  complement instrumentation now.
- **Subliminal learning caveat**: the Nature-published effect (Cloud et al.)
  may be substantially a LoRA artifact — disappears under full finetuning
  (arXiv:2606.00831); channel location determines auditability
  (arXiv:2606.22019).

## Parked: Vogel's Qwen introspection result through the J-lens (2026-07-12)

Vogel, "Small Models Can Introspect, Too" (vgel.me/posts/qwen-introspection/):
Qwen2.5-Coder-32B, repeng PCA concept vectors ("cat"/"bread") injected via the
KV cache of earlier turns (layers 18–33), steering removed before the detection
question; introspection measured as steered-vs-unsteered logit shift on yes/no
(informative architecture prompt: +0.372% → +52.3% "yes" shift; Lorem-Ipsum and
wrong-location controls fail, so not a generic yes-bias). Logit lens over all 64
layers: a "yes" signal hill at layers 46–52 (only steered + informed prompt), a
broader hill 52–62 (present even unsteered), and strong suppression of "yes" in
the final two layers. Precursor/companion to the Latent Introspection paper
(arXiv:2602.20031).

The J-lens redo is more informative for a specific reason: it discriminates two
suppression mechanisms the logit lens conflates.
- **(a) Active gating**: the mid-stack detection signal is workspace-transmitted
  (high p_l for the yes/detection tokens through the band) and then killed late
  — the Macar et al. default-"no" gate acting on workspace content.
- **(b) Never transmitted**: the mid-stack hill is a breadcrumb — logit-lens
  visible but low J-conductance throughout; the "final-layer suppression" is
  then an artifact of premature decoding, not a gate.
Distinguishing (a) from (b) is exactly the workspace-vs-complement question
from avenue #2 above — the two parked threads converge.

Practical notes for the eventual run: near the final layers J_l converges to
the raw unembedding row, so J-lens ≈ logit lens exactly where Vogel saw the
suppression — the informative delta is the 46–62 hills, i.e. the late-band
region. Fit the lens through the late layers (the per-layer readout covers any
fitted layer; only the aggregate is band-restricted 40–90%). "yes"/"no" are
single tokens, so `jlens/yes`-style readout probes apply directly, and gate
scalars already ride the readout channel. Open question for the harness: vgel's
injection is KV-cache-persistent (steer turn 1, ask unsteered in turn 2) —
check what saklas's conversational path preserves across generate calls before
designing the replication.

## Speculative footnote (marked as such)

The clamping result operationalizes a psychodynamic stack with unusual fidelity:
workspace = access consciousness; plan representations = preconscious
(retrievable on demand); the complement = unconscious *whose only route to
speech is re-derivation into workspace form* — influence through transformation
into acceptable content, never directly. The introspection gate (default-"no",
trained by post-training, ablatable) is then a literal repression mechanism,
and the +53%/+75% elicitation results are the band moving under intervention.
Not load-bearing for any experiment above; but the subliminal-band experiment
and the sleeper-direction experiment are, jointly, a test of how far the
analogy actually carries.
