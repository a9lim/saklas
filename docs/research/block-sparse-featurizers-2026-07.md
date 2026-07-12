# Block-sparse featurizers → unsupervised manifold discovery for saklas

*Research digest, 2026-07-12. Prompted by Fel et al. 2606.25234 ("can we do this
for LLMs?"). Verification: Fel et al. read at primary source (full HTML);
SASA (2606.06333) abstract fetched; Engels et al. (2405.14860) prior knowledge +
search-verified; the SAE-scaling paper (2509.02565) is [snippet] only.*

**Verdict: viable — the LLM version already exists (SASA, June 2026). The
unclaimed territory is the steering/probing side, which is exactly the half
saklas already ships. The complement is unusually clean: BSF/SASA have
unsupervised discovery and no intervention runtime; saklas has the full
subspace steering + probe runtime and no unsupervised discovery.**

---

## The paper (Fel et al., arXiv:2606.25234)

"Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept
Manifolds." Vision (DINOv3, InceptionV1, SDXL). The move: generalize SAEs by
grouping dictionary atoms into blocks of dimension b and enforcing sparsity at
the block level, so the atomic unit of interpretation is a subspace, not a
direction. A concept then has two readouts — **presence** (block norm ‖z_m‖₂,
the analogue of an SAE activation) and **position** (the within-block code,
i.e. where on the concept manifold this token sits).

Three variants, shared linear decoder `x̂ = zD`:

- **Vanilla BSF** — free untied encoder; hard block projection Π_k keeps the k
  blocks of largest ‖z_m‖₂, zeroes the rest; reconstruction-only loss.
- **Grassmannian BSF** — tied; each block D_m ∈ St(b,d) an orthonormal Stiefel
  frame; `z = Π_k(γ x Dᵀ)` with a scalar energy-compensation γ.
- **Group Lasso BSF** — free encoder + block soft-threshold
  `sh_θ(z)_m = max(1 − θ/‖z_m‖₂, 0) z_m` (prox of the ℓ₂,₁ norm).

Manifold coordinates are extracted post hoc: collect each active block's
contributions `m_m = z_m D_m` over the token stream, PCA them; the PCA frame is
the concept's natural basis (their curve-detector block recovers the full
orientation circle, with Fourier modes ω=1/2/3 at 59/18/12% of variance).

**The MDL argument** is the honest core: framing the code as compression, block
selection costs `log₂(G choose k)` bits vs an SAE's `log₂(Gb choose kb)` —
BSFs beat b=1 SAEs on total description length at every dictionary width on
DINOv3, with optimal b = 2–4. Task-derived distortion floors (classification
tolerates R²≈0.8, depth needs ≈0.9) anchor the comparison so reconstruction
alone doesn't decide it.

**Steering** (SDXL): clamp a block's code to waypoints on a Kohonen SOM fit to
the block's empirical code density — i.e. move only within the subspace's
high-density region. Note this is *cruder* than saklas's kernel: no
along/onto split, no tube thickness, grid waypoints instead of a fitted
surface.

Authors' own hedge on language: "applying block sparsity directly to language
or to video would mismatch the prior… I suspect the matched object should
change accordingly." They cite parallel LLM work already making the relaxation
(Dalili & Mahdavi = SASA below; Hindupur et al. 2025).

## The LLM side is already demonstrated (SASA, arXiv:2606.06333)

Dalili & Mahdavi, "Subspace-Aware Sparse Autoencoders," June 2026. Same
relaxation, LLM-native: decoder subspaces + Top-s group gating + a
**nuclear-norm rank-adaptive regularizer** so each block chooses its effective
rank. Theory: reconstructing an intrinsically ≥2-D feature with 1-D decoders
needs exponentially many atoms (the formal version of feature splitting);
block size r ≥ d_i restores polynomial sample complexity. Empirics on GPT-2 +
Mistral-7B: reduced splitting and absorption, improved monosemanticity,
matches/exceeds standard SAEs on roughly half the token budget.
**No intervention/steering experiments at all.**

Supporting landscape:

- **Engels et al. (2405.14860)** — ground truth that irreducible multi-dim
  features exist in LLMs: weekday/month *circles* in GPT-2 and Mistral-7B,
  found by clustering SAE decoder directions, causally implicated in modular
  date arithmetic. Token-level, not centroid-level (this matters — see below).
- **SAE scaling under feature manifolds (2509.02565)** [snippet] — SAEs reduce
  loss by tiling manifolds with many sparsely-activating latents at the
  expense of rarer features; part of SAE scaling pathology is manifold
  tiling. Block sparsity is the natural structural fix.
- Group-SAE (2410.21508) is a different "group" (layer-grouping for training
  efficiency) — not this.

## Why saklas is the natural host

A discovered block **is** a `LayerSubspace`: block span = the subspace,
within-block code distribution = discover coords, empirical code density =
what the fuzzy-manifold σ-field / tube fit already models. Mapping:

| BSF/SASA object | saklas object (existing) |
|---|---|
| block span D_m | `LayerSubspace.basis` |
| block norm ‖z_m‖₂ (presence) | probe `:fraction` / `:membership` channels |
| within-block code (position) | `ProbeReading.coords` / `%` position |
| SOM high-density waypoints | σ-field tube fit (strictly richer) |
| clamp-to-waypoint steering | `subspace_inject` along/onto |
| on-manifold vs arbitrary direction | `experiment naturalness` (built-in eval) |

Saklas independently converged on the "concept = subspace + position +
thickness" ontology from the steering side; BSF/SASA arrived at it from the
dictionary-learning side. What saklas lacks is unsupervised discovery (our
manifolds are authored or fit from *labeled* node corpora); what they lack is
a steering/probe runtime. As of this sweep, **unsupervised manifold discovery
feeding a real intervention runtime exists nowhere in the literature.**

## The science hook: centroid-level vs token-level geometry

Our replicated "gemma flattens structured concept geometry" results
(circumplex λ2/λ1=0.31; taxonomy flat r≈0.15; months auto-fits flat) are all
**centroid-level** — one pooled point per concept corpus. Engels' rings are
**token-level**. Pooling a ring that individual tokens traverse can average it
into a blob, so the flattening findings and the ring findings don't actually
contradict — they measure different objects. A BSF/SASA fit on gemma directly
tests whether the flattening is real or a pooling artifact, and dovetails with
two standing TODOs: the months/days re-fit under the fixed clustered-ring
detector, and the cross-model "does Qwen ring?" question.

## Risks / gotchas

- **The prior may genuinely mismatch language** (the Fel hedge). Vision
  manifolds come from continuous transformation groups (orientation, hue,
  lighting); language has fewer obvious continuous parameters. Engels found a
  handful of circles, not hundreds. Expect most blocks to resolve effectively
  rank-1 — SASA's rank-adaptivity is the right response (let data pick rank
  per block); per-block effective rank is the thing to measure. A mostly-flat
  dictionary would *corroborate* the gemma-flattening line, not waste the
  effort — and flat blocks steer on the fast affine path anyway.
- **Rogue dims.** Block selection by raw L2 norm on an LLM residual stream
  will be massive-activation-dominated — the exact failure the whitener
  exists for. Selection must happen in normalized/whitened space (the vision
  paper never hits this; DINOv3 stats are tamer). This is our home turf.
- **SAE training hygiene transfers**: input normalization, dead-latent
  resuscitation becomes a block-level aux loss.
- **Compute is fine.** Vanilla BSF is a ~30-line delta over a TopK SAE.
  gemma-2-2b (d=2304) or gemma-3-4b (d=2560), one workspace-band layer,
  G≈8–16k blocks × b=4 (32–65k latents, decoder ~85–170M params), a few
  hundred million streamed tokens — a 4090 job. Stream activations rather
  than storing (200M tokens × d=2560 fp16 ≈ 1 TB raw).

## Plan (phased; each phase is a go/no-go gate for the next)

**Phase 0 — post-hoc blockification (days; zero training).** Cluster the
decoder directions of an existing SAELens SAE on gemma (Engels-style: cosine +
spectral clustering), fit within-cluster PCA on codes over a token stream,
look for irreducible multi-dim clusters (weekday/month rings first, then
unknowns). Uses the existing `sae` runtime for harvest. Deliverable: does
gemma ring at the token level? If yes → live target for Phase 1. If no →
sharpened flattening finding; take the question to Qwen before training
anything.

**Phase 1 — train the featurizer (≈week part-time + 4090 runs).** TopK-block
SAE with SASA's rank-adaptive regularizer and whitened block-norm selection,
one gemma layer, optionally warm-started from Phase-0 clusters (cuts token
budget). Run the paper's MDL comparison vs b=1 — the honest
language-viability arbiter — plus per-block effective-rank histogram.

**Phase 2 — the saklas bridge (the novel bit; medium feature).** A
`discovered` manifold source: import a block as a fitted artifact (span +
coords + σ from code density), steerable via `%` / `subspace_inject`,
probeable through existing channels, `experiment naturalness` as the
on-manifold-vs-straight-chord eval. Steering/probe machinery needs near-zero
change; the work is the import path + artifact format extension.

## Sources

- Fel et al., "Structuring Sparsity: Block-Sparse Featurizers Capture Visual
  Concept Manifolds" — arXiv:2606.25234 (primary, full text)
- Dalili & Mahdavi, "Subspace-Aware Sparse Autoencoders" (SASA) —
  arXiv:2606.06333 (abstract fetched)
- Engels et al., "Not All Language Model Features Are One-Dimensionally
  Linear" — arXiv:2405.14860
- "Understanding sparse autoencoder scaling in the presence of feature
  manifolds" — arXiv:2509.02565 [snippet]
- Group-SAE — arXiv:2410.21508 (disambiguation only)
- Hindupur et al. 2025 (cited by Fel as parallel LLM work; not independently
  verified this sweep)
