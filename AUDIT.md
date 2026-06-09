# saklas engine audit

Date: 2026-06-09. Tree: `dev` @ `ca79755` (clean). Scope: text-generation
performance, steering quality, probe/monitor quality, extraction quality —
`saklas/core/`, `saklas/io/`, tests, against ARCHITECTURE.md and the AGENTS.md
invariants as spec.

Method: four independent deep-read audits (one per area), each instructed to
treat code as ground truth and docs as claims; every high-severity finding was
then re-verified directly against source in a second pass. Provenance tags:

- **[direct]** — re-verified line-by-line in the second pass; treat as fact.
- **[auditor-numeric]** — the area auditor executed the real code path
  (CPU/synthetic) and observed the number quoted; structurally confirmed in
  the second pass.
- **[analytic]** — derived from code structure; correct unless an unstated
  invariant intervenes; flagged where an empirical sweep should confirm.

Severity: **high** = wrong behavior or wrong numbers shipped silently;
**med** = biased/degraded/fragile; **low** = robustness, ergonomics, drift.

---

## 1. Summary

Four stories, in decreasing urgency:

**1. The gain-recalibration blast radius (S1–S3, S11).** Commit `462a779`
(2026-06-06) bumped `_MANIFOLD_ALONG_GAIN` 0.125 → 16.0 when the affine push
target became whitened-unit. The calibration sweep covered affine push terms
only (`formal%formal`, `personas%caveman`, `personas%hacker`). Three other
consumers of the same constant were never renormalized:

| consumer | calibrated? | current behavior |
|---|---|---|
| affine push (vectors, poles, `%`-on-flat) | yes (3 targets, gemma-4-12b) | works as documented |
| ablation axes (`!x`, κ=1) | no | sign-flips and **amplifies** the ablated component (~−15·q, norm-cap-saturated) |
| curved `%` along | no | ~130× overdrive on raw authoring-coordinate offsets; foot lands deep in RBF extrapolation |
| `onto` (`_MANIFOLD_GAIN = 0.5`) | calibrated 1 day *before* the bump | its operating point (eff_along ≈ 0.04) no longer exists |

`!refusal` today *amplifies* the refusal direction until the norm cap bites.
Every spectral/authored curved fit produces norm-cap soup at any user along
≳ 0.1. Both are latent only because the bundled `personas`/`emotions` resolve
flat on gemma-4-12b and `!` is rarely exercised end-to-end. This is the single
highest-leverage fix cluster in the codebase.

**2. The gate-channel frame problem (P1, P2, P6, P7, P8).** The Mahalanobis
read math is verified correct, but the conventions layered on top are not:
discover-layout coordinates have unconventioned sign and fit-dependent scale
(so `@when:confident.uncertain > 0.4` can mean "when *uncertain*" on the next
fit, against a ±10 instead of ±1 pole), `@label`/`~label` channels only exist
when the label lands in the top-N nearest (so `<`-gates structurally cannot
fire in their intended region), and `label_scale` silently changes meaning
(spacing vs tube thickness) for curved+σ probes.

**3. The statistical floor (E3, E4, P3, E5, P4).** Everything sits on 48
single-token samples per node centroid and 96 neutral rows against
D = 2560–5376. The whitener is ridge-Euclidean outside a ≤95-dim span; its
trace-based λ is rogue-dominated and over-shrinks real whitened separation
(6.4× in a synthetic demo). DLS has no noise margin, so roughly half of
pure-noise layers pass the straddle test and receive steering budget. EV
layer-weighting is exactly vacuous for every 2-node probe. None of these are
bugs; together they are the cheapest global SNR multiplier available, and most
of the fix is configuration plus ~3 localized changes.

**4. Performance is in good shape; the gates aren't.** The documented hot-path
invariants substantially hold (§7). The real wins found: prefill computes
full-vocab logits for every prompt position (`logits_to_keep` never passed),
the curved transport does a CPU SVD round-trip per layer per token on MPS, the
sampler pays 2 avoidable host syncs per token, and the monitor's incremental
path retains far more per token than advertised. Separately, the throughput
regression test doesn't exercise what it claims and the ablation perf test is
structurally red (probes a tag that no longer exists).

Also in the lifecycle layer: `--no-dls` is a dead flag, the `_role-<slug>`
tensor variant is documented but never written by any fit, and fits are not
invalidated when `baseline_prompts.json` or the neutral corpus change — the
only two paths found where extraction silently serves a *wrong* (not merely
noisy) artifact.

---

## 2. Steering correctness and calibration

### S1. `!x` ablation is anti-ablation at production gain — high [direct]

`hooks.py:1001` sets `eff_along_L = shares[L] * _MANIFOLD_ALONG_GAIN` (= 16.0,
`hooks.py:580`) for the merged affine group; the kernel (`manifold.py:2348`)
computes `p_new = q + along·(target − κ·q)`. For an ablation axis (κ=1,
target≈0) that is `p_new = (1 − eff_along)·q ≈ −15·q`: the "ablated" component
is sign-flipped and amplified until `norm_cap = 3·‖h‖` saturates.
Auditor-numeric check through the real `synthesize_subspace → apply_to_model →
subspace_inject` chain: component along d̂ went **+2.08 → −6.37** (uncapped
−31), ‖h‖ 2.13 → 6.38 (cap-saturated). The documented semantics
(`h' = h − α(h·d̂ − μ·d̂)d̂`, bare `!x` ⇒ α=1) require the κ-branch multiplier
to be the user's α (~1), not the share-weighted slide gain. Under the old gain
0.125 this was a too-weak 12.5% ablation; the bump flipped it.

Fix: decouple the collapse fraction from the slide gain. Change the kernel to
`p_new = q + along·target − κ_frac·q`, with `synthesize_subspace` baking
`κ_frac = κ_mask · α_ablate` clamped [0,1], gain multiplying only
`along·target`. Update `_single_affine_fast`, the recompose drop condition (a
pure-ablation group has `along·target = 0` but κ_frac ≠ 0 — must not be
dropped), and the κ carry in `_orthogonalize_affine_against`. Add an
end-to-end test asserting `!x` drives the component to ~0 at production gain.

### S2. `AblationTerm.coeff` parsed but never consumed — med [direct]

`steering_expr.py:183` carries the coefficient; the dispatch
(`session.py:2791–2799`) reads only `entry.target`/`entry.trigger`, and
`synthesize_subspace`'s `ablate` argument (`manifold.py:1890`) has no coeff
slot. `0.5*!x` ≡ `!x`. Folds into the S1 fix (κ_frac = coeff).

### S3. Curved `%` along over-driven ~130× — high [direct]

`hooks.py:891–894`: `eff_along = along * shares[L] * 16.0` for curved
manifolds, but a curved term's `target`/`origin` are raw authoring coordinates
— the whitened-unit normalization happens only in `synthesize_subspace`, which
curved terms bypass (`add_manifold`). `translate_foot` then moves the foot by
≈ 16·share·along × the full origin→target authoring distance.
Auditor-numeric on a real `fit_layer_subspace` fit with nodes spanning [−1,1]:
user along 0.3 → foot at coord **+8.6** (deep r³ extrapolation, output pinned
at the 3‖h‖ cap); along 1.0 → +31. On a periodic `BoxDomain` axis the
overshoot wraps mod period — the steering position aliases around the circle.
`CustomDomain.clamp_position` is the identity (`manifold.py:577` — its
`bounds` field is stored but never enforced), so discover-spectral fits have
no containment except the norm cap. The docs' "[0,1] slide fraction" semantics
existed at gain 0.125 and are gone. `_MANIFOLD_GAIN = 0.5` (onto) was
calibrated 2026-06-05 at eff_along ≈ 0.04 — an operating point that no longer
exists.

Fix: a separate curved gain (≈ 0.125 restores the fraction semantics), or
define a whitened-unit curved budget (scale `target − origin` by
`1/‖J·(target−origin)‖_M` per layer). Then re-run the onto sweep at the new
operating point. Interim mitigation: clamp curved eff_along to [0,1].

### S4. Composed multi-term share profile dominated by the larger-raw-magnitude term — med [analytic]

`manifold.py:2056–2058`: the per-layer budget profile is `‖Σ cf·wd‖_M` of the
*raw* coeff-weighted displacement, while the target is built from
whitened-unit fragments (`:2063–2067`). Single-term: the normalization cancels
and behavior matches docs. Two terms with very different whitened node
distances (documented spread 1.15..72): the cross-layer profile is essentially
the far term's, so the near term's direction is spent on the far term's layer
distribution — the spread the whitened-unit fix removed from the target
survives in the profile under composition. Fix: compose per-fragment
normalized profiles, e.g. `share_L = ‖Σᵢ cfᵢ·wdᵢ/‖wdᵢ‖_M‖_M`.

### S5. Vector form vs `%`-label form of the same concept get different layer profiles — med [analytic]

The vector path folds through `folded_vector_directions` (`vectors.py:757`,
baked dir = δ̂·share_fit) and re-measures that already-whitened magnitude in M
again at synthesis (`manifold.py:2058`) — double-whitening the profile
(∝ share_fit,L·‖δ̂_L‖_M). The `%`-label path reads real `node_coords` —
single-whitened (∝ ‖d_L‖_M). Same concept, two cross-layer budget shapes.
Fix: have the vector path push real node geometry (`δ̂_L`, real Euclidean
‖δ_L‖ coord) so both forms enter `synthesize_subspace` in activation units.

### S6. Per-layer degenerate fallback mixes metrics across layers — low [direct quote]

`manifold.py:2068–2072`: when whitened-unit fragments cancel at one layer
(`‖tc‖ < eps`), that layer falls back to raw-Euclidean target *and* Euclidean
share while siblings stay whitened — the mixed cross-layer profile the module
docstring (`manifold.py:1953–1957`) claims is gated against. Fix: emit
share 0 / target 0 on per-layer degeneracy (the layer genuinely cancels).

### S7. Share normalization before curved-overlap drops; post-strip weakening — low [analytic]

`hooks.py:967–968` normalizes shares to mean 1 before `:989–990` drops layers
whose affine span sits inside a curved span, so survivors' mean ≠ 1 for that
group. And `_orthogonalize_affine_against` (`hooks.py:619–625`) strips the
curved-span component from the push displacement, silently shrinking that
term's whitened norm below α (uncalibrated, undocumented strength change); the
κ carry `(M*M)@kappa` can fall below 1 on full-ablation axes (partial ablation
where full was requested — currently masked by S1).

### S8. `_frame_rotation_transport` degenerate-frame claims overstated — low [auditor-numeric]

Docstring (`manifold.py:2190–2198`) claims "exactly the identity …
norm-preserving and lossless". Measured: identity-at-rest error ~1.2e-5
(benign fp noise, fires every token); a rank-deficient old frame (collinear
RBF Jacobian columns) *does* transport in the "90°" plane and loses norm
(‖Hn‖ 1.288 → 1.173), contradicting the comment. Behavior is arguably the
right geometry; the comments are wrong and the degenerate case is untested.

### S9. Pure-ablation share is ‖sum of rows‖, not summed magnitudes — low [direct quote]

`manifold.py:2078` sums ablation rows before taking the magnitude, vs
ARCHITECTURE §4's "summed ablation-row whitened magnitude". `!a + !b` with
anti-correlated directions under-weights those layers. Cosmetic until S1 is
fixed.

### S10. Multi-group sequential caps compound; order dependence — low [analytic]

`hooks.py:450–484` applies trigger groups sequentially with a per-group
3·‖h‖ cap — two active groups can jointly scale ‖h‖ up to 9×; an ablation axis
in group 2 overlapping group 1's push direction collapses the just-added push,
in dict-iteration order. Cross-group affine spans are never orthogonalized
(only affine-vs-curved is). Acceptable semantics; needs a doc note and a
composition test.

### S11. Calibration provenance and the extrapolation map — med [direct]

What 16.0 was calibrated on: a live gemma-4-12b α-sweep over three affine
targets, thinking off, MPS (commit `462a779`; no script committed). The
committed `scratch_*.py` files are all older-generation experiments
(pre-bump gain sweeps at 0.04–0.14; kernel-shape ablations on caveman only).
Empirically untested: any other model; curved `%` (S3 — known-broken
analytically); `!` (S1 — known-broken); ≥2-term composition (S4); norm-cap
fire rate per layer (whether the cap silently flattens the intended per-layer
profile on high-share layers — one instrumented sweep logging `scale < 1`
frequency inside `_soft_norm_cap` resolves it); depth compounding (push
offsets sum linearly across layers ≈ GAIN·α·n_layers whitened units along the
concept vs nodes at 1–70 units — `along` is a register-intensity dial, not a
geometric position; downstream-block amplification of early-layer offsets is
unmeasured — a per-layer ΔlogP probe sweep would resolve it).

### S12. Persona-fan frontier: where each proposed fix lands — design note

(a) *Sphere the between-persona covariance at fit time* lands in
`derive_pca_coords` (`manifold.py:2511`, rescale consensus-Gram eigen-axes to
unit variance) and/or `fit_affine_subspace` node_coords. Requires refitting
persona tensors; shifts `personas[i]` gate thresholds, `@label` distances,
`~label` bandwidths, webui layouts. α calibration survives via the
whitened-unit target. (b) *Strip the shared axis from the target at steer
time* lands at dispatch (`_affine_manifold_push`, session.py:366, or
per-fragment in `synthesize_subspace` before unit normalization) — no artifact
change, no gate change, and the whitened-unit normalization automatically
re-inflates the subdominant identity component to the full GAIN·α budget. Risk:
the shared axis carries genuine persona-ness register; make the strip fraction
a term parameter. (b) is the cheap experiment; (a) the durable fix if (b)
validates. Note (a) composes with P1 (layout orientation) — do them together
if both land.

---

## 3. Probe / monitor quality

### P1. Discover-layout gate coordinates: arbitrary sign, fit-dependent scale — high [direct]

`derive_pca_coords` (`manifold.py:2552–2570`) returns
`coords = evecs[:, :k] * evals.sqrt()` with eigenvector sign explicitly
unconventioned (code comment: "up to per-axis sign, immaterial for a layout" —
immaterial for steering, load-bearing for gates) and magnitude = consensus
whitened separation. The monitor's gate coordinate maps the per-layer read
onto this shared layout (`monitor.py:1815–1867`, applied at `:387–393`), and
`flat_scalars` emits it as the bare/`[i]` channels (`:1254–1258`).
`orient_to=0` sign-fixes the per-layer *basis* (`manifold.py:1326–1330`), but
the read maps onto the unoriented *layout*. Auditor-numeric: across 6 seeded
fits of the same 2-node concept, node 0's coordinate came out
`−10.3, −9.3, −10.0, +9.8, −8.8, −9.3` — sign flips between fits, magnitude
~10 not 1. Docs promise "pole-normalized: 1.0 at the positive node". Every
bundled concept probe, `personas`, and `emotions` go through this path.
`@when:confident.uncertain > 0.4` can silently mean "when uncertain" on the
next fit/model, against an arbitrary scale.

Fix: orient each layout axis at fit time so node 0's coordinate is
non-negative (lift `orient_to` to the layout), and normalize rank-1 layouts to
±1 (or divide the gate channel by node-0's layout coord at attach) so the
documented semantics become true. Migration note: coord-form `%` positions are
in layout units; sign orientation is the correctness-critical half.

### P2. `@label`/`~label` channels exist only in top-N — `<`-gates can't fire — high [direct]

`flat_scalars` (`monitor.py:1260–1263`) emits `@label`/`~label` only for the
top-N nearest (default 3, `monitor.py:23`); `ProbeGate` lookup of an absent
key reports inactive, never raises (`triggers.py:155–159`). So
`@when:personas@hacker < -2` ("steer when *far* from hacker") goes inactive
precisely when far from hacker (out of top-3 of 107) — never fires in its
intended region. Same for `~label <` gates and the reserved `@neutral`.
Fix: at gate install, force-emit the gated label's channel regardless of rank
(the batched path already computes all `(P, Kmax)` distances — one extra named
column is free), or refuse/warn on `<`-gates targeting rank-suppressed
channels.

### P3. Whitener ridge λ is rogue-dominated; over-shrinks all typical directions — med [auditor-numeric]

`mahalanobis.py:210–223`: `λ = (‖X‖²_F/(N·D)) · ridge_scale` = mean
per-channel variance = trace/D. On real LMs trace is dominated by the
massive-activation channels, so λ ≫ typical-direction variance and
`Σ_reg ≈ λI` on the non-rogue subspace: rogue dims are correctly suppressed,
but there is **no relative whitening among non-rogue dims** — which is where
the Fisher fit's claimed benefit lives. Synthetic demo: one rogue channel
(σ=100) in D=256 → λ ≈ 41 crushed a true whitened internode separation of
3.16 to 0.49 (6.4× suppression). With N=96 ≪ D the empirical Σ is rank ≤ 95,
so >98% of directions see only the ridge. Everything downstream — Fisher
bases, shares, consensus Grams (hence discover layouts), monitor coords,
`@label` distances, gate scales — is a function of this λ.

Fix: shrink toward `diag(Σ̂)` instead of `(trace/D)·I` — a diagonal Λ keeps
the Woodbury form at the same O(ND) cost and actually whitens typical dims; or
at minimum λ from the **median** per-channel variance; or Ledoit-Wolf
intensity (closed-form). Pair with a denser neutral corpus (E4).
Interaction: P11 below — the off-span M-leakage bias is currently negligible
*because* the heavy ridge makes M-proj ≈ Euclidean-proj. Fixing the ridge
un-suppresses that bias; re-measure P11's experiment after.

### P4. EV layer-weighting vacuous at K=2; weak layers contribute amplified noise at full weight — med [direct]

Cross-layer aggregation weights by normalized `explained_variance`
(`monitor.py:2020–2031`); the whitened ev_ratio at K=2 is identically 1.0
(one nonzero eigenvalue of a (2,2) Gram — confirmed empirically), so all 17
bundled concept probes aggregate as an **unweighted mean over all fit
layers**. Worse: the domain-frame map's slope is layout_scale /
per-layer-separation, so a layer's mapped read noise ∝ 1/`mahalanobis_share_L`
— the least-signal layers contribute the most noise, undamped (DLS only drops
non-straddling layers). Fix: weight by inverse read variance ∝ share², already
baked per layer in every artifact; one-line change in
`_attach_manifold_probe`. Experiment: per-layer coord std over the cached
neutral stack, aggregate std under EV vs share² weights.

### P5. Aggregate pooling reads one position before the extraction frame — med [direct]

`capture[k]` is the state that *produced* token k (`hooks.py:27–38`, i.e.
post-consumption of token k−1). `score`'s aggregate pools
`rows[last_content_index(generated_ids)]` (`session.py:3046–3047`) — the
state *before* the last content token entered the context. Extraction pools
the state *at* the last content token (after consumption). The
extraction-frame row is `rows[agg_idx+1]`, which exists whenever a terminal
EOS forward ran. Live and aggregate paths agree with each other (the
"bit-identical" claim is intra-path and true) but both sit one token off the
frame the probes were fit in — worst on short responses (the read is then
nearly the end-of-prompt state). Fix: pool `min(agg_idx+1, n_forwards−1)`.
Experiment: score both indices over a corpus of gens; expect material deltas
on short responses.

### P6. `label_scale` for curved+σ fits is tube thickness, not label spacing — med [direct quote]

`monitor.py:2115–2122`: with a σ-field the per-node bandwidths are within-node
thickness, and `label_scale = median(bandwidths)` (`:2047–2059`) — typically
≪ NN spacing, inflating reported `@label` distances by spacing/thickness. The
documented contract ("median node nearest-neighbor whitened distance",
threshold portability across probes) holds only for flat/no-σ fits. Fix:
compute `label_scale` from NN spacings of `node_white` unconditionally (the
spacing code at `:2124–2130` already exists); keep σ for the assignment τ.

### P7. `:membership` ≡ 1.0 on every flat probe, emitted as a gate channel — low-med [direct quote]

`monitor.py:395, 1110, 1264`. AGENTS.md's own example
(`@when:emotions:membership > 0.6`) is constant-true on gemma-4-12b where
`emotions` resolves flat. Fix: omit the channel for affine probes (missing key
⇒ gate inactive — the safe default) or warn at gate install.

### P8. `~label` calibration K-dependent; arbitrary for K=1 rays — low-med [direct quote]

`monitor.py:2099–2161`: a `> 0.5` threshold means very different things over
3 candidates vs 108; K=1 rays get τ = 1.0 whitened-unit with no relation to
any layout scale (and the docstring claims K=1 returns `(None, None)` while
the guard is `K < 1`). Document per-probe baselines or report posterior odds
vs uniform.

### P9. Batched fraction has a catastrophic-cancellation path — low

`monitor.py:967` computes `x_m2` by differencing large fp32 quantities where
the per-probe path uses the centered form (`:2192–2194`). Degrades exactly
where fraction is most sensitive (h ≈ mean); rare in practice (rogue mass).

### P10. Curved readings differ across capture modes — low

Warm-started (incremental) vs cold (aggregate-only/full) foot solves can give
different curved coords at the same token (test tolerance 2e-2 acknowledges
it). Wrong-basin sticking is adequately mitigated by the nearest-node safety
restart. Document the cross-mode tolerance; no code change needed.

### P11. Read-at-node off-span M-leakage — low, currently negligible [auditor-numeric]

The affine coord map can't absorb off-span node residual + neutral component
leaking through Σ⁻¹ cross-coupling; measured tiny (coord −0.004 at neutral)
*because* the heavy ridge (P3) suppresses it. Re-measure if P3 lands; add the
assertion test (score the cached neutral stack; mean coord ≈ 0, `@neutral` ≈ 0).

### P12. Tail-ring depth-8 clamp silently pools the wrong token past 8 trailing specials — low

`session.py:169`; only a risk for channel-heavy formats (gpt-oss). Emit a
one-time warning when the clamp engages.

### P13. Gate-threshold ergonomics summary

`:fraction` ∈ [0,1]: stable, good. `@label` in label-spacings: good design,
undermined by P2 (presence) and P6 (units). Bare `<probe>`/`<probe>[i]`: the
worst channel — P1's arbitrary sign + fit-dependent scale despite being the
documented primary gate. `~label`: K-dependent baseline (P8); `:membership`:
vacuous on flat (P7).

---

## 4. Extraction quality

### E1. `--no-dls` is a dead flag — high [direct]

`session.py:950` stores `self._dls = bool(dls)`; that is the only read-free
occurrence in the package (grep-verified — nothing consults it).
`ManifoldExtractionPipeline.fit` calls `compute_dls_axes` unconditionally
(`extraction.py:754–758`). CLI `--no-dls` (`cli/runners.py:152`) flows into a
stored-but-never-read attribute. Both docs describe an opt-out that doesn't
exist. Fix: thread `dls` through to the pipeline and skip/keep-all in the flat
branch; test that all axes survive with `dls=False`. (Required for the E5
ablation experiment anyway.)

### E2. Staleness chain excludes two fit inputs — high [direct on mechanism]

`ManifoldFolder.nodes_sha256` (`io/manifolds.py:791–855`) hashes corpora +
fit_mode/hyperparams + roles/kinds + template hash, and nothing else. Two
silent-wrong shapes: (a) a user `baseline_prompts.json` override with the same
k=48 pools existing corpora against mismatched prompts with no error (the
multiple-of-k check can't catch same-length swaps), and cache hits keep
serving tensors whose elicitation inputs changed; (b) the neutral corpus
correctly invalidates the whitener and `layer_means` on its own hash, but
**no fitted manifold invalidates** (`extraction.py:241–245` checks only
nodes_sha + sae_revision) — after a neutral regen, every fit's Fisher basis,
share, DLS keep-set, and discover coords are stale relative to the metric the
monitor now reads with. Fix: fold the resolved-prompts hash and the
neutral-cache hash into the sidecar comparison (~20 lines + tests); at minimum
document `pack clear` after neutral regen.

### E3. Per-node power is 48 single-token samples and `extract()` can't raise it — med-high [direct]

`extract()` has no `samples_per_prompt` in its signature; `manifold generate`
and `regenerate_bundled.py` default to 1; one pooled token per response. Each
centroid is a 48-sample mean in D=2560+; most exposed: the 2-node δ̂, authored
curved fits (which **exact-interpolate** the noisy centroids by contract —
`smoothing=None` hardwired for authored, `extraction.py:507`,
`manifold.py:1484` — chasing exactly the noise discover-spectral's GCV was
added to suppress), and `select_topology` reading geometry off K≈20 noisy
points (faint-cycle contrast threshold 1.08 is within centroid-noise reach).
Fixes, cheap → expensive: expose `samples_per_prompt` on `extract()`/CLI;
bump bundled defaults (1 → 2–4; cost is linear and one-time); allow
`--smoothing` for authored fits. The deciding experiment: split-half per-layer
whitened cosine of δ̂ for the same concept extracted under two seeds — one
number tells you whether 48 is enough.

### E4. The whitener is rank-limited: N=96 ≪ D — med-high [direct on mechanism]

Σ built from the 96-row neutral stack; rank ≤ 95, so all whitened quantities
are genuinely Fisher only inside the neutral-variation span and ridge-scaled
Euclidean outside it. Rogue cancellation still works (rogue dims are reliably
in-span) — which is why whitening verifiably helped — but the "signal-weighted
consensus" claim and share calibration inherit ridge-Euclidean behavior
off-span. Fix: `--neutral-samples 8–10` (~770–960 rows); record N and
per-layer effective rank in whitener provenance. Pairs with P3 (shrinkage
target). Stale docstring says N=90 (`mahalanobis.py:26–27, 38`).

### E5. DLS straddle has no margin; ~half of pure-noise layers pass — med [direct quote]

`vectors.py:591`: `projs.min() < 0.0 and projs.max() > 0.0` — a −1e-6
projection keeps the axis, and for a no-signal layer the sign is coin-flip
(~50% pass rate). Survivors get `share ∝ ‖δ_noise‖_M` — a uniform nonzero
floor the mean-1 normalization then feeds at the expense of signal layers.
Fix: (a) require `|projs| > z·SE` (within-node SE is nearly free to collect
during centroid pooling — the σ-field pass already computes within-node second
moments for curved fits); (b) share shrinkage
`share² ← max(0, ‖δ‖²_M − 2·tr(M_R)/n_samples)`. Directly sharpens steering
at fixed α.

### E6. Last-token-only pooling; masked-mean is one small step away — med [analytic]

Pooling keeps 1 of ~50–250 response tokens. `_capture_all_hidden_states`
already does per-row gathers in-hook (`vectors.py:72–97`) and the render knows
where the assistant turn starts; a masked mean over
`[response_start, content_end]` is a `(B,T)` mask matmul with the same `(B,D)`
retention. Plausible 5–10× effective-sample gain per response even with token
autocorrelation. Last-token is the attention-weighted summary by design, so
this changes what's extracted: run as an A/B (split-half direction stability +
steering coherence at matched α), not a silent swap.

### E7. Degenerate responses pool the prompt's last token; filtering inconsistent — med [direct on mechanism]

An empty/whitespace response makes `last_content_index` walk back through the
turn markers into the **user prompt's** last content token — prompt-state
contaminates the node centroid. `generate_responses` never filters/retries
empties (`session.py:1503`); `_validate_discover_corpora` rejects blanks at
authoring (`io/manifolds.py:1240` — so an empty generation crashes `extract()`
*after* the full corpus run, no retry), but fit-time `node_groups()` only
checks `isinstance(s, str)` (`:783`) — hand-edited/merged folders with blanks
pool silently. No refusal-pattern filter either. Fix: bounded regenerate-on-
empty in `generate_responses`; enforce non-blank in `node_groups`; optionally
log a per-node refusal-count advisory.

### E8. SAE fits mix spaces in two places — med [direct quote]

(a) DLS gets SAE-reconstructed centroids but raw-space ν
(`extraction.py:451–462` vs `:754–758`) — the sign test can flip on
reconstruction bias, not concept polarity. (b) Monopolar: `c` reconstructed,
`nu` raw (`:371–380`) — `c − nu` carries SAE reconstruction error as if it
were concept signal. Fix: reconstruct ν through the SAE at both sites (or
document the asymmetry as intentional).

### E9. Restricted-choice scorer: per-candidate cut bias + degenerate-candidate pathology — med [direct]

`scoring.py:139–144`: each candidate's scored span starts at *its own*
divergence from `prefix_ids`. A candidate whose tokenization merges the prefix
tail (`" Mon"+"day"` class) is charged `log P` of prefix-tail text the
unmerged candidates aren't — the restricted-choice softmax compares quantities
with different conditioning baselines. Fix: one set-wide `cut = min(cuts)` so
every candidate is scored from the same conditioning point. Second
(`:181–184`): a candidate adding no distinct token records `sum_logprob = 0.0`,
which dominates the softmax (everything else is negative) → probability ≈ 1
for a whitespace-absorbed candidate; should be excluded/−inf with a warning.
Third: `score_inputs()` emits a `suffix` field nothing consumes. Coverage:
**no test exercises `score_choices`** (only a mocked return). Verified
positive: the raw-distribution claim holds (plain fp32 `log_softmax`, no
sampler knobs) and steering does apply during the scoring forward — though
`@response`/probe-gated terms are silently inactive there (all-prefill);
worth a docstring note.

### E10. `_role-<slug>` tensor variant is never written by any fit — med [direct]

The pipeline always writes the raw filename
(`extraction.py:222`: `tensor_filename(model_id, release=sae_release)` — no
call site in the package passes `role=`). So `extract(role="x")` writes a
role-baselined fit into the **raw** slot of the canonical folder
(indistinguishable from and clobbering a standard fit), and the returned
`"<name>:role-<slug>"` selector can't load (`_ensure_manifold_loaded` rejects
the variant, `session.py:2183–2189`). Docs describe the variant as a working
surface. Fix: pass `role=` into `tensor_filename` in the pipeline and add
`role-*` to the load dispatch, or delete the variant from docs/grammar.

### E11. Faint-ring fallback can false-positive on a near-closed noisy arc — med-low [direct on thresholds]

Guards (`manifold.py:3231–3236, 3275–3354`): closure passes when the endpoint
gap < 2× median spacing; 2-NN recall ≥ 0.90 tolerates the two endpoint misses
for K ≥ 10; contrast ≥ 1.08 passes any smooth arc. A detected ring wins
outright over GCV (`:3554–3557`, deliberate). Tests cover arcs only to 300° at
K ≤ 12 — a noisy 330–350° horseshoe is untested and would be forced periodic.
Fix: add near-closed-arc cases; consider requiring the spectral fundamental
eigenpair to wind monotonically with the tour order as a fifth guard.

### E12. RBF saddle solve and poisedness have no conditioning guard — low

`torch.linalg.solve` on the symmetric-indefinite saddle with no condition
estimate; poisedness via `matrix_rank` at default tolerance
(`manifold.py:654, 693`) — a nearly-coplanar layout passes rank and yields
garbage weights silently. Cheap: warn when `cond(M)` > ~1e8 (K ≤ ~130, cheap
to compute), suggest a smoothing floor.

### E13. `_pca_basis` rank-0 path returns an arbitrary direction — low

`manifold.py:1238–1239`: `R = max(1, …)` forces R=1 at rank 0 (identical
centroids); QR of a near-zero column yields a numerically arbitrary unit
vector, and DLS may keep it by chance (E5). Return an empty basis / drop the
layer.

### E14. Consensus Gram has no robustness to a pathological layer — low

Plain mean over layers (`extraction.py:527–529, 625–627`); the drop-out
argument covers weak layers, not inflated ones. Cheap: warn when one layer
contributes > X% of `tr(Ḡ)`; consider a trimmed mean behind a hyperparam.

### E15. Pooling edge cases — low

`special_token_ids` unions `added_tokens_encoder` (`vectors.py:142–145`) —
added *content* vocabulary gets wrongly skipped at walkback. Base-model branch
tokenizes with `add_special_tokens=False` (no BOS — mildly OOD for BOS-trained
base models). All-special sequences floor to index 0 silently. Right-padding
exactness itself is correct (mask built and passed; right-aligned positions
valid; pooled position attends only real tokens) — but no GPU parity test pins
padded == unpadded capture; add one to the GPU suite.

---

## 5. Text-generation performance

### G1. `logits_to_keep` never passed: full-vocab logits for every prompt position — high [direct]

Zero occurrences of `logits_to_keep` in the package. Prefill forwards
(`generation.py:991–1005`) compute lm_head over the whole prompt and keep only
the last row; `session.py:3952` (`cache_prefix`) discards the logits entirely;
extraction capture (`vectors.py:111–119`) never reads them. Cost: for a
2k-token prompt on a 256k-vocab model, ~2.7 TFLOP of wasted matmul (~15–20% of
a 4B model's prefill) plus a `[1, T, V]` transient (~1 GB bf16); extraction
capture batches pay `[16, T, V]` on MPS unified memory; the no-KV-cache
fallback (talkie) pays it **every decode step**. Fix: pass `logits_to_keep=1`
(signature-gated once per model; HF's own `generate` does this).
`Gemma3ForCausalLM.forward` accepts it on the installed transformers (5.10.1).

### G2. Curved transport does a CPU SVD round-trip per layer per token on MPS — high (curved only) [direct on mechanism]

`_svd_mps_safe` (`manifold.py:2159–2172`) is an explicit
`a.cpu() → svd → .to(device)`, called from `_frame_rotation_transport`
unconditionally on every curved fire. A curved fit covering ~30 layers ⇒ ~30
pipeline stalls per token (each a Metal command-buffer flush) — this, not the
documented foot solve, dominates curved decode on MPS. The matrices are tiny
(n ≤ 3). Fix: closed-form/device-side small-n SVD (n=1 trivial, n=2 closed
form, n=3 Jacobi sweep). Also profile whether `torch.linalg.solve` in
`_gn_step` (`manifold.py:4190`) stays on-device. Violates the documented
"no CPU sync" hook invariant. (Note: at present every curved fire is also
mis-gained per S3 — fix S3 first or the profiling target is unrepresentative.)

### G3. Sampler pays two avoidable data-dependent syncs per token — med-high [direct]

`generation.py:397–399`: `valid = row_probs > 0;
top_idx[0][valid], row_probs[valid]` — boolean-mask indexing forces a
device→host sync to size the output, twice, on every sampled token, steered or
not. With the mandatory `int(next_token.item())` (`:1131`) and the optional
entropy `.item()` (`:1126`), a loom-attached gen pays ~3–4 syncs/token where
1–2 would do. Fix: don't filter zeros — `torch.multinomial` never draws
zero-weight entries, the first entry is clamped positive, and the entropy sum
is unaffected; filter only in `_sampler_logprob_vector` (off-hot-path). Bonus:
fixed-shape candidate tensors (≤1024) help any future graph capture.

### G4. Incremental monitor scoring: a forced sync inside the forward + heavy reading retention — med [direct on mechanism]

The step sink fires inside the max-probe-layer forward hook and ends in
`.cpu().tolist()` (`monitor.py:1038–1043`) — the host stalls mid-forward, then
queues the remaining layers + lm_head, serializing scoring with the model
forward every token. Per-token device work ~300 kernel launches (Woodbury +
block-diag matmuls + fresh `torch.zeros` accumulators, `monitor.py:938–941`);
`_incremental_readings` retains full `ProbeReading`s including the
`*_per_layer` dicts (19 probes × ~90 entries/token — O(T) Python-object growth
reaching hundreds of MB over long gens; gates only ever consume
`flat_scalars`, which never reads per-layer traces). Three cheap wins:
preallocate the accumulators; strip `*_per_layer` from retained rows unless
the inspector needs them; move the host transfer out of the forward hook to
just before the next gate check (`generation.py:1016–1017`) so the lm_head
overlaps the scoring matmuls.

### G5. Both GPU perf gates are mis-aimed; one is dead — med [direct]

(a) `test_throughput_regression` (`tests/test_smoke.py:316–361`) steers one
always-active merged affine subspace — exactly the `_single_affine_fast` path
— with **no monitor attached** despite the "Steered + monitored timing"
comment. The 85% gate never exercises the general hook path, gated/phased
triggers, curved manifolds, or incremental capture+scoring; vanilla also runs
first, handing the steered run warm caches. (b) `TestAblationPerformance`
(`:391–423`) constructs the session with `probes=["affect"]` — no bundled
manifold has carried the `affect` tag since the 4.0 regen (tags are
epistemic/alignment/register/cultural) — so `assert len(probes) >= 2` fails
structurally on any GPU run. Fix: (b) use a live tag; (a) add a second timed
leg with probes + a gate attached, and alternate run order.

### G6. Compiled steered generation recompiles per α; stale comment — med (CUDA opt-in) [analytic]

`hooks.py:359–363` stores `float(along)`/`float(onto)`; under `torch.compile`
hook-state floats specialize as constants → an α change between generations
(every `generate_sweep` step) is a guard miss/recompile, and under
reduce-overhead a CUDA-graph recapture. `model.py:803–809` still documents the
old 0-dim-tensor pinning mechanism whose method no longer exists. Fix: carry
along/onto/kappa as 0-dim device tensors in `_single_affine_fast`
(`subspace_inject` already types them `float | Tensor`). Also: in-place logit
mutations (`generation.py:1057–1059, 521, 1072`) on graph-owned buffers are
the classic cudagraph hazard — consumption order looks safe but is untested;
AGENTS.md itself requests the CUDA validation pass.

### G7. Aggregate-tail ring allocates per token — med-low [direct on mechanism]

`hooks.py:127–135`: `bucket_ref.append(src.clone())` + `list.pop(0)` per layer
per forward — a fresh `(D,)` device clone + list churn every token in the mode
sold as "zero per-token cost". The incremental depth-1 path got the
preallocated-`copy_` treatment; the ring didn't. Fix: preallocate `(depth, D)`
per layer with a rolling index.

### G8. Full-retention finalize re-scores token-by-token, twice in the worst case — low-med [direct on mechanism]

`_per_token_coord_stream` (`monitor.py:1267–1288`) runs the full `_score_full`
(including topk-nearest + soft assignment + a `.cpu().tolist()`) per token and
keeps only `coords[0]` — O(T) syncs at finalize. A `return_hidden` +
loom-attached gen scores per token twice (live tap + finalize). Fix: batch the
flat path over T (`_woodbury_apply` already accepts `[n, D]`), one transfer.

### G9. Curved probes recompute the Woodbury apply flat probes already did — low-med [direct on mechanism]

`_layer_geometry` inside `_score_probe_full` (`monitor.py:2191`) recomputes
`Σ⁻¹h` per curved probe per layer; `_score_flat_batched` (`:955`) computed the
same product for the same layer moments in the same `_score_full` call. Fix:
cache `sih` per (layer, hidden-id) within a call.

### G10. Per-generation StaticCache allocation; no reuse — low

`generation.py:808–829` builds a full-size StaticCache per generation under
`cuda_graphs=True` (GBs at 12B/long context). Pool/reuse when capacity
suffices. Note `compile`/`cuda_graphs` both default False — out of the box no
CUDA run uses either; standing missed opportunity per docs, not a bug.

### G11. `generate_batch` is sequential — low (structural)

`session.py:5345–5360` loops `_generate_core` row-by-row. True B>1 decode is
blocked by `HiddenCapture` hardcoding `h[0, -1, :]` (`hooks.py:109`), the
`[1, V]` sampler, and single-stream penalty/stop/thinking machines. The
prefix-KV share recovers prefill cost only. Big win for sweep/eval workloads;
large change.

### G12. Minor per-token churn — low

`generation.py:1072` casts `bias_val.to(logits.dtype)` per token when
`logit_bias` is set (pre-cast once); `monitor.py:715` rebuilds
`tuple(self._probes.keys())` per scoring call; `_PenaltyState.add`
(`generation.py:531–533`) does a tiny H2D per new unique token
(penalties only).

**Worst-common-case per-token chain (loom-attached, probes on, merged-affine
steering, MPS):** steered forward (fast-path inject, ~10 kernels) →
max-probe-layer hook: capture `copy_` + score sink (~300 kernels, **1 forced
sync**, ~100–200 µs Python reading build) → remaining layers + lm_head →
`nan_to_num_`/`clamp_` → topk/top-p → **2 masked-select syncs** → multinomial
→ **`.item()` sync** → **entropy `.item()` sync** → decode via cached table →
tap → next. ≈5 host syncs/token against a theoretical 1.

---

## 6. Doc–code drift (collected)

- AGENTS.md (root, "Performance invariants"): `_MANIFOLD_GAIN = 1.0` — code
  says 0.5 (`hooks.py:538`); `saklas/core/AGENTS.md` has the correct value.
- AGENTS.md + ARCHITECTURE §3.6: `--no-dls` opts out — dead flag (E1).
- AGENTS.md grammar: `!x` ⇒ `h' = h − α(h·d̂ − μ·d̂)d̂`, bare α=1 — code uses
  α ≈ 16·share and drops the coeff (S1, S2).
- `hooks.py:765–767` (`add_manifold` docstring): along "clamped to [0,1] …
  slides the foot geodesically" — true of the user knob, false of the applied
  value (S3).
- ARCHITECTURE §5.3: onto gain "calibrated on the `emotions%dominant` sweep" —
  that calibration predates the along-gain bump; its operating point is gone
  (S3).
- ARCHITECTURE §5.1 / `manifold.py:2190–2198`: transport "exactly the
  identity … norm-preserving and lossless" — approximate; fails rank-deficient
  frames (S8).
- ARCHITECTURE §4: "summed ablation-row whitened magnitude" vs
  magnitude-of-sum (S9).
- `manifold.py:1953–1957`: "never mixing the two metrics across layers" vs the
  per-layer degenerate fallback (S6).
- ARCHITECTURE §6.1 / `monitor.py:92–96`: coords "pole-normalized, 1.0 at the
  positive node" — discover layouts are sign- and scale-arbitrary (P1).
- io/AGENTS.md + root AGENTS.md: `_role-<slug>` tensor variant — never written
  by any fit (E10).
- `CustomDomain` stores and round-trips `bounds` nothing enforces
  (`manifold.py:547–555, 577–578`).
- `mahalanobis.py:26–27, 38`: "N=90" — actual neutral corpus is 96; docs
  reference `from_neutral_cache` — actual name `from_cache` (`:255`).
- AGENTS.md "Monitor capture … only the latest per-layer slice + ProbeReading
  rows retained" — the rows include full `*_per_layer` traces; the framing
  undersells O(T) growth (G4).
- `tests/test_smoke.py:338` "Steered + monitored timing" — no monitor attached
  (G5).
- `monitor.py:52`: "Shared by Monitor and Monitor".

---

## 7. Verified clean

Worth as much as the findings — these were checked, not assumed:

**Kernel and composition.** The `h = mean + h_par + h_perp` decomposition;
the affine analytic shortcut; H_o kept verbatim under all ops; along→onto
order; fp32 throughout; soft cap as the only norm guard. Translate-not-
collapse preserves per-token spread (push offsets are q-independent; verified
additive composition across groups). σ-field onto: σ=0 reproduces `(1−o)`
exactly, never expands inside the tube. `_frame_rotation_transport` on
well-conditioned frames: identity at rest (~1e-5), norm-preserving under real
rotations. Single-term whitened-unit calibration algebra checks out
(per-layer displacement `GAIN·α·wd_L/mean(wn)`, linear in α,
target-scale-independent). κ-mask derivation order-robust; `_ortho_basis`
scale-free, ordered push-first; mean-1 normalization with all-zero guard;
opposing-term cancellation degrades to a benign no-op. Orthogonality
mechanics (curved-curved 1e-3 check, affine-vs-curved strip of basis and
displacement, full-containment drop). Trigger semantics (gated inactive in
prefill, `when` ⇒ prompt=False, missing probe ⇒ inactive; scores written
post-forward for next-step gates). `_single_affine_fast` ≡ general path for
its shape.

**Monitor and whitener internals.** `_layer_geometry`: `c = M_R⁻¹BΣ⁻¹x` is
exactly the M-orthogonal projection coefficient; fraction genuinely ∈ [0,1];
the batched path reproduces the per-probe path. Woodbury exact for the ridge
estimator; SPD Cholesky with jitter fallback; fp32 end-to-end on the read
path; non-finite exclusion makes `covers_all` trustworthy; legacy non-fp32
cache refused. Neutral pseudo-candidate mechanics correct and reads ≈0 at the
neutral mean. `label_scale` as a uniform rescale preserves `nearest` ranking.
Incremental row alignment (prefill row = token-0 reading; terminal-EOS trim)
consistent across modes at the same index. Read frame ≡ steering write frame
(same fitted Manifold handed to both; steering displacements register 1:1 in
gate coords).

**Extraction.** PCA@2 ≡ DiM exactly; the whitened K=2 fit is the Σ⁻¹δ
discriminant via `G = XΣ⁻¹Xᵀ` → QR (span-preserving); `orient_to=0` applied
before node_coords; ev_ratio is the whitened retained fraction. Right-padded
batched capture semantics (mask passed; per-row in-hook gather; positions
valid). One shared `last_content_index` definition; `response[i] ↔
prompt[i%k]` alignment enforced at every pooling site. The `_LENGTH_DIRECTIVE`
common-mode design is genuinely symmetric across all four sites. Monopolar
fold (non-SAE) correct, hard errors when ν/whitener missing.
`nodes_sha256` sensitivity to everything it covers is implemented and tested.
GCV machinery correct (formula, λ=0 exclusion, edf ≥ K → inf). Discover
coords: node-mean centering, eigh symmetrization, disconnected-graph error,
harmonic dedup, PH triangle cap. DLS N=2 parity and degenerate-row exclusion.
Scorer's raw-distribution claim; steering active during scoring.

**Generation.** Top-p via topk with the 1024 cap, no full-vocab sort.
`inference_mode` coverage complete on the generation path. Flat roster scored
in one batched sweep with one host transfer (better than the docs' phrasing).
The three capture modes select exactly as documented; aggregate-only really
does zero per-token scoring; gate callbacks reuse the step-sink reading.
Steering hooks transient with clean detach; prefix-KV share exact-match
guarded and steering-aware. Stop-sequence matching bounded; EOS/token-text/
think-delimiter caches all real. MPS discipline (end-of-loop sync, per-chunk
empty_cache) as documented.

---

## 8. Test-coverage gaps

The recurring shape: the math has good unit coverage in the Euclidean/toy
regime; the **production regime** (whitened, N ≪ D, pipeline-fitted artifacts,
production gains) is CI-invisible.

- The whitened branch of `synthesize_subspace` has zero tests (the suite
  passes no whitener) — the whitened-unit target, whitened share, S6 fallback
  are all untested.
- No test runs the Monitor over a **pipeline-fitted** manifold (hand-built
  `node_coords = ±1` everywhere — which is exactly what masked P1). The test
  whitener is N ≥ D and isotropic; the production N ≪ D rogue regime is
  untested.
- Ablation through `apply_to_model` at production eff_along: plumbing-only
  coverage (would have caught S1).
- `_orthogonalize_affine_against` (κ carry, target strip, layer drop): zero
  direct tests. `_frame_rotation_transport` degenerate frames: untested.
- Curved `%` at production gain; periodic wrap under eff_along > 1;
  `translate_foot` as such: untested.
- No gate test for `<`-gates against rank-suppressed channels (P2); no
  ridge/N-sensitivity test (P3/E4); no live-vs-extraction frame test (P5,
  existing tests assert the current index as correct); no curved+σ
  `label_scale` unit test (P6).
- `score_choices`: no functional test (E9). No GPU parity test for padded vs
  unpadded capture (E15). Topology tests stop at 300° arcs (E11).
- Norm-cap fire-rate instrumentation: absent everywhere (S11).
- Both GPU perf gates mis-aimed/dead (G5).

---

## 9. Recommended sequence

**P0 — correctness (small diffs, silent-corruption class):**
1. S1+S2: decouple κ collapse from slide gain; carry the ablation coeff.
2. S3: separate/normalized curved gain; interim [0,1] clamp; re-run onto sweep.
3. P1: orient + pole-normalize discover layouts (one localized change at fit).
4. P2: force-emit gated label channels (or refuse `<`-gates on suppressed
   channels).
5. E1 wire `--no-dls`; E10 wire or delete the role variant; G5 fix both perf
   gates. All three are "the surface claims something the engine doesn't do".

**P1 — quality floor (config + ~3 localized changes, then measure):**
6. E2: fold prompts-hash + neutral-hash into staleness.
7. P3+E4: diagonal shrinkage target for λ; `--neutral-samples 8–10`; record
   effective rank. Re-measure P11 after.
8. E3: expose `samples_per_prompt` on `extract()`; bump bundled defaults; run
   the split-half δ̂-cosine experiment to set defaults from data.
9. P4: share²-weighted layer aggregation. E5: margin-aware DLS + share
   shrinkage (measure with the now-working `--no-dls`).
10. P5 aggregate index; P6 unconditional spacing-based `label_scale`; P7/P8
    channel hygiene. S4/S5: per-fragment share composition + vector-path real
    coords.

**P2 — performance (independent of the above except G2→S3):**
11. G1 `logits_to_keep=1` everywhere (biggest single perf win; also helps
    extraction memory).
12. G3 sampler de-sync; G4 monitor incremental cluster (prealloc, strip
    per-layer traces, move the transfer); G7 ring prealloc; G8/G9 finalize
    batching + Woodbury cache.
13. G2 device-side small-n SVD (after S3 so profiling is representative).
    G6 tensor-pinned α + the CUDA validation pass.

**P3 — durable:** the §8 test additions (priority: whitened synthesize suite,
pipeline-fitted monitor suite, production-gain ablation/curved e2e, frame
test), the §6 doc sweep, E6 masked-mean A/B, E11 arc cases, S12 experiment (b)
for the persona fan.

The through-line: the engine's core math is sound and the documented design
is mostly real in code. What this audit found is concentrated where the
2026-06 recalibration outran its blast radius (S1–S3), where conventions on
top of correct math were never pinned (P1, P2, P6), and where the statistical
floor was set by defaults nobody re-derived (E3–E5, P3–P4). All three clusters
are cheap relative to the engine they sit on.
