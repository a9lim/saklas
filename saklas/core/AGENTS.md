# core/

Engine layer: model loading, vector extraction, the unified subspace/manifold
fit + injection, trait/manifold monitoring, session orchestration, the
generation loop, and the loom conversation tree. The cross-cutting design lives
in the repo-root `ARCHITECTURE.md`; this file is the per-module map.

The 4.0 backend lowers **every** steering term — vectors, poles, `~`/`|`
projections, `!` ablations, and affine/curved `%` — to one per-layer
`inject_three_op` (along/onto) call. There is no angular/additive mode, no
`injection_mode`/`theta_max`, no `_STEER_GAIN`, and no per-fit lever.

## model.py

HF causal LM loading. `_LAYER_ACCESSORS` maps `model_type` → layer-list accessor
(`def`s, not lambdas). `_TESTED_ARCHS` gates a one-time `UserWarning` when
`model_type` isn't known-working. Cascading fallbacks on attention impl (SDPA →
eager), dtype, device. `_compile_with_probe` wraps `torch.compile` with a 2-token
prefill+decode warmup so inductor/Triton failures surface at load as a caught
warning + eager fallback (`use_static_cuda_launcher` forced off — Gemma-4 + torch
2.12 kernel-arg-mismatch bug). `_load_text_from_multimodal` extracts text-only
sub-models (Ministral-as-Mistral3), strips `language_model.` prefixes,
dequantizes FP8. `patch_torch_for_mps()` installs two lazy MPS-only workarounds
(`torch.histc` integer→float for MoE routing; `torch.ldexp` MXFP4 round-trip
through CPU honoring `out=`).

## vectors.py

Low-level extraction primitives. One forward pass per prompt;
`_capture_all_hidden_states` hooks every layer at once. `_encode_and_capture_all`
pools from the **last content token** — `last_content_index` walks back past both
`tokenizer.all_special_ids` and `tokenizer.added_tokens_encoder` values to skip
trailing chat-template markers (where outlier channels dominate). `special_token_ids`
+ `last_content_index` are the one canonical "last non-special token" definition
shared by every single-state readout (extraction, vector aggregate, manifold
aggregate); `encode_and_capture_stack` is the full-`[T,D]` companion for the
manifold monitor. `_render_and_tokenize_for_capture(..., role=, model_type=)` is
the shared render+tokenize+walkback front half. Contrastive diffs are differenced
in **fp32**.

`extract_difference_of_means` is the **only** vector extractor (contrastive PCA
is gone). `_capture_diffs_for_pairs` runs the forward loop and returns diffs +
per-layer pos/neg means + SAE layer set; the direction is `mean(diffs)` fp32.
`_share_bake_and_warn` applies the DLS keep set, share-bakes retained layers
(`stored_L = d̂_L × ref_norm_L × score_L / Σ score_L`), and emits the diagnostics
warning. Bake **score** = `‖direction‖_M / ref_norm` when a `whitener` covers
every scored layer (Mahalanobis), else `‖·‖₂ / ref_norm` (Euclidean) — all-or-
nothing via `covers_all`. SAE: the extractor runs in feature space when passed a
`SaeBackend` (encode → direction → decode), and the SAE branch ignores the
whitener. Returns `(profile, diagnostics)`.

DLS (Selective Steering, Dang & Ngo 2026 Eq. 9): `compute_dls_axes(node_centroids,
bases, layer_means)` is the **N-node straddle** core — keep axis `d̂ᵣ` at layer L
iff `{(cᵢ − ν)·d̂ᵣ}` straddles zero. `compute_dls_mask_per_axis` (bipolar sugar,
the per-axis pos/neg form) and `compute_dls_mask` (scalar, the legacy
opposite-sign product test) are thin wrappers, bit-identical at R=1. The subspace
apply path consumes the per-axis keep set by slicing the basis
(`LayerSubspace.select_axes`); an all-fail layer drops.

`_compute_layer_diagnostics` returns `evr`, `intra_pair_variance_{mean,std}`,
`inter_pair_alignment`, `diff_principal_projection`, persisted through
`save_profile` → `Profile.diagnostics` and surfaced by `vector why`.
`compute_layer_means` / `compute_neutral_activations` build the probe-centering
baseline + the whitener's neutral cache. `project_profile(base, onto, operator,
*, whitener=None)` is the per-layer `~`/`|` projection (LEACE under a whitener,
Gram-Schmidt otherwise; metric all-or-nothing via `covers_all`).

**Subspace-native fold (built, tested, not yet the production vector path).**
`fold_vector_to_subspace` / `_fold_centroids_to_affine_manifold` capture pos/neg
centroids and fold them into an affine `R=1` `Manifold`; `fold_directions_to_subspace`
is the monopolar sibling (a merge/projection direction → a one-pole ray);
`folded_vector_directions` is the reverse view. These implement "a vector *is* a
2-node affine manifold". The live extraction/steering path for vector concepts
still uses baked `Profile`s folded at dispatch (`session._vector_push_fragment`).

## extraction.py

`ExtractionPipeline` — custom-concept orchestration: tensor cache → statements →
scenarios → pairs → extract → save. Dependencies via the `ModelHandle` /
`PackWriter` protocols (`SaklasSession` satisfies both). `DEFAULT_EXTRACTION_METHOD
= "dim"` and difference-of-means is the only method (`_method_label` no longer
dispatches on PCA). Tensor-cache hit re-checks `statements_sha256` against the
sibling `statements.json` (drift → re-extract, not raise). `extract()` emits
`VectorExtracted`.

`ManifoldExtractionPipeline.fit(folder, *, sae=...)` is the manifold/subspace
fitter. Cache-hit on `nodes_sha256` (folds in corpus + `{domain, node_coords}`
authored / `{fit_mode, hyperparams}` discover). Else pools per-node centroids
(`compute_node_centroid`) and dispatches on `fit_mode`:
- **`pca`** (flat / discover): per layer `fit_affine_subspace` (μ-centered PCA
  basis — whitened/Fisher when the whitener covers all fit layers, Euclidean
  otherwise — neutral-anchored frame, real per-layer `node_coords`), then per-axis
  DLS (`compute_dls_axes` → `select_axes`) across all fit layers, then the
  μ-centered `subspace_share` bake.
- **`authored`/`spectral`** (curved): per layer `fit_layer_subspace` (PCA frame +
  RBF surface), share bake, and a per-layer `origin` (`invert_parameterization` of
  the neutral mean — curved only; flat's foot is coord 0).
Discover mode derives per-model coords via `discover_coords` at a reference layer
first. SAE coverage is fail-fast (`SaeCoverageError`). Bakes `explained_variance`,
`mahalanobis_share` (+ `share_metric`), `origin`. **No lever** — torn out.
Whitener resolution is deferred past the cache-hit return and gated all-or-nothing.
Emits `ManifoldExtracted`.

## sae.py

SAE backend. `SaeBackend` (runtime-checkable Protocol): `encode_layer`/
`decode_layer`, `release`, `revision`, `layers`. `MockSaeBackend` for CPU tests;
`SaeLensBackend` the concrete adapter. `load_sae_backend(release, *, revision,
model_id, device, dtype)` queries SAELens, validates base-model compatibility,
resolves per-layer sae_ids (`_canonical_layer_map` narrowest-width), gates
`sae_lens` imports so the module loads without `[sae]`. Errors:
`SaeBackendImportError`, `SaeReleaseNotFoundError` (difflib suggestions).

## mahalanobis.py

`LayerWhitener` holds per-layer centered neutrals `X_L ∈ ℝ^(N,D)` + the Woodbury
inverse `K_L = (NλI + XXᵀ)⁻¹`; `apply_inv(layer, v) = (1/λ)(v − Xᵀ K X v)` in
O(ND), no D×D. Ridge `λ_L = (‖X_L‖²_F / (N·D)) · ridge_scale`. Built lazily via
`from_neutral_activations` / `from_cache(model_id)`; neutrals cached **fp32** (the
project-wide invariant — fp16's 65504 ceiling overflows gemma-3's late layers to
±inf, poisoning Σ). Any layer whose centered acts or `K` come back non-finite is
*excluded*, so `covers_all` is trustworthy as "finite factors everywhere" — the
all-or-nothing gate shared by extraction, manifold fit, projection, monitor, and
`vector compare`. Primitives: `mahalanobis_cosine`, `leace_project`, `apply_inv`,
`subspace_gram(layer, B) = B Σ⁻¹ Bᵀ` (the (R,R) reduced inverse covariance behind
whitened share + whitened manifold reads), `woodbury_factors` (device-resident
factors for the monitor's inline per-token apply). `SaklasSession.whitener` is a
lazy property; `bootstrap_probes` builds it eagerly and wires it into both
monitors via `set_whitener`.

## manifold.py

Pure-tensor (fp32, no session/IO) subspace + manifold math. Goodfire "Manifold
Steering" (arXiv 2605.05115), generalized to arbitrary intrinsic dim/topology.

Domains: `ManifoldDomain` ABC + `BoxDomain` (open/periodic axes), `SphereDomain`
(Sⁿ chordal), `CustomDomain` (explicit immersion; also the identity carrier for
discover coords and synthesized affine subspaces). `domain_from_spec`/`to_spec`
round-trip the tagged union.

`LayerSubspace` — `mean`, `basis` (+ the curved RBF triple + unit-box
normalization). `is_affine ⇔ node_params is None`. Flat layers carry `node_coords`
(K,R) — the real neutral-anchored per-layer node positions (steer-target source).
`affine(mean, basis, node_coords=)`, `select_axes(kept)` (per-axis DLS prune),
`eval_at`, `jacobian_at`, `rbf_params()` (raises on a flat subspace).

`_pca_basis(X, *, n_components, whitener, layer)` — μ-centered PCA: Euclidean SVD,
or the whitened/Fisher generalized eigenproblem `(S_b, Σ)` via the Woodbury Σ⁻¹
(`G = X Σ⁻¹ Xᵀ`, directions `Σ⁻¹ Xᵀ a`, re-orthonormalized via QR so the hot path
is untouched). Shared by `fit_affine_subspace` (flat) and `fit_layer_subspace`
(curved). The **basis caveat** (do not break PCA@2 ≡ DiM): always μ-center the
scatter, never anchor-center it. `fit_affine_subspace` returns `(subspace,
mu_coords, ev_ratio)` and neutral-anchors the frame (`mean = P_basis(neutral)`,
real `node_coords = (centroids − neutral)·basisᵀ`); `orient_to` fixes the sign.
`subspace_share(mu_coords, basis, whitener, layer)` is the μ-centered (anchor-
independent) per-layer budget weight (`DEFAULT_N_COMPONENTS = 64`).

`Manifold` — domain + per-layer `LayerSubspace`s + `node_labels`/`node_coords`/
`node_roles` + the bakes `explained_variance`/`mahalanobis_share`/`origin`. The
`Profile` analogue. `manifold_point`, `tangent` (analytic RBF Jacobian),
`resolve_position` (coord payload or label), `nearest_node_{index,label,role}`.

`decompose(h, mean, basis) → (h_par_c, h_perp)` — the shared centered
decomposition behind injection + the read monitor. `synthesize_subspace(push,
ablate, neutral_means)` composes the active steering term set into one
`SynthesizedSubspace` per layer (orthonormal merged basis via `_ortho_basis`,
push-before-ablation ordering; `target_coord = B @ Σ coeffᵢ·poleᵢ`; `share =
‖Δ‖`). `inject_three_op(h, subspace, domain, target_coord, foot_seed, along,
onto)` is **the** injection: affine analytic shortcut (foot = q, geodesic slide,
`H_o` kept) vs curved per-token GN foot-follow (along transports `H_n` normal at
the new foot, onto scales it `(1−o)`, `H_o` kept). `norm_cap = 3·‖h‖` is the only
norm guard. `invert_parameterization` is the cold/eval-only damped-LM nearest-
point projection.

`save_manifold`/`load_manifold` round-trip the per-model tensor (`layer_<L>.{mean,
basis[,node_params,rbf_weights,poly_coeffs,coord_offset,coord_scale]}` + shared
`node_coords` + optional `origin`) + sidecar (`explained_variance_per_layer`,
`mahalanobis_share_per_layer`, `origin_per_layer`, `share_metric`,
`subspace_metric` — no `lever_per_layer`). `transfer_manifold` re-bakes share in
target space (no lever). Discover: `derive_pca_coords` (cumulative-variance
prefix) / `derive_spectral_coords` (Laplacian eigenmaps, eigenvalue-ratio cliff)
/ `discover_coords` dispatcher, with `PcaDiagnostics`/`SpectralDiagnostics`.
Naturalness eval: `to_hellinger`, `bhattacharyya_distance`, `fit_behavior_manifold`,
`trajectory_naturalness`, `compute_node_behavior_centroid`,
`compute_trajectory_distributions`.

## hooks.py

`SteeringHook` carries per-layer groups `(trigger, subspace, domain, target_coord,
origin_coord, along, onto)` and runs each active one through `inject_three_op`
(`_apply_manifold_groups`). A cheap pre-check skips inactive steps. Per-token foot
state `_manifold_feet` (cold → seed at `origin`, `_MANIFOLD_COLD_GN_STEPS = 4` GN
steps; warm → one step). No composed-tensor fast path: a steered layer always
runs the slow (ctx-consulting) hook, so per-step triggers / probe gates work
uniformly; `all_fast_path()` is true only unsteered (the StaticCache / graph-
capture eligibility signal).

`SteeringManager` owns `subspaces` (dispatch-synthesized merged affine, one per
trigger group, via `add_subspace`) + `manifolds` (curved, via `add_manifold`).
`apply_to_model` lowers both to per-layer entries: it share-weights (mean-1
normalized — `_manifold_layer_shares` prefers the baked `mahalanobis_share`, else
the Euclidean `‖eval_rbf(node_params)‖_F`), orthogonalizes the affine subspace
against curved spans (`_orthogonalize_affine_against` — curved wins shared
directions), and enforces `_CURVED_ORTHO_TOL = 1e-3` between two curved manifolds
(`OverlappingManifoldError`). Gain: `_MANIFOLD_GAIN = 1.0`, `eff_along_L = share_L
· base` (affine: α already in `target_coord`; curved: × clamped user `along`).
**No lever / N, no `[0,1]` clamp / water-fill on `along`** (a high-share layer is
meant to overshoot; `norm_cap` bounds it). `onto` stays clamped `[0,1]`.
`reset_manifold_feet` cold-starts followers per generation.

`HiddenCapture` — `attach`/`detach`/`stacked`/`latest_per_layer`. Incremental
mode (`set_incremental`) overwrites a length-1 bucket per layer so device memory
stays O(layers·D); fires the step sink after the highest hooked layer. The session
opts in for the common monitored case, wiring `score_single_token_tensor →
_incremental_rows`; `_finalize_generation` builds `(agg, per_token)` from those
rows bit-identically to `score_per_token`. Manifold probes + `return_hidden` force
full retention.

## monitor.py

`TraitMonitor` scores vector probes against per-layer hidden states in the
**whitened (Mahalanobis) cosine** when the whitener covers every probed layer,
else Euclidean cosine for all (all-or-nothing via `covers_all`; never a per-layer
mix). The probe weight stays `‖baked‖₂` (the bake already folded the Mahalanobis
score in). `_ensure_cache` precomputes whitened probe directions + Mahalanobis
norms + device-resident Woodbury factors; the hot path is one matmul + a cheap
per-token `Σ⁻¹h_c` apply, zero `.item()`. `_layer_sims` is the shared per-layer
kernel. Entry points: `score_per_token` (primary), `score_single_token{,_per_layer,
_tensor}`, `score_stack`, `measure`, live-mean `begin/update/end_live`.

`ManifoldMonitor` is the manifold read peer (accepts a whitener too).
`add_probe(name, manifold, *, top_n)` caches per-layer reduced `(K,R)` node coords.
Per token, EV-weighted across layers: **fraction** (`‖h_par_c‖/‖h−mean‖`, the
subspace share ∈ [0,1]) + **nearest** node. Whitened readout
(`AttachedManifoldProbe.whitened`, per probe, all-or-nothing) switches both to
Mahalanobis forms (M-orthogonal share, `M_R`-metric cdist). `score_single_token`
returns `ManifoldTokenReading`s; `flat_scalars` flattens to `{"<name>:fraction",
"<name>@<label>": −distance}` for the probe-gate machinery. `score_aggregate`
(end-of-gen) pools the last non-special token, runs the channels, and recovers
authoring coords + residual via `invert_parameterization`. `measure` is the
one-shot text helper.

## triggers.py

`Trigger` (frozen): phase flags + optional `ProbeGate`. `Trigger.active(ctx)`
consults flags and, when gated, `ctx.probe_scores[gate.probe]` against `score <op>
threshold`. Factories `first`/`after`/`when`. `TriggerContext.probe_scores` is
filled by the per-step callback, cleared on `reset()`. Gated triggers are inactive
during prefill and for missing probes (no raise). `ProbeGate` is frozen/hashable
so identical gates compose under equality; `ProbeGate.probe` is the canonical
scalar key from whichever monitor supplied it.

## cuda_graphs.py

CUDA-graphs / `StaticCache`. Eligible only for **unsteered** generation (any
steering hook forces the slow path). `is_cuda_graphs_supported(model, device)`
probes viability + caches keyed by underlying module id / device / dtype;
`make_static_cache` sizes to `prompt_len + max_new_tokens + offset`; `warn_once`
logs the fallback reason.

## generation.py

Token-by-token decode + KV cache under `torch.inference_mode()`. Models that
ignore `past_key_values` (e.g. talkie) flip `no_cache_mode` (O(N²), one warning).
`GenerationConfig` (frozen) holds session defaults; per-call `SamplingConfig`
composes into a local copy. Top-p via `torch.topk` (not full-vocab sort); `top_k`
(default 1024 cap) applied before top-p. `generate_steered` accepts `seed`, `stop`,
`logit_bias`, presence/frequency penalties, `logprobs`, `score_callback` (probe
gates), `forced_prefix`, `steering_active`. `forced_prefix` forces the first N
decode tokens while the multinomial draw still runs (re-seeding stays bit-
identical through the fork). `fork_from_token` / `prefill_assistant` /
`append_user_turn` / `append_assistant_turn` are the fork / commit entry points.
`detect_base_model` (no chat template) routes flat (`raw=True`) generation.
Per-message role substitution (`role_templates.apply_with_per_turn_roles`) backs
the roleplay scaffold; `_active_role` is the steering-driven role.
`supports_thinking` / `_detect_think_delimiters` round-trips Qwen/Gemma
`enable_thinking`, falling back to channel (gpt-oss) / bracket (Mistral-3)
detectors; `ThinkingState` / `GenerationState` drive the streaming state machine.

## session.py

`SaklasSession` owns the model, profile registry (`_profiles`), both monitors,
`SteeringManager`, `HiddenCapture`, generation defaults (`session.config`), the
loom tree, and a synchronous `events: EventBus`. `from_pretrained(model_id, *,
device, dtype, quantize, probes, projection_metric=..., dls=..., compile=...,
cuda_graphs=...)` does the HF load + probe bootstrap + layer-mean compute — there
is no `injection_mode`/`theta_max` param. `__init__` takes a pre-loaded model.
Both call `materialize_bundled()` + `selectors.invalidate()` early. `_N_PAIRS =
45`, `_N_PAIRS_PER_SCENARIO = 5`.

`generate` / `generate_stream` are keyword-only and accept `steering: str |
Steering | None` (dicts rejected), `sampling`, `thinking=None`, plus loom args
(`parent_node_id`, `n`, `recipe_override`); both return `RunSet` (`.first` is the
`GenerationResult`). `session.steering(value)` coerces via `Steering.from_value`,
materializes `ProjectedTerm`s into derived profiles (`_materialize_projections` →
`project_profile` with `session.whitener`, Gram-Schmidt fallback), and pushes a
per-scope entries dict onto a LIFO stack so nested scopes compose. The per-call
`projection_metric` rides a parallel `_steering_override_stack: list[str | None]`
walked top-down by `_resolve_projection_metric` (inner wins; default Mahalanobis).
`_resolve_pole_aliases` + the cache-hit auto-load (`_try_autoload_vector`) let
HTTP clients steer bundled probes without pre-registration.

`_compose_steering_entries` is the 4.0 dispatch (`ARCHITECTURE.md` §4): classify
each entry — `AblationTerm` → ablation fragment; `ManifoldTerm` → affine `%` joins
the merge (`_affine_manifold_push`, label-form) or curved `%` → `add_manifold`;
plain `(alpha, trigger)` → rank-1 push (`_vector_push_fragment`, unit dir +
`‖d_L‖` coord) — group push+ablate by trigger, `synthesize_subspace` per group,
`add_subspace`. `_install_composed_steering` → `apply_to_model`. There are **no**
persistent hooks. Manifold-implied roles aggregate under soft-warn +
highest-`|coeff|`-wins (`RoleBaselineMismatchWarning`).

`_generate_core` owns the `_gen_lock` re-entry guard, `_gen_phase` lifecycle
(`IDLE`/`PREAMBLE`/`RUNNING`/`FINALIZING`, via `session.gen_state`), the steering
context, `_begin_capture`/`_end_capture`, `_finalize_generation`, and teardown.
Monitor scoring is in-flight (no second forward pass); `SamplingConfig(return_hidden=
True)` widens capture to every layer and lands `GenerationResult.hidden_states`,
re-scorable via `session.score_hidden`. Manifold probes ride the same plumbing:
`add_manifold_probe(selector, *, as_name=None, top_n=3)` resolves via
`_ensure_manifold_loaded` and registers on `_manifold_monitor`;
`measure_manifold(text, *, names=None)` is the one-shot text scorer;
`_begin_capture` widens to the union of vector + manifold probe layers; the gate
callback merges `TraitMonitor.score_single_token` and
`ManifoldMonitor.score_single_token`/`flat_scalars` into
`TriggerContext.probe_scores`; `_finalize_generation` calls
`ManifoldMonitor.score_aggregate` into `GenerationResult.manifold_readings`.
`session.lock` is the server-owned `asyncio.Lock` (distinct from `_gen_lock`).
`generate_batch`/`generate_sweep` return `RunSet` (sweep builds the Cartesian
product as loom siblings). Hot-path events: `GenerationStarted`/`SteeringApplied`/
`SteeringCleared`/`ProbeScored`/`GenerationFinished` + `VectorExtracted`/
`ManifoldExtracted`; threaded subscribers hop via `loop.call_soon_threadsafe`.

## loom.py

`LoomTree` — the engine-side conversation tree. Nodes are turns, children are
alternative continuations, the active path is the model's context.
`messages_for(leaf)` renders the path as v2 chat messages (`with_labels=True` adds
each node's `role_label`); `flat_text(leaf)` is the base-model analogue (one
continuous string, no roles). Node ids are 26-char ULIDs. Primitives: edit,
branch, navigate, delete_subtree, `regenerate`. `Recipe` is the per-node
reproducibility receipt (steering expression, sampling, thinking, seed, probe set
+ per-probe sha256); `Recipe.overlay`/`invert_steering`/`compose_modifier` back
the auto-regen modes. Per-node token blobs live in memory during streaming;
`to_dict` omits them, `save` writes a gzip sidecar. Mutators raise
`MutationDuringGenerationError` (409) on conflict; `UnknownNodeError` (404) /
`InvalidNodeOperationError` (400) otherwise.

## tree_filter.py

Filter grammar for tree pruning — distinct from the steering `@when:` grammar
(that gates per-step readings; this gates per-node aggregates). Clauses are
`<agg>:<probe> <op> <threshold>`; `agg_op` ∈ `agg`/`any`/`last`; multi-clause is
AND. `parse_filter` → `FilterClause`; `FilterClause.evaluate`; `filter_tree`
backs `LoomTree.filter_by_expr`. `FilterParseError` on any parse problem.

## loom_diff.py

Cross-branch diff primitives. `text_diff` (word-level via
`difflib.SequenceMatcher` → aligned `DiffSpan`s), `readings_diff` (per-probe `Δ =
b − a`, sorted by `abs(delta)`), `per_token_diff` (byte-offset alignment),
`steering_delta` (compact `"+0.2 calm"` edge label via the shared grammar).

## transcript.py

Transcript export/import for loom paths. `SAKLAS_TRANSCRIPT_VERSION = 1`;
`to_yaml`/`from_yaml`. Import modes: **default** (new top-level branch), **here**
(child of active), **merge** (attach the non-matching tail at the deepest matching
prefix). Guards: model mismatch refuses `--merge` (`TranscriptModelMismatch`),
system-prompt mismatch banners, probe-hash drift warns / raises
`TranscriptProbeDriftError` under `--strict`.

## joint_logprobs.py

Cross-branch joint-logprob computation for the loom NodeCompareDrawer.
`compute_joint_logprobs(session, a_id, b_id)` reconstructs the shared prefix
(`_shared_prefix_len`, not re-tokenization), force-replays each branch under its
recipe, and assembles per-aligned-position rows with both branches' chosen-token
logprobs + the cross-branch evaluation (`_compute_rows`, alignment via
`per_token_diff`). `approx_kl` is top-K-truncated `KL(P_A‖P_B)`. Cached on
`session._joint_logprob_cache`; held under the session lock.

## results.py

`GenerationResult`, `RunSet`, `TokenAlt`, `TokenEvent`, `ProbeReadings`,
`ManifoldTokenReading`, `ManifoldAggregate`, `ResultCollector`. `RunSet` is the
list-like multi-run shape (`node_ids`/`grid`/`.first`/`.to_collector()`/
`.to_dataframe()`). `TokenEvent` carries `thinking`, `logprob`, `top_alts`,
`finish_reason`, `scores` (per-probe sims, live-stream-gated), `perplexity`,
`manifold_readings`. `GenerationResult` carries `prompt_tokens`, `finish_reason`,
optional `logprobs`, `manifold_readings`, and `applied_steering` (the canonical
expression, round-trips through `parse_expr`). `ManifoldTokenReading` (`fraction`,
`nearest`) is per-token; `ManifoldAggregate` (`fraction_mean`/`fraction_per_layer`/
`nearest`/`coords`/`coords_per_layer`/`residual_mean`/`residual_per_layer`) is
end-of-gen. `to_dict()` omits `hidden_states`.

## histogram.py

`HIST_BUCKETS = 16`; `bucketize(norms, buckets)` collapses sorted per-layer norms
into evenly-sized groups. Used by the TUI WHY footer + CLI `vector why`.

## sampling.py / steering.py / steering_expr.py / events.py / errors.py / profile.py

`SamplingConfig` — per-call frozen config with `merged_with`. `Steering` — frozen;
`from_value` accepts `str | Steering | None` (strings route through the shared
parser, dicts rejected); the only per-call override field is `projection_metric`
(`None` = inherit). `events.py` — synchronous `EventBus` + event dataclasses.
`errors.py` — `SaklasError` base; every saklas exception multi-inherits through it
while keeping its stdlib MRO, so `except SaklasError` catches the family and
`except ValueError`/`RuntimeError` at existing sites still works. `user_message()`
returns `(http_status, text)`. `SteeringExprError` lives here (re-exported from
`steering_expr.py`) with subclasses `ManifoldArityError` (wrong coord count, in
`add_manifold`) and `OverlappingManifoldError` (two curved manifolds overlap at a
layer, in `apply_to_model`). `profile.py` — `Profile` wraps `dict[int, Tensor]`
with `.layers`/`.save`/`.load`/`.to_gguf`/`.merged`/`.merged_with`/`.promoted_to`/
`.cosine_similarity(other, *, per_layer=False, whitener=None)` (magnitude-weighted)/
`.projected_away`; empty layer intersection raises `ProfileError`.

`steering_expr.py` hosts the unified grammar (`parse_expr(text, *, namespace=None)`
→ `Steering`; `format_expr` round-trips; `referenced_selectors` for install-time
checks). Term markers: `ProjectedTerm(coeff, trigger, operator, base, onto)`
(materialized into derived profiles before the hook layer), `AblationTerm` (`!atom`,
default coeff 1.0, doesn't compose with `~`/`|` — lowered through
`synthesize_subspace`'s ablation path at dispatch), `ManifoldTerm` (`along`,
`onto`, position; the third `toward` slot is removed — `_expand_three_op_coeffs`
yields a 1- or 2-tuple). Probe gates (`@when:<probe><op><threshold>`) accept three
identifier shapes — vector (`angry.calm`), manifold fraction (`circumplex:fraction`),
manifold label (`circumplex@elated`) — all stored verbatim in `ProbeGate.probe` so
the runtime gate is identical; the parser is the only place the discrimination
lives.
