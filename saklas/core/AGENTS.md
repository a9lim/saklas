# core/

Engine layer: model loading, the unified manifold/subspace fit + injection, the
unified `Monitor`, session orchestration, the generation loop, and the
loom conversation tree. The cross-cutting design lives in the repo-root
`ARCHITECTURE.md`; this file is the per-module map.

There is one artifact family — the manifold — and one injection kernel. A
steering vector is the 2-node `pca` case; concept extraction and manifold fitting
share `ManifoldExtractionPipeline`. Every steering term — vectors, poles, `~`/`|`
projections, `!` ablations, and affine/curved `%` — lowers to one per-layer
`subspace_inject` (along/onto) call. There is no angular/additive mode, no
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

Low-level capture, pooling, DLS, the vector⇄subspace fold, and profile I/O. **No
`extract_difference_of_means`** — extraction authors a 2-node `pca` manifold now
(`extraction.py`); what survives here is the capture + fold + projection
machinery that pipeline and dispatch both consume.

One forward pass per response; `_capture_all_hidden_states` hooks every layer at
once. Capture is **conversational** (4.0 / A2): a corpus item is an assistant
*response* to a fixed baseline *prompt*, so `_encode_and_capture_all(model,
tokenizer, prompt, response, layers, device, *, role=, model_type=, system_msg=)`
renders `[system: directive, user: prompt, assistant: response]` (`system_msg`
defaults to the shared `_LENGTH_DIRECTIVE` — the brevity directive only, *not* the
generation persona — standard assistant label; matches the generation framing so
the response isn't OOD and the directive cancels as common-mode against neutral)
and pools from the response's **last content token** — `last_content_index` walks
back past both `tokenizer.all_special_ids` and `tokenizer.added_tokens_encoder`
values to skip trailing chat-template markers (where outlier channels dominate).
`special_token_ids` + `last_content_index` are the one canonical "last non-special
token" definition shared by every single-state readout (centroid pooling, vector
aggregate, manifold aggregate). `_render_and_tokenize_for_capture(tokenizer,
prompt, response, device, *, role=, model_type=)` is the shared
render+tokenize+walkback front half; `role` substitutes the assistant label only
when an explicit per-node role override is set (the persona-baselined fit), the
swap-back default being the standard assistant. `_load_baseline_prompts` loads the
shared A2 user prompts (user override → bundled `saklas/data/baseline_prompts.json`,
48 prompts); `_neutral_pairs` pairs the neutral corpus to those prompts
(`response[i] ↔ prompt[i % k]`). `compute_layer_means` /
`compute_neutral_activations` build the probe-centering baseline + the whitener's
neutral cache, pooling conversational `(prompt, response)` pairs via `_neutral_pairs`.

DLS (Selective Steering, Dang & Ngo 2026 Eq. 9): `compute_dls_axes(node_centroids,
bases, layer_means)` is the **N-node straddle** core — keep axis `d̂ᵣ` at layer L
iff `{(cᵢ − ν)·d̂ᵣ}` straddles zero. At K=2 (stacked `[μ_pos, μ_neg]`) it
reproduces a difference-of-means steering vector's keep set bit for bit (the
legacy opposite-sign product test). The subspace
apply path consumes the per-axis keep set by slicing the basis
(`LayerSubspace.select_axes`); an all-fail layer drops.

**The vector⇄subspace fold (the production vector path).**
`fold_directions_to_subspace(name, directions, neutral_means)` folds an arbitrary
per-layer direction (a `merge` linear-combination, a `~`/`|` projection, or a
folded bundled concept) into a neutral-anchored affine `R=1` `Manifold` — a
one-pole ray (the live 2-node-vector read goes through extraction's whitened
`fit_affine_subspace`; the legacy bipolar-centroid fold was retired in the
Mahalanobis-only collapse). `folded_vector_directions(manifold)` is the **reverse view**:
`{L: δ̂_L · share_L}`, the baked-direction view of a 2-node affine manifold, used
to back the `Profile`-returning surface (`extract()`, `subspace compare`/`why`,
GGUF export) without a second stored representation. It raises on a curved or
multi-dim manifold. Dispatch lowers every plain vector term through
`fold_directions_to_subspace` → `_affine_manifold_push` (`session.py`) onto the
merged affine subspace.

`save_profile`/`load_profile` round-trip a baked `dict[int, Tensor]` to
`.safetensors` + a slim `.json` sidecar (stamped with `PACK_FORMAT_VERSION` from
`io/packs.py`); they back the per-model `layer_means`/`neutral_activations`/
alignment caches and the folded-profile interchange. `project_profile(base, onto,
operator, *, whitener=...)` is the per-layer `~`/`|` projection — closed-form
LEACE, Mahalanobis-only: the whitener is **required** and must cover every
projected layer (`covers_all`), else `WhitenerError` (no Euclidean path).

## extraction.py

`ManifoldExtractionPipeline.fit(folder, *, sae=...)` is **the** extraction
pipeline (concept extraction and manifold fitting are the same thing — a vector is
a 2-node `pca` manifold). Dependencies via the `ModelHandle` protocol (`model` /
`tokenizer` / `layers` / `device` / `model_id` / `_run_generator` /
`generate_responses`) + an `EventBus` for `ManifoldExtracted`; `SaklasSession`
satisfies it implicitly. Cache-hit on the sidecar `nodes_sha256` (folds in corpus
+ `{domain, node_coords}` authored / `{fit_mode, hyperparams}` discover) **and**
`sae_revision`. Else reads `_load_baseline_prompts()` and pools per-node
conversational centroids (`compute_node_centroid(..., responses, baseline_prompts,
role=node_role, model_type=)`), threading the folder's `node_kinds` /
`node_roles` into the `Manifold` + sidecar metadata, then dispatches on
`fit_mode`:

- **`pca`** (flat / discover; the 2-node-vector case): derive per-model coords
  (`discover_coords` over the layer-agnostic consensus Gram — `mean_L` of each
  layer's whitened, node-mean-centered `(K,K)` Gram, signal-weighted so the
  layout draws on whichever layers carry the concept, no reference layer) when
  discover, then per layer
  `fit_affine_subspace` (μ-centered PCA basis — whitened/Fisher; the whitener must
  cover every fit layer or the fit raises `WhitenerError`, there is no Euclidean
  activation-space fit — neutral-anchored frame, real
  per-layer `node_coords`), per-axis DLS straddle across all fit layers
  (`compute_dls_axes` → `select_axes`), then the μ-centered `subspace_share` bake.
  The per-layer subspace dim is the `max_dim`-capped layout dim (the affine span
  *is* the layout — no `max_subspace_dim` for pca); the origin is the neutral
  projection (neutral-anchored frame — no `anchor_origin` knob).
- **monopolar** (a `pca` folder with `K == 1`): a structural early branch — a
  1-node flat fit is meaningless (needs `k+1 ≥ 2` poised nodes), so the engine
  reads it as concept-vs-neutral. Folds `concept − ν` (ν = `handle.layer_means`)
  into a 1-node neutral-anchored ray via `fold_directions_to_subspace` (raw δ̂
  basis, whitened share when `covers_all`), bypassing discover-coords / per-layer
  pca / DLS entirely; `method = "manifold_monopolar"`. Raises if ν is unavailable.
- **`authored`/`spectral`** (curved): per layer `fit_layer_subspace` (PCA frame +
  RBF surface), μ-centered share bake, and a per-layer `origin`
  (`invert_parameterization` of the neutral mean — curved only; flat's foot is
  coord 0).

`--sae` reconstructs each centroid through the SAE before the fit (fail-fast
`SaeCoverageError`); the fitted subspace is model-space regardless. Bakes
`explained_variance`, `mahalanobis_share` (+ `share_metric`/`subspace_metric`),
`origin`. **No lever.** Discover coords come from the **consensus Gram** —
`mean_L` of each layer's whitened, node-mean-centered `(K,K)` Gram
`X̃_L Σ_L⁻¹ X̃_Lᵀ` (signal-weighted, layer-agnostic; no reference layer) — and the
same per-layer Grams' traces are stamped into the sidecar as
`node_spread_per_layer` (`{str(L): tr(G_L)}`), the concept's whitened
signal-by-layer profile. Diagnostic only (nothing runtime branches on it; absent →
empty dict), surfaced by `manifold show`; computed for every K≥2 fit (authored too),
distinct from `mahalanobis_share` (the same whitened spread restricted to the
steerable subspace). Whitener resolution is deferred past the cache-hit return
and gated all-or-nothing on `covers_all`. `min_nodes(k) = 2k+1` for curved; a flat
`pca` fit needs only `k+1` (so K=2/k=1 is a steering vector). Emits
`ManifoldExtracted`. (Note: `extract.py`'s old concept `ExtractionPipeline` and
the `_capture_diffs_for_pairs` DiM apparatus are gone; the `discover-pca` 2-node
fit replaces them.)

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
`from_neutral_activations`, `from_cache(model_id)` (requires cached layer means),
or `from_neutral_cache(model_id)` (derives per-layer means from cached neutrals for
no-model-load transfer rebakes); neutrals cached **fp32** (the project-wide
invariant — fp16's 65504 ceiling overflows gemma-3's late layers to
±inf, poisoning Σ). Any layer whose centered acts or `K` come back non-finite is
*excluded*, so `covers_all` is trustworthy as "finite factors everywhere" — the
all-or-nothing gate shared by extraction, manifold fit, projection, monitor, and
`subspace compare`. Primitives: `mahalanobis_cosine`, `mahalanobis_norm`,
`leace_project`, `apply_inv`, `subspace_gram(layer, B) = B Σ⁻¹ Bᵀ` (the (R,R)
reduced inverse covariance behind whitened share + whitened manifold reads),
`woodbury_factors` (device-resident factors for the monitor's inline per-token
apply). `SaklasSession.whitener` is a lazy property; session construction wires it
into the unified `Monitor` via `set_whitener`.

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
`node_roles`/`node_kinds` + the bakes `explained_variance`/`mahalanobis_share`/
`origin` + `feature_space`/`metadata`. The `Profile` analogue. `node_kinds`
(abstract/concrete) is generation-only provenance — it selects the system template
+ elicitation role label at authoring time, but is NOT consumed at fit; it
round-trips through the save/load sidecar. `manifold_point`, `tangent` (analytic
RBF Jacobian), `resolve_position` (coord payload or label),
`nearest_node_{index,label,role}`.

`decompose(h, mean, basis) → (h_par_c, h_perp)` — a standalone centered-decomposition
helper (exported but off the hot path; `subspace_inject` and the monitor each
decompose inline). `synthesize_subspace(push,
ablate, neutral_means)` composes the active steering term set into one
`SynthesizedSubspace` per layer (orthonormal merged basis via `_ortho_basis`,
push-before-ablation ordering; `target_coord = B @ Σ coeffᵢ·poleᵢ`; `share =
‖Δ‖`). `subspace_inject(h, subspace, domain, target_coord, foot_seed, along,
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
`compute_trajectory_distributions`, `compute_node_centroid`.

## hooks.py

`SteeringHook` carries per-layer groups `(trigger, subspace, domain, target_coord,
origin_coord, along, onto)` and runs each active one through `subspace_inject`
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
stays O(layers·D); fires the step sink after the highest hooked layer. It is
currently **off** — the session keeps full retention and `_finalize_generation`
scores via `score_per_token`. Under the unified full reading there is no no-sync
incremental coord-row fast path to short-circuit (every read is a full per-probe
reading from the captured stack), so the only-latest-hidden memory win is gone by
design; manifold probes + `return_hidden` already force full retention regardless.

## monitor.py

`Monitor` — **one** read-side class (the unification of the former `TraitMonitor` +
`ManifoldMonitor`) that reads every probe shape as whitened subspace coordinates,
flat and curved alike. Hook-driven, fp32, Mahalanobis-only (`covers_all` or
`WhitenerError`). **One** read path: `_score_full` loops every probe through
`_score_probe_full`, which loops the probe's shared fit layers through
`_layer_geometry` and EV-weights across them — there is no batched-affine fast path
and no flat/curved field asymmetry. The research-tool priority is full per-token
information (nearest, curved coords, residual, per-layer), not the throughput a
batched matmul bought. Each layer's `_layer_geometry` yields the M-orthogonal
**fraction** `sqrt(gᵀ M_R⁻¹ g)/‖x‖_M` (`g = B Σ⁻¹ x`), the whitened query for the
**nearest** `M_R`-metric node, and the M-projection reduced coord `c = M_R⁻¹ g`.
From there:

- a **flat** (affine) probe recovers `coords` through the affine reduced→domain
  map (`coord_S @ c + coord_b`); off-surface `residual` is identically `0`.
  **domain-frame**: each layer's reduced coord is in that layer's `‖δ_L‖` units,
  mapped to the shared domain *before* EV-averaging (reference = each node's
  whitened read-coord). At rank-1 this is the pole-normalized coordinate (`1.0` at
  the positive node); a 1-node fold (monopolar `extract`, ad-hoc `probe()`) is the
  K=1 ray (`coord_b = 0`). A 2-node concept axis *and* the rank-8 `personas` fan.
- a **curved** probe recovers `coords` via the `invert_parameterization`
  nearest-point foot solve (run **per token** now — the accepted live cost) plus a
  real normalized off-surface `residual`.

`score_aggregate` (end-of-gen) pools the last non-special token per layer and calls
the *same* `_score_probe_full`, so the aggregate at a token index is bit-identical
to the live read at that token. `add_probe(name, manifold, *, top_n)` /
`remove_probe(name)` attach/detach any shape. The read shares
`_build_whitened_factors` (per-layer `_LayerWhiten` build), `_attach_manifold_probe`
(node cache + EV weights), and `_layer_geometry`. `__init__`'s `layer_means` is
vestigial on the hot path — the readout centers on each fit's own
`LayerSubspace.mean`. `probe_layers` is the capture-widening union
(`attached_layers` is a back-compat alias for the surfaces that consumed the former
`ManifoldMonitor`).

`flat_scalars` (one staticmethod) writes the gate channels from a readings dict:
`"<name>"` (= coords axis 0) + `"<name>[i]"` per axis, `"<name>:fraction"`,
`"<name>@<label>": −distance` per nearest node. Every probe — flat and curved — now
carries coords *and* nearest, so flat probes expose `@label` similarity gates too
(`@when:personas@hacker`) and the gate grammar is uniform. Entry points:
`score_per_token` (primary — returns `(aggregate readings, per-token axis-0 coord
stream)`), `score_single_token{,_per_layer}` (`_per_layer` is a view over the
reading's `coords_per_layer` backing the loom heatmap), `measure_from_hidden`,
`score_stack`, live-mean `begin/update/end_live`. History/stats are
per-coordinate-axis (`axis_stats`); the TUI-facing scalar helpers report axis 0.
The bundled roster is the fitted 2-node `Manifold`s themselves
(`_bootstrap_manifold_probes` — no fold). The one-shot re-render text scorer
(`measure`) is gone — every read source is live hooks scoring captured hidden
states, no second forward pass.

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
detectors; the internal `_ThinkState` machine + `GenerationState` drive streaming.

## session.py

`SaklasSession` owns the model, the in-memory profile registry (`_profiles`), the
loaded-manifold registry (`_manifolds`), the unified `Monitor` (`_monitor`),
`SteeringManager`, `HiddenCapture`, generation defaults (`session.config`), the loom
tree, and a synchronous `events: EventBus`. `from_pretrained(model_id, *, device,
dtype, quantize, probes, system_prompt, max_tokens=1024, dls=True, compile=False,
compile_mode=None, cuda_graphs=False, return_top_k=0)`
does the HF load + layer-mean compute + probe bootstrap — there is no
`injection_mode`/`theta_max` param. `__init__` takes a pre-loaded model. Both call
`materialize_bundled_manifolds()` + `selectors.invalidate()` early.
`_RESPONSE_MAX_TOKENS = 256` caps each in-character response (4.0 / A2
elicitation). Module-level helpers `_manifold_is_affine` / `_affine_manifold_push`
(per-layer basis rows + node-coord targets for a flat manifold) back the dispatch.
Conversational-elicitation helpers `_KIND_TEMPLATES` / `_article` / `_system_for`
(the per-kind system prompt) / `_role_for` (the swapped assistant-role label:
abstract → `someone_{slug}`, concrete → `{slug}`) author each node's corpus.

`extract(concept, baseline, *, kind="abstract", ...)` authors a 2-node
discover-`pca` manifold (`create_discover_manifold_folder`,
`hyperparams={"max_dim": 1}`) and fits it via `ManifoldExtractionPipeline`,
returning `(canonical_name, folded_vector_directions(manifold))`. The corpus is
generated conversationally — `generate_responses(concepts, kinds, *, roles=None,
samples_per_prompt=1, …)` has each pole answer the shared baseline prompts *in
character*, the concept riding both the system prompt (`_system_for`) and the
swapped assistant-role label (`_role_for`); responses are emitted samples-outer /
prompts-inner so `response[i] ↔ prompt[i % k]`. **Bipolar** (`baseline` set)
authors a 2-node folder and generates both poles. **Monopolar** (`baseline=None`)
authors a genuinely **1-node** folder (only the concept pole is generated); the
pipeline folds `concept − ν` into a 1-node neutral-anchored ray (see
`extraction.py`), neutral implicit via `layer_means`. `generate_neutral_responses`
is the neutral-corpus sibling (no system, standard label). `extract_vector_from_corpora(..., kind=...)` is the corpus-in sibling
(hand-authored pairs skip generation; corpora pooled conversationally, each length
a multiple of the baseline prompt set); `extract_manifold(folder)` is the
multi-node delegate that returns a `Manifold`. All gate against `GenState.IDLE`
(fitting runs forward passes). `_fit_vector_manifold` is the shared tail. (The
training-free `clone_from_corpus` / `io.cloning` path is removed in 4.0.)

`generate` / `generate_stream` are keyword-only and accept `steering: str |
Steering | None` (dicts rejected), `sampling`, `thinking=None`, plus loom args;
both return `RunSet` (`.first` is the `GenerationResult`). `session.steering(value)`
coerces via `Steering.from_value`, materializes `ProjectedTerm`s into derived
profiles (`_materialize_projections` → `project_profile` with `session.whitener`,
which is Mahalanobis-only — a `~`/`|` term on a session with no covering whitener
raises `WhitenerError`), and pushes a per-scope entries dict onto a LIFO stack so
nested scopes compose. There is no per-call projection-metric override.
`_resolve_pole_aliases` canonicalizes + sign-flips bare poles through the manifold
tier and routes variants.

**Steering resolution (manifold-first).** `_ensure_profile_registered(name)`
resolves a direction from, in order: (1) an in-memory baked direction already in
`_profiles` (ad-hoc `extract`/`merge`/projection results); (2) a fitted
2-node `pca` manifold on disk — `_try_fold_manifold` → `_ensure_manifold_loaded`
(load `[ns/]name[:variant]`, raw or `sae-<release>`) + `folded_vector_directions`,
memoized into `_profiles`; (3) a stale (`< PACK_FORMAT_VERSION`) legacy
`vectors/<ns>/<name>/` folder — `_port_stale_legacy_vector` ports it to a 2-node
manifold file-only (no tensor yet) and raises with the exact `manifold fit` command
to run. `_bootstrap_manifold_probes(categories)` is the probe-roster bootstrap:
for each `default/` manifold tagged in a requested category, fit-or-load the
2-node `Manifold` and hand it to the `Monitor` (which reads it as a rank-1
coordinate — no fold; replaces the old `io.probes_bootstrap.bootstrap_probes`).

`_compose_steering_entries` is the dispatch (`ARCHITECTURE.md` §4): classify each
entry — `AblationTerm` → ablation fragment; `ManifoldTerm` → affine `%` joins the
merge (`_affine_manifold_push`) or curved `%` → `add_manifold`; plain `(alpha,
trigger)` → `_ensure_profile_registered` → `fold_directions_to_subspace` →
`_affine_manifold_push` push fragment — group push+ablate by trigger,
`synthesize_subspace` per group, `add_subspace`. `_install_composed_steering` →
`apply_to_model`. There are **no** persistent hooks. Manifold-implied roles
aggregate under soft-warn + highest-`|coeff|`-wins (`RoleBaselineMismatchWarning`).

`_generate_core` owns the `_gen_lock` re-entry guard, `_gen_phase` lifecycle
(`IDLE`/`PREAMBLE`/`RUNNING`/`FINALIZING`, via `session.gen_state`), the steering
context, `_begin_capture`/`_end_capture`, `_finalize_generation`, and teardown.
Monitor scoring is in-flight (no second forward pass); `SamplingConfig(return_hidden=
True)` widens capture to every layer and lands `GenerationResult.hidden_states`,
re-scorable via `session.score_hidden`. Probes ride the same plumbing:
`add_probe(selector, *, as_name=None, top_n=3)` resolves via `_resolve_probe_manifold`
(a 2-node `pca` concept folds to a rank-1 probe via `_fold_profile_probe`; a
multi-node manifold attaches whole) and registers on the unified `_monitor`;
`remove_probe(name)` detaches. `_begin_capture` widens to `_monitor.probe_layers()`;
the gate callback runs one `_monitor.score_single_token` through `flat_scalars` into
`TriggerContext.probe_scores` (one key space —
`"<name>"`/`"<name>[i]"`/`"<name>:fraction"`/`"<name>@<label>"`);
`_finalize_generation` calls `_monitor.score_aggregate` into
`GenerationResult.manifold_readings`.
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
`ManifoldReading`, `ResultCollector`. `RunSet` is the
list-like multi-run shape (`node_ids`/`grid`/`.first`/`.to_collector()`/
`.to_dataframe()`). `TokenEvent` carries `thinking`, `logprob`, `top_alts`,
`finish_reason`, `scores` (per-probe `ManifoldReading`s — the full readings,
live-stream-gated), `perplexity`, `manifold_readings`. `GenerationResult`
carries `prompt_tokens`, `finish_reason`, optional `logprobs`, `readings`
(per-probe `ProbeReadings`), `manifold_readings`, and `applied_steering` (the
canonical expression, round-trips through `parse_expr`). `ManifoldReading`
(`coords`/`fraction`/`nearest`/`residual` + `fraction_per_layer`/`coords_per_layer`/
`residual_per_layer`) is the **single** reading shape for both the live per-token
stream and the end-of-gen aggregate (the aggregate is the reading pooled at the
last-content token). Every field is populated for flat and curved fits alike;
`residual` is `0` for a flat fit (the surface fills its subspace) and the
normalized off-surface distance for a curved fit. `ProbeReadings` is vectorized per
coordinate axis (`mean`/`std`/`min`/`max`/`delta_per_gen` are `tuple[float,...]`,
`per_generation` a list of coord tuples). `to_dict()` omits `hidden_states`.

## histogram.py

`HIST_BUCKETS = 16`; `bucketize(norms, buckets)` collapses sorted per-layer norms
into evenly-sized groups. Used by the TUI WHY footer + CLI `subspace why`.

## sampling.py / steering.py / steering_expr.py / events.py / errors.py / profile.py

`SamplingConfig` — per-call frozen config with `merged_with`. `Steering` — frozen;
`from_value` accepts `str | Steering | None` (strings route through the shared
parser, dicts rejected); it carries no per-call metric override (`~`/`|` projection
is Mahalanobis-only). `events.py` — synchronous `EventBus` + event dataclasses.
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
`onto`, position; `_expand_along_onto_coeffs` yields a 1- or 2-tuple). Probe gates
(`@when:<probe><op><threshold>`) accept three identifier shapes — vector
(`confident.uncertain`), manifold fraction (`pad:fraction`), manifold label
(`pad@happy`) — all stored verbatim in `ProbeGate.probe` so the runtime gate is
identical; the parser is the only place the discrimination lives.
