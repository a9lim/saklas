# core/

The engine layer: model loading, capture, the manifold fit, the injection
kernel, the read-side instruments, session orchestration, the decode loop, and
the loom tree. The cross-cutting design lives in the repo-root
`ARCHITECTURE.md`; this file is the per-module ownership map.

Two claims hold throughout. **One artifact family** — the manifold. A steering
vector is its 2-node `pca` case, so concept extraction and manifold fitting are
one pipeline (`extraction.py`). **One injection kernel** — vectors, poles,
`~`/`|` projections, `!` ablations, and affine or curved `%` all lower to
per-layer `subspace_inject` calls carrying an along/onto pair
(`manifold.py`).

## model.py

HF causal-LM loading and per-architecture wiring. `ArchProfile` /
`_LAYER_ACCESSORS` map `model_type` → layer-list accessor (module-level `def`s,
not lambdas); `_TESTED_ARCHS` gates a one-time `UserWarning` on an untested
architecture. `LoadPlan` → `_resolve_load_plan` → `_load_with_fallbacks`
cascades over attention implementation (SDPA → eager), dtype, and device.
`_compile_with_probe` wraps `torch.compile` with a prefill+decode warmup so an
inductor/Triton failure surfaces at load as a caught warning plus eager
fallback. `_load_text_from_multimodal` extracts a text-only sub-model
(Ministral-as-Mistral3), strips `language_model.` prefixes, and dequantizes FP8.
`patch_torch_for_mps()` installs two lazy MPS-only workarounds (`torch.histc`
integer→float for MoE routing; `torch.ldexp` MXFP4 round-trip through CPU
honoring `out=`).

`get_unembedding(model)` returns `W_U` (`[vocab, d]`) and `get_final_norm` the
pre-unembedding norm module, found as a sibling of the `get_layers` ModuleList
(`norm`/`final_layernorm`/`ln_f`/`final_norm`) rather than a second per-arch
table — both exist for the Jacobian-lens readout, and nothing else in saklas
touches the unembedding outside the model's own forward.
`loaded_model_fingerprint` / `model_source_fingerprint` are the identities every
cached artifact keys on; `workspace_layer_indices` is the 40–90% band helper.

## static_cache.py

StaticCache detection, construction, and fallback — the `torch.compile` enabler
on **every** backend, not just CUDA (MPS takes StaticCache and the compile win
on top). `is_static_cache_supported` is the device-agnostic viability probe,
cached by underlying module id (through `torch.compile`'s `_orig_mod` wrapper),
device, and dtype; `is_cuda_graphs_supported` is the CUDA-only gate on top,
deciding `reduce-overhead` vs the fusion-only `default` compile mode;
`make_static_cache` is the single factory (it early-initializes layer buffers so
Transformers marks stable K/V addresses outside Dynamo, and flags sliding layers
that cannot slide this generation so the constant mask stays in the graph);
`warn_once` logs a fallback reason once per model. The user-facing knob is
spelled `cuda_graphs=`. Steering eligibility is decided at the steering layer
(`SteeringManager.all_fast_path` / `static_steerable`), not here.

## capture.py

Hidden-state capture, pooling, DLS, the vector⇄subspace fold, projection, and
the capture-mode vocabulary.

`CaptureMode` + `CaptureState` are the per-generation capture contract:
one enum (`INCREMENTAL`, `LEAN_INCREMENTAL`, `AGGREGATE_ONLY`, `GATING_SUBSET`,
`FULL`) plus the orthogonal `persistent` flag and the gated subset/keys, so
illegal combinations (incremental *and* aggregate-only) are unrepresentable.
`SaklasSession._begin_capture` sets it and every `_score_*` dispatch keys off
it; the modes trade only *when/how often* scoring runs and what memory the
capture keeps — every read is a full per-probe `ProbeReading` either way.

Capture runs in **right-padded batches**: `_encode_and_capture_all_batch`
renders + tokenizes a chunk, right-pads to a common length (attention-masked;
pool indices are unchanged because real tokens stay left-aligned), transfers
ids/mask once, and runs one `_capture_all_hidden_states` forward that pools each
row at its last-content index *inside the hook* (per-row gather → `(B, D)` per
layer, never `(B, T, D)`). Rendering, special-token walkback, and padding stay on
CPU. `_encode_and_capture_all` is the single-pair sibling;
`_render_and_tokenize_for_capture` the shared render+tokenize+walkback front
half; `_ReusablePooledCapture` the buffer. `_CaptureComplete` aborts the model
wrapper after the last requested block, so final norm and the LM head never run.
`_CAPTURE_BATCH = 16` is the starting chunk, grown toward `_CAPTURE_BATCH_MAX =
64` on clean batches.

Capture is **conversational**: a corpus item is an assistant *response* to a
fixed baseline *prompt*, rendered as `[system: directive, user: prompt,
assistant: response]` and pooled from the response's last content token.
`_LENGTH_DIRECTIVE` ("Answer in one short paragraph.") is the sole system prompt
at capture — the generation persona is generation-only, so the directive is
common-mode that cancels against neutral while matching capture framing to
generation framing. `role=` substitutes the assistant label only under an
explicit per-node role override (the persona-baselined fit); `roles=` carries
mixed per-row substitutions for fit-wide batches. `_load_baseline_prompts` loads
the 48 shared prompts (user override → bundled package data); `_neutral_pairs`
aligns the neutral corpus to them (`response[i] ↔ prompt[i % k]`);
`compute_neutral_activations` builds the single per-model neutral artifact, and
the probe-centering baseline is its per-layer `X.mean(0)` — there is no separate
layer-mean pass or cache. `special_token_ids` + `last_content_index` are the one
canonical "last non-special token" definition every single-state readout shares.

`compute_dls_axes(node_centroids, bases, layer_means)` is Discriminative Layer
Selection (Selective Steering, Dang & Ngo 2026 Eq. 9) as an N-node straddle:
keep axis `d̂ᵣ` at layer L iff `{(cᵢ − ν)·d̂ᵣ}` straddles zero — same-side layers
encode concept *intensity*, not polarity. At K=2 this is exactly the pos/neg
opposite-sign test. The apply path consumes the keep set by slicing the basis
(`LayerSubspace.select_axes`); an all-fail layer drops.

`fold_directions_to_subspace(name, directions, neutral_means)` folds an
arbitrary per-layer direction (a merge, a `~`/`|` projection, a folded bundled
concept) into a neutral-anchored affine `R=1` `Manifold` — a one-pole ray.
`folded_directions(manifold)` is the reverse view `{L: δ̂_L · share_L}`, the
baked-direction shape backing every `Profile`-returning surface (`extract()`,
`manifold compare`/`why`, GGUF export) without a second stored representation;
it raises on a curved or multi-dim manifold, and `is_foldable_vector_manifold`
is the predicate. `project_profile(base, onto, operator, *, whitener)` is the
per-layer `~`/`|` projection — closed-form LEACE, Mahalanobis-only: the whitener
is required and must cover every projected layer (`covers_all`) or `WhitenerError`.

## mahalanobis.py

`LayerWhitener` holds per-layer centered neutrals `X_L ∈ ℝ^(N,D)` plus the
Woodbury inverse `K_L = (NλI + XXᵀ)⁻¹`; `apply_inv(layer, v) = (1/λ)(v − Xᵀ K X
v)` in O(ND), never D×D. Ridge `λ_L = (‖X_L‖²_F / (N·D)) · ridge_scale`. Built
lazily from `from_neutral_activations` (in-memory) or `from_cache(model_id)` —
the offline loader that resolves the atomic `neutral_activations.json` pointer to
its immutable per-layer fp32 shards without a model load and derives the
centering mean as `X.mean(0)`. Neutrals are cached **fp32**: fp16's 65504 ceiling
overflows gemma-3's late layers to ±inf and poisons Σ. Any layer whose centered
activations or `K` come back non-finite is *excluded*, which is what makes
`covers_all` trustworthy as "finite factors everywhere" — the all-or-nothing gate
shared by extraction, manifold fit, projection, the monitor, and `manifold
compare`. Primitives: `mahalanobis_cosine`, `mahalanobis_norm`, `leace_project`,
`apply_inv`, `subspace_gram(layer, B) = B Σ⁻¹ Bᵀ` (the reduced `(R,R)` inverse
covariance behind whitened share and whitened manifold reads), and
`woodbury_factors` (device-resident factors for the monitor's inline per-token
apply). `SaklasSession.whitener` is a lazy property wired into the `Monitor` via
`set_whitener`.

## manifold.py

Pure-tensor (fp32, no session/IO) subspace and manifold math — Goodfire
"Manifold Steering" (arXiv 2605.05115) generalized to arbitrary intrinsic
dimension and topology. The on-disk tensor codec (`save_manifold`/`load_manifold`)
and the disk-backed `ActivationRowStore` row spool live in
`io/manifold_tensors.py` — io importing core's dataclasses is the correct
layering arrow; fit-capture math here lazy-imports the store.

**Domains.** `ManifoldDomain` ABC + `BoxDomain` (per-axis open or periodic:
boxes/disks, cylinders, n-tori), `SphereDomain` (Sⁿ, chordal), `CustomDomain`
(explicit immersion; also the identity carrier for discover coords and
synthesized affine subspaces). `domain_from_spec` / `validate_domain_spec` /
`normalize_domain_spec` round-trip the tagged union.

**Subspaces.** `LayerSubspace` carries `mean`, `basis`, the curved RBF triple +
unit-box normalization, and (flat) real neutral-anchored `node_coords` (K,R) —
the steer-target source. `is_affine ⇔ node_params is None`. `affine(...)`,
`select_axes(kept)` (per-axis DLS prune), `eval_at`, `jacobian_at`,
`rbf_params()` (raises on flat), `sigma_at`. `manifold_is_affine(manifold)` is
the public flat/curved predicate.

`_pca_basis(X, *, n_components, whitener, layer)` is μ-centered PCA: Euclidean
SVD, or the whitened/Fisher generalized eigenproblem `(S_b, Σ)` via the Woodbury
Σ⁻¹ (`G = X Σ⁻¹ Xᵀ`, directions `Σ⁻¹ Xᵀ a`, re-orthonormalized by QR so the hot
path is untouched). Basis caveat: **always μ-center the scatter, never
anchor-center it** — that is what keeps PCA@2 ≡ difference-of-means.
`fit_affine_subspace` returns `(subspace, mu_coords, ev_ratio)`, neutral-anchors
the frame (`mean = P_basis(neutral)`, `node_coords = (centroids − neutral)·basisᵀ`),
and `orient_to` fixes the sign. `subspace_share(mu_coords, basis, whitener,
layer)` is the μ-centered, anchor-independent per-layer budget weight
(`DEFAULT_N_COMPONENTS = 64`). Callers can pass a precomputed Gram and `Σ⁻¹X`, so
Fisher directions reuse `Aᵀ(Σ⁻¹X)` without a second Woodbury apply. The Euclidean
SVD branch exists for the behavior-space naturalness fit (output-distribution
space, no rogue activation dims); every activation-space caller requires the
whitener.

`Manifold` is the `Profile` analogue: domain + per-layer `LayerSubspace`s +
`node_labels`/`node_coords`/`node_roles`/`node_kinds` + the bakes
`mahalanobis_share`/`origin` + `feature_space`/`metadata`. `node_kinds`
(abstract/concrete/custom) is generation-time provenance — it selects the system
template and elicitation role label at authoring, is never consumed at fit, and
round-trips through the sidecar. `manifold_point`, `tangent` (analytic RBF
Jacobian), `resolve_position` (coord payload or label),
`nearest_node_{index,label,role}`.

**Composition + injection.** `synthesize_subspace(push, ablate, neutral_means, *,
whitener=None)` composes the active term set into one `SynthesizedSubspace` per
layer (orthonormal merged basis via `_ortho_basis`, push before ablation). With a
covering whitener the push is **whitened-normalized**, which is what makes
`along` a scale-stable strength knob: `share = ‖Δ‖_M` (the whitened displacement,
feeding the mean-1 cross-layer profile) and `target_coord = Σᵢ coeffᵢ·(B@dirᵢ)/‖dirᵢ‖_M`
(each fragment a whitened-unit direction scaled by its user coefficient). Node
distance therefore does not set push magnitude — a tight bipolar pole and a far
persona centroid land on the same whitened budget (`Σ_L eff_along_L =
gain·n_layers`), linear in α. Without a covering whitener (CPU stub, degenerate
fit) it falls back to the raw-Euclidean `target_coord = B @ Σ coeffᵢ·poleᵢ`,
`share = ‖Δ‖₂`. Per-axis κ coefficients come out of the same call: 0 on push
axes, the exact signed ablation-operator eigenvalues on the ablation complement
(partial/repeated/non-orthogonal terms preserved; the tiny eigensolve CPU-hops on
MPS).

`subspace_inject(h, subspace, domain, target_coord, foot_seed, along, onto)` is
**the** injection. The affine path is analytic: foot = the projected coord,
translate by the fixed offset with per-axis κ collapsing ablation axes
(`p_new = q + a·(target − κ·q)`), off-subspace residual kept verbatim. The curved
path runs a warm-started per-token Gauss-Newton foot follow, translating the foot
and transporting the off-surface residual `H_n` to the new foot by the minimal
orthogonal principal-angle rotation between the old and new tangent frames
(`_frame_rotation_transport` — an exact identity when the foot does not move, so
the curved path is identity at `along=0` regardless of foot accuracy). `onto`
shrinks `H_n` toward the zero-thickness wire on a σ-less fit, or toward the local
fuzzy σ-tube when the fit carries one. Keeping `h_perp` verbatim is what lets a
vector and N orthogonal manifolds compose with zero cross-talk. MGS
orthonormalization plus a CPU-hopped `n×n` SVD (`_svd_mps_safe`) keep it
MPS-safe; `norm_cap = 3·‖h‖` guards the curved path only.

**Penalized smoothing.** `fit_rbf_smoothed(node_params, values, *, smoothing)` is
the thin-plate/Duchon generalization of `fit_rbf_interpolant`: the penalized
saddle `[E+λI Q; Qᵀ 0][w;c]=[y;0]`. `smoothing="auto"` GCV-selects λ
(`_gcv_select_lambda` → `_rbf_smoother_matrix`, GCV `K·RSS/(K−edf)²`, edf
`tr S_λ`); `0`/`None` delegates to the exact interpolant bit-for-bit; a float
pins λ. Fitted weight shapes are unchanged, so the hot-path `eval_rbf` is
untouched — only the coefficients shrink. `fit_layer_subspace` takes `smoothing=`
and reports the chosen λ/edf through its `rbf_info` out-dict.
`prepare_rbf_fit_plan` factors the node-layout-only work (kernel/polynomial
blocks, QR/eigensystem, λ grid, fixed-λ LU) once per manifold; every layer and
`fit_sigma_field` reuse the `RbfFitPlan` while keeping non-aliased persistent
tensors. CPU/fp32 — the saddle is MPS-unsafe.

**Fuzzy-manifold σ-field (curved fits only).** Optional per-layer tube thickness:
the surface carries a within-node off-surface spread `σ(z)` instead of being a
zero-thickness wire. `compute_manifold_node_stats` produces the layer-major row
spool; `compute_store_reduced_covariances` streams it through bounded fp32
chunks after the basis exists, centering out-of-place (stable at large
common-mode offsets, and never mutating an fp32 mmap view) and accumulating each
node's reduced `(R,R)` covariance.
`compute_node_reduced_covariance_from_rows` serves standalone callers without
retained rows. `fit_sigma_field` reduces the covariance to one off-surface scalar
per node (`_off_surface_var`: the normal-complement trace via the surface tangent,
divided by the normal degrees of freedom from the tangent's *actual* local rank,
including rank-deficient folds where `R <= n`; one batched economy SVD supplies
both rank and projector) and fits a separate `log σ` RBF over the same normalized
`node_params`, stored on `LayerSubspace.{sigma_rbf_weights,sigma_poly_coeffs}`
(absent ⇒ `has_sigma` False ⇒ `sigma_at` returns 0 ⇒ the plain `(1 − o)`
collapse). `sigma_at` costs one extra `eval_rbf` on the curved path. v1 is
isotropic — one scalar per node.

`transfer_manifold_subspaces(src, alignment, *, whitener, from_model, to_model)`
is the pure-tensor core of cross-model transfer: map points/means through
`M_L x + b_L` and basis directions through `M_L`, QR-orthonormalize the mapped
rows, transform the affine/RBF reduced coefficients by the exact companion map,
re-bake the Mahalanobis share in target space (target whitener required), and
transform `origin`. Rank collapse is rejected, and a curved transfer is rejected
when the companion map is not an isometry — a scalar sigma field cannot encode an
anisotropic tube. Folder orchestration around it lives in
`io/manifold_lifecycle.py::transfer_manifold`.

`invert_parameterization` is the cold/eval-only damped-LM nearest-point
projection; `decompose(h, mean, basis)` is a standalone centered decomposition
(exported, off the hot path — `subspace_inject` and the monitor each decompose
inline).

## topology.py

Coordinate discovery and topology selection for discover-mode manifolds. Pure
tensor math on the `(K, K)` node scatter — no model, no IO; it depends on
`manifold.py`, never the reverse, and `extraction.py` is the one production
consumer.

**Coordinate derivation.** `derive_pca_coords` eigendecomposes the consensus Gram
and keeps the smallest prefix whose cumulative variance crosses `var_threshold`
(capped at `max_dim`); `derive_spectral_coords` reads pairwise distances off the
same Gram, runs Laplacian eigenmaps on a symmetric k-NN graph
(`_knn_adjacency` → `_laplacian_eigen`), and picks `k` at the eigenvalue-ratio
cliff. `discover_coords` dispatches, with `PcaDiagnostics`/`SpectralDiagnostics`.
The derivation is **layer-agnostic** — the consensus Gram is `mean_L` of each
layer's whitened, node-mean-centered `(K,K)` Gram, so whitening puts every layer
in common units and a layer that does not separate the nodes drops out on its
own. `neutral_layout_coord` places the per-model neutral mean into a flat layout
by landmark MDS, so `% 0,…,0` reads as neutral; subtracting it is a pure
translation and leaves steering untouched (cardinal weights are
translation-invariant).

**Topology selection** (`fit_mode="auto"`). `select_topology` makes two
deliberately decoupled decisions — a single reconstruction score has a dimension
bias, so the highest-dim candidate would always win. **(a) flat vs curved**: the
flat affine and curved RBF fits are scored by effective-dof-penalized GCV
(`_ols_gcv_score` / `_rbf_gcv_score`) in a shared whitened/Fisher reduced metric,
with the curved candidate floored to the flat candidate's dim so the two compete
at matched expressiveness (the spectral cliff systematically undershoots, and an
under-dimensioned curved fit reads flat). **(b) periodic axes**: Vietoris–Rips H1
persistent homology (`_rips_h1_persistence` boundary-matrix reduction →
`_count_persistent_loops`) counts loops robustly — a circle, a 6:1 ellipse, and a
noisy circle all read as one loop; a 2-torus as two; a blob/arc/line as zero —
and the spectral eigenpairs supply the angle coordinates
(`_detect_periodic_axes`, with `_is_angular_harmonic` deduplicating a circle's
`cos kθ` harmonics). Returns a `TopologyChoice` (`fit_mode`/`coords`/`domain` +
ranked `TopologyCandidate`s for the sidecar). Sphere is **authored-only** — the
least reliable topology to detect from few centroids.

PH counts loops by hole size, so `_faint_cycle_coords` is the guarded
single-cycle fallback, run only when PH counts zero. Off a greedy+2-opt tour
(`_nn_tour`) it accepts two sampling regimes, both gated `7 ≤ K ≤ 128` and both
requiring graded growth and 1-D-ness: **uniform** — a small cyclic modulation on a
near-equidistant heap, accepted on low 2-NN max degree, near-uniform tour-edge
closure, and local recall; and **clustered** — tight clumps spaced around the loop
(the sampling real concept families have), where tour edges go bimodal and
closure/recall fail though the loop is real, accepted instead on ≥2 decisively
bimodal, mutually regular inter-cluster gaps plus a real far antipode. It returns
a uniform `2π·rank/K` `S¹` coordinate in the recovered cyclic *order* — topology,
not metric, is what a periodic domain needs. The regression suite pins the
specificity (~0% false positives on random Gaussian heaps, grids, fans, arcs,
lines, blobs) and clustered-ring recall; the bimodality guard trades two
documented false negatives (a very loose cluster heap approaching uniform, an
ellipse beyond 6:1) for that specificity. A gapped ring is geometrically the same
point cloud as an open arc, so it correctly stays non-periodic.

## naturalness.py

Behavior-space naturalness eval — the one part of the manifold evaluation
pipeline that drives a live model forward, which is why it is not in the
pure-tensor `manifold.py`. `to_hellinger`, `bhattacharyya_distance`,
`fit_behavior_manifold`, `trajectory_naturalness` are pure-tensor helpers over
`manifold.py` primitives; `compute_node_behavior_centroid` and
`compute_trajectory_distributions` (with `_next_token_distribution`) are the two
that call `model(...)`. Consumed by `cli/runners/experiment.py`.

## extraction.py

`ManifoldExtractionPipeline.fit(folder, *, sae=..., layer_indices=...,
fit_mode=..., hyperparams=..., force=..., dls=...)` is **the** extraction
pipeline. Dependencies arrive structurally through the runtime-checkable
`ModelHandle` protocol (`model`/`tokenizer`/`layers`/`device`/`model_id`/
`_run_generator`/`generate_responses`) plus an `EventBus` for
`ManifoldExtracted`; `SaklasSession` satisfies it implicitly.

**Cache identity.** The sidecar `nodes_sha256` folds in labels, corpus, and
`{domain, node_coords}` (authored) or `{fit_mode, hyperparams}` (discover), plus
the resolved `template_ref` content; the full cache key adds `sae_revision`,
token-exact capture identity (baseline prompts and tokenizer render included),
the loaded-model fingerprint, and the fitted-layer set. The mandatory Mahalanobis
whitener is resolved *after* the fitted-tensor fast path but *before* any
activation capture, so a missing or partial neutral cache cannot waste a full
model pass. `resolve_fit_layer_indices`, `prepare_capture_rows`, and
`offline_fit_identity` are the pieces of that key derivable with no weights
loaded; `io.manifold_lifecycle.preflight_manifold_fit_noop` consumes them so the
weight-free no-op and the fit cannot drift on what a corpus hashes to.

**Capture.** `compute_manifold_node_stats` pools the whole roster in one row
stream that crosses node boundaries, so short nodes fill shared forward batches;
OOM halves the active batch and a clean run grows it back. Centroid-only fits
reduce by node on device before transfer; raw curved fits retain source-dtype rows
in a layer-major mmap spool for the later covariance instead of capturing twice or
holding the roster in fp32 RAM. A token-exact per-model **capture cache (format
v4)** lets domain, topology, and smoothing refits skip model forwards. Its
identity includes node boundaries; its digest metadata validates centroid payloads
plus exact per-layer row tensors; payloads are immutable generation-named
safetensors shards, one per layer, so a disjoint top-up writes only new shards and
one damaged layer recaptures only itself. Publication fsyncs payloads and
directory entries, writes a recovery journal, atomically replaces and
directory-fsyncs the authoritative JSON pointer, and only then GCs superseded
generations — a failed publication preserves the prior good pointer, and the next
fit adopts a complete crash-left journal under the capture-stem lock without
rerunning capture. Layer coverage is unioned (full→subset needs no forward;
overlapping subsets capture only missing layers), disjoint cached and fresh stores
combine by view, and `force` drops only the requested layers. Cache groups prune
oldest-first past 8 GiB (`SAKLAS_MANIFOLD_CAPTURE_CACHE_GB`); `pack clear` /
`pack rm` remove referenced groups.

**Fit modes.** `layer_indices` accepts an explicit set or the canonical 40–90%
workspace band; the folder's `node_kinds`/`node_roles` thread into the `Manifold`
and sidecar; then `_resolve_fit_geometry` dispatches:

- **`pca`** (flat; the 2-node-vector case) — derive per-model coords over the
  consensus Gram when discover, then per layer `fit_affine_subspace` (whitened /
  Fisher basis, neutral-anchored frame, real `node_coords`), per-axis DLS straddle
  across all fit layers, then the μ-centered `subspace_share` bake. The per-layer
  subspace dim *is* the `max_dim`-capped layout dim (the affine span is the
  layout), and the origin is the neutral projection. The shared display layout is
  re-anchored on `neutral_layout_coord`.
- **monopolar** (a `pca` folder with `K == 1`) — a structural early branch: a
  flat fit needs `k+1 ≥ 2` poised nodes, so the engine reads a single node as
  concept-vs-neutral and folds `concept − ν` into a 1-node neutral-anchored ray
  via `fold_directions_to_subspace` (raw δ̂ basis, whitened share when
  `covers_all`), bypassing discover coords, per-layer PCA, and DLS.
  `method = "manifold_monopolar"`. Raises if ν is unavailable.
- **`authored`/`spectral`** (curved) — per layer `fit_layer_subspace` (PCA frame +
  RBF surface), the μ-centered share bake, and a per-layer `origin`
  (`invert_parameterization` of the neutral mean; a flat fit's foot is coord 0).
  `spectral` passes `smoothing` (default `"auto"` → GCV) into `fit_rbf_smoothed`;
  authored stays exact, so an authored node is an exact steering target. Per-layer
  λ/edf ride the sidecar as `rbf_smoothing_per_layer`.
- **`auto`** (discover) — `select_topology` picks the geometry per model; the
  resolved `effective_fit_mode` drives the same flat/curved fit, and the sidecar
  records `resolved_fit_mode`, `topology_winner`, and the ranked
  `topology_candidates`. A curved winner carries its already-factorized
  `RbfFitPlan` from scoring into the final layer fit.

Curved manifolds skip DLS (no pos/neg polarity — per-layer signal is the
apply-time share alone). `min_nodes(k) = 2k+1` (`io/manifold_folder.py`) is the
curved poisedness floor; a flat `pca` fit needs only `k+1`. `--sae` reconstructs
each centroid through the SAE before the fit (fail-fast `SaeCoverageError`), one
layer resident at a time, popping raw centroid layers as decoded replacements
appear so both full rosters never coexist; the fitted subspace is model-space
regardless, so the hook never touches the SAE. The sidecar also stamps
`node_spread_per_layer` (`{str(L): tr(G_L)}`) — the concept's pre-DLS whitened
signal-by-layer profile, diagnostic only (nothing branches on it), distinct from
`mahalanobis_share` (the same spread restricted to the steerable subspace).

**Locking.** The long fit lock is target-scoped and stored outside the removable
folder, so two model/variant targets in one folder compute concurrently while
readers keep the previously published artifact. The folder-manifest lock covers
only the authoring snapshot (which hashes exact nodes plus the resolved
template/baseline and target clear epochs) and the final compare-and-swap
publication, so authoring edits, scoped clears, and rm/recreate all prevent stale
publication; discover fit-mode/hyperparameter overrides merge into `manifold.json`
inside that same lock, never as an unlocked pre-fit rewrite. Every fitted
tensor/sidecar read, replacement, clear, transfer, and folder removal takes a
digest-named pair lock under the *namespace parent*, never inside the removable
folder, and folder-wide lifecycle operations take the manifest lock then all
affected pair locks in sorted order — so an `rm` or HF stage-swap cannot delete a
lock inode or tear a read transaction. Payloads stage and fsync before a
sidecar-first pair commit; an interrupted pair is a cache miss or is repaired at
the next target fit. Downstream topology/covariance work holds a PID-backed
process lease that prune/GC honors without the exclusive stem lock, and pruning
runs only after the current transaction releases under a directory prune lock plus
one victim lock at a time. A failed fit after an override evicts only that
namespace/variant's resident manifold, folded profile, attached probe, and prefix
state.

Fit uses `ManifoldFolder.load(..., verify_manifest=False)`: it hashes the live
corpus into the capture and final identities and validates the requested tensor
without rereading every historical payload. Runtime fitted-tensor use, install,
and push keep strict integrity verification; publication hashes each newly
committed pair exactly once.

## sae.py, sae_training.py

`SaeBackend` is a tiny runtime-checkable Protocol —
`encode_layer`/`decode_layer`/`feature_count`/`feature_direction`, `release`,
`revision`, `layers`. `MockSaeBackend` serves CPU tests; `SaeLensBackend` adapts
provider releases; `LocalSaeBackend` adapts Saklas-trained artifacts.
`load_sae_backend(release, *, revision, model_id, device, dtype)` resolves
`local:<name>` without importing SAELens; otherwise it queries SAELens, validates
base-model compatibility, and resolves per-layer sae_ids through
`_canonical_layer_map` — Neuronpedia-hosted first, then narrowest width, then
smallest L0, because hosting supplies the label and the `maxActApprox` metadata
channel an unhosted pick would silently lose. Registry resolution is eager,
weights are lazy with a one-layer resident cache, and a valid fitted-tensor hit
never imports SAELens. An explicit `revision` is passed only when the installed
loader exposes `revision=`; otherwise loading raises rather than stamping a pin it
did not honor. Errors: `SaeBackendImportError`, `SaeReleaseNotFoundError`
(difflib suggestions).

The capture surface is the transformer-block output, so only
residual-post/block-output SAEs are valid: discovery omits explicitly named
attention/MLP/transcoder families (`_release_hook_kind`), and loaded hook metadata
plus decoder width are validated (`_validate_residual_hook`,
`validate_residual_width`) before a release becomes resident.
`select_runtime_layer` chooses the layer nearest 65% model depth (workspace
preferred); `list_sae_releases` discovers local artifacts and compatible registry
rows without loading weights; `sae_device_str` normalizes the device string.

`sae/<id>` steering reads `W_dec[id]`; feature probes read the encoder channel
outside the `Monitor`. The probe/gate channel is **normalized strength** —
`activation / maxActApprox` ∈ ~[0,1], the Neuronpedia corpus-max unit, so features
read apples-to-apples like the lens probes' mean fitted-layer probability; raw
activation is the unit only when no metadata is cached (offline or unlisted
feature), and `ScalarReading.unit` says which. Feature metadata is fetched lazily
from Neuronpedia at validate/pin time and through a batch backfill the dashboard
calls between generations — never inside the decode loop — and persists through
`io/sae.py::save_sae_feature_meta`.

`sae_training.py::train_residual_sae` trains a native one-layer ReLU SAE from
block-output token activations under model inference mode; decoder rows are
unit-normalized so L1 cannot be evaded by rescaling.

## jlens.py, jlens_fit.py

The Jacobian lens (Gurnee et al., "Verbalizable Representations Form a Global
Workspace in Language Models", Transformer Circuits 2026): `J_l =
E[∂h_final/∂h_l]` per source layer. `jlens.py` holds everything on a runtime path
and imports nothing beyond torch and the error taxonomy, so a per-step import is a
`sys.modules` lookup; the estimator lives in `jlens_fit.py` and stays reachable as
`jlens.fit_jacobian_lens` through a lazy module `__getattr__`, so importing the
readout never drags the backward-pass code in.

**Artifact + readout (`jlens.py`).** `JacobianLens` holds the fp32 matrices:
`transport(h, layer)` maps a residual into the final basis; `token_direction(v,
unembed)` is `W_U[v] @ J_l` per layer — the profile-shaped direction behind
`jlens/<word>` steering atoms; `lens_logits` is the full readout `W_U · norm(J_l
h)` (matvec in the unembed's own dtype — an fp32 `W_U` copy would be GBs).
`readout_probabilities` is the shared calibration primitive, computed once per
logits matrix and passed to pinned probes, per-layer cards, and the aggregate
through the `*_from_probabilities` helpers. `aggregate_readout(logits, depths,
top_k)` aggregates a stacked `[L, vocab]` readout: per-layer softmax calibrates
away the cross-layer logit scale, then per token `strength = mean_l p_l(v)` plus a
depth center of mass and spread weighted by the same `p_l(v)`. The readout is
sharp, not diffuse, and what changes over depth is *which* token leads, so a
token's probability profile over depth is its depth signal; `p_l` is the one unit
behind every lens statistic. Top-k selects on aggregated full-vocab strength (a
per-layer top-k union would miss a mid-pack-everywhere token) and CoM/spread are
evaluated only for the selected columns, returned as `[(vocab_id, strength, com,
spread)]` in one batched host transfer. `token_readout_stats(logits, depths,
token_ids)` is the single-token restriction of the same calibration — the math
behind `jlens/<word>` probe readings and gate scalars — returning `(strength, com,
spread, per_layer[p_l])`. `resolve_word_token` maps a word to its single vocab id
(leading-space piece first, decode-and-compare check, `MultiTokenWordError` with
the pieces otherwise). `sparse_nonneg_decompose` is the J-space split: greedy
pursuit against the never-materialized dictionary `W_U J_l` (scores are the
composed matvec, normalized by chunk-computed atom norms; only selected rows
form), with a tiny projected-gradient NNLS re-solve per step, returning
`JSpaceDecomposition(layer, share, tokens)`. `JacobianLens.merge` is the
non-mutating n_prompts-weighted combiner, `merge_into` recycles a caller-owned
tail, `union_layers` combines same-corpus layer shards. `SKIP_FIRST_POSITIONS =
16` (attention sinks) and `DEFAULT_CHECKPOINT_EVERY = 25` live here. Errors:
`JacobianLensError` (422), `LensNotFittedError` (404), `MultiTokenWordError`
(400).

**Estimator (`jlens_fit.py`).** `fit_jacobian_lens` is the **only backward-pass
code in saklas**. Everything else runs under `inference_mode`, and inference
tensors never re-enter autograd, so the fit builds its own `torch.enable_grad()`
forward with a grad-seeding output hook on the first fitted block. Consecutive
ragged prompts share one right-padded graph (`prompt_batch`; CPU/CUDA default 4,
MPS 2) and batched VJPs recover `dim_batch` output rows per backward without
replicating the forward. For output dimension `r` the cotangent is injected at
every valid target position, source positions are averaged within a prompt, and
prompt Jacobians are summed — exact equal-prompt weighting even for ragged
lengths; the first `SKIP_FIRST_POSITIONS` and the final position are excluded from
both cotangents and the source mean. Backends without batched-VJP coverage fall
back to exact unreplicated scalar VJPs, and `SAKLAS_JLENS_VJP=replicated` keeps the
reference replicated estimator. Grads come from `torch.autograd.grad(final,
sources)` — never `backward()` + `retain_grad()`, whose `.grad` accumulation across
the multi-backward loop would corrupt the rows — which also stops the graph walk at
the shallowest requested source, so a band-restricted fit never backprops below its
lowest layer. A terminal hook captures the target residual and aborts the forward
before final norm and the full-vocabulary head.

**Memory + failure discipline.** Row blocks stage in byte-budgeted device+host
stripes (up to `_ROW_STRIPE = 256`, never below one active VJP block) that
validate and commit directly into the persistent CPU fp32 accumulator; allocation
failure halves the stripe independently of prompt width. CUDA double-buffers
stripes with two ordered event-tracked slots so D2H overlaps the next backward,
discarding an uncommitted suffix on transfer failure and dropping to one slot
before narrowing estimator batches. A later VJP OOM rebuilds the graph at the
first uncommitted row, and prompt/dimension widths halve independently. On MPS the
pass loop drains the command queue every `_MPS_SYNC_EVERY_PASSES` (4) — Metal
reports queue exhaustion as an *asynchronous* command-buffer error that silently
zeroes work rather than raising, so an unsynced loop corrupts the fit — with a
zero-row guard before any stripe commits. Cancellation is checked before and after
every output-dimension block, draining queues and pending transfers before
abandoning partial sums. Teardown removes the fit hooks, clears their retained
graph and stripe buffers, and flushes the allocator cache: a long-lived server
must not carry the backward working set into the next decode.

**Checkpoint + resume.** `checkpoint_accumulator_cb` fires every
`DEFAULT_CHECKPOINT_EVERY` prompts without fracturing a healthy microbatch; the io
layer normalizes live sums and merges any prefix one layer at a time while
streaming fp32, so checkpointing never materializes a second complete lens.
Checkpoints are self-contained (`base_n_prompts=0`) and stamp the token-id hash of
the prefix actually consumed, so a later corpus extension resumes honestly.
Payloads and their directory entries are durable before the pointer publishes, and
the checkpoint stays the recovery point until the commit succeeds. Resume
transfers ownership of the loaded prefix matrices into the estimator, scales them
to raw sums in place, and accumulates the tail into that same storage — one full
fp32 matrix set, not a base plus an equally large tail. Sidecars are compared
before either payload is paged, so a farther self-contained checkpoint wins
without materializing the older durable lens, and a checkpoint that passes header
preflight but fails digest/finite validation falls back to the durable prefix.
When the terminal checkpoint is already the complete lens, finalization re-fsyncs
its immutable per-layer generations and repoints the durable sidecar rather than
rewriting the artifact. Persisting a missing-layer union reuses existing shard
pointers and serializes only newly fitted layers; because one corpus-progress
identity covers every layer, a prefix extension of a strict durable subset is
rejected before payload load (request the full layer set, or `force=True`). The
fit is compute-bound; restricting source layers is the one real wall-time lever.

## hooks.py

`HiddenCapture` — the capture buffer. Public surface:
`attach`/`attach_persistent`/`attach_batch_tail`/`detach`/`clear`, the retention
setters `set_retention`/`set_incremental`/`set_aggregate_tail`/`set_tail_with_sink`,
the readers `stacked`/`latest_per_layer`/`per_layer_buckets`/`tail_slice_at`/
`batch_tail_slice_at`, the post-forward `fire_step_sink`/`ingest_persistent`,
selective-prefill `set_prompt_positions`/`prompt_stacked`, and `is_transient()`
(true iff transient per-gen forward hooks are registered — the compiled-clean
routing gate). No caller reaches into private state: `per_layer_buckets()` is a
plain attribute return (zero per-token cost on the WS read path), and
`set_tail_with_sink(depth, sink)` is the only setter that arms a deep tail ring
*and* a per-token sink together.

`SteeringHook` carries per-layer groups `(trigger, subspace, domain,
target_coord, origin_coord, along, onto, kappa)` and dispatches on the shape
`recompose` armed, cheapest first — the first three consult no `TriggerContext`
at all:

1. `_const_single` — one always-active pure-push affine group: a single in-place
   `hidden.add_(c)`.
2. `_single_affine_lowrank` — the same group with an `!` ablation: the fixed push
   plus a projection restricted to the nonzero-κ rows (the common single-ablation
   case is an elementwise dot + axpy, avoiding MPS's expensive tiny-matmul
   dispatch).
3. `_single_affine_fast` — the general single-affine fallback: one
   `subspace_inject` + `copy_`, no group loop, no foot state.
4. the general path — multiple groups, any curved manifold, or a gated/phased
   trigger: read `_ctx`, skip inactive groups, thread the per-token foot
   (`_manifold_feet`; cold seeds at `origin` with `_MANIFOLD_COLD_GN_STEPS = 4`
   Gauss-Newton steps, warm takes one).

Cases 1–3 are a fixed tensor-op sequence identical every decode step, which is
what makes a *steered* generation StaticCache / `torch.compile` eligible. Case 4
keeps triggers and probe gates dynamic and forces the eager DynamicCache path.
`subspace_inject` returns fp32 and the hook's `copy_` does the model-dtype
downcast, so there is no per-fire cast temporary.

`SteeringManager` owns `subspaces` (dispatch-synthesized merged affine, one per
trigger group, via `add_subspace`) and `manifolds` (curved, via `add_manifold`).
`apply_to_model` lowers both to per-layer entries: share-weight (mean-1
normalized via `_normalize_shares_mean1`; `_manifold_layer_shares` prefers the
baked `mahalanobis_share`, else the Euclidean `‖eval_rbf(node_params)‖_F`),
orthogonalize the affine subspace against curved spans
(`_orthogonalize_affine_against` — curved wins shared directions, carrying κ
through the re-orthonormalization), and enforce `_CURVED_ORTHO_TOL = 1e-3`
between two curved manifolds (`OverlappingManifoldError`).
`reset_manifold_feet` cold-starts followers per generation.

**Compiled offsets.** `install_persistent_offset_hooks` /
`install_persistent_capture_hooks` build branchless per-layer buffers *before*
`torch.compile`, so they are traced into the captured graph;
`adopt_compiled_offsets` hands them to the manager. `compute_static_offsets`
returns `{}` for unsteered, a `{layer: (D,)}` map for the static-affine pure-push
case, and `None` for anything needing the per-token kernel (curved `%`, a gate or
phase trigger, an `!` ablation). `write_compiled_offsets` pushes them in place
(`copy_` — reassigning would change tensor identity and force a retrace) and
zeroes untouched layers; `detach_transient_hooks` then removes the ctx-consulting
hooks so nothing double-applies. Compiled/eager parity is structural: both paths
contract the same lowering (`_lower_affine_subspaces`) through the same
`_affine_push_offset`.

`all_fast_path()` is true iff no *transient* steering hook is attached — both
genuinely unsteered and steering carried by the persistent offset buffers, so it
is a graph-capture eligibility signal, not an "is this unsteered" test (callers
needing that distinction read the session's `_steering_uses_compiled_offsets`
alongside it). `static_steerable()` is true when every attached hook is the
static single-affine fast path. The session ORs them into `generate_steered`'s
`use_static_cache`.

**Gain.** The per-layer share normalizes to mean 1 (`Σ_L share_L = n_layers`) and
`eff_along_L = share_L · gain`. The along-gain is path-specific.
`_SUBSPACE_GAIN = 16.0` scales the **affine** path, where the whitened-unit target
makes push a free, overshoot-safe magnitude; it is live-calibrated on gemma-4-12b
so `α ≈ 0.5` lands the coherent band for tight concepts and personas alike and
`α ≈ 1.0` is the strong/over-steer zone where a hard persona breaks (dial α down
per target). `_MANIFOLD_ALONG_GAIN = 4.0` scales the **curved** path, where the
target is raw node coords so `eff_along` is a *fraction of the way to the node*
(`1.0` lands on it, `norm_cap` bounds off-domain RBF extrapolation); calibrated on
a clean stateless gemma-4-12b `months%january` sweep. For **periodic `BoxDomain`**
fits `eff_along` is clamped to `[0,1]` and share-weighting is dropped
(uniform per layer), because share × gain sends many layers past 1 on a ring and
each would wrap to a different node; non-periodic curved fits keep the
share-weighted unclamped path. `_MANIFOLD_ONTO_GAIN = 0.5` scales `onto` only,
calibrated on a gemma-4-12b `emotions%dominant` onto sweep so `onto ∈ [0,1]` is a
usable dial with `1.0` a coherent ceiling rather than the over-steer edge;
`eff_onto_L = clamp(onto · share_L · _MANIFOLD_ONTO_GAIN, 0, 1)`. There is no
lever and no `[0,1]` clamp or water-fill on affine `along` — a high-share layer is
meant to overshoot, and the de-rogued whitened coords keep it controlled.
`_SUBSPACE_GAIN` remains tagged a prototype.

## monitor.py, monitor_attach.py

`Monitor` is the one read-side class for geometry probes: every probe is a
`Manifold` and every read — live per token and the end-of-generation aggregate —
produces one full `ProbeReading` (`coords` + `fraction` + `nearest` + `residual` +
`assignment` + `membership`, plus per-layer traces and the per-axis
`depth_com`/`depth_spread`). Hook-driven, fp32, Mahalanobis-only: the whitener
must cover every probed layer (`covers_all`) or scoring raises `WhitenerError` —
there is no Euclidean readout, because on real LMs the Euclidean metric is
rogue-dominated and would be a wrong answer, not a degraded one.

One read shape, two execution paths. The whole **flat** roster is scored together
per layer in `_score_flat_batched` — one `Σ⁻¹h` Woodbury apply, stacked /
block-diagonal matmuls, and a single host transfer, scattered into global
per-probe slots. **Curved** probes run the per-probe `_score_probe_full`
(`invert_parameterization` foot solve, warm-started across decode tokens from the
previous foot when `enable_curved_warm` is set — the sequential live path).
`_layer_geometry` (module-level in `monitor_attach.py`, called directly on the hot
path) yields the M-orthogonal **fraction** `sqrt(gᵀ M_R⁻¹ g)/‖x‖_M` (`g = B Σ⁻¹ x`),
the whitened query for the nearest `M_R`-metric node, and the M-projection reduced
coord `c = M_R⁻¹ g`. From there a flat probe recovers `coords` through the affine
reduced→domain map (off-surface `residual` identically 0) and a curved probe
through the foot solve with a real normalized `residual`. Coordinates are
**domain-frame**: each layer's reduced coord is in that layer's `‖δ_L‖` units and
is mapped to the shared domain *before* share-averaging, so at rank-1 it is the
pole-normalized coordinate (`1.0` at the positive node). `_depth_stats` stamps
depth center of mass and spread at both assembly sites from mass
`share_weight_L · |coord_L[axis]|` over depths `layer/(n_layers−1)` — pure
host-side arithmetic over values already transferred; the `n_layers` ctor kwarg
supplies the denominator, and unset leaves the stats empty. There is no
`layer_means` input: the readout centers on each fit's own `LayerSubspace.mean`
and the whitener carries the neutral statistics the metric needs.

The **neutral anchor competes in `nearest`** as a virtual candidate
(`NEUTRAL_LABEL = "neutral"`): every fit is neutral-anchored, so neutral is a
point in the same whitened metric as the nodes (`_LayerWhiten.neutral_white` — the
zero vector for an affine fit, the baked `origin` mapped through
`eval_at → basis → chol` for a curved one), never a stored node. The per-probe
path appends it as the `K`-th cdist row; the batched flat path gets it free from
the zero pad column. `_attach_manifold_probe` sets `inject_neutral = NEUTRAL_LABEL
not in node_labels`, so a real node named `neutral` keeps sole ownership.

`flat_scalars` (one staticmethod) is the single gate emitter, writing from a
readings dict: `"<name>"` (coords axis 0) and `"<name>[i]"` per axis,
`"<name>:fraction"`, `"<name>@<label>"` = −distance per nearest node,
`"<name>~<label>"` = soft-assignment probability, `"<name>:membership"`. Nearest
distance is reported in units of the probe's **typical label spacing**
(`AttachedManifoldProbe.label_scale`, the median node nearest-neighbor whitened
distance), so a threshold reads as "within N label-spacings" and transfers across
probes where a raw whitened distance would not; because the scale is a per-probe
constant, `nearest` still *ranks* by raw distance. Every probe — flat and curved —
carries coords and nearest, so the gate grammar is uniform.

**Fuzzy reads** are the soft counterpart of `nearest`/`residual`: `assignment` is
a `softmax(−d²/(2τ²) − R·log(τ))` node posterior (a proper isotropic R-D
Gaussian-mixture posterior with uniform prior) and `membership` is the
`exp(−residual²/2σ²)` tube-fit density ∈ `[0,1]` (`1.0` for flat / σ-less fits).
The per-candidate bandwidth τ and the precomputed log-volume bias `−R·log(τ)` are
computed at attach (`_compute_assign_bandwidth`): τ is a curved fit's σ-field
mapped into the whitened metric or a flat fit's local layout scale, and the bias
is the Gaussian normalization without which a diffuse-τ candidate swallows
probability regardless of distance. The hot path gains one softmax and one add
over distances it already computes, folded into the existing single host transfer.

Entry points: `score_per_token` (primary — `(aggregate readings, per-token axis-0
coord stream)`), `score_single_token{,_per_layer}`, `measure_from_hidden`,
`score_stack`, `score_aggregate` (pools the last content token and runs the *same*
`_score_probe_full`, so the aggregate is bit-identical to the live read at that
index), `plan_gate_scalars`/`score_planned_gate_scalars` (the gated-subset path
with its plan cache — the large-K guard), `add_probe`/`remove_probe`,
`probe_geometry`. `Monitor.history` keeps per-probe coordinate tuples
(`accumulate_readings` folds an aggregate in; the in-flight gate/stream path
deliberately does not touch it).

`monitor_attach.py` holds the once-per-`add_probe` algebra, entirely off the hot
path: `AttachedManifoldProbe`, `_LayerWhiten`, `_build_whitened_factors`,
`_attach_manifold_probe`, `_compute_assign_bandwidth`, `_layer_geometry`, the
affine-coord helpers, and `_woodbury_apply`. `monitor.py` imports back from it; it
must never import `monitor.py`.

## instruments/

The read-side protocol — the read analogue of `SteeringComposer`. One contract
over three families: **geometry** (Monitor subspace probes), **lens** (the
J-lens readout channel), **sae** (feature reads). `session.instruments` is the
registry `{"geometry", "lens", "sae"} → Instrument`, and it is the family
*enumeration*: the probe-hash roster, gate preflight, and the server's per-family
dispatch all derive from it, so a fourth family is one registry entry plus its
implementation. `session.geometry`/`session.lens`/`session.sae` are the typed
public faces.

`types.py` — the shared vocabulary. **`GateRef`** + `parse_gate_ref` is the ONE
place probe-gate key discrimination lives (it inverts the family
`Monitor.flat_scalars` emits); the per-step runtime lookup stays a plain `dict.get`
on the verbatim string, so `GateRef` exists for *composition preflight*.
`validate_gate_channels` rejects a channel a family can never produce —
`@when:sae/123:membership` is `UnsupportedProbeChannelError` (400) at compose time
instead of a silently inactive gate; a *supported* channel merely absent this step
stays quietly inactive. **`ScalarReading`** is the honest one-channel reading for
lens and SAE: `value` + an explicit `unit` (`mean_token_probability` /
`activation_over_max` / `raw_activation`) + `per_layer` + a `DepthSummary` that
carries its mass `basis`, because `depth_com` means three mathematically unrelated
things across families. Its `to_dict()` is the family-native wire shape inside the
measurement envelope; `to_probe_reading()` is the compatibility projection, used
only at the two compat boundaries (`TokenEvent.probe_readings` and the
OpenAI/Ollama vendor extension) via `as_probe_reading`. `reading_axis0` /
`reading_per_layer_axis0` are the cross-family accessors; `scalar_gate_keys` is
`flat_scalars`' single-axis counterpart, emitting only `<name>` and `<name>[0]`.
Also here: `InstrumentPlan` (declared capture demand, not mechanics),
`InstrumentPrep`/`LensPrep` (the generation-boundary source snapshot),
`InstrumentBinding` (the immutable per-generation source/spec snapshot),
per-family `LiveState` subclasses (`GeometryLiveState`/`LensLiveState`/
`SaeLiveState` — user intent, deliberately not unified with runtime residency),
`ReadRequest`, `AGG_TAIL_DEPTH = 8`, and `next_prep_token` (one process-wide
sequence).

`protocol.py` — the `Instrument` / `InstrumentRun` contracts. A capture
transaction is the uniform sequence **`close_run → prepare → plan → bind`** at
both generation boundaries: source refresh strictly precedes planning, because
adoption may rewrite live probe specs, and a plan or freeze read off the live
registry could pair the prep's lens with a replacement's layers. Every `prepare`
raises on a still-bound run (a stale pin would short-circuit the very refresh the
step exists for); every prep carries a per-preparation `token` its plan echoes;
`bind` refuses a plan/prep pair whose tokens or families disagree, and prep-less
or foreign-family calls raise. `InstrumentRun` owns the generation-scoped state —
the frozen `binding`, step stashes, per-generation flags — and `observe(step_id)`
memoizes **only while bound**, since the idle passthrough run persists
indefinitely and a repeated step id with different hidden states must not read
stale. Full-roster reads prime the memo (`prime_observation`); partial reads
(gating scalar subsets, `coords_only` lean rows, `only=` restrictions) never do —
a partial reading served as the full `observe` result is the completeness trap.
`observe_aggregate` backs geometry's finalize aggregate and the batch per-row
lens/SAE aggregates. Every capture transaction reaches
`session._close_instrument_runs()` through nested `finally`s (the generation, the
batch, the joint-logprob replay), so a raising hook detach cannot leak a bound run
and pin a stale lens between generations.

Each family owns a **reentrant leaf `state_lock`** (outer locks such as
`_gen_lock` / `_model_exclusive` are always taken first, and it is never taken on
a per-token path). It covers roster mutation (attach/detach/`try_detach`, the
SAE metadata backfill's whole-dict replacement, the geometry fit-promotion and
failed-override walks, the lazy `set_whitener` factor rebuild) against the
coherent read surfaces (`names`/`specs`/`manifolds`/`probe_layers`/`probe_hash`/
`live_layers`, the `plan`/`bind` roster reads, the idle-run reads) — an un-locked
reader iterating a live roster raises under a concurrent detach. The lens's whole
`prepare` snapshot is one atomic transaction under that lock, so specs and live
device stack cannot come from two different lens identities.

**Mutation timing.** Lens and SAE measurement guards consult
`_measurement_specs()`, so a mid-generation detach cannot change a bound
generation's roster — it applies at the next generation. Geometry has no roster
snapshot (Monitor scoring walks the live probe dict), so `GeometryInstrument`
attach/detach take `_model_exclusive` and a served detach during a running
generation **rejects with retry-shortly semantics** rather than racing.

`lens.py` — `LensInstrument`. Owns the probe registry, the live-readout runtime
(device-resident `J_stack` + unembed/norm modules), the per-forward stash and
display readings, the per-generation active flag and disk-identity pin, and the
read surfaces: `attach`/`detach`/`try_detach`/`specs`/`probe_layers`,
`score_probes` (readout-channel `ScalarReading` synthesis), `gate_scalars`
(once-per-forward logits + stash the display step reuses), `score_aggregate`
(tail-ring pooled last content token), `live_readout_step`, `authored_capture`
(per-row computation only), `enable_live`/`disable_live`/`live_layers`,
`token_readout`, `probe_hash` (the `jlens-readout-v2` identity digest), and
`validate_gate` (strength axis only). Shared J-lens *primitives* — the logits-row
builder, depth/decode caches, the transport stack, the `jlens` disk-identity
property — stay on the session, reached through the session back-ref exactly like
`SteeringComposer`.

`sae.py` — `SaeInstrument`, the same shape: probe registry, live-discovery config
(`{layer, source}`), per-forward stash (one encode shared by gates, pinned probes,
and the live display), per-generation flag, the same read surfaces, and
`probe_hash` (`sae-readout-v2`, unit-carrying). Backend *residency*
(`_sae_backend`/`_sae_layer`/`_sae_width`, `_require_sae`, `_encode_sae_hidden`),
the Neuronpedia metadata cache and fetchers, the train/load/unload lifecycle, and
the offline `sae_token_readout` stay session-side — residency is runtime state
shared with steering atoms, not probe intent. **An SAE probe emits exactly one
channel**: `<name>` / `<name>[0]`, the normalized strength. There is no
`:fraction` or `:membership` — a feature activation is not a geometry reading, and
a gate on those channels is a preflight error, never a silently constant
comparison. The binding resolves `max_act` at bind, so a metadata backfill cannot
change a running generation's strength unit.

`geometry.py` — `GeometryInstrument`, the thin adapter giving the Monitor family
the same face: `attach` (the exclusive-GPU whitener touch + resolve +
`monitor.add_probe` flow), `detach`/`try_detach`, `specs`, `names`, `manifolds`,
`plan`/`bind`, `live_state`/`set_live` (the CAA live toggle), `token_readout`,
`validate_gate` (every channel — axes, fraction, membership, distance,
assignment), and `probe_hash` (the baked-tensor / subspace-geometry digest). The
Monitor engine and the capture modes stay in `monitor.py` and session/HiddenCapture
state — folding them in would combine an orchestration extraction with an engine
rewrite.

## triggers.py, steering.py, steering_expr.py, steering_composer.py

`triggers.py` — `Trigger` (frozen): phase flags plus an optional `ProbeGate`.
`Trigger.active(ctx)` consults the flags and, when gated,
`ctx.probe_scores[gate.probe]` against `score <op> threshold`. Factories
`first`/`after`/`when`. `TriggerContext.probe_scores` is filled by the per-step
callback and cleared on `reset()`; a gated trigger is inactive during prefill and
for a missing probe (no raise). `ProbeGate` is frozen and hashable so identical
gates compose under equality, and `ProbeGate.probe` is the canonical scalar key
verbatim.

`steering.py` — `Steering`, frozen: `alphas: {name: alpha_or_entry}` plus an
optional `thinking` override and a default `trigger`. `Steering.from_value`
accepts `str | Steering | None` only; a dict raises. An entry can carry its own
trigger as an `(alpha, trigger)` tuple. It carries no per-call metric override —
`~`/`|` projection is Mahalanobis-only.

`steering_expr.py` — the unified grammar and IR compiler, one parser plus
formatter for every input surface (Python, YAML, HTTP, CLI). The module docstring
carries the authoritative grammar; the shapes that matter here: `@<preset>` phase
tags, counted decode windows `@first:N` / `@after:N` (the `:N` is required and is
what separates `@after:5` from the bare post-thinking `@after`), probe gates
`@when:<probe><op><threshold>`, and their composition `@<phase>&when:...`.
Between them these cover every `Trigger` a public factory builds, so
`parse_expr(format_expr(s)) == s` round-trips; a `Trigger` hand-built outside that
set has no grammar form and `format_expr` **raises** rather than emitting an
unparseable string. Term markers: `ProjectedTerm(coeff, trigger, operator, base,
onto)` (materialized into derived profiles before the hook layer), `AblationTerm`
(`!atom`, default coeff 1.0, does not compose with `~`/`|`/`%`), `ManifoldTerm`
(`along`, `onto`, position; `_expand_along_onto_coeffs` yields a 1- or 2-tuple).
Gate probe strings are stored verbatim in `ProbeGate.probe` across every channel
shape, so the runtime gate is a plain dict lookup and the parser is the only place
the discrimination lives. Bare atoms resolve through
`io.selectors.resolve_bare_atom`'s ladder (a bipolar pole is itself a node label,
lowered to an affine `%` push); `referenced_selectors` backs install-time checks.

`steering_composer.py` — `SteeringComposer`, the steering-resolution and stack
collaborator, instantiated once in `SaklasSession.__init__` after the Monitor. It
solely owns the LIFO stack (`_stack`), projection materialization, pole-alias
resolution, artifact registration, steering lowering, and hook install/rebuild.
It reaches session state through `self._session` at push/pop frequency, off the
per-token path; the one near-hot method, `build_gating_score_callback`, binds
`capture`/`monitor` to locals before the inner `def`, so the per-token gate path
gains no indirection. Push/pop call the session's `_rebuild_steering_hooks`
forwarder so stub overrides take effect. `compose_steering_entries` is the
dispatch: `AblationTerm` → ablation fragment; `ManifoldTerm` → affine `%` joins
the merge (`_affine_manifold_push`) or curved `%` → `add_manifold`; a plain
`(alpha, trigger)` → `ensure_profile_registered` → `fold_directions_to_subspace`
→ push fragment. Push and ablate fragments group by trigger,
`synthesize_subspace` runs per group, and `install_composed_steering` →
`apply_to_model`. There are **no** persistent steering hooks in the composed
sense — the transient hooks are removed after each generation, and the compiled
offset buffers are the separate static path. Manifold-implied roles aggregate
under soft-warn + highest-`|coeff|`-wins (`RoleBaselineMismatchWarning`). One
structured `_gated_refs()` pass (over `parse_gate_ref`) drives gate discovery and
channel-validates lens/SAE-attached gate references at generation preflight.

## scene.py

The cast-model turn stitcher. **Template autopsy** renders sentinel-content probe
conversations through the live chat template once per model and mechanically
slices out a `TurnGrammar`: prelude, per-seat `SeatWrapper`s (open / label site /
close), `SystemShape` (real turn vs gemma-style fold, or unsupported), generation
appendices (`gen_extra` under `enable_thinking=False`, `gen_extra_thinking` under
True; `None` means thinking-mode stitching is unsupported), content/system trim
flags probed with padded sentinels, `last_assistant_special`, think delimiters,
and the empirically probed `strips_history_thinking` (strip families render a
turn's committed thinking only while it is the last turn before the gen header).
`render_scene(grammar, turns, *, system, gen_seat, gen_label, gen_thinking)`
stitches arbitrary `(seat, label, text, thinking)` sequences — alternation is not
required, and labels are placed in *constructed* headers, so the label→turn
binding is positional by construction and a splice-collision class is
structurally impossible — plus a trailing generation header on either seat.
`validate_turn_grammar` is the load-bearing gate: byte-exact round-trip against
the template's own render (plain / gen / padded / system / closed-on-assistant /
thinking-gen cases). A passing family's alternating renders are bit-identical
through the stitcher, which is the extraction-baseline contract; a failing family
falls back with a warning (`SceneGrammarError`). `render_scene_raw` is the
marker-mode fallback (base models, Mistral) and the `build_chat_input`
base-model branch. `SaklasSession.scene_grammar` is the lazy per-session
autopsy+validation (`None` = fallback), and `build_chat_input(scene=, gen_seat=)`
routes through the stitcher when a grammar is present — the only path that can
open a user-seat generation prompt. Errors: `SceneRenderError`,
`SceneGrammarError`, `SceneThinkingUnsupportedError`.

## generation.py, generation_finalizer.py, token_callback.py, token_payloads.py, measurements.py

`generation.py` — token-by-token decode with KV cache under
`torch.inference_mode()`. A model that ignores `past_key_values` (e.g. talkie)
flips `no_cache_mode` (O(N²), one warning). `GenerationConfig` (frozen) holds
session defaults and a per-call `SamplingConfig` composes into a local copy.
Top-p runs via `torch.topk`, not a full-vocab sort; `top_k` (default 1024 cap) is
a hard candidate-pool cap applied before top-p (llama.cpp/Ollama order).
`generate_steered` accepts `seed`, `stop`, `logit_bias`, presence/frequency
penalties, `logprobs`, `score_callback` (probe gates), `step_callback`,
`forced_prefix`, `steering_active`, and `want_perplexity`. `forced_prefix` forces
the first N decode tokens while the multinomial draw still runs, so re-seeding
stays bit-identical through the fork. `want_perplexity=False` skips the per-token
entropy `.item()` host sync when no consumer surfaces perplexity. Stop-sequence
matching keeps only a bounded tail of the emitted text (`_stop_keep =
max(len(s))−1` chars), so the per-token match is O(tail+emit). `detect_base_model`
routes flat (`raw=True`) generation; `supports_thinking` /
`_detect_think_delimiters` round-trip Qwen/Gemma `enable_thinking` with channel
(gpt-oss) and bracket (Mistral-3) fallbacks, and the `_ThinkState` machine plus
`GenerationState` drive streaming.

**Step identity.** The decode loop owns ONE `step_id` per forward —
`len(generated_ids)` before that forward — and hands the same value to the capture
sink (`step_callback`), the gate callback (`score_callback`), and the token tap
(the internal 7-argument `StepTokenCallback`). Buffered partial-UTF-8 emits carry
the *flushing* forward's step, matching their reading semantics. The instrument
workers' per-forward stashes are step-keyed (`stash["step"] == step_id`), so
staleness is structural and reuse is idempotent; the matrix-granular gate→display
reuse (band logits, partial row overlap, the shared SAE encode) rides beneath that
predicate.

`token_callback.py` — the typed contracts. The public `TokenCallback` is
six-argument; the internal `StepTokenCallback` adds the trailing `step_id`, which
`SaklasSession._token_tap` absorbs before invoking user callbacks.
`TokenConsumer` + `TokenConsumerOptions` (`live_scores`, `per_layer_scores`,
`lens_readout`, `sae_readout`, `perplexity`) are how a consumer *declares* what it
reads, which is what lets the session skip work nothing consumes;
`consumer_options` reads them off a callback.

`token_payloads.py` — `TokenProbePayload` shapes per-token read data for the
callbacks (`build_token_probe_payload`, `merge_readings`, `to_token_payload`),
folding geometry/lens/SAE channels into one payload without a second softmax.

`measurements.py` — **the** wire record for read-side data, carried by the native
WS `token`/`done` frames, loom token rows, and the token-replay endpoints:
`{version, scope: token|aggregate|replay, provenance: captured|replayed, scores,
per_layer_scores, instruments{geometry, lens, sae}}`, built by
`build_measurements`. Each family block carries its own `binding` (source +
recipe steering), its family-native `readings`, and its native `readout` (lens
per-layer + aggregate; SAE features). The flat `scores`/`per_layer_scores` views
are the cross-family axis-0 projection.

`generation_finalizer.py` — `finalize_generation(session, ...)` decodes the
response, scores probes per the capture mode, updates session side effects, and
builds the `GenerationResult`.

## session.py

`SaklasSession` owns the model, the in-memory profile registry (`_profiles`), the
loaded-manifold registry (`_manifolds`), the `Monitor` (`_monitor`),
`SteeringManager`, `HiddenCapture`, the three instruments, generation defaults
(`session.config`), the loom tree, and a synchronous `events: EventBus`. Steering
resolution and the LIFO stack live on the `SteeringComposer`;
`ensure_manifold_loaded` and `ensure_profile_registered` are the single public
artifact-registration API, with narrow `_push_steering`/`_pop_steering`
forwarders where the session boundary is meaningful.

`from_pretrained(model_id, *, device, dtype, quantize, probes, system_prompt,
max_tokens=1024, dls=True, compile=False, compile_mode=None, cuda_graphs=False,
return_top_k=0, on_progress=None)` does the HF load, neutral/whitener wiring, and
probe bootstrap; `__init__` takes a pre-loaded model. Both call
`materialize_bundled_manifolds()` + `selectors.invalidate()` early.
`_RESPONSE_MAX_TOKENS = 256` caps each in-character response;
`_CORPUS_GEN_BATCH = 16` is the corpus-generation batch. Module-level
`_affine_manifold_push` builds per-layer basis rows + node-coord targets for a
flat manifold; `_KIND_TEMPLATES` / `_article` / `_system_for` / `_role_for` author
each node's elicitation frame (abstract → `someone_{slug}`, concrete → `{slug}`,
`custom` → no role swap — the custom system carries the frame).

**Extraction.** `extract(concept, baseline, *, kind="abstract", ...)` authors a
2-node discover-`pca` manifold (`hyperparams={"max_dim": 1}`) and fits it,
returning `(canonical_name, folded_directions(manifold))`. The corpus comes from
`generate_responses(concepts, kinds, *, roles=None, custom_system=None,
samples_per_prompt=1, ...)`, which has each pole answer the shared baseline
prompts in character with the concept riding both the system prompt and the
swapped assistant-role label; responses are emitted samples-outer/prompts-inner so
`response[i] ↔ prompt[i % k]`. Generation is batched — `_run_generator_batch`
left-pads a chunk into one `model.generate` with per-row independent sampling, and
`_run_generator` stays the single-shot `ModelHandle` seam. A monopolar
(`baseline=None`) call authors a genuinely 1-node folder.
`generate_neutral_responses` is the neutral-corpus sibling (no system beyond the
directive, standard label); `extract_from_corpora` is the corpus-in sibling;
`fit(folder)` is the multi-node delegate returning a `Manifold`;
`_fit_concept_manifold` is the shared tail. All gate against `GenState.IDLE`.

**Steering resolution (manifold-first).** `ensure_profile_registered(name)`
resolves a direction from, in order: an in-memory baked direction already in
`_profiles`; the reserved `jlens/` namespace (`register_jlens_direction` resolves
the word through the fitted lens, raising `LensNotFittedError` /
`MultiTokenWordError` rather than falling through to extraction); then a fitted
2-node `pca` manifold on disk (`try_fold_manifold` → `ensure_manifold_loaded` →
`folded_directions`, memoized into `_profiles`).
`_bootstrap_manifold_probes(categories, *, include_fitted_defaults)` is the
probe-roster bootstrap in one pass, two tiers: **tagged concept axes** (each
`default/` manifold in a requested category is fit-or-loaded and handed to the
Monitor under its bare name) and, when `probes is None`, **fitted multi-node
defaults** — attach-only, never fitting, so a 107-node manifold cannot block
startup. Both frontends get the same roster; an explicit `probes=[...]` list skips
the multi-node sweep.

**Generation.** `generate`/`generate_stream` are keyword-only, accept `steering:
str | Steering | None`, and return `RunSet` (`.first` is the
`GenerationResult`). `_generate_core` owns the `_gen_lock` re-entry guard, the
`_gen_phase` lifecycle (`IDLE`/`PREAMBLE`/`RUNNING`/`FINALIZING` via
`session.gen_state`), the steering context, `_begin_capture`/`_end_capture`,
finalization, and teardown — all through `_generation_transaction`, so the
gen-lock release sits behind the outermost `finally`. `_resolve_read_demand`
produces one frozen `ReadDemand` (`need_per_token`, `per_token_full_consumer`,
`gating_only_probes`, per-family gate keys, `lean_per_token`,
`final_probe_aggregate`, live flags, `capture_prompt`) so the caller and the
capture planner cannot form different opinions about the same generation;
`_begin_capture` then runs the instrument transaction, unions the declared
`latest_layers`/`tail_layers` plans, and picks the physical `CaptureMode`
(retention-mode selection stays session-side because the
`INCREMENTAL → set_tail_with_sink` upgrade is cross-instrument resource sharing).
A lens-only gate does not force per-token *monitor* scoring.
`generate_batch`/`generate_sweep` return `RunSet` (sweep builds the Cartesian
product as loom siblings). `generate_batch` shares one prefill across rows with a
common token prefix when steering is prefill-inactive (a term steers prefill iff
`trigger.prompt and gate is None`), which also gates push/pop cache invalidation;
an alpha sweep steers the prompt by construction, so it re-prefills per row.
Hot-path events: `GenerationStarted`/`SteeringApplied`/`SteeringCleared`/
`ProbeScored`/`GenerationFinished`/`ManifoldExtracted`; threaded subscribers hop
via `loop.call_soon_threadsafe`. `session.lock` is the server-owned
`asyncio.Lock`, distinct from `_gen_lock`.

**Probes.** `add_probe(selector, *, as_name=None, top_n=3)` routes a `jlens/` or
`sae/` selector to its instrument and everything else to the geometry instrument
(a 2-node `pca` concept folds to a rank-1 probe; a multi-node manifold attaches
whole). `remove_probe(name)` tries lens, then SAE, then geometry.
`session.probes` and `session.probe_hashes` are the roster views, built from each
instrument's locked snapshot and `probe_hash`.

**Authored-prefill capture.** Loom-attached generation also scores visible
authored spans on the prefill that already consumes them:
`_pending_authored_prompt_targets` selects authored node channels with no prior
token capture, `_match_authored_prompt_targets` locates their exact content-token
sequences in the rendered prompt and maps token `j` to its producer state at
`j-1` (the same semantics generated rows use), and `HiddenCapture` clones only
those rows at each active probe/lens/SAE layer. A KV-prefix hit translates full
positions into suffix-local rows and leaves cached producers untouched; selective
prompt capture forces transient hooks, because persistent buffers carry only the
latest slice. The first post-forward callback scores the rows in prompt order,
writes one `capture_authored` loom mutation per populated channel, then runs the
ordinary final-prompt sink. There is no second transformer forward. Captured
authored rows are immutable — rerolls reuse their data.

**Jacobian-lens surface.** `session.jlens` validates sidecar/live-weight identity
before loading, refreshes or evicts an already-resident lens when an external
process replaces its generation, and verifies each layer's payload digest while
promoting it. A generation refreshes that disk identity once at the capture
boundary and pins the resident lens through token scoring, gates, and final
aggregation; per-token paths never reopen the sidecar or its shards, and an
external pointer switch becomes visible at the next generation boundary.
`fit_jlens(prompts, ...)` pre-filters too-short prompts (so the saved `n_prompts`
counts consumed prompts exactly — what makes resume slicing sound), hashes the
filtered corpus, resumes a matching partial fit by default (`force=True`
restarts), checkpoints through the io layer, and gates under both a cross-process
per-model fit lock and `_model_exclusive` (forward *and* backward passes).
`jlens_readout(prompt, layers=, positions=, top_k=, aggregate=)` is the offline
readout; `register_jlens_direction(word)` lands `W_U[v] @ J_l` in `_profiles` for
the steering branch only; `jspace_decompose(selector, k=, layers=)` splits any
steerable direction against the lens dictionary.

**Token replay.** `jlens_token_readout(node_id, raw_index, ...)`,
`sae_token_readout`, and `geometry_token_readout` are the three loom-anchored
replays behind the dashboard token drilldown, all the same shape: fork-style
validation, rebuild the node's exact prompt render (`_prepare_input` — stamped
role labels + recipe thinking; `raw=` selects the flat render), append
`raw_token_ids[:raw_index]`, run one capture forward at the needed layers, and
read at the position that *produced* the clicked token. `apply_steering` replays
under the node's recipe steering — exact for always-active affine terms, since the
slide is position-independent; phase/gated terms do not reproduce on a bare
forward. The steering scope opens OUTSIDE `_model_exclusive` (composer push/pop
take `_gen_lock` blocking, so nesting would self-deadlock — the same ordering
`score_choices` uses). Geometry's replay is post-hoc by construction: it reads
aggregate-only generations and probes attached after the fact, and forces curved
warm-start off (a one-shot read must not seed from a prior generation's foot).

**Live toggles.** `live_probe_scores`/`set_live_probe_scores(bool)` — when off,
`_generate_core` masks every per-token monitor consumer at the source, so
generations run aggregate-only capture; probe gates still force the subset they
need. `enable_live_lens`/`disable_live_lens`/`live_lens_layers` — the selected
layers' `J_l` go device-resident, join the capture-widen union, force transient
capture routing, and arm a bounded tail ring when no probes are attached;
`_live_lens_readout_step` runs at the token tap **post-forward**, so no new
forward hooks are installed and `static_steerable` is untouched. The default layer
set is every fitted layer. Live-SAE discovery shares the generation's resolved
logit-alternative top-k width (falling back to `_DEFAULT_READOUT_TOP_K = 8` when
alternatives are disabled), so the live SAE config stores only its resident
`{layer, source}`.

## loom.py, loom_diff.py, tree_filter.py, transcript.py, joint_logprobs.py

`loom.py` — `LoomTree`, the engine-side conversation tree: nodes are turns,
children are alternative continuations, the active path is the model's context.
Node ids are 26-char ULIDs. `messages_for(leaf)` renders the path as chat messages
(`with_labels=True` adds each node's `role_label` and, when set, `thinking_text`
under a `"thinking"` key for the scene stitcher); `flat_text(leaf)` is the
base-model analogue. Five primitives: edit, branch, navigate, delete_subtree,
`regenerate`. `Recipe` is the per-node reproducibility receipt (steering
expression, sampling, thinking, seed, probe set + per-probe sha256), with
`overlay`/`invert_steering`/`compose_modifier` behind the auto-regen modes.
`LoomNode.thinking_text` is the turn's verbatim thinking block. `tree.cast:
dict[label, CastMember]` is the cast roster — `set_cast_member`/
`remove_cast_member` are decoration-tier (no conflict check), validate the label
as a role slug, and emit `LoomMutated(op="cast")` without node ids. Composition at
generation is the session's `_apply_cast_defaults`, the weakest tier.

`TREE_FORMAT_VERSION = 2` and `TOKEN_SIDECAR_FORMAT_VERSION = 2` are both
required **exactly**: tree, node, cast, and token-sidecar readers demand the
complete current writer shape and do not infer defaults from omitted fields.
Explicit nullable values (a transcript-imported `raw_token_ids=None`) round-trip
as null; `cast` is a required key (an empty roster emits `{}`). Visible token rows
own the canonical `measurements` envelope — the decode tap creates it for
generated tokens and selective prefill capture creates the same shape for authored
tokens, both stamping instrument source plus recipe steering inside the per-family
`binding`. `TokenScoreDict` rows are stored opaquely (the sidecar reader validates
only the node-level `tokens`/`thinking_tokens`/`raw_token_ids` fields), so token
rows ride the compressed sidecar and explicit save/load preserves the rich
channels without a separate cache. `set_authored_token_scores` installs one
authored channel atomically and emits `LoomMutated(op="capture_authored")`;
unlike the hot-loop `append_token` it advances the tree revision. Per-node token
blobs live in memory during streaming — `to_dict` omits them, `save` writes a gzip
sidecar. Mutators raise `MutationDuringGenerationError` (409) on conflict,
`UnknownNodeError` (404) / `InvalidNodeOperationError` (400) otherwise.

`loom_diff.py` — cross-branch diff primitives: `text_diff` (word-level via
`difflib.SequenceMatcher` → aligned `DiffSpan`s), `readings_diff` (per-probe
`Δ = b − a`, sorted by `abs(delta)`, carrying both originals so "moved" and
"appeared" stay distinguishable), `per_token_diff` (byte-offset alignment), and
`steering_delta` (a compact `"+0.2 calm"` edge label through the shared grammar).

`tree_filter.py` — the filter grammar for tree pruning, deliberately distinct from
the steering `@when:` grammar because the scalars differ: `@when:` gates per-step
readings during generation, this gates per-node aggregates the monitor stamped at
finalize. Clauses are `<agg>:<probe> <op> <threshold>` with `agg_op ∈
agg`/`any`/`last`; multi-clause is AND. `parse_filter` → `FilterClause` →
`filter_tree`, behind `LoomTree.filter_by_expr`. `FilterParseError` on any parse
problem.

`transcript.py` — transcript export/import for loom paths.
`SAKLAS_TRANSCRIPT_VERSION = 2` (per-turn `speaker:` = `role_label`, `thinking:` =
`thinking_text`, a top-level `cast:` block); `to_yaml`/`from_yaml` require that
exact version. Import modes: **default** (new top-level branch), **here** (child
of active), **merge** (attach the non-matching tail at the deepest matching
prefix). A turn with a `recipe` re-attaches as a *generated* node in its recorded
seat — provenance is recipe presence, the cast model's invariant — and the cast
block merges into the tree roster (the session's member wins, with a
`cast_conflict` guard note). Guards: a model mismatch refuses merge
(`TranscriptModelMismatch`), a system-prompt mismatch banners, and probe-hash
drift warns or raises `TranscriptProbeDriftError` under strict.

`joint_logprobs.py` — cross-branch joint logprobs for the loom comparison
surface. `compute_joint_logprobs(session, a_id, b_id)` reconstructs the shared
prefix (`_shared_prefix_len`, not re-tokenization), force-replays each branch
under its recipe, and assembles per-aligned-position rows with each branch's own
chosen-token logprob, the cross-branch evaluation, and `rank_changed` (the
canonical "steering shifted the head of the distribution" signal). `approx_kl` is
top-K-truncated `KL(P_A‖P_B)` — documented as approximate, since the tail is
unobserved. Cached on the session keyed by sorted `(a_id, b_id)`, invalidated by
tree mutations that rename or delete the nodes.

## scoring.py

Restricted-choice completion scoring — the logit read of a template, the
counterpart to a manifold fit (the manifold pools each candidate's slot-filled
*activation*; this reads its slot-filled *logprob*). `score_choices(session,
messages, choices, *, assistant_prefix="", labels=None, steering=None)` returns a
`ChoiceScores` set: per candidate `{n_tokens, sum_logprob, mean_logprob}` plus the
restricted-choice softmax over each (`prob_sum`, `prob_mean`) — length bias is
real when candidates differ in token count, so neither view is silently chosen.
Scoring runs against the **raw** model distribution (plain `log_softmax`,
temperature 1, no top-k/p), so the probabilities are the model's beliefs, not a
sampler reshaping. One batched teacher-forced forward per `_SCORE_BATCH` (16)
chunk — vocab is ~256k, so an unbounded batch would blow memory — with
`logsumexp` + a gather avoiding a second vocab-sized tensor. `_shared_prefix_len`
recovers each completion span, absorbing the boundary-token merge. `steering=`
wraps the forward in `session.steering(...)`: the distributional before/after.
`score_template(session, template, *, steering)` runs it over a
`TemplateFolder.score_inputs()` and returns one `ChoiceScores` per context.

## results.py, profile.py, sampling.py, events.py, errors.py, naming.py, role_templates.py, histogram.py

`results.py` — `GenerationResult`, `RunSet`, `TokenAlt`, `TokenEvent`,
`ProbeReadings`, `ProbeReading`, `ResultCollector`. `RunSet` is the list-like
multi-run shape (`node_ids`/`grid`/`.first`/`.to_collector()`/`.to_dataframe()`).
`TokenEvent` carries `thinking`, `logprob`, `top_alts`, `finish_reason`,
`perplexity`, `measurements` (the envelope), and `probe_readings` — a separate
typed field feeding the OpenAI/Ollama `x-saklas-probe-readings` vendor extension
with per-probe readings merged across families. `GenerationResult` carries
`prompt_tokens`, `finish_reason`, optional `logprobs`, `probe_readings`, and
`applied_steering` (the canonical expression, round-tripping through
`parse_expr`); `to_dict()` omits `hidden_states`. `ProbeReading`
(`coords`/`fraction`/`nearest`/`residual` + `assignment`/`membership` +
`fraction_per_layer`/`coords_per_layer`/`residual_per_layer` +
`depth_com`/`depth_spread`) is the **single** geometry reading shape for both the
live per-token stream and the end-of-gen aggregate; `residual` is 0 for a flat fit
and the normalized off-surface distance for a curved one, and the depth stats are
empty when the reading has no per-layer trace (lean modes) or the Monitor was not
given `n_layers`. `ProbeReadings` is an explicit multi-run summary shape for
notebook plots, not embedded in a generation result.

`profile.py` — `Profile` wraps `dict[int, Tensor]` with
`.layers`/`.save`/`.load`/`.to_gguf`/`.merged`/`.merged_with`/`.promoted_to`/
`.cosine_similarity(other, *, per_layer=False, whitener=None)`
(magnitude-weighted)/`.projected_away`. It also owns `save_profile`/`load_profile`
— the safetensors + slim JSON sidecar format for the per-model neutral/alignment
caches and folded-profile interchange. The sidecar is
`{format_version, saklas_version, method, tensor_sha256, provenance}`; readers
require exactly `PROFILE_FORMAT_VERSION` (`io/integrity.py`). An empty layer
intersection raises `ProfileError`.

`sampling.py` — `SamplingConfig`, the frozen per-call config with `merged_with`;
`None` on a field means "use the session default", composed at entry without
mutating `session.config`, so one session serves many concurrent requests without
request-local overrides leaking.

`events.py` — the synchronous `EventBus` and every payload dataclass it carries
(`Event` is the closed union). Subscribers run on the emit thread; a route that
needs an event loop hops inside its own callback.

`errors.py` — `SaklasError` is the base; every saklas exception multi-inherits
through it while keeping its stdlib MRO, so `except SaklasError` catches the
family and `except ValueError`/`RuntimeError` at existing sites still works.
`user_message()` returns `(http_status, text)`, defaulting to `(500, str(self))`.
This module is the single home for the engine's cross-boundary error classes —
defining them beside each raiser would make a low-level module like `mahalanobis`
or `profile` an import dependency of everything that catches its errors. Several
modules re-export the class they raise (`from saklas.core.mahalanobis import
WhitenerError` works); those aliases are import compatibility, not a second home.
`SteeringExprError` (400) covers parse failures; `SteeringCompositionError` (422)
covers post-parse composition failures with subclasses `ManifoldArityError` (wrong
coord count, raised in `add_manifold`) and `OverlappingManifoldError` (two curved
spans collide at a layer, raised in `apply_to_model`).
`UnsupportedProbeChannelError` (400) is the gate-preflight rejection.

`naming.py` — `canonical_concept_name` and the slug/bipolar-pole grammar. Pure
string normalization with no model or IO, so `core.session` and `io.selectors`
share it without an import cycle.

`role_templates.py` — custom assistant-role rendering. Most HF chat templates
hardcode role branches, so the universal trick is **render-then-splice**: render
with the standard `assistant` role, string-replace the per-family role header in
the rendered string, then tokenize — which keeps token-index stability for the
trigger system. `ROLE_HEADERS` is the `model_type` → `<before><label><after>`
registry; `apply_with_role` / `apply_with_per_turn_roles` are the entry points.
Mistral-3 has no role label in its rendered string, so it maps to `None` and
raises `RoleSubstitutionUnsupportedError`.

`histogram.py` — `HIST_BUCKETS = 16` and `bucketize(norms, buckets)`, collapsing
sorted per-layer norms into evenly sized groups for `manifold why` and notebook
plots.

## Invariants

- **Hot-path hooks**: no Python allocation, no `.item()`, no CPU sync, in-place
  only. The whole decode loop runs under `torch.inference_mode()`; the J-lens
  estimator is the only backward-pass code in the package.
- **Norms use fp32** — fp16 sum-of-squares overflows at hidden_dim ≥ 2048. This
  covers fit-time direction norms, the per-position norms inside
  `subspace_inject`, centroid differences, and the cached neutral activations.
- **Mahalanobis-only.** The activation-space fit, `~`/`|` projection, monitor
  reads, cross-model transfer, and `manifold compare` all require a covering
  whitener (`covers_all`) and raise `WhitenerError` otherwise. There is no
  Euclidean fallback.
- **One injection kernel.** Every steered layer runs `subspace_inject`. The
  static single-affine cases are a fixed tensor-op sequence, so StaticCache and
  `torch.compile` capture are eligible for unsteered (`all_fast_path`) *and*
  static-affine steered (`static_steerable`) generation; curved, gated, or phased
  steering keeps the ctx-consulting general path on DynamicCache and eager. Curved
  `%` is materially the slowest path — a per-token Gauss-Newton foot solve plus
  frame-rotation transport, an `n ≥ 2` fit hopping to CPU for the SVD on MPS, and
  no compile eligibility.
- **Share baked at fit**, normalized to mean 1 at apply. No norm preservation
  (`onto` is meant to shrink `‖h‖`); the curved path's `norm_cap = 3·‖h‖` is the
  only bound and the affine fast path carries none.
- **Top-p via `torch.topk`**, never a full-vocab sort; `top_k` caps the candidate
  pool before top-p.
- **Monitor capture is hook-driven**, inline with generation — no second forward
  pass. Per-token scoring runs **post-forward** (`fire_step_sink` after `model()`
  returns), not inside the capture hook, so the host-side read does not drain the
  device pipeline mid-forward. Scoring is conditional on declared need via
  `ReadDemand` → `CaptureMode`.
- **Steering hooks are transient** — composed before generation, removed after.
  The persistent compiled-offset buffers are the separate static path and are
  zeroed when unsteered.
- **Instrument ordering** is `close_run → prepare → plan → bind`, enforced rather
  than advisory; every capture transaction closes its runs through nested
  `finally`s. Instrument `state_lock`s are reentrant leaf locks, taken after
  `_gen_lock`/`_model_exclusive` and never on a per-token path.
- **MPS discipline** — diffs on CPU, `torch.mps.empty_cache()` between extraction
  batches, end-of-loop sync to dodge Metal command-buffer reuse, and periodic
  queue drains in the J-lens fit (Metal reports queue exhaustion asynchronously by
  silently zeroing work).

## Deliberately absent

Rejected designs, listed once so they are not re-proposed: there is no
angular/additive injection mode, no `injection_mode`/`theta_max`/`_STEER_GAIN`,
and no per-fit lever or `N` correction; no Euclidean activation-space fit,
projection, or monitor read; no `anchor_origin` knob (every frame is
neutral-anchored) and no `max_subspace_dim` for a flat `pca` fit (the affine span
*is* the layout); no `[0,1]` clamp or water-fill on affine `along`; no separate
`layer_means` cache or forward pass; no second forward pass for monitor reads and
no re-render text scorer; no `extract_difference_of_means` or standalone
`{positive, negative}` corpus artifact; no `pca` tensor-filename variant; no
SAE `:fraction`/`:membership` channels; no `cache_dir=` parameter (set
`$SAKLAS_HOME`); and no dict input to `Steering`.
