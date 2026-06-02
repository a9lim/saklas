# saklas backend architecture

How saklas extracts, composes, injects, and reads steering signal. This is
the as-built reference for the unified subspace/manifold engine: one artifact
family, one injection kernel, one set of read primitives. It describes what the
code does today — the per-layer math, the dispatch path, the calibration — at
enough depth to modify the engine without re-deriving it from the source.

The companion `AGENTS.md` files are the file-by-file maps; this document is the
cross-cutting story those maps assume.

---

## 1. The one idea

There is a single artifact family, split on one structural axis: **flat vs
curved**.

- A **flat subspace** is an affine frame — a `mean` (D,) plus an orthonormal
  `basis` (R, D) — with no interpolant. The reduced coordinates *are* the
  authoring coordinates (identity map): `eval_at(c) = c @ basis + mean`.
- A **curved manifold** is the same affine frame plus an `r³` polyharmonic RBF
  interpolant surface through the per-node centroids, fit either from authored
  coordinates or from a spectral embedding.

The unifying facts:

- **PCA on two centroids is difference-of-means.** Center two pole centroids →
  `±δ/2`; the sole principal axis is `δ̂`. A steering *vector* is a rank-1 flat
  subspace, and the vector pipeline is "fit a 2-node subspace".
- **Whitened PCA on two centroids is `Σ⁻¹δ`** (the LDA / Fisher discriminant).
  The de-rogued "aggressive vector" direction is just *which PCA metric the fit
  uses* — gated by the same whitener-coverage check every other surface uses.

The three bundled defaults map cleanly onto the taxonomy:

| artifact        | nodes | rank | structure | fit             |
|-----------------|-------|------|-----------|-----------------|
| a DiM vector    | 2     | 1    | flat      | PCA (= DiM)     |
| `personas`      | 101   | ~8   | flat      | discover PCA    |
| `pad`           | 15    | 3    | curved    | authored coords |

Every steering term — vectors, bare poles, `~`/`|` projections, `!` ablations,
and `%` manifold positions — lowers at generation time to a single per-layer
injection: `subspace_inject` (`core/manifold.py`). There is no angular/additive
mode split, no separate vector hook, and no per-fit gain knob. The flat case
takes an analytic shortcut inside that kernel; the curved case runs a per-token
nearest-point foot-follower.

> **Live state.** The 26 bundled *vector* concepts are stored and steered as
> 2-node `pca` `Manifold` artifacts under `manifolds/` (4.0 6b/6d) — the
> "a vector *is* the K=2 affine subspace" convergence, shipped. `personas` (flat)
> and `pad` (curved) are `Manifold` artifacts too. User-authored / ad-hoc vector
> directions (`extract`/`clone`/`merge`, `~`/`|` projections) are still baked
> `Profile`s, folded into rank-1 subspace fragments at dispatch by
> `fold_directions_to_subspace` (`core/vectors.py`); legacy `vectors/` packs
> migrate to manifolds via port-on-detect (6e). All paths lower to the same kernel.

---

## 2. In-memory representations

`core/profile.py` — **`Profile`**: a typed `dict[int, Tensor]` wrapper (per-layer
baked steering direction) with `merged`, `projected_away`, `cosine_similarity`,
and safetensors `save`/`load`. This is the live storage form of a DiM vector.
The baked tensor's per-layer *magnitude* carries the layer's share (§3.3).

`core/manifold.py` — the manifold/subspace types:

- **`LayerSubspace`** — one layer's reduced frame. `mean` (D,), `basis` (R, D);
  for a curved layer also `node_params`/`rbf_weights`/`poly_coeffs` +
  `coord_offset`/`coord_scale` (the unit-box normalization the RBF kernel
  needs). `is_affine` ⇔ `node_params is None`. A flat layer additionally carries
  `node_coords` (K, R) — the **real, neutral-anchored** per-layer node positions
  `(cᵢ − ν)·basisᵀ`, the steer-target source. `affine(mean, basis,
  node_coords=)` builds a flat one; `select_axes(kept)` prunes its basis rows
  (per-axis DLS); `eval_at`, `jacobian_at` evaluate; `rbf_params()` returns the
  curved triple and raises on a flat subspace (a guardrail against routing a flat
  subspace through RBF math).
- **`Manifold`** — `name`, `domain` (a `ManifoldDomain`), `node_labels`,
  `node_coords` (K, n) shared authoring layout, `layers: dict[int,
  LayerSubspace]`, plus calibration bakes `explained_variance`,
  `mahalanobis_share`, `origin` (per-layer authoring-coord foot of the neutral
  mean, curved only), `node_roles`, `feature_space`. The analogue of `Profile`
  for manifold steering. `manifold_point`, `tangent`, `resolve_position` (coord
  payload or node-label string), `nearest_node_{index,label,role}`.
- **`SynthesizedSubspace`** — the dispatch-time analogue of a fitted `Manifold`:
  the entire active steering expression composed into *one* per-layer affine
  subspace + `target_coord` + `share`. Built by `synthesize_subspace` (§4).

`ManifoldDomain` (ABC) defines the geometry: `BoxDomain` (per-axis open or
periodic — boxes, disks, cylinders, n-tori), `SphereDomain` (Sⁿ, chordal
metric), `CustomDomain` (explicit immersion; also the identity-embedded carrier
for discover coords and for synthesized affine subspaces). The RBF interpolant
only ever needs pairwise embedded distances and an embedding Jacobian, so any
embeddable topology works without touching the kernel.

### On-disk layout (`~/.saklas/`, override `$SAKLAS_HOME`)

The two roots are **separate**: vectors under `vectors/`, manifolds under
`manifolds/`.

```
~/.saklas/
  neutral_statements.json
  vectors/<ns>/<concept>/
    pack.json  scenarios.json  statements.json
    <safe_model>.safetensors (+ .json sidecar)     # baked DiM Profile
    <safe_model>_sae-<rel>.safetensors             # SAE-space variant
    <safe_model>_from-<safe_src>.safetensors       # cross-model transfer
    <safe_model>_role-<slug>.safetensors           # role-augmented
  models/<safe_model>/
    layer_means.{safetensors,json}                 # probe-centering baseline
    neutral_activations.{safetensors,json}         # 90 prompts × layers, fp32
    alignments/<safe_src>.{safetensors,json}       # Procrustes map
  manifolds/<ns>/<name>/
    manifold.json  nodes/NN_<label>.json  scenarios.json
    <safe_model>.safetensors (+ .json sidecar)     # fitted per-layer subspaces
```

Tensor-filename variants (`io/paths.py`) are `raw` (canonical), `sae-<release>`,
`from-<safe_src>`, `role-<slug>` — at most one kind per file, mutually
exclusive. There is no `pca` variant and no method suffix; difference-of-means
is the only vector extraction method. `pack.json.files` / `manifold.json.files`
carry sha256 maps verified on load.

---

## 3. Extraction

### 3.1 Forward capture and pooling

`core/vectors.py` owns the low-level capture. One forward pass per prompt;
`_capture_all_hidden_states` hooks every layer at once. Pooling is from the
**last content token** — `last_content_index` walks back from the final position
past both `tokenizer.all_special_ids` and `tokenizer.added_tokens_encoder`
values, so trailing chat-template markers (which carry outlier "rogue" channel
activations) never get pooled. `special_token_ids` + `last_content_index` are
the single canonical definition of "last non-special token", shared by every
single-state readout (extraction, vector aggregate, manifold aggregate) so the
discipline can't drift per-site. `encode_and_capture_stack` is the full-`[T,D]`
companion for the manifold monitor.

### 3.2 Difference-of-means vectors

`extract_difference_of_means` is the **only** vector extractor.
`_capture_diffs_for_pairs` runs the contrastive forward loop and returns
per-layer pos/neg running means + the per-pair diffs; the per-layer direction is
`mean(diffs)` in **fp32** (fp16 sum-of-squares overflows at hidden_dim ≥ 2048,
and loses precision differencing close vectors). Contrastive pairs come from
`generate_statements` (§3.7): two independent per-concept corpora zipped pos/neg
(the former moment-paired path is removed — the centroid difference is identical
either way).

### 3.3 Share-baking

Each layer's unit direction is scaled to that layer's mean activation norm, then
multiplied by `scoreᵢ / Σ scores` over the DLS-retained layers:

```
stored_L = d̂_L × ref_norm_L × score_L / Σ_L score_L
```

So the per-layer *magnitude* of the stored tensor encodes the cross-layer
weighting. The dispatch fold reads `‖stored_L‖` back out as the layer share
(§4); llama.cpp's uniform GGUF scalar reproduces the per-layer weighting for
free (`io/gguf_io.py`). The bake **score** is the Mahalanobis norm
`‖mean_diff‖_M / ref_norm` when a whitener covers every scored layer, else the
Euclidean `‖·‖₂ / ref_norm`. The choice is all-or-nothing (`covers_all`): the
two scales differ by a per-layer `1/√λ_L` factor that does not cancel from the
cross-layer-normalized share, so mixing metrics across layers is incoherent. The
metric used is recorded in the sidecar `bake` field.

### 3.4 The whitener

`core/mahalanobis.py` — **`LayerWhitener`** holds per-layer centered neutral
activations `X_L ∈ ℝ^(N, D)` and the small Woodbury inverse `K_L = (NλI +
XXᵀ)⁻¹`, so `apply_inv(layer, v) = Σ_reg⁻¹ v = (1/λ)(v − Xᵀ K X v)` in O(ND)
without ever materializing a D×D matrix. Ridge `λ_L = (‖X_L‖²_F / (N·D)) ·
ridge_scale`. Built lazily from the cached neutral activations
(`from_neutral_activations` in-memory, `from_cache(model_id)` from disk).

Neutral activations are cached **fp32** — the project-wide invariant. fp16's
65504 ceiling overflows gemma-3's extreme late-layer channels to ±inf, which
poisons the covariance (`λ = inf → K = nan`); fp32 has the range and keeps the
whitener bit-reproducible across the cache boundary.
`from_neutral_activations` *excludes* any layer whose centered activations or
regularized inverse come back non-finite, leaving it uncovered. So
`covers_all(layers)` — the all-or-nothing coverage gate shared by extraction,
manifold fit, projection, the monitor, and `vector compare` — is trustworthy as
"finite factors everywhere": either the whole probed/scored set is whitened or
none of it is.

Primitives: `apply_inv`, `mahalanobis_cosine`, `leace_project` (closed-form
LEACE single-direction projector), `subspace_gram(layer, B) = B Σ⁻¹ Bᵀ` (the
(R,R) reduced-space inverse covariance backing whitened share + whitened
manifold reads; reduces to `I_R` for orthonormal `B` when `Σ = I`), and
`woodbury_factors(layer, *, device, dtype)` (device-resident factors for the
monitor's per-token inline `Σ⁻¹h` apply).

### 3.5 The subspace/manifold fit

`ManifoldExtractionPipeline.fit` (`core/extraction.py`) pools each node's mean
activation (`compute_node_centroid`, last-content-token pooling), then dispatches
on `manifold.json::fit_mode`:

- **`pca`** (flat / discover) → per layer `fit_affine_subspace` →
  `LayerSubspace.affine`.
- **`authored`** / **`spectral`** (curved) → per layer `fit_layer_subspace` →
  PCA frame + RBF surface.

Both derive the per-layer basis through one shared routine, `_pca_basis`.

**Basis selection (`_pca_basis`).** The input is the `(K, D)` **μ-centered**
centroid scatter `X = centroids − mean(centroids)`.
- *Euclidean* (default, or partial whitener coverage): ordinary SVD of `X`;
  `basis = Vh[:R]`, `ev_ratio` = retained fraction of raw inter-node variance.
- *Whitened / Fisher* (whitener covers `layer`): the generalized eigenproblem
  `(S_b, Σ)` solved via the low-rank Woodbury `Σ⁻¹` — eigvecs `a` of `G = X Σ⁻¹
  Xᵀ` (K×K), directions `vᵣ = Σ⁻¹ Xᵀ aᵣ`, re-expressed in a Euclidean-
  orthonormal basis via QR (span-preserving, so the steering hot path operates
  on an ordinary orthonormal frame). `ev_ratio` = retained fraction of *whitened*
  between-variance. `R = min(n_components, K−1, rank)`, `n_components` default 64.

Whitened PCA maximizes `vᵀS_b v / vᵀΣv` instead of raw `vᵀS_b v`. On real LMs raw
PCA chases the massive-activation (rogue) channels — most variance, little
concept signal — leaving the subspace, its `mean`, and the steering direction
rogue-dominated. The Fisher ratio divides each direction by its background
variance `vᵀΣv`, so rogue dims (huge background variance) cancel out — the exact
cancellation difference-of-means gets for free by differencing. The de-rogued
subspace barely overlaps the rogue-dominated `mean`, so the running `‖h‖` swing
that plagued raw fits collapses for free. This is the same metric, the same
`covers_all` gate, as the DiM bake and the monitor.

**The neutral-anchor invariant (the load-bearing rule).** Every fit — flat and
curved — stores `mean = P_basis(neutral_L)`: the *projection of the layer's
neutral mean into the span*, not the raw neutral vector. Node coordinates are
stored relative to that anchor: `coordsᵢ = (centroidᵢ − neutral) @ basisᵀ`.
Consequences:
- Neutral → reduced-coord 0 in the span. For a flat subspace the surface *is* the
  span, so the affine origin is implicitly 0 (no stored origin); `!` ablation
  targets 0.
- Coords are *real* (true projected distances), so the per-layer signal weight is
  intrinsic: a node sits at distance ∝ `‖δ_L‖` from the origin, so a fixed slide
  fraction displaces more where the concept signal is bigger. This is what makes
  a separate per-layer gain "lever" redundant (§5.4).
- All affine subspaces share one anchor, so composition is concatenation (§4),
  with no per-term frame conversion.

**Basis caveat (do not break PCA@2 ≡ DiM).** The *frame* (mean + coords) anchors
at neutral, but the *basis* comes from the **μ-centered** scatter, never the
anchor-centered one. At K=2 the μ-centered SVD's sole axis is exactly
`δ̂ = unit(c₀ − c₁)`; anchor-centering would inject `(μ − neutral)` as a spurious
axis and, if the neutral offset dominates (e.g. dog.cat's shared "animalness"),
return the offset instead of `δ̂`. `fit_affine_subspace` also takes `orient_to`
(default node 0) to flip basis rows to a deterministic sign convention, so K=2 /
node-0-is-pos reproduces the DiM `+δ̂` orientation.

`fit_affine_subspace` returns `(LayerSubspace.affine(mean, basis, node_coords),
mu_coords, ev_ratio)`, where `mu_coords = (centroids − μ)·basisᵀ` is the
μ-centered (anchor-independent) reduced coords used for the budget share.
`fit_layer_subspace` normalizes the embedded domain coords to the unit box and
fits `fit_rbf_interpolant` — a dense symmetric-*indefinite* saddle system solved
with `torch.linalg.solve` (never Cholesky; node counts are tiny, scipy is not
pulled in). At n=1 over an open axis the `r³` polyharmonic spline reproduces the
natural cubic spline exactly.

### 3.6 Per-layer signal weighting at fit time

- **Flat → per-axis DLS.** Discriminative Layer Selection (Selective Steering,
  Dang & Ngo 2026 Eq. 9) generalizes from layers to axes. `compute_dls_axes`
  keeps axis `d̂ᵣ` at layer L iff the node projections `{(cᵢ − ν)·d̂ᵣ}` *straddle
  zero* (both signs present) — the axis discriminates node *position* across the
  baseline rather than encoding a common offset/intensity. At N=2 this is exactly
  the pos/neg opposite-sign test (`compute_dls_mask`/`compute_dls_mask_per_axis`,
  bit-identical at R=1); for N>2 (personas) it prunes per-axis. The kept set is
  baked into the stored basis via `LayerSubspace.select_axes`, so the steer path
  needs no separate mask. An all-fail layer is dropped (matching the folded-
  vector path).
- **Curved → no DLS.** A manifold has no pos/neg polarity, so the opposite-sign
  test is undefined; per-axis pruning would force an RBF re-fit. Per-layer signal
  is instead handled entirely by the apply-time share.

### 3.7 Calibration bakes

- **`mahalanobis_share`** — `subspace_share(mu_coords, basis, whitener, layer)` =
  `sqrt(Σ_k coords_kᵀ M_R coords_k)` (whitened, `M_R = subspace_gram`) or
  `‖coords‖_F` (Euclidean). The whitened/Euclidean spread of the node centroids
  around their *own* mean, restricted to the subspace — **anchor-independent**
  (μ-centered), so it measures signal *spread*, not where neutral sits. At
  K=2/R=1 this is `‖δ_L‖_M / √2`, so the normalized per-layer profile is the DiM
  bake profile exactly. Baked when the whitener covers all fit layers, gated
  alongside the whitened basis; `share_metric` records which. This is the
  per-layer budget weight at apply time. (Coords supply target *positions*; share
  supplies the *budget*.)
- **`origin`** (curved only) — `invert_parameterization` of the neutral mean onto
  the surface, per layer, in authoring coords. The cold-start foot seed for the
  per-token follower and the slide target of `!`. Flat subspaces store none (foot
  = span-coord 0).
- **`explained_variance`** — recorded as a fit-quality diagnostic only; it no
  longer drives gain.

There is **no lever bake.** `layer_lever`, `Manifold.lever`, `lever_per_layer`,
`_MIN_MANIFOLD_LEVER`, and the `base/N` division are all gone (§5.4 explains
why).

### 3.8 Discover-mode coordinate derivation

When the user supplies labeled corpora but no coordinate system, `fit_mode` is
`pca` or `spectral` and coordinates are derived per-model at fit time from the
pooled centroids at a reference layer (`core/manifold.py`):

- **`derive_pca_coords`** — SVD of the centered centroids; pick the smallest
  prefix whose cumulative variance crosses `var_threshold` (default 0.70), capped
  at `max_dim` (default 8); project onto those `k` directions.
- **`derive_spectral_coords`** — symmetric (union) k-NN graph, heat-kernel edge
  weights, eigendecompose the symmetric normalized Laplacian, drop the smallest
  eigenvalue, pick `k` by the eigenvalue-*ratio* cliff `argmax(λ_{k+1}/λ_k)` (the
  absolute gap over-picks on S¹). A disconnected graph raises, naming the
  component count and pointing at `--k-nn`/`--method pca`.

Both wrap the derived coords in `CustomDomain(k)` with identity embedding and
proceed through `fit_affine_subspace` (pca) or `fit_layer_subspace` (spectral)
unchanged. Per-model coordinates are the architectural consequence: a Gemma fit
and a Qwen fit produce different node layouts for the same corpus heap. The
derived layout is stored as `node_coords` in the per-model safetensors; the
diagnostics (`PcaDiagnostics`/`SpectralDiagnostics`) ride into the sidecar.

### 3.9 SAE feature space

`--sae <release>` reconstructs each centroid (or pos/neg stack) through the SAE
before the fit/extract: encode → decode → fit in *model* space, so the hook never
touches the SAE. `core/sae.py` wraps SAELens (`load_sae_backend`); coverage is
fail-fast (`SaeCoverageError` before the pooling loop). The SAE branch ignores
the whitener (residual-stream Σ doesn't apply in feature space).

### 3.10 Statement generation

`session.generate_statements(concepts, *, scenarios=None, n_scenarios,
statements_per_cell, on_scenarios=None, on_corpus=None, neutrals=False,
role=None)` is the unified corpus generator. One LLM call writes shared
scenarios, then one call per `(scenario, concept)` cell fills the corpus.
Scenario-sharing across the row is load-bearing — without it per-concept
centroids mix concept signal with scenario signal, and discover layouts surface
scenario as the dominant axis. A literal-concept (anti-allegory) directive is
present in every prompt builder, keeping non-human axes (deer/wolf, brick/
feather) literal rather than collapsing into human-social metaphor (tests assert
its presence, not byte-identity). Returns `{concept: [statements]}`. The DiM
extractor calls it with `[pos, neg]` and zips the two corpora into pairs;
discover authoring (`manifold generate`, the HTTP route) wraps it into a
freshly written discover folder. `on_scenarios`/`on_corpus` are streaming sinks
the resumable discover writers (`io/manifolds.py`) use for big rosters.

---

## 4. Composition (dispatch-time synthesis)

The engine never holds "one manifold per concept". At each generation
`session._compose_steering_entries` classifies the active steering expression and
composes the unified backend:

| term                         | contributes                                                   |
|------------------------------|---------------------------------------------------------------|
| `c x` / pole                 | push fragment: unit baked dir `(1, D)`, target `[‖d_L‖]`, coeff `c` |
| `c a~b` / `a\|b`             | push fragment: the derived projected direction (materialized via `project_profile`), folded like a vector |
| `!x`                         | ablation fragment: x's per-layer directions, target → origin (0) |
| `c M%label` (affine M)       | push fragment: M's per-layer basis rows, target = node's `LayerSubspace.node_coords[idx]`, coeff `c` |
| `c[,o] M%pos` (curved M)     | a separate two-op term via `add_manifold` (along=c, onto=o)  |

`fold_directions_to_subspace` (4.0 6b, replacing the old `_vector_push_fragment`
shim) splits a resolved direction `Profile` into per-layer `(unit_dir, [‖d_L‖])`,
so the synthesizer's `Δ = coeff · (coord @ basis) = coeff · d_L` reproduces the
baked direction exactly. `_affine_manifold_push` reads the
fitted manifold's per-layer `node_coords` (label-form only — coord-form on an
affine manifold has no interpolant). Projection terms are materialized to derived
`Profile`s first (`_materialize_projections` → `project_profile`, LEACE under a
whitener, Gram-Schmidt otherwise) and then folded like vectors.

Push + ablation fragments are grouped **by trigger**. Each trigger group is
composed by `synthesize_subspace(push, ablate, neutral_means)` into one
`SynthesizedSubspace`. Per layer (over the union of layers any term touches):

1. Flatten every present push fragment's basis rows, then every ablation
   fragment's rows, into one ordered list (push first); orthonormalize the union
   (`_ortho_basis`) → the merged `(R, D)` basis `B`. Ordering is load-bearing:
   push-before-ablation keeps the push displacement inside the earlier rows, so
   the ablation-only axes (the orthogonal complement) get a target ≈ 0 for free.
2. World push displacement `Δ = Σ_push coeffᵢ·(coord_targetᵢ @ basis_rowsᵢ)`;
   `target_coord = B @ Δ` is its coordinate in the merged basis.
3. `share = ‖Δ‖` (a pure-ablation layer weights by the summed ablation-row
   magnitude instead).

Because `B` is orthonormal, `‖target_coord‖ = ‖Δ‖`, so the per-layer budget
weight and the steered coordinate sit on one consistent scale. The result is
registered via `SteeringManager.add_subspace`; curved `%` terms register via
`add_manifold`. `add_subspace`/`add_manifold` see exactly one merged affine
subspace per trigger group plus zero or more curved manifolds.

### Orthogonality model

At `apply_to_model` time:

- **Two curved manifolds** sharing a layer must be (near-)orthogonal
  (`max |cosine| ≤ _CURVED_ORTHO_TOL = 1e-3`); overlapping ones raise
  `OverlappingManifoldError`. Each overwrites its own in-subspace component, so
  overlapping spans would clobber each other.
- **The merged affine subspace** is orthogonalized against every curved span at
  that layer (`_orthogonalize_affine_against`: strip the curved-span component
  from the affine basis rows *and* the push displacement, re-orthonormalize). The
  curved manifold wins the shared directions; the affine slide handles the
  complement. If the affine span lies entirely inside the curved span, that layer
  is dropped for the affine subspace. Affine-vs-affine never checks — it always
  merges.

The off-*subspace* residual is always kept verbatim, so the merged affine
subspace and N orthogonal curved manifolds compose with zero cross-talk — each
operates only inside its own span.

---

## 5. Steering / injection

### 5.1 The kernel

`subspace_inject(h, subspace, domain, target_coord, foot_seed, along, onto)`
(`core/manifold.py`) is the single injection. It decomposes
`h = mean + h_par + h_perp` against the layer's affine subspace (`decompose`),
where `h_par` is the in-subspace component and `h_perp = H_o` the off-subspace
residual, then applies two operations:

- **along (`a ∈ ℝ`)** — slide the projected foot from its current position toward
  `target_coord`, geodesically in authoring-coord space (`domain.geodesic`), and
  transport the off-*manifold*-in-subspace residual `H_n` to stay normal at the
  new foot (project onto the new tangent's normal space, renorm to the preserved
  `‖H_n‖`). Tangential/directional. This replaces the old additive chord — by
  sliding *on the surface* it never cuts through off-manifold low-density space.
- **onto (`o ∈ [0,1]`)** — scale `H_n` by `(1 − o)`: collapse the off-surface
  in-subspace residual onto the surface. Vacuous when the surface fills its
  subspace.

`H_o` (the off-subspace residual) is **always kept verbatim** — the former third
op (`toward`, which scaled `H_o`) is gone, because it scaled the orthogonal
complement of *this* subspace, i.e. every composing neighbor's span, breaking
orthogonal composition. The R-dim `~`/`|` semantics are instead recovered by
routing those operators into the merged affine subspace as push/ablation axes
(§4).

All subspace arithmetic runs in reduced (R-dim) coordinates; because `basis` is
orthonormal, `‖H_n_reduced‖ = ‖H_n‖` exactly, so the cost is O(R) not O(D). fp32
throughout. Order is fixed **along → onto** (the transport must precede `onto`
scaling the transported residual). A soft cap `‖h_new‖ ≤ norm_cap·‖h‖`
(`norm_cap = 3.0`) is the only norm guard — `onto` is *meant* to shrink
`‖h − mean‖`, so there is no global norm preservation.

**Flat (affine) shortcut.** When `subspace.is_affine` the surface fills the
subspace: reduced coords *are* authoring coords (identity), the foot is `q`
exactly, `H_n ≡ 0`, and `onto` is vacuous. So the kernel skips the Gauss-Newton
foot solve, the RBF eval, and the tangent Gram-solve — it slides `q` toward
`target` (linear, since the synthesized domain is `CustomDomain`) and keeps `H_o`
verbatim. This is load-bearing for throughput: a folded vector is the common
case and the curved per-token solve would blow the throughput invariant.

**Curved foot-following.** The nearest-point foot on the surface is a function of
the running activation, so it is tracked across tokens rather than re-solved.
`SteeringHook._manifold_feet[i]` holds the previous token's refined foot (`None`
= cold). Cold → seed at the per-layer `origin O_L` and take
`_MANIFOLD_COLD_GN_STEPS = 4` Gauss-Newton steps (the prefill fire converges the
foot across the whole prompt window); warm → one step from the carried foot. The
last position's foot is stashed for the next token. `invert_parameterization` (a
damped Levenberg-Marquardt solve, warm-started from each query's nearest fit
node, batched over queries × restarts) is the cold/eval-only nearest-point
projection — never the per-token hot path.

### 5.2 The hook

`core/hooks.py` — **`SteeringHook`** carries the per-layer groups
`(trigger, subspace, domain, target_coord, origin_coord, along, onto)` and runs
each active one through `subspace_inject` in `_apply_manifold_groups`. A
cheap pre-check skips the work when no group is active this step (e.g. an
`AFTER_THINKING` group during prefill). There is no composed-tensor fast path:
every steered layer runs the (ctx-consulting) slow hook, so per-step triggers and
probe gates work uniformly. `all_fast_path()` is true only for the unsteered path
— which is the sole `torch.compile` / StaticCache graph-capture eligibility
signal (`core/cuda_graphs.py`).

**`SteeringManager`** owns the hooks + the per-generation `TriggerContext`. It
holds `subspaces` (the dispatch-synthesized merged affine subspaces, one per
trigger group) and `manifolds` (curved terms). `apply_to_model` lowers both to
per-layer `subspace_inject` entries, orthogonalizes affine against curved
(§4), and recomposes the hooks. `reset_manifold_feet` cold-starts every
follower at each generation start.

### 5.3 Gain

The per-layer share is normalized to **mean 1** (`Σ_L share_L = n_layers`), *not*
sum 1, then:

- **Merged affine subspace** (`add_subspace`): `eff_along_L = share_L ·
  _MANIFOLD_GAIN`, `onto = 0`. The coefficient α is already folded into
  `target_coord` by `synthesize_subspace` (`Δ = Σ coeffᵢ·poleᵢ`, required for
  multi-term composition), so α scales *where* the slide lands and `share_L·base`
  scales *how far* the slide goes. α is unclamped (it sets a target position).
- **Curved manifold** (`add_manifold`): `eff_along_L = along · share_L · base`,
  `eff_onto_L = clamp(onto · share_L · base, 0, 1)`; the target is the full node
  position, so `along` is the slide fraction (the historic `%` knob), clamped to
  `[0,1]` at apply time. `onto` stays a `[0,1]` collapse fraction.

`_MANIFOLD_GAIN = 1.0` — one gain constant for both modes. There is **no `[0,1]`
clamp and no water-fill on `along`**: with mean-1 share a typical layer already
sits at `eff_along ≈ base`, so a clamp would saturate nearly every layer and
erase the share distribution. High-signal layers are allowed to overshoot the
target; the de-rogued whitened coords keep the overshoot controlled, and the
kernel's `norm_cap = 3·‖h‖` is the only backstop. `onto` keeps its clamp (a
collapse fraction past 1 inverts the residual).

Geometrically `eff_along` is the geodesic-lerp fraction of the in-subspace
component toward the target, so `base ≈ 1.0` means a *typical layer fully
replaces* its in-subspace component with the target. The constant is calibrated
on the gemma whitened smoke; α clamps to `[0,1]` so `base` is the strength
ceiling. Per-persona strength variance persists — a hard persona peaks near its
coherence edge at α ≈ 1 where a robust one still has room; tune α per target.

### 5.4 Why mean-1 share and no lever

Under **sum-1** share, `Σ eff_along_L = base`, so `base` was the *total* slide
budget spread across `n_layers` (≈ `base/n_layers` per layer — the source of the
"base=1 is too gentle" reading). Under **mean-1** share, `eff_along_L ≈ base` per
*typical* layer, so `base` is a clean dimensionless per-layer slide fraction,
**n_layers-invariant**. That invariance is what restores `A ⊂ B` consistency:
steering to `(1)` on a rank-1 subspace and `(1,0,…)` on a rank-R superset moves
the shared axis identically.

The **lever** `N = Σ_L share_L · f_L` (with `f_L = E_neutral[‖h_par_c‖/‖h‖]`) was
a normalized-coord-era correction: when targets were unit/±1 and the magnitude
lived in a normalization factor, dividing the gain by the captured-norm fraction
equalized the per-α effect across subspace dimension. Neutral-anchored **real**
coords put the magnitude back in the target, so dividing by `N` became a spurious
*second* correction. It also broke `A ⊂ B` consistency (a larger subspace has a
larger lever ⇒ smaller `base/N` ⇒ weaker steering) and was catastrophic on
whitened low-rank fits (`N ≈ 1e-4` for a de-rogued rank-1 vector ⇒ `base/N ≈
10⁴`, saturating every layer into full-replacement collapse). With the lever
removed, a rank-1 vector and a rank-8 fit land in the same α-band with no per-fit
retuning. The lever is torn out everywhere; EV survives as a diagnostic only.

---

## 6. Probing / reads

`core/monitor.py` — two read-side monitors, both hook-driven (inline with
generation, one matmul per layer, no second forward pass), both fp32, both with
no `.item()` per token.

### 6.1 TraitMonitor (vector probes)

Scores per-layer probe directions against the running hidden state. The per-layer
similarity is the **whitened (Mahalanobis) cosine** `⟨V, h_c⟩_M / (‖V‖_M
‖h_c‖_M)` — matching the metric the default DiM bake and `~`/`|` projection use.
The metric is all-or-nothing via `covers_all`: Mahalanobis when the wired
whitener covers every probed layer, else plain Euclidean cosine for all (never a
per-layer mix — that would fold cosines from two metrics into one aggregate). The
per-layer probe *weight* stays `‖baked‖₂` (the bake already folded the
Mahalanobis score into the magnitude, so re-whitening would double-count).
`_ensure_cache` precomputes the whitened probe directions + their Mahalanobis
norms + device-resident Woodbury factors, so the hot path is one matmul plus a
cheap per-token `Σ⁻¹h_c` apply. Entry points: `score_per_token` (primary,
returns `(aggregate, per_token)`), `score_single_token{,_per_layer,_tensor}`
(inline, SSE / probe-gate callback), `score_stack`, `measure` (one-shot text).

### 6.2 ManifoldMonitor (manifold probes)

The read-side peer. `add_probe(name, manifold, *, top_n)` pre-caches each layer's
node values in reduced `(K, R)` coords. Two channels per token, EV-weighted
across layers:
- **fraction** — `‖h_par_c‖ / ‖h − mean‖`, the share of the centered activation
  living in the manifold's subspace, in `[0,1]`.
- **nearest** — the nearest node by reduced-coord `cdist`.

**Whitened readout** (built per probe at attach when the whitener covers every
layer of that manifold): both channels switch to their Mahalanobis forms —
fraction is the M-orthogonal subspace share `sqrt(gᵀ M_R⁻¹ g)/‖x‖_M` (`g = B Σ⁻¹
x`), nearest runs in the `M_R` metric — the read-side analogue of the whitened/
Fisher subspace the manifold was *fitted* in. `score_aggregate` is the
end-of-gen slow path: pools the last non-special token per layer (same discipline
as extraction), runs the channels, and additionally calls
`invert_parameterization` to recover authoring coords + normalized residual.
`flat_scalars` flattens readings to `{"<name>:fraction": float, "<name>@<label>":
−distance}` (signed so "larger = closer"); the session merges these into
`TriggerContext.probe_scores` so probe gates can address them.

### 6.3 Probe gates / triggers

`core/triggers.py` — `Trigger` (frozen) carries phase flags + an optional
`ProbeGate`. `Trigger.active(ctx)` consults the phase flags and, when gated,
`ctx.probe_scores[gate.probe]` against `score <op> threshold`. The gate key is
the canonical scalar key from whichever monitor supplied it (`"angry.calm"` from
TraitMonitor, `"pad:fraction"` / `"pad@elated"` from
ManifoldMonitor) — the runtime lookup is identical; only the parser knows the
difference. Gated triggers report inactive during prefill (no reading yet) and
for missing probes (no raise).

---

## 7. Grammar (`core/steering_expr.py`)

`parse_expr(text) → Steering`; `format_expr` round-trips. Every input surface
(Python, YAML, HTTP, TUI, `vector merge`) speaks it.

```
expr     := term (("+" | "-") term)*
term     := [coeff "*"?] ["!"] selector ["@" trigger]
selector := atom (("~" | "|") atom | "%" position)?
position := signed_num ("," signed_num)* | label
atom     := [ns "/"] NAME ["." NAME] [":" variant]
trigger  := preset | "when" ":" probe op NUM
```

`+`/`−` add terms, `*` attaches a coefficient, `~` projects onto a direction
(keep the shared component), `|` projects orthogonal (remove it), `!`
mean-ablates (`h' = h − α(h·d̂ − μ·d̂)d̂`). `%` places a generation at a manifold
position — `<coord_list>` or `<label>` (sugar for that node's coords). The
manifold coefficient slot is `along[,onto]` (1- or 2-tuple); the former third
`toward` slot is removed. Variants are `raw`/`sae[-<release>]`/`role[-<slug>]`/
`from[-<src>]` (no `pca`). A bare slug resolves through
`io.selectors.resolve_bare_name`: first as an installed bipolar pole, then as a
manifold node label (synthesizing a label-form `ManifoldTerm`); cross-tier
ambiguity raises. Term types (`ProjectedTerm`/`AblationTerm`/`ManifoldTerm` +
plain tuples) survive as parse-time markers the dispatch synthesizer consumes.

**Role substitution.** A node's `role` (or a `:role-<slug>` vector variant) pools
that node's centroid under a chat-template assistant-label substitution
(`core/role_templates.py`), producing role-baselined activation space (a persona
manifold). At steer time `nearest_node_role` picks the role and pipes it through
`session._active_role` so the generation prefill applies the same substitution —
steer baseline equals extract baseline. Multiple terms implying distinct roles
compose under soft-warn + highest-`|coeff|`-wins (`RoleBaselineMismatchWarning`).
Mistral-3/talkie lack a substitutable label and raise
`RoleSubstitutionUnsupportedError` at fit time.

---

## 8. Cross-model transfer

`io/alignment.py` — per-layer orthogonal Procrustes (`fit_alignment`, SVD for
matched dim, rectangular least-squares otherwise; both center first) maps
`M_L : ℝ^D_src → ℝ^D_tgt`. `transfer_profile` applies `M_L @ v_src` per layer and,
with a target whitener covering the transferred layers, re-bakes each magnitude
to its *target* Mahalanobis norm so the share is in the target metric.
`transfer_manifold` (`io/manifolds.py`) maps a fitted manifold's per-layer
`mean → M_L mean` and `basis → basis @ M_Lᵀ`, leaves the RBF + `node_coords`
untouched (subspace/authoring-coordinate space, invariant under the model-space
map), re-bakes the Mahalanobis **share** in target space (no lever — it's gone),
clears `origin` (per-layer foot of the *source* neutral), and writes the
`_from-<safe_src>` variant. Alignments cache under the *target* model dir.

---

## 9. Naturalness eval

The paper's validation half (`saklas experiment naturalness`,
`core/manifold.py`): fit a *behavior-space* manifold over node-corpus output
distributions mapped to Hellinger space (`p ↦ √p`), generate, re-run the model
over the generated text to recover its behavioral trajectory, and report the
per-step Bhattacharyya distance of that trajectory to the behavior manifold — low
is natural, high flags off-manifold "teleportation". `--compare-linear` scores a
straight-chord additive baseline alongside.

---

## 10. Frontiers and known limitations

- **Persona identity vs. intensity (deferred).** The fitted `personas` subspace
  is a near-1-D fan: ~80–87% of between-node displacement variance lies on a
  single "persona-ness / distance-from-default-assistant" axis every persona
  loads on; per-persona identity is real but subdominant (~15–20%, orthogonal).
  Geodesic-`along` to a node's absolute coords is dominated by the shared axis,
  so distinct personas can express the same "generic intense persona" register.
  This is *not* the rogue problem — whitening verifiably worked (the dominant
  axis carries 2–4% of its energy on rogue channels vs ~100% in raw space). The
  fix is about how steering *accesses* the subdominant identity directions (sphere
  the between-persona covariance at fit time; strip the shared axis from the
  target), not the metric.
- **`base_gain` magnitude.** `_MANIFOLD_GAIN = 1.0` is committed but provisional
  and coupled to the personas frontier: the coherence cliff is `eff_along → 1.0`
  (full in-subspace replacement), so a strong-but-safe constant for rank-8 fits
  sits *below* 1.0. The rank-1 vector path likely tolerates more. One constant,
  no per-fit knob.
- **Subspace-native vector storage.** The fold functions that represent a vector
  as a 2-node affine `Manifold` are tested (author→fit→steer loop) but not the
  production extraction/storage path; vector concepts still ship as baked
  `Profile`s. Converging them collapses the last `vectors/`-vs-`manifolds/`
  representational seam.
- **MPS non-determinism.** Metal kernels are not bitwise deterministic even at
  temperature 0, so run-to-run wording jitters; compare qualitatively.

---

*Pure manifold/subspace math lives in `core/manifold.py` (fp32, dependency-free,
no session/IO coupling). The whitener is `core/mahalanobis.py`; extraction
`core/vectors.py` + `core/extraction.py`; dispatch + injection
`core/session.py` + `core/hooks.py`; reads `core/monitor.py`; grammar
`core/steering_expr.py`.*
