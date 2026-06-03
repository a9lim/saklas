# saklas backend architecture

How saklas extracts, composes, injects, and reads steering signal. This is the
as-built reference for the unified subspace/manifold engine: **one artifact
family, one extraction pipeline, one injection kernel, one set of read
primitives**. It describes what the code does today — the per-layer math, the
dispatch path, the calibration — at enough depth to modify the engine without
re-deriving it from the source.

The companion `AGENTS.md` files are the file-by-file maps; this document is the
cross-cutting story those maps assume.

---

## 1. The one idea

There is a single artifact family — the **manifold** — split on one structural
axis: **flat vs curved**.

- A **flat subspace** is an affine frame — a `mean` (D,) plus an orthonormal
  `basis` (R, D) — with no interpolant. The reduced coordinates *are* the
  authoring coordinates (identity map): `eval_at(c) = c @ basis + mean`.
- A **curved manifold** is the same affine frame plus an `r³` polyharmonic RBF
  interpolant surface through the per-node centroids, fit either from authored
  coordinates or from a spectral embedding.

The unifying facts:

- **A steering vector is a 2-node flat manifold.** Center two pole centroids →
  `±δ/2`; the sole principal axis of the μ-centered scatter is `δ̂`. So
  difference-of-means *is* "PCA on two centroids", and the vector pipeline is "fit
  a 2-node flat subspace". `session.extract("angry.calm")` authors a 2-node
  discover-`pca` manifold (node 0 = `angry`, node 1 = `calm`) and fits it; the
  in-memory `Profile` callers get back is a *folded view* of that fit
  (`folded_vector_directions`). There is no separate baked-DiM artifact, no
  `statements.json`, no `vectors/` storage — the concept's only on-disk form is
  the 2-node manifold under `manifolds/`.
- **Whitened PCA on two centroids is `Σ⁻¹δ`** (the LDA / Fisher discriminant).
  The de-rogued "aggressive vector" direction is just *which PCA metric the fit
  uses* — gated by the same whitener-coverage check every other surface uses.

The bundled defaults map cleanly onto the taxonomy:

| artifact            | nodes | rank | structure | fit_mode  |
|---------------------|-------|------|-----------|-----------|
| a concept vector    | 2     | 1    | flat      | `pca`     |
| `cultural`          | 9     | ~4   | flat      | `pca`     |
| `register`          | 15    | ~6   | flat      | `pca`     |
| `personas`          | 101   | ~8   | flat      | `pca`     |
| `pad`               | 15    | 3    | curved    | `authored`|

Every steering term — vectors, bare poles, `~`/`|` projections, `!` ablations,
and `%` manifold positions — lowers at generation time to a single per-layer
injection: `subspace_inject` (`core/manifold.py`). There is no angular/additive
mode split, no separate vector hook, and no per-fit gain knob. The flat case
takes an analytic shortcut inside that kernel; the curved case runs a per-token
nearest-point foot-follower.

There are four `fit_mode`s: `authored` (user gives domain + coords; curved),
`pca` / `spectral` (discover — labeled corpora only, coords derived per-model;
`pca` flat, `spectral` curved), and `baked` (corpus-less, a precomputed direction
from `subspace merge`). The first three flow through the fit pipeline; `baked`
ships an already-fit tensor.

---

## 2. In-memory representations

`core/profile.py` — **`Profile`**: a typed `dict[int, Tensor]` wrapper (per-layer
baked steering direction) with `merged`, `projected_away`, `cosine_similarity`,
and safetensors `save`/`load`. It is the in-memory steering-vector shape — the
folded view of a 2-node manifold, or an ad-hoc `extract`/`merge` /
projection result — but it is no longer a *storage* form for a concept. The baked
tensor's per-layer *magnitude* carries the layer's share (§3.7).

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
  mean, curved only), `node_roles`, `feature_space`, and a free `metadata` dict.
  The analogue of `Profile` for manifold steering. `manifold_point`, `tangent`,
  `resolve_position` (coord payload or node-label string), `nearest_node_{index,
  label,role}`.
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

There is **one** artifact root: `manifolds/`. (`vectors/` is read only to detect
and port pre-4.0 packs.)

```
~/.saklas/
  neutral_statements.json                          # organic responses to the baseline prompts
  baseline_prompts.json                            # user override for the shared A2 prompts
  manifolds/<ns>/<name>/
    manifold.json                                  # name, source, fit_mode, files{sha256}, per-node {label,kind,role?},
                                                   #   + domain/coords (authored) | hyperparams (discover)
    nodes/NN_<label>.json                          # one response list per node (response[i] ↔ baseline_prompt[i % k])
    <safe_model>.safetensors (+ .json sidecar)     # fitted per-layer subspaces; discover/baked
                                                   #   also carry node_coords (the derived layout)
    <safe_model>_sae-<rel>.safetensors             # SAE-space fit
    <safe_model>_from-<safe_src>.safetensors       # cross-model transfer
    <safe_model>_role-<slug>.safetensors           # role-augmented
  models/<safe_model>/
    layer_means.{safetensors,json}                 # probe-centering baseline
    neutral_activations.{safetensors,json}         # 90 prompts × layers, fp32
    alignments/<safe_src>.{safetensors,json}       # Procrustes map
  vectors/<ns>/<concept>/                          # LEGACY (pre-4.0) only — ported on touch
```

Tensor-filename variants (`io/paths.py`) are `raw` (canonical), `sae-<release>`,
`from-<safe_src>`, `role-<slug>` — at most one kind per file, mutually exclusive.
There is no `pca` variant and no method suffix; difference-of-means is the only
vector extraction method. `manifold.json.files` carries a sha256 map verified on
load (`io/manifolds.py` + the `packs.py` integrity helpers).

The `manifold.json::fit_mode` discriminates the folder shape and the fit path:
`authored` (curved, user coords) · `pca` (flat, derived coords) · `spectral`
(curved, derived coords) · `baked` (corpus-less, a frozen direction from
`subspace merge`).

---

## 3. Extraction

Extraction is a single pipeline, `ManifoldExtractionPipeline.fit`
(`core/extraction.py`), because concept extraction and manifold fitting are the
same operation: pool per-node centroids, fit a per-layer subspace, bake the
per-layer share, write the per-model tensor. A 2-node `pca` fit is a steering
vector; an N-node fit is a manifold. The session wraps it: `extract` /
`extract_vector_from_corpora` author a 2-node `pca` folder then fit;
`extract_manifold` fits an authored/discover folder directly.

### 3.1 Forward capture and pooling

`core/vectors.py` owns the low-level capture. One forward pass per statement;
`_capture_all_hidden_states` hooks every layer at once. Pooling is from the
**last content token** — `last_content_index` walks back from the final position
past both `tokenizer.all_special_ids` and `tokenizer.added_tokens_encoder`
values, so trailing chat-template markers (which carry outlier "rogue" channel
activations) never get pooled. `special_token_ids` + `last_content_index` are
the single canonical definition of "last non-special token", shared by every
single-state readout (centroid pooling, vector aggregate, manifold aggregate) so
the discipline can't drift per-site. For instruction-tuned models the statement
is wrapped as an assistant turn so the capture happens in the model's actual
generation regime; a `role=` substitutes the assistant-role label
(`core/role_templates.py`) for persona/role-baselined fits.
`encode_and_capture_stack` is the full-`[T,D]` companion for the manifold
monitor. `compute_node_centroid` (`core/manifold.py`) pools a node's corpus into
one fp32 mean per layer.

### 3.2 A steering vector as a 2-node fit

`session.extract(concept, baseline, *, kind="abstract")`:

1. Splits the composite (`angry.calm` → pos `angry`, neg `calm`). A **monopolar**
   concept (`baseline=None`) authors a genuinely **1-node** folder; the engine
   recognizes the single-node `pca` shape and fits it against the model's neutral
   mean ν (`layer_means`) as the implicit negative pole — folding `concept − ν`
   into a 1-node neutral-anchored ray (step 3a below). Neutral is sourced per-model
   at fit, never a stored corpus.
2. Generates the pole corpora via `generate_responses` (§3.10) — in-character
   responses to the shared baseline prompts, one corpus per pole — unless a cached
   folder exists. Monopolar generates only the concept pole.
3. Authors a discover-`pca` folder via `create_discover_manifold_folder` with
   `node_corpora = {pos_label: [...], neg_label: [...]}` (one entry for monopolar),
   `node_kinds`, and `hyperparams = {"max_dim": 1, "var_threshold": 0.7}` — so a
   bipolar fit is a rank-1 flat subspace.
   - **3a (monopolar).** When the folder is a single `pca` node, the pipeline takes
     an early branch: pool the one concept centroid, compute `concept − ν` per
     layer, and `fold_directions_to_subspace` it into a 1-node ray (raw δ̂ basis,
     whitened share when the whitener covers the layers) — no discover-coords, no
     per-layer PCA, no DLS. The bare-label tier resolves `0.5 <concept>` to this
     node, so it steers exactly like a bipolar pole.
4. Fits via `ManifoldExtractionPipeline` and returns `(canonical_name,
   folded_vector_directions(manifold))`.

`extract_vector_from_corpora` is the corpus-in sibling (hand-authored pairs —
skips generation). Both emit `VectorExtracted`; the `Profile` they return
is the folded view, not a separately stored tensor.

`folded_vector_directions(manifold)` reverses the fold: `{L: δ̂_L · share_L}`, the
baked-direction equivalent, used to back the `Profile`-returning surface
(`extract()`, `subspace compare`/`why`, GGUF export) without a second stored
representation. It raises on a curved or multi-dim manifold.

### 3.3 Basis selection (`_pca_basis`)

Both flat (`fit_affine_subspace`) and curved (`fit_layer_subspace`) derive the
per-layer basis through one routine, `_pca_basis(X, *, n_components, whitener,
layer)`. The input is the `(K, D)` **μ-centered** centroid scatter `X = centroids
− mean(centroids)` (`DEFAULT_N_COMPONENTS = 64`):

- *Euclidean* (no whitener — the behavior-space naturalness fit, which lives in
  output-distribution space where there are no rogue activation dims): ordinary
  SVD of `X`; `basis = Vh[:R]`, `ev_ratio` = retained fraction of raw inter-node
  variance. Activation-space callers always pass a covering whitener (§3.4) and
  raise without one, so they never take this branch.
- *Whitened / Fisher* (whitener covers `layer`): maximize the LDA objective
  `vᵀS_b v / vᵀΣv` — the generalized eigenproblem `(S_b, Σ)` with `S_b = XᵀX`
  (between-node scatter) and `Σ` the residual-stream covariance. Solved via the
  low-rank Woodbury `Σ⁻¹`: eigvecs `a` of `G = X Σ⁻¹ Xᵀ` (K×K), directions
  `vᵣ = Σ⁻¹ Xᵀ aᵣ`, then re-expressed in a Euclidean-orthonormal basis via QR
  (span-preserving, so the steering hot path operates on an ordinary orthonormal
  frame). `ev_ratio` = retained fraction of *whitened* between-variance.
  `R = min(n_components, K−1, rank)`.

**Why whiten.** On real LMs raw PCA chases the massive-activation (rogue) channels
— most variance, little concept signal — leaving the subspace, its `mean`, and
the steering direction rogue-dominated. The Fisher ratio divides each direction by
its background variance `vᵀΣv`, so rogue dims (huge background variance) cancel —
the exact cancellation difference-of-means gets for free by differencing. The
de-rogued subspace barely overlaps the rogue-dominated `mean`, so the running
`‖h‖` swing that plagues raw fits collapses for free. This is the same metric, the
same `covers_all` gate, as the share bake (§3.7) and the monitor (§6).

**PCA@2 ≡ DiM (the spine identity).** At K=2 the μ-centered scatter is
`X = [+δ/2; −δ/2]` with `δ = c₀ − c₁`; the sole right-singular vector is
`δ̂ = unit(δ)` exactly. So a 2-node Euclidean fit reproduces the
difference-of-means direction bit-for-bit. The whitened K=2 fit instead returns
`Σ⁻¹δ` re-orthonormalized — the LDA discriminant, the de-rogued vector. The
`orient_to=0` argument flips basis rows to a deterministic sign convention so node
0 (the positive pole) lands at `+δ̂`, matching the historical `+δ̂` DiM
orientation.

### 3.4 The whitener

`core/mahalanobis.py` — **`LayerWhitener`** holds per-layer centered neutral
activations `X_L ∈ ℝ^(N, D)` and the small Woodbury inverse `K_L = (NλI +
XXᵀ)⁻¹`, so `apply_inv(layer, v) = Σ_reg⁻¹ v = (1/λ)(v − Xᵀ K X v)` in O(ND)
without ever materializing a D×D matrix. Ridge `λ_L = (‖X_L‖²_F / (N·D)) ·
ridge_scale`. Built lazily from the cached neutral activations
(`from_neutral_activations` in-memory, `from_cache(model_id)` from disk with
cached layer means, `from_neutral_cache(model_id)` from disk with means derived
from the neutral cache for no-model-load transfer rebakes).

Neutral activations are cached **fp32** — the project-wide invariant. fp16's
65504 ceiling overflows gemma-3's extreme late-layer channels to ±inf, which
poisons the covariance (`λ = inf → K = nan`); fp32 has the range and keeps the
whitener bit-reproducible across the cache boundary.
`from_neutral_activations` *excludes* any layer whose centered activations or
regularized inverse come back non-finite, leaving it uncovered. So
`covers_all(layers)` — the all-or-nothing coverage gate shared by extraction,
manifold fit, projection, the monitor, transfer, and `subspace compare` — is
trustworthy as "finite factors everywhere": either the whole probed/scored set is
whitened, or the activation-space caller raises `WhitenerError`. There is no
Euclidean fallback — Mahalanobis-only, because on real LMs the Euclidean metric is
rogue-dominated (a wrong answer, not a degraded one).

Primitives: `apply_inv`, `mahalanobis_norm`, `mahalanobis_cosine`, `leace_project`
(closed-form LEACE single-direction projector), `subspace_gram(layer, B) = B Σ⁻¹
Bᵀ` (the (R,R) reduced-space inverse covariance backing whitened share + whitened
manifold reads; reduces to `I_R` for orthonormal `B` when `Σ = I`), and
`woodbury_factors(layer, *, device, dtype)` (device-resident factors for the
monitor's per-token inline `Σ⁻¹h` apply).

### 3.5 The neutral-anchor invariant + the basis caveat

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
  a separate per-layer gain "lever" redundant (§5.5).
- All affine subspaces share one anchor, so composition is concatenation (§4)
  with no per-term frame conversion.

**The basis caveat (do not break PCA@2 ≡ DiM).** The *frame* (mean + coords)
anchors at neutral, but the *basis* comes from the **μ-centered** scatter, never
the anchor-centered one. At K=2 the μ-centered SVD's sole axis is exactly
`δ̂ = unit(c₀ − c₁)`; anchor-centering would inject `(μ − neutral)` as a spurious
axis and, if the neutral offset dominates (e.g. dog.cat's shared "animalness"),
return the offset instead of `δ̂`. `fit_affine_subspace` returns `(subspace,
mu_coords, ev_ratio)`, where `mu_coords = (centroids − μ)·basisᵀ` is the
μ-centered (anchor-independent) reduced coords used for the budget share.

### 3.6 Per-layer signal weighting at fit time (DLS)

- **Flat → per-axis DLS.** Discriminative Layer Selection (Selective Steering,
  Dang & Ngo 2026 Eq. 9) generalizes from layers to axes. `compute_dls_axes`
  keeps axis `d̂ᵣ` at layer L iff the node projections `{(cᵢ − ν)·d̂ᵣ}` *straddle
  zero* (both signs present) — the axis discriminates node *position* across the
  baseline rather than encoding a common offset/intensity. At K=2 this is exactly
  the pos/neg opposite-sign test (`compute_dls_mask`/`compute_dls_mask_per_axis`,
  bit-identical at R=1); for K>2 (personas, register) it prunes per-axis. The
  flat fit runs the straddle over all fit layers at once and slices the basis via
  `LayerSubspace.select_axes`; an all-fail layer is dropped (matching the
  folded-vector path). `--no-dls` keeps every axis.
- **Curved → no DLS.** A manifold has no pos/neg polarity, so the opposite-sign
  test is undefined; per-axis pruning would force an RBF re-fit. Per-layer signal
  is instead handled entirely by the apply-time share.

### 3.7 The fit and the calibration bakes

`fit_affine_subspace(centroids, *, neutral_mean, whitener, layer, orient_to,
n_components)` builds the flat frame (μ-centered basis, neutral anchor, real
`node_coords`, sign orientation). `fit_layer_subspace` adds the RBF surface for
the curved path: it normalizes the embedded domain coords to the unit box and
fits `fit_rbf_interpolant` — a dense symmetric-*indefinite* saddle system solved
with `torch.linalg.solve` (never Cholesky; node counts are tiny, scipy is not
pulled in). At n=1 over an open axis the `r³` polyharmonic spline reproduces the
natural cubic spline exactly.

The per-tensor bakes:

- **`mahalanobis_share`** — `subspace_share(mu_coords, basis, whitener, layer)` =
  `sqrt(Σ_k coords_kᵀ M_R coords_k)` (whitened, `M_R = subspace_gram`) or
  `‖coords‖_F` (Euclidean). The whitened/Euclidean spread of the node centroids
  around their *own* mean, restricted to the subspace — **anchor-independent**
  (μ-centered), so it measures signal *spread*, not where neutral sits. At
  K=2/R=1 this is `‖δ_L‖_M / √2`, so the normalized per-layer profile is the DiM
  bake profile exactly. The activation-space fit always whitens (it raises without
  a covering whitener), so `share_metric`/`subspace_metric` read `mahalanobis` — the
  one exception is a monopolar fit's `subspace_metric`, which labels its raw-δ̂ basis
  `euclidean` (a basis label, not a fallback). This is the per-layer budget weight at
  apply time. (Coords supply target *positions*; share supplies the *budget*.)
- **`origin`** (curved only) — `invert_parameterization` of the neutral mean onto
  the surface, per layer, in authoring coords. The cold-start foot seed for the
  per-token follower and the slide target of `!`. Flat subspaces store none (foot
  = span-coord 0; routing a flat subspace through `invert_parameterization` would
  also hit `rbf_params()` and raise).
- **`explained_variance`** — recorded as a fit-quality diagnostic only; it does
  not drive gain.

For a vector, the activation-space magnitude lives in the real `node_coords`
(`coord_+ − coord_− = ‖δ_L‖`), so a fixed slide fraction displaces proportionally
more where the concept signal is bigger — there is no separate `ref_norm` factor.
The folded view `folded_vector_directions` returns `δ̂_L · share_L` for the
scale-invariant surfaces (`compare`/`why`) and GGUF export, where llama.cpp's
uniform control-vector scalar reproduces the relative per-layer weighting
(`io/gguf_io.py`). There is **no lever bake** (§5.5 explains why).

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

`anchor_origin` (pca only) post-translates the derived coords so a named node
lands at `(0, …, 0)` — RBF/affine interpolation is exact at fit nodes, so
`<manifold>%0,…,0` reproduces that anchor node's behavior. Both wrap the derived
coords in `CustomDomain(k)` with identity embedding and proceed through
`fit_affine_subspace` (pca) or `fit_layer_subspace` (spectral) unchanged.
Per-model coordinates are the architectural consequence: a Gemma fit and a Qwen
fit produce different node layouts for the same corpus heap, stored as
`node_coords` in the per-model safetensors; the diagnostics
(`PcaDiagnostics`/`SpectralDiagnostics`) ride into the sidecar.

The 2-node-vector case (§3.2) is exactly discover-`pca` with `max_dim=1`: `k`
collapses to 1, the floor is `k+1 = 2` (a flat affine subspace needs only `k+1`
poised centroids, unlike the curved `min_nodes(k) = 2k+1`), and the single derived
axis is `δ̂`.

### 3.9 SAE feature space

`--sae <release>` reconstructs each centroid (encode → decode) through the SAE
before the fit — a denoised, sparse-feature-supported centroid — and restricts the
fit to the SAE's covered layers. The fit happens in *model* space, so the hook
never touches the SAE. `core/sae.py` wraps SAELens (`load_sae_backend`); coverage
is fail-fast (`SaeCoverageError` before the pooling loop). The SAE branch still
whitens with the residual-stream whitener (the centroids are decoded back to model
space before the fit).

### 3.10 Conversational corpus generation (A2)

`session.generate_responses(concepts, kinds, *, roles=None, samples_per_prompt=1,
max_new_tokens=256, on_progress=None)` is the unified corpus generator. For each
`(concept, kind)` the model answers the **shared baseline prompts**
(`saklas/data/baseline_prompts.json`, 64) *in character*: the concept rides a
system prompt (`_system_for`, by kind — abstract → "someone {c}", concrete →
"{art} {c}") and a swapped assistant-role elicitation label (`_role_for`). It
always role-swaps (no system-only fallback). Responses are emitted samples-outer /
prompts-inner, so `response[i] ↔ baseline_prompt[i % k]` — the alignment
`compute_node_centroid` and the node corpus files assume (length a multiple of
`k`). The shared prompts hold topic common-mode across nodes, replacing the old
per-manifold scenario set (so no `scenarios.json` is written). Returns
`{concept: [response, ...]}`. The vector path calls it with `[pos, neg]`;
discover authoring (`manifold generate`, the HTTP route) loops it per node, writing
each via the resumable discover writers (`io/manifolds.py`).
`session.generate_neutral_responses` is the neutral sibling (empty system, standard
label) that fills `neutral_statements.json` with organic responses to the same
prompts. Extraction pools the swapped-back `[user, assistant]` pairs in
standard-assistant space (the node `role`, when set, overrides the label for a
persona-baselined fit).

---

## 4. Composition (dispatch-time synthesis)

The engine never holds "one manifold per concept term". At each generation
`session._compose_steering_entries` classifies the active steering expression and
composes the unified backend:

| term                         | contributes                                                   |
|------------------------------|---------------------------------------------------------------|
| `c x` / pole                 | push fragment: fold the resolved direction → unit baked dir `(1, D)`, target `[‖d_L‖]`, coeff `c` |
| `c a~b` / `a\|b`             | push fragment: the derived projected direction (materialized via `project_profile`), folded like a vector |
| `!x`                         | ablation fragment: x's per-layer directions, target → origin (0) |
| `c M%label` (affine M)       | push fragment: M's per-layer basis rows, target = node's `LayerSubspace.node_coords[idx]`, coeff `c` |
| `c[,o] M%pos` (curved M)     | a separate two-op term via `add_manifold` (along=c, onto=o)  |

**Resolving a direction (manifold-first).** A plain vector term resolves through
`session._ensure_profile_registered`, in order: (1) an in-memory baked direction
already in `_profiles` (ad-hoc `extract`/`merge`/projection results); (2) a
fitted 2-node `pca` manifold on disk — `_try_fold_manifold` loads it
(`_ensure_manifold_loaded`) and folds via `folded_vector_directions`, memoizing the
result; (3) a stale (`< PACK_FORMAT_VERSION`) legacy `vectors/<ns>/<name>/` folder
— ported to a 2-node manifold file-only (`_port_stale_legacy_vector`), then the
call raises with the exact `manifold fit` command (porting carries no tensor; it
re-fits lazily because fitting can't re-enter the generation lock from dispatch).

**Folding to a push fragment.** `fold_directions_to_subspace(name, directions,
neutral_means)` (`core/vectors.py`) folds a resolved per-layer direction into a
neutral-anchored affine `R=1` `Manifold` — a one-pole ray with `basis = d̂_L`,
`node_coords = [[‖d_L‖]]`, and `share = ‖d_L‖_M`. `session._affine_manifold_push`
then splits it into per-layer `(unit_dir rows, [‖d_L‖] coord)`, so the
synthesizer's `Δ = coeff · (coord @ basis) = coeff · d_L` reproduces the baked
direction exactly. An affine `%` term reads the fitted manifold's per-layer
`node_coords` directly (label-form only — coord-form on an affine manifold has no
interpolant). Projection terms are materialized to derived `Profile`s first
(`_materialize_projections` → `project_profile`: closed-form LEACE, Mahalanobis-only
— the session whitener must cover every projected layer, else `WhitenerError`) and
then folded like vectors.

**Grouping + synthesis.** Push + ablation fragments are grouped **by trigger**.
Each trigger group is composed by `synthesize_subspace(push, ablate,
neutral_means)` into one `SynthesizedSubspace`. Per layer (over the union of layers
any term touches):

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
weight and the steered coordinate sit on one consistent scale. Each merged
subspace registers via `SteeringManager.add_subspace`; curved `%` terms register
via `add_manifold`. So `add_subspace`/`add_manifold` see exactly one merged affine
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
  `‖H_n‖`). Tangential/directional. This replaces an additive chord — by sliding
  *on the surface* it never cuts through off-manifold low-density space.
- **onto (`o ∈ [0,1]`)** — scale `H_n` by `(1 − o)`: collapse the off-surface
  in-subspace residual onto the surface. Vacuous when the surface fills its
  subspace.

`H_o` (the off-subspace residual) is **always kept verbatim** — there is no third
op scaling the orthogonal complement of *this* subspace (which would be every
composing neighbor's span, breaking orthogonal composition). The R-dim `~`/`|`
semantics are recovered by routing those operators into the merged affine subspace
as push/ablation axes (§4).

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
sum 1 (`_manifold_layer_shares` prefers the baked `mahalanobis_share`, else the
Euclidean `‖eval_rbf(node_params)‖_F`), then:

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
replaces* its in-subspace component with the target. α clamps to `[0,1]` so `base`
is the strength ceiling. Per-persona strength variance persists — a hard persona
peaks near its coherence edge at α ≈ 1 where a robust one still has room; tune α
per target.

### 5.4 Worked dispatch trace (a folded vector)

`0.3 angry.calm`:

1. `_ensure_profile_registered("angry.calm")` loads the 2-node `pca` manifold,
   folds it (`folded_vector_directions` → `{L: δ̂_L·share_L}`), memoizes.
2. `fold_directions_to_subspace` rebuilds a neutral-anchored `R=1` ray:
   `basis = δ̂_L`, `node_coords = [[‖d_L‖]]`, `share = ‖d_L‖_M`.
3. `_affine_manifold_push` → push fragment `(δ̂_L rows, [‖d_L‖] coord, coeff 0.3)`.
4. `synthesize_subspace`: `B = δ̂_L`, `Δ_L = 0.3·‖d_L‖·δ̂_L`,
   `target_coord = [0.3·‖d_L‖]`, `share = 0.3·‖d_L‖`.
5. `add_subspace` → `apply_to_model`: `share_L` normalized to mean 1,
   `eff_along_L = share_L · 1.0`.
6. Per token, `subspace_inject` (affine shortcut): foot `q = (h − mean)·δ̂`,
   slide `q ← q + eff_along·(target − q)`, write `h ← mean + (slid q)·δ̂ + H_o`.

### 5.5 Why mean-1 share and no lever

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
no `.item()` per token. Both ride the same `HiddenCapture` plumbing as
generation; `session._begin_capture` widens capture to the union of vector +
manifold probe layers.

### 6.1 TraitMonitor (vector probes)

Scores per-layer probe directions against the running hidden state. The per-layer
similarity is the **whitened (Mahalanobis) cosine** `⟨V, h_c⟩_M / (‖V‖_M
‖h_c‖_M)` — matching the metric the fit + `~`/`|` projection use. The metric is
Mahalanobis-only and gated all-or-nothing via `covers_all`: the wired whitener must
cover every probed layer, else `_ensure_cache` raises `WhitenerError` (there is no
Euclidean path — a partial mix would fold cosines from two metrics into one
aggregate). The per-layer
probe *weight* stays `‖baked‖₂` (the bake already folded the Mahalanobis score
into the magnitude, so re-whitening would double-count). `_ensure_cache`
precomputes the whitened probe directions + their Mahalanobis norms +
device-resident Woodbury factors, so the hot path is one matmul plus a cheap
per-token `Σ⁻¹h_c` apply. The bundled probe roster comes from folding the fitted
2-node manifolds tagged in each requested category
(`session._bootstrap_manifold_probes`), so the probe directions are the same
de-rogued directions steering uses. Entry points: `score_per_token` (primary,
returns `(aggregate, per_token)`), `score_single_token{,_per_layer,_tensor}`
(inline, SSE / probe-gate callback), `score_stack`, `measure` (one-shot text).

### 6.2 ManifoldMonitor (manifold probes)

The read-side peer. `add_probe(name, manifold, *, top_n)` pre-caches each layer's
node values in reduced `(K, R)` coords. Two channels per token, EV-weighted
across layers:
- **fraction** — `‖h_par_c‖ / ‖h − mean‖`, the share of the centered activation
  living in the manifold's subspace, in `[0,1]`.
- **nearest** — the nearest node by reduced-coord `cdist`.

**Whitened readout** (Mahalanobis-only — built per probe at attach; the whitener
must cover every layer of that manifold, else `_build_whitened` raises
`WhitenerError`): both channels are their Mahalanobis forms — fraction is the
M-orthogonal subspace share `sqrt(gᵀ M_R⁻¹ g)/‖x‖_M` (`g = B Σ⁻¹ x`), nearest runs
in the `M_R` metric — the read-side analogue of the whitened/Fisher subspace the
manifold was *fitted* in. There is no Euclidean readout. `score_aggregate` is the
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
TraitMonitor, `"pad:fraction"` / `"pad@elated"` from ManifoldMonitor) — the
runtime lookup is identical; only the parser knows the difference. Gated triggers
report inactive during prefill (no reading yet) and for missing probes (no raise).

---

## 7. Grammar (`core/steering_expr.py`)

`parse_expr(text) → Steering`; `format_expr` round-trips. Every input surface
(Python, YAML, HTTP, TUI, `subspace merge`) speaks it.

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
manifold coefficient slot is `along[,onto]` (1- or 2-tuple). Variants are
`raw`/`sae[-<release>]`/`role[-<slug>]`/`from[-<src>]` (no `pca`). A bare slug
resolves through `io.selectors.resolve_bare_name`: first as a bipolar-pole node of
a 2-node `pca` manifold (`resolve_manifold_name`/`resolve_manifold_label`), then as
a multi-node manifold node label (synthesizing a label-form `ManifoldTerm`);
cross-tier ambiguity raises. Term types (`ProjectedTerm`/`AblationTerm`/
`ManifoldTerm` + plain tuples) survive as parse-time markers the dispatch
synthesizer consumes.

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
`M_L : ℝ^D_src → ℝ^D_tgt`. `transfer_profile` applies `M_L @ v_src` per layer and
re-bakes each magnitude to its *target* Mahalanobis norm so the share is in the
target metric. The target whitener is **required** and must cover the transferred
layers (Mahalanobis-only — a missing/partial whitener raises `WhitenerError`;
generate neutral activations for the target model first, there is no Euclidean
rebake). `transfer_manifold` (`io/manifolds.py`) maps a fitted manifold's per-layer
`mean → M_L mean` and `basis → basis @ M_Lᵀ`, leaves the RBF + `node_coords`
untouched (subspace/authoring-coordinate space, invariant under the model-space
map), re-bakes the Mahalanobis **share** in target space (same whitener
requirement; no lever — it's gone), clears `origin` (per-layer foot of the
*source* neutral), and writes the
`_from-<safe_src>` variant. Since a vector is a 2-node `pca` manifold, `subspace
transfer` routes to this one transfer path. Alignments cache under the *target*
model dir.

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
- **Discover-coord transfer.** Cross-model Procrustes for discover *coordinates*
  is deferred (a Gemma layout and a Qwen layout aren't comparable without it);
  `transfer_manifold` maps the per-layer subspaces but leaves the per-model
  `node_coords` as-is. Authored manifolds (shared coords) transfer cleanly.
- **MPS non-determinism.** Metal kernels are not bitwise deterministic even at
  temperature 0, so run-to-run wording jitters; compare qualitatively.

---

*Pure manifold/subspace math lives in `core/manifold.py` (fp32, dependency-free,
no session/IO coupling). The whitener is `core/mahalanobis.py`; capture + fold +
projection `core/vectors.py`; the fit pipeline `core/extraction.py`; dispatch +
injection `core/session.py` + `core/hooks.py`; reads `core/monitor.py`; grammar
`core/steering_expr.py`.*
