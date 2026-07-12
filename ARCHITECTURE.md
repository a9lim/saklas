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
  a 2-node flat subspace". `session.extract("formal.casual")` authors a 2-node
  discover-`pca` manifold (node 0 = `formal`, node 1 = `casual`) and fits it; the
  in-memory `Profile` callers get back is a *folded view* of that fit
  (`folded_vector_directions`). There is no separate baked-DiM artifact, no
  `statements.json`, no `vectors/` storage — the concept's only on-disk form is
  the 2-node manifold under `manifolds/`.
- **Whitened PCA on two centroids is `Σ⁻¹δ`** (the LDA / Fisher discriminant).
  The de-rogued "aggressive vector" direction is just *which PCA metric the fit
  uses* — gated by the same whitener-coverage check every other surface uses.

The complete bundled defaults map cleanly onto the taxonomy. Package-data folders
with missing declared node corpora are skipped by materialization, so in-progress
generation output such as a partial `emotions` run is not exposed as a default
manifold:

| artifact            | nodes | rank | structure | fit_mode   |
|---------------------|-------|------|-----------|------------|
| a concept vector    | 2     | 1    | flat      | `pca`      |
| `personas`          | 107   | ~8   | flat (auto-resolved) | `auto`     |
| `emotions`          | 20    | 3    | per-model (flat on gemma-4-12B) | `auto`     |

Every steering term — vectors, bare poles, `~`/`|` projections, `!` ablations,
and `%` manifold positions — lowers at generation time to a single per-layer
injection: `subspace_inject` (`core/manifold.py`). There is no angular/additive
mode split, no separate vector hook, and no per-fit gain knob. The flat case
takes an analytic shortcut inside that kernel; the curved case runs a per-token
nearest-point foot-follower.

There are four `fit_mode`s: `authored` (user gives domain + coords; curved),
`pca` / `spectral` (discover — labeled corpora only, coords derived per-model;
`pca` flat, `spectral` curved), and `baked` (corpus-less, a precomputed direction
from `manifold bake`). The first three flow through the fit pipeline; `baked`
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
  LayerSubspace]`, plus the calibration bakes `mahalanobis_share`, `origin`
  (per-layer authoring-coord foot of the neutral mean, curved only),
  `node_roles`, `feature_space`, and a free `metadata` dict.
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

There is **one** artifact root: `manifolds/`.

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
    <safe_model>_role-<slug>.safetensors           # reserved; role fits validate the canonical tensor
  models/<safe_model>/
    neutral_activations.json                       # atomic neutral-cache shard pointer
    neutral_activations.layer-L.gen-*.safetensors # neutral corpus × one layer, fp32
    alignments/<safe_src>.json                     # atomic factorized-affine shard pointer
    alignments/<safe_src>.layer-L.gen-*.safetensors # one bounded factor shard per layer
    jlens/active.json                              # active local/external source
    jlens/local/default/manifest.json              # Saklas-fitted lens pointer
    jlens/local/default/jlens.layer-L.gen-*.safetensors
    jlens/bindings/neuronpedia.json                # pinned HF reference only
    sae/active.json                                # active local/external source
    sae/local/<name>/manifest.json                 # Saklas-trained SAE
    sae/local/<name>/layer-L.gen-*.safetensors
    sae/bindings/<release>.json                    # SAELens/HF reference only
```

Tensor-filename variants (`io/paths.py`) are `raw` (canonical), `sae-<release>`,
`from-<safe_src>`, `role-<slug>` — at most one kind per file, mutually exclusive.
There is no `pca` variant and no method suffix; difference-of-means is the only
vector extraction method. `manifold.json.files` carries a sha256 map verified on
load (`io/manifold_folder.py` + the `packs.py` integrity helpers).

The `manifold.json::fit_mode` discriminates the folder shape and the fit path:
`authored` (curved, user coords) · `pca` (flat, derived coords) · `spectral`
(curved, derived coords) · `baked` (corpus-less, a frozen direction from
`manifold bake`).

---

## 3. Extraction

Extraction is a single pipeline, `ManifoldExtractionPipeline.fit`
(`core/extraction.py`), because concept extraction and manifold fitting are the
same operation: pool per-node centroids, fit a per-layer subspace, bake the
per-layer share, write the per-model tensor. A 2-node `pca` fit is a steering
vector; an N-node fit is a manifold. The session wraps it: `extract` /
`extract_vector_from_corpora` author a 2-node `pca` folder then fit;
`fit` fits an authored/discover folder directly.

### 3.1 Forward capture and pooling

`core/vectors.py` owns the low-level capture. Statements are captured in
**right-padded batches** (`_encode_and_capture_all_batch`, `_CAPTURE_BATCH`
pairs per forward; `_encode_and_capture_all` is the single-pair sibling);
rendering, special-token walkback, and padding remain on CPU, then the complete
ids/mask batch crosses to the accelerator once (no per-row H2D/D2H round-trip);
`_capture_all_hidden_states` hooks every layer at once and pools each row at its
last-content index *inside the hook* (per-row gather), so only `(B, D)` per layer
is retained, never `(B, T, D)`, and the MPS allocator flush is amortized per
chunk. Right-padding is exact for capture — causal attention at a row's pool
position sees only real tokens to its left, so the pooled state matches the
unpadded forward. Pooling is from the
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
`encode_and_capture_stack` is the full-`[T,D]` companion for the monitor.
`compute_node_centroid` (`core/manifold.py`) pools one node's corpus. The fit
pipeline instead uses `compute_manifold_node_stats`: a fit-wide stream fills
batches across node boundaries, tokenizes once, buckets by length, grows clean
batches up to 64, and adapts down after OOM. Capture terminates after the last
selected transformer layer, skipping unused upper blocks / final norm / LM head.
Centroid-only fits reduce by node before transfer. Raw curved fits write
source-dtype rows to a layer-major mmap spool; sigma covariance projects that
spool layer-major in bounded chunks, so it needs neither a second model pass, a
resident fp32 hidden roster, nor one small hidden-dimension GEMM per node; token-exact per-model centroid/row
caches include baseline/tokenizer-render identity and node boundaries. The final
fitted-tensor cache remains the first fast path; after it misses, the mandatory
Mahalanobis whitener is checked before this expensive capture begins. Format v4
stores independently digested, immutable generation-named per-layer centroid
and row shards: scoped fits verify/map only requested layers, disjoint top-ups
write only missing shards, row publication computes its tensor-domain digest in
the same bounded payload traversal; non-current caches miss and are recaptured normally.
Each generation fsyncs its payloads before a recovery journal and atomic pointer;
the pointer directory is fsynced before superseded generations are collected.
The prior pointer remains authoritative on failure, and a complete orphaned
generation is adopted under the stem lock on the next fit without another model
pass. Coverage is unioned for
subset/top-up reuse; cached and newly captured row stores compose as zero-copy
layer views for the covariance pass. Forced subset fits recapture only their
selected layers and carry all other v4 pointers forward. The exclusive stem lock
ends after cache publication; a live PID lease protects mapped rows through the
later covariance pass, and deferred auto-curved row top-ups merge the latest
pointer when they briefly reacquire the lock. Oldest-group pruning runs after the active stem transaction
and locks each victim before deletion (skipping live leases), past a configurable disk bound;
geometry-only refits skip capture entirely. Fitted tensors carry the
loaded-model fingerprint and selected layer set in their cache identity.
Long compute is serialized only per exact model/variant target; folder locks
cover an exact-input snapshot and revision-CAS publication, so readers and other
targets continue while fitting. Scoped clear epochs and a folder artifact id
prevent a paused fit from undoing clear or publishing into an rm/recreated folder.
Every destructive authoring reset or HF replacement follows the same manifest →
sorted stable-pair lock order as lifecycle deletion before it removes or swaps a
folder. Corpus-less baked writers publish each tensor/sidecar proof as part of
the folder transaction; if manifest proof publication fails, an identical
ordinary retry replaces the unproven pair and resumes a matching multi-model
merge from its verified prefix. Merge model discovery carries its verified,
folded components into preparation, so it never repeats artifact hash/load/fold
work merely to rediscover the shared model intersection.

### 3.2 A steering vector as a 2-node fit

`session.extract(concept, baseline, *, kind="abstract")`:

1. Splits the composite (`formal.casual` → pos `formal`, neg `casual`). A **monopolar**
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
(`extract()`, `manifold compare`/`why`, GGUF export) without a second stored
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
manifold fit, projection, the monitor, transfer, and `manifold compare` — is
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
  baseline rather than encoding a common offset/intensity. At K=2 (stacked
  `[μ_pos, μ_neg]`) this is exactly the pos/neg opposite-sign test, bit-identical
  to a difference-of-means vector's keep set; for K>2 (e.g. `personas`) it
  prunes per-axis. The
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
fits the exact or penalized interpolant. `prepare_rbf_fit_plan` builds the
layout-only kernel/polynomial blocks, QR/eigensystem, λ grid, and fixed-λ LU once;
every activation layer and the sigma field reuse it. Centered `Σ⁻¹X` rows and
Grams are likewise computed once per layer and shared by discovery, topology,
Fisher PCA, and neutral-layout anchoring. The dense saddle is symmetric-
*indefinite* (never Cholesky; scipy is not pulled in). At n=1 over an open axis
the `r³` polyharmonic spline reproduces the natural cubic spline exactly.

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
  The unified `Monitor` **also** read-weights by it (normalized to sum 1, not
  mean 1): combining each layer's geometry into one cross-layer reading, the layer
  carrying the most steering budget is the most reliable to read from — so one
  baked quantity drives both the steer side and the read side. (This replaced an
  `explained_variance` read weight, which normalized away the per-layer signal
  *magnitude* the pooling needs — and was identically 1.0 for every 2-node
  concept, so it weighted the whole bipolar roster uniformly.)
- **`origin`** (curved only) — `invert_parameterization` of the neutral mean onto
  the surface, per layer, in authoring coords. The cold-start foot seed for the
  per-token follower and the slide target of `!`. Flat subspaces store none (foot
  = span-coord 0; routing a flat subspace through `invert_parameterization` would
  also hit `rbf_params()` and raise).

For a vector, the activation-space magnitude lives in the real `node_coords`
(`coord_+ − coord_− = ‖δ_L‖`), so a fixed slide fraction displaces proportionally
more where the concept signal is bigger — there is no separate `ref_norm` factor.
The folded view `folded_vector_directions` returns `δ̂_L · share_L` for the
scale-invariant surfaces (`compare`/`why`) and GGUF export, where llama.cpp's
uniform control-vector scalar reproduces the relative per-layer weighting
(`io/gguf_io.py`). There is **no lever bake** (§5.5 explains why).

### 3.8 Discover-mode coordinate derivation

When the user supplies labeled corpora but no coordinate system, `fit_mode` is
`pca` or `spectral` and coordinates are derived per-model at fit time. The
derivation is **layer-agnostic**: there is no reference layer. Each fit layer
contributes its whitened, node-mean-centered `(K, K)` Gram `X̃_L Σ_L⁻¹ X̃_Lᵀ`
(via `LayerWhitener.subspace_gram`), and the **consensus Gram** is their mean
`Ḡ = mean_L X̃_L Σ_L⁻¹ X̃_Lᵀ`. Whitening puts every layer in common
(background-σ) units, so the raw average is **signal-weighted**: a layer where
the nodes aren't separated contributes a near-zero Gram and falls out of the
mean, while a layer where the concept is strongly represented dominates — the
layout draws on whichever layers the concept actually lives in, without picking a
band by hand. The `(K, K)` Gram is the layer-invariant object, which is what
makes averaging across heterogeneous-norm layers well-posed. Both methods embed
that one consensus Gram (`core/manifold.py`):

- **`derive_pca_coords`** — eigendecompose `Ḡ` (`torch.linalg.eigh`, the Gram
  analogue of SVD-of-centered-centroids: eigenvalues are the component variances,
  eigenvectors scaled by `√λ` are the scores `U S`); pick the smallest prefix
  whose cumulative variance crosses `var_threshold` (default 0.70), capped at
  `max_dim` (default 8).
- **`derive_spectral_coords`** — read pairwise distances off the Gram
  (`d²_ij = Ḡ_ii + Ḡ_jj − 2 Ḡ_ij`, the mean per-layer Mahalanobis distance =
  the distance in concatenated-whitened space), then symmetric (union) k-NN graph,
  heat-kernel edge weights, eigendecompose the symmetric normalized Laplacian,
  drop the smallest eigenvalue, pick `k` by the eigenvalue-*ratio* cliff
  `argmax(λ_{k+1}/λ_k)` (the absolute gap over-picks on S¹). A disconnected graph
  raises, naming the component count and pointing at `--k-nn`/`--method pca`.

PCA embeds `Ḡ` linearly and spectral embeds it locally, but it is the **same**
consensus geometry — `derive_pca_coords`'s eigendecomposition and
`derive_spectral_coords`'s distances both come from one `Ḡ`, which is the
Gram of the concatenated-whitened centroid stack.

The per-layer summands of the consensus are a free diagnostic: each layer's
`tr(G_L) = Σ_k ‖x̃_k‖²_M` is the total whitened between-node spread at layer `L`,
in background-σ² units (comparable across layers). The fit stamps these into the
sidecar as `node_spread_per_layer` — the concept's whitened signal-by-layer
profile, surfaced by `pack show`. It is diagnostic only (nothing runtime
branches on it) and is computed for every K≥2 fit, authored included. It is the
*full-space* sibling of the apply-time `mahalanobis_share` (§3.7), which is the
same whitened spread restricted to the fitted steerable subspace; a layer where
`tr(G_L)` is large but the share is small is one whose subspace is dropping concept
signal (the fitted subspace captures little of the between-node spread).

The derived coords come out PCA-mean-centered and wrap in `CustomDomain(k)` with
identity embedding, then proceed through `fit_affine_subspace` (pca) or
`fit_layer_subspace` (spectral) unchanged. There is no `anchor_origin` knob: the
steer-time origin is always the projection of the per-model neutral mean onto the
subspace (the affine fit neutral-anchors the frame — coord 0 in each layer's real
reduced frame is neutral, §3.7), so the manifold's origin already carries the
principled "no behavioral shift from neutral" meaning for free. Per-model
coordinates are the architectural consequence: a Gemma fit and a Qwen fit produce
different node layouts for the same corpus heap, stored as `node_coords` in the
per-model safetensors; the diagnostics (`PcaDiagnostics`/`SpectralDiagnostics`)
ride into the sidecar.

The 2-node-vector case (§3.2) is exactly discover-`pca` with `max_dim=1`: `k`
collapses to 1, the floor is `k+1 = 2` (a flat affine subspace needs only `k+1`
poised centroids, unlike the curved `min_nodes(k) = 2k+1`), and the single derived
axis is `δ̂`.

### 3.9 SAE feature space

`--sae <release>` reconstructs each centroid (encode → decode) through the SAE
before the fit — a denoised, sparse-feature-supported centroid — and restricts the
fit to the SAE's covered layers. The fit happens in *model* space, so the hook
never touches the SAE. `core/sae.py` wraps SAELens (`load_sae_backend`); registry
resolution is eager, weights load lazily with only one layer resident, and a valid
fitted tensor cache hit does not import/load SAELens. Coverage
is fail-fast (`SaeCoverageError` before the pooling loop). The SAE branch still
whitens with the residual-stream whitener (the centroids are decoded back to model
space before the fit). Reconstruction transfers the raw centroid roster one layer
at a time: once a decoded `(K,D)` replacement exists, that layer's raw `(K,D)`
stack is released. A multi-node SAE fit therefore never retains complete raw and
reconstructed `K×D×L` rosters beside the later whitened rows; the one-node
monopolar branch consumes its raw centroid before this ownership transfer.

### 3.10 Conversational corpus generation (A2)

`session.generate_responses(concepts, kinds, *, roles=None, samples_per_prompt=1,
max_new_tokens=256, on_progress=None)` is the unified corpus generator. For each
`(concept, kind)` the model answers the **shared baseline prompts**
(`saklas/data/baseline_prompts.json`, 48) *in character*: the concept rides a
system prompt (`_system_for`, by kind — abstract → "someone {c}", concrete →
"{art} {c}") led by a shared one-paragraph length directive (`_LENGTH_DIRECTIVE`)
to keep responses from rambling past the token cap, and a swapped assistant-role
elicitation label (`_role_for`). It always role-swaps (no system-only fallback).
Responses are emitted samples-outer /
prompts-inner, so `response[i] ↔ baseline_prompt[i % k]` — the alignment
`compute_node_centroid` and the node corpus files assume (length a multiple of
`k`). The shared prompts hold topic common-mode across nodes, replacing the old
per-manifold scenario set (so no `scenarios.json` is written). Returns
`{concept: [response, ...]}`. The vector path calls it with `[pos, neg]`;
discover authoring (`manifold generate`, the HTTP route) loops it per node, writing
each via the resumable discover writers (`io/manifold_authoring.py`).
`session.generate_neutral_responses` is the neutral sibling (the length directive
as its *only* system — no persona — standard label) that fills
`neutral_statements.json` with organic responses to the same prompts. The
directive is the only framing neutral shares with the node corpora (it leads every
node system too), so it cancels at extraction while leaving neutral the
default-voice reference the contrast subtracts against. Extraction pools the
swapped-back `[system: directive, user, assistant]` turns in standard-assistant
space — **the persona is generation-only, not reconstructed at capture**, so a
node centroid sees only the brevity directive (matching the framing the response
was generated under; the node `role`, when set, overrides the assistant label for
a persona-baselined fit).

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
`session.ensure_profile_registered`, in order: (1) an in-memory baked direction
already in `_profiles` (ad-hoc `extract`/`merge`/projection results); (2) a
fitted 2-node `pca` manifold on disk — `_try_fold_manifold` loads it
(`ensure_manifold_loaded`) and folds via `folded_vector_directions`, memoizing the
result. Pre-manifold vector folders are not part of the current runtime format.

**Folding to a push fragment.** `fold_directions_to_subspace(name, directions,
neutral_means)` (`core/vectors.py`) folds a resolved per-layer direction into a
neutral-anchored affine `R=1` `Manifold` — a one-pole ray with `basis = d̂_L`,
`node_coords = [[‖d_L‖]]`, and `share = ‖d_L‖_M`. `session._affine_manifold_push`
then splits it into per-layer `(unit_dir rows, [‖d_L‖] coord)`, so the
synthesizer's `Δ = coeff · (coord @ basis) = coeff · d_L` reproduces the baked
direction exactly. An affine `%` term in **label form** reads the fitted manifold's
per-layer `node_coords[idx]` directly; in **coord form** it maps the free authoring
coordinate into each layer's reduced frame by cardinal RBF interpolation over the
node layout (`manifold.rbf_cardinal_weights` on the shared `node_coords`, solved
once in authoring space), so the per-layer target is the layout blend
`node_coords_L.T @ w`. The weights are exact at the nodes (`w = e_idx`), so coord
form placed at a node's coords reproduces label form and interpolates between nodes
off-node — the flat analogue of a curved `%` following its RBF surface instead of a
straight chord. Projection terms are materialized to derived `Profile`s first
(`_materialize_projections` → `project_profile`: closed-form LEACE, Mahalanobis-only
— the session whitener must cover every projected layer, else `WhitenerError`) and
then folded like vectors.

**Grouping + synthesis.** Push + ablation fragments are grouped **by trigger**.
Each trigger group is composed by `synthesize_subspace(push, ablate,
neutral_means, *, whitener)` into one `SynthesizedSubspace`. Per layer (over the
union of layers any term touches):

1. Flatten every present push fragment's basis rows, then every ablation
   fragment's rows, into one ordered list (push first); orthonormalize the union
   (`_ortho_basis`) → the merged `(R, D)` basis `B`.
2. **Whitened-normalized push** (the session always passes `self.whitener`; gated
   all-or-nothing on `covers_all`). Each push fragment's raw neutral→node
   direction `dirᵢ = coord_targetᵢ @ basis_rowsᵢ` is unit-normalized **in the
   Mahalanobis metric** (`dirᵢ/‖dirᵢ‖_M`) and scaled by its user coeff, then
   summed: `target_coord = Σᵢ coeffᵢ·(B@dirᵢ)/‖dirᵢ‖_M`. This *strips the raw node
   distance* from the direction (only the unit whitened bearing survives) while
   keeping α as the strength. The ablation-only axes carry `target ≈ 0`.
3. Per-axis collapse mask `κ` (R,): `0` on the push span, `1` on the
   ablation-only complement, derived as `κ = 1 − ‖proj onto push span‖²` (robust
   to orthonormalization order). The kernel *translates* push axes by the fixed
   offset but *collapses* `κ=1` axes toward 0 — a `target ≈ 0` alone no longer
   removes an ablated direction (translate-by-offset would leave it untouched), so
   κ is what does the ablation post translate-not-collapse.
4. `share = ‖Δ‖_M` — the **whitened** magnitude of the raw coeff-weighted
   displacement `Δ = Σ coeffᵢ·dirᵢ`; this is the per-layer *profile* (the absolute
   node-distance scale cancels under apply-time mean-1 normalization, leaving only
   the relative across-layer shape). A pure-ablation layer weights by the summed
   ablation-row whitened magnitude instead.

**Why whitened.** Steering by the raw-Euclidean node distance (the prior `target =
B@Δ`, `‖target‖ = ‖Δ‖₂`) coupled `along` to an accident of where each centroid
sits: a tight bipolar pole sits ~0.3 from neutral while a far persona centroid
sits ~17 — a ~100× spread — so a single `along` gain over-pushed far nodes into
incoherence and left near ones doing nothing (`0.5 formal%formal` ≈ dead,
`0.5 personas%caveman` slammed). Raw-Euclidean distance is the un-whitened,
rogue-dominated metric the rest of the engine avoids; it leaked back in through
`node_coords` as the `%` target. Whitened-unit normalization removes it: every
target lands on **one uniform whitened budget** (`Σ_L eff_along_L = gain·n_layers`),
linear in α, concentrated by the per-layer profile — so `along` finally means the
same thing across ranks/targets. (Whitening the *direction* toward a fixed node is
otherwise metric-invariant — only the calibration changes — so the push still aims
at the node centroid, just measured in std-units.) A whitener-absent path (CPU
stub / degenerate fit) falls back to the raw-Euclidean `target_coord = B@Δ`,
`share = ‖Δ‖₂` unchanged. Each merged subspace registers via
`SteeringManager.add_subspace`; curved `%` terms register via `add_manifold`. So
`add_subspace`/`add_manifold` see exactly one merged affine subspace per trigger
group plus zero or more curved manifolds.

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
`h = mean + h_par + h_perp` against the layer's affine subspace inline,
where `h_par` is the in-subspace component and `h_perp = H_o` the off-subspace
residual, then applies two operations:

- **along (`a ∈ ℝ`)** — **translate** the projected foot by the fixed
  neutral→target offset scaled by `a` (`domain.translate_foot`), *not* a lerp onto
  the absolute target. Every token's foot shifts by the same displacement, which
  preserves the per-token in-subspace spread and keeps strong steer coherent —
  collapsing each foot onto the absolute target instead erases that spread and
  degenerates into looping (the kernel ablation that motivated the
  translate-not-collapse change). It then transports the
  off-*manifold*-in-subspace residual `H_n` from the tangent frame at the old foot
  to the frame at the new foot by the **minimal orthogonal (principal-angle)
  rotation** between the two tangent subspaces (`_frame_rotation_transport`):
  orthonormalize each frame (modified Gram-Schmidt — `torch.linalg.qr` is
  unimplemented on MPS), take the principal angles from the SVD of their `n×n`
  overlap, and rotate each pair of principal vectors `aᵢ → bᵢ` in its own plane.
  This is *exactly* the identity when the foot doesn't move (`p_new == p`, i.e.
  `along=0`), so the curved path is identity at rest **regardless of foot-solve
  accuracy**, and it is norm-preserving and lossless. (The former
  project-onto-normal + renorm was neither: it discarded the residual's
  tangential-at-the-foot component every fire — which never vanishes at an
  approximate foot — corrupting any off-neutral activation by 20–150% with *zero*
  steering, compounding across layers into degenerate looping.) Tangential/
  directional; by moving *on the surface* it never cuts through off-manifold
  low-density space. A per-axis collapse mask `κ` (§4) overrides this for ablation
  axes (`κ=1`): those collapse toward 0 instead of translating.
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
(`norm_cap = 3.0`) is the only norm guard on this curved path — `onto` is *meant*
to shrink `‖h − mean‖`, so there is no global norm preservation.

**Flat (affine) shortcut.** When `subspace.is_affine` the surface fills the
subspace: reduced coords *are* authoring coords (identity), the foot is `q`
exactly, `H_n ≡ 0`, and `onto` is vacuous. So the kernel skips the Gauss-Newton
foot solve, the RBF eval, and the tangent Gram-solve — it computes
`p_new = q + a·(target − κ·q)` and keeps `H_o` verbatim. The per-axis κ from
`synthesize_subspace` selects per axis: `κ=0` push axes translate by the fixed
`a·target` offset (preserving per-token spread), `κ=1` ablation axes do
`q + a·(0 − q)` (collapse the component toward 0), so push and ablation share one
analytic op. This is load-bearing for throughput: a folded vector is the common
case and the curved per-token solve would blow the throughput invariant. The
affine branch also drops the `norm_cap` — the displacement `(p_new − q)@basis` is
bounded and added to a large-norm residual, so it can't push `‖h_new‖` past the
cap, and skipping it saves two per-fire full-width norm reductions.

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
`AFTER_THINKING` group during prefill). The dominant case — one always-active
(`Trigger.BOTH`) affine group — is precomputed at `recompose` into
`_single_affine_fast`, and `hook_fn` short-circuits to it: one analytic slide + an
in-place write, consulting no `TriggerContext` and threading no foot state.
Curved / gated / phased steering runs the general (ctx-consulting) path. Two
eligibility signals feed `core/cuda_graphs.py`: `all_fast_path()` (unsteered — no
hooks) and `static_steerable()` (every hook is the static-affine fast path). Both
make `torch.compile` / StaticCache graph capture eligible — StaticCache never
bypasses the forward hooks, so the static-affine injection still applies, and its
fixed tensor-op sequence is capturable. (StaticCache-with-steering is CUDA-only;
the through-the-hook graph capture wants a CUDA validation pass.)

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
  _SUBSPACE_GAIN`, `onto = 0`. The coefficient α is already folded into
  `target_coord` by `synthesize_subspace` (`Δ = Σ coeffᵢ·poleᵢ`, required for
  multi-term composition), so α scales the *size* of the translate offset and
  `share_L·base` scales the per-layer slide fraction. α is unclamped (it sets the
  offset magnitude).
- **Curved manifold** (`add_manifold`): `eff_along_L = along · share_L ·
  _MANIFOLD_ALONG_GAIN`, `eff_onto_L = clamp(onto · share_L · _MANIFOLD_ONTO_GAIN, 0,
  1)`; the target is the full node position, so `along` is the slide fraction (the
  historic `%` knob), clamped to `[0,1]` at apply time. `onto` stays a `[0,1]`
  collapse fraction.

**Three gain constants.** `_SUBSPACE_GAIN = 16.0` (live-calibrated) scales
**along** on the **affine** path (whitened-unit target, free push magnitude);
`_MANIFOLD_ALONG_GAIN = 4.0` scales **along** on the **curved** path (raw node-coord
target, so `eff_along` is a fraction-to-node — `1.0` lands on it);
`_MANIFOLD_ONTO_GAIN = 0.5` now scales **onto** only
(the off-surface collapse share-weight, curved-only — calibrated on the
gemma-4-12b `emotions%dominant` onto sweep: at `1.0` even `onto=0.5` fragmented and
`onto=1.0` collapsed into looping, since collapsing the off-surface residual
erases the per-token spread; `0.5` makes `onto∈[0,1]` a usable dial with `1.0` a
coherent ceiling). The split is the
translate-not-collapse consequence: a fixed-offset translate is *unbounded* where
the old lerp-onto-target saturated (the offset compounds across layers rather than
landing on the target), so the slide gain runs ~an order of magnitude below the
old collapse gain. **Whitened-target recalibration:** since the affine target is
now a *whitened-unit* direction (§5.3), the avg per-layer whitened push is `GAIN·α`,
independent of which target — where the prior raw-Euclidean target scaled the push
by each node's distance from neutral (caveman ~17, formal ~0.3, a ~100× spread
that left near targets dead). The old `0.125` was tuned against that ~17 scale, so
on the unit scale it under-pushes ~100×; `16.0` is live-calibrated on a gemma-4-12b
α-sweep (`formal%formal`, `personas%caveman`, `personas%hacker`) so the recommended
α≈0.5 lands at effective gain `GAIN·α ≈ 8` — clearly steering every target while
staying coherent for the most fragile persona. Coherence ceilings vary ~2× per
target (hacker shatters at effective ~12, caveman ~17, formal past ~22 — the §10
per-persona variance a scalar gain can't unify), so α≈1.0 is the strong/over-steer
zone where hard personas break. Committed but tagged a prototype in the source.

There is **no `[0,1]` clamp and no water-fill on `along`**: a high-signal layer is
*meant* to overshoot, the de-rogued whitened coords keep the overshoot controlled,
and on the curved path `norm_cap = 3·‖h‖` is the only backstop (the affine fast
path relies on the controlled coords alone). `onto` keeps its clamp
(a collapse fraction past 1 inverts the residual). Per-persona strength variance
persists — a hard persona peaks near its coherence edge where a robust one still
has room; tune α per target.

### 5.4 Worked dispatch trace (a folded vector)

`0.3 formal.casual`:

1. `ensure_profile_registered("formal.casual")` loads the 2-node `pca` manifold,
   folds it (`folded_vector_directions` → `{L: δ̂_L·share_L}`), memoizes.
2. `fold_directions_to_subspace` rebuilds a neutral-anchored `R=1` ray:
   `basis = δ̂_L`, `node_coords = [[‖d_L‖]]`, `share = ‖d_L‖_M`.
3. `_affine_manifold_push` → push fragment `(δ̂_L rows, [‖d_L‖] coord, coeff 0.3)`.
4. `synthesize_subspace` (whitened, session whitener covers all): `B = δ̂_L`, raw
   dir `‖d_L‖·δ̂_L` whitened-unit-normalized then ×coeff →
   `target_coord = [0.3 / ‖δ̂_L‖_M]` (so `‖target@B‖_M = 0.3`), profile
   `share = ‖0.3·d_L‖_M = 0.3·‖d_L‖_M`, `κ = [0]` (pure push).
5. `add_subspace` → `apply_to_model`: `share_L` normalized to mean 1,
   `eff_along_L = share_L · 16.0`. Avg per-layer whitened push = `16.0·0.3` —
   target-independent (a far persona node lands on the same budget).
6. Per token, `subspace_inject` (affine shortcut): foot `q = (h − mean)·δ̂`,
   translate `q ← q + eff_along·(target − κ·q) = q + eff_along·target` (κ=0),
   write `h ← mean + (translated q)·δ̂ + H_o`.

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

`core/monitor.py` — **one** read-side monitor, `Monitor` (the unification of the
former `TraitMonitor` + `ManifoldMonitor`), hook-driven (inline with generation,
no second forward pass), fp32. It reads a fitted subspace and emits one
reading shape (`ProbeReading` — `coords` + `fraction` + `nearest` + `residual`,
plus the `*_per_layer` traces) for flat and curved probes alike — the read-side
peer of the unified `subspace_inject` kernel. A 2-node concept axis, the 107-node
`personas` fan, and the `emotions` affect surface are all just probes on the one
monitor. It rides the same `HiddenCapture` plumbing as generation;
`session._begin_capture` widens capture to the union of every probe's layers
(`Monitor.probe_layers`; `attached_layers` survives as an alias for the server/TUI
surfaces that consumed the former `ManifoldMonitor`).

### 6.1 One read shape (every field, every token), two execution paths

`_score_full` produces one full `ProbeReading` per probe — **no flat/curved field
asymmetry**, the full per-token information (nearest, curved coords, residual,
per-layer) is the research-tool priority. Execution is no-redundancy: the whole
**flat** roster is scored together in `_score_flat_batched` (one `Σ⁻¹h` Woodbury
apply + stacked / block-diagonal matmuls + a single host transfer per layer,
scattered into global per-probe slots), while **curved** probes run the per-probe
`_score_probe_full` foot solve — warm-started across decode tokens from the previous
foot (`enable_curved_warm`) on the sequential live path. Per-token scoring is itself
**conditional**: when nothing consumes a per-token reading (no probe gate, no live
stream) the session skips it and pools the aggregate once at the last content token
(`score_aggregate`); a gate / stream consumer gets per-token incremental scoring.
Per layer `_layer_geometry` returns the M-orthogonal **fraction**
`sqrt(gᵀ M_R⁻¹ g)/‖x‖_M` (`g = B Σ⁻¹ x`), the whitened query for the **nearest**
`M_R`-metric node, and the M-projection reduced coord `c = M_R⁻¹ g`. From there:

- a **flat** (affine) probe recovers `coords` through the affine reduced→domain
  map (`coord_S @ c + coord_b`); off-surface `residual` is identically `0` (the
  surface fills its subspace);
- a **curved** probe recovers `coords` through the `invert_parameterization`
  nearest-point foot solve, which also yields a real normalized off-surface
  `residual`.

Coords are **domain-frame**: each layer's raw reduced coord lives in that layer's
own `‖δ_L‖` units, so it is mapped to the shared domain *before* EV-averaging; the
reference is each node's *whitened* read-coord, not the Euclidean `node_coords`. At
rank-1 this is the pole-normalized coordinate — `1.0` at the positive node, signed,
unbounded past it. A 1-node fold (a monopolar `extract`, an ad-hoc `probe()`
direction) is the K=1 ray case (`coord_b = 0`, anchored through the origin). Curved
coords now fill **per token** too — the foot solve runs in the decode loop, the
accepted throughput cost of the live readout.

The read is Mahalanobis-only, gated all-or-nothing via `covers_all` (the whitener
is built per probe at attach by `_build_whitened_factors` — a missing / non-covering
whitener raises `WhitenerError`, no Euclidean path), sharing the per-layer
`_LayerWhiten` build, the attach-time node cache (`_attach_manifold_probe`), and
`_layer_geometry`. `__init__`'s `layer_means` is vestigial on the hot path — the
readout centers on each fit's own `LayerSubspace.mean`.

### 6.2 Entry points + gate channels

`add_probe(name, manifold, *, top_n)` attaches any probe shape; `remove_probe`
detaches. Scoring entry points all share `_score_probe_full`: `score_per_token`
(primary — returns `(aggregate readings, per-token axis-0 coord stream)`),
`score_single_token{,_per_layer}` (inline, SSE / probe-gate callback — `_per_layer`
is a view over the reading's `coords_per_layer` backing the loom heatmap),
`measure_from_hidden`, `score_stack`, and `score_aggregate` (end-of-gen — pools the
last non-special token per layer and calls the *same* `_score_probe_full`, so the
aggregate at a token index is bit-identical to the live read at that token), plus
the live-mean trio `begin/update/end_live`. The one-shot re-render text scorer
(`measure`) is gone — every read source is live hooks scoring captured hidden
states. The bundled probe roster is the fitted 2-node manifolds tagged in each
requested category — plus, in the default (`probes is None`) roster, every
already-fitted bundled multi-node manifold (`personas`, `emotions`) attached
under its `default/<name>` selector — handed in directly by
`session._bootstrap_manifold_probes` (no fold), so the read frame is the same
de-rogued subspace steering uses.

`flat_scalars` (one staticmethod) flattens a readings dict to the probe-gate
channels: `"<name>"` (= coords axis 0) and `"<name>[i]"` per coord axis,
`"<name>:fraction"`, and `"<name>@<label>" = −distance` per nearest node (signed so
"larger = closer"). Under the unified full reading every probe — flat and curved —
carries coords *and* nearest, so flat probes now expose `@label` similarity gates
too (`@when:personas@hacker`) and the gate grammar is uniform. The session merges
these into `TriggerContext.probe_scores`. (For the common monitored case — probes
attached, no `return_hidden` — capture runs incremental: each token is scored live
and only the latest per-layer hidden slice + per-token `ProbeReading` rows are
retained, O(layers·D) not O(T·layers·D). `return_hidden` falls back to full
retention + `score_per_token`; the reads are full `ProbeReading`s either way — the
win is memory, not throughput.)

### 6.3 Probe gates / triggers

`core/triggers.py` — `Trigger` (frozen) carries phase flags + an optional
`ProbeGate`. `Trigger.active(ctx)` consults the phase flags and, when gated,
`ctx.probe_scores[gate.probe]` against `score <op> threshold`. The gate key is the
canonical scalar key `Monitor.flat_scalars` emits — a coordinate axis
(`"confident.uncertain"` = axis 0, or `"personas[3]"` = axis 3 via the
`@when:<probe>[<i>]` grammar), a subspace fraction (`"emotions:fraction"`), or a label
similarity (`"emotions@happy"`). Every shape composes with a leading `<ns>/`
segment (`"jlens/fake"`, `"default/emotions@happy"`), stored verbatim. The
runtime lookup is identical for every shape; only
the parser knows the difference. Gated triggers report inactive during prefill (no
reading yet) and for missing probes (no raise).

### 6.4 The Jacobian lens (verbalizable-workspace reads)

A second, per-model read primitive alongside the whitened probe reads (Gurnee
et al., Transformer Circuits 2026). `core/jlens.py` fits
`J_l = E[∂h_final/∂h_l]` per source layer over a web-text corpus — the only
backward passes in saklas (the estimator seeds an autograd leaf at the first
fitted block's output under `torch.enable_grad()` and reads per-layer grads with
`torch.autograd.grad`; everything else stays `inference_mode`). The default fit
uses exact ragged prompt microbatches (CPU/CUDA default 4, MPS 2) plus batched
VJPs (`is_grads_batched=True`) for `ceil(d_model/dim_batch)` output-dim blocks,
with an exact scalar fallback and env-overridable replicated reference mode
(`SAKLAS_JLENS_VJP`). Transparent mean-position probe identities collapse each
source derivative inside autograd from `[rows,B,T,D]` to `[rows,B,D]` while
leaving the forward and upstream gradient unchanged. A final-block hook stops before norm + LM head. Every
backend transfers byte-budgeted, allocation-adaptive row stripes directly into
the CPU accumulator; an OOM
rebuilds the graph at the first uncommitted row. Self-contained checkpoints
(`jlens.partial.*`) are written as immutable per-layer shards directly from raw
accumulator sums, avoiding a second full fp32 lens and supporting repeated
interruption or missing-layer top-up resume. A resumed prefix is converted from
average to weighted sum in place and becomes the tail estimator's accumulator,
so resume retains one full fp32 lens rather than two. Sidecar progress is compared
before payload load, so a farther self-contained checkpoint displaces the older
durable/resident matrices without a transient two-lens peak; failed checkpoint
digest validation falls back to the durable prefix only after releasing the bad
payload. Checkpoints carry the token-id hash of their consumed prefix, so an
interrupted shorter corpus can be extended without fabricating the future full
hash or restarting. Sparse layer top-ups reuse the unchanged durable shard
pointers and write only new matrices. A transaction-scoped verified-pointer
proof avoids rehashing just-loaded reuse shards; unverified callers still hash.
Fresh same-corpus subset no-ops materialize only their requested v6 shards and
leave the full durable union published. Since v6 records one corpus progress for
all layers, extending a strict subset of a durable superset is rejected before
matrix load unless the caller requests the full union or explicitly forces
replacement; carrying unrequested old-prefix layers would make that progress
false. Missing-layer top-ups preserve the full unchanged union, while selected
resume views release their source containers before estimator entry.
The streamed
safetensors writer never retains a second complete tensor mapping. Normal corpus extension
resumes from an exact token-id prefix; the default dataset is commit-pinned;
exact source/live-model fingerprints invalidate mutable revisions. A complete
terminal checkpoint is fsynced and promoted at finalization instead of being
rewritten or immediately rehashed; otherwise each fp32 layer shard is streamed once, with its payload
digest verified on final and checkpoint loads. Pointer-directory fsync follows
pointer unlink before checkpoint/full-artifact shard GC and follows pointer
publication before old-generation GC; fit preflight reaps crash-left streamed
temporaries. Exact no-op recovery removes a
checkpoint left by a crash after final publication only when the final pointer
provably subsumes its corpus, layers, estimator policy, and effective progress.
The runtime source (`io/lens.py` + `io/lens_sources.py`) is either a
Saklas-owned local fit under `models/<safe>/jlens/local/default/` or a pinned
external binding whose provider payload remains in the Hugging Face cache.
Both adapt to the same `JacobianLens` and support four
consumers with zero hot-path cost when unused:

- the **readout** `softmax(W_U · norm(J_l h))` (`session.jlens_readout`
  offline; `enable_live_lens` per decode step at the token tap — no forward
  hooks, so steering fast-path/compile eligibility is untouched);
- **steering atoms**: `jlens/<word>` lowers to the per-layer direction
  `W_U[v] @ J_l` registered lazily as an ordinary profile — it folds, pushes,
  ablates, and projects like any extracted vector, but restricted to the
  workspace band (40–90% depth): in the motor regime the direction converges
  on the raw unembedding row, so pushing there is token-forcing, not concept
  induction (live-verified: unrestricted, it shatters into token loops at
  every α). Lens atoms run hotter than concept vectors — α≈0.3 is the
  gemma-3-4b sweet spot;
- **probes + gates**: a `jlens/<word>` probe reads the *readout channel*, not
  a whitened coordinate — `add_probe` lands it in the session lens-probe
  registry (never the Monitor; no whitener, no direction fold), and the
  reading is the token's standing in the band readout: ONE channel,
  `coords = (strength,)` — the mean band **probability** `mean_l p_l(v)`
  (the `@when:jlens/<word>` gate channel — [0,1], the workspace
  `strength`, one number across every card and layer; a within-layer max
  normalization is not apples-to-apples, and the depth CoM weights by the
  same `p_l`, so `p_l` is the one unit behind every lens statistic). The
  synthesized `ProbeReading` carries per-layer `(p_l,)`, and the live
  top-k display wire reports per-layer softmax probabilities too. Scoring is post-forward on the lens path
  (display rides the live-lens step's logits; gate scalars compute once per
  forward in the gating callback and stash for the display step; the
  finalize aggregate pools the capture tail ring) — a lens gate forces its
  own per-step compute regardless of the live toggles, and a lens-only gate
  does not force per-token monitor scoring. The steering direction and the
  probe read are deliberately different objects: pushing acts on the
  activation, the probe asks the paper's question — *how disposed is the
  model to say this word*;
- the **J-space decomposition** (`sparse_nonneg_decompose`): greedy pursuit of
  any direction against the dictionary `W_U J_l` (never materialized), the
  per-layer *verbalizable share* + contributing tokens.

### 6.5 Sparse-autoencoder runtime (feature reads)

`core/sae.py` serves a common backend protocol from two owners. Fit-time
`--sae` still reconstructs manifold node centroids through a lazily loaded SAE;
the live runtime keeps one selected encoder/decoder pair resident on a session.
The default hook layer is the release-covered layer nearest 65% model depth,
preferring the 40–90% workspace band; `--layer` may select another covered
layer. Published SAELens weights remain in the Hugging Face cache, with only a
small binding and optional Neuronpedia labels under `sae/bindings/`. Native
`sae train` fits a residual-post ReLU SAE over an arbitrary Saklas-supported HF
model and stores its fp32 weights under `sae/local/<name>/`. `sae/active.json`
selects either owner for later sessions; both lower to the same runtime backend.

The runtime has the same three consumers as the J-lens, but is single-layer:

- **readout**: `SAE.encode(h_L)` after the decode forward, followed by top-k;
  the WebSocket `sae_readout` channel carries feature id, raw post-nonlinearity
  activation, and an optional cached label. It reuses the existing capture tap
  and adds no hook. `sae_token_readout` is the loom-prefix replay variant;
- **steering atoms**: `sae/<id>` lazily registers decoder row `W_dec[id]` as a
  one-layer ordinary profile. It lowers through the same affine synthesis and
  `subspace_inject` path as every other vector. `!sae/<id>` is therefore the
  existing directional mean-ablation, not an encode-clamp-decode operator;
- **probes + gates**: `add_probe("sae/<id>")` lands in the session SAE-probe
  registry, not the `Monitor`. Its single coordinate is the feature activation,
  so `@when:sae/<id>>N` compares the SAE's own units. A gate forces one encode
  per forward even with live discovery off and does not force monitor scoring.
  The finalize aggregate pools the last content token from the capture tail.

One encode is shared by live top-k, pinned probes, and SAE gate scalars on a
step. Activation values are comparable over tokens for one feature, not across
features; the dashboard scales each card against its own recent maximum.

---

## 7. Grammar (`core/steering_expr.py`)

`parse_expr(text) → Steering`; `format_expr` round-trips. Every live steering
surface (Python, YAML, HTTP, TUI) speaks it. `manifold bake` parses that grammar's
namespace-qualified additive scalar subset; it rejects dynamic terms and
Mahalanobis `~`/`|` projections because no identity-matched whitener is loaded.

```
expr     := term (("+" | "-") term)*
term     := [coeff "*"?] ["!"] selector ["@" trigger]
selector := atom (("~" | "|") atom | "%" position)?
position := signed_num ("," signed_num)* | label
atom     := [ns "/"] NAME ["." NAME] [":" variant] | "sae" "/" INT
trigger  := preset | "when" ":" probe op NUM
probe    := [ns "/"] NAME ["." NAME] ["[" INT "]"]   # vector probe (jlens/fake = readout strength), optional coord axis
          | [ns "/"] NAME ":" "fraction" | [ns "/"] NAME "@" NAME   # manifold channels
          | "sae" "/" INT                                 # feature activation
```

`+`/`−` add terms, `*` attaches a coefficient, `~` projects onto a direction
(keep the shared component), `|` projects orthogonal (remove it), `!`
mean-ablates (`h' = h − α(h·d̂ − μ·d̂)d̂`). `%` places a generation at a manifold
position — `<coord_list>` or `<label>` (sugar for that node's coords). The
manifold coefficient slot is `along[,onto]` (1- or 2-tuple). Variants are
`raw`/`sae[-<release>]`/`role[-<slug>]`/`from[-<src>]` (no `pca`). A bare slug
resolves in `core/steering_expr`: first as a bipolar pole/name of a 2-node `pca`
manifold (`resolve_pole`/`resolve_manifold_name`), then via
`io.selectors.resolve_bare_name` (the manifold-label tier) as a multi-node manifold
node label (synthesizing a label-form `ManifoldTerm`); cross-tier ambiguity raises.
The `jlens` and `sae` namespaces are reserved. `jlens/<word>` resolves through
the fitted Jacobian lens (§6.4); `sae/<id>` admits an integer RHS and resolves
through the resident SAE (§6.5). No manifold may be authored under either.
Term types (`ProjectedTerm`/`AblationTerm`/
`ManifoldTerm` + plain tuples) survive as parse-time markers the dispatch
synthesizer consumes.

**Role substitution.** A node's `role` (or a `:role-<slug>` vector variant) pools
that node's centroid under a chat-template assistant-label substitution
(`core/role_templates.py`), producing role-baselined activation space (a persona
manifold). At steer time `nearest_node_role` picks the role and pipes it through
`session._active_role` so the generation prefill applies the same substitution —
steer baseline equals extract baseline. Multiple terms implying distinct roles
compose under soft-warn + highest-`|coeff|`-wins (`RoleBaselineMismatchWarning`).
Mistral-3 lacks a substitutable label and raises
`RoleSubstitutionUnsupportedError` at fit time.

---

## 8. Cross-model transfer

`io/alignment.py` — per-layer compact affine alignment (`fit_alignment`,
row-space orthogonal Procrustes for matched dim, rectangular minimum-norm
least-squares otherwise; both low-rank factorized) maps points as
`A_L(x)=M_L x+b_L`. `transfer_profile` applies only `M_L @ v_src` per layer and
re-bakes each magnitude to its *target* Mahalanobis norm so the share is in the
target metric. The target whitener is **required** and must cover the transferred
layers (Mahalanobis-only — a missing/partial whitener raises `WhitenerError`;
generate neutral activations for the target model first, there is no Euclidean
rebake). `transfer_manifold` (`io/manifold_lifecycle.py`, with the pure-tensor core lifted to
`core/manifold.py::transfer_manifold_subspaces`) maps a fitted manifold's points
and mean through `A_L` while basis directions use `M_L`, then QR-orthonormalizes
the mapped basis and transforms every affine/RBF reduced coefficient by the exact
companion map. A collapsed span is rejected; curved transfer is rejected when
the companion map would make its scalar tube thickness anisotropic. It re-bakes the
Mahalanobis **share** in target space (same whitener
requirement; no lever — it's gone), transforms each curved neutral-foot
`origin` through the same companion map, and writes the
`_from-<safe_src>` variant. Since a vector is a 2-node `pca` manifold, `manifold
transfer` routes to this one transfer path. Alignment cache v5 stores each
layer's linear map as two rank-sized factors plus translation in an immutable
shard under the *target* model dir. The atomic pointer's identity and every
declared header are checked before selective payload materialization; selected
shards are read once for digest + decode. Stable
per-model neutral-capture locks and directional alignment-fit
locks span cache recheck through publication (including both serial model
loads), so two cold transfer commands do not repeat the same capture/fit work.
The materializing neutral loader returns the sidecar validated in that same
transaction, and cold alignment prep builds the target whitener directly from
the already-resident target rows. A cold fill publishes a complete neutral cache,
then narrows both in-memory seed rosters as soon as requested shared coverage is
known and releases unrequested tensor owners before Procrustes. The model-free preflight materializes the target
neutral rows exactly once: an alignment hit builds the whitener from them, while
an absent/corrupt alignment materializes the proven source rows too and fits,
scores, and publishes Procrustes entirely offline. Neither outcome loads model
weights or reopens/re-digests the target neutral artifact.

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
- **`base_gain` magnitude.** Steering now uses **three** constants:
  `_SUBSPACE_GAIN = 16.0` (along on the affine path), `_MANIFOLD_ALONG_GAIN = 4.0`
  (along on the curved path — a fraction-to-node scale), and
  `_MANIFOLD_ONTO_GAIN = 0.5` (onto / off-surface collapse only). Both are committed but
  tagged prototypes. `_SUBSPACE_GAIN` was bumped ~130×
  from the prior `0.125` when the affine target became whitened-unit (§5.3): the
  avg per-layer whitened push is now `GAIN·α` for *every* target, where the old
  raw-Euclidean target scaled it by each node's distance from neutral (a ~100×
  spread that left tight targets like `formal%formal` dead while `personas%caveman`
  slammed). `16.0` is live-calibrated on a gemma-4-12b α-sweep so α≈0.5 (effective
  `GAIN·α ≈ 8`) clearly steers concepts and personas alike while staying coherent
  for fragile personas; the per-target *coherence* ceiling (~2× variance, §10), not
  the geometric scale, now caps it — α≈1.0 over-steers hard personas. `_MANIFOLD_ONTO_GAIN`
  was likewise calibrated on the gemma-4-12b `emotions%dominant`
  onto sweep: at `1.0` (combined with a directional push) `onto=0.5` already
  fragmented and `onto=1.0` collapsed into looping — collapsing the off-surface
  residual erases the per-token spread, the same failure translate-not-collapse
  avoids — so `0.5` puts `onto∈[0,1]` at a usable range with `1.0` a coherent
  ceiling. Still one global per-op constant, no per-fit knob; per-persona strength
  variance is handled by α, not a per-target gain.
- **Discover-coord transfer.** Cross-model Procrustes for discover *coordinates*
  is deferred (a Gemma layout and a Qwen layout aren't comparable without it);
  `transfer_manifold` maps the per-layer subspaces but leaves the per-model
  `node_coords` as-is. Authored manifolds (shared coords) transfer cleanly.
- **MPS non-determinism.** Metal kernels are not bitwise deterministic even at
  temperature 0, so run-to-run wording jitters; compare qualitatively.
- **J-lens single-token vocabulary.** The lens (§6.4) can only name concepts
  that are single tokens: `jlens/<word>` raises on multi-token words, and a
  concept the model represents diffusely across pieces won't surface cleanly in
  the readout (the paper's own stated limitation). Multi-token direction
  synthesis is an open extension. The probe/gate channel is paper-native
  (per-layer softmax of the readout — one strength/probability axis);
  one wrinkle inherited from the softmax: a token nowhere near any layer's
  top has vanishing readout mass, so its depth CoM is numerically
  meaningless (the same degeneracy `aggregate_readout` has below top-k —
  read CoM only when strength is clearly nonzero).

---

*Pure manifold/subspace math lives in `core/manifold.py` (fp32, dependency-free,
no session/IO coupling). The whitener is `core/mahalanobis.py`; capture + fold +
projection `core/vectors.py`; the fit pipeline `core/extraction.py`; dispatch +
injection `core/session.py` + `core/hooks.py`; reads `core/monitor.py`; grammar
`core/steering_expr.py`.*
