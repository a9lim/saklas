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
2.12 kernel-arg-mismatch bug). StaticCache is early-initialized before a
compiled prefill so Transformers marks its K/V addresses outside Dynamo;
leaving its lazy init inside the graph disables CUDA graphs and recompiles once
per fresh cache. Ordinary generations reuse one reset-in-place StaticCache per
session so its K/V identities stay graph-stable. Persistent offset and capture
hooks are shared by CUDA and MPS compiled sessions; dynamic/curved/gated hook
topologies route to the eager original. CUDA graph trees are thread-local, so a
stream worker routes to eager while same-thread generation may replay the graph.
`_load_text_from_multimodal` extracts text-only
sub-models (Ministral-as-Mistral3), strips `language_model.` prefixes,
dequantizes FP8. `patch_torch_for_mps()` installs two lazy MPS-only workarounds
(`torch.histc` integer→float for MoE routing; `torch.ldexp` MXFP4 round-trip
through CPU honoring `out=`). `get_unembedding(model)` returns `W_U`
(`get_output_embeddings().weight`, `[vocab, d]`) and `get_final_norm(model)` the
pre-unembedding norm module (found as a sibling of the `get_layers` ModuleList —
`norm`/`final_layernorm`/`ln_f`/`final_norm` — rather than a second per-arch
table); both exist for the Jacobian lens readout, nothing else in saklas touches
the unembedding outside the model's own forward.

## jlens.py

The Jacobian lens (Gurnee et al. 2026): `J_l = E[∂h_final/∂h_l]` per source
layer, averaged over positions ≥ `SKIP_FIRST_POSITIONS` (16, attention sinks)
and a text corpus. `fit_jacobian_lens(model, tokenizer, prompts, layer_modules,
…)` is the **only backward-pass code in saklas**: everything else runs under
`inference_mode`, and with frozen params + integer inputs no autograd graph
exists at all, so the fit runs under `torch.enable_grad()` with returning
forward hooks that attach transparent zero-valued mean-position probe leaves;
the first fitted block also detaches its output (reuses `get_layers`, zero
per-arch wiring).
Consecutive ragged prompts share one right-padded graph (`prompt_batch`,
CPU/CUDA default 4, MPS 2) and `ceil(d_model/dim_batch)` backwards. Each backward selects one block of output
dims, sums those dims over the valid target positions, and calls
`torch.autograd.grad(..., is_grads_batched=True)` so `dim_batch` VJPs come from a
single unreplicated graph while preserving equal-prompt weighting. Autograd
returns each source as `[dim_batch,B,D]`, not `[dim_batch,B,T,D]`: the probe
identity collapses the valid source-position mean inside backward while passing
the full residual gradient upstream unchanged.
`SAKLAS_JLENS_VJP=replicated` restores the reference replicated-prompt estimator,
and `auto` falls back to exact replicated row-batch VJPs when a backend lacks vmap
coverage. Grads come from `torch.autograd.grad(final,
sources)` — NOT `backward()` + `retain_grad()`, whose `.grad` accumulation across
the multi-backward loop would corrupt the rows — which also stops the graph walk
at the shallowest requested source, so a `--layers`-restricted fit never
backprops below its lowest layer. A terminal hook stops the forward after the
final transformer residual, before final norm / LM head. Every backend uses
byte-budgeted device + host row stripes (up to `_ROW_STRIPE`, never smaller than
one active VJP block) that validate and commit directly into the persistent CPU
accumulator. Allocation failure halves the stripe independently of prompt width;
a later dim-batch OOM releases an oversized persistent slot before retry. CUDA
uses two ordered event-tracked slots
to overlap D2H with the next backward stripe, discards an entire uncommitted
suffix on transfer failure, and drops to one slot before narrowing estimator
batches when the overlap buffer itself causes OOM. If a later VJP OOMs the graph
is rebuilt at the first uncommitted row. A fully unsynced
MPS loop can still let the CPU enqueue too far
ahead, so the pass loop drains the queue every `_MPS_SYNC_EVERY_PASSES` (4), and
zero rows raise before a stripe is committed. Cooperative cancellation is also
checked before and after every output-dimension VJP block (not merely after a
complete prompt microbatch); accelerator queues and pending CUDA transfers are
drained before the active group's partial sums are abandoned, so the web cancel
control remains responsive even when `prompt_batch > 1`. Prompt/dimension widths halve
independently on OOM and stay below the proven failure ceiling. The fit is
compute-bound; explicit `source_layers` restriction is the one real wall-time
lever (a previously benchmarked 40–90% subset measured 1.73× faster).
Estimator teardown removes the fit hooks, clears their retained autograd graph
and accelerator stripe buffers, then flushes the MPS/CUDA allocator cache. A
long-lived server must not carry the backward-pass working set into the next
decode.
`checkpoint_accumulator_cb` fires every `DEFAULT_CHECKPOINT_EVERY` (25) prompts
for allocation-light resumable fits: IO normalizes live sums and merges any
prefix one layer at a time while streaming fp32, so checkpointing does not
materialize another complete fp32 lens. Checkpoints are self-contained
(`base_n_prompts=0`) and survive repeated interruptions even beside an older
full artifact. Resume transfers ownership of the loaded prefix matrices into
the estimator, scales them to raw sums in place, and accumulates the tail into
that same storage; it therefore holds one full fp32 matrix set, not a base plus
an equally large tail. Durable/checkpoint sidecars are compared before either
payload is paged; a farther self-contained checkpoint wins without first
materializing the older durable lens (and any superseded resident/device stack
is evicted before the checkpoint load). A checkpoint that passes header
preflight but fails full digest/finite validation falls back to the durable
prefix after the failed payload is released. Cadence never fractures a healthy
prompt microbatch. Checkpoint payload files and their directory entries are
durable before the checkpoint pointer is published. When the terminal checkpoint
is already the complete lens, finalization defensively re-fsyncs its immutable
per-layer tensor generations and atomically repoints the durable sidecar instead
of converting/writing the same artifact again. Each checkpoint stamps the
token-id hash of the prefix actually consumed, so a later corpus extension can
resume honestly from that prefix. Successful loads/writes also yield an
in-transaction pointer proof: terminal promotion and missing-layer shard reuse
skip a redundant full rehash only while that exact pointer remains under the fit
lock; unverified public callers retain defensive digest validation. The checkpoint remains the
recovery point until that pointer commit succeeds.
`JacobianLens.merge` is the non-mutating n_prompts-weighted combiner;
`merge_into` recycles a caller-owned
tail; `union_layers` combines same-corpus layer shards. Persisting a
missing-layer union reuses the existing v6 shard pointers and serializes only
the newly fitted layers. A same-session superset remains the full resident lens
when `fit_jlens(..., source_layers=...)` returns a narrower no-op view; a fresh
subset no-op materializes only the requested shards and leaves the full durable
union untouched. Because v4 has one corpus-progress identity for every layer, a
prefix extension of a strict durable subset is rejected before payload load:
the caller must request the full durable layer set or use `force=True` to
explicitly replace it. Missing-layer top-ups load/preserve the full same-corpus
union, while resume drops superseded containers before backward so only the
selected accumulator owns its matrices. Exact no-op paths also reap a crash-left checkpoint only when the validated
final pointer matches its semantics/layers and has reached at least its effective
`base_n_prompts + n_prompts` progress.

`JacobianLens` holds the fp32 matrices: `transport(h, layer)` maps a residual
into the final basis; `token_direction(v, unembed)` is `W_U[v] @ J_l` per layer
— the profile-shaped direction behind `jlens/<word>` atoms; `lens_logits`
(free function) is the full readout `W_U · norm(J_l h)` (matvec in the
unembed's own dtype — a fp32 W_U copy would be GBs). `aggregate_readout(logits,
depths, top_k)` is the **layer-aggregation** of a stacked `[L, vocab]` readout:
per-layer softmax (calibrates away the cross-layer logit scale), then per token
`strength = mean_l p_l(v)` (mean fitted-layer probability) and a depth center of mass +
std weighted by the same per-layer probability `p_l(v)` — the selected-layer readout is
sharp, and what changes over depth is *which* token leads, so a token's
probability profile over depth is its depth signal (a diffuse noise layer's
vote is discounted by its own lack of mass; the former within-layer salience
gave it a full vote). Top-k by aggregated full-vocab strength (a per-layer top-k union would
miss a mid-pack-everywhere token); CoM/spread are evaluated only for those
selected columns, then returned as `[(vocab_id, strength, com, spread)]` in one
batched host transfer. `readout_probabilities` is the shared calibration
primitive; `_live_lens_readout_step` computes it once per logits matrix and
passes the result to pinned probes, per-layer cards, and the aggregate through
the `*_from_probabilities` helpers. `token_readout_stats(logits, depths,
token_ids)` is the **single-token restriction** of the same calibration — read
at pinned vocabulary ids instead of top-k selection, returning per id
`(strength, com, spread, per_layer[p_l])` where `strength = mean_l p_l(v)`
(∈ [0,1], the ONE probe/gate/display channel — apples-to-apples across
tokens and layers; the depth-CoM mass is the same `p_l`, matching
`aggregate_readout`) — the math behind
`jlens/<word>` probe readings and gate scalars. `resolve_word_token` maps
a word to its single vocab id (leading-space piece first, decode-and-compare
sanity check, `MultiTokenWordError` with the pieces otherwise).
`sparse_nonneg_decompose` is the J-space split: greedy pursuit against the
dictionary `W_U J_l` — never materialized (scores are the composed matvec,
normalized by chunk-computed atom norms; only selected rows form), coefficient
re-solve as a tiny projected-gradient NNLS per step — returning
`JSpaceDecomposition(layer, share, tokens)`. Errors: `JacobianLensError` (422),
`LensNotFittedError` (404), `MultiTokenWordError` (400), all `SaklasError`s.

## capture.py

Low-level capture, pooling, DLS, the vector⇄subspace fold, and profile I/O. **No
`extract_difference_of_means`** — extraction authors a 2-node `pca` manifold now
(`extraction.py`); what survives here is the capture + fold + projection
machinery that pipeline and dispatch both consume.

Capture runs in **right-padded batches** — `_encode_and_capture_all_batch(model,
tokenizer, prompts, responses, layers, device, *, role=, model_type=,
system_msg=)` renders + tokenizes the chunk, right-pads to a common length
(attention-masked; pool indices unchanged since real tokens stay left-aligned),
then transfers the finished ids/mask once and runs one
`_capture_all_hidden_states` forward that pools each row at its
last-content index *inside the hook* (per-row gather → `(B, D)` per layer, never
`(B, T, D)`). Rendering, special-token walkback, and padding stay on CPU—there
is no per-row H2D followed by a blocking `.tolist()` D2H. `role=` is uniform;
`roles=` carries mixed per-row substitutions for fit-wide batches.
`compute_node_centroid` / `compute_neutral_activations` chunk by
`_CAPTURE_BATCH` and amortize the MPS `empty_cache` per chunk.
Fit-wide manifold capture tokenizes once, sorts rows by token length, grows clean
batches up to `_CAPTURE_BATCH_MAX`, and stops the model after the last selected
layer so unused upper blocks, final norm, and LM head never run. Its layer-major
centroid accumulators are normalized in place and transferred directly to fit
ownership, avoiding a second full `layers × nodes × hidden_dim` fp32 roster.
`_encode_and_capture_all` is the single-pair sibling. `_capture_all_hidden_states`
hooks every layer at once and accepts an `int` (single) or `(B,)` tensor (per-row)
`pool_index`. Capture is **conversational** (4.0 / A2): a corpus item is an
assistant *response* to a fixed baseline *prompt*, so `_encode_and_capture_all(model,
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
prompt, response, *, role=, model_type=)` is the shared
render+tokenize+walkback front half; `role` substitutes the assistant label only
when an explicit per-node role override is set (the persona-baselined fit), the
swap-back default being the standard assistant. `_load_baseline_prompts` loads the
shared A2 user prompts (user override → bundled `saklas/data/baseline_prompts.json`,
48 prompts); `_neutral_pairs` pairs the neutral corpus to those prompts
(`response[i] ↔ prompt[i % k]`). `compute_neutral_activations` builds the whitener's
neutral cache (one per-model artifact), pooling conversational `(prompt, response)`
pairs via `_neutral_pairs`; the probe-centering baseline is its per-layer `X.mean(0)`
(`bootstrap_layer_means`), so there is no separate layer-mean forward pass or cache.

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
Mahalanobis-only collapse). `folded_directions(manifold)` is the **reverse view**:
`{L: δ̂_L · share_L}`, the baked-direction view of a 2-node affine manifold, used
to back the `Profile`-returning surface (`extract()`, `manifold compare`/`why`,
GGUF export) without a second stored representation. It raises on a curved or
multi-dim manifold. Dispatch lowers every plain vector term through
`fold_directions_to_subspace` → `_affine_manifold_push` (`session.py`) onto the
merged affine subspace.

`profile.save_profile`/`profile.load_profile` own the baked-profile wire format:
a `dict[int, Tensor]` to `.safetensors` + a slim `.json` sidecar (stamped with
the exact `PROFILE_FORMAT_VERSION` from `io/packs.py`) for the per-model
`layer_means`/`neutral_activations`/alignment caches and folded-profile
interchange. `project_profile(base, onto, operator, *, whitener=...)` is the
per-layer `~`/`|` projection — closed-form LEACE, Mahalanobis-only: the whitener
is **required** and must cover every projected layer (`covers_all`), else
`WhitenerError` (no Euclidean path).

## extraction.py

`ManifoldExtractionPipeline.fit(folder, *, sae=...)` is **the** extraction
pipeline (concept extraction and manifold fitting are the same thing — a vector is
a 2-node `pca` manifold). Dependencies via the `ModelHandle` protocol (`model` /
`tokenizer` / `layers` / `device` / `model_id` / `_run_generator` /
`generate_responses`) + an `EventBus` for `ManifoldExtracted`; `SaklasSession`
satisfies it implicitly. Cache-hit on the sidecar `nodes_sha256` (folds in labels, corpus
+ `{domain, node_coords}` authored / `{fit_mode, hyperparams}` discover),
`sae_revision`, token-exact capture identity (baseline prompts + tokenizer
render included), loaded-model fingerprint, and fitted-layer set. A mandatory
Mahalanobis whitener is resolved after that final fitted-tensor fast path but
before any activation capture, so a missing or partial neutral cache cannot
waste a full model pass. Else pools the complete
roster through `compute_manifold_node_stats`: one row stream crosses node
boundaries so short template nodes fill shared forward batches; OOM halves the
active batch and a clean run grows it back. Centroid-only fits reduce by node on
device before transfer; raw curved fits retain source-dtype rows in a layer-major
mmap spool for later covariance instead of capturing the corpus twice or holding
the whole roster in fp32 RAM. A token-exact per-model capture cache lets domain,
topology, and smoothing refits skip model forwards; its identity includes node
boundaries, and its digest metadata validates centroid payloads plus exact
per-layer row tensors; subset fits map/hash only requested row layers and validate
only requested centroid/row layers. Capture-cache format v4 persists immutable
generation-named safetensors shards, one per layer, so a disjoint top-up writes
only the new shards instead of copying/replacing a multi-GiB container. Publication fsyncs
shard payloads and their directory entries, writes a recovery journal, atomically
replaces and directory-fsyncs the authoritative JSON pointer, and only then
garbage-collects superseded generations. A failed publication therefore
preserves the prior good pointer, while the next fit adopts a complete crash-left
journal under the capture-stem lock without rerunning model capture. Cached and
freshly captured disjoint
layer stores combine by view (ownership is transferred to the composite), so
the sigma covariance pass does not copy the full row roster into a third mmap.
Each row shard validates independently, so one damaged layer recaptures only
itself; publication streams each row payload once while computing its exact
tensor-domain digest, and speculative auto-fit rows do not enter durable coverage
until publish.
The shared per-model capture stem is protected by `artifact_lock`
only across read/top-up/publish; the downstream topology/covariance fit retains a
PID-backed process lease that prune/GC honors without holding the exclusive stem
lock. Safetensors stage through unique same-directory tempfiles. `force` drops
only the requested layers from the loaded v4 pointer, so a forced subset keeps
all unselected shard identities. Deferred auto-curved row top-ups reload and
merge the latest pointer under the reacquired stem lock. Layer coverage is
unioned/topped up, so full→subset needs no forward
and overlapping subsets capture only missing layers. Cache groups prune oldest-first past 8 GiB
(`SAKLAS_MANIFOLD_CAPTURE_CACHE_GB`) and `pack clear` / `pack rm` remove referenced
group. Pruning runs only after the current stem transaction releases, under a
directory prune lock plus one victim lock at a time, so eviction cannot remove
another fit's active cache or introduce cross-stem lock inversion. `layer_indices` accepts an explicit set or the canonical 40–90% workspace
band. It threads the folder's `node_kinds` / `node_roles` into the `Manifold` +
sidecar metadata, then dispatches on `fit_mode`.
The long fit lock is target-scoped and stored outside the removable folder.
Folder-manifest locks cover only override/snapshot and final publication: the
snapshot hashes exact nodes plus the resolved template/baseline and target clear
epochs; publication compare-and-swaps that revision. Readers and unrelated model
targets therefore proceed during compute, while authoring edits, scoped clears,
and rm/recreate prevent stale publication. Tensor and sidecar payloads are both
staged+fsynced before a sidecar-first pair commit; any interrupted mixed/orphan
pair is a cache miss or is repaired at the next target fit.
Every fitted tensor/sidecar read, replacement, clear, transfer, and folder
removal uses a digest-named pair lock under the namespace parent (not inside the
removable manifold folder). Folder-wide lifecycle operations take the manifest
lock then all affected pair locks in sorted order, so `rm`/HF stage-swap cannot
delete a lock inode or tear a cache-hit proof/read transaction.
If an override changes the manifest and the fit then fails, the session evicts
only that namespace/variant's resident manifold, folded profile, attached probe,
and prefix state; validation failures that did not write preserve consumers.

Fit uses `ManifoldFolder.load(..., verify_manifest=False)`: it hashes the live
corpus into the capture/final identities and validates the requested tensor, but
does not reread every historical model/SAE payload. Metadata-only authoring and
lifecycle routing (source-tier reads for refresh/removal) also skip the payload
walk. Runtime fitted-tensor use, install, and push retain strict integrity
verification; publication hashes each newly committed pair exactly once.

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
  projection (neutral-anchored frame — no `anchor_origin` knob). The shared
  display layout (`Manifold.node_coords`) comes out PCA-mean-centered (origin =
  node centroid), so step 4a re-anchors it on neutral via `neutral_layout_coord`
  — the landmark-MDS projection of neutral into the consensus-PCA layout
  (`cₙ = node_coords⁺ · g_ν`, `g_ν` neutral's whitened node-mean-centered
  cross-Gram column). Subtracting it is a **pure translation** (cardinal weights
  are translation-invariant, so `_affine_manifold_push` steering is unchanged),
  putting the layout origin at neutral's projection so `% 0,…,0` reads as neutral
  and the rack sliders share the geometry plot's whitened origin. Flat only — a
  curved layout's neutral is the per-layer `origin` foot, not a layout coord.
- **monopolar** (a `pca` folder with `K == 1`): a structural early branch — a
  1-node flat fit is meaningless (needs `k+1 ≥ 2` poised nodes), so the engine
  reads it as concept-vs-neutral. Folds `concept − ν` (ν = `handle.layer_means`)
  into a 1-node neutral-anchored ray via `fold_directions_to_subspace` (raw δ̂
  basis, whitened share when `covers_all`), bypassing discover-coords / per-layer
  pca / DLS entirely; `method = "manifold_monopolar"`. Raises if ν is unavailable.
- **`authored`/`spectral`** (curved): per layer `fit_layer_subspace` (PCA frame +
  RBF surface), μ-centered share bake, and a per-layer `origin`
  (`invert_parameterization` of the neutral mean — curved only; flat's foot is
  coord 0). The `spectral` path passes `smoothing` (default `"auto"` → GCV) into
  the penalized `fit_rbf_smoothed`; authored stays exact (node = exact target).
  Per-layer λ/edf ride the sidecar as `rbf_smoothing_per_layer`.
- **`auto`** (discover): `select_topology` (`core/manifold.py`) picks the
  geometry per-model — flat `pca` vs curved `spectral` by GCV, plus periodic
  `BoxDomain` axes via persistent homology. The chosen `effective_fit_mode`
  drives the same flat/curved per-layer fit below; the resolved mode + the ranked
  topology candidates land in the sidecar (`resolved_fit_mode`,
  `topology_winner`, `topology_candidates`). A curved winner carries the
  already-factorized `RbfFitPlan` from scoring into the final layer fit instead
  of repeating layout QR/eigh/LU. Sphere is authored-only.

`--sae` reconstructs each centroid through the SAE before the fit (fail-fast
`SaeCoverageError`); the fitted subspace is model-space regardless. Bakes
`mahalanobis_share` (+ `share_metric`/`subspace_metric`), `origin`. **No lever.**
For multi-node SAE fits, reconstruction pops raw centroid layers as their decoded
replacements are produced, so raw and reconstructed full rosters do not coexist;
the `per_node` view is built only by the K=1 monopolar branch.
Discover coords come from the **consensus Gram** —
`mean_L` of each layer's whitened, node-mean-centered `(K,K)` Gram
`X̃_L Σ_L⁻¹ X̃_Lᵀ` (signal-weighted, layer-agnostic; no reference layer) — and the
same per-layer Grams' traces are stamped into the sidecar as
`node_spread_per_layer` (`{str(L): tr(G_L)}`), the concept's pre-DLS whitened
signal-by-layer profile. Diagnostic only (nothing runtime branches on it; absent →
empty dict), surfaced by `manifold show`; computed for every K≥2 fit (authored too),
distinct from `mahalanobis_share` (the same whitened spread restricted to the
steerable subspace). Whitener resolution is deferred past the cache-hit return
and gated all-or-nothing on `covers_all`. Each layer's centered `Σ⁻¹X` and Gram
are computed once and reused by discovery, topology scoring, the Fisher basis,
and neutral-layout anchoring. Curved fits share one layout-only `RbfFitPlan`
(kernel/polynomial blocks, QR/eigensystem, λ grid, fixed-λ LU) across every layer
and the sigma field. `min_nodes(k) = 2k+1` for curved; a flat
`pca` fit needs only `k+1` (so K=2/k=1 is a steering vector). Emits
`ManifoldExtracted`. (Note: `extract.py`'s old concept `ExtractionPipeline` and
the `_capture_diffs_for_pairs` DiM apparatus are gone; the `discover-pca` 2-node
fit replaces them.)

## sae.py

SAE backend. `SaeBackend` (runtime-checkable Protocol): `encode_layer`/
`decode_layer`/`feature_count`/`feature_direction`, `release`, `revision`,
`layers`. `MockSaeBackend` for CPU tests; `SaeLensBackend` adapts provider
releases and `LocalSaeBackend` adapts Saklas-trained artifacts.
`load_sae_backend(release, *, revision, model_id, device, dtype)` resolves
`local:<name>` without importing SAELens; otherwise it queries SAELens, validates base-model compatibility,
resolves per-layer sae_ids (`_canonical_layer_map` — Neuronpedia-hosted first,
then narrowest width, then smallest L0: hosting supplies the label +
`maxActApprox` metadata channel, so an unhosted pick would silently lose
normalization), gates
`sae_lens` imports so local-only flows avoid initializing the provider package.
Registry resolution is eager but weights are lazy with a one-layer resident cache; a valid fitted-tensor
hit does not import SAELens at all. An explicit `revision` is passed only when
the installed loader exposes `revision=`; otherwise loading raises instead of
stamping a pin it did not honor. Errors:
`SaeBackendImportError`, `SaeReleaseNotFoundError` (difflib suggestions).
The live runtime keeps one selected encoder/decoder layer resident;
`select_runtime_layer` chooses nearest 65% model depth (workspace preferred),
and `list_sae_releases` discovers both local artifacts and compatible registry
rows without weights. `sae_training.py::train_residual_sae` trains a native
one-layer ReLU SAE from block-output token activations under model inference
mode; decoder rows are unit-normalized so L1 cannot be evaded by rescaling.
`sae/<id>` steering reads `W_dec[id]`; feature probes read the encoder channel
outside the `Monitor`. The probe/gate channel is **normalized strength** —
`activation / maxActApprox` ∈ ~[0,1], the Neuronpedia corpus-max unit, so
features read apples-to-apples like the lens probes' mean fitted-layer probability;
raw activation only when no metadata is cached (offline / unlisted feature).
Metadata (`{label, max_act}` per feature, `session._sae_feature_meta`) is
fetched lazily from Neuronpedia — at validate/pin time
(`_fetch_sae_feature_meta`) and via the batch backfill
`fetch_sae_feature_meta(ids)` (the dashboard calls it between generations
with the ids the live top-k surfaced; a fetch never runs in the decode
loop) — and persisted through `io/sae.py::save_sae_feature_meta`. Live
top-k rows and `sae_token_readout` features carry `max_act` beside the raw
activation so clients render both units. The capture surface is the transformer-block output, so
only residual-post/block-output SAEs are valid: discovery omits explicitly named
attention/MLP/transcoder families, and loaded hook metadata plus decoder width are
validated before a release becomes resident.

## mahalanobis.py

`LayerWhitener` holds per-layer centered neutrals `X_L ∈ ℝ^(N,D)` + the Woodbury
inverse `K_L = (NλI + XXᵀ)⁻¹`; `apply_inv(layer, v) = (1/λ)(v − Xᵀ K X v)` in
O(ND), no D×D. Ridge `λ_L = (‖X_L‖²_F / (N·D)) · ridge_scale`. Built lazily via
`from_neutral_activations` (in-memory) or `from_cache(model_id)` (the single
offline loader — resolves the atomic `neutral_activations.json` pointer to its
immutable per-layer fp32 shards, no model load, and
derives the per-layer centering mean from the cached neutrals as `X.mean(0)`;
backs `manifold compare` + the cross-model transfer rebake); neutrals cached
**fp32** (the project-wide
invariant — fp16's 65504 ceiling overflows gemma-3's late layers to
±inf, poisoning Σ). Any layer whose centered acts or `K` come back non-finite is
*excluded*, so `covers_all` is trustworthy as "finite factors everywhere" — the
all-or-nothing gate shared by extraction, manifold fit, projection, monitor, and
`manifold compare`. Primitives: `mahalanobis_cosine`, `mahalanobis_norm`,
`leace_project`, `apply_inv`, `subspace_gram(layer, B) = B Σ⁻¹ Bᵀ` (the (R,R)
reduced inverse covariance behind whitened share + whitened manifold reads),
`woodbury_factors` (device-resident factors for the monitor's inline per-token
apply). `SaklasSession.whitener` is a lazy property; session construction wires it
into the unified `Monitor` via `set_whitener`.

## manifold.py

Pure-tensor (fp32, no session/IO) subspace + manifold math. Goodfire "Manifold
Steering" (arXiv 2605.05115), generalized to arbitrary intrinsic dim/topology.
The on-disk **tensor codec** (`save_manifold`/`load_manifold`) and the disk-backed
`ActivationRowStore` row spool live in the io layer now
(`io/manifold_tensors.py`) — that persistence lived here only because the codec
needs these dataclasses, and io importing core's types is the correct layering
arrow. Fit-capture math (`compute_manifold_node_stats` etc.) still produces the
spool, so it lazy-imports `ActivationRowStore` from io (the established core→io
reference pattern).

Domains: `ManifoldDomain` ABC + `BoxDomain` (open/periodic axes), `SphereDomain`
(Sⁿ chordal), `CustomDomain` (explicit immersion; also the identity carrier for
discover coords and synthesized affine subspaces). `domain_from_spec`/`to_spec`
round-trip the tagged union.

`LayerSubspace` — `mean`, `basis` (+ the curved RBF triple + unit-box
normalization). `is_affine ⇔ node_params is None`. Flat layers carry `node_coords`
(K,R) — the real neutral-anchored per-layer node positions (steer-target source).
`affine(mean, basis, node_coords=)`, `select_axes(kept)` (per-axis DLS prune),
`eval_at`, `jacobian_at`, `rbf_params()` (raises on a flat subspace).
`manifold_is_affine(manifold)` — public module-level flat/curved predicate.

`_pca_basis(X, *, n_components, whitener, layer)` — μ-centered PCA: Euclidean SVD,
or the whitened/Fisher generalized eigenproblem `(S_b, Σ)` via the Woodbury Σ⁻¹
(`G = X Σ⁻¹ Xᵀ`, directions `Σ⁻¹ Xᵀ a`, re-orthonormalized via QR so the hot path
is untouched). Shared by `fit_affine_subspace` (flat) and `fit_layer_subspace`
(curved). The **basis caveat** (do not break PCA@2 ≡ DiM): always μ-center the
scatter, never anchor-center it. `fit_affine_subspace` returns `(subspace,
mu_coords, ev_ratio)` and neutral-anchors the frame (`mean = P_basis(neutral)`,
real `node_coords = (centroids − neutral)·basisᵀ`); `orient_to` fixes the sign.
`subspace_share(mu_coords, basis, whitener, layer)` is the μ-centered (anchor-
independent) per-layer budget weight (`DEFAULT_N_COMPONENTS = 64`). Fit callers
can pass a precomputed Gram and `Σ⁻¹X`; Fisher directions then use
`Aᵀ(Σ⁻¹X)` without another Woodbury application.

`Manifold` — domain + per-layer `LayerSubspace`s + `node_labels`/`node_coords`/
`node_roles`/`node_kinds` + the bakes `mahalanobis_share`/`origin` +
`feature_space`/`metadata`. The `Profile` analogue. `node_kinds`
(abstract/concrete/custom) is generation-only provenance — it selects the system template
+ elicitation role label at authoring time, but is NOT consumed at fit; it
round-trips through the save/load sidecar. `manifold_point`, `tangent` (analytic
RBF Jacobian), `resolve_position` (coord payload or label),
`nearest_node_{index,label,role}`.

`decompose(h, mean, basis) → (h_par_c, h_perp)` — a standalone centered-decomposition
helper (exported but off the hot path; `subspace_inject` and the monitor each
decompose inline). `synthesize_subspace(push,
ablate, neutral_means, *, whitener=None)` composes the active steering term set
into one `SynthesizedSubspace` per layer (orthonormal merged basis via
`_ortho_basis`, push-before-ablation ordering). When a covering `whitener` is
supplied (the session always passes `self.whitener`; gated all-or-nothing on
`covers_all`) the push is **whitened-normalized** so `along` is a scale-stable
strength knob: `share = ‖Δ‖_M` (whitened displacement → the mean-1 cross-layer
*profile*) and `target_coord = Σᵢ coeffᵢ·(B@dirᵢ)/‖dirᵢ‖_M` (each fragment a
**whitened-unit** direction, node-distance stripped, scaled by its user coeff).
This removes the ~100× push spread that came from steering by the raw-Euclidean
node distance (a tight bipolar pole sits ~0.3 from neutral, a far persona
centroid ~17, so one `along` gain couldn't calibrate both); every target now
lands on one uniform whitened budget (`Σ_L eff_along_L = gain·n_layers`),
linear in α, concentrated where the signal lives. Whitener-absent (CPU stub /
degenerate fit) falls back to the raw-Euclidean `target_coord = B @ Σ coeffᵢ·poleᵢ`,
`share = ‖Δ‖₂`. `subspace_inject(h, subspace, domain, target_coord, foot_seed, along,
onto)` is **the** injection: affine analytic shortcut (foot = q, translate by the
fixed `a·target` offset with per-axis κ collapsing ablation axes — `p_new = q +
a·(target − κ·q)` — `H_o` kept) vs curved per-token GN foot-follow (along
translates the foot and transports `H_n` to the new foot by the minimal
orthogonal principal-angle rotation between the old/new tangent frames
(`_frame_rotation_transport` — exact identity when the foot doesn't move, so the
curved path is identity at `along=0` regardless of foot accuracy; replaced the
project-onto-normal + renorm that corrupted off-neutral activations every fire),
onto shrinks `H_n` toward the zero-thickness wire on legacy/no-σ fits or toward
the local fuzzy σ-tube when present, `H_o` kept). MGS orthonormalization + a CPU-hopped
`n×n` SVD keep it MPS-safe (`linalg.qr`/`svd` are unimplemented/`fallback` on
Metal). `synthesize_subspace` emits per-axis κ coefficients: 0 on push axes and
the exact signed ablation-operator eigenvalues on the ablation complement
(partial/repeated/non-orthogonal terms preserved; the tiny eigensolve CPU-hops
for MPS). Hook-time gain compensation keeps `along·κ` equal to the user α.
`norm_cap = 3·‖h‖` is the only norm guard. `invert_parameterization` is the cold/eval-only damped-LM nearest-
point projection.

`save_manifold`/`load_manifold` (the codec — **moved to `io/manifold_tensors.py`**)
round-trip the per-model tensor (`layer_<L>.{mean,
basis[,affine_map][,node_params,rbf_weights,poly_coeffs,coord_offset,coord_scale]}` + shared
`node_coords` + optional `origin`) + sidecar (`mahalanobis_share_per_layer`,
`origin_per_layer`, `share_metric`, `subspace_metric` — no
`explained_variance_per_layer`, no `lever_per_layer`). `transfer_manifold_subspaces(src,
alignment, *, whitener, from_model, to_model)` is the pure-tensor core of the
cross-model affine transfer: maps points/means through `M_L x+b_L` and basis
directions through `M_L`, QR-orthonormalizes the mapped rows, and transforms the
affine/RBF reduced coefficients by the exact companion map. Rank collapse is
rejected; a curved transfer is rejected when the companion map is not an
isometry because the current scalar sigma field cannot encode an anisotropic
tube. It re-bakes the Mahalanobis
**share** in target space via `subspace_share` (target whitener **required** —
`WhitenerError` on a missing / partial one, no Euclidean rebake), transforms
`origin` through the same companion map, and returns the transferred `Manifold`.
The folder read/write orchestration around it (load the source tensor, write the
`_from-<safe_src>` variant, patch the sidecar) stays in
`io/manifold_lifecycle.py::transfer_manifold`. Discover: `derive_pca_coords`
(cumulative-variance prefix) / `derive_spectral_coords` (Laplacian eigenmaps,
eigenvalue-ratio cliff)
/ `discover_coords` dispatcher, with `PcaDiagnostics`/`SpectralDiagnostics`;
`_laplacian_eigen` is the shared graph→Laplacian→eigh core.

**Penalized smoothing.** `fit_rbf_smoothed(node_params, values, *, smoothing)`
is the thin-plate/Duchon generalization of `fit_rbf_interpolant`: the penalized
saddle `[E+λI Q; Qᵀ 0][w;c]=[y;0]` (penalty `λ·wᵀEw`). `smoothing="auto"`
GCV-selects λ (`_gcv_select_lambda` → `_rbf_smoother_matrix` hat `S_λ`, GCV
`K·RSS/(K−edf)²`, edf `tr S_λ`); `0`/`None` delegates to the exact interpolant
bit-for-bit; a float is a fixed λ. The fitted weight shapes are unchanged so
`eval_rbf` (hot path) is untouched — only the coefficients shrink. `fit_layer_subspace`
takes `smoothing=` (curved discover passes it; authored stays exact) and surfaces
the chosen λ/edf via the `rbf_info` out-dict. `prepare_rbf_fit_plan` factors the
node-layout-only work once per manifold; every layer and `fit_sigma_field` reuse
it while retaining non-aliased persistent tensors. CPU/fp32 (the saddle is
MPS-unsafe).

**Fuzzy-manifold σ-field (curved only).** Optional per-layer *tube thickness*:
the surface stops being a zero-thickness wire and carries a within-node off-surface
spread `σ(z)`. Raw curved fits retain activation rows from the shared centroid
pass; `compute_store_reduced_covariances` streams the layer-major spool through
bounded fp32 chunks after the basis exists, centering out-of-place before the
projection (stable at large common-mode activation offsets and never mutating an
fp32 mmap view), avoiding both a second model pass and `nodes × layers` small
hidden-dimension GEMMs while accumulating each
node's reduced `(R,R)` covariance. `compute_node_reduced_covariance_from_rows`
and `compute_node_reduced_covariance` remain for standalone callers
without retained rows; `fit_sigma_field`
reduces it to one off-surface scalar per node (`_off_surface_var` — the
normal-complement trace via the surface tangent `_reduced_tangent`, divided by
the normal degrees of freedom from the tangent's **actual local rank**, including
rank-deficient folds where `R <= n`; one batched economy SVD supplies both rank
and the tangent projector) and fits a
**separate** `log σ` RBF over the same normalized `node_params`, stored on
`LayerSubspace.{sigma_rbf_weights,sigma_poly_coeffs}` (None ⇒ `has_sigma` False ⇒
`sigma_at` returns 0 ⇒ exact legacy). `sigma_at(embedded)` is the one extra
`eval_rbf` on the curved path. Curved+raw fits only (SAE/flat skip; the
extraction pipeline gates on `effective_fit_mode != "pca" and sae_backend is
None`); summary rides the sidecar as `sigma_field_per_layer`. Consumed by soft
`onto` (`subspace_inject` shrinks `H_n` toward `σ(z)` — the tube — instead of to 0)
and the monitor's membership/assignment reads. v1 is **isotropic** (one scalar/node;
diagonal/full later).

**Topology selection (`fit_mode="auto"`).** `select_topology(stacks, layer_grams,
consensus_gram, *, whitener, …)` picks the discover geometry per-model in two
decoupled decisions (decoupling dodges the dimension bias that makes a single
reconstruction score always crown the highest-dim candidate): **(a) flat vs
curved** — GCV of the flat affine (`pca`) vs curved RBF (`spectral`) fit in a
shared whitened-reduced target metric (`_ols_gcv_score`/`_rbf_gcv_score`), the
curved candidate floored to the **flat candidate's dim** (`min_dim=k_flat`) so the
two compete at matched expressiveness — the spectral eigenvalue-ratio cliff
systematically undershoots (one dominant Fiedler mode picks k=1), and without the
floor a curved manifold linearly embedded in a `k_flat`-plane reads flat (the flat
affine fit reconstructs the in-plane curve, the under-dimensioned curved fit
can't, losing reconstruction it would win at matched dim — the flat-bias the
the topology regression suite surfaced);
**(b) periodic axes** — Vietoris–Rips H1
*persistent homology* (`_rips_h1_persistence` boundary-matrix reduction →
`_count_persistent_loops`, essential loops at `eps_max=2ε_c`) counts the loops
(ellipse/noise-robust — a circle and a 6:1 ellipse both read as one loop), the
spectral eigenpairs coordinate them (`_detect_periodic_axes`, `_is_angular_harmonic`
dedups a circle's `cos kθ` harmonics), routing to a periodic `BoxDomain`. Returns a
`TopologyChoice` (`fit_mode`/`coords`/`domain` + ranked `TopologyCandidate`s for
the sidecar). Sphere is **authored-only** — not an auto candidate. PH counts loops
by *hole size*, so two kinds of real ring slip under its threshold;
`_faint_cycle_coords` is the complementary single-cycle fallback
(`_detect_periodic_axes` runs it only when PH counts zero), recovering the cyclic
order in either of two sampling regimes off a greedy+2-opt tour (`_nn_tour`):
**uniform** — a faint ring (small cyclic modulation on a near-equidistant heap,
e.g. day-of-week centroids at ~16% modulation): too thin a hole for PH but
near-equidistant, so the classic guards fire — **1-D** (symmetric 2-NN max-degree
≤ 3), **closed** (tour edges near-uniform, `max/median < 2.0`), **local**
(tour-neighbours among the two nearest, recall ≥ 0.90); and **clustered** — tight
clumps spaced around the loop, the sampling real concept families have
(months→seasons, days→weekday/weekend): the tour edges go **bimodal** (tiny
intra-cluster, big inter-cluster) so closure/recall fail though the loop is real,
accepted instead when the inter-cluster gaps are **≥2** in number, **decisively
bimodal** (smallest gap ≥ 3.5× the small-edge scale — the guard that screens a
diffuse low-D random cloud whose many accidental long edges only marginally clear
the gap cutoff), **mutually regular** (`max/min ≤ 2.5`), and the loop has a **real
far antipode** (tour-antipode/tour-neighbour mean distance ≥ 2.5 — rejects a blob/
fan). Both regimes also require **graded** growth (`d(sep=2)/d(sep=1) ≥ 1.08`) and
1-D-ness at the looser clustered bound (degree ≤ 4 — a tight clump reaches 4),
returning a uniform `2π·rank/K` `S¹` coordinate in the recovered cyclic *order*
(exact spacing dropped — topology, not metric, is what the periodic domain needs).
Gated `7 ≤ K ≤ 128`. Validated by the periodic-topology regression tests for specificity
(~0% false-positive on random Gaussian heaps K≥9, and on grids/fans/arcs/blobs/
lines) and clustered-ring sensitivity (100% recall for tight-to-moderate clumps);
the bimodality guard trades two documented false-negatives — a very-loose cluster
heap approaching uniform, and an eccentric ellipse `> 6:1` — for that 0% FP rate.
A **gapped ring** (a uniform ring with one missing sector) is geometrically the
*same point cloud* as an open arc, so it correctly stays non-periodic — the
closure guard rejecting both is right, not a miss.
## naturalness.py

Behavior-space naturalness eval, extracted from `manifold.py` to restore its
pure-tensor contract (naturalness drives a live model forward pass, which has no
place in a geometry module): `to_hellinger`, `bhattacharyya_distance`,
`fit_behavior_manifold`, `trajectory_naturalness`,
`compute_node_behavior_centroid`, `compute_trajectory_distributions`,
`_next_token_distribution`. Import sites in `cli/runners.py` and
`tests/test_naturalness.py` point here. `compute_node_centroid` remains in
`manifold.py` (shared primitive — activation centroid, no model).

## hooks.py

`SteeringHook` carries per-layer groups `(trigger, subspace, domain, target_coord,
origin_coord, along, onto)` and runs each active one through `subspace_inject`
(`_apply_manifold_groups`). A cheap pre-check skips inactive steps. Per-token foot
state `_manifold_feet` (cold → seed at `origin`, `_MANIFOLD_COLD_GN_STEPS = 4` GN
steps; warm → one step). The dominant case — exactly one always-active
(`Trigger.BOTH`) affine group — is precomputed at `recompose` into
`_single_affine_fast`: pure push becomes one constant `add_`; mixed
push+ablation precomputes that constant and projects only the nonzero-κ ablation
rows in model dtype (the common single-ablation case is an elementwise dot +
axpy, avoiding MPS's expensive tiny matmul dispatch); only the fallback runs one
`subspace_inject` + `copy_`. There is no group loop, trigger re-check, or foot
seed on these paths. `all_fast_path()` is
true only unsteered (no hooks); `static_steerable()` is true when *every* hook is
that static-affine fast path — those are the two StaticCache / graph-capture
eligibility signals (`session.use_static_cache` ORs them). A curved / gated /
phased hook consults ctx + foot state per step and disqualifies both. `subspace_inject`
returns its fp32 result and the hook's `copy_` does the model-dtype downcast (no
per-fire downcast temp). For curved fits with a fuzzy σ-field, `onto` lands inside
the learned tube rather than collapsing every activation onto the mean wire.

`SteeringManager` owns `subspaces` (dispatch-synthesized merged affine, one per
trigger group, via `add_subspace`) + `manifolds` (curved, via `add_manifold`).
`apply_to_model` lowers both to per-layer entries: it share-weights (mean-1
normalized — `_manifold_layer_shares` prefers the baked `mahalanobis_share`, else
the Euclidean `‖eval_rbf(node_params)‖_F`), orthogonalizes the affine subspace
against curved spans (`_orthogonalize_affine_against` — curved wins shared
directions), and enforces `_CURVED_ORTHO_TOL = 1e-3` between two curved manifolds
(`OverlappingManifoldError`). Gain: three constants — `_SUBSPACE_GAIN = 16.0`
scales `along` on the **affine** path (whitened-unit target → free push *magnitude*,
overshoot-safe; live-calibrated — see below); `_MANIFOLD_ALONG_GAIN = 4.0`
scales `along` on the **curved** path, where the target is raw node coords so
`eff_along` is a *fraction of the way to the node* (`1.0` lands on it; `norm_cap`
bounds off-domain RBF extrapolation). Clean-stateless-calibrated on a gemma-4-12b
`months_loop%january` sweep: `along=1.0` → `eff_along≈4` lands the vivid coherent
winter sweet spot. For **periodic `BoxDomain` fits** `eff_along` is now clamped
and share-weighting dropped: `eff_along = max(0, min(1, along·_MANIFOLD_ALONG_GAIN))`,
uniform per layer, so no layer wraps past the target node. Non-periodic curved fits
keep the share-weighted unclamped path (`eff_along = share_L · gain`, which can
exceed 1 on high-share layers and is bounded only by `norm_cap`).
`_MANIFOLD_ONTO_GAIN = 0.5`
scales `onto` only (calibrated on the gemma-4-12b `emotions%dominant` onto sweep — at
`1.0` even `onto=0.5` fragmented and `onto=1.0` collapsed; `0.5` makes `onto∈[0,1]` a
usable dial with `1.0` a coherent ceiling). `eff_along_L = share_L · gain` (affine:
`gain=16`, α already in `target_coord`; curved: `gain=4.0` × clamped user `along`);
`eff_onto_L = clamp(onto · share_L · _MANIFOLD_ONTO_GAIN, 0, 1)`. **No lever / N, no `[0,1]` clamp / water-fill on
`along`** (a high-share layer is meant to overshoot; `norm_cap` bounds it). `onto`
stays clamped `[0,1]` (beyond 1 would overshoot through the wire/tube).
(`_SUBSPACE_GAIN` jumped ~130× *up* from the prior `0.125` when the affine
target went whitened-unit — the avg per-layer whitened push is now `GAIN·α`,
target-independent, where it used to scale with each node's raw-Euclidean distance
from neutral; `16.0` is live-calibrated on a gemma-4-12b α-sweep so `α ≈ 0.5`
clearly steers concepts *and* personas while staying coherent for the fragile ones
(effective `GAIN·α ≈ 8`; coherence ceilings vary ~2× per target — §10 — so a hard
persona breaks by `α ≈ 1.0`). Tagged a prototype.)
`reset_manifold_feet` cold-starts followers per generation.

`HiddenCapture` — public API: `attach`/`attach_persistent`/`detach`/`clear`,
`stacked`/`latest_per_layer`/`per_layer_buckets`/`tail_slice_at`, the mode setters
`set_incremental`/`set_aggregate_tail`/`set_tail_with_sink`, the post-forward
`fire_step_sink`/`ingest_persistent`, selective-prefill
`set_prompt_positions`/`prompt_stacked`, and `is_transient()` (true iff transient
per-gen forward hooks are registered — the compiled-clean routing gate). No
caller reaches into `_per_layer`/`_handles`/`_step_sink`: `per_layer_buckets()`
returns the raw bucket dict (a plain attribute return — **zero per-token cost** on
the WS read path), `set_tail_with_sink(depth, sink)` arms a deep tail ring AND a
per-token sink together (neither single setter gives both — `set_incremental`
forces length-1 and drops the ring, `set_aggregate_tail` installs no sink), and
`is_transient()` replaces the `not _handles` check.

The session picks the capture mode in `_begin_capture` from `need_per_token` and
stores it as one `CaptureMode` on a `CaptureState` dataclass (`session.py`;
`mode` + orthogonal `persistent` + the GATING_SUBSET `gating_subset`/`gating_keys`
— replacing the five correlated booleans that used to make illegal combinations
representable). The `_score_*` dispatch keys off `mode`:

- **`INCREMENTAL`** (`set_incremental`) — a full-reading live consumer wants
  per-token readings: overwrites a single preallocated length-1 buffer per layer
  via `copy_` (zero per-step capture allocation, O(layers·D) memory). The step
  sink scores each token live — but it now fires **post-forward** (`generate_steered`
  calls `HiddenCapture.fire_step_sink` after `model()` returns, not from inside the
  hook at the max probe layer; FIX F1), so the host-side score read no longer drains
  the device pipeline mid-forward. The per-token `ProbeReading` rows back
  `_score_incremental`'s (aggregate, per-token).
- **`LEAN_INCREMENTAL`** (FIX F2) — the live consumers
  read only the axis-0 coord (the SSE trait stream / loom probe row), no nearest /
  assignment / per-layer trace and no probe gate: the post-forward step sink scores
  each token `coords_only=True` (skips the big-K nearest norm + assignment softmax +
  per-layer host reconstruction) into the per-token coord stream, while a bounded
  tail ring lets `_score_lean_incremental` re-score the **full** aggregate once at
  finalize. Built on `set_tail_with_sink`, like the gating-subset path.
- **`AGGREGATE_ONLY`** (`set_aggregate_tail`) — probes attached but *nothing*
  consumes a per-token reading (no gate, no loom row, no trait stream, no live
  scores — e.g. a stateless server gen): keeps a bounded tail ring
  (`_AGG_TAIL_DEPTH = 8`) and runs NO step sink, so the decode loop pays zero
  per-token scoring; `_finalize_generation`/`_score_aggregate_only` pool the last
  *content* token once via `tail_slice_at` (the ring is deep enough to walk back
  past trailing specials) and one `Monitor.score_aggregate`.
- **`GATING_SUBSET`** (FIX #4) — per-token scoring is needed *only* to feed probe
  gates (`set_tail_with_sink`): the sink scores just the gated subset's scalars per
  token (into `_incremental_gate_scores`) while the tail ring lets finalize pool the
  FULL roster once. `CaptureState.aggregate_only` covers this mode too — its
  per-token rows feed only the gate, so it shares the aggregate-only finalize.
- **`FULL`** (append) — `return_hidden` (widen) or any non-incremental
  read, plus the degenerate no-probe / capture-disabled state: distinct clones per
  step so `stacked()` builds the full `[T, D]`; `score_per_token`.

Every read is a full per-probe `ProbeReading` either way; the modes trade only
*when/how often* scoring runs (per token vs once) and memory (length-1 vs tail
ring vs full stack).

Loom-attached generation also scores visible **authored** spans on the prefill
that already consumes them. `_pending_authored_prompt_targets` selects authored
node channels with no prior token capture; `_match_authored_prompt_targets`
locates their exact content-token sequences in the full rendered prompt and maps
token `j` to its producer state at `j-1` (the same semantics generated rows use).
`HiddenCapture` clones only those selected prompt rows at each active
probe/J-LENS/SAE layer, not the full prompt stack. A KV-prefix hit translates
full positions into suffix-local rows and leaves cached producers untouched;
selective prompt capture forces transient hooks because persistent buffers carry
only the latest slice. The first post-forward callback scores the rows in prompt
order, writes one `capture_authored` loom mutation per populated channel, then
runs the ordinary final-prompt sink. There is no second transformer forward, but
the requested monitor/SAE/J-LENS readout work still runs per authored token.
Captured authored rows are immutable: rerolls reuse their original data, while a
standalone append remains plain until a later generation actually prefills it.

## monitor.py

`Monitor` — **one** read-side class (the unification of the former `TraitMonitor` +
`ManifoldMonitor`) that reads every probe shape as whitened subspace coordinates,
flat and curved alike. Hook-driven, fp32, Mahalanobis-only (`covers_all` or
`WhitenerError`). **One** read shape, two execution paths: `_score_full` scores the
whole *flat* roster together in `_score_flat_batched` (one `Σ⁻¹h` Woodbury apply +
stacked / block-diagonal matmuls + a single host transfer per layer, scattered into
global per-probe slots) and runs each *curved* probe through `_score_probe_full`
(the per-probe `invert_parameterization` foot solve, warm-started across decode
tokens from the previous foot when `enable_curved_warm` is set — the sequential
live path). No flat/curved field asymmetry — both yield the full `ProbeReading`
(the research-tool priority is full per-token information: nearest, curved coords,
residual, per-layer). Both assembly sites also stamp the per-axis
`depth_com`/`depth_spread` stats via the module-level `_depth_stats` (mass
`share_weight_L · |coord_L[axis]|`, depths `layer/(n_layers−1)` — the
`n_layers` ctor kwarg the session supplies; unset ⇒ stats stay empty). Pure
host-side arithmetic over values the reading already transferred — zero tensor
cost. `score_aggregate` pools one token (the last content token)
and runs the per-probe `_score_probe_full`, so the aggregate is bit-identical to
the live read at that index. Each layer's `_layer_geometry` yields the M-orthogonal
**fraction** `sqrt(gᵀ M_R⁻¹ g)/‖x‖_M` (`g = B Σ⁻¹ x`), the whitened query for the
**nearest** `M_R`-metric node, and the M-projection reduced coord `c = M_R⁻¹ g`.
From there:

- a **flat** (affine) probe recovers `coords` through the affine reduced→domain
  map (`coord_S @ c + coord_b`); off-surface `residual` is identically `0`.
  **domain-frame**: each layer's reduced coord is in that layer's `‖δ_L‖` units,
  mapped to the shared domain *before* share-averaging (reference = each node's
  whitened read-coord). At rank-1 this is the pole-normalized coordinate (`1.0` at
  the positive node); a 1-node fold (monopolar `extract`, ad-hoc `probe()`) is the
  K=1 ray (`coord_b = 0`). A 2-node concept axis *and* the rank-8 `personas` fan.
- a **curved** probe recovers `coords` via the `invert_parameterization`
  nearest-point foot solve (run **per token** now — the accepted live cost) plus a
  real normalized off-surface `residual`.

The **neutral anchor competes in `nearest`** as a virtual candidate (`NEUTRAL_LABEL`)
— every fit is neutral-anchored, so neutral is a point in the same whitened metric
as the nodes (`_LayerWhiten.neutral_white`: the zero vector for an affine fit, the
baked per-layer `origin` mapped through `eval_at → basis → chol` for a curved one),
never a stored node. The per-probe path appends it as the `K`-th row of the cdist;
the batched flat path gets it for free (the zero `node_pad` column already holds the
affine neutral, so its distance falls out as `‖cq‖` — one reserved valid column, no
hot-loop change). When the activation sits closer to the origin than to any node,
`nearest` reports `("neutral", dist)`. `_attach_manifold_probe` sets
`inject_neutral = NEUTRAL_LABEL not in node_labels`, so a real node named `neutral`
keeps sole ownership of the label.

`score_aggregate` (end-of-gen) pools the last non-special token per layer and calls
the *same* `_score_probe_full`, so the aggregate at a token index is bit-identical
to the live read at that token. `add_probe(name, manifold, *, top_n)` /
`remove_probe(name)` attach/detach any shape. The read shares
`_build_whitened_factors` (per-layer `_LayerWhiten` build), `_attach_manifold_probe`
(node cache + per-layer Mahalanobis-share read weights), and `_layer_geometry`. `__init__`'s `layer_means` is
vestigial on the hot path — the readout centers on each fit's own
`LayerSubspace.mean`. `probe_layers` is the capture-widening union.

`flat_scalars` (one staticmethod) writes the gate channels from a readings dict:
`"<name>"` (= coords axis 0) + `"<name>[i]"` per axis, `"<name>:fraction"`,
`"<name>@<label>": −distance` per nearest node (including the reserved
`"<name>@neutral"` when the neutral anchor wins a top-N slot). The nearest
distance is reported in units of the probe's **typical label spacing** —
`AttachedManifoldProbe.label_scale`, the median node nearest-neighbor whitened
distance (a robust single per-probe scalar derived from the same per-node
bandwidths `_compute_assign_bandwidth` produces). Raw whitened distance spans
~60× across fits, so a bare `@label` threshold wasn't portable; `d/label_scale`
("typical label-spacings away") transfers. Because the scale is a per-probe
constant, `nearest` still ranks by **raw** distance (literally nearest — distinct
from the density-aware `~<label>` assignment, which keeps per-candidate `τ`).
Every probe — flat and curved — carries coords *and* nearest, so flat probes
expose `@label` similarity gates too (`@when:personas@hacker`,
`@when:personas@neutral`) and the gate grammar is uniform.

**Fuzzy reads** (the soft/distributional view of `nearest`/`residual`): every
`ProbeReading` also carries `assignment` (a `softmax(−d²/(2τ²) − R·log(τ))` node
posterior — a proper isotropic R-D Gaussian-mixture posterior with uniform node
prior, the distributional counterpart to argmax `nearest`) and `membership`
(`exp(−residual²/2σ²)` tube-fit density ∈ `[0,1]`, `1.0` for flat / no-σ fits).
The per-candidate bandwidth `τ` AND the precomputed Gaussian log-volume bias
`−R·log(τ)` are computed at attach (`_compute_assign_bandwidth` →
`AttachedManifoldProbe.{assign_bandwidth, assign_logvol_bias}`): `τ` is a curved
fit's σ-field mapped into the whitened metric (×`√(tr(M_R)/R)`) or a flat fit's
local layout scale (each node's nearest-neighbor whitened distance), and the
log-volume bias is the missing Gaussian normalization (without it the bare
`−d²/2τ²` softmax has a broadest-node-wins pathology — a diffuse-τ candidate
sits near logit 0 while crisp-τ candidates have strongly-negative logits, so the
wide node swallows probability regardless of distance; the bias inverts this to
the standard "tight when near, broad when far" mixture behavior). `R` is the
manifold's per-layer subspace rank (rank-uniform across a fit's layers). Soft
assignment is uniform flat+curved with the hot path gaining only one softmax +
one add over the distances it already computes (folded into the single host
transfer in both `_score_probe_full` and the batched `_score_flat_batched`).
`flat_scalars` emits these as `"<name>~<label>"` (assignment prob) +
`"<name>:membership"`. Entry points:
`score_per_token` (primary — returns `(aggregate readings, per-token axis-0 coord
stream)`), `score_single_token{,_per_layer}` (`_per_layer` is a view over the
reading's `coords_per_layer` backing the loom heatmap), `measure_from_hidden`,
`score_stack`, live-mean `begin/update/end_live`. History/stats are
per-coordinate-axis (`axis_stats`); the scalar compatibility helpers report axis 0.
The bundled roster is the fitted 2-node `Manifold`s themselves
(`_bootstrap_manifold_probes` — no fold). The one-shot re-render text scorer
(`measure`) is gone — every read source is live hooks scoring captured hidden
states, no second forward pass.

## monitor_attach.py

Attach-time probe algebra, extracted from `monitor.py` (which it shrank by ~620
lines, to ~2057 L). Runs once per `add_probe` call, entirely off the hot path.
Hosts: `AttachedManifoldProbe`, `_LayerWhiten`, `_build_whitened_factors`,
`_attach_manifold_probe`, `_compute_assign_bandwidth`, `_layer_geometry`, the
affine-coord helpers, and `_woodbury_apply`. The hot path in `monitor.py` imports
`_layer_geometry`, `AttachedManifoldProbe`, `_build_whitened_factors`, and
`_attach_manifold_probe` back from this module (no circular dependency —
`monitor_attach` imports only `mahalanobis` and `manifold`).

## instruments/

The read-side instrument protocol (phase 1 of the 5.x unification — the
read-side analogue of `SteeringComposer`). One contract over the three read
families: geometry (Monitor subspace probes), the Jacobian-lens readout
channel, SAE feature reads. `types.py` is the shared vocabulary:
**`GateRef`** (structured probe-gate reference; `parse_gate_ref` is the ONE
place the scalar-key discrimination lives — it inverts the key family
`Monitor.flat_scalars` emits; the hot-path runtime lookup stays the verbatim
string) + `validate_gate_channels` (composition-preflight rejection of
channels a family can never produce — `@when:sae/123:membership` raises
`UnsupportedProbeChannelError` (400) instead of sitting silently inactive;
a *supported* channel temporarily absent this step still reads inactive);
**`ScalarReading`** (the honest one-channel reading for lens/SAE, with an
explicit `unit` — `mean_token_probability` / `activation_over_max` /
`raw_activation` — and a `DepthSummary` that carries its mass `basis`,
because `depth_com` means three mathematically unrelated things across
families; `to_probe_reading()` is the phase-1 wire bridge reproducing the
historical synthesized-`ProbeReading` shape bit-for-bit until the phase-2
measurement envelope lands); **`InstrumentPlan`** (declared capture demand,
not mechanics — the session planner unions plans and picks physical
retention; the `INCREMENTAL → set_tail_with_sink` upgrade is
cross-instrument resource sharing and stays session-side);
**`InstrumentPrep`** / **`LensPrep`** (the generation-boundary source
snapshot, produced by `prepare` and consumed by `plan` + `bind` — every
prep carries a per-preparation `token` the derived plan echoes and bind
compares; the lens prep additionally carries the disk refresh + pin,
the authoritative spec/live snapshot derived from the prepared
identity, and the sidecar fingerprint the binding stamps);
**`InstrumentBinding`** (immutable per-generation source/spec snapshot, so
un-locked mutations like the SAE metadata backfill can't change what a
running generation measures); per-family `LiveConfig` dataclasses (user
intent, discriminated — deliberately NOT unified with runtime residency:
lens "enabled" is not device residency, SAE residency precedes its toggle).
`protocol.py` holds the `Instrument` / `InstrumentRun` contracts —
`InstrumentRun.observe` memoizes by `step_id` while bound (the production
gate→display reuse is the workers' stash mechanism, run-scoped by
construction; see the post-review hardening notes below), `observe_many`
covers batch generation, and authored-prefill orchestration (token matching,
`j-1` producer semantics, loom persistence) deliberately stays session-side.
Family implementations land incrementally: lens + SAE instruments extract
their session.py regions; geometry stays a thin adapter over `Monitor` (the
four capture modes are session/HiddenCapture state, not Monitor internals).

`lens.py` — **`LensInstrument`**, the extracted J-lens family (the former
~675-line session region): owns the probe registry (`probes`), the
live-readout runtime dict (`live` — device-resident `J_stack` + unembed/norm
modules), the per-forward stash (`step_stash` + `last_step_readings`), the
per-generation flags (`active_for_generation`) and disk-identity pin
(`generation_lens`/`generation_lens_active`), and the read surfaces:
`attach`/`detach`/`specs`/`probe_layers`, `score_probes` (readout-channel
`ProbeReading` synthesis), `gate_scalars` (once-per-forward band logits +
stash for display reuse), `score_aggregate` (tail-ring pooled last content
token), `live_readout_step` (token-tap display; gate-stash row reuse),
`authored_capture` (per-row computation only — token matching and loom
persistence stay session-side), `enable_live`/`disable_live`/`live_layers`,
`probe_hash` (the `jlens-readout-v2` identity digest), and `validate_gate`
(strength axis only). Shared J-lens *primitives* (`_jlens_logits_rows`,
depth/decode caches, the transport stack, the `jlens` disk-identity
property) stay on the session — the instrument reaches them through the
session back-ref exactly like `SteeringComposer`. The session re-exposes the
instrument's state under the historical private names
(`_lens_probes`/`_live_lens`/`_lens_step_stash`/`_last_lens_step_readings`/
`_live_lens_active_for_generation`/`_generation_jlens{,_active}`) via
delegating properties, and `_lens_instrument` itself is a lazy property
(first touch constructs it — ctor only stores the session ref) so narrow
test stubs that skip `__init__` self-heal.

`sae.py` — **`SaeInstrument`**, the extracted SAE family (the former
~560-line session region), same shape: probe registry, live-discovery
config (`live` — `{layer, top_k, source}`), per-forward stash (one encode
shared by gates, pinned probes, and the live display on a step),
per-generation flag, and the read surfaces (`attach`/`detach`/`specs`,
`score_probes`/`probe_values`, `gate_scalars`, `score_aggregate`,
`live_readout_step`, `authored_capture`, `enable_live`/`disable_live`/
`is_live`, `probe_hash` — the `sae-readout-v2` unit-carrying digest,
`validate_gate` — strength axis only). Backend *residency*
(`_sae_backend`/`_sae_layer`/`_sae_width`, `_require_sae`,
`_encode_sae_hidden`), the Neuronpedia metadata cache/fetchers, the
train/load/unload lifecycle, and the offline `sae_token_readout` stay
session-side (residency is runtime state shared with steering atoms, not
probe intent). Session re-exposes state via the same delegating-property +
lazy-instrument pattern. **Deliberate 5.x semantics change:** SAE gate
scalars no longer emit the fake `:fraction = 0.0` / `:membership = 1.0`
constants — only the real strength channel (`<name>`/`<name>[0]`); a gate
on a channel the family can never produce is a composition-preflight error
(`validate_gate` → `UnsupportedProbeChannelError`), wired at the composer
in the plan/run slice.

`geometry.py` — **`GeometryInstrument`**, the thin adapter giving the
Monitor family the same face as its siblings: `attach` (the exclusive-GPU
whitener-touch + resolve + `monitor.add_probe` flow), `detach`, `specs`,
`names`, `validate_gate` (every channel — axes, fraction, membership,
distance, assignment), and `probe_hash` (the baked-tensor / subspace-
geometry digest). The Monitor engine and the four capture modes stay
untouched (folding them in would combine an orchestration extraction with
an engine rewrite). `session.instruments` is the uniform registry —
`{"geometry", "lens", "sae"}` → instrument — behind the probe-hash roster,
gate preflight, and the server's phase-2 instrument enumeration.

**Public faces (P3):** `session.instruments` is the family registry;
`session.lens` / `session.sae` return the instrument objects themselves —
the typed public API for probe attach/detach, live toggles, and scoring.
Artifact/source lifecycle (`fit_jlens`, `select_jlens_source`,
`jlens_readout`, token replay; `train_sae`/`load_sae`/`unload_sae`/
`sae_info`) stays on the session: source management is not measurement.
The flat `session._score_*`/`enable_live_*` delegates remain as internal
forwarders for the server/CLI call sites. `LensInstrument`/`SaeInstrument`/
`GeometryInstrument`/`UnsupportedProbeChannelError` are exported from
`saklas/__init__.py`.

The composer's four gate-key walks collapsed onto one structured
`_gated_refs()` pass (`parse_gate_ref`), which also **channel-validates**
lens/SAE-attached gate references at generation preflight
(`UnsupportedProbeChannelError`, 400) and fixes the old
`re.split("[\\[:@~]")` truncation of variant-suffixed probe names.

Both formerly deferred slices are landed. **Plan-driven capture:** every
family implements `plan(prep) -> InstrumentPlan` and
`_begin_capture` (plus the batch preamble) unions the declared
`latest_layers`/`tail_layers` instead of hand-rolled per-family branches;
retention-mode selection stays session-side (the
`INCREMENTAL → set_tail_with_sink` upgrade is cross-instrument resource
sharing). **Source prep:** `prepare(ReadRequest) -> InstrumentPrep` is
the generation-boundary source step, run after `close_run` — the lens
family reads the disk-refreshing `session.jlens`
getter under pin demand there (the one formula both boundaries reduce
to: a live readout, or attached probes with a final aggregate or a lens
gate) and returns the `LensPrep` carrying the pin **and the
authoritative spec/live/fingerprint snapshot** (spec layers derived from
the prepared lens identity — a pin-demanded lens without
`source_layers` is a hard prepare failure, and narrow test stubs use
real-shaped `SimpleNamespace(source_layers=…)` lenses; the live runtime
dict by reference; the sidecar identity for the binding); geometry/SAE
preps carry the request forward. The whole lens snapshot is ONE atomic
transaction under `LensInstrument.state_lock` — a reentrant leaf lock
(outer locks like `_gen_lock`/`_model_exclusive` are always taken
first) that also serializes the getter's refresh/adopt/evict,
`has_compatible_jlens`, `_adopt_fitted_jlens`/`_evict_resident_jlens`,
probe attach/detach, and the live toggles — so the snapshot cannot tear
mid-`prepare` (sol's round-4 P1: without it, an adoption landing
between the refresh and the live-state read paired lens A's specs with
lens B's live device stack). `plan` and `bind` consume the prep only
— never the live registry — so an interleaved adoption inside the
prepare→bind window (the un-locked `has_compatible_jlens` on the
session-info route can trigger one from another thread) cannot pair the
prep's pinned lens with the replacement's rewritten layers (sol's
round-3 P1; regression tests pin both interleavings). Every family's
`prepare` raises on a still-bound run (a stale pin would short-circuit
the very refresh the step exists for). **Formal runs:** `bind(plan,
prep)` — prep mandatory, family provenance validated on BOTH plan and
prep (a bare `bind(plan)`, a wrong-family prep, or a wrong-family
`LensPrep` all raise), and the plan must echo the prep's
per-preparation `token` (`InstrumentPlan.prep_token` — same-family
plan/prep crossing from different prepare() calls raises, sol's round-4
P2) — freezes an `InstrumentBinding` from the prep's
specs and the prep's `fingerprint` (a bind-time live identity read
could stamp a concurrently adopted replacement onto a run pinned to the
older lens; the SAE binding resolves `max_act` at bind, so the un-locked
Neuronpedia backfill can no longer change a running generation's
strength unit — sol's race), installs the prep's pin/live decisions, and
returns the family's
run (`LensRun`/`SaeRun`/`GeometryRun`), which owns the instrument-side
generation-scoped state: step stashes, display readings, active flags,
the lens disk-identity pin, the prepare-time live-state snapshot, and
the `observe` step-memo (memoized only
while bound — the idle run persists indefinitely, so it never memoizes).
Geometry's curved warm feet stay in the Monitor engine and the four
capture modes stay session/HiddenCapture wiring. Instruments keep the
historical state names as delegating properties over `current_run` (an
idle passthrough run backs out-of-generation reads — measurement specs
come from `_measurement_specs()`: the frozen binding when bound, the
live registry when idle; the live-readout state symmetrically from
`_measurement_live()`: the run's prepare-time snapshot when bound, so a
bound run keeps reading the live stack that matches its pin even after
an adoption rebuilds the instrument-level `live`).

Post-review hardening (sol, gaslamp thread `instrument-protocol`):

- **Ordering contract** — a capture transaction is the uniform protocol
  sequence `close_run → prepare → plan → bind` at both generation
  boundaries (`_begin_capture` and the batch preamble): the defensive
  `_close_instrument_runs()` first, then `prepare` takes the lens disk
  refresh + pin (`session.jlens`, whose adoption path REWRITES live
  probe layer lists) and snapshots specs/live against that identity, and
  `plan`/`bind` consume the snapshot — a plan or freeze read off the
  live registry can pair the prep's lens with a concurrently adopted
  replacement's layers and KeyError in the transport stack. Pin demand
  is the registry boolean, not plan
  emptiness (probes whose lens vanished still pin the validated-missing
  state). Formerly the refresh/pin dance was session special-casing
  around `bind(lens=, pinned=)` kwargs; it is protocol shape now
  (`LensInstrument.prepare`), and the ordering is enforced, not
  advisory: prep-less or foreign-family plan/bind calls raise.
- **Every capture transaction closes its runs, exception-hard** — the
  generation finallys, the batch finally, and the joint-logprob replay
  all reach `_close_instrument_runs()` through nested finallys, so a
  raising hook detach cannot skip run closure, and the gen-lock release
  sits behind the outermost finally (an earlier teardown exception must
  not wedge the session).  `close_run` is protocol surface (the
  bind/close asymmetry is what let the replay leak bound runs, pinning a
  stale lens between generations).
- **Mutations apply next generation — lens/SAE only.** Their measurement
  guards (score/aggregate/live/authored/batch-row) consult
  `_measurement_specs()`, so an un-locked mid-generation detach (the
  synchronous DELETE route) cannot change a bound generation's roster.
  Geometry has **no roster snapshot yet** (Monitor scoring walks the live
  `_probes` dict), so `GeometryInstrument.detach` takes
  `_model_exclusive` like `attach` and a served detach during a running
  generation **rejects with retry-shortly semantics** instead of racing;
  `session.remove_probe` routes geometry removals through the instrument.
  A true Monitor roster snapshot (extending the next-generation contract
  to geometry) is deferred — it is engine surgery, not guard cleanup.

Per-step gate→display matrix reuse remains the stash mechanism inside
the workers, run-scoped by construction; `observe(step_id)` is
protocol-complete but the hot paths are not yet routed through it.

## triggers.py

`Trigger` (frozen): phase flags + optional `ProbeGate`. `Trigger.active(ctx)`
consults flags and, when gated, `ctx.probe_scores[gate.probe]` against `score <op>
threshold`. Factories `first`/`after`/`when`. `TriggerContext.probe_scores` is
filled by the per-step callback, cleared on `reset()`. Gated triggers are inactive
during prefill and for missing probes (no raise). `ProbeGate` is frozen/hashable
so identical gates compose under equality; `ProbeGate.probe` is the canonical
scalar key from whichever monitor supplied it.

## cuda_graphs.py

CUDA-graphs / `StaticCache`. Eligible for **unsteered** generation
(`all_fast_path`) **and** for the static single-affine steered case
(`static_steerable` — a `Trigger.BOTH` affine slide, no ctx/foot/gate);
curved / gated / phased / ablation steering forces the eager DynamicCache
path. `is_cuda_graphs_supported(model, device)`
probes viability + caches keyed by underlying module id / device / dtype;
`make_static_cache` sizes to `prompt_len + max_new_tokens + offset`; `warn_once`
logs the fallback reason.

## scene.py

The cast-model turn stitcher: **template
autopsy** renders sentinel-content probes through the live chat template and
slices out a `TurnGrammar` — prelude, per-seat `SeatWrapper`s (open/label
site/close), system shape (real turn vs gemma-style fold, or unsupported —
gemma-2), generation appendices (`gen_extra` under `enable_thinking=False`,
`gen_extra_thinking` under True; `None` = thinking-mode stitching
unsupported), content/system trim flags (probed with padded sentinels —
gemma/llama/qwen3.5 `| trim` content; trim applies to turn *text* before
thinking/fold composition), `last_assistant_special` (qwen3 inserts an empty
think scaffold into the final assistant turn of a *closed* render — that
shape falls back to the template), think delimiters + the empirically-probed
`strips_history_thinking` (strip families render a turn's committed thinking
only while it is the last turn before the gen header — "lasts one turn", a9
convention 3). `render_scene(grammar, turns, *, system, gen_seat, gen_label,
gen_thinking)` stitches arbitrary `(seat, label, text, thinking)` sequences —
alternation NOT required, labels placed in *constructed* headers (the
`_splice_occurrences` label-collision class is structurally impossible) —
plus a trailing generation header on either seat. `validate_turn_grammar` is
the **load-bearing gate**: byte-exact round-trip against the template's own
render (plain / gen / padded / system / closed-on-assistant / thinking-gen
cases); a passing family's alternating renders are bit-identical through the
stitcher (extraction baseline contract), a failing family falls back with a
warning (`SceneGrammarError`). `render_scene_raw` is the marker-mode
fallback (base models, mistral) — also the `build_chat_input` base-model
branch. `SaklasSession.scene_grammar` is the lazy per-session
autopsy+validation (None = fallback); `build_chat_input(scene=, gen_seat=)`
routes through the stitcher when a grammar is present, byte-identical on
every legacy render, and is the only path that can open a user-seat
generation prompt. Live-validated on 11 real templates (gemma-2/3/4,
llama-3.2, qwen2.5/3/3.5, talkie; GLM-4.7-Flash falls back — braids think
markers into turn structure).

## generation.py

Token-by-token decode + KV cache under `torch.inference_mode()`. Models that
ignore `past_key_values` (e.g. talkie) flip `no_cache_mode` (O(N²), one warning).
`GenerationConfig` (frozen) holds session defaults; per-call `SamplingConfig`
composes into a local copy. Top-p via `torch.topk` (not full-vocab sort); `top_k`
(default 1024 cap) applied before top-p. `generate_steered` accepts `seed`, `stop`,
`logit_bias`, presence/frequency penalties, `logprobs`, `score_callback` (probe
gates), `forced_prefix`, `steering_active`, `want_perplexity`. `forced_prefix`
forces the first N decode tokens while the multinomial draw still runs (re-seeding
stays bit-identical through the fork). `want_perplexity=False` skips the per-token
entropy `.item()` host sync (one sync/token) when no consumer surfaces perplexity —
the session passes False for stateless gens that aren't loom-attached / opted-in
(server streaming never reads per-token ppl); `cand_logp` is still computed when
`logprobs` needs it. Stop-sequence matching keeps only a bounded tail of the
emitted text (`_stop_keep = max(len(s))−1` chars) instead of growing the full
completion, so the per-token match is O(tail+emit) not the old O(n²) concat. The
`SaklasSession` fork/append entry points (`fork_from_token` /
`prefill_assistant` / `append_user_turn` / `append_assistant_turn`, all in
`session.py`) build on this decode loop. Matching structural role + role label
coalesces authored spans in place; ordinary one-shot generation reuses a
matching leaf and forces its existing text as the prefix, while explicit fan /
fork / prefill operations retain sibling semantics. `detect_base_model` (no
chat template) routes flat (`raw=True`) generation.
Per-message role substitution (`role_templates.apply_with_per_turn_roles`) backs
the roleplay scaffold; `_active_role` is the steering-driven role.
`supports_thinking` / `_detect_think_delimiters` round-trips Qwen/Gemma
`enable_thinking`, falling back to channel (gpt-oss) / bracket (Mistral-3)
detectors; the internal `_ThinkState` machine + `GenerationState` drive streaming.

## session.py

`SaklasSession` owns the model, the in-memory profile registry (`_profiles`), the
loaded-manifold registry (`_manifolds`), the unified `Monitor` (`_monitor`),
`SteeringManager`, `HiddenCapture`, generation defaults (`session.config`), the loom
tree, and a synchronous `events: EventBus`. Steering resolution + the LIFO steering
stack live on a `SteeringComposer` collaborator (`core/steering_composer.py`),
instantiated exactly once in `__init__` after `_monitor`. The composer solely owns
the stack (`_stack`), projection materialization, pole alias resolution, steering
lowering, and hook installation. `ensure_manifold_loaded` and
`ensure_profile_registered` are the session's single public artifact-registration
API. The session keeps narrow `_push_steering`/`_pop_steering` context forwarders
and generation helpers where the session boundary is meaningful.
The composer reaches session state through `self._session` at push/pop frequency, off
the per-token path; the one near-hot method, `build_gating_score_callback`, binds
`capture`/`monitor` to locals and reads the session capture state through the back-ref
exactly as before, so the per-token gate path gains no new indirection. Push/pop call
the session's `_rebuild_steering_hooks` forwarder (not the composer's own) so stub
overrides take effect. `from_pretrained(model_id, *, device,
dtype, quantize, probes, system_prompt, max_tokens=1024, dls=True, compile=False,
compile_mode=None, cuda_graphs=False, return_top_k=0)`
does the HF load + layer-mean compute + probe bootstrap — there is no
`injection_mode`/`theta_max` param. `__init__` takes a pre-loaded model. Both call
`materialize_bundled_manifolds()` + `selectors.invalidate()` early.
`_RESPONSE_MAX_TOKENS = 256` caps each in-character response (4.0 / A2
elicitation). Module-level helper `_affine_manifold_push`
(per-layer basis rows + node-coord targets for a flat manifold) backs the dispatch;
`core.manifold.manifold_is_affine` is the flat/curved predicate used across surfaces.
Conversational-elicitation helpers `_KIND_TEMPLATES` / `_article` / `_system_for`
(the per-kind system prompt — `custom` takes a caller-supplied template) /
`_role_for` (the swapped assistant-role label: abstract → `someone_{slug}`,
concrete → `{slug}`, `custom` → `None` (no swap — the custom system carries the
frame, pooled in standard-assistant space)) author each node's corpus.

`extract(concept, baseline, *, kind="abstract", ...)` authors a 2-node
discover-`pca` manifold (`create_discover_manifold_folder`,
`hyperparams={"max_dim": 1}`) and fits it via `ManifoldExtractionPipeline`,
returning `(canonical_name, folded_directions(manifold))`. The corpus is
generated conversationally — `generate_responses(concepts, kinds, *, roles=None,
custom_system=None, samples_per_prompt=1, …)` has each pole answer the shared baseline prompts *in
character*, the concept riding both the system prompt (`_system_for`) and the
swapped assistant-role label (`_role_for`); responses are emitted samples-outer /
prompts-inner so `response[i] ↔ prompt[i % k]`. Generation is **batched** —
`_run_generator_batch` left-pads a `_CORPUS_GEN_BATCH` chunk of prompts into one
`model.generate` (per-row independent sampling), replacing the per-prompt loop;
`_run_generator` stays as the single-shot `ModelHandle` seam. **Bipolar** (`baseline` set)
authors a 2-node folder and generates both poles. **Monopolar** (`baseline=None`)
authors a genuinely **1-node** folder (only the concept pole is generated); the
pipeline folds `concept − ν` into a 1-node neutral-anchored ray (see
`extraction.py`), neutral implicit via `layer_means`. `generate_neutral_responses`
is the neutral-corpus sibling (no system, standard label). `extract_from_corpora(..., kind=...)` is the corpus-in sibling
(hand-authored pairs skip generation; corpora pooled conversationally, each length
a multiple of the baseline prompt set); `fit(folder)` is the multi-node delegate
that returns a `Manifold`. All gate against `GenState.IDLE`
(fitting runs forward passes). `_fit_concept_manifold` is the shared tail. (The
training-free `clone_from_corpus` / `io.cloning` path is removed in 4.0.)

`generate` / `generate_stream` are keyword-only and accept `steering: str |
Steering | None` (dicts rejected), `sampling`, `thinking=None`, plus loom args;
both return `RunSet` (`.first` is the `GenerationResult`). `session.steering(value)`
coerces via `Steering.from_value`, materializes `ProjectedTerm`s into derived
profiles (`SteeringComposer.materialize_projections` → `project_profile` with `session.whitener`,
which is Mahalanobis-only — a `~`/`|` term on a session with no covering whitener
raises `WhitenerError`), and pushes a per-scope entries dict onto a LIFO stack so
nested scopes compose. There is no per-call projection-metric override.
`SteeringComposer.resolve_pole_aliases` canonicalizes bare poles via `io.selectors.canonicalize_atom`
(the sign-flip is retired — a bare pole resolves through the manifold-label tier as
a `%` push) and routes variants.

**Steering resolution (manifold-first).** `SteeringComposer.ensure_profile_registered(name)`
resolves a direction from, in order: (1) an in-memory baked direction already in
`_profiles` (ad-hoc `extract`/`merge`/projection results); (1b) the reserved
`jlens/` namespace — `register_jlens_direction` resolves the word through the
fitted Jacobian lens (raising `LensNotFittedError`/`MultiTokenWordError`, never
falling through to extraction); (2) a fitted
2-node `pca` manifold on disk — `try_fold_manifold` → `ensure_manifold_loaded`
(load `[ns/]name[:variant]`, raw or `sae-<release>`) + `folded_directions`,
memoized into `_profiles`. `_bootstrap_manifold_probes(categories, *, include_fitted_defaults)` is the
probe-roster bootstrap — one pass, two tiers. **Tagged concept axes**: for each
`default/` manifold tagged in a requested category, fit-or-load the 2-node
`Manifold` and hand it to the `Monitor` (rank-1 coordinate, no fold; replaces the
old `io.probes_bootstrap.bootstrap_probes`), registered under the **bare** name.
**Fitted multi-node defaults** (`include_fitted_defaults`, set when
`probes is None`): sweep every bundled `default/` manifold and additionally attach
any *already fitted* for the model and not already attached (`personas`,
`emotions`) under the qualified `default/<name>` selector — attach-only (never
fits; an unfitted one logs a skip), so a 107-node manifold can't block startup.
This folds the former serve-only `_attach_default_manifold_probes` into the
construction-time pass, so both frontends (serve / programmatic) get the
same roster; an explicit `probes=[...]` category list skips the multi-node sweep.

`SteeringComposer.compose_steering_entries` is the dispatch (`ARCHITECTURE.md` §4): classify each
entry — `AblationTerm` → ablation fragment; `ManifoldTerm` → affine `%` joins the
merge (`_affine_manifold_push`) or curved `%` → `add_manifold`; plain `(alpha,
trigger)` → `ensure_profile_registered` → `fold_directions_to_subspace` →
`_affine_manifold_push` push fragment — group push+ablate by trigger,
`synthesize_subspace` per group, `add_subspace`. `install_composed_steering` →
`apply_to_model`. There are **no** persistent hooks. Manifold-implied roles
aggregate under soft-warn + highest-`|coeff|`-wins (`RoleBaselineMismatchWarning`).

`_generate_core` owns the `_gen_lock` re-entry guard, `_gen_phase` lifecycle
(`IDLE`/`PREAMBLE`/`RUNNING`/`FINALIZING`, via `session.gen_state`), the steering
context, `_begin_capture`/`_end_capture`, `_finalize_generation`, and teardown.
Monitor scoring is in-flight (no second forward pass); `SamplingConfig(return_hidden=
True)` widens capture to every layer and lands `GenerationResult.hidden_states`,
re-scorable via `session.score_hidden`. Probes ride the same plumbing:
`add_probe(selector, *, as_name=None, top_n=3)` routes a reserved
`jlens/<word>` selector to the lens-probe registry (`_add_lens_probe` — the
readout channel, see "Lens probes" below), else resolves via
`_resolve_probe_manifold`
(a 2-node `pca` concept folds to a rank-1 probe via `_fold_profile_probe`; a
multi-node manifold attaches whole) and registers on the unified `_monitor`;
`remove_probe(name)` detaches either kind. `geometry_token_readout(node_id,
raw_index, *, apply_steering=, raw=)` is the geometry family's token replay —
the loom-anchored sibling of `jlens_token_readout`/`sae_token_readout` (same
prompt rebuild + one capture forward at the roster's layer union + the same
steering-scope-outside-`_model_exclusive` ordering), scored through
`Monitor.score_single_token` at the position that produced the clicked token
(curved warm-start forced off — a one-shot read must not seed from a prior
generation's foot). Post-hoc by construction: it reads aggregate-only
generations and probes attached after the fact; no attached probe raises
`ValueError`. `_begin_capture` widens to `_monitor.probe_layers()`
and picks the `CaptureMode` from `need_per_token` (gate ∨ loom row ∨ trait stream ∨
live scores ∨ per-layer persist — see hooks.py `HiddenCapture`), storing it on the
`CaptureState` dataclass (`mode` + `persistent` + the GATING_SUBSET subset/keys —
the legal-by-construction replacement for the former five correlated booleans):
per-token incremental scoring when something consumes a per-token reading, else
aggregate-only (no per-token scoring, pool once at finalize). The gate callback runs one
`_monitor.score_single_token` through `flat_scalars` into
`TriggerContext.probe_scores` (one key space —
`"<name>"`/`"<name>[i]"`/`"<name>:fraction"`/`"<name>@<label>"`);
`_finalize_generation` calls `_score_incremental` / `_score_aggregate_only` /
`score_per_token` (per mode) into `GenerationResult.probe_readings`.
`session.lock` is the server-owned `asyncio.Lock` (distinct from `_gen_lock`).
`generate_batch`/`generate_sweep` return `RunSet` (sweep builds the Cartesian
product as loom siblings). `generate_batch` auto-shares one prefill across rows
when they have a common token prefix and the steering is prefill-inactive
(`_maybe_cache_batch_prefix` → `cache_prefix` → consumed in `_generate_core`): the
`_prefix_cache` KV is unsteered, so reuse is valid exactly when no steering term
touches the prompt region — `_steering_active_in_prefill` (a term steers prefill
iff `trigger.prompt and gate is None`) gates both the consume path and steering
push/pop invalidation (a prefill-inactive scope preserves the cache, so a
`@response`-steered batch still shares the prefill; a default-`BOTH` batch re-prefills
per row). The alpha *sweep* steers the prompt at varying strength by construction
(`BOTH` terms), so it can't share a prefill — that's expected, not a gap. Hot-path
events: `GenerationStarted`/`SteeringApplied`/
`SteeringCleared`/`ProbeScored`/`GenerationFinished` +
`ManifoldExtracted`; threaded subscribers hop via `loop.call_soon_threadsafe`.

**Jacobian-lens surface.** `session.jlens` validates v6 sidecar/live-weight
identity before loading, refreshes or evicts an already-resident lens when an
external process replaces/removes its generation, then verifies the payload digest
while promoting each layer (`io/lens.py`, like `whitener`). Generation refreshes
that disk identity once in `_begin_capture` (or the compatible-batch preamble) and
pins the resulting resident lens through token scoring, gates, and final
aggregation; per-token paths never reopen the sidecar or its layer shards, and an
external pointer switch becomes visible at the next generation boundary.
`fit_jlens(prompts, …)` pre-filters too-short
prompts (so the saved `n_prompts` counts consumed prompts exactly — what makes
resume slicing sound), hashes the filtered corpus, resumes a matching partial
fit by default (`force=True` restarts), checkpoints via the io layer, and gates
under both a cross-process per-model fit lock and `_model_exclusive` (forward AND
backward passes). Shared readout helpers:
`_jlens_logits_rows` (the one bmm + unembed matvec over `(layer, hidden)` rows
— per-layer top-k and aggregate both consume it), `_jlens_topk_rows` (accepts
precomputed `logits=`), `_jlens_aggregate_rows` (`aggregate_readout` + decode),
`_jlens_depths` (`layer/(n_layers−1)`), `_jlens_decode_id` (the one cache-backed
single-token decode).
`jlens_readout(prompt, layers=, positions=, top_k=, aggregate=)` is the offline
readout (captures via `_capture_all_hidden_states`, default final position
only); `aggregate=True` returns `(per_layer, per-position aggregate)` from the
same logits, with both the aggregate and matrix covering every requested fitted
layer.
`jlens_token_readout(node_id, raw_index, *, layers=, top_k=, apply_steering=,
raw=)` is the loom-anchored readout behind the dashboard drilldown's j-lens
tab: fork-style validation (assistant node, `raw_token_ids`, range), rebuild
the node's exact prompt render via `_prepare_input` (stamped role labels +
recipe thinking; `raw=` selects the flat render — raw-ness isn't stamped on
the node, the caller supplies it), append `raw_token_ids[:raw_index]`, one
capture forward, per-layer top-k + the all-layer `aggregate` block at the
final position — the forward that
*produced* the clicked token. `apply_steering` replays under the node's
recipe steering; the steering scope opens OUTSIDE `_model_exclusive`
(`SteeringComposer.push`/`pop` take `_gen_lock` blocking — nesting would
self-deadlock; same ordering as `score_choices`).
`register_jlens_direction(word)` lands `W_U[v] @ J_l` in `_profiles` under
`jlens/<word>` — the resolver behind the lazy `jlens/` *steering* branch
(`ensure_profile_registered`) only. `jspace_decompose(selector, k=,
layers=)` resolves any steerable direction and splits it against the lens
dictionary.

**Lens probes (readout channel).** `add_probe("jlens/<word>")` routes to the
session lens-probe registry (`_lens_probes: name -> {word, token_id, layers}`,
all fitted layers; `_add_lens_probe` validates the artifact + single-token word
and pre-warms the device transport stack) — **never** the Monitor, never a
direction fold, no whitener. The reading is the readout-channel synthesis of
`ProbeReading` (`_score_lens_probes`, now a thin delegating forwarder to
`LensInstrument.score_probes`, over `jlens.token_readout_stats`):
`coords = (strength,)` — the ONE channel, mean fitted-layer probability
`mean_l p_l(v)` (the gate channel `@when:jlens/<word>` and the workspace
card's `strength`) — `coords_per_layer[l] = (p_l,)`, probability-mass-weighted
`depth_com`, geometry fields defaulted. The live
per-layer top-k display wire carries per-layer softmax probabilities too
(`_live_lens_readout_step` softmaxes before top-k; monotone, so the
ranking is unchanged), so every lens surface reads one unit. Three read sites,
one math: per-step display readings ride the live-lens step's one calibrated
probability matrix
(`_last_lens_step_readings`, merged into every populated payload channel via
`TokenProbePayload.merge_readings`); gate scalars come from
`_score_lens_gate_scalars` in the gating score callback (once per forward,
`Monitor.flat_scalars` over the synthesized readings, and the computed logits
plus probabilities are stashed in `_lens_step_stash` for the display step to
reuse — the gate
callback runs before the token tap); the finalize aggregate
(`_score_lens_probes_aggregate`, called from `generation_finalizer`) pools
the last content token from the capture tail ring like `_score_aggregate_only`.
Capture accounting: fitted lens-probe layers join the `_begin_capture` union;
the INCREMENTAL branch swaps `set_incremental` for `set_tail_with_sink` when
lens probes are attached (finalize needs the ring); the lens-only tail branch
covers `_lens_probes` as well as the live lens; and a **lens-only** gate does
not force per-token *monitor* scoring — `need_per_token` keys on
monitor-attached gate keys (`gated_probe_keys`), while
`SteeringComposer.gated_lens_probe_keys` detects lens gates for the callback
merge. `probe_hashes`/`_probe_hash` stamp lens probes with a deterministic
identity digest (`jlens-readout-v2`, model, word, token id, layers).

**CAA live toggle.** `live_probe_scores` / `set_live_probe_scores(bool)` —
when off, `_generate_core` masks every per-token monitor consumer at the
source (trait stream, loom probe rows, live token scores, per-layer persist,
subspace-coords persist, the tap's `needs_scores`), so generations run
aggregate-only capture; probe gates still force the subset they need.
Surfaced as `POST .../instruments/geometry/live` + session info `live_probe_scores`.

**Live lens** (`enable_live_lens`/`disable_live_lens`/
`live_lens_layers`): the selected layers' `J_l` go device-resident, join the
capture-widen union in `_begin_capture` (which also forces transient — not
persistent — capture routing, and arms a bounded tail ring when no probes are
attached), and `_live_lens_readout_step` runs at the token tap post-forward —
no new forward hooks, `static_steerable` untouched — returning `(per_layer,
aggregate, token_ids)`. The tap passes those into
`TokenProbePayload.to_token_payload` (`lens=`/`lens_aggregate=`/`lens_token_ids=`),
which folds them — with the same probabilities, no second softmax — into the
5.x measurement envelope's `instruments.lens.readout` block (`build_measurements`);
`generate_stream` forwards that envelope onto `TokenEvent.measurements`. The
aggregate covers every live layer, and its
top-k width follows the generation's resolved logit-alternative
`return_top_k` (8 only when alternatives are disabled). The default layer set
is **every fitted layer**; pass an explicit `layers` list to trade coverage for
device memory. `saklas serve`
auto-enables the live lens at startup when the artifact exists (serve-side
policy; the library stays opt-in).

## loom.py

Visible token rows own the canonical `measurements` envelope (5.x — the record
that replaced the `captured` key and the six per-token scalar aliases). The
decode tap creates it for generated tokens; selective prefill capture creates
the same shape for authored tokens from the producer rows that were already
forward passed. Both stamp instrument source plus recipe steering inside the
envelope's per-family `binding` and live directly on the loom node. The
`TokenScoreDict` rows are stored opaquely (the sidecar reader validates only the
node-level `tokens`/`thinking_tokens`/`raw_token_ids` fields, not the inner
`measurements` shape), so the `TOKEN_SIDECAR_FORMAT_VERSION` did not change; a
pre-5.x row still carrying a `captured` key loads fine but reads empty in a
`measurements`-aware client (clean break — no migration). Because token rows
already ride the compressed token sidecar, explicit loom save/load preserves
these rich historical channels without a separate cache or schema path. `set_authored_token_scores` installs one authored
channel atomically and emits `LoomMutated(op="capture_authored")`; unlike the
hot-loop `append_token`, it advances the tree revision so clients receive the
updated authored node.

`LoomTree` — the engine-side conversation tree. Nodes are turns, children are
alternative continuations, the active path is the model's context.
`messages_for(leaf)` renders the path as v2 chat messages (`with_labels=True` adds
each node's `role_label` plus, when set, `thinking_text` under a `"thinking"`
key for the scene stitcher); `flat_text(leaf)` is the base-model analogue (one
continuous string, no roles). Node ids are 26-char ULIDs. Primitives: edit,
branch, navigate, delete_subtree, `regenerate`. `Recipe` is the per-node
reproducibility receipt (steering expression, sampling, thinking, seed, probe set
+ per-probe sha256); `Recipe.overlay`/`invert_steering`/`compose_modifier` back
the auto-regen modes. `LoomNode.thinking_text` is the turn's verbatim thinking
block (committed or decoded-at-finalize), plain-string, main-JSON persisted.
**Cast roster** (phase 3): `tree.cast: dict[label, CastMember]` —
`set_cast_member`/`remove_cast_member` are decoration-tier (no conflict check),
validate the label as a role slug, emit `LoomMutated(op="cast")` (no node ids —
clients refetch the roster), and ride `to_dict`/`save` as the required `cast`
key (`tree_format` is exactly 2; an empty roster emits `{}`). Tree, node, cast,
and v2 token-sidecar readers require the complete current writer shape; they do
not infer defaults from omitted legacy fields. Explicit nullable values such as
transcript-imported `raw_token_ids=None` round-trip as null. Composition at generation is the session's
`_apply_cast_defaults` — the weakest tier, before regen overrides. Per-node
token blobs live in memory during streaming; `to_dict` omits them, `save`
writes a gzip sidecar. Mutators raise `MutationDuringGenerationError` (409) on
conflict; `UnknownNodeError` (404) / `InvalidNodeOperationError` (400)
otherwise.

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

Transcript export/import for loom paths. `SAKLAS_TRANSCRIPT_VERSION = 2`
(the cast model: per-turn `speaker:` = `role_label`, `thinking:` =
`thinking_text`, top-level `cast:` block). `to_yaml`/`from_yaml` require that
exact version. Import modes: **default** (new top-level
branch), **here** (child of active), **merge** (attach the non-matching tail at
the deepest matching prefix). Import re-attaches a recipe-bearing turn as a
*generated* node in its recorded seat (`begin_assistant(seat=turn.role)`) and
merges the cast block into the tree roster (session's member wins on conflict,
`cast_conflict` guard note). Guards: model mismatch refuses `--merge`
(`TranscriptModelMismatch`), system-prompt mismatch banners, probe-hash drift
warns / raises `TranscriptProbeDriftError` under `--strict`.

## joint_logprobs.py

Cross-branch joint-logprob computation for the loom NodeCompareDrawer.
`compute_joint_logprobs(session, a_id, b_id)` reconstructs the shared prefix
(`_shared_prefix_len`, not re-tokenization), force-replays each branch under its
recipe, and assembles per-aligned-position rows with both branches' chosen-token
logprobs + the cross-branch evaluation (`_compute_rows`, alignment via
`per_token_diff`). `approx_kl` is top-K-truncated `KL(P_A‖P_B)`. Cached on
`session._joint_logprob_cache`; held under the session lock.

## scoring.py

Restricted-choice completion scoring — the logit read of a template (the
counterpart to a manifold fit). `score_choices(session, messages, choices, *,
assistant_prefix="", labels=None, steering=None)` scores each candidate
completion's conditional logprob given the conversation context and returns a
`ChoiceScores` set — per candidate `{n_tokens, sum_logprob, mean_logprob}` plus the
restricted-choice softmax (`prob_sum` over the joint logprobs, `prob_mean` over the
length-normalized ones; both reported, neither silently chosen). Scoring is against
the **raw** model distribution (plain `log_softmax`, temperature 1, no top-k/p), so
the probabilities are the model's beliefs, not a sampler reshaping. One batched
teacher-forced forward per `_SCORE_BATCH` (=16) chunk — vocab is ~256k, so an
unbounded batch would blow memory; `logsumexp` + a gather avoid materializing a
second vocab-sized tensor. The completion span per choice is recovered with
`_shared_prefix_len` (reused from `joint_logprobs.py`), which absorbs the
boundary-token merge. `steering=` wraps the forward in `session.steering(...)` — the
distributional before/after read. `score_template(session, template, *, steering)`
runs it over a `TemplateFolder.score_inputs()` and returns one `ChoiceScores` per
context. Surfaced as `session.score_choices` / `session.score_template`.

## results.py

`GenerationResult`, `RunSet`, `TokenAlt`, `TokenEvent`, `ProbeReadings`,
`ProbeReading`, `ResultCollector`. `RunSet` is the
list-like multi-run shape (`node_ids`/`grid`/`.first`/`.to_collector()`/
`.to_dataframe()`). `TokenEvent` carries `thinking`, `logprob`, `top_alts`,
`finish_reason`, `perplexity`, `probe_readings` (per-probe `ProbeReading`s — the
compat channel feeding the OpenAI/Ollama `x-saklas-probe-readings` vendor
extension and the traits SSE, merged across families), and `measurements` (the
5.x envelope — geometry/lens/sae `instruments` + flat `scores`/`per_layer_scores`
views, `build_measurements`; replaced the former `lens_readout`/`lens_aggregate`/
`sae_readout` fields). `GenerationResult`
carries `prompt_tokens`, `finish_reason`, optional `logprobs`,
`probe_readings`, and `applied_steering` (the
canonical expression, round-trips through `parse_expr`). `ProbeReading`
(`coords`/`fraction`/`nearest`/`residual` + `assignment`/`membership` +
`fraction_per_layer`/`coords_per_layer`/`residual_per_layer` +
`depth_com`/`depth_spread`) is the **single**
reading shape for both the live per-token stream and the end-of-gen aggregate (the
aggregate is the reading pooled at the last-content token). `assignment`
(`[(label, prob)]` soft node posterior) and `membership` (tube-fit density, default
`1.0`) are the fuzzy-manifold readout — empty/`1.0` defaults keep serialization
back-compat. `depth_com`/`depth_spread` are the per-axis depth center of mass
(+ std) of the per-layer coordinate trace — mass `share_weight_L · |coord_L|`,
depths `layer/(n_layers−1)` — computed at both monitor assembly sites
(`monitor.py::_depth_stats`); empty when the reading has no per-layer trace
(lean modes) or the `Monitor` wasn't given `n_layers`. Every field is populated for flat and curved fits alike;
`residual` is `0` for a flat fit (the surface fills its subspace) and the
normalized off-surface distance for a curved fit. `ProbeReadings` remains an
explicit multi-run summary shape for notebook plots; it is not embedded in a
generation result. `to_dict()` omits `hidden_states`.

## histogram.py

`HIST_BUCKETS = 16`; `bucketize(norms, buckets)` collapses sorted per-layer norms
into evenly-sized groups. Used by CLI `manifold why` and notebook plots.

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
`.projected_away`; it also owns `save_profile`/`load_profile` for the safetensors
sidecar format. Empty layer intersection raises `ProfileError`.

`steering_expr.py` hosts the unified grammar (`parse_expr(text, *, namespace=None)`
→ `Steering`; `format_expr` round-trips; `referenced_selectors` for install-time
checks). Term markers: `ProjectedTerm(coeff, trigger, operator, base, onto)`
(materialized into derived profiles before the hook layer), `AblationTerm` (`!atom`,
default coeff 1.0, doesn't compose with `~`/`|` — lowered through
`synthesize_subspace`'s ablation path at dispatch), `ManifoldTerm` (`along`,
`onto`, position; `_expand_along_onto_coeffs` yields a 1- or 2-tuple). Probe gates
(`@when:<probe><op><threshold>`) accept coordinate axes
(`confident.uncertain`, `personas[3]`), manifold fraction
(`emotions:fraction`), label distance (`emotions@happy`), soft assignment
(`personas~hacker`), tube membership (`emotions:membership`), and SAE/J-lens
channels — all stored verbatim in `ProbeGate.probe` so the runtime gate is
identical; the parser is the only place the discrimination lives.
