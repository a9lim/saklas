# io/

Persistence + distribution: pack + manifold format, HF hub, GGUF, cloning,
alignment, and the path/selector/cache plumbing the rest of saklas runs on.
Vectors live under `vectors/`, manifolds under `manifolds/` — separate roots.

## paths.py

Every `~/.saklas/` path resolves through `saklas_home()` (honors `$SAKLAS_HOME`).
Helpers: `vectors_dir`, `models_dir`, `manifolds_dir`, `concept_dir(ns, name)`,
`manifold_dir(ns, name)`, `model_dir(id)`, `neutral_statements_path`,
`safe_model_id` (`/` → `__`), `ensure_within(root, *parts)` (path-traversal
barrier).

Owns the tensor-filename variant scheme. A folder can hold several tensors per
model, distinguished by filename suffix — exactly one *kind* per file:

- `<safe_model>.safetensors` — raw DiM (canonical)
- `<safe_model>_sae-<release>.safetensors` — DiM in SAE feature space
- `<safe_model>_from-<safe_src>.safetensors` — cross-model transfer
- `<safe_model>_role-<slug>.safetensors` — role-augmented extraction

`tensor_filename(model_id, *, release=None, transferred_from=None, role=None)` +
`sidecar_filename(...)` construct (the three kind kwargs are mutually exclusive);
`parse_tensor_filename(name) → (safe_model, variant)` inverts, variant ∈ `None` /
`sae-<release>` / `from-<safe_src>` / `role-<name>`. Separators
`_VARIANT_SEP_SAE`/`_FROM`/`_ROLE`. There is **no `pca` variant and no method
suffix** — difference-of-means is the only vector extraction method.

## packs.py

`PackMetadata` + `Sidecar` + `ConceptFolder`. `PACK_FORMAT_VERSION = 3` (4.0: a
vector *is* a 2-node `pca` manifold, so v2 `vectors/` packs are legacy —
`scripts/upgrade_packs.py` ports statements-bearing folders to `manifolds/` and
re-stamps tensor-only ones; `_all_concepts` silently skips un-migrated v2);
`PackMetadata.load` raises `PackFormatError` on a stale *or* newer `format_version`
(symmetric ceiling). `NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$`. Required pack.json
fields: `name`, `description`, `version`, `license`, `tags`, `recommended_alpha`,
`source`, `files`.

`ConceptFolder.load` verifies every file in `pack.json.files` against disk
(`verify_integrity`, with a `(size, mtime_ns)` fingerprint cache), requires at
least one of statements.json / a tensor, and demands a sidecar beside every
`.safetensors`. It globs `*.safetensors` + `*.gguf`; safetensors wins on a
same-stem conflict.

`Sidecar` carries `method` / `saklas_version` / `statements_sha256` plus optional
`components` (merge provenance), `diagnostics_by_layer` (`vector why`), the
transfer fields (`source_model_id` / `alignment_map_hash` /
`transfer_quality_estimate`), and `role` (set on `_role-<name>` tensors). `method`
is free-form; production writes `difference_of_means`, `dim_sae`,
`procrustes_transfer`, `merge`, `imported`, `gguf_import` (and the cache sidecars'
`layer_means`/`neutral_activations`/`procrustes_alignment`). No code branches on
it.

Helpers shared with `hf.py`/`session.py`: `synthesize_pack_metadata`,
`hash_folder_files`/`hash_file`, `enumerate_variants(folder, model_id) →
{raw | sae-<release> | from-<safe_src> | role-<name>: path}`, `is_stale` /
`version_mismatch` / `merge_components_status`, `bundled_concept_names`.
`materialize_bundled()` copies bundled package data into `vectors/default/<concept>/`
(copy-on-miss; re-copies a stale `pack.json` with a `.bak`; overwrites
`statements.json` only when the user's copy hashes equal to the bundled one).

## manifolds.py

On-disk format for manifold artifacts — `~/.saklas/manifolds/<ns>/<name>/`, a peer
of `vectors/` (N labeled nodes on a domain, not a single bipolar concept).
`MANIFOLD_FORMAT_VERSION = 5` (decoupled from `PACK_FORMAT_VERSION`); rejects pre-v3
and newer-than-local. v4 added `explained_variance_per_layer`, v5 added
`origin_per_layer`; the fitted-tensor sidecar also carries
`mahalanobis_share_per_layer` + `share_metric` + `subspace_metric` (all written by
`core/manifold.save_manifold`; there is **no `lever_per_layer`** — the lever is
gone). `min_nodes(n) = 2n+1`. Two folder shapes share the format, discriminated by
`manifold.json::fit_mode`:

- `"authored"` — user supplies `domain` + per-node `{label, coords}`.
- `"pca"` / `"spectral"` — discover mode: nodes carry `{label}` only, coords are
  derived per-model at fit time and stored in the safetensors; `hyperparams` feeds
  the picker.

`ManifoldFolder.load` validates the format version, the `NAME_REGEX` and per-node
`_LABEL_REGEX = ^[a-z][a-z0-9_-]{0,63}$` (labels drop `.` — reserved as the
bipolar separator and the `%label` lexer would mis-read it), branches on
`fit_mode`, enforces `min_nodes` on authored folders (discover at fit time),
verifies `files`, and demands a sidecar per fitted tensor. `source` field
(`local`/`bundled`/`hf://...`) mirrors `PackMetadata.source`. `node_groups()` reads
`nodes/NN_<label>.json` in order; `nodes_sha256()` is the staleness key — hashes
`{corpus, domain, node_coords}` (authored) / `{corpus, fit_mode, hyperparams}`
(discover), folding in any non-`None` node role. `ManifoldSidecar` is the lean
per-tensor JSON (`method` round-trips `manifold_pca`/`manifold_sae` authored,
`manifold_discover_{pca,spectral,sae}` discover, `manifold_procrustes_transfer`
transfer); tensor save/load itself lives in `core/manifold.py`.

Authoring: `create_manifold_folder` (authored webui/HTTP path, returns
`(folder, advisories)`), `create_discover_manifold_folder`
(`_sanitize_hyperparams` drops cross-method keys at the IO boundary). Streaming
companions for big rosters (a crash keeps finished nodes):
`init_discover_manifold_folder` + `append_discover_manifold_node` +
`write_manifold_scenarios`; `plan_discover_generation → DiscoverGenerationPlan` is
the shared resume/add-nodes planner (deliberately bypasses `ManifoldFolder.load`
so it can inspect a partial). `merge_discover_manifolds` unions ≥2 discover
sources' node corpora into a fresh *unfitted* folder (authored sources / label
collisions / mixed-mode-without-override raise). `update_manifold_folder`
re-authors authored folders.

Lifecycle (addressed by `(namespace, name)`, not the concept `Selector`):
`remove_manifold_folder` (returns `rematerializes_on_restart` for `default/`/
bundled), `clear_manifold_tensors(... model_scope=None, variant="all")` (filter
raw/sae/from/all; keeps `manifold.json` + corpus), `refresh_manifold` (unscoped:
`local` → skip, `bundled` → re-materialize, `hf://` → re-pull; scoped → drop one
model's fit). `transfer_manifold(folder, *, from_model, to_model, alignment,
whitener=None, ...)` — pure-io: applies a caller-supplied per-layer Procrustes
`alignment` to the fitted subspaces (`mean → M_L mean`, `basis → basis @ M_Lᵀ`),
leaves the RBF + `node_coords` untouched, re-bakes the Mahalanobis **share** in
target space when a target whitener covers the transferred layers (no lever),
clears `origin` (per-layer source-neutral foot), carries source `subspace_metric`,
writes the `_from-<safe_src>` variant. `manifold_summary(folder)` is the
session-independent serializer shared by `manifold show -j` + the HTTP
summary route. `iter_manifold_folders`, `bundled_manifold_names`,
`materialize_bundled_manifolds`. Per-node `role` (slug `[a-z0-9._-]+`) rides
through the fit to `compute_node_centroid` for role-baselined centroids;
family-unsupported raises `RoleSubstitutionUnsupportedError` at fit time.

## selectors.py

Selector grammar shared by `cache_ops` and `core.session` (lives in `io` so
neither imports up into `cli`). `Selector(kind, value, namespace)` with kinds
`name` / `tag` / `namespace` / `model` / `all`; `parse(raw)` handles `ns/name`,
the prefixes, and a trailing `:variant` via `_VARIANT_REGEX = ^(raw | sae[-…] |
role[-…] | from[-…])$` — **no `pca`**. `resolve(selector)` walks `vectors_dir()`
into `ResolvedConcept`s (module-level cache); `resolve_pole(raw, namespace=) →
(canonical, sign, match, variant)` is the bare-pole alias pipeline.
`resolve_manifold_label(label, *, namespace=) → ResolvedManifoldLabel | None`
memoizes its own `manifolds_dir()` walk; `resolve_bare_name(raw, *, namespace=) →
(pole_hit, manifold_hit)` is the unified tier (pole first, then manifold label,
cross-tier collision raises). `invalidate()` clears both caches — mutating code
must call it. `parse_args(tokens)` splits a token list into one concept selector +
one optional `model:` scope.

## cache_ops.py

Pure data layer behind `pack install/refresh/clear/rm/ls/search/push` +
`pack export gguf`. Every function returns structured results (`ConceptRow` /
`ConceptInfo` / `PackListResult` / `HfRow`); the CLI renders.

- `install(target, as_, *, force, statements_only)` — HF coord or local folder.
- `refresh(selector, *, model_scope)` — re-pull from `pack.json.source` (scoped =
  drop the per-model tensor pair; `local` skipped; `bundled` re-copies; `hf://`
  re-pulls). `refresh_neutrals()` rewrites `neutral_statements.json`.
- `delete_tensors(selector, model_scope, *, variant="all")` — filter
  `raw`/`sae`/`from`/`all`.
- `uninstall(selector, *, yes)` — refuses broad selectors without `yes`.
- `export_gguf(...)` — single concept; refuses in-place export for bundled.
- `push(selector, ...)` — single concept (default `variant="raw"`); delegates to
  `hf.push_pack`.
- `list_concepts` / `pack_info` / `search_remote_packs` — HF failures land in the
  result, never raise.

## hf.py / hf_manifolds.py

HF distribution as **model** repos (`repo_type="model"`; safetensors is hub-native,
`base_model` frontmatter gives reverse-link discoverability). `split_revision`
parses `owner/name@rev`. `stage_verify_swap` (`staging.py`) owns the shared
stage-verify-swap choreography for HF installs: build under `.staging/`, recover
`.bak` on interrupted swaps, then promote atomically with best-effort restore.
`pull_pack` supplies pack-specific staging/validation; native if a `pack.json` is
present, else `_install_synthesized_pack` (scans tensors, writes
`method="imported"` sidecars — repeng-style GGUF-only repos install with zero
prep). `push_pack(..., variant="all")` stages a filtered copy + one
`upload_folder`; the model card carries `library_name: saklas`, `saklas-pack`
tags, deduped `base_model:`, and `base_model_relation: adapter`.
`resolve_target_coord` picks `<whoami>/<name>`.

`hf_manifolds.py` is the manifold counterpart (`saklas-manifold` tag). `pull_manifold`
uses the same stage-verify-swap but **rejects** a repo with no `manifold.json` (the
geometry can't be inferred from a bare tensor dump). `push_manifold(..., variant=
"raw")` always uploads the corpus (a manifold can't re-fit without it) and filters
tensors; `search_manifolds`/`fetch_manifold_info` fill the picker fields
(`domain_label`, `node_count`, `fit_mode`, `tensor_models`); `install_manifold`
orchestrates HF pulls + local copies.

## gguf_io.py

`write_gguf_profile(profile, path, *, model_hint)` + `read_gguf_profile(path)`,
matching llama.cpp's control-vector convention (`general.architecture =
"controlvector"`, `direction.<idx>` fp32). Lazy `gguf` import raises
`GGUFNotInstalled`. Because shares are baked into the tensor magnitudes,
llama.cpp's uniform `--control-vector-scaled` scalar reproduces saklas's per-layer
weighting with no per-layer metadata.

## probes_bootstrap.py

`load_defaults()` walks `vectors/default/` into `{tag: [concept]}`.
`bootstrap_layer_means(...)` loads/computes the per-layer probe-centering means.
`bootstrap_probes(model, tokenizer, layers, model_info, categories, *,
whitener=None, layer_means=None, dls=True)` loads cached probe tensors (raising
`StaleSidecarError` on `statements.json` drift unless `SAKLAS_ALLOW_STALE=1`) and
extracts the rest — **difference-of-means only** (no `method=` param). A `whitener`
enables Mahalanobis bake scoring (sidecar `bake: "mahalanobis"`, else
`"euclidean"`); `layer_means` + `dls` feed the DLS mask.

## cloning.py

Training-free persona cloning. `clone_from_corpus(session, path, name, *,
n_pairs=90, seed=None, batch_size=5, force=False)` reads one-utterance-per-line,
samples exemplars, pairs each against a model-generated *neutralized* rewrite, then
delegates extraction + save to `session.extract(DataSource(pairs=...))`. Lands in
`local/<name>/` with `tags += ["cloned"]` and `corpus_sha256`/`n_pairs`/
`batch_size`/`seed` provenance. Errors: `CorpusTooShortError` / `CorpusTooLongError`
/ `InsufficientPairsError`.

## merge.py

Offline direction merging into a corpus-less `fit_mode="baked"` manifold.
`merge_into_manifold(name, expression, model, *, force, strict)` reads each
component (a fitted 2-node `pca` manifold folded down, or a legacy vector pack),
linearly combines the per-layer directions, folds the result to a one-pole ray
(`fold_directions_to_subspace`), and writes a baked manifold to
`manifolds/local/<name>/` — one fitted tensor per shared model, all sharing one
`manifold.json`. The shared grammar's `|` (project-away) is accepted, triggers +
bare un-namespaced poles rejected. `project_away` / `linear_sum` / `shared_models`;
the baked sidecar carries `method="merge"` + per-component `components` provenance.

## alignment.py

Cross-model probe alignment via per-layer Procrustes.
`load_or_compute_neutral_activations(...)` is the disk-cached per-model neutrals
(90 prompts × layers, **fp32** — the project-wide invariant; self-heals legacy
bf16/fp16/non-finite caches). These are what the Mahalanobis whitener builds its
covariance from. `fit_alignment(src, tgt, *, min_shared_layers=10) → {layer: M_L}`
(orthogonal Procrustes for matched dim, rectangular least-squares otherwise; both
center first); `alignment_quality` is per-layer R². `transfer_profile(profile,
alignment_map, *, source_model_id, ..., whitener=None)` applies `M_L @ v_src` per
layer (uncovered layers dropped, `method="procrustes_transfer"`), re-scaling each
magnitude to its *target* Mahalanobis norm when a target whitener covers the
layers. The fitted map round-trips under the *target* model dir
(`models/<safe_tgt>/alignments/<safe_src>.…`).

## atomic.py / datasource.py

`atomic.py` — `write_bytes_atomic` / `write_json_atomic`: stage to `<path>.tmp` in
the same directory, `fsync`, then `os.replace`. `datasource.py` — `DataSource`
normalizes contrastive pairs from raw lists, JSON, CSV, HF datasets, or curated
bundled concepts (`DataSource.curated(concept)` triggers `materialize_bundled` and
reads `vectors/default/<concept>/statements.json`).
