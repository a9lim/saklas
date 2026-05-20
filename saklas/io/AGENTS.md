# io/

Persistence + distribution: pack format, HF hub, GGUF, cloning, alignment,
and the path/selector/cache plumbing the rest of saklas runs on.

## paths.py

Every `~/.saklas/` path resolves through `saklas_home()` (honors `$SAKLAS_HOME`).
Helpers: `vectors_dir`, `models_dir`, `neutral_statements_path`, `concept_dir(ns, name)`,
`model_dir(model_id)`, `safe_model_id` (`/` → `__`).

Owns the tensor-filename variant scheme. A concept folder can hold multiple
baked tensors per model, distinguished by filename suffix:

- `<safe_model>.safetensors` — raw DiM (canonical default)
- `<safe_model>_pca.safetensors` — legacy raw PCA
- `<safe_model>_sae-<release>.safetensors` — DiM in SAE feature space
- `<safe_model>_sae-<release>_pca.safetensors` — PCA in SAE feature space
- `<safe_model>_from-<safe_src>.safetensors` — cross-model transfer

`tensor_filename(model_id, *, release=None, transferred_from=None, method="dim")`
and `sidecar_filename(...)` construct; `parse_tensor_filename(name)` inverts,
returning `(safe_model, variant)` where `variant` is `None` (canonical DiM),
`"pca"`, `"sae-<release>"`, `"sae-<release>-pca"`, or `"from-<safe_src>"`. The
`_sae-`/`_from-` literals are kind separators; `_pca` is a method suffix applied
last and stripped first on parse. `release` and `transferred_from` are mutually
exclusive; `transferred_from` rejects `method="pca"` (transfers preserve the
source method). `_KNOWN_METHODS = {"dim", "pca"}`.

## packs.py

`PackMetadata` + `Sidecar` + `ConceptFolder`. `PACK_FORMAT_VERSION = 2`;
`PackMetadata.load` raises `PackFormatError` on a stale `format_version` (with a
`scripts/upgrade_packs.py` hint) and on a newer-than-local one. `NAME_REGEX =
^[a-z][a-z0-9._-]{0,63}$`. Required pack.json fields: `name`, `description`,
`version`, `license`, `tags`, `recommended_alpha`, `source`, `files`.

`ConceptFolder.load` verifies every file in `pack.json.files` against disk
(`verify_integrity`, with an in-process `(size, mtime_ns)` fingerprint cache),
requires at least one of statements.json / a tensor, and demands a sidecar
beside every `.safetensors`. It globs `*.safetensors` + `*.gguf`; safetensors
wins on a same-stem conflict (native, carries the sidecar). `sidecar(sid)`
raises `KeyError` for GGUF-only entries.

`Sidecar` carries `method` / `saklas_version` / `statements_sha256` plus
optional `components` (merge provenance), `diagnostics_by_layer` (`vector why`),
and transfer fields `source_model_id` / `alignment_map_hash` /
`transfer_quality_estimate`. `method` round-trips `difference_of_means` /
`dim_sae` / `contrastive_pca` / `pca_center_sae` / `procrustes_transfer` /
`merge` / `imported`; no production code branches on it.

Helpers shared with `hf.py` and `session.py`: `synthesize_pack_metadata(...)`
builds a `PackMetadata` with `files` hashed from on-disk contents;
`hash_folder_files` / `hash_file` for hashing; `enumerate_variants(folder,
model_id)` returns `{variant_key: path}` keyed `raw` / `pca` /
`sae-<release>` / `sae-<release>-pca` / `from-<safe_src>`. `is_stale` /
`version_mismatch` / `merge_components_status` / `merge_components_stale` are
staleness checks.

`materialize_bundled()` copies bundled package data into `~/.saklas/`:
`neutral_statements.json` and each `saklas/data/vectors/<concept>/` →
`vectors/default/<concept>/`. Copy-on-miss for fresh installs. On a stale
`format_version` it re-copies the shipped `pack.json` in place (writing a
`.bak`), and overwrites `statements.json` only when the user's copy hashes
equal to the bundled one (canonical-JSON comparison) — a user-edited statements
file is preserved with an INFO log. Per-model tensors stay put.

## atomic.py

`write_bytes_atomic` / `write_json_atomic`: stage to `<path>.tmp` in the same
directory, `fsync`, then `os.replace`. Same-dir staging is required for atomic
replace. A crash leaves an orphan `.tmp` outside the manifest — harmless.

## selectors.py

Selector grammar shared by `cache_ops` and `core.session` (lives in `io` so
neither imports up into `cli`). `Selector(kind, value, namespace)` with kinds
`name` / `tag` / `namespace` / `model` / `all`; `default` aliases to
`namespace/default`. `parse(raw)` handles `ns/name`, `tag:`/`namespace:`/`model:`
prefixes, and a trailing `:variant` (`raw` | `pca` | `sae[-<release>]`, via
`_VARIANT_REGEX`). `resolve(selector)` walks `vectors_dir()` into
`ResolvedConcept`s through a module-level cache — mutating code must call
`invalidate()`.

`resolve_pole(raw, namespace=None) -> (canonical, sign, match, variant)` is the
pole-alias pipeline: a bare pole on either side of an installed bipolar concept
resolves to the full composite with `sign` ±1 (caller multiplies the user
alpha). Cross-namespace or cross-canonical collisions raise
`AmbiguousSelectorError`. `parse_args(tokens)` splits a token list into one
concept selector + one optional `model:` scope.

## cache_ops.py

Pure data layer behind `pack install/refresh/clear/rm/ls/search/push` and
`pack export gguf`. Every function returns structured results (`ConceptRow` /
`ConceptInfo` / `PackListResult` / `HfRow`); the CLI does rendering.

- `install(target, as_, *, force, statements_only)` — HF coord (`ns/name[@rev]`)
  or local folder; `install_folder` for the copy path. `statements_only` strips
  tensors after install.
- `refresh(selector, *, model_scope)` — re-pull from `pack.json.source`. Scoped
  refresh deletes just the per-model tensor pair (re-extracts on next use);
  `source=local` is silently skipped; `bundled` re-copies from package data;
  `hf://` re-pulls. `refresh_neutrals()` rewrites `neutral_statements.json`.
- `delete_tensors(selector, model_scope, *, variant="all")` — variant filter
  `raw` / `sae` / `all`.
- `uninstall(selector, *, yes)` — removes the whole folder; refuses broad
  selectors (`all`, bare `namespace:`) without `yes=True`. Bundled concepts
  re-materialize on next session init.
- `export_gguf(selector, *, model_scope, output, model_hint)` — single concept
  only. Refuses in-place export for bundled concepts (their folder is restored
  on refresh — the GGUF would vanish); pass `--output` outside the pack folder.
  `_resolve_model_hint` derives `controlvector.model_hint` from
  `transformers.AutoConfig.model_type`.
- `push(selector, ...)` — single concept only; refuses `source=bundled`/`hf://`
  without `--as` or `--force`. Rehashes disk state, then delegates to
  `hf.push_pack`.
- `list_concepts` / `pack_info` / `search_remote_packs` — local + HF merged
  listings; HF failures land in the result, never raise.

## hf.py

HF distribution. Packs are **model** repos (`repo_type="model"` exclusively) —
safetensors is model-hub-native and `base_model` frontmatter gives reverse-link
discoverability. `split_revision` parses `owner/name@rev` into `(coord, rev)`;
any git ref works. `_download` upgrades the error message when the user points
at a dataset repo.

`pull_pack(coord, target_folder, *, force, revision)` uses stage-verify-swap: the
pack is built under `<target>.staging/`, integrity-verified there, then
atomically swapped (`target → .bak`, `staging → target`, rmtree `.bak`); a crash
mid-swap is recoverable from `.bak`. If the repo has a `pack.json` it installs
as a native pack with `source` rewritten to the `hf://` coord. If not,
`_install_synthesized_pack` fabricates one: scans `*.safetensors`/`*.gguf`,
writes `method="imported"` sidecars for bare safetensors, slugs a name from the
repo via `NAME_REGEX`. Repeng-style GGUF-only repos install with zero prep.

`push_pack(folder, coord, *, private, include_statements, include_tensors,
model_scope, tag_version, dry_run, variant="all")` stages a filtered copy
(README + `.gitattributes` + filtered pack.json), then one `upload_folder`.
`variant` filters tensors `raw` / `sae` / `from` / `all`. Model card carries
`library_name: saklas`, merged tags (`saklas-pack`, `activation-steering`,
`steering-vector`, + pack tags), a deduped `base_model:` list, and
`base_model_relation: adapter`. `resolve_target_coord` picks `<whoami>/<name>`
unless `--as` overrides. `search_packs` / `fetch_info` query the hub without a
full download.

## gguf_io.py

`write_gguf_profile(profile, path, *, model_hint)` + `read_gguf_profile(path)`,
matching llama.cpp's control-vector convention: `general.architecture =
"controlvector"`, `controlvector.model_hint`, `controlvector.layer_count`,
`direction.<layer_idx>` tensors as fp32. Lazy `gguf` import raises
`GGUFNotInstalled` (ImportError subclass) with an install hint;
`read_gguf_profile` returns `(profile, {method: "gguf_import", ...})`.

Because shares are baked into the tensor magnitudes at extraction, llama.cpp's
uniform `--control-vector-scaled` scalar reproduces saklas's per-layer weighting
with no per-layer metadata. Repeng unit-normed GGUFs round-trip too (uniform
injection — the semantic they were exported with).

## probes_bootstrap.py

`load_defaults()` runs `materialize_bundled()`, then walks `vectors/default/`
into `{tag: [concept_name, ...]}`. `bootstrap_layer_means(...)` loads or computes
per-layer mean activations for probe centering at
`models/<id>/layer_means.{safetensors,json}`; stale when `neutral_statements.json`
changes.

`bootstrap_probes(..., *, method="dim", whitener=None, layer_means=None,
dls=True)`: loads cached probe tensors (raising `StaleSidecarError` when
`statements.json` changed since extraction, unless `SAKLAS_ALLOW_STALE=1`) and
extracts the rest. `method` selects DiM (`extract_difference_of_means`) vs PCA
(`extract_contrastive`). `whitener` (a `LayerWhitener`) enables Mahalanobis-
flavored share scoring for DiM — sidecars then carry `bake: "mahalanobis"`,
else `"euclidean"`; PCA ignores it. `layer_means` + `dls` feed the
discriminative-layer-selection mask. MPS cache flushed between probes.

## cloning.py

Training-free persona cloning. `clone_from_corpus(session, path, name, *,
n_pairs=90, seed=None, batch_size=5, force=False)` reads a one-utterance-per-line
text file, samples `n_pairs` exemplars, and pairs each against a model-generated
*neutralized* rewrite (generated in batches of `batch_size`). It owns corpus
hashing, exemplar sampling, rewrite batching, fit-checking, and the cache
short-circuit, then delegates extraction + save to `session.extract(DataSource(
pairs=...))`. Result lands in `local/<name>/`; final `pack.json` carries
`corpus_sha256` / `n_pairs` / `batch_size` / `seed` and `tags += ["cloned"]`.

Cache key: `sha256(corpus) + n_pairs + batch_size + seed`, compared against the
existing `pack.json`; `force=True` bypasses. Errors: `CorpusTooShortError` (<10
usable lines), `CorpusTooLongError` (batch + budget overflows context),
`InsufficientPairsError` (too few pairs survived parsing).

## merge.py

Offline vector merging into a distributable single-vector pack.
`merge_into_pack(name, expression, model, *, force, strict)` writes a
tensors-only pack to `local/<name>/`. `expression` uses the shared steering
grammar from `core.steering_expr` (`+`/`-`/coefficient/`|`); merge accepts only
`|` for project-away and rejects triggers and bare un-namespaced poles.
`project_away` removes one direction per layer; `linear_sum` sums components
over the layer intersection (`strict=True` errors on dropped layers).
`shared_models` returns models every term has a tensor for. Saved sidecars carry
`method="merge"` and per-component `components` provenance.

## alignment.py

Cross-model probe alignment via per-layer Procrustes.
`load_or_compute_neutral_activations(...)` is the disk-cached per-model neutrals
at `models/<id>/neutral_activations.{safetensors,json}` — 90 prompts × layers,
stored fp16, promoted to fp32 on load (Procrustes wants fp32 SVD). Same hash
check as `layer_means` decides staleness.

`fit_alignment(src_acts, tgt_acts, *, min_shared_layers=10) -> {layer: M_L}`
fits `M_L : ℝ^D_src → ℝ^D_tgt` per shared layer — orthogonal Procrustes (SVD)
for matched dim, rectangular least-squares for mismatched. Both center first.
`AlignmentError` below `min_shared_layers`. `alignment_quality` is per-layer R²;
its median becomes `Sidecar.transfer_quality_estimate`.

`transfer_profile(profile, alignment_map, *, source_model_id,
transfer_quality_estimate=None)` applies `M_L @ v_src` per layer (uncovered
layers dropped), tagging the result `method="procrustes_transfer"`.
`alignment_cache_path` / `save_alignment_map` / `load_alignment_map` round-trip
the fitted map under `models/<safe_tgt>/alignments/<safe_src>.{safetensors,json}`
— under the *target* dir so deleting a target wipes its alignments. Transferred
profiles land at the target's `_from-<safe_src>` tensor path.

## datasource.py

`DataSource` normalizes contrastive pairs from raw lists, JSON, CSV, HF
datasets, or curated bundled concepts (`DataSource.curated(concept)` triggers
`materialize_bundled` and reads `vectors/default/<concept>/statements.json`).
