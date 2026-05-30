# io/

Persistence + distribution: pack format, HF hub, GGUF, cloning, alignment,
and the path/selector/cache plumbing the rest of saklas runs on.

## paths.py

Every `~/.saklas/` path resolves through `saklas_home()` (honors `$SAKLAS_HOME`).
Helpers: `vectors_dir`, `models_dir`, `neutral_statements_path`, `concept_dir(ns, name)`,
`model_dir(model_id)`, `manifolds_dir`, `manifold_dir(ns, name)`, `safe_model_id` (`/` → `__`).

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

## manifolds.py

On-disk format for manifold-steering artifacts — `~/.saklas/manifolds/<ns>/<name>/`, a peer of `vectors/` (a manifold is N labeled nodes on a domain, not a single bipolar concept, so not a `ConceptFolder`). `MANIFOLD_FORMAT_VERSION = 4` is decoupled from concept packs' `PACK_FORMAT_VERSION`; pre-v3 (1-D cyclic-spline) manifolds are rejected with a pointer to `scripts/upgrade_manifolds.py`. v4 adds the optional per-layer `explained_variance_per_layer: {str(idx): float}` sidecar field used by additive-mode manifold steering's fit-quality normalization — bumping the version triggers `materialize_bundled_manifolds` to refresh the bundled manifest on next session start, and the EV field gets populated on the next fit; v3 sidecars load fine with empty EV, just without the quality-normalization boost. Two folder shapes share the format, discriminated by `manifold.json::fit_mode`:

- `"authored"` (default, historical) — user supplies `domain` + per-node `{label, coords}`; `domain_from_spec` builds the geometry; soft `UserWarning`s fire for poisedness deficiency and near-flat non-periodic axes.
- `"pca"` / `"spectral"` — discover mode. `domain` is absent and nodes carry `{label}` only — coords are derived per-model at fit time by `core/manifold.discover_coords`, wrapped in `CustomDomain(k)` and stored in the per-model safetensors. `hyperparams` on the folder feeds the picker.

`ManifoldFolder.load` validates format version (the manifold *name* against `NAME_REGEX`, each node *label* against the stricter `_LABEL_REGEX = ^[a-z][a-z0-9_-]{0,63}$` — labels drop `.` because the dot is reserved as the bipolar-pole separator and the steering-expr lexer addresses a label via `%label`, so a dotted label is neither typable nor resolvable; the error names it a "grammar-addressable identifier"), branches on `fit_mode`, enforces `min_nodes(n) = 2n+1` on authored folders (discover folders enforce it at fit time, once `k` is picked), verifies the `files` manifest when populated, and exposes the `is_discover` discriminator for downstream branching. The four label-validation sites (authored-load, discover-load, `_validate_authored_nodes`, `_validate_discover_corpora`) all share `_LABEL_REGEX`. `ManifoldFolder` carries a `source` field (`"local"` default / `"bundled"` / `"hf://<owner>/<name>[@rev]"`) mirroring `PackMetadata.source` — stamped by `pull_manifold`, preserved across re-fits by `write_metadata` (omitted from the manifest only when `"local"`, so non-HF folders stay byte-identical to the pre-source shape), and read by `refresh_manifold` to pick the re-pull path. `node_groups()` reads `nodes/NN_<label>.json` (one JSON statement list per node) in order; `nodes_sha256()` is the per-tensor staleness key — for authored folders it hashes `{corpus, domain, node_coords}`; for discover folders it hashes `{corpus, fit_mode, hyperparams}`. The field name is unchanged for back-compat with v3 sidecars (the spec called for a `fit_inputs_sha256` rename but the impl folds the new semantics into the existing key). `write_metadata` re-hashes and rewrites `manifold.json` after a fit. The `files` manifest covers only fitted tensors + sidecars — the node corpus is user-editable (editing it is the re-fit trigger) so it is deliberately *not* integrity-hashed (`hash_manifold_files`). `ManifoldSidecar` is the lean per-tensor JSON (`method` / `domain` / `node_count` / `nodes_sha256` / optional SAE keys; plus `fit_mode` / `hyperparams` / `diagnostics` on discover fits) — separate from `packs.Sidecar`, whose concept-extraction fields are meaningless here. `ManifoldSidecar.method` round-trips `manifold_pca` / `manifold_sae` (authored) and `manifold_discover_pca` / `manifold_discover_spectral` / `manifold_discover_sae` (discover). Tensor save/load itself lives in `core/manifold.py` — discover fits additionally stash `node_coords` (the derived per-model layout) in the same safetensors. `ManifoldFormatError` on any malformed folder.

`iter_manifold_folders(namespace=None)` is the shared discovery walk — yields `(namespace, ManifoldFolder)` for every installed manifold, skipping malformed folders; both the CLI (`_iter_manifold_folders` is a thin wrapper) and the server share it so neither re-implements the walk. `create_manifold_folder(namespace, name, description, domain_spec, nodes)` is the authored webui/HTTP path: it validates name/namespace/labels against `NAME_REGEX`, builds the domain via `domain_from_spec`, enforces `min_nodes` and per-node coord arity + non-empty statement lists, writes `manifold.json` (with an empty `files` manifest) + the `nodes/` corpus, and returns `(folder, advisories)` — the captured `_warn_authoring_quality` warnings so the UI can flag a poisedness-deficient layout before a fit. `create_discover_manifold_folder(namespace, name, description, *, fit_mode, node_corpora, hyperparams=None, node_roles=None)` is the discover-mode authoring path: nodes are `{label: [statements]}` with no coords, `fit_mode in {"pca","spectral"}` is the discriminator, and `_sanitize_hyperparams(fit_mode, hyperparams)` drops cross-method keys (e.g. `k_nn` on a PCA folder) at the IO boundary so a stray hyperparam never lands in the manifest and crashes the dispatcher at fit time. `update_manifold_folder(folder, *, description=None, nodes=None)` re-authors authored folders in place — a `nodes` argument fully replaces the node list and corpus; existing fitted tensors are kept (they go stale, the next fit overwrites). `FileExistsError` on an existing folder, `ManifoldFormatError` on bad input.

Lifecycle (the manifold analogue of pack rm / clear / refresh in `cache_ops`; addressed by `(namespace, name)`, not the concept `Selector`/`resolve` machinery): `remove_manifold_folder(namespace, name)` rmtrees the folder and returns `{namespace, name, source, removed, rematerializes_on_restart}` (the flag is true for `default/`-namespace or `source=="bundled"` folders, which respawn on next session init via `materialize_bundled_manifolds`). `clear_manifold_tensors(namespace, name, model_scope=None, *, variant="all")` deletes the per-model `<safe>*.safetensors` + sidecars (filter `raw`/`sae`/`all`, the same three `delete_tensors` accepts; `model_scope` is a safe-id narrowing to one model's fit, mirroring `delete_tensors`'s positional `model_scope` — `None` clears every model) while keeping `manifold.json` + the `nodes/` corpus, then re-hashes the `files` manifest — it loads the folder *before* unlinking (a populated manifest would fail integrity on a reload after deletion) and re-hashes from disk afterward. `refresh_manifold(namespace, name, *, model_scope=None, force=True)` mirrors `cache_ops.refresh`: unscoped it dispatches on the `source` tier — `local` (or anything without an upstream) → `"skipped"`, `bundled` / `default/` → `materialize_bundled_manifolds()` (`"bundled"`), `hf://...` → re-pull via `pull_manifold` (`"hf"`); a `model_scope` arg instead does a scoped refresh — drops just that model's fit pair (delegates to `clear_manifold_tensors`) so it re-fits on next use, returning `"scoped"`. All raise `FileNotFoundError` when the manifold isn't installed. The server's inline `delete_manifold` route can refactor to `remove_manifold_folder`.

`transfer_manifold(folder, *, from_model, to_model, alignment, transfer_quality_estimate=None, force=False)` is the manifold analogue of `vector transfer` — pure-io: the caller supplies the per-layer Procrustes `alignment` map (`{layer: M_L}`, the shape `alignment.fit_alignment` produces; building it needs both models loaded, a session/CLI concern), and this function only *applies* it. It loads the source fit at `<folder>/<safe_from>.safetensors`, maps each covered layer's affine subspace (`mean → M_L @ mean`, `basis → basis @ M_L^T` — both live in model space) into target space, leaves the RBF interpolant + `node_coords` untouched (they're subspace/authoring-coordinate-space, invariant under the model-space map), drops uncovered layers (mirrors `transfer_profile`), and `save_manifold`s to the `_from-<safe_src>` filename variant (`paths.tensor_filename(to_model, transferred_from=from_model)`). The sidecar is patched post-save with `source_model_id` + `transfer_quality_estimate` (`save_manifold`'s fixed key allow-list doesn't carry them), method tagged `manifold_procrustes_transfer`, and the folder `files` manifest re-hashed. `FileNotFoundError` (no source fit), `ManifoldFormatError` (empty alignment / no covered layer), `FileExistsError` (transferred tensor exists, no `force`).

`manifold_summary(folder)` is the session-independent serializer both `vector manifold show -j` (CLI) and `GET /saklas/v1/manifolds/{ns}/{name}` (server) can share in Wave 2 — pure-io, no session needed. Returns `namespace` (from the folder's parent dir) / `name` / `description` / `source` / `fit_mode` / `is_discover` / `domain` / `domain_label` / `intrinsic_dim` / `min_nodes` / `node_count` / `node_labels` / `node_coords` / `node_roles` / `hyperparams` / `fitted_models` (safe-ids with a tensor on disk) / `tensor_variants` (`{safe_model: [variant_key, ...]}`, `raw`/`sae-<rel>`/`from-<safe_src>`). For a discover folder the on-disk geometry is empty (`domain == {}`, `intrinsic_dim == 0`, `min_nodes is None`, `node_coords == []`) — the derived per-model layout lives in the fitted safetensors, which a session-aware caller lifts separately (kept out of this pure-io read).

`merge_discover_manifolds(target_namespace, target_name, target_description, *, sources, fit_mode=None, hyperparams=None, force=False)` unions the *nodes* of two or more discover-mode source folders into a fresh discover folder — the manifold analogue of `merge_into_pack` for vectors, but on node corpora rather than steering directions. Restricted to discover sources by design: authored manifolds carry user-declared geometry that isn't mergeable without a shared coordinate system. Label collisions across sources raise `ManifoldFormatError` (refuse over silent renames — labels carry provenance the user cares about). `fit_mode` defaults to the sources' shared mode when they agree; sources with mixed modes require an explicit override. `hyperparams=None` inherits from the first source; an explicit dict replaces wholesale. `_sanitize_hyperparams` runs at the IO boundary through the inner `create_discover_manifold_folder` call. The merged folder is written *unfitted* — the natural next step is `saklas vector manifold discover <merged>` (or `POST .../fit`) to derive coords from the combined heap. `FileNotFoundError` on missing source, `FileExistsError` on destination conflict without `force=True`, `ManifoldFormatError` for authored sources / collisions / mixed-mode-without-override.

Nodes optionally carry a per-node `role` field (slug `[a-z0-9._-]+`). Authored nodes: `{label, coords, statements, role?}`. Discover nodes: `{label}` on disk plus `role?` (the `node_roles` kwarg on `create_discover_manifold_folder` is the `{label: role|None}` mapping). When set, the role rides through `ManifoldExtractionPipeline.fit` → `compute_node_centroid(role=...)` → `_encode_and_capture_all` for chat-template substitution, producing role-baselined centroids — the persona-manifold geometry. `nodes_sha256` folds in any non-`None` role so a role edit invalidates a cached fit. `ManifoldFolder.node_roles` is the index-aligned in-memory list (defaults all-`None` for legacy folders); `_node_payload_authored` / `_node_payload_discover` emit the field only when non-`None` so non-role folders are byte-identical to the pre-A-phase shape. `_validate_node_role` is the shared slug-shape gate; family-unsupported (Mistral-3 / talkie) raises `RoleSubstitutionUnsupportedError` at fit time, not at folder write time (the folder is model-agnostic).

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
`ResolvedConcept`s through a module-level cache; `resolve_manifold_label`
memoizes its own walk of `manifolds_dir()` in a peer cache. `invalidate()`
clears both — mutating code must call it.

`resolve_pole(raw, namespace=None) -> (canonical, sign, match, variant)` is the
pole-alias pipeline: a bare pole on either side of an installed bipolar concept
resolves to the full composite with `sign` ±1 (caller multiplies the user
alpha). Cross-namespace or cross-canonical collisions raise
`AmbiguousSelectorError`.

`resolve_manifold_label(label, namespace=None) -> ResolvedManifoldLabel | None`
scans a memoized index (the all-namespace `manifolds_dir()` walk, keyed on the
root like `resolve`'s concept cache; filtered in-memory by namespace + label)
for a manifold whose `node_labels` contains `label`.
Returns a `ResolvedManifoldLabel(namespace, manifold_name, label)` on a unique
hit, `None` on miss, raises `AmbiguousSelectorError` on cross-manifold
collision. `resolve_bare_name(raw, namespace=None) -> (pole_hit, manifold_hit)`
is the unified tier above both — tries pole resolution first, then
manifold-label resolution, raises on cross-tier collision (the same slug
matches both a pole and a manifold node). The grammar (`core/steering_expr`)
calls it on plain bare slugs to synthesize a label-form `ManifoldTerm` when
the slug isn't an installed pole — that's how `0.7 pirate` parses to
`0.7 local/persona%pirate`. Namespace-qualified / variant-suffixed / bipolar
(`.`-containing) atoms skip the manifold tier.

`parse_args(tokens)` splits a token list into one
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

## hf_manifolds.py

The manifold-side counterpart to `hf.py` — HF *model*-repo convention again, but
the tagging convention diverges (`saklas-manifold` instead of `saklas-pack`).
`pull_manifold(coord, target_folder, *, force, revision)` uses the same
stage-verify-swap discipline `pull_pack` does: snapshot-download into
`<target>.staging/`, validate the staged folder through `ManifoldFolder.load`
(checks format version + `NAME_REGEX`/`_LABEL_REGEX` + the `files` integrity
manifest when populated), stamp `manifold.json::source = hf://coord[@rev]` (so
`refresh_manifold` re-pulls from the same place — mirrors how `pull_pack`
rewrites `pack.json.source`, persisted across re-fits by `write_metadata`),
then atomic-rename into place. Unlike packs, a manifold repo with no
`manifold.json` is rejected — the geometry can't be inferred from a bare
safetensors dump, so refuse rather than synthesize. `_install_manifold` is the
allow-listed file copier (`manifold.json`, `scenarios.json`, `*.safetensors`,
`*.json` at root; `nodes/*.json` under `nodes/`; everything else like
`README.md`/`.gitattributes` is skipped). `search_manifolds(query)` queries
`HfApi.list_models` filtered by `saklas-manifold` (with the same legacy-arg
fallback `search_packs` uses), then calls `fetch_manifold_info` per result to
fill in the manifold-specific fields the picker needs (`domain_label`,
`node_count`, `fit_mode`, `tensor_models`). `fetch_manifold_info(coord)` is the
cheap probe: pulls only `manifold.json` plus the repo's file listing, computes
`domain_label` from the parsed domain spec (or `discover-<fit_mode>` when the
folder is discover-mode), returns row dicts ready for display. Top-level
`install_manifold(target, as_, *, force)` orchestrates HF pulls and local
folder copies — wraps `pull_manifold` for HF coords (`<ns>/<name>[@rev]`) and
delegates to `_install_local_manifold` for local folder paths. `ManifoldInstallConflict`
is the 409-shaped error mirroring `cache_ops.InstallConflict`. `push_manifold(folder,
coord, *, private=False, model_scope=None, variant="all", dry_run=False)` is the
HF upload, mirroring `hf.push_pack` exactly in shape (stage a filtered copy, add
`README.md` + `.gitattributes`, one `upload_folder`; returns `(repo_url,
commit_sha)`, sha `None` on dry-run). Divergences from the pack push: the model
card carries the `saklas-manifold` tag (not `saklas-pack`) and a manifold-shaped
body (domain / fit_mode / node labels rather than a recommended alpha); the
corpus (`manifold.json` + `nodes/*`) is *always* uploaded since a manifold can't
re-fit without it (so a corpus-only push of an unfitted manifold is valid, unlike
a pack); per-model `<safe>.safetensors` + sidecars are filtered by `model_scope`
(safe-id) and `variant` (`raw`/`sae`/`from`/`all`, same as `push_pack`). The
staged `manifold.json::files` map is patched directly to the freshly-hashed staged
file set before upload (a model/variant filter changes the file set; re-loading
the verbatim-copied manifest would fail integrity against the excluded tensors).
`_render_manifold_card` / `_manifold_sidecar_stem_to_hf_coord` /
`_manifold_variant_matches` are the helpers (parallel to `hf._render_model_card`
/ `_sidecar_stem_to_hf_coord` / `push_pack`'s `_variant_matches`).

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
