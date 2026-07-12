# io/

Persistence + distribution. Concepts and steering manifolds are the same artifact
now — labeled nodes on a domain, fit per-model — so `manifolds.py` is the on-disk
format and `hf_manifolds.py` the HF distribution path. The old pack
format/distribution surface (`PackMetadata`/`ConceptFolder`/`pull_pack`/the
`cache_ops` install layer/`datasource.py`) is gone; `packs.py` / `cache_ops.py` /
`hf.py` are thin shared-primitive remnants. Everything lives under
`~/.saklas/manifolds/`; `vectors/` is read only to port pre-4.0 packs.

## paths.py

Every `~/.saklas/` path resolves through `saklas_home()` (honors `$SAKLAS_HOME`).
Helpers: `manifolds_dir`, `manifold_dir(ns, name)`, `templates_dir`, `models_dir`, `model_dir(id)`,
`neutral_statements_path`, `baseline_prompts_path` (user override for the shared A2
baseline user prompts; falls back to bundled `saklas/data/baseline_prompts.json`),
`safe_model_id` (ordinary Hub ids preserve `/` → `__`; ambiguous/local ids use
a reversible `_z` base64url tier), `ensure_within(root, *parts)` (path-traversal
barrier). `vectors_dir` / `concept_dir` survive only for the legacy-port scan — no
current writer targets them.

`sae.py` owns the small live-runtime metadata cache under
`models/<safe>/sae/`: one release/layer identity sidecar plus the lazily
fetched per-feature Neuronpedia metadata (`<release>-features.json` —
`{id: {label, max_act}}`, where `max_act` is `maxActApprox`, the strength
unit that normalizes the readout channel to 0..1; a legacy labels-only
`-labels.json` still reads through). SAE weights remain in the Hugging Face
cache.

Owns the tensor-filename variant scheme. A manifold folder can hold several
fitted tensors per model, distinguished by filename suffix — exactly one *kind*
per file:

- `<safe_model>.safetensors` — raw DiM (canonical)
- `<safe_model>_sae-<release>.safetensors` — fit in SAE feature space
- `<safe_model>_from-<safe_src>.safetensors` — cross-model transfer
- `<safe_model>_role-<slug>.safetensors` — role-augmented (reserved: the filename
  round-trips, but `extract --role` bakes the role into the corpus and writes the
  canonical tensor, so no `_role-` file is emitted yet)

`tensor_filename(model_id, *, release=None, transferred_from=None, role=None)` +
`sidecar_filename(...)` construct (the three kind kwargs are mutually exclusive);
`parse_tensor_filename(name) → (safe_model, variant)` inverts, variant ∈ `None` /
`sae-<release>` / `from-<safe_src>` / `role-<name>`. Separators
`_VARIANT_SEP_SAE`/`_FROM`/`_ROLE`. There is **no `pca` variant and no method
suffix** — difference-of-means is the only vector extraction method.

## packs.py

Shared pack-format *primitives* only — the format/distribution surface is gone.
What remains: `NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$` (manifolds reuse it),
`hash_file` / `hash_folder_files` / `verify_integrity` (the sha256 integrity
helpers behind the neutral/layer-means/alignment caches and the manifold integrity
manifest), `PackFormatError`, and `PACK_FORMAT_VERSION = 3` — the *legacy-vector
migration sentinel*: a `vectors/` pack whose `pack.json.format_version` is below it
is legacy and ported to a 2-node `pca` manifold on touch
(`scripts/upgrade_packs.py` / `SteeringComposer.port_stale_legacy_vector`). Also stamped
onto the profile-cache sidecars `profile.save_profile` writes
(`vectors.save_profile` remains a compatibility alias).

## manifolds.py

The on-disk format for every concept + steering manifold —
`~/.saklas/manifolds/<ns>/<name>/`. `MANIFOLD_FORMAT_VERSION = 5` (decoupled from
`PACK_FORMAT_VERSION`); rejects any `format_version` below 5. `min_nodes(n) = 2n+1`
(the curved-fit poisedness floor). Five `fit_mode`s share the class, discriminated
by `manifold.json::fit_mode`:

- **`authored`** — user supplies `domain` + per-node `{label, coords}`. Curved RBF.
- **`pca`** / **`spectral`** / **`auto`** (discover) — nodes carry `{label}` only;
  coords are derived per-model at fit time and stored in the safetensors;
  `hyperparams` feeds the picker. `pca` is flat (also the 2-node vector case),
  `spectral` curved, `auto` lets `core.manifold.select_topology` pick flat-vs-
  curved (GCV) plus periodic `BoxDomain` axes (H1 persistent homology) per-model.
  All three are `is_discover`; `_HYPERPARAMS_BY_MODE` whitelists each (auto's
  union includes `smoothing` + `persistence_frac`), and `nodes_sha256` folds
  `fit_mode` + hyperparams so re-fit is deterministic.
- **`baked`** — corpus-less: a precomputed direction written by `manifold bake`.
  No node corpus, no re-fit; `BakedManifoldError` guards corpus-requiring calls.

`ManifoldFolder.load` validates the format version, `NAME_REGEX`, and per-node
`_LABEL_REGEX = ^[a-z][a-z0-9_-]{0,63}$` (labels drop `.` — reserved as the
bipolar separator and the `%label` lexer would mis-read it), branches on
`fit_mode`, enforces `min_nodes` on authored folders (discover at fit time),
verifies `files`, and demands a sidecar per fitted tensor. `source`
(`local`/`bundled`/`hf://...`) and `tags` ride `manifold.json`. Each node entry
also carries an optional `kind` ∈ {`abstract`, `concrete`, `custom`} (`_validate_node_kind`;
`custom` = a caller-supplied generation system prompt, no role swap),
parallel to the optional `role` — generation-only provenance (it selects the
generation system template + elicitation role label) that the fit never consumes;
it rides `ManifoldFolder.node_kinds` / `ManifoldSidecar.node_kinds` and is emitted
only when set by `_node_payload_authored` / `_node_payload_discover`.
`node_groups()` reads `nodes/NN_<label>.json` in order; `nodes_sha256()` is the
staleness key — hashes `{corpus, domain, node_coords}` (authored) / `{corpus,
fit_mode, hyperparams}` (discover) / a baked sentinel, folding in any non-`None`
node role and any non-`None` node kind.
`ManifoldSidecar` is the lean per-tensor JSON (`method` round-trips
`manifold_pca`/`manifold_sae` authored, `manifold_discover_{pca,spectral,sae}`
discover, `merge` baked, `manifold_procrustes_transfer` transfer + the
share/subspace metrics, fit_mode, hyperparams, diagnostics, and
`node_spread_per_layer` — the whitened between-node spread `{str(L): tr(G_L)}`,
a diagnostic concept-signal-by-layer profile, empty on pre-4.0 fits); the tensor
save/load itself lives in `core/manifold.py`. `hash_manifold_files` reuses
`packs.hash_file` for the per-file sha256 integrity manifest. After the first
manifest population, `ManifoldFolder.update_file_hashes` hashes only the tensor
and sidecar just replaced; already-verified historical variants are preserved.

A node corpus is now a list of conversational *responses* (`list[str]`) aligned to
the shared A2 baseline user prompts — `response[i]` answers `baseline_prompt[i % k]`
(`baseline_prompts_path`), so a corpus length must be a multiple of `k`. The shared
baseline prompts are global (bundled `saklas/data/baseline_prompts.json`), not
per-manifold, so the generation path no longer writes `scenarios.json` and no
longer calls `write_manifold_scenarios`. Fresh discover authoring has no
`scenarios=` parameter; only legacy vector-folder migration writes
`scenarios.json` while porting old packs, and the standalone read/write helpers
remain for compatibility tests and old folders.

Authoring: `create_manifold_folder` (authored webui/HTTP path, returns `(folder,
advisories)`), `create_discover_manifold_folder` (`sanitize_hyperparams` drops
cross-method keys at the IO boundary, gated by `_HYPERPARAMS_BY_MODE`; takes
`node_roles=` / `node_kinds=` maps),
`create_baked_manifold_folder` + `save_baked_manifold_tensor` (the `subspace
merge` target — one fitted tensor per model, all sharing one `manifold.json`).
`port_legacy_vector_folder` ports a stale `vectors/<ns>/<name>/` pack to a 2-node
`pca` discover folder (file-only — no tensors carried; they re-fit lazily).
Streaming companions for big rosters (a crash keeps finished nodes):
`init_discover_manifold_folder` (also takes `node_kinds=`) +
`append_discover_manifold_node`; `plan_discover_generation → DiscoverGenerationPlan`
(also takes `node_kinds=`) is
the shared resume/add-nodes planner (deliberately bypasses `ManifoldFolder.load` so
it can inspect a partial). `merge_discover_manifolds` unions ≥2 discover sources'
node corpora into a fresh *unfitted* folder (authored sources / label collisions /
mixed-mode-without-override raise). `update_manifold_folder` re-authors authored
folders.

Lifecycle (addressed by `(namespace, name)`, not a concept `Selector`):
`remove_manifold_folder` (returns `rematerializes_on_restart` for `default/`/
bundled), `clear_manifold_tensors(... model_scope=None, variant="all")` (filter
raw/sae/from/all; keeps `manifold.json` + corpus), `refresh_manifold` (unscoped:
`local` → skip, `bundled` → re-materialize, `hf://` → re-pull; scoped → drop one
model's fit), `transfer_manifold(folder, *, from_model, to_model, alignment,
whitener, ...)` (folder read/write orchestration only — the subspace math itself
moved to `core.manifold.transfer_manifold_subspaces`). It loads the source tensor,
hands the loaded `Manifold` + caller-supplied per-layer Procrustes `alignment` +
target whitener to the core compute (which maps `mean → M_L mean`,
`basis → basis @ M_Lᵀ`, leaves RBF + `node_coords` untouched, re-bakes the
Mahalanobis **share** in target space — target whitener **required**, `WhitenerError`
otherwise; no Euclidean rebake — and clears `origin`), then writes the
`_from-<safe_src>` variant + patches the transfer-provenance sidecar. The core
function raises a plain `ValueError` when the alignment covers no fitted layer; this
function surfaces it as `ManifoldFormatError` (the `WhitenerError`, a `ValueError`
subclass, propagates verbatim via a `SaklasError`-first `except`). The lazy core
imports are now just `load_manifold` / `save_manifold` /
`transfer_manifold_subspaces` — no `LayerSubspace` / `eval_rbf` / `subspace_share` /
`mahalanobis.WhitenerError`. `manifold_summary(folder)` is the session-independent
serializer shared by `pack show -j` + the HTTP summary route. Ordinary
discovery/summary routing (`iter_manifold_folders`, `manifold_summary`, HTTP
lookup) loads metadata with `verify_manifest=False`; install/push/fitted-tensor
use stays strict. `bundled_manifold_names`, `materialize_bundled_manifolds`
(copy-on-miss into `default/` for complete package-data folders only, plus a
re-copy when the bundled manifest hash drifts or the on-disk `format_version`
predates `MANIFOLD_FORMAT_VERSION`). Bundle-drift comparison runs on the
**`files`-stripped** canonical payload (`_manifest_content_sha256`) — the
integrity map accumulates per-model fit proofs locally
(`update_file_hashes`), so comparing it against the shipped manifest
misread every fitted bundled manifold as a bundle update and the refresh
clobbered the proofs, orphaning the tensors (the strict loader refuses a
tensor with no proof). On a genuine bundle update the refresh **carries
forward** `files` entries for still-present non-bundle-shipped artifacts
(fitted tensors + sidecars) — proofs still verify on every load, and
`nodes_sha256` staleness remains the thing that decides whether an old
fit is current. Per-node `role`
(slug `[a-z0-9._-]+`) rides the fit to `compute_node_centroid` for role-baselined
centroids; family-unsupported raises `RoleSubstitutionUnsupportedError` at fit time.

**Templated manifolds** carry a `template_ref: str | None` (a
`templates.py::TemplateFolder` selector) instead of node corpora authored by hand:
`create_manifold_from_template(ns, name, *, template_ref, fit_mode, …)` resolves the
template, expands its `values × contexts` into per-value node corpora (via
`TemplateFolder.node_corpora()`), and writes a normal discover folder that stores
both the derived corpus (`nodes/`) and the `template_ref`. The fit
(`core/extraction.py`) resolves the ref to use the template's **multi-turn
contexts** as the per-node elicitation prefixes; `nodes_sha256()` folds the resolved
template's `sha256()` so a context/value edit re-fits. The template is the authoring
source of truth; the corpus is its materialization. There is **no embedded
`template` block** — the pre-migration `{slot, values, pairs}` block, `expand_template`,
`_validate_template_block`, and `create_templated_manifold_folder` are gone.

## templates.py

The standalone templated-completion artifact — `~/.saklas/templates/<ns>/<name>/
template.json`, peer to a manifold. `TemplateFolder` = `{name, slot, values,
contexts:[TemplateContext{turns:[{role,content}], assistant}]}`,
`TEMPLATE_FORMAT_VERSION = 1`. Invariant (`_validate_body` / `_validate_context`):
the slot appears **exactly once** in each context's final `assistant` string and in
no history turn; the last history turn must be `user` (the slotted assistant turn
follows it). Derived views: `node_labels()` (slugged values), `node_corpora()`
(`{label: [slot-filled assistant per context]}` — the manifold corpus,
`corpus[i] ↔ contexts[i]`), `score_inputs()` (per-context `{messages,
assistant_prefix, suffix, choices, labels}` — the scorer feed, slot split via
`partition`). `sha256()` hashes `slot × values × contexts` (excludes
description/tags) — the staleness key folded into a referencing manifold's
`nodes_sha256`. Lifecycle: `create_template_folder` (validates whole body before
writing — no half-built folder), `resolve_template(selector)` (bare name searches
all namespaces, `AmbiguousTemplateError` on collision, `TemplateNotFoundError` on
miss), `iter_template_folders`, `remove_template_folder`. `_slug_value` mirrors the
manifold node-label slug; `_LABEL_REGEX` is redefined locally so the
`manifolds → templates` import direction stays acyclic.

**Bundled templates.** `bundled_template_names` / `materialize_bundled_templates`
mirror the manifold materializer for the template kind — copy-on-miss of
`saklas/data/templates/<name>/template.json` into `~/.saklas/templates/default/`,
re-copy on bundle drift (canonical hash or `format_version`), process-scope no-op.
**Ordering**: every bootstrap site runs this *before*
`materialize_bundled_manifolds`, because a templated bundled manifold
(`default/<name>`) resolves its `template_ref` at fit (a hard
`TemplateNotFoundError` otherwise; `nodes_sha256` degrades to the ref string). A
template-derived bundled manifold authors deterministically (model-free) — write
the template, then `create_manifold_from_template` derives the corpus — so there
is no model-generation step. No template-derived bundled manifold ships at present
(colors was the first candidate, pulled).

## selectors.py

Selector grammar shared by `core.session` and the CLI (lives in `io` so neither
imports up into `cli`). A concept *is* a manifold now: `all_concepts` walks
`manifolds_dir()` via `iter_manifold_folders`, and `ResolvedConcept.folder` is the
manifold folder. `Selector(kind, value, namespace)` with kinds `name` / `tag` /
`namespace` / `model` / `all`; `parse(raw)` handles `ns/name`, the prefixes, and a
trailing `:variant` via `_VARIANT_REGEX = ^(raw | sae[-…] | role[-…] | from[-…])$`
— **no `pca`**. `resolve(selector)` filters the memoized walk; `model:X` matches
any manifold with a fitted tensor for X.

Bare-pole resolution moved entirely to the manifold tier (a bipolar concept is a
2-node `pca` manifold). **`resolve_bare_atom(concept, *, namespace=,
typed_namespace=, variant=) → ResolvedBareAtom`** is the **single owner** of the
whole bare-atom tier ladder (ordering + cross-tier arbitration) — `core/steering_expr`
calls it once instead of hand-sequencing the tiers. It returns a tagged
`ResolvedBareAtom(kind ∈ {label, name, pole})`: (1) **label** tier — a bare
dot-free slug (`variant=="raw"`, no typed namespace) matching a node label, via
`resolve_bare_name`; (2) **name** tier — a `variant=="raw"` 2-node `pca` manifold
*name* (dotted `formal.casual` skips tier 1), via `resolve_manifold_name`,
resolving to node 0 (the `orient_to=0` + pole); (3) **pole** tier — neither
matched, so `canonicalize_atom` peels the `:variant` suffix + canonicalizes the
slug. Every bipolar pole is itself a node label, so a bare pole resolves through
tier 1 as an affine `%` push. The retired `resolve_pole` folded into
**`canonicalize_atom(raw) → (canonical, variant)`** (the pure slug + variant peel —
no `match`/`sign` slots, since the bipolar sign-flip is gone); the external
canonicalizer consumers (`SteeringComposer.resolve_pole_aliases`, `tui/app`, `cli/runners`)
call it directly. The underlying tier steps stay public:
`resolve_manifold_label(label, *, namespace=)` finds a node by label across
installed manifolds; `resolve_manifold_name(name, *, namespace=)` resolves a 2-node
`pca` manifold's *name* to node 0; `resolve_bare_name(raw, *, namespace=) →
ResolvedManifoldLabel | None` is *just* the manifold-label tier (delegates to
`resolve_manifold_label`, raising on cross-manifold collision). Three memoized walks
(`_concepts_cache`/`_manifold_labels_cache`/`_manifold_names_cache`) keyed on
`manifolds_dir()`; `invalidate()` clears all three — mutating code must call it.
`parse_args(tokens)` splits a token list into one concept selector + one optional
`model:` scope.

## cache_ops.py / hf.py

Both gutted to the surface that survives the collapse.

`cache_ops.py` is now just GGUF export: `export_gguf_manifold(ns, name, *,
model_scope, output, model_hint)` folds a fitted 2-node `pca` manifold to a single
direction (`folded_vector_directions`) and writes a llama.cpp control-vector GGUF
(one `.gguf` per model; refuses in-place export for bundled manifolds);
`_resolve_model_hint` derives `controlvector.model_hint` from the base model's
`AutoConfig.model_type`. The old pack data layer
(install/refresh/clear/ls/search/push + `ConceptRow`/`PackListResult`) is gone.

`hf.py` is the generic HF surface `hf_manifolds.py` builds on: `HFError`,
`split_revision(owner/name@rev)`, the monkeypatchable `_hf_snapshot_download` /
`_hf_hub_download` / `_hf_api` indirections, and `resolve_target_coord(name, as_)`
(`--as owner/name` wins, else `whoami()/<name>`). All pack-shaped distribution
(`pull_pack`/`push_pack`/`search_packs`/`fetch_info`) is gone.

## hf_manifolds.py

HF distribution for manifolds (`saklas-manifold` tag, `repo_type="model"` —
safetensors is hub-native, `base_model` frontmatter gives reverse-link
discoverability). `pull_manifold` uses the shared `staging.stage_verify_swap` and
**rejects** a repo with no `manifold.json` (the geometry can't be inferred from a
bare tensor dump) — but `_port_legacy_pack_dir` first salvages a legacy
`saklas-pack` repo (`pack.json` + `statements.json`) into a 2-node `pca` manifold
on install, so old vector packs still install. `push_manifold(..., variant="raw")`
always uploads the corpus (a manifold can't re-fit without it) and filters tensors;
the model card (`_render_manifold_card`) carries `library_name: saklas`,
`saklas-manifold` tags, deduped `base_model:`, `base_model_relation: adapter`.
`search_manifolds`/`fetch_manifold_info` fill the picker fields (`domain_label`,
`node_count`, `fit_mode`, `tensor_models`); `install_manifold` orchestrates HF
pulls + `_install_local_manifold` copies. `ManifoldInstallConflict` on an existing
folder without `force`.

## gguf_io.py

`write_gguf_profile(profile, path, *, model_hint)` + `read_gguf_profile(path)`,
matching llama.cpp's control-vector convention (`general.architecture =
"controlvector"`, `direction.<idx>` fp32). Lazy `gguf` import raises
`GGUFNotInstalled`. Because shares are baked into the tensor magnitudes,
llama.cpp's uniform `--control-vector-scaled` scalar reproduces saklas's per-layer
weighting with no per-layer metadata.

## probes_bootstrap.py

`load_default_manifolds()` walks `manifolds/default/` into `{tag: [manifold_name]}`
(the category-grouped probe roster; triggers `materialize_bundled_manifolds`).
`bootstrap_layer_means(...)` returns the per-layer probe-centering means as
`X.mean(0)` of the per-model neutral-activation cache
(`load_or_compute_neutral_activations`) — same corpus the whitener covariance is
built from, so there is no separate `layer_means.safetensors` cache (a cold model
pays one neutral-corpus forward loop, not two). The old `bootstrap_probes` is gone — the session
sources bundled probe directions by folding fitted 2-node manifolds
(`session._bootstrap_manifold_probes`).

## merge.py

Offline direction merging into a corpus-less `fit_mode="baked"` manifold.
`merge_into_manifold(name, expression, model, *, force, strict)` resolves each
component to a per-layer direction by folding a fitted 2-node `pca` manifold
(`folded_vector_directions`), linearly combines the directions (`linear_sum`,
over the union of their layer coverage), folds the result to a one-pole ray
(`fold_directions_to_subspace`), and writes a baked manifold to
`manifolds/local/<name>/` — one fitted tensor per shared model
(`create_baked_manifold_folder` / `save_baked_manifold_tensor`), all sharing one
`manifold.json`. Only namespace-qualified additive/subtractive scalar terms are
accepted. Triggers, `!`, `%`, multi-coefficients, and `~`/`|` are rejected;
projection is Mahalanobis-only and cannot be reproduced without a live whitener.
`shared_models(expression)` returns the models every term has a fitted tensor
for. The baked sidecar carries `method="merge"`, the unanimous component model
fingerprint, and per-component `components` provenance. Legacy projected bakes
are rejected at load and must be rebuilt.

## alignment.py

Cross-model probe alignment via per-layer Procrustes.
`load_or_compute_neutral_activations(...)` is the disk-cached per-model neutrals
(the neutral corpus × layers, **fp32** — the project-wide invariant; exact loaded
model + rendered-token + layer-schema identity, payload-digest verified; self-heals
legacy bf16/fp16/non-finite caches). These are what the Mahalanobis whitener builds its
covariance from. `validate_neutral_cache_metadata` checks the digest plus
safetensors key/shape/dtype header without paging payloads for an exact-repeat
preflight; the materializing loader additionally checks finiteness.
`fit_alignment(src, tgt, *, min_shared_layers=10) → {layer: M_L}`
(orthogonal Procrustes for matched dim, rectangular least-squares otherwise; both
center first); `alignment_quality` is per-layer R². `transfer_profile(profile,
alignment_map, *, source_model_id, ..., whitener)` applies `M_L @ v_src` per
layer (uncovered layers dropped, `method="procrustes_transfer"`), re-scaling each
magnitude to its *target* Mahalanobis norm. The target whitener is **required** and
must cover the transferred layers (`WhitenerError` otherwise; Mahalanobis-only — no
Euclidean transfer). The fitted map binds both validated neutral-cache identities,
both model fingerprints, and its own payload digest under the *target* model dir
(`models/<safe_tgt>/alignments/<safe_src>.…`). `transfer_manifold`
(`manifolds.py`) is the manifold counterpart.

## lens.py

The per-model Jacobian-lens artifact — `models/<safe_model_id>/jlens.
{safetensors,json}`, peer to the neutral-activation cache and shaped like it
(`layer_<idx>` tensor keys + atomic JSON sidecar). `LENS_FORMAT_VERSION = 3`;
a wrong version, missing/mismatched payload digest, non-finite tensors, or a corrupt sidecar all log a warning and
read as "no lens" (`load_lens → None`) rather than crash — the caller decides
whether to error (`LensNotFittedError` with the `lens fit` hint) or re-fit.
Storage is **fp16** (deliberately unlike the neutral cache's fp32 invariant:
J entries are O(1) so range is no constraint, and nothing here feeds a
covariance inversion), promoted to fp32 on load. The sidecar records `method`,
`n_prompts`, `d_model`, `source_layers`, the corpus spec + token-id sha256 (the
resume/staleness key), optional raw-corpus sha/count metadata for model-load-free
no-op checks, `seq_len`, `dim_batch`, `skip_first_positions`, exact model
source/live-weight identities, tensor sha256, and the model's
layer count (needed to prove `all`/`workspace` coverage without loading it).
Loading uses `safe_open` one layer at a time, so fp16 source storage is released
as the fp32 lens is materialized rather than coexisting as a full mapping. Resumable
checkpoints live beside the full artifact as `jlens.partial.{safetensors,json}`:
the estimator writes a self-contained averaged checkpoint directly from raw sums,
merging a prior prefix one layer at a time during fp16 conversion. This avoids a
second full fp32 lens at checkpoint cadence and makes repeated interruptions
independent of an older full artifact (`base_n_prompts=0`). Finalization promotes
an already-complete terminal checkpoint without rewriting its matrices, or writes
the full artifact durably and removes an incomplete checkpoint. Metadata-only
final/checkpoint preflight rejects incompatible corpora/layers before matrix IO.
`lens_paths` /
`lens_checkpoint_paths` / `save_lens` / `save_lens_checkpoint_accumulator` /
`save_lens_checkpoint` (compatibility) / `load_lens` /
`load_lens_checkpoint_sidecar` / `load_lens_checkpoint` /
`promote_lens_checkpoint` / `remove_lens`; the fit itself lives in `core/jlens.py`.

## atomic.py / staging.py

`atomic.py` — `write_bytes_atomic` / `write_json_atomic`: stage to a same-directory
temp file, `fsync`, then `os.replace`. `staging.py` — `stage_verify_swap` is the
shared HF-install choreography: build under `.staging/`, recover `.bak` on
interrupted swaps, then promote atomically (`target → .bak`, `.staging → target`)
with best-effort restore. (`datasource.py` is gone — extraction takes node corpora
directly.)
