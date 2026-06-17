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
`safe_model_id` (`/` → `__`), `ensure_within(root, *parts)` (path-traversal
barrier). `vectors_dir` / `concept_dir` survive only for the legacy-port scan — no
current writer targets them.

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
(`scripts/upgrade_packs.py` / `session._port_stale_legacy_vector`). Also stamped
onto the profile-cache sidecars `vectors.save_profile` writes.

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
also carries an optional `kind` ∈ {`abstract`, `concrete`} (`_validate_node_kind`),
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
`packs.hash_file` for the per-file sha256 integrity manifest.

A node corpus is now a list of conversational *responses* (`list[str]`) aligned to
the shared A2 baseline user prompts — `response[i]` answers `baseline_prompt[i % k]`
(`baseline_prompts_path`), so a corpus length must be a multiple of `k`. The shared
baseline prompts are global (bundled `saklas/data/baseline_prompts.json`), not
per-manifold, so the generation path no longer writes `scenarios.json` and no
longer calls `write_manifold_scenarios` (the helper still exists and round-trips an
explicit `scenarios=` corpus, but generation does not feed it).

Authoring: `create_manifold_folder` (authored webui/HTTP path, returns `(folder,
advisories)`), `create_discover_manifold_folder` (`_sanitize_hyperparams` drops
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
whitener, ...)` (pure-io: applies a caller-supplied per-layer Procrustes
`alignment` to the fitted subspaces — `mean → M_L mean`, `basis → basis @ M_Lᵀ` —
leaves RBF + `node_coords` untouched, re-bakes the Mahalanobis **share** in target
space — the target whitener is **required** and must cover the transferred layers
(`WhitenerError` otherwise; no Euclidean rebake) — clears `origin`, writes the
`_from-<safe_src>` variant). `manifold_summary(folder)` is the session-independent
serializer shared by `pack show -j` + the HTTP summary route.
`iter_manifold_folders`, `bundled_manifold_names`, `materialize_bundled_manifolds`
(copy-on-miss into `default/` for complete package-data folders only, plus a
re-copy when the bundled manifest hash drifts or the on-disk `format_version`
predates `MANIFOLD_FORMAT_VERSION`). Per-node `role`
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
imports up into `cli`). A concept *is* a manifold now: `_all_concepts` walks
`manifolds_dir()` via `iter_manifold_folders`, and `ResolvedConcept.folder` is the
manifold folder. `Selector(kind, value, namespace)` with kinds `name` / `tag` /
`namespace` / `model` / `all`; `parse(raw)` handles `ns/name`, the prefixes, and a
trailing `:variant` via `_VARIANT_REGEX = ^(raw | sae[-…] | role[-…] | from[-…])$`
— **no `pca`**. `resolve(selector)` filters the memoized walk; `model:X` matches
any manifold with a fitted tensor for X.

Bare-pole resolution moved entirely to the manifold tier (a bipolar concept is a
2-node `pca` manifold): `resolve_pole(raw, namespace=)` only peels the `:variant`
suffix + canonicalizes (always `match=None`, `sign=+1`).
`resolve_manifold_label(label, *, namespace=)` finds a node by label across
installed manifolds; `resolve_manifold_name(name, *, namespace=)` resolves a 2-node
`pca` manifold's *name* (e.g. `formal.casual`) to node 0 (the `orient_to=0` + pole) —
the vector-composite read path. `resolve_bare_name(raw, *, namespace=) →
ResolvedManifoldLabel | None` is *just* the manifold-label tier (it delegates to
`resolve_manifold_label`, raising on cross-manifold collision); it knows nothing of
poles. The tier ordering lives in the caller (`core/steering_expr`): a bare dot-free
slug hits `resolve_bare_name` (the label tier) first, then the composite-name tier
(`resolve_manifold_name`, for a dotted `formal.casual`), then `resolve_pole`
canonicalization — every bipolar pole is itself a node label, so a bare pole
resolves through the label tier as an affine `%` push. Three memoized walks
(`_concepts_cache`/`_manifold_labels_cache`/`_manifold_names_cache`) keyed on
`manifolds_dir()`; `invalidate()` clears all three — mutating code must call it.
`parse_args(tokens)` splits a token list into one concept selector + one optional
`model:` scope.

## cache_ops.py / hf.py

Both gutted to the surface that survives the collapse.

`cache_ops.py` is now just GGUF export: `_export_gguf_manifold(ns, name, *,
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
`|` project-away via `project_away`), folds the result to a one-pole ray
(`fold_directions_to_subspace`), and writes a baked manifold to
`manifolds/local/<name>/` — one fitted tensor per shared model
(`create_baked_manifold_folder` / `save_baked_manifold_tensor`), all sharing one
`manifold.json`. The shared grammar's `|` is accepted; triggers + bare
un-namespaced poles are rejected. `shared_models(expression)` returns the models
every term has a fitted tensor for. The baked sidecar carries `method="merge"` +
per-component `components` provenance.

## alignment.py

Cross-model probe alignment via per-layer Procrustes.
`load_or_compute_neutral_activations(...)` is the disk-cached per-model neutrals
(the neutral corpus × layers, **fp32** — the project-wide invariant; self-heals legacy
bf16/fp16/non-finite caches). These are what the Mahalanobis whitener builds its
covariance from. `fit_alignment(src, tgt, *, min_shared_layers=10) → {layer: M_L}`
(orthogonal Procrustes for matched dim, rectangular least-squares otherwise; both
center first); `alignment_quality` is per-layer R². `transfer_profile(profile,
alignment_map, *, source_model_id, ..., whitener)` applies `M_L @ v_src` per
layer (uncovered layers dropped, `method="procrustes_transfer"`), re-scaling each
magnitude to its *target* Mahalanobis norm. The target whitener is **required** and
must cover the transferred layers (`WhitenerError` otherwise; Mahalanobis-only — no
Euclidean transfer). The fitted map round-trips under the *target* model dir
(`models/<safe_tgt>/alignments/<safe_src>.…`). `transfer_manifold`
(`manifolds.py`) is the manifold counterpart.

## atomic.py / staging.py

`atomic.py` — `write_bytes_atomic` / `write_json_atomic`: stage to a same-directory
temp file, `fsync`, then `os.replace`. `staging.py` — `stage_verify_swap` is the
shared HF-install choreography: build under `.staging/`, recover `.bak` on
interrupted swaps, then promote atomically (`target → .bak`, `.staging → target`)
with best-effort restore. (`datasource.py` is gone — extraction takes node corpora
directly.)
