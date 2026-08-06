# io/

Persistence and distribution: exact on-disk schemas, the selector grammar,
integrity, atomic publication, Hugging Face transport, and artifact lifecycle.
All Saklas-owned state lives under `~/.saklas/` and resolves through
`paths.saklas_home()`.

`io` imports `core`'s pure dataclasses (`Manifold`, `LayerSubspace`,
`ManifoldDomain`) — the correct layering arrow. `core/manifold.py` stays
pure-tensor geometry with no filesystem coupling, so codecs and filesystem
transactions live here.

Three artifact families: the **manifold** (per-concept, the steering artifact),
the **template** (slot + values + contexts), and the **per-model sources**
(neutral/alignment caches, Jacobian lens, SAEs).

## Shared primitives

### paths.py

`saklas_home()` honors `$SAKLAS_HOME`. Helpers: `manifolds_dir`, `manifold_dir`,
`templates_dir`, `models_dir`, `model_dir`, `neutral_statements_path`,
`baseline_prompts_path` (user override for the shared baseline user prompts,
falling back to bundled `saklas/data/baseline_prompts.json`).
`ensure_within(root, *parts)` is the path-traversal barrier — namespaces, names,
model ids, and manifest-relative filenames arrive from HTTP bodies, CLI args, and
downloaded manifests, so the string is re-checked where it becomes a path.

`safe_model_id` / `unsafe_model_id` round-trip a model id through one reversible
`_z`-prefixed urlsafe-base64 component; `encode_release_id` / `decode_release_id`
do the same in lowercase Base32 for SAE releases and source model ids. Both
reject non-canonical input.

This module owns the tensor-filename variant scheme. A manifold folder holds
several fitted tensors per model, at most one *kind* per file:
`<safe_model>.safetensors` (canonical difference-of-means),
`_sae-<encoded_release>`, `_from-<encoded_src>` (cross-model transfer).
`tensor_filename` / `sidecar_filename` construct (the two kind kwargs are
mutually exclusive) and `parse_tensor_filename → (safe_model, variant)` inverts;
components escape the separators before concatenation so a model id containing
one stays unambiguous, and a non-canonical identity parses as `None`.
`VARIANT_SUFFIX_RE` is the single source of truth for the selector-level
`:variant` suffix (`raw` | `sae[-…]` | `role[-…]` | `from[-…]`), imported by
`io.selectors` and `cli.runners`. `role-` exists in the selector grammar only:
`extract --role` bakes the role into the corpus and writes the canonical raw
tensor.

### integrity.py

`NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$` is the one artifact-name grammar
(manifolds, selectors, templates, local lens/SAE names). `hash_file` /
`verify_integrity` are the sha256 helpers behind every integrity manifest;
`verify_integrity` routes each manifest-relative path through `ensure_within` and
counts an escaping entry as a failure rather than reading off-tree. Its
stat-identity fingerprint cache is purely an optimization — first load and any
stat change still run the full sha256.

`PROFILE_FORMAT_VERSION = 6` stamps the sidecars `core/profile.py::save_profile`
writes. That schema is exactly `{format_version, saklas_version, method,
tensor_sha256, provenance}` — four identity fields plus one free-form JSON-safe
`provenance` object, so a producer records anything else there rather than
growing the set. `method` is closed: `profile`, `manifold_pca` (the folded 2-node
manifold view `session.extract` returns), `merge`. `tensor_sha256` binds the
exact payload bytes, and writer-stamped fields are re-stamped on save so
`Profile.load(p).save(q)` round-trips. No cache rides this version — the
neutral-activation and alignment caches carry their own schemas and versions in
`alignment.py` — so the only reader a bump reaches is a user-saved `Profile`,
and the reader's version check leads so a stale file is told which version it
is and what to re-run, not which field the exact-set schema missed first.

### atomic.py / staging.py / shards.py

`atomic.py` — `write_bytes_atomic` / `write_json_atomic` stage to a
same-directory tempfile, `fsync`, then `os.replace` (same-dir staging is
required: `os.replace` is atomic only within a filesystem). `fsync_directory`
makes a directory entry durable. `artifact_lock` is the cross-process lock;
`ReleasableArtifactLock` lets a short cache transaction run inside a longer fit,
and `artifact_process_lease` / `artifact_has_live_lease` protect mapped immutable
shards after that lock is released (stale PID markers are reaped).

`staging.py` — `stage_verify_swap`: recover a `.bak` when the destination is
missing, wipe stale staging, build a fully validated `.staging/` tree, then
promote (`target → .bak`, `.staging → target`) with best-effort restore.

`shards.py` — the immutable-generation primitive three per-model families share
(the neutral and alignment caches in `alignment.py`, the local J-lens in
`lens.py`). One `<stem>.layer-<L>.gen-<uuid>.safetensors` payload per layer plus
a JSON sidecar naming the live generation per layer under `tensor_files`;
publication is a single atomic pointer replace, so a reader only ever sees a
complete generation set. `shard_paths` requires a bare filename (a
manifest-supplied `..` or absolute path is rejected) and takes
`require_exact_keys=False` for a scoped subset read. `cleanup_generations` runs
only once the surviving pointer is durable, and logs rather than raises — an
uncollected generation is inert, a collected-too-early one is data loss.
`fit_lock` keys on a `<stem>.fit` sibling so the expensive transaction spans model
construction without blocking short pointer reads. Sidecar *schemas* and format
versions stay private per family; sharing only the path/GC/lock discipline keeps
crash recovery from drifting between them.

### source_registry.py

`ActiveSourceRegistry` is the `active.json` selection shared by the lens and SAE
families: a `{format_version, model_id, kind, name}` payload written atomically
under the artifact lock. `read` validates every field and returns `None` on any
deviation — a stale or hand-edited selection must not stop a model from starting.
`write` validates kind, name grammar, and on-disk existence, so a selection never
dangles. `clear_if_active` unpublishes when the selection names exactly the
source being removed. Per-family: the binding schemas the selection points at and
the loaders that make it resident.

### bootstrap.py

`materialize_bundled_artifacts()` is the single bootstrap call every entry point
makes. It runs `materialize_bundled_templates()` **before**
`materialize_bundled_manifolds()`: a bundled manifold may `template_ref` a
bundled `default/<name>` template, and the fit resolves that ref (a hard
`TemplateNotFoundError` otherwise, with `nodes_sha256` degrading to the bare ref
string). Owning the order in one place keeps a new bootstrap site from getting it
wrong.

`canonical_payload_sha256(data, *, strip_keys=())` is the bundle-drift hash both
materializers use — canonical JSON so cosmetic differences compare equal, with
`strip_keys` dropping top-level keys carrying local transaction state.
Unparseable bytes fall back to a raw sha256, so hand-edited content reads as
drift rather than being silently overwritten.

## The manifold artifact

`~/.saklas/manifolds/<ns>/<name>/` holds `manifold.json`, `nodes/NN_<label>.json`
corpora, and per-model fitted `.safetensors` + `.json` sidecar pairs. The format
splits across four modules; `manifolds.py` re-exports them as the stable
`from saklas.io.manifolds import X` surface.

### manifold_folder.py — the format core

The dependency root: dataclasses, validators, `manifold.json` load/save,
integrity, and the locks. `MANIFOLD_FORMAT_VERSION = 10`; readers and writers
require exactly 10. A `manifold.json` at any other version is a hard
`ManifoldFormatError`; a *fitted sidecar* at another version is **stale, not
corrupt** — `ManifoldFolder.load` leaves its stem unregistered and the manifold
reads as "not fitted for that model", a clean cache miss the next fit
overwrites. Raising there would take the whole artifact out of the selector
index on a format bump, unfitting its labels, probes, and steering along with
the one expired tensor. Corrupt (rather than merely old) sidecars still raise,
and a corpus-less `baked` manifold keeps the strict read because it has nothing
to re-fit from. `min_nodes(n) = 2n+1` is the curved-fit poisedness floor (a
flat `pca` fit needs only `k+1`). Node labels validate against `_LABEL_REGEX =
^[a-z][a-z0-9_-]{0,63}$` — stricter than `NAME_REGEX` because `.` is the bipolar
separator and the steering lexer addresses a label as `%label`, so a dotted label
could be neither typed nor resolved unambiguously. The manifold *name* keeps
`NAME_REGEX`.

Five `fit_mode`s share the class: **`authored`** (user supplies `domain` +
per-node `{label, coords}`; curved RBF); the discover trio **`pca`** (flat, also
the 2-node vector case) / **`spectral`** (curved) / **`auto`** (geometry chosen
per-model by `core.topology.select_topology`), where nodes carry `{label}` only
and coords are derived at fit time into the safetensors; and **`baked`** —
corpus-less, a precomputed direction from `manifold bake` that never re-fits, with
`BakedManifoldError` guarding corpus-requiring calls.

`sanitize_hyperparams(fit_mode, hyperparams)` is the single per-mode whitelist
(`auto` takes the union of its candidates' knobs plus `persistence_frac`). A key
the selected dispatcher would ignore **raises** at the authoring/override
boundary rather than being silently discarded, so the manifest never records a
knob that did nothing.

`ManifoldFolder.load(folder, *, verify_manifest=True)` validates the format
version, name, labels, and node shape, branches on `fit_mode`, enforces
`min_nodes` on authored folders, verifies the `files` manifest, and demands a
sidecar per fitted tensor. `verify_manifest=False` serves metadata-only
inventory, authoring, lifecycle routing, and summaries, so listing scales with
metadata size rather than artifact bytes; runtime, install, and push callers keep
the full walk. Structural checks and tensor/sidecar pairing run either way.

A node corpus is a `list[str]` of responses aligned to the shared baseline
prompts (`response[i]` answers `baseline_prompt[i % k]`, so its length is a
multiple of `k`). Each node also carries an optional `role` (the assistant-role
substitution its centroid is pooled under — a real fit input) and `kind` ∈
{`abstract`, `concrete`, `custom`} (generation-time provenance the fit never
consumes).

`nodes_sha256(*, resolved_template_sha256=None)` is the staleness key: node
labels, every corpus file's bytes, `{domain, node_coords}` (authored) or
`{fit_mode, hyperparams}` (discover), `node_roles`, `node_kinds`, and — for a
templated manifold — the resolved template's content hash, falling back to the
ref string when it genuinely cannot be resolved. A baked folder hashes a
provenance-only sentinel.

`ManifoldSidecar` is the lean per-tensor JSON;
`validate_manifold_sidecar_payload` enforces the exact schema by set equality, so
no field is ever synthesized on read. `method` is a closed discriminator that
`MANIFOLD_METHOD_FIT_MODES` couples to `fit_mode` (`manifold_pca` /
`manifold_sae` / `manifold_monopolar{,_sae}` → authored;
`manifold_discover_{pca,spectral,auto}` → their pinned mode;
`manifold_discover_sae` → any discover mode; `merge` / `folded_vector` → baked;
`manifold_procrustes_transfer` → any), and provenance blocks are validated
all-or-nothing against it: `auto` carries the resolved mode plus ranked
`topology_candidates`, `merge` carries `components` + `bake_policy`, a transfer
carries source model id + fingerprint, and an `sae-*` feature space carries a
release + per-layer feature ids — each absent on every other kind of fit.
Four fields exist so a refit can be proven unnecessary *off-model*:
`model_source_fingerprint` (the pre-load-verifiable checkpoint identity —
distinct from `source_model_fingerprint`, which names the other model a
cross-model transfer came from), `capture_version`, `capture_render_sha256`
(the tokenizer-render half of `capture_sha256` — node partition plus each
rendered row's token ids and pool index), and `baseline_prompts_sha256`. See
"weight-free fit preflight" under `manifold_lifecycle.py`.
`node_spread_per_layer` is measured *before* DLS, so its keys are the
evaluated-layer roster and may strictly contain `fitted_layers`, which is what
lets a cache prove a layer was evaluated rather than accidentally omitted.

`_locked_manifest(folder)` serializes manifest read-modify-write per folder.
`manifold_pair_lock(tensor)` locks one tensor/sidecar pair at a stable
digest-named path in the *parent* namespace directory, not inside the removable
folder, so `rm` and stage-swap refreshes cannot unlink a held lock inode and let
a later process acquire a fresh one for the same logical pair.
`update_file_hashes(*paths)` hashes only the files just replaced, preserving
proofs for other variants. `manifold.json` also carries three local-only
transaction keys — `files`, `artifact_id`, `fit_epochs` — stripped from
bundle-drift comparisons and from pushes.

### manifold_tensors.py — fitted-tensor codec + row spool

`save_manifold` / `load_manifold(path, *, verify_manifest=True)` round-trip the
safetensors payload (per-layer mean/basis/optional `affine_map`, or the curved
RBF/σ triple, plus shared `node_coords` and optional `origin`) and its sidecar
under the folder pair lock, staged through same-directory tempfiles with fsync +
`os.replace`. `_replace_manifold_file` is the atomic-replace seam publication
failure-injection tests monkeypatch; `_load_manifold_locked` is the pair-locked
read half `core/extraction.py`'s cache-hit fast path reuses. The tensor-derived
layer roster must equal the sidecar's `fitted_layers`.

`ActivationRowStore` is the temporary layer-major mmap row spool a curved fit
uses once the centroid-derived basis exists — one shared-memory tensor per layer
in the model output dtype (lossless relative to the source residual; fp32
promotion happens only when covariance math consumes a node slice), keeping
`nodes × responses × layers × d_model` off the accelerator. `persist` / `load` /
`load_shards` / `combine_disjoint` back the sharded per-model capture cache.

### manifold_authoring.py — discovery + the write path

`iter_manifold_folders(namespace=None)` walks the tree metadata-only and skips
malformed folders, so one bad artifact cannot break a listing.
`RESERVED_NAMESPACES = {jlens, sae}` — both are lazily-resolved per-model
steering tiers, and a folder under either name would shadow that resolution.

- `create_manifold_folder(...) → (folder, advisories)` — the authored path;
  advisories are soft poisedness/flat-axis warnings, surfaced before a fit is paid
  for. `update_manifold_folder` re-authors one.
- `create_discover_manifold_folder(..., node_roles=, node_kinds=)` — labeled
  corpora only.
- `create_manifold_from_template(ns, name, *, template_ref, fit_mode, …)` —
  resolves the template, expands `values × contexts` through
  `TemplateFolder.node_corpora()`, and writes an ordinary discover folder storing
  both the derived corpus and the `template_ref`. The template is the authoring
  source of truth; the corpus is its materialization.
- `create_baked_manifold_folder` + `save_baked_manifold_tensor` — the `manifold
  bake` target: one fitted tensor per model, all sharing one `manifold.json`. Each
  pair records its manifest proof before the call returns, and an
  identical-producer retry repairs a pair left unproven by a crash mid-publication.
- `init_discover_manifold_folder` + `append_discover_manifold_node` — streaming
  companions for big rosters, so a crash keeps the finished nodes.
  `plan_discover_generation → DiscoverGenerationPlan` is the shared resume /
  add-nodes planner; it bypasses `ManifoldFolder.load` so it can inspect a partial
  folder, but still requires that partial manifest to carry the exact current
  version, fields, name, and label-only node shape.
- `merge_discover_manifolds` unions ≥2 discover sources' corpora into a fresh
  *unfitted* folder (authored sources, label collisions, and mixed modes without an
  explicit override raise).

### manifold_lifecycle.py — rm / clear / refresh / transfer / summary

Addressed by `(namespace, name)`, not a `Selector`. `manifold.json::source`
(`local` / `bundled` / `hf://…`) is the tier that decides refresh behavior.
`remove_manifold_folder` returns `rematerializes_on_restart` for bundled
`default/` folders; `clear_manifold_tensors(..., model_scope=None,
variant="all")` filters raw/sae/from/all and keeps `manifold.json` + corpus;
`refresh_manifold` unscoped skips `local`, re-materializes `bundled`, and
re-pulls `hf://`, while scoped it drops one model's fit. A scoped clear
increments only its own model/variant fit epoch, invalidating a paused matching
fit without discarding unrelated target work; rm/recreate changes the folder's
`artifact_id`.

Destructive paths take the manifest lock then the affected pair locks in
deterministic sorted order — as do whole-folder force authoring and HF stage
swaps — so they cannot unlink a tensor while a fitted reader owns its logical
pair lock. Shared per-layer capture shards are collected only after the last
fitted-sidecar owner disappears, under the capture-stem lock fitting uses, and
only when no live PID lease is consuming mapped rows. Stem enumeration is
deliberately independent of sidecar readability — clear and rm are repair
surfaces, so a corrupt sidecar must not make a content-addressed capture
undiscoverable — with ownership proven globally under the stem lock immediately
before deletion.

`transfer_manifold(folder, *, from_model, to_model, alignment, whitener, …)` is
folder read/write orchestration only: it hands the loaded `Manifold`, the
caller-supplied compact affine alignment, and the target whitener to
`core.manifold.transfer_manifold_subspaces`, then writes the `_from-<safe_src>`
variant and patches the transfer-provenance sidecar. The target whitener is
required (`WhitenerError` otherwise, no Euclidean rebake), and the core function's
plain `ValueError` for an alignment covering no fitted layer surfaces here as
`ManifoldFormatError`. `preflight_transfer_manifold → TransferSourceProof` proves
the source and rejects only a trusted existing target before any model or
alignment work; the backend repeats the proof authoritatively inside its own
transaction.

`preflight_manifold_fit_noop(folder, *, model_id, layer_indices, sae, fit_mode,
hyperparams, quantize, device) → ManifoldFitProof | None` is the **weight-free
fit preflight** — the manifold counterpart of `lens fit`'s model-free no-op, and
the reason `manifold extract` / `manifold fit` no longer pay a model load to
discover they had nothing to do. It proves, in the order the fit keys on them:
`nodes_sha256` (recomputed from the folder); `model_source_fingerprint`
(recomputed from the published config / local checkpoint files and the load
representation); `model_fingerprint`, which is *not* recomputable off-model and
is therefore **bridged** — the neutral-activation cache was written by a real
load and binds this checkpoint source to the loaded fingerprint it produced, the
same bridge `alignment.py`'s `_proven_sidecar` walks; `capture_version` +
`capture_render_sha256` + `baseline_prompts_sha256`, re-rendered here from
tokenizer + prompts + corpus, which with the two fingerprints covering the
model-identity half make `capture_sha256` equality follow; `fit_policy_version`;
and the fitted-layer set, resolved against the *config's* layer count
(`core.model.config_model_shape`). The identity digests come from
`core.extraction.offline_fit_identity` so the preflight and the fit cannot drift
on what a corpus hashes to.

`None` means **unproven**, never "stale": an SAE fit (whose key needs a resolved
backend fingerprint), a discover override (which rewrites `manifold.json` inside
the fit's own lock, so the visible hash is not the one the fit compares), a
checkpoint with no provable source, a missing neutral cache, a config with no
layer count, an unresolvable template, a corpus a templated fit would re-derive,
or a tokenizer that will not load all return it, and the caller must fall
through to the ordinary model-loading fit. A partial proof is the one thing this
must never emit — it would serve a stale fit after a baseline-prompt edit.
Nothing here mutates: no manifest rewrite, no capture-cache recovery, no
promotion.

`manifold_summary(folder, *, include_fits=False)` is the session-independent
serializer shared by `pack show -j` and the HTTP summary route: identity, source,
tags, `template_ref`, fit mode, geometry, node layout, hyperparameters, and which
models have a fitted tensor (`fitted_models` + per-model `tensor_variants`). A
discover folder carries no on-disk geometry — coords are per-model, in the
safetensors — so `domain` is `{}`, `intrinsic_dim` is `0`, `node_coords` is `[]`.
`template_ref` is surfaced because it is what a user needs before hand-editing
`nodes/`: a re-derive would overwrite the edit. `fitted` appears only under
`include_fits` — one `manifold_fit_summary` block per fitted tensor *stem*,
unreadable sidecars skipped rather than failing the summary — because it costs
one sidecar read per stem: an inspection surface wants it, a listing rendering
many folders does not.

### manifolds.py — public barrel + bundled materialization

Re-exports the four submodules under one import surface, and physically hosts
bundled materialization: `_materialized_home` is monkeypatched by attribute path,
so the guard and the function reading it stay together.

`materialize_bundled_manifolds()` is copy-on-miss into `manifolds/default/`,
JSON-only (per-model fits are user-side), and a per-home process-scoped no-op
after the first call — which sidesteps the ambiguity between "bundle changed
under the user" and "user changed the manifest via a CLI override" within one
invocation. Three paths: fresh install, bundle update (canonical manifest hash
differs, or the on-disk `format_version` is not current), or skip.
`bundled_manifold_names()` advertises only folders whose `manifold.json` and
every declared node file are present, so an interrupted regeneration cannot
expose a partial folder as a default manifold.

Drift comparison runs on the **local-state-stripped** canonical payload: `files`
accumulates per-model fit proofs locally, so hashing it would misread every fit
as a bundle update. A genuine bundle update carries `files` entries forward for
still-present artifacts the bundle does not ship (fitted tensors + sidecars) —
proofs are re-verified on every load, and `nodes_sha256` remains what decides
whether a fit is current — but re-copies node corpora unconditionally, since a
node-level edit is meaningful only against a specific bundle version and keeping
it would mix two corpora. The materializer drops the selector index itself.

## selectors.py

The selector grammar shared by `core.session` and the CLI. It lives in `io` so
neither imports up into `cli`. A concept *is* a manifold, so every view projects
from a walk of `manifolds_dir()`.

`Selector(kind, value, namespace)` with kinds `name` / `tag` / `namespace` /
`model` / `all`. `parse(raw)` handles `ns/name`, the prefixes, the `default` and
`all` aliases, and a trailing `:variant` validated against `VARIANT_SUFFIX_RE`.
`resolve(selector)` filters the index; `model:X` matches any manifold with a
fitted tensor for X, regardless of variant. `parse_args(tokens)` splits a token
list into one concept selector plus one optional `model:` scope.

**One memoized walk**, keyed by `manifolds_dir()`, produces a `_ManifoldIndex`
with three views: `concepts` (every installed manifold), `labels` (the flattened
`(namespace, manifold, node label)` index), and `names` (the 2-node `pca` subset
addressable by manifold name). A single compound steering expression hits all
three, so the tree is walked once and the views built in memory. `invalidate()`
drops it, and **every io mutator calls it for itself** — authoring, lifecycle,
install/pull, bundled materialization — so callers carry no invalidation duty.

`resolve_bare_atom(concept, *, namespace=, typed_namespace=, variant=) →
ResolvedBareAtom` is the **single owner** of the bare-atom tier ladder, ordering
and arbitration included; `core/steering_expr` calls it once instead of
hand-sequencing tiers. The result is tagged `kind ∈ {label, name, pole}`:

1. **label** — `resolve_manifold_label`, gated to a plain bare slug
   (`variant == "raw"`, no typed namespace, no `.`); a hit routes to a label-form
   `<manifold>%<label>` push. The lookup is a folder scan, not a fitted-tensor
   check, so an authored-but-unfitted manifold's labels still resolve.
2. **name** — `resolve_manifold_name`, for a `variant == "raw"` name that *is* a
   2-node `pca` manifold (`formal.casual` — the `.` skips tier 1), resolving to
   node 0, the positive pole.
3. **pole** — `canonicalize_atom(raw) → (canonical, variant)`, the pure slug +
   variant peel. Every bipolar pole is itself a node label and has already
   resolved through tier 1, so this tier carries genuine fresh-vector, variant,
   and dotted-name references.

A collision in either manifold tier raises `AmbiguousSelectorError` and
propagates; any other failure falls through to tier 3, where a genuine error
re-surfaces with the canonical message. `resolve_manifold_label` and
`resolve_manifold_name` stay public as the underlying steps.

## templates.py

The standalone templated-completion artifact —
`~/.saklas/templates/<ns>/<name>/template.json`, a peer of a manifold.
`TEMPLATE_FORMAT_VERSION = 2`. `TemplateFolder = {name, slot, values,
contexts:[TemplateContext{turns:[{role, content}], assistant}]}`.

**Invariant** (`_validate_body` / `_validate_context`): the slot appears exactly
once in each context's final `assistant` string and in no history turn — history
is shared common-mode across the values, so the slot lives only where the value
is read — and the last history turn must be `user`.

Derived views: `node_labels()` (slugged values), `node_corpora()` (`{label:
[slot-filled assistant per context]}`, `corpus[i] ↔ contexts[i]` — the manifold
corpus), `score_inputs()` (per-context `{messages, assistant_prefix, suffix,
choices, labels}` — the scorer feed, split at the slot). `sha256()` hashes
`slot × values × contexts`, excluding description/tags; it is the staleness key
folded into a referencing manifold's `nodes_sha256`.

Lifecycle: `create_template_folder` (validates the whole body before writing, so
there is no half-built folder), `resolve_template(selector)` (a bare name
searches all namespaces; `AmbiguousTemplateError` on collision,
`TemplateNotFoundError` on miss), `iter_template_folders`,
`remove_template_folder`. `_LABEL_REGEX` is redefined locally so the
`manifolds → templates` import direction stays acyclic.

**Bundled templates are dormant by design.** `bundled_template_names` /
`materialize_bundled_templates` mirror the manifold materializer, but no template
ships bundled, so the lister returns `[]` and the materializer no-ops. The
mechanism stays because a template-derived bundled manifold is the one artifact
that authors deterministically and model-free — write the template, and
`create_manifold_from_template` derives the corpus with no generation step. The
templates-before-manifolds ordering that makes it shippable lives in
`io.bootstrap`.

## bake.py

Offline direction merging into a corpus-less `fit_mode="baked"` manifold.
`merge_into_manifold(name, expression, model, *, force, strict,
expected_model_fingerprint)` folds each component (a fitted 2-node `pca`
manifold) to a direction via `core.capture.folded_directions`, combines them with
`linear_sum` over the union of their layer coverage (an absent component
contributes zero rather than deleting the layer, matching live composition), folds
the result to a one-pole ray, and writes `manifolds/local/<name>/` — one fitted
tensor per shared model, all sharing one `manifold.json`.
`shared_models(expression)` reports the models every term has a fitted tensor for.

Only namespace-qualified additive/subtractive scalar terms are accepted.
Triggers, `!`, `%`, multi-coefficients, and `~`/`|` are rejected: projection is
Mahalanobis-only and needs an identity-matched live whitener, so substituting an
offline Euclidean projection would change semantics. Component variants are
checked against the loaded manifold's provenance — a role-baselined component must
be selected through its `:role-*` alias, a `:from-*` component must match the
recorded source model.

A same-expression retry resumes only an incomplete destination whose proven
prefix matches the component tensor hashes, coefficients, model fingerprints, and
bake policy; a fully proven destination keeps ordinary exists semantics. The baked
sidecar carries `method="merge"`, the unanimous component model fingerprint,
`bake_policy`, and per-component `components` provenance.

## alignment.py

The per-model neutral-activation cache and the cross-model Procrustes alignment
cache, both built on `shards.py`.

`load_or_compute_neutral_activations(...)` is the disk-cached neutral corpus,
`[N, D]` per layer in **fp32** — the project-wide range invariant, since late
residual channels overflow fp16 and would poison every covariance-derived read.
It is the single per-model neutral artifact: the whitener's covariance and the
probe-centering mean both come from it. Identity covers the exact loaded model,
the rendered tokens, and the layer schema, with verified payload digests;
non-current or corrupt caches miss and are replaced. Cache format 4 is an atomic
JSON pointer to one immutable fp32 shard per layer, and
`load_validated_neutral_cache(..., requested_layers=)` digests and materializes
only the selected shards.

Locking is load-bearing: the per-model `neutral_fit_lock` spans cache recheck,
**model construction**, capture, and publication, and the directional
`alignment_fit_lock` spans both model loads and the second cache recheck — so two
concurrent cold transfers never duplicate the shared model load, the neutral
forwards, or the Procrustes fit.

`load_or_fit_transfer_alignment(src, tgt, *, force, label, requested_layers=None)
→ (M, quality_per_layer, map_path, source_identity, target_identity,
target_whitener, target_layer_means)` is the public single-flight orchestrator
over those primitives; `manifold transfer`'s runner is a thin caller. Identity is
proven without hashing any tensor payload; a complete cached-map hit drops the
source seed rows and never builds the source model; the roster narrows to
`requested_layers ∩ available`, releasing unrequested tensor owners even when a
cold fill had to capture a full roster first; a cache writer racing the metadata
preflight restarts the directional transaction under the already-held outer lock.
On no-shared-layers or `AlignmentError` it exits with the caller's `label` prefix.

`fit_alignment(src, tgt, *, min_shared_layers=10, …) → {layer: LayerAlignment}`
runs row-space orthogonal Procrustes for a matched dim and rectangular
minimum-norm least squares otherwise, both retained as low-rank factors, plus the
fitted translation `b_L = mean_tgt − M_L mean_src`; `alignment_quality` is
per-layer R². The map binds both validated neutral-cache identities, both model
fingerprints, and its own payload digest under
`models/<safe_tgt>/alignments/<safe_src>.json` (cache format 5): one immutable
factor shard per layer, atomically pointer-switched. Existing factor pointers
carry forward and only missing requested layers are fitted and written, so a
narrow transfer never digests, materializes, fits, or whitens unrelated layers.
`-f` recomputes the alignment and transfer output but does not recapture an exact
neutral cache.

## lens.py / lens_sources.py

`lens.py` owns the Saklas-fitted per-model Jacobian lens:
`models/<safe_model_id>/jlens/local/default/manifest.json`, an atomic pointer to
immutable per-layer `jlens.layer-<L>.gen-<uuid>.safetensors` generations.
`LENS_FORMAT_VERSION = 6`, required exactly. Storage is **fp32**, which keeps the
estimator accumulator lossless across save and resume. The fit itself is
`core/jlens_fit.py::fit_jacobian_lens`.

The sidecar records the estimator settings (`method`, `n_prompts`, `d_model`,
`source_layers`, `seq_len`, `dim_batch`, `skip_first_positions`), the corpus spec
+ token-id sha256 that is the resume/staleness key, exact model source and
live-weight identities, per-layer filenames + sha256, and the model's layer count
— enough to prove `all` / `workspace` coverage without loading it. A wrong
version, a missing or mismatched digest, non-finite tensors, or a corrupt sidecar
log a warning and read as "no lens" (`load_lens → None`) rather than crashing;
the caller decides whether to raise `LensNotFittedError` or re-fit. Loading uses
`safe_open` one shard at a time, so only requested matrices materialize, and
missing-layer top-ups carry unchanged shard pointers forward after rehashing
every reuse candidate.

Checkpoints use the same generation-pointer scheme under `checkpoint.json`. The
estimator writes a self-contained averaged checkpoint from raw sums, merging any
prior prefix one layer at a time during fp32 streaming, so repeated interruptions
stay independent of any older artifact. Finalization promotes a complete terminal
checkpoint by pointing the durable sidecar at its immutable generations without
rewriting them, accepting only current-version checkpoints whose every shard
matches its declared digest. Final publication and checkpoint unlink are separate
crash points, so exact no-op recovery removes a leftover checkpoint only once the
validated final artifact proves matching fit semantics, layer coverage, and at
least its effective prompt progress; newer or different-corpus checkpoints stay
resumable.

Durability: payloads and directory entries are durable before any pointer is
published, and the pointer's parent directory is fsynced both before old shards
are collected and before an unlinked pointer's shards are removed, so power loss
cannot roll back to a pointer whose payload is gone. A per-model `jlens.fit` lock
spans preflight, estimator and checkpoint work, publication, and removal;
metadata-only preflight rejects incompatible corpora or layers before any matrix
IO. The default FineWeb-Edu streamer accepts a cancellation event, and on the
server path Hub resolution and iteration run in a spawn-only subprocess so a
blocked provider call can be terminated — spawn because forking an already-loaded
MPS process is unsafe and a detached thread leaves non-daemon workers behind.
Every lens `*_sha256` is a canonical lowercase 64-hex digest, and a
positive-progress checkpoint must carry the digest of its consumed token prefix.

Surface: `lens_paths` / `lens_checkpoint_paths` / `save_lens` /
`save_lens_checkpoint_accumulator` / `load_lens` / `load_lens_checkpoint_sidecar`
/ `load_lens_checkpoint` / `promote_lens_checkpoint` /
`remove_subsumed_lens_checkpoint` / `remove_lens`.

`lens_sources.py` owns `jlens/active.json` through `LENS_SOURCES`, an
`ActiveSourceRegistry` over the kinds `local` and `huggingface` (both addressed by
an artifact-name slug), plus commit-pinned external bindings under
`jlens/bindings/<name>.json` at `LENS_SOURCE_FORMAT_VERSION = 1`. It also owns
Neuronpedia discovery against `neuronpedia/jacobian-lens` and the weights-only
adapter that reads `.pt` payloads offline from the Hugging Face cache without
copying them into `SAKLAS_HOME`. Its Hub calls go through `io.hf`.

## sae.py / sae_artifacts.py

`sae.py` is the SAE family's source registry and local metadata cache.
`SAE_SOURCES` is an `ActiveSourceRegistry` (`SAE_SOURCE_FORMAT_VERSION = 1`) over
the kinds `local` and `saelens` — a local name obeys `NAME_REGEX`, a provider
release id is provider-shaped and only has to be non-blank.
`sae_source_release` is the one place the public `local:<name>` prefix convention
is applied, `load_active_sae` the one place a selection resolves to
`(release, provider_metadata)`. Under `models/<safe>/sae/bindings` it stores the
release/layer runtime binding plus the lazily fetched per-feature Neuronpedia
metadata (`<release>-features.json`, `{id: {label, max_act}}`, where `max_act` is
`maxActApprox` — the unit that normalizes the SAE strength channel to 0..1), both
at `SAE_RUNTIME_FORMAT_VERSION = 3`. Provider weights stay in the SAELens/Hugging
Face cache. `sae_artifacts.py` owns Saklas-trained fp32 weights under
`sae/local/<name>/` with their own manifest (`LOCAL_SAE_FORMAT_VERSION = 1`), and
never writes into a provider cache.

## hf.py / hf_manifolds.py

`hf.py` is the single Hugging Face seam: `HFError`,
`split_revision(owner/name@rev)` (names are `NAME_REGEX`-restricted, so `@` is
unambiguous), the three monkeypatchable indirections `_hf_snapshot_download` /
`_hf_hub_download` / `_hf_api`, and `resolve_target_coord(name, as_)`. Both
`hf_manifolds.py` and `lens_sources.py` import those three under exactly these
names, so patching either module's attribute blocks the network on that path. New
Hub work belongs here.

`hf_manifolds.py` is manifold distribution: the `saklas-manifold` tag on a
`repo_type="model"` repo, because safetensors is hub-native and `base_model`
frontmatter gives reverse-link discoverability. `pull_manifold` builds through
`stage_verify_swap` and **rejects** a repo with no `manifold.json` — the geometry
cannot be inferred from a bare tensor dump. `push_manifold(folder, coord, *,
private, model_scope, variant="raw", dry_run) → (repo_url, commit_sha)` always
uploads the corpus (a manifold cannot re-fit without it) and filters fitted pairs
by model scope and variant; the card carries `library_name: saklas`, the
`saklas-manifold` tags, deduped `base_model:`, and `base_model_relation: adapter`.
`search_manifolds` / `fetch_manifold_info` fill the picker fields under a fixed
row cap. `install_manifold(target, as_, *, force, on_progress)` orchestrates the
HF pull and the local-folder copy, with `ProgressCallback` narrating five stages
(resolve → download → validate → stage → swap) for the CLI and the SSE route.
`ManifoldInstallConflict` (409) is raised on an existing destination without
`force`.

Both HF and local staged installs rewrite `manifold.json::name` and every fitted
sidecar's repeated `name` to the destination basename, re-hash those sidecars,
then re-validate — so `--as ns/name` changes runtime identity, not just the
directory. A pair is eligible for that rewrite only when both its tensor and
original sidecar already carry source-manifest proofs; the rename never blesses a
partial pair. `push_manifold` snapshots its source under one
manifest-then-sorted-pair transaction, freezing the tensor candidate list before
those pair locks are acquired and releasing every local lock before any Hub
request. A force install whose source is already the resolved destination is an
exact no-op rather than self-deletion.

## gguf_io.py

The GGUF codec plus the manifold export driver — the only interchange export
saklas emits. `write_gguf_profile(profile, path, *, model_hint)` /
`read_gguf_profile(path)` round-trip a per-layer direction dict through
llama.cpp's control-vector convention (`general.architecture = "controlvector"`,
`controlvector.model_hint`, `controlvector.layer_count`, `direction.<idx>` fp32).
The `gguf` package is optional; the lazy import raises `GGUFNotInstalled`. Because
per-layer shares are baked into the stored magnitudes, llama.cpp's uniform
`--control-vector-scaled` scalar reproduces saklas's layer weighting with no extra
metadata slot; a unit-normed repeng vector reads back as uniform-per-layer, the
semantic it was exported with.

`export_gguf_manifold(ns, name, *, model_scope, output, model_hint)` folds each
fitted raw 2-node `pca` tensor to a single direction and writes one `.gguf` per
model. The preflight is metadata-only and each *selected* tensor is
integrity-checked when loaded, so unrelated models, SAE variants, and prior
exports are never eagerly hashed. In-folder export is refused for a bundled
manifold, whose folder is restored on refresh, and otherwise extends the folder's
integrity manifest with the new `.gguf` entries only, leaving the selector index
valid. `_resolve_model_hint` derives llama.cpp's architecture string from the
base model's `AutoConfig.model_type`, with `--model-hint` as the escape hatch.

## probes_bootstrap.py

`load_default_manifolds()` walks `manifolds/default/` into `{tag:
[manifold_name]}` — the category-grouped probe roster — triggering bundled
materialization first. `bootstrap_layer_means(...)` returns the per-layer
probe-centering means as `X.mean(0)` of the neutral-activation cache: the same
corpus, pooling, and dtype the whitener's covariance is built from, so there is no
separate `layer_means` artifact and a cold model pays one neutral-corpus forward
loop instead of two. The bundled probe *directions* are sourced by the session
folding fitted 2-node manifolds, not by this module.

## Deliberately absent

Rejected designs, listed once so they do not get re-proposed:

- **No pack format or datasource layer.** Concepts *are* manifolds and extraction
  takes node corpora directly; there is no `PackMetadata` / `ConceptFolder` /
  `pull_pack` surface.
- **No `pca` tensor variant and no method suffix.** Difference-of-means is the
  only vector extraction method, so the canonical raw tensor carries no method
  tag. `role-` is a selector suffix only, never a tensor-file kind.
- **No Euclidean fallback** in the fit, the share bake, or the transfer rebake. A
  missing or non-covering whitener raises `WhitenerError`; on real LMs the
  Euclidean metric is rogue-dominated, so a fallback would be a wrong answer
  rather than a degraded one.
- **No separate `layer_means` cache** — the neutral-activation cache is the one
  per-model neutral artifact.
- **No embedded `template` block** in `manifold.json`; a templated manifold
  carries `template_ref` plus its derived corpus.
- **No profile-level cross-model transfer.** Transfer runs through
  `transfer_manifold` → `core.manifold.transfer_manifold_subspaces`.
- **No caller-side selector invalidation.** Every io mutator calls
  `selectors.invalidate()` itself.
