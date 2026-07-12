# cli/

Nine-verb root parser
(`tui`/`serve`/`manifold`/`pack`/`experiment`/`config`/`template`/`lens`/`sae`).
`manifold` is the
unified compute surface (extract/generate/from-template/fit/bake/merge/transfer/
compare/why); `pack` is the lifecycle/distribution verb (ls/show/install/search/
push/rm/clear/refresh/export gguf); `template` owns the standalone
templated-completion artifact (create/ls/show/score/rm); `lens` owns the
per-model Jacobian-lens artifact (fit/show/top/decompose/rm). There is no
`vector` alias
— install via `pack install` and export via `pack export gguf`. Split across:
- `cli/main.py` — entry point, `parse_args`, `main`, `_COMMAND_RUNNERS` dispatch
- `cli/parsers.py` — `_build_root_parser` + every `_build_X_parser`, the verb tables
- `cli/runners.py` — every `_run_X` plus the shared helpers below
- `cli/config_file.py` — `ConfigFile` dataclass + `compose` / `apply_flag_overrides`
  / `ensure_vectors_installed`

`main()` dispatches via `_COMMAND_RUNNERS[cmd]`. Bare `saklas` (or a bare verb
with no subverb) prints help and exits 0, not argparse's exit 2.

## Verb nesting

- `manifold` = the unified compute surface
  (`extract`/`generate`/`fit`/`bake`/`merge`/`transfer`/`compare`/`why`),
  hand-dispatched by `_run_manifold` (table `_MANIFOLD_VERBS`). `fit` absorbs the
  former `discover` verb: its positional is a name-or-folder, `_run_manifold_fit`
  resolves it and reads `fit_mode`, and the discover hyperparams
  (`--method`/`--max-dim`/`--var-threshold`/`--k-nn`/`--bandwidth`/
  `--max-subspace-dim`) apply only when the resolved folder is discover-mode
  (`pca`/`spectral`), erroring against an authored folder. `bake` (the former
  `merge` compute verb) lands a corpus-less baked manifold from a steering
  expression via `merge_into_manifold`; the present `merge` is the *distinct*
  corpus-union of discover-mode manifolds — two different verbs now.
  `from-template` (`_run_manifold_from_template`) is pure-IO: it resolves a
  standalone template and writes a discover folder that derives its corpus from
  the template + stores `template_ref` (replaces the former `manifold template`
  embedded-block verb). The extract/generate/fit/transfer verbs load a model.
- `pack` = lifecycle + distribution
  (`ls`/`show`/`install`/`search`/`push`/`rm`/`clear`/`refresh`/`export`),
  hand-dispatched by `_run_pack` (table `_PACK_VERBS`). `export gguf <name>` folds
  a fitted 2-node `pca` manifold to a steering vector and writes a llama.cpp
  control-vector GGUF, via `cache_ops.export_gguf_manifold`. These verbs are
  pure-IO over `~/.saklas/manifolds/`, addressed by `(namespace, name)` pairs —
  no model load (except none; `search`/`install` hit HF).
- Bare-name resolution is shared by both dispatchers and splits by intent: verbs
  addressing an *existing* manifold (`manifold transfer`, `pack clear`/`refresh`/
  `rm`/`export`/`show`) resolve cross-namespace via `_resolve_manifold_ns_name`
  (reaching bundled `default/`, raising on collision/miss); verbs authoring a
  *fresh* folder (`manifold merge` target, `pack push`) default a bare name to
  `local/` via `_split_manifold_ns_name` — `manifold generate` does the same bare
  → `local/` split inline.
- `experiment` = repeatable research runs (`fan`, `transcript run`, `naturalness`)
  via `_EXPERIMENT_VERBS`; `_run_experiment` hand-dispatches.
- `config` = show / validate.
- `template` = the standalone templated-completion artifact
  (`create`/`ls`/`show`/`score`/`rm`) via `_TEMPLATE_VERBS`; `_run_template`
  hand-dispatches. `create` reads a `--contexts` JSON file (each entry a multi-turn
  `{turns, assistant}` or the single-turn `{user, assistant}` sugar normalized by
  `_normalize_context_entry`) and writes `~/.saklas/templates/<ns>/<name>/`;
  `ls`/`show`/`rm` are pure-IO. `score <name> -m MODEL [-S EXPR] [--by sum|mean]`
  loads a model and prints the per-context restricted-choice value distribution
  (`session.score_template`), steering-aware via `-S`. Bare names default to
  `local/` (`_split_manifold_ns_name`); `score`/`show`/`rm` resolve cross-namespace
  via `resolve_template`.
- `lens` = the per-model Jacobian-lens artifact
  (`fit`/`show`/`top`/`decompose`/`rm`) via `_LENS_VERBS`; `_run_lens`
  hand-dispatches (`@_saklas_error_exit`-wrapped, like `_run_experiment` — the
  lens error family carries `user_message()`). `fit`/`top`/`decompose` load a
  model with `probes=[]` (no default probe bootstrap); `show`/`rm` are pure-IO
  over `models/<safe_id>/jlens.*`. `fit` sources its corpus from `--corpus FILE`
  (one document per line, or JSONL with a `text` field) or streams the default
  fineweb-edu sample via the optional `datasets` dependency
  (`_load_lens_corpus`), and resumes a matching partial fit by default. The
  model is a **positional** on `fit`/`show`/`top`/`rm` (the artifact is
  per-model); `decompose` takes a selector positional + required `-m`.

## Config loading

`_load_effective_config(args)` (in `runners.py`) is the shared entry point every
`-c`-taking subcommand calls. It composes `~/.saklas/config.yaml` + explicit `-c`
files via `ConfigFile.effective(extras, include_default=True)`, runs
`apply_flag_overrides` for CLI-supplied values, then stamps in place: `args.model`
(if YAML supplied it), `args.temperature`, `args.top_p`, `args.thinking`,
`args.system_prompt`, `args.max_tokens`, `args.config_vectors`, plus the YAML-only
knobs `args.compile` / `args.cuda_graphs` / `args.top_k_alts` (from YAML
`return_top_k`) — each only when the matching CLI flag is unset (CLI wins). Finally
calls `ensure_vectors_installed`. There is **no** `method`/`injection_mode`/
`theta_max`/`projection_metric` threading — those config keys were removed with the
unified injection kernel and the Mahalanobis-only collapse.

`ConfigFile.load` parses the YAML, warns on unknown keys, and validates the
`vectors:` value (a single steering expression) through `parse_expr` at load time.
`compose` overrides field-by-field (later wins; `vectors` wholesale). Known keys
(`_KNOWN_KEYS`): `model`, `vectors`, `thinking`, `temperature`, `top_p`,
`max_tokens`, `system_prompt`, `compile`, `cuda_graphs`,
`return_top_k`. `ensure_vectors_installed` walks the raw expression via
`referenced_selectors`, auto-installing HF concepts and materializing `default/`
ones; `strict=True` raises on any unresolvable reference.

## Session construction + warmup

`_make_session(args)` builds the `SaklasSession` via `from_pretrained`, threading
`device`, `quantize`, `probes`, `system_prompt`, `max_tokens`, `dls`, `compile`,
`cuda_graphs`, `return_top_k`. There is no projection-metric knob (`~`/`|` is
Mahalanobis-only), no injection-mode resolution, and no `--legacy` conflict check.
`_warmup_session`
runs a 32-token stateless `session.generate(...)` so dynamo's shape promotion fires
on a realistic prefill before the user's first request; called from `tui` and
`serve` after `_setup_steering_vectors`. There is no serve-side probe-attach step:
the default probe roster — tagged concept axes plus every fitted bundled multi-node
manifold (`personas`, `emotions`) — is attached at session construction
(`_bootstrap_manifold_probes`, `core/session.py`) in every frontend, so the
dashboard rack opens already watching them. `_resolve_probes` maps unset / `all` →
`None` (the session's default-roster signal), `none` / `[]` → `[]`, and an explicit
category list through verbatim (tagged concepts only, no multi-node sweep).

## Flags

`tui`/`serve` share model-loading args (`model`, `-q/--quantize`, `-d/--device`,
`-p/--probes`), the injection block (`_add_injection_args`), the logit block
(`_add_logit_args`), and config args (`_add_config_args`: `-c/--config` repeatable,
`-s/--strict`).

- **Injection block** (`_add_injection_args`, on `tui`/`serve`/`experiment fan`/
  `transcript run`/`naturalness`): `--no-dls`, `--compile`, `--cuda-graphs`. All
  argparse-default to `None`/`False`; YAML fills unset values, session defaults
  (DLS on / compile + cuda-graphs **off**) win otherwise. `~`/`|` projection is
  Mahalanobis-only — there is **no** `--projection-metric`/`--steer-mode`/
  `--theta-max`/`--legacy`.
- **Logit block** (`_add_logit_args`): `--top-k-alts N` (→ session
  `SamplingConfig.return_top_k`). The `[0,256]` bound is validated only on the YAML
  `return_top_k` key, not on this flag (a plain `type=int`).
- `tui`: `model` optional (a `-c` config with `model:` can supply it); `--max-tokens`
  default 1024.
- `serve`: `-H/--host` (default `0.0.0.0`), `-P/--port` (8000), `-S/--steer EXPR`,
  `-C/--cors ORIGIN` (repeatable), `-k/--api-key` (falls back to `$SAKLAS_API_KEY`),
  `--no-web`.
- `manifold extract`: positional `concept` (one concept or two poles, `nargs="+"`),
  `-m/--model`, `-f/--force` (re-authors the pole corpora + bypasses the tensor
  cache), `--sae RELEASE`, `--role SLUG` (mutually exclusive
  with `--sae`; the role bakes into the node corpora and writes the **canonical**
  tensor — no `_role-` suffix — while returning a `:role-<slug>` name tail; slug
  `[a-z0-9._-]+`), `--namespace NS` (destination; unset → `local/`). There is **no
  `--method`/`--legacy`** — difference-of-means (a 2-node `pca` fit) is the only
  method. The loaded session/pipeline validates model, corpus, tokenizer, role,
  SAE-transform, and manifest identity before accepting a cache hit; extraction
  constructs that session with `probes=[]` so probe bootstrap is not part of
  artifact training.
- `manifold generate`: `name` + `--concepts C...` (required, ≥2),
  `[--kind {abstract,concrete,custom}] [--system TEMPLATE] [--samples-per-prompt K]
  [--seed INT] [--role-per-node] [-m] [-f]`. LLM-authors a discover folder via
  `session.generate_responses` — each node's corpus is in-character responses to
  the shared A2 baseline prompts (`--kind` selects the system template +
  elicitation role label: abstract → `someone {c}`, concrete → `{art} {c}`,
  `custom` → the `--system` template ({c} = concept, no role swap, required when
  `--kind custom`); `--samples-per-prompt` is responses per baseline prompt,
  default 1). `--role-per-node` doubles each concept slug as that node's
  assistant-role substitution → a persona manifold.
- `manifold fit`: positional `target` (a manifold name *or* a folder path;
  `_run_manifold_fit` resolves it and reads `fit_mode`), `-m/--model`, `-f/--force`,
  `--sae RELEASE`, `--layers L1,L2,...|workspace|all`
  (default all; subset artifacts contain only those injection layers), plus the discover hyperparams
  `--method pca|spectral|auto`, `--max-dim N`, `--min-dim N`, `--var-threshold T`,
  `--k-nn K`, `--bandwidth SIGMA`, `--max-subspace-dim R`, `--smoothing auto|0|LAMBDA`,
  `--persistence-frac F`. `-f/--force` bypasses the per-model tensor cache and
  re-pools/re-fits unconditionally — threaded `args.force` → `session.fit` →
  `ManifoldExtractionPipeline.fit(force=)`, which skips the `nodes_sha256`
  cache-hit. Needed because `manifold fit` (unlike `manifold extract -f`) doesn't
  re-author the corpus, so without it an unchanged corpus always cache-hits and a
  code-level fit change (e.g. a topology-selection fix) can't be picked up. An
  authored folder runs `ManifoldExtractionPipeline` directly; a discover folder
  (`pca`/`spectral`/`auto`) passes only the supplied override patch into
  `session.fit`; the pipeline merges and writes it under the same manifest lock
  as cache-key derivation and fit publication. Supplying a discover hyperparam
  against an authored folder is an error. Cache validation happens after model
  load so the sidecar can be checked against the actually loaded weight
  fingerprint (a mutable model id alone cannot prove a hit); actual fit sessions
  use `probes=[]` so artifact training does not eagerly
  bootstrap the unrelated probe roster.
  `--method auto` defers flat-vs-curved + periodic-axis selection to
  `select_topology` per-model. `--max-subspace-dim` caps the per-layer RBF subspace
  dim for the curved spectral fit (argparse-default `None` → engine 64) and is
  dropped by `sanitize_hyperparams` for `--method pca` — a flat fit's subspace dim
  is its `--max-dim` layout dim. `--min-dim` (spectral only) floors the intrinsic
  dim the eigenvalue-ratio cliff picks (set `--min-dim == --max-dim` to pin it,
  e.g. PAD's 3); ignored for `--method pca`. `--smoothing` (curved only: GCV `auto` / exact `0`
  / fixed λ) sets the penalized-RBF regularization; `--persistence-frac` (auto
  only) is the H1 loop-significance threshold. This verb folds the former separate
  `discover` verb.
- `manifold bake`: `name` + `expression`, `-f`, `-s/--strict`, `-m/--model`. Lands
  a corpus-less baked manifold via `merge_into_manifold`; accepts only
  namespace-qualified additive/subtractive scalar terms. Dynamic terms and
  Mahalanobis `~`/`|` projections are rejected without a live whitener.
- `manifold merge`: `name` + `sources` (1+), `-f`. Unions the node corpora of
  discover-mode manifolds into a fresh folder — distinct from `bake` (which lowers a
  steering expression).
- `manifold transfer`: `name`, `--from SRC` / `--to TGT` (required), `-f`, `-j`.
  Fits/loads a Procrustes alignment and writes the target's `from-<safe_src>`
  **manifold** tensor via `transfer_manifold` — one transfer path (the old
  vector-side transfer bridge is gone). The runner materializes the target's
  proven neutral cache only for source-tensor layers and builds the whitener from
  those resident rows. A cached alignment loads no model; if requested map layers
  are missing and both
  neutral-cache source identities remain proven, the runner fits and saves the
  missing factors offline from those selected cached matrices, carrying forward
  the other immutable factor shards. The per-model neutral single-flight lock is
  acquired before model construction, so distinct concurrent alignments sharing
  a cold model do not duplicate that model load. `-f` recomputes the requested
  alignment factors and target transfer but reuses exact neutral caches; it does
  not force the 90-prompt neutral capture. A first cold capture still publishes
  the complete reusable neutral artifact, but once shared coverage is known the
  runner narrows both in-memory seed rosters to requested layers and releases
  each full roster before Procrustes. Transferred shares are re-baked in the
  target Mahalanobis metric;
  **mandatory** — a missing/unusable target cache raises `WhitenerError` (the runner
  exits 1 with a regenerate-neutrals hint), there is no Euclidean rebake.
- `manifold compare`: `concepts` (1+), `-m/--model` (required), `-v`, `-j`,
  `--ridge-scale` (1.0). No `--metric`/`--legacy` — compare is **Mahalanobis-only**.
  1-arg ranks all installed against the target, 2-arg pairwise, 3+ an N×N matrix; it
  loads `LayerWhitener.from_cache` up front (a miss is fatal — there is no Euclidean
  path). Concepts fold from their 2-node manifolds.
- `manifold why`: `concept`, `-m/--model` (required), `-j`. Per-layer `‖baked‖`
  histogram (16 buckets) + sidecar diagnostics.
- `pack`: lifecycle + inspection. `ls [--namespace NS] [-v|-j]`, `show <name> [-j]`,
  `install <target> [-a NS/N] [-f]` (pulls an HF manifold or copies a local folder,
  salvaging a legacy saklas-pack repo), `search [query] [-v|-j]`,
  `push <selector> [-a OWNER/N] [-m] [-p/--private] [-d/--dry-run]
  [--variant raw|sae|all]` (HF upload, CLI-only), `rm <selector> [-y]`,
  `clear <selector> [-m] [--variant raw|sae|all]` (drop per-model fitted tensors;
  default `all`), `refresh <selector> [-m]`,
  `export gguf <name> [-m MODEL] [-o PATH] [--model-hint HINT]` (folds a fitted
  2-node `pca` manifold to a vector and writes a control-vector GGUF). The only
  surviving `--method` flag is the manifold `pca`/`spectral` one (on
  `manifold fit`/`manifold merge`).
- `experiment fan`: `model` + `prompt`, `-g/--grid CONCEPT=ALPHAS` (required,
  repeatable), `-S/--base-steering`, `--max-tokens` (256), `-j`. Runs the grid
  through `generate_sweep`.
- `experiment transcript run`: `path` + optional `model`, `--max-tokens` (256).
  `transcript` is not a top-level verb.
- `experiment naturalness`: `model` + `prompt`, `--manifold FOLDER` / `-S/--steer
  EXPR` (required), `--compare-linear`, `--max-tokens` (128), `-j`.
  Its behavior-manifold preflight consumes only authoring geometry/corpus, so it
  does not hash unrelated fitted payloads.
- `config show`/`validate` — flags as in `config_file`.
- `lens fit`: positional `model`, `--corpus FILE`, `--prompts N` (100),
  `--seq-len T` (128), `--dim-batch K` (8; total backward work is K-invariant,
  so the knob trades memory for per-pass overhead), `--prompt-batch B`
  (consecutive ragged prompts per graph; CPU/CUDA default 4, MPS 2; both widths
  halve independently on OOM and stay below a proven ceiling),
  `--checkpoint-every N` (25,
  writes a self-contained checkpoint directly from the live accumulator; the
  full artifact is written durably once at finalization),
  `--layers L1,L2,...|workspace` (restrict source layers — skips
  all forward-graph and backward work below the lowest one, the one real
  wall-time lever; `sample` is rejected for fitting because it still includes
  layer 0 and is artifact-size/debug only), `-f/--force` (restart from zero
  instead of resuming), `-d`, `-q`. All five numeric fit flags are positive-only.
  An exact metadata-only no-op can run before corpus streaming/model load when
  the sidecar's immutable model-source identity, pinned default-dataset revision
  (or custom raw-corpus hash), layer coverage, and tensor digest all match;
  otherwise loaded token IDs + live weights decide. That exact proof also reaps
  a crash-left checkpoint when the durable final artifact provably subsumes it.
  A superset stored lens satisfies
  narrower layer requests without refit (a fresh session reads only those
  shards and leaves the durable union intact), and missing layers are fitted as
  a checkpointed/resumable top-up while preserving the same-corpus union. A
  normal 100→1000 corpus extension resumes when the saved token-id hash matches
  the new prefix; extending only a strict subset of a v4 durable superset is
  rejected because v4 has one progress field for all layers — request the full
  durable set or use `-f` for explicit replacement. Model
  fingerprint mismatch forces a clean fit.
- `lens show`: positional `model`, `-j`.
- `lens top`: positionals `model` + `prompt` (raw text, no chat template),
  `-k/--top-k` (8), `--layers L1,L2,...` (default: 9 evenly spaced fitted
  layers), `--position P` (repeatable, negative ok; default final position),
  `-d`, `-q`, `-j`. Output leads with the layer-aggregated block (`token ·
  strength · com ±spread`, computed over the workspace-band subset of the
  displayed layers via `jlens_readout(aggregate=True)` — same forward), then
  the per-layer matrix; JSON carries both under `aggregate` / `layers`.
- `lens decompose`: positional `selector`, `-m/--model` (required),
  `-k/--top-k` (16 — the sparsity budget), `--layers L1,L2,...`, `-d`, `-q`,
  `-j`.
- `lens rm`: positional `model`, `-y/--yes`.

## Error handling

`@_saklas_error_exit` wraps the top-level runners (including `_run_tui`): any
escaping `SaklasError` prints `user_message()` to stderr and exits with
`min(2, status // 100)`.
