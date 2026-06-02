# cli/

Eight-verb root parser
(`tui`/`serve`/`pack`/`subspace`/`manifold`/`vector`/`experiment`/`config`). 4.0
promoted `subspace` (flat — the old `vector` extract/merge/clone/compare/why/
transfer) and `manifold` (the old `vector manifold *` subtree) to top-level
verbs; `vector` is now a **deprecated alias** that parses the old tree
identically and dispatches to the new runners with a one-time stderr notice
(`_run_vector` → `_SUBSPACE_RUNNERS` / `_run_manifold`). Split across:
- `cli/main.py` — entry point, `parse_args`, `main`, `_COMMAND_RUNNERS` dispatch
- `cli/parsers.py` — `_build_root_parser` + every `_build_X_parser`, the verb tables
- `cli/runners.py` — every `_run_X` plus the shared helpers below
- `cli/config_file.py` — `ConfigFile` dataclass + `compose` / `apply_flag_overrides`
  / `ensure_vectors_installed`
- `cli/output.py` — text/JSON formatters for `pack ls` / `pack search`

`main()` dispatches via `_COMMAND_RUNNERS[cmd]`. Bare `saklas` (or a bare verb
with no subverb) prints help and exits 0, not argparse's exit 2. `subspace` and
`manifold` are top-level verbs; `vector` is the deprecated alias (`_run_vector`
warns then routes to `_SUBSPACE_RUNNERS` / `_run_manifold`).

## Verb nesting

- `pack` = distribution (install/refresh/clear/rm/ls/search/push/export) via
  `_PACK_VERBS` / `_PACK_BUILDERS` / `_PACK_RUNNERS`
- `subspace` = flat-artifact computation (extract/merge/clone/compare/why/transfer)
  via `_SUBSPACE_VERBS` / `_SUBSPACE_BUILDERS` / `_SUBSPACE_RUNNERS` (`_run_subspace`).
- `manifold` = the steering-manifold subtree promoted to top-level
  (`fit`/`discover`/`generate`/`merge`/`install`/`search`/`push`/`rm`/
  `clear`/`refresh`/`transfer`/`ls`/`show`), hand-dispatched by `_run_manifold`
  (`_build_manifold_parser` reuses `_build_vector_manifold`, so the verb and the
  `vector manifold` alias parse identically).
  The fit/discover/generate/transfer verbs load a model; the lifecycle verbs and
  `ls`/`show` are pure-IO over `~/.saklas/manifolds/`, addressed by
  `(namespace, name)` pairs (not the concept `Selector`/`resolve` machinery).
  Bare-name resolution splits by intent: verbs addressing an *existing* manifold
  (`clear`/`refresh`/`rm`/`transfer`, `discover`/`show`) resolve cross-namespace
  via `_resolve_manifold_ns_name` (reaching bundled `default/`, raising on
  collision/miss); verbs authoring a *fresh* folder (`generate`, `merge` target,
  `push`) default a bare name to `local/` via `_split_manifold_ns_name`.
- `experiment` = repeatable research runs (`fan`, `transcript run`, `naturalness`)
  via `_EXPERIMENT_VERBS` / `_EXPERIMENT_BUILDERS`; `_run_experiment` hand-dispatches.
- `config` = show / validate.

## Config loading

`_load_effective_config(args)` (in `runners.py`) is the shared entry point every
`-c`-taking subcommand calls. It composes `~/.saklas/config.yaml` + explicit `-c`
files via `ConfigFile.effective(extras, include_default=True)`, runs
`apply_flag_overrides` for CLI-supplied values, then stamps in place: `args.model`
(if YAML supplied it), `args.temperature`, `args.top_p`, `args.thinking`,
`args.system_prompt`, `args.max_tokens`, `args.config_vectors`, plus the YAML-only
knobs `args.projection_metric` / `args.compile` / `args.cuda_graphs` /
`args.top_k_alts` (from YAML `return_top_k`) — each only when the matching CLI flag
is unset (CLI wins). Finally calls `ensure_vectors_installed`. There is **no**
`method`/`injection_mode`/`theta_max` threading — those config keys were removed
with the unified injection kernel.

`ConfigFile.load` parses the YAML, warns on unknown keys, and validates the
`vectors:` value (a single steering expression) through `parse_expr` at load time.
`compose` overrides field-by-field (later wins; `vectors` wholesale). Known keys
(`_KNOWN_KEYS`): `model`, `vectors`, `thinking`, `temperature`, `top_p`,
`max_tokens`, `system_prompt`, `projection_metric`, `compile`, `cuda_graphs`,
`return_top_k`. `ensure_vectors_installed` walks the raw expression via
`referenced_selectors`, auto-installing HF concepts and materializing `default/`
ones; `strict=True` raises on any unresolvable reference.

## Session construction + warmup

`_make_session(args)` builds the `SaklasSession` via `from_pretrained`, threading
`device`, `quantize`, `probes`, `system_prompt`, `max_tokens`, `projection_metric`
(default `"mahalanobis"`), `dls`, `compile`, `cuda_graphs`, `return_top_k`. There
is no injection-mode resolution and no `--legacy` conflict check. `_warmup_session`
runs a 32-token stateless `session.generate(...)` so dynamo's shape promotion fires
on a realistic prefill before the user's first request; called from `tui` and
`serve` after `_setup_steering_vectors`. `_attach_default_manifold_probes(session)`
runs in `_run_serve` after `create_app`, gated on the dashboard being mounted
(`web_enabled`) — it attaches the bundled `default/personas` + `default/pad`
as read-side probes (fitted-for-model only; an unfitted one is skipped with a hint).

## Flags

`tui`/`serve` share model-loading args (`model`, `-q/--quantize`, `-d/--device`,
`-p/--probes`), the injection block (`_add_injection_args`), the logit block
(`_add_logit_args`), and config args (`_add_config_args`: `-c/--config` repeatable,
`-s/--strict`).

- **Injection block** (`_add_injection_args`, on `tui`/`serve`/`experiment fan`/
  `transcript run`/`naturalness`): `--projection-metric {mahalanobis,euclidean}`,
  `--no-dls`, `--compile`, `--cuda-graphs`. All argparse-default to `None`/`False`;
  YAML fills unset values, session defaults (mahalanobis / DLS on / compile +
  cuda-graphs **off**) win otherwise. There is **no** `--steer-mode`/`--theta-max`/
  `--legacy`.
- **Logit block** (`_add_logit_args`): `--top-k-alts N` (0–256, → session
  `SamplingConfig.return_top_k`).
- `tui`: `model` optional (a `-c` config with `model:` can supply it); `--max-tokens`
  default 1024.
- `serve`: `-H/--host` (default `0.0.0.0`), `-P/--port` (8000), `-S/--steer EXPR`,
  `-C/--cors ORIGIN` (repeatable), `-k/--api-key` (falls back to `$SAKLAS_API_KEY`),
  `--no-web`.
- `subspace extract`: positional `concept` (one concept or two poles, `nargs="+"`),
  `-m/--model`, `-f/--force` (pre-deletes the existing tensor + threads
  `force_statements=True`), `--sae RELEASE`, `--sae-revision REV`, `--role SLUG`
  (mutually exclusive with `--sae`; writes a `_role-<slug>` tensor + returns a
  `:role-<slug>` tail; slug `[a-z0-9._-]+`), `--namespace NS` (destination; unset →
  `local/`). There is **no `--method`/`--legacy`** — difference-of-means is the only
  method.
- `subspace merge`: `name` + `expression`, `-f`, `-s/--strict`, `-m/--model`.
- `subspace clone`: `corpus_path`, `-N/--name` (required), `-m/--model`,
  `-n/--n-pairs` (90), `--seed`, `-f`.
- `subspace compare`: `concepts` (1+), `-m/--model` (required), `-v`, `-j`,
  `--metric {euclidean,mahalanobis}` (default `None` → resolves to `mahalanobis`),
  `--ridge-scale` (1.0, mahalanobis only). No `--legacy`. 1-arg ranks all installed
  against the target, 2-arg pairwise, 3+ an N×N matrix; the mahalanobis path loads
  `LayerWhitener.from_cache` up front (a miss is fatal — no silent Euclidean
  fallback).
- `subspace why`: `concept`, `-m/--model` (required), `-j`. Per-layer `‖baked‖`
  histogram (16 buckets) + sidecar diagnostics.
- `subspace transfer`: `concept`, `--from SRC` / `--to TGT` (required), `-f`, `-j`.
  Fits/loads a Procrustes alignment, writes the target's `from-<safe_src>` tensor.
- `manifold`: top-level `fit`/`discover`/`generate`/`merge`/`install`/`search`/
  `push`/`rm`/`clear`/`refresh`/`transfer`/`ls`/`show`. `fit <folder>` runs
  `ManifoldExtractionPipeline` on an authored folder; `discover <name>
  [--method pca|spectral] [--max-dim N] [--var-threshold T] [--k-nn K]
  [--bandwidth SIGMA] [--max-subspace-dim R] [--sae REL]` fits a discover-mode
  folder (writes any CLI hyperparam override into `manifold.json` atomically
  *before* the fit; `--max-subspace-dim` caps the per-layer PCA dim,
  argparse-default `None` → engine 64). `generate <name> --concepts C...
  [--n-scenarios N] [--statements-per-concept K] [--seed INT] [--role-per-node]
  [-m] [-f]` LLM-authors a discover folder via `session.generate_statements`
  (`--role-per-node` doubles each concept slug as that node's assistant-role
  substitution → a persona manifold). Lifecycle/distribution verbs mirror their
  `pack`/`vector` precedents flag-for-flag. The only surviving `--method` flag is
  the manifold `pca`/`spectral` one (on `discover`/`merge`).
- `experiment fan`: `model` + `prompt`, `-g/--grid CONCEPT=ALPHAS` (required,
  repeatable), `-S/--base-steering`, `--max-tokens` (256), `-j`. Runs the grid
  through `generate_sweep`.
- `experiment transcript run`: `path` + optional `model`, `--max-tokens` (256).
  `transcript` is not a top-level verb.
- `experiment naturalness`: `model` + `prompt`, `--manifold FOLDER` / `-S/--steer
  EXPR` (required), `--compare-linear`, `--max-tokens` (128), `-j`.
- `pack install`/`refresh`/`clear`/`rm`/`ls`/`search`/`push`/`export gguf`,
  `config show`/`validate` — flags as in their `cache_ops`/`hf` backends.

## Error handling

`@_saklas_error_exit` wraps the top-level runners (including `_run_tui`): any
escaping `SaklasError` prints `user_message()` to stderr and exits with
`min(2, status // 100)`.
