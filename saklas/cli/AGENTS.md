# cli/

Six-verb root parser
(`tui`/`serve`/`subspace`/`manifold`/`experiment`/`config`). `subspace` is the
flat-artifact computation surface (extract/merge/compare/why/transfer);
`manifold` owns the steering-manifold tree. There is no `pack` verb and no `vector`
alias — pack distribution folded into the manifold artifact, so install via
`manifold install` and export via `manifold export gguf`. Split across:
- `cli/main.py` — entry point, `parse_args`, `main`, `_COMMAND_RUNNERS` dispatch
- `cli/parsers.py` — `_build_root_parser` + every `_build_X_parser`, the verb tables
- `cli/runners.py` — every `_run_X` plus the shared helpers below
- `cli/config_file.py` — `ConfigFile` dataclass + `compose` / `apply_flag_overrides`
  / `ensure_vectors_installed`

`main()` dispatches via `_COMMAND_RUNNERS[cmd]`. Bare `saklas` (or a bare verb
with no subverb) prints help and exits 0, not argparse's exit 2.

## Verb nesting

- `subspace` = flat-artifact computation (extract/merge/compare/why/transfer)
  via `_SUBSPACE_VERBS` / `_SUBSPACE_RUNNERS` (`_run_subspace`).
- `manifold` = the steering-manifold tree
  (`fit`/`discover`/`generate`/`merge`/`install`/`search`/`push`/`rm`/
  `clear`/`refresh`/`transfer`/`ls`/`show`/`export`), hand-dispatched by
  `_run_manifold` (table `_MANIFOLD_VERBS`). `export gguf <name>` folds a fitted
  2-node `pca` manifold to a steering vector and writes a llama.cpp control-vector
  GGUF (the successor to the old `pack export gguf`), via
  `cache_ops._export_gguf_manifold`. The fit/discover/generate/transfer verbs load
  a model; the lifecycle verbs and `ls`/`show` are pure-IO over
  `~/.saklas/manifolds/`, addressed by `(namespace, name)` pairs. Bare-name
  resolution splits by intent: verbs addressing an *existing* manifold
  (`clear`/`refresh`/`rm`/`transfer`, `discover`/`show`) resolve cross-namespace via
  `_resolve_manifold_ns_name` (reaching bundled `default/`, raising on
  collision/miss); verbs authoring a *fresh* folder (`generate`, `merge` target,
  `push`) default a bare name to `local/` via `_split_manifold_ns_name`.
- `experiment` = repeatable research runs (`fan`, `transcript run`, `naturalness`)
  via `_EXPERIMENT_VERBS`; `_run_experiment` hand-dispatches.
- `config` = show / validate.

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
`serve` after `_setup_steering_vectors`. `_attach_default_manifold_probes(session)`
runs in `_run_serve` after `create_app`, gated on the dashboard being mounted
(`web_enabled`) — it attaches each bundled `default/<name>` manifold already fitted
for the loaded model as a read-side probe (`personas` / `pad` plus any fitted
concept axes); an unfitted one is skipped with a one-line hint.

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
- **Logit block** (`_add_logit_args`): `--top-k-alts N` (0–256, → session
  `SamplingConfig.return_top_k`).
- `tui`: `model` optional (a `-c` config with `model:` can supply it); `--max-tokens`
  default 1024.
- `serve`: `-H/--host` (default `0.0.0.0`), `-P/--port` (8000), `-S/--steer EXPR`,
  `-C/--cors ORIGIN` (repeatable), `-k/--api-key` (falls back to `$SAKLAS_API_KEY`),
  `--no-web`.
- `subspace extract`: positional `concept` (one concept or two poles, `nargs="+"`),
  `-m/--model`, `-f/--force` (re-authors the pole corpora + bypasses the tensor
  cache), `--sae RELEASE`, `--sae-revision REV`, `--role SLUG` (mutually exclusive
  with `--sae`; writes a `_role-<slug>` tensor + returns a `:role-<slug>` tail; slug
  `[a-z0-9._-]+`), `--namespace NS` (destination; unset → `local/`). There is **no
  `--method`/`--legacy`** — difference-of-means (a 2-node `pca` fit) is the only
  method.
- `subspace merge`: `name` + `expression`, `-f`, `-s/--strict`, `-m/--model`. Lands
  a corpus-less baked manifold via `merge_into_manifold`.
- `subspace compare`: `concepts` (1+), `-m/--model` (required), `-v`, `-j`,
  `--ridge-scale` (1.0). No `--metric`/`--legacy` — compare is **Mahalanobis-only**.
  1-arg ranks all installed against the target, 2-arg pairwise, 3+ an N×N matrix; it
  loads `LayerWhitener.from_cache` up front (a miss is fatal — there is no Euclidean
  path). Concepts fold from their 2-node manifolds.
- `subspace why`: `concept`, `-m/--model` (required), `-j`. Per-layer `‖baked‖`
  histogram (16 buckets) + sidecar diagnostics.
- `subspace transfer`: `concept`, `--from SRC` / `--to TGT` (required), `-f`, `-j`.
  A vector is a 2-node `pca` manifold, so `_run_transfer` delegates to
  `_run_manifold_transfer` (the `concept` positional is bridged onto `name`) — one
  transfer path. It fits/loads a Procrustes alignment and writes the target's
  `from-<safe_src>` **manifold** tensor via `transfer_manifold`. The runner calls
  `_target_whitener_from_neutral_cache` (→ `LayerWhitener.from_neutral_cache`) on the
  target model's cached `neutral_activations.safetensors` (no model load on a cache
  hit) so transferred shares are re-baked in the target Mahalanobis metric;
  **mandatory** — a missing/unusable target cache raises `WhitenerError` (the runner
  exits 1 with a regenerate-neutrals hint), there is no Euclidean rebake.
- `manifold`: top-level `fit`/`discover`/`generate`/`merge`/`install`/`search`/
  `push`/`rm`/`clear`/`refresh`/`transfer`/`ls`/`show`/`export`. `fit <folder>` runs
  `ManifoldExtractionPipeline` on an authored folder; `discover <name>
  [--method pca|spectral] [--max-dim N] [--var-threshold T] [--k-nn K]
  [--bandwidth SIGMA] [--max-subspace-dim R] [--sae REL]` fits a discover-mode
  folder (writes any CLI hyperparam override into `manifold.json` atomically
  *before* the fit; `--max-subspace-dim` caps the per-layer RBF subspace dim for
  the curved spectral fit, argparse-default `None` → engine 64, and is dropped by
  `_sanitize_hyperparams` for `--method pca` — a flat fit's subspace dim is its
  `--max-dim` layout dim). `generate <name> --concepts C...
  [--kind {abstract,concrete}] [--samples-per-prompt K] [--seed INT]
  [--role-per-node] [-m] [-f]` LLM-authors a discover folder via
  `session.generate_responses` — each node's corpus is in-character responses to
  the shared A2 baseline prompts (`--kind` selects the system template +
  elicitation role label; `--samples-per-prompt` is responses per baseline prompt,
  default 1). `--role-per-node` doubles each concept slug as that node's
  assistant-role substitution → a persona manifold. `install <target> [-a NS/N]
  [-f]` pulls an HF
  manifold or copies a local folder (and salvages a legacy saklas-pack repo).
  `export gguf <name> [-m MODEL] [-o PATH] [--model-hint HINT]` folds a fitted
  2-node `pca` manifold to a vector and writes a control-vector GGUF. The only
  surviving `--method` flag is the manifold `pca`/`spectral` one (on
  `discover`/`merge`).
- `experiment fan`: `model` + `prompt`, `-g/--grid CONCEPT=ALPHAS` (required,
  repeatable), `-S/--base-steering`, `--max-tokens` (256), `-j`. Runs the grid
  through `generate_sweep`.
- `experiment transcript run`: `path` + optional `model`, `--max-tokens` (256).
  `transcript` is not a top-level verb.
- `experiment naturalness`: `model` + `prompt`, `--manifold FOLDER` / `-S/--steer
  EXPR` (required), `--compare-linear`, `--max-tokens` (128), `-j`.
- `config show`/`validate` — flags as in `config_file`.

## Error handling

`@_saklas_error_exit` wraps the top-level runners (including `_run_tui`): any
escaping `SaklasError` prints `user_message()` to stderr and exits with
`min(2, status // 100)`.
