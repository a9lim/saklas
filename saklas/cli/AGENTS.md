# cli/

Six-verb root parser
(`tui`/`serve`/`manifold`/`pack`/`experiment`/`config`). `manifold` is the unified
compute surface (extract/generate/fit/bake/merge/transfer/compare/why); `pack` is
the lifecycle/distribution verb (ls/show/install/search/push/rm/clear/refresh/
export gguf). There is no `vector` alias — install via `pack install` and export
via `pack export gguf`. Split across:
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
  corpus-union of discover-mode manifolds — two different verbs now. The
  extract/generate/fit/transfer verbs load a model.
- `pack` = lifecycle + distribution
  (`ls`/`show`/`install`/`search`/`push`/`rm`/`clear`/`refresh`/`export`),
  hand-dispatched by `_run_pack` (table `_PACK_VERBS`). `export gguf <name>` folds
  a fitted 2-node `pca` manifold to a steering vector and writes a llama.cpp
  control-vector GGUF, via `cache_ops._export_gguf_manifold`. These verbs are
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
  cache), `--sae RELEASE`, `--sae-revision REV`, `--role SLUG` (mutually exclusive
  with `--sae`; the role bakes into the node corpora and writes the **canonical**
  tensor — no `_role-` suffix — while returning a `:role-<slug>` name tail; slug
  `[a-z0-9._-]+`), `--namespace NS` (destination; unset → `local/`). There is **no
  `--method`/`--legacy`** — difference-of-means (a 2-node `pca` fit) is the only
  method.
- `manifold generate`: `name` + `--concepts C...` (required, ≥2),
  `[--kind {abstract,concrete}] [--samples-per-prompt K] [--seed INT]
  [--role-per-node] [-m] [-f]`. LLM-authors a discover folder via
  `session.generate_responses` — each node's corpus is in-character responses to
  the shared A2 baseline prompts (`--kind` selects the system template +
  elicitation role label; `--samples-per-prompt` is responses per baseline prompt,
  default 1). `--role-per-node` doubles each concept slug as that node's
  assistant-role substitution → a persona manifold.
- `manifold fit`: positional `target` (a manifold name *or* a folder path;
  `_run_manifold_fit` resolves it and reads `fit_mode`), `-m/--model`, `-f/--force`,
  `--sae RELEASE`, `--sae-revision REV`, plus the discover hyperparams
  `--method pca|spectral`, `--max-dim N`, `--var-threshold T`, `--k-nn K`,
  `--bandwidth SIGMA`, `--max-subspace-dim R`. An authored folder runs
  `ManifoldExtractionPipeline` directly; a discover folder (`pca`/`spectral`) has
  any supplied hyperparam written into `manifold.json` atomically *before* the fit.
  Supplying a discover hyperparam against an authored folder is an error.
  `--max-subspace-dim` caps the per-layer RBF subspace dim for the curved spectral
  fit (argparse-default `None` → engine 64) and is dropped by `_sanitize_hyperparams`
  for `--method pca` — a flat fit's subspace dim is its `--max-dim` layout dim. This
  verb folds the former separate `discover` verb.
- `manifold bake`: `name` + `expression`, `-f`, `-s/--strict`, `-m/--model`. Lands
  a corpus-less baked manifold via `merge_into_manifold`.
- `manifold merge`: `name` + `sources` (1+), `-f`. Unions the node corpora of
  discover-mode manifolds into a fresh folder — distinct from `bake` (which lowers a
  steering expression).
- `manifold transfer`: `name`, `--from SRC` / `--to TGT` (required), `-f`, `-j`.
  Fits/loads a Procrustes alignment and writes the target's `from-<safe_src>`
  **manifold** tensor via `transfer_manifold` — one transfer path (the old
  vector-side transfer bridge is gone). The runner calls
  `_target_whitener_from_neutral_cache` (→ `LayerWhitener.from_neutral_cache`) on the
  target model's cached `neutral_activations.safetensors` (no model load on a cache
  hit) so transferred shares are re-baked in the target Mahalanobis metric;
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
- `config show`/`validate` — flags as in `config_file`.

## Error handling

`@_saklas_error_exit` wraps the top-level runners (including `_run_tui`): any
escaping `SaklasError` prints `user_message()` to stderr and exits with
`min(2, status // 100)`.
