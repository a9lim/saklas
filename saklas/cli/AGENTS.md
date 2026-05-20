# cli/

Six-verb root parser (`tui`/`serve`/`pack`/`vector`/`experiment`/`config`) split across:
- `cli/main.py` — entry point, `parse_args`, `main`, `_COMMAND_RUNNERS` dispatch
- `cli/parsers.py` — `_build_root_parser` + every `_build_X_parser`
- `cli/runners.py` — every `_run_X` plus the shared helpers below
- `cli/config_file.py` — `ConfigFile` dataclass + `compose` / `apply_flag_overrides` / `ensure_vectors_installed`
- `cli/output.py` — text/JSON formatters for `pack ls` / `pack search`

`main()` dispatches via `_COMMAND_RUNNERS[cmd]`. Bare `saklas` (or a bare verb with no subverb) prints help and exits 0, not argparse's exit 2.

## Verb nesting

- `pack` = distribution (install/refresh/clear/rm/ls/search/push/export) via `_PACK_VERBS` / `_PACK_BUILDERS` / `_PACK_RUNNERS`
- `vector` = computation (extract/merge/clone/compare/why/transfer) via `_VECTOR_VERBS` / `_VECTOR_BUILDERS` / `_VECTOR_RUNNERS`
- `experiment` = repeatable research runs (`fan`, `transcript run`) via `_EXPERIMENT_VERBS` / `_EXPERIMENT_BUILDERS`; `_run_experiment` hand-dispatches the two verbs
- `config` = show / validate

## Config loading

`_load_effective_config(args)` (in `runners.py`) is the shared entry point every subcommand that takes `-c` calls. It composes `~/.saklas/config.yaml` + explicit `-c` files via `ConfigFile.effective(extras, include_default=True)`, runs `apply_flag_overrides` for CLI-supplied values, then stamps in place: `args.model` (if YAML supplied it), `args.temperature`, `args.top_p`, `args.thinking`, `args.system_prompt`, `args.max_tokens`, `args.config_vectors`, plus YAML-only knobs `args.method` / `args.injection_mode` / `args.theta_max` / `args.projection_metric` / `args.no_compile` / `args.no_cuda_graphs` / `args.top_k_alts` — each only when the matching CLI flag is unset (CLI wins). Finally calls `ensure_vectors_installed`.

`ConfigFile.load` parses the YAML, warns on unknown keys, and validates the `vectors:` value (a single steering expression string) through `saklas.core.steering_expr.parse_expr` at load time, storing the raw text. Re-parsing into a `Steering` happens on consumption. `compose` overrides field-by-field, later configs winning; `vectors` overrides wholesale (no concatenation). Known keys: `model`, `vectors`, `thinking`, `temperature`, `top_p`, `max_tokens`, `system_prompt`, `extraction_method`, `injection_mode`, `theta_max`, `projection_metric`, `compile`, `cuda_graphs`, `return_top_k`.

`ensure_vectors_installed` walks the raw expression via `referenced_selectors`, auto-installing HF-namespaced concepts and materializing `default/` ones; `strict=True` raises on any unresolvable reference instead of warning.

## Session construction + warmup

`_make_session(args)` builds the `SaklasSession` via `from_pretrained`, resolving probe categories, injection mode, projection metric, DLS, compile, and CUDA-graph settings off `args`. It enforces the `--legacy` conflict checks (mutually exclusive with `--steer-mode`, `--projection-metric`, `--no-dls`). `_warmup_session` runs a single-token `session.generate("Hi", sampling=SamplingConfig(max_tokens=1), stateless=True)` so the first real request is fast (`serve` only).

## Flags

`tui` and `serve` share model-loading args (`model`, `-q/--quantize`, `-d/--device`, `-p/--probes`), the injection block (`_add_injection_args`), the logit block (`_add_logit_args`), and config args (`_add_config_args`: `-c/--config` repeatable, `-s/--strict`).

- Injection block (`tui`/`serve`/`experiment fan`/`transcript run`): `--steer-mode {angular,additive}`, `--theta-max RAD`, `--projection-metric {mahalanobis,euclidean}`, `--no-dls`, `--legacy`, `--no-compile`, `--no-cuda-graphs`. All argparse-default to `None`/`False`; YAML fills unset values, session defaults (angular / π/2 / mahalanobis / DLS on / compile + cuda-graphs auto-on) win otherwise.
- Logit block: `--top-k-alts N` (0–256, default unset → session default 0). Sets the session-level `SamplingConfig.return_top_k`.
- `tui`: `model` is optional (a `-c` config with `model:` can supply it); `--max-tokens` default 1024.
- `serve`: `-H/--host` (default `0.0.0.0`), `-P/--port` (default 8000), `-S/--steer EXPR`, `-C/--cors ORIGIN` (repeatable), `-k/--api-key` (falls back to `$SAKLAS_API_KEY`), `--no-web` (skip the dashboard mount at `/`).
- `vector extract`: positional `concept` (one concept or two poles), `-m/--model`, `-f/--force`, `--method {dim,pca}` (default `dim`; `pca` writes the legacy `_pca` filename suffix), `--legacy` (≡ `--method pca`, mutually exclusive with `--method`), `--sae RELEASE`, `--sae-revision REV`. `--method`/`--legacy` resolve through `_resolve_legacy_method`.
- `vector merge`: positional `name` + `expression` (a steering expression, e.g. `"0.3 ns/a + 0.5 ns/a~ns/b"`), `-f/--force`, `-s/--strict`, `-m/--model`.
- `vector clone`: positional `corpus_path`, required `-N/--name`, `-m/--model`, `-n/--n-pairs` (default 90), `--seed`, `-f/--force`.
- `vector compare`: positional `concepts` (1+ selectors), required `-m/--model`, `-v/--verbose`, `-j/--json`, `--metric {euclidean,mahalanobis}` (default `mahalanobis`), `--ridge-scale FLOAT` (default 1.0, mahalanobis only), `--legacy` (≡ `--metric euclidean`). 1-arg mode ranks all installed against the target, 2-arg is pairwise, 3+ prints an N×N matrix. The mahalanobis path loads `LayerWhitener.from_cache(model_id)` up front; a missing whitener cache is fatal (no silent Euclidean fallback).
- `vector why`: positional `concept`, required `-m/--model`, `-j/--json`. Prints a per-layer `||baked||` histogram (16 buckets) plus diagnostics when the sidecar carries them.
- `vector transfer`: positional `concept`, required `--from SRC_MODEL` / `--to TGT_MODEL`, `-f/--force`, `-j/--json`. Fits/loads a Procrustes alignment and writes a transferred tensor at the target's `from-<safe_src>` variant.
- `experiment fan`: positional `model` + `prompt`, required repeatable `-g/--grid CONCEPT=ALPHAS`, `-S/--base-steering EXPR`, `--max-tokens` (default 256), `-j/--json`. Runs the alpha grid through `session.generate_sweep`; JSON mode emits `RunSet.to_dict()`.
- `experiment transcript run`: positional `path` + optional `model` (falls back to the transcript's embedded `model_id`), `--max-tokens` (default 256). Replays each user turn and reports per-turn readings drift. `transcript` is not a top-level verb.
- `pack install`: `target`, `-s/--statements-only`, `-a/--as NS/NAME`, `-f/--force`.
- `pack refresh`: `selector` (or the literal `neutrals`), `-m/--model`.
- `pack clear`: `selector`, `-m/--model`, `-y/--yes` (required for broad selectors), `--variant {raw,sae,all}` (default `all`).
- `pack rm`: `selector`, `-y/--yes` (required for broad selectors).
- `pack ls`: optional `selector`, `-j/--json`, `-v/--verbose` — local-only, no HF query.
- `pack search`: optional `query`, `-j/--json`, `-v/--verbose` — the HF-remote verb.
- `pack push`: `selector`, `-a/--as OWNER/NAME`, `-p/--private`, `-m/--model`, `-s/--statements-only`, `-n/--no-statements`, `-t/--tag-version`, `-d/--dry-run`, `-f/--force`, `--variant {raw,sae,all}` (default `raw` — SAE variants carry stronger provenance, so sharing them is opt-in).
- `pack export gguf`: `selector`, `-m/--model`, `-o/--output`, `--model-hint`.
- `config show`: `-c/--config` (extra YAML), `-m/--model` (override), `--no-default` (skip `~/.saklas/config.yaml`). `config validate`: positional `file` — exit 0 valid, 2 invalid.

`--legacy` is a single-flag preset: on `tui`/`serve` it forces `injection_mode="additive"`, `extraction_method="pca"`, `projection_metric="euclidean"`, `dls=False`; on `vector extract` it forces `--method pca`; on `vector compare` it forces `--metric euclidean`. Conflicting per-flag controls on the same verb error at parse/runner time before model load.

## Error handling

`@_saklas_error_exit` wraps the top-level runners (not `tui`): any escaping `SaklasError` prints `user_message()` to stderr and exits with `min(2, status // 100)`.
