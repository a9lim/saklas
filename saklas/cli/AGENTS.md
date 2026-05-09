# cli/

Five-verb root parser (`tui`/`serve`/`pack`/`vector`/`config`) split across:
- `cli/main.py` — entry point, `_build_root_parser`, `_load_effective_config`, `_warmup_session`
- `cli/parsers.py` — every `_build_X_parser`
- `cli/runners.py` — every `_run_X`
- `cli/config_file.py` — `ConfigFile` + `effective()` / `resolve_poles()` / `to_yaml()`

## Verb nesting

- `pack` = distribution (install/refresh/clear/rm/ls/search/push/export) via `_PACK_BUILDERS` / `_PACK_RUNNERS` tables
- `vector` = computation (extract/merge/clone/compare/why) via `_VECTOR_BUILDERS` / `_VECTOR_RUNNERS`
- `config` = show / validate

## Config loading

`_load_effective_config(args)` is the shared entry point every subcommand that takes `-c` calls. Composes `~/.saklas/config.yaml` + explicit `-c` files and stamps `args.config_vectors` (a steering expression string, or `None`) / `args.temperature` / `args.top_p` / `args.max_tokens` / `args.method` (extraction) / `args.injection_mode` / `args.theta_max` in place. The `vectors:` YAML key is a single steering expression — parsed through `saklas.core.steering_expr.parse_expr` which resolves bare poles (`wolf → deer.wolf @ -0.5`) via `io.selectors.resolve_pole`. `ConfigFile.load` validates the expression at load time and stores the raw string; re-parsing happens on consumption.

## Warmup

`_warmup_session` issues a single-token `session.generate(sampling=SamplingConfig(max_tokens=1), stateless=True)` — no `session.config` mutation.

## Flags

- `tui` / `serve`: `--steer-mode {angular,additive}` (default `angular`), `--theta-max RAD` (default π/2 ≈ 1.5708), and `--projection-metric {mahalanobis,euclidean}` (default `mahalanobis`, since v2.1) flow into `SaklasSession.from_pretrained` via `_make_session` and stamp the session-level injection mode + runtime projection metric. All three default to `None` on argparse; YAML `injection_mode:`, `theta_max:`, and `projection_metric:` win when the matching CLI flag is unset; session defaults (angular / π/2 / mahalanobis) win otherwise.
- `serve`: `--host/-H`, `--port/-P`, `--steer/-S EXPR`, `--cors/-C`, `--api-key/-k`, plus `-c/--config`. `--steer` takes one steering expression string (the shared grammar); combine multiple terms with `+`/`-`.
- `vector extract`: `--method {dim,pca}` (default `dim`) selects the per-layer extractor. Tensor filenames diverge: `--method dim` writes to `<safe_model>.safetensors` (canonical); `--method pca` writes to `<safe_model>_pca.safetensors` (legacy). Both can coexist on disk; the steering grammar's `:pca` variant addresses the legacy tensor.
- `vector compare`: `--metric {euclidean,mahalanobis}` (default `mahalanobis` since v2.1). Mahalanobis path requires `~/.saklas/models/<id>/{layer_means,neutral_activations}.safetensors` to exist on disk; `LayerWhitener.from_cache(model_id)` raises a `WhitenerError` with a populating-command hint when the cache is missing (this is fatal — `compare --metric mahalanobis` doesn't silently fall back to Euclidean since that would hide the missing cache). `--ridge-scale FLOAT` (default 1.0) tunes the ridge multiplier on the regularized covariance.
- `vector merge`: positional `expression` argument — a steering expression such as `"0.3 ns/a + 0.4 ns/b"` or `"0.5 ns/a~ns/b"` for projection-removal. The comma-separated legacy form is gone.
- `--no-dls` (v2.1): disables the discriminative-layer-selection mask at extraction time. `--legacy` already implies this; passing both errors out at parse time. The mask itself lives in `saklas.core.vectors.compute_dls_mask` and is evaluated against the cached `layer_means`; without `layer_means` (e.g. `probes=[]` sessions whose neutrals haven't been computed yet) the helper silently keeps all layers. See `saklas/core/AGENTS.md` for the algorithm.
- `--legacy` (v2.0 backcompat preset): on `tui`/`serve` flips `injection_mode="additive"`, `extraction_method="pca"`, `projection_metric="euclidean"`, and `dls=False` (all passed through to `SaklasSession.from_pretrained`); on `vector extract` flips `--method pca`; on `vector compare` flips `--metric euclidean` (overriding the v2.1 mahalanobis default). Mutually exclusive with the per-flag controls on the same verb (combination errors at parse time before model load — `--legacy + --steer-mode`, `--legacy + --projection-metric`, `--legacy + --no-dls`, `--legacy + --method`, `--legacy + --metric` all reject). `_resolve_legacy_method(args)` (in `runners.py`) is the shared helper for the conflict check on `vector extract`.
- `pack ls` is local-only; `pack search` is the HF-remote verb.
- `pack rm` replaces `uninstall`; `pack ls` replaces `list`.

## SAE flags

`vector extract` accepts `--sae RELEASE` (required value — no implicit default, since SAELens ships many releases per base model) and `--sae-revision REV` (optional HF revision pin). Written tensor lands at `<concept>/<safe_model>_sae-<release>.safetensors`; returned canonical name from `session.extract` carries a `:sae-<release>` suffix so subsequent `session.steering` calls address the SAE variant uniquely.

`pack push --variant {raw,sae,all}` defaults to `raw` — SAE variants carry stronger provenance requirements (release + revision + per-layer sae_ids), so sharing them is opt-in. `pack clear --variant {raw,sae,all}` defaults to `all` — clearing a stale extraction should wipe every flavor unless scoped.
