# cli/

Five-verb root parser (`tui`/`serve`/`pack`/`vector`/`config`) split across:
- `cli/main.py` — entry point, `_build_root_parser`, `_load_effective_config`, `_warmup_session`
- `cli/parsers.py` — every `_build_X_parser`
- `cli/runners.py` — every `_run_X`
- `cli/selectors.py` — selector grammar + `resolve_pole` (renamed from `cli_selectors.py`)
- `cli/config_file.py` — `ConfigFile` + `effective()` / `resolve_poles()` / `to_yaml()`

## Verb nesting

- `pack` = distribution (install/refresh/clear/rm/ls/search/push/export) via `_PACK_BUILDERS` / `_PACK_RUNNERS` tables
- `vector` = computation (extract/merge/clone/compare/why) via `_VECTOR_BUILDERS` / `_VECTOR_RUNNERS`
- `config` = show / validate

## Config loading

`_load_effective_config(args)` is the shared entry point every subcommand that takes `-c` calls. Composes `~/.saklas/config.yaml` + explicit `-c` files and stamps `args.config_vectors` (a steering expression string, or `None`) / `args.temperature` / `args.top_p` / `args.max_tokens` in place. The `vectors:` YAML key is a single steering expression — parsed through `saklas.core.steering_expr.parse_expr` which resolves bare poles (`wolf → deer.wolf @ -0.5`) via `cli.selectors.resolve_pole`. `ConfigFile.load` validates the expression at load time and stores the raw string; re-parsing happens on consumption.

## Warmup

`_warmup_session` issues a single-token `session.generate(sampling=SamplingConfig(max_tokens=1), stateless=True)` — no `session.config` mutation.

## Flags

- `serve`: `--host/-H`, `--port/-P`, `--steer/-S EXPR`, `--cors/-C`, `--api-key/-k`, plus `-c/--config`. `--steer` takes one steering expression string (the shared grammar); combine multiple terms with `+`/`-`.
- `vector merge`: positional `expression` argument — a steering expression such as `"0.3 ns/a + 0.4 ns/b"` or `"0.5 ns/a~ns/b"` for projection-removal. The comma-separated legacy form is gone.
- `pack ls` is local-only; `pack search` is the HF-remote verb.
- `pack rm` replaces `uninstall`; `pack ls` replaces `list`.

## SAE flags

`vector extract` accepts `--sae RELEASE` (required value — no implicit default, since SAELens ships many releases per base model) and `--sae-revision REV` (optional HF revision pin). Written tensor lands at `<concept>/<safe_model>_sae-<release>.safetensors`; returned canonical name from `session.extract` carries a `:sae-<release>` suffix so subsequent `session.steering` calls address the SAE variant uniquely.

`pack push --variant {raw,sae,all}` defaults to `raw` — SAE variants carry stronger provenance requirements (release + revision + per-layer sae_ids), so sharing them is opt-in. `pack clear --variant {raw,sae,all}` defaults to `all` — clearing a stale extraction should wipe every flavor unless scoped.
