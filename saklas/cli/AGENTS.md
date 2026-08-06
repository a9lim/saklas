# cli/

The eight-verb command line. `serve` runs the HTTP server; `manifold` is the
compute surface, `pack` the manifold lifecycle, `template` the standalone
templated-completion artifact; `lens` and `sae` are parallel per-model source
lifecycles (`fit` vs `train`, then fetch/ls/show/use/rm, with lens keeping
top/decompose); `experiment` runs repeatable research runs and `config`
inspects setup YAML.

## Module map

| File | Owns |
|---|---|
| `main.py` | `parse_args`, `main`, the `_COMMAND_RUNNERS` dispatch |
| `parsers.py` | `_build_root_parser` + every `_build_*` builder, the verb tables, the shared flag blocks |
| `config_file.py` | `ConfigFile` + `compose` / `apply_flag_overrides` / `ensure_vectors_installed` |
| `alpha_grid.py` | `parse_alpha_list` — comma lists, `start:stop:step`, `linspace(a,b,n)` for `experiment fan` |
| `runners/shared.py` | Cross-group helpers: session construction, effective config, probe/progress resolution, name resolvers, the verb menu, the error decorator |
| `runners/{serve,manifold,pack,template,config,lens,sae,experiment}.py` | That group's `_run_*` functions plus its `_*_RUNNERS` dispatch dict |
| `runners/__init__.py` | Barrel: re-exports every runner and helper as the `saklas.cli.runners.<name>` import/patch surface |

The barrel is load-bearing. Submodule runners call the session/config helpers
*through the package object* (`import saklas.cli.runners as _pkg`;
`_pkg._make_session(...)`), so `monkeypatch.setattr("saklas.cli.runners.<name>",
…)` reaches them. The names swapped this way are `_make_session`,
`_print_startup`, `_print_model_info`, `_load_effective_config`, the `_fold_*`
folders, and `_load_or_fit_transfer_alignment` (re-exported from `io.alignment`,
which owns the single-flight locking, under that private CLI name).

## Dispatch

`main()` routes `args.command` through `_COMMAND_RUNNERS`. Every verb group
hand-dispatches its own subverb from a `(verb, description)` table in
`parsers.py` (`_MANIFOLD_VERBS`, `_PACK_VERBS`, `_TEMPLATE_VERBS`,
`_LENS_VERBS`, `_SAE_VERBS`, `_EXPERIMENT_VERBS`, plus `_TRANSCRIPT_VERBS` in
`runners/experiment.py`). That table drives both parser registration and the
bare-verb help, so the two cannot drift; the `_*_RUNNERS` dict keys off the same
verb strings.

A bare `saklas`, or a verb group with no subverb, prints help and exits **0** —
not argparse's exit 2. Every group renders that menu through one helper,
`_print_verb_menu(group, verbs)` (usage line, width-aligned verb table, `-h`
pointer); `config` is the one group that prints its own usage inline. Unknown
subverbs never reach a runner — every verb is a registered subparser, so
argparse rejects them first.

`@_saklas_error_exit` wraps all eight top-level runners (and serve's two
live-readout policy helpers): an escaping `SaklasError` prints `user_message()`
to stderr and exits `min(2, status // 100)`.

## Name resolution

Manifolds are addressed by `(namespace, name)` pairs, and the split depends on
intent:

- **Addressing an existing manifold** — `manifold transfer` and `pack
  rm`/`clear`/`refresh`/`export` call `_resolve_manifold_ns_name`; `manifold
  fit` by name calls its path-returning wrapper `_resolve_manifold_folder`. It
  materializes bundled artifacts, returns an explicit `ns/name` verbatim
  (leaving existence to the io backend), and walks installed folders for a bare
  name so it can reach `default/`. Miss exits 1, collision exits 2. `pack show`
  implements the same cross-namespace and ambiguity semantics inline.
- **Authoring a fresh folder** — `manifold from-template`, `manifold merge`
  (target *and* sources), `pack push`, and `template create` use
  `_split_manifold_ns_name`, which defaults a bare name to `local/`; `manifold
  generate` does the same split inline.

`_iter_manifold_folders` calls `io.bootstrap.materialize_bundled_artifacts`
before walking, because CLI verbs resolve folders *before* any session exists —
without it a first `saklas manifold fit default/personas` would miss the bundled
folder.

Templates resolve their own way: `_split_manifold_ns_name` for `create`,
`io.templates.resolve_template` (cross-namespace) for `show`/`rm`/`score` and
for `manifold from-template`'s source.

## Config loading

`_load_effective_config(args, *, default_max_tokens=1024)` is the shared entry
point for every `-c`-taking subcommand. It composes `~/.saklas/config.yaml` plus
explicit `-c` files via `ConfigFile.effective(extras, include_default=True)`
(later overrides earlier), applies CLI overrides, then stamps `args` in place:
`model` (only if YAML supplied it), `temperature`, `top_p`, `thinking`,
`system_prompt`, `max_tokens`, `config_vectors`, plus the YAML-only knobs
`compile`, `cuda_graphs`, and `top_k_alts` (from YAML `return_top_k`). CLI wins
throughout — a YAML value only fills a field the flag left unset. It finishes by
calling `ensure_vectors_installed`.

`max_tokens` follows that same precedence, which is why the verbs carrying
`--max-tokens` argparse-default it to `None`: the loader must be able to tell
"unset" from an explicit value. `default_max_tokens` is the caller's floor when
neither CLI nor YAML supplied one — 1024 for `serve` (which has no flag), 256
for `experiment fan` and `experiment transcript run`, 128 for `experiment
naturalness`.

`ConfigFile.load` parses the YAML, rejects unknown keys, and type/range-checks
every field. Known keys (`_KNOWN_KEYS`): `model`, `vectors`, `thinking`,
`temperature`, `top_p`, `max_tokens`, `system_prompt`, `compile`,
`cuda_graphs`, `return_top_k`. `vectors:` is a single steering expression,
validated at load time through `core.steering_expr.parse_expr` so a typo fails
at config load rather than at generation. `compose` overrides field-by-field
(`vectors` wholesale — later files replace the expression, they do not
concatenate).

`ensure_vectors_installed` walks the raw expression via `referenced_selectors`
so namespace-qualified references keep their install coordinates, materializes
bundled artifacts, auto-installs HF manifolds, and reports unresolvable ones;
`strict=True` (`-s/--strict`) raises instead of warning. A bare reference counts
as resolved when `io.selectors.resolve_manifold_label` or `resolve_manifold_name`
finds it (an ambiguity is a match, not a miss).

## Session construction + narration

`_make_session(args, *, load_probes=True)` builds the session via
`from_pretrained`, threading `device`, `quantize`, `probes`, `system_prompt`,
`max_tokens`, `dls` (`not --no-dls`), `compile`, `cuda_graphs`, `return_top_k`
(from `--top-k-alts`, `None → 0`), and `on_progress`.

`_resolve_probes` maps unset / `all` → `None` (the session's default-roster
signal: tagged concept axes plus every fitted bundled multi-node manifold),
`none` / `[]` → `[]`, and an explicit category list through verbatim. There is
no serve-side probe-attach step — the roster is attached at session
construction, so the dashboard rack opens already watching it.

Artifact authoring never bootstraps an unrelated roster: `manifold extract` /
`manifold fit` pass `load_probes=False`, and `lens fit`/`top`/`decompose` and
`sae train` construct sessions directly with `probes=[]`. `serve`, `manifold
generate`, the `experiment` verbs, and `template score` take the default roster.

`_progress_printer(args)` is the one narration sink: an indented `  <message>`
writer, or `None` when `-j` is set so a JSON payload stays parseable.
`_make_session` passes it as the session's `on_progress` — construction is slow
(HF load, neutral/whitener capture, bundled fits) and without it looks like a
hang. Long operations pass their own callback alongside (`session.fit`,
`fit_jlens`, `train_sae`, `install_manifold`).

`serve` startup order: effective config → model check → `_make_session` →
config `vectors:` then `-S/--steer` through `_setup_steering_vectors` (which
pre-extracts and registers every referenced atom) → `create_app` →
`_warmup_session` (a 32-token stateless generation so dynamo's shape promotion
fires on a realistic prefill; serve is its only caller) →
`_enable_serve_live_lens_if_compatible` → (web only)
`_enable_serve_live_sae_if_available` → `uvicorn.run`. The lens helper adopts a
cached source whose identity matches the loaded weights; the SAE helper prefers
an already-active source, else `_best_serve_sae_release` (provider-hosted and
Neuronpedia-labelled first, canonical over broad variant sets, curated base over
its `-all` sibling, name as the stable tie-break). `--no-web` skips implicit SAE
discovery and download entirely.

## Shared flag blocks

- `_add_dls_arg` — `--no-dls` alone. Carried by `manifold extract` and
  `manifold fit`: an artifact-authoring verb has no use for generation-time
  compile state.
- `_add_injection_args` — `--no-dls` plus `--compile` and `--cuda-graphs`. On
  `serve`, `experiment fan`, `experiment transcript run`, `experiment
  naturalness`. All default `None`/`False`; YAML fills unset values, session
  defaults (DLS on, compile and cuda-graphs off) win otherwise.
- `_add_logit_args` — `--top-k-alts N`, the session default for
  `SamplingConfig.return_top_k`. Enforced in `[0, 256]` by both CLI and YAML.
  On `serve` and `experiment fan`.
- `_add_config_args` — `-c/--config PATH` (repeatable) and `-s/--strict`.

## Reserved short flags

One letter, one meaning, tree-wide:

| Flag | Long form | Flag | Long form |
|---|---|---|---|
| `-m` | `--model` | `-f` | `--force` |
| `-d` | `--device` | `-j` | `--json` |
| `-q` | `--quantize` | `-v` | `--verbose` |
| `-p` | `--probes` | `-y` | `--yes` |
| `-c` | `--config` | `-s` | `--strict` |
| `-S` | `--steer` | `-a` | `--as` |
| `-g` | `--grid` | `-o` | `--output` |
| `-H` | `--host` | `-P` | `--port` |
| `-C` | `--cors` | | |

`-k` is the one double-booked letter: `--api-key` on `serve`, `--top-k` on `lens
top` / `lens decompose`. They never share a parser.

`pack push` keeps `--private` and `--dry-run` long-only: `-p` and `-d` are taken
above, and `push` is the one verb that publishes — muscle memory must not flip
visibility or skip the upload. The manifold compute verbs (`extract`,
`generate`, `fit`) expose no `-d`/`-q` at all; they
`set_defaults(quantize=None, device="auto", probes=None)`.

## Per-verb flags

**serve** — `model` (optional when YAML supplies it), `-q`, `-d`, `-p`,
`-H/--host` (`0.0.0.0`), `-P/--port` (`[1, 65535]`, 8000), `-S/--steer EXPR`,
`-C/--cors ORIGIN` (repeatable), `-k/--api-key` (falls back to
`$SAKLAS_API_KEY`), `--no-web`, plus the injection, logit, and config blocks.

**manifold extract** — `concept` (`nargs="+"`; one concept or two poles, 3+ is
an error), `-m`, `-f` (re-authors the pole corpora and bypasses the tensor
cache), `--sae RELEASE`, `--role SLUG` (mutually exclusive with `--sae`; the
role bakes into the node corpora and writes the *canonical* tensor while
returning a `:role-<slug>` name tail), `--kind {abstract,concrete,custom}`
(default `abstract`), `--system TEMPLATE` (dest `custom_system`; required when
`--kind custom`), `--namespace NS` (unset → `local/`), `--no-dls`. `--kind`
selects the generation system template and elicitation role label — the same
knob `manifold generate` carries. Cache-hit validation is the model-free
preflight or the loaded session/pipeline — never bare file existence, which
proves nothing about model, corpus, tokenizer, role, SAE transform, or manifest
identity (see `manifold fit` below for the shared preflight).

**manifold generate** — `name`, `--concepts C...` (required; the runner rejects
fewer than 2), `--kind`, `--system`, `--samples-per-prompt K` (1),
`--description TEXT`, `--seed INT`, `--role-per-node`, `-m`, `-f`.
`--role-per-node` doubles each concept slug as that node's assistant-role
substitution, producing a persona manifold. Generation resumes: the runner plans
first (`plan_discover_generation`), so an already-complete folder costs no model
load and a partial one regenerates only its missing nodes.

**manifold from-template** — `template`, `--name MANIFOLD` (default: the
template name), `--fit-mode {pca,spectral,auto}` (default `auto`), `--max-dim
N`, `--var-threshold T`, `--description`, `-f`. Pure IO: resolves the template,
writes a discover folder carrying the corpus plus `template_ref`, no model.

**manifold fit** — `target` (a manifold name *or* a folder path;
`_run_manifold_fit` resolves it and reads `fit_mode`), `-m`, `-f`, `--layers
L1,L2,…|workspace|all` (default all), `--sae RELEASE`, and the discover
hyperparams `--method {pca,spectral,auto}`, `--max-dim N`, `--min-dim N`,
`--var-threshold T`, `--k-nn K`, `--bandwidth SIGMA`, `--max-subspace-dim R`,
`--smoothing auto|0|LAMBDA`, `--persistence-frac F`, plus `--no-dls`.

Hyperparams apply only to a discover folder (`pca`/`spectral`/`auto`); supplying
any against an authored folder exits 2, mirroring the server's 400. The runner
passes only the supplied override patch into `session.fit`, and the pipeline
merges and publishes it under the same manifest lock that derives the cache key
— the CLI never rewrites `manifold.json` unlocked. `--min-dim`,
`--max-subspace-dim`, and `--smoothing` are curved-fit knobs and are rejected
for `--method pca` (a flat fit's subspace dim *is* its `--max-dim` layout dim);
`--persistence-frac` is `auto`-only.

`-f` bypasses the per-model tensor cache and re-pools unconditionally — needed
because `fit`, unlike `extract -f`, does not re-author the corpus, so an
unchanged corpus always cache-hits and a code-level fit change could never be
picked up.

Without `-f`, both `fit` and `extract` run the model-free exact no-op preflight
(`_try_manifold_fit_noop_preflight` → `io.preflight_manifold_fit_noop`) **before**
`_make_session`: on a proven no-op they print the cache-hit line and exit 0
having loaded no weights. A mutable model id still proves nothing — the
preflight establishes sidecar integrity, corpus / role / template identity, the
token-exact tokenizer render, and the checkpoint→loaded fingerprint bridge, and
returns "unproven" for anything it cannot establish (an `--sae` fit, any
discover override, a checkpoint with no provable source, a missing neutral
cache). Unproven falls through to the ordinary loaded fit, whose cache check
still runs against the actually-loaded weight fingerprint. `extract` additionally
declines the preflight on a `--role` baseline the existing corpus does not carry,
so that stays the loaded path's error rather than a silent hit.

**manifold bake** — `name`, `expression`, `-f`, `-s/--strict`, `-m`. Lands a
corpus-less baked manifold via `io.bake.merge_into_manifold`; only
namespace-qualified additive/subtractive scalar terms are accepted, since
dynamic terms and Mahalanobis projections need a live model.

**manifold merge** — `name`, `sources` (the runner requires ≥ 2),
`--description`, `--method {pca,spectral,auto}` (override the merged fit_mode;
required when sources disagree), `-f`. Unions discover-mode node *corpora* into
a fresh unfitted folder — the corpus analogue of `bake`, which lowers a steering
expression instead.

**manifold transfer** — `name`, `--from SRC` / `--to TGT` (both required), `-f`,
`-j`. Preflights the source tensor, obtains the Procrustes alignment through
`io.alignment.load_or_fit_transfer_alignment` (a thin call — single-flight
locking, cache proof, and retry-on-race live in `io`), then writes the target's
`from-<safe_src>` tensor via `transfer_manifold`. `-f` recomputes alignment and
transfer but reuses exact neutral caches. Re-baking shares in the target
Mahalanobis metric is mandatory: a missing or unusable target cache raises
`WhitenerError` and the runner exits 1 with a regenerate-neutrals hint.

**manifold compare** — `concepts` (1+), `-m` (required), `-v`, `-j`,
`--ridge-scale FLOAT` (1.0). Loads `LayerWhitener.from_cache` up front and exits
1 on a miss — compare is Mahalanobis-only, so a missing neutral cache surfaces
directly instead of degrading. One arg ranks every installed fitted manifold
against the target, two are pairwise, three or more produce an N×N matrix;
bulk selectors (`tag:`/`namespace:`/`all`) expand to names and inherit any
`:variant` suffix. Concepts fold from their 2-node manifolds.

**manifold why** — `concept`, `-m` (required), `-j`. Text output is the per-layer
`‖baked‖` histogram, 16 buckets, terminal-width aware; `-j` emits the total
layer count plus one `{layer, magnitude}` row per layer.

**pack** — `ls [--namespace NS] [-v] [-j]`; `show <name> [-j]`; `install
<target> [-a NS/NAME] [-f]` (HF coord `<ns>/<name>[@revision]` or a local folder
path); `search [query] [-v] [-j]`; `push <selector> [-a OWNER/NAME] [-m]
[--private] [--dry-run] [--variant raw|sae|all]` (default `raw` — SAE variants
carry stronger provenance, so sharing them is opt-in); `rm <selector> -y`;
`clear <selector> [-m] [--variant raw|sae|all]` (default `all`); `refresh
<selector> [-m]`; `export gguf <name> [-m] [-o PATH] [--model-hint HINT]`.

`rm` requires `-y` in **every** namespace, bundled `default/` included — a
`local/` manifold cost real extraction and fit time, so the guard is uniform,
not per-namespace; without it the runner exits 2. These verbs are pure IO over
`~/.saklas/manifolds/` and load no model (`install` and `search` hit HF).
`export gguf` folds a fitted 2-node `pca` manifold to one direction via
`io.gguf_io.export_gguf_manifold` and rejects multi-node or curved fits.

**experiment fan** — `model`, `prompt`, `-g/--grid CONCEPT=ALPHAS` (required,
repeatable), `-S/--steer EXPR` (the fixed expression composed under each grid
row), `--max-tokens` (256), `-q`, `-d`, `-p`, `-j`, plus the injection, logit,
and config blocks. `--base-steering` is a hidden alias on the same dest,
suppressed from `--help` so `-S/--steer` reads identically on every verb that
takes one. Grid alphas parse through `alpha_grid.parse_alpha_list`.

**experiment transcript run** — `path`, optional `model` (defaults to the
transcript's own `model_id`), `--max-tokens` (256), `-q`, `-d`, `-p`, plus the
injection and config blocks. `-s/--strict` comes from the config block but here
gates "refuse on probe drift".

**experiment naturalness** — `model`, `prompt`, `--manifold FOLDER` (required),
`-S/--steer EXPR` (required), `--compare-linear`, `--max-tokens` (128), `-q`,
`-d`, `-p`, `-j`, plus the injection and config blocks. The behavior-manifold
preflight loads the folder with `verify_manifest=False`: it consumes only
authoring geometry and corpus, so unrelated fitted payloads are not hashed.

**config** — `show [-c PATH …] [-m MODEL] [--no-default]`, `validate <file>`.
`validate` dry-runs vector resolvability without installing anything and exits 2
on any failure.

**template** — `create <name> --slot TOKEN --values V... --contexts FILE
[--description] [-f]`; `ls [-j]`; `show <name> [-j]`; `score <name> -m MODEL
[-S EXPR] [--by sum|mean] [-d] [-q] [-j]`; `rm <name> -y`. `create` reads a JSON
list of contexts, each a multi-turn `{turns, assistant}` or the single-turn
`{user, assistant}` sugar normalized by `_normalize_context_entry`. `score` is
the only template verb that loads a model, printing the per-context
restricted-choice value distribution (`--by` picks the joint or the
length-normalized ranking statistic); the rest are pure IO. `rm` without `-y`
exits 2.

**lens** — the model is a **positional** on `fit`/`fetch`/`ls`/`show`/`use`/
`top`/`rm` (the artifact is per-model); `decompose` takes a selector positional
and a required `-m`.

- `fit <model> [--corpus FILE] [--prompts N] [--seq-len T] [--dim-batch K]
  [--prompt-batch B] [--checkpoint-every N] [--layers …] [-f] [-d] [-q]`.
  Defaults: 100 prompts, seq-len 128, dim-batch 8, prompt-batch 4 (2 on MPS),
  checkpoint every 25. All numeric fit flags are positive-only. `--corpus` reads
  one document per line (a JSONL line with a `text` field also works, each
  sliced to `_LENS_DOC_CHARS = 4000` before tokenization); unset, it streams the
  default web-text sample via the optional `datasets` dependency. `--layers`
  accepts explicit indices, `workspace`, or `all`; `sample` is rejected because
  it still includes layer 0 and is artifact-size/debug only, not a wall-time
  lever. `-f` restarts from zero; otherwise the fit resumes. A model-free exact
  no-op runs before corpus streaming and model load
  (`_try_lens_fit_noop_preflight`, serialized under `io.lens.lens_fit_lock`):
  it matches the sidecar's source fingerprint, seq-len, layer coverage, corpus
  identity, and payload digests, then reaps a crash-left checkpoint the
  validated final artifact provably subsumes.
- `fetch <model> [source] [--revision REV] [--repo REPO] [-f] [-j]` — pure IO,
  no model load. Source defaults to `neuronpedia`, revision to `main`, repo to
  `neuronpedia/jacobian-lens`. Provider bytes stay in the Hugging Face cache;
  Saklas writes only the pinned binding.
- `ls <model> [-j]`, `show <model> [source] [-j]`, `use <model> <source>` —
  sources are `local:default` or `neuronpedia`.
- `top <model> <prompt> [-k K] [--layers …] [--position P] [-d] [-q] [-j]` —
  raw prompt, no chat template. `-k` defaults to 8, layers to every fitted
  layer, `--position` is repeatable and accepts negatives (default: final
  position). Output leads with the layer-aggregated block (`token · strength ·
  com ±spread`) then the per-layer matrix; JSON carries both under `aggregate`
  and `layers`, whose per-layer rows report the same probability unit the
  aggregate averages.
- `decompose <selector> -m MODEL [-k K] [--layers …] [-d] [-q] [-j]` — `-k` is
  the sparsity budget, default 16.
- `rm <model> [source] [-y]` — external removal forgets only the binding and
  never purges Hugging Face cache bytes. Without `-y` it prompts interactively
  (unlike `pack rm` / `template rm`, which exit 2).

**sae** — mirrors lens argument order, model positional first; sources are
`local:NAME` or `saelens:RELEASE`.

- `train <model> <name>` — native residual-post ReLU fit with `--corpus`,
  `--layer` (default: nearest 65% depth), `--tokens` (1,000,000), `--seq-len`
  (128), `--batch-size` (8), `--width` / `--expansion` (8), `--learning-rate`
  (3e-4), `--l1` (1e-3), `--dead-threshold` (1e-6), `--seed` (0), `-f`, `-d`,
  `-q`, `-j`.
- `fetch <model> saelens:RELEASE [--layer L] [--revision REV] [-j]` — pure IO
  like `lens fetch`, with **no `-d`/`-q`**: every check `load_sae` makes for an
  external release is provable from the release entry and the published config
  (compatibility, covered layers, decoder row vs hidden size), so the
  transformer is never instantiated.
- `ls <model> [-j]`, `show <model> [source] [-j]`, `use <model> <source>`.
- `rm <model> [source] [-y]` — **no `-j`**: removal reports one line of prose,
  matching `lens rm`. Prompts interactively without `-y`.

## Deliberately absent

Each of these belongs to a design the engine does not have; do not propose them.

- No `vector` alias verb. No `discover` verb — `manifold fit` takes a
  name-or-folder and reads its `fit_mode`. No top-level `transcript` verb — it
  nests under `experiment`. No `manifold template` verb — authoring is
  `manifold from-template`, and the manifold carries a `template_ref`.
- No `argv[0]` peeking and no bare-model fallback: `saklas google/gemma-2-2b-it`
  is an argparse error.
- No `--steer-mode` / `--theta-max` / `--legacy` / `--injection-mode` /
  `--projection-metric` anywhere. There is one injection kernel, and `~`/`|`
  projection is Mahalanobis-only.
- No `--metric` on `manifold compare`, and no Euclidean fallback for compare,
  transfer, or any activation-space fit.
- `--method` belongs only to discover-mode fitting (`manifold fit`, `manifold
  merge`). Extraction has no method knob — difference-of-means, a 2-node `pca`
  fit, is the only one.
- No `method` / `injection_mode` / `theta_max` / `projection_metric` config
  keys to thread through `_load_effective_config`, and no gain / lever / origin
  flags: the only steering knob on the CLI is `--no-dls`.
- No serve-side probe-attach step — probes are attached at session construction.
- No `-j` on `sae rm` / `lens rm`, and no `-d`/`-q` on `sae fetch` /
  `lens fetch` (both are pure IO).
- No `DIAGNOSTICS` block in `manifold why`.
