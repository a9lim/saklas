# CLAUDE.md

## What this is

`saklas` is a Python library + TUI + OpenAI-compatible server for activation steering and trait monitoring on HuggingFace causal LMs. Extracts per-layer steering profiles via contrastive PCA, applies them per-generation through forward hooks, monitors activations against probe vectors. Three frontends — `SaklasSession` (programmatic), `saklas serve` (HTTP), `saklas` (Textual TUI) — all share the same engine.

## Commands

```bash
pip install -e ".[dev]"                # editable install + pytest
pip install -e ".[cuda]"               # bitsandbytes + flash-attn (CUDA only)
pip install -e ".[serve]"              # fastapi + uvicorn
saklas <model_id>                       # launch TUI
saklas serve <model_id>                 # OpenAI-compatible API
saklas install <target> [-s|-a NS/N|-f] # HF coord or folder; -s = statements only
saklas refresh <selector>               # re-pull concept(s); `refresh neutrals` for neutral_statements.json
saklas clear <selector> [-m MODEL]      # delete per-model tensors (keeps statements)
saklas uninstall <selector> [-y]        # fully remove concept folder (bundled respawns on next run)
saklas list [selector] [-i|-j|-v]       # lists installed + HF; -i = installed only
saklas merge <name> <components> [-m]   # merge: saklas merge bard default/happy:0.3,...
pytest tests/                           # all tests (CPU tests run anywhere)
pytest tests/test_smoke.py              # GPU smoke (downloads gemma-3-4b-it, ~8GB)
```

Selector grammar (shared by `refresh`/`clear`/`uninstall`/`list`): `<name>`, `<ns>/<name>`, `tag:<t>`, `namespace:<ns>`, `default`, `all`. Bare names resolve across namespaces and raise `AmbiguousSelectorError` on collision. Subcommands take a single selector positional plus `-m/--model` where model scope is meaningful (refresh, clear, merge). `refresh` silently skips `source=local` concepts so `refresh all` doesn't crash against a cache containing user-authored vectors. `refresh neutrals` is a reserved form that rewrites `~/.saklas/neutral_statements.json` from the bundled copy. `uninstall` refuses broad selectors (`all`, `namespace:`) without `-y`; bundled concepts re-materialize on next session init.

## PyPI release

Package name `saklas`, import name `saklas`. Version lives in **two places** — bump both or `saklas.__version__` drifts from PyPI:

- `pyproject.toml` — `[project] version`
- `saklas/__init__.py` — `__version__`

Releases are automated: merging a version bump to `main` triggers `.github/workflows/release.yml`, which tags `v$VERSION`, builds, publishes via trusted publishing, and cuts a GitHub release. Push without bumping → workflow no-ops. `saklas/data/**/*.json` ships via `[tool.setuptools.package-data]`; the user cache under `~/.saklas/` is materialized on first run.

## Cache layout

All state under `~/.saklas/` (override via `SAKLAS_HOME`):

```
~/.saklas/
  neutral_statements.json                 # user-editable (copy-on-miss from package)
  vectors/
    default/<concept>/                    # bundled (source=bundled, -r default)
      pack.json                           # name, description, tags, files{sha256}
      statements.json                     # contrastive pairs array
      <safe_model_id>.safetensors         # per-model tensor
      <safe_model_id>.json                # slim sidecar: method/scores/saklas_version/statements_sha256
    local/<concept>/                      # user-authored (source=local, -r skips)
    <hf_owner>/<concept>/                 # HF-pulled (source=hf://<owner>/<name>[@revision])
  models/<safe_model_id>/
    layer_means.safetensors               # per-model probe-centering baseline
    layer_means.json                      # sidecar hashes neutral_statements.json
```

Integrity: `pack.json.files` is a sha256 map verified on every `ConceptFolder.load`. Tensor sidecars record `statements_sha256` at extraction; mismatch flags `is_stale` and the CLI warns.

HF distribution: packs live as **model repos** (not datasets) because safetensors is model-hub-native and `base_model` frontmatter creates reverse-link discoverability from the base model's hub page. `saklas/hf.py` uses `repo_type="model"` exclusively — no dataset fallback. `saklas -i owner/name@revision` pins to any git ref (tag, branch, or commit SHA); `hf.split_revision` parses `@`, threads `revision` through `_download`/`pull_pack`/`fetch_info`, and records it in `source = "hf://owner/name@v1.2.0"`. `cache_ops._refresh` re-splits the stored source so pinned installs re-pull the same revision — pinning is pinning, not "follow latest." `@` is unambiguous because `NAME_REGEX` forbids it in concept names.

**Upgrade gotcha.** `materialize_bundled` is copy-on-miss for `neutral_statements.json`, so existing users keep their old file on upgrade. Run `saklas refresh neutrals` after a release that changes neutrals to force-copy the bundled version — layer means auto-recompute on next session init via the hash check in `bootstrap_layer_means`.

## Architecture

Five layers: **model/vector**, **steering/monitoring**, **session**, **TUI**, **server**.

### Model + vector
- `model.py` — Loads HF causal LMs. `_LAYER_ACCESSORS` maps `model_type` → layer-list accessor (`def`, not lambdas); adding an architecture = one entry. Cascading fallbacks on attention impl (SDPA → eager), dtype, and device. `_load_text_from_multimodal` extracts the text-only sub-model from checkpoints like Ministral-tagged-as-Mistral3, strips `language_model.` prefixes from safetensors shards, and dequantizes FP8 inline. Patches `torch.histc` for MPS MoE routing.
- `vectors.py` — One forward pass per prompt (no batching); `_capture_all_hidden_states` hooks every layer in a single pass. `_encode_and_capture_all` pools from the **last content token**, walking back past `tokenizer.all_special_ids` to skip trailing chat-template markers (Llama's `<|eot_id|>`, Gemma's `<end_of_turn>`, Qwen's `<|im_end|>`) whose hidden states are disconnected from content. Contrastive diffs are cast to **float32** before differencing — fp16 loses precision on close activation vectors and produces degenerate SVD inputs. Multi-pair: per-layer SVD, scored by explained variance ratio. Single-pair: scored by `diff_norm / activation_norm` (same ~0.01–0.4 range as EVR so the two modes are comparable). MPS cache flushed between pairs. Returns a **profile**: `dict[int, (Tensor, score)]` covering every layer. `compute_layer_means` averages over 45 neutral prompts for centering. `_template_overhead_cache` is keyed by `id(tokenizer)` — safe only when one tokenizer lives for the session lifetime.
- `probes_bootstrap.py` — Walks `~/.saklas/vectors/default/` after `materialize_bundled()`. 21 bundled probes (19 bipolar + 2 monopolar) ship as per-concept folders in `saklas/data/vectors/`, categorized via `pack.json` tags (`affect`, `epistemic`, `alignment`, `register`, `social_stance`, `cultural`). `bootstrap_layer_means` caches at `~/.saklas/models/<id>/layer_means.safetensors`, stale when `neutral_statements.json` hash changes. MPS cache flushed between probe extractions.

### Steering + monitoring
- `hooks.py` — `SteeringHook` adds a pre-composed vector to hidden states in-place. `SteeringManager` groups vectors by layer, orthogonalizes per layer (Gram-Schmidt), one hook per active layer. Vectors register via `add_vector(name, profile, alpha)`; activation is controlled by the session clearing and re-adding, not by a toggle flag. **Per-profile normalization:** `effective_alpha = alpha * score_i * (_REF_SCORE / mean_score)`. Raw PCA scores differ several-fold between architectures, so without normalization the same user alpha would hit a dead zone on one model and a coherence cliff on another. Dividing by `mean_score` preserves relative per-layer emphasis; re-multiplying by `_REF_SCORE = 1/32` pins the alpha scale so that α≈0.5 is the coherent nuanced band and α≈1.0 is past the cliff across the bundled architectures. `_REF_SCORE` was last calibrated against gemma-4-31b-it with the v1.2 regenerated pairs — recalibrate if the bundled statements or their topical breadth change materially.
- `monitor.py` — `TraitMonitor` scores all probes against a set of per-layer hidden states. `_score_probes` mean-centers (subtracting the per-layer neutral means) before computing score-weighted cosine similarities (both norms: `(h @ v) / (|h| * |v|)`). `total_w` accumulates only layers present in the hidden-state dict. Two entry points: `measure(model, tokenizer, layers, text)` runs a forward pass and scores from the last-content-token hidden states; `measure_from_hidden(hidden_per_layer)` scores pre-captured states directly. History accumulates across generations. `probe_names`, `profiles`, `layer_means` are properties (no defensive copies).

### Session
- `session.py` — `SaklasSession` owns the model, vector registry (`_profiles`), monitor, generation config, and conversation history. **Vectors are registered without alphas** via `steer(name, profile)`; alphas are supplied per-generation via `generate(input, alphas={"name": 0.5})`. Orthogonalize and thinking are per-call. No persistent steering hooks between generations. `thinking=True` is gated by `supports_thinking` and the session decodes only the response portion for history. `generate()` / `generate_stream()` accept OpenAI-parity sampling: `stateless` (skip history + snapshot/restore monitor around `measure()`), `seed`, `stop`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs`. The server always passes `stateless=True`; the TUI uses the stateful path. `_generation_preamble()` dedupes setup shared between blocking and streaming. After generation a **separate** forward pass runs `monitor.measure()` to match probe extraction's last-content-token pooling. Streaming uses a worker thread; the worker emits `None` as the done sentinel in its `finally`. The on_token callback carries `(text, thinking, token_id, logprob, top_logprobs)`. Full custom-concept pipeline: cache → curated dataset → statement cache → model-generated pairs → contrastive PCA → save. Curated concepts save under `~/.saklas/vectors/default/<c>/`; user concepts under `~/.saklas/vectors/local/<c>/`. Statement caches are model-independent (stored alongside the tensor folder as `statements.json`), so generated pairs are reused across models. Every save refreshes the owning `pack.json.files` sha256 map via `_update_local_pack_files`. `_N_PAIRS = 45` is the default for both bundled statements and user generation. `_PAIR_RE` is compiled at module level. `MIN_ELAPSED_FOR_RATE` is a public constant.
- `datasource.py` — `DataSource` normalizes contrastive pairs from curated names, JSON, CSV, HF datasets, or raw lists.
- `results.py` — `GenerationResult`, `TokenEvent`, `ProbeReadings` dataclasses with `to_dict()`. `TokenEvent` has `thinking: bool`, `logprob`, `top_logprobs`, `finish_reason`. `GenerationResult` has `prompt_tokens`, `finish_reason` (`"stop"`/`"length"`/`"stop_sequence"`), optional `logprobs`. `ProbeReadings`: `mean`, `std`, `min`, `max`, `per_generation`, `delta_per_gen`. `ResultCollector` accumulates for batch export (dicts/JSONL/CSV/DataFrame).

### Generation loop
- `generation.py` — Token-by-token with KV cache, wrapped in `torch.inference_mode()`. Top-p via `torch.topk` (k capped at 1024). In-place ops (`logits.div_()`, etc.) throughout. MPS sync at end of loop prevents Metal command buffer reuse crashes. `generate_steered` accepts `seed`, `stop` (list), `logit_bias` (pre-built `(idx, val)` tensors), `presence_penalty`/`frequency_penalty` (tracked via a per-token `completion_counts` dict, applied to raw logits before temperature), and `logprobs` (0 = chosen only, >0 = top-k alternatives via `torch.log_softmax(logits.float())` before sampling; gated off the hot path when `None`). `state.finish_reason` defaults to `"length"` and flips to `"stop"` on EOS/external-stop or `"stop_sequence"` on stop-string match; stop matching runs against accumulated non-thinking response text per emitted token and trims the current emit to the pre-stop prefix. The `None` end-of-generation sentinel is **not** emitted by `generate_steered` — the session worker or TUI closure puts it on the queue in its `finally`, so pending actions see the final conversation state. `supports_thinking(tokenizer)` delegates to `_detect_think_delimiters`, which returns `(start_id, end_id, response_start_id, starts_in_thinking)` by rendering a round-trip assistant message through the template — works across families (Qwen `</think>`, Gemma `<channel|>`, gpt-oss `<|channel|>…<|message|>`, etc.) without hardcoded delimiter strings. `response_start_id` handles channel-based formats where multiple tokens separate thinking from response. Generation loop uses a state machine: idle → preamble → thinking → response_preamble → done. Tokens that decode to `\ufffd` (partial UTF-8) are marked `None` in the token table and buffered until a complete token follows, then flushed via `tokenizer.decode(pending_ids)`. `_tok_key(tokenizer)` is a shared cache-key helper used by `_eos_cache`, `_token_table_cache`, `_think_delim_cache`. `GenerationState` has `stop_requested` (Event), `token_queue` (SimpleQueue), `thinking_end_idx`, `finish_reason`.

### TUI
- `tui/app.py` — Thin frontend over `SaklasSession`. Owns local alpha/enabled/orthogonalize/thinking state, passes through to session per call. Thinking defaults ON for models that support it (Ctrl+T toggles). Polls at ~15 FPS. Commands: `/steer`, `/probe`, `/clear`, `/rewind`, `/sys`, `/temp`, `/top-p`, `/max`, `/help`. Mid-generation interruption: any conflicting action (Ctrl+R, new message, `/steer`, `/probe`, `/clear`, `/rewind`) stops the current generation and defers execution via `_pending_action`; `_poll_generation` dispatches once the worker finishes and the `None` sentinel is consumed. Panel focus uses index constants (`_LEFT`, `_CHAT`, `_TRAIT`).
- `tui/utils.py` — Shared helpers; `build_bar(value, max, width)` renders filled/empty bar pairs.
- `tui/vector_panel.py`, `chat_panel.py`, `trait_panel.py` — Panels. `_AssistantMessage` renders thinking tokens in a collapsible section. `trait_panel._nav_items` is `list[str]` (probe names directly).

### API server
- `server.py` — FastAPI app factory. OpenAI-compatible (`/v1/models`, `/v1/chat/completions`, `/v1/completions`) plus saklas management (`/v1/saklas/vectors`, `/v1/saklas/probes`, `/v1/saklas/session`). Thin HTTP layer — all routes call the session with `stateless=True`. `_SamplingBase` pydantic model (shared by chat and completions) accepts the full OpenAI surface: `stop`, `seed`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs` (bool for chat, int for completions), `top_logprobs`, `stream_options.include_usage`, `max_completion_tokens` (aliased to `max_tokens` via model validator), plus accept-and-ignore for `user`, `n`, `response_format`, `messages[].name`. `ChatMessage._flatten_content` accepts multimodal arrays and concatenates text parts (non-text rejected). Responses include real `usage` counts, accurate `finish_reason` from `session._gen_state.finish_reason`, per-request `created` timestamps. `_stream_generation` emits a first-chunk `{role: "assistant"}` delta for chat; final chunk's `finish_reason` comes from gen state; optional usage chunk follows before `[DONE]` when `stream_options.include_usage` is set. Single `app.state.gen_lock` (`asyncio.Lock`) serializes both non-streaming (`async with`) and streaming (held for the full stream with 5-minute timeout → 503); requests queue FIFO rather than 409ing. Bearer auth via `SAKLAS_API_KEY` env / `--api-key` CLI is applied as an app-level dependency; unset = open. `RequestValidationError` handler maps pydantic errors to OpenAI shape (`type: "invalid_request_error"`, `param`, `code`). Thinking tokens stream as `reasoning_content` in the chat delta. Logprobs render via `_render_logprobs_chat` which decodes token ids through the tokenizer. Vector extraction streaming is synchronous (progress replayed after completion). **Not supported:** tool calling, strict JSON/json_schema mode, `/v1/embeddings`.
- `cli.py` — Subcommand-based dispatch. Peek `argv[0]`: if it's in `_SUBCOMMANDS = {serve, install, refresh, clear, uninstall, list, merge}`, route to that subcommand's parser; otherwise treat argv as bare-TUI args. Each subcommand has its own parser builder (`_build_<cmd>_parser`) and runner (`_run_<cmd>`), wired together via the `_RUNNERS` dict. `_print_startup(args)` shared between TUI and serve. `serve` accepts `--host/-H`, `--port/-P`, `--steer/-S name:alpha`, `--cors/-C`, `--api-key/-k`. `install`: `target` positional + `--statements-only/-s`, `--as/-a`, `--force/-f`. `refresh <selector>`: `--model/-m` for scoped refresh; `selector == "neutrals"` is a reserved form routing to `cache_ops.refresh_neutrals()`. `clear <selector>`: `--model/-m`, `--yes/-y` (required for `all`/`namespace:`). `uninstall <selector>`: `--yes/-y` (required for broad selectors; bundled concepts respawn). `list [selector]`: `--installed/-i` (skip HF), `--json/-j`, `--verbose/-v`; HF is queried by default. `merge <name> <components>`: `--model/-m`, `--force/-f`, `--strict/-s`. Bare TUI: `<model>` positional + `-q/-d/-p/-s/--max-tokens` from `_add_common_args`, plus `-c/--config` (repeatable) and `--strict`. Cache ops no longer compose with model load — each is a standalone action. Config loading via `-c` composes YAML files, auto-installs missing vectors, and the composed vector map lands on `args.config_vectors` for `_run_tui`. `print_migration_notice_if_needed()` flags legacy `probes/cache` / `~/.liahona` detritus at startup.

## Performance rules

These gate the throughput regression test (steered ≥ 85% of vanilla tok/s):

- **Hot-path hooks**: no Python allocation, no `.item()`, no CPU sync, in-place mutation only.
- **`torch.inference_mode()`** wraps the entire generation loop.
- **In-place ops**: `logits.div_()`, `logits.clamp_()`, `probs.div_()`.
- **Top-p via `torch.topk`**, not full-vocab sort. `k = min(1024, vocab)`.
- **Norms use `.float()`** — fp16 sum-of-squares overflows for hidden_dim ≥ 2048.
- **Vectors scaled to mean hidden-state norm at extraction, gated per-layer at apply time** by `score_i * (_REF_SCORE / mean_score)`. User alpha scale is architecture-invariant: α≈0.5 coherent, α≈1.0 cliff.
- **Monitor runs a separate forward pass** after generation and pools from the last content token (matching probe extraction). Mean-centered cosine sims remove baseline bias, bounded to [-1, 1]. The extra pass trades throughput for scoring consistency.
- **Steering hooks are transient** — composed before generation, removed after.
- **Contrastive diffs in float32** — fp16 subtraction between close vectors loses precision.
- **MPS discipline** — diffs kept on CPU (SVD runs there anyway), `torch.mps.empty_cache()` between extraction passes, `torch.device(target)` model loading to avoid unified-memory RSS spike.

## Bundled statements

21 curated concepts at **n=45 pairs each**, stored in `saklas/data/vectors/<canonical>/statements.json`. Most are **bipolar** with composite names (`happy.sad`, `masculine.feminine`, `high_context.low_context`): generated via `Speaker A IS <pos> / Speaker B IS <neg>` to produce sharp axes where negative α is a real coherent pole, not "absence of X." Two monopolar exceptions (`agentic`, `manipulative`) use `Speaker B is unrelated`. Both modes produce topically disjoint pairs — broader contrastive directions than minimal-word-swap. `_REF_SCORE = 1/32` is calibrated against the bipolar-dominant 21-probe pack; recalibrate if the generation mode or topical breadth changes materially. 45 neutral statements in `saklas/data/neutral_statements.json` follow the same generation discipline (affect-neutral, topically diverse, no stylistic voice). All generated via `scripts/regenerate_bundled_statements.py` on gemma-4-31b-it. The script reads an explicit `BIPOLAR`/`MONOPOLAR` manifest (not a folder walk) so concept identity is authoritative; `--purge` wipes the data dir before regeneration.

**Canonical naming rule.** `saklas.session.canonical_concept_name(concept, baseline=None)` slugs each pole via `[^a-z0-9]+ → _` (so `high-context` → `high_context`) and joins bipolar poles with the separator `.` from `BIPOLAR_SEP`. `session.extract()` returns `(canonical_name, profile)`; all cache paths, pack folders, and registry keys use the canonical name. `/steer happy - sad` and `/steer happy.sad` resolve to the same vector. `NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$` — `.` and `_` allowed, `@` still forbidden (HF revision separator). `.` chosen over `~` because HF Hub repo names reject `~` and packs round-trip through HF as model repos.

**Pole aliasing.** `cli_selectors.resolve_pole(raw, namespace=None)` scans `_all_concepts()` across every namespace and returns `(canonical, sign, match)`. Typing a bare pole name (e.g. `wolf`) on top of an installed bipolar pack (`bob/deer.wolf`) returns `("deer.wolf", -1, <resolved>)`; the caller multiplies the user's α by `sign` before storing. Monopolar exact matches and composite-literal inputs pass through unchanged. Ambiguity (multiple installed packs match under different canonicals, or the same canonical across multiple namespaces) raises `AmbiguousSelectorError` — the user disambiguates with `ns/name`. The explicit bipolar form `/steer pos - neg` skips resolution entirely so user-declared poles always win.

**Extract lookup broadened.** `session.extract()` no longer restricts its curated lookup to `default/` — it scans `cli_selectors._all_concepts()` for the canonical name across every namespace, so `/steer bob/deer.wolf` hits an HF-pulled pack without needing to be bundled.

Pair-count note: going below n=32 inflates mean PCA scores by ~38% (small-sample bias), which destabilizes `_REF_SCORE / mean_score` normalization and breaks cross-concept alpha consistency. n=45 is the measured sweet spot — 12% mean inflation vs n=60, no layer-direction flips, 25% faster extraction.

## Supported architectures

`model.py:_LAYER_ACCESSORS`. Add new architecture = add one entry. 53 entries covering: llama (1–4), mistral/ministral/mixtral, gemma (1–4, text+recurrent), phi/phi3/phimoe, qwen (1–3.5, moe variants), cohere (1–2), deepseek (v2/v3), starcoder2, olmo (1–3, moe), glm (3–4, moe), granite/granitemoe, nemotron, stablelm, gpt2/neo/j/bigcode/neox/oss, bloom, falcon/falcon_h1, mpt, dbrx, opt.

## Testing

GPU-required (CUDA or MPS): `test_smoke.py`, `test_session.py`. Downloads `google/gemma-3-4b-it` (~8GB) on first run. `device="auto"` picks cuda > mps > cpu. MPS runs ~3–5× slower, so `test_extraction_fast_enough` uses a backend-specific budget (10s CUDA / 60s MPS).

CPU-only: `test_paths`, `test_packs`, `test_selectors`, `test_cache_ops`, `test_hf`, `test_merge`, `test_config_file`, `test_cli_flags`, `test_probes_bootstrap`, `test_results`, `test_datasource`, `test_server`. Cover: pack format, slim sidecars, integrity/staleness, selector grammar, cache ops (including `-r` skipping locals), HF wrappers (mocked), merge composition, config loading, CLI flag parsing, profile extraction, steering effect, hook cleanup, save/load roundtrip, monitor history, throughput regression, `build_chat_input`, `bootstrap_probes`, DataSource parsing, ResultCollector export, monitor scoring (`measure_from_hidden`, history accumulation, sparkline growth), SaklasSession lifecycle/generation/streaming, API server endpoints/streaming/CLI parsing.
