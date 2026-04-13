# CLAUDE.md

## What this is

`saklas` is a Python library and TUI for activation steering + trait monitoring on HuggingFace causal LMs. Extracts steering vectors (contrastive PCA / RepE), applies them per-generation via forward hooks, monitors activations against probe vectors. Usable headlessly via `SaklasSession` or interactively via Textual TUI.

## Commands

```bash
pip install -e ".[dev]"          # editable install + pytest
pip install -e ".[cuda]"         # bitsandbytes + flash-attn (CUDA only)
pip install -e ".[research]"     # datasets + pandas (for API users)
pip install -e ".[serve]"        # fastapi + uvicorn (for API server)
saklas <model_id>               # launch TUI
saklas serve <model_id>         # launch OpenAI-compatible API server
python -m saklas <model_id>     # alt entry point
saklas -x <selector>            # delete tensors for matched concepts (keeps statements)
saklas -r <selector>            # re-pull concepts from source (bundled or HF)
saklas -i <ns>/<concept>        # install a pack from HF or a local folder path
saklas -l [selector]            # list or info-dump packs (installed + HF); exits
pytest tests/test_smoke.py -v    # GPU smoke tests (downloads gemma-3-4b-it ~8GB; CUDA or MPS)
pytest tests/ -v                 # all tests (non-CUDA tests run anywhere)
```

## PyPI release

Package name is `saklas`. Import name stays `saklas`.

```bash
# bump version in pyproject.toml, then:
python3 -m build                   # creates dist/saklas-X.Y.Z.tar.gz + .whl
twine upload dist/*                # uploads using ~/.pypirc token
```

`saklas/data/**/*.json` is included via `[tool.setuptools.package-data]`. The user cache under `~/.saklas/` is not shipped; it's materialized on first run by copying `saklas/data/` into `~/.saklas/`.

## Cache layout

All cache state lives under `~/.saklas/` (override via `SAKLAS_HOME` env var):

```
~/.saklas/
  neutral_statements.json                 # user-editable; materialized from saklas/data/
  vectors/
    default/                              # bundled-from-pip concepts (source=bundled, -r default)
      <concept>/
        pack.json                         # name, description, tags, recommended_alpha, source, files{sha256}
        statements.json                   # bare contrastive-pair list
        <safe_model_id>.safetensors       # extracted per-model tensor
        <safe_model_id>.json              # slim sidecar: method/scores/saklas_version/statements_sha256
    local/                                # user-authored / merged concepts (source=local, -r errors)
    <hf_owner>/                           # HF-pulled concepts (source=hf://<owner>/<name>)
  models/
    <safe_model_id>/                      # per-model derived artifacts
      layer_means.safetensors
      layer_means.json
```

Namespace resolution is optional: bare selectors (`happy`) resolve across all
namespaces and raise `AmbiguousSelectorError` on collision. Fully-qualified
selectors (`a9lim/happy`) are always unambiguous. Selector grammar shared by
`-r`, `-x`, `-l`, `parse_args`: `<name>`, `<ns>/<name>`, `tag:<t>`, `namespace:<ns>`,
`model:<id>`, `default`, `all`. `model:` is a resource scope that AND-combines
with the concept selector.

Integrity: every `pack.json` carries a `files` map of relative-path → sha256.
`ConceptFolder.load` verifies the map on every load. Per-tensor sidecars record
`statements_sha256` at extraction time; when the current statements hash
differs, `is_stale` flags the tensor and the CLI warns.

Signing is out of scope for Story A. `pack.json` reserves null `signature` and
`signature_method` fields as forward-compatibility hooks for v2 Ed25519 TOFU
signing (see `docs/superpowers/specs/2026-04-12-story-a-portability-design.md`).

## Architecture

Five layers: **model/vector**, **steering/monitoring**, **session API**, **TUI**, **API server**.

### Model + Vector layer
- `model.py` — Loads HF causal LMs. `_LAYER_ACCESSORS` maps `model_type` to layer-list accessor functions (`def`, not lambdas); add new architectures here. Cascading fallbacks: SDPA → eager attention, dtype → fp16/fp32, device → CPU (for weight conversion) then `.to(target)`. `_load_text_from_multimodal`: extracts text-only model from multimodal checkpoints (e.g. Ministral tagged as Mistral3) — creates model with `torch.device(target)`, loads safetensors shards with `language_model.` prefix stripping, dequantizes FP8 weights inline. Patches `torch.histc` for MPS integer tensor support (MoE routing). Uses `log.info()` for device/memory reporting.
- `vectors.py` — Per-prompt forward passes (no batching). `_capture_all_hidden_states` hooks every layer in one pass. `_encode_and_capture_all` handles tokenization, chat-template wrapping, and pools from the last content token — walks back past `tokenizer.all_special_ids` to skip trailing chat-template markers (Llama's `<|eot_id|>`, Gemma's `<end_of_turn>`, Qwen's `<|im_end|>`) whose hidden states are disconnected from content. No attention capture — the last content token's hidden state is already an attention-weighted summary of prior positions (what the model uses for next-token prediction). `extract_contrastive`: 2N passes for N pairs, casts to float32 before differencing (fp16 subtraction loses precision), `_mps` and `diff_device` computed once at top of function. Multi-pair: per-layer SVD extracts first principal component, scored by explained variance ratio. Single-pair: scores as `diff_norm / activation_norm` (ratio of contrast to activation magnitude), producing values in the same range as explained-variance-ratio (~0.01–0.4) so single-pair and multi-pair profiles contribute comparably. Normalization iterates `profile.items()` directly (no separate scores dict). MPS cache flushed between pairs. Returns a **profile**: `dict[int, (Tensor, score)]` mapping every layer to direction + signal strength. Profiles saved as `.safetensors` + `.json` sidecar (`save_profile` builds both dicts in a single loop). `compute_layer_means`: 45 neutral prompts → per-layer mean hidden state for centering. `_template_overhead_cache` is keyed by `id(tokenizer)` — safe only when one tokenizer lives for the session lifetime (object IDs can be reused after GC).
- `probes_bootstrap.py` — Walks `~/.saklas/vectors/default/` (after triggering first-run `materialize_bundled()`). 28 bundled probes live as per-concept folders under `saklas/data/vectors/<concept>/` in the wheel. `load_defaults()` groups concepts by `pack.json.tags`. `bootstrap_layer_means`: loads or computes per-layer mean activations, cached at `~/.saklas/models/<safe_model_id>/layer_means.safetensors` with a slim sidecar. Stale if `~/.saklas/neutral_statements.json` has changed since extraction. MPS cache flushed between probe extractions.

### Steering + Monitoring layer
- `hooks.py` — `SteeringHook` adds pre-composed vector to hidden states in-place. `SteeringManager` groups vectors by layer, orthogonalizes per layer (Gram-Schmidt), one hook per active layer. Vectors registered via `add_vector(name, profile, alpha)` — activation controlled by the session clearing and re-adding vectors, not by a toggle flag. Per-profile score normalization: `effective_alpha = alpha * score_i * (_REF_SCORE / mean_score)`. Raw PCA explained-variance-ratios differ 3× or more between architectures (diffuse small models like Llama-3.2-3B score ~0.07 while gemma-3-4b scores ~0.17 for the same statements), so without normalization the same `alpha` would hit a dead zone on one model and a coherence cliff on another. Dividing by the profile's mean preserves relative per-layer emphasis; re-multiplying by `_REF_SCORE = 0.125` pins the alpha scale so that α≈0.4–0.8 is the coherent nuanced band and α≈1.0 is past the cliff across every bundled architecture (verified on llama-3.2-3b-it, gemma-3-4b-it, Qwen3.5-4B, Ministral-3-8B).
- `monitor.py` — `TraitMonitor` scores all probes against a set of per-layer hidden states. Core scoring logic in `_score_probes(hidden_per_layer)`: mean-centers hidden states (subtracting per-layer means computed from neutral prompts) before computing score-weighted cosine similarities (both norms: `(h @ v) / (|h| * |v|)`) against probe vectors. `total_w` only accumulates scores for layers present in the hidden-state dict. One value per probe per generation, computed and recorded in a single loop. Two entry points: `measure(model, tokenizer, layers, text)` runs a forward pass (via `_encode_and_capture_all`) and feeds the last-content-token hidden states to `_score_probes`; `measure_from_hidden(hidden_per_layer)` scores from pre-captured hidden states directly (no forward pass). The session and TUI use `measure()` after generation, running a separate forward pass to score probes consistently with how probe vectors were extracted. History accumulates across generations (not reset per call). `probe_names` is a property over `_raw_profiles.keys()` (no separate list). `profiles` and `layer_means` properties return the underlying dicts directly (no defensive copies).

### Session API layer
- `session.py` — `SaklasSession` is the programmatic API and the TUI's backend. Owns model, vector registry (`_profiles`), monitor, generation config, conversation history. Public accessors: `model_id` property, `has_vector(name)` method, `model_info` dict. Key design: **vectors are registered without alphas** via `steer(name, profile)`, alphas are supplied per-generation via `generate(input, alphas={"name": 0.5})`. No persistent steering hooks between generations. Orthogonalize and thinking are per-call parameters. `thinking=True` enables thinking/reasoning mode for models that support it (gated by `supports_thinking`); the session decodes only the response portion (`generated_ids[thinking_end_idx:]`) for conversation history. `_generation_preamble()` deduplicates setup shared between blocking and streaming paths (use_thinking, prepare_input, gen_state reset, apply_steering). After generation, probe scoring runs a separate forward pass via `monitor.measure()` to match the last-content-token pooling used during probe extraction. Streaming uses a worker thread that puts `None` on the queue in its `finally` block as the done sentinel. Full extraction pipeline: cache → curated dataset → statement cache → model-generated pairs → contrastive PCA → save. Curated concepts save under `~/.saklas/vectors/default/<concept>/<safe_model>.safetensors`; user-extracted concepts save under `~/.saklas/vectors/local/<tag>/`. Statement caches are model-independent (stored alongside the tensor folder as `statements.json`), so generated pairs are reused across models. Every save also refreshes the owning `pack.json` `files` sha256 map via `_update_local_pack_files`. `_PAIR_RE` is compiled at module level. `MIN_ELAPSED_FOR_RATE` is a public constant.
- `datasource.py` — `DataSource` normalizes contrastive pairs from curated names, JSON, CSV, HF datasets, or raw Python lists. No `description` attribute.
- `results.py` — `GenerationResult`, `TokenEvent`, `ProbeReadings` dataclasses with `to_dict()`. `TokenEvent` has a `thinking: bool` flag for thinking-mode tokens. `ProbeReadings` fields: `mean`, `std`, `min`, `max`, `per_generation` (one value per `measure()` call), `delta_per_gen` (mean absolute per-generation change). `ResultCollector` accumulates results for batch export (dicts, JSONL, CSV, DataFrame).

### Generation loop
- `generation.py` — Token-by-token with KV cache. Top-p via `torch.topk` (k capped at 1024). `torch.inference_mode()` wraps entire loop. MPS sync at end of generation prevents Metal command buffer reuse crashes. The `None` end-of-generation sentinel is **not** emitted by `generate_steered` — the session streaming worker puts it on the queue in its `finally` block; the TUI's `_generate` closure puts it on the queue after updating `_messages`, so pending actions see the final conversation state. `supports_thinking(tokenizer)` delegates to `_detect_think_delimiters` and returns whether the result is non-trivial. `_detect_think_delimiters(tokenizer)` returns `(start_id, end_id, response_start_id, starts_in_thinking)` by rendering a round-trip assistant message through the template — works across model families (Qwen `</think>`, Gemma `<channel|>`, gpt-oss `<|channel|>…<|message|>`, etc.) without hardcoding delimiter strings. `response_start_id` handles channel-based formats where multiple tokens separate thinking from response (e.g. `<|channel|>response<|message|>`); it is `None` for formats where response follows immediately after the end delimiter. When `thinking=True`, the generation loop uses a state machine: idle → preamble (start delimiter + channel label suppressed) → thinking → response_preamble (post-thinking channel label suppressed, channel-based only) → done. For Qwen-style models where `<think>` is in the generation prompt, starts directly in thinking state. For Gemma-style models where the model explicitly opens a channel, starts idle and enters thinking only if the start token is generated. Token table uses `tokenizer.decode([id])` for correct byte-level BPE rendering (handles both SentencePiece `▁` and GPT-2 `Ġ` space markers). Tokens that decode to `\ufffd` (partial UTF-8 byte sequences, e.g. multi-token emoji) are marked `None` in the table and buffered during generation until a complete-token follows, then flushed via `tokenizer.decode(pending_ids)`. `on_token(text, is_thinking, token_id)` callback passes the actual token ID for single tokens (`-1` for multi-byte buffer flushes). `_tok_key(tokenizer)` is a shared helper for cache key construction used by `_eos_cache`, `_token_table_cache`, and `_think_delim_cache`. `GenerationState` has `stop_requested` (Event), `token_queue` (SimpleQueue), and `thinking_end_idx` — no `is_generating` event.

### TUI layer (Textual)
- `tui/app.py` — Thin frontend over `SaklasSession`. Owns local alpha/enabled/orthogonalize/thinking state, passes through to session at generation time. Thinking defaults ON for models that support it (Ctrl+T toggles). Polls at ~15 FPS. Commands: `/steer`, `/probe`, `/clear`, `/rewind`, `/sys`, `/temp`, `/top-p`, `/max`, `/help`. Mid-generation interruption: any action that conflicts with generation (Ctrl+R, new message, `/steer`, `/probe`, `/clear`, `/rewind`) stops the current generation and defers execution via `_pending_action`; `_poll_generation` dispatches the pending action once the worker thread finishes and the `None` sentinel is consumed. Panel focus uses index constants (`_LEFT`, `_CHAT`, `_TRAIT`) instead of string comparisons. `_adjust_config(attr, delta, lo, hi)` unifies the temperature/top-p adjustment actions. `_parse_args` captures the first parseable float as a scalar (no accumulation list).
- `tui/utils.py` — Shared TUI helpers. `build_bar(value, max_value, width)` renders a filled/empty bar pair.
- `tui/vector_panel.py` — Model info, vectors with alpha bars, generation config. Imports `build_bar` from `tui.utils`.
- `tui/chat_panel.py` — Message log, status bar, input field. `_AssistantMessage` renders thinking tokens in a collapsible section (expanded during streaming, collapsed on finalize) above the main response. Owns `append_token(token)` and `append_thinking_token(token)` methods for self-contained string accumulation.
- `tui/trait_panel.py` — Per-probe bars + sparklines, sort modes. `_nav_items` is `list[str]` (probe names directly, no type-tag tuples). Imports `build_bar` from `tui.utils`.

### API server layer
- `server.py` — FastAPI app factory. OpenAI-compatible endpoints (`/v1/models`, `/v1/chat/completions`, `/v1/completions`) plus saklas-specific management (`/v1/saklas/vectors`, `/v1/saklas/probes`, `/v1/saklas/session`). Thin HTTP layer over `SaklasSession` — no business logic. Uses `session.model_id` for model identification. Model `created` timestamp captured once at app creation (`app.state.created_ts`). Steering params passed per-request via `steer` key in request body (includes `alphas`, `orthogonalize`, `thinking`), merged with server-startup defaults. `thinking` is wired through both chat and completions routes. `_stream_generation` is a unified SSE generator parameterized on `object_type` and `format_delta` — used by both chat and completions streaming. When thinking is enabled, streaming chat responses emit thinking tokens as `reasoning_content` in the delta (following OpenAI convention). Single session, 409 on concurrent generation. Vector extraction streaming is intentionally synchronous (progress replayed after completion) — the JSON path is the primary interface.
- `cli.py` — Dispatches `saklas serve` vs default TUI/cache/list modes. `_print_startup(args)` shared between TUI and serve. `serve` subcommand accepts `--host`, `--port`, `--steer name:alpha`, `--cors`. Main TUI parser takes the Story A flag grammar: `-r` (refresh), `-x` (clear-tensors), `-i` (install), `-l` (list — exits), `-m <name> <components>` (merge), `-C <path>` (config file), `--strict`. Cache ops compose with model load: `saklas -r default gemma-2-2b-it` refreshes then launches. `-l` is exit-only. `-r`/`-x`/`-i` use `action="append"` single-token semantics; compound selectors require repeated flags (e.g. `-x tag:emotion -x model:gemma-2-2b-it`). Config loading via `-C` composes multiple YAML files, applies flag overrides, auto-installs missing vectors (`ensure_vectors_installed`), and the composed vector map lands on `args.config_vectors` for `_run_tui` to register. `print_migration_notice_if_needed()` runs once at startup to flag legacy `probes/cache` / `~/.liahona` detritus.

## Performance rules

These matter for the throughput regression test (steered >= 85% of vanilla tok/s):

- **Hot-path hooks**: No Python allocation, no `.item()`, no CPU sync. In-place mutation only.
- **`torch.inference_mode()`** wrapping the entire generation loop.
- **In-place ops**: `logits.div_()`, `logits.clamp_()`, `probs.div_()`.
- **`torch.topk`** for top-p, not full-vocab sort. `k` capped at `min(1024, vocab)`.
- **Norm computations use `.float()`** — fp16 sum-of-squares overflows for hidden_dim >= 2048.
- **Vectors scaled to mean hidden-state norm** at each extraction layer, then gated per-layer by `score_i * (_REF_SCORE / mean_score)` at apply time (see `hooks.py`). The user-facing alpha is normalized so α≈0.5 is the coherent nuanced band and α≈1.0 is past the collapse cliff regardless of backbone.
- **Monitor pools from the last content token**: after generation, a separate forward pass (`_encode_and_capture_all`) captures hidden states and pools from the last non-special token at each layer, matching probe extraction. Mean-centered cosine similarities (both norms) remove baseline bias, bounded to [-1, 1]. The extra forward pass trades throughput for scoring consistency with probe extraction.
- **Steering hooks are transient**: composed before generation, removed after. No persistent hooks between generations.
- **Contrastive diffs in float32** — fp16 subtraction between close activation vectors loses precision, producing degenerate SVD inputs. Cast to float32 before differencing.
- **MPS memory discipline** — diffs kept on CPU (SVD runs there anyway). `torch.mps.empty_cache()` between forward passes in extraction loops. Model loading via `torch.device(target)` avoids CPU RSS spike on unified memory.

## Supported architectures

`model.py:_LAYER_ACCESSORS`. Add new architecture = add one entry. Currently: llama, llama4, llama4_text, mistral, mistral4, ministral, ministral3, mixtral, gemma, gemma2, gemma3, gemma3_text, gemma4, gemma4_text, recurrent_gemma, phi, phi3, phimoe, qwen, qwen2, qwen2_moe, qwen3, qwen3_moe, qwen3_5, qwen3_5_text, qwen3_5_moe, cohere, cohere2, deepseek_v2, deepseek_v3, starcoder2, olmo, olmo2, olmo3, olmoe, glm, glm4, glm4_moe_lite, granite, granitemoe, nemotron, stablelm, gpt2, gpt_neo, gptj, gpt_bigcode, bloom, falcon, falcon_h1, gpt_neox, gpt_oss, mpt, dbrx, opt.

## Testing

Smoke and session tests require a GPU backend (CUDA or Apple Silicon MPS) and download `google/gemma-3-4b-it` on first run. `device="auto"` picks cuda > mps > cpu via `detect_device`. MPS runs roughly 3–5× slower than CUDA for this model, so `test_extraction_fast_enough` uses a backend-specific timing budget (10s CUDA, 60s MPS). Non-GPU tests (`test_paths.py`, `test_packs.py`, `test_selectors.py`, `test_cache_ops.py`, `test_hf.py`, `test_merge.py`, `test_config_file.py`, `test_cli_flags.py`, `test_probes_bootstrap.py`, `test_results.py`, `test_datasource.py`, `test_server.py`) run anywhere. Coverage: pack format, slim sidecars, integrity/staleness, selector grammar, cache ops, HF wrappers (mocked), merge composition, config loading, CLI flag parsing, profile extraction, steering effect, hook cleanup, save/load roundtrip, monitor history, throughput regression, `build_chat_input`, `bootstrap_probes`, DataSource parsing, ResultCollector export, monitor scoring (`measure_from_hidden`, history accumulation, sparkline growth), SaklasSession lifecycle/generation/streaming, API server endpoints/streaming/CLI parsing.
