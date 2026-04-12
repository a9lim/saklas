# CLAUDE.md

## What this is

`liahona` is a Python library and TUI for activation steering + trait monitoring on HuggingFace causal LMs. Extracts steering vectors (contrastive PCA / RepE), applies them per-generation via forward hooks, monitors activations against probe vectors. Usable headlessly via `LiahonaSession` or interactively via Textual TUI.

## Commands

```bash
pip install -e ".[dev]"          # editable install + pytest
pip install -e ".[cuda]"         # bitsandbytes + flash-attn (CUDA only)
pip install -e ".[research]"     # datasets + pandas (for API users)
pip install -e ".[serve]"        # fastapi + uvicorn (for API server)
liahona <model_id>               # launch TUI
liahona serve <model_id>         # launch OpenAI-compatible API server
python -m liahona <model_id>     # alt entry point
pytest tests/test_smoke.py -v    # CUDA smoke tests (downloads gemma-2-2b-it ~5GB)
pytest tests/ -v                 # all tests (non-CUDA tests run anywhere)
```

## PyPI release

Package name is `liahona-ai` (`liahona` is blocked by PyPI). Import name stays `liahona`.

```bash
# bump version in pyproject.toml, then:
python3 -m build                   # creates dist/liahona_ai-X.Y.Z.tar.gz + .whl
twine upload dist/*                # uploads using ~/.pypirc token
```

`probes/defaults.json` is included via `[tool.setuptools.package-data]`. The `probes/cache/` directory (model-specific tensors) is not shipped.

## Architecture

Five layers: **model/vector**, **steering/monitoring**, **session API**, **TUI**, **API server**.

### Model + Vector layer
- `model.py` — Loads HF causal LMs. `_LAYER_ACCESSORS` maps `model_type` to layer-list accessor lambdas; add new architectures here. Cascading fallbacks: SDPA → eager attention, dtype → fp16/fp32, device → CPU (for weight conversion) then `.to(target)`. `_load_text_from_multimodal`: extracts text-only model from multimodal checkpoints (e.g. Ministral tagged as Mistral3) — creates model with `torch.device(target)`, loads safetensors shards with `language_model.` prefix stripping, dequantizes FP8 weights inline. Patches `torch.histc` for MPS integer tensor support (MoE routing).
- `vectors.py` — Per-prompt forward passes (no batching). `_capture_all_hidden_states` hooks every layer in one pass. `_encode_and_capture_all` handles tokenization, chat-template wrapping, attention-weighted pooling (falls back to last-token pooling on OOM). `extract_contrastive`: 2N passes for N pairs, casts to float32 before differencing (fp16 subtraction loses precision), diffs kept on CPU for MPS. Per-layer SVD extracts first principal component, scored by explained variance ratio. MPS cache flushed between pairs. Returns a **profile**: `dict[int, (Tensor, score)]` mapping every layer to direction + signal strength. Profiles saved as `.safetensors` + `.json` sidecar. `compute_layer_means`: 45 neutral prompts → per-layer mean hidden state for centering.
- `probes_bootstrap.py` — Loads/extracts probe profiles per `liahona/probes/defaults.json`. 28 probes across 5 categories (emotion, personality, safety, cultural, gender). `bootstrap_layer_means`: loads or computes per-layer mean activations, cached as `_LAYERMEANS.safetensors` per model. MPS cache flushed between probe extractions.

### Steering + Monitoring layer
- `hooks.py` — `SteeringHook` adds pre-composed vector to hidden states in-place. `SteeringManager` groups vectors by layer, orthogonalizes per layer (Gram-Schmidt), one hook per active layer.
- `monitor.py` — `TraitMonitor` runs a single post-generation forward pass over the generated text using attention-weighted pooling. Mean-centers hidden states (subtracting per-layer means computed from neutral prompts) before computing score-weighted cosine similarities against probe vectors. One value per probe per generation. No hooks on the model during generation.

### Session API layer
- `session.py` — `LiahonaSession` is the programmatic API and the TUI's backend. Owns model, vector registry (`_profiles`), monitor, generation config, conversation history. Key design: **vectors are registered without alphas** via `steer(name, profile)`, alphas are supplied per-generation via `generate(input, alphas={"name": 0.15})`. No persistent steering hooks between generations. Orthogonalize and thinking are per-call parameters. `thinking=True` enables thinking/reasoning mode for models that support it (gated by `supports_thinking`); the session decodes only the response portion (`generated_ids[thinking_end_idx:]`) for conversation history. Full extraction pipeline: cache -> curated dataset -> statement cache -> model-generated pairs -> contrastive PCA -> save. Statement cache is model-independent (stored under `datasets/cache/`), so generated pairs are reused across models.
- `datasource.py` — `DataSource` normalizes contrastive pairs from curated names, JSON, CSV, HF datasets, or raw Python lists.
- `results.py` — `GenerationResult`, `TokenEvent`, `ProbeReadings` dataclasses with `to_dict()`. `TokenEvent` has a `thinking: bool` flag for thinking-mode tokens. `ResultCollector` accumulates results for batch export (dicts, JSONL, CSV, DataFrame).

### Generation loop
- `generation.py` — Token-by-token with KV cache. Top-p via `torch.topk` (k capped at 1024). `torch.inference_mode()` wraps entire loop. MPS sync at end of generation prevents Metal command buffer reuse crashes. The `None` end-of-generation sentinel is **not** emitted by `generate_steered` — the TUI's `_generate` closure puts it on the queue after updating `_messages`, so pending actions see the final conversation state. `supports_thinking(tokenizer)` checks if the chat template has `enable_thinking` support. `_detect_think_delimiters(tokenizer)` returns `(start_id, end_id, starts_in_thinking)` by rendering a round-trip assistant message through the template — works across model families (Qwen `</think>`, Gemma `<channel|>`, etc.) without hardcoding delimiter strings. When `thinking=True`, the generation loop uses a state machine: idle → preamble (start delimiter + channel label suppressed) → thinking → done. For Qwen-style models where `<think>` is in the generation prompt, starts directly in thinking state. For Gemma-style models where the model explicitly opens a channel, starts idle and enters thinking only if the start token is generated. Token table uses `tokenizer.decode([id])` for correct byte-level BPE rendering (handles both SentencePiece `▁` and GPT-2 `Ġ` space markers). Tokens that decode to `\ufffd` (partial UTF-8 byte sequences, e.g. multi-token emoji) are marked `None` in the table and buffered during generation until a complete-token follows, then flushed via `tokenizer.decode(pending_ids)`.

### TUI layer (Textual)
- `tui/app.py` — Thin frontend over `LiahonaSession`. Owns local alpha/enabled/orthogonalize/thinking state, passes through to session at generation time. Thinking defaults ON for models that support it (Ctrl+T toggles). Polls at ~15 FPS. Commands: `/steer`, `/probe`, `/clear`, `/rewind`, `/sys`, `/temp`, `/top-p`, `/max`, `/help`. Mid-generation interruption: any action that conflicts with generation (Ctrl+R, new message, `/steer`, `/probe`, `/clear`, `/rewind`) stops the current generation and defers execution via `_pending_action`; `_poll_generation` dispatches the pending action once the worker thread finishes and the `None` sentinel is consumed.
- `tui/vector_panel.py` — Model info, vectors with alpha bars, generation config.
- `tui/chat_panel.py` — Message log, status bar, input field. `_AssistantMessage` renders thinking tokens in a collapsible section (expanded during streaming, collapsed on finalize) above the main response.
- `tui/trait_panel.py` — Per-probe bars + sparklines, sort modes.

### API server layer
- `server.py` — FastAPI app factory. OpenAI-compatible endpoints (`/v1/models`, `/v1/chat/completions`, `/v1/completions`) plus liahona-specific management (`/v1/liahona/vectors`, `/v1/liahona/probes`, `/v1/liahona/session`). Thin HTTP layer over `LiahonaSession` — no business logic. Steering params passed per-request via `steer` key in request body (includes `alphas`, `orthogonalize`, `thinking`), merged with server-startup defaults. When thinking is enabled, streaming chat responses emit thinking tokens as `reasoning_content` in the delta (following OpenAI convention). Single session, 409 on concurrent generation. SSE streaming for chat/completions and vector extraction progress.
- `cli.py` — Dispatches `liahona serve` vs default TUI mode. `serve` subcommand accepts `--host`, `--port`, `--steer name:alpha`, `--cors`.

## Performance rules

These matter for the throughput regression test (steered >= 85% of vanilla tok/s):

- **Hot-path hooks**: No Python allocation, no `.item()`, no CPU sync. In-place mutation only.
- **`torch.inference_mode()`** wrapping the entire generation loop.
- **In-place ops**: `logits.div_()`, `logits.clamp_()`, `probs.div_()`.
- **`torch.topk`** for top-p, not full-vocab sort. `k` capped at `min(1024, vocab)`.
- **Norm computations use `.float()`** — fp16 sum-of-squares overflows for hidden_dim >= 2048.
- **Vectors scaled to mean hidden-state norm** at each extraction layer. Alpha directly represents the fraction of activation magnitude (e.g. alpha=0.15 means 15% perturbation at high-signal layers).
- **Monitor is post-generation**: single forward pass after generation, no hooks during generation. Mean-centered cosine similarities remove baseline bias.
- **Steering hooks are transient**: composed before generation, removed after. No persistent hooks between generations.
- **Contrastive diffs in float32** — fp16 subtraction between close activation vectors loses precision, producing degenerate SVD inputs. Cast to float32 before differencing.
- **MPS memory discipline** — diffs kept on CPU (SVD runs there anyway). `torch.mps.empty_cache()` between forward passes in extraction loops. Attention capture falls back to last-token pooling on OOM. Model loading via `torch.device(target)` avoids CPU RSS spike on unified memory.

## Supported architectures

`model.py:_LAYER_ACCESSORS`. Add new architecture = add one entry. Currently: llama, llama4, llama4_text, mistral, mistral4, ministral, ministral3, mixtral, gemma, gemma2, gemma3, gemma3_text, gemma4, gemma4_text, recurrent_gemma, phi, phi3, phimoe, qwen, qwen2, qwen2_moe, qwen3, qwen3_moe, qwen3_5_text, qwen3_5_moe, cohere, cohere2, deepseek_v2, deepseek_v3, starcoder2, olmo, olmo2, olmo3, olmoe, glm, glm4, granite, granitemoe, nemotron, stablelm, gpt2, gpt_neo, gptj, gpt_bigcode, bloom, falcon, falcon_h1, gpt_neox, gpt_oss, mpt, dbrx, opt.

## Testing

Smoke tests require CUDA and download `google/gemma-2-2b-it` on first run. Non-CUDA tests (`test_results.py`, `test_datasource.py`, `test_server.py`) run anywhere. Session tests (`test_session.py`) require CUDA. Coverage: profile extraction, steering effect, hook cleanup, save/load roundtrip, monitor history, throughput regression, `build_chat_input`, `bootstrap_probes`, DataSource parsing, ResultCollector export, LiahonaSession lifecycle/generation/streaming, API server endpoints/streaming/CLI parsing.
