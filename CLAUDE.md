# CLAUDE.md

## What this is

`steer` is a local activation steering + trait monitoring TUI for HuggingFace causal LMs. Extracts steering vectors (ActAdd/CAA), injects them via forward hooks during generation, monitors activations against probe vectors in real time. Textual-based three-column TUI.

## Commands

```bash
pip install -e ".[dev]"          # editable install + pytest
pip install -e ".[cuda]"         # bitsandbytes + flash-attn (CUDA only)
steer <model_id>                 # launch TUI
python -m steer <model_id>       # alt entry point
pytest tests/test_smoke.py -v    # smoke tests (CUDA, downloads gemma-2-2b-it ~5GB)
```

`bitsandbytes` is an optional dependency (`pip install -e ".[bnb]"`). Non-CUDA installs work without it.

## Architecture

Three layers: **model/vector**, **generation**, **TUI**.

### Model + Vector layer
- `model.py` — Loads HF causal LMs. `_LAYER_ACCESSORS` maps `model_type` to layer-list accessor lambdas; add new architectures here. `_text_config()` resolves `hidden_size` through multimodal `text_config` wrappers. `get_model_info()` returns model metadata including `param_count`.
- `vectors.py` — `extract_actadd`: separate forward passes for concept and baseline (no batching — avoids fully-masked padding corruption). Baseline must be a semantically meaningful contrastive term (not empty string) for the steering direction to be useful. `extract_caa`: contrastive pairs in a single batched pass. `extract_actadd_batched`: batched ActAdd for probe bootstrap with per-concept `ref_norm`. All extraction uses `torch.inference_mode()`, `use_cache=False`. Hook-based hidden state capture via `_capture_hidden_states_single`/`_batched` for robustness with multimodal wrappers. When using `output_hidden_states` fallback, index is `layer_idx + 1` (index 0 = embedding). Vectors saved as `.safetensors` + `.json` sidecar.
- `probes_bootstrap.py` — Loads/extracts probe vectors per `steer/probes/defaults.json`, cached under `steer/probes/cache/{model_name}/`. Cache misses from `FileNotFoundError` are silent; other load errors log a warning and re-extract. Categories: emotion, personality, safety, cultural, gender.

### Steering + Monitoring layer
- `hooks.py` — `SteeringHook` adds a pre-composed vector to hidden states in-place via `register_forward_hook`. Handles both tuple and bare tensor output. `recompose` uses stack+matmul. `SteeringManager` groups vectors by layer, handles orthogonalization (Gram-Schmidt), manages hook lifecycle. `set_layer(name, layer_idx)` only updates the index — `apply_to_model()` handles all hook attach/detach.
- `monitor.py` — `TraitMonitor` attaches a read-only forward hook. Pre-stacks probes into a unit-normalized matrix; one matmul per token yields cosine similarities. GPU buffer batches results and auto-grows (doubles) when full; `flush_to_cpu()` transfers to CPU history and updates running stats (count/sum/sum_sq/min/max/first/last). `has_pending_data()` checks if buffer has unflushed data. `get_current()`/`get_previous()` read from CPU history only — caller must `flush_to_cpu()` first. `get_stats(name)` returns pre-computed O(1) stats dict. `add_probe`/`remove_probe` flush and resize the GPU buffer.

### Generation loop
- `generation.py` — Token-by-token with KV cache. Top-p sampling via `torch.topk` (k computed once before the loop from `model.config.vocab_size`). Single `torch.inference_mode()` wraps the entire loop. Logits sanitized with `nan_to_num` + clamp to [-100, 100]. Zero-probability guard: top token clamped to min 1e-8 prob (branchless, no GPU→CPU sync). EOS checks both `model.generation_config.eos_token_id` and `tokenizer.eos_token_id`. Empty/partial generations on stop are not appended to chat history.

### TUI layer (Textual)
Three-column layout: left (1fr) | center (2fr) | right (1fr).

- `tui/app.py` — `SteerApp` orchestrates everything. Caches `self._device`/`self._dtype`/`self._device_str` on init. Polls token queue + monitor at ~15 FPS; trait panel only updates when `has_pending_data()` is true (no idle CPU waste). VRAM polled every ~1s during generation, cached when idle. Panel focus system: Tab/Shift+Tab cycles focus between left/chat/trait panels; arrows routed to focused panel; hjkl (alpha/layer) are global (work from any panel except chat input). Key handling uses `on_key` override (not BINDINGS) for Tab, arrows, hjkl, Enter — Textual's Screen/Input intercept these before app-level bindings fire. Generation stats (token count, tok/s, elapsed) tracked and frozen on completion, reset on new generation. A/B compare (`Ctrl+A`) guarded by `_ab_in_progress` flag — blocks steering mutations during unsteered generation. Probe categories loaded from `defaults.json` at init, passed to `TraitPanel`. Commands: `/steer <concept> <baseline> [alpha] [layer]`, `/clear`, `/sys`, `/temp`, `/top-p`, `/max`, `/probes`, `/help`.
- `tui/vector_panel.py` — `LeftPanel` widget: model info, vectors with inline alpha bars and layer position visualizer, generation config with visual bars, keybinding reference. All vectors show full state; selected vector gets expanded detail.
- `tui/chat_panel.py` — `ChatPanel`: message log, status bar (gen indicator, token progress, tok/s, elapsed, prompt tokens, VRAM), input field. `_AssistantMessage(Static)` subclass tracks streaming text via `chat_text` attribute.
- `tui/trait_panel.py` — `TraitPanel`: accepts `categories` dict from `defaults.json` (no hardcoded probe list). Collapsible categories, per-probe bars + inline mini sparklines (8 chars), expandable stats row using pre-computed running stats from `TraitMonitor`. Keyboard navigation through probes and categories via `nav_up`/`nav_down`/`nav_enter`. Sort modes: name/magnitude/change. Renders via single `Static.update()` call (not DOM rebuild) for 15 FPS poll compatibility. Only re-renders when new monitor data arrives (`has_pending_data()` gate).
- `tui/styles.tcss` — Three-column CSS with `.focused` class for panel highlight.

### Data flow
1. User submits message → `ChatPanel.UserSubmitted` → `SteerApp._start_generation()`
2. Worker thread runs `generate_steered()` → forward hooks inject steering vectors + record monitor similarities
3. Poll timer (~15 FPS) drains token queue, updates status bar stats, flushes monitor buffer to CPU, updates trait panel with current values + histories

## Performance rules

These matter for the throughput regression test (steered >= 85% of vanilla tok/s):

- **Hot-path hooks**: No Python allocation, no `.item()`, no CPU sync. In-place mutation, return `output` directly. Both hooks handle bare tensor (Gemma 4) and tuple output.
- **`torch.inference_mode()`** everywhere, wrapping the entire generation loop (not per-token).
- **In-place ops in generation**: `logits.div_(temperature)`, `logits.clamp_()`, `probs.div_()`.
- **`torch.topk`** for top-p, not full-vocab sort. `k` computed once before the loop.
- **No `.item()` in the sampling loop** — zero-probability fallback uses `clamp(min=1e-8)` on the top token (branchless, GPU-only).
- **Norm computations in extraction use `.float()`** — fp16 sum-of-squares overflows for hidden_dim >= 2048. Hard requirement.
- **No mean-centering** in extraction — `(pos - center) - (neg - center) == pos - neg`.
- **Vectors scaled to 10% of mean hidden-state norm** at the extraction layer. Uniform across architectures regardless of absolute norm scale.
- **Device/dtype cached** in `SteerApp.__init__`. Never call `next(model.parameters())` in handlers or poll loops.
- **Monitor**: `flush_to_cpu()` before `get_current()`/`get_previous()`. Both methods read CPU history only. GPU buffer auto-grows when full. Running stats updated incrementally in flush — O(1) stats access, no per-render history iteration. TUI polls gated on `has_pending_data()` to avoid idle CPU waste.
- **`recompose`**: `torch.stack` + broadcasted multiply + `.sum()`, not Python `sum()`.

## Supported architectures

`model.py:_LAYER_ACCESSORS`. Add new architecture = add one entry. Common patterns: `model.model.layers` (most), `model.transformer.h` (GPT-2/Bloom/Falcon), `model.model.language_model.layers` (Gemma 3/4 multimodal). Currently: llama, llama4, llama4_text, mistral, mixtral, gemma, gemma2, gemma3, gemma3_text, gemma4, gemma4_text, recurrent_gemma, phi, phi3, phimoe, qwen, qwen2, qwen2_moe, qwen3, qwen3_moe, cohere, cohere2, deepseek_v2, deepseek_v3, starcoder2, olmo, olmo2, olmoe, glm, glm4, granite, granitemoe, nemotron, stablelm, gpt2, gpt_neo, gptj, gpt_bigcode, bloom, falcon, gpt_neox, mpt, dbrx, opt.

## Testing

Smoke tests require CUDA and download `google/gemma-2-2b-it` on first run. Coverage: vector extraction (valid shape + finite norm), steering effect, hook cleanup, save/load roundtrip, monitor history, throughput regression (steered >= 85% of vanilla tok/s).
