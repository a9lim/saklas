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

`bitsandbytes` is optional (`pip install -e ".[bnb]"`). Non-CUDA installs work without it.

## Architecture

Three layers: **model/vector**, **steering/monitoring**, **TUI**.

### Model + Vector layer
- `model.py` — Loads HF causal LMs. `_LAYER_ACCESSORS` maps `model_type` to layer-list accessor lambdas; add new architectures here. `_text_config()` resolves `hidden_size` through multimodal `text_config` wrappers.
- `vectors.py` — Separate forward passes per prompt (no batching — multimodal models produce corrupted hidden states with padded batches). Shared `_encode_and_capture` helper handles tokenization, BOS fallback, forward pass, and mean-pooling. `extract_actadd`: two passes. `extract_caa`: 2N passes for N contrastive pairs, averaged. Hook-based hidden state capture via `_capture_hidden_states_single`. Both accept optional `device` param to avoid `next(model.parameters()).device` overhead. Vectors saved as `.safetensors` + `.json` sidecar. Cache filenames are sanitized (`re.sub` on unsafe chars).
- `probes_bootstrap.py` — Loads/extracts probe vectors per `steer/probes/defaults.json`, cached under `steer/probes/cache/{model_name}/`. `defaults.json` maps category name to list of probe names; dataset file is `{probe_name}.json`. Categories: emotion (8), personality (7), safety (4), cultural (6), gender (3) — 28 probes total. Clear `steer/probes/cache/` to force re-extraction.

### Steering + Monitoring layer
- `hooks.py` — `SteeringHook` adds a pre-composed vector to hidden states in-place via `register_forward_hook`. Handles both tuple and bare tensor output. `SteeringManager` groups vectors by layer, handles orthogonalization (Gram-Schmidt), manages hook lifecycle.
- `monitor.py` — `TraitMonitor` attaches a read-only forward hook. Pre-stacks probes into a unit-normalized matrix; one matmul per token yields cosine similarities. GPU buffer auto-grows; `flush_to_cpu()` transfers to CPU history and updates running stats. `has_pending_data()` gates TUI polling.

### Generation loop
- `generation.py` — Token-by-token with KV cache. Top-p sampling via `torch.topk` (k capped at 1024). Single `torch.inference_mode()` wraps the entire loop. Attention mask used only for prefill, `None` thereafter. EOS checks both `generation_config` and `tokenizer`. Greedy/sampling share unified token-accounting code.

### TUI layer (Textual)
Three-column layout: left (1fr) | center (2fr) | right (1fr).

- `tui/app.py` — Orchestrates everything. Polls at ~15 FPS; trait panel only updates on `has_pending_data()`. Status bar updates gated on change (skips idle repaints). Panel focus via Tab/Shift+Tab. Commands: `/steer`, `/probe`, `/clear`, `/rewind`, `/sys`, `/temp`, `/top-p`, `/max`, `/help`. `/steer` and `/probe` share `_extract_vector_worker` for the CAA extraction pipeline and cache. Curated dataset fallback when concept matches a probe name in `defaults.json`.
- `tui/vector_panel.py` — Model info, vectors with inline alpha bars, generation config, keybinding reference.
- `tui/chat_panel.py` — Message log, status bar (tok/s, elapsed, VRAM), input field.
- `tui/trait_panel.py` — Per-probe bars + sparklines, stats row (mu, sigma, lo, hi, delta/tok), sort modes, keyboard navigation, `Ctrl+D` to remove probes.

## Performance rules

These matter for the throughput regression test (steered >= 85% of vanilla tok/s):

- **Hot-path hooks**: No Python allocation, no `.item()`, no CPU sync. In-place mutation only.
- **`torch.inference_mode()`** wrapping the entire generation loop.
- **In-place ops**: `logits.div_()`, `logits.clamp_()`, `probs.div_()`.
- **`torch.topk`** for top-p, not full-vocab sort. `k` capped at `min(1024, vocab)`, computed once before the loop.
- **No `.item()` in sampling** — zero-probability fallback uses `clamp(min=1e-8)` (branchless, GPU-only).
- **Norm computations use `.float()`** — fp16 sum-of-squares overflows for hidden_dim >= 2048.
- **Vectors scaled to 10% of mean hidden-state norm** at the extraction layer.
- **Device/dtype cached** in `SteerApp.__init__`. Never call `next(model.parameters())` in handlers.
- **Monitor hook**: branchless `last_state / norm.clamp(min=1e-8)` — no CPU sync. `flush_to_cpu()` vectorizes stats across probe dimension. TUI polls gated on `has_pending_data()`. History uses `deque(maxlen=8)` to cap memory (only sparkline and current/previous consume it).
- **TUI panel caches**: Widget references cached in `on_mount()` — no `query_one()` in render/poll paths. Stats lines and sort order cached with identity/value checks. Bar rendering shared via `_build_bar()` helper.
- **SteeringManager**: Name→index dict for O(1) lookup in `set_alpha`/`set_layer`/`toggle_vector`.
- **`recompose`**: `torch.stack` + broadcasted multiply + `.sum()`, not Python `sum()`.

## Supported architectures

`model.py:_LAYER_ACCESSORS`. Add new architecture = add one entry. Currently: llama, llama4, llama4_text, mistral, mixtral, gemma, gemma2, gemma3, gemma3_text, gemma4, gemma4_text, recurrent_gemma, phi, phi3, phimoe, qwen, qwen2, qwen2_moe, qwen3, qwen3_moe, cohere, cohere2, deepseek_v2, deepseek_v3, starcoder2, olmo, olmo2, olmoe, glm, glm4, granite, granitemoe, nemotron, stablelm, gpt2, gpt_neo, gptj, gpt_bigcode, bloom, falcon, gpt_neox, mpt, dbrx, opt.

## Testing

Smoke tests require CUDA and download `google/gemma-2-2b-it` on first run. Coverage: vector extraction (ActAdd + CAA), steering effect, hook cleanup (verifies unsteered baseline match), save/load roundtrip, monitor history, throughput regression, `build_chat_input`, `bootstrap_probes` cache path.
