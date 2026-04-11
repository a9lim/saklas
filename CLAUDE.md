# CLAUDE.md

## What this is

`steer` is a local activation steering + trait monitoring TUI for HuggingFace causal LMs. Extracts steering vectors (PCA on contrastive differences), injects them via forward hooks during generation, monitors activations against probe vectors in real time. Textual-based three-column TUI.

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
- `vectors.py` — Separate forward passes per prompt (no batching — multimodal models produce corrupted hidden states with padded batches). `_capture_all_hidden_states` hooks every layer in a single forward pass, returning `{layer_idx: hidden_state}`. `_encode_and_capture_all` handles tokenization, chat-template wrapping (instruct models get assistant-role framing; base models get raw strings), BOS fallback, and attention-weighted pooling (last token's self-attention from the last layer, averaged across heads; falls back to last-token pooling). `extract_contrastive`: 2N passes for N contrastive pairs, captures all layers per pass. Per-layer SVD extracts the first principal component; each layer is scored by explained variance ratio (σ₁/Σσᵢ). Layers below 10% of max score are dropped. Returns a **profile**: `dict[int, (Tensor, score)]` mapping layer indices to direction vectors and signal strengths. Single-pair case uses raw diff norm as score proxy. Profiles saved as multi-tensor `.safetensors` (keys `layer_{i}`) + `.json` sidecar with scores. Cache filenames are sanitized (`re.sub` on unsafe chars), format: `{concept}.safetensors` (no layer suffix).
- `probes_bootstrap.py` — Loads/extracts probe profiles per `steer/probes/defaults.json`, cached under `steer/probes/cache/{model_name}/`. Returns `dict[str, profile]`. `defaults.json` maps category name to list of probe names; dataset file is `{probe_name}.json`. Categories: emotion (8), personality (7), safety (4), cultural (6), gender (3) — 28 probes total. Clear `steer/probes/cache/` to force re-extraction.

### Steering + Monitoring layer
- `hooks.py` — `SteeringHook` adds a pre-composed vector to hidden states in-place via `register_forward_hook`. Handles both tuple and bare tensor output. `SteeringManager.add_vector(name, profile, alpha)` takes a profile; `apply_to_model` iterates each vector's profile entries with effective alpha = `alpha * score`, groups by layer, orthogonalizes per layer group (Gram-Schmidt), recomposes into one hook per active layer. No user-facing layer selection — layers are determined automatically by extraction scores.
- `monitor.py` — `TraitMonitor` accepts probe profiles. Each probe monitors its **peak layer** (highest score in its profile). Hooks are grouped by peak layer — one hook per distinct layer, each with its own probe sub-matrix and GPU buffer. Per-layer hook: one matmul per token yields cosine similarities. `flush_to_cpu()` merges across layer buffers into unified history and running stats. `has_pending_data()` gates TUI polling. `add_probe`/`remove_probe` accept profiles and trigger full hook rebuild.

### Generation loop
- `generation.py` — Token-by-token with KV cache. Top-p sampling via `torch.topk` (k capped at 1024). Single `torch.inference_mode()` wraps the entire loop. Attention mask used only for prefill, `None` thereafter. EOS checks both `generation_config` and `tokenizer`. Greedy/sampling share unified token-accounting code.

### TUI layer (Textual)
Three-column layout: left (1fr) | center (2fr) | right (1fr).

- `tui/app.py` — Orchestrates everything. Polls at ~15 FPS; trait panel only updates on `has_pending_data()`. Status bar updates gated on change (skips idle repaints). Panel focus via Tab/Shift+Tab. Commands: `/steer "concept" [alpha]`, `/probe "concept"`, `/clear`, `/rewind`, `/sys`, `/temp`, `/top-p`, `/max`, `/help`. No layer arguments — extraction automatically profiles all layers. `/steer` and `/probe` share `_extract_vector_worker` for the contrastive extraction pipeline and cache. Curated dataset fallback when concept matches a probe name in `defaults.json`.
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
- **Vectors scaled to 10% of mean hidden-state norm** at each extraction layer.
- **Device/dtype cached** in `SteerApp.__init__`. Never call `next(model.parameters())` in handlers.
- **Monitor hooks**: One hook per distinct peak-layer, each with its own probe sub-matrix and GPU buffer. Per-hook: branchless `last_state / norm.clamp(min=1e-8)` — no CPU sync. `flush_to_cpu()` iterates layer buffers, vectorizes stats across probe dimension. TUI polls gated on `has_pending_data()`. History uses `deque(maxlen=8)` to cap memory (only sparkline and current/previous consume it).
- **TUI panel caches**: Widget references cached in `on_mount()` — no `query_one()` in render/poll paths. Stats lines and sort order cached with identity/value checks. Bar rendering shared via `_build_bar()` helper.
- **SteeringManager**: Name→index dict for O(1) lookup in `set_alpha`/`toggle_vector`.
- **`recompose`**: `torch.stack` + broadcasted multiply + `.sum()`, not Python `sum()`.

## Supported architectures

`model.py:_LAYER_ACCESSORS`. Add new architecture = add one entry. Currently: llama, llama4, llama4_text, mistral, mixtral, gemma, gemma2, gemma3, gemma3_text, gemma4, gemma4_text, recurrent_gemma, phi, phi3, phimoe, qwen, qwen2, qwen2_moe, qwen3, qwen3_moe, cohere, cohere2, deepseek_v2, deepseek_v3, starcoder2, olmo, olmo2, olmoe, glm, glm4, granite, granitemoe, nemotron, stablelm, gpt2, gpt_neo, gptj, gpt_bigcode, bloom, falcon, gpt_neox, mpt, dbrx, opt.

## Testing

Smoke tests require CUDA and download `google/gemma-2-2b-it` on first run. Coverage: profile extraction (all-layer contrastive PCA), steering effect, hook cleanup (verifies unsteered baseline match), profile save/load roundtrip, multi-layer monitor history, throughput regression, `build_chat_input`, `bootstrap_probes` cache path.
