# CLAUDE.md

## What this is

`steer` is a Python library and TUI for activation steering + trait monitoring on HuggingFace causal LMs. Extracts steering vectors (contrastive PCA / RepE), applies them per-generation via forward hooks, monitors activations against probe vectors. Usable headlessly via `SteerSession` or interactively via Textual TUI.

## Commands

```bash
pip install -e ".[dev]"          # editable install + pytest
pip install -e ".[cuda]"         # bitsandbytes + flash-attn (CUDA only)
pip install -e ".[research]"     # datasets + pandas (for API users)
steer <model_id>                 # launch TUI
python -m steer <model_id>       # alt entry point
pytest tests/test_smoke.py -v    # CUDA smoke tests (downloads gemma-2-2b-it ~5GB)
pytest tests/ -v                 # all tests (non-CUDA tests run anywhere)
```

## Architecture

Four layers: **model/vector**, **steering/monitoring**, **session API**, **TUI**.

### Model + Vector layer
- `model.py` — Loads HF causal LMs. `_LAYER_ACCESSORS` maps `model_type` to layer-list accessor lambdas; add new architectures here.
- `vectors.py` — Per-prompt forward passes (no batching). `_capture_all_hidden_states` hooks every layer in one pass. `_encode_and_capture_all` handles tokenization, chat-template wrapping, attention-weighted pooling. `extract_contrastive`: 2N passes for N pairs, per-layer SVD extracts first principal component, scored by explained variance ratio. Returns a **profile**: `dict[int, (Tensor, score)]` mapping every layer to direction + signal strength. Profiles saved as `.safetensors` + `.json` sidecar. `compute_layer_means`: 30 neutral prompts → per-layer mean hidden state for centering.
- `probes_bootstrap.py` — Loads/extracts probe profiles per `steer/probes/defaults.json`. 28 probes across 5 categories (emotion, personality, safety, cultural, gender). `bootstrap_layer_means`: loads or computes per-layer mean activations, cached as `_LAYERMEANS.safetensors` per model.

### Steering + Monitoring layer
- `hooks.py` — `SteeringHook` adds pre-composed vector to hidden states in-place. `SteeringManager` groups vectors by layer, orthogonalizes per layer (Gram-Schmidt), one hook per active layer.
- `monitor.py` — `TraitMonitor` runs a single post-generation forward pass over the generated text using attention-weighted pooling. Mean-centers hidden states (subtracting per-layer means computed from neutral prompts) before computing score-weighted cosine similarities against probe vectors. One value per probe per generation. No hooks on the model during generation.

### Session API layer
- `session.py` — `SteerSession` is the programmatic API and the TUI's backend. Owns model, vector registry (`_profiles`), monitor, generation config, conversation history. Key design: **vectors are registered without alphas** via `steer(name, profile)`, alphas are supplied per-generation via `generate(input, alphas={"name": 1.5})`. No persistent steering hooks between generations. Orthogonalize is a per-call parameter. Full extraction pipeline: cache -> curated dataset -> statement cache -> model-generated pairs -> contrastive PCA -> save.
- `datasource.py` — `DataSource` normalizes contrastive pairs from curated names, JSON, CSV, HF datasets, or raw Python lists.
- `results.py` — `GenerationResult`, `TokenEvent`, `ProbeReadings` dataclasses with `to_dict()`. `ResultCollector` accumulates results for batch export (dicts, JSONL, CSV, DataFrame).

### Generation loop
- `generation.py` — Token-by-token with KV cache. Top-p via `torch.topk` (k capped at 1024). `torch.inference_mode()` wraps entire loop.

### TUI layer (Textual)
- `tui/app.py` — Thin frontend over `SteerSession`. Owns local alpha/enabled/orthogonalize state, passes through to session at generation time. Polls at ~15 FPS. Commands: `/steer`, `/probe`, `/clear`, `/rewind`, `/sys`, `/temp`, `/top-p`, `/max`, `/help`.
- `tui/vector_panel.py` — Model info, vectors with alpha bars, generation config.
- `tui/chat_panel.py` — Message log, status bar, input field.
- `tui/trait_panel.py` — Per-probe bars + sparklines, stats, sort modes.

## Performance rules

These matter for the throughput regression test (steered >= 85% of vanilla tok/s):

- **Hot-path hooks**: No Python allocation, no `.item()`, no CPU sync. In-place mutation only.
- **`torch.inference_mode()`** wrapping the entire generation loop.
- **In-place ops**: `logits.div_()`, `logits.clamp_()`, `probs.div_()`.
- **`torch.topk`** for top-p, not full-vocab sort. `k` capped at `min(1024, vocab)`.
- **Norm computations use `.float()`** — fp16 sum-of-squares overflows for hidden_dim >= 2048.
- **Vectors scaled to 10% of mean hidden-state norm** at each extraction layer.
- **Monitor is post-generation**: single forward pass after generation, no hooks during generation. Mean-centered cosine similarities remove baseline bias.
- **Steering hooks are transient**: composed before generation, removed after. No persistent hooks between generations.

## Supported architectures

`model.py:_LAYER_ACCESSORS`. Add new architecture = add one entry. Currently: llama, llama4, llama4_text, mistral, mixtral, gemma, gemma2, gemma3, gemma3_text, gemma4, gemma4_text, recurrent_gemma, phi, phi3, phimoe, qwen, qwen2, qwen2_moe, qwen3, qwen3_moe, cohere, cohere2, deepseek_v2, deepseek_v3, starcoder2, olmo, olmo2, olmoe, glm, glm4, granite, granitemoe, nemotron, stablelm, gpt2, gpt_neo, gptj, gpt_bigcode, bloom, falcon, gpt_neox, mpt, dbrx, opt.

## Testing

Smoke tests require CUDA and download `google/gemma-2-2b-it` on first run. Non-CUDA tests (`test_results.py`, `test_datasource.py`) run anywhere. Session tests (`test_session.py`) require CUDA. Coverage: profile extraction, steering effect, hook cleanup, save/load roundtrip, monitor history, throughput regression, `build_chat_input`, `bootstrap_probes`, DataSource parsing, ResultCollector export, SteerSession lifecycle/generation/streaming.
