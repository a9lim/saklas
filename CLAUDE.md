# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`steer` is a local activation steering + trait monitoring TUI for HuggingFace transformer models. It loads a causal LM, extracts steering vectors (ActAdd or CAA), injects them via forward hooks during generation, and monitors activations against probe vectors in real time — all through a Textual terminal UI.

## Commands

```bash
pip install -e ".[dev]"          # Install in editable mode with test deps
pip install -e ".[dev,flash]"    # Also install flash-attn (CUDA only)
steer <model_id>                 # Launch TUI (e.g. steer google/gemma-2-9b-it)
python -m steer <model_id>       # Alternative entry point
pytest tests/test_smoke.py -v    # Smoke tests (requires CUDA + downloads gemma-2-2b-it ~5GB)
pytest tests/test_smoke.py -v -k "test_actadd_returns_unit_vector"  # Single test
```

## Architecture

The system has three layers: **model/vector infrastructure**, **generation loop**, and **TUI**.

### Model + Vector layer
- `model.py` — Loads HF causal LMs with quantization/device/compile options. `_LAYER_ACCESSORS` dict maps `model_type` strings to layer-list accessors; add new architectures here. `_text_config()` helper resolves `hidden_size` from multimodal configs that nest it under `text_config`.
- `vectors.py` — Two extraction methods: `extract_actadd` (single concept + baseline batched into one forward pass, mean-pool + L2-normalize the difference) and `extract_caa` (contrastive pairs batched into a single forward pass, same pipeline). Batched variant `extract_actadd_batched` for probe bootstrap. All extraction uses `torch.inference_mode()`. Vectors saved as `.safetensors` + `.json` metadata sidecar.
- `probes_bootstrap.py` — On startup, loads or extracts probe vectors per `steer/probes/defaults.json`. Probes are cached under `steer/probes/cache/{model_name}/`. Categories: emotion, personality, safety, cultural, gender. CAA probes expect dataset files in `steer/datasets/`.

### Steering + Monitoring layer
- `hooks.py` — `SteeringHook` registers a `register_forward_hook` on a layer, adding a pre-composed vector to hidden states in-place. Handles both tuple output (`output[0].add_()`) and bare tensor output (e.g. Gemma 4 decoder layers return a raw tensor). `recompose` uses stack+matmul to compose multiple vectors. `SteeringManager` groups vectors by layer, handles orthogonalization (Gram-Schmidt), and manages hook lifecycle.
- `monitor.py` — `TraitMonitor` attaches a read-only forward hook at a single layer. Pre-stacks probe vectors into a unit-normalized matrix; the hook normalizes the hidden state once, then a single matmul yields cosine similarities (no per-probe division). Guards against zero/NaN hidden state norms by writing zeros instead of NaN. GPU buffer batches results; `flush_to_cpu()` transfers to CPU history on TUI poll — call it once before reading sparklines, not per-probe.

### Generation loop
- `generation.py` — Token-by-token generation with KV cache, top-p sampling via `torch.topk` (avoids full-vocab sort), stop control via `threading.Event`. A single `torch.inference_mode()` context wraps the entire generation loop (not per-token — this matters for performance). Runs in a worker thread; tokens flow to TUI via `queue.SimpleQueue`. Steering hooks are transparent to the generation loop (they modify hidden states via forward hooks). Logits are clamped to [-100, 100] before softmax to prevent inf/NaN from extreme steering. `build_chat_input` falls back to plain tokenization for base models without a chat template.

### TUI layer (Textual)
- `tui/app.py` — `SteerApp` orchestrates everything. Caches `self._device`/`self._dtype` on init (never call `next(model.parameters())` outside `__init__`). Polls token queue + monitor at ~15 FPS. In-chat commands: `/steer`, `/clear`, `/system`, `/temp`, `/probes`. Keybindings for alpha/layer adjustment, orthogonalization toggle, A/B comparison.
- `tui/chat_panel.py`, `vector_panel.py`, `trait_panel.py` — Display widgets. Trait panel has collapsible categories, sort modes (name/magnitude/change), and sparkline display.
- `tui/styles.tcss` — Textual CSS.

### Key data flow
1. User types message → `ChatPanel.UserSubmitted` → `SteerApp._start_generation()`
2. Worker thread runs `generate_steered()` → each forward pass triggers steering hooks (add vectors) and monitor hook (record cosine similarities)
3. TUI poll timer drains token queue into chat panel, flushes monitor buffer, updates trait bars/sparklines

## Performance conventions

These matter for the throughput regression test (steered ≥85% of vanilla tok/s):

- **Hot-path hooks (`hook_fn`, `_hook`)**: No Python allocation, no `.item()`, no CPU sync. Return `output` directly after in-place mutation — never build new tuples. Both hooks handle bare tensor output (Gemma 4) and tuple output (most models). The monitor hook branches on `norm > 1e-8` to guard against zero/NaN states — this is a necessary GPU→CPU sync but only on degenerate inputs.
- **`torch.inference_mode()`** over `torch.no_grad()` everywhere (generation, extraction). In the generation loop, the context wraps the entire loop — never re-enter per token.
- **In-place ops in generation loop**: Use `logits.div_(temperature)` not `logits / temperature` — avoids a tensor allocation per token. The logits view is not reused after sampling.
- **`torch.topk`** for top-p sampling, not full-vocab sort.
- **Vector extraction math**: Difference vectors are computed as `pos - neg` directly. Do not re-introduce mean-centering — it's algebraically redundant (`(pos - center) - (neg - center) == pos - neg`). CAA uses a single batched forward pass for all positive+negative pairs.
- **Device/dtype**: Cached as `self._device`/`self._dtype` in `SteerApp.__init__`. Never call `next(model.parameters())` in action handlers or poll loops.
- **Monitor flush**: Call `flush_to_cpu()` first in the poll tick, then read `get_current()`/`get_previous()` from CPU history. Never call `get_current()`/`get_previous()` before flushing — they hit the GPU buffer and cause extra syncs.
- **Vector composition**: `recompose` uses `torch.stack` + broadcasted multiply + `.sum()`, not Python `sum()` over tensors.

## Supported architectures

Defined in `model.py:_LAYER_ACCESSORS`. Most models use `model.model.layers`; GPT-2 family uses `model.transformer.h`; multimodal wrappers (gemma3, gemma4) use `model.model.language_model.layers`; a few have unique paths (GPT-NeoX, MPT/DBRX, OPT). Adding a new architecture = adding one entry to that dict. Currently supported: llama, llama4, llama4_text, mistral, mixtral, gemma, gemma2, gemma3, gemma3_text, gemma4, gemma4_text, recurrent_gemma, phi, phi3, phimoe, qwen, qwen2, qwen2_moe, qwen3, qwen3_moe, cohere, cohere2, deepseek_v2, deepseek_v3, starcoder2, olmo, olmo2, olmoe, glm, glm4, granite, granitemoe, nemotron, stablelm, gpt2, gpt_neo, gptj, gpt_bigcode, bloom, falcon, gpt_neox, mpt, dbrx, opt.

## Testing

Smoke tests require CUDA and download `google/gemma-2-2b-it` on first run. Tests cover vector extraction, steering effect, hook cleanup, save/load roundtrip, monitor history, and throughput regression (steered must be ≥85% of vanilla tok/s).
