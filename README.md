# steer

Local activation steering and trait monitoring for HuggingFace transformer models, with a real-time terminal UI.

Load a model, extract steering vectors, adjust them live, and watch how activations shift across 28 behavioral probes — all from your terminal.

## What it does

- **Activation steering**: Extract and inject steering vectors (ActAdd or Contrastive Activation Addition) into any layer during generation. Adjust strength in real time.
- **Trait monitoring**: Track cosine similarity between model activations and 28 probe vectors across 5 categories (emotion, personality, safety, cultural, gender) as tokens are generated. Visualized as live bars, sparklines, and running statistics.
- **Custom vectors and probes**: Extract your own steering vectors or monitoring probes from any concept via `/steer` and `/probe` commands — uses LLM-generated contrastive pairs or falls back to curated datasets.
- **A/B comparison**: Generate the same prompt with and without steering to see the effect side-by-side.
- **Orthogonalization**: Gram-Schmidt orthogonalize multiple steering vectors to reduce interference.

## Requirements

- Python 3.11+
- PyTorch 2.2+
- A HuggingFace causal LM
- CUDA GPU recommended. MPS (Apple Silicon) and CPU work but without quantization or flash attention.

## Install

```bash
pip install -e .

# With bitsandbytes quantization (CUDA only):
pip install -e ".[bnb]"

# With flash attention (CUDA only):
pip install -e ".[flash]"

# Both:
pip install -e ".[cuda]"

# With test dependencies:
pip install -e ".[dev]"
```

## Usage

```bash
steer google/gemma-2-9b-it
steer mistralai/Mistral-7B-Instruct-v0.3 -q 4bit
steer meta-llama/Llama-3.1-8B-Instruct --probes emotion personality safety
```

### CLI options

| Flag | Description |
|------|-------------|
| `model` | HuggingFace model ID or local path |
| `-q`, `--quantize` | `4bit` or `8bit` (CUDA only, via bitsandbytes) |
| `-d`, `--device` | `auto` (default), `cuda`, `mps`, or `cpu` |
| `-p`, `--probes` | Probe categories to load: `all`, `none`, `emotion`, `personality`, `safety`, `cultural`, `gender` (default: all) |
| `-s`, `--system-prompt` | System prompt for chat |
| `-m`, `--max-tokens` | Max tokens per generation (default: 1024) |
| `-c`, `--cache-dir` | Cache directory for extracted vectors |

## TUI

### Layout

```
+------------------+----------------------------------+------------------+
|  VECTORS         |                                  |  TRAIT MONITOR   |
|  > happy caa L21 |          Chat                    |  Emotion         |
|    formal caa L18|                                  |    happy #### .42|
|                  |                                  |    sad   ##- -.15|
|  CONFIG          |                                  |  Personality     |
|  temp ####- 0.7  |                                  |    honest ### .31|
|  top-p #### 0.9  |                                  |    verbose ##-.18|
|                  |                                  |                  |
|  KEYS            |  Type a message...               |                  |
+------------------+----------------------------------+------------------+
```

### Keybindings

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Cycle panel focus |
| `Left` / `Right` | Adjust alpha (strength) |
| `Up` / `Down` | Navigate vectors / probes (in focused panel) |
| `Ctrl+D` | Remove selected vector or probe |
| `Ctrl+T` | Toggle selected vector on/off |
| `Ctrl+O` | Toggle orthogonalization |
| `Ctrl+A` | A/B compare (steered vs unsteered) |
| `Ctrl+R` | Regenerate last response |
| `Ctrl+S` | Cycle trait sort mode (name / magnitude / change) |
| `Escape` | Stop generation |
| `Ctrl+Q` | Quit |

### Chat commands

| Command | Description |
|---------|-------------|
| `/steer "concept" [layer] [alpha]` | Steering vector via contrastive pairs (e.g. `/steer "happy" 18 2.5`) |
| `/steer "concept" - "baseline" [layer] [alpha]` | Steering with explicit baseline (e.g. `/steer "formal" - "casual"`) |
| `/probe "concept" [layer]` | Add a monitoring probe (e.g. `/probe "sarcastic"`) |
| `/probe "concept" - "baseline" [layer]` | Probe with explicit baseline |
| `/clear` | Clear chat history and reset probe stats |
| `/rewind` | Undo last user message and its response |
| `/sys <prompt>` | Set system prompt |
| `/temp <value>` | Set temperature |
| `/top-p <value>` | Set top-p |
| `/max <value>` | Set max tokens |
| `/help` | Show available commands |

For `/steer` and `/probe`, if the concept matches a built-in probe name (e.g. "happy", "refusal"), the curated contrastive pair dataset is used automatically. Otherwise, pairs are generated via the loaded model.

## Probe library

28 probes across 5 categories, each backed by 60 curated contrastive pairs:

| Category | Probes |
|----------|--------|
| **Emotion** | happy, angry, fearful, surprised, disgusted, excited, sad, calm |
| **Personality** | sycophantic, honest, creative, formal, verbose, authoritative, confident, uncertain |
| **Safety** | refusal, deceptive, hallucinating |
| **Cultural** | western, hierarchical, direct, contextual, religious, traditional |
| **Gender** | masculine, agentic, paternal |

Probes are extracted on first run and cached per model under `steer/probes/cache/`.

## Supported architectures

Llama (1-4), Mistral, Mixtral, Gemma (1-4), Phi (1-3), PhiMoE, Qwen (1-3), Qwen2-MoE, Qwen3-MoE, Cohere (1-2), DeepSeek (V2-V3), StarCoder2, OLMo (1-2), OLMoE, GLM (3-4), Granite, GraniteMoE, Nemotron, StableLM, GPT-2, GPT-Neo, GPT-J, GPT-BigCode, GPT-NeoX, Bloom, Falcon, MPT, DBRX, OPT, RecurrentGemma.

Adding a new architecture requires one entry in `model.py:_LAYER_ACCESSORS`.

## Steering methods

**ActAdd** (Turner et al., 2023): Difference between a concept prompt and a contrastive baseline at a target layer. Two forward passes.

**Contrastive Activation Addition** (Rimsky et al., 2023): Averages over multiple matched positive/negative prompt pairs for more robust vectors. The `/steer` and `/probe` commands generate these pairs automatically using the loaded model, or use curated datasets for built-in probe concepts.

## How the monitor works

A read-only forward hook at the penultimate layer computes cosine similarity between the last token's hidden state and each probe vector via a single matrix multiply. Results accumulate in a GPU buffer and batch-transfer to CPU on the TUI poll cycle (~15 FPS). The throughput target is >=85% of vanilla generation speed with steering and monitoring active.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```

Requires CUDA. Downloads `google/gemma-2-2b-it` (~5 GB) on first run. Covers vector extraction, steering effect, hook cleanup, save/load roundtrip, monitor correctness, and throughput regression.
