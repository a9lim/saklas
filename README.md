# steer

Local activation steering and trait monitoring for HuggingFace transformer models, with a real-time terminal UI.

Load a model, extract steering vectors, adjust them live, and watch how activations shift across dozens of behavioral probes — all from your terminal.

## What it does

- **Activation steering**: Extract and inject steering vectors (ActAdd or Contrastive Activation Addition) into any layer during generation. Adjust strength and layer placement in real time.
- **Trait monitoring**: Track cosine similarity between model activations and a library of probe vectors (emotion, personality, safety, cultural, gender) as tokens are generated. Visualized as live bars and sparklines.
- **A/B comparison**: Generate the same prompt with and without steering to see the effect side-by-side.
- **Orthogonalization**: Gram-Schmidt orthogonalize multiple steering vectors to reduce interference.

## Requirements

- Python 3.11+
- PyTorch 2.2+
- A HuggingFace causal LM (tested with Gemma, Llama, Mistral, Phi, Qwen)
- CUDA GPU recommended. MPS (Apple Silicon) and CPU work but without quantization or torch.compile.

## Install

```bash
pip install -e .

# With flash attention (CUDA only):
pip install -e ".[flash]"

# With test dependencies:
pip install -e ".[dev]"
```

## Usage

```bash
steer google/gemma-2-9b-it
steer mistralai/Mistral-7B-Instruct-v0.3 -q 4bit
steer meta-llama/Llama-3.1-8B-Instruct --device cuda --probes emotion personality safety
```

### CLI options

| Flag | Description |
|------|-------------|
| `model` | HuggingFace model ID or local path |
| `-q`, `--quantize` | `4bit` or `8bit` (CUDA only, via bitsandbytes) |
| `--device` | `auto` (default), `cuda`, `mps`, or `cpu` |
| `--no-compile` | Skip torch.compile |
| `--probes` | Probe categories to load: `emotion` `personality` `safety` `cultural` `gender` (default: emotion personality) |
| `-s`, `--system-prompt` | System prompt for chat |
| `--max-tokens` | Max tokens per generation (default: 512) |
| `--cache-dir` | Cache directory for extracted vectors |

## TUI

### Layout

```
┌─────────────────────────────────┬──────────────────────┐
│                                 │  STEERING VECTORS    │
│          Chat                   │  > happy   a=+1.0 L13│
│                                 │    concise a=+0.5 L18│
│                                 ├──────────────────────┤
│                                 │  CONTROLS            │
│                                 │  Alpha ████░░░ +1.0  │
│                                 │  Layer 13 / 26       │
│                                 ├──────────────────────┤
│                                 │  TRAIT MONITOR        │
│                                 │  ▸ Emotion            │
│                                 │    happy  ████████ +0.42│
│                                 │    sad    ██░░░░░░ -0.15│
│  Type a message...              │  ▾ Personality (10)  │
└─────────────────────────────────┴──────────────────────┘
```

### Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+N` | Add steering vector |
| `Ctrl+D` | Remove selected vector |
| `Ctrl+T` | Toggle selected vector on/off |
| `←` / `→` | Adjust alpha (strength) |
| `↑` / `↓` | Change injection layer |
| `O` | Toggle orthogonalization |
| `Ctrl+A` | A/B compare (steered vs unsteered) |
| `Ctrl+R` | Regenerate last response |
| `S` | Cycle trait sort mode (name/magnitude/change) |
| `Ctrl+P` | Add probe |
| `Ctrl+Q` | Quit |

### Chat commands

| Command | Description |
|---------|-------------|
| `/steer <concept> [layer] [alpha]` | Add a steering vector (e.g. `/steer happy 18 0.8`) |
| `/clear` | Clear chat history |
| `/system <prompt>` | Set system prompt |
| `/temp <value>` | Set temperature |
| `/probes` | List active probes |

## Supported architectures

Llama, Mistral, Gemma, Gemma 2, Phi, Phi-3, Qwen, Qwen2, Qwen2-MoE, GPT-NeoX.

## Steering methods

**ActAdd** (Turner et al., 2023): Extracts the difference between a concept prompt and an empty baseline at a given layer. Fast, single forward pass per concept.

**Contrastive Activation Addition** (Rimsky et al., 2023): Averages over multiple positive/negative prompt pairs for stronger, more robust vectors. Requires a dataset JSON file with `{"pairs": [{"positive": ..., "negative": ...}]}`.

## How the monitor works

A read-only forward hook at the penultimate layer computes cosine similarity between the last token's hidden state and each probe vector via a single matrix multiply. Results accumulate in a GPU buffer and batch-transfer to CPU on the TUI poll cycle (~15 FPS), so the monitor adds negligible overhead. The throughput target is ≥85% of vanilla generation speed with 3 steering vectors and 15 probes active.

## Tests

```bash
pytest tests/test_smoke.py -v
```

Requires CUDA. Downloads `google/gemma-2-2b-it` (~5 GB) on first run. Covers vector extraction, steering effect, hook cleanup, save/load roundtrip, monitor correctness, and throughput regression.
