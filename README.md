# steer

Activation steering and trait monitoring for HuggingFace transformer models. Extract steering vectors, apply them during generation with per-call alpha control, and monitor how activations shift across behavioral probes.

Two interfaces: a **Python API** for scripted experiments and batch sweeps, and a **terminal UI** for interactive exploration.

## Python API

```python
from steer import SteerSession, DataSource, ResultCollector

with SteerSession("google/gemma-2-2b-it", device="cuda") as session:
    # Extract a steering vector
    happy_profile = session.extract("happy")       # uses curated dataset
    session.steer("happy", happy_profile)           # register (no alpha yet)

    # Generate with steering
    result = session.generate(
        "What makes a good day?",
        alphas={"happy": 2.0},
    )
    print(result.text)
    print(result.readings)  # probe monitor data

    # A/B comparison — just omit alphas
    unsteered = session.generate("What makes a good day?")

    # Sweep alphas
    collector = ResultCollector()
    for alpha in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        session.clear_history()
        result = session.generate(
            "Describe a sunset.",
            alphas={"happy": alpha},
        )
        collector.add(result, alpha=alpha)
    collector.to_csv("sweep_results.csv")
```

### Key concepts

**Vectors are registered without alphas.** `session.steer(name, profile)` stores the vector. `session.generate(input, alphas={"name": 1.5})` applies it for that generation only. No persistent hooks live on the model between calls.

**Orthogonalization is per-call.** `session.generate(input, alphas={...}, orthogonalize=True)` applies Gram-Schmidt to the active vectors for that generation only.

**Multiple vectors compose naturally:**

```python
session.steer("happy", happy_profile)
session.steer("formal", formal_profile)

# Apply both
result = session.generate("Hello.", alphas={"happy": 2.0, "formal": 1.0})

# Apply only one
result = session.generate("Hello.", alphas={"happy": 2.0})

# Apply none
result = session.generate("Hello.")
```

### SteerSession reference

```python
session = SteerSession(
    model_id,                        # HuggingFace model ID or local path
    device="auto",                   # "auto", "cuda", "mps", "cpu"
    quantize=None,                   # "4bit", "8bit", or None
    probes=None,                     # list of categories, or None for all
    system_prompt=None,              # default system prompt
    max_tokens=1024,                 # max tokens per generation
    cache_dir=None,                  # vector cache directory
)

# Vector extraction
profile = session.extract("happy")                  # curated dataset
profile = session.extract("empathy", baseline="apathy")  # contrastive
profile = session.extract([("pos", "neg"), ...])     # raw pairs
profile = session.extract(DataSource.csv("pairs.csv"))
profile = session.load_profile("saved.safetensors")
session.save_profile(profile, "output.safetensors")

# Model-generated contrastive pairs
pairs = session.generate_pairs("curiosity")  # list[(str, str)]

# Vector registry
session.steer("name", profile)     # register
session.unsteer("name")            # remove
session.vectors                    # dict of registered profiles

# Generation
result = session.generate("prompt", alphas={"name": 1.5}, orthogonalize=False)
for token in session.generate_stream("prompt", alphas={"name": 1.5}):
    print(token.text, end="", flush=True)

# Monitoring
session.monitor("honest")                   # curated probe
session.monitor("custom", custom_profile)    # from profile
session.unmonitor("honest")

# State
session.config.temperature = 0.8   # also: top_p, max_new_tokens, system_prompt
session.history                    # conversation messages
session.last_result                # most recent GenerationResult
session.model_info                 # model metadata
session.stop()                     # interrupt generation
session.rewind()                   # drop last exchange
session.clear_history()            # clear conversation
```

### Structured output

```python
result = session.generate("prompt", alphas={"happy": 2.0})
result.text              # decoded output
result.tokens            # token IDs
result.token_count       # number of tokens
result.tok_per_sec       # generation speed
result.elapsed           # seconds
result.vectors           # {"happy": 2.0} — snapshot of alphas used
result.readings          # {"probe_name": ProbeReadings} if probes active
result.to_dict()         # plain Python types, JSON-serializable
```

### DataSource formats

```python
from steer import DataSource

ds = DataSource.curated("happy")                                   # bundled
ds = DataSource.json("pairs.json")                                 # steer schema
ds = DataSource.csv("pairs.csv", positive_col="pos", negative_col="neg")
ds = DataSource.huggingface("user/dataset", split="train[:100]")   # requires datasets
ds = DataSource.from_pairs([("positive text", "negative text")])
```

### ResultCollector

```python
collector = ResultCollector()
collector.add(result, concept="happy", alpha=2.0, run_id=1)

collector.to_dicts()               # list of flat dicts
collector.to_jsonl("results.jsonl")
collector.to_csv("results.csv")
collector.to_dataframe()           # requires pandas
```

Probe readings flatten to columns: `probe_honest_mean`, `probe_honest_std`, etc. Vector alphas flatten to `vector_happy_alpha`.

## Terminal UI

```bash
steer google/gemma-2-9b-it
steer mistralai/Mistral-7B-Instruct-v0.3 -q 4bit
steer meta-llama/Llama-3.1-8B-Instruct --probes emotion personality
```

### CLI options

| Flag | Description |
|------|-------------|
| `model` | HuggingFace model ID or local path |
| `-q`, `--quantize` | `4bit` or `8bit` (CUDA only) |
| `-d`, `--device` | `auto` (default), `cuda`, `mps`, or `cpu` |
| `-p`, `--probes` | Categories: `all`, `none`, `emotion`, `personality`, `safety`, `cultural`, `gender` |
| `-s`, `--system-prompt` | System prompt |
| `-m`, `--max-tokens` | Max tokens per generation (default: 1024) |
| `-c`, `--cache-dir` | Vector cache directory |

### Layout

```
+--------------------+----------------------------------+------------------+
|  VECTORS           |                                  |  TRAIT MONITOR   |
|  > happy  +2.5     |          Chat                    |  Emotion         |
|    formal +1.0     |                                  |    happy #### .42|
|                    |                                  |    sad   ##- -.15|
|  CONFIG            |                                  |  Personality     |
|  temp ####- 0.7    |                                  |    honest ### .31|
|  top-p #### 0.9    |                                  |                  |
|                    |  Type a message...               |                  |
+--------------------+----------------------------------+------------------+
```

### Keybindings

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Cycle panel focus |
| `Left` / `Right` | Adjust alpha |
| `Up` / `Down` | Navigate vectors / probes |
| `Enter` | Toggle vector on/off |
| `Backspace` / `Delete` | Remove selected vector or probe |
| `Ctrl+O` | Toggle orthogonalization |
| `Ctrl+A` | A/B compare (steered vs unsteered) |
| `Ctrl+R` | Regenerate last response |
| `Ctrl+S` | Cycle trait sort mode |
| `[` / `]` | Adjust temperature |
| `{` / `}` | Adjust top-p |
| `Escape` | Stop generation |
| `Ctrl+Q` | Quit |

### Chat commands

| Command | Description |
|---------|-------------|
| `/steer "concept" [alpha]` | Extract and register steering vector |
| `/steer "concept" - "baseline" [alpha]` | Contrastive steering |
| `/probe "concept"` | Add monitoring probe |
| `/probe "concept" - "baseline"` | Contrastive probe |
| `/clear` | Clear history and reset probes |
| `/rewind` | Undo last exchange |
| `/sys <prompt>` | Set system prompt |
| `/temp <value>` | Set temperature |
| `/top-p <value>` | Set top-p |
| `/max <value>` | Set max tokens |

Concepts matching built-in probe names use curated datasets automatically. Otherwise, pairs are generated by the loaded model.

## Probe library

28 probes across 5 categories, each backed by ~60 curated contrastive pairs:

| Category | Probes |
|----------|--------|
| **Emotion** | happy, angry, fearful, surprised, disgusted, excited, sad, calm |
| **Personality** | honest, creative, formal, verbose, authoritative, confident, uncertain |
| **Safety** | sycophantic, refusal, deceptive, hallucinating |
| **Cultural** | western, hierarchical, direct, contextual, religious, traditional |
| **Gender** | masculine, agentic, paternal |

Probes are extracted on first run and cached per model under `steer/probes/cache/`.

## Install

```bash
pip install -e .                   # base install
pip install -e ".[dev]"            # + pytest
pip install -e ".[cuda]"           # + bitsandbytes + flash-attn
pip install -e ".[research]"       # + datasets + pandas (for API)
pip install -e ".[bnb]"            # + bitsandbytes only
```

Requires Python 3.11+, PyTorch 2.2+. CUDA recommended; MPS and CPU work without quantization.

## Supported architectures

39 architectures via `model.py:_LAYER_ACCESSORS`. Adding a new one = one lambda entry.

Llama (1-4), Mistral, Mixtral, Gemma (1-4), Phi (1-3), PhiMoE, Qwen (1-3), Qwen2-MoE, Qwen3-MoE, Cohere (1-2), DeepSeek (V2-V3), StarCoder2, OLMo (1-2), OLMoE, GLM (3-4), Granite, GraniteMoE, Nemotron, StableLM, GPT-2, GPT-Neo, GPT-J, GPT-BigCode, GPT-NeoX, Bloom, Falcon, MPT, DBRX, OPT, RecurrentGemma.

## How it works

### Steering vectors

**Representation Engineering** (Zou et al., 2023): For each contrastive pair, captures attention-weighted hidden states at every layer. Computes pos-neg differences, extracts the first principal component per layer via batched SVD. Each layer is scored by explained variance ratio. The result is a multi-layer profile — no manual layer selection. Scores weight each layer's contribution during generation.

### Monitor

After each generation, a single forward pass over the generated text produces attention-weighted hidden states per layer (same pooling as extraction). Each layer's hidden state is mean-centered — subtracting the per-layer mean computed from 30 neutral prompts — to remove baseline projection bias that otherwise makes raw cosine similarities uninformative. Score-weighted cosine similarities against probe vectors produce one value per probe per generation.

Layer means are computed once per model and cached as `_LAYERMEANS.safetensors` alongside probe vectors. No hooks run on the model during generation — monitoring is purely post-generation.

## Tests

```bash
pytest tests/ -v                   # all tests
pytest tests/test_results.py tests/test_datasource.py -v  # no GPU needed
pytest tests/test_smoke.py -v      # CUDA required
```

CUDA tests download `google/gemma-2-2b-it` (~5 GB) on first run. Non-CUDA tests (`test_results.py`, `test_datasource.py`) run anywhere.
