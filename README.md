# saklas

[![PyPI](https://img.shields.io/pypi/v/saklas)](https://pypi.org/project/saklas/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/saklas/)

Activation steering and trait monitoring for HuggingFace transformer models. Extract steering vectors, apply them during generation with per-call alpha control, and monitor how activations shift across behavioral probes.

Three interfaces: a **Python API** for scripted experiments and batch sweeps, an **OpenAI-compatible API server** for drop-in use with any OpenAI SDK client, and a **terminal UI** for interactive exploration.

## Python API

```python
from saklas import SaklasSession, DataSource, ResultCollector

with SaklasSession("google/gemma-2-2b-it", device="cuda") as session:
    # Extract a steering vector
    happy_profile = session.extract("happy")       # uses curated dataset
    session.steer("happy", happy_profile)           # register (no alpha yet)

    # Generate with steering
    result = session.generate(
        "What makes a good day?",
        alphas={"happy": 0.2},
    )
    print(result.text)
    print(result.readings)  # probe monitor data

    # A/B comparison — just omit alphas
    unsteered = session.generate("What makes a good day?")

    # Sweep alphas
    collector = ResultCollector()
    for alpha in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        session.clear_history()
        result = session.generate(
            "Describe a sunset.",
            alphas={"happy": alpha},
        )
        collector.add(result, alpha=alpha)
    collector.to_csv("sweep_results.csv")
```

### Key concepts

**Vectors are registered without alphas.** `session.steer(name, profile)` stores the vector. `session.generate(input, alphas={"name": 0.5})` applies it for that generation only. Alpha directly represents the fraction of mean hidden-state norm (e.g. 0.5 = 50% perturbation at high-signal layers). No persistent hooks live on the model between calls.

**Orthogonalization is per-call.** `session.generate(input, alphas={...}, orthogonalize=True)` applies Gram-Schmidt to the active vectors for that generation only.

**Thinking mode is per-call.** For models that support it (Qwen 3.5, QwQ, Gemma 4, etc.), `session.generate(input, thinking=True)` enables the model's built-in reasoning trace. Thinking delimiters are detected automatically from the chat template — no hardcoded tokens. Thinking tokens are separated from the response — `result.text` contains only the final answer, while streaming via `generate_stream` yields `TokenEvent` objects with `thinking=True` for the reasoning trace.

**Multiple vectors compose naturally:**

```python
session.steer("happy", happy_profile)
session.steer("formal", formal_profile)

# Apply both
result = session.generate("Hello.", alphas={"happy": 0.2, "formal": 0.1})

# Apply only one
result = session.generate("Hello.", alphas={"happy": 0.2})

# Apply none
result = session.generate("Hello.")
```

### SaklasSession reference

```python
session = SaklasSession(
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
result = session.generate("prompt", alphas={"name": 0.5}, orthogonalize=False)
result = session.generate("prompt", thinking=True)  # enable reasoning trace
for token in session.generate_stream("prompt", alphas={"name": 0.5}):
    if token.thinking:
        print(f"[think] {token.text}", end="", flush=True)
    else:
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
result = session.generate("prompt", alphas={"happy": 0.2})
result.text              # decoded output
result.tokens            # token IDs
result.token_count       # number of tokens
result.tok_per_sec       # generation speed
result.elapsed           # seconds
result.vectors           # {"happy": 0.2} — snapshot of alphas used
result.readings          # {"probe_name": ProbeReadings} if probes active
result.to_dict()         # plain Python types, JSON-serializable
```

### DataSource formats

```python
from saklas import DataSource

ds = DataSource.curated("happy")                                   # bundled
ds = DataSource.json("pairs.json")                                 # saklas schema
ds = DataSource.csv("pairs.csv", positive_col="pos", negative_col="neg")
ds = DataSource.huggingface("user/dataset", split="train[:100]")   # requires datasets
ds = DataSource(pairs=[("positive text", "negative text")])
```

### ResultCollector

```python
collector = ResultCollector()
collector.add(result, concept="happy", alpha=0.2, run_id=1)

collector.to_dicts()               # list of flat dicts
collector.to_jsonl("results.jsonl")
collector.to_csv("results.csv")
collector.to_dataframe()           # requires pandas
```

Probe readings flatten to columns: `probe_honest_mean`, `probe_honest_std`, etc. Vector alphas flatten to `vector_happy_alpha`.

## API Server

Serve a steered model as an OpenAI-compatible HTTP endpoint. Works with the OpenAI Python/JS SDK, LangChain, `curl`, or anything that speaks the OpenAI API.

```bash
pip install -e ".[serve]"
saklas serve google/gemma-2-9b-it --steer cheerful:0.2 --port 8000
```

### Usage with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Uses server-default steering (cheerful=0.2 from --steer flag)
resp = client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)

# Override steering per-request
resp = client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"steer": {"alphas": {"cheerful": 0.4}, "orthogonalize": True, "thinking": True}},
)

# Streaming
for chunk in client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Serve CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `model` | required | HuggingFace model ID or local path |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `-q`, `--quantize` | None | `4bit` or `8bit` |
| `-d`, `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `-p`, `--probes` | all | Probe categories to bootstrap |
| `-s`, `--system-prompt` | None | Default system prompt |
| `-m`, `--max-tokens` | `1024` | Max tokens per generation |
| `--steer` | None | Pre-load vector, repeatable. `name:alpha` or `name` |
| `--cors` | None | CORS origin, repeatable |

### Endpoints

**OpenAI-compatible:**
- `GET /v1/models` — list loaded model
- `GET /v1/models/{model_id}` — model details
- `POST /v1/chat/completions` — chat (streaming + non-streaming)
- `POST /v1/completions` — text completion (streaming + non-streaming)

**Vector management:**
- `GET /v1/saklas/vectors` — list registered vectors
- `POST /v1/saklas/vectors/extract` — extract a new vector (streams progress via SSE)
- `POST /v1/saklas/vectors/load` — load from `.safetensors` file
- `DELETE /v1/saklas/vectors/{name}` — remove a vector

**Probe management:**
- `GET /v1/saklas/probes` — list active probes + last readings
- `GET /v1/saklas/probes/defaults` — available default probes by category
- `POST /v1/saklas/probes/{name}` — activate a probe
- `DELETE /v1/saklas/probes/{name}` — deactivate a probe

**Session management:**
- `GET /v1/saklas/session` — current config, model info, default alphas
- `PATCH /v1/saklas/session` — update temperature, top_p, max_tokens, system_prompt
- `POST /v1/saklas/session/clear` — clear conversation history
- `POST /v1/saklas/session/rewind` — undo last exchange

Full API docs available at `http://localhost:8000/docs` when the server is running.

Probe readings are returned as an extra `probe_readings` field in generation responses — standard clients ignore it, aware clients get inline monitoring data.

## Terminal UI

```bash
saklas google/gemma-2-9b-it
saklas mistralai/Mistral-7B-Instruct-v0.3 -q 4bit
saklas meta-llama/Llama-3.1-8B-Instruct --probes emotion personality
```

### CLI options

| Flag | Description |
|------|-------------|
| `model` | HuggingFace model ID or local path |
| `-q`, `--quantize` | `4bit` or `8bit` (CUDA only) |
| `-d`, `--device` | `auto` (default), `cuda`, `mps`, or `cpu` |
| `-p`, `--probes` | Categories: `all`, `none`, `emotion`, `personality`, `safety`, `cultural`, `gender` |
| `-s`, `--system-prompt` | System prompt |
| `--max-tokens` | Max tokens per generation (default: 1024) |
| `-i`, `--install <target>` | Install a pack from HF coordinate (`<ns>/<name>`) or local folder path |
| `-r`, `--refresh <selector>` | Re-pull concept(s) from source (repeatable) |
| `-x`, `--clear-tensors <selector>` | Delete tensors for matched concepts; keeps `statements.json` (repeatable) |
| `-l`, `--list [<selector>]` | List or show info about installed + HF packs; exits after printing |
| `-m`, `--merge <name> <components>` | Merge vectors: `-m bard default/happy:0.3,user/archaic:0.4` |
| `-C`, `--config <path>` | Load setup YAML (repeatable; later files override earlier) |
| `--strict` | With `-C`: fail hard on missing vectors |

**Selectors** (used by `-r`, `-x`, `-l`): bare name (resolves across namespaces; errors on ambiguity), `<ns>/<name>`, `tag:<tag>`, `namespace:<ns>`, `model:<id>`, `default`, `all`.

### Cache layout

saklas stores all caches and user-editable data under `~/.saklas/`:

```
~/.saklas/
  neutral_statements.json
  vectors/
    default/              # bundled probes (pip-installed, refreshable via -r default)
    local/                # user-authored concepts and merged packs
    <ns>/                 # packs pulled from Hugging Face (ns = HF repo owner)
  models/
    <safe_model_id>/
      layer_means.safetensors
```

Set `SAKLAS_HOME=/custom/path` to override the root directory.

### Layout

```
+--------------------+----------------------------------+------------------+
|  VECTORS           |                                  |  TRAIT MONITOR   |
|  > happy  +0.10    |          Chat                    |  Emotion         |
|    formal +0.06    |                                  |    happy #### .42|
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
| `Ctrl+T` | Toggle thinking mode (models that support it) |
| `Ctrl+A` | A/B compare (steered vs unsteered) |
| `Ctrl+R` | Regenerate last response (interrupts if generating) |
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

All commands that touch the model (`/steer`, `/probe`) or modify history (`/clear`, `/rewind`) interrupt any in-progress generation and execute once it stops. Sending a new message mid-generation also stops the current response and submits immediately after.

Concepts matching built-in probe names use curated datasets from `~/.saklas/vectors/default/<concept>/statements.json`. Otherwise, pairs are generated by the loaded model and cached under `~/.saklas/vectors/local/<concept>/statements.json` — subsequent extractions of the same concept (even with a different model) reuse the cached statements.

## Probe library

28 probes across 5 categories, each backed by ~60 curated contrastive pairs:

| Category | Probes |
|----------|--------|
| **Emotion** | happy, angry, fearful, surprised, disgusted, excited, sad, calm |
| **Personality** | honest, creative, formal, verbose, authoritative, confident, uncertain |
| **Safety** | sycophantic, refusal, deceptive, hallucinating |
| **Cultural** | western, hierarchical, direct, contextual, religious, traditional |
| **Gender** | masculine, agentic, paternal |

Probes are extracted on first run and cached per model under `~/.saklas/vectors/default/<concept>/<safe_model_id>.safetensors`.

## Install

```bash
pip install saklas             # base
pip install saklas[serve]      # + fastapi + uvicorn (for API server)
pip install saklas[research]   # + datasets + pandas (for API)
```

Requires Python 3.11+, PyTorch 2.2+. Works on Linux, macOS, and Windows.

### From source

```bash
pip install -e .                   # base install
pip install -e ".[dev]"            # + pytest
pip install -e ".[serve]"          # + fastapi + uvicorn
pip install -e ".[research]"       # + datasets + pandas
```

### Quantization and flash-attn (experimental)

The `cuda` and `bnb` extras install `bitsandbytes` and/or `flash-attn` for 4-bit/8-bit quantization and fused attention. These depend on platform-specific CUDA toolchains and may not build cleanly on all systems. Support is only guaranteed for the vanilla (unquantized) install.

```bash
pip install saklas[bnb]       # bitsandbytes only
pip install saklas[cuda]      # bitsandbytes + flash-attn (Linux only, needs CUDA_HOME)
```

From source, `flash-attn` requires build isolation disabled:

```bash
pip install torch psutil setuptools wheel && pip install -e ".[cuda]" --no-build-isolation
```

## Supported architectures

53 architectures via `model.py:_LAYER_ACCESSORS`. Adding a new one = one function entry.

Llama (1-4), Mistral (1, 4), Ministral (1, 3), Mixtral, Gemma (1-4), Phi (1-3), PhiMoE, Qwen (1-3.5), Qwen2-MoE, Qwen3-MoE, Qwen3.5-MoE, Cohere (1-2), DeepSeek (V2-V3), StarCoder2, OLMo (1-3), OLMoE, GLM (3-4), Granite, GraniteMoE, Nemotron, StableLM, GPT-2, GPT-Neo, GPT-J, GPT-BigCode, GPT-NeoX, GPT-OSS, Bloom, Falcon, Falcon-H1, MPT, DBRX, OPT, RecurrentGemma.

## How it works

### Steering vectors

**Representation Engineering** (Zou et al., 2023): For each contrastive pair, captures attention-weighted hidden states at every layer. Computes pos-neg differences, extracts the first principal component per layer via batched SVD. Each layer is scored by explained variance ratio. The result is a multi-layer profile — no manual layer selection. Scores weight each layer's contribution during generation.

### Custom steering vectors

When you steer on a concept that isn't in the curated probe library, saklas generates its own contrastive pairs using the loaded model, then extracts a vector from them. The pipeline:

1. **Statement generation** — the model writes contrastive statement pairs in batches, each batch seeded by a different specificity lens (unique facts/lore, physical traits, social dynamics, inner life, concrete routines). The prompt forces concept-specific detail — names, terminology, sensory descriptions that only apply to the target concept — and explicitly rejects generic statements that could work for anything similar.
2. **Caching** — generated pairs are saved under `~/.saklas/vectors/local/<concept>/statements.json`. Pairs are model-independent, so a different model reuses the same cached statements.
3. **Extraction** — pairs feed into the standard contrastive PCA pipeline (per-layer SVD, explained variance scoring).

This means `/steer "anything"` works — personality traits, religions, animals, emotions, fictional characters, "man who ate too much spaghetti." The vector captures what's distinctive about the concept, not generic associations.

To clear cached tensors for a concept (e.g. to re-extract with a different statements set), use `saklas -x <concept>`. To remove the statements themselves, delete the folder under `~/.saklas/vectors/local/` manually.

### Monitor

After generation, a separate forward pass over the generated text produces attention-weighted hidden states at every layer — the same pooling used during probe extraction. This ensures probe scores are computed against the same kind of representation the probes were trained on.

Each layer's hidden state is mean-centered — subtracting the per-layer mean computed from 45 neutral prompts — to remove baseline projection bias that otherwise makes raw cosine similarities uninformative. Score-weighted cosine similarities against probe vectors produce one value per probe per generation. Probe history accumulates across generations, enabling sparklines and running statistics.

Layer means are computed once per model and cached at `~/.saklas/models/<safe_model_id>/layer_means.safetensors`. They auto-invalidate when `~/.saklas/neutral_statements.json` is edited.

## Tests

```bash
pytest tests/ -v                   # all tests
pytest tests/test_results.py tests/test_datasource.py tests/test_server.py -v  # no GPU needed
pytest tests/test_smoke.py -v      # CUDA required
```

CUDA tests download `google/gemma-2-2b-it` (~5 GB) on first run. Non-CUDA tests (`test_results.py`, `test_datasource.py`, `test_server.py`) run anywhere.
