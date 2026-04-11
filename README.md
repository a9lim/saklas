# liahona

Activation steering and trait monitoring for HuggingFace transformer models. Extract steering vectors, apply them during generation with per-call alpha control, and monitor how activations shift across behavioral probes.

Three interfaces: a **Python API** for scripted experiments and batch sweeps, an **OpenAI-compatible API server** for drop-in use with any OpenAI SDK client, and a **terminal UI** for interactive exploration.

## Python API

```python
from liahona import LiahonaSession, DataSource, ResultCollector

with LiahonaSession("google/gemma-2-2b-it", device="cuda") as session:
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

**Vectors are registered without alphas.** `session.steer(name, profile)` stores the vector. `session.generate(input, alphas={"name": 0.15})` applies it for that generation only. Alpha directly represents the fraction of mean hidden-state norm (e.g. 0.15 = 15% perturbation at high-signal layers). No persistent hooks live on the model between calls.

**Orthogonalization is per-call.** `session.generate(input, alphas={...}, orthogonalize=True)` applies Gram-Schmidt to the active vectors for that generation only.

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

### LiahonaSession reference

```python
session = LiahonaSession(
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
result = session.generate("prompt", alphas={"name": 0.15}, orthogonalize=False)
for token in session.generate_stream("prompt", alphas={"name": 0.15}):
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
from liahona import DataSource

ds = DataSource.curated("happy")                                   # bundled
ds = DataSource.json("pairs.json")                                 # liahona schema
ds = DataSource.csv("pairs.csv", positive_col="pos", negative_col="neg")
ds = DataSource.huggingface("user/dataset", split="train[:100]")   # requires datasets
ds = DataSource.from_pairs([("positive text", "negative text")])
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
liahona serve google/gemma-2-9b-it --steer cheerful:0.2 --port 8000
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
    extra_body={"steer": {"alphas": {"cheerful": 0.4}, "orthogonalize": True}},
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
- `GET /v1/liahona/vectors` — list registered vectors
- `POST /v1/liahona/vectors/extract` — extract a new vector (streams progress via SSE)
- `POST /v1/liahona/vectors/load` — load from `.safetensors` file
- `DELETE /v1/liahona/vectors/{name}` — remove a vector

**Probe management:**
- `GET /v1/liahona/probes` — list active probes + last readings
- `GET /v1/liahona/probes/defaults` — available default probes by category
- `POST /v1/liahona/probes/{name}` — activate a probe
- `DELETE /v1/liahona/probes/{name}` — deactivate a probe

**Session management:**
- `GET /v1/liahona/session` — current config, model info, default alphas
- `PATCH /v1/liahona/session` — update temperature, top_p, max_tokens, system_prompt
- `POST /v1/liahona/session/clear` — clear conversation history
- `POST /v1/liahona/session/rewind` — undo last exchange

Full API docs available at `http://localhost:8000/docs` when the server is running.

Probe readings are returned as an extra `probe_readings` field in generation responses — standard clients ignore it, aware clients get inline monitoring data.

## Terminal UI

```bash
liahona google/gemma-2-9b-it
liahona mistralai/Mistral-7B-Instruct-v0.3 -q 4bit
liahona meta-llama/Llama-3.1-8B-Instruct --probes emotion personality
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

Probes are extracted on first run and cached per model under `liahona/probes/cache/`.

## Install

```bash
pip install -e .                   # base install
pip install -e ".[dev]"            # + pytest
pip install -e ".[serve]"          # + fastapi + uvicorn (for API server)
pip install torch psutil setuptools wheel && pip install -e ".[cuda]" --no-build-isolation  # Linux/WSL only
pip install -e ".[research]"       # + datasets + pandas (for API)
pip install -e ".[bnb]"            # + bitsandbytes only
```

Requires Python 3.11+, PyTorch 2.2+. CUDA recommended; MPS and CPU work without quantization.

### Windows (WSL2 recommended)

`flash-attn` does not build on native Windows, and `bitsandbytes` has limited Windows support. For full CUDA functionality, use WSL2 — it gives you a Linux environment with direct GPU access.

**One-time setup** (from PowerShell as Administrator):

```powershell
wsl --install
```

Restart when prompted. This installs Ubuntu by default. On first launch, WSL will ask you to create a username and password.

Verify GPU access with `nvidia-smi`. If it shows your GPU, you're set.

**Clone and install liahona**:

```bash
sudo apt install -y python3-venv git
git clone <repo-url> ~/liahona && cd ~/liahona

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch psutil setuptools wheel

# flash-attn needs CUDA_HOME to find nvcc. PyTorch installs CUDA under
# /usr/local — find your version with: ls /usr/local/ | grep cuda
export CUDA_HOME=/usr/local/cuda-13
pip install -e ".[cuda]" --no-build-isolation

# Log in to HuggingFace to avoid download throttling and access gated models
huggingface-cli login
```

**Returning to an existing install**:

```bash
wsl                                # enter WSL from any Windows terminal
cd ~/liahona
source .venv/bin/activate
liahona google/gemma-2-2b-it      # ready to go
```

**Native Windows (no WSL)** — if you don't need quantization or flash-attn, liahona works with full-precision models out of the box:

```bash
pip install -e .
```

For quantization only (no flash-attn), `bitsandbytes` has experimental Windows support:

```bash
pip install -e ".[bnb]"
```

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
