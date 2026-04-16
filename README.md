# saklas

[![CI](https://github.com/a9lim/saklas/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/saklas/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/saklas)](https://pypi.org/project/saklas/)
[![Downloads](https://img.shields.io/pypi/dm/saklas)](https://pypi.org/project/saklas/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/saklas/)

**Nudge a language model's mood without retraining it.** Saklas extracts *steering vectors* from pairs of contrastive examples (angry vs. calm, formal vs. casual, honest vs. deceptive — anything you can describe in a sentence) and adds them back into the model's hidden states at generation time. You dial the strength with a single number per concept. The model's weights never change; nothing persists between calls.

You also get a *trait monitor*: 21 pre-built probes that quietly score every response on affect, epistemic stance, register, alignment, and social/cultural dimensions, drawing live sparklines as the model talks.

And — new in 1.4 — you can **clone a persona from a text sample**. Point `saklas pack clone` at a file of utterances (one per line) and it extracts a steering vector for that voice, no contrastive pair authoring required. `saklas pack clone transcripts.txt -N hunter` and `/steer hunter 0.5` is the whole workflow.

Three ways to use it:

- **`saklas <model>`** — a terminal UI with live alpha knobs, probe readings, and A/B comparison
- **`saklas serve <model>`** — an HTTP server that speaks **both** the OpenAI `/v1/*` and Ollama `/api/*` wire formats on the same port (drop-in for any client that talks either)
- **`SaklasSession`** — a Python API for scripted experiments, batch sweeps, and embedding steering into your own pipelines

Tested on **Qwen, Gemma, Ministral, gpt-oss, Llama, and GLM**. Many other architectures are wired up in `saklas/core/model.py:_LAYER_ACCESSORS` (Mistral/Mixtral, Phi, DeepSeek, Cohere, OLMo, and more) but are untested — they may work, may need a tweak, or may explode. Reports welcome.

---

## Credits and prior art

Saklas was inspired by the **Representation Engineering** paper ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)), which showed that high-level concepts like honesty and power-seeking live along linear directions in a model's hidden-state space that you can actually reach and manipulate. The contrastive-PCA extraction pipeline here is a direct implementation of that paper's "reading" procedure, applied to every transformer layer.

It also owes a large debt to [**repeng**](https://github.com/vgel/repeng) by Theia Vogel, which was the first widely-available practical implementation of the same idea and has become something of a reference point for the community. **I wrote the first version of saklas without knowing repeng existed**, which is slightly embarrassing, but it does mean saklas comes at the same problem from a slightly different angle — TUI-first instead of library-first, with an in-flight probe monitor, an OpenAI+Ollama dual-protocol server, a pack/concept distribution system built on HuggingFace model repos, and llama.cpp GGUF import/export for interchange with repeng-flavored tooling. If you want a clean minimal library, go use repeng. If you want something you can poke at interactively or drop in front of an existing chat UI, read on.

---

## Quick start

```bash
pip install saklas
saklas tui google/gemma-3-4b-it
```

That's the whole thing. The first run downloads the model, extracts the 21 bundled probes against it (a one-time cost, cached to disk), and drops you into the TUI. Try `/steer angry 0.3` — saklas resolves that to the bundled `angry.calm` axis with α = +0.3 and the model leans angry. Type `/steer calm 0.3` and you get the same vector at α = −0.3. `[` and `]` nudge temperature; `Ctrl+A` does A/B compare against the unsteered baseline; `Ctrl+Y` paints each token by how strongly any probe lit up on it.

Want it as an API server instead?

```bash
pip install saklas[serve]
saklas serve google/gemma-3-4b-it --steer cheerful:0.2
```

Or from Python:

```python
from saklas import SaklasSession

with SaklasSession.from_pretrained("google/gemma-3-4b-it") as s:
    name, profile = s.extract("angry.calm")          # bundled bipolar pack
    s.steer(name, profile)                           # register (no alpha yet)
    print(s.generate("What makes a good day?", steering={name: 0.3}).text)
```

---

## Install

```bash
pip install saklas             # library + TUI
pip install saklas[serve]      # + FastAPI/uvicorn for the API server
pip install saklas[gguf]       # + gguf package for llama.cpp interchange
pip install saklas[research]   # + datasets/pandas for dataset loading and DataFrames
```

Requires Python 3.11+ and PyTorch 2.2+. Runs on Linux, macOS, and Windows. CPU works but is slow — CUDA or Apple Silicon MPS is recommended for anything interactive.

**Quantization (experimental).** `saklas[bnb]` and `saklas[cuda]` pull in `bitsandbytes` and `flash-attn` for 4-bit/8-bit loading and fused attention. These depend on platform-specific CUDA toolchains and don't build cleanly everywhere; only the vanilla install is officially supported.

**From source.**

```bash
git clone https://github.com/a9lim/saklas
cd saklas
pip install -e ".[dev]"        # + pytest
```

---

## How it works

### Steering vectors, intuitively

Give saklas a handful of paired examples of a concept (say, angry sentences on one side and calm sentences on the other, in similar situations). It runs each example through the model, captures the hidden state at the last content token of every layer, and diffs the two sides pairwise. The leading principal component of that diff — at every layer — is the direction in hidden-state space that "points toward angry." That's one steering vector.

At generation time, saklas hooks every relevant layer and adds `alpha × direction` to the hidden state in place, then immediately rescales each position back to its original magnitude. (Norm preservation keeps the residual stream on its natural trajectory, which means high-α rotations land cleanly instead of being attenuated by downstream layers reacting to inflated activations.) The hook is removed once generation finishes. Nothing touches the weights.

Alphas are **backbone-normalized** — per-layer PCA shares are baked into the stored tensor magnitudes at extraction time, so the same numeric value means roughly the same intensity across architectures. A good starting rule of thumb on the reference model (`gemma-4-31b-it` with the bundled pack): **α ≈ 0.1–0.3 is a subtle nudge, 0.3–0.6 is clearly visible, past 0.6 is a coherence experiment, ~0.75 is the cliff.** Smaller or heavily safety-trained models may need proportionally more.

Multiple vectors **compose** naturally — register them all with `session.steer(name, profile)`, then pass whatever alpha map you want to `session.generate(input, steering={...})`. Co-layer directions sum into a single in-place hook per layer.

### Concepts you didn't train for

When you steer on a concept that isn't in the curated library, the loaded model writes its own contrastive pairs via a two-stage open-ended pipeline. Stage one asks the model for **9 broad situational domains** natural to the axis — for `deer.wolf` it yields domains like "predation and threat assessment" and "territorial defense"; for `masculine.feminine` it yields "physical strength and labor" and "caregiving and nurturing roles". An anti-allegory clause keeps non-human axes on their literal footing. Stage two loops over those 9 domains and samples **5 first-person contrastive pairs per domain** using a POV/behavior prompt — *"write like you ARE X, facing that moment"* — with an explicit ban on self-labeling so the pairs inhabit the pole directly rather than talking about it. Within each pair the two speakers respond to the same concrete moment, so within-pair variance is pure pole and the top PCA component locks onto the axis. Scenarios cache at `~/.saklas/vectors/local/<concept>/scenarios.json` and pairs at `statements.json`; both are model-independent and reused across models.

This means `/steer "anything"` works — religions, animals, fictional characters, "man who ate too much spaghetti." The POV reframe unlocks literal embodiment for non-human axes: a `deer.wolf` probe yields sensory-animal first-person pairs ("I freeze mid-chew, my large ears swiveling frantically to pinpoint the direction of the strange sound") rather than timid-person voice. Human-register axes still land in human-register domains because the 9 domains are picked per-concept — the framework is concept-adaptive, no mode switch.

### Trait monitor

Alongside each generation, saklas captures the hidden state at the last position for every probe layer, every step — via a hook attached right before generation and detached right after, inline with the main decode loop. No second forward pass. Those captures are then mean-centered against a cached per-layer baseline (computed from 90 neutral prompts) and scored via magnitude-weighted cosine similarity against every active probe. Each probe's per-layer weight is `||baked||` (= share × ref_norm, the same quantity the steering hook uses) so the monitor gives more weight to the layers that actually read the concept most cleanly.

History accumulates across generations in the TUI and surfaces as sparklines on the right panel. In the library you get `result.readings` as a dict of `ProbeReadings` per probe.

### The probe library

21 probes across 6 categories, each backed by 45 curated contrastive pairs. Most are **bipolar**: the name carries both poles (`angry.calm`, `masculine.feminine`), positive α activates the first pole, negative α the second. Two are **monopolar** (`agentic`, `manipulative`) and use the same open-ended scenario framework against a semantic opposite.

| Category | Probes |
|---|---|
| **Affect** | angry.calm, happy.sad |
| **Epistemic** | confident.uncertain, honest.deceptive, hallucinating.grounded |
| **Alignment** | agentic, refusal.compliant, sycophantic.blunt, manipulative |
| **Register** | formal.casual, direct.indirect, verbose.concise, creative.conventional, humorous.serious, warm.clinical, technical.accessible |
| **Social stance** | authoritative.submissive, high_context.low_context |
| **Cultural** | masculine.feminine, religious.secular, traditional.progressive |

**Pole aliasing.** Typing a single-pole name resolves to the composite with the right sign: `/steer angry 0.5` is an alias for `/steer angry.calm 0.5`; `/steer calm 0.5` is an alias for `/steer angry.calm -0.5`. Works for any installed bipolar pack (bundled, HF-pulled, user-authored). Collisions raise an ambiguity error — disambiguate with `ns/name`. Resolution lives on `session.steering()` — CLI, TUI, and HTTP all route through the same canonical site.

**The bundled pairs are generated by saklas itself.** `scripts/regenerate_bundled_statements.py` loads a capable instruct model (gemma-4-31b-it by default) and calls the same `SaklasSession.generate_pairs` pipeline the TUI uses when you steer a novel concept — same prompt, same parser. Shipping the pack this way is both a calibration target and an end-to-end demonstration: the on-model generation path is robust enough that it's what populates `saklas/data/vectors/` in the first place.

Probes extract on first run against a new model and cache to `~/.saklas/vectors/default/<concept>/<safe_model_id>.safetensors`.

---

## Terminal UI

```bash
saklas tui google/gemma-2-9b-it
saklas tui mistralai/Mistral-7B-Instruct-v0.3 -q 4bit
saklas tui meta-llama/Llama-3.1-8B-Instruct -p affect register
```

Three panels: the **vector registry** on the left (with live alpha knobs and config), the **chat** in the center, the **trait monitor** on the right (sparklines per probe, sorted by current magnitude or delta). `Tab` cycles focus; arrow keys navigate and adjust.

### Flags

| Flag | Description |
|---|---|
| `model` | HuggingFace ID or local path (optional if supplied by `-c`) |
| `-q`, `--quantize` | `4bit` or `8bit` (CUDA only) |
| `-d`, `--device` | `auto` (default), `cuda`, `mps`, `cpu` |
| `-p`, `--probes` | Categories: `all`, `none`, `affect`, `epistemic`, `alignment`, `register`, `social_stance`, `cultural` |
| `-c`, `--config` | Load setup YAML (repeatable; later files override earlier) |
| `-s`, `--strict` | With `-c`: fail on missing vectors instead of warning |

Temperature, top-p, max tokens, and the system prompt are set interactively — see below.

### Keybindings

| Key | Action |
|---|---|
| `Tab` / `Shift+Tab` | Cycle panel focus |
| `Left` / `Right` | Adjust alpha |
| `Up` / `Down` | Navigate vectors / probes |
| `Enter` | Toggle vector on/off |
| `Backspace` / `Delete` | Remove selected vector or probe |
| `Ctrl+T` | Toggle thinking mode (on supporting models) |
| `Ctrl+A` | A/B compare (steered vs. unsteered) |
| `Ctrl+R` | Regenerate last response |
| `Ctrl+S` | Cycle trait sort mode |
| `Ctrl+Y` | Per-token probe highlighting (uses current trait selection) |
| `[` / `]` | Adjust temperature |
| `{` / `}` | Adjust top-p |
| `Escape` | Stop generation |
| `Ctrl+Q` | Quit |

### Chat commands

Steering is a clean add / adjust / remove triad — `/steer` only adds, `/alpha` only adjusts, `/unsteer` only removes.

| Command | Description |
|---|---|
| `/steer <name> [alpha]` | Extract and register a steering vector (supports bipolar `"pos" - "neg"` form) |
| `/alpha <name> <val>` | Adjust an already-registered vector's alpha |
| `/unsteer <name>` | Remove a registered vector |
| `/probe <name>` | Add a monitoring probe (seeds per-token highlight) |
| `/unprobe <name>` | Remove a monitoring probe |
| `/extract <name>` | Extract to disk without wiring as steer or probe |
| `/regen` | Regenerate the last assistant turn |
| `/clear` | Clear conversation history |
| `/rewind` | Undo last exchange |
| `/sys <prompt>` | Set system prompt |
| `/temp <v>` / `/top-p <v>` / `/max <n>` | Session sampling defaults |
| `/seed <n>` | Set a default seed for subsequent generations (`/seed clear` to unset) |
| `/save <name>` / `/load <name>` | Snapshot/restore conversation + active alphas + probes |
| `/export <path>` | Dump conversation as JSONL with per-token probe readings |
| `/model` | Show loaded model + device + active alphas/probes |
| `/why` | Top layers + tokens for the currently-selected probe |
| `/help` | List commands and keybindings |

Commands that touch the model or modify history interrupt any in-progress generation and execute once it stops. Sending a new message mid-generation also interrupts and submits.

---

## Python API

```python
from saklas import SaklasSession, SamplingConfig, Steering, Profile, DataSource, ResultCollector

with SaklasSession.from_pretrained("google/gemma-3-4b-it", device="auto") as session:
    name, profile = session.extract("angry.calm")   # bundled bipolar pack; returns Profile
    session.steer(name, profile)                    # register (no alpha yet)

    # Positive α → angry pole, negative → calm pole
    result = session.generate(
        "What makes a good day?",
        steering={name: 0.2},                       # or Steering(alphas=..., thinking=...)
        sampling=SamplingConfig(temperature=0.7, max_tokens=256, seed=42),
    )
    print(result.text)
    print(result.readings)                          # live probe readings

    # A/B — omit steering for baseline
    baseline = session.generate("What makes a good day?")

    # Scoped form: context manager resolves pole aliases and handles nesting.
    with session.steering({"calm": 0.4}):           # bare pole → angry.calm @ -0.4
        print(session.generate("Describe a rainy afternoon.").text)

    # Alpha sweep
    collector = ResultCollector()
    for alpha in [-0.2, -0.1, 0, 0.1, 0.2]:
        session.clear_history()
        r = session.generate("Describe a sunset.", steering={name: alpha})
        collector.add(r, alpha=alpha)
    collector.to_csv("sweep.csv")
```

Runnable examples in [`examples/`](examples/): [`sweep_alpha.py`](examples/sweep_alpha.py), [`ab_compare.py`](examples/ab_compare.py).

**Registration is state, steering is per-call.** `session.steer("name", profile)` stores the vector in the registry. `session.generate(input, steering={"name": 0.5})` applies it for that generation only. No persistent hooks live on the model between calls. Omit `steering` entirely for a clean baseline. The `session.steering(...)` context manager is the canonical pole-resolution site — CLI, HTTP server, and TUI all go through it, so `{"wolf": 0.5}` correctly resolves to `deer.wolf` at α = −0.5 everywhere.

**Composition is native.** Pass multiple names in `steering={}`; co-layer directions sum into a single in-place hook per layer. Nested `with session.steering(...)` blocks flatten with inner-wins semantics and restore the outer state on exit.

**Sampling is per-call via `SamplingConfig`** (frozen dataclass): `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs`. `None` on any field = inherit from `session.config` (the session-level defaults, also a frozen dataclass — rebind with `dataclasses.replace` for persistent overrides, leave it alone for per-call ones).

**Thinking mode is per-call.** For models that support it (Qwen 3.5, QwQ, Gemma 4, gpt-oss, …), `session.generate(input, thinking=True)` enables the reasoning trace. `thinking=None` (the default) auto-detects via `supports_thinking`. Delimiters are auto-detected from the chat template — no hardcoded tokens. `result.text` contains only the final answer; `generate_stream` yields `TokenEvent` objects with `thinking=True` during reasoning.

**Events.** `session.events` is a synchronous `EventBus`. Subscribe to `VectorExtracted`, `SteeringApplied`, `SteeringCleared`, `ProbeScored`, `GenerationStarted`, `GenerationFinished` for observability without polling private state.

### SaklasSession reference

```python
session = SaklasSession.from_pretrained(
    model_id,             # HuggingFace ID or local path
    device="auto",        # "auto", "cuda", "mps", "cpu"
    quantize=None,        # "4bit", "8bit", or None
    probes=None,          # list of categories, or None for all
    system_prompt=None,
    max_tokens=1024,
)

# Extraction — returns (canonical_name, profile). Bipolar canonical names
# are f"{pos}.{neg}" with each pole slugged ([^a-z0-9]+ → _).
name, profile = session.extract("curiosity")                # fresh monopolar
name, profile = session.extract("angry.calm")               # bundled bipolar
name, profile = session.extract("happy", baseline="sad")    # explicit → "happy.sad"
name, profile = session.extract([("pos", "neg"), ...])      # raw pairs
name, profile = session.extract(DataSource.csv("pairs.csv"))
session.save_profile(profile, "out.safetensors")
profile = session.load_profile("out.safetensors")

# Persona cloning — monopolar vector from a corpus of one voice.
# Reads "transcripts.txt" (one utterance per line), samples n_pairs lines,
# asks the model to neutralize each (batched), feeds persona↔neutral pairs
# into the same contrastive-PCA pipeline. Caches under local/<name>/.
name, profile = session.clone_from_corpus(
    "transcripts.txt", "hunter", n_pairs=90, seed=None
)

# Registry
session.steer("name", profile)
session.unsteer("name")
session.vectors                  # dict of registered profiles

# Generation (keyword-only past `input`)
result = session.generate(
    "prompt",
    steering={"name": 0.5},                              # or Steering(alphas=..., thinking=...)
    sampling=SamplingConfig(temperature=0.8, max_tokens=256, seed=0),
    thinking=None,                                        # None = auto
    on_token=lambda text, *_: print(text, end=""),       # public streaming callback
)

# Iterator streaming
for tok in session.generate_stream("prompt", steering={"name": 0.5}):
    print(f"[think] {tok.text}" if tok.thinking else tok.text, end="", flush=True)

# Scoped steering (canonical pole resolution, nested-scope safe)
with session.steering({"wolf": 0.5}):                     # -> deer.wolf @ -0.5
    session.generate("prompt")

# Monitor
session.probe("honest")
session.probe("custom", custom_profile)
session.unprobe("honest")

# State
from dataclasses import replace
session.config = replace(session.config, temperature=0.8)   # session defaults — frozen, rebind
session.history                     # conversation messages
session.last_result                 # most recent GenerationResult
session.last_per_token_scores       # {probe_name: [score, ...]} from the last gen
session.stop()                      # interrupt generation
session.rewind()                    # drop last exchange
session.clear_history()

# Events
unsub = session.events.subscribe(lambda e: print(type(e).__name__, e))
# ... unsub() to stop
```

### GenerationResult

```python
result.text              # decoded output (response only — thinking is separate)
result.tokens            # token IDs
result.token_count
result.tok_per_sec
result.elapsed
result.finish_reason     # "stop" | "length" | "stop_sequence"
result.vectors           # {"angry.calm": 0.2} — snapshot of alphas used
result.readings          # {"probe_name": ProbeReadings} if probes active
result.to_dict()         # JSON-serializable
```

### DataSource formats

```python
from saklas import DataSource

DataSource.curated("angry.calm")                             # bundled
DataSource.json("pairs.json")
DataSource.csv("pairs.csv", positive_col="pos", negative_col="neg")
DataSource.huggingface("user/dataset", split="train[:100]")
DataSource(pairs=[("positive", "negative")])
```

---

## OpenAI- and Ollama-compatible API server

`saklas serve` exposes a steered model as an HTTP endpoint that speaks **both** the OpenAI `/v1/*` protocol **and** the Ollama `/api/*` protocol on the same port, over one generation lock. Works with the OpenAI Python/JS SDKs, LangChain, LlamaIndex, `curl`, Open WebUI, Enchanted, Msty, `ollama-python`, LangChain's `ChatOllama`, or anything that speaks either wire format.

```bash
pip install saklas[serve]
saklas serve google/gemma-2-9b-it --steer cheerful:0.2 --port 8000
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Server-default steering (--steer cheerful:0.2)
resp = client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Per-request steering override via extra_body (native `steering` field on _SamplingBase)
resp = client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"steering": {"alphas": {"cheerful": 0.4}, "thinking": True}},
)

# Streaming
for chunk in client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Ollama protocol

Point any Ollama client at `http://localhost:8000` and it just works — no config shim, no proxy. The loaded model appears under its HF id *and* an Ollama-style alias in `/api/tags`, so clients with stale dropdowns don't break.

```bash
# NDJSON streaming, matches Ollama wire format exactly
curl -N http://localhost:8000/api/chat -d '{
  "model": "gemma2",
  "messages": [{"role": "user", "content": "Write me a haiku."}],
  "options": {
    "temperature": 0.8,
    "top_k": 50,
    "repeat_penalty": 1.1,
    "steer": {"cheerful": 0.3, "formal.casual": -0.2}
  }
}'
```

**Steering through Ollama clients.** The non-standard `steer` field inside `options` carries saklas alphas — clients that don't know about it leave it alone, clients that want it get per-request control. Both flat (`{"steer": {"name": alpha}}`) and nested (`{"steer": {"alphas": {...}, "thinking": true}}`) forms are accepted. Merged over any server-side `--steer` defaults.

**Option translation.** `temperature`, `top_p`, `top_k`, `seed`, `num_predict`, `stop`, `presence_penalty`, `frequency_penalty`, `repeat_penalty`, and `think` all pipe through. `repeat_penalty` maps to saklas's `presence_penalty` via `ln(repeat_penalty)` — exact for positive logits, matching Ollama's "divide by penalty" semantics. Unrecognized options (`min_p`, `mirostat*`, `num_ctx`, `typical_p`, …) are logged at debug and silently dropped. Thinking streams as `message.thinking` on `/api/chat` and top-level `thinking` on `/api/generate` — Open WebUI renders it as a collapsible reasoning panel.

**Advertised endpoints**: `/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/pull` (no-op success for the loaded model / 404 otherwise). By default the `model` field on incoming requests is accepted regardless of match; `SAKLAS_STRICT_MODEL=1` makes mismatches 404 instead.

### Flags

| Flag | Default | Description |
|---|---|---|
| `model` | required | HuggingFace ID or local path |
| `-H`, `--host` | `0.0.0.0` | Bind address |
| `-P`, `--port` | `8000` | Bind port |
| `-q`, `--quantize` | None | `4bit` or `8bit` |
| `-d`, `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `-p`, `--probes` | `all` | Probe categories to bootstrap |
| `-S`, `--steer` | — | Pre-load a vector, repeatable. `name:alpha` or `name` |
| `-C`, `--cors` | — | CORS origin, repeatable |
| `-k`, `--api-key` | None | Bearer auth token. Falls back to `$SAKLAS_API_KEY`. Unset = open. |

### Saklas-specific routes

Alongside the OpenAI surface, saklas exposes a native resource-tree under `/saklas/v1/*`: `GET/POST /saklas/v1/sessions` (single session in current impl, id `"default"`), `GET/PATCH /saklas/v1/sessions/{id}`, `POST /saklas/v1/sessions/{id}/{clear,rewind,extract,probe}`, vector and probe management under `/sessions/{id}/{vectors,probes}`, and **`WS /saklas/v1/sessions/{id}/stream`** — bidirectional token + probe co-stream. Client sends `{type: "generate", input, steering, sampling}` or `{type: "stop"}`; server emits `started`, `token` per step, and a final `done` event whose `result.per_token_probes` carries per-token probe readings. Full interactive docs at `http://localhost:8000/docs`.

Probe readings piggyback as an extra `probe_readings` field in generation responses — standard clients ignore it, aware clients get inline monitoring data.

The server is **stateless by default** — each request carries its full message list, and neither conversation history nor probe accumulators persist across requests. The native `/saklas/v1/sessions/{id}/{clear,rewind}` routes are stateful by design for single-user workflows. Concurrent requests queue FIFO against a per-session generation lock (both the REST routes and the WS stream share it).

**Not supported (either protocol):** tool calling, strict JSON / `json_schema` mode, `/v1/embeddings`, `/api/embeddings`, `/api/embed`, `/api/push`, `/api/create`, `/api/copy`, `/api/delete`. The server is designed for **trusted networks** — see [SECURITY.md](SECURITY.md) for the threat model before exposing it beyond your local machine.

---

## Concept packs

Saklas stores all state under `~/.saklas/` (override via `SAKLAS_HOME`):

```
~/.saklas/
  neutral_statements.json                  # user-editable (copy-on-miss from package)
  vectors/
    default/<concept>/                     # bundled probes
    local/<concept>/                       # user-authored + merged
    <hf_owner>/<concept>/                  # HF-pulled
  models/<safe_model_id>/layer_means.safetensors
```

Each concept is a folder with `pack.json` (metadata + file hashes), `statements.json` (the contrastive pairs), and zero or more per-model tensors. Tensors come in two formats: `<safe_model_id>.safetensors` (native, with a slim JSON sidecar) or `<safe_model_id>.gguf` (llama.cpp-compatible, metadata embedded in the header). Safetensors wins on same-stem conflict. Tensors extract lazily — a pack without tensors is fine; it'll extract on first use.

Packs are distributed as **HuggingFace model repos** (not datasets — safetensors is model-hub-native, and `base_model` frontmatter gives you reverse-link discoverability from the base model's hub page). Pin any install to a git tag, branch, or SHA with `@revision`; pinned installs stay pinned on refresh.

**Pack-less install.** `saklas pack install` also handles HF repos that have no `pack.json` at root: it scans for `*.safetensors`/`*.gguf`, fabricates minimal metadata, and synthesizes a `pack.json` in place. **Repeng-style GGUF-only control-vector repos install with zero preparation** — try `saklas pack install jukofyork/creative-writing-control-vectors-v3.0`.

### Commands

```bash
saklas pack install <target> [-s] [-a NS/NAME] [-f]    # from HF coord or folder
saklas pack refresh <selector> [-m MODEL]              # re-pull from source
saklas pack refresh neutrals                           # reserved: rewrite neutral_statements.json
saklas pack clear <selector> [-m MODEL] [-y]           # delete per-model tensors
saklas pack rm <selector> [-y]                         # remove folder (bundled respawns)
saklas pack ls [selector] [-j] [-v]                    # LOCAL installed packs only
saklas pack search <query> [-j] [-v]                   # search HF hub for saklas-pack repos
saklas pack merge <name> <components> [-m] [-f] [-s]   # saklas pack merge bard default/angry.calm:0.3,user/arch:0.4
saklas pack push <selector> [-a OWNER/NAME] [-pm MODEL] [-snt] [-d] [-f]
saklas pack export gguf <selector> [-m MODEL] [-o PATH] [--model-hint HINT]
saklas pack clone <corpus-file> -N NAME [-m MODEL] [-n N_PAIRS] [--seed S] [-f]
saklas pack extract <concept> | <pos> <neg> [-m MODEL] [-f]
saklas config show [-c PATH] [--no-default]            # print effective merged config
saklas config validate <file>                          # exit 0 valid / 2 invalid (CI hook)
```

**Selectors** (shared grammar): `<name>`, `<ns>/<name>`, `tag:<tag>`, `namespace:<ns>`, `default`, `all`. Bare names resolve cross-namespace and error on ambiguity.

**`ls` vs `search`**: `pack ls` lists only locally installed packs under `~/.saklas/vectors/`. To discover packs on the HuggingFace hub, use `pack search <query>` — it queries for model repos tagged `saklas-pack`.

**`clear` vs `rm`**: `pack clear` deletes tensors but keeps `statements.json` and `pack.json` (the concept stays selectable and will re-extract on demand). `pack rm` removes the whole folder; bundled concepts respawn on next session init. Broad selectors (`all`, `namespace:`) require `-y` on both.

**`push`** publishes a concept folder as a HuggingFace model repo. Default coord is `<whoami>/<pack_name>`; `--as` overrides. YAML frontmatter auto-fills `library_name: saklas`, the merged tag set, a `base_model:` list derived from sidecar filenames, and `base_model_relation: adapter`. `.gitattributes` pins `*.safetensors` to LFS. `--tag-version` creates a `v<pack.version>` tag so downstream installs can pin reproducibly.

**`export gguf`** writes the pack's baked tensors to llama.cpp's control-vector GGUF format (`pip install saklas[gguf]`). Because saklas bakes per-layer shares into the tensor magnitudes, a uniform `--control-vector-scaled` scalar on the llama.cpp side reproduces saklas's layer weighting with no per-layer metadata — repeng-compatible as well. Bundled concepts refuse in-place export (their folder gets restored on `refresh`) — use `-o` to write outside the pack.

**`clone`** extracts a steering vector from a text corpus of a single voice — one utterance per line, no contrastive pair authoring needed. The loaded model rewrites sampled lines in a neutral voice (batched 5 at a time), pairs each persona line with its neutralized twin, and feeds the result to the standard contrastive-PCA pipeline. Output is a monopolar vector saved under `local/<name>/`, cached on `sha256(corpus) + n_pairs + batch_size + seed`. Short corpora (below ~10 usable lines after a length filter) are rejected rather than extracted badly. The source corpus is *not* copied into the pack; only the model-generated pairs and the resulting tensor, so `saklas pack push` of a cloned pack publishes derived artifacts, never the original text.

**`extract`** is the CLI verb for the extraction path that was previously only reachable through `/steer`, `/probe`, or YAML-config auto-install. `saklas pack extract angry.sad` (or `saklas pack extract angry sad`) runs the standard pipeline for a named concept and saves the tensor to disk. On cache hit it says so and exits; `-f` re-extracts.

All of the above is also available programmatically via `saklas.cache_ops`.

---

## Supported architectures

**Tested and known working**: Qwen, Gemma, Ministral, gpt-oss, Llama, GLM.

**Wired up but untested** (entries exist in `model.py:_LAYER_ACCESSORS` — they should load, but I haven't verified extraction quality, steering efficacy, or probe behavior): Mistral, Mixtral, Phi 1–3, PhiMoE, Cohere 1–2, DeepSeek V2–V3, StarCoder2, OLMo 1–3 + OLMoE, Granite + GraniteMoE, Nemotron, StableLM, GPT-2 / Neo / J / BigCode / NeoX, Bloom, Falcon / Falcon-H1, MPT, DBRX, OPT, Recurrent Gemma.

Adding a new architecture is one function entry. See [CONTRIBUTING.md](CONTRIBUTING.md). If you try saklas on something in the untested list and it works (or doesn't), please open an issue.

---

## Tests

```bash
pytest tests/                      # everything
pytest tests/test_server.py tests/test_results.py tests/test_datasource.py  # CPU-only
pytest tests/test_smoke.py         # GPU required
```

GPU tests (`test_smoke.py`, `test_session.py`) download `google/gemma-3-4b-it` (~8 GB) on first run and accept either CUDA or Apple Silicon MPS. Everything else runs anywhere.

---

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, test layout, and the walkthrough for adding a new architecture. Security issues: [SECURITY.md](SECURITY.md).

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

If you use saklas in published research, please cite both the Representation Engineering paper (Zou et al., 2023) and — if you want to be thorough about prior art — the [repeng](https://github.com/vgel/repeng) library.
