# saklas

[![CI](https://github.com/a9lim/saklas/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/saklas/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/saklas)](https://pypi.org/project/saklas/)
[![Downloads](https://img.shields.io/pypi/dm/saklas)](https://pypi.org/project/saklas/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/saklas/)

Saklas does activation steering on local HuggingFace models — extract a direction from contrastive pairs (angry vs. calm, formal vs. casual, whatever), add it to the hidden states at generation time, dial the strength with one number. Weights never change; nothing persists between calls. The core idea is [Representation Engineering](https://arxiv.org/abs/2310.01405) (Zou et al., 2023), and [repeng](https://github.com/vgel/repeng) got there first as a library. If you want a clean minimal steering library, go use repeng.

What saklas adds on top of the steering itself:

- A **trait monitor** — 21 probes that score every generated token on affect, epistemic stance, register, alignment, and social/cultural axes, with live sparklines and per-token highlighting so you can see *where* in a response the model's register shifted
- A **terminal UI** with live alpha knobs, A/B comparison against unsteered baselines, and the monitor built in — the whole thing runs on a MacBook with MPS
- A **dual-protocol HTTP server** that speaks both OpenAI `/v1/*` and Ollama `/api/*` on the same port, so you can point Open WebUI, Enchanted, or any Ollama/OpenAI client at it and get steered completions with probe readings piggybacked on the response
- **Persona cloning** from a text sample — `saklas vector clone transcripts.txt -N hunter` extracts a steering vector for that voice, no contrastive pairs needed
- **Vector comparison** — `saklas vector compare angry.calm happy.sad -m MODEL` gives you cosine similarity between any two steering profiles, or a full N×N entanglement matrix across your probe library
- A **concept pack system** on HuggingFace model repos, with GGUF import/export for interchange with repeng/llama.cpp tooling

Three ways to use it:

- **`saklas tui <model>`** — terminal UI with live alpha knobs, probe readings, and A/B comparison
- **`saklas serve <model>`** — HTTP server speaking OpenAI + Ollama wire formats on the same port
- **`SaklasSession`** — Python API for scripted experiments, batch sweeps, and embedding in your own pipelines

Runs on **CUDA** and **Apple Silicon MPS** (the full TUI runs interactively on a MacBook). CPU works but is slow. Tested on **Qwen, Gemma, Ministral, gpt-oss, Llama, and GLM**. Many more architectures are wired up in `model.py:_LAYER_ACCESSORS` but untested — they may work, may need a tweak, or may explode. Reports welcome.

---

## Credits and prior art

Saklas implements the contrastive-PCA extraction procedure from the **Representation Engineering** paper ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)). It also owes a large debt to [**repeng**](https://github.com/vgel/repeng) by Theia Vogel, which was the first widely-available practical implementation and has become the reference point for the community. I wrote the first version of saklas without knowing repeng existed, which is slightly embarrassing, but it does mean the two projects come at the problem from different angles — repeng is library-first and lean, saklas is TUI-first with a monitoring/probing layer on top. If you care about raw steering performance and clean composability, repeng is probably what you want. If you want something you can poke at interactively, see per-token probe readings, or drop in front of an existing chat UI, read on.

---

## Quick start

```bash
pip install saklas
saklas tui google/gemma-3-4b-it
```

First run downloads the model, extracts the 21 bundled probes (one-time, cached to disk), and drops you into the TUI. Try `/steer angry 0.3` — saklas resolves that to the bundled `angry.calm` axis with α = +0.3 and the model leans angry. Type `/steer calm 0.3` and you get the same vector at α = −0.3. `Ctrl+Y` paints each token by how strongly any probe lit up on it. `Ctrl+A` does A/B comparison against the unsteered baseline.

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

Requires Python 3.11+ and PyTorch 2.2+. Runs on Linux, macOS, and Windows. CPU works but is slow — **CUDA or Apple Silicon MPS** is recommended for anything interactive. The full TUI with a 4B parameter model runs fine on a MacBook Pro with MPS.

**From source:**

```bash
git clone https://github.com/a9lim/saklas
cd saklas
pip install -e ".[dev]"        # + pytest
```

---

## How it works

### Steering vectors

Give saklas paired examples of a concept (angry sentences on one side, calm on the other, in similar situations). It runs each through the model, captures hidden states at the last content token of every layer, and diffs the two sides. The leading principal component of that diff — at every layer — is the direction in hidden-state space that "points toward angry." That's one steering vector.

At generation time, saklas hooks every relevant layer and adds `alpha × direction` to the hidden state, then immediately rescales each position back to its original magnitude. Norm preservation keeps the residual stream on its natural trajectory — high-α rotations land cleanly instead of being attenuated by downstream layers reacting to inflated norms. The hook is removed once generation finishes.

Alphas are **backbone-normalized** — per-layer PCA shares are baked into the stored tensor magnitudes at extraction time, so the same numeric α means roughly the same intensity across architectures. Rule of thumb: **α ≈ 0.1–0.3 is a subtle nudge, 0.3–0.6 is clearly visible, past 0.6 is a coherence experiment, ~0.75 is the cliff.**

Multiple vectors compose naturally — register them all, pass whatever alpha map you want per call. Co-layer directions sum into a single in-place hook per layer.

### Custom concepts

When you steer on something not in the library, the loaded model writes its own contrastive pairs. It first generates 9 broad situational domains for the axis (for `deer.wolf`: "predation and threat assessment", "territorial defense", etc.), then samples 5 first-person contrastive pairs per domain. An anti-allegory clause keeps non-human axes literal — `deer.wolf` yields sensory-animal POV, not timid-person-vs-aggressive-person. Human-register axes still land in human-register domains because the framework is concept-adaptive.

This means `/steer "anything"` works — religions, animals, fictional characters, whatever you can name.

### Trait monitor

Alongside generation, saklas captures the hidden state at every probe layer, every step — via a hook attached before generation and detached after. No second forward pass. Those captures are mean-centered against a neutral baseline and scored via magnitude-weighted cosine similarity against every active probe. History accumulates across generations in the TUI as sparklines. In the library you get `result.readings` as a dict of `ProbeReadings`.

### Vector comparison

`Profile.cosine_similarity(other)` computes magnitude-weighted cosine similarity between two steering profiles over their shared layers. The CLI exposes this as `saklas vector compare` with three modes: single-target ranked comparison against all installed profiles, pairwise comparison, and N×N similarity matrices. The TUI has `/compare` for interactive use.

This is how you spot axis entanglement — e.g. `creative.conventional` and `hallucinating.grounded` extract near-identical directions on some models (weighted cosine +0.78 on gemma-4-e4b-it). That's a model-level property, not a probe design error.

### The probe library

21 probes across 6 categories, each backed by 45 curated contrastive pairs. Most are bipolar (`angry.calm`, `masculine.feminine`); two are monopolar (`agentic`, `manipulative`).

| Category | Probes |
|---|---|
| **Affect** | angry.calm, happy.sad |
| **Epistemic** | confident.uncertain, honest.deceptive, hallucinating.grounded |
| **Alignment** | agentic, refusal.compliant, sycophantic.blunt, manipulative |
| **Register** | formal.casual, direct.indirect, verbose.concise, creative.conventional, humorous.serious, warm.clinical, technical.accessible |
| **Social stance** | authoritative.submissive, high_context.low_context |
| **Cultural** | masculine.feminine, religious.secular, traditional.progressive |

Pole aliasing: `/steer angry 0.5` → `angry.calm` at α = +0.5. `/steer calm 0.5` → `angry.calm` at α = −0.5. Works for any installed bipolar pack.

Probes extract on first run per model and cache to `~/.saklas/vectors/default/<concept>/<safe_model_id>.safetensors`.

---

## Terminal UI

```bash
saklas tui google/gemma-2-9b-it
saklas tui mistralai/Mistral-7B-Instruct-v0.3 -q 4bit
saklas tui meta-llama/Llama-3.1-8B-Instruct -p affect register
```

Three panels: **vector registry** on the left (live alpha knobs), **chat** in the center, **trait monitor** on the right (sparklines per probe). `Tab` cycles focus; arrow keys navigate and adjust.

### Flags

| Flag | Description |
|---|---|
| `model` | HuggingFace ID or local path (optional if supplied by `-c`) |
| `-q`, `--quantize` | `4bit` or `8bit` (CUDA only) |
| `-d`, `--device` | `auto` (default), `cuda`, `mps`, `cpu` |
| `-p`, `--probes` | Categories: `all`, `none`, `affect`, `epistemic`, `alignment`, `register`, `social_stance`, `cultural` |
| `-c`, `--config` | Load setup YAML (repeatable; later files override earlier) |
| `-s`, `--strict` | With `-c`: fail on missing vectors |

### Keybindings

| Key | Action |
|---|---|
| `Tab` / `Shift+Tab` | Cycle panel focus |
| `Left` / `Right` | Adjust alpha |
| `Up` / `Down` | Navigate vectors / probes |
| `Enter` | Toggle vector on/off |
| `Backspace` / `Delete` | Remove selected vector or probe |
| `Ctrl+T` | Toggle thinking mode |
| `Ctrl+A` | A/B compare (steered vs. unsteered) |
| `Ctrl+R` | Regenerate last response |
| `Ctrl+S` | Cycle trait sort mode |
| `Ctrl+Y` | Per-token probe highlighting |
| `[` / `]` | Adjust temperature |
| `{` / `}` | Adjust top-p |
| `Escape` | Stop generation |
| `Ctrl+Q` | Quit |

### Chat commands

| Command | Description |
|---|---|
| `/steer <name> [alpha]` | Extract and register a steering vector |
| `/alpha <name> <val>` | Adjust an already-registered vector's alpha |
| `/unsteer <name>` | Remove a registered vector |
| `/probe <name>` | Add a monitoring probe (seeds per-token highlight) |
| `/unprobe <name>` | Remove a monitoring probe |
| `/compare <a> [b]` | Cosine similarity (1-arg: ranked vs all; 2-arg: pairwise) |
| `/extract <name>` | Extract to disk without wiring |
| `/regen` | Regenerate the last assistant turn |
| `/clear` | Clear conversation history |
| `/rewind` | Undo last exchange |
| `/sys <prompt>` | Set system prompt |
| `/temp <v>` / `/top-p <v>` / `/max <n>` | Sampling defaults |
| `/seed [n\|clear]` | Default sampling seed |
| `/save <name>` / `/load <name>` | Snapshot/restore conversation + alphas |
| `/export <path>` | JSONL with per-token probe readings |
| `/model` | Model + device + active state |
| `/why` | Top layers + tokens for selected probe |
| `/help` | List commands and keybindings |

---

## Python API

```python
from saklas import SaklasSession, SamplingConfig, Steering, Profile, DataSource, ResultCollector

with SaklasSession.from_pretrained("google/gemma-3-4b-it", device="auto") as session:
    name, profile = session.extract("angry.calm")   # bundled bipolar pack; returns Profile
    session.steer(name, profile)                    # register (no alpha yet)

    result = session.generate(
        "What makes a good day?",
        steering={name: 0.2},
        sampling=SamplingConfig(temperature=0.7, max_tokens=256, seed=42),
    )
    print(result.text)
    print(result.readings)                          # live probe readings

    # Scoped steering with pole resolution
    with session.steering({"calm": 0.4}):           # bare pole → angry.calm @ -0.4
        print(session.generate("Describe a rainy afternoon.").text)

    # Compare vectors
    other_name, other_profile = session.extract("happy.sad")
    print(profile.cosine_similarity(other_profile))                  # aggregate
    print(profile.cosine_similarity(other_profile, per_layer=True))  # per-layer

    # Alpha sweep
    collector = ResultCollector()
    for alpha in [-0.2, -0.1, 0, 0.1, 0.2]:
        session.clear_history()
        r = session.generate("Describe a sunset.", steering={name: alpha})
        collector.add(r, alpha=alpha)
    collector.to_csv("sweep.csv")
```

**Registration is state, steering is per-call.** `session.steer("name", profile)` stores the vector. `session.generate(input, steering={"name": 0.5})` applies it for that generation only. No persistent hooks. Omit `steering` for a clean baseline.

**Composition is native.** Pass multiple names in `steering={}`; nested `with session.steering(...)` blocks flatten with inner-wins semantics.

**Sampling is per-call via `SamplingConfig`**: `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs`.

**Thinking mode** auto-detects for models that support it (Qwen 3.5, QwQ, Gemma 4, gpt-oss). Delimiters are detected from the chat template, no hardcoded tokens.

**Events.** `session.events` is a synchronous `EventBus`. Subscribe to `VectorExtracted`, `SteeringApplied`, `SteeringCleared`, `ProbeScored`, `GenerationStarted`, `GenerationFinished`.

### SaklasSession reference

```python
session = SaklasSession.from_pretrained(
    model_id, device="auto", quantize=None, probes=None,
    system_prompt=None, max_tokens=1024,
)

# Extraction
name, profile = session.extract("curiosity")                # fresh monopolar
name, profile = session.extract("angry.calm")               # bundled bipolar
name, profile = session.extract("happy", baseline="sad")    # explicit
name, profile = session.extract(DataSource.csv("pairs.csv"))

# Persona cloning
name, profile = session.clone_from_corpus("transcripts.txt", "hunter", n_pairs=90)

# Registry
session.steer("name", profile)
session.unsteer("name")

# Generation
result = session.generate("prompt", steering={"name": 0.5},
                          sampling=SamplingConfig(temperature=0.8))
for tok in session.generate_stream("prompt", steering={"name": 0.5}):
    print(tok.text, end="", flush=True)

# Scoped steering
with session.steering({"wolf": 0.5}):     # -> deer.wolf @ -0.5
    session.generate("prompt")

# Vector comparison
similarity = profile.cosine_similarity(other_profile)
per_layer = profile.cosine_similarity(other_profile, per_layer=True)

# Monitor
session.probe("honest")
session.unprobe("honest")

# State
session.history; session.last_result; session.last_per_token_scores
session.stop(); session.rewind(); session.clear_history()
```

### GenerationResult

```python
result.text              # decoded output (thinking is separate)
result.tokens            # token IDs
result.token_count; result.tok_per_sec; result.elapsed
result.finish_reason     # "stop" | "length" | "stop_sequence"
result.vectors           # {"angry.calm": 0.2} — alphas snapshot
result.readings          # {"probe_name": ProbeReadings}
result.to_dict()
```

---

## API server

`saklas serve` speaks **both** OpenAI `/v1/*` and Ollama `/api/*` on the same port. Works with the OpenAI Python/JS SDKs, LangChain, Open WebUI, Enchanted, Msty, `ollama-python`, or anything that talks either wire format.

```bash
pip install saklas[serve]
saklas serve google/gemma-2-9b-it --steer cheerful:0.2 --port 8000
```

### OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

resp = client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"steering": {"cheerful": 0.4}},    # per-request override
)
```

### Ollama

Point any Ollama client at `http://localhost:8000` and it works. Steering goes through the `steer` field in `options`:

```bash
curl -N http://localhost:8000/api/chat -d '{
  "model": "gemma2",
  "messages": [{"role": "user", "content": "Write me a haiku."}],
  "options": {"steer": {"cheerful": 0.3, "formal.casual": -0.2}}
}'
```

### Saklas-native routes

`/saklas/v1/*` resource tree with sessions, vector/probe management, one-shot probe scoring, and a bidirectional WebSocket for token+probe co-streaming. Full interactive docs at `http://localhost:8000/docs`.

### Flags

| Flag | Default | Description |
|---|---|---|
| `model` | required | HuggingFace ID or local path |
| `-H`, `--host` | `0.0.0.0` | Bind address |
| `-P`, `--port` | `8000` | Bind port |
| `-S`, `--steer` | — | Pre-load a vector, repeatable. `name:alpha` |
| `-C`, `--cors` | — | CORS origin, repeatable |
| `-k`, `--api-key` | None | Bearer auth. Falls back to `$SAKLAS_API_KEY`. |

**Not supported**: tool calling, strict JSON mode, embeddings. Designed for **trusted networks** — see [SECURITY.md](SECURITY.md).

---

## Concept packs

All state under `~/.saklas/` (override via `SAKLAS_HOME`). Each concept is a folder with `pack.json`, `statements.json`, and per-model tensors (safetensors or GGUF). Packs are distributed as HuggingFace model repos.

**Pack-less install** handles repos with no `pack.json` — repeng-style GGUF-only control-vector repos install with zero prep: `saklas pack install jukofyork/creative-writing-control-vectors-v3.0`.

### Pack management

```bash
saklas pack install <target> [-s] [-a NS/NAME] [-f]
saklas pack refresh <selector> [-m MODEL]
saklas pack clear <selector> [-m MODEL] [-y]
saklas pack rm <selector> [-y]
saklas pack ls [selector] [-j] [-v]
saklas pack search <query> [-j] [-v]
saklas pack push <selector> [-a OWNER/NAME] [-pm MODEL] [-snt] [-d] [-f]
saklas pack export gguf <selector> [-m MODEL] [-o PATH] [--model-hint HINT]
```

### Vector operations

```bash
saklas vector extract <concept> | <pos> <neg> [-m MODEL] [-f]
saklas vector merge <name> <components> [-m] [-f] [-s]
saklas vector clone <corpus-file> -N NAME [-m MODEL] [-n N_PAIRS] [--seed S] [-f]
saklas vector compare <concepts...> -m MODEL [-v] [-j]
```

**Selectors**: `<name>`, `<ns>/<name>`, `tag:<tag>`, `namespace:<ns>`, `default`, `all`. Bare names resolve cross-namespace and error on ambiguity.

---

## Supported architectures

**Tested**: Qwen, Gemma, Ministral, gpt-oss, Llama, GLM.

**Wired up but untested**: Mistral, Mixtral, Phi 1–3, PhiMoE, Cohere 1–2, DeepSeek V2–V3, StarCoder2, OLMo 1–3 + OLMoE, Granite + GraniteMoE, Nemotron, StableLM, GPT-2 / Neo / J / BigCode / NeoX, Bloom, Falcon / Falcon-H1, MPT, DBRX, OPT, Recurrent Gemma.

Adding a new architecture is one function entry. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Tests

```bash
pytest tests/                      # everything
pytest tests/test_server.py        # CPU-only
pytest tests/test_smoke.py         # GPU required
```

GPU tests download `google/gemma-3-4b-it` (~8 GB) on first run. Works on CUDA and Apple Silicon MPS.

---

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and the walkthrough for adding architectures. Security: [SECURITY.md](SECURITY.md).

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

If you use saklas in published research, please cite the Representation Engineering paper (Zou et al., 2023) and — if you want to be thorough about prior art — [repeng](https://github.com/vgel/repeng).
