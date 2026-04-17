# saklas

[![CI](https://github.com/a9lim/saklas/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/saklas/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/saklas)](https://pypi.org/project/saklas/)
[![Downloads](https://img.shields.io/pypi/dm/saklas)](https://pypi.org/project/saklas/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/saklas/)

Saklas is a library for activation steering and trait probing on local HuggingFace models. You point it at a concept — angry vs. calm, formal vs. casual, any axis you can name — and it generates contrastive pairs, extracts a direction from them, and adds that direction to the model's hidden states at generation time. The base model is never modified, so you can dial steering strength up and down per call with a single number.

It's built on Representation Engineering ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)), the same paper [repeng](https://github.com/vgel/repeng) implements. The main feature is a terminal UI with live steering controls and a built-in trait monitor that scores every generated token against any probe concept you care about, with live averages and sparklines so you can see *where* in a response a trait shifts. There's also a dual-protocol HTTP server that speaks both OpenAI `/v1/*` and Ollama `/api/*` on the same port, so Open WebUI, Enchanted, or any Ollama/OpenAI client can talk to a steered model with no changes. Persona cloning works on any text sample — point it at a corpus and it'll extract a voice/style vector without hand-labeled pairs.

Three ways to use it:

- **`saklas tui <model>`** — the terminal UI. Live alpha knobs, probe readings, A/B compare against the unsteered baseline.
- **`saklas serve <model>`** — HTTP server speaking OpenAI + Ollama on the same port.
- **`SaklasSession`** — Python API for scripted experiments and embedding in your own pipelines.

Runs on **CUDA** and **Apple Silicon MPS**; the full TUI is comfortably interactive on a MacBook. CPU works but is slow. Tested on **Qwen, Gemma, Ministral, gpt-oss, Llama, and GLM**. A lot more architectures are wired up in `model.py:_LAYER_ACCESSORS` and should work out of the box — if you try one, let me know how it went.

---

## Credits

The contrastive-PCA approach comes from the Representation Engineering paper ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)). [repeng](https://github.com/vgel/repeng), by Theia Vogel, is the well-known implementation in this space and is what most people reach for. Saklas covers the same core idea from a different angle: repeng is lean and library-first; saklas is TUI-first with monitoring and a chat-server bundled in. Both are AGPL; both are worth your time.

---

## Quick start

```bash
pip install saklas
saklas tui google/gemma-3-4b-it
```

First run downloads the model and extracts the 21 bundled probes (one-time, cached to disk), then drops you into the TUI. Try `/steer angry 0.3` — saklas resolves that to the bundled `angry.calm` axis at α = +0.3 and the model leans angry. `/steer calm 0.3` gives you the same vector at α = −0.3. `Ctrl+Y` colors each generated token by how strongly any probe lit up on it. `Ctrl+A` does A/B comparison against the unsteered baseline.

As an API server:

```bash
pip install saklas[serve]
saklas serve google/gemma-3-4b-it --steer cheerful:0.2
```

From Python:

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
pip install saklas[sae]        # + sae-lens for SAE-backed extraction
```

Requires Python 3.11+ and PyTorch 2.2+. Runs on Linux, macOS, and Windows. CPU works but is slow; CUDA or Apple Silicon MPS is recommended for anything interactive. A 4B model is comfortable on a MacBook Pro with MPS.

From source:

```bash
git clone https://github.com/a9lim/saklas
cd saklas
pip install -e ".[dev]"        # + pytest
```

---

## How it works

### Steering vectors

You give saklas paired examples of a concept — angry sentences on one side, calm on the other, describing similar situations. It runs each through the model, grabs the hidden state at the last content token of every layer, and diffs the two sides. The leading principal component of that diff (at every layer) is the direction that points toward "angry." That's one steering vector.

At generation time, saklas hooks every relevant layer and adds `alpha × direction` to the hidden state, then rescales the result back to its original magnitude. Norm preservation keeps the residual stream on its usual trajectory so the rotation lands cleanly instead of getting attenuated by downstream layers reacting to an inflated norm. The hook is removed once generation finishes.

Alphas are backbone-normalized: per-layer PCA shares are baked into the tensor magnitudes at extraction, so the same numeric α means roughly the same intensity across architectures. Rough guide:

- **0.1–0.3** — subtle nudge
- **0.3–0.6** — clearly visible in the output
- **0.6+** — starts pulling the model away from coherent output
- **~0.75** — generation tends to fall apart

Multiple vectors compose. Register them all, pass whatever alpha map you want per call; co-layer directions sum into a single in-place hook per layer.

### SAE-backed extraction (optional)

Install `saklas[sae]` and pass `--sae <release>` to `vector extract` to run contrastive PCA in sparse-autoencoder feature space instead of raw residual-stream space. saklas routes through SAELens, so any published release it covers — GemmaScope, Eleuther Meta-LLaMA-3.1 SAEs, Joseph Bloom's, Apollo/Goodfire — works day one. The output plugs into the same hook, monitor, and pack infrastructure as raw PCA; the only visible difference is a `:sae-<release>` suffix on the concept name so raw and SAE flavors can coexist.

```bash
saklas vector extract honest.deceptive -m google/gemma-2-2b-it \
  --sae gemma-scope-2b-pt-res-canonical
```

Then steer the SAE variant explicitly (Python, config, or TUI):

```python
with session.steering({"honest:sae": 0.3}):
    session.generate("...")
```

```yaml
# ~/.saklas/config.yaml
vectors:
  "honest:sae": 0.3
```

```
/steer --sae honest 0.3          # TUI: picks the unique SAE variant
/steer --sae gemma-scope-2b-pt-res-canonical honest 0.3   # explicit release
```

SAE profiles are subset-layer — only layers the release covers — and share-baking redistributes over the covered subset automatically. The 21 bundled concepts ship raw-PCA only; users opt into SAE extraction per-concept.

By default steering fires on every token — prompt prefill, thinking section, response. A `Trigger` narrows that window. Pass one as a per-call default, or attach a different trigger per concept in the same call:

```python
from saklas import Steering, Trigger

# Steer only the response, never the prompt or the thinking section
session.generate("...", steering=Steering(
    alphas={"warm": 0.4},
    trigger=Trigger.AFTER_THINKING,
))

# Mix regimes per concept
session.generate("...", steering=Steering(alphas={
    "honest": 0.3,                              # default trigger (everywhere)
    "warm":   (0.4, Trigger.AFTER_THINKING),    # per-entry override
}))
```

Presets cover the common cases (`BOTH`, `GENERATED_ONLY`, `PROMPT_ONLY`, `AFTER_THINKING`, `THINKING_ONLY`); `Trigger.first(n)` and `Trigger.after(n)` give you a windowed application by generation step; constructing the dataclass directly handles arbitrary regimes. When every entry uses `Trigger.BOTH` the hook path collapses to the v1.4 fast path, so default generations pay nothing for the machinery.

### Custom concepts

When you steer on something not in the library, the loaded model writes its own contrastive pairs. It first sketches 9 broad situational domains for the axis (for `deer.wolf`: "predation and threat assessment", "territorial defense", etc.), then samples 5 first-person contrastive pairs per domain. An anti-allegory clause in the prompt keeps non-human axes literal, so `deer.wolf` yields sensory-animal POV rather than timid-person-vs-aggressive-person. Human-register axes still land in human-register domains because the framework is concept-adaptive.

In practice this means `/steer <anything>` works — religions, animals, fictional characters, whatever you can name.

### Trait monitor

While generating, saklas captures the hidden state at every probe layer at every step through a hook attached before generation and detached after. No second forward pass. The captures are mean-centered against a neutral baseline and scored by magnitude-weighted cosine similarity against every active probe. History accumulates across generations in the TUI as sparklines; in the library you get `result.readings` as a dict of `ProbeReadings`.

### Vector comparison

`Profile.cosine_similarity(other)` gives you magnitude-weighted cosine similarity between two steering profiles over their shared layers. The CLI exposes three modes: single-target ranked comparison against all installed profiles, pairwise comparison, and N×N similarity matrices. The TUI has `/compare` for interactive use.

This is how you spot axis entanglement. For example, `creative.conventional` and `hallucinating.grounded` extract near-identical directions on some models (weighted cosine +0.78 on gemma-4-e4b-it). That's the model itself encoding both axes on the same underlying direction, not a probe design error.

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

Three panels: vector registry on the left (live alpha knobs), chat in the center, trait monitor on the right (sparklines per probe). `Tab` cycles focus; arrow keys navigate and adjust.

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
| `/steer <pos> . <neg> [alpha]` | Same, bipolar form (period delimiter) |
| `/alpha <name> <val>` | Adjust an already-registered vector's alpha |
| `/unsteer <name>` | Remove a registered vector |
| `/probe <name>` | Add a monitoring probe (seeds per-token highlight) |
| `/probe <pos> . <neg>` | Same, bipolar form |
| `/unprobe <name>` | Remove a monitoring probe |
| `/compare <a> [b]` | Cosine similarity (1-arg: ranked vs all; 2-arg: pairwise) |
| `/extract <name>` | Extract to disk without wiring |
| `/extract <pos> . <neg>` | Same, bipolar form |
| `/regen` | Regenerate the last assistant turn |
| `/clear` | Clear conversation history |
| `/rewind` | Undo last exchange |
| `/sys <prompt>` | Set system prompt |
| `/temp <v>` / `/top-p <v>` / `/max <n>` | Sampling defaults |
| `/seed [n\|clear]` | Default sampling seed |
| `/save <name>` / `/load <name>` | Snapshot/restore conversation + alphas |
| `/export <path>` | JSONL with per-token probe readings |
| `/model` | Model + device + active state |
| `/help` | List commands and keybindings |

A **WHY footer** at the bottom of the trait panel shows the top-5 layers (by `||baked||`) and the live top/bottom emitted tokens (by signed score) for whichever probe is selected — driven by selection, no command needed.

The **chat status footer** shows generation progress (token bar against `max_tokens`), live tok/s, elapsed, VRAM, and a **context bar** (prompt + emitted tokens against the model's context window, cyan/yellow/red as it fills). All bars in the UI share one width via `saklas.tui.utils.BAR_WIDTH`.

Bipolar poles don't need quotes: `/steer a dog . a pair of cats 0.4` parses as `pos="a dog", neg="a pair of cats", alpha=0.4`. Whitespace around the period is what splits, so `dog.cat` stays a single canonical name (the bundled-pack form).

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

Registration is state; steering is per-call. `session.steer("name", profile)` stores the vector. `session.generate(input, steering={"name": 0.5})` applies it for that generation only. No persistent hooks. Omit `steering` for a clean baseline.

Composition is native. Pass multiple names in `steering={}`; nested `with session.steering(...)` blocks flatten with inner-wins semantics.

Sampling is per-call via `SamplingConfig`: `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs`.

Thinking mode auto-detects for models that support it (Qwen 3.5, QwQ, Gemma 4, gpt-oss). Delimiters come from the chat template, not a hardcoded list.

`session.events` is a synchronous `EventBus`. Subscribe to `VectorExtracted`, `SteeringApplied`, `SteeringCleared`, `ProbeScored`, `GenerationStarted`, `GenerationFinished`.

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

`saklas serve` speaks both OpenAI `/v1/*` and Ollama `/api/*` on the same port. Works with the OpenAI Python/JS SDKs, LangChain, Open WebUI, Enchanted, Msty, `ollama-python`, and anything else that talks either wire format.

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

`/saklas/v1/*` is a resource tree with sessions, vector and probe management, one-shot probe scoring, a bidirectional WebSocket for token+probe co-streaming, and a live traits SSE endpoint (`GET /saklas/v1/sessions/{id}/traits/stream`) that streams per-token probe scores in real time during any active generation. Interactive docs at `http://localhost:8000/docs`.

### Flags

| Flag | Default | Description |
|---|---|---|
| `model` | required | HuggingFace ID or local path |
| `-H`, `--host` | `0.0.0.0` | Bind address |
| `-P`, `--port` | `8000` | Bind port |
| `-S`, `--steer` | — | Pre-load a vector, repeatable. `name:alpha` |
| `-C`, `--cors` | — | CORS origin, repeatable |
| `-k`, `--api-key` | None | Bearer auth. Falls back to `$SAKLAS_API_KEY`. |

Not supported: tool calling, strict JSON mode, embeddings. The server is designed for trusted networks — see [SECURITY.md](SECURITY.md).

---

## Concept packs

All state lives under `~/.saklas/` (override via `SAKLAS_HOME`). Each concept is a folder with `pack.json`, `statements.json`, and per-model tensors (safetensors or GGUF). Packs are distributed as HuggingFace model repos.

Pack-less install handles repos with no `pack.json`, so repeng-style GGUF-only control-vector repos install with zero prep: `saklas pack install jukofyork/creative-writing-control-vectors-v3.0`.

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
saklas vector why <concept> -m MODEL [-n N] [--all] [-j]
```

Merge supports projection: `a~b:0.5` removes b's direction from a before scaling. For example, `saklas vector merge dehallu default/creative.conventional~default/hallucinating.grounded:0.8` gives you creative writing with the hallucination axis projected out.

Selectors: `<name>`, `<ns>/<name>`, `tag:<tag>`, `namespace:<ns>`, `default`, `all`. Bare names resolve cross-namespace and error on ambiguity.

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

If you use saklas in published research, please cite the Representation Engineering paper (Zou et al., 2023) and [repeng](https://github.com/vgel/repeng).
