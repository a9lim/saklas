# saklas

[![CI](https://github.com/a9lim/saklas/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/saklas/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/saklas)](https://pypi.org/project/saklas/)
[![Downloads](https://img.shields.io/pypi/dm/saklas)](https://pypi.org/project/saklas/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/saklas/)

<p align="center">
  <video src="https://github.com/a9lim/saklas/raw/main/docs/demo.mp4" controls muted playsinline width="720">
    Demo video at <a href="docs/demo.mp4">docs/demo.mp4</a>.
  </video>
</p>

Saklas is a library for activation steering and probing on local HuggingFace models. You give it any concept, from "angry" to "bacterium", and it automatically generates contrastive pairs, extracts a direction from them, and then steers the model's hidden states along that direction when it's time to generate text. The model itself isn't touched, so you can change the steering strength as you go.

Saklas is built on Representation Engineering ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)), the same paper [repeng](https://github.com/vgel/repeng) implements. It has three frontends:

- **`saklas serve <model>`**: web UI at `http://localhost:8000/`. The same port is also compatible with OpenAI `/v1/*` and Ollama `/api/*`. Pass `--no-web` for API-only mode.
- **`saklas tui <model>`**: TUI for terminal use.
- **`SaklasSession`**: Python API for scripted experiments.

It runs on both CUDA and Apple Silicon MPS, and it runs comfortably on a MacBook. Tested to work on Qwen, Gemma, Ministral, GPT-OSS, Llama, GLM, and Talkie, with significantly more architectures experimentally wired up.

---

## Quick start

```bash
pip install saklas
saklas serve google/gemma-4-31b-it
```

Then once it loads, open `http://localhost:8000/`. The first run downloads the model and extracts the 26 bundled probes, which may take a few minutes. 

The same port is also compatible with the OpenAI API format:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(
    model="google/gemma-4-31b-it",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"steering": "0.4 cheerful"},
)
```

If you prefer a terminal:

```bash
pip install saklas
saklas tui google/gemma-4-31b-it
```

Or directly from Python:

```python
from saklas import SaklasSession

with SaklasSession.from_pretrained("google/gemma-4-31b-it") as s:
    name, profile = s.extract("angry.calm")          # bundled bipolar pack
    s.steer(name, profile)                           # register (no alpha yet)
    print(s.generate("What makes a good day?", steering=f"0.3 {name}").text)
```

---

## Reporting issues

If you notice any errors while using the program, please update to the most recent version. If the error still persists, please open an issue. This project is a work in progress and I am actively finding and fixing bugs.

---

## Credits

The contrastive-pair approach comes from the Representation Engineering paper ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)). [repeng](https://github.com/vgel/repeng) by Theia Vogel is the well-known implementation in this space. Saklas implements the same idea from a different angle: repeng is lean and more of a library, saklas is a full workbench with monitoring, branching chat, and a server built in. Both are worth your time!

Since v2.1 the default extractor is difference-of-means (DiM) per [Im & Li, 2025](https://arxiv.org/abs/2502.02716), and the per-layer share allocation runs in the Mahalanobis metric (whitened against per-model activation covariance) instead of the v1.x Euclidean magnitude. The legacy v1.x stack (PCA extraction, additive steering, Euclidean shares and cosine) is available all together via `--legacy` on `tui`, `serve`, `vector extract`, and `vector compare`, or piecemeal via `--method pca`, `--steer-mode additive`, and `--metric euclidean` on the relevant verbs.

---

## Install

```bash
pip install saklas             # everything needed to run it
pip install saklas[gguf]       # adds the gguf package for llama.cpp interchange
pip install saklas[research]   # adds datasets and pandas for dataset loading and DataFrames
pip install saklas[notebook]   # adds plotly, pandas, and kaleido for Jupyter figure helpers
pip install saklas[sae]        # adds sae-lens for SAE-backed extraction
```

This requires Python 3.11+ and PyTorch 2.2+. It should run on Linux, macOS, and Windows. CUDA or Apple Silicon MPS is recommended for anything interactive.

From source:

```bash
git clone https://github.com/a9lim/saklas
cd saklas
pip install -e ".[dev]"        # + pytest
```

---

## How it works

### Steering vectors

Saklas takes any concept, and generates pairs of sentences that best represent it. It then runs them through the model, and averages the differences of the hidden states at each layer; this gives you the vector representing the concept. When it's time to generate text, saklas rotates the residual stream toward the concept direction at every layer, weighted so that α=1 means full alignment.

### Built-in probe library

There are 26 default probes across 7 categories. Each probe is built from only 45 contrastive pairs generated using the program's pipeline.

| Category | Probes |
|---|---|
| **Affect** | angry.calm, happy.sad, fearful.unflinching |
| **Epistemic** | confident.uncertain, honest.deceptive, hallucinating.grounded, curious.disinterested |
| **Alignment** | agentic, refusal.compliant, sycophantic.blunt, manipulative |
| **Register** | formal.casual, direct.indirect, verbose.concise, creative.conventional, humorous.serious, warm.clinical, technical.accessible |
| **Social stance** | authoritative.submissive, high_context.low_context, self.other |
| **Cultural** | masculine.feminine, religious.secular, traditional.progressive, individualist.collectivist |
| **Identity** | ai.human |

Poles are aliased for easy use: `/steer angry 0.5` → `angry.calm` at α = +0.5. `/steer calm 0.5` → `angry.calm` at α = −0.5. This works for any installed bipolar pack.

### Triggers

By default steering fires on every token. The grammar's `@trigger` token attaches a per-term override:

```python
# Steer only the response, never the prompt or the thinking section
session.generate("...", steering="0.4 warm@after")

# Mix regimes per concept
session.generate("...", steering="0.3 honest + 0.4 warm@after")

# Projection: steer honest with sycophancy removed
session.generate("...", steering="0.3 honest|sycophantic")
```

The grammar tokens `@both`, `@response`, `@before`, `@after`, and `@thinking` let you choose when steering is applied. `Trigger.first(n)` and `Trigger.after(n)` let you express token-window ranges. If you want arbitrary combinations, you should pass a pre-built `Steering`.

### Ablation

Prefix a concept with `!` to ablate it: at every covered layer, the component of the residual stream along the concept's direction is replaced with the baseline mean.

### SAE-backed extraction (experimental)

> **Experimental** This pipeline is not as tested as the raw extraction path. 

Install `saklas[sae]` and pass `--sae <release>` to `vector extract` to run extraction in feature space. Saklas routes through SAELens, so any published release it covers (GemmaScope, Eleuther Meta-LLaMA-3.1 SAEs, Joseph Bloom's, Apollo, Goodfire) should be supported.

### Cross-model probe transfer

Probes are extracted per (model, concept). To use a probe extracted on one model with a different model, run `saklas vector transfer --from SRC --to TGT NAME`. Transferred profiles can coexist with native ones: use `/steer 0.3 angry:from-google__gemma-4-31b-it` to select the transferred variant explicitly if both exist.

---

## Web UI

```bash
pip install saklas
saklas serve google/gemma-4-31b-it
```

Open `http://localhost:8000/`. 

You can send messages in four ways. Hitting enter usually commits your message as a user node and triggers the model to generate, or while the selected node is a user node, prefills the model's turn and then has it generate from that. `Ctrl+Enter` (or `Alt+Enter`) usually commits your message without running the model, or when the selected node is a user node, lets you submit the model's turn for it. Submissions during generation get queued.

You can select a probe (or `surprise (logprob)`) to color tokens by score live; you can compare two probes at the same time as well. All generated tokens are clickable; clicking displays all probe scores at all layers for that one token. You can also see top-token alternatives the model considered in the menu. The activation atlas extends this across the whole conversation.

---

## Terminal UI

```bash
saklas tui google/gemma-2-9b-it
saklas tui mistralai/Mistral-7B-Instruct-v0.3 -q 4bit
saklas tui meta-llama/Llama-3.1-8B-Instruct -p affect register
```

### Flags

| Flag | Description |
|---|---|
| `model` | HuggingFace ID or local path (optional if supplied by `-c`) |
| `-q`, `--quantize` | `4bit` or `8bit` (CUDA only) |
| `-d`, `--device` | `auto` (default), `cuda`, `mps`, `cpu` |
| `-p`, `--probes` | Categories: `all`, `none`, `affect`, `epistemic`, `alignment`, `register`, `social_stance`, `cultural` |
| `--steer-mode` | `angular` (default) or `additive` (legacy v1.x add-and-rescale path) |
| `--theta-max` | Max rotation angle for angular mode (radians; default π/2 ≈ 1.5708) |
| `--legacy` | v2.0 backcompat preset: PCA extraction with additive injection. Mutually exclusive with `--steer-mode`. |
| `-c`, `--config` | Load setup YAML |
| `-s`, `--strict` | With `-c`: fail on missing vectors |

### Keybindings

| Key | Action |
|---|---|
| `Tab` / `Shift+Tab` | Cycle panel focus |
| `Left` / `Right` | Adjust alpha finely |
| `Shift+Left` / `Shift+Right` | Adjust alpha coarsely |
| `Up` / `Down` | Navigate vectors or probes |
| `Enter` | Toggle vector on/off |
| `Backspace` / `Delete` | Remove selected vector or probe |
| `Ctrl+T` | Toggle thinking mode |
| `Ctrl+A` | Toggle auto-regen side-by-side comparison |
| `Ctrl+R` | Regenerate last response |
| `Ctrl+S` | Cycle trait sort mode |
| `Ctrl+Y` / `Ctrl+Shift+Y` | Cycle per-token highlight: off → probe → surprise |
| `Ctrl+L` | Open the loom tree screen |
| `Ctrl+E` / `Ctrl+B` | Edit or branch the active loom node |
| `Ctrl+N` / `Ctrl+D` | Navigate by prefix or request guarded subtree delete |
| `[` / `]` | Adjust temperature |
| `{` / `}` | Adjust top-p |
| `Escape` | Stop generation |
| `Ctrl+Q` | Quit |

### Chat commands

| Command | Description |
|---|---|
| `/steer <expression>` | Apply a steering expression (grammar: `0.5 honest + 0.3 warm@after`, `0.5 honest:sae`, `0.5 a\|b`, …) |
| `/alpha <val> <name>` | Adjust an already-registered vector's alpha |
| `/unsteer <name>` | Remove a registered vector |
| `/probe <name>` | Extract and register a probe vector |
| `/probe <pos> . <neg>` | Same, bipolar form |
| `/unprobe <name>` | Remove a registered probe vector |
| `/compare <a> [b]` | Cosine similarity (1-arg: ranked vs all; 2-arg: pairwise) |
| `/extract <name>` | Extract to disk without registering |
| `/extract <pos> . <neg>` | Same, bipolar form (only path for new bipolar extraction) |
| `/regen [N] [mode]` | Regenerate the last assistant turn, optionally as N siblings or with a recipe override |
| `/fan <vector> <alphas>` | Generate an alpha grid as loom siblings |
| `/auto-regen [on\|off\|mode]` | Configure the side-by-side comparison modifier |
| `/tree` | Open the loom tree screen |
| `/edit <text>` / `/branch [text]` | Mutate or branch the active loom node |
| `/nav <prefix>` / `/del yes` | Navigate by node prefix or delete the active subtree after confirmation |
| `/prune <filter-expr>` | Dim nonmatching loom nodes by aggregate probe readings |
| `/diff <id1> <id2> [--full]` | Compare branch text and reading deltas |
| `/diff --siblings` | Compare assistant siblings under the active user parent |
| `/clear` | Clear conversation history |
| `/rewind` | Undo last exchange |
| `/sys <prompt>` | Set system prompt |
| `/temp <v>` / `/top-p <v>` / `/max <n>` | Sampling defaults |
| `/seed [n\|clear]` | Default sampling seed |
| `/save <name>` / `/load <name>` | Save/restore the full loom conversation tree |
| `/export <path>` | JSONL with per-token probe readings |
| `/model` | Model + device + active state |
| `/help` | List commands and keybindings |

If you want to extract a vector for two poles, use `/extract a dog . a pair of cats`. The TUI parses around the space-period-space delimiter. `dog.cat` stays a single name.

---

## Python API

```python
from saklas import SaklasSession, SamplingConfig, Steering, Profile, DataSource, ResultCollector

with SaklasSession.from_pretrained("google/gemma-3-4b-it", device="auto") as session:
    name, profile = session.extract("angry.calm")   # bundled bipolar pack; returns Profile
    session.steer(name, profile)                    # register (no alpha yet)

    result = session.generate(
        "What makes a good day?",
        steering=f"0.2 {name}",
        sampling=SamplingConfig(temperature=0.7, max_tokens=256, seed=42),
    )
    # generate() returns a RunSet. Single-run attributes delegate to .first.
    print(result.text)
    print(result.readings)                          # live probe readings
    print(result.applied_steering)                  # canonical expression receipt

    # Scoped steering with pole resolution
    with session.steering("0.4 calm"):              # bare pole → angry.calm @ -0.4
        print(session.generate("Describe a rainy afternoon.").text)

    # Compare vectors
    other_name, other_profile = session.extract("happy.sad")
    print(profile.cosine_similarity(other_profile))                  # aggregate
    print(profile.cosine_similarity(other_profile, per_layer=True))  # per-layer

    # Alpha sweep
    collector = ResultCollector()
    for alpha in [-0.2, -0.1, 0, 0.1, 0.2]:
        session.clear_history()
        r = session.generate("Describe a sunset.", steering=f"{alpha} {name}")
        collector.add(r.first, alpha=alpha)
    collector.to_csv("sweep.csv")
```

Registration is state and steering is per-call. `session.steer("name", profile)` stores the vector; `session.generate(input, steering="0.5 name")` applies it for that generation. Without `steering` you get a clean baseline.

You can compose concepts with `+`, `-`, `@trigger`, `|`, or `~`. Every surface (Python, YAML, HTTP, TUI, `vector merge`) parses the same expression language. Nested `with session.steering(...)` blocks get flattened, inner wins on key collision.

Sampling is per-call via `SamplingConfig`: `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs`.

Thinking mode auto-detects for models that support it (Qwen 3.5, Gemma 4, gpt-oss, etc). The delimiters are detected from the chat template.

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
result = session.generate("prompt", steering="0.5 name",
                          sampling=SamplingConfig(temperature=0.8))
for tok in session.generate_stream("prompt", steering="0.5 name"):
    print(tok.text, end="", flush=True)

# Scoped steering
with session.steering("0.5 wolf"):        # -> deer.wolf @ -0.5
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
result.vectors           # {"angry.calm": 0.2}: alphas snapshot
result.readings          # {"probe_name": ProbeReadings}
result.to_dict()
```

---

## Notebook helpers

```bash
pip install saklas[notebook]
```

```python
from saklas import SaklasSession, ResultCollector
from saklas.notebook import plot_alpha_sweep, plot_probe_correlation, plot_layer_norms, plot_trait_history

with SaklasSession.from_pretrained("google/gemma-3-4b-it") as s:
    results = s.generate_sweep("Describe a sunset.", sweep={"happy.sad": [-0.4, 0.0, 0.4]})
    plot_alpha_sweep(results.to_collector()).show()
```

Four plotly figure builders: `plot_alpha_sweep`, `plot_probe_correlation`, `plot_layer_norms`, and `plot_trait_history`. Each accepts the structured types saklas already returns (`ResultCollector`, `dict[str, Profile]`, `Profile`, `dict[str, ProbeReadings]`) and returns a plotly `Figure` you can render inline, export to HTML via `.write_html()`, or save as PNG via `.write_image()`. The `to_dataframe(...)` helper coerces results into pandas DataFrames for ad-hoc analysis.

## Batched generation

```python
results = session.generate_batch(["What's a good day?", "Describe a sunset.", "Tell me a joke."], steering="0.4 cheerful")
sweep = session.generate_sweep("Describe a rainy day.", sweep={"happy.sad": [-0.4, 0.0, 0.4]})
```

Both return `RunSet`: an ordered, list-like result set with `node_ids`, `grid`, `.first`, `.to_collector()`, and `.to_dataframe()`. The native HTTP route for the same shape is `POST /saklas/v1/sessions/{id}/experiments/fan`, with body `{prompt, grid, base_steering?, sampling?, thinking?, raw?}`.

## API server

`saklas serve` supports both OpenAI `/v1/*` and Ollama `/api/*` on the same port. It should work with the OpenAI Python and JS SDKs, LangChain, Open WebUI, Enchanted, Msty, `ollama-python`, and anything else that talks either wire format.

```bash
pip install saklas
saklas serve google/gemma-2-9b-it --steer "0.2 cheerful" --port 8000
```

### OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

resp = client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"steering": "0.4 cheerful"},       # per-request expression
)
```

### Ollama

Point any Ollama client at `http://localhost:8000` and it should work. Steering goes through the `steer` field in `options`:

```bash
curl -N http://localhost:8000/api/chat -d '{
  "model": "gemma2",
  "messages": [{"role": "user", "content": "Write me a haiku."}],
  "options": {"steer": "0.3 cheerful - 0.2 formal.casual"}
}'
```

### Saklas-native routes

`/saklas/v1/*` is a resource tree with sessions, vector and probe management, one-shot probe scoring, a bidirectional WebSocket for token plus probe co-streaming, and a live traits SSE endpoint (`GET /saklas/v1/sessions/{id}/traits/stream`) that streams per-token probe scores in real time during any active generation. Interactive docs at `http://localhost:8000/docs`.

### Flags

| Flag | Default | Description |
|---|---|---|
| `model` | required | HuggingFace ID or local path |
| `-H`, `--host` | `0.0.0.0` | Bind address |
| `-P`, `--port` | `8000` | Bind port |
| `-S`, `--steer` | None | Default steering expression, e.g. `"0.2 cheerful"` |
| `--steer-mode` | `angular` | `angular` (default) or `additive` (legacy v1.x path) |
| `--theta-max` | `π/2` | Max rotation angle for angular mode (radians) |
| `--legacy` | off | v2.0 backcompat preset: PCA extraction with additive injection. Mutually exclusive with `--steer-mode`. |
| `-C`, `--cors` | None | CORS origin, repeatable |
| `-k`, `--api-key` | None | Bearer auth. Falls back to `$SAKLAS_API_KEY`. |

Not supported: tool calling, strict JSON mode, embeddings. The server is designed for trusted networks, please see [SECURITY.md](SECURITY.md).

---

## Concept packs

All state lives under `~/.saklas/` (override via `SAKLAS_HOME`). Each concept is a folder with `pack.json`, `statements.json`, and per-model tensors. Packs are distributed as HuggingFace model repos.

Packless install handles repos with no `pack.json`, so repeng-style GGUF-only control-vector repos install cleanly: `saklas pack install jukofyork/creative-writing-control-vectors-v3.0`.

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
saklas vector extract <concept> | <pos> <neg> [-m MODEL] [-f] [--method dim|pca] [--sae RELEASE [--sae-revision REV]] [--legacy]
saklas vector merge <name> <expression> [-m] [-f] [-s]
saklas vector clone <corpus-file> -N NAME [-m MODEL] [-n N_PAIRS] [--seed S] [-f]
saklas vector compare <concepts...> -m MODEL [-v] [-j] [--metric euclidean|mahalanobis] [--legacy]
saklas vector why <concept> -m MODEL [-j]
```

`--method dim` (default) writes to `<concept>/<safe_model>.safetensors`; `--method pca` writes to `<concept>/<safe_model>_pca.safetensors` so the two can coexist on disk. Address either at steer time with the variant suffix: `0.3 honest` picks the canonical (DiM) tensor, `0.3 honest:pca` picks the legacy PCA tensor. The `--legacy` shorthand sets `--method pca` for v2.0 reproduction.

Merge expressions share the steering grammar. Terms combine with `+` or `-`, coefficients lead each term, `~` keeps the component aligned with another direction, and `|` projects another direction's component out. For example, `saklas vector merge dehallu "0.8 default/creative.conventional|default/hallucinating.grounded"` gives you creative with hallucination projected out.

Selectors: `<name>`, `<ns>/<name>`, `tag:<tag>`, `namespace:<ns>`, `default`, `all`. Bare names resolve across namespaces and error if ambiguous.

---

## Supported architectures

**Tested**: Qwen, Gemma, Ministral, gpt-oss, Llama, GLM, Talkie.

**Wired up but untested**: Mistral, Mixtral, Phi 1–3, PhiMoE, Cohere 1–2, DeepSeek V2–V3, StarCoder2, OLMo 1–3 plus OLMoE, Granite plus GraniteMoE, Nemotron, StableLM, GPT-2, GPT-Neo, GPT-J, GPT-BigCode, GPT-NeoX, Bloom, Falcon, Falcon-H1, MPT, DBRX, OPT, Recurrent Gemma.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for adding an architecture.

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

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup. For security, please see [SECURITY.md](SECURITY.md).

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

If you use Saklas in published research, please also cite the Representation Engineering paper (Zou et al., 2023) and [repeng](https://github.com/vgel/repeng).
