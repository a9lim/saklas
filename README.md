# saklas

[![CI](https://github.com/a9lim/saklas/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/saklas/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/saklas)](https://pypi.org/project/saklas/)
[![Downloads](https://img.shields.io/pypi/dm/saklas)](https://pypi.org/project/saklas/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/saklas/)

Saklas is a library for activation steering and trait monitoring on local HuggingFace models. You give it a concept, from "formal" to "a pirate" to "happy", and it generates contrastive examples, extracts a direction from the model's hidden states, and then steers along that direction when it's time to generate text. The model itself is never modified, so the steering strength is just a number you change per call.

Every steering signal in saklas is one artifact: the **manifold**, a set of labelled nodes fit to a per-layer subspace. A plain steering vector is the two-node case. A 107-persona fan and a 20-mood affect surface are the many-node case. The same `%` operator that picks a pole picks a point anywhere on a fitted surface, so you can blend between personas or slide along an emotional gradient with the same grammar you use for a single vector.

Saklas is built on Representation Engineering ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)), the same paper [repeng](https://github.com/vgel/repeng) implements. There are three frontends over one engine:

- **`saklas serve <model>`**: a web dashboard at `http://localhost:8000/`. The same port also speaks OpenAI `/v1/*` and Ollama `/api/*`, plus a native `/saklas/v1/*` API. Pass `--no-web` for API-only mode.
- **`saklas tui <model>`**: a terminal UI.
- **`SaklasSession`**: a Python API for scripted experiments.

It runs on both CUDA and Apple Silicon MPS, and it runs comfortably on a MacBook. Tested on Qwen, Gemma, Ministral, gpt-oss, Llama, GLM, and Talkie, with a lot more architectures experimentally wired up.

---

## Quick start

```bash
pip install saklas
saklas serve google/gemma-4-31b-it
```

Once it loads, open `http://localhost:8000/`. The first run downloads the model and fits the 17 bundled concept probes, which can take a few minutes.

The same port speaks the OpenAI API format:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(
    model="google/gemma-4-31b-it",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"steering": "0.4 warm"},
)
```

If you prefer a terminal:

```bash
saklas tui google/gemma-4-31b-it
```

Or directly from Python:

```python
from saklas import SaklasSession

with SaklasSession.from_pretrained("google/gemma-4-31b-it") as s:
    name, profile = s.extract("formal.casual")       # a bundled bipolar concept
    print(s.generate("What makes a good day?", steering=f"0.3 {name}").first.text)
```

---

## Reporting issues

If you hit an error, please update to the most recent version first. If it still happens, please open an issue. This project is a work in progress and I am actively finding and fixing bugs.

---

## Credits

The contrastive-pair approach comes from the Representation Engineering paper ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)). [repeng](https://github.com/vgel/repeng) by Theia Vogel is the well-known implementation in this space. Saklas takes the same idea from a different angle: repeng is lean and more of a library, saklas is a full workbench with monitoring, branching chat, and a server built in. Both are worth your time!

Extraction is difference-of-means per [Im & Li, 2025](https://arxiv.org/abs/2502.02716), which at two poles is exactly the leading principal component. The per-layer strength allocation runs in the Mahalanobis metric (whitened against per-model activation covariance) so the channels that dominate raw activation norms don't swamp the signal. Manifold steering follows [Goodfire's work](https://arxiv.org/abs/2605.05115).

---

## Install

```bash
pip install saklas                  # everything needed to run it
pip install saklas[gguf]            # adds the gguf package for llama.cpp interchange
pip install saklas[sae]             # adds SAELens for external SAE releases
pip install saklas[research]        # adds datasets and pandas for dataset loading and DataFrames
pip install saklas[notebook]        # adds plotly, pandas, and kaleido for Jupyter figure helpers
pip install saklas[cuda]            # adds bitsandbytes and HF kernels for CUDA acceleration
pip install saklas[cuda-experimental]  # the above plus flash-attn
```

This needs Python 3.11+ and PyTorch. It should run on Linux, macOS, and Windows. CUDA or Apple Silicon MPS is recommended for anything interactive.

From source:

```bash
git clone https://github.com/a9lim/saklas
cd saklas
pip install -e ".[dev]"             # plus pytest
```

---

## How it works

### Steering vectors

Saklas takes a concept and generates contrastive examples. For a bipolar concept like `formal.casual`, it has the model answer a fixed set of baseline prompts in character as each pole. It runs those through the model, pools the hidden state at the last content token of every layer, and subtracts the two poles. That difference-of-means direction, which is the leading principal component when there are exactly two poles, is the steering vector. The per-layer strength is allocated in a whitened (Mahalanobis) metric. When it's time to generate, saklas hooks each layer and slides the hidden state's component inside the concept's subspace toward the concept, leaving the rest of the hidden state alone. The coefficient is per-call, and the model itself is never touched.

### The manifold

A steering vector is the two-node case of a manifold: two labelled nodes (the poles) fit to a per-layer subspace. Add more nodes and you get a surface you can place a generation anywhere on. The `%` operator does the placing. `0.5 formal` slides toward the `formal` pole; `0.5 personas%pirate` slides toward the pirate node of the 107-persona fan; `emotions%0.3,0.8,0.0` places the generation at a point in the pleasure-arousal-dominance affect space. Everything lowers to the same per-layer subspace injection at generation time, so a vector and several manifolds compose with no cross-talk.

### Bundled concepts

There are 17 bundled bipolar concepts across 4 categories, plus two many-node manifolds.

| Category | Concepts |
|---|---|
| **Epistemic** | confident.uncertain, honest.deceptive, curious.disinterested |
| **Alignment** | refusing.compliant, sycophantic.blunt, sincere.manipulative |
| **Register** | formal.casual, direct.indirect, verbose.concise, creative.conventional, humorous.serious, warm.clinical, technical.accessible |
| **Cultural** | masculine.feminine, individualist.collectivist, traditional.progressive, religious.secular |

Each bipolar concept has two poles, and either pole is a name you can steer toward. `0.5 formal` is `formal.casual` at +0.5, `0.5 casual` is the same concept at −0.5. This works for any installed bipolar concept.

The two many-node manifolds are **`personas`** (107 archetype nodes from `assistant` to `vandal`, from Anthropic's [Assistant Axis paper](https://arxiv.org/abs/2601.10387)) and **`emotions`** (20 mood nodes over pleasure-arousal-dominance affect space). These ship without per-model tensors because they are heavy to fit, so they attach as probes only once you have fit them for your model.

### The steering grammar

Every live steering surface (Python, YAML, HTTP, the TUI) speaks the same grammar.
`manifold bake` accepts its namespace-qualified additive scalar subset.

```python
session.generate("...", steering="0.3 honest + 0.4 warm")   # add two concepts
session.generate("...", steering="0.5 formal - 0.2 verbose") # subtract one
session.generate("...", steering="0.3 honest|sycophantic")   # honest with sycophancy projected out
session.generate("...", steering="0.3 honest~confident")     # keep only the shared component
session.generate("...", steering="!sycophantic")             # ablate sycophancy
session.generate("...", steering="0.5 personas%pirate")      # place on a manifold node
```

`+` and `-` combine terms, a leading number is the coefficient (default 0.5), `~` projects onto a direction, `|` projects orthogonal to it, `!` ablates a concept by replacing its component with the baseline mean, and `%` places a generation at a point on a manifold.

### Triggers

By default steering fires on every token. The `@trigger` token attaches a per-term override:

```python
# Steer only the response, never the prompt or the thinking section
session.generate("...", steering="0.4 warm@after")

# Mix regimes per concept
session.generate("...", steering="0.3 honest + 0.4 warm@after")
```

`@both`, `@response`, `@before`, `@after`, `@thinking`, `@prompt`, and `@generated` choose when steering applies. A probe gate fires steering only on the tokens where a live reading crosses a threshold: `0.4 warm@when:confident.uncertain>0.4` steers warmth only while the confidence probe reads above 0.4.

### Manifold steering

A straight line between two states cuts through regions of activation space the model never actually visits, so the intermediate generations come out garbled or collapse to sameness. When you want to steer through a sequence of related states, like an emotional gradient or a blend between two personas, manifold steering fits a smooth surface through the activation centroids of the labelled nodes and steers along it instead.

```python
# Slide toward the pirate persona at strength 0.5
session.generate("Describe a forest.", steering="0.5 personas%pirate")

# A point in pleasure-arousal-dominance affect space, response only
session.generate("How was your day?", steering="0.7 emotions%0.3,0.8,0.0@response")
```

The `%` coefficient slot is `along[,onto]`: `along` is how far to slide toward the position, and `onto` (curved surfaces only) collapses the off-surface component toward the learned tube. `saklas experiment naturalness` scores how far a steered run drifts off the model's natural behavior, so you can check a manifold run against a plain linear one.

### SAE-backed extraction (experimental)

> **Experimental.** This path is less tested than raw extraction.

Pass `--sae <source>` to `manifold extract` or `manifold fit` to run extraction in feature space. `saelens:<release>` uses a published SAELens release (and requires `saklas[sae]`); `local:<name>` uses an SAE trained by Saklas. The fitted subspace is always model-space, so the generation hook never touches the SAE.

### Cross-model transfer

Manifolds are fit per (model, concept). To use one fit on one model with a different model, run `saklas manifold transfer <name> --from SRC --to TGT`, which Procrustes-aligns the source fit into the target's activation space. Transferred fits coexist with native ones: address the transferred variant explicitly with the `:from-<safe_src>` suffix if both exist.

---

## Web UI

```bash
saklas serve google/gemma-4-31b-it
```

Open `http://localhost:8000/`.

The chat is a branching loom tree, so any turn can fork into siblings. Hitting Enter commits your message as a user node and runs the model, or, when the selected node is a user node, prefills the model's turn and generates from there. Holding Ctrl, Cmd, or Option while you submit commits without running the model. Submissions during generation get queued.

You can select a probe (or `surprise (logprob)`) to color tokens by score live, and you can compare two probes at once. Every generated token is clickable, and clicking shows all probe scores at all layers for that token, plus the top alternatives the model considered. The probe inspector renders each probe's geometry in the whitened metric, and the activation atlas extends the per-token view across the whole conversation.

---

## Terminal UI

```bash
saklas tui google/gemma-2-9b-it
saklas tui mistralai/Mistral-7B-Instruct-v0.3 -q 4bit
saklas tui meta-llama/Llama-3.1-8B-Instruct -p epistemic register
```

### Flags

| Flag | Description |
|---|---|
| `model` | HuggingFace ID or local path (optional if supplied by `-c`) |
| `-q`, `--quantize` | `4bit` or `8bit` (CUDA only) |
| `-d`, `--device` | `auto` (default), `cuda`, `mps`, `cpu` |
| `-p`, `--probes` | Categories: `all`, `none`, `epistemic`, `alignment`, `register`, `cultural` |
| `--max-tokens` | Default max generation tokens (default 1024) |
| `--no-dls` | Disable discriminative layer selection at extraction time |
| `-c`, `--config` | Load setup YAML (repeatable, later overrides earlier) |
| `-s`, `--strict` | With `-c`: fail on missing concepts |

### Keybindings

| Key | Action |
|---|---|
| `Tab` / `Shift+Tab` | Cycle panel focus |
| `Left` / `Right` | Adjust alpha finely |
| `Shift+Left` / `Shift+Right` | Adjust alpha coarsely |
| `Up` / `Down` | Navigate vectors or probes |
| `Enter` | Toggle vector on or off |
| `Backspace` / `Delete` | Remove selected vector or probe |
| `Ctrl+T` | Toggle thinking mode |
| `Ctrl+A` | Toggle auto-regen side-by-side comparison |
| `Ctrl+R` | Regenerate last response |
| `Ctrl+S` | Cycle trait sort mode |
| `Ctrl+Y` / `Ctrl+Shift+Y` | Cycle per-token highlight: off, probe, surprise |
| `Ctrl+O` | Toggle chat and raw render mode |
| `Ctrl+L` | Open the loom tree screen |
| `Ctrl+E` / `Ctrl+B` | Edit or branch the active loom node |
| `Ctrl+N` / `Ctrl+D` | Navigate by prefix or request guarded subtree delete |
| `Ctrl+Enter` / `Alt+Enter` | Commit without generating |
| `[` / `]` | Adjust temperature |
| `{` / `}` | Adjust top-p |
| `Escape` | Stop generation |
| `Ctrl+Q` | Quit |

### Chat commands

| Command | Description |
|---|---|
| `/steer <expression>` | Apply a steering expression (`0.5 honest + 0.3 warm@after`, `0.5 personas%pirate`, `0.5 honest:sae`, …) |
| `/alpha <val> <name>` | Adjust an already-registered vector's alpha |
| `/unsteer <name>` | Remove a registered vector |
| `/probe <name>` | Extract and attach a concept probe |
| `/probe <pos> . <neg>` | Same, bipolar form |
| `/unprobe <name>` | Detach a probe |
| `/manifold-probe <selector>` | Attach a many-node manifold as a probe (`emotions`, `personas`, `ns/name`) |
| `/manifold fit <folder>` | Fit an authored manifold pack |
| `/extract <name>` / `/extract <pos> . <neg>` | Extract to disk without attaching (`--role <slug>` for a persona-baselined fit) |
| `/pairs <name>` | Extract from hand-written `positive \| negative` pairs |
| `/compare <a> [b]` | Cosine similarity (1-arg: ranked vs all; 2-arg: pairwise) |
| `/regen [N] [mode]` | Regenerate the last assistant turn, optionally as N siblings or with a recipe override |
| `/fan <vector> <alphas>` | Generate an alpha grid as loom siblings |
| `/auto-regen [mode]` | Configure the side-by-side comparison modifier |
| `/tree` | Open the loom tree screen |
| `/edit <text>` / `/branch [text]` | Mutate or branch the active loom node |
| `/nav <prefix>` / `/del [yes]` | Navigate by node prefix or delete the active subtree after confirmation |
| `/star` / `/note <text>` / `/path` | Mark, annotate, or print the active node's path |
| `/prune <filter-expr>` | Dim nonmatching loom nodes by aggregate probe readings |
| `/diff <id1> <id2> [--full]` / `/diff --siblings` | Compare branch text and reading deltas |
| `/commit <text>` | Commit a turn without generating |
| `/clear` / `/rewind` | Clear history or undo the last exchange |
| `/sys <prompt>` | Set system prompt |
| `/temp <v>` / `/top-p <v>` / `/max <n>` / `/seed [n\|clear]` | Sampling defaults |
| `/save <name>` / `/load <name>` | Save or restore the full loom conversation tree |
| `/export <path>` | JSONL with per-token probe readings |
| `/model` / `/help` | Model and device state, or list commands and keybindings |

To extract a concept from two poles, use `/extract a dog . a pair of cats`. The TUI parses around the space-period-space delimiter, so `dog.cat` stays a single name.

---

## Python API

```python
from saklas import SaklasSession, SamplingConfig, Steering, Profile, ResultCollector

with SaklasSession.from_pretrained("google/gemma-3-4b-it", device="auto") as session:
    name, profile = session.extract("honest.deceptive")   # returns (canonical_name, Profile)

    result = session.generate(
        "What makes a good day?",
        steering=f"0.3 {name}",
        sampling=SamplingConfig(temperature=0.7, max_tokens=256, seed=42),
    ).first
    print(result.text)
    print(result.readings)                          # live probe readings
    print(result.applied_steering)                  # canonical expression receipt

    # Scoped steering with pole and manifold-label resolution
    with session.steering("0.4 casual"):            # bare pole resolves to formal.casual at -0.4
        print(session.generate("Describe a rainy afternoon.").first.text)
    with session.steering("0.5 pirate"):            # bare label resolves to personas%pirate
        print(session.generate("Describe a forest.").first.text)

    # Compare concepts
    other_name, other_profile = session.extract("warm.clinical")
    print(profile.cosine_similarity(other_profile))                  # aggregate
    print(profile.cosine_similarity(other_profile, per_layer=True))  # per-layer

    # Alpha sweep
    results = session.generate_sweep("Describe a sunset.", sweep={name: [-0.3, 0.0, 0.3]})
    results.to_collector().to_csv("sweep.csv")
```

Steering is per-call. `session.generate(prompt, steering="0.5 name")` applies it for that one generation; without `steering` you get a clean baseline. If you want to register a profile and toggle it across calls, `session.steer(name, profile)` stores it and `session.unsteer(name)` removes it.

`generate`, `generate_stream`, and `session.steering()` accept `str | Steering | None` only. A string is a steering expression; a dict raises. You can compose concepts with `+`, `-`, `@trigger`, `|`, `~`, `!`, and `%`. Nested `with session.steering(...)` blocks compose, and the inner scope wins on a key collision.

Sampling is per-call via `SamplingConfig`: `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs`, `return_hidden`.

Thinking mode auto-detects for models that support it (Qwen 3.5, Gemma 4, gpt-oss, and others). The delimiters come from the chat template.

`session.events` is a synchronous `EventBus`. Subscribe to `VectorExtracted`, `SteeringApplied`, `SteeringCleared`, `ProbeScored`, `GenerationStarted`, and `GenerationFinished`.

### SaklasSession reference

```python
session = SaklasSession.from_pretrained(
    model_id, device="auto", quantize=None, probes=None,
    system_prompt=None, max_tokens=1024,
)

# Extraction
name, profile = session.extract("curiosity")                  # monopolar, against the neutral mean
name, profile = session.extract("honest.deceptive")           # bundled bipolar
name, profile = session.extract("warm", baseline="clinical")  # explicit poles
name, profile = session.extract_vector_from_corpora(positives, negatives)  # hand-written corpora

# Manifold fitting
manifold = session.fit("manifolds/local/mood")   # fit a multi-node manifold, returns a Manifold

# Registry (optional; steering is usually per-call)
session.steer("name", profile)
session.unsteer("name")

# Generation
result = session.generate("prompt", steering="0.5 name",
                          sampling=SamplingConfig(temperature=0.8))
for tok in session.generate_stream("prompt", steering="0.5 name"):
    print(tok.text, end="", flush=True)

# Probes
session.add_probe("honest.deceptive")
session.add_probe("emotions")            # a many-node manifold attaches whole
session.remove_probe("honest.deceptive")

# Restricted-choice scoring (the logit read, steering-aware)
scores = session.score_choices(messages, ["Monday", "Tuesday"], steering="0.5 confident")

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
result.readings          # {"probe_name": ProbeReadings}
result.applied_steering  # canonical expression receipt
result.to_dict()
```

---

## Notebook helpers

```bash
pip install saklas[notebook]
```

```python
from saklas import SaklasSession
from saklas.notebook import plot_alpha_sweep, plot_probe_correlation, plot_layer_norms, plot_trait_history

with SaklasSession.from_pretrained("google/gemma-3-4b-it") as s:
    results = s.generate_sweep("Describe a sunset.", sweep={"warm.clinical": [-0.4, 0.0, 0.4]})
    plot_alpha_sweep(results.to_collector()).show()
```

There are four plotly figure builders: `plot_alpha_sweep`, `plot_probe_correlation`, `plot_layer_norms`, and `plot_trait_history`. Each takes the structured types saklas already returns and gives back a plotly `Figure` you can render inline, export to HTML with `.write_html()`, or save as PNG with `.write_image()`. The `to_dataframe(...)` helper coerces results into pandas DataFrames for ad-hoc analysis.

---

## Batched generation

```python
results = session.generate_batch(["What's a good day?", "Describe a sunset.", "Tell me a joke."], steering="0.4 warm")
sweep = session.generate_sweep("Describe a rainy day.", sweep={"warm.clinical": [-0.4, 0.0, 0.4]})
```

Both return a `RunSet`: an ordered, list-like result set with `node_ids`, `grid`, `.first`, `.to_collector()`, and `.to_dataframe()`. The native HTTP route for the same shape is `POST /saklas/v1/sessions/{id}/experiments/fan`, with body `{prompt, grid, base_steering?, sampling?, thinking?, raw?}`.

---

## API server

`saklas serve` speaks OpenAI `/v1/*` and Ollama `/api/*` on the same port. It should work with the OpenAI Python and JS SDKs, LangChain, Open WebUI, Enchanted, Msty, `ollama-python`, and anything else that talks either wire format.

```bash
saklas serve google/gemma-2-9b-it --steer "0.2 warm" --port 8000
```

### OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

resp = client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"steering": "0.4 warm"},       # per-request expression
)
```

### Ollama

Point any Ollama client at `http://localhost:8000` and it should work. Steering goes through the `steer` field in `options`:

```bash
curl -N http://localhost:8000/api/chat -d '{
  "model": "gemma2",
  "messages": [{"role": "user", "content": "Write me a haiku."}],
  "options": {"steer": "0.3 warm - 0.2 formal.casual"}
}'
```

### Saklas-native routes

`/saklas/v1/*` is a resource tree with sessions, steering and probe management, manifold install and browse, restricted-choice scoring, a bidirectional WebSocket for token-plus-probe co-streaming, and a live traits SSE endpoint (`GET /saklas/v1/sessions/{id}/traits/stream`) that streams per-token probe scores during any active generation. Interactive docs at `http://localhost:8000/docs`.

### Flags

| Flag | Default | Description |
|---|---|---|
| `model` | required | HuggingFace ID or local path |
| `-H`, `--host` | `0.0.0.0` | Bind address |
| `-P`, `--port` | `8000` | Bind port |
| `-S`, `--steer` | None | Default steering expression, e.g. `"0.2 warm"` |
| `-C`, `--cors` | None | CORS origin, repeatable |
| `-k`, `--api-key` | None | Bearer auth; falls back to `$SAKLAS_API_KEY` |
| `--no-web` | off | Skip the dashboard mount, API-only mode |

It does not do tool calling, strict JSON mode, or embeddings. The server is meant for trusted networks. Please see [SECURITY.md](SECURITY.md).

---

## Concept packs

All state lives under `~/.saklas/` (override with `$SAKLAS_HOME`). Each manifold is a folder under `~/.saklas/manifolds/<ns>/<name>/` with a `manifold.json`, per-node corpus files, and per-model fitted tensors. Packs are distributed as HuggingFace model repos.

Packless install handles repos with no manifest, so repeng-style GGUF-only control-vector repos install cleanly:

```bash
saklas pack install jukofyork/creative-writing-control-vectors-v3.0
```

### Pack lifecycle and distribution

```bash
saklas pack ls [-v|-j]                                   # list installed manifolds
saklas pack show <name> [-j]                             # inspect one
saklas pack install <target> [-a NS/NAME] [-f]          # HF coord or local folder
saklas pack search <query> [-j|-v]                       # search HF for saklas-manifold repos
saklas pack push <name> [-a OWNER/NAME] [-m MODEL] [--variant raw|sae|all]
saklas pack refresh <name> [-m MODEL]                    # re-pull or re-fit
saklas pack clear <name> [-m MODEL] [--variant raw|sae|all]   # delete fitted tensors
saklas pack rm <name> [-y]                               # remove the folder
saklas pack export gguf <name> [-m MODEL] [-o PATH] [--model-hint HINT]
```

### Manifold operations

```bash
saklas manifold extract <concept> | <pos> <neg> [-m MODEL] [--sae RELEASE] [--role SLUG] [-f]
saklas manifold generate <name> --concepts C... [--kind abstract|concrete]
saklas manifold from-template <template> [--name MANIFOLD] [--fit-mode auto|pca|spectral]
saklas manifold fit <name> | <folder> [-m MODEL] [--sae REL] [--method pca|spectral|auto]
saklas manifold bake <name> <expression> [-m MODEL]
saklas manifold merge <name> <src...> [-f]
saklas manifold transfer <name> --from SRC --to TGT [-f]
saklas manifold compare <concepts...> -m MODEL
saklas manifold why <concept> -m MODEL [-j]
```

`manifold extract` writes a 2-node concept; `manifold bake` lands a corpus-less manifold from namespace-qualified additive/subtractive scalar terms, for example `saklas manifold bake balanced "0.8 default/honest.deceptive - 0.2 default/sycophantic.blunt"`. Dynamic terms and `~`/`|` projection are rejected offline because live projection is Mahalanobis-only and requires the loaded model's whitener.

Selectors are shared across surfaces: `<name>`, `<ns>/<name>`, `tag:<tag>`, `namespace:<ns>`, `model:<m>`, `default`, `all`, optionally suffixed with a variant (`:raw`, `:sae`, `:sae-<release>`, `:from-<safe_src>`, `:role-<slug>`). Bare names resolve across namespaces and error if ambiguous.

---

## Jacobian lens

The Jacobian lens ([Gurnee et al. 2026](https://transformer-circuits.pub/2026/workspace/index.html)) reads out what an intermediate activation is disposed to make the model *say*: a per-layer matrix transports the residual into the final-layer basis, and the model's own unembedding decodes it into ranked vocabulary tokens. The paper shows these verbalizable directions form the model's global workspace — the small subspace its flexible reasoning routes through.

Fetch an official Neuronpedia lens when the model is supported, or fit one
locally, then read, steer, gate, and decompose:

```bash
saklas lens fetch google/gemma-3-4b-it                    # provider payload stays in HF cache
# unsupported model:
saklas lens fit org/model --prompts 100                   # local/default; resumes if interrupted
saklas lens ls google/gemma-3-4b-it
saklas lens use google/gemma-3-4b-it neuronpedia
saklas lens top google/gemma-3-4b-it "Fact: the currency used in the country shaped like a boot is"
saklas lens decompose confident.uncertain -m google/gemma-3-4b-it   # how verbalizable is a concept vector?
saklas serve google/gemma-3-4b-it -S "!jlens/fake"        # ablate a lens direction
```

`jlens/<word>` works anywhere a concept does: `0.3 jlens/orange` steers toward a token's lens direction, `!jlens/fake` ablates it, and `@when:jlens/fake > 0.4` gates another term on it. Lens directions run hotter than concept vectors — start around 0.3 rather than 0.5. In the TUI, `/lens` streams the top workspace tokens per layer live while the model generates. The fit needs a web-text corpus — pass `--corpus FILE` or `pip install 'saklas[hf]'` to stream a default sample.

---

## Sparse autoencoders

Use a published SAELens release, or train a local residual-post SAE when the
model has no compatible release:

```bash
pip install 'saklas[sae]'
saklas sae fetch google/gemma-3-4b-it saelens:gemma-scope-2-4b-it-res

# Unsupported model: train on FineWeb-Edu, or pass --corpus FILE.
pip install 'saklas[hf]'
saklas sae train org/model my-sae --layer 20 --tokens 1000000
saklas sae use org/model local:my-sae
saklas sae ls org/model
saklas serve org/model
```

Saklas owns artifacts it computes: local lenses and SAEs live below
`$SAKLAS_HOME/models/<model>/jlens/local/` and `sae/local/`. External lens and
SAE weights remain in Hugging Face/SAELens caches; Saklas stores only a pinned
binding and the active-source selection. `lens rm` / `sae rm` therefore delete
local payloads or forget bindings, but never purge provider caches.

The dashboard SAE tab loads one release into the running session, streams its
top feature activations, and pins features as probes. `0.3 sae/9143` steers on
decoder row 9143; `!sae/9143` uses saklas's ordinary directional
mean-ablation; `@when:sae/9143 > 3` gates on the raw post-nonlinearity encoder
activation. V1 keeps one deterministic hook layer resident (nearest 65% depth,
workspace band preferred); pass `--layer` to `sae fetch` or `sae train` to select a different
covered layer. Feature labels are fetched lazily from Neuronpedia when the
selected SAELens registry entry provides an id, and reads remain offline-first.

---

## Supported architectures

**Tested**: Qwen, Gemma, Ministral, gpt-oss, Llama, GLM, Talkie.

**Wired up but untested**: Mistral, Mixtral, Phi 1 to 3, PhiMoE, Cohere 1 and 2, DeepSeek V2 and V3, StarCoder2, OLMo 1 to 3 plus OLMoE, Granite plus GraniteMoE, Nemotron, StableLM, GPT-2, GPT-Neo, GPT-J, GPT-BigCode, GPT-NeoX, Bloom, Falcon, Falcon-H1, MPT, DBRX, OPT, Recurrent Gemma.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for adding an architecture; it's usually a single accessor entry.

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

If you use saklas in published research, please also cite the Representation Engineering paper (Zou et al., 2023) and [repeng](https://github.com/vgel/repeng).
