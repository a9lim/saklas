# saklas

[![CI](https://github.com/a9lim/saklas/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/saklas/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/saklas)](https://pypi.org/project/saklas/)
[![Downloads](https://img.shields.io/pypi/dm/saklas)](https://pypi.org/project/saklas/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/saklas/)

Saklas is a local workbench for mechanistic interpretability on large language
models.

It comes with a local dashboard, along with a Python API, and a server compatible
with both OpenAI and Ollama.

## Quick start

```bash
pip install saklas
saklas serve google/gemma-3-4b-it
```

Open [http://localhost:8000](http://localhost:8000).

The first launch downloads the model and fits the 17 bundled concept probes. This
can take a while; they get stored in `~/.saklas/` for future launches.

For NVIDIA CUDA:

```bash
pip install "saklas[cuda,flash]"
saklas serve google/gemma-3-4b-it --device cuda
```

## WebUI

### Threads

Every authored or generated turn becomes a node in a branching tree. You can reroll
from any point and the tree lets you save and load conversations.

The bottom of the panel has the standard sampling controls: temperature, top-p,
top-k, repetition and presence penalties, and everything else you'd expect.

### Completion modes

Chat mode renders the model's template as turns with collapsible thinking. 
Roles are handled dynamically:

- **you write** selects the role the text you write will be appended as;
- **model writes** selects the role the model continues as (or `none` to just append);
- the cast button lets you assign steering to specific roles.

Compatible chat templates support arbitrary role labels beyond `user` and `assistant`.

Raw mode exposes one raw buffer for base models.

### Token-level inspection

Tokens can be highlighted by probe scores or **surprise (logprob)**.

Clicking on a token opens one detail drawer with four views:

- **geometry** — attached probe readings across layers;
- **logits** — the chosen token and captured alternatives with logprobs, plus
  token forking;
- **SAE** — sparse feature activations;
- **J-lens** — the aggregate workspace and layer-by-vocabulary readout.

The detail cursor can walk tokens, thinking/response segments, and turns without
closing the drawer. Historical rows use their captured measurements when present
and can replay the producing prefix for newly attached instrumentation.

Clicking on the info icon by a probe shows a fitted concept's geometry layer by layer.

### Instruments

The right side of the workbench has four tabs.

| Tab | Purpose |
|---|---|
| **Subspace** | Flat fitted subspaces |
| **Manifold** | Curved fitted surfaces |
| **SAE** | SAE features |
| **Lens** | Jacobian-lens workspace |

### Analysis tools

Press `Cmd+K` or `Ctrl+K` elsewhere to open a menu with further options:

- build, fit, merge, install, and inspect manifolds;
- author and score restricted-choice templates;
- manage the cast and open steering workflows;
- inspect correlations and pairwise layer geometry;
- check model, device, source, authentication, and server health;
- open the built-in help surface.

## Concepts, subspaces, and manifolds

In Saklas, you extract concepts as **manifolds** or **subspaces**.

- A 1D flat subspace is just a steering vector.
- A higher-rank flat subspace is a group of orthogonal steering vectors.
- A curved manifold fits a nonlinear surface and lets you steer along it.

### Bundled concepts

Saklas comes with 17 concept pairs that are attached as probes by default:

| Category | Concepts |
|---|---|
| **Epistemic** | `confident.uncertain`, `honest.deceptive`, `curious.disinterested` |
| **Alignment** | `refusing.compliant`, `sycophantic.blunt`, `sincere.manipulative` |
| **Register** | `formal.casual`, `direct.indirect`, `verbose.concise`, `creative.conventional`, `humorous.serious`, `warm.clinical`, `technical.accessible` |
| **Cultural** | `masculine.feminine`, `individualist.collectivist`, `traditional.progressive`, `religious.secular` |

Three larger concept sets ship as well, but aren't fitted by default:

- **`personas`** — 107 personas;
- **`emotions`** — 20 emotional states;
- **`months`** — 12 months.

## Steering expressions

Every surface uses the same expression format. The same recipe can be used
across Python, YAML, OpenAI, Ollama, or the native API.

```text
0.3 honest + 0.4 warm
0.5 formal - 0.2 verbose
0.3 honest|sycophantic
0.3 honest~confident
!sycophantic
0.5 personas%pirate
0.7,0.4 months%january@response
0.4 warm@when:confident.uncertain>0.4
0.3 jlens/orange + 0.2 sae/9143
```

| Syntax | Meaning |
|---|---|
| `+`, `-` | Add or subtract terms |
| leading number | Steering coefficient; omitted terms default to `0.5` |
| `~` | Keep the component shared with another direction |
| `\|` | Remove the component shared with another direction |
| `!` | Mean-ablate a direction |
| `%label` or `%x,y,…` | Choose a named node or coordinates on a manifold |
| `@response`, `@prompt`, `@thinking`, … | Restrict the token phase where a term applies |
| `@when:<probe><op><value>` | Apply a term only while a live probe gate is true |

Manifold coefficients use two coordinates: `along` and `onto`. `along` controls
movement within the manifold toward the target; `onto` reduces the off-surface
component inside the manifold's fitted tube.

## How Saklas works

### Extraction

Saklas first has the model answer a shared set of baseline prompts as each concept,
then it takes the resulting hidden states and fits them to either a curved manifold
or a flat subspace.

Layer allocation uses a Mahalanobis metric estimated from neutral activations.
Discriminative layer selection removes flat axes that fail to straddle the neutral
baseline across the fitted nodes.

The full data flow, artifact boundaries, instrument protocol, and concurrency
invariants are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

### Monitoring

A reading includes fitted coordinates, the centered activation's subspace share,
nearest nodes, and, for curved manifolds, residual and tube membership.

### Jacobian lens

The Jacobian lens implementation follows Gurnee et al.'s
[work](https://transformer-circuits.pub/2026/workspace/index.html).
Saklas lets you use one of the published J-lens artifacts, or fit your own.

### Sparse autoencoders

Saklas can either use a published SAELens release or train a local SAE.

## Installation

Saklas requires Python 3.11 or newer and PyTorch 2.2 or newer. CUDA or Apple
Silicon MPS is strongly recommended for interactive use; CPU is supported for
smaller models and non-GPU workflows.

```bash
pip install saklas
```

The base package includes the HTTP server, the prebuilt Svelte WebUI, and SAELens.
Optional extras add specialized workflows:

| Extra | Adds |
|---|---|
| `flash` | FlashAttention 2 for supported NVIDIA CUDA models; stable and tested |
| `cuda` | `bitsandbytes` quantization and Hugging Face `kernels` acceleration |
| `hf` | `datasets` for streamed J-LENS and SAE corpora |
| `gguf` | GGUF import/export support |
| `research` | NumPy, SciPy, scikit-learn, pandas, Matplotlib, and image helpers |
| `notebook` | Plotly, pandas, and Kaleido notebook helpers |
| `pandas` | pandas-only dataframe export helpers |
| `dev` | Test, lint, type-check, and build tooling |

Extras can be combined:

```bash
pip install "saklas[cuda,flash]"       # full tested NVIDIA path
pip install "saklas[hf,research]"      # dataset-backed research workflows
pip install "saklas[notebook]"         # interactive figures
```

`cuda` and `flash` are Linux/NVIDIA CUDA extras. FlashAttention is selected
automatically when installed; there is no runtime flag to enable it.

From source:

```bash
git clone https://github.com/a9lim/saklas
cd saklas
pip install -e ".[dev]"
```

## Running the server

```bash
saklas serve MODEL [options]
```

Common options:

| Option | Default | Purpose |
|---|---:|---|
| `-d`, `--device` | `auto` | `cuda`, `mps`, `cpu`, or automatic selection |
| `-q`, `--quantize` | none | `4bit` or `8bit` bitsandbytes quantization on CUDA |
| `-p`, `--probes` | `all` | Bundled probe categories, `all`, or `none` |
| `-H`, `--host` | `0.0.0.0` | Bind address |
| `-P`, `--port` | `8000` | Bind port |
| `-S`, `--steer` | none | Default steering expression |
| `--top-k-alts` | `0` | Alternative tokens captured at each decode step |
| `--compile` | off | Opt into `torch.compile` after Saklas probes the path |
| `--cuda-graphs` | off | Pair static cache and CUDA graph capture with `--compile` |
| `-k`, `--api-key` | none | Require bearer authentication; also reads `$SAKLAS_API_KEY` |
| `--no-web` | off | Run the APIs without mounting the dashboard |

`serve` and every subcommand that accepts `-c/--config` read
`~/.saklas/config.yaml` first and then compose any explicit `-c PATH` files on top.
For example:

```yaml
model: google/gemma-3-4b-it
vectors: "0.3 honest + 0.2 warm"
temperature: 0.8
top_p: 0.9
max_tokens: 512
return_top_k: 8
```

Inspect the resolved configuration with `saklas config show` and validate a file
with `saklas config validate path.yaml`.

## Command-line artifact workflows

The CLI has eight top-level verbs. The WebUI covers the interactive versions of
most workflows; the CLI is useful for reproducible preparation, distribution, and
batch work.

| Verb | Role |
|---|---|
| `serve` | Launch the WebUI and the three HTTP protocol surfaces |
| `manifold` | Extract, generate, fit, bake, merge, transfer, compare, or diagnose manifolds |
| `pack` | List, inspect, install, search, push, clear, refresh, remove, or export manifold packs |
| `experiment` | Run alpha fans, replay transcripts, and evaluate naturalness |
| `config` | Show or validate composed configuration |
| `template` | Create and score restricted-choice completion templates |
| `lens` | Fit, fetch, select, read, decompose, or remove Jacobian lenses |
| `sae` | Train, fetch, select, inspect, or remove SAE sources |

Representative commands:

```bash
# Extract and fit a two-pole concept for one model
saklas manifold extract patient impatient -m google/gemma-3-4b-it

# Fit a bundled many-node manifold
saklas manifold fit personas -m google/gemma-3-4b-it

# Install or publish manifold packs through Hugging Face
saklas pack search creativity
saklas pack install OWNER/REPO
saklas pack push local/patient.impatient -a OWNER/REPO -m google/gemma-3-4b-it

# Fetch an official J-LENS or fit one locally
saklas lens fetch google/gemma-3-4b-it
saklas lens fit org/model --prompts 100

# Fetch an SAE or train a local source
saklas sae fetch google/gemma-3-4b-it saelens:gemma-scope-2-4b-it-res
saklas sae train org/model my-sae --layer 20 --tokens 1000000
```

Run `saklas <verb> -h` and `saklas <verb> <subcommand> -h` for the complete flag
surface.

## HTTP APIs

The same `saklas serve` process exposes four surfaces on one port:

- `/` — the Saklas WebUI;
- `/v1/*` — OpenAI-compatible models and chat completions;
- `/api/*` — Ollama-compatible generation and chat;
- `/saklas/v1/*` — native sessions, loom trees, probes, manifolds, templates,
  SAE/J-LENS lifecycle and replay, SSE, and token-plus-measurement WebSockets.

Interactive OpenAPI documentation is available at
[http://localhost:8000/docs](http://localhost:8000/docs).

### OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Describe a rainy afternoon."}],
    extra_body={"steering": "0.3 warm + 0.2 concise"},
)
print(response.choices[0].message.content)
```

### Ollama client

```bash
curl -N http://localhost:8000/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Write a short haiku."}],
  "options": {"steer": "0.3 warm - 0.2 formal.casual"}
}'
```

Saklas targets a trusted local machine or lab network. It is not a hardened
multi-tenant inference service. If you bind it beyond a trusted host, read
[SECURITY.md](SECURITY.md), set an API key, and add TLS, rate limits, request
limits, and isolation outside Saklas.

## Python API

```python
from saklas import SaklasSession, SamplingConfig

with SaklasSession.from_pretrained(
    "google/gemma-3-4b-it",
    device="auto",
    return_top_k=8,
) as session:
    name, profile = session.extract("patient", baseline="impatient")
    session.add_probe(name)

    result = session.generate(
        "How should I learn a difficult skill?",
        steering=f"0.3 {name} + 0.2 concise",
        sampling=SamplingConfig(
            temperature=0.8,
            top_p=0.9,
            max_tokens=256,
            seed=42,
        ),
    ).first

    print(result.text)
    print(result.applied_steering)
    print(result.probe_readings[name].coords)
```

`generate` and `generate_stream` accept the same steering expression as the WebUI.
Generation returns a list-like `RunSet`; `.first` is convenient for a single
completion. `GenerationResult` carries text, token IDs, throughput and timing,
finish reason, the canonical applied expression, captured log probabilities, and
aggregate probe readings.

Batch and sweep helpers return the same result shape:

```python
batch = session.generate_batch(
    ["Describe a sunset.", "Describe a storm."],
    steering="0.3 warm",
)

sweep = session.generate_sweep(
    "Describe a forest.",
    sweep={"warm.clinical": [-0.4, 0.0, 0.4]},
)
```

Restricted-choice scoring evaluates a candidate distribution directly under the
model, optionally with steering:

```python
scores = session.score_choices(
    [{"role": "user", "content": "The first weekday is"}],
    ["Monday", "Tuesday", "Wednesday"],
    steering="0.3 confident",
)
```

Notebook helpers are available from `saklas.notebook` after installing
`saklas[notebook]`: `plot_alpha_sweep`, `plot_probe_correlation`,
`plot_layer_norms`, `plot_trait_history`, and `to_dataframe`.

## Model support

Saklas has end-to-end tested paths for:

- Qwen 2, Qwen 3, and Qwen 3.5, including supported text and MoE variants;
- Gemma 2, Gemma 3, and Gemma 4, including text-only extraction from supported
  multimodal checkpoints;
- Mistral 3 and Ministral 3;
- Llama, GLM, gpt-oss, and Talkie.

Additional architectures are wired through the generic residual-layer interface,
including Mixtral, Phi, Cohere, DeepSeek, OLMo, Granite, Nemotron, GPT-2-family,
Falcon, MPT, DBRX, OPT, and others. Saklas emits a warning when an architecture is
wired but has not been exercised end to end.

CUDA and Apple Silicon MPS both have real-model smoke coverage. Model-specific
features still depend on the checkpoint: chat/role experiments require a compatible
chat template, official SAE and J-LENS sources cover only some models, and
FlashAttention depends on the model's Transformers attention implementation.

## State and distribution

Saklas keeps local state under `~/.saklas/`; set `$SAKLAS_HOME` to move it. The
store contains authored manifolds, per-model fits and neutral statistics, local
SAE/J-LENS artifacts, source bindings, and templates. Conversation saves are
explicit browser-downloaded JSON files (or caller-selected `LoomTree.save()`
paths); they are not autosaved under `~/.saklas/`.

Manifold packs are folders with metadata, node corpora, integrity hashes, and
optional fitted tensors. They can be installed from a local path or distributed as
Hugging Face model repositories. A fitted two-node PCA manifold can also be
exported as a llama.cpp control-vector GGUF with `saklas pack export gguf`.

Treat model repositories and downloaded artifacts as executable or otherwise
untrusted input. Saklas verifies declared artifact hashes, but integrity is not
authorship.

## Development

```bash
pip install -e ".[dev]"

ruff check .
pyright
pytest -q -m "not gpu"
pytest -q tests/
python -m build
```

GPU smoke tests may download model weights:

```bash
pytest -q tests/test_smoke.py
```

The WebUI is a Svelte 5 + Vite application in `webui/`. Its compiled bundle under
`saklas/web/dist/` is committed package data and ships in the wheel.

```bash
cd webui
npm ci
npm run check
npm run build
git diff --exit-code ../saklas/web/dist
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions and adding a
model architecture.

## Research lineage and credits

Saklas builds on Representation Engineering
([Zou et al., 2023](https://arxiv.org/abs/2310.01405)).
[repeng](https://github.com/vgel/repeng) by Theia Vogel is the best-known compact
implementation of that approach; Saklas takes the workbench route, adding live
monitoring, manifold geometry, branching experiments, and server protocols.

Two-pole extraction uses difference-of-means following
[Im & Li, 2025](https://arxiv.org/abs/2502.02716). Manifold steering follows
[Goodfire's manifold work](https://arxiv.org/abs/2605.05115). The `personas`
source is derived from the framing in Anthropic's
[Assistant Axis paper](https://arxiv.org/abs/2601.10387). J-LENS support implements
the verbalizable-workspace method of
[Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html).

If you use Saklas in published research, please cite the relevant upstream methods
alongside the Saklas version and exact model checkpoint you used.

## Issues and security

Please update to the latest Saklas release before filing a bug. Include the model
ID, device, dtype or quantization mode, Saklas version, and a minimal reproduction
in [GitHub Issues](https://github.com/a9lim/saklas/issues).

Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).

## License

Saklas is licensed under AGPL-3.0-or-later. See [LICENSE](LICENSE).
