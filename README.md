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

Saklas is a library for activation steering and trait probing on local HuggingFace models. You give it any concept, from "angry" to "bacterium", and it automatically generates contrastive pairs, extracts a direction from them, and then steers the model's hidden states along that direction when it's time to generate text. The model itself isn't touched, so you can change the steering strength as you go.

Saklas is built on Representation Engineering ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)), the same paper [repeng](https://github.com/vgel/repeng) implements. It has three frontends over one engine: a Svelte web cockpit, a keyboard-native terminal workbench, and a Python API. The cockpit ships pre-built in the wheel and mounts at `http://localhost:8000/` by default on every `saklas serve`. It has a branching conversation tree, live steering and probe racks, per-token highlighting, an activation atlas, a token-drilldown heatmap, and a bidirectional WebSocket that co-streams tokens and probe scores; the terminal workbench covers the same surface keyboard-first. The HTTP server speaks OpenAI `/v1/*` and Ollama `/api/*` on the same port, so Open WebUI, Enchanted, or any other OpenAI or Ollama client can talk to a steered model without changes. Persona cloning works on any text sample: point it at a corpus and it pulls out a voice and style vector without hand-labeled pairs.

Three ways to use it:

- **`saklas serve <model>`**: web UI at `http://localhost:8000/`, mounted by default. The same port also speaks OpenAI `/v1/*` and Ollama `/api/*`, so existing clients work unchanged. Pass `--no-web` for API-only mode.
- **`saklas tui <model>`**: keyboard-native terminal workbench, for SSH and headless work.
- **`SaklasSession`**: Python API for scripted experiments and pipelines.

It runs on CUDA and Apple Silicon MPS. The full TUI has been tested to run comfortably on a MacBook. CPU does work but it's slow. Tested on Qwen, Gemma, Ministral, gpt-oss, Llama, and GLM. A lot more architectures are wired up in `saklas/core/model.py:_LAYER_ACCESSORS` but have not been tested; if you try one, please let me know how it went.

---

## Reporting issues

If you notice any errors while using the program, please update to the most recent version and reinstall the hooks. If it still persists, please open an issue. This project is a work in progress and I am actively finding and fixing bugs.

---

## Credits

The contrastive-pair approach comes from the Representation Engineering paper ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)). [repeng](https://github.com/vgel/repeng) by Theia Vogel is the well-known implementation in this space and is what most people might reach for. Saklas implements the same idea from a different angle: repeng is lean and more of a library, saklas is a full workbench with monitoring, branching chat, a web cockpit, a TUI, and a chat server bundled in. Both are worth your time!

Since v2.1 the default extractor is difference-of-means (DiM) per [Im & Li, 2025](https://arxiv.org/abs/2502.02716), and the per-layer share allocation runs in the Mahalanobis metric (whitened against per-model activation covariance) instead of the v1.x Euclidean magnitude. The legacy v1.x stack (PCA extraction, additive steering, Euclidean shares and cosine) is available all together via `--legacy` on `tui`, `serve`, `vector extract`, and `vector compare`, or piecemeal via `--method pca`, `--steer-mode additive`, and `--metric euclidean` on the relevant verbs.

---

## Quick start

```bash
pip install saklas
saklas serve google/gemma-3-4b-it
```

Open `http://localhost:8000/`. The first run downloads the model and extracts the 26 bundled probes. The cockpit boots with the steering rack empty; click `+ steering` to pick a vector and add it, or paste a steering expression like `0.4 angry` into the expression bar. Tokens stream live with probe-tinted highlighting, the loom sidebar tracks every branch you generate, and the inspector rack on the right has live alpha sliders, signed-value displays, and a per-layer reading strip per probe.

The same port also speaks OpenAI and Ollama, so existing clients work without changes:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"steering": "0.4 cheerful"},
)
```

If you prefer a terminal:

```bash
pip install saklas
saklas tui google/gemma-3-4b-it
```

`/steer 0.4 angry` applies the built-in `angry.calm` vector at α = +0.4 and the model leans angry. `/steer 0.4 calm` gives you the same vector at α = −0.4. `Ctrl+Y` colors each generated token by how strongly the selected probe lit up on it. `Ctrl+A` toggles auto-regen: by default it opens a two-column comparison with an unsteered shadow, and `/auto-regen <mode>` can switch the comparison to inverted, reseeded, hot, cool, or custom recipe variants.

From Python:

```python
from saklas import SaklasSession

with SaklasSession.from_pretrained("google/gemma-3-4b-it") as s:
    name, profile = s.extract("angry.calm")          # bundled bipolar pack
    s.steer(name, profile)                           # register (no alpha yet)
    print(s.generate("What makes a good day?", steering=f"0.3 {name}").text)
```

---

## Install

```bash
pip install saklas             # everything: TUI, HTTP server, web cockpit
pip install saklas[gguf]       # adds the gguf package for llama.cpp interchange
pip install saklas[research]   # adds datasets and pandas for dataset loading and DataFrames
pip install saklas[notebook]   # adds plotly, pandas, and kaleido for Jupyter figure helpers
pip install saklas[sae]        # adds sae-lens for SAE-backed extraction
```

FastAPI and uvicorn are base dependencies as of v3.x, and the web UI bundle ships pre-built inside the wheel, so a plain `pip install saklas` is all you need to get the cockpit at `/` on every `saklas serve`.

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

Saklas takes pairs of sentences and runs them through the model, and then subtracts the two sides. By default it averages those differences at each layer (difference-of-means, or DiM); this gives you the direction from one pole to the other. The legacy contrastive-PCA path (first principal component of the diffs at each layer) is still available via `--method pca` if you want to compare or reproduce v1.x results.

When it's time to generate text, the default injection mode is angular: at each layer, saklas rotates the residual stream toward the concept direction. The per-layer rotations are share-weighted so that the cumulative rotation across the residual stream sums to `|α| × π/2`. So α=1 fully aligns the residual with the concept, α=0.5 lands at 45° cumulative, and α=0 is a no-op. The legacy additive path (add `α × direction` then rescale back to the original norm, with the v1.x `_STEER_GAIN = 2.0`) is still available via `--steer-mode additive` and is bit-identical to v1.x.

Roughly, for coherent steering under angular:

- **0.0–0.2**: barely visible
- **0.2–0.5**: coherent steered
- **0.5–0.7**: strong, often a clear behavior shift
- **0.7–1.0**: saturation territory; expect coherence to break near 1.0

These are rough bands measured on one model and one concept (gemma-4-e4b-it on `angry.calm`); your mileage will vary across models and concepts. Please sweep on your own setup before settling on a number.

When multiple vectors are selected, they compose by summing into a single rotation per layer; cooperating vectors compound, opposing ones cancel.

### Ablation

Prefix a concept with `!` to mean-ablate it: at every covered layer and token position, the component of the residual stream along the concept direction is replaced with the neutral-baseline mean. Bare `!refusal` fully replaces; `0.5 !refusal` blends halfway. Ablation composes with additive steering (`0.3 honest + !sycophantic`) and runs ablate-then-add inside the hook.

```python
with session.steering("!refusal"):
    out = session.generate("How do I break into my neighbor's house?")
```

Ablation is a runtime operation on activations: no new tensors land on disk, and the monitor's probe score for an ablated concept drops to near zero by construction.

### SAE-backed extraction (experimental)

> **Experimental** This pipeline is not as tested as the raw extraction path. α was measured and calibrated on raw extraction and may not cleanly transfer. Quality also depends on which SAE release you pick. I would recommend using a low α (0.1–0.2) and sweeping. For production use the raw pipeline should be the default.

Install `saklas[sae]` and pass `--sae <release>` to `vector extract` to run extraction in sparse-autoencoder feature space (DiM by default; pass `--method pca` for the v1.x SAE-PCA path). Saklas routes through SAELens, so any published release it covers (GemmaScope, Eleuther Meta-LLaMA-3.1 SAEs, Joseph Bloom's, Apollo, Goodfire) should be supported. The output uses the same hook backend as raw extraction.

```bash
saklas vector extract honest.deceptive -m google/gemma-2-2b-it \
  --sae gemma-scope-2b-pt-res-canonical
```

Then steer the SAE variant:

```python
with session.steering("0.3 honest:sae"):
    session.generate("...")
```

```yaml
# ~/.saklas/config.yaml
vectors: "0.3 honest:sae"
```

```
/steer 0.3 honest:sae                                   # TUI: unique SAE variant on disk
/steer 0.3 honest:sae-gemma-scope-2b-pt-res-canonical   # TUI: explicit release
```

SAE profiles only include the layers the release covers.

By default steering fires on every token. The grammar's `@trigger` token attaches a per-term trigger override:

```python
# Steer only the response, never the prompt or the thinking section
session.generate("...", steering="0.4 warm@after")

# Mix regimes per concept
session.generate("...", steering="0.3 honest + 0.4 warm@after")

# Projection: steer honest with sycophancy removed
session.generate("...", steering="0.3 honest|sycophantic")
```

The grammar tokens `@both`, `@response`, `@before`, `@after`, and `@thinking` map to the preset constants `BOTH`, `GENERATED_ONLY`, `PROMPT_ONLY`, `AFTER_THINKING`, and `THINKING_ONLY`. `Trigger.first(n)` and `Trigger.after(n)` let you express token-window ranges. If you want arbitrary combinations, you should pass a pre-built `Steering`.

### Custom concepts

When you steer on something not in the built-in library, the model writes its own contrastive pairs. It first comes up with 9 domains for the axis (for `deer.wolf`, it comes up with "predation and threat assessment", "territorial defense", etc.), and then writes 5 contrastive pairs per domain. 

This means `/steer <anything>` works: religions, animals, fictional characters, anything you can name.

One caveat on custom axes: when the two poles are asymmetric (one specific and one generic, or one that reads more naturally in the reversed order than the order you typed), the model sometimes flips A and B during pair generation, so the statements you asked for as the positive pole end up under `negative` and vice versa. The tensor still extracts cleanly, it just points the wrong way, and `+α` steers toward what you called the negative pole. Balanced axes like the bundled ones don't trip this; it shows up mainly on asymmetric pairs like `human.artificial_intelligence`. If a custom axis does the opposite of what you expect, open `~/.saklas/vectors/local/<concept>/statements.json` and check whether the `positive` entries actually read as the pole you asked for. If they're reversed, swap `positive` and `negative` in the file and re-run extraction, or just flip the pole order in your call.

### Trait monitor

While generating, saklas records the hidden state at every probe layer and every step. They are mean-centered against a neutral baseline and then scored by weighted cosine similarity against every active probe. You can see the history as a sparkline in the TUI; in the library you get `result.readings` as a dict of `ProbeReadings`.

### Cross-model probe transfer

Probes are extracted per (model, concept). To use a probe extracted on one model with a different model, run `saklas vector transfer --from SRC --to TGT NAME`. Saklas computes neutral activations on both models, fits a per-layer alignment between them, and writes a transferred tensor at the target's `_from-<safe_src>` variant path. Transferred profiles coexist with native ones; `/steer 0.3 angry:from-google__gemma-3-4b-it` picks the transferred variant explicitly when both exist.

The transfer carries a `transfer_quality_estimate` in its sidecar (median per-layer R² across shared layers). Values near 1.0 mean the linear map captures the cross-model geometry; values below 0.5 mean transferred probes will be noisy. Visible in `pack ls -v` and the web UI.

### Probe quality diagnostics

Every contrastive extraction emits per-layer metrics alongside the tensor: explained variance ratio (or score proxy under DiM), intra-pair variance, inter-pair alignment, and diff-to-direction projection. `saklas vector why <concept> -m MODEL` shows them as a quality stoplight below the layer histogram. A soft warning fires at extraction time when the median across layers looks degenerate.

Bundled probes extracted before v1.6 don't carry diagnostics on disk. Please run `saklas pack refresh <selector> -m MODEL` to backfill.

### Vector comparison

`Profile.cosine_similarity(other)` gives you weighted cosine similarity between two steering profiles over their shared layers. Pass `whitener=session.whitener` to score in the Mahalanobis metric instead of Euclidean. The CLI has three modes: ranked comparison of one selected vector against all installed profiles, direct pairwise comparison, and N×N similarity matrices, with `--metric mahalanobis` to switch metrics. The TUI has `/compare` for interactive use.

This lets you find concepts that are correlated. For example, on Gemma 4 the model encodes `masculine.feminine` and `traditional.progressive` along the same direction (+0.53 to +0.59 weighted cosine across the 31b and e4b checkpoints), and `hallucinating.grounded` overlaps with `humorous.serious` (+0.53 to +0.66). Steering one nudges the other; the entanglement is in the model's representation, not in the probe extraction.

### The probe library

There are 26 default probes across 7 categories. Each probe is built from 45 contrastive pairs generated using the program's pipeline.

| Category | Probes |
|---|---|
| **Affect** | angry.calm, happy.sad, fearful.unflinching |
| **Epistemic** | confident.uncertain, honest.deceptive, hallucinating.grounded, curious.disinterested |
| **Alignment** | agentic, refusal.compliant, sycophantic.blunt, manipulative |
| **Register** | formal.casual, direct.indirect, verbose.concise, creative.conventional, humorous.serious, warm.clinical, technical.accessible |
| **Social stance** | authoritative.submissive, high_context.low_context, self.other |
| **Cultural** | masculine.feminine, religious.secular, traditional.progressive, individualist.collectivist |
| **Identity** | ai.human |

Poles are aliased: `/steer angry 0.5` → `angry.calm` at α = +0.5. `/steer calm 0.5` → `angry.calm` at α = −0.5. This works for any installed bipolar pack.

Probes extract on first run per model and cache to `~/.saklas/vectors/default/<concept>/<safe_model_id>.safetensors`.

---

## Web UI

```bash
pip install saklas
saklas serve google/gemma-3-4b-it
```

Open `http://localhost:8000/`. The cockpit is dense by design: a workspace rail on the far left, an optional loom tree sidebar, branch canvas plus chat in the center, an inspector rack on the right, and a status footer pinned to the bottom. Everything the TUI does is here through buttons, sliders, menus, and drawers, plus a few things the TUI structurally can't show at once. The visual system is a dark "volcanic glass" theme: sharp geometry, magma-red primary actions, blue data and status accents, dense scan-friendly panels.

**Per-token highlighting** lives directly on the chat tokens. Pick a probe (or `surprise (logprob)`) from the probe rack and tokens tint by score live as they stream; compare two probes via a two-stripe overlay. Every token is clickable regardless of highlight state; clicking opens a layer × probe heatmap drawer for that one token (thinking-block tokens drill into the thinking stream, response tokens into the response stream). The activation atlas drawer extends the same view across the whole active conversation, plus captured top-token alternatives when enabled.

**The steering rack** has one strip per loaded vector: enable toggle, alpha slider with a 0 detent, signed alpha display, trigger pill, variant chip, and a menu for projection, ablation, duplicate, or copy expression. Project-onto and project-orthogonal-to open an inline modal that takes a target concept name. The rack also has an explicit vector input and `apply vector` button; type a concept name, hit Enter, and saklas extracts or loads the vector and adds it to the rack. `browse` opens the vector picker, and the full recipe builder drawer exposes coefficients, variants, projections, ablations, and triggers in one dense editor. The canonical steering expression renders below the rack and is paste-editable.

**The probe rack** is symmetric: one strip per active probe, with a live sparkline, current value bar, signed value display, and an always-visible per-layer reading strip (one heatmap cell per covered layer, tinted by the probe's score at that layer for the most recent token). The whole row is the click target for highlight selection. `+ probe` opens a probe picker that mirrors the steering picker.

**The loom** surfaces branching conversation state instead of hiding alternates in a flat log. The branch canvas shows the active path, siblings under the current parent, and children of the active node, with buttons for `regenerate N`, alpha-grid fan-out, sibling comparison, and child comparison. The loom sidebar adds tree navigation, edit, branch, delete, star, note, filter, pin, and transcript workflows. Branches carry the recipe that produced them, so steering experiments remain reproducible.

**Auto-regen** generalizes the old A/B compare. The default mode produces an unsteered shadow response, but the same topbar control can also invert steering, reseed, run hotter or cooler, or apply a custom partial recipe. Two-column chat rendering shows the primary path beside the pinned or auto-regenerated comparison path.

**Send modifiers.** `Ctrl+Enter` (or `Cmd+Enter`) commits a turn without running a model: a user turn under the active node, or, when the active node is a user node, an authored assistant turn under it. `Ctrl`-clicking the send button does the same, and the send-button label updates live as you hold the modifier. Separately, typing into a user-rooted context generates an answer-prefill instead of a new user turn, seeding the next assistant reply with whatever you type before the model takes over.

**Pending queue.** Submissions during an in-flight generation (chat sends, commits, regen, rack and sampling changes) defer rather than racing the stream. Each queued submission renders as a dim chip above the input with a per-bubble cancel; the up-arrow walks back through the queue and lets you edit any pending item in place. `Esc` cancels an in-flight gen and leaves the queue intact, so you can interrupt and replan without losing your follow-ups.

**Analysis drawers.** The workspace rail and topbar open drawers for the experiment lab, activation atlas, recipe builder, advanced sampling, packs, session and auth, health, extract, load, compare, system prompt, model info, help, export, vector merge, corpus clone, transcript import and export, the N×N correlation matrix across loaded vectors and active probes, the cross-layer pairwise cosine matrix between any two named vectors or probes, and the layer-norm explorer. Reference drawers span the union of registered steering vectors and active probes.

The loom tree cache and highlight selection persist to `localStorage` per model for fast reloads; the server tree remains authoritative after bootstrap. Chat input recall is intentionally in-memory only.

The cockpit mounts by default on every `saklas serve`. Pass `--no-web` for API-only mode on a proxied or production deployment where `/` already belongs to something else.

The web UI is a Svelte 5 and Vite single-page app that ships pre-built in the wheel. Source lives at `webui/`; `cd webui && npm ci && npm run build` regenerates the bundle into `saklas/web/dist/` if you want to iterate on it.

---

## Terminal UI

```bash
saklas tui google/gemma-2-9b-it
saklas tui mistralai/Mistral-7B-Instruct-v0.3 -q 4bit
saklas tui meta-llama/Llama-3.1-8B-Instruct -p affect register
```

There are three panels: a workbench/rack on the left, chat in the center, and a probe rack plus layer map on the right. `Tab` cycles between panels, arrow keys navigate within each panel, and `Ctrl+L` opens the full-screen loom tree when you want more room for branch navigation.

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

The right panel's layer map shows `||baked||` for the selected probe as a compact histogram across model layers. It is static for the selected profile; per-token probe activity lives in the chat highlight itself.

The footer in the chat panel shows generation progress, live tok/s, elapsed, and the running perplexity of the token stream (geometric mean of the configured sampler distribution after temperature, top-k, and top-p renormalization).

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

**Tested**: Qwen, Gemma, Ministral, gpt-oss, Llama, GLM.

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
