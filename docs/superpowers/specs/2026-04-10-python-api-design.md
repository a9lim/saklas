# Python API & Headless Mode for steer

**Date:** 2026-04-10
**Status:** Draft

## Goal

Make steer usable as a Python library for researchers running scripted experiments, batch sweeps, and notebook exploration — without losing the interactive TUI. The API becomes the single backend; the TUI becomes a thin frontend over it.

## Audience

**Primary:** Researchers who want to script batch jobs — extract vectors across models, steer generation on benchmark prompts, collect probe readings into structured output. They work in Jupyter notebooks and Python scripts.

**Secondary:** Tool builders who embed steer's capabilities into their own apps and pipelines. They benefit from the same clean API once it stabilizes.

## Scope

steer extracts, steers, monitors, and emits structured results. Evaluation (perplexity, toxicity, benchmarks) is the researcher's problem. steer provides hooks and structured output so researchers can plug in their own metrics.

No new CLI subcommands. The TUI remains the only CLI entry point. All headless/batch work goes through the Python API.

---

## Component 1: SteerSession

The central orchestrator. Owns the model, steering manager, monitor, conversation history, and generation config. Replaces the orchestration logic currently in `tui/app.py`.

### Construction

```python
from steer import SteerSession

session = SteerSession("google/gemma-2-2b-it", device="auto", quantize="4bit")
# Also a context manager:
with SteerSession("meta-llama/Llama-3-8B-Instruct") as session:
    ...
```

Parameters: `model_id` (str), `device` (str, default "auto"), `quantize` (str | None, default None), `probes` (list[str] | None, default all categories), `system_prompt` (str | None), `max_tokens` (int, default 1024), `cache_dir` (str | None).

Construction calls `load_model`, `get_layers`, `detect_device`, `get_model_info`, and optionally `bootstrap_probes` — the same sequence `cli.py` performs today.

### Vector extraction

```python
# Curated dataset by name
happy_profile = session.extract("happy")

# Raw pairs
profile = session.extract([("Be formal", "Be casual"), ("Dear sir", "Hey dude")])

# DataSource (see Component 2)
profile = session.extract(DataSource.csv("my_pairs.csv"))

# Load pre-extracted
profile = session.load_profile("path/to/saved.safetensors")

# Save
session.save_profile(profile, "path/to/output.safetensors", metadata={"concept": "happy"})
```

`extract()` accepts: a string (curated dataset name lookup), a `list[tuple[str, str]]`, or a `DataSource`. Internally normalizes to pairs and calls `extract_contrastive`. Results are cached by concept name.

### Steering

```python
session.steer("happy", happy_profile, alpha=1.5)
session.set_alpha("happy", 2.0)
session.toggle("happy")          # disable without removing
session.unsteer("happy")         # remove entirely
session.orthogonalize()          # re-orthogonalize all active vectors
session.clear_vectors()          # remove all
```

These delegate to `SteeringManager`. `steer()` calls `manager.add_vector()` then `manager.apply_to_model()`. `set_alpha()` and `toggle()` call the corresponding manager methods. A separate steering lock allows mid-generation alpha adjustment.

### Monitoring

```python
session.monitor("honest")                        # curated probe by name
session.monitor("custom_probe", custom_profile)  # from a profile
session.unmonitor("honest")
```

Delegates to `TraitMonitor.add_probe()` / `remove_probe()`. Probe names that match curated datasets auto-extract (with caching).

### State queries

```python
session.vectors        # dict[str, VectorState]  — name -> (profile, alpha, enabled)
session.probes         # dict[str, ProbeState]    — name -> (profile, latest readings)
session.generation     # GenerationState | None   — in-progress generation info
session.config         # GenerationConfig         — the existing dataclass from generation.py
session.model_info     # dict                     — name, params, arch, device, dtype
session.history        # list[dict]               — conversation message log
session.last_result    # GenerationResult | None  — result of most recent generation
```

All safe to read from any thread. The TUI renders from these instead of maintaining parallel state.

### Generation

Two modes sharing the same underlying loop:

```python
# Blocking (scripts/notebooks)
result = session.generate("What is the meaning of life?")
result = session.generate([
    {"role": "system", "content": "You are a poet."},
    {"role": "user", "content": "Write about rain."},
])
# Returns GenerationResult (see Component 3)

# Streaming (TUI / interactive)
for token in session.generate_stream("Tell me about rain"):
    print(token.text, end="", flush=True)
    # token.readings available per-token if probes active
# session.last_result has the final GenerationResult
```

Streaming is the primitive; blocking drains the iterator. Both append to `session.history`.

### Generation control

```python
session.stop()           # interrupt in-flight generation (sets stop flag)
session.rewind()         # drop last assistant turn from history
session.clear_history()  # clear all conversation history
```

### Config mutation

```python
session.config.temperature = 0.8
session.config.top_p = 0.9
session.config.max_tokens = 512
session.config.system_prompt = "You are helpful."
```

### Lifecycle

```python
session.close()  # removes all hooks, frees model references
```

Also supports context manager protocol.

---

## Component 2: DataSource

Normalizes contrastive pairs from multiple formats into `list[tuple[str, str]]`.

```python
from steer import DataSource

ds = DataSource.curated("happy")                    # steer's bundled datasets
ds = DataSource.json("my_pairs.json")               # steer JSON schema
ds = DataSource.csv("pairs.csv")                    # default cols: positive, negative
ds = DataSource.csv("pairs.csv", positive_col="good", negative_col="bad")
ds = DataSource.huggingface("user/dataset", positive_col="pos", negative_col="neg")
ds = DataSource.huggingface("user/dataset", split="train[:100]")
ds = DataSource.from_pairs([("Be happy", "Be sad")])

# Interface
ds.pairs          # list[tuple[str, str]]
ds.name           # str (inferred from filename/dataset or provided via kwarg)
ds.description    # str | None
```

### Design decisions

- **Eager materialization.** Pairs are loaded on construction. Contrastive datasets are small — no streaming needed.
- **`session.extract()` normalizes internally.** Strings become `DataSource.curated()`, lists become `DataSource.from_pairs()`, DataSources pass through. No separate method per format.
- **HuggingFace is optional.** `DataSource.huggingface()` raises ImportError if `datasets` not installed.
- **Name inference.** `DataSource.json("empathy.json")` infers name `"empathy"`. `DataSource.curated("happy")` uses `"happy"`. Raw pairs default to `"custom"`. All overridable via `name` kwarg.
- **Column defaults.** CSV and HuggingFace default to `positive`/`negative` columns.

---

## Component 3: Structured output

### ProbeReadings

```python
@dataclass
class ProbeReadings:
    per_token: list[float]     # cosine similarity at each generated token
    mean: float
    std: float
    min: float
    max: float
    delta_per_tok: float       # mean change between consecutive tokens

    def to_dict(self) -> dict
```

### GenerationResult

```python
@dataclass
class GenerationResult:
    text: str                              # decoded output
    tokens: list[int]                      # raw token ids
    token_count: int
    tok_per_sec: float
    elapsed: float                         # seconds
    readings: dict[str, ProbeReadings]     # probe_name -> readings (empty dict if no probes)
    vectors: dict[str, float]              # snapshot of active vectors + alphas at generation time

    def to_dict(self) -> dict              # everything as plain Python types
```

### TokenEvent

```python
@dataclass
class TokenEvent:
    text: str                              # decoded token text
    token_id: int
    index: int                             # position in sequence
    readings: dict[str, float] | None      # per-probe cosine sim, None if no probes
```

### Design decisions

- **`GenerationResult.vectors` snapshots steering state at generation time.** Each result in a sweep records what was active.
- **`to_dict()` returns plain Python types only.** No tensors, no numpy. Compatible with `json.dumps`, pandas, wandb.
- **`readings` is an empty dict when no probes are active.** Not None. Avoids null checks.
- **No aggregation methods on these classes.** That's ResultCollector's job.

---

## Component 4: ResultCollector

Accumulates results across runs for export.

```python
from steer import ResultCollector

collector = ResultCollector()

for alpha in [0.5, 1.0, 1.5, 2.0]:
    session.steer("happy", happy_profile, alpha=alpha)
    for prompt in prompts:
        result = session.generate(prompt)
        collector.add(result, prompt=prompt, concept="happy", alpha=alpha)
    session.unsteer("happy")

# Access
collector.results          # list[dict]

# Export
collector.to_dicts()       # list[dict] — flat, serializable
collector.to_jsonl("results.jsonl")
collector.to_csv("results.csv")
df = collector.to_dataframe()  # requires pandas
```

### How `add()` works

```python
collector.add(result, **tags)
```

- `result` is a `GenerationResult`. Its fields get flattened into the row.
- `**tags` are arbitrary key-value pairs (prompt, concept, alpha, model_name, run_id, etc.). Become columns.
- Probe readings flatten: `readings["honest"].mean` -> column `probe_honest_mean`, etc.
- Active vectors flatten: `vectors["happy"]` -> column `vector_happy_alpha`.

### Design decisions

- **Researcher controls the outer loop.** ResultCollector has no opinions about what a "sweep" looks like.
- **Flat column namespace.** Probes prefixed `probe_`, vectors prefixed `vector_`. Natural for CSV/DataFrame.
- **Pandas is optional.** `to_dataframe()` raises ImportError. `to_dicts()`, `to_jsonl()`, `to_csv()` use stdlib only.
- **No aggregation.** No `.mean_by()`, no `.plot()`. That's pandas territory.

---

## Component 5: Package structure

### New files

```
steer/
├── session.py          # SteerSession (NEW)
├── datasource.py       # DataSource (NEW)
├── results.py          # GenerationResult, TokenEvent, ProbeReadings, ResultCollector (NEW)
├── model.py            # unchanged
├── vectors.py          # unchanged
├── generation.py       # unchanged
├── hooks.py            # unchanged
├── monitor.py          # unchanged
├── probes_bootstrap.py # unchanged
├── cli.py              # minor change: constructs SteerSession, passes to SteerApp
├── __init__.py         # exports SteerSession, DataSource, ResultCollector, result types
├── tui/
│   ├── app.py          # REWRITTEN: thin frontend over SteerSession
│   ├── chat_panel.py   # minor changes: reads from session.history
│   ├── vector_panel.py # minor changes: reads from session.vectors
│   └── trait_panel.py  # minor changes: reads from session.probes
```

### Public API

```python
# steer/__init__.py
from steer.session import SteerSession
from steer.datasource import DataSource
from steer.results import GenerationResult, TokenEvent, ProbeReadings, ResultCollector

__all__ = [
    "SteerSession",
    "DataSource",
    "GenerationResult",
    "TokenEvent",
    "ProbeReadings",
    "ResultCollector",
]
```

### Dependencies

No new required dependencies. Optional extras:

- `pip install steer[hf]` — pulls `datasets` for `DataSource.huggingface()`
- `pip install steer[pandas]` — pulls `pandas` for `ResultCollector.to_dataframe()`
- `pip install steer[research]` — pulls both

---

## Component 6: Thread safety and generation lifecycle

### Generation lock

One generation at a time, enforced by a non-blocking lock:

```python
self._gen_lock = threading.Lock()

def generate(self, input, **kwargs):
    if not self._gen_lock.acquire(blocking=False):
        raise RuntimeError("Generation already in progress")
    try:
        return self._generate_blocking(input, **kwargs)
    finally:
        self._gen_lock.release()
```

`generate_stream()` follows the same pattern. Calling either while generation is in-flight raises immediately.

### Stop

`session.stop()` sets a flag checked per-token by the generation loop (same mechanism as `GenerationState.stop_flag` today). The in-flight call returns a partial `GenerationResult`.

### Steering lock

Separate from the generation lock. `steer()`, `unsteer()`, `set_alpha()`, `toggle()`, `orthogonalize()` acquire the steering lock. This allows mid-generation alpha adjustment (same as the TUI's arrow-key behavior).

### State reads

`session.vectors`, `session.probes`, `session.history`, `session.config`, `session.model_info` are safe to read from any thread at any time. Immutable snapshots or protected by lightweight locks.

### Monitor flush

- **Blocking mode:** `generate()` calls `monitor.flush_to_cpu()` after the loop completes, packs readings into `GenerationResult`.
- **Streaming mode:** Each `TokenEvent` carries the latest readings from the highest-layer hook's buffer advance. Final `GenerationResult` available via `session.last_result`.

### TUI integration

```python
# tui/app.py worker thread
def _generation_worker(self):
    for token in self.session.generate_stream(messages):
        self.post_message(TokenReceived(token))
    self.post_message(GenerationComplete())
```

The TUI's 15 FPS poll timer for token draining is replaced by Textual's message system. Each `TokenReceived` triggers a UI update. Monitor readings arrive with each token event.

---

## What changes vs. what doesn't

### Changes

| Component | Nature of change |
|-----------|-----------------|
| `tui/app.py` | Substantial rewrite. All orchestration moves to SteerSession. Becomes ~400 lines of Textual layout, keyboard handling, and session method dispatch. |
| `cli.py` | Minor. Constructs SteerSession, passes to SteerApp(session). |
| `__init__.py` | Exports new public API. |
| Panel files | Minor. Read from session state instead of app attributes. |

### No changes

| Component | Why |
|-----------|-----|
| `model.py` | Session wraps it, doesn't modify it. |
| `vectors.py` | Session calls `extract_contrastive`, `save_profile`, `load_profile` as-is. |
| `generation.py` | Session wraps `generate_steered` and `build_chat_input`. |
| `hooks.py` | Session owns a `SteeringManager` and calls its methods. |
| `monitor.py` | Session owns a `TraitMonitor` and calls its methods. |
| `probes_bootstrap.py` | Session calls `bootstrap_probes` on init. |
| Datasets / probes | Data files unchanged. |
| CLI arguments | Same args, same behavior. |
| TUI keyboard shortcuts | Same shortcuts, routed to session methods. |

### Test impact

Existing smoke tests continue to work (they test core modules directly). New tests needed for:

- SteerSession lifecycle (construct, extract, steer, generate, close)
- DataSource format parsing (each classmethod)
- ResultCollector accumulation and export
- Thread safety (concurrent state reads during generation)
- TUI integration (session passed in, commands dispatch correctly)
