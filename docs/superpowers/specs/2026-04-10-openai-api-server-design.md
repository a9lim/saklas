# OpenAI-Compatible API Server for Steer

## Summary

Add a headless HTTP server mode (`steer serve`) that exposes an OpenAI-compatible API backed by `SteerSession`. Clients can use any OpenAI SDK, LangChain, or `curl` to interact with a steered model. Steering and probe management are exposed through additional REST endpoints under `/v1/steer/`.

## CLI

New subcommand — existing `steer <model>` (TUI) is unchanged.

```
steer serve <model_id> [flags]
```

**Flags** (all optional):
| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `--quantize`, `-q` | None | `4bit` or `8bit` |
| `--device`, `-d` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `--probes`, `-p` | all | Probe categories to bootstrap |
| `--system-prompt`, `-s` | None | Default system prompt |
| `--max-tokens`, `-m` | `1024` | Default max tokens |
| `--cache-dir`, `-c` | package default | Vector/probe cache dir |
| `--steer` | None | Pre-load vectors: `name:alpha` or `name` (alpha=0) repeatable |
| `--cors` | None | CORS origin(s), repeatable. Omit = no CORS. |

`--steer` accepts either `name:alpha` (pre-register with default alpha) or just `name` (register with alpha 0, client overrides per-request). The `name` resolves through the same pipeline as the TUI's `/steer` command — curated dataset, cache, or extraction.

## Dependencies

Add `fastapi` and `uvicorn[standard]` as a new optional dependency group:

```toml
[project.optional-dependencies]
serve = ["fastapi>=0.110", "uvicorn[standard]>=0.29"]
```

No new core dependencies. If a user runs `steer serve` without the extras installed, fail with a clear message pointing to `pip install steer[serve]`.

## Architecture

### New files

| File | Purpose |
|------|---------|
| `steer/server.py` | FastAPI app factory, all route definitions, request/response models |

### Modified files

| File | Change |
|------|--------|
| `steer/cli.py` | Add `serve` subcommand via argparse subparsers. Existing `steer <model>` becomes the implicit default (no breaking change). |
| `pyproject.toml` | Add `serve` optional dependency group |

### No changes to

`session.py`, `generation.py`, `hooks.py`, `monitor.py`, `vectors.py`, `model.py`, `results.py`, `datasource.py`, `probes_bootstrap.py`, `tui/`.

The server is a thin HTTP layer over `SteerSession`. All business logic stays in the session.

## Concurrency Model

Single `SteerSession` instance. The model can only run one generation at a time (GPU is serialized).

- **Generation endpoints** (`/v1/chat/completions`, `/v1/completions`): If a generation is already in progress, return `409 Conflict` immediately. No request queuing.
- **Management endpoints** (vectors, probes, session): These don't touch the model during generation and can be served concurrently. Exception: `POST /v1/steer/vectors/extract` runs forward passes — it also returns 409 if generation is active, and blocks generation while extracting.
- **Extraction** (`POST /v1/steer/vectors/extract`): Long-running. Streams progress messages via SSE (`text/event-stream`). Client can request blocking JSON response instead via `Accept: application/json`.

## Endpoints

### OpenAI-Compatible

#### `GET /v1/models`

Returns the loaded model in OpenAI's list format.

```json
{
  "object": "list",
  "data": [
    {
      "id": "google/gemma-2-9b-it",
      "object": "model",
      "created": 1712700000,
      "owned_by": "local"
    }
  ]
}
```

#### `GET /v1/models/{model_id:path}`

Returns a single model object. 404 if the model_id doesn't match the loaded model.

#### `POST /v1/chat/completions`

Standard OpenAI chat completions request, plus optional `steer` extension.

**Request body:**

```json
{
  "model": "google/gemma-2-9b-it",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "stream": false,
  "steer": {
    "alphas": {"cheerful": 0.2, "formal": 0.1},
    "orthogonalize": true
  }
}
```

`model` is accepted but ignored (single-model server). `steer` is optional — omit it to use server defaults. `steer.alphas` merges with (and overrides) server defaults.

**Non-streaming response:**

```json
{
  "id": "steer-abc123",
  "object": "chat.completion",
  "created": 1712700000,
  "model": "google/gemma-2-9b-it",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello there!"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 42,
    "total_tokens": 42
  },
  "probe_readings": {
    "cheerful": {"mean": 0.82, "std": 0.05, "min": 0.71, "max": 0.93}
  }
}
```

`probe_readings` is our extension — standard clients ignore it. `prompt_tokens` is 0 because we don't track input token count separately (could be added later).

**Streaming response** (`"stream": true`):

SSE stream. Each event is a `chat.completion.chunk`:

```
data: {"id":"steer-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"steer-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Probe readings are included in the final chunk (the one with `finish_reason`):

```json
{"id":"steer-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"probe_readings":{...}}
```

#### `POST /v1/completions`

Raw text completions (no chat template). Same structure as OpenAI's completions API.

**Request body:**

```json
{
  "model": "google/gemma-2-9b-it",
  "prompt": "The meaning of life is",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": false,
  "steer": {
    "alphas": {"cheerful": 0.2}
  }
}
```

**Response:** Same shape as chat completions but with `"object": "text_completion"` and `choices[].text` instead of `choices[].message`.

**Implementation note:** `SteerSession.generate()` accepts a list of message dicts. For raw completions, we pass `[{"role": "user", "content": prompt}]` without a system prompt and skip chat template wrapping. This requires a small addition — a `raw` parameter on `_prepare_input` that skips `build_chat_input` and just tokenizes directly. Alternatively, we tokenize in the server and pass `input_ids` — but `generate()` takes strings/messages, not tensors. Simplest path: add a `raw: bool = False` parameter to `generate()`/`generate_stream()` that, when true, tokenizes the input without chat template. This is a minor session.py change.

**Revision**: On reflection, this touches session.py which we said we wouldn't change. Two options:
1. Tokenize in the server and call `generate_steered` directly (bypasses session, duplicates orchestration logic).
2. Add the `raw` flag to session (small, clean, useful for the programmatic API too).

Option 2 is better — it's a one-line branch in `_prepare_input` and makes the programmatic API more capable. We'll allow this targeted session.py change.

### Vector Management — `/v1/steer/vectors`

#### `GET /v1/steer/vectors`

List registered steering vectors.

```json
{
  "vectors": {
    "cheerful": {
      "layers": [0, 1, 2, "..."],
      "top_layers": [
        {"layer": 14, "score": 0.89},
        {"layer": 15, "score": 0.85}
      ],
      "default_alpha": 0.2
    }
  }
}
```

`top_layers` shows the 5 highest-signal layers by explained variance score. `default_alpha` reflects the server-startup default (0 if loaded without alpha).

#### `POST /v1/steer/vectors/extract`

Extract a new steering vector.

**Request body:**

```json
{
  "name": "cheerful",
  "source": "cheerful",
  "baseline": null,
  "alpha": 0.15,
  "register": true
}
```

`source` can be:
- A concept name string (goes through full extraction pipeline)
- An object `{"pairs": [["positive", "negative"], ...]}` for raw pairs

`alpha` sets the server default alpha for this vector. `register` (default true) auto-registers the vector for steering.

**Streaming response** (default, `Accept: text/event-stream`):

```
event: progress
data: {"message": "Generating contrastive pairs for 'cheerful'..."}

event: progress
data: {"message": "Extracting contrastive profile (45 pairs)..."}

event: done
data: {"name": "cheerful", "layers": 26, "top_layer": 14, "top_score": 0.89}
```

**Blocking response** (`Accept: application/json`): waits until done, returns the `done` payload directly.

#### `POST /v1/steer/vectors/load`

Load a vector from a saved `.safetensors` file.

```json
{
  "name": "cheerful",
  "path": "/path/to/cheerful.safetensors",
  "alpha": 0.15
}
```

#### `DELETE /v1/steer/vectors/{name}`

Unsteer and unregister a vector. Returns 204 on success, 404 if not found.

### Probe Management — `/v1/steer/probes`

#### `GET /v1/steer/probes`

List active probes and their last readings.

```json
{
  "probes": {
    "cheerful": {
      "active": true,
      "last_reading": {"mean": 0.82, "std": 0.05, "min": 0.71, "max": 0.93}
    }
  }
}
```

#### `GET /v1/steer/probes/defaults`

List available default probe names grouped by category.

```json
{
  "defaults": {
    "emotion": ["happiness", "sadness", "anger", "fear", "surprise", "disgust"],
    "personality": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"],
    "safety": ["toxicity", "deception", "manipulation", "bias", "harmfulness"]
  }
}
```

#### `POST /v1/steer/probes/{name}`

Activate a probe. If the probe name matches a default, it's bootstrapped automatically. Otherwise, extraction runs (same as vectors).

```json
{
  "profile_path": null
}
```

`profile_path` is optional — provide a `.safetensors` path to load a custom probe profile. Omit to use defaults/extraction.

Returns 200 with probe info on success.

#### `DELETE /v1/steer/probes/{name}`

Deactivate a probe. Returns 204, or 404 if not active.

### Session Management — `/v1/steer/session`

#### `GET /v1/steer/session`

```json
{
  "model": "google/gemma-2-9b-it",
  "model_info": {
    "model_type": "gemma2",
    "num_layers": 26,
    "hidden_dim": 2304,
    "vram_used_gb": 5.2
  },
  "config": {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "system_prompt": null
  },
  "default_alphas": {"cheerful": 0.2},
  "history_length": 4
}
```

#### `PATCH /v1/steer/session`

Update generation config.

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "system_prompt": "You are a helpful assistant."
}
```

All fields optional — only provided fields are updated.

#### `POST /v1/steer/session/clear`

Clear conversation history and probe readings. Returns 204.

#### `POST /v1/steer/session/rewind`

Rewind last turn. Returns 204, or 400 if history is empty.

## Request/Response Models

All request/response bodies are Pydantic models defined in `server.py`. This gives us automatic validation and OpenAPI docs at `/docs`.

## Error Handling

Standard HTTP status codes with JSON error bodies:

```json
{
  "error": {
    "message": "Generation already in progress",
    "type": "conflict",
    "code": 409
  }
}
```

| Status | When |
|--------|------|
| 400 | Invalid request (bad alphas, unknown vector name, empty history for rewind) |
| 404 | Unknown model, vector, or probe |
| 409 | Generation/extraction already in progress |
| 422 | Pydantic validation failure (automatic from FastAPI) |
| 500 | Unexpected errors |

## Conversation State

The server maintains a single conversation via `SteerSession._history`. Each `/v1/chat/completions` request appends to this history (matching the TUI behavior). Clients that want stateless completions should call `POST /v1/steer/session/clear` between requests, or use `/v1/completions` which doesn't maintain history.

This is a deliberate choice — it matches the mental model of "one person talking to one model" (like Ollama). Multi-session support is out of scope for v1.

## Session.py Change

One targeted change: add `raw: bool = False` to `generate()` and `generate_stream()`. When `raw=True`, `_prepare_input` tokenizes the input string directly (`self._tokenizer.encode()`) instead of wrapping it in chat template via `build_chat_input`. This supports `/v1/completions` cleanly without duplicating generation orchestration in the server.

## Testing

- Unit tests for request/response model serialization (no GPU needed)
- Unit tests for route logic with a mocked `SteerSession` (no GPU needed)  
- Integration test: start server, hit `/v1/models`, verify response shape (needs GPU)
- Integration test: `/v1/chat/completions` streaming and non-streaming (needs GPU)
- Integration test: vector extract → steer → generate → probe readings round-trip (needs GPU)

## What's Explicitly Out of Scope

- Multi-session / multi-user support
- Authentication / API keys
- Request queuing (409 on conflict, not queue)
- `/v1/embeddings` endpoint
- Batch API
- Function calling / tool use
- `logprobs` field in responses
