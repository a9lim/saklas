# server/

Dual-protocol HTTP: OpenAI `/v1/*` + Ollama `/api/*` + native `/saklas/v1/*` on the same port.

## app.py

FastAPI factory + OpenAI route handlers. `create_app(session, default_steering=None, api_key=None)` mounts OpenAI (`/v1/models`, `/v1/chat/completions`, `/v1/completions`) + calls `register_saklas_routes` and `register_ollama_routes`. `default_steering` is a pre-built `Steering` or `None`; per-request expressions compose over it at the key level.

Thin HTTP — all routes call `session.generate(..., sampling=SamplingConfig(...), steering=Steering(...))` directly. `_SamplingBase` pydantic shared by chat/completions: `stop`, `seed`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs` (bool chat / int completions), `top_logprobs`, `stream_options.include_usage`, `max_completion_tokens` (aliased to `max_tokens` via model validator), native `steering` top-level field (a steering expression string — the shared grammar), native `thinking` field (`None` = auto). Dict-shaped steering payloads are rejected at the pydantic layer.

Accept-and-ignore: `user`, `n`, `response_format: {"type": "text"}`, empty `tools: []` / `tool_choice: "none"` (LangChain compat — non-empty rejected via `_check_langchain_compat` → 400). `ChatMessage._flatten_content` concatenates text parts of multimodal arrays (non-text rejected via `UnsupportedContentError`).

Responses carry real `usage`, accurate `finish_reason` from `session._gen_state.finish_reason`, per-request `created`. `_stream_generation` emits first-chunk `{role: "assistant"}` delta; final chunk's `finish_reason` comes from gen state; optional usage chunk before `[DONE]` when `stream_options.include_usage`.

**`session.lock`** (`asyncio.Lock`, distinct from threading `_gen_lock`) serializes non-streaming (`async with session.lock`) and streaming (held for full stream with 5-minute timeout → 503) **across both OpenAI and Ollama routes**; requests queue FIFO rather than 409ing. `acquire_session_lock(session)` is the bounded async context manager both route families use.

Bearer auth via `SAKLAS_API_KEY` / `--api-key` applied as app-level dependency covering HTTP and WebSocket routes; `_require_auth` + `_check_bearer` for HTTP; `ws_auth_ok(websocket)` called before `websocket.accept()` in WS handlers (closes 1008 on fail). Unset API key = open.

Global `_on_saklas_error` handler maps `SaklasError` subclass → HTTP status, path-dispatching Ollama vs OpenAI error shapes. `RequestValidationError` handler maps pydantic errors to OpenAI shape. Thinking tokens stream as `reasoning_content` in chat delta. `_render_logprobs_chat` decodes token ids via tokenizer. `saklas.io.hf::_download` upgrades the error when a user points at a dataset repo instead of a model repo.

`SAKLAS_STRICT_MODEL=1` makes the `model` field on OpenAI + Ollama requests 404 on mismatch; unset = accept any name.

**Not supported (either protocol)**: tool calling, strict JSON/`json_schema` mode, embeddings. `server/openai.py` is currently a re-export facade over `app.py` (full split of OpenAI routes deferred to avoid regression risk).

## saklas_api.py

Native `/saklas/v1/*` resource tree, mounted by `register_saklas_routes(app)`. Session-shaped API with single-session impl: one session id `"default"` (and the loaded model id also resolves).

Routes:
- `GET/POST /saklas/v1/sessions` — list / idempotent create (POST body accepted but model mismatch logs a warning and returns existing)
- `GET/PATCH/DELETE /saklas/v1/sessions/{id}` — info / update defaults / no-op 204
- `POST /saklas/v1/sessions/{id}/{clear,rewind}`
- Vector management under `/sessions/{id}/vectors` — list / get / load-from-disk / delete
- `POST /sessions/{id}/extract` — async via `asyncio.to_thread(session.extract, ...)`, SSE progress when `Accept: text/event-stream`, JSON otherwise
- Probe management under `/sessions/{id}/probes` — list / defaults / activate / deactivate
- `POST /sessions/{id}/probe` — one-shot scoring via `monitor.measure(model, tokenizer, layers, text)` under the session lock; no generation required

**Killer feature — `WS /saklas/v1/sessions/{id}/stream`**: bidirectional token+probe co-stream.
- Client sends `{type: "generate", input, steering, sampling, thinking, stateless, raw}` or `{type: "stop"}`
- Server emits `{type: "started", generation_id}`, `{type: "token", text, thinking, token_id}` per token, `{type: "done", result: {text, tokens, finish_reason, usage, per_token_probes: [{token_idx, probes}]}}`
- Per-token probes assembled post-stream from `session._last_per_token_scores` in the `done` event

**Live traits SSE — `GET /saklas/v1/sessions/{id}/traits/stream`**: per-token probe scores in real time during any active generation. Uses inline per-token scoring (`TraitMonitor.score_single_token`) gated behind `session._trait_subscribers > 0`; zero overhead when no SSE clients are connected. Events: `start`, `token` (with `{idx, text, thinking, probes}`), `done` (with aggregate). Connection stays open across generations; multiple clients supported via `session._trait_queues` (list of `(loop, asyncio.Queue)` pairs, `threading.Lock`-protected).

**Concurrency design**: `session.generate` runs in a worker thread via `asyncio.to_thread`; `on_token` bridges to asyncio via `loop.call_soon_threadsafe(queue.put_nowait, event)`; the handler coroutine races `queue.get()` against `websocket.receive_json()` via `asyncio.wait(..., FIRST_COMPLETED)` so incoming `{type: "stop"}` can call `session.stop()` without blocking the token forwarder. Session lock held for full generate turn so concurrent WS clients serialize FIFO. WS stays open across turns.

Old `/v1/saklas/*` routes removed with no aliases.

## ollama.py

Ollama-compatible shim mounted by `register_ollama_routes(app)`, reusing `session` / `default_steering` / `session.lock` / auth. Advertises `/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/pull` (no-op success for loaded model / 404 otherwise), + 501 stubs for `/api/push`, `/api/create`, `/api/copy`, `/api/delete`, `/api/embeddings`, `/api/embed`.

NDJSON streaming (`application/x-ndjson`) matches Ollama wire format. `/api/show.template` reflects real HF Jinja `tokenizer.chat_template` (honest over useful-looking). `/api/generate` omits the `context` field (vs an empty list saklas can't round-trip).

**Model aliasing** is hybrid:
- `_HF_TO_OLLAMA_ALIASES` overrides where Ollama's catalogue rounds differently (Gemma-2-2b is 2.6B but Ollama advertises `gemma2:2b`) or where `model_type` lacks a version suffix (Llama).
- `_infer_aliases(session)` falls back to `<family>:<size>` from `session.model_info`. `_normalise_family` strips `_text`/`_moe`/`forcausallm`; `_size_tag` rounds ≥10B to integer B, <10B keeps one decimal.
- Overrides win; inference fills gaps for new architectures.

**Option translation** (`_resolve_options`): `temperature`, `top_p`, `top_k`, `seed`, `num_predict`→`max_tokens`, `stop`, `presence_penalty`, `frequency_penalty`, `repeat_penalty`, `steer`. `repeat_penalty` maps to `presence_penalty` via `ln(repeat_penalty)` — equivalent to Ollama's "divide positive logits by penalty" (count-independent, bounded). Anything else in `options` (`min_p`, `mirostat*`, `num_ctx`, `typical_p`, …) logged at debug and silently skipped.

**Steering passthrough** via non-standard `steer` field inside `options` (or top-level): a steering expression string using the shared grammar (`"0.5 honest + 0.3 warm@after"`). Composes over `default_steering` at the key level (per-request overrides default). Non-string payloads raise a clear error. Clients that don't know about it pass through unchanged — Open WebUI / Enchanted / LangChain's `ChatOllama` work out of the box.

**Thinking** streams as `message.thinking` (chat) / top-level `thinking` (generate). `_duration_stats` splits wall time proportionally between `prompt_eval_duration` + `eval_duration` in ns. `_finish_to_done_reason` maps saklas `stop_sequence` → Ollama `stop`.
