# server/

Dual-protocol HTTP on one port: OpenAI `/v1/*`, Ollama `/api/*`, native `/saklas/v1/*`. One model per server; generation across all three protocols serializes on a single `asyncio.Lock`.

## app.py

FastAPI factory + OpenAI route handlers. `create_app(session, default_steering=None, cors_origins=None, api_key=None, *, web=False)` registers the OpenAI routes, then calls `register_ollama_routes` and `register_saklas_routes`; mounts the Svelte SPA last (after the API routes so its catch-all can't shadow them) when `web=True` (`saklas serve` default-on, `--no-web` and library callers off). `default_steering` is a pre-built `Steering` or `None`; per-request steering expressions compose over it at the key level (request keys win, default-only keys pass through, explicit empty string clears).

OpenAI routes: `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions`, `POST /v1/completions`. Thin HTTP — handlers call `session.generate` / `generate_stream` with `SamplingConfig` + `Steering` directly, never mutating `session.config`.

`_SamplingBase` (pydantic, shared by chat/completions): `stop`, `seed`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs` (bool for chat / int for completions), `top_logprobs`, `stream_options.include_usage`, `max_completion_tokens` (aliased onto `max_tokens` via model validator), native `steering` field (a steering expression string in the shared grammar), native `thinking` field (`None` = auto via `supports_thinking`). Steering is a string only — dict-shaped payloads are rejected at the pydantic layer.

Accept-and-ignore: `user`, `n`, `response_format: {"type": "text"}`, empty `tools: []` / `tool_choice` in `{"none", "auto"}`. Non-empty `tools`, `tool_choice` outside `none`/`auto`, and non-`text` `response_format` are rejected via `_check_langchain_compat` → 400 (LangChain compat). `ChatMessage._flatten_content` concatenates text parts of OpenAI multimodal arrays; non-text parts raise `UnsupportedContentError`.

Responses carry real `usage`, `finish_reason` from `session._gen_state.finish_reason`, per-request `created`, and a `probe_readings` block. `_stream_generation` emits a first-chunk `{role: "assistant"}` delta, takes `finish_reason` from gen state on the final chunk, and emits an optional usage chunk before `[DONE]` when `stream_options.include_usage`. Thinking tokens stream as `reasoning_content` in the chat delta. `_render_logprobs_chat` / `_render_logprobs_completions` build the two OpenAI logprobs shapes from `result.logprobs` (alt text comes off `TokenAlt`, not a re-tokenize).

Auth: bearer token from `SAKLAS_API_KEY` / `--api-key`, applied as an app-level dependency over HTTP and WebSocket routes. `_require_auth` + `_check_bearer` gate HTTP; `ws_auth_ok(websocket)` is called before `websocket.accept()` in WS handlers (close 1008 on fail). Unset key = open server.

`SAKLAS_STRICT_MODEL` (`1`/`true`/`yes`/`on`) makes the `model` field 404 on a name mismatch across OpenAI and Ollama routes; unset accepts any name. The accepted set is the HF id plus any Ollama-style aliases.

`acquire_session_lock(session)` is a bounded (`SESSION_LOCK_TIMEOUT_SECONDS` = 300) async context manager over `session.lock`. Non-streaming handlers take the lock plainly; streaming handlers hold it for the full stream via `acquire_session_lock` and emit a 503 on timeout. Requests queue FIFO rather than 409. `session.lock` (`asyncio.Lock`, server-owned) is distinct from the threading `_gen_lock` inside the engine.

`_on_saklas_error` maps any `SaklasError` to an HTTP status and picks the Ollama (`{"error": msg}`) vs OpenAI error shape by path prefix. `RequestValidationError` is mapped to the OpenAI error shape.

Not supported by either compat protocol: tool calling, JSON-schema/structured-output mode, embeddings.

`server/openai.py` is a re-export facade over `app.py` so callers can `from saklas.server.openai import ChatCompletionRequest` without reaching into `app`.

## saklas_api.py

Native `/saklas/v1/*` resource tree, mounted by `register_saklas_routes(app)`. URL paths carry `{session_id}` for a multi-session shape, but the impl is single-session: the one session has id `"default"`, and the loaded model id also resolves to it; everything else 404s.

Packs (top-level, not under a session):
- `GET /saklas/v1/packs` — locally installed packs as JSON via `cache_ops.list_concepts(None, hf=False)`. Local-only, off the network.
- `GET /saklas/v1/packs/search?q=&limit=` — HF-hub search proxy via `cache_ops.search_remote_packs`; returns structured rows. Missing `huggingface_hub` → 503, HF transport error → 502.
- `POST /saklas/v1/packs` body `{target, as?, force?, statements_only?}` — wraps `cache_ops.install` in a worker thread. `target` is an HF coord `ns/name[@rev]` or local folder. `InstallConflict` → 409, `ValueError` → 400, missing target → 404.

Sessions:
- `GET/POST /saklas/v1/sessions` — list / idempotent create (POST body accepted; a model mismatch warns and returns the existing session).
- `GET/PATCH/DELETE /saklas/v1/sessions/{id}` — info / update session defaults / no-op 204.
- `POST /saklas/v1/sessions/{id}/{clear,rewind}`.

Vectors under `/sessions/{id}/vectors`:
- `GET` list, `GET /{name}` profile JSON, `POST` load-from-disk, `DELETE /{name}` (also drops the name from `default_steering`).
- `POST /extract` — runs `session.extract` in `asyncio.to_thread`; SSE progress when `Accept: text/event-stream`, JSON otherwise.
- `POST /vectors/merge` body `{name, expression}` — wraps `merge_into_pack` (model-scoped, `force=True`), loads and registers the merged profile, held under `session.lock`. Returns the same JSON as `GET /vectors/{name}`. `MergeError` → 400.
- `POST /vectors/clone` body `{name, corpus_path, n_pairs?, seed?, baseline?}` — wraps `session.clone_from_corpus` in `asyncio.to_thread` under `session.lock`; auto-registers on success. SSE branch emits only `done`/`error` (no progress hook in the clone path). Missing corpus → 404.
- `GET /vectors/{name}/diagnostics` — 16-bucket `||baked||` histogram plus per-layer magnitudes and the `diagnostics_by_layer` / `diagnostics_summary` blocks when the profile carries them. 404 when the vector isn't registered.
- `GET /vectors/pairwise?a=&b=` — cross-layer cosine matrix between two named vectors / probes. Same pool as `/correlation` (steering vectors ∪ active probes). Returns `{a, b, layers_a, layers_b, matrix, model}` where `matrix[i][j]` is the raw cosine between `a`'s `layers_a[i]` and `b`'s `layers_b[j]`; near-zero norms and shape mismatches land as `null` so the heatmap renders empty cells instead of meaningless cosines. Registered *before* `GET /vectors/{name}` so the literal path wins the routing match.

Probes under `/sessions/{id}/probes`: list / defaults / activate / deactivate. `POST /sessions/{id}/probe` body `{text, probes?}` — one-shot scoring via `monitor.measure` under the session lock, no generation.

`GET /sessions/{id}/correlation?names=a,b,c` — N×N magnitude-weighted cosine matrix across loaded steering vectors and active probes (a steering vector wins a name collision over a same-named probe). Default covers everything; `names` restricts the subset. Returns `{names, matrix, layers_shared}`.

Loom tree under `/sessions/{id}/tree`: full-tree GET, active-path GET, and navigate / edit / branch / delete / star / note / reset mutations, plus `edge_label`, `filter`, branch `diff`, `joint_logprobs`, and transcript `transcript` / `transcript/load`. Mutations run the tree's conflict checks, so edit/delete/reset return 409 when they would corrupt an in-flight generation.

### POST /sessions/{id}/experiments/fan

JSON alpha grid over one prompt. Body `{prompt, grid: {name: [alphas]}, base_steering?, sampling?, thinking?, raw?}`. The grid is validated server-side (empty dict / empty alpha list → 400), then `session.generate_sweep(..., stateless=False)` runs in a worker thread under `session.lock`. Returns `{kind, total, node_ids, rows}` where each row carries `idx`, `alpha_values`, `node_id`, and a `result` subset.

### GET /sessions/{id}/traits/stream — live traits SSE

Per-token probe scores in real time during any active generation. Uses inline per-token scoring (`TraitMonitor.score_single_token`) gated behind registered trait queues — zero overhead when no client is connected. The connection stays open across generations; multiple clients are supported. Events: `start` (`{generation_id}`), `token` (`{idx, text, thinking, probes}`), `done` (`{generation_id, finish_reason, aggregate}`), plus `: heartbeat` every 15 s when idle.

### WS /saklas/v1/sessions/{id}/stream — token + probe co-stream

Bidirectional WebSocket. Only `session_id == "default"` is reachable (HF ids contain `/` and the path param isn't `:path`).

Client → server: `{type: "stop"}`, or `{type: "generate", input, steering, sampling, thinking, stateless, raw, parent_node_id?, n?, recipe_override?}`. Three special generate modes:
- **Logit fork** — `{fork_node_id, fork_raw_index, fork_alt_token_id}` routes to `session.fork_from_token`: replays the source node's raw decode prefix, forces the alt token, resamples under the node's stamped recipe. `input`/`steering`/`sampling`/`n` are ignored; all three fork fields must travel together (else 400). The fork slices on the `raw_index` carried by each `token` event.
- **Answer-prefill** — `{prefill_node_id, prefill_text}` routes to `session.prefill_assistant`: `prefill_node_id` is a user node, `prefill_text` is tokenized into a forced decode prefix, and the seeded assistant reply lands as a sibling under it. `input` and the `fork_*` fields are ignored; `steering`/`sampling`/`n` ride through; `thinking` is forced off. Both fields must travel together; a message can't be both a fork and a prefill (400 either way).
- **Commit (no-generation send)** — `{commit_role, commit_text, parent_node_id?}` routes to `session.append_user_turn` or `session.append_assistant_turn`. `commit_role="user"` lands a user turn under `parent_node_id` (or active node when omitted); `commit_role="assistant"` lands an authored assistant turn under the user node identified by `parent_node_id` (required). Short-circuits the n-way / streaming machinery entirely: emits a single `started` (with `node_id=null`) followed by `done` carrying the new node id under `result.{kind="commit", role, text, node_id}`. No token frames in between. Mutually exclusive with fork and prefill (400 on mix). `input`/`steering`/`sampling`/`thinking`/`n` are ignored.
- **Recipe override** — `recipe_override` is a built-in mode string (`unsteered`/`inverted`/`reseed`/`cool`/`hot`) or a partial-recipe expression (`seed=42, temperature=1.5`), resolved against the parent recipe by the engine. Ignored when `None`.

`n>1` fans out N sibling assistant nodes on one shared user parent, generated serially, each with a deterministic derived seed.

Server → client events:
- `{type: "started", generation_id, node_id: null, sibling_index, sibling_count}` — `node_id` is filled in lazily by the first token.
- `{type: "node_created", node_id, parent_id, role, rev}` — emitted on `begin_assistant`/`branch`/`add_user` so the client can pre-allocate a render slot.
- `{type: "tree_mutated", op, rev, added, removed, updated, active_node_id}` — `added`/`updated` are full node objects.
- `{type: "token", text, thinking, token_id, node_id, raw_index?, logprob?, top_alts?, scores?, per_layer_scores?}` — per token. `logprob`/`top_alts` appear when the engine captured them (logprobs or `return_top_k` requested); `scores`/`per_layer_scores` appear when probes are loaded.
- `{type: "done", result, node_id, sibling_index, sibling_count}` — `result` carries `text`, `tokens`, `finish_reason`, `usage`, `per_token_probes`, `mean_logprob`, `mean_surprise`.
- `{type: "error", message, code, ...}` — a steering-expression / validation error sends an error frame and keeps the connection open; other failures close with 1011.

Concurrency: one perpetual reader task owns `websocket.receive_json()` and feeds a shared `incoming` queue (the `websockets` library forbids overlapping receives). `tree_mutated` / `node_created` events ride a connection-level `LoomMutated` subscription forwarded by a dedicated task. All sends go through one `asyncio.Lock` so two tasks can't interleave bytes. Per generate turn, `session.generate_stream` runs in a worker thread (`asyncio.to_thread`); `on_token` bridges to asyncio via `loop.call_soon_threadsafe`; the handler races the token queue against `incoming` (`asyncio.wait(FIRST_COMPLETED)`) so an in-flight `{type: "stop"}` can call `session.stop()` without blocking. `session.lock` is held for the full N-way batch so concurrent WS clients serialize FIFO. The WS stays open across turns; a mid-batch stop cancels the current sibling and skips remaining ones.

## ollama.py

Ollama-compatible shim mounted by `register_ollama_routes(app)`, reusing `session` / `default_steering` / `session.lock` / app-level auth. Routes: `/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/pull` (no-op success for the loaded model, 404 otherwise), `HEAD /` (liveness probe), and 501 stubs for `/api/push`, `/api/create`, `/api/copy`, `/api/delete`, `/api/embeddings`, `/api/embed`.

Streaming responses are NDJSON (`application/x-ndjson`), matching the Ollama wire format. `/api/show.template` reflects the real HF Jinja `tokenizer.chat_template` (honest over a fake Go template). `/api/generate` omits the `context` field (saklas can't round-trip it).

Model aliasing is hybrid: `_HF_TO_OLLAMA_ALIASES` overrides where Ollama's catalogue rounds differently (Gemma-2-2b is 2.6B but advertised `gemma2:2b`) or where `model_type` lacks a version suffix (Llama); otherwise `_infer_aliases` falls back to `<family>:<size>` from `session.model_info` (`_normalise_family` strips `_text`/`_moe`/`forcausallm`, `_size_tag` rounds ≥10B to integer B and keeps one decimal below). Overrides win; inference fills gaps.

Option translation (`_resolve_options`) recognizes `temperature`, `top_p`, `top_k`, `seed`, `num_predict`→`max_tokens`, `stop`, `presence_penalty`, `frequency_penalty`, `repeat_penalty`, `steer`. `repeat_penalty` maps to `presence_penalty` via `ln(repeat_penalty)` (Ollama's "divide positive logits by penalty", count-independent). Everything else in `options` (`min_p`, `mirostat*`, `num_ctx`, `typical_p`, …) is logged at debug and dropped.

Steering passes through a non-standard `steer` field inside `options` (or top-level): a steering expression string in the shared grammar, composed over `default_steering` at the key level. Non-string `steer` raises a clear error. Clients that don't know about it pass through unchanged, so Open WebUI / Enchanted / LangChain's `ChatOllama` work as-is. A top-level `think` bool toggles thinking.

Thinking streams as `message.thinking` (chat) / top-level `thinking` (generate). `_duration_stats` splits wall time proportionally between `prompt_eval_duration` and `eval_duration` (ns). `_finish_to_done_reason` maps saklas `stop_sequence` → Ollama `stop`. `SAKLAS_STRICT_MODEL` makes a `model` mismatch 404; option resolution is hoisted into the route handler so a bad `steer` expression returns a clean 400 before `StreamingResponse` flushes headers.
