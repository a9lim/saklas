# server/

Dual-protocol HTTP on one port: OpenAI `/v1/*`, Ollama `/api/*`, native
`/saklas/v1/*`. One model per server; generation across all three protocols
serializes on a single `asyncio.Lock`. There is **no `/saklas/v1/packs*` surface**
— concepts are manifolds, so distribution rides the manifold routes (install) and
the vectors-under-session routes (extract/merge); HF upload stays CLI-only.

## Module map

`app.py` registers the OpenAI routes inline, then calls
`register_ollama_routes(app)` (`ollama.py`) and `register_saklas_routes(app)`
(`saklas_api.py`), then mounts the Svelte SPA last (so its catch-all can't shadow
the API). `register_saklas_routes` is the native-tree orchestrator — it delegates
to sub-module registrars and registers a few route groups in place:

- `manifold_routes.register_manifold_routes` — `/saklas/v1/manifolds/*`
- `manifold_probe_routes.register_manifold_probe_routes` — `/saklas/v1/manifold-probes/*`
- `session_routes.register_session_routes` — `/saklas/v1/sessions` CRUD + clear/rewind
- `probe_routes.register_probe_routes` — `/sessions/{id}/probes/*` (list / defaults / activate / deactivate)
- `experiment_routes.register_experiment_routes` — `/sessions/{id}/experiments/fan`
- `traits_routes.register_traits_routes` — `/sessions/{id}/traits/stream` (SSE)
- in place: the loom `tree/*` routes, the `/sessions/{id}/vectors/*` routes, and
  the `WS /sessions/{id}/stream`

`server/openai.py` is a re-export facade over `app.py`. `server/sse.py` and
`server/ws_events.py` are the shared SSE / WS plumbing.

## app.py (OpenAI)

FastAPI factory + OpenAI handlers. `create_app(session, default_steering=None,
cors_origins=None, api_key=None, *, web=False)` — `web=True` mounts the dashboard
(`saklas serve` default-on, `--no-web` and library callers off). `default_steering`
is a pre-built `Steering` or `None`; per-request steering expressions compose over
it at the key level (request keys win, default-only keys pass through, explicit
empty string clears).

OpenAI routes: `GET /v1/models`, `GET /v1/models/{id}`, `POST
/v1/chat/completions`, `POST /v1/completions`. Thin HTTP — handlers call
`session.generate` / `generate_stream` with `SamplingConfig` + `Steering`, never
mutating `session.config`.

`_SamplingBase` (pydantic, shared by chat/completions): `stop`, `seed`,
`logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs` (bool for chat /
int for completions), `top_logprobs`, `stream_options.include_usage`,
`max_completion_tokens` (aliased onto `max_tokens`), native `steering` field (a
steering expression string), native `thinking` field (`None` = auto via
`supports_thinking`). Steering is a string only — dict payloads are rejected at the
pydantic layer.

Accept-and-ignore: `user`, `n`, `response_format: {"type": "text"}`, empty `tools:
[]` / `tool_choice` in `{"none", "auto"}`. Non-empty `tools`, `tool_choice` outside
`none`/`auto`, and non-`text` `response_format` are rejected via
`_check_langchain_compat` → 400. `ChatMessage._flatten_content` concatenates text
parts of OpenAI multimodal arrays; non-text parts raise `UnsupportedContentError`.

Responses carry real `usage`, `finish_reason` from
`session._gen_state.finish_reason`, per-request `created`, and a `probe_readings`
block. `_stream_generation` emits a first-chunk `{role: "assistant"}` delta, takes
`finish_reason` from gen state on the final chunk, and emits an optional usage
chunk before `[DONE]`. Thinking tokens stream as `reasoning_content`.
`_render_logprobs_chat` / `_render_logprobs_completions` build the two OpenAI
logprobs shapes from `result.logprobs` (alt text off `TokenAlt`, not a re-tokenize).

Manifold-probe readings ride each choice under the vendor-prefixed
`x-saklas-manifold-readings` extension, keyed by attached probe name. Non-streaming
carries the aggregate (`ManifoldAggregate.to_dict()`); each streamed chunk carries
the per-token reading (`ManifoldTokenReading.to_dict()`) when at least one probe is
attached and `live_scores` is on; the final chunk carries the aggregate. The field
is omitted entirely when no manifold probe is attached, so OpenAI clients that
don't read the extension see no shape change. `_manifold_reading_aggregate(session)`
and `_manifold_token_readings(event)` are the shared helpers (also imported by
`ollama.py`).

Auth: bearer token from `SAKLAS_API_KEY` / `--api-key`, applied as an app-level
dependency over HTTP and WebSocket routes. `_require_auth` + `_check_bearer` gate
HTTP; `ws_auth_ok(websocket)` is called before `websocket.accept()` (close 1008 on
fail) and accepts either an `Authorization: Bearer ...` header or the
browser-dashboard fallback `?token=...`. Unset key = open server.
`SAKLAS_STRICT_MODEL` (`1`/`true`/`yes`/`on`) makes the `model` field 404 on a name
mismatch across OpenAI and Ollama; unset accepts any name.

`acquire_session_lock(session)` is a bounded (`SESSION_LOCK_TIMEOUT_SECONDS` = 300)
async context manager over `session.lock`. Non-streaming handlers take it plainly;
streaming handlers hold it for the full stream and emit a 503 on timeout. Requests
queue FIFO. `session.lock` (`asyncio.Lock`, server-owned) is distinct from the
threading `_gen_lock` inside the engine. `_on_saklas_error` maps any `SaklasError`
to an HTTP status and picks the Ollama vs OpenAI error shape by path prefix;
`RequestValidationError` maps to the OpenAI shape. Not supported by either compat
protocol: tool calling, JSON-schema/structured-output mode, embeddings.

## saklas_api.py (native tree orchestrator + in-place routes)

URL paths carry `{session_id}` for a multi-session shape, but the impl is
single-session: the one session has id `"default"`, and the loaded model id also
resolves to it; everything else 404s.

**SSE error-frame scrubbing (info-disclosure discipline).** Long-running SSE
workers (`/extract`, manifold `generate` / `fit`) must NOT surface raw `str(e)`
in the terminal `error` frame — Python exception strings routinely echo
filesystem paths and traceback fragments. The catch-all `except Exception` logs
the full traceback server-side and sends the generic `{"message": "<op> failed",
"code": type(e).__name__}`. Typed branches that already send a safe message
(`SaklasError.user_message()`, the manifold poisedness
`ValueError`/`ConcurrentExtractionError`) keep their messages.
`server/sse.py::progress_sse_response` is the shared queue-driven SSE worker — it
owns the progress/done/error frame loop, the `call_soon_threadsafe` bridge,
cancellation cleanup, and the generic catch-all; route modules supply only the job
body + any typed safe-message formatter.

### session_routes.py

- `GET/POST /saklas/v1/sessions` — list / idempotent create (a model mismatch warns
  and returns the existing session). `_session_info` carries `is_base_model` plus
  `role_substitution_supported` / `user_role_supported` (against `ROLE_HEADERS` /
  `USER_ROLE_HEADERS` for the resolved `model_type`) so the webui can gate roles.
- `GET/PATCH/DELETE /saklas/v1/sessions/{id}` — info / update defaults / no-op 204.
- `POST /saklas/v1/sessions/{id}/{clear,rewind}`.

### manifold_routes.py

`_manifold_json` is the wire serializer behind every detail-returning route. It
builds the *session-independent* fields via `io.manifolds.manifold_summary(folder)`
— the same serializer CLI `manifold show -j` emits — so the shared keys
(`namespace`/`name`/`description`/`source`/`fit_mode`/`is_discover`/`node_count`/
`node_labels`/`node_roles`/`hyperparams`/`fitted_models`/`tensor_variants` + authored
`domain`/`domain_label`/`intrinsic_dim`/`min_nodes`/`node_coords`) match across CLI
and server. The route then layers *session-aware* extras: `fitted_for_session` /
`stale`, and — for a discover folder fitted on the loaded model — the materialized
per-model geometry (`domain`/`intrinsic_dim`/`node_coords`) lifted from the
per-model sidecar/tensor via `_resolve_intrinsic_dim` + a `load_manifold` read.

- `GET /manifolds` — every installed manifold (domain spec, intrinsic dim, node
  labels/coords, `min_nodes`, `fit_mode`, `hyperparams`, `fitted_for_session`,
  `stale`). Unfitted discover folders report `domain == {}` / `intrinsic_dim == 0`;
  `node_coords == []` at list level (lives on the detail route).
- `GET /manifolds/{ns}/{name}` — one manifold + per-node statements + per-tensor fit
  detail (discover fits carry `fit_mode` + `hyperparams` + `diagnostics`; discover
  nodes carry the derived per-model coords or `null`). 404 on missing.
- `GET /manifolds/search?q=&limit=` — HF-hub search proxy via
  `io.hf_manifolds.search_manifolds`; `{query, results}`. Missing
  `huggingface_hub` → 503, HF transport error → 502.
- `POST /manifolds` — author an *authored* artifact (`create_manifold_folder`;
  box/sphere domain; per-node `role` optional). Returns detail + `advisories`. 409
  on existing folder, 400 malformed.
- `POST /manifolds/discover` — author a *discover* artifact
  (`create_discover_manifold_folder`; nodes carry no coords; `_sanitize_hyperparams`
  drops cross-method keys).
- `POST /manifolds/merge` — `merge_discover_manifolds` (discover-mode sources only:
  authored geometry isn't mergeable without a shared coordinate system). Pools node
  corpora + roles into one unfitted discover folder; pair with `/fit`. Label
  collisions / mixed `fit_mode` without override / <2 sources / dest conflict →
  400/409.
- `POST /manifolds/install` — `io.hf_manifolds.install_manifold` in a worker thread
  under `session.lock`. `InstallManifoldRequest` carries `as` as a true wire field
  (`Field(alias="as")` + `populate_by_name`). `_refuse_if_busy(session)` first (409
  on an in-flight engine gen-lock). `ManifoldInstallConflict` → 409, missing → 404,
  bad input → 400, `huggingface_hub` missing → 503, HF error → 502. Returns the same
  detail JSON as `GET .../{ns}/{name}`. There is deliberately **no install/push
  parity gap to worry about**: `push_manifold` is wired only into CLI `manifold
  push` — HF upload stays CLI-only, no `POST .../push` route.
- `POST /manifolds/generate` — LLM-authors a discover folder via
  `session.generate_responses` (A2 conversational extraction — each concept
  answers the shared baseline prompts in character, one corpus per node) under
  the session lock; SSE progress on `Accept: text/event-stream`, JSON otherwise.
  `GenerateManifoldRequest` body carries `kind: "abstract"|"concrete"` +
  `samples_per_prompt: int` (no `n_scenarios`/`statements_per_concept` — A2 has
  no scenarios, so `scenarios.json` provenance is no longer written).
  `role_per_node=true` → persona manifold.
- `PATCH /manifolds/{ns}/{name}` — `update_manifold_folder` (serializes against an
  in-flight fit). Existing tensors go stale, not deleted.
- `DELETE /manifolds/{ns}/{name}` — `remove_manifold_folder` (single source of truth
  shared with CLI `manifold rm`) under `session.lock`; 409 when a fit holds the
  gen-lock, 404 pre-lock. Response `{namespace, name, source, removed,
  rematerializes_on_restart}`.
- `POST /manifolds/{ns}/{name}/fit` — `session.extract_manifold` under the lock; SSE
  / JSON. Discover folders accept `fit_mode` / `hyperparams` overrides, written
  atomically into `manifold.json` (after `_sanitize_hyperparams`) *before* the fit;
  authored folders reject them with 400. Poisedness `ValueError` →
  `code: "PoisednessError"`; `ConcurrentExtractionError` → 409. Steering a fitted
  manifold needs no route — a `%` term loads it lazily on scope entry.

### manifold_probe_routes.py

Read-side counterpart to manifold steering. `GET /manifold-probes` lists every
attached probe (`{name, manifold, top_n, layers, node_labels, node_count, domain,
intrinsic_dim, feature_space}`). `POST /manifold-probes` body `{selector, name?,
top_n?}` wraps `session.add_manifold_probe` (selector rides the same
`[ns/]name[:variant]` shape `%` steering consumes — probe and steering share the
lazy-load cache). `DELETE /manifold-probes/{name}` detaches.

### probe_routes.py

Vector probes under `/sessions/{id}/probes`: list / defaults / activate / deactivate.
The one-shot text-scoring endpoints (`POST .../probe`, `POST .../manifold-probe`)
were removed in 4.0 — they re-rendered arbitrary text out of conversation context
via the now-deleted `monitor.measure` / `session.measure_manifold`. Live per-token
scoring during generation is unchanged: it rides the traits SSE stream and the
manifold-probe reading extensions on the OpenAI / Ollama / WS paths, which use
live hooks, not a re-render pass.

### Vectors under `/sessions/{id}/vectors` (in `saklas_api.py`)

- `GET` list, `GET /{name}` profile JSON, `POST` load-from-disk, `DELETE /{name}`
  (also drops the name from `default_steering`).
- `GET /{name}/diagnostics` — 16-bucket `‖baked‖` histogram + per-layer magnitudes +
  the `diagnostics_by_layer` blocks when present.
- `GET /vectors/pairwise?a=&b=` — cross-layer **whitened** cosine matrix between two
  named vectors / probes. Mahalanobis-only (no `metric` param, no Euclidean path):
  whitened cosine is single-layer, so each cell is whitened in `a`'s row-layer frame.
  `session.whitener` must cover every row-layer of `a`, else 409 (regenerate the
  neutral cache). Registered *before* `GET /vectors/{name}` so the literal path wins.
- `POST /extract` — in `asyncio.to_thread`; SSE / JSON. `_coerce_corpora`
  normalizes `source`: a concept name routes to `session.extract` (a composite
  name fits a 2-node `pca`; a monopolar name with no baseline fits the 1-node
  neutral-anchored ray), while two pole corpora (`{positive, negative}` /
  `{pairs: [...]}` / a bare single `{positive, negative}`) route to
  `session.extract_vector_from_corpora` and land a 2-node `pca` manifold.
  `namespace` controls the destination; `force`
  bypasses the tensor cache. There is no `/extract/preview` (the A0
  scenario/preview machinery was removed — A2 has no scenarios).
- `POST /vectors/merge` body `{name, expression}` — wraps `merge_into_manifold`
  (model-scoped, `force=True`): lands a corpus-less baked manifold, folds the fitted
  tensor back to a steering Profile, registers it. `_refuse_if_busy` first (409).
  `MergeError` → 400. (Cloning was removed in 4.0 — no `/vectors/clone` route.)

`GET /sessions/{id}/correlation?names=…` — N×N Mahalanobis-cosine matrix across
loaded steering vectors and active probes (a steering vector wins a name collision
over a same-named probe). Mahalanobis-only: passes `session.whitener` to
`cosine_similarity`; a missing whitener is 409, and a pair the whitener doesn't
fully cover lands as `null`. Default covers everything; `names` restricts.

### Loom tree (in `saklas_api.py`)

`/sessions/{id}/tree`: full-tree GET, active-path GET, and navigate / edit / branch /
delete / star / note / reset mutations, plus `edge_label`, `filter`, branch `diff`,
`joint_logprobs`, and `transcript` / `transcript/load`. Mutations run the tree's
conflict checks (409 when they would corrupt an in-flight generation).

### experiment_routes.py

`POST /sessions/{id}/experiments/fan` — JSON alpha grid over one prompt. Body
`{prompt, grid: {name: [alphas]}, base_steering?, sampling?, thinking?, raw?}`. The
grid is validated server-side (empty → 400), then `session.generate_sweep(...,
stateless=False)` runs in a worker thread under the lock. Returns `{kind, total,
node_ids, rows}`.

### traits_routes.py — live traits SSE

`GET /sessions/{id}/traits/stream` — per-token probe scores in real time during any
active generation, via inline `TraitMonitor.score_single_token` gated behind
registered trait queues (zero overhead when no client is connected). Stays open
across generations; multiple clients supported. Events: `start`
(`{generation_id}`), `token` (`{idx, text, thinking, probes}`), `done`
(`{generation_id, finish_reason, aggregate}`), `: heartbeat` every 15 s when idle.

### WS /saklas/v1/sessions/{id}/stream (in `saklas_api.py`)

Bidirectional WebSocket; only `session_id == "default"` is reachable (HF ids contain
`/`). Client → server: `{type: "stop"}`, or `{type: "generate", input, steering,
sampling, thinking, stateless, raw, parent_node_id?, n?, recipe_override?}`. The
`sampling` block (`WSSamplingParams` → `_build_sampling` → `SamplingConfig`) carries
`user_role`/`assistant_role` — the per-message role-substitution labels, stamped
onto the produced loom nodes and rendered faithfully per-turn. Special generate
modes:
- **Logit fork** (`fork_node_id`/`fork_raw_index`/`fork_alt_token_id`) →
  `session.fork_from_token`: replays the source node's raw decode prefix, forces the
  alt token, resamples under the node's stamped recipe. All three fork fields must
  travel together (400 otherwise).
- **Answer-prefill** (`prefill_node_id`/`prefill_text`) → `session.prefill_assistant`:
  the seeded assistant reply lands as a sibling under a user node. `thinking` forced
  off.
- **Commit (no-generation send)** (`commit_role`/`commit_text`) →
  `session.append_user_turn` / `append_assistant_turn`. `raw=true` lifts the
  user-under-user guard. Emits one `started` (node_id=null) + one `done` carrying the
  new node under `result.{kind="commit", role, text, node_id}`. No token frames.
- **Recipe override** — a built-in mode string (`unsteered`/`inverted`/`reseed`/
  `cool`/`hot`) or a partial-recipe expression (`seed=42, temperature=1.5`).

Fork / prefill / commit are mutually exclusive (400 on mix). `n>1` fans out N sibling
assistant nodes on one shared user parent, generated serially with deterministic
derived seeds. Server → client: `started` (node_id filled lazily by the first
token), `node_created`, `tree_mutated`, `token` (per token — `logprob`/`top_alts`
when captured, `scores`/`per_layer_scores` when probes are loaded, `manifold_readings`
`Record<name, {fraction, nearest}>` when any manifold probe is attached, computed
inline off `session._capture._per_layer`), `done` (`result` with `text`, `tokens`,
`finish_reason`, `usage`, `per_token_probes`, `mean_logprob`, `mean_surprise`,
`manifold_readings` aggregate), `error` (validation errors keep the connection open;
other failures close 1011).

Concurrency: one perpetual reader task owns `receive_json()` and feeds a shared
`incoming` queue; `tree_mutated`/`node_created` ride a connection-level
`LoomMutated` subscription; all sends go through one `asyncio.Lock`. Per generate
turn, `generate_stream` runs in a worker thread; `on_token` bridges to asyncio via
`call_soon_threadsafe`; the handler races the token queue against `incoming` so an
in-flight `{type: "stop"}` calls `session.stop()` without blocking. Non-stop frames
mid-generation hold in a per-connection deferred deque and drain after the turn.
`session.lock` is held for the full N-way batch so concurrent WS clients serialize
FIFO.

## ollama.py

Ollama-compatible shim (`register_ollama_routes`), reusing `session` /
`default_steering` / `session.lock` / app-level auth. Routes: `/api/version`,
`/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/pull`
(no-op success for the loaded model, 404 otherwise), `HEAD /`, and 501 stubs for
`/api/push`, `/api/create`, `/api/copy`, `/api/delete`, `/api/embeddings`,
`/api/embed`.

Streaming responses are NDJSON. `/api/show.template` reflects the real HF Jinja
`tokenizer.chat_template`. `/api/generate` omits `context` (saklas can't round-trip
it). Model aliasing is hybrid: `_HF_TO_OLLAMA_ALIASES` overrides where Ollama's
catalogue rounds differently or `model_type` lacks a version suffix; otherwise
`_infer_aliases` falls back to `<family>:<size>` from `session.model_info`.
`_resolve_options` recognizes `temperature`, `top_p`, `top_k`, `seed`,
`num_predict`→`max_tokens`, `stop`, `presence_penalty`, `frequency_penalty`,
`repeat_penalty` (→ `presence_penalty` via `ln(repeat_penalty)`), `steer`;
everything else (`min_p`, `mirostat*`, …) is logged at debug and dropped. Steering
passes through a non-standard `steer` field inside `options` (or top-level) — a
steering expression composed over `default_steering` at the key level; non-string
`steer` raises a clean 400 before headers flush. A top-level `think` bool toggles
thinking, streamed as `message.thinking` (chat) / top-level `thinking` (generate).
`_duration_stats` splits wall time between `prompt_eval_duration` and
`eval_duration`; `_finish_to_done_reason` maps `stop_sequence` → `stop`.
`SAKLAS_STRICT_MODEL` 404s a `model` mismatch. Manifold-probe readings ride under
the top-level `x-saklas-manifold-readings` key (non-streaming = aggregate, each
NDJSON chunk = per-token reading, final chunk = aggregate); absent when no probe is
attached.
