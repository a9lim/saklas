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
to sub-module registrars (and owns the shared request bodies + serializer helpers
those registrars import):

- `manifold_routes.register_manifold_routes` — `/saklas/v1/manifolds/*`
- `template_routes.register_template_routes` — `/saklas/v1/templates/*` (templated-completion artifact + scorer)
- `session_routes.register_session_routes` — `/saklas/v1/sessions` CRUD + clear/rewind
- `tree_routes.register_tree_routes` — the loom `/sessions/{id}/tree/*` routes
- `vector_routes.register_vector_routes` — the `/sessions/{id}/vectors/*` routes + `/sessions/{id}/correlation`
- `probe_routes.register_probe_routes` — `/sessions/{id}/probes/*` (unified: list / defaults / attach / detach — every probe shape)
- `experiment_routes.register_experiment_routes` — `/sessions/{id}/experiments/fan`
- `traits_routes.register_traits_routes` — `/sessions/{id}/traits/stream` (SSE)
- `lens_routes.register_lens_routes` — `/sessions/{id}/lens/*` (token readout +
  the live-lens toggle)
- `ws_stream.register_ws_stream` — the `WS /sessions/{id}/stream` co-stream engine

`server/sse.py`, `server/streaming.py`, and `server/ws_events.py` are the shared
SSE / WS plumbing. `server/streaming.py` owns the end-of-stream derivation shared by
all three streaming protocols — `stream_finalizer(session, result)` yields
`(finish_reason, usage, probe_agg)`, and `probe_reading_aggregate(session, result)` /
`_usage_dict(result)` are its result-parameterized parts (the SSE, NDJSON, and WS
finalization sites each format the triple into their own wire shape).

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

Probe readings ride each choice under the vendor-prefixed
`x-saklas-probe-readings` extension, keyed by attached probe name. Both the
aggregate and each streamed per-token chunk carry the same `ProbeReading.to_dict()`
shape (the unification: aggregate is the reading pooled at the last-content token).
Non-streaming carries the aggregate; each streamed chunk carries the per-token
reading when at least one probe is attached and `live_scores` is on; the final
chunk carries the aggregate. The field is omitted entirely when no probe is
attached, so OpenAI clients that don't read the extension see no shape change.
`_probe_reading_aggregate(session)`
and `_probe_token_readings(event)` are imported aliases over
`server.request_helpers.probe_reading_aggregate` /
`server.request_helpers.probe_token_readings`, shared with `ollama.py`.

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

## Native route modules and schemas

`saklas_api.py` is now only the native-route registrar plus a backcompat re-export
surface for old imports. New route-specific request bodies and serializers live
beside their route groups:

- `native_common.py` — single-session id resolution.
- `session_models.py` — session request bodies and `session_info`.
- `vector_models.py` — vector/extract/bake request bodies and vector serializers.
- `tree_models.py` — loom-tree request bodies and tree serializers.
- `ws_models.py` — WebSocket request bodies, sampling conversion, token/result helpers.
- `experiment_models.py` — experiment request bodies.

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
  and returns the existing session). `session_info` carries `is_base_model` plus
  `role_substitution_supported` / `user_role_supported` (against `ROLE_HEADERS` /
  `USER_ROLE_HEADERS` for the resolved `model_type`) and the resolved
  `default_assistant_role` / `default_user_role`, so the webui can gate roles,
  plus `jlens_fitted` (a `lens_paths` existence check gating the drilldown's
  j-lens tab — deliberately not the lazy `session.jlens` load),
  `live_lens_layers` (the live workspace readout's resolved layer list, `null`
  while off; coerced so stub sessions read as off), and `live_probe_scores`
  (the CAA live toggle; coerced so stub sessions read as the default-on).
- `GET/PATCH/DELETE /saklas/v1/sessions/{id}` — info / update defaults / no-op 204.
- `POST /saklas/v1/sessions/{id}/{clear,rewind}`.

### manifold_routes.py

`_manifold_json` is the wire serializer behind every detail-returning route. It
builds the *session-independent* fields via `io.manifolds.manifold_summary(folder)`
— the same serializer CLI `pack show -j` emits — so the shared keys
(`namespace`/`name`/`description`/`source`/`tags`/`fit_mode`/`is_discover`/`node_count`/
`node_labels`/`node_roles`/`node_kinds`/`hyperparams`/`fitted_models`/`tensor_variants` + authored
`domain`/`domain_label`/`intrinsic_dim`/`min_nodes`/`node_coords`) match across CLI
and server. The summary carries `tags` (the manifold's category) on the wire. The route then layers *session-aware* extras: `fitted_for_session` /
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
  (`create_discover_manifold_folder`; nodes carry no coords; `sanitize_hyperparams`
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
  `GenerateManifoldRequest` body carries `concepts: list[str]` (≥2, one node per
  concept), `kind: "abstract"|"concrete"` + `samples_per_prompt: int` (no
  `n_scenarios`/`statements_per_concept` — A2 has no scenarios, so `scenarios.json`
  provenance is no longer written). `role_per_node=true` → persona manifold.
- `PATCH /manifolds/{ns}/{name}` — `update_manifold_folder` (serializes against an
  in-flight fit). Existing tensors go stale, not deleted.
- `DELETE /manifolds/{ns}/{name}` — `remove_manifold_folder` (single source of truth
  shared with CLI `pack rm`) under `session.lock`; 409 when a fit holds the
  gen-lock, 404 pre-lock. Response `{namespace, name, source, removed,
  rematerializes_on_restart}`.
- `POST /manifolds/{ns}/{name}/fit` — `session.fit` under the lock; SSE
  / JSON. Discover folders accept `fit_mode` / `hyperparams` overrides, written
  atomically into `manifold.json` (after `sanitize_hyperparams`) *before* the fit;
  authored folders reject them with 400. Poisedness `ValueError` →
  `code: "PoisednessError"`; `ConcurrentExtractionError` → 409. Steering a fitted
  manifold needs no route — a `%` term loads it lazily on scope entry.

The `POST /manifolds/templated` route survives as a **bridge** (back-compat for the
webui's templated-manifold builder): it writes a standalone template
(single-turn contexts from the `{user, assistant}` pairs) then a manifold that
`template_ref`-erences it. Multi-turn contexts + the scorer ride the dedicated
template routes.

### template_routes.py

The standalone templated-completion artifact (`io.templates`). Lifecycle is
pure-IO; `score` runs the loaded model. `_template_detail` serializes
`TemplateFolder.summary()` + the full `contexts`.

- `GET /templates` — list (`summary` + `namespace` per row).
- `GET /templates/{ns}/{name}` — detail incl. `contexts`. 404 on missing.
- `POST /templates` — `create_template_folder` (`CreateTemplateRequest`: slot,
  values, multi-turn `contexts:[{turns, assistant}]`, `force`). 409 on existing,
  400 on a validation failure (slot in a history turn, slot count ≠ 1, last turn
  not user, …).
- `DELETE /templates/{ns}/{name}` — `remove_template_folder`; 404 on missing.
- `POST /templates/{ns}/{name}/score` — `session.score_template` in
  `asyncio.to_thread` under `acquire_session_lock` (503 if locked). Body
  `{steering?}`; returns `{template, namespace, steering, contexts:
  [ChoiceScores.to_dict()]}` — the per-context restricted-choice value
  distribution, steering-aware. 404 on a missing/ambiguous template, 400 on a
  scoring/steering-expr failure (scrubbed to `type(e).__name__`).

### probe_routes.py

The read-side counterpart to manifold steering — one unified probe collection
under `/sessions/{id}/probes`, covering every probe shape (a 2-node concept axis is
the rank-1 case, a discover / curved fit the rank-R case). The pre-4.0 split (vector
probes by name; manifold probes by selector on a separate route) collapsed with the
monitor unification onto the session's single `Monitor`.

- `GET /probes` lists every attached probe via `session._monitor.attached_probes()`
  (`{name, manifold, top_n, layers, node_labels, node_count, domain, intrinsic_dim,
  feature_space}`). `_probe_info` also emits `is_affine` (the flat/curved
  discriminator the client classifies subspace-vs-manifold on, via
  `core.manifold.manifold_is_affine` — `core.session._manifold_is_affine` is a
  back-compat alias — defensively guarded → `False` on any read failure) and
  `node_coords` (the per-node layout backing the client mini-map,
  `null` when the manifold has none materialized).
- `GET /probes/defaults` returns the default roster.
- `POST /probes` body `{selector, name?, top_n?}` → `session.add_probe(selector,
  as_name=name, top_n=top_n)`, 201 + the probe info. The selector rides the same
  `[ns/]name[:variant]` shape `%` steering consumes — probe and steering share the
  lazy-load cache. 400 on an empty selector, 404 on `FileNotFoundError`, 400 on
  `KeyError`/`ValueError`, and any other `SaklasError` maps through
  `user_message()` (a `jlens/<word>` selector's `LensNotFittedError` → 404).
- `DELETE /probes/{name}` → `session.remove_probe`, 204; 404 if not attached
  (either roster — monitor or lens).
- **Lens probes** — a `jlens/<word>` selector lands in the session lens-probe
  registry (the READOUT channel), not the Monitor; `GET /probes` appends those
  rows via `_lens_probe_info` (shape-compatible with `_probe_info` plus the
  `lens: true` discriminator, `word`, `token_id`; `feature_space: "readout"`,
  `intrinsic_dim: 1` — the one `strength` axis, no `node_coords`),
  and `POST /probes` returns the same shape for a lens attach. `GET
  .../geometry` 404s on a lens probe (no subspace geometry behind it).
- `POST /probes/live` body `{enabled}` → `session.set_live_probe_scores` under
  the session lock — the **CAA live toggle**: off ⇒ per-token monitor scoring
  is disabled for UI/trait/loom consumers (aggregate-only capture; probe gates
  still force the subset they need). Session info reports the state as
  `live_probe_scores` (coerced default-on for stub sessions).
- `GET /probes/{name:path}/geometry` → `session._monitor.probe_geometry(name)`, the
  static per-layer geometry (centroids, neutral anchor, PCA rotation for rank≥3,
  curve/surface overlay) backing the dashboard probe-inspector plot; 404 if not
  attached.

The one-shot text-scoring endpoints (`POST .../probe`, `POST .../manifold-probe`)
were removed in 4.0 — they re-rendered arbitrary text out of conversation context
via the now-deleted `monitor.measure` / `session.measure_manifold`. Live per-token
scoring during generation is unchanged: it rides the traits SSE stream and the
probe reading extensions on the OpenAI / Ollama / WS paths, which use live hooks,
not a re-render pass.

### Vectors under `/sessions/{id}/vectors` (in `vector_routes.py`)

- `GET` list, `GET /{name}` profile JSON, `POST` load-from-disk, `DELETE /{name}`
  (also drops the name from `default_steering`).
- `GET /{name}/diagnostics` — 16-bucket `‖baked‖` histogram + per-layer magnitudes +
  the `diagnostics_by_layer` blocks when present.
- `GET /vectors/pairwise?a=&b=` — cross-layer **whitened** cosine matrix between two
  named vectors / probes. Mahalanobis-only (no `metric` param, no Euclidean path):
  whitened cosine is single-layer, so each cell is whitened in `a`'s row-layer frame.
  `session.whitener` must cover every row-layer of `a`, else 409 (regenerate the
  neutral cache). Registered *before* `GET /vectors/{name}` so the literal path wins.
- `POST /extract` — in `asyncio.to_thread`; SSE / JSON. `coerce_corpora`
  normalizes `source`: a concept name routes to `session.extract` (a composite
  name fits a 2-node `pca`; a monopolar name with no baseline fits the 1-node
  neutral-anchored ray), while two pole corpora (`{positive, negative}` /
  `{pairs: [...]}` / a bare single `{positive, negative}`) route to
  `session.extract_vector_from_corpora` and land a 2-node `pca` manifold.
  `namespace` controls the destination; `force` bypasses the tensor cache;
  `auto_register` (wire field `register`, default true) steers the result in as a
  vector on success. There is no `/extract/preview` (the A0
  scenario/preview machinery was removed — A2 has no scenarios).
- `POST /vectors/bake` (`BakeVectorRequest`) body `{name, expression}` — wraps
  `merge_into_manifold` (model-scoped, `force=True`): lands a corpus-less baked
  manifold, folds the fitted tensor back to a steering Profile, registers it. The
  server mirror of CLI `manifold bake`. `_refuse_if_busy` first (409).
  `MergeError` → 400. (Cloning was removed in 4.0 — no `/vectors/clone` route.)

`GET /sessions/{id}/correlation?names=…` — N×N Mahalanobis-cosine matrix across
loaded steering vectors and active probes (a steering vector wins a name collision
over a same-named probe). Mahalanobis-only: passes `session.whitener` to
`cosine_similarity`; a missing whitener is 409, and a pair the whitener doesn't
fully cover lands as `null`. Default covers everything; `names` restricts.

### Loom tree (in `tree_routes.py`)

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

### lens_routes.py — Jacobian-lens routes

`POST /sessions/{id}/lens/token/validate` body `{word}` — a read-only tokenizer
check for the dashboard's J-lens STEER and PROBE add forms. Returns
`{word, token_id}` only when the word round-trips as exactly one vocabulary
token; multi-token words return 400. It never registers a direction or probe;
the engine registration boundaries still revalidate the invariant.

`GET /sessions/{id}/lens/token-readout?node_id=&raw_index=[&top_k=8][&steered=
true][&raw=false][&layers=csv]` — the workspace readout at one decode step of a
loom node (`session.jlens_token_readout` in `asyncio.to_thread` under
`acquire_session_lock`, 503 on timeout): the per-layer J-lens top-k matrix at
the forward that produced the clicked token, each row
`{layer, in_band, tokens:[{token, id, logprob}]}` sorted ascending (`in_band` =
the 40–90% workspace band), plus the layer-aggregated `aggregate:
[{token, strength, com, spread}]` block (per-layer softmax → mean band
probability + salience-weighted depth center of mass; band-restricted,
strength-descending; `[]` from a session dict without the key). `steered` (default on) replays under the node's
recipe steering — `steered=false` is the unsteered counterfactual; `raw` marks
a flat-buffer node (raw-ness isn't stamped server-side, the client's render
mode supplies it). Errors: `LensNotFittedError`/`UnknownNodeError` → 404,
`InvalidNodeOperationError`/bad `layers`/`top_k` → 400, other `SaklasError`s →
their `user_message()` status. Discovery rides
the session-info `jlens_fitted` field (a `lens_paths` existence check — never
the ~GB lazy artifact load).

`POST /sessions/{id}/lens/live` body `{enabled, layers?, top_k?=5}` — toggle
the **live** workspace readout (`session.enable_live_lens` /
`disable_live_lens` under `acquire_session_lock`, so it never races an
in-flight stream and applies to the next generation). Enabling moves the
selected layers' `J_l` device-resident; `layers` omitted enables every
fitted layer in the 40–90% band (the TUI `/lens` default). Returns `{enabled,
layers}` (the resolved list). While enabled, the native WS `token` frame
carries the per-step matrix as `lens_readout` (see ws_stream below) and
session info reports the layer list as `live_lens_layers` (`null` while off —
the dashboard's WORKSPACE-panel rehydration read). Errors:
`LensNotFittedError` → 404, bad `layers` → 400, `top_k` outside `[1, 50]` →
400. `saklas serve` auto-enables the live lens at startup when the artifact
exists (`_run_serve`, `top_k=8`), so the dashboard opens hot; the toggle
still disables per session.

`POST /sessions/{id}/lens/fit` body `{prompts?=100, seq_len?, layers?=
"workspace", force?=false}` — kick off the **background lens fit** (the
dashboard's "fit j-lens" button; the former CLI-only policy). 202 + the
initial status; one fit at a time (409 while running); `layers="sample"`
rejected (not fittable). The job streams the default fineweb-edu corpus
(`io.lens.stream_default_lens_corpus` — needs the optional `datasets` dep,
else a clean typed error), runs `session.fit_jlens` (resume-by-default,
checkpointed) in a worker thread, and auto-enables the full-band live lens
when the artifact lands. Deliberately **polled, not SSE**: the fit is hours
of wall clock and progress must survive page reloads. `GET
/sessions/{id}/lens/fit` returns `{running, prompts_done, prompts_total,
message, error, started_at, finished_at, live_layers}` (per-prompt counts
parsed from the engine's `prompt N/M` progress lines). Error text follows
the SSE scrubbing discipline (typed `user_message()`, else exception type
only). Generations attempted while the fit holds the model raise through
the ordinary busy path — surfaced client-side by the sticky WS error toast.

### traits_routes.py — live traits SSE

`GET /sessions/{id}/traits/stream` — per-token probe scores in real time during any
active generation, via inline `Monitor.score_single_token` gated behind
registered trait queues (zero overhead when no client is connected). Stays open
across generations; multiple clients supported. Events: `start`
(`{generation_id}`), `token` (`{idx, text, thinking, probes}`), `done`
(`{generation_id, finish_reason, aggregate}`), `: heartbeat` every 15 s when idle.

### WS /saklas/v1/sessions/{id}/stream (in `ws_stream.py`)

Bidirectional WebSocket; only `session_id == "default"` is reachable (HF ids contain
`/`). Client → server: `{type: "stop"}`, or `{type: "generate", input, steering,
sampling, thinking, stateless, raw, parent_node_id?, n?, recipe_override?}`. The
`sampling` block (`WSSamplingParams` → `build_sampling` → `SamplingConfig`) carries
`user_role`/`assistant_role` (the per-message role-substitution labels, stamped
onto the produced loom nodes and rendered faithfully per-turn), plus `return_top_k`
(per-request top-K-alts override) and `persist_per_layer_scores` (the per-layer
heatmap opt-in that gates the `per_layer_scores` token channel). Special generate
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
when captured, `scores`/`per_layer_scores` when probes are loaded, `probe_readings`
`Record<name, {fraction, nearest}>` when any probe is attached, computed
inline off `session._capture._per_layer`, and `lens_readout`
`Record<layerStr, [token, score][]>` + `lens_aggregate`
`[token, strength, com, spread][]` while the live lens is on — the WS
`_on_token` stamps `_saklas_wants_lens_readout` so the engine computes the
step readout, and `build_token_event` copies the token tap's `lens` /
`lens_aggregate` slots onto
the frame), `done` (`result` with `text`, `tokens`,
`finish_reason`, `usage`, `per_token_probes`, `mean_logprob`, `mean_surprise`,
`probe_readings` aggregate), `error` (validation errors keep the connection open;
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
`SAKLAS_STRICT_MODEL` 404s a `model` mismatch. Probe readings ride under
the top-level `x-saklas-probe-readings` key (non-streaming = aggregate, each
NDJSON chunk = per-token reading, final chunk = aggregate); absent when no probe is
attached.
