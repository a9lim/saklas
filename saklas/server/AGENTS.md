# server/

Dual-protocol HTTP on one port: OpenAI `/v1/*`, Ollama `/api/*`, native
`/saklas/v1/*`. One model per server; generation across all three protocols
serializes on a single `asyncio.Lock`. There is **no `/saklas/v1/packs*` surface**
— concepts are manifolds, so distribution and node-union merge ride the manifold
routes while session profile routes own extract/bake; HF upload stays CLI-only.

## Module map

`app.py` registers the OpenAI routes inline, then calls
`register_ollama_routes(app)` (`ollama.py`) and `register_saklas_routes(app)`
(`native_routes.py`), then mounts the Svelte SPA last (so its catch-all can't shadow
the API). `register_saklas_routes` is the native-tree orchestrator — it delegates
to sub-module registrars (and owns the shared request bodies + serializer helpers
those registrars import):

- `manifold_routes.register_manifold_routes` — `/saklas/v1/manifolds/*`
- `template_routes.register_template_routes` — `/saklas/v1/templates/*` (templated-completion artifact + scorer)
- `session_routes.register_session_routes` — `/saklas/v1/sessions` CRUD + clear/rewind
- `tree_routes.register_tree_routes` — the loom `/sessions/{id}/tree/*` routes
- `profile_routes.register_profile_routes` — the `/sessions/{id}/profiles/*` routes + `/sessions/{id}/extract` + `/sessions/{id}/correlation`
- `probe_routes.register_probe_routes` — `/sessions/{id}/probes/*` (unified: list / defaults / attach / detach — every probe shape)
- `traits_routes.register_traits_routes` — `/sessions/{id}/traits/stream` (SSE)
- `instrument_routes.register_instrument_routes` — the unified `/sessions/{id}/instruments/*`
  family over the three read instruments (`geometry`/`lens`/`sae`): listing, per-family
  live toggle, source list/switch, the polled background-preparations resource
  (lens `fetch`/`fit`, sae `load`/`train`), token-readout replay, and the family
  extras (`token/validate`, `features/metadata`, `features/validate`). Replaces the
  former `lens_routes` / `sae_routes` groups and `POST /probes/live`.
- `ws_stream.register_ws_stream` — the `WS /sessions/{id}/stream` co-stream engine

`server/sse.py`, `server/streaming.py`, `server/ws_events.py`, and
`server/background_job.py` are the shared SSE / WS / polled-job plumbing.
`server/background_job.py::BackgroundJob` is the polled-job counterpart to
`sse.progress_sse_response`: it centralizes the scaffolding the four background
routes share (J-lens `fit` + `fetch`, SAE `train` + `load`) — the mutable
`app.state.<name>` status dict, the `app.state.<name>_task` handle, the optional
`app.state.<name>_cancel` `threading.Event`, the group mutual-exclusion 409
(`share_group` / `refuse_if_busy` — lens fetch XOR fit, sae load XOR train), the
start/status/cancel triad, `launch` (guaranteed `running`/`finished_at`
finalization), and shutdown `stop`. Those `app.state` attributes remain the live
source of truth (the job reads/writes through `app.state`, never a cached copy),
so shutdown hooks and callers that reassign them are honored. `make_progress_hook`
builds the regex progress callback (each job keeps its own vocabulary — prompts
for the fit, tokens for the train) and `scrub_job_error` carries the same
error-scrubbing discipline as the SSE worker onto the polled surface (typed
`SaklasError.user_message()`, else the exception type only; a cooperative-cancel
settles to `cancelled`). `server/streaming.py` owns the end-of-stream derivation shared by
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

`native_routes.py` registers the native route groups. Route-specific request
bodies and serializers live beside their route groups:

- `native_common.py` — exact single-session id resolution and the shared
  `NativeRequest` base. Every native request model inherits this base and rejects
  unknown fields (`extra="forbid"`); nested native objects are strict too. The
  OpenAI/Ollama protocol models remain protocol-specific.
- `session_models.py` — session request bodies and `session_info`.
- `profile_models.py` — profile/extract/bake request bodies (`ExtractRequest`, `BakeProfileRequest`) and profile serializers.
- `tree_models.py` — loom-tree request bodies and tree serializers.
- `ws_models.py` — WebSocket request bodies, sampling conversion, token/result helpers.

URL paths carry `{session_id}` for a multi-session shape, but the impl is
single-session: the one session has exactly id `"default"`; everything else 404s.

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
body + any typed safe-message formatter. The **polled** background jobs (J-lens
`fit` / `fetch`, SAE `train` / `load`) share the identical discipline via
`server/background_job.py::scrub_job_error`, which writes the scrubbed
message/error into the job's status dict instead of an SSE frame.

### session_routes.py

- `GET/POST /saklas/v1/sessions` — list / idempotent create (a model mismatch warns
  and returns the existing session). `session_info` carries `is_base_model` plus
  `role_substitution_supported` / `user_role_supported` (against `ROLE_HEADERS` /
  `USER_ROLE_HEADERS` for the resolved `model_type`) and the resolved
  `default_assistant_role` / `default_user_role`, so the webui can gate roles,
  plus `jlens_fitted` (`has_compatible_jlens`: v6 shard sidecar/payload plus loaded
  weight identity, gating the drilldown's j-lens tab without the lazy fp32 load),
  `live_lens_layers` (the live J-lens readout's resolved layer list, `null`
  while off), and `live_probe_scores` (the CAA live toggle). Session metadata is
  serialized from the exact live `SaklasSession` contract; production does not
  coerce incomplete test doubles or inspect lens sidecars as a fallback.
  Scene-grammar capabilities (the cast model; `_scene_capabilities`):
  `scene_mode` (validated grammar present — gates the
  seat toggle + free commit seating), `thinking_input_supported` (family think
  delimiters — gates the committed-thinking box), `strips_history_thinking`
  (drives the composer's "lasts one turn" warning).
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
  parity gap to worry about**: `push_manifold` is wired only into CLI `pack
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
  gen-lock, 404 pre-lock; referenced activation-capture groups are removed too. Response `{namespace, name, source, removed,
  rematerializes_on_restart}`.
- `POST /manifolds/{ns}/{name}/fit` — `session.fit` under the lock; SSE
  / JSON. Discover folders accept `fit_mode` / `hyperparams` overrides; the
  pipeline merges them into `manifold.json` (after `sanitize_hyperparams`) inside
  the same cross-process manifest transaction as cache-key derivation and fit;
  authored folders reject them with 400. `force=true` bypasses tensor/capture
  hits. `layers` optionally names explicit
  transformer indices or `"workspace"`/`"all"`; the fitted tensor sidecar pins
  that layer set. Poisedness `ValueError` →
  `code: "PoisednessError"`; `ConcurrentExtractionError` → 409. Steering a fitted
  manifold needs no route — a `%` term loads it lazily on scope entry.

`POST /manifolds/from-template` derives a discover manifold from an existing
standalone template selected by `template_ref`. Template authoring stays on
`POST /templates`; the manifold route only materializes its value/context corpus
and records the canonical template reference.

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
  `core.manifold.manifold_is_affine` — defensively guarded → `False` on any read
  failure) and
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
  The **CAA live toggle** moved to `POST /instruments/geometry/live` (was
  `POST /probes/live`); session info still reports the state as
  `live_probe_scores`.
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

### Profiles under `/sessions/{id}/profiles` (in `profile_routes.py`)

The 5.x rename of the former `/vectors` group (a steering vector is the 2-node
`pca` case of a profile). `session.profiles` is the canonical registry
(`dict[name, dict[int, Tensor]]`); the routes wrap each entry in a `Profile` for
the wire.

- `GET /profiles` list (`{"profiles": [...]}`), `GET /profiles/{name}` profile JSON,
  `DELETE /profiles/{name}` (also drops the name from `default_steering`).
- `GET /profiles/pairwise?a=&b=` — cross-layer **whitened** cosine matrix between two
  named profiles / probes. Mahalanobis-only (no `metric` param, no Euclidean path):
  whitened cosine is single-layer, so each cell is whitened in `a`'s row-layer frame.
  `session.whitener` must cover every row-layer of `a`, else 409 (regenerate the
  neutral cache). Registered *before* `GET /profiles/{name}` so the literal path wins.
- `POST /extract` — in `asyncio.to_thread`; SSE / JSON (path unchanged). The exact
  request shape is `{concept, baseline?, sae?, role?, namespace?, force?}` and routes
  to `session.extract`: a concept with a baseline fits a 2-node `pca`; a monopolar
  concept fits the 1-node neutral-anchored ray. The resulting manifold's folded
  profile is always registered in the live session. `namespace` controls the
  destination and `force` bypasses the tensor cache. There is no `/extract/preview` (the A0
  scenario/preview machinery was removed — A2 has no scenarios).
- `POST /profiles/bake` (`BakeProfileRequest`) body `{name, expression}` — wraps
  `merge_into_manifold` (model-scoped, `force=True`): lands a corpus-less baked
  manifold, folds the fitted tensor back to a steering Profile, registers it. The
  server mirror of CLI `manifold bake`. `_refuse_if_busy` first (409).
  `MergeError` → 400. (Cloning was removed in 4.0 — no `/profiles/clone` route.)

`GET /sessions/{id}/correlation?names=…` — N×N Mahalanobis-cosine matrix across
loaded steering vectors and active probes (a steering vector wins a name collision
over a same-named probe). Mahalanobis-only: passes `session.whitener` to
`cosine_similarity`; a missing whitener is 409, and a pair the whitener doesn't
fully cover lands as `null`. Default covers everything; `names` restricts.

### Loom tree (in `tree_routes.py`)

`/sessions/{id}/tree`: full-tree GET, active-path GET, and navigate / edit / branch /
delete / star / note / reset mutations, plus `edge_label`, `filter`, branch `diff`,
`joint_logprobs`, and `transcript` / `transcript/load`. Mutations run the tree's
conflict checks (409 when they would corrupt an in-flight generation). Joint
logprob reads use the session-owned `joint_logprob_cache` directly.
Cast roster (phase 3): `GET .../tree/cast` (label → member), `PUT
.../tree/cast/{label}` body `{steering?, thinking?, seed?, notes?}` (label-slug
+ expression syntax validated up front, 400 on either), `DELETE
.../tree/cast/{label}` (204, absent = no-op). Roster ops are decoration-tier
(never 409); the roster also rides the full-tree GET (`cast` key when
non-empty) and the `op="cast"` `tree_mutated` frame. `tree/branch` takes an
optional `role` override — with the engine's scene mode this is the seat-swap
branch primitive.

### instrument_routes.py — the unified `/instruments` family

One route tree over `session.instruments` (`geometry`/`lens`/`sae`). It supersedes
the former per-family `/lens/*` and `/sae/*` groups **and** `POST /probes/live`.
Auth / `acquire_session_lock` / `SaklasError.user_message()` / `background_job`
scrubbing disciplines are copied verbatim from the routes it replaces.
`{family}` outside `geometry`/`lens`/`sae` → 404.

- `GET /sessions/{id}/instruments` — enumerate the three families. Per family
  `{family, live, source, probes, capabilities}`. **live** is family-discriminated:
  geometry `{enabled: session.live_probe_scores}`; lens `{enabled: bool,
  layers: session.live_lens_layers}`; sae `{enabled: session.live_sae, layer,
  top_k}` (layer/top_k from the live config when on, else `null`). **source**:
  lens = the active source label (`session._active_jlens_source_label()`), sae =
  the resident release normalized to `saelens:`/`local:` (from `session.sae_info`),
  geometry = `null`. **capabilities** declares per-family support so clients never
  guess: `{sources: bool, preparations: [ops], token_readout: bool,
  source_switch: bool}` (geometry `False/[]/False/False`; lens
  `True/["fetch","fit"]/True/True`; sae `True/["load","train"]/True/False`).

- `POST .../instruments/{family}/live` — uniform body `{enabled, layers?, top_k?}`.
  geometry → `session.set_live_probe_scores(enabled)` (layers/top_k → 400); lens →
  `enable_live_lens(layers=…)` / `disable_live_lens()` under the session lock
  (top_k → 400); sae → `enable_live_sae(top_k=…)` / `disable_live_sae()` (layers →
  400). Returns the family's resolved live state (same shape as the GET listing's
  live field). **Replaces** `POST /probes/live` and `POST /lens/live`.

- `GET .../instruments/{family}/sources` — lens: the prepared-sources listing
  (`list_lens_sources`, `path` stripped). sae: `{sources, releases}` merging
  prepared sources (`list_sae_sources`) with provider release candidates
  (`list_sae_releases`) so the dashboard sees both prepared and
  still-needs-fetching rows. geometry: 404 (no source lifecycle).

- `PUT .../instruments/{family}/source` body `{source}` — lens only (synchronous
  switch: the old `POST /lens/use` semantics — lock + derived-state eviction +
  auto-enable live). sae → 409 (source switching loads weights → use preparations
  `load`); geometry → 404.

- `POST/GET/DELETE .../instruments/{family}/preparations` — the unified
  background-job resource over `background_job.py` (still polled, never SSE;
  resumable/cancel semantics preserved). POST body `{operation, …op fields}`:
  lens supports `fetch` (old `/lens/fetch`) and `fit` (old `/lens/fit`), sae
  supports `load` (old `/sae/load`) and `train` (old `/sae/train`); the fields are
  re-parsed into the exact old per-op request models (unknown/invalid → 400).
  GET returns the running-or-last job status in a common shape:
  `{state: idle|running|done|error, operation, progress: {current, total, unit} |
  null, message, error, started_at, finished_at, cancellable, …op extras}`
  (fit `unit="prompts"`, train `unit="tokens"`; fetch/load → `progress null` +
  message; extras: fit/fetch `live_layers`, load `release`/`info`, train
  `name`/`info`). Mutual exclusions preserved exactly (lens fetch XOR fit, sae
  load XOR train → 409). DELETE: lens cancels a running fit (fetch isn't
  cancellable → 409 when only a fetch runs / nothing runs); sae cancels a running
  train, else unloads the resident SAE (the old `DELETE /sae/load` teardown). The
  jobs keep the historical `app.state.lens_fit` / `lens_fetch` / `sae_load` /
  `sae_train` attributes and the `_stop_lens_fit` / `_stop_sae_train` shutdown
  hooks. geometry → 404.

- `GET .../instruments/{family}/token-readout?node_id=&raw_index=[&top_k=][&steered=
  ][&raw=][&layers=]` — the loom token-drilldown readout, wrapped in the 5.x
  `measurements` **replay** envelope: `{measurements: {version, scope: "replay",
  provenance: "replayed", instruments: {<family>: {binding: {source, steering},
  readout: {…}}}}}` built via `core.measurements.build_measurements`
  (`steered=false` → `binding.steering` null). lens = the old
  `session.jlens_token_readout` (per-layer top-k `readout.layers` + the all-layer
  `readout.aggregate`); sae = `session.sae_token_readout` (`readout.features`).
  geometry: 404. Same lock/error mapping as before
  (`LensNotFittedError`/`UnknownNodeError` → 404, `InvalidNodeOperationError`/bad
  `layers`/`top_k` → 400).

- Family extras (moved under the family): `POST .../instruments/lens/token/validate`
  (`{word}` → `{word, token_id}`, read-only single-token check; multi-token → 400),
  `POST .../instruments/sae/features/metadata` (the Neuronpedia backfill:
  `{ids}` ≤64 → `session.fetch_sae_feature_meta`, no session lock — network +
  disk cache only), and `POST .../instruments/sae/features/validate` (was
  `POST /sae/feature/validate`).

**Deleted routes** (clean break, no aliases): `POST /probes/live`,
`GET /lens/sources`, `POST /lens/use`, `POST/GET /lens/fetch`,
`POST/GET/DELETE /lens/fit`, `POST /lens/live`, `POST /lens/token/validate`,
`GET /lens/token-readout`, `GET /sae/sources`, `GET /sae/releases`,
`POST/GET/DELETE /sae/load`, `POST/GET/DELETE /sae/train`, `POST /sae/live`,
`POST /sae/feature/validate`, `POST /sae/features/metadata`,
`GET /sae/token-readout`. `lens_routes.py` and `sae_routes.py` are gone.

### traits_routes.py — live traits SSE

`GET /sessions/{id}/traits/stream` — per-token probe scores in real time during any
active generation, via inline `Monitor.score_single_token` gated behind
registered trait queues (zero overhead when no client is connected). Stays open
across generations; multiple clients supported. Events: `start`
(`{generation_id}`), `token` (`{idx, text, thinking, probes}`), `done`
(`{generation_id, finish_reason, aggregate}`), `: heartbeat` every 15 s when idle.

### WS /saklas/v1/sessions/{id}/stream (in `ws_stream.py`)

Bidirectional WebSocket; only the exact `session_id == "default"` is accepted.
The dashboard composer sends `{type:"submit", text?, authored_role?,
generated_role?, steering?, sampling?, thinking?, authored_thinking?, raw?,
parent_node_id?, n?, recipe_override?}`. Roles use canonical
`"user"|"assistant"` names. Text requires `authored_role`; omit
`generated_role` for commit-only;
omit text for a bare continuation. The server commits text once, then fans
generated siblings from that node, so authored and generated roles are explicit
and independent.

The compatibility and specialist client frame is `{type: "stop"}`, or
`{type: "generate", input, steering,
sampling, thinking, stateless, raw, parent_node_id?, n?, recipe_override?,
generate_seat?}`. `generate_seat` (`"user"|"assistant"`, default assistant) is
the cast model's seat selector: `"user"` renders the generation prompt as a
user-seat header (labeled by `sampling.user_role`) and lands the node with
`role="user"` + a stamped recipe — generated is provenance, not a seat; needs
the session's validated scene grammar (`SceneRenderError` 400 otherwise).
`input: null` is a continue — no committed turn, the model speaks next from
`parent_node_id` (or the active leaf), enabling a/a and u/u sequences. The
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
- **Commit (no-generation send)** (`commit_role`/`commit_text`, optional
  `commit_thinking`) → `session.append_user_turn` / `append_assistant_turn`.
  `raw=true` lifts the user-under-user guard (scene mode lifts both commit-parent
  guards engine-side — u/u and a/a authored shapes land on validated families).
  `commit_thinking` is a committed thinking block stored on
  `LoomNode.thinking_text` and rendered through the family think delimiters
  (400 `SceneThinkingUnsupportedError` when the family can't carry it;
  suppressed in raw mode). Emits one `started` (node_id=null) + one `done`
  carrying the new node under `result.{kind="commit", role, text, node_id}`.
  No token frames.
- **Recipe override** — a built-in mode string (`unsteered`/`inverted`/`reseed`/
  `cool`/`hot`) or a partial-recipe expression (`seed=42, temperature=1.5`).

Fork / prefill / commit are mutually exclusive (400 on mix). `n>1` fans out N sibling
assistant nodes on one shared user parent, generated serially with deterministic
derived seeds.

A roster mutation (`LoomMutated(op="cast")`) forwards as a `tree_mutated`
frame with empty `added`/`updated` plus a `cast` key inlining the full roster
(`{label: {recipe?, notes?}}`), so clients reconcile without a refetch.

Server → client: `started` (node_id filled lazily by the first token),
`tree_mutated`, `token` (per token — `logprob`/`top_alts`/`perplexity` when
captured, plus the token tap's 5.x **`measurements`** envelope — the single
JSON-safe read-side record, the same object appended to the loom row: `version`
/ `scope="token"` / `provenance`, the flat `scores`/`per_layer_scores` views,
and per-family `instruments` (`geometry`/`lens`/`sae`, each with its attached-probe
`readings` and — for lens/sae — the native `readout` discovery surface plus a
`binding` recording source + recipe steering)). The token frame carries **no**
legacy per-token aliases: the pre-5.x top-level `scores`/`per_layer_scores`/
`probe_readings`/`lens_readout`/`lens_aggregate`/`sae_readout`/`captured` keys
are gone — everything lives inside `measurements`. The stream wraps `_on_token`
in a typed `TokenConsumer` whose options request the live readouts;
`build_token_event` forwards `measurements` verbatim rather than rebuilding it
from loom rows. `done` (`result` with `text`, `tokens`, `finish_reason`,
`usage`, `mean_logprob`, `mean_surprise`, the `probe_readings` aggregate **and**
an aggregate-scope `measurements` envelope — `probe_measurements_aggregate`
splits the result's readings by family: geometry = Monitor probes, lens =
`session.lens_probe_names`, sae = `session.sae_probe_names`, with source/steering
binding from the live configs), `error` (validation errors keep the connection
open; other failures close 1011).

Concurrency: one perpetual reader task owns `receive_json()` and feeds a shared
`incoming` queue; `tree_mutated` ride a connection-level
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
