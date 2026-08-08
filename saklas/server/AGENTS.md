# server/

Dual-protocol HTTP on one port: OpenAI `/v1/*`, Ollama `/api/*`, native
`/saklas/v1/*`, and the Svelte dashboard at `/`. One model per server;
generation across all three protocols serializes on one `asyncio.Lock`
(`session.lock`, server-owned), distinct from the engine's threading
`gen_lock`.

## Module map

| Module | Owns |
|---|---|
| `app.py` | FastAPI factory, OpenAI routes, auth, the session lock, the three error envelopes |
| `ollama.py` | The `/api/*` shim (`register_ollama_routes`) |
| `native_routes.py` | `register_saklas_routes` — mounts the native sub-registrars in order |
| `native_common.py` | Single-session id resolution, the strict `NativeRequest` base, `refuse_if_busy`, the shared extraction error policy |
| `response_models.py` | The response `TypedDict`s every non-streaming native route annotates — the schema `scripts/generate_webui_types.py` renders `webui/src/lib/types.gen.ts` from |
| `session_routes.py` / `session_models.py` | `/sessions` CRUD, clear/rewind, steering validation, `session_info` |
| `manifold_routes.py` | `/manifolds/*` — authoring, merge, install, generate, fit, search |
| `template_routes.py` | `/templates/*` — the templated-completion artifact + its scorer |
| `probe_routes.py` | `/sessions/{id}/probes/*` — one attach/detach/list surface over all three probe families |
| `profile_routes.py` / `profile_models.py` | `/sessions/{id}/profiles/*`, `/extract`, `/correlation` |
| `tree_routes.py` / `tree_models.py` | The loom `/sessions/{id}/tree/*` routes and cast roster |
| `instrument_routes.py` | `/sessions/{id}/instruments/*` — the table-driven read-family tree |
| `ws_stream.py` / `ws_models.py` / `ws_events.py` | `WS /sessions/{id}/stream`, its schemas, its frame builders |
| `sse.py`, `streaming.py`, `background_job.py`, `request_helpers.py`, `model_names.py` | Shared plumbing |

`create_app` registers OpenAI routes, then Ollama, then the native tree, and
mounts the SPA **last** so its catch-all cannot shadow an API path.
`server/__init__.py` re-exports `create_app`, `acquire_session_lock`,
`ws_auth_ok`, `SESSION_LOCK_TIMEOUT_SECONDS`.

## Shared plumbing

- **`request_helpers.py`** — the one construction site for every protocol's
  request lowering: `build_sampling_config` (normalizes `stop` from string or
  sequence, collapses the two OpenAI `logprobs` shapes, treats empty role
  labels as unset), `parse_request_steering` + `merge_steering` (per-request
  expression composed over the server default at the key level; an explicit
  empty string clears and must reach the session as an *empty* `Steering`, not
  `None`, so the cast roster can't refill it), `flatten_content`,
  `probe_token_readings`, `strict_model_enabled`.
- **`streaming.py`** — `stream_finalizer(session, result)` derives the
  end-of-stream triple `(finish_reason, usage, probe_agg)` all three streaming
  protocols share; each formats it into its own wire shape.
  `probe_reading_aggregate` unions the Monitor roster with `session.lens.names`
  / `session.sae.names`, so readout-channel probes keep their aggregates.
- **`sse.py`** — `progress_sse_response` is the queue-driven SSE worker
  (progress/done/error frames, `call_soon_threadsafe` bridge, a `: heartbeat`
  comment every `HEARTBEAT_SECONDS = 15` of idle so a proxy read timeout can't
  drop a slow job, and cancellation cleanup that awaits the real worker so a
  disconnected client can't release `session.lock` mid-write). It takes the
  lock **unbounded** — the jobs it drives are inherently unbounded.
  `sse_or_json` negotiates SSE on `Accept: text/event-stream` vs one JSON body
  under the bounded lock, with typed error handling explicit per branch
  (`error_formatter` for the frame, `json_errors` for the status table).
- **`background_job.py`** — `BackgroundJob` is the polled counterpart for the
  four long preparations. State lives on `app.state.<name>` / `<name>_task` /
  `<name>_cancel` and is read through `app.state` on every access, never
  cached, so shutdown hooks and callers that reassign it are honored.
  `share_group` ties mutually-exclusive jobs (lens fetch XOR fit, sae fetch XOR
  train) into one 409 group; `launch` guarantees `running`/`finished_at`
  finalization; `make_progress_hook` parses each job's own progress vocabulary.
- **`model_names.py`** — HF-id → Ollama alias resolution.
  `_HF_TO_OLLAMA_ALIASES` overrides where Ollama's catalogue rounds
  differently or `model_type` carries no version suffix; otherwise
  `_infer_aliases` falls back to `<family>:<size>`. `known_model_names` is the
  lowercased accept set both protocols check in strict mode.

**Error scrubbing (info-disclosure discipline).** A long-running job must not
surface raw `str(e)` — Python exception strings routinely echo filesystem paths
and traceback fragments. The catch-all logs the full traceback server-side and
sends `{"message": "<op> failed", "code": type(e).__name__}`; typed branches
with an already-safe message (`SaklasError.user_message()`, the manifold
poisedness `ValueError`, the install conflicts) keep theirs. `scrub_job_error`
applies the same rule to the polled surface, writing into the status dict
instead of a frame; a cooperative cancel settles to `cancelled`, no error.

## app.py — factory, OpenAI, auth, errors

`create_app(session, default_steering=None, cors_origins=None, api_key=None, *,
web=False)`. `web=True` mounts the dashboard (`saklas serve` default-on,
`--no-web` and library callers off). Routes: `GET /v1/models`,
`GET /v1/models/{id}`, `POST /v1/chat/completions`, `POST /v1/completions`.
Thin HTTP — handlers call `session.generate` / `generate_stream` with a
`SamplingConfig` + `Steering` and never mutate `session.config`.

`_SamplingBase` (pydantic, shared by chat/completions): `stop`, `seed`,
`logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs` (bool for
chat, int for completions), `top_logprobs`, `stream_options.include_usage`,
`max_completion_tokens` (folded onto `max_tokens`), the native `steering` field
(an expression **string** — dicts are rejected at the pydantic layer), and the
native `thinking` field (`None` = auto via `supports_thinking`).

Accept-and-ignore: `user`, `n`, `response_format: {"type": "text"}`, empty
`tools: []`, `tool_choice` in `{"none", "auto"}`. Non-empty `tools`, other
`tool_choice` values, and non-`text` `response_format` are rejected by
`_check_langchain_compat` → 400. `ChatMessage` concatenates the text parts of
OpenAI multimodal arrays; a non-text part raises `UnsupportedContentError`.

Responses carry real `usage`, `finish_reason` off `result.finish_reason`, and
per-request `created`. `_stream_generation` emits a first-chunk
`{role: "assistant"}` delta, takes the finish reason from `stream_finalizer` on
the final chunk, optionally emits a usage chunk, and always terminates with
`[DONE]` — including after an in-band error frame, because OpenAI clients wait
for the sentinel. Thinking tokens stream as `reasoning_content`. A wired-in
`Request` stops generation once the client disconnects, and
`stream_iter.close()` in a `finally` tears the engine worker thread down
deterministically. `_render_logprobs_chat` / `_render_logprobs_completions`
build the two OpenAI logprobs shapes off `result.logprobs`, reading alt text
from `TokenAlt` rather than re-tokenizing.

Probe readings ride each **choice** under the vendor-prefixed
`x-saklas-probe-readings` extension, keyed by probe name, in
`ProbeReading.to_dict()` shape; the aggregate is that reading pooled at the
last content token. The field is omitted when nothing is attached, so clients
that don't read it see no shape change. Both compat protocols request
`live_scores=False` / `live_readouts=False`, so the final chunk's aggregate is
what carries data.

**Auth.** Bearer token from `SAKLAS_API_KEY` / `--api-key`, applied as an
app-level dependency over HTTP and WebSocket routes. `_require_auth` +
`_check_bearer` gate HTTP; `ws_auth_ok(websocket)` runs **before**
`websocket.accept()` (close 1008 on failure) and accepts either an
`Authorization: Bearer …` header or the browser-dashboard `?token=…` fallback,
since browser WebSocket constructors cannot set headers. Unset key = open
server. `SAKLAS_STRICT_MODEL` (`1`/`true`/`yes`/`on`) 404s a `model` mismatch
across OpenAI and Ollama against `known_model_names`; unset accepts any name.

**Locking.** `acquire_session_lock(session)` is a bounded
(`SESSION_LOCK_TIMEOUT_SECONDS = 300`) async context manager yielding
`True`/`False`. Non-streaming handlers take it plainly; streaming handlers hold
it for the full stream and emit a 503 on timeout. Requests queue FIFO.

**Error envelopes.** Three, discriminated by path prefix in `_protocol_error`:
Ollama `{"error": "<msg>"}` under `/api/`, native `{"detail": "<msg>"}` under
`/saklas/v1/`, OpenAI `{"error": {message, type, param, code}}` otherwise, with
`openai_error_type` mapping 409 → `conflict`, other 4xx →
`invalid_request_error`, else `server_error`. `_on_saklas_error` routes any
`SaklasError` through `user_message()`; `RequestValidationError` lands as 400
in the request's own envelope; `_on_http_exception` flattens a native
`HTTPException.detail` to a **string** so `/saklas/v1/*` clients never have to
guess between a string, a dict, and a list of pydantic errors.

## Native tree conventions

URL paths carry `{session_id}` for a multi-session shape, but the impl is
single-session: `SINGLE_SESSION_ID = "default"`; everything else 404s via
`resolve_session_id`. Every native request model inherits `NativeRequest`
(`extra="forbid"`), nested native objects included; the OpenAI/Ollama protocol
models stay protocol-specific.

`refuse_if_busy(session)` is a non-blocking `gen_lock` probe → 409. It guards
the mutating manifold routes and the profile bake: `session.lock` orders native
mutations against each other, but an SSE fit or extract whose request was
cancelled leaves its worker thread — and the `gen_lock` — alive past the
cancel.

**Status taxonomy.**

| Status | Means |
|---|---|
| 400 | Malformed body or an unsupported knob (`RequestValidationError`, a rejected preparation field, `top_k` on a live toggle, a bad `layers` list) |
| 404 | Unknown session, artifact, probe, node, or instrument family; a capability the family does not have |
| 409 | A conflict that retrying can clear: busy job group, engine gen-lock held, existing folder, non-covering whitener, SAE source switch |
| 422 | `SteeringCompositionError` — parsed and resolved, but the loaded geometry can't honor it (`ManifoldArityError`, `OverlappingManifoldError`) |
| 503 | Bounded session-lock timeout |

`native_common.extraction_error_frame` / `extraction_json_errors` are the
shared typed policy for the two routes that drive `ManifoldExtractionPipeline`
(`POST /extract`, `POST .../fit`): the retryable conflicts
(`ConcurrentExtractionError`, `ManifoldAuthoringChangedError`) → 409 /
`code: "Conflict"`, an authoring-grade `ValueError` → 400 with
`code: "PoisednessError"` when it is the RBF poisedness failure, everything
else to the catch-all scrubber.

### session_routes.py

- `GET/POST /saklas/v1/sessions` — list / idempotent create (a model mismatch
  warns and returns the existing session).
- `GET/PATCH/DELETE /saklas/v1/sessions/{id}` — info / update defaults / no-op
  204. PATCH reads pydantic's fields-set for `top_k` so an explicit `null`
  (disable) is distinguishable from an omitted field (preserve).
- `POST /saklas/v1/sessions/{id}/steering/validate` — parse, resolve, and
  **dry-install** an authored expression: `parse_expr`, then entering
  `session.steering(parsed)` under the session lock and rolling straight back,
  so a green validation cannot hide a rack-time failure. A `SaklasError` comes
  back **in band** as `{valid: false, expression, error}` — a form result, not
  a failed request.
- `POST /saklas/v1/sessions/{id}/{clear,rewind}` (204; rewind 400s on empty
  history).

`session_info` is serialized from the exact live `SaklasSession` contract —
production does not coerce incomplete test doubles or inspect artifact sidecars
as a fallback. It carries `is_base_model`, `supports_thinking` /
`thinking_is_optional`, `jlens_fitted` (`has_compatible_jlens`: shard
sidecar/payload validity plus loaded-weight identity, gating the drilldown's
j-lens tab without the lazy fp32 load), the role capabilities
(`role_substitution_supported` / `user_role_supported` against `ROLE_HEADERS` /
`USER_ROLE_HEADERS` for the resolved `model_type`, plus `default_assistant_role`
/ `default_user_role`), the scene-grammar capabilities `scene_mode` /
`thinking_input_supported` / `strips_history_thinking`, and — the read plane's
ONE representation — an **`instruments`** key holding the same per-family
blocks `GET .../instruments` lists.

### manifold_routes.py

A manifold is a top-level artifact, so these routes live outside
`/sessions/{id}`; `fit` is the exception that needs the loaded model.
`_manifold_json` is the wire serializer behind every detail-returning route: it
builds the session-independent fields via
`io.manifolds.manifold_summary(folder)` — byte-identical to what CLI
`pack show -j` emits for the shared keys — then layers the session-aware
extras: `fitted_for_session`, `stale` (sidecar `nodes_sha256` vs the folder,
plus loaded-model fingerprint), `resolved_fit_mode` (a fitted discover folder's
actual flat/curved family, `None` for an unfitted `auto`), and — for a discover
folder fitted on the loaded model — the materialized per-model geometry
(`domain` / `domain_label` / `intrinsic_dim` / `min_nodes` / `node_coords`)
lifted from the sidecar and tensor. `full=True` adds per-node statements and
per-tensor fit detail (`method`, `feature_space`, `node_count`,
`nodes_sha256`, `fit_mode`, optional `hyperparams` / `diagnostics`).

The **list route stays cheap**: `_fitted_geometry` opens the safetensors with
`safe_open` and reads the header key set plus the one small `(K, n)`
`node_coords` entry — the flat/curved discriminator is the *absence* of any
`layer_<L>.node_params` key — instead of deserializing every per-layer payload.
An unreadable artifact reports unresolved geometry rather than failing the
listing.

- `GET /manifolds` — every installed manifold with per-session fit status.
  `GET /manifolds/{ns}/{name}` — detail; 404 missing, 400 malformed folder.
- `GET /manifolds/search?q=&limit=` — HF-hub proxy via
  `io.hf_manifolds.search_manifolds`; `{query, results}`. Missing
  `huggingface_hub` → 503, HF transport error → 502.
- `POST /manifolds` — author an *authored* artifact (box/sphere domain,
  per-node `role` optional). Returns detail + `advisories` (soft poisedness /
  flat-axis warnings, so a deficient layout surfaces before a fit is paid for).
  409 existing, 400 malformed.
- `POST /manifolds/discover` — author a *discover* artifact (nodes carry
  `{label, statements, role?}`, no coords).
- `POST /manifolds/from-template` — derive a discover folder from a standalone
  template selected by `template_ref`; template authoring stays on
  `POST /templates`. 404 unknown template, 409 ambiguous or existing.
- `POST /manifolds/merge` — `merge_discover_manifolds`, discover-mode sources
  only (authored geometry isn't mergeable without a shared coordinate system).
  Pools node corpora + roles into one unfitted discover folder; pair with
  `/fit`. `<2` sources, label collisions, mixed `fit_mode` without an override,
  or a destination conflict → 400/404/409.
- `POST /manifolds/install` — `install_manifold` in a worker thread,
  **SSE-capable** through `sse_or_json` (a manifold repo carries a per-model
  safetensors payload, so download + stage/verify/swap is a long operation).
  `InstallManifoldRequest` carries `as` as a true wire field
  (`Field(alias="as")` + `populate_by_name`). `refuse_if_busy` runs before the
  worker starts writing. `ManifoldInstallConflict` → 409, missing → 404, bad
  input → 400, `huggingface_hub` missing → 503, HF error → 502. Both branches
  return the detail JSON `GET .../{ns}/{name}` ships.
- `POST /manifolds/generate` — LLM-authors a discover folder via
  `session.generate_responses` under the session lock; SSE or JSON.
  `concepts` (≥2, one node each), `kind` ∈ `abstract|concrete|custom` with
  `custom_system` **required** for `custom` (400 otherwise, matching
  `/extract`), `samples_per_prompt`, `role_per_node=true` → persona manifold.
  The default **resumes** via `plan_discover_generation` — fills missing nodes,
  appends concepts new to the roster; `force` is the clean slate.
- `PATCH /manifolds/{ns}/{name}` — `update_manifold_folder` under the lock +
  `refuse_if_busy`, so a corpus rewrite can't race a fit reading `nodes/`.
  Existing tensors go stale, not deleted.
- `DELETE /manifolds/{ns}/{name}` — `remove_manifold_folder`, the single source
  of truth shared with CLI `pack rm`. Response `{namespace, name, source,
  removed, rematerializes_on_restart}`.
- `POST /manifolds/{ns}/{name}/fit` — `session.fit` under the lock; SSE or
  JSON. Discover folders accept `fit_mode` / `hyperparams` overrides that the
  pipeline merges into `manifold.json` inside the same manifest transaction
  that derives the cache key and publishes the fit; authored folders reject
  them. `layers` optionally names explicit indices or `workspace`/`all`, and
  the sidecar pins that set. `force` bypasses tensor/capture hits.

Every mutating route calls `_evict_manifold` so a delete or re-fit leaves no
stale in-memory `Manifold` behind either the bare or the qualified grammar key.
Steering a fitted manifold needs no route — a `%` term loads it lazily on scope
entry.

### template_routes.py

The standalone templated-completion artifact (`io.templates`). Lifecycle is
pure-IO; `score` runs the loaded model.

- `GET /templates` — list (`summary` + `namespace` per row).
- `GET /templates/{ns}/{name}` — detail incl. the full `contexts`; 404 on
  missing.
- `POST /templates` — `create_template_folder` (slot, values, multi-turn
  `contexts: [{turns, assistant}]`, description, tags, force). 409 on existing,
  400 on a validation failure (slot in a history turn, slot count ≠ 1, …).
- `DELETE /templates/{ns}/{name}` — 200 `{namespace, name, removed}`; 404 on
  missing.
- `POST /templates/{ns}/{name}/score` — `session.score_template` in
  `asyncio.to_thread` under the bounded lock (503 if locked). Body
  `{steering?}`; returns `{template, namespace, steering, contexts:
  [ChoiceScores.to_dict()]}` — the per-context restricted-choice distribution,
  steering-aware. 404 on a missing/ambiguous template, 400 on a
  scoring/steering failure (scrubbed to `type(e).__name__`).

### probe_routes.py

One unified collection covering every probe shape. Rows are discriminated by an
explicit **`family`** key and carry only what that family can actually produce —
a client keys off `family`, never off an out-of-band flag on an otherwise
geometry-shaped row.

- `GET /probes` — Monitor rows (`family: "geometry"`, `manifold`, `top_n`,
  `layers`, `node_labels`, `node_count`, `domain`, `intrinsic_dim`,
  `feature_space`, `is_affine` from `core.manifold.manifold_is_affine` — the
  flat/curved discriminator the client classifies subspace-vs-manifold on — and
  `node_coords`, `null` when no per-model layout is materialized), then pinned
  lens rows (`family: "lens"`, `layers`, `intrinsic_dim: 1`,
  `feature_space: "readout"`, `word`, `token_id`) and SAE rows
  (`family: "sae"`, the resident `layer`, `intrinsic_dim: 1`,
  `feature_space: "sae-readout"`, `feature_id`, `label`, `max_act` — the
  strength unit, `null` when the reading is a raw activation).
- `GET /probes/defaults` — the default roster.
- `POST /probes` `{selector, name?, top_n?}` → `session.add_probe`, 201 + the
  row in its family's shape. The selector rides the same `[ns/]name[:variant]`
  grammar `%` steering consumes, so probe and steering share the lazy-load
  cache. 400 on an empty selector or `KeyError`/`ValueError`, 404 on
  `FileNotFoundError`, otherwise `SaklasError.user_message()` (a `jlens/<word>`
  selector's `LensNotFittedError` → 404).
- `DELETE /probes/{name}` → `session.remove_probe`, 204; 404 when the name is
  in none of the three rosters.
- `GET /probes/{name:path}/geometry` → `session.monitor.probe_geometry(name)`:
  per-layer centroids, the neutral anchor, a top-3 PCA rotation at rank ≥ 3,
  and a curved fit's curve/surface overlay — all in the whitened frame the
  reads use, so the live per-token point overlays directly. 404 when not
  attached, which includes every readout-channel probe (no subspace behind
  one). `defaults` registers before this greedy path so it still resolves.

### profile_routes.py

`session.profiles` is the canonical registry; the routes wrap each entry in a
`Profile` for the wire. Comparison reads go through
`session.analytics_profile` / `analytics_names` — cached **CPU snapshots**
built once under the exclusive-GPU lock, so a polled endpoint never issues an
MPS→CPU copy on a threadpool thread that could race a model op on PyTorch's
single non-thread-safe command buffer.

- `GET /profiles`, `GET /profiles/{name}`, `DELETE /profiles/{name}` (also
  drops the name from `default_steering`).
- `GET /profiles/pairwise?a=&b=` — cross-layer **whitened** cosine matrix
  between two named profiles or probes. Mahalanobis-only: whitened cosine is a
  single-layer operation, so each cell is whitened in `a`'s row-layer frame —
  exact on the layer-aligned diagonal, an A-frame read off it. The whitener
  must cover every row-layer of `a`, else 409 (regenerate the neutral cache); a
  registered name whose snapshot isn't built yet is also 409, retryable.
  Registered **before** `GET /profiles/{name}` so the literal path wins.
- `GET /sessions/{id}/correlation?names=…` — N×N Mahalanobis-cosine matrix
  across loaded profiles and active probes (a profile wins a name collision
  over a same-named probe). A request-scope cache holds `(v, Σ⁻¹v, vᵀΣ⁻¹v)` per
  name/layer so the symmetric matrix costs one Woodbury apply per entry rather
  than one per pair. Missing whitener → 409; a pair it doesn't fully cover, or
  a name whose snapshot isn't ready, lands as `null`.
- `POST /extract` — `session.extract` in `asyncio.to_thread`, SSE or JSON (the
  JSON branch also returns the collected `progress` lines). Body
  `{concept, baseline?, kind, custom_system?, sae?, role?, namespace?, force?}`
  — `kind` ∈ `abstract|concrete|custom` with `custom_system` required for
  `custom`, matching `/manifolds/generate`. A concept with a baseline fits a
  2-node `pca`; a monopolar concept fits the 1-node neutral-anchored ray. The
  folded profile is always registered on the live session.
- `POST /profiles/bake` `{name, expression}` — the HTTP face of
  `SaklasSession.bake`, which owns the whole sequence (model-scoped merge into
  a corpus-less `fit_mode="baked"` manifold, fold back to a `Profile`,
  registration). The session method computes the loaded-weight fingerprint, so
  a bake whose components were fitted against different weights is refused here
  exactly as in Python. `refuse_if_busy` first; `MergeError` → 400.

### tree_routes.py

`/sessions/{id}/tree`: full-tree `GET` and `PUT` (atomic restore, delegating
schema validation to `LoomTree.from_dict` and refusing a model mismatch —
saved token ids and stamped recipes are model-specific), active-path `GET`, and
navigate / edit / branch / delete / star / note / reset mutations, plus
`edge-label`, `filter`, branch `diff`, `joint-logprobs`, and `transcript` /
`transcript/load`. Mutations run the tree's conflict checks (409 when they
would corrupt an in-flight generation); navigate, star, and note are
decoration-tier and never conflict. `tree/branch` takes an optional `role`
override (`user|assistant|system`) — with the engine's scene mode this is the
seat-swap branch primitive.

`joint-logprobs` force-replays both branches under their stamped recipes and
returns per-aligned-position cross-evaluation records; results live in
`session.joint_logprob_cache` keyed by the sorted `(a_id, b_id)` pair,
double-checked under the lock, invalidated by edits/deletes/finalize.

Cast roster: `GET .../tree/cast`, `PUT .../tree/cast/{label}`
`{steering?, thinking?, seed?, notes?}` (label slug and expression syntax
validated up front, 400 on either), `DELETE .../tree/cast/{label}` (204, absent
= no-op) — all decoration-tier. `cast_json` serializes the **effective** roster
with an `origin` per label (`configured`, `structural` for `user`/`assistant`,
or `observed`) and rides the full-tree GET plus every `tree_mutated` frame, so
clients reconcile identity without a refetch or provenance inference.

### instrument_routes.py — the unified `/instruments` family

One route tree over `session.instruments` (`geometry` / `lens` / `sae`).
Dispatch is **table-driven**, not a chain of `if family == …` branches: the
`CAPABILITIES` dict declares what each family supports, and an unsupported
operation answers from the declaration (including its status and message).
`require_family` validates `{family}` against the registry itself, so the table
and the registry cannot disagree about which families exist. Response shapes
live in `response_models.py` with the rest of the native tree's, and
`tests/test_measurements_envelope.py` pins the envelope's key sets.

| Family | `sources` | `preparations` | `token_readout` | `source_switch` |
|---|---|---|---|---|
| `geometry` | no (404) | none (404) | yes | no (404) |
| `lens` | yes | `fetch`, `fit` | yes | yes |
| `sae` | yes | `fetch`, `train` | yes | no (409 → run `fetch` as a preparation) |

Preparation operation names match the CLI verbs exactly on both surfaces.

- `GET /sessions/{id}/instruments` — the three families, each as
  `{family, live, source, probes, capabilities}` built by the ONE
  `family_block`; `session_info` embeds the same list. `live` is the family's
  own `LiveState.to_dict()` (geometry `{enabled}`, lens `{enabled, layers}`,
  sae `{enabled, layer, source}`). `source` is `instrument.active_source` — the
  same resolver that stamps the source onto every measurement binding, so the
  listing agrees with persisted rows rather than answering from a
  prepared-sources scan.
- `POST .../instruments/{family}/live` `{enabled, layers?}` →
  `instrument.set_live(enabled, **extras)`; the *instrument* rejects an extra it
  can't honor (`TypeError` → 400), so there is one rejection rule rather than
  three. `top_k` is refused for every family up front — readout width is
  generation state shared with `return_top_k`/alts, never an instrument-local
  dial. Returns the resolved live block.
- `GET .../instruments/{family}/sources` — lens: `list_lens_sources` with
  `path` stripped. sae: `{sources, releases}`, merging prepared sources with
  provider release candidates so the dashboard sees both prepared and
  still-needs-fetching rows.
- `PUT .../instruments/{family}/source` `{source}` — lens only, synchronous:
  lock, live off, `select_jlens_source`, live back on, returning
  `{source, live_layers}`. 404 missing, 400 malformed, 409 not fitted or a
  running lens job, 503 lock timeout.
- `POST/GET/DELETE .../instruments/{family}/preparations` — the unified
  background-job resource, polled, never SSE. POST body `{operation, …fields}`
  (202); fields are re-parsed into the per-operation model, with a flattened
  `"field: message"` string on rejection (400). GET returns the running-or-last
  job in one shape: `{state: idle|running|done|error, operation, progress:
  {current, total, unit} | null, message, error, started_at, finished_at,
  cancellable, …extras}` — fit `unit="prompts"`, train `unit="tokens"`,
  fetch/load `progress: null` plus a message; extras are lens `live_layers` /
  `source`, sae `release` / `name` / `info`. Group exclusions 409. DELETE:
  lens cancels a running fit (a fetch is not cancellable, so a fetch-only or
  idle state 409s); sae cancels a running train, else unloads the resident SAE
  under the lock.
- `GET .../instruments/{family}/token-readout?node_id=&raw_index=[&top_k=]
  [&steered=][&raw=][&layers=]` — resolve → validate → dispatch to
  `instrument.token_readout`, which owns the whole replay **including** its
  `scope="replay"` measurements envelope, so this route reshapes nothing. A
  knob a family cannot honor comes back as `ValueError` → 400 rather than being
  silently dropped (`top_k`/`layers` on geometry, `layers` on sae).
  `LensNotFittedError`/`UnknownNodeError` → 404,
  `InvalidNodeOperationError` → 400, 503 on lock timeout.
- Family extras: `POST .../instruments/lens/token/validate` (`{word}` →
  `{word, token_id}`, read-only; multi-token → 400),
  `POST .../instruments/sae/features/validate` (`{id}`), and
  `POST .../instruments/sae/features/metadata` (`{ids}`, 1..64 → the
  Neuronpedia label + `maxActApprox` backfill; network and disk cache only, so
  it deliberately takes no session lock).

Jobs keep the `app.state.lens_fit` / `lens_fetch` / `sae_load` / `sae_train`
attributes and register `_stop_lens_fit` / `_stop_sae_train` shutdown hooks.

### WS /saklas/v1/sessions/{id}/stream (ws_stream.py)

Bidirectional; only the exact `session_id == "default"` is accepted, and
`ws_auth_ok` runs before `accept()`.

**Inbound.** Exactly three frame types, all validated by pydantic in the reader
before dispatch.

- `{type: "submit", …}` — **the one authored-turn contract on this wire**:
  `text` + `authored_role` appends a span; `generated_role` then optionally
  asks the model to continue from it; with no text, `generated_role` alone is a
  bare continue from the selected leaf. Also carries `steering`, `sampling`,
  `thinking`, `authored_thinking`, `raw`, `parent_node_id`, `n`,
  `recipe_override`. `_normalize_submit` lowers it onto the `generate` schema:
  append-only takes the no-decode `session.append_turn` path; append+generate
  keeps the authored turn for one atomic commit inside the generation worker
  and then generates from `input=None`.
- `{type: "generate", …}` — the specialist / compatibility frame: `input`
  (string, message list, or `null` for a continue), `steering`, `sampling`,
  `thinking`, `stateless`, `raw`, `parent_node_id`, `n`, `recipe_override`, the
  fork triple (`fork_node_id`/`fork_raw_index`/`fork_alt_token_id`), the
  prefill pair (`prefill_node_id`/`prefill_text`), and `generate_seat`
  (`"user"|"assistant"`, default assistant — `"user"` renders the prompt under
  a user-seat header and lands the node `role="user"` with a stamped recipe;
  generated is provenance, not a seat, and it needs the validated scene
  grammar). It carries **no `commit_*` vocabulary** — authored turns are
  `submit`'s job. Field-consistency rules (whole fork group together, prefill
  needs its text, fork and prefill mutually exclusive) live in the model
  validators, so the schema is the single description of a well-formed frame;
  `PydanticCustomError` keeps those messages verbatim.
- `{type: "stop"}` — signals `session.stop()` mid-generation; a no-op when idle.

`WSInputMessage` is `{role, content, label?}` — `label` is the per-turn cast
label the scene stitcher renders into the constructed header, so a dashboard
auto-regen shadow of a custom-labelled turn replays under the prompt it is
shadowing. `WSSamplingParams` adds `user_role`/`assistant_role` (per-message
role substitution, stamped on the produced nodes), `return_top_k`,
`persist_per_layer_scores`, and `persist_subspace_coords`.

**Outbound.** `started` (`generation_id`, `node_id` filled lazily by the first
token, `sibling_index`, `sibling_count`); `tree_mutated` (`op`, `rev`, `added`,
`removed`, `updated`, `active_node_id`, plus the inlined `cast` roster);
`token`; `done`; `error`.

The `token` frame carries `text`, `thinking`, `token_id`, `node_id`,
`raw_index`, and `logprob`/`perplexity`/`top_alts` when captured, plus the
**`measurements`** envelope — the single JSON-safe read-side record, the same
object appended to the loom row: `version`, `scope: "token"`, `provenance`, the
flat `scores`/`per_layer_scores` views, and per-family `instruments`
(`geometry`/`lens`/`sae`, each with its attached-probe `readings` and — for
lens/sae — the native `readout` discovery surface plus a `binding` recording
source + recipe steering). The tap owns that payload; `build_token_event`
forwards it verbatim rather than reconstructing it from persisted loom rows,
which would create a second wire authority and mask tap bugs.

The `done` frame's `result` carries `text`, `tokens`, `finish_reason`, `usage`,
`mean_logprob`, `mean_surprise`, and an **aggregate-scope `measurements`
envelope only** — the engine builds it at finalize
(`GenerationResult.measurements`), so the server neither re-splits readings by
family nor reprojects the lens/SAE `ScalarReading`s. A `stop` landing mid-turn
rewrites the wire `finish_reason` to `cancelled` (UI-only; the engine's
canonical `stop` stays on the loom recipe). A commit-only `submit` emits one
`started` (`node_id: null`) and one `done` whose result is `{role, text,
node_id, finish_reason: "stop", mean_logprob: null, mean_surprise: null}` — no
token frames.

**One error convention.** Every rejection is an `error` frame with the same
five keys (`message`, `code`, `status`, `node_id`, `sibling_index`) on a
connection that stays **open** — a 400-grade mistake must not close the socket,
and FastAPI's `SaklasError` handler does not apply to WebSocket routes, so the
handler catches them explicitly. Pydantic rejections render as
`"field: message"` joined by `"; "` — the same convention
`app._on_validation_error` gives native REST bodies — with **every** error
reported, not just the first, because `input` is a union whose real failure is
the second branch error. Only an unexpected exception closes (1011).

**Concurrency.** One perpetual reader task owns `receive_json()` and feeds a
shared `incoming` queue — the underlying `websockets` `recv_in_progress` flag
makes overlapping receives a `RuntimeError`, so no other task reads the socket.
All sends go through one `asyncio.Lock`. `tree_mutated` events ride a
connection-level `LoomMutated` subscription forwarded by its own task, and
`done` waits for its node's `finalize_assistant` delta to be forwarded first,
so a client never sees a completed-but-empty assistant node. Per generate turn
`generate_stream` runs in a worker thread, `on_token` bridges to asyncio via
`call_soon_threadsafe`, and the handler races the token queue against
`incoming` so an in-flight `stop` is honored without blocking; non-stop frames
mid-generation hold in a deferred deque and drain after the turn.
`session.lock` is held for the full N-way batch so concurrent clients serialize
FIFO; `n>1` fans siblings out serially under deterministic derived seeds, and
an error inside one sibling aborts the rest of the fan.

## ollama.py

Ollama-compatible shim reusing `session` / `default_steering` / `session.lock`
/ app-level auth. Routes: `/api/version`, `/api/tags`, `/api/ps`, `/api/show`,
`/api/chat`, `/api/generate`, `/api/pull` (no-op success for the loaded model,
404 otherwise), `HEAD /` (bodyless 200 liveness probe), and 501 stubs for
`/api/push`, `/api/create`, `/api/copy`, `/api/delete`, `/api/embeddings`,
`/api/embed`.

Streaming responses are NDJSON (`application/x-ndjson`). `/api/show.template`
reflects the real HF Jinja `tokenizer.chat_template` — clients that parse it
fail either way, so the honest template beats a meaningless Go placeholder.
`/api/generate` omits `context` (saklas can't round-trip it). `_resolve_options`
recognizes `temperature`, `top_p`, `top_k`, `seed`, `num_predict`→`max_tokens`,
`stop`, `presence_penalty`, `frequency_penalty`, `repeat_penalty` (→
`presence_penalty` via `ln(repeat_penalty)`: Ollama divides positive logits by
the penalty, which is subtracting a per-seen-token constant), and `steer`;
everything else (`min_p`, `mirostat*`, `num_ctx`, …) is logged at debug and
dropped. Steering rides a non-standard `steer` field inside `options` or at the
top level — an expression string composed over `default_steering` at the key
level; a non-string raises a clean 400 before headers flush. A top-level
`think` bool wins over the expression's flag and streams as `message.thinking`
(chat) / top-level `thinking` (generate). `_duration_stats` splits measured
wall time between `prompt_eval_duration` and `eval_duration`;
`_finish_to_done_reason` maps `stop_sequence` → `stop`. Probe readings ride the
top-level `x-saklas-probe-readings` key, absent when nothing is attached.

## Deliberately absent

Each of these is a refused design, not a gap. Do not re-propose them.

- **No `/saklas/v1/packs*` surface** and no `POST .../push`. Concepts are
  manifolds, so distribution rides the manifold routes; HF upload is CLI-only
  (`pack push`).
- **No traits SSE stream.** Live per-token reads ride the WS `measurements`
  envelope and the OpenAI/Ollama reading extensions.
- **No per-family `/lens/*` or `/sae/*` groups and no `POST /probes/live`.**
  Everything lives under `/instruments/{family}/…`, with no aliases.
- **No one-shot text-scoring endpoints** (`POST .../probe`,
  `POST .../manifold-probe`): they re-render arbitrary text out of conversation
  context. Scoring is a live-hook read or a `token-readout` replay.
- **No `/extract/preview` and no `/profiles/clone`.**
- **No per-token aliases on the WS `token` frame** — no top-level `scores` /
  `per_layer_scores` / `probe_readings` / `lens_readout` / `lens_aggregate` /
  `sae_readout` / `captured`. Everything is inside `measurements`.
- **No flat instrument keys on `session_info`** — no `live_lens_layers`,
  `live_sae`, `live_probe_scores`, `sae_loaded`, `sae_info`. The `instruments`
  block is the one representation.
- **Neither compat protocol supports** tool calling, JSON-schema /
  structured-output mode, or embeddings.
- **No multi-session server.** The `{session_id}` segment exists for the shape;
  only `"default"` resolves.
