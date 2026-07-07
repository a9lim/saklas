# saklas audit - 2026-07-07

Scope: adversarial source audit of generation while steering/probing, API
streaming and progress routes, J-lens surfaces, and cleanup/unification
opportunities. I used three subagents in parallel:

- Kepler: server/API/session concurrency and protocol correctness.
- Helmholtz: generation hot path, steering/probing performance, J-lens.
- Russell: redundancy, dead bridges, and module-boundary cleanup.

I reconciled their findings against the live source. The initial audit pass did
not load a model or exercise GPU/MPS/CUDA paths; the remediation pass below did
run targeted model/GPU-backed regressions where noted.

## Highest-priority correctness and concurrency findings

- [x] **A1 - Streaming and progress cancellation can release `session.lock`
  while the underlying worker is still running.**

  Impact: a disconnected OpenAI/Ollama stream or cancelled progress SSE can
  leave generation/extraction/fit work running in a worker thread after the
  async route exits its `session.lock` scope. A later request can then enter the
  same session while the old job is still mutating session state, using model
  hooks, or occupying `_gen_lock`.

  Evidence:
  - `generate_stream()` only calls `worker.join(timeout=5.0)` during close,
    then returns control to the route (`saklas/core/session.py:6006`).
  - OpenAI streaming closes the iterator in the `async with
    acquire_session_lock(session)` scope, but the close path is bounded
    (`saklas/server/app.py:558`, `saklas/server/app.py:617`).
  - Ollama streaming has the same pattern (`saklas/server/ollama.py:702`,
    `saklas/server/ollama.py:792`).
  - Progress SSE cancels the async wrapper and exits `async with lock`; jobs
    using `asyncio.to_thread` can keep their underlying worker running
    (`saklas/server/sse.py:45`, `saklas/server/sse.py:75`,
    `saklas/server/vector_routes.py:392`,
    `saklas/server/manifold_routes.py:976`).

  Fix sketch: make the lock lifetime owned by the actual worker/job, not just
  the response generator. For generation streams, either join the worker to
  completion on close or move streams to an explicit job object that owns both
  cancellation and lock release. For progress SSE, avoid cancelling a
  `to_thread` job without a corresponding worker completion/abort handshake.

  Fixed 2026-07-07: `generate_stream()` now joins the worker to completion on
  close and exposes its own result; progress SSE now awaits unfinished worker
  tasks under the lock instead of cancelling the wrapper while `to_thread`
  continues.

- [x] **A2 - Stop-sequence trimming leaves final probe aggregates aligned to
  hidden states after the visible response ended.**

  Impact: `GenerationResult.text` can be trimmed before the stop delimiter, but
  `probe_readings` and `readings` are still scored at the last generated content
  token from `generated_ids`. If the stop sequence is visible text, the final
  aggregate can include the stop delimiter or a partial delimiter the client
  never saw.

  Evidence:
  - Stop handling trims `state.response_text` and sets
    `finish_reason = "stop_sequence"` while `generated_ids` still contains the
    token that completed the match (`saklas/core/generation.py:1417`).
  - Final text uses `state.response_text` for stop-sequence completions
    (`saklas/core/session.py:4753`).
  - Incremental and aggregate-only scoring still pick
    `last_content_index(generated_ids, tokenizer)` (`saklas/core/session.py:3613`,
    `saklas/core/session.py:3645`).
  - The result stores those aggregate readings unchanged
    (`saklas/core/session.py:4781`, `saklas/core/session.py:4852`,
    `saklas/core/session.py:4860`).

  Fix sketch: have generation record the visible response endpoint as a raw
  token/forward index when a stop sequence fires. Final scoring should use that
  visible endpoint, or the last non-thinking entry in `state.emit_map`, instead
  of recomputing from untrimmed `generated_ids`.

  Fixed 2026-07-07: `GenerationState` records `response_aggregate_index` on
  stop-sequence matches; finalization threads it through full, incremental,
  lean, and aggregate-only probe scoring. No-visible-token stop matches now
  produce empty per-probe aggregates rather than pooling the hidden stop token.

- [x] **A3 - Requested `@when:<probe>@label` and
  `@when:<probe>~label` gates can be false solely because the requested label
  is outside the monitor's current top-N.**

  Impact: a gate against a specific manifold label looks like an exact channel,
  but the score is only emitted if that label survives the monitor's top-N
  nearest/assignment ranking. Missing score keys make the trigger inactive, so
  gates can silently fail even when the requested label's exact distance or
  soft-assignment value should satisfy the threshold.

  Evidence:
  - `score_gate_scalars()` parses exact requested distance and assignment
    labels (`saklas/core/monitor.py:727`).
  - It then uses `probe.top_n` and only emits `topk` labels
    (`saklas/core/monitor.py:867`).
  - `Trigger.active()` treats a missing gate score as inactive
    (`saklas/core/triggers.py:155`).
  - The current behavior is encoded in
    `tests/test_manifold_gate.py:212`.

  Fix sketch: for gate scoring, gather exact requested label indices directly
  from the full distance/probability vectors. For live incremental scoring,
  let the gate callback request exact scalar keys rather than depending on
  `flat_scalars()` top-N output.

  Fixed 2026-07-07: gate-only scoring now gathers requested `@label` distances
  and `~label` assignment probabilities directly by label index, independent of
  `probe.top_n`.

- [x] **A4 - Ollama finalization reads mutable session-global result state
  instead of the local generation result.**

  Impact: non-streaming Ollama has the local `result`, but derives
  `done_reason` and probe aggregates from `session.generation_state` and
  `session.last_result`. Streaming also reads `session.last_result` after
  iterator close. Under the cancellation bug in A1, or any future multi-job
  route, final metadata can be stale or belong to a different request.

  Evidence:
  - Non-streaming stores `result = session.generate(...)`
    (`saklas/server/ollama.py:614`), then reads
    `session.generation_state.finish_reason` (`saklas/server/ollama.py:623`)
    and `_probe_reading_aggregate(session)` (`saklas/server/ollama.py:632`).
  - The aggregate helper is explicitly bound to `session.last_result`
    (`saklas/server/app.py:300`).
  - Streaming reads `result = session.last_result`
    (`saklas/server/ollama.py:803`).

  Fix sketch: derive final Ollama fields from the local `result` returned by
  the generation call. Mirror `stream_finalizer(session, result)` style for
  both protocols, and make probe aggregate helpers accept an explicit result.

  Fixed 2026-07-07: OpenAI/Ollama helpers now accept explicit results;
  non-streaming Ollama derives `done_reason` and probe aggregates from the local
  result, and streaming finalizers prefer the stream iterator's own result.

- [x] **A5 - J-lens cache loading accepts malformed tensor shapes until a
  later matmul crashes.**

  Impact: `load_lens()` advertises self-healing behavior for unusable caches,
  but validates only format version and finiteness. A cache with the wrong
  tensor rank, non-square tensors, a sidecar/tensor `d_model` mismatch, missing
  layers, or extra layers can load successfully and fail later in live readout,
  steering atoms, or decomposition.

  Evidence:
  - `load_lens()` parses tensors into `jacobians` and constructs
    `JacobianLens` without shape checks (`saklas/io/lens.py:81`,
    `saklas/io/lens.py:102`).
  - `JacobianLens.__init__` stores tensors as-is
    (`saklas/core/jlens.py:127`).
  - The first hard failure is later in `hidden @ J.T`
    (`saklas/core/jlens.py:157`).

  Fix sketch: validate that sidecar `d_model > 0`, each tensor is finite,
  2-D, square, exactly `(d_model, d_model)`, and the tensor layer keys match
  sidecar `source_layers` when present. Invalid caches should log and return
  `None`. Add targeted `tests/test_lens_io.py` coverage.

  Fixed 2026-07-07: loader validates sidecar `d_model`, nonempty
  `source_layers`, tensor/sidecar layer equality, finite values, and exact
  `(d_model, d_model)` tensor shapes; malformed caches return `None`.

## Performance findings for generation while steering/probing

- [x] **B1 - `generate_batch()` and fan generation are serialized wrappers
  around `_generate_core()`.**

  Impact: prefix KV reuse helps prefill, but decoding, steering hooks, capture,
  probe scoring, and finalization run one row at a time. This is the largest
  throughput ceiling for evals and steering/probe sweeps where rows share the
  same sampling config and steering expression.

  Evidence:
  - Fan, batch, and sweep generation now flow through a shared serial executor
    whose loop calls `_generate_core()` once per row
    (`saklas/core/session.py:200`, `saklas/core/session.py:218`).
  - `generate_batch()` now has a fast branch for compatible stateless rows,
    including aggregate probe reads, but incompatible rows still route through
    the shared serial executor (`saklas/core/session.py:6049`,
    `saklas/core/session.py:6527`).
  - Corpus generation already has a true batched helper using one
    `model.generate` over the batch (`saklas/core/session.py:2320`,
    `saklas/core/session.py:2397`).

  Fix sketch: first unify serial batch/fan orchestration behind one runner so
  behavior is centralized. Then add a stateless batched decode path for rows
  with compatible sampling/steering, batched `HiddenCapture`, and per-row
  monitor aggregation.

  Progress 2026-07-07: fan, batch, and sweep orchestration now share
  `_run_serial_generation_jobs()` (`saklas/core/session.py:200`,
  `saklas/core/session.py:5766`, `saklas/core/session.py:6023`,
  `saklas/core/session.py:6249`). This removes three divergent serial loops and
  creates one branch point for true batched decode paths.

  Progress 2026-07-07 (second pass): `generate_batch()` now opportunistically
  runs compatible stateless, no-probe batches through one padded
  `model.generate()` call (`saklas/core/session.py:6049`,
  `saklas/core/session.py:6357`, `saklas/core/session.py:6527`). The branch keeps
  cancellation via a `StoppingCriteria`, supports always-active steering hooks,
  and falls back to the serial path for live lens, hidden capture requests,
  logprobs/top-k returns, stop strings, seeded per-row sampling, penalties, and
  phased/gated triggers. Coverage asserts both the one-call fast path and an
  under-initialized probe fallback (`tests/test_batch.py:299`,
  `tests/test_batch.py:334`).

  Progress 2026-07-07 (third pass): compatible stateless batches with attached
  probes now stay on the batched decode path too. `HiddenCapture` has a bounded
  batched tail ring that stores per-forward `[B, D]` slices
  (`saklas/core/hooks.py:175`, `saklas/core/hooks.py:373`); the batch path
  attaches it for probe layers, pools each row's final content token, and builds
  the same stateless aggregate `ProbeReadings` shape as the serial finalizer
  (`saklas/core/session.py:6147`, `saklas/core/session.py:6191`,
  `saklas/core/session.py:6400`, `saklas/core/session.py:6469`). Coverage asserts
  one `model.generate()` call plus distinct per-row aggregate probe values
  (`tests/test_batch.py:338`).

  Progress 2026-07-07 (fourth pass): deterministic fan-out (`generate(..., n>1)`
  with effective `temperature <= 0`) now batches by repeating the prompt through
  the same compatible `generate_batch` fast path and returning a `kind="fan"`
  `RunSet` (`saklas/core/session.py:5776`, `saklas/core/session.py:5822`).
  Coverage asserts one `model.generate()` call for a three-way greedy fan
  (`tests/test_batch.py:362`).

  Progress 2026-07-07 (fifth pass): greedy fast batches no longer reject a
  caller-supplied seed, because `temperature <= 0` does not consume RNG
  (`saklas/core/session.py:6103`, `saklas/core/session.py:6229`). This unlocks
  seeded deterministic batch/fan calls while preserving serial behavior for
  stochastic seeded sampling (`tests/test_batch.py:326`,
  `tests/test_batch.py:343`, `tests/test_batch.py:399`). Degenerate stateless
  greedy sweeps whose rows resolve to the same steering expression now reuse the
  same batch fast path and return the original sweep grid/callback shape
  (`saklas/core/session.py:6717`, `saklas/core/session.py:6888`;
  `tests/test_batch.py:830`). Stochastic fan-out remains serial because the
  existing contract derives per-sibling seeds and the installed Transformers
  `generate()` API has no per-row RNG-generator surface. Distinct-alpha steering
  sweeps remain serial because the active steering manager composes one hook
  coefficient set per layer for the whole batch; batching those rows correctly
  would require a batch-shaped steering-hook contract rather than a wrapper
  change.

- [x] **B2 - Live J-lens readout performs per-layer vocab matvecs every token
  and disables the persistent capture route.**

  Impact: with five live lens layers, every token does five `d x d` transports,
  five vocab matvecs, and five top-k calls. Live lens also widens capture to
  lens layers and forces transient capture when `_live_lens` is enabled, losing
  the compiled-clean persistent capture path.

  Evidence:
  - The docstring describes one `d x d` matvec plus one vocab matvec per
    selected layer (`saklas/core/session.py:2020`).
  - `_live_lens_readout_step()` loops layers and performs `h @ J.T`, final norm,
    vocab matvec, and top-k per layer (`saklas/core/session.py:2090`).
  - `_begin_capture()` adds live lens layers to the capture set and disallows
    persistent capture while live lens is on (`saklas/core/session.py:3351`,
    `saklas/core/session.py:3375`).

  Fix sketch: batch the selected-layer transport/readout where possible
  (`torch.stack`/`bmm` or a layer-major tensor) and extend persistent capture
  buffers to cover live lens layers, so live lens does not force transient
  hooks.

  Fixed 2026-07-07: live J-lens readout now batches selected-layer transport,
  final norm, vocab matvec, and top-k in a layer-major tensor path. Live-lens
  capture also reuses the compiled-clean persistent capture buffers when they
  are available, so enabling the lens no longer forces transient hooks.

- [x] **B3 - Prefix-cache reuse and StaticCache/compiled decode are mutually
  exclusive in the current router, despite comments implying a StaticCache
  prefix path exists.**

  Impact: cached-prefix requests cannot use the StaticCache/compiled fast path
  today. That may be an acceptable tradeoff, but the comments in
  `generate_steered()` say a non-`None` `past_key_values` is expected to already
  be a StaticCache. The session router only enables StaticCache when
  `cached_pkv is None`.

  Evidence:
  - `generate_steered()` describes non-`None` `past_key_values` as already
    StaticCache-sized (`saklas/core/generation.py:790`).
  - `_run_generation_loop()` requires `cached_pkv is None` before enabling
    StaticCache (`saklas/core/session.py:5216`).

  Fix sketch: either implement StaticCache-backed prefix entries with explicit
  `cache_position` and enough decode headroom, or update docs/comments so the
  router clearly says prefix reuse chooses DynamicCache.

  Fixed 2026-07-07: prefix-cache entries can now be StaticCache-backed when the
  caller requests it and supplies enough decode headroom. The generation router
  preserves the StaticCache/compiled path for cached static entries, rejects
  static prefix hits when the active generation is not static-eligible, and
  safely misses oversized static entries so a freshly sized cache can be used.
  Batch prefix warming requests static entries for unsteered static-capable
  sessions while keeping DynamicCache prefix reuse for response-phase steering
  cases.

- [x] **B4 - Compiled sessions install persistent offset and capture hooks on
  every layer before compile.**

  Impact: stable hook topology is useful for compile, but even narrow probe
  rosters and unprobed compiled generations carry all-layer hook bodies:
  `add_` for steering offsets and `copy_` for capture. This may be fine on some
  backends, but it is a persistent tax on the exact decode path intended to be
  fastest.

  Evidence:
  - Compile setup installs persistent hooks before compile
    (`saklas/core/session.py:758`).
  - Offset hooks are registered for every layer
    (`saklas/core/hooks.py:1490`, `saklas/core/hooks.py:1515`).
  - Capture hooks are also designed as per-layer persistent hooks
    (`saklas/core/hooks.py:1522`).

  Fix sketch: split compiled modes into "no capture", "known probe subset", and
  "dynamic/future probes", or defer compile until the default probe layer set is
  known. Measure before changing; this is a hot-path tradeoff, not automatically
  a bug.

  Fixed 2026-07-07: added the safe "no capture" compiled mode. Sessions
  constructed with explicit `probes=[]` still install persistent offset hooks
  for static-affine steering, but skip the all-layer persistent capture
  `copy_` hooks and buffers. Default/named-probe sessions keep the existing
  capture hooks so probed generation can still ride the compiled-clean capture
  path; later ad-hoc probes on a `probes=[]` session fall back to transient
  capture.

- [x] **B5 - Stop-sequence-only calls force the token tap but avoid the token
  table, causing per-token tokenizer decode fallback.**

  Impact: stop matching needs token text, so `_need_tap` is true when
  `stop_list` is present. But `_tap_has_text_consumer` is false unless there is
  a real text/logprob/trait/probe consumer, so `cache_token_text=False`; the
  generation loop falls back to `tokenizer.decode([tid])` per emitted token.

  Evidence:
  - Stop-sequence-only generation sets `_need_tap`
    (`saklas/core/session.py:5524`) but not necessarily
    `_tap_has_text_consumer` (`saklas/core/session.py:5532`).
  - `generate_steered()` documents `cache_token_text` as disabled for
    stop-sequence-only callers (`saklas/core/generation.py:814`).
  - `_decode_piece()` then falls back to `tokenizer.decode([tid])`
    (`saklas/core/generation.py:1045`).

  Fix sketch: add a tiny lazy token-text LRU or a stop-aware incremental
  decoder. That keeps the bounded stop-match cost low without requiring a full
  vocab decode table.

  Fixed 2026-07-07: stop-only decoding now uses a per-generation lazy token text
  cache, avoiding repeated `tokenizer.decode([tid])` calls without building the
  full vocab token table.

## Cleanup, simplification, and dead-bridge findings

- [x] **C1 - WebSocket token-event rendering can rescore probes from private
  capture state on the token path.**

  Impact: the WebSocket stream explicitly requests live probe scores from the
  token tap, but `ws_events` still falls back to `session._capture._per_layer`
  and calls `score_single_token()` itself if the payload is missing. That is a
  private reach-in and can duplicate monitor geometry work on a hot path.

  Evidence:
  - WebSocket generation marks `_saklas_wants_live_scores` and
    `_saklas_wants_per_layer_scores` on the callback
    (`saklas/server/ws_stream.py:573`).
  - `ws_events` falls back to `session._capture._per_layer` and
    `mf_monitor.score_single_token(...)` (`saklas/server/ws_events.py:66`).

  Fix sketch: remove the fallback or make it a debug-only path. If fallback is
  kept, use a public capture accessor and avoid rescoring when
  `_last_token_probe_payload` is absent.

  Fixed 2026-07-07: `ws_events` now serializes only the token-tap payload and
  no longer reaches into `session._capture._per_layer` or calls
  `score_single_token()` itself.

- [x] **C2 - The generation conductor in `session.py` still owns too many
  responsibilities.**

  Impact: `SaklasSession` remains the main coupling point for decoding,
  steering stack composition, capture routing, monitor scoring, live lens,
  loom persistence, event shaping, result finalization, and streaming state.
  This makes hot-path edits hard to verify and encourages private cross-module
  reach-ins.

  Evidence:
  - `session.py` is over 6k lines in this checkout.
  - `_token_tap` spans logprobs, mean surprise, probe payloads, live lens, loom
    token append, user callbacks, and trait queues (`saklas/core/session.py:5372`).
  - `_finalize_generation()` decodes text, scores probes, trims hidden states,
    builds results, updates monitor history, and writes loom events
    (`saklas/core/session.py:4738`).
  - SteeringComposer extraction left many thin session forwarders
    (`saklas/core/session.py:3028`, `saklas/core/session.py:3287`).

  Fix sketch: do not do a big-bang split. Extract one typed token-event/probe
  payload builder and one result-finalization/scoring collaborator first, then
  update WebSocket/SSE consumers to use the typed payload instead of session
  private state.

  Fixed 2026-07-07: extracted token-level probe payload shaping into
  `core/token_payloads.py` (`TokenProbePayload` +
  `build_token_probe_payload`) and moved generation result finalization/scoring
  into `core/generation_finalizer.py`. `SaklasSession._token_tap` now delegates
  probe payload construction, and `_finalize_generation` is a compatibility
  wrapper over the finalizer collaborator.

- [x] **C3 - Server protocol modules depend on each other's private helpers
  and on CLI internals.**

  Impact: OpenAI, Ollama, WebSocket, native routes, and CLI diagnostics are
  still tangled. This raises the cost of changing one protocol and makes route
  behavior harder to reason about in isolation.

  Evidence:
  - Ollama imports private helpers from `server.app`
    (`saklas/server/ollama.py:23`).
  - OpenAI strict model naming imports Ollama `_aliases_for`
    (`saklas/server/app.py:738`).
  - WebSocket streaming imports private app helpers
    (`saklas/server/ws_stream.py:27`).
  - The server WHY route imports CLI `_summarize_diagnostics`
    (`saklas/server/vector_routes.py:485`, implementation at
    `saklas/cli/runners.py:976`).

  Fix sketch: extract shared `server/sampling.py`, `server/steering_request.py`,
  `server/auth.py`, `server/model_names.py`, and a core diagnostics summary
  helper. Protocol modules should import shared helpers, not each other.

  Fixed 2026-07-07: diagnostics summarization moved from private `cli.runners`
  into `core.histogram.summarize_diagnostics`; model aliases moved to
  `server.model_names`; request flattening, request-steering merge, sampling
  construction, strict-model checking, and probe-reading serialization moved to
  `server.request_helpers`. OpenAI, Ollama, WebSocket, and vector WHY routes no
  longer import those behaviors from each other's private helper surfaces.

- [x] **C4 - `saklas_api.py` is still a cross-route schema/helper bucket.**

  Impact: request bodies and helpers for sessions, vectors, WebSockets, tree,
  joint-logprobs, and experiments live together, then route modules import
  private names from it. This file is a second route layer without ownership.

  Evidence:
  - Request bodies begin as a broad bucket in `saklas/server/saklas_api.py:43`.
  - `session_routes` imports session-specific models and helpers from the
    bucket (`saklas/server/session_routes.py:12`).
  - `vector_routes` imports vector schemas and private helpers from the same
    bucket (`saklas/server/vector_routes.py:23`).

  Fix sketch: move schemas beside their route groups. Leave `saklas_api.py` as
  an app registrar/backcompat import surface only, or retire it once imports are
  moved.

  Fixed 2026-07-07: `saklas_api.py` is now a slim native-route registrar plus
  explicit backcompat re-exports. Route-owned schemas/helpers moved to
  `native_common.py`, `session_models.py`, `vector_models.py`, `tree_models.py`,
  `ws_models.py`, and `experiment_models.py`; session/vector/tree/probe/lens/
  traits/experiment/WebSocket routes import those owners directly.

- [x] **C5 - `scenarios.json` support remains on the main discover-authoring
  path even though current A2 generation no longer writes per-manifold
  scenarios.**

  Impact: current authoring still exposes and writes `scenarios`, but comments
  say 4.0 conversational generation uses global baseline prompts and no longer
  writes per-manifold scenarios. This is legacy compatibility mixed into the
  main authoring path.

  Evidence:
  - `create_discover_manifold_folder()` accepts `scenarios`
    (`saklas/io/manifold_authoring.py:346`).
  - The main authoring function writes `scenarios.json` when present
    (`saklas/io/manifold_authoring.py:445`).
  - The comment says 4.0 conversational generation no longer writes
    per-manifold scenarios (`saklas/io/manifold_authoring.py:664`).
  - Legacy vector migration is the active producer
    (`saklas/io/manifold_authoring.py:627`).

  Fix sketch: quarantine scenario read/write under legacy migration and HF
  compatibility. Keep current discover authoring focused on corpus,
  `template_ref`, node roles/kinds, and fit metadata.

  Fixed 2026-07-07: `create_discover_manifold_folder()` no longer exposes or
  writes `scenarios=`; the internal scenario-writing path is used only by
  `port_legacy_vector_folder()` while porting old vector packs. Standalone
  `read_manifold_scenarios()` / `write_manifold_scenarios()` remain for legacy
  folder compatibility.

- [x] **C6 - `/manifolds/templated` is a stale bridge that duplicates the
  template route plus `manifold from-template`, and it overwrites
  unconditionally.**

  Impact: the bridge accepts only simple user/assistant pairs, while the
  dedicated template route owns the richer template artifact. It also writes
  both template and manifold with `force=True`, bypassing explicit overwrite
  semantics.

  Evidence:
  - The bridge schema is pair-oriented (`saklas/server/manifold_routes.py:128`).
  - It calls `create_template_folder(..., force=True)` and
    `create_manifold_from_template(..., force=True)`
    (`saklas/server/manifold_routes.py:588`,
    `saklas/server/manifold_routes.py:593`).
  - The dedicated template creation route has its own explicit request surface
    (`saklas/server/template_routes.py:45`).

  Fix sketch: deprecate the bridge, or route it through a shared
  template-to-manifold service with explicit `force` semantics and parity with
  the real template API.

  Fixed 2026-07-07: the bridge now has explicit `force` semantics and defaults
  to conflict instead of overwriting both the template and manifold
  unconditionally.

- [x] **C7 - SteeringComposer extraction left test-pinned private shims on
  `SaklasSession`.**

  Impact: the composer exists, but session still exposes a long layer of private
  forwarding methods, some kept because tests monkeypatch them. This preserves
  old coupling and makes it harder to reason about which object owns steering
  resolution.

  Evidence:
  - Session constructs `SteeringComposer` (`saklas/core/session.py:1169`).
  - Multiple methods are documented as thin forwarders to the composer, with
    tests called out as callers (`saklas/core/session.py:3028`,
    `saklas/core/session.py:3035`, `saklas/core/session.py:3287`,
    `saklas/core/session.py:3304`).

  Fix sketch: update tests to target `SteeringComposer` or public helpers
  directly. Remove forwarding shims in small batches once tests stop pinning the
  old session-private surface.

  Fixed 2026-07-07: removed stale session-only forwarders for projection
  materialization, pole-alias resolution, stack flattening, gated-probe key
  discovery, low-level profile folding/legacy-port helpers, and composed
  steering install. Production steering setup now calls `SteeringComposer`
  directly where possible, and tests that were pinning port-on-detect,
  probe-gate detection, and stack flattening now target the composer-owned
  behavior instead.

- [x] **C8 - Profile serialization ownership is split across three layers.**

  Impact: `Profile.save/load`, `vectors.save_profile/load_profile`, and
  `SaklasSession.save_profile/load_profile` all participate in one artifact
  format. That makes format changes harder to localize.

  Evidence:
  - `Profile.save()` delegates to `saklas.core.vectors.save_profile`
    (`saklas/core/profile.py:158`).
  - Actual serialization lives in `vectors.py`
    (`saklas/core/vectors.py:761`, `saklas/core/vectors.py:846`).
  - Session wraps the same operations again (`saklas/core/session.py:2802`).

  Fix sketch: make `core/profile.py` the owner of profile serialization. Leave
  `vectors.save_profile/load_profile` as narrow compatibility aliases until
  callers are migrated.

  Fixed 2026-07-07: `core.profile.save_profile/load_profile` now own the
  safetensors + sidecar format, `Profile.save/load` call them directly, session
  loading imports from `core.profile`, and `core.vectors.save_profile/load_profile`
  are narrow compatibility aliases.

## Lower-severity compatibility and validation findings

- [x] **D1 - OpenAI responses always include top-level `probe_readings`, even
  when empty.**

  Impact: choice-level `x-saklas-probe-readings` is conditional, but the
  top-level compatibility field is emitted as `{}` even when no probes are
  present. This is harmless for tolerant clients, but it is noisier than the
  extension behavior and can drift from documented "only when present"
  semantics.

  Evidence:
  - Streaming final chunk always includes `"probe_readings":
    compat_probe_readings` (`saklas/server/app.py:637`).
  - Non-streaming chat and completions always include the top-level key
    (`saklas/server/app.py:855`, `saklas/server/app.py:903`).
  - The choice-level manifold extension is conditional in the same code paths
    (`saklas/server/app.py:631`, `saklas/server/app.py:852`,
    `saklas/server/app.py:900`).

  Fix sketch: only attach the top-level compatibility field if the dict is
  non-empty, or document that it is intentionally always present.

  Fixed 2026-07-07: OpenAI streaming and non-streaming responses now attach the
  legacy top-level `probe_readings` field only when the compatibility dict is
  nonempty.

- [x] **D2 - `lens fit --layers ""` reaches an internal empty-source failure
  instead of a clear CLI validation error.**

  Impact: an empty `--layers` string parses to an empty list. Bounds validation
  accepts it, and later code calls `min(sources)` or reads the first device row,
  producing an internal error rather than a user-facing message.

  Evidence:
  - `_parse_layer_list("")` returns `[]` (`saklas/cli/runners.py:2303`).
  - `fit_jacobian_lens()` accepts empty `sources` because the bounds check uses
    `any(...)` (`saklas/core/jlens.py:357`).
  - The first failure is `min(sources)` or `next(iter(dev_rows.values()))`
    (`saklas/core/jlens.py:465`, `saklas/core/jlens.py:507`).

  Fix sketch: reject an explicit empty list immediately after parsing or in
  `fit_jacobian_lens()` with a clear error such as "`--layers` must name at
  least one source layer".

  Fixed 2026-07-07: CLI parsing rejects an empty `--layers` list with exit 2
  and a clear message; `fit_jacobian_lens()` also rejects empty programmatic
  `source_layers`.

- [x] **D3 - Newly extracted in-memory profiles can lose to installed-manifold
  name collisions during steering parse.**

  Impact: extracting `local/formal.casual` returns the canonical profile name
  `formal.casual`, and `session.steer(name, profile)` registers that exact
  in-memory profile. If a bundled/default manifold of the same name also exists,
  a later `session.generate(..., steering=f"0.1 {name}")` can fail with
  `AmbiguousSelectorError` before the session profile registry is considered.

  Evidence:
  - Plain steering strings parsed through `Steering.from_value()` without any
    session registry context (`saklas/core/steering.py:85`).
  - The installed-manifold composite-name tier raises on cross-namespace
    duplicate names before falling through to a plain profile key
    (`saklas/core/steering_expr.py:997`).
  - The regression is covered by
    `tests/test_steering_expr.py:464` and
    `tests/test_session.py:155`.

  Fixed 2026-07-07: `parse_expr()` accepts a session-local `profile_names`
  override, and session-owned parse sites pass the live profile registry
  (`saklas/core/steering_expr.py:1089`, `saklas/core/session.py:2928`,
  `saklas/core/session.py:5023`). Exact registered profile keys now shadow the
  installed-manifold tiers for that parse only.

- [x] **D4 - `score_hidden()` can mix CPU hidden states with GPU monitor
  Woodbury factors.**

  Impact: `return_hidden=True` returns CPU hidden-state captures, and
  `score_hidden(result.hidden_states, per_token=True)` should round-trip those
  captures. The batched flat monitor cache keyed itself to the hidden-state
  device but kept `X`/`K_inv` on the attached probe's device, producing a CPU/GPU
  matmul crash on MPS/CUDA.

  Evidence:
  - `_ensure_flat_cache()` receives the hidden-state device
    (`saklas/core/monitor.py:655`).
  - Before the fix, the cache moved per-probe means/bases but not shared
    Woodbury factors (`saklas/core/monitor.py:1144`).
  - The round-trip path is covered by `tests/test_session.py:277`.

  Fixed 2026-07-07: flat cache rebuild now moves `X` and `K_inv` onto the active
  scoring device, and the curved helper also device-aligns its mean, basis,
  Woodbury factors, and precomputed `s_mean`
  (`saklas/core/monitor.py:1144`, `saklas/core/monitor_attach.py:657`).

## Checks run

Main process:

- `git status --short --branch` - clean `dev...origin/dev` before editing this
  file.
- `.venv/bin/python -m compileall -q saklas tests`
- `.venv/bin/ruff check saklas/core/generation.py saklas/core/session.py saklas/server/app.py saklas/server/ollama.py saklas/server/streaming.py saklas/core/monitor.py`
- `.venv/bin/ruff check saklas/core/generation.py saklas/core/session.py saklas/core/monitor.py saklas/core/jlens.py saklas/io/lens.py saklas/server/app.py saklas/server/ollama.py saklas/server/sse.py saklas/server/ws_events.py saklas/server/vector_routes.py saklas/server/manifold_routes.py saklas/core/histogram.py saklas/cli/runners.py tests/test_generation.py tests/test_manifold_monitor.py tests/test_lens_io.py tests/test_jlens.py tests/test_cli_flags.py tests/test_server.py tests/test_server_manifold_probes.py tests/test_saklas_api.py`
- `.venv/bin/pytest -q tests/test_generation.py tests/test_manifold_monitor.py::test_gate_scalar_fraction_label_assignment_skip_curved_foot tests/test_manifold_monitor.py::test_gate_scalar_requested_labels_ignore_probe_top_n tests/test_lens_io.py tests/test_jlens.py::test_source_layers_must_precede_final tests/test_jlens.py::test_source_layers_must_not_be_empty tests/test_jlens_session.py tests/test_cli_flags.py::test_lens_layers_empty_string_errors tests/test_server.py::TestOllamaApi::test_chat_non_streaming_done_reason_comes_from_result tests/test_server_manifold_probes.py::TestOpenAIProbeExtension::test_chat_completion_absent_when_no_probes tests/test_saklas_api.py::TestManifoldRoutes::test_create_templated`
  - result: `49 passed`, one Starlette deprecation warning.
- `.venv/bin/ruff check saklas/server/app.py saklas/server/ollama.py saklas/server/ws_stream.py saklas/server/request_helpers.py saklas/server/model_names.py saklas/core/session.py saklas/core/profile.py saklas/core/vectors.py saklas/io/manifold_authoring.py tests/test_manifolds_io.py tests/test_manifold_probe_session.py tests/test_saklas_api.py tests/test_boundary_guard.py tests/test_profile.py tests/test_packs.py`
  - result: `All checks passed`.
- `.venv/bin/pytest -q tests/test_manifold_probe_session.py::test_begin_capture_live_lens_uses_persistent_capture_when_available tests/test_manifolds_io.py::test_create_discover_does_not_write_scenarios tests/test_vector_migration.py::TestPortLegacyVectorFolder::test_scenarios_ported_from_dict_form tests/test_profile.py::test_save_metadata_override_merges_on_top_of_self_metadata tests/test_format_version.py tests/test_packs.py::test_save_load_profile_roundtrip_slim_sidecar tests/test_gguf_io.py::test_load_profile_dispatches_on_extension tests/test_server.py::TestOllamaApi::test_tags_advertises_aliases_for_known_model tests/test_server.py::TestOllamaApi::test_chat_non_streaming_done_reason_comes_from_result tests/test_saklas_api.py::TestWebSocket::test_bad_steering_does_not_kill_connection tests/test_boundary_guard.py`
  - result: `16 passed`, one Starlette deprecation warning.
- `.venv/bin/pytest -q tests/test_generation.py tests/test_manifold_monitor.py::test_gate_scalar_fraction_label_assignment_skip_curved_foot tests/test_manifold_monitor.py::test_gate_scalar_requested_labels_ignore_probe_top_n tests/test_lens_io.py tests/test_jlens.py::test_source_layers_must_precede_final tests/test_jlens.py::test_source_layers_must_not_be_empty tests/test_jlens_session.py tests/test_cli_flags.py::test_lens_layers_empty_string_errors tests/test_server.py::TestOllamaApi::test_tags_advertises_aliases_for_known_model tests/test_server.py::TestOllamaApi::test_chat_non_streaming_done_reason_comes_from_result tests/test_server_manifold_probes.py::TestOpenAIProbeExtension::test_chat_completion_absent_when_no_probes tests/test_saklas_api.py::TestManifoldRoutes::test_create_templated tests/test_saklas_api.py::TestWebSocket::test_bad_steering_does_not_kill_connection tests/test_manifold_probe_session.py::test_begin_capture_live_lens_uses_persistent_capture_when_available tests/test_manifolds_io.py::test_create_discover_does_not_write_scenarios tests/test_vector_migration.py::TestPortLegacyVectorFolder::test_scenarios_ported_from_dict_form tests/test_profile.py::test_save_metadata_override_merges_on_top_of_self_metadata tests/test_format_version.py tests/test_packs.py::test_save_load_profile_roundtrip_slim_sidecar tests/test_gguf_io.py::test_load_profile_dispatches_on_extension tests/test_boundary_guard.py`
  - result: `64 passed`, one Starlette deprecation warning.

Subagent-reported targeted test:

- `.venv/bin/pytest -q tests/test_manifold_gate.py tests/test_manifold_monitor.py::test_gate_scalar_fraction_label_assignment_skip_curved_foot tests/test_batch.py::TestPrefixCacheEligibility::test_batch_common_prefix_detection_keeps_scalar_walk_on_cpu`
  - result: `25 passed`, one Starlette deprecation warning.
- `.venv/bin/ruff check saklas/core/session.py tests/test_batch.py`
  - result: `All checks passed`.
- `.venv/bin/pytest -q tests/test_batch.py::TestPrefixCacheEligibility::test_static_prefix_hit_requires_static_eligibility_and_headroom tests/test_batch.py::TestPrefixCacheEligibility::test_generation_loop_keeps_static_cache_on_static_prefix_hit tests/test_batch.py::TestPrefixCacheEligibility::test_cache_prefix_can_build_static_cache_entry`
  - result: `3 passed`, one Starlette deprecation warning.
- `.venv/bin/pytest -q tests/test_batch.py`
  - result: `30 passed`, one Starlette deprecation warning.
- `.venv/bin/ruff check saklas/core/session.py saklas/core/steering_composer.py saklas/core/steering_expr.py saklas/core/AGENTS.md saklas/io/packs.py saklas/io/AGENTS.md tests/test_steering_context.py tests/test_projection.py tests/test_vector_migration.py tests/test_probe_gate.py tests/test_manifold_probe_session.py`
  - result: `All checks passed`.
- `.venv/bin/pytest -q tests/test_steering_context.py tests/test_projection.py tests/test_vector_migration.py tests/test_probe_gate.py tests/test_manifold_probe_session.py::test_gating_callback_emits_probe_scalars tests/test_manifold_probe_session.py::test_gating_callback_empty_capture_returns_empty tests/test_jlens_grammar.py::test_resolve_probe_manifold_routes_jlens_to_profile_fold tests/test_jlens_decompose.py::test_session_jspace_decompose_on_registered_profile`
  - result: `113 passed`, one SWIG deprecation warning.
- `.venv/bin/ruff check saklas/server/AGENTS.md saklas/server/saklas_api.py saklas/server/native_common.py saklas/server/session_models.py saklas/server/vector_models.py saklas/server/ws_models.py saklas/server/tree_models.py saklas/server/experiment_models.py saklas/server/session_routes.py saklas/server/vector_routes.py saklas/server/tree_routes.py saklas/server/experiment_routes.py saklas/server/ws_stream.py saklas/server/probe_routes.py saklas/server/lens_routes.py saklas/server/traits_routes.py tests/test_saklas_api.py tests/test_server_manifold_probes.py`
  - result: `All checks passed`.
- `.venv/bin/pytest -q tests/test_saklas_api.py`
  - result: `64 passed`, one Starlette deprecation warning.
- `.venv/bin/pytest -q tests/test_server_manifold_probes.py tests/test_vectors_diagnostics_api.py`
  - result: `34 passed`, one Starlette deprecation warning.
- `.venv/bin/pytest -q tests/test_server.py::TestLensTokenReadout::test_session_info_carries_jlens_fitted tests/test_saklas_api.py tests/test_server_manifold_probes.py tests/test_vectors_diagnostics_api.py`
  - result: `99 passed`, one Starlette deprecation warning.
- `.venv/bin/ruff check saklas/core/session.py tests/test_compile_capture_modes.py`
  - result: `All checks passed`.
- `.venv/bin/pytest -q tests/test_compile_capture_modes.py`
  - result: `1 passed`.
- `.venv/bin/pytest -q tests/test_persistent_capture.py tests/test_compile_capture_modes.py`
  - result: `5 passed`.
- `.venv/bin/ruff check saklas/core/session.py saklas/core/generation_finalizer.py saklas/core/token_payloads.py tests/test_token_payloads.py`
  - result: `All checks passed`.
- `.venv/bin/pytest -q tests/test_token_payloads.py tests/test_generation.py`
  - result: `14 passed`, one SWIG deprecation warning.
- `.venv/bin/pytest -q tests/test_generation.py tests/test_token_payloads.py tests/test_manifold_probe_session.py::test_generation_result_carries_probe_readings_field tests/test_server_manifold_probes.py::TestWebSocketProbeReadings`
  - result: `18 passed`, one Starlette deprecation warning.
- `.venv/bin/ruff check saklas/core/session.py saklas/core/steering.py saklas/core/steering_expr.py saklas/core/steering_composer.py saklas/core/joint_logprobs.py saklas/core/monitor.py saklas/core/monitor_attach.py tests/test_batch.py tests/test_session.py tests/test_steering_expr.py`
  - result: `All checks passed`.
- `.venv/bin/pytest -q tests/test_batch.py::TestPrefixCacheEligibility tests/test_steering_expr.py::test_registered_profile_can_shadow_installed_manifold`
  - result: `16 passed`, one Starlette deprecation warning.
- `.venv/bin/pytest -q tests/test_session.py::TestGeneration::test_generate_with_alphas tests/test_session.py::test_return_hidden_round_trip`
  - result: `2 passed`.
- `.venv/bin/pytest -q tests/test_batch.py tests/test_sweep_loom.py tests/test_steering_expr.py::test_registered_profile_can_shadow_installed_manifold tests/test_session.py`
  - result: `62 passed`, `1 skipped`, one Starlette deprecation warning.

Not run:

- Full `pytest`.
- Live HTTP streaming reproduction tests.
