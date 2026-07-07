# Saklas Audit 2

Date: 2026-07-07

Scope: adversarial source audit of implementation bugs, generation-time
steering/probing performance, API/frontend consistency, and cleanup/dead-code
opportunities. Completed subagent passes covered generation/steering/probing,
CLI-server-web surfaces, and redundancy/drift. A fourth manifold/extraction pass
did not return a usable report before shutdown, so no unverified findings from it
are included here.

## Findings

### 1. High: exact label and assignment gates become path-dependent when full per-token probing is active

Evidence:

- `saklas/core/session.py:5570` treats loom/live/persist consumers as a full
  per-token path, and `saklas/core/session.py:5579` / `:5584` only computes
  `gating_only_probes` and `gating_probe_keys` when there is no full consumer.
- In the full path, `saklas/core/steering_composer.py:738` feeds gates from
  `monitor.flat_scalars(incremental_readings[-1])`.
- Full monitor readings truncate `nearest` and `assignment` to `top_n` at
  `saklas/core/monitor.py:470` and `:511`; `flat_scalars` only emits
  `name@label` and `name~label` keys for labels present in those truncated lists
  at `saklas/core/monitor.py:1634` and `:1636`.
- Missing gate keys go inactive at `saklas/core/triggers.py:157`.
- The gate-only path intentionally avoids this truncation by using
  `score_gate_scalars` at `saklas/core/session.py:3442`, and
  `tests/test_manifold_monitor.py:558` asserts requested labels ignore `top_n`.

Impact:

`@when:personas@hacker` or `@when:personas~hacker` can fire during gate-only
generation, then silently stop firing when a simultaneous full per-token consumer
is enabled, solely because that label was not in the token's truncated top-N
readout. This makes steering behavior depend on observability mode.

Suggested fix:

Always compute the requested `gating_probe_keys` when probe gating is active. In
the gating callback, source exact gate keys from `monitor.score_gate_scalars` even
when full incremental readings also exist, or merge those exact keys into the
flattened full reading. Add a regression with `top_n=1`, a gate on a non-nearest
label, and a full per-token consumer.

### 2. Medium: compiled-offset steering skips the logit NaN/inf clamp

Evidence:

- The MPS compiled fast path detaches transient hooks and writes steering into
  persistent offset buffers at `saklas/core/steering_composer.py:908-912`.
- Those offsets are real steering: computed in
  `saklas/core/hooks.py:1475-1509`, written in `:1511-1524`, and applied by the
  persistent hook at `:1572-1575`.
- Detaching transient hooks clears `self.hooks` at `saklas/core/hooks.py:1541-1543`.
- Generation passes `steering_active=bool(self._steering.hooks)` at
  `saklas/core/session.py:5284`.
- The logit sanitize/clamp only runs when `steering_active` is true at
  `saklas/core/generation.py:1229-1231`.

Impact:

Static affine steering on the compiled-offset path can still push fp16 activations
into bad logits, but the safety clamp is skipped exactly because the steering is
carried by persistent offsets instead of transient hooks.

Suggested fix:

Pass a predicate that means "any steering is currently applied", for example
`bool(self._steering.hooks) or self._steering_uses_compiled_offsets`, into
`generate_steered`. Add a focused unit around the argument plumbing or a fake
compiled-offset session.

### 3. Medium: `POST /manifolds/{ns}/{name}/fit` mutates the manifest before acquiring the session lock

Evidence:

- `saklas/server/manifold_routes.py:927-960` loads and rewrites
  `manifold.json` for discover-mode `fit_mode` / `hyperparams` overrides.
- The SSE branch does not acquire the session lock until
  `progress_sse_response(session.lock, ...)` at `saklas/server/manifold_routes.py:994`.
- The JSON branch acquires the lock later at `saklas/server/manifold_routes.py:1003`.

Impact:

Concurrent fit requests serialize the expensive model work, but the input
manifest rewrite races before that serialization. Request A can fit using request
B's hyperparameters, or both can observe an interleaved manifest state.

Suggested fix:

Move the manifest load/sanitize/atomic-write into the locked job, immediately
before `session.fit`, for both SSE and JSON paths. Keep the existing atomic write,
but make it part of the same critical section as the fit.

### 4. Medium: malformed Ollama options can escape as 500s

Evidence:

- `_resolve_options` casts `presence_penalty` and `frequency_penalty` with bare
  `float(...)` at `saklas/server/ollama.py:222-223`.
- It raises a bare `ValueError` for non-string `steer` at
  `saklas/server/ollama.py:242-246`.
- It casts `num_predict` with bare `int(...)` at
  `saklas/server/ollama.py:262`.
- `/api/chat` and `/api/generate` call `_resolve_options` before starting a
  response at `saklas/server/ollama.py:756` and `:773`.

Impact:

Client mistakes such as `{"options": {"num_predict": "many"}}` or a non-string
`steer` are request validation errors, but they can become server errors instead
of Ollama-shaped 400s.

Suggested fix:

Catch `TypeError` / `ValueError` around option resolution and return a 400, or
raise a `SaklasError` subtype covered by the app exception handlers. Add request
tests for malformed `num_predict`, penalties, and `steer`.

### 5. Medium-low: stateless zero-token generations with probes can leak prior readings

Evidence:

- `saklas/core/generation_finalizer.py:61` only builds stateless per-run probe
  readings when `session._monitor.probe_names and generated_ids`.
- If `generated_ids` is empty, the function falls through to
  `readings = session.build_readings()` at `saklas/core/generation_finalizer.py:101`.
- `build_readings` returns accumulated monitor history from prior generations at
  `saklas/core/session.py:4832-4847`.
- `GenerationConfig.max_new_tokens` is allowed to be zero by the core loop shape
  (`saklas/core/generation.py:993` documents the degenerate no-forward case).

Impact:

A stateless call that produces no tokens can return the session's historical
`ProbeReadings` instead of an empty per-call reading. That violates stateless
isolation and can confuse API clients that rely on `readings` as current-run data.

Suggested fix:

When `stateless=True`, return `{}` whenever there are no generated tokens to score.
Keep `build_readings()` only for stateful generation history. Add a finalizer unit
with probes attached, prior monitor history, `stateless=True`, and
`generated_ids=[]`.

### 6. Medium-low: `PATCH /sessions/{id}` accepts `thinking` but drops it

Evidence:

- `PatchSessionRequest` includes `thinking` at
  `saklas/server/session_models.py:22-28`.
- `patch_session` applies temperature, top-p, top-k, max tokens, and system prompt
  only at `saklas/server/session_routes.py:56-67`.
- `session_config_dict` omits `thinking` at `saklas/server/session_models.py:31-39`.
- The web sampling strip sends `thinking` defaults (`webui/src/panels/SamplingStrip.svelte:190`),
  so the UI believes the value is persisted.

Impact:

The endpoint returns 200 while silently ignoring a field exposed by the schema and
web client. Users can toggle a default that does not survive the round trip.

Suggested fix:

Either make thinking a real persisted session default and echo it in
`session_config_dict`, or remove it from the PATCH schema/web persistence path and
keep it explicitly per-call.

### 7. Medium-low: native session model-id aliases are unreachable for slash-bearing HF ids

Evidence:

- `resolve_session_id` accepts either `"default"` or `session.model_id` at
  `saklas/server/native_common.py:13-18`.
- Native routes use a normal path segment like
  `/saklas/v1/sessions/{session_id}` at `saklas/server/session_routes.py:39`, not
  a path converter.

Impact:

Normal HuggingFace model ids such as `google/gemma-...` contain `/`, so the
advertised model-id alias cannot match the route as written. Only `default` is
reliably reachable.

Suggested fix:

Document/support only `default`, or introduce an explicit slash-safe alias or
query parameter. Avoid blindly changing all session routes to `{session_id:path}`
without checking subroute shadowing.

### 8. Performance: gate-only scoring still captures every attached probe layer

Evidence:

- `_begin_capture` starts from `self._monitor.probe_layers()` at
  `saklas/core/session.py:3376`.
- `probe_layers()` is the union of all attached probe layers
  (`saklas/core/monitor.py:186`).
- In `GATING_SUBSET`, only scoring is narrowed (`saklas/core/session.py:3442`);
  the full tail ring is still kept at `saklas/core/session.py:3447-3450` so
  finalization can compute the full aggregate once.

Impact:

For a pure control call with many attached probes but only one scalar gate, decode
still copies/captures the layer union for the full roster on every token. That is
correct for calls that need final full probe aggregates, but it leaves a real
performance win on the table for generation while steering/probing only for gates.

Suggested fix:

Add an explicit "gate-only, no final probe aggregate" mode or request flag. In
that mode, derive capture layers from the gated probes/keys instead of the full
roster, and return no full `probe_readings` aggregate.

### 9. Performance: live J-lens readout rebuilds the transport stack every token

Evidence:

- `enable_live_lens` moves each selected `J_l` to device once in a dict at
  `saklas/core/session.py:2159-2165`.
- `_live_lens_readout_step` then loops each token, calls `.to(torch.float32)` for
  each selected `J_l`, and stacks them at `saklas/core/session.py:2206-2217`.

Impact:

Live lens already performs expensive per-layer `J_l h` and vocab top-k work; the
per-token cast/stack churn adds avoidable allocations and dispatch overhead on the
same interactive path where users are simultaneously generating, steering, and
probing.

Suggested fix:

Precompute a stable `layers` list plus a single stacked fp32 transport tensor in
`enable_live_lens`, or cache stacks by selected layer subset. At step time, gather
only hidden rows for layers that have a latest slice and index into the cached
transport stack.

### 10. Observability: batched fast-path `tok_per_sec` is per-row tokens divided by batch wall time

Evidence:

- The fast path measures one `elapsed` around the whole `model.generate` batch at
  `saklas/core/session.py:6433-6436`.
- It reuses that same elapsed value for every row at
  `saklas/core/session.py:6473-6487`.
- Result construction computes `tok_per_sec = len(generated_ids) / elapsed` at
  `saklas/core/session.py:6159-6162` and
  `saklas/core/generation_finalizer.py:28-34`.

Impact:

For batch size N, each row reports a rate based on the whole batch wall time, so
per-result `tok_per_sec` is not comparable to serial generation and can understate
aggregate throughput by roughly the batch size. This is an instrumentation issue,
not a generation bug.

Suggested fix:

Expose batch-level throughput separately, or mark per-row `elapsed` / `tok_per_sec`
as row latency rather than throughput. If keeping the current shape, consider
adding a `batch_tok_per_sec` metric to batch results or experiment summaries.

### 11. Cleanup: `_manifold_is_affine` remains load-bearing internally after promotion

Evidence:

- The old alias is retained in `saklas/core/session.py:534-537`.
- `saklas/core/steering_composer.py:41-43` imports the underscore alias and uses
  it at `saklas/core/steering_composer.py:831`.
- `tests/test_boundary_guard.py:159-171` marks `_manifold_is_affine` as a
  promoted old name but only scans frontend dirs (`tui`, `server`, `cli`), so the
  internal use is invisible to the guard.

Impact:

The public promotion is incomplete: new internal code can keep depending on the
compat alias while the regression guard still passes.

Suggested fix:

Import `manifold_is_affine` directly from `saklas.core.manifold` in
`steering_composer`. Then either expand the guard to cover internal modules or
explicitly document the alias as a public compatibility promise.

### 12. Cleanup/docs: active web UI README documents removed API routes

Evidence:

- `webui/README.md:105` says diagnostics powers `saklas vector why`.
- `webui/README.md:110` advertises `/vectors/{merge,clone}`.
- `webui/README.md:111` advertises `/saklas/v1/packs*`.
- Current web AGENTS notes say there is no `/saklas/v1/packs*` and no
  `/vectors/clone` at `saklas/web/AGENTS.md:33`.

Impact:

The active web UI README points contributors at pre-4.0 routes and commands. This
is stale documentation on a live development surface, not just archived notes.

Suggested fix:

Update the README to `saklas manifold why`, `/vectors/bake`, and the current
`/manifolds` install/search routes. Consider a small route-doc parity test for
removed endpoints if this drift keeps recurring.

### 13. Cleanup/docs: root AGENTS has version and `--method` drift

Evidence:

- `AGENTS.md:19` says the current version is 4.0.0, while
  `saklas/__init__.py:8` is 4.2.0.
- `AGENTS.md:59` documents `manifold fit --method`, but `AGENTS.md:107` says
  there is no `--method` surface.
- The live parser exposes `--method` for fit and merge at
  `saklas/cli/parsers.py:353-357` and `:516-519`.

Impact:

The contributor contract contradicts the live CLI. New work against this file can
remove or avoid a valid surface because the doc says it does not exist.

Suggested fix:

Remove the volatile "currently ..." version parenthetical or update it as part of
release work. Scope the "no `--method`" sentence to removed vector extraction /
steering-mode knobs, not discover-manifold fit/merge.

### 14. Low: WebSocket `n <= 0` is silently coerced to one generation

Evidence:

- `saklas/server/ws_stream.py:322` sets `n = msg.n if msg.n and msg.n > 0 else 1`.
- The intended validation block at `saklas/server/ws_stream.py:323-328` is then
  unreachable for explicit `n=0` or negative values.

Impact:

Invalid client input is treated as "generate once", which can make UI bugs or
malformed clients produce real work instead of a clear 400.

Suggested fix:

Default only when `msg.n is None`; reject explicit values below 1.

### 15. Low: Ollama `/api/push` points users at a non-existent CLI verb

Evidence:

- The error message says to use `saklas manifold push` at
  `saklas/server/ollama.py:416`.
- `manifold` verbs exclude `push` at `saklas/cli/parsers.py:76-86`, while `pack`
  owns lifecycle publishing and includes `push` (`saklas/cli/parsers.py:92+`).

Impact:

Users who hit the unimplemented Ollama push route receive an incorrect recovery
command.

Suggested fix:

Change the message to `saklas pack push`.

### 16. Low/product consistency: custom generated manifolds are CLI/server-only, not web-expressible

Evidence:

- The server accepts `kind: "custom"` plus `custom_system` at
  `saklas/server/manifold_routes.py:171-174`.
- The web type allows only `"abstract" | "concrete"` at
  `webui/src/lib/types.ts:501-508`.
- The builder UI only offers abstract/concrete radio buttons at
  `webui/src/drawers/ManifoldBuilderDrawer.svelte:408-409` and `:1171-1172`.

Impact:

One authoring mode is supported by CLI/API but invisible to the web builder. That
may be intentional power-user scope, but the types make it look unsupported rather
than deliberately CLI/API-only.

Suggested fix:

Either add a custom kind + system-prompt field to the web builder, or explicitly
document it as CLI/API-only and keep the web type narrowed by design.

## Verification Notes

- Completed subagent passes: generation/steering/probing hot path;
  CLI/server/web/API consistency; redundancy/dead-code/docs drift.
- Local verification: source inspection with `rg`, `nl`, and targeted reads of
  `ARCHITECTURE.md`, `saklas/core/AGENTS.md`, server/web AGENTS, generation,
  session, hooks, monitor, steering composer, server route, and web UI files.
- A subagent ran `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
  tests/test_server.py tests/test_saklas_api.py tests/test_web.py
  tests/test_cli_verbs.py tests/test_logits.py`, yielding `199 passed`.
- Local isolated repro confirmed the stateless zero-token finalizer leak with a
  stub session. No model-loading or GPU tests were run for this audit.
