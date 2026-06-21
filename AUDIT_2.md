# saklas audit — round two — 2026-06-21

A second sweeping, opinionated pass, run after the round-one remediation
(`AUDIT.md`) landed. Deliberately targets what round one did not look at: the
two unaudited large files (`core/manifold.py` 4686 L, `core/monitor.py` 2676 L,
`cli/runners.py` 2596 L), numerical edge cases, protocol fidelity, and taste.
Findings carry `file:line` evidence; the Tier-1 items were verified against the
code by hand. Checkboxes track remediation.

**Status legend:** `[ ]` open · `[~]` in progress · `[x]` done · `[-]` won't fix

## The shape of it

Round one didn't leave rot — it left a **propagation gap**. Almost every real
issue is the same move: a good pattern was established in the primary path and
not carried to its siblings. T1.2's stream-close went to OpenAI but not Ollama;
T1.1's bounded lock went to two paths but not the other twelve; the F1
post-forward scoring fix went to `session.py` but not the `joint_logprobs.py`
replay; T2.0's boundary guard bans `session._` but not module-level privates.

The orthogonal theme is **epoch residue**: the code says "concepts *are*
manifolds now," but *vector* is still load-bearing across the public surface,
and `TokenEvent` carries the same data under two names pending a "deferred
frontend rewire." Neither is decay — it's the residue of fast, correct
iteration.

The big-file verdict matches round one's finding about `session.py`:
`manifold.py`, `runners.py`, and the rest of `monitor.py` are **not** god
objects. Don't decompose them. There are exactly two clean seams worth cutting.

---

## Tier 1 — Real bugs / latent failures (verified)

- [x] **T1.1 — Ollama streaming leaks the GPU worker on disconnect.** *[confirmed]* — DONE: threaded `request` into `_stream_chat_or_generate`, added per-token `is_disconnected()` early-out + `finally: stream_iter.close()`, and a terminating `done:true` frame on both error paths. Test updated.
  `server/ollama.py:708–767` wraps `generate_stream` in `try/except
  ConcurrentGenerationError/SaklasError` with **no `finally: stream_iter.close()`
  and no `await request.is_disconnected()`**. A client that drops mid-stream keeps
  the engine generating to EOS, holding `_gen_lock` — and since all three
  protocols share the lock, it stalls every OpenAI and native request too. This is
  the exact `_stream_generation` fix from T1.2, never mirrored to Ollama.
  **Fix:** thread `request: Request` into `_stream_chat_or_generate`; add the
  per-token `is_disconnected()` early-out + a `finally` that calls
  `stream_iter.close()` if present. Also emit a final `done:true` frame on the
  error paths (754/760) so ollama-python / ChatOllama don't stall waiting for it.

- [x] **T1.2 — Twelve unbounded `async with session.lock` sites.** *[confirmed]* — DONE: all 12 routed through `acquire_session_lock` (300s → 503); the two WS-internal sites surface a `SessionLocked` WS error frame instead of an HTTP 503; long generate/fit JSON paths bound the acquire (SSE path unchanged).
  T1.1 bounded the two non-streaming chat paths via `acquire_session_lock`; these
  were left bare: `tree_routes.py:148,217`, `ws_stream.py:436,505`,
  `vector_routes.py:411,439`, `manifold_routes.py:655,696,822,850,888,992`. The
  dangerous ones are the `manifold generate`/`fit` JSON paths (822, 992) — they
  hold the lock for the entire generation/fit (minutes), so a client that fires the
  non-SSE path and disconnects pins the lock with no 503.
  **Fix:** route the quick I/O mutations through `acquire_session_lock` (300s →
  503); for the long generate/fit JSON paths, either bound the same way or reject
  the non-SSE path with a 400 "use SSE for long-running operations."

- [x] **T1.3 — Three exceptions escape the `SaklasError` family.** *[confirmed]* — DONE: `AlignmentError`/`WebUINotBuilt`/`AlphaListError` now multi-inherit `SaklasError` (stdlib base first) with `user_message()`; added `ManifoldNotFoundError`/`ManifoldExistsError` (404/409) and swapped the 5 bare `FileNotFoundError`/`FileExistsError` raises in `io/manifold_lifecycle.py`.
  AGENTS.md asserts every saklas exception multi-inherits through `SaklasError`.
  Violated by `AlignmentError(ValueError)` (`io/alignment.py:163`),
  `WebUINotBuilt(RuntimeError)` (`web/routes.py:18`), `AlphaListError(ValueError)`
  (`tui/loom_helpers.py:30`). The first two are user-facing (`manifold transfer`,
  dashboard-not-built) and **bypass the server's `_on_saklas_error` handler →
  unformatted 500 + traceback**.
  **Fix:** add `SaklasError` to the three MROs with `user_message()` →
  `(422, …)` / `(500, …)` / `(400, …)`. The stdlib base stays first so existing
  `except ValueError`/`RuntimeError` sites keep working. Consider also promoting
  the bare `FileNotFoundError`/`FileExistsError` raises in
  `io/manifold_lifecycle.py` to `ManifoldNotFoundError(FileNotFoundError,
  SaklasError)` / `ManifoldExistsError(FileExistsError, SaklasError)` for 404/409.

- [-] **T1.4 — F1 fix not propagated to the joint-logprob replay.** — WON'T FIX (not a bug).
  On close reading, `joint_logprobs.py:573` already fires `gating_callback()` in the
  same post-forward slot the session path uses: after `_call_model` returns (line 570),
  before KV-cache extraction (577) and logits read (581) — bit-identical ordering to
  `generation.py`'s `score_callback` (post-`model()`, pre-logits). The audit agent's
  "drains the pipeline" framing applies equally to both paths and isn't the F1 concern
  (which was syncing *mid-forward* from inside a hook). No change warranted.

- [x] **T1.5 — Affine fallback in `_manifold_layer_shares` crashes flat manifolds.** *[confirmed]* — DONE: the Euclidean fallback now branches on `sub.is_affine` (uses `‖node_coords‖`, 1.0 when absent) and only calls `rbf_params()`/`eval_rbf` on curved subspaces.
  `hooks.py:972` — the Euclidean share fallback (comment: "for CPU test stubs")
  unconditionally calls `sub.rbf_params()`, documented in `core/AGENTS.md` as
  **raising on a flat subspace**. So it breaks for exactly the affine manifolds
  (every 2-node concept, `personas`) it claims to support. Production never hits it
  (whitener mandatory → baked share always present), so **medium**, not a prod
  crash — but it blocks CPU-test instantiation of flat manifolds without a neutral
  cache, against the comment's promise.
  **Fix:** guard on `sub.is_affine`; use `‖sub.node_coords‖` as the affine spread.

- [x] **T1.6 — `-O` strips invariant-guarding asserts.** *[confirmed class]* — DONE (with T3.1/T3.2): `manifold.py:4316` + the moved `compute_node_behavior_centroid` assert → `raise SaklasError`; `monitor.py:556` → early-return of the empty `ProbeReading`.
  `manifold.py:4316` (corrupt-sidecar affine load), `manifold.py:4608`,
  `monitor.py:556` (narrowing before a `torch.cat` that would otherwise `TypeError`
  on `None`). Under `python -O` these become silent-wrong-answer / confusing
  downstream crashes.
  **Fix:** convert the few that guard real invariants to `if … is None: raise
  SaklasError(...)`.

---

## Tier 2 — The taste headline: epoch residue

- [x] **T2.1 — Retire "vector" from the public Python surface.** — DONE (additive): `session.profiles` already existed (returns the raw mutable registry, used pervasively) so `vectors`' docstring now points to it as canonical; `GenerationResult.steering_alphas` property added aliasing the `vectors` field (field + wire key kept for compat); `steering_composer.snapshot_steering_alphas` got a clarifying docstring noting a `ManifoldTerm`'s coeff is its `along` slide-fraction, not an additive alpha (no behavior change). Wire/CSV keys deliberately left unchanged.
  `session.vectors` returns `dict[str, Profile]` (`session.py:1233`);
  `GenerationResult.vectors` (`results.py:167`) is a `{name: coeff}` alpha dict that
  also mislabels `ManifoldTerm.along` as additive strength
  (`steering_composer.py:932`). In a world where "a vector is the 2-node flat case
  of a manifold," a user reaches for `session.vectors`, finds `Profile`s, and burns
  time reconciling the name.
  **Fix (additive):** add `session.profiles` as canonical (deprecated `vectors`
  property shim); rename `GenerationResult.vectors` → `steering_alphas` with a
  deprecated `vectors` property; update `ResultCollector` column names. Leave the
  `/vectors/*` wire routes for compat. Ship the rename, drop shims next major.

- [x] **T2.2 — `TokenEvent` carries the same data twice.** — DONE: collapsed to one `probe_readings` field + a `scores` read-only property shim. VERIFIED behavior-preserving: the two payload keys (`readings`/`probe_readings`) are the same `agg` object, and the old `probe_readings` gate `_wants_live_token_scores` is set to `bool(live_scores)` (session.py:5386) — i.e. always equal to the `readings` population condition, so they never diverged. WS path reads the payload directly (untouched). Construction site + two tests migrated.
  `results.py:321,332` — `scores` and `probe_readings` are the same `dict[str,
  ProbeReading]`; the comment admits "kept distinct for the deferred frontend
  rewire." Every streaming consumer branches `event.scores or event.probe_readings`
  and the server normalizes `payload.get("readings") or payload.get("probe_readings")`.
  **Fix:** collapse to `probe_readings`; add a one-release `scores` property shim;
  migrate the `or`-branching consumers (`session.py:5371`, `app.py:1440`,
  `ws_events.py:76`).

- [x] **T2.3 — Public return types aren't exported.** — DONE: `ChoiceScores`/`ChoiceScore`, `parse_expr`/`format_expr`, `ManifoldTerm`/`ProjectedTerm`/`AblationTerm`, `SelectorError`/`AmbiguousSelectorError`, `ManifoldNotRegisteredError`/`VectorNotRegisteredError` added to `saklas/__init__.py` (`_EXPORTS`/`__all__`/`TYPE_CHECKING`). Import smoke-test passes.
  `score_choices`/`score_template` are public, but `ChoiceScores`/`ChoiceScore`
  (`core/scoring.py`) aren't in `__init__.py`. Same for `parse_expr`/`format_expr`,
  `SelectorError`/`AmbiguousSelectorError`, `ManifoldNotRegisteredError`, and the
  expression term types (`ManifoldTerm`/`ProjectedTerm`/`AblationTerm`).
  **Fix:** add them to `_EXPORTS`/`__all__`/`TYPE_CHECKING` in `saklas/__init__.py`.
  Consider moving `ManifoldNotRegisteredError`/`VectorNotRegisteredError` from
  `session.py` to `core/errors.py`.

- [x] **T2.4 — Boundary guard's module-level blind spot.** — DONE: promoted `_manifold_is_affine` → `core/manifold.py::manifold_is_affine` (session keeps a back-compat alias for `steering_composer`), `_export_gguf_manifold` → `export_gguf_manifold`, `_all_concepts` → `all_concepts` (alias kept; tests monkeypatch the module name), `_sanitize_hyperparams` → `sanitize_hyperparams` (fixed the underscore-in-`__all__` contradiction); all import sites in server/tui/cli/tests updated. `scoring.py`/`joint_logprobs.py` switched to public `session.model`/`tokenizer`. `test_boundary_guard.py` got a targeted regression check for the four names (a blanket module-level ban was scoped out — it would flag legitimate intra-frontend helpers; the candidates are recorded for a future decision).
  T2.0 banned `session._` instance reach-ins; module-level underscored functions
  still cross layers: `_manifold_is_affine` (imported by `manifold_routes.py:29`,
  `probe_routes.py:25`), `_export_gguf_manifold`/`_all_concepts`/
  `_sanitize_hyperparams` (imported by `runners.py`). `_sanitize_hyperparams` is
  even in `io.manifolds.__all__` — an underscore name in `__all__` is a
  contradiction. `scoring.py:122` / `joint_logprobs.py:463` also use
  `session._model`/`_tokenizer` where public `session.model`/`tokenizer` exist.
  **Fix:** promote the four to public names in their natural homes
  (`_manifold_is_affine` → `core/manifold.py::manifold_is_affine`); update import
  sites; switch the two core files to the public model/tokenizer properties; extend
  `test_boundary_guard.py` to flag module-level privates.

---

## Tier 3 — Two surgical extractions (and a "don't")

`manifold.py`, `runners.py`, and the rest of `monitor.py` are **not** god objects
— four agents independently declined to split them. Don't re-propose those
decompositions. Cut exactly these two seams:

- [x] **T3.1 — `manifold.py` → `core/naturalness.py` (~186 L).** — DONE: 7 naturalness-only functions moved (`to_hellinger`, `bhattacharyya_distance`, `fit_behavior_manifold`, `trajectory_naturalness`, `compute_node_behavior_centroid`, `compute_trajectory_distributions`, `_next_token_distribution`); `compute_node_centroid` left (shared primitive); import sites in `cli/runners.py` + `tests/test_naturalness.py` repointed; no re-exports.
  The module's first line claims "pure tensor math, no session/IO coupling," then
  `manifold.py:4500–4686` (the behavior-manifold/naturalness cluster) **calls
  `model(...)` directly** — raw HF forwards with KV cache + sampling. It's the one
  block violating the contract, imports `SamplingConfig`, and is called only by the
  naturalness experiment.
  **Fix:** extract `to_hellinger`, `bhattacharyya_distance`, `fit_behavior_manifold`,
  `trajectory_naturalness`, `compute_node_behavior_centroid`,
  `compute_trajectory_distributions`, `_next_token_distribution` to
  `core/naturalness.py`; update import sites; restore the pure-tensor contract.

- [x] **T3.2 — `monitor.py` → `core/monitor_attach.py` (~560 L).** — DONE: attach-time block (lines 2091–2677: `AttachedManifoldProbe`, `_LayerWhiten`, `_build_whitened_factors`, `_attach_manifold_probe`, `_compute_assign_bandwidth`, `_layer_geometry`, the affine helpers + `_woodbury_apply`) moved; monitor.py 2677→2057 L; `_layer_geometry`/`AttachedManifoldProbe`/`_build_whitened_factors`/`_attach_manifold_probe` imported back (no cycle — monitor_attach imports only mahalanobis + manifold). The M7 per-layer-R bandwidth concern left as a deferred note. T5.1 monitor items (double-invalidate, `_woodbury_apply` docstring) folded in here.
  The bottom third (`AttachedManifoldProbe`, `_LayerWhiten`, `_build_whitened_factors`,
  `_attach_manifold_probe`, `_compute_assign_bandwidth`, `_layer_geometry`,
  `monitor.py:2091–2677`) is attach-time algebra with **zero hot-path coupling** —
  runs once per `add_probe`, never during decode. Pure mechanical move; drops
  `monitor.py` to ~1600 L. `_layer_geometry` re-exports back for the hot path.

> Also worth a hard look during T3.2: `_compute_assign_bandwidth` takes a single
> `R` from the first layer (`monitor.py:2616`) while `_subspace_coords_for` notes
> rank can vary per layer after DLS prune — a silent log-volume-bias error on
> mixed-rank flat fits. Either compute the bias per-layer or assert rank
> uniformity and raise clearly.

---

## Tier 4 — Tests don't cover the hard parts

- [x] **T4.1 — Two capture modes have zero coverage.** — DONE: `test_finalize_lean_incremental_probe_path` + `test_finalize_gating_subset_probe_path` in `tests/test_generation.py` assert the tail-ring aggregate is the FULL reading (not a lean row) and the correct branch dispatch; spot-checked to fail when the behavior is broken.
  `LEAN_INCREMENTAL` and `GATING_SUBSET` (the F2/FIX-4 conditional-scoring paths the
  server and TUI actually select) appear in tests only as import lines; only `FULL`
  and `INCREMENTAL` are exercised. Add two ~30-line CPU tests mirroring
  `test_finalize_incremental_probe_path_does_not_stack_capture`.

- [x] **T4.2 — The geometrically-hard code is untested + the gain is unguarded.** — DONE (gain) / ALREADY-COVERED (topology): added `test_affine_steer_coeff_half_displaces_in_sane_whitened_band` (a 0.5 steer lands in `[1.0, 25.0]` whitened units — catches a 10× gain error either way without pinning the calibration). Topology turned out already well-covered by `tests/test_manifold_topology.py` (flat/line/curve/circle/ellipse/torus-T2/faint-ring fallback/false-positive-rate/PH/harmonic-dedup) — no new tests needed.
  `select_topology` (PH, the single-cycle faint-ring fallback), the curved
  foot-solver, and the gain constants have no behavioral guard —
  `_SUBSPACE_GAIN`/`_MANIFOLD_ALONG_GAIN` are tagged "due for recalibration" but
  nothing catches a 10× error. Add a CPU test: a synthetic whitened subspace, a 0.5
  steer lands in `[0.2, 3.0]` whitened units. Add a `select_topology` test over a
  synthetic ring vs flat fan.

- [x] **T4.3 — The webui is untested and carries a 600-line corpse.** — DONE (delete path): confirmed `parseExpression` had no caller (only `serializeExpression` is used) and deleted it + its exclusively-owned lexer/parser/helpers (`expression.ts` 705 → 130 L). `npm run check` clean; `npm run build` leaves the committed `dist/` bundle byte-identical (the dead code was never bundled). Vitest setup deliberately NOT added (out of scope for a dead-code removal).
  No Vitest, no `test` script. `webui/src/lib/expression.ts::parseExpression` (a
  full grammar parser with a documented round-trip invariant) has **no caller** —
  dead, and free to drift from the Python `steering_expr`.
  **Fix:** delete it, OR add Vitest + a round-trip test against the Python
  `test_steering_expr` fixtures and wire it into deserialization. Decide and commit.

- [x] **T4.4 — Shared fakes never consolidated → a vacuous test.** — DONE (partial, by design): fixed the concrete bug — `test_web.py` mocked `session._monitor` while the server reads public `session.monitor` (vacuous assertions) → now assigns through the public surface. Created `tests/_fakes.py::make_mock_session`. The 4 server mock factories were NOT force-merged — only the genuinely-shared wiring lives in `_fakes.py`; `test_server`/`test_saklas_api`/`test_vectors_diagnostics_api` keep their divergent factories (Ollama-specific model_info, real trait-queue/EventBus logic, private-vs-public monitor attrs) with documented reasons rather than a lossy merge.
  The T4.3-followup `tests/_fakes.py` never landed; four server test files each
  define a drifting `_mock_session()`. One (`test_web.py:71`) mocks
  `session._monitor` while the server reads public `session.monitor`, so those
  assertions pass against an unconfigured auto-mock.
  **Fix:** `tests/_fakes.py` with one `make_mock_session()`; migrate the four files;
  fix the `_monitor` → `monitor` assignment.

---

## Tier 5 — Cheap hygiene (batchable)

- [x] **T5.1 — Trivia sweep.** — DONE: monitor items (double `_invalidate_flat_cache`, `_woodbury_apply` docstring) in Phase 2; `f"-{x*-1.0:g}"` → `abs(x)`; `_resolve_atom`'s vestigial `default_namespace` param + always-`+1` `sign_flip` return removed (3 call sites simplified); `_reduced_tangent` deleted (truly dead), `_off_surface_var` marked `# test-only` (used by `test_manifold_math.py`).
  Original list: double `_invalidate_flat_cache()` (`monitor.py:1926–1927`);
  `_woodbury_apply` docstring "Shared by `Monitor` and `Monitor`" (`monitor.py:54`);
  `f"-{x * -1.0:g}"` → `-x`/`abs(x)` (`steering_expr.py:1203`); dead singular
  `_reduced_tangent`/`_off_surface_var` (test-only, move or mark);
  vestigial `default_namespace`/`sign_flip` in `_resolve_atom` (`steering_expr.py:762`).
- [x] **T5.2 — CLI consistency.** — DONE (6 fixes): `serve` now accepts its model from `-c config.yaml` (`nargs="?"` + post-config guard, symmetric with `tui`); `template rm` resolves bare names cross-namespace via `resolve_template`; `config show` emits `compile`/`cuda_graphs`/`return_top_k`; `pack search` exits 1 on network/import failure; `_setup_steering_vectors` errors go to stderr; `template ls/show/score` `--json` unified to `dest="json_output"`. The `_boot_session` extraction was scoped out (larger refactor, noted below). Original list: `serve` can't take its model from `-c config.yaml`
  while `tui` can (`parsers.py:15`, asymmetric `nargs`); `template rm` resolves bare
  names to `local/` only while `show`/`score` cross-namespace (`runners.py:1792`);
  `config show` drops `compile`/`cuda_graphs`/`return_top_k` (`config_file.py:79`);
  `pack search` exits 0 on network failure (`runners.py:1881,1884`); error prints to
  stdout not stderr (`runners.py:282,295`); `template` uses `args.json` while the
  rest uses `args.json_output` (`parsers.py:781,786,814`). The CLI agent's
  `_boot_session()` extraction folds the serve/tui asymmetry away — worth it.
- [ ] **T5.3 — GCV perf (optional).** `_gcv_select_lambda` (`manifold.py:745`) runs 40
  sequential LU solves per layer per candidate; an eigendecomposition-once approach
  makes it 40 scalar ops. Dominant cost of `auto` fit on large discover manifolds;
  not blocking.

## Still open from round one

- [x] **Periodic `eff_along` clamp** (`hooks.py:1205`, documented deferred in AGENTS.md). — DONE: periodic `BoxDomain` fits now drop share-weighting and clamp `eff_along` to `[0,1]` (`max(0, min(1, along·_MANIFOLD_ALONG_GAIN))`, uniform per layer); non-periodic curved fits keep the share-weighted unclamped path. Test redesign: the 3 share-weighting tests repointed to a new non-periodic `_manifold_open` helper (their mechanics still hold there), + a new `test_manager_periodic_clamps_and_drops_share_weighting`. The AGENTS.md "Deferred fix"/"CAVEAT" note can now be dropped (docs pass).
  On periodic fits (`months`), share-weighted `eff_along` can exceed 1 and wrap the
  ring, so `_MANIFOLD_ALONG_GAIN = 4.0` rides a coherent wrap rather than being a
  safe magnitude — fragile to any share/gain change. **Fix:** clamp curved
  `eff_along` to `[0,1]` + drop share-weighting on periodic `BoxDomain` axes (the
  three lines the AGENTS.md note specifies). NOTE: a first cut was implemented +
  reverted during Phase 1 — it's correct but forces a real test redesign (3
  `test_manifold_steering.py` tests use a *periodic* `_manifold` helper to assert
  the share-weighting / unclamped-along mechanics, which the fix removes only for
  periodic fits). Needs its own focused phase: repoint those 3 tests to a
  non-periodic curved helper (the mechanics still hold there) + add a periodic
  clamp/uniform test. Deferred, not dropped.

---

## Recommended order

1. **Tier 1** — the latent/operational bugs. Independent files, mostly parallel.
2. **Tier 3** — the two extractions; mechanical, make the geometry testable.
3. **Tier 2** — the public-surface cleanup as one deliberate additive pass.
4. **Tier 4** — tests + webui.
5. **Tier 5** + the periodic clamp — opportunistic hygiene.

**Resist:** splitting `manifold.py`/`runners.py` wholesale (the seams aren't real),
and any big-bang public rename that removes old names in one shot — every Tier-2
fix is additive with a one-release shim. The lesson from round one held: the call
is **sequencing and propagation**, not scope.
