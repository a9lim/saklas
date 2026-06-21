# saklas audit — 2026-06-21

A sweeping, opinionated pass for accumulated debt. Verified findings with
`file:line` evidence and concrete fixes. Checkboxes track remediation.

**Status legend:** `[ ]` open · `[~]` in progress · `[x]` done · `[-]` won't fix

## The shape of it

Disciplined, mature codebase — full pyright + ruff + pre-commit, zero stray
`TODO`/`FIXME`, CI that diffs the committed Svelte bundle. The `core/` engine is
well-factored *at the module level*. The rot is concentrated and structural, with
one dominant shape: **clean primitives, god-object conductors, joined by a
fictional frontend/engine boundary.** Everything below ladders off that.

---

## Tier 1 — Real issues (fix regardless of taste)

- [x] **T1.1 — Non-streaming generation lock is unbounded.** *[confirmed bug]* — DONE: both non-streaming paths (`app.py::_run_blocking`, `ollama.py::_run_and_build_chat_response`) now use `acquire_session_lock` → 503.
  `server/app.py:801` (`_run_blocking`) and `server/ollama.py:600` use bare
  `async with session.lock`; streaming, `experiment fan`, `template score`, and
  Ollama-streaming all use the bounded `acquire_session_lock` (→ 503 after 300s).
  A non-streaming `/v1/chat/completions` request **queues forever** behind a stuck
  generation while its streaming sibling cleanly 503s.
  **Fix:** route both non-streaming generation paths through `acquire_session_lock`.
  (`server/AGENTS.md` already describes the bounded behavior the code lacks — doc drift.)

- [x] **T1.2 — Streaming lifecycle has no disconnect detection.** *[robustness]* — DONE: `_stream_generation` got both `finally: stream_iter.close()` and a per-token `await request.is_disconnected()` early-out (`request` injected into the OpenAI handlers, defaults `None` for library callers).
  `_stream_generation` (`server/app.py:567`) holds the bounded lock for the entire
  stream and `for event in stream_iter:` has no `await request.is_disconnected()`
  check (the pattern exists in `traits_routes.py:51`). A client that disconnects
  mid-stream keeps the GPU busy until generation naturally ends; teardown rides
  CPython refcount finalization (`session.py:5868` `finally` → stop-flag +
  `worker.join(timeout=5.0)`) rather than an explicit close.
  **Fix:** explicit `is_disconnected()` poll + `stream_iter.close()` in a `finally`.

- [x] **T1.3 — `nodes_sha256` swallows every exception.** *[staleness-correctness]* — DONE: `except Exception` narrowed to `(TemplateNotFoundError, AmbiguousTemplateError)`; any other error now propagates instead of producing a stale-but-passing hash.
  `io/manifolds.py:849` wraps template resolution in a bare `except Exception:` and
  falls back to hashing the ref string, so a renamed/broken template silently
  degrades the staleness key — **a stale fit survives a template edit** with no warning.
  **Fix:** narrow to `TemplateNotFoundError`/`AmbiguousTemplateError`.

- [x] **T1.4 — `_per_token_probes` shortest-list `min`.** *[low]* — DONE: rows now driven by `n_tokens`; the existing per-probe `i < len(vals)` guard omits short probes per-row instead of truncating all.
  `server/saklas_api.py:662` — a lagging probe with fewer recorded tokens silently
  truncates the WS `done` payload's `per_token_probes`.
  **Fix:** use the per-probe `i < len(vals)` guard already in the comprehension.

---

## Tier 2 — Systemic: god objects behind a fictional boundary

Three of the four largest files are accretion-point god objects:

| Class/file | Size | Symptom |
|---|---|---|
| `SaklasSession` (`core/session.py`) | 6182 L, ~127 methods | model + extraction + steering + capture + scoring + probes + loom + history |
| `SaklasApp` (`tui/app.py`) | 5007 L, ~130 methods | dispatch + gen lifecycle + loom + extraction workers + A/B shadow + input history |
| `io/manifolds.py` | 3017 L | format I/O + authoring + lifecycle + cross-model transfer + bundled materialization |

- [x] **T2.0 — PREREQUISITE: promote a public engine API; ban `session._` in frontends.** — DONE: added public surface on `SaklasSession` (`monitor`/`manifolds`/`profiles`/`model_metadata`/`generation_state`/`gen_lock`/`joint_logprob_cache`/`loom_conflict_check`/`ensure_manifold_loaded`/`ensure_profile_registered`); migrated all frontend reach-ins (tui/app.py, server/*); added `tests/test_boundary_guard.py` (tokenize-based, empty allowlist). Verified: 2204 passed, pyright 0 errors.
  The TUI reaches into `session._monitor`/`_profiles`/`_manifolds`/
  `_ensure_manifold_loaded`/`_gen_state.emit_map` at **45+ sites**; the server into
  `session._gen_state`/`_last_result`/`_capture._per_layer`. You cannot safely
  decompose `SaklasSession` while two frontends depend on its private shape.
  **Fix (low-risk, additive):** promote the privates the frontends need to public API
  (`session.monitor` property, `loaded_profiles()`, `ensure_manifold()`, `model_info`,
  typed gen-state read), migrate call sites, add a lint rule banning `session._`
  outside `core/`. This is the gate for every other Tier-2 item.

- [x] **T2.1 — Split `io/manifolds.py`** — DONE: 3017 → 413 (re-export shim + shared constants)
  + `manifold_folder.py` (1053, format core), `manifold_authoring.py` (1203, create/merge),
  `manifold_lifecycle.py` (493, rm/clear/refresh/transfer/materialize). `transfer_manifold`'s
  compute lifted to `core/manifold.py::transfer_manifold_subspaces` (pure-tensor); io keeps only
  the folder orchestration. Shim re-exports every public name — zero import sites changed.
  Verified: 2242 passed, pyright 0 errors.

- [x] **T2.2 — Decompose `SaklasSession`** — DONE (partial, by design): extracted **`SteeringComposer`**
  (`core/steering_composer.py`, 936 L; ~550 L off `session.py`, push/pop-frequency so zero hot-path
  cost; owned `_steering_stack` moved with it). The plan agent **declined the other three with
  principled reasons** rather than force cosmetic splits: `CaptureCoordinator` — state is read/written
  by the per-token `_token_tap`, splitting hits a perf-protected loop; `Extractor` — already behind
  the `ModelHandle` protocol (session *is* the handle), extraction would only add shims;
  `LoomController` — generation-coupled half calls `_generate_core`/`generate`. **Finding:** the
  engine is more decoupled than the god-object framing assumed; only the steering subsystem was a
  clean seam. Verified: 2248 passed, pyright 0 errors.

- [x] **T2.3 — Decompose `SaklasApp`** — DONE: `app.py` 5007 → 3164 (−37%). Extracted
  `extraction_controller.py` (1159), `loom_controller.py` (861), `input_history_controller.py` (347);
  Textual `on_*`/`action_*`/bindings stay on the App as thin forwarders to controller logic.
  `GenerationController` declined (gen-lifecycle state is woven into Textual's per-token message pump).
  Each extraction behavior-reviewed for stranded cross-reads of the interlocking flags. Verified:
  2248 passed, pyright 0 errors.

- [x] **T2.4 — Make the capture-mode state machine legal-by-construction.**  — DONE: `CaptureMode` enum + `CaptureState` dataclass replace the boolean bag; public `HiddenCapture` accessors replace the private reach-ins (0 left).
  `_begin_capture` sets ~5 correlated booleans (`session.py:3220–3366`) that a 5-branch
  if/elif keeps consistent by hand — illegal combinations are representable. Replace with
  a `CaptureMode` enum + `CaptureState` dataclass; `_score_*` keys off the enum. Also
  removes the reach-into-`HiddenCapture`-privates smell (`session.py:3297,3326,5083,5261`
  poke `_step_sink`/`_handles`/`_per_layer` because the public API has wrong seams — the
  comment at 3290 admits it).

---

## Tier 3 — Duplication clusters (small, contained wins)

- [x] **T3.1 — 3× near-identical `_score_*`** (`session.py:3442/3474/3511`). Same empty-  — DONE: `_extract_coord_stream` + `_empty_readings`; the three `_score_*` are thin wrappers.
  `ProbeReading` literal, same `coords[0] if … else 0.0` loop; `_score_lean_incremental`
  is `_score_incremental`'s loop + a call to `_score_aggregate_only`.
  → one `_extract_coord_stream` + one `_empty_readings`.
- [x] **T3.2 — `extract` vs `extract_vector_from_corpora`** (`session.py:1705` vs `1825`)  — DONE: shared `_author_and_fit_2node` tail.
  are 90% identical, differing only in corpus source. → shared `_author_and_fit_2node`.
- [x] **T3.3 — 6× lock/phase/finally boilerplate** (`session.py:1248,1757,1854,1964,3684,5165`).  — DONE: `@contextmanager _model_exclusive(op)` at all 6 sites.
  → one `@contextmanager _model_exclusive(op_name)`.
- [x] **T3.4 — 3× protocol translation.** `_build_steering` (`saklas_api.py:498`) is a  — DONE: `_build_steering` collapsed into `_merge_steering`; shared `server/streaming.py::stream_finalizer` for SSE/NDJSON/WS.
  verbatim re-impl of `_merge_steering` (`app.py:361`); SSE/NDJSON/WS each re-derive
  finish_reason+usage+probe-aggregate. → collapse `_build_steering`; extract a shared
  `StreamFinalizer`.
- [x] **T3.5 — Split `saklas_api.py` (2339 L).** Registrar already delegates 6 modules;  — DONE: `saklas_api.py` split → `tree_routes.py`/`vector_routes.py`/`ws_stream.py`.
  extract the in-place outliers (`tree_routes.py`, `vector_routes.py`, `ws_stream.py`).
- [x] **T3.6 — Panel selection-cursor** reimplemented in `vector_panel.py:116` and  — DONE: `tui/selectable.py` SelectableListWidget base + worker decorator + `_commit_turn`.
  `trait_panel.py:130`; **worker boilerplate** duplicated across ~20 `run_worker` sites
  (`_start_commit_user`/`_start_commit_assistant` byte-identical).
  → `SelectableListWidget` base + `@worker_with_status` decorator.
- [x] **T3.7 — Selector tier-ladder is split-brain.** Resolution *steps* in  — DONE: one `resolve_bare_atom` owns the tier ladder; `resolve_pole` retired.
  `io/selectors.py`, *ordering/arbitration* in `core/steering_expr.py:979–1038`,
  documented in three places because no single function owns it. `resolve_pole`
  (`selectors.py:209`) is vestigial — 3 of 4 return slots dead.
  → one `resolve_bare_atom()` in `selectors.py`.

---

## Tier 4 — Hygiene

- [x] **T4.1 — Prune `scripts/`** — DONE: kept the 3 maintenance scripts; archived 7 doc-cited
  experiments to `scripts/experiments/` with the `docs/notes` + `docs/plans` links updated;
  `git rm`'d the 15 zero-reference orphans.
- [x] **T4.2 — `scripts/out/` is not gitignored** (confirmed; lone untracked dir). — DONE.
- [x] **T4.3 — Tests: add `conftest.py` + one shared fake-model factory.** — DONE: `tests/conftest.py`
  with `FakeLogitsModel`/`CharTokenizer`/`make_logits_model`; migrated `test_joint_logprobs` +
  `test_naturalness`. (Minor follow-up: could move the shared classes to a `tests/_fakes.py`
  helper to match the `_whitener.py` precedent — currently in conftest, works + green.)
- [x] **T4.4 — Test the two untested high-value modules.** — DONE: new `tests/test_hf_manifolds.py`
  (HF layer mocked, no network) and `tests/test_scoring.py` (CPU `score_choices`/`score_template`
  math via the shared fake).
- [x] **T4.5 — CI: add a coverage gate + Python-version matrix** — DONE: 3.11/3.12/3.13 matrix
  (`fail-fast: false`); informational `--cov=saklas --cov-report=term-missing` (no fail-under,
  `pytest-cov` added to `[dev]`).
- [ ] **T4.6 — Docs-as-map.** AGENTS.md (59KB, auto-loaded every session) + ARCHITECTURE.md
  (63KB) co-maintain the same gain constants and engine deep-dives in parallel; concrete
  drift already hit (T1.1, `server/AGENTS.md`). Make AGENTS.md a map (what-this-is, subtree
  pointers, commands, one-paragraph-per-subsystem); move engine internals into
  ARCHITECTURE.md. Target ≤20KB. Subtree AGENTS.md files are right-sized — leave them.

---

## Recommended order

1. **T1.1 + T1.2** — the two lock/stream issues. Small, concrete, latent.
2. **T2.0** — promote public API, ban `session._` in frontends. Low-risk, additive; gates everything.
3. **T4 hygiene** — scripts purge, gitignore, conftest + shared fake, coverage gate. Cheap, makes later steps safer.
4. **T2.1** — `io/manifolds.py` split + lift `transfer_manifold`. Lowest-risk decomposition; proves the pattern.
5. **T2.2 + T2.4** — `SteeringComposer` extraction + capture-mode enum. Cleanest engine seams.
6. **T3** duplication sweep — opportunistic, alongside the above.
7. **T4.6** docs-as-map — once the drift is confirmed.

**Resist:** starting with `tui/app.py` (T2.3, highest effort / lowest unit-test payoff) or
any big-bang rewrite. The god objects grew correctly by accreting real features; T2.0's
boundary work is what makes incremental extraction safe. The call is **sequencing, not scope.**
