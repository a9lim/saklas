# Audit follow-ups

Plan for the items selected from the codebase audit. Phases are ordered by ship order, not by audit tier — atomic IO and version single-sourcing land first because they're mechanical wins with low blast radius; the session.py extraction refactor lands last because it touches the most surface area.

Each phase is meant to be one or two PRs. Phases are independent unless noted. 1.5.0 is in flight; everything below is post-1.5.0 work.

---

## Phase 1 — Atomic IO and pack write safety

**Audit refs:** Tier 1 #1, #2, #4, #5.

**Why:** Plain `open(p, "w")` followed by `json.dump` corrupts the pack on SIGKILL or ENOSPC. `pull_pack()` runs the integrity check on the temp dir, then copies to the destination — a partial copy leaves a corrupt installed pack with no signal. `materialize_bundled()` overwrites a user's `pack.json` and `statements.json` in place with no log line. The format-version-from-future error tells users to run an upgrade script that can only go forward.

### 1a. Atomic JSON writes

Add a single helper, e.g. `saklas/io/atomic.py`:

```
def write_json_atomic(path: Path, payload: dict) -> None:
    # write to <path>.tmp in same dir, fsync, os.replace
def write_bytes_atomic(path: Path, data: bytes) -> None:
    # same shape for tensor sidecars / gitattributes
```

Use it at every call site that currently writes JSON or small files into `~/.saklas`:

- `saklas/io/packs.py:114` — `PackMetadata.write()`
- `saklas/io/packs.py:150` — `Sidecar.write()`
- `saklas/io/packs.py:385` — neutral statements copy on miss (binary copy; use `write_bytes_atomic` or a staging temp)
- `saklas/io/packs.py:415` — concept folder materialization
- `saklas/io/hf.py:167` and `:256` — `pull_pack` file copies (better handled by 1b below, but worth the wrapper for the JSON sidecar at `:268`)
- `saklas/io/hf.py:498`, `:501` — staging area `.gitattributes` / `README.md`
- `saklas/io/cache_ops.py:142`, `:164`, `:224` — install/copy paths

`hf.push_pack()` (`hf.py:439-536`) already stages then commits; that's the model. The local writes need the same discipline.

Same dir is important — `os.replace()` is only atomic on the same filesystem. `~/.saklas` is one volume so this is fine, but the helper should `Path.parent` the temp file rather than `tempfile.NamedTemporaryFile()` in `/tmp`.

### 1b. `pull_pack` integrity on the destination, not the temp

`saklas/io/hf.py:136-170`. Note that this is a bigger change than a copy wrapper: `pull_pack` currently deletes the target *before* any verified install (`hf.py:139-143`), so destination-atomicity requires reordering the overwrite protocol, not just an `os.replace()` on the JSON.

Concrete shape:

1. Compute target dir, but don't touch it yet.
2. Stage the whole pack into a sibling `<target>.staging/` (or `~/.saklas/.pull-tmp/<concept>/` — same volume either way), running every file copy and metadata write there.
3. Run `verify_integrity()` against the staging dir.
4. If a previous install exists at `<target>`, rename it to `<target>.bak` (atomic).
5. Rename `<target>.staging/` to `<target>` (atomic).
6. On success, rmtree `<target>.bak`. On failure between steps 4 and 5, restore from `.bak`.

`os.rename` of a directory is atomic on the same volume. The `.bak` window gives us crash-recoverable rollback. This shape also unblocks 1c (the same staging discipline applies to bundled materialization).

### 1c. `materialize_bundled` doesn't blow away user edits

`saklas/io/packs.py:365-417`. When `format_version` drift triggers an upgrade:

1. Compare on-disk `statements.json` against the bundled version (sha256 of canonical-JSON form).
2. If they differ, log a warning at INFO and skip the statements rewrite — only update `pack.json`. Add a marker `pack.json.user_modified = true` to make the skip stable across runs.
3. Always write a `pack.json.bak` next to the new `pack.json` for one upgrade cycle.
4. Emit a single INFO log line per upgrade: `materialize_bundled: upgraded <ns>/<name> v1 → v2 (format_version)`.

This preserves the documented behavior (auto-upgrade on stale format_version) without silently destroying customization.

### 1d. Format-version-from-future error message

`saklas/io/packs.py:67-73`. Branch on direction:

- `fmt_ver < PACK_FORMAT_VERSION`: current message ("run scripts/upgrade_packs.py").
- `fmt_ver > PACK_FORMAT_VERSION`: new message ("this pack was created by a newer saklas (format vN > local vM); upgrade saklas, or pass `--force-legacy` to attempt to load anyway").

`--force-legacy` doesn't have to actually exist yet — calling it out gives users a phrase to search for when we add it.

### Validation

- Existing `tests/test_packs.py` covers format_version and integrity; extend to cover (a) crash-during-write recovery (write half a JSON via the helper's `*.tmp`, kill before rename, then `ConceptFolder.load()` should still see the prior good version), (b) `materialize_bundled` skipping rewrite when statements differ, (c) clearer message on future format_version.
- Manual: `pkill -9` saklas mid-`pack install` and confirm next `pack ls` is clean.

### Rollback

Each sub-item is a separate commit. The atomic-write helper is purely additive; reverting any individual call site is safe.

---

## Phase 2 — Statements hash is a hard contract

**Audit ref:** Tier 1 #3.

**Why:** `probes_bootstrap.py:111-118` warns when `sidecar.statements_sha256` doesn't match the live `statements.json` and proceeds. Hand-edit statements after extraction and you silently use stale tensors. The check is in one code path; `is_stale()` exists in `packs.py:335-339` but isn't load-gated.

### Approach

1. In `ConceptFolder.load()` (or a tightly scoped wrapper), call `is_stale()` and raise a new `StaleSidecarError(SaklasError, ValueError)`.
2. The exception message names the concept, the model_id, and the concrete remediation: `saklas pack refresh <ns>/<name> -m <model>`.
3. Provide an opt-out env var or kwarg `allow_stale=True` on the loader for advanced workflows. Don't expose it as a CLI flag yet — make it conscious.
4. Convert the existing warning at `probes_bootstrap.py:115-118` into the exception path; remove the silent-proceed branch.

### Validation

- `tests/test_packs.py` already exercises sidecar staleness; flip the existing test to expect raise instead of warn.
- New test: extract, modify `statements.json`, attempt to load, expect `StaleSidecarError` with model_id in message.
- Smoke: run a TUI session against an installed pack with a hand-edited statements file and confirm the error is legible.

### Gotchas

This is a behavior break. Anyone running with hand-edited statements gets an error on next load. Worth a CHANGELOG line.

---

## Phase 3 — Single-source version via `[project] dynamic`

**Audit ref:** Tier 2.

**Why:** `pyproject.toml:7` and `saklas/__init__.py:3` both hardcode `"1.5.0"`. Six modules consume `__version__` at runtime: `core/vectors.py:508`, `io/probes_bootstrap.py:92`, `io/gguf_io.py:149`, `io/hf.py:263`, `server/ollama.py:445`, `cli/runners.py:531`. The CI consistency check (`.github/workflows/ci.yml:45-65`) catches drift but only after it's introduced.

### Approach

Use the PEP 621 dynamic-attr direction (literal stays in `__init__.py`, pyproject reads it via setuptools) rather than the inverse. Dynamic-attr is the more standard pure-Python pattern, doesn't need an `importlib.metadata` fallback for editable / not-yet-installed source trees, and doesn't introduce a runtime dependency on the install metadata being present.

### Steps

1. `saklas/__init__.py` — leave `__version__ = "1.5.0"` as the single literal. Six consumers stay unchanged.
2. `pyproject.toml`:
   ```toml
   [project]
   # remove: version = "1.5.0"
   dynamic = ["version"]

   [tool.setuptools.dynamic]
   version = {attr = "saklas.__version__"}
   ```
3. `.github/workflows/ci.yml:45-65` — delete the consistency check job; it's redundant once the literal lives in one place.
4. `.github/workflows/release.yml:19` already reads `pyproject.toml`'s `[project] version`. With dynamic, that key isn't a literal anymore. Switch the read to `python -c "import saklas; print('version=' + saklas.__version__)"` after a `pip install -e .` step.

### Validation

- `python -m build` produces a wheel whose metadata `Version:` matches `saklas.__version__`. Phase 9a is the gate that confirms this on every PR.
- `pip install dist/saklas-*.whl && python -c "import saklas; print(saklas.__version__)"` round-trips.
- Bump `__version__` to `1.5.1`, run CI, confirm release workflow tags `v1.5.1`.

### Gotchas

`setuptools.dynamic.attr` parses the AST without executing the module on modern setuptools (verified against ≥81). `saklas/__init__.py` does eager re-exports including `from saklas.core.session import SaklasSession`, which transitively imports torch — but AST parsing reads the literal at `__init__.py:3` without running any imports. The Phase 9a wheel build is the validation gate; if it ever breaks because a future setuptools changes resolution semantics, fall back to `[tool.setuptools.dynamic] version = {file = "saklas/_version.txt"}` with a one-line `_version.txt` and `__init__.py` reading it via `importlib.resources`. Either way, the goal is one literal.

---

## Phase 4 — `SaklasError.user_message()` centralization

**Audit ref:** Tier 4 #13.

**Why:** Three surfaces translate exceptions to user-facing strings independently. `server/app.py:514-529` maps exception type to HTTP status + JSON shape; `cli/runners.py` does ad-hoc `print(..., file=sys.stderr); sys.exit(N)` in at least four places (`:116`, `:287`, `:439`, `:723`); `tui/app.py` does `add_system_message(str(exc))`. When an error message changes, three places drift.

### Approach

Add a method on `SaklasError`:

```python
class SaklasError(Exception):
    def user_message(self) -> tuple[int, str]:
        # default: (500/exit-1, str(self))
        ...
```

Subclasses override the int (HTTP status / exit code) and optionally rewrite the string. The convention: return code mapping is "HTTP-style" (`400` for bad input, `409` for conflict, `404` for not found, `500` for unexpected); CLI maps the int via `min(2, code // 100)` or a tiny table; TUI ignores the code.

Surfaces:

- `server/app.py:514-529` — replace the type-switch with `code, msg = exc.user_message(); return JSONResponse(..., status_code=code)`.
- `cli/runners.py` — wrap each runner in `try / except SaklasError as e: code, msg = e.user_message(); print(msg, file=sys.stderr); sys.exit(...)`. Best done as a decorator on each runner function.
- `tui/app.py` — wherever errors are surfaced, call `e.user_message()[1]` for the string.

### Migration

The default `user_message` returning `str(self)` keeps current behavior for any subclass that doesn't override. Override the four or five subclasses where the message benefits from being more user-facing (`AmbiguousSelectorError` should mention `disambiguate with ns/name`, `StaleSidecarError` from Phase 2 should mention `pack refresh`, etc).

### Validation

- Tests in `tests/test_server.py` and `tests/test_cli_flags.py` already pin some error messages — they should keep passing.
- New test: `assert SaklasError("x").user_message() == (500, "x")` and one override per subclass.

---

## Phase 5 — Layer decoupling: selectors and `cache_ops` formatting

**Audit refs:** Tier 4 #14, #15.

**Why:** `saklas/cli/selectors.py` defines the selector grammar and `resolve()`, and `saklas/io/cache_ops.py` imports it. `io/` depending on `cli/` is the wrong direction. The same coupling appears at `saklas/core/session.py:904`, `:1241`, `:1288`, where extraction reaches across into CLI — a problem Phase 7 inherits if we don't fix it here. Separately, `cache_ops.py` is 727 LOC and mixes cache mutation with `print(_json.dumps(...))` calls (`:543`, `:590`, `:612`, `:676`, `:700`); a programmatic caller of `cache_ops` shouldn't be making side-effect prints.

### 5a. Move selector grammar to a layer that both can import

Move from `saklas/cli/selectors.py` to `saklas/io/selectors.py`. CLI keeps a thin wrapper that adds argparse-friendly parsing helpers but delegates the grammar.

`saklas/cli/selectors.py:271 LOC` — most of it is the grammar; what stays is the argparse adapters, `format_resolution_error`, and similar.

After this move:
- `saklas/io/cache_ops.py` imports from `saklas.io.selectors` (same layer).
- `saklas/core/session.py:904`, `:1241`, `:1288` import from `saklas.io.selectors` (downward dependency, allowed).
- `saklas/cli/selectors.py` shrinks to argparse glue + re-exports for back-compat during the transition.

### 5b. Split cache mutation from CLI presentation

Move from `saklas/io/cache_ops.py` to `saklas/cli/runners.py` (or a new `saklas/cli/output.py`):

- `_print_list`, `_print_hf_rows`, `_row_from_concept`, and the `print(_json.dumps(...))` branches at `:543`, `:590`, `:612`, `:676`, `:700`.

`cache_ops.py` functions return structured results (`list[ConceptRow]`, `InstallResult` dataclasses); the CLI runner formats them. This is also a prerequisite for Phase 9c's T20 ruff rule, which would otherwise flag the intentional prints.

After both moves, `cache_ops.py` should be roughly 400-450 LOC and contain only IO + state mutation.

### Validation

- Existing `tests/test_packs.py` and `tests/test_cli_flags.py` cover both surfaces. After the move, packs tests should not import `cli.runners`; CLI tests should not import `io.cache_ops` directly.
- One new lint check (or just a CI grep): `grep -rn "import.*cli" saklas/io/ saklas/core/` returns nothing.

---

## Phase 6 — Generation state machine

**Audit ref:** dropped from initial cut, restored on Codex's pushback.

**Why:** `SaklasSession` and the TUI both track an ad-hoc `_gen_active: bool`. On the session it's tangled with `_gen_lock`, `HiddenCapture` lifecycle, the steering scope context manager, and `monitor.begin_live()` / `end_live()` — five separate handles whose ordering invariants are encoded only in the cleanup paths (`session.py:1841-1978`, plus the defense check at `:1433` inside `_apply_steering`). On the TUI, `_gen_active` is read or written across 13 sites (`tui/app.py:132, 301, 361, 367, 413, 508, 757, 761, 774, 806, 851, 890, 1408, 1443, 1455`) — most of those are "is a gen running right now?" reads that should defer to the session. A boolean per surface is the wrong shape for a workflow with five ordered states and two surfaces that need to agree on which state we're in.

This is also the cleanest possible prerequisite for Phase 7. ExtractionPipeline shouldn't run while a generation is in flight, and the check is much cleaner against a typed enum than against a boolean shared with two finally-blocks.

### 6a. Session-side `GenState` enum

In `saklas/core/session.py`, replace `self._gen_active: bool` with:

```python
class GenState(IntEnum):
    IDLE = 0
    PREAMBLE = 1   # lock held, steering scope entered, capture not yet attached
    RUNNING = 2    # capture attached, monitor live, generate_steered active
    FINALIZING = 3 # capture detached, monitor end_live pending, lock not yet released
```

Transitions in `_generate_core` (line numbers below are pre-refactor):

| Site | Old | New |
| --- | --- | --- |
| `:1841-1846` (lock acquire + `_gen_active = True`) | bool flip | `IDLE → PREAMBLE` |
| `:1932` (after capture attach + monitor live + steering ctx reset) | implicit | `PREAMBLE → RUNNING` |
| `:1946` (inner `finally`, capture detached + steering scope exited) | implicit | `RUNNING → FINALIZING` |
| `:1975-1977` (outer `finally`, monitor end_live + lock release) | `_gen_active = False` | `FINALIZING → IDLE` |

The defense check at `:1843` becomes `if self._gen_state is not GenState.IDLE: raise ConcurrentGenerationError(...)`. The check at `:1433` inside `_apply_steering` becomes `if self._gen_state not in (GenState.PREAMBLE, GenState.RUNNING):`.

The threading `_gen_lock` stays — the enum doesn't replace the primitive, it makes the field that protects against re-entering after acquire typed and self-documenting. Lock + state is the right shape; the bug class the enum prevents is "we set a flag at one transition and forgot to clear it at another."

### 6b. Expose state to the TUI; collapse the duplicate boolean

Add a public read-only property on `SaklasSession`:

```python
@property
def gen_state(self) -> GenState: ...
```

And a convenience: `session.is_generating -> bool` returning `gen_state is not GenState.IDLE`.

Replace the TUI's `self._gen_active: bool` (`tui/app.py:132`). The 13 sites split:

- "Am I generating right now?" reads (most of them: `:301, 361, 367, 413, 508, 757, 761, 851, 1443, 1455`) → `self._session.is_generating`.
- Writes that flip the flag for UI-only reasons (`:774, 806, 890, 1408`) — these are tracking the *UI's* gen lifecycle, which differs slightly from the session's because the TUI counts a gen as "still going" until the `("done",)` sentinel lands on the local queue, even if the session has already returned to IDLE. Keep one minimal UI-side boolean for that — call it `_ui_gen_active` and document the distinction. It should not gate any session call; only UI-side things like Ctrl+R behavior.

So we go from "two booleans, one ambiguous" to "one typed session state plus one explicit UI flag with a one-line docstring."

### Validation

- New `tests/test_session.py::test_gen_state_transitions` constructs a session, asserts state is `IDLE`, runs a generation with an `on_token` callback that asserts state is `RUNNING`, asserts state returns to `IDLE` on success.
- New test: raise from inside the generation worker, assert state still returns to `IDLE`.
- Existing `test_concurrent_generation_rejected` covers the re-entry guard; should keep passing with the enum check.
- TUI tests in `test_tui_commands.py` should keep passing — the read-property change is mechanical.

### Sequencing

- Independent of Phases 1–5 and 9.
- **Hard prerequisite for Phase 7.** Phase 7 will touch session.py heavily; we don't want the extraction refactor and a state-machine refactor in the same session.py churn window.

### Gotchas

- Server side (`server/app.py`, `server/saklas_api.py`, `server/ollama.py`) doesn't read `_gen_active` directly — it uses `session.lock` (asyncio) and the threading lock indirectly. So Phase 6 is library-internal + TUI; the server changes nothing.
- `_gen_active` appears in a docstring at `session.py:1433-1434` describing the contract for `_apply_steering`. Update the docstring text alongside the check.

---

## Phase 7 — Extract `ExtractionPipeline` from `session.py`

**Audit ref:** Tier 4 #16.

**Why:** `saklas/core/session.py` is 2121 LOC. Roughly 900 LOC (audit cited 377-1280) handle concept extraction: folder probing, statement caching, scenario generation, pair generation, contrastive PCA invocation, pack updates. This is a self-contained pipeline tangled with session state.

### Approach

Take the structural shape, not the back-reference shape. The session passes in the dependencies extraction needs; the pipeline holds none of them as inherited "session" state. This is more upfront design than `def __init__(self, session): self._session = session`, but it pays off in testability and in clean boundaries against Phase 6's state machine.

The dependencies extraction actually needs (audited from `session.py:377-1280`):

- A `ModelHandle` carrying `model: PreTrainedModel`, `tokenizer: PreTrainedTokenizerBase`, `device: torch.device`, `dtype: torch.dtype`, `model_id: str`, `layers: list` (the `_layers` accessor result). Used by `_run_generator` (`:424-442`), SAE setup (`:836-842`), and the cache / DataSource / bundled / fresh paths (`:876-899, :955-969, :1054-1061`).
- A `PackWriter` protocol exposing `local_concept_folder(name) -> Path`, `promote_profile(profile, name)`, `update_local_pack_files(folder)`. These are the three private session methods extraction calls; lifting them into a protocol makes it explicit.
- A `VectorRegistry` protocol exposing `add(name, profile)`, `__contains__(name)`. Today this is `session._profiles`; the registry interface lets ExtractionPipeline write back without touching session internals.
- The session's `EventBus` for `VectorExtracted` emission.

```python
# saklas/core/extraction.py
class ExtractionPipeline:
    def __init__(
        self,
        model_handle: ModelHandle,
        pack_writer: PackWriter,
        registry: VectorRegistry,
        events: EventBus,
    ) -> None: ...

    def extract(
        self,
        concept: str,
        *,
        scenarios: list[str] | None = None,
        reuse_scenarios: bool = False,
        force_statements: bool = False,
        sae: SaeBackend | None = None,
    ) -> tuple[str, Profile]: ...
```

`SaklasSession.extract()` becomes a thin delegate that constructs (or holds) a pipeline and forwards. The session also gates the call: `if self.gen_state is not GenState.IDLE: raise ConcurrentExtractionError(...)`. This is exactly what Phase 6's typed state buys us — the gate is a one-line enum check, not "is this boolean set right now."

`ModelHandle`, `PackWriter`, and `VectorRegistry` start as runtime-checkable `Protocol`s (mirroring `SaeBackend`'s pattern in `saklas/core/sae.py`). The session implements them implicitly. Tightening into formal classes is a follow-up if the protocols prove leaky.

### Validation

- `tests/test_session.py` already exercises extraction end-to-end. It should keep passing without modification — this is a pure move at the API level.
- `saklas/core/session.py` LOC drops to ~1.2K; check via `wc -l`.
- New focused `tests/test_extraction_pipeline.py` constructs an `ExtractionPipeline` against a `MockModelHandle` (mirroring the `MockSaeBackend` pattern) and verifies cache-hit short-circuits, scenario reuse, and `force_statements`. CPU-only — far easier to write against the extracted class than against the full session.
- `grep -rn "import.*cli" saklas/core/extraction.py` returns nothing (validates Phase 5's prerequisite).

### Sequencing

This is the biggest churn in the plan. Land it last. Hard prerequisites:

- **Phase 5** — extraction's import of `cli.selectors` (`session.py:904, :1241, :1288`) has to move before extraction can leave session.py without dragging the wrong-direction CLI dependency along.
- **Phase 6** — the state-machine gate (`if self.gen_state is not GenState.IDLE`) replaces ad-hoc boolean checks the back-reference shape would otherwise inherit.

By the time we get here, Phase 1 has stabilized the IO writes the pipeline relies on, Phase 4's error-translation contract makes it easier to bubble extraction errors cleanly, Phase 5 has moved selectors to `io/`, and Phase 6 has typed the generation state we need to gate against.

### Gotchas

- Make sure `ModelHandle` is a *handle*, not a copy. The pipeline must see the same model object the session uses; otherwise device-side state diverges. Pass by reference, document the lifetime tie.
- `_run_generator` reads `self._tokenizer` and uses chat templates. The handle's `tokenizer` is the live one the session also holds — same object identity, not a clone.
- The `EventBus` is shared, not owned. Extraction emits `VectorExtracted`; subscribers (TUI trait panel, server SSE) keep working because the bus hasn't moved.

---

## Phase 8 — TUI slash command registry

**Audit ref:** Tier 4 #17.

**Why:** `saklas/tui/app.py:309-450` is a ~140-line if/elif chain dispatching slash commands. Each branch duplicates "missing argument" and "unknown command" error formatting.

### Approach

Registry-driven dispatch in `saklas/tui/commands.py`:

```python
@dataclass(frozen=True)
class SlashCommand:
    name: str
    handler: Callable[..., Awaitable[None]]
    usage: str
    min_args: int
    max_args: int | None  # None = unbounded

COMMANDS: dict[str, SlashCommand] = {
    "/steer": SlashCommand("/steer", _handle_steer, "Usage: /steer <expr>", 1, None),
    ...
}
```

`_handle_command` becomes:

```python
def _handle_command(self, raw: str) -> None:
    name, *args = raw.split()
    cmd = COMMANDS.get(name)
    if cmd is None:
        self.add_system_message(f"unknown command: {name}")
        return
    if len(args) < cmd.min_args or (cmd.max_args is not None and len(args) > cmd.max_args):
        self.add_system_message(cmd.usage)
        return
    asyncio.create_task(cmd.handler(self, *args))
```

Help output (`/help`) becomes a one-liner over `COMMANDS.values()`.

### Validation

- `tests/test_tui_commands.py` (689 LOC) covers slash dispatch. Should keep passing; if it doesn't, the new dispatcher's argument-count handling has drifted from the old branches.
- Manual: every documented slash command still works; `/help` shows them all.

---

## Phase 9 — CI hygiene

**Audit refs:** Tier 5 #19, #20, #21.

These are independent low-cost wins. Bundle into one CI PR.

### 9a. Build wheel on PRs

`.github/workflows/ci.yml` — add a job:

```yaml
build:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v6
    - uses: actions/setup-python@v6
      with: { python-version: "3.12" }
    - run: pip install build && python -m build
    - run: pip install dist/saklas-*.whl && python -c "import saklas; print(saklas.__version__)"
```

This is also the validation gate for Phase 3: setuptools.dynamic.attr is documented to read the AST without executing the module, but `__init__.py` does eager re-exports (`SaklasSession` at `:17`) that transitively import torch, so we shouldn't take it on faith. Land 9a *with or before* Phase 3 — if the wheel build fails on `python -m build` because dynamic-attr resolution tries to execute `__init__.py`, fall back to the `_version.txt` shape. Without 9a in place, a Phase 3 regression won't surface until release.

### 9b. `@pytest.mark.gpu` on tests that need a GPU

The marker is registered in `pyproject.toml:76-78` but no test carries it. Walk `tests/test_smoke.py` and `tests/test_session.py` (and any other test that conditionally skips on no-CUDA/MPS) and add `@pytest.mark.gpu` at the function level.

CI already runs `-m "not gpu"`, so the practical change is that the marker becomes accurate documentation. Side benefit: `pytest -m gpu` is now a meaningful local command for verifying GPU coverage.

### 9c. `T20` ruff rule + pyright in pre-commit

`pyproject.toml:90`:

```toml
[tool.ruff.lint]
select = ["E", "F", "W", "T20"]
```

T20 catches `print` and `pdb.set_trace` / `breakpoint()`. Phase 5 moved the intentional `print()` calls out of `cache_ops.py`, so this lands cleanly here.

`.pre-commit-config.yaml`:

```yaml
- repo: https://github.com/RobertCraigie/pyright-python
  rev: v1.1.400
  hooks:
    - id: pyright
      additional_dependencies: [torch, transformers, fastapi, textual, pyyaml]
```

Pyright in pre-commit catches the public-API regressions before they hit CI. The dependency list mirrors what CI installs.

### Validation

- New `build` job goes green on its own PR.
- Run `pytest -m gpu` locally; should select exactly the tests that need a GPU.
- `pre-commit run --all-files` clean.

---

## Sequencing summary

| Phase | Audit refs | Blast radius | Sequencing |
| --- | --- | --- | --- |
| 1 — Atomic IO | T1 #1, #2, #4, #5 | low | first |
| 2 — Statements hash hard | T1 #3 | medium (behavior break) | with 1 |
| 3 — `dynamic = ["version"]` | T2 | low | with or after 9a |
| 4 — `user_message()` | T4 #13 | low | independent |
| 5 — Selectors + cache_ops split | T4 #14, #15 | medium | before 7, before 9c |
| 6 — Generation state machine | restored | medium (touches session.py + TUI) | before 7 |
| 7 — `ExtractionPipeline` | T4 #16 | high (largest churn) | after 5 and 6, last |
| 8 — TUI command registry | T4 #17 | low | independent |
| 9a — Build wheel on PR | T5 #19 | low | with or before 3 |
| 9b — `@pytest.mark.gpu` | T5 #20 | low | independent |
| 9c — T20 + pyright pre-commit | T5 #21 | low | after 5 |

A reasonable cut against the next few releases: Phase 1 + 3 + 9a/9b bundle naturally as a hardening patch (mechanical, low blast radius). Phase 2 + 4 + 5 + 8 + 9c bundle as a cleanup release (Phase 2 is a behavior break, Phase 4 is a small public-API addition, the rest is structure). Phase 6 and Phase 7 are each substantial enough to deserve their own minor — Phase 6 because it changes a typed contract spanning session and TUI, Phase 7 because it's the largest churn in the plan and depends on 5 and 6. Exact version numbers are yours to call.
