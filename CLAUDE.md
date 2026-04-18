# CLAUDE.md

## What this is

`saklas` is a Python library + Textual TUI + dual-protocol HTTP server (OpenAI `/v1/*` **and** Ollama `/api/*` on the same port) for activation steering and trait monitoring on HuggingFace causal LMs. Representation Engineering via contrastive PCA; per-call alpha control; no model mutation. Three frontends over one engine: `SaklasSession` (programmatic), `saklas serve` (HTTP), `saklas tui` (TUI).

Version lives in **two places** — bump both or `saklas.__version__` drifts from PyPI:
- `pyproject.toml` → `[project] version`
- `saklas/__init__.py` → `__version__`

Releases: merge bump to `main` → `.github/workflows/release.yml` tags `v$VERSION`, builds, publishes via trusted publishing, cuts a GitHub release. Push without bumping → no-op.

## Subtree docs

Deep internals live in subtree `CLAUDE.md` files — Claude Code auto-loads these when working in the corresponding directory. Only consult them when editing that layer.

- `saklas/core/CLAUDE.md` — model loading, vectors, hooks, monitor, session, generation loop
- `saklas/server/CLAUDE.md` — OpenAI / Ollama / native `/saklas/v1/*` routes
- `saklas/tui/CLAUDE.md` — slash commands, highlighting, panels, status footer
- `saklas/cli/CLAUDE.md` — verb dispatch, config loading, warmup
- `saklas/io/CLAUDE.md` — packs, HF distribution, GGUF, cloning, datasource

## Commands

```bash
pip install -e ".[dev]"                         # editable + pytest
pip install -e ".[serve]"                       # fastapi + uvicorn
pip install -e ".[gguf]"                        # llama.cpp GGUF I/O
pip install -e ".[cuda]"                        # bitsandbytes + flash-attn (Linux/CUDA)
pip install -e ".[sae]"                         # SAELens-backed SAE extraction (sae-lens)
saklas tui <model_id>                           # TUI (explicit subcommand)
saklas serve <model_id>                         # OpenAI + Ollama API (dual-protocol)
saklas pack install <target> [-s|-a NS/N|-f]    # HF coord or folder; -s = statements only
saklas pack refresh <selector> [-m MODEL]       # re-pull; `refresh neutrals` is reserved
saklas pack clear <selector> [-m MODEL] [--variant raw|sae|all]  # delete per-model tensors (keep statements)
saklas pack rm <selector> [-y]                  # remove folder (bundled respawns)
saklas pack ls [selector] [-j|-v]               # LOCAL installed packs only
saklas pack search <query> [-j|-v]              # search HF hub for saklas-pack repos
saklas pack push <selector> [-a OWNER/NAME] [-p] [-m MODEL] [-s|-n] [-t] [-d] [-f] [--variant raw|sae|all]
saklas pack export gguf <selector> [-m MODEL] [-o PATH] [--model-hint HINT]
saklas vector extract <concept>|<pos> <neg> [-m MODEL] [-f] [--sae RELEASE [--sae-revision REV]]
saklas vector merge <name> <expression> [-m]    # shared steering grammar: "0.3 ns/a + 0.5 ns/b~ns/c"
saklas vector clone <corpus> -N NAME [-m MODEL] [-n N_PAIRS] [--seed S] [-f]
saklas vector compare <concepts...> -m MODEL [-v] [-j]       # cosine similarity between vectors
saklas vector why <concept> -m MODEL [-n N] [--all] [-j]     # top layers by ||baked||
saklas config show [-c PATH ...] [--no-default] [-m MODEL]
saklas config validate <file>
pytest tests/                                   # all; GPU tests gated on CUDA/MPS
```

Root parser has exactly five verbs — `tui`, `serve`, `pack`, `vector`, `config`. No `argv[0]` peeking, no top-level verb aliases, no bare-TUI fallback — `saklas google/gemma-2-2b-it` is an argparse error. Bare `saklas`/`saklas pack`/`saklas vector`/`saklas config` print help + exit 0.

Every subcommand that takes `-c/--config` auto-loads `~/.saklas/config.yaml` first, then composes any explicit `-c` files on top (later overrides earlier). `ConfigFile.effective(extras, include_default=...)` is the single entry point. `ConfigFile.vectors` is a steering expression string — parsed lazily through `saklas.core.steering_expr.parse_expr`, which resolves bare poles (`wolf → deer.wolf @ -0.5`) via `cli.selectors.resolve_pole` and produces the canonical `Steering` IR every surface speaks.

Selector grammar (shared): `<name>`, `<ns>/<name>`, `tag:<t>`, `namespace:<ns>`, `default`, `all`, optionally suffixed with `:<variant>` where `<variant>` is `raw` (default), `sae` (unique SAE variant for the concept), or `sae-<release>` (specific SAELens release). Bare names resolve cross-namespace and raise `AmbiguousSelectorError` on collision; bare `:sae` raises `AmbiguousVariantError` when more than one SAE release is extracted for a concept. `pack refresh` silently skips `source=local`; `pack refresh neutrals` rewrites `~/.saklas/neutral_statements.json` from the bundled copy. `pack rm` refuses broad selectors without `-y`; bundled concepts respawn on next session init. `pack ls` is local-only; HF hub search is `pack search`.

## SAE pipeline (optional)

`saklas vector extract <concept> --sae <release>` runs contrastive PCA in SAE feature space rather than raw residual-stream space, then decodes back to model space. Uses SAELens as the loader, so any published release it covers (GemmaScope, Eleuther, Joseph Bloom's, Apollo/Goodfire) works on day one. Install with `pip install -e ".[sae]"`.

Output tensors coexist with raw-PCA tensors in the same concept folder: `<model>.safetensors` (raw) alongside `<model>_sae-<release>.safetensors` (SAE). Select at steer time with the `:sae[-<release>]` suffix in the shared steering expression grammar — `session.steering("0.3 honest:sae")` from Python, `vectors: "0.3 honest:sae"` in the config file, `/steer 0.3 honest:sae` in the TUI. A bare `:sae` picks the unique SAE variant; explicit `:sae-<release>` picks one when multiple coexist.

SAE profiles are subset-layer (only layers the release covers). Share-baking redistributes over the covered subset — hook math and monitor scoring are unchanged. Release selection at extract time requires an explicit SAELens release name (SAELens ships many per base model — `gemma-scope-2b-pt-res`, `-mlp`, `-att`, etc. — and there's no sensible implicit default); if multiple SAEs exist per layer within a release, saklas picks the narrowest-width and emits a warning.

Sidecar JSON records `sae_release`, `sae_revision`, `sae_ids_by_layer` alongside the usual `method: "pca_center_sae"` / `statements_sha256` fields. `pack push --variant sae|all` opts into sharing SAE tensors (push defaults to `raw` so provenance-stronger SAE flavors don't ship accidentally); `pack clear --variant raw|sae|all` scopes deletions (defaults to `all`).

## Python API (v2.0)

Every surface — Python, YAML, HTTP, TUI, `vector merge` — speaks the same steering-expression grammar out of `saklas.core.steering_expr`:

```
expr     := term (("+" | "-") term)*
term     := [coeff "*"?] selector ["@" trigger]
selector := atom (("~" | "|") atom)?
atom     := [ns "/"] NAME ["." NAME] [":" variant]
trigger  := before | after | both | thinking | response | prompt | generated
variant  := raw | sae | sae-<release>
```

`+`/`-` add terms, `*` attaches a coefficient, `~` projects onto a direction (keeps the shared component), `|` projects orthogonal (removes the shared component). `@trigger` tags a per-term trigger override. `:variant` routes to SAE tensors.

```python
from saklas import SaklasSession, SamplingConfig, Steering, Trigger, Profile
from saklas import SaklasError, EventBus, SteeringApplied, ProbeScored

with SaklasSession.from_pretrained("google/gemma-3-4b-it", device="auto") as session:
    name, profile = session.extract("angry.calm")     # Profile, not dict
    session.steer(name, profile)

    result = session.generate(
        "What makes a good day?",
        steering=f"0.3 {name}",
        sampling=SamplingConfig(temperature=0.7, max_tokens=256, seed=42),
        thinking=None,                                 # None = auto via supports_thinking
    )

    with session.steering("0.5 wolf"):                # resolves to deer.wolf @ -0.5
        result = session.generate("Describe a forest.")

    # Per-term triggers and projections are first-class in the grammar.
    result = session.generate(
        "Solve this, then answer politely.",
        steering="0.3 honest + 0.4 warm@after - 0.2 sycophantic|honest",
    )

    # Pre-built Steering objects are still accepted for typed construction.
    result = session.generate(
        "Solve this, then answer politely.",
        steering=Steering(alphas={
            "honest": 0.3,                             # BOTH (default)
            "warm":   (0.4, Trigger.AFTER_THINKING),   # per-entry
        }),
    )

    for tok in session.generate_stream("Tell me a story.", steering=f"0.2 {name}"):
        print(f"[think] {tok.text}" if tok.thinking else tok.text, end="", flush=True)

    session.events.subscribe(lambda e: print("event:", type(e).__name__))
    print(session.last_result.applied_steering)        # canonical expression receipt
```

Hard-break notes vs v1.4:
- `generate` / `generate_stream` / `session.steering()` accept `str | Steering | None`. Dict inputs are no longer accepted — pass an expression string or build a `Steering` directly.
- `extract()` returns `(name, Profile)`, not `(name, dict)`. `Profile`'s dict-compat surface (`__getitem__`/`__iter__`/`len`/`items`) means most v1 loop code still works.
- `SaklasSession.__init__` takes a pre-loaded `PreTrainedModel`; construct with `SaklasSession.from_pretrained(model_id, ...)`.
- `session.config = replace(session.config, temperature=0.8)` works for session-level defaults (`GenerationConfig` is trimmed to session-level fields only); **per-call overrides should use `SamplingConfig`**, not rebind `session.config`.
- `session.lock` is an `asyncio.Lock` owned by the server layer. Library-only callers never touch it.
- Every saklas-raised exception is a `SaklasError` subclass while preserving its stdlib MRO, so `except SaklasError` catches the whole family and `except ValueError` / `except RuntimeError` at existing sites still works.
- `GenerationResult.applied_steering` carries the canonical expression string that produced the result (or `None` for unsteered generations) — round-trips through `saklas.core.steering_expr.parse_expr`.

## Cache layout

All state under `~/.saklas/` (override via `SAKLAS_HOME`):

```
~/.saklas/
  neutral_statements.json              # user-editable (copy-on-miss from package)
  vectors/
    default/<concept>/                 # bundled (source=bundled)
      pack.json                        # name, description, tags, files{sha256}
      scenarios.json                   # 9 broad domains used as pair-gen seed
      statements.json                  # contrastive pairs
      <safe_model_id>.safetensors      # baked tensor (native)
      <safe_model_id>.json             # slim sidecar: method/saklas_version/statements_sha256
      <safe_model_id>.gguf             # optional llama.cpp parallel
    local/<concept>/                   # user-authored (source=local; `refresh` skips)
    <hf_owner>/<concept>/              # HF-pulled (source=hf://owner/name[@rev])
  models/<safe_model_id>/
    layer_means.safetensors            # probe-centering baseline
    layer_means.json                   # hash of neutral_statements.json
```

- `pack.json.files` sha256 map verified on every `ConceptFolder.load`.
- Shares are baked into tensor magnitudes at extraction, so sidecars carry no `scores` field.
- **Dual-format tensors**: `ConceptFolder` globs `*.safetensors` + `*.gguf`. Safetensors wins on same-stem conflict (native, carries sidecar). GGUF metadata lives in the header — `sidecar(sid)` raises `KeyError` for GGUF-only entries. `vectors.load_profile` dispatches on extension.
- **Upgrade gotcha**: `materialize_bundled` is copy-on-miss for `neutral_statements.json` and missing concept folders, **but auto-upgrades bundled concepts in place when the on-disk `pack.json.format_version` is explicitly stale** (v1.x → v2.0). Overwrites shipped `pack.json` + `statements.json`; leaves per-model tensors. Releases changing neutrals still need `saklas pack refresh neutrals`; layer means auto-recompute via hash check.

## Performance invariants

These gate `test_session.py::test_throughput` (steered ≥ 85% of vanilla tok/s):

- **Hot-path hooks**: no Python allocation, no `.item()`, no CPU sync, in-place only. Norm preservation adds two `torch.linalg.vector_norm(dtype=float32)` calls + one in-place `mul_` per steered layer per step.
- **`torch.inference_mode()`** wraps the entire generation loop.
- **In-place ops**: `logits.div_()`, `logits.clamp_()`, `probs.div_()`, `hidden.add_()` / `mul_()`.
- **Top-p via `torch.topk`**, not full-vocab sort. `k = min(config.top_k or 1024, vocab)`; `top_k` is a hard cap applied before top-p (matches llama.cpp/Ollama).
- **Norms use fp32** — fp16 sum-of-squares overflows at hidden_dim ≥ 2048. Applies to extraction-time direction norms **and** the norm-preserving hook's pre/post captures.
- **Shares baked at extraction, flat scalar at apply.** Stored direction = `unit_direction * ref_norm * (score / sum(scores))` over retained layers (first/last 2 dropped per `drop_edges=(2,2)` default — see `core/vectors.py`); hook math is `user_alpha * _STEER_GAIN * sum(baked)`. Per-unit-α injection is invariant across layer count and absolute score magnitude. `_STEER_GAIN = 2.0` calibrated on gemma-4-31b-it with the bundled 21-probe pack post-edge-drop; coherent band ≈ α 0.3–0.85, cliff ≈ α 0.95+. Cross-model validation on Qwen3.5-9B and Llama-3.2-3B-Instruct confirms the cliff transfers within ±0.1α.
- **Norm preservation is unconditional.** Every steered layer's output rescaled back to pre-injection per-position magnitude. The v1.3.1 sweep confirmed norm preservation made injection **more** efficient per unit α — unconstrained addition also inflates magnitude which implicitly attenuates directional change.
- **Monitor capture is hook-driven, inline with generation.** `HiddenCapture` hooks union of probe layers for the run; `score_per_token` scores from captured tensors afterward — no second forward pass. Per-layer weight = `||baked||` (= share × ref_norm). Per-layer stacked cache — one matmul per layer scores all probes simultaneously; one `.cpu().tolist()` per measure regardless of probe count.
- **Steering hooks are transient** — composed before generation, removed after.
- **Contrastive diffs in float32** — fp16 subtraction between close vectors loses precision.
- **MPS discipline** — diffs kept on CPU (SVD runs there anyway), `torch.mps.empty_cache()` between extraction passes, `torch.device(target)` model loading to avoid unified-memory RSS spike.

## Tested architectures

`_TESTED_ARCHS` frozenset in `core/model.py` — emit one-time `UserWarning` on load when `model_type` isn't in this set. **Known working**: `qwen2`, `qwen3`, `qwen3_5` (+ `_text`/`_moe`), `gemma2`, `gemma3` (+ `_text`), `gemma4` (+ `_text`), `mistral3`, `ministral3`, `gpt_oss`, `llama`, `glm`. **Wired up but untested**: mistral/mixtral, phi/phi3/phimoe, cohere 1–2, deepseek v2/v3, starcoder2, olmo 1–3 + moe, granite/granitemoe, nemotron, stablelm, gpt2/neo/j/bigcode/neox, bloom, falcon/falcon_h1, mpt, dbrx, opt, recurrent_gemma. Adding a new architecture = one entry in `_LAYER_ACCESSORS`. **Be careful claiming breadth in user-facing docs** — 54 accessor entries is not the same as 54 architectures working.

## Bundled concepts

21 curated concepts at **n=45 pairs each** (9 scenarios × 5 pairs/scenario), stored in `saklas/data/vectors/<canonical>/`. 19 bipolar + 2 monopolar (`agentic`, `manipulative`). Authoritative list is the `BIPOLAR`/`MONOPOLAR` manifest in `scripts/regenerate_bundled_statements.py`.

Categories:
- `affect`: angry.calm, happy.sad
- `epistemic`: confident.uncertain, honest.deceptive, hallucinating.grounded
- `alignment`: refusal.compliant, sycophantic.blunt, agentic, manipulative
- `register`: formal.casual, direct.indirect, verbose.concise, creative.conventional, humorous.serious, warm.clinical, technical.accessible
- `social_stance`: authoritative.submissive, high_context.low_context
- `cultural`: masculine.feminine, religious.secular, traditional.progressive

**Known axis entanglement**: `creative.conventional ↔ hallucinating.grounded` extract near-identical directions (weighted cosine +0.78 on gemma-4-e4b-it). Steering "creative" also steers toward "hallucinating." Model-level entanglement, not probe design — document for users.

**Scenario framework** (`scripts/regenerate_bundled_statements.py`, generated on gemma-4-31b-it):
- `generate_scenarios`: 9 broad situational *domains* per concept (2–6 words), shared across both poles. Load-bearing **anti-allegory clause** in the scenario prompt keeps non-human axes on their literal footing.
- `generate_pairs`: loops domains × 5 first-person contrastive pairs each. **POV/behavior framing** ("write like you ARE X, facing that moment") generalizes across human and non-human concepts; **self-label ban** prevents `deer → sheepish-deerlike-person` leak.
- Unlocks non-human axes (`deer.wolf`, `bacterium.virus`, `brick.feather`, `the_sun.the_moon`) as literal rather than human-allegory.

**Extraction pipeline on cache miss** (`SaklasSession.extract`): if `statements.json` exists (curated or local) → extract from it and skip generation. Else: generate scenarios → save → generate pairs → save → contrastive PCA → save tensor. Tensor cache always short-circuits upstream. Flags: `scenarios=[...]` (explicit, bypass scenario gen), `reuse_scenarios=True` (when regenerating pairs), `force_statements=True` (ignore statements cache). `reuse_scenarios` defaults False — scenarios are cheap.

90 neutral statements in `saklas/data/neutral_statements.json` follow the same affect-neutral, topically-diverse discipline.

Pair-count note: <n=32 inflates mean PCA scores ~38% (small-sample bias). Share-baking is mean-magnitude-invariant so this no longer breaks cross-concept α consistency — but n=45 remains the measured sweet spot (no layer-direction flips, ~25% faster than n=60).

## Canonical naming + pole aliasing

`session.canonical_concept_name(concept, baseline=None)` slugs poles via `[^a-z0-9]+ → _` (`high-context` → `high_context`) and joins bipolar via `BIPOLAR_SEP = "."`. `session.extract()` returns `(canonical_name, profile)`. `/steer happy . sad` and `/steer happy.sad` resolve to the same vector.

`NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$`; `.` and `_` allowed, `@` forbidden (HF revision separator). `.` chosen over `~` because HF Hub repo names reject `~`.

`saklas.cli.selectors.resolve_pole(raw, namespace=None)` scans `_all_concepts()` across every namespace, returns `(canonical, sign, match)`. Typing a bare pole (`wolf`) on top of an installed `bob/deer.wolf` returns `("deer.wolf", -1, …)`; caller multiplies user α by `sign`. Monopolar exact matches and composite literals pass through unchanged. Ambiguity → `AmbiguousSelectorError` (disambiguate with `ns/name`). Explicit bipolar form (`/steer pos . neg`) skips resolution so user-declared poles always win.

## Package layout

`saklas/{core,io,cli,server,tui}/`. `saklas/__init__.py` pins public re-exports: `SaklasSession`, `SaklasError`, `Profile`, `ProfileError`, `SamplingConfig`, `Steering`, `EventBus` + event dataclasses, `DataSource`, `GenerationResult`, `TokenEvent`, `ProbeReadings`, `ResultCollector`. Consumers doing `from saklas import X` keep working; anything reaching into private submodule paths is expected to break (hard-break refactor).

## Testing

**GPU-required** (CUDA or MPS): `test_smoke.py`, `test_session.py`. Downloads `google/gemma-3-4b-it` (~8GB) on first run. `device="auto"` picks cuda > mps > cpu. MPS runs ~3–5× slower, so `test_extraction_fast_enough` uses a backend-specific budget (10s CUDA / 60s MPS). `test_session` covers construction (21 probes auto-loaded), steering (`extract` returns `Profile`), cloning, generation + readings, thinking mode, `ResultCollector` JSONL/CSV. `test_smoke` owns `test_throughput_regression` (steered ≥ 85% of vanilla tok/s).

**CPU-only** (424 tests): `test_paths`, `test_packs`, `test_profile`, `test_sampling`, `test_steering`, `test_steering_context`, `test_events`, `test_format_version`, `test_selectors`, `test_canonical_name`, `test_cache_ops`, `test_hf`, `test_merge`, `test_gguf_io`, `test_config_file`, `test_cli_flags`, `test_probes_bootstrap`, `test_results`, `test_datasource`, `test_cloning` (CPU-safe paths only), `test_server`, `test_saklas_api`, `test_tui_commands`. Cover core v2 dataclasses, steering context semantics, pack format integrity + staleness, selector grammar, HF wrappers (mocked), GGUF roundtrip, config loading + `resolve_poles`, monitor scoring, five-verb CLI dispatch, OpenAI/Ollama/native servers, TUI slash command dispatch.
