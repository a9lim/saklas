# AGENTS.md

## What this is

`saklas` is a Python library + Textual TUI + dual-protocol HTTP server (OpenAI `/v1/*` **and** Ollama `/api/*` on the same port) for activation steering and trait monitoring on HuggingFace causal LMs. Representation Engineering via difference-of-means (default since v2.1; contrastive-PCA via `--method pca` for legacy parity), with angular (rotation-based) injection by default and additive + norm-preserving available via `--steer-mode additive`. Per-call alpha control, no model mutation. Three frontends over one engine: `SaklasSession` (programmatic), `saklas serve` (HTTP), `saklas tui` (TUI).

Version lives in `saklas/__init__.py` as `__version__`. `pyproject.toml` reads it dynamically via `version = {attr = "saklas.__version__"}`, so there's only one place to bump.

Releases: merge bump to `main` → `.github/workflows/release.yml` tags `v$VERSION`, builds, publishes via trusted publishing, cuts a GitHub release. Push without bumping → no-op.

## Subtree docs

Deep internals live in subtree `AGENTS.md` files — Claude Code auto-loads these when working in the corresponding directory. Only consult them when editing that layer.

- `saklas/core/AGENTS.md` — model loading, vectors, hooks, monitor, session, generation loop
- `saklas/server/AGENTS.md` — OpenAI / Ollama / native `/saklas/v1/*` routes
- `saklas/tui/AGENTS.md` — slash commands, highlighting, panels, status footer
- `saklas/cli/AGENTS.md` — verb dispatch, config loading, warmup
- `saklas/io/AGENTS.md` — packs, HF distribution, GGUF, cloning, datasource
- `saklas/web/AGENTS.md` — webui mount, wire protocol, source layout, reactivity gotchas

## Commands

```bash
pip install -e ".[dev]"                         # editable + pytest
pip install -e ".[serve]"                       # fastapi + uvicorn
pip install -e ".[gguf]"                        # llama.cpp GGUF I/O
pip install -e ".[cuda]"                        # bitsandbytes + flash-attn (Linux/CUDA)
pip install -e ".[sae]"                         # SAELens-backed SAE extraction (sae-lens)
saklas tui <model_id> [--steer-mode {angular,additive}] [--theta-max RAD]
saklas serve <model_id> [--steer-mode {angular,additive}] [--theta-max RAD]
saklas pack install <target> [-s|-a NS/N|-f]    # HF coord or folder; -s = statements only
saklas pack refresh <selector> [-m MODEL]       # re-pull; `refresh neutrals` is reserved
saklas pack clear <selector> [-m MODEL] [--variant raw|sae|all]  # delete per-model tensors (keep statements)
saklas pack rm <selector> [-y]                  # remove folder (bundled respawns)
saklas pack ls [selector] [-j|-v]               # LOCAL installed packs only
saklas pack search <query> [-j|-v]              # search HF hub for saklas-pack repos
saklas pack push <selector> [-a OWNER/NAME] [-p] [-m MODEL] [-s|-n] [-t] [-d] [-f] [--variant raw|sae|all]
saklas pack export gguf <selector> [-m MODEL] [-o PATH] [--model-hint HINT]
saklas vector extract <concept>|<pos> <neg> [-m MODEL] [-f] [--method dim|pca] [--sae RELEASE [--sae-revision REV]]
saklas vector merge <name> <expression> [-m]    # shared steering grammar: "0.3 ns/a + 0.5 ns/b~ns/c"
saklas vector clone <corpus> -N NAME [-m MODEL] [-n N_PAIRS] [--seed S] [-f]
saklas vector compare <concepts...> -m MODEL [-v] [-j]       # cosine similarity between vectors
saklas vector why <concept> -m MODEL [-j]                    # per-layer ||baked|| as 16-bucket histogram
saklas config show [-c PATH ...] [--no-default] [-m MODEL]
saklas config validate <file>
pytest tests/                                   # all; GPU tests gated on CUDA/MPS
```

Root parser has exactly five verbs — `tui`, `serve`, `pack`, `vector`, `config`. No `argv[0]` peeking, no top-level verb aliases, no bare-TUI fallback — `saklas google/gemma-2-2b-it` is an argparse error. Bare `saklas`/`saklas pack`/`saklas vector`/`saklas config` print help + exit 0.

Every subcommand that takes `-c/--config` auto-loads `~/.saklas/config.yaml` first, then composes any explicit `-c` files on top (later overrides earlier). `ConfigFile.effective(extras, include_default=...)` is the single entry point. `ConfigFile.vectors` is a steering expression string — parsed lazily through `saklas.core.steering_expr.parse_expr`, which resolves bare poles (`wolf → deer.wolf @ -0.5`) via `cli.selectors.resolve_pole` and produces the canonical `Steering` IR every surface speaks.

Selector grammar (shared): `<name>`, `<ns>/<name>`, `tag:<t>`, `namespace:<ns>`, `default`, `all`, optionally suffixed with `:<variant>` where `<variant>` is `raw` (canonical DiM, default since v2.1), `pca` (legacy PCA tensor at `<safe>_pca.safetensors`), `sae` (unique SAE variant for the concept), or `sae-<release>` (specific SAELens release). Bare names resolve cross-namespace and raise `AmbiguousSelectorError` on collision; bare `:sae` raises `AmbiguousVariantError` when more than one SAE release is extracted for a concept. `pack refresh` silently skips `source=local`; `pack refresh neutrals` rewrites `~/.saklas/neutral_statements.json` from the bundled copy. `pack rm` refuses broad selectors without `-y`; bundled concepts respawn on next session init. `pack ls` is local-only; HF hub search is `pack search`.

## Extraction methods (v2.1)

Two extractors share `_capture_diffs_for_pairs` (forward-pass capture) and `_share_bake_and_warn` (edge-drop + share-baking + diagnostics warning) in `saklas/core/vectors.py`; only the per-layer direction computation differs:

- `extract_difference_of_means` (default, `--method dim`): per-layer direction = `mean(diffs_per_layer[L])`. Score = `||direction||_M / ref_norm` when a `whitener` is wired (Mahalanobis bake, v2.1 default), `||direction||_2 / ref_norm` otherwise (Euclidean fallback). Sidecar `method` label `"difference_of_means"` (raw) / `"dim_sae"` (SAE feature space) plus a `bake` field `"mahalanobis"` / `"euclidean"`. Im & Li 2025 / AxBench 2025 motivation for DiM; Mahalanobis-flavored share allocation generalizes the metric off Euclidean magnitude onto the per-model activation distribution.
- `extract_contrastive` (legacy, `--method pca`): per-layer direction = first principal component of the diffs (batched SVD). Score = explained-variance ratio (metric-invariant — Mahalanobis branch ignores the whitener). Sidecar label `"contrastive_pca"` / `"pca_center_sae"` with `bake: "euclidean"` — bit-identical to v2.0.x output.

Tensor filenames carry the method as a suffix: `<safe>.safetensors` (raw DiM, canonical), `<safe>_pca.safetensors` (raw PCA legacy), `<safe>_sae-<release>.safetensors` (DiM in SAE feature space), `<safe>_sae-<release>_pca.safetensors` (PCA in SAE feature space, legacy). `tensor_filename(model_id, *, release=None, transferred_from=None, method="dim")` and `parse_tensor_filename` round-trip; `enumerate_variants` keys are `raw` / `pca` / `sae-<release>` / `sae-<release>-pca` / `from-<safe_src>`. Existing v1.x PCA tensors at the canonical `<safe>.safetensors` path keep loading — sidecar method tells the runtime what's stored. Bake variants share the canonical filename; the sidecar `bake` field disambiguates (no `_mbake` filename suffix because v2.1 default-flips DiM bake, so on-disk tensors after first re-extraction are uniformly Mahalanobis-baked).

## Bake metric (v2.1)

Under DiM, score = `||mean_diff||_M / ref_norm` makes per-layer `ref_norm` cancel from the hook share: angular reads `share_L_hook = ||baked_L|| / Σ ||baked||`, which expands to `share_L = ||mean_diff_L||_M / Σ_L' ||mean_diff_L'||_M` — pure Mahalanobis magnitude weighting across retained layers, same algebraic shape as the Euclidean form `||m_L||_2 / Σ ||m_L'||_2`. Direction parametrization unchanged (`unit_2(mean_diff) × ref_norm`), so additive magnitudes stay calibrated for `_STEER_GAIN`. Layers where the contrastive signal sits in low-variance directions get more share (concept-meaningful per the metric); layers where the diff is amplified by high-variance noise get less.

`LayerWhitener` (`saklas/core/mahalanobis.py`) holds per-layer centered neutral activations `X_L ∈ ℝ^(N=90, D)` and the small Woodbury inverse `K_L = (NλI + X X^T)^{-1} ∈ ℝ^(N, N)`; `Σ_reg^{-1} v = (1/λ)(v − X^T K X v)` is O(ND) per matvec without materializing D×D. Ridge `λ_L = (||X_L||_F² / (N · D)) · ridge_scale` (mean diagonal of the un-regularized sample covariance). No new persistent cache: built lazily from the existing `~/.saklas/models/<id>/{layer_means,neutral_activations}.safetensors` pair via `LayerWhitener.from_neutral_activations` (in-memory) or `LayerWhitener.from_cache(model_id)` (disk-only, used by `vector compare --metric mahalanobis`). `SaklasSession` exposes a lazy `session.whitener` property that builds on first access; `bootstrap_probes` builds it eagerly before extraction so v2.1 defaults are end-to-end Mahalanobis on the bootstrap path.

`project_profile(base, onto, operator, *, whitener=None)` and `Profile.cosine_similarity(other, *, whitener=None)` accept the same primitive. With `whitener=None` they fall back to Euclidean Gram-Schmidt and standard cosine; with a whitener, `|`/`~` use the closed-form LEACE projector for a single direction (Belrose et al., 2023, arXiv 2306.03819) and cosine uses `<u,v>_M = u^T Σ^{-1} v`. Layers absent from the whitener fall back to Euclidean per layer (whitener may legitimately cover a subset, e.g. SAE-only releases).

Runtime `|`/`~` projection at steering time (`_materialize_projections` in `session.py`) defaults to Mahalanobis since v2.1 — `project_profile` is called with `session.whitener`, switching `~` and `|` to the closed-form LEACE projector. The session-level default is set by `SaklasSession(projection_metric="mahalanobis")` (the new kwarg, default `"mahalanobis"`); per-call overrides ride on `Steering.projection_metric: Literal["mahalanobis", "euclidean"] | None = None` (None = inherit). YAML `projection_metric:` and CLI `--projection-metric {mahalanobis,euclidean}` on `tui`/`serve` mirror the kwarg. `--legacy` flips it to `"euclidean"` (plain Gram-Schmidt, the v2.0/v2.1 behavior). When the whitener is unavailable for a layer (no neutral-activation cache yet), `project_profile` falls back to Euclidean per-layer transparently. Steering recipes like `0.3 honest|sycophantic` produce more aggressive concept removal under the new default — LEACE removes all linearly-decodable `sycophantic` information from `honest`, vs. Euclidean's literal-direction subtraction. Use `--legacy` (or per-call `projection_metric="euclidean"`) to reproduce the older shape.

Empirical (gemma-4-e4b-it / `default/angry.calm`, single-seed sweep): Mahalanobis bake redistributes top-share layers from the v2.0 Euclidean `[9-13]` mid-stack to `[14-16, 25-26]`. Cross-architecture spot-check on Qwen3.6-27B concentrates top share at very late layers `[57-61]` with peak/mean=3.57 (vs e4b's 1.64); coherent steering survives at α=+0.6 with surface-form intact, suggesting the late-layer concentration is a real architecture difference rather than metric pathology, but the distinguishing experiment (re-running Qwen with Euclidean bake) hasn't been done yet.

## Discriminative Layer Selection (v2.1)

The `drop_edges=(2,2)` heuristic is gone. Layer selection at extraction time is now data-driven via centered DLS (Selective Steering, Dang & Ngo 2026, Eq. 9): a layer is kept iff `(μ_pos − μ_neutral) · d̂` and `(μ_neg − μ_neutral) · d̂` have opposite signs. Layers where both class means project to the same side of the neutral baseline along `d̂` encode concept *intensity* rather than concept *polarity* — they inflate share without aiding discrimination.

`compute_dls_mask(mu_pos, mu_neg, directions, layer_means)` lives in `saklas/core/vectors.py`; both extractors call it and pass the kept set to `_share_bake_and_warn` (which now takes `keep_set: set[int] | None` instead of `edge_idx`). `layer_means=None` disables centering; `dls=False` skips the mask entirely. When every layer fails the discriminative check the helper warns and falls back to keep-all rather than emptying the profile (degenerate concept on this model).

Empirical incidence on the bundled `default/angry.calm`: gemma-4-e4b-it 11/42 dropped (`[5, 8, 31–34, 37–41]`); Qwen3.6-27B 13/64 dropped (mostly contiguous `[40, 49–60]`, with the documented late-layer concentration at L61–L63 surviving — DLS sharpens the existing peak rather than removing it). Cross-architecture sanity checks pending on smaller / non-emotional concepts.

CLI: `--no-dls` opts out at extraction time; `--legacy` bundles `dls=False` alongside the other v2.0 flags.

## Backcompat (`--legacy`)

Single-flag preset on `tui`, `serve`, `vector extract`, and `vector compare` that bundles the pre-v2.1 stack: PCA extraction (`--method pca`), additive injection (`--steer-mode additive`), Euclidean cosine (`--metric euclidean`), Euclidean `~`/`|` projection (`--projection-metric euclidean`), and DLS off (`--no-dls`-equivalent — the v2.0 `edge_drop` heuristic is gone, so `--legacy` keeps every layer rather than re-implementing the removed shape). Mutually exclusive with the per-flag controls on the same verb (passing both `--legacy` and `--method dim` errors out at parse time before model load). For `tui`/`serve`, `--legacy` flips `extraction_method="pca"`, `projection_metric="euclidean"`, and `dls=False` on the session so first-run probe bootstrap *and* runtime projection match the v2.0 stack; for `vector extract`, it flips `--method` only; for `vector compare`, it flips `--metric` only (which defaults to `mahalanobis`).

`SaklasSession.from_pretrained` and `__init__` gain `extraction_method: Literal["dim", "pca"] = "dim"`; `bootstrap_probes` accepts `method=` and `whitener=`. Sidecars carry the bake choice in the `bake` field so cross-version diagnostics know which scoring drove the magnitudes on disk.

## SAE pipeline (optional)

`saklas vector extract <concept> --sae <release>` runs extraction in SAE feature space (DiM by default; `--method pca` for the legacy SAE-PCA path). Encodes the per-pair pos/neg stacks through `SaeBackend.encode_layer`, runs the chosen extractor in feature space, decodes the resulting direction back through `SaeBackend.decode_layer`, then hands off to share-baking. Uses SAELens as the loader, so any published release it covers (GemmaScope, Eleuther, Joseph Bloom's, Apollo/Goodfire) works on day one. Install with `pip install -e ".[sae]"`.

Output tensors coexist with raw tensors in the same concept folder. Select at steer time with the `:sae[-<release>]` suffix in the shared steering expression grammar — `session.steering("0.3 honest:sae")` from Python, `vectors: "0.3 honest:sae"` in the config file, `/steer 0.3 honest:sae` in the TUI. A bare `:sae` picks the unique SAE variant; explicit `:sae-<release>` picks one when multiple coexist.

SAE profiles are subset-layer (only layers the release covers). Share-baking redistributes over the covered subset — hook math and monitor scoring are unchanged. Release selection at extract time requires an explicit SAELens release name (SAELens ships many per base model — `gemma-scope-2b-pt-res`, `-mlp`, `-att`, etc. — and there's no sensible implicit default); if multiple SAEs exist per layer within a release, saklas picks the narrowest-width and emits a warning.

Sidecar JSON records `sae_release`, `sae_revision`, `sae_ids_by_layer` alongside the usual `method` / `statements_sha256` fields. `pack push --variant sae|all` opts into sharing SAE tensors (push defaults to `raw` so provenance-stronger SAE flavors don't ship accidentally); `pack clear --variant raw|sae|all` scopes deletions (defaults to `all`).

## Injection modes (v2.1)

Set per session via `SaklasSession.from_pretrained(injection_mode="angular"|"additive", theta_max=π/2)` or per call via `Steering(injection_mode=..., theta_max=...)`. CLI flags `--steer-mode` and `--theta-max` on `tui`/`serve`; YAML keys `injection_mode:` and `theta_max:`. `Steering.injection_mode = None` (default) inherits the session-level setting; nested `session.steering(...)` scopes resolve inner-wins via a parallel LIFO override stack.

- **Angular (default)**: per layer L, `θ_L = share_L × ||composed_unit_sum||_L × θ_max` where `composed_unit_sum_L = Σ_i α_i × (baked_i_L / ||baked_i_L||)` and `share_L = ||baked_L|| / Σ_L' ||baked_L'||`. Cumulative budget across the residual stream sums to `|α| × θ_max`, so `α=1` ↔ full π/2 rotation toward the concept, `α=0.5` ↔ 45° cumulative. Hot path is a Givens rotation in the (h_unit, d_perp_unit) plane: norm-preserving exactly, no rescale. Cached `_d_hat` / `_theta` / `_cos_t` / `_sin_t` at `recompose` time so hook_fn has zero `.item()` calls.
- **Additive (legacy)**: `effective_alpha = user_alpha × _STEER_GAIN`; per-layer `composed_L = effective_alpha × baked_L` (share built into magnitude). Hook math is `vector_norm → add_(composed) → vector_norm.clamp_(1e-6) → mul_(ratio)`. Bit-identical to v1.x when explicitly selected. `_STEER_GAIN = 2.0` calibrated on gemma-4-31b-it; coherent band ≈ α 0.3–0.85, cliff ≈ α 0.95+. Cross-model validation on Qwen3.5-9B and Llama-3.2-3B-Instruct confirms the cliff transfers within ±0.1α.

Empirical angular band (gemma-4-e4b-it / `default/angry.calm`): under v2.0 Euclidean bake (`seed=42`), coherent ≈ α [-1.0, +0.7], cliff at α=+1.0. Under v2.1 Mahalanobis bake (`seed=1234`), wobble appears earlier on the angry side (token leakage at α=+0.6, mid-cliff at +0.8), cliff still at α=+1.0; calm-side band unchanged. Single-seed evidence both ways — multi-seed comparison still owed before claiming "better steering." Asymmetry between calm and angry sides is consistent with the documented `angry.calm ↔ authoritative.submissive` entanglement on this checkpoint (cosine +0.40 under both metrics on Mahalanobis-baked tensors, so the entanglement is real semantic overlap rather than a metric artifact).

## Python API (v2.1)

Every surface — Python, YAML, HTTP, TUI, `vector merge` — speaks the same steering-expression grammar out of `saklas.core.steering_expr`:

```
expr        := term (("+" | "-") term)*
term        := [coeff "*"?] ["!"] selector ["@" trigger]    # coeff optional; omit → 0.5 (additive) or 1.0 (ablation)
selector    := atom (("~" | "|") atom)?
atom        := [ns "/"] NAME ["." NAME] [":" variant]  # NAME uses _ for multi-word
trigger     := preset | gate
preset      := before | after | both | thinking | response | prompt | generated
gate        := "when" ":" probe_atom op NUM
probe_atom  := NAME ["." NAME]
op          := > | >= | < | <=
variant     := raw | sae | sae-<release>
```

`+`/`-` add terms, `*` attaches a coefficient, `~` projects onto a direction (keeps the shared component), `|` projects orthogonal (removes the shared component), `!` mean-ablates the concept from the residual stream at hook time (`h' = h - α(h·d̂ - μ·d̂)d̂`; bare `!x` defaults to α=1.0, fully replace). `@<preset>` tags a per-term trigger override. `@when:<probe><op><threshold>` (v2.1) is a probe-gated trigger — fires only on decode steps where the named monitor probe's last reading satisfies the comparison; implicit `prompt=False` since probe scores aren't available during prefill. `:variant` routes to SAE tensors. `!` cannot compose with `~` or `|`. Compound triggers (`@<preset>&when:...`) are programmatic-only in v2.1 — build the `Trigger` directly via `dataclasses.replace` for "after-thinking AND when:angry>0.4" semantics.

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

    # Probe-gated trigger (v2.1): fire calm-steering only when the angry
    # probe reads above 0.4 during decoding.  ``score_callback`` runs
    # after every forward to refresh ``ctx.probe_scores``; gates skip
    # firing during prefill (no monitor reading yet).
    result = session.generate(
        "What's the weather like?",
        steering="0.5 calm@when:angry.calm>0.4",
    )

    # Pre-built Steering objects are still accepted for typed construction.
    result = session.generate(
        "Solve this, then answer politely.",
        steering=Steering(alphas={
            "honest": 0.3,                             # BOTH (default)
            "warm":   (0.4, Trigger.AFTER_THINKING),   # per-entry
        }),
    )

    # Hidden-state round-trip — opt-in per call; off by default.
    result = session.generate(
        "Count to three.",
        sampling=SamplingConfig(max_tokens=16, return_hidden=True),
    )
    # {layer_idx: [T, D] tensor on CPU, T == len(result.tokens)}
    hiddens = result.hidden_states
    # Re-score the same tensors against registered probes.
    agg, per_token = session.score_hidden(hiddens, per_token=True)

    for tok in session.generate_stream("Tell me a story.", steering=f"0.2 {name}"):
        print(f"[think] {tok.text}" if tok.thinking else tok.text, end="", flush=True)

    session.events.subscribe(lambda e: print("event:", type(e).__name__))
    print(session.last_result.applied_steering)        # canonical expression receipt
```

Hard-break notes vs v1.4:
- `generate` / `generate_stream` / `session.steering()` accept `str | Steering | None` only. Dict inputs raise `TypeError`.
- `extract()` returns `(name, Profile)`, not `(name, dict)`. `Profile` exposes the full mapping interface (`__getitem__`/`__iter__`/`__len__`/`__contains__`/`items`/`keys`/`values`) plus the typed surface (`layers`, `metadata`, `weight_at`, `save`/`load`, `merged`, `projected_away`, `cosine_similarity`), so iteration-style v1 loop code still works.
- `session.steer(name, profile)` and `session.save_profile(profile, path)` accept `Profile` only — bare dicts are rejected. `session.vectors` returns `dict[str, Profile]`.
- `SaklasSession.__init__` takes a pre-loaded `PreTrainedModel`; construct with `SaklasSession.from_pretrained(model_id, ...)`. The `cache_dir=` parameter is gone — set `SAKLAS_HOME` to override paths.
- `session.config = replace(session.config, temperature=0.8)` works for session-level defaults (`GenerationConfig` is trimmed to session-level fields only); **per-call overrides should use `SamplingConfig`**, not rebind `session.config`.
- `session.lock` is an `asyncio.Lock` owned by the server layer. Library-only callers never touch it.
- `SteeringApplied.entries` is a required `dict[str, tuple[float, Trigger]]` — subscribers can't rely on it being `None`.
- Every saklas-raised exception is a `SaklasError` subclass while preserving its stdlib MRO, so `except SaklasError` catches the whole family and `except ValueError` / `except RuntimeError` at existing sites still works.
- `GenerationResult.applied_steering` carries the canonical expression string that produced the result (or `None` for unsteered generations) — round-trips through `saklas.core.steering_expr.parse_expr`.

Additive-only changes since v2.1:
- Default extractor flipped to DiM. Existing PCA tensors at the canonical `<safe>.safetensors` path keep loading (sidecar method tells the runtime); fresh extractions write DiM. Pass `--method pca` to opt into legacy contrastive PCA, which writes to the suffixed `<safe>_pca.safetensors` so both can coexist.
- Default injection flipped to angular (Givens rotation toward `d̂`). `Steering` gains optional `injection_mode` and `theta_max` fields (both `None` = inherit session default). `SaklasSession.from_pretrained` and `SaklasSession.__init__` gain `injection_mode="angular"`, `theta_max=π/2` defaults. `--steer-mode` and `--theta-max` on `tui`/`serve`; YAML keys `injection_mode:` and `theta_max:`.
- `_STEER_GAIN = 2.0` only multiplies under `injection_mode="additive"`. Under angular, user α maps directly to a rotation angle (`Σ_L θ_L = |α| × θ_max`).

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

- **Hot-path hooks**: no Python allocation, no `.item()`, no CPU sync, in-place only. Angular path runs ~1 vector_norm + 1 dot + a handful of muls/subs in fp32 + an in-place `copy_` per layer per step; additive path keeps the v1.x two `vector_norm` + `add_` + `mul_(ratio)` rescale.
- **`torch.inference_mode()`** wraps the entire generation loop.
- **In-place ops**: `logits.div_()`, `logits.clamp_()`, `probs.div_()`, `hidden.add_()` / `mul_()` / `copy_()`.
- **Top-p via `torch.topk`**, not full-vocab sort. `k = min(config.top_k or 1024, vocab)`; `top_k` is a hard cap applied before top-p (matches llama.cpp/Ollama).
- **Norms use fp32** — fp16 sum-of-squares overflows at hidden_dim ≥ 2048. Applies to extraction-time direction norms **and** the per-position norms inside both injection paths.
- **Shares baked at extraction, mode-specific at apply.** Stored direction = `unit_direction * ref_norm * (score / sum(scores))` over retained layers (first/last 2 dropped per `drop_edges=(2,2)` default — see `core/vectors.py`). Additive: per-layer `composed = user_alpha * _STEER_GAIN * baked_L`, share is automatic via magnitude. Angular: per-layer `effective_alpha_L = user_alpha * share_L` (share computed from `||baked_L||` at `apply_to_model` time), so cumulative `Σ_L θ_L` sums to `|α| * θ_max` regardless of layer count.
- **Norm preservation is mode-specific.** Additive: explicit per-position rescale via `vector_norm` pre/post. Angular: rotation is exactly norm-preserving by construction (Givens in the `(h_unit, d_perp_unit)` plane), no rescale. Near-aligned positions (`||d_perp|| < 1e-6`) get a `torch.where` no-op fallback to avoid the `cos_t * h_unit + sin_t * 0` shrinkage pathology.
- **Monitor capture is hook-driven, inline with generation.** `HiddenCapture` hooks union of probe layers for the run; `score_per_token` scores from captured tensors afterward — no second forward pass. Per-layer weight = `||baked||` (= share × ref_norm). Per-layer stacked cache — one matmul per layer scores all probes simultaneously; one `.cpu().tolist()` per measure regardless of probe count.
- **Steering hooks are transient** — composed before generation, removed after.
- **Contrastive diffs in float32** — fp16 subtraction between close vectors loses precision.
- **MPS discipline** — diffs kept on CPU (SVD runs there anyway), `torch.mps.empty_cache()` between extraction passes, `torch.device(target)` model loading to avoid unified-memory RSS spike.

## Tested architectures

`_TESTED_ARCHS` frozenset in `core/model.py` — emit one-time `UserWarning` on load when `model_type` isn't in this set. **Known working**: `qwen2`, `qwen3`, `qwen3_5` (+ `_text`/`_moe`), `gemma2`, `gemma3` (+ `_text`), `gemma4` (+ `_text`), `mistral3`, `ministral3`, `gpt_oss`, `llama`, `glm`, `talkie`. **Wired up but untested**: mistral/mixtral, phi/phi3/phimoe, cohere 1–2, deepseek v2/v3, starcoder2, olmo 1–3 + moe, granite/granitemoe, nemotron, stablelm, gpt2/neo/j/bigcode/neox, bloom, falcon/falcon_h1, mpt, dbrx, opt, recurrent_gemma. Adding a new architecture = one entry in `_LAYER_ACCESSORS`. Architectures whose custom modeling ignores `past_key_values` (e.g. the original `lewtun/talkie-1930-13b-it-hf` port) auto-fall back to no-KV-cache generation in `generate_steered` — correct but O(N²), one-time warning. The `a9lim/talkie-1930-13b-it-hf-cached` fork uses the standard KV-cache path. **Be careful claiming breadth in user-facing docs** — 54 accessor entries is not the same as 54 architectures working.

## Bundled concepts

24 curated concepts at **n=45 pairs each** (9 scenarios × 5 pairs/scenario), stored in `saklas/data/vectors/<canonical>/`. 22 bipolar + 2 monopolar (`agentic`, `manipulative`). Authoritative list is the `BIPOLAR`/`MONOPOLAR` manifest in `scripts/regenerate_bundled_statements.py`.

Categories:
- `affect`: angry.calm, happy.sad, fearful.unflinching
- `epistemic`: confident.uncertain, honest.deceptive, hallucinating.grounded, curious.disinterested
- `alignment`: refusal.compliant, sycophantic.blunt, agentic, manipulative
- `register`: formal.casual, direct.indirect, verbose.concise, creative.conventional, humorous.serious, warm.clinical, technical.accessible
- `social_stance`: authoritative.submissive, high_context.low_context
- `cultural`: masculine.feminine, religious.secular, traditional.progressive, individualist.collectivist

**Known axis entanglements** (cross-model robust on gemma-4-31b-it / gemma-4-e4b-it; weighted cosine via `vector compare`):
- `masculine.feminine ↔ traditional.progressive` (+0.53 / +0.59) — model treats Hofstede's MAS and traditionalism as the same direction
- `hallucinating.grounded ↔ humorous.serious` (+0.66 / +0.53) — humor reads as off-grounded creative weirdness
- `angry.calm ↔ authoritative.submissive` (+0.78 / +0.54) — anger encodes as dominance

Model-specific (not robust across the two checkpoints): on e4b, `individualist.collectivist ↔ high_context.low_context` lands at -0.47 (collectivist ↔ high-context per Hall) but is essentially zero on 31b. On 31b only, `authoritative.submissive` anchors a "stern/dominant" macro-cluster that pulls in sycophantic, refusal, warm, and hallucinating at |cos| ≥ 0.7 — does not transfer to e4b.

These are **model-level** entanglements, not probe-design failures — document for users. Effective rank of the 24 probes is ~11.1 dims on 31b, ~14.0 on e4b (out of theoretical 24).

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

**GPU-required** (CUDA or MPS): `test_smoke.py`, `test_session.py`. Downloads `google/gemma-3-4b-it` (~8GB) on first run. `device="auto"` picks cuda > mps > cpu. MPS runs ~3–5× slower, so `test_extraction_fast_enough` uses a backend-specific budget (10s CUDA / 60s MPS). `test_session` covers construction (24 probes auto-loaded), steering (`extract` returns `Profile`), cloning, generation + readings, thinking mode, `ResultCollector` JSONL/CSV. `test_smoke` owns `test_throughput_regression` (steered ≥ 85% of vanilla tok/s).

**CPU-only** (678 tests): `test_paths`, `test_packs`, `test_profile`, `test_sampling`, `test_steering`, `test_steering_context`, `test_events`, `test_format_version`, `test_selectors`, `test_canonical_name`, `test_cache_ops`, `test_hf`, `test_merge`, `test_gguf_io`, `test_config_file`, `test_cli_flags`, `test_probes_bootstrap`, `test_results`, `test_datasource`, `test_cloning` (CPU-safe paths only), `test_server`, `test_saklas_api`, `test_tui_commands`. Cover core v2 dataclasses, steering context semantics, pack format integrity + staleness, selector grammar, HF wrappers (mocked), GGUF roundtrip, config loading + `resolve_poles`, monitor scoring, five-verb CLI dispatch, OpenAI/Ollama/native servers, TUI slash command dispatch.
