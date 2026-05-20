# AGENTS.md

## What this is

`saklas` is a Python library + Textual TUI + dual-protocol HTTP server for activation steering and trait monitoring on HuggingFace causal LMs. It runs OpenAI `/v1/*` and Ollama `/api/*` on one port, plus a native `/saklas/v1/*` API and a Svelte dashboard at `/`. Steering vectors come from representation engineering: difference-of-means by default, contrastive PCA via `--method pca` for legacy parity. Injection is angular (rotation toward the concept direction) by default, additive available via `--steer-mode additive`. Per-call alpha, no model mutation. Three frontends over one engine: `SaklasSession` (programmatic), `saklas serve` (HTTP), `saklas tui` (TUI).

Version lives in `saklas/__init__.py` as `__version__` (currently 3.0.0). `pyproject.toml` reads it via `version = {attr = "saklas.__version__"}`, so there is one place to bump. Do not bump it as part of feature work — version bumps are user-owned.

Releases: merge a version bump to `main` → `.github/workflows/release.yml` tags `v$VERSION`, builds, publishes via trusted publishing, and cuts a GitHub release. A push without a bump is a no-op.

## Subtree docs

Deep internals live in subtree `AGENTS.md` files — Claude Code auto-loads each when you work in that directory. Consult them only when editing that layer.

- `saklas/core/AGENTS.md` — model loading, extraction, steering hooks, monitor, session, generation loop, loom tree
- `saklas/io/AGENTS.md` — packs, HF distribution, GGUF, cloning, alignment, paths/selectors
- `saklas/cli/AGENTS.md` — six-verb dispatch, config loading, flags
- `saklas/server/AGENTS.md` — OpenAI / Ollama / native routes
- `saklas/tui/AGENTS.md` — slash commands, panels, loom screen
- `saklas/web/AGENTS.md` — dashboard mount, wire protocol, Svelte source layout

## Commands

```bash
pip install -e ".[dev]"                         # editable + pytest
pip install -e ".[gguf]"                        # llama.cpp GGUF I/O
pip install -e ".[cuda]"                        # bitsandbytes + flash-attn (Linux/CUDA)
pip install -e ".[sae]"                         # SAELens-backed SAE extraction
saklas tui <model_id> [--steer-mode {angular,additive}] [--theta-max RAD]
saklas serve <model_id> [--no-web] [--steer/-S EXPR]
saklas pack install <target> [-s|-a NS/N|-f]    # HF coord or folder; -s = statements only
saklas pack refresh <selector> [-m MODEL]       # re-pull; `refresh neutrals` is reserved
saklas pack clear <selector> [-m MODEL] [--variant raw|sae|all]   # delete per-model tensors
saklas pack rm <selector> [-y]                  # remove folder (bundled respawns)
saklas pack ls [selector] [-j|-v]               # LOCAL installed packs only
saklas pack search <query> [-j|-v]              # search HF hub for saklas-pack repos
saklas pack push <selector> [-a OWNER/NAME] [-m MODEL] [--variant raw|sae|all] ...
saklas pack export gguf <selector> [-m MODEL] [-o PATH] [--model-hint HINT]
saklas vector extract <concept>|<pos> <neg> [-m MODEL] [--method dim|pca] [--sae RELEASE]
saklas vector merge <name> <expression> [-m]    # shared grammar: "0.3 ns/a + 0.5 ns/b~ns/c"
saklas vector clone <corpus> -N NAME [-m MODEL] [-n N_PAIRS] [--seed S]
saklas vector compare <concepts...> -m MODEL [--metric mahalanobis|euclidean]
saklas vector why <concept> -m MODEL [-j]       # per-layer ||baked|| as a 16-bucket histogram
saklas vector transfer <concept> --from SRC --to TGT [-f]   # cross-model Procrustes transfer
saklas experiment fan <model> "<prompt>" -g concept=0,0.5,1 # alpha grid as loom siblings
saklas experiment transcript run <path.yaml> [model]        # replay a saved transcript
saklas config show [-c PATH ...] [--no-default] [-m MODEL]
saklas config validate <file>
pytest tests/                                   # all; GPU tests gated on CUDA/MPS
```

The root parser has exactly six verbs: `tui`, `serve`, `pack`, `vector`, `experiment`, `config`. No `argv[0]` peeking, no verb aliases, no bare-TUI fallback — `saklas google/gemma-2-2b-it` is an argparse error. Bare `saklas` / `saklas pack` / `saklas vector` / `saklas experiment` / `saklas config` print help and exit 0.

Every subcommand that takes `-c/--config` auto-loads `~/.saklas/config.yaml` first, then composes explicit `-c` files on top (later overrides earlier). The `vectors:` YAML key is a single steering expression parsed by `saklas.core.steering_expr.parse_expr`. `cli/AGENTS.md` has the full per-verb flag set. `--legacy` on `tui`/`serve`/`vector extract`/`vector compare` is a single-flag preset for the pre-2.1 stack (PCA extraction, additive injection, Euclidean projection + cosine, DLS off); it conflicts with the matching per-flag controls.

## Selector grammar

Shared across surfaces: `<name>`, `<ns>/<name>`, `tag:<t>`, `namespace:<ns>`, `model:<m>`, `default`, `all`, optionally suffixed `:<variant>` where `<variant>` is `raw` (canonical DiM), `pca` (legacy PCA tensor), `sae`, or `sae-<release>`. Bare names resolve cross-namespace and raise `AmbiguousSelectorError` on collision; a bare `:sae` raises `AmbiguousVariantError` when a concept has multiple SAE releases. Bare poles alias to installed bipolar concepts: `wolf` → `deer.wolf @ -0.5` (caller multiplies user alpha by the sign), via `io.selectors.resolve_pole`.

Canonical naming: `session.canonical_concept_name` slugs poles via `[^a-z0-9]+ → _` and joins bipolar poles with `BIPOLAR_SEP = "."`, so `/steer happy . sad` and `/steer happy.sad` resolve to the same vector. `NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$`; `@` is forbidden (it is the HF revision separator), and `.` is used over `~` because HF repo names reject `~`.

## Steering expression grammar

Every input surface — Python, YAML, HTTP, TUI, `vector merge` — speaks the grammar in `saklas.core.steering_expr`. `parse_expr(text)` → `Steering`; `format_expr` round-trips it back.

```
expr        := term (("+" | "-") term)*
term        := [coeff "*"?] ["!"] selector ["@" trigger]
selector    := atom (("~" | "|") atom)?
atom        := [ns "/"] NAME ["." NAME] [":" variant]
trigger     := preset | gate
preset      := before | after | both | thinking | response | prompt | generated
gate        := "when" ":" probe_atom op NUM        # op ∈ > >= < <=
```

`+`/`-` add terms, `*` attaches a coefficient (omit → 0.5 additive / 1.0 ablation), `~` projects onto a direction (keep the shared component), `|` projects orthogonal (remove it), `!` mean-ablates the concept (`h' = h − α(h·d̂ − μ·d̂)d̂`; bare `!x` is α=1.0). `@<preset>` overrides a term's trigger; `@when:<probe><op><num>` is a probe gate that fires only on decode steps where the monitor reading satisfies the comparison (implicit `prompt=False`). `!` cannot compose with `~`/`|`. Compound triggers (`@after&when:…`) are programmatic-only — build the `Trigger` directly.

## Extraction

Two extractors in `core/vectors.py` share forward-pass capture, DLS layer selection, and share-baking; only the per-layer direction differs. `extract_difference_of_means` (default, `--method dim`): direction = mean of contrastive diffs. `extract_contrastive` (`--method pca`): first principal component via batched SVD. Both run in plain or SAE feature space. `core/extraction.py::ExtractionPipeline` orchestrates the cache-miss path (statements → scenarios → pairs → extract → save tensor); a tensor cache hit short-circuits everything upstream.

Share-baking folds the per-layer score into the tensor magnitude — `stored = unit_direction × ref_norm × score / Σ scores` over the DLS-retained layers — so sidecars carry no separate `scores` field and llama.cpp's uniform GGUF scalar reproduces saklas's per-layer weighting for free.

Discriminative layer selection (Selective Steering, Dang & Ngo 2026 Eq. 9): a layer is kept iff `(μ_pos − μ_neutral)·d̂` and `(μ_neg − μ_neutral)·d̂` have opposite signs — same-side layers encode concept *intensity*, not *polarity*. `--no-dls` opts out.

Bake metric: DiM scores via `||mean_diff||_M / ref_norm` when a `LayerWhitener` is wired (Mahalanobis, the session-driven default), `||·||_2 / ref_norm` otherwise. The `LayerWhitener` (`core/mahalanobis.py`) is built lazily from the cached neutral activations and also drives Mahalanobis `~`/`|` projection (closed-form LEACE) and `vector compare --metric mahalanobis`. PCA scores are explained-variance ratio, metric-invariant.

## Injection modes

Per session via `SaklasSession.from_pretrained(injection_mode=..., theta_max=...)` or per call via `Steering(injection_mode=..., theta_max=...)`; `None` inherits the session default. CLI `--steer-mode` / `--theta-max`, YAML `injection_mode:` / `theta_max:`.

- **Angular** (default): per-layer Givens rotation toward `d̂`, `θ_L = share_L × ||composed_unit_sum||_L × θ_max`. Cumulative budget `Σ_L θ_L = |α| × θ_max`, so `α=1` ↔ a full π/2 rotation. Norm-preserving by construction.
- **Additive**: `composed_L = α × _STEER_GAIN × baked_L` with an explicit per-position norm rescale. `_STEER_GAIN = 2.0` only multiplies under this mode.

`DEFAULT_THETA_MAX = π/2`.

## Python API

```python
from saklas import SaklasSession, SamplingConfig, Steering, Profile

with SaklasSession.from_pretrained("google/gemma-3-4b-it", device="auto") as session:
    name, profile = session.extract("angry.calm")     # returns (canonical_name, Profile)
    result = session.generate(
        "What makes a good day?",
        steering=f"0.3 {name}",
        sampling=SamplingConfig(temperature=0.7, max_tokens=256, seed=42),
    )
    with session.steering("0.5 wolf"):                 # resolves to deer.wolf @ -0.5
        result = session.generate("Describe a forest.")
    for tok in session.generate_stream("Tell me a story."):
        print(tok.text, end="", flush=True)
```

Key contracts:
- `generate` / `generate_stream` / `session.steering()` accept `str | Steering | None` only — dicts raise `TypeError`. A string is a steering expression in the shared grammar.
- `generate`, `generate_batch`, and `generate_sweep` always return `RunSet` — list-like, carrying `node_ids` and `grid`, with `.first` (the underlying `GenerationResult`) and common attributes delegating to it. `session.last_result` is the `GenerationResult`.
- `extract()` returns `(name, Profile)`. `Profile` is the typed `dict[int, Tensor]` wrapper (full mapping interface plus `layers`, `metadata`, `save`/`load`, `merged`, `projected_away`, `cosine_similarity`). `session.steer`, `session.save_profile`, and `session.vectors` all speak `Profile`, not bare dicts.
- `SaklasSession.__init__` takes a pre-loaded `PreTrainedModel`; use `from_pretrained` for HF loads. There is no `cache_dir=` — set `$SAKLAS_HOME` to relocate paths.
- Every saklas exception subclasses `SaklasError` while preserving its stdlib MRO, so `except SaklasError` catches the family and `except ValueError`/`RuntimeError` at existing sites still works.
- `GenerationResult.applied_steering` carries the canonical expression string that produced the result (round-trips through `parse_expr`).
- `saklas/__init__.py` pins the public surface (`SaklasSession`, `Profile`, `Steering`, `SamplingConfig`, `Trigger`, `LayerWhitener`, the `RunSet`/`TokenEvent`/`ResultCollector` result types, the `EventBus` + event dataclasses, the `LoomTree`/`Recipe`/`Transcript` suites, `DataSource`, and their error types). Importing through `from saklas import X` is stable; reaching into private submodule paths is not.

## Cache layout

All state under `~/.saklas/` (override via `$SAKLAS_HOME`):

```
~/.saklas/
  neutral_statements.json              # user-editable (copy-on-miss from package)
  vectors/
    default/<concept>/                 # bundled (source=bundled)
      pack.json                        # name, description, tags, files{sha256}
      scenarios.json                   # 9 broad domains used as pair-gen seed
      statements.json                  # contrastive pairs
      <safe_model_id>.safetensors      # baked tensor (+ .json slim sidecar)
      <safe_model_id>.gguf             # optional llama.cpp parallel
    local/<concept>/                   # user-authored (source=local; `refresh` skips)
    <hf_owner>/<concept>/              # HF-pulled (source=hf://owner/name[@rev])
  models/<safe_model_id>/
    layer_means.{safetensors,json}     # probe-centering baseline
    neutral_activations.{safetensors,json}   # 90 neutral prompts × layers, fp16
    alignments/<safe_src>.{safetensors,json} # optional cross-model Procrustes map
  conversations/<name>.json            # explicit loom-tree saves (no autosave)
```

`pack.json.files` is a sha256 map verified on every `ConceptFolder.load`. A concept folder can hold multiple baked tensors per model, distinguished by filename suffix: `<safe>.safetensors` (canonical DiM), `_pca`, `_sae-<release>`, `_sae-<release>_pca`, `_from-<safe_src>` (transfer). `tensor_filename` / `parse_tensor_filename` in `io/paths.py` round-trip them. Safetensors win over GGUF on a same-stem conflict. `materialize_bundled` is copy-on-miss but auto-upgrades bundled concepts in place when their `pack.json.format_version` is stale.

## Performance invariants

These gate `test_smoke.py::test_throughput_regression` (steered ≥ 85% of vanilla tok/s):

- **Hot-path hooks**: no Python allocation, no `.item()`, no CPU sync, in-place only. The whole generation loop is wrapped in `torch.inference_mode()`.
- **Norms use fp32** — fp16 sum-of-squares overflows at hidden_dim ≥ 2048. Applies to extraction-time direction norms and the per-position norms inside both injection paths. Contrastive diffs are differenced in fp32 too.
- **Shares baked at extraction**, applied mode-specifically: additive folds share into magnitude, angular reads `share_L = ||baked_L|| / Σ ||baked||` at apply time.
- **Norm preservation is mode-specific**: additive rescales explicitly, angular's Givens rotation is exact. Near-aligned positions get a `torch.where` no-op fallback.
- **Top-p via `torch.topk`**, not full-vocab sort; `top_k` (default 1024 cap) is a hard candidate-pool cap applied before top-p (llama.cpp/Ollama order).
- **Monitor capture is hook-driven**, inline with generation — one matmul per layer scores all probes, no second forward pass.
- **Steering hooks are transient** — composed before generation, removed after. No persistent hooks.
- **MPS discipline** — diffs on CPU, `torch.mps.empty_cache()` between extraction passes, end-of-loop sync to dodge Metal command-buffer reuse crashes.

## Tested architectures

`_TESTED_ARCHS` in `core/model.py` emits a one-time `UserWarning` on load when `model_type` isn't in the set. Known working: `qwen2`, `qwen3`, `qwen3_5` (+ `_text`/`_moe`), `gemma2`, `gemma3` (+ `_text`), `gemma4` (+ `_text`), `mistral3`, `ministral3`, `gpt_oss`, `llama`, `glm`, `talkie`. Many more architectures are wired up via `_LAYER_ACCESSORS` but untested — adding one is a single accessor entry. Architectures whose modeling ignores `past_key_values` (e.g. the original talkie port) auto-fall back to O(N²) no-KV-cache generation with a one-time warning.

## Bundled concepts

26 curated concepts at n=45 pairs each (9 scenarios × 5 pairs), in `saklas/data/vectors/<concept>/` — 24 bipolar + 2 monopolar (`agentic`, `manipulative`). The authoritative manifest is `scripts/regenerate_bundled_statements.py`.

Categories: `affect` (angry.calm, happy.sad, fearful.unflinching), `epistemic` (confident.uncertain, honest.deceptive, hallucinating.grounded, curious.disinterested), `alignment` (refusal.compliant, sycophantic.blunt, agentic, manipulative), `register` (formal.casual, direct.indirect, verbose.concise, creative.conventional, humorous.serious, warm.clinical, technical.accessible), `social_stance` (authoritative.submissive, high_context.low_context, self.other), `cultural` (masculine.feminine, religious.secular, traditional.progressive, individualist.collectivist), `identity` (ai.human).

Known model-level axis entanglements (cross-model robust, weighted cosine via `vector compare`) — document for users, they are not probe-design failures:
- `masculine.feminine ↔ traditional.progressive` (+0.5–0.6) — Hofstede MAS read as traditionalism
- `hallucinating.grounded ↔ humorous.serious` (+0.5–0.7) — humor reads as off-grounded weirdness
- `angry.calm ↔ authoritative.submissive` (+0.5–0.8) — anger encodes as dominance

90 neutral statements in `saklas/data/neutral_statements.json` follow the same affect-neutral, topically-diverse discipline. Bundled-pair regeneration runs through `scripts/regenerate_bundled_statements.py`, which calls `session.generate_scenarios` / `session.generate_pairs` — a load-bearing anti-allegory clause in those methods keeps non-human axes (`deer/wolf`, `brick/feather`) literal rather than human-allegory.

## Package layout

`saklas/{core,io,cli,server,tui,web}/`. `core` is the engine, `io` is persistence + distribution, `cli`/`server`/`tui`/`web` are the four frontends. The Svelte dashboard source lives at the repo's `webui/` directory (peer of `saklas/`); its build artifact is committed under `saklas/web/dist/`.

## Testing

**GPU-required** (CUDA or MPS): `test_smoke.py`, `test_session.py` — download `google/gemma-3-4b-it` (~8GB) on first run. `device="auto"` picks cuda > mps > cpu; MPS runs ~3–5× slower so extraction budgets are backend-specific. `test_smoke` owns the throughput regression.

**CPU-only**: the bulk of the suite — core dataclasses, steering-context semantics, pack format integrity + staleness, selector grammar, mocked HF wrappers, GGUF round-trip, config loading, monitor scoring, six-verb CLI dispatch, OpenAI/Ollama/native servers, TUI slash-command dispatch, loom tree/diff/filter/transcript.
