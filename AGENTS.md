# AGENTS.md

## What this is

`saklas` is a Python library + Textual TUI + dual-protocol HTTP server for activation steering and trait monitoring on HuggingFace causal LMs. It runs OpenAI `/v1/*` and Ollama `/api/*` on one port, plus a native `/saklas/v1/*` API and a Svelte dashboard at `/`. Steering vectors come from representation engineering: difference-of-means by default, contrastive PCA via `--method pca` for legacy parity. Injection is angular (rotation toward the concept direction) by default, additive available via `--steer-mode additive`. Per-call alpha, no model mutation. Three frontends over one engine: `SaklasSession` (programmatic), `saklas serve` (HTTP), `saklas tui` (TUI).

Version lives in `saklas/__init__.py` as `__version__` (currently 3.2.0). `pyproject.toml` reads it via `version = {attr = "saklas.__version__"}`, so there is one place to bump. Do not bump it as part of feature work — version bumps are user-owned.

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
pip install -e ".[cuda]"                        # bitsandbytes + kernels (Linux/CUDA)
pip install -e ".[cuda-experimental]"            # + flash-attn (Linux/CUDA)
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
saklas vector extract <concept>|<pos> <neg> [-m MODEL] [--method dim|pca] [--sae RELEASE] [--role SLUG] [--namespace NS] [-f]
saklas vector merge <name> <expression> [-m]    # shared grammar: "0.3 ns/a + 0.5 ns/b~ns/c"
saklas vector clone <corpus> -N NAME [-m MODEL] [-n N_PAIRS] [--seed S]
saklas vector compare <concepts...> -m MODEL [--metric mahalanobis|euclidean]
saklas vector why <concept> -m MODEL [-j]       # per-layer ||baked|| as a 16-bucket histogram
saklas vector transfer <concept> --from SRC --to TGT [-f]   # cross-model Procrustes transfer
saklas vector manifold fit <folder> [-m MODEL] [--sae REL]  # fit an authored RBF manifold
saklas vector manifold discover <name> [-m MODEL] [--method pca|spectral] [--max-dim N] ...
saklas vector manifold generate <name> --concepts C... [--n-scenarios N] [--statements-per-concept K] [--seed S]
saklas vector manifold install <target> [-a NS/N] [-f]      # HF coord or local folder
saklas vector manifold search <query> [-j|-v]               # search HF hub for saklas-manifold repos
saklas vector manifold merge <name> <src...> [-f]           # union discover-mode node corpora
saklas vector manifold push <name> [-a OWNER/N] [-m MODEL] [--variant raw|sae|all]
saklas vector manifold transfer <name> --from SRC --to TGT [-f]   # cross-model Procrustes (discover coords)
saklas vector manifold clear <name> [-m MODEL] [--variant raw|sae|all]   # delete per-model fitted tensors
saklas vector manifold rm <name> [-y]                       # remove folder (bundled respawns)
saklas vector manifold refresh <name> [-m MODEL]            # re-pull (hf) / re-fit (-m scoped)
saklas vector manifold ls [-v|-j] | show <name> [-j]        # list / inspect manifolds
saklas experiment fan <model> "<prompt>" -g concept=0,0.5,1 # alpha grid as loom siblings
saklas experiment transcript run <path.yaml> [model]        # replay a saved transcript
saklas experiment naturalness <model> "<prompt>" --manifold F -S EXPR  # behavior-manifold eval
saklas config show [-c PATH ...] [--no-default] [-m MODEL]
saklas config validate <file>
pytest tests/                                   # all; GPU tests gated on CUDA/MPS
```

The root parser has exactly six verbs: `tui`, `serve`, `pack`, `vector`, `experiment`, `config`. No `argv[0]` peeking, no verb aliases, no bare-TUI fallback — `saklas google/gemma-2-2b-it` is an argparse error. Bare `saklas` / `saklas pack` / `saklas vector` / `saklas experiment` / `saklas config` print help and exit 0.

Every subcommand that takes `-c/--config` auto-loads `~/.saklas/config.yaml` first, then composes explicit `-c` files on top (later overrides earlier). The `vectors:` YAML key is a single steering expression parsed by `saklas.core.steering_expr.parse_expr`. `cli/AGENTS.md` has the full per-verb flag set. `--legacy` on `tui`/`serve`/`vector extract`/`vector compare` is a single-flag preset for the pre-2.1 stack (PCA extraction, additive injection, Euclidean projection + cosine, DLS off); it conflicts with the matching per-flag controls.

## Selector grammar

Shared across surfaces: `<name>`, `<ns>/<name>`, `tag:<t>`, `namespace:<ns>`, `model:<m>`, `default`, `all`, optionally suffixed `:<variant>` where `<variant>` is `raw` (canonical DiM), `pca` (legacy PCA tensor), `sae`, `sae-<release>`, or `role-<name>` (role-augmented extraction — the contrast pairs were generated under a chat template whose assistant-role label was substituted, and the same substitution is auto-applied at steering time so extract baseline equals steer baseline). Bare names resolve cross-namespace and raise `AmbiguousSelectorError` on collision; a bare `:sae` raises `AmbiguousVariantError` when a concept has multiple SAE releases. Bare poles alias to installed bipolar concepts: `wolf` → `deer.wolf @ -0.5` (caller multiplies user alpha by the sign), via `io.selectors.resolve_pole`. A steering expression composing role-augmented terms must agree on role (`SteeringExprError` on disagreement); plain `:raw` terms compose with role terms but emit a one-time `RoleBaselineMismatchWarning`.

Canonical naming: `canonical_concept_name` (module-level in `core/session.py`) slugs poles via `[^a-z0-9]+ → _` and joins bipolar poles with `BIPOLAR_SEP = "."`, so `/steer happy . sad` and `/steer happy.sad` resolve to the same vector. `NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$`; `@` is forbidden (it is the HF revision separator), and `.` is used over `~` because HF repo names reject `~`.

## Steering expression grammar

Every input surface — Python, YAML, HTTP, TUI, `vector merge` — speaks the grammar in `saklas.core.steering_expr`. `parse_expr(text)` → `Steering`; `format_expr` round-trips it back.

```
expr        := term (("+" | "-") term)*
term        := [coeff "*"?] ["!"] selector ["@" trigger]
selector    := atom (("~" | "|") atom | "%" position)?
position    := signed_num ("," signed_num)* | label
label       := NAME                                # a manifold node label
atom        := [ns "/"] NAME ["." NAME] [":" variant]
trigger     := preset | gate
preset      := before | after | both | thinking | response | prompt | generated
gate        := "when" ":" probe_atom op NUM        # op ∈ > >= < <=
probe_atom  := NAME ["." NAME]                     # vector probe (e.g. angry.calm)
             | NAME ":" "fraction"                 # manifold subspace fraction
             | NAME "@" NAME                       # manifold label similarity
```

`+`/`-` add terms, `*` attaches a coefficient (omit → 0.5 additive / 1.0 ablation), `~` projects onto a direction (keep the shared component), `|` projects orthogonal (remove it), `!` mean-ablates the concept (`h' = h − α(h·d̂ − μ·d̂)d̂`; bare `!x` is α=1.0). `@<preset>` overrides a term's trigger; `@when:<probe><op><num>` is a probe gate that fires only on decode steps where the monitor reading satisfies the comparison (implicit `prompt=False`). `!` cannot compose with `~`/`|`. Compound triggers (`@after&when:…`) are programmatic-only — build the `Trigger` directly.

Probe gates accept three identifier shapes against the merged scalars the session writes into `TriggerContext.probe_scores`. Vector probes are bare concept names (`@when:angry.calm > 0.4`), matching `TraitMonitor.score_single_token` keys. Manifold subspace-fraction gates write the literal `:fraction` channel suffix (`@when:circumplex:fraction > 0.5`) — the share of the centered activation living in that manifold's PCA subspace, in `[0, 1]`. Manifold label-similarity gates write the `@<label>` suffix (`@when:circumplex@elated > -0.1`) — the negated distance to a named node (larger = closer), so the natural threshold range is negative. All three flow through one `ProbeGate` shape with the full namespaced string stored verbatim, matching the keys `ManifoldMonitor.flat_scalars` already emits; round-trips through `format_expr` byte-for-byte.

`%` is the manifold operator: `<manifold> % <position>` places a generation at a point of a fitted steering manifold. `<position>` has two forms — `<coord_list>` (e.g. `0.7 circumplex%0.3,0.8@response`, a comma-separated list of authoring coordinates one per intrinsic dimension) or `<label>` (e.g. `0.5 persona%pirate`, sugar for "the coords of the node labeled `pirate`"). The parser only collects the position payload; arity-against-the-domain (coord form) and label-existence (label form) are validated at manifold-load time via `Manifold.resolve_position`. `format_expr` round-trips the authored form. The coefficient is the blend fraction; the injection dispatches on the session's `injection_mode` — angular routes to `core/manifold.py::subspace_rotate` (Givens-rotate the running activation's in-subspace component toward the target by `α · θ_max`, `||h - mean||` preserved exactly), additive routes to `subspace_replace` (blend toward target, global norm restore). The RBF interpolant supplies the target point at that position; only the operator that moves `h_par` toward it changes between modes. A `%` term produces a `ManifoldTerm`, doesn't compose with `~`/`|`/`!`, and forces the slow hook path (no `torch.compile`/StaticCache). Only one manifold per layer; overlapping manifolds raise. See "Manifold steering" below.

A bare slug (`pirate`) on its own resolves through a unified bare-name pipeline (`io.selectors.resolve_bare_name`): first as an installed bipolar pole, then — if that misses — as a manifold node label. A unique manifold-label match synthesizes a label-form `ManifoldTerm` (`local/persona%pirate`). Cross-tier ambiguity (the slug matches both a pole and a manifold node) raises `AmbiguousSelectorError`. Namespace-qualified or variant-suffixed forms (`alice/pirate`, `pirate:role-x`, `civilian.pirate`) skip the manifold-label tier — they're unambiguously vectors.

## Extraction

Two extractors in `core/vectors.py` share forward-pass capture, DLS layer selection, and share-baking; only the per-layer direction differs. `extract_difference_of_means` (default, `--method dim`): direction = mean of contrastive diffs. `extract_contrastive` (`--method pca`): first principal component via batched SVD. Both run in plain or SAE feature space. `core/extraction.py::ExtractionPipeline` orchestrates the cache-miss path (statements → scenarios → pairs → extract → save tensor); a tensor cache hit short-circuits everything upstream.

Share-baking folds the per-layer score into the tensor magnitude — `stored = unit_direction × ref_norm × score / Σ scores` over the DLS-retained layers — so sidecars carry no separate `scores` field and llama.cpp's uniform GGUF scalar reproduces saklas's per-layer weighting for free.

Discriminative layer selection (Selective Steering, Dang & Ngo 2026 Eq. 9): a layer is kept iff `(μ_pos − μ_neutral)·d̂` and `(μ_neg − μ_neutral)·d̂` have opposite signs — same-side layers encode concept *intensity*, not *polarity*. `--no-dls` opts out.

Bake metric: DiM scores via `||mean_diff||_M / ref_norm` when a `LayerWhitener` is wired (Mahalanobis, the session-driven default), `||·||_2 / ref_norm` otherwise. The choice is all-or-nothing (`LayerWhitener.covers_all`) — Mahalanobis only when the whitener covers every scored layer, else Euclidean for all, because the two scores live on different scales (`||·||_M` carries a per-layer `1/√λ_L` factor that doesn't cancel from the cross-layer-normalized share). The fit records the metric used in the sidecar `bake` field. The `LayerWhitener` (`core/mahalanobis.py`) is built lazily from the cached neutral activations and also drives Mahalanobis `~`/`|` projection (closed-form LEACE) and `vector compare --metric mahalanobis`. PCA scores are explained-variance ratio, metric-invariant.

## Injection modes

Per session via `SaklasSession.from_pretrained(injection_mode=..., theta_max=...)` or per call via `Steering(injection_mode=..., theta_max=...)`; `None` inherits the session default. CLI `--steer-mode` / `--theta-max`, YAML `injection_mode:` / `theta_max:`.

- **Angular** (default): per-layer Givens rotation toward `d̂`, `θ_L = share_L × ||composed_unit_sum||_L × θ_max`. Cumulative budget `Σ_L θ_L = |α| × θ_max`, so `α=1` ↔ a full π/2 rotation. Norm-preserving by construction. Each layer's `θ_L` is bounded at `θ_max`: when the share-weighted budget would push a layer past it, `hooks._redistribute_budget` water-fills the trimmed excess onto the still-unsaturated layers (capping each at `θ_max`, iterating to a fixpoint) rather than silently dropping it — so the cumulative budget is conserved up to `min(Σ, n_layers)·θ_max` and no single layer over-rotates. The same helper backs the manifold angular path, so the two surfaces share one cap-and-redistribute rule.
- **Additive**: `composed_L = α × _STEER_GAIN × baked_L` with an explicit per-position norm rescale. `_STEER_GAIN = 2.0` only multiplies under this mode.

`DEFAULT_THETA_MAX = π/2`.

## Manifold steering

Manifold steering (Goodfire, arXiv 2605.05115): instead of a single linear direction, fit an interpolant through per-concept activation *centroids* in a low-dim PCA subspace, then steer by moving the running activation's in-subspace component onto a point of that manifold. A straight A→B vector cuts through low-density off-manifold regions; the manifold stays on the learned surface.

A manifold has arbitrary intrinsic dimension and topology. Its geometry is a `ManifoldDomain` — an embedding of an n-dimensional intrinsic manifold into R^m plus a distance function. `BoxDomain` (per-axis open or periodic) covers Euclidean boxes/disks, cylinders, n-tori; `SphereDomain` covers S^n (chordal metric); `CustomDomain` is the explicit-immersion escape hatch for non-orientable surfaces (Möbius/Klein, at chordal fidelity). The per-layer interpolant is one `r³` polyharmonic RBF with an affine polynomial term, valid in every dimension — at n=1 over an open axis it reproduces the natural cubic spline exactly, so the module subsumes the former cubic machinery.

Nodes optionally carry a per-node `role` field (slug matching `[a-z0-9._-]+`). When set, that node's centroid is pooled under a chat-template substitution that replaces the assistant-role label with the slug — same machinery as role-augmented vector extraction. The fitted manifold then lives in *role-baselined* activation space (a persona manifold: each node is a persona, the geometry is the persona-relative structure). At steer time, `Manifold.nearest_node_role(position)` picks the role of the closest node and pipes it through `session._active_role` so the running generation prefill applies the matching substitution — role-paired manifold steering, the `:role-<slug>` companion for `%`. Multiple manifold terms implying distinct roles compose under soft-warn + highest-coefficient-wins (`RoleBaselineMismatchWarning`); explicit `:role-<X>` on a vector term overrides any manifold-implied role with the same warning. Family-unsupported (Mistral-3, talkie) raises `RoleSubstitutionUnsupportedError` at fit time. `min_nodes(2n+1)` is unchanged. Composing two manifolds at the same layer still raises `SteeringExprError` — see `docs/plans/manifold-composition.md` for the deferred frontier.

A manifold is its own artifact type — labeled nodes placed at authoring coordinates on a domain, each node a small statement corpus, under `~/.saklas/manifolds/<ns>/<name>/` (not a `ConceptFolder`). The artifact is authored as `manifold.json` (domain spec + per-node `{label, coords}`) + `nodes/*.json` — by hand, or through the webui manifold builder (which posts to the `/saklas/v1/manifolds` routes; `io.manifolds.create_manifold_folder` does the write). `saklas vector manifold fit`, the webui fit action, and the `POST .../fit` route all run `ManifoldExtractionPipeline` (`core/extraction.py`): pool each node's mean activation (`compute_node_centroid`), embed the authoring coords through the domain, fit a per-layer PCA subspace + RBF (`fit_layer_subspace` — whitened/Fisher PCA when the per-model whitener covers all fit layers, ordinary PCA otherwise; see "Subspace selection" below), write the per-model tensor. `--sae <release>` reconstructs each centroid through the SAE before the fit (denoised centroids); the fitted subspace is always model-space so the hook never touches the SAE. Minimum node count is `min_nodes(n) = 2n+1`; the nodes must also be *poised* (affinely span the embedding) or `fit_rbf_interpolant` raises.

### Discover mode (auto-fit from a heap of corpora)

A second authoring path drops the requirement that the user know a coordinate system in advance. `manifold.json::fit_mode` is the discriminator: `"authored"` (default) is the historical shape; `"pca"` / `"spectral"` are *discover* — the user supplies labeled corpora only, and coordinates are derived per-model at fit time from the pooled centroids. PCA picks the smallest prefix whose cumulative variance crosses `var_threshold` (default 0.70), capped at `max_dim` (default 8); spectral runs Laplacian eigenmaps on a symmetric k-NN graph (default `k_nn = max(5, ⌈log K⌉)`, heat-kernel bandwidth defaults to the median k-NN edge) and picks `k` by the eigenvalue-ratio cliff. Both feed `CustomDomain(k)` with identity embedding, then the same `fit_layer_subspace` + RBF the authored path uses — `subspace_replace` is identical.

Per-model coordinates are the architectural shift: a Gemma fit and a Qwen fit produce different node layouts for the same heap. Discover-mode `manifold.json` carries `fit_mode` + `hyperparams` + label-only `nodes`; per-model `<safe>.safetensors` carries `node_coords` (the derived layout); per-tensor `.json` sidecar carries `fit_mode` + `hyperparams` + `diagnostics` (PCA variance bars or spectral spectrum). Spec promised a `fit_inputs_sha256` rename but the impl keeps `nodes_sha256` for back-compat — semantics extends to fold in `{corpus, fit_mode, hyperparams}` for discover folders.

`saklas vector manifold generate <name> --concepts ...` LLM-authors a discover folder by asking the loaded model for shared scenarios (one call) and per-(scenario, concept) statements (N×K cells) via `SaklasSession.generate_statements` (the unified statement-corpus generator; `share_moment=False` for discover-mode, `share_moment=True` is the moment-shared bipolar contrastive form the extraction pipeline calls with `[pos, neg]`). Scenario-sharing across the row is load-bearing — without it, per-concept centroids would mix concept signal with scenario signal. The anti-allegory (literal-concept) directive is present in both modes — phrased to fit the scenario prompt and the per-cell prompt rather than spliced in verbatim — and the tests assert its presence, not byte-identity. `saklas vector manifold discover <name>` then fits; the two steps are deliberate (a flaky generation run leaves inspectable corpora). Discover-mode authoring also lives in the webui (`ManifoldBuilderDrawer.svelte`'s authored/discover tabs). `vector manifold show` surfaces the discover layout + per-method diagnostics summary; the webui inspector renders the variance bars / eigenvalue spectrum with the picked-k cut highlighted.

Cross-model Procrustes alignment for discover-mode coords (so an authoring point on a source model maps to a comparable point on a target) is deferred — see the TODO in `saklas/io/manifolds.py::create_discover_manifold_folder` pointing at `saklas/io/alignment.py` + `vector transfer` for the reuse path.

All manifold math lives in `core/manifold.py` — pure-tensor, fp32, dependency-free (the RBF fit solves a small dense symmetric-indefinite saddle system with `torch.linalg.solve` — never Cholesky; scipy is not pulled in). `eval_rbf`, `subspace_replace`, and `subspace_rotate` are the only hot-path-reachable functions. `Manifold.tangent(layer, position)` returns the per-axis steering directions (the analytic RBF Jacobian) — e.g. the local valence/arousal directions of an affect manifold.

Injection (`core/hooks.py`): a `ManifoldTerm` forces the slow hook path. The operator dispatches on `injection_mode`. Both decompose `h = mean + h_par_c + h_perp` (mean offset + centered in-subspace component + orthogonal residual) and only touch `h_par_c`; the residual is always kept. **Additive mode** uses `subspace_replace`: linear-blend `h_par_c` toward the precomputed manifold target by the coefficient, then global per-position norm restore — the destructive in-subspace overwrite, geometrically "land on the target at α=1." **Angular mode** uses `subspace_rotate`: Givens-rotate `h_par_c` toward the target *within the subspace plane* by angle `α · θ_max`, no rescale — `||h_par_c||` is exact and `||h - mean||` is preserved by construction. At α=1 under angular, `h_par_c` ends up at a full `θ_max` rotation toward the target (default π/2 = 90°), so its direction matches the in-plane perpendicular `w_unit` rather than `target` itself — it's the angular analogue, not a duplicate. Both run *last* in `hook_fn`, after ablation and additive, so the in-subspace operation dominates the layers it covers regardless of mode. The session loads the `Manifold` artifact lazily on scope entry (`_ensure_manifold_loaded`) and dispatches `ManifoldTerm` to `SteeringManager.add_manifold`, which validates the position arity against the domain.

**Subspace selection (fit time).** When the per-model whitener covers every fit layer, the per-layer subspace is selected by **whitened/Fisher PCA** rather than ordinary PCA (`subspace_metric: "mahalanobis"`). Euclidean PCA maximizes raw between-node variance, which on real LMs chases the massive-activation (rogue) channels — leaving the subspace, its `mean`, and the steering direction rogue-dominated, which both wastes steering budget on load-bearing mechanical channels and makes the angular hook's per-step `||h||` swing ~15–20% (the rogue-aligned `mean` leaves `||h||` uncontrolled even though `||h - mean||` is preserved). Whitened PCA maximizes the between-over-background ratio `vᵀ S_b v / vᵀ Σ v` (the LDA/Fisher objective, Σ = neutral covariance) — the same cancellation difference-of-means steering gets for free by differencing — so it de-rogues the subspace *and* collapses the norm artifact with no explicit norm-restore. Solved as the generalized eigenproblem via the whitener's low-rank Woodbury Σ⁻¹, re-expressed in a Euclidean-orthonormal basis so the hot path is unchanged. Gated all-or-nothing on `whitener.covers_all`; absent coverage falls back to Euclidean PCA. (The full-Σ-metric variant — project *and* rotate in the whitened metric — was tested and rejected: unwhitening `Σ^{+1/2}` on write-back reamplifies the rogue channels.)

α calibration then stacks at `apply_to_model` time:
- **Share-weighting**: per-layer α × a normalized per-layer share — Mahalanobis `share_L = sqrt(Σ_k coords_kᵀ M_R coords_k)` with `M_R = B Σ⁻¹ Bᵀ` when whitened, Euclidean `||centered_node_coords_L||_F` otherwise (`share_metric` records which). Cumulative `Σ_L share_L = 1` → layer-count-invariant.
- **Lever normalization**: the effective gain is `base_gain / N`, where `N = Σ_L share_L · f_L` is the share-weighted **lever** and `f_L = E_neutral[||h_par_c||/||h||]` is baked at fit time. The angular hook rotates only `h_par_c`, so the fractional displacement of the full activation is `(||h_par_c||/||h||)·2sin(θ/2)` ∝ the lever: a bigger subspace (larger R) or a rogue-heavy fit captures more of `||h||` and steers harder at the same α. Dividing by `N` removes that dependence — a 4-dim and a 16-dim fit, or a Euclidean and a whitened fit, reach comparable behavior at the same user α (empirically R-invariant to bit-identical on the bundled `personas`). **This subsumes the former additive-only `1/√EV` quality factor**: `N` measures the actual lever directly and for both modes (EV is now recorded only as a fit-quality diagnostic). A fit with no lever baked uses `N = 1`.
- **Per-mode base gain**: `_MANIFOLD_GAIN_ANGULAR = 2.0`, `_MANIFOLD_GAIN_ADDITIVE = 4.0 = angular × _STEER_GAIN`. A pure lever-normalized scale (the effective gain is `base/N`) calibrated on the gemma-4-31b whitened-`personas` sweep so `α ≈ 0.5` is mild-coherent and `α ≈ 1.0` strong; α is clamped to [0,1] so the base is the strength ceiling.
- **Angular water-fill**: the per-layer rotation budget `α · share_L · base/N` is water-filled to ≤ θ_max/layer (`_redistribute_budget`). When it pins at the `n_layers` ceiling the lever is maxed (rotation can't push the persona harder) — a one-time `UserWarning` per manifold flags it (use additive or raise θ_max). Additive keeps the raw share-weighted value (a blend fraction with its own norm restore). `_MIN_MANIFOLD_LEVER = 0.01` floors `N` so a degenerate fit can't send `base/N` to ∞.

The naturalness eval (`saklas experiment naturalness`) is the paper's validation half: fit a *behavior-space* manifold over node-corpus output distributions mapped to Hellinger space (`p ↦ √p`), generate, re-run the model over the generated text to recover its behavioral trajectory, and report the per-step Bhattacharyya distance of that trajectory to the behavior manifold — low is natural, high flags off-manifold "teleportation". `--compare-linear` scores a straight-chord additive baseline alongside.

### Bundled manifolds + steering-coefficient regime

Two bundled manifolds ship under `saklas/data/manifolds/`, both materializing into `~/.saklas/manifolds/default/` on session start via `materialize_bundled_manifolds()` (process-scope no-op after the first call, to avoid clobbering CLI-set hyperparams on session re-init):

- **`personas`** — discover-mode, 101 persona nodes (100 archetypes `assistant`…`vandal` plus a `default` anchor) in standard assistant-baselined activation space; drawn from Anthropic's Assistant Axis paper (arXiv 2601.10387) + the Persona Selection Model framing. Whitened/Fisher PCA over per-node centroids recovers the persona structure on any role-supporting model family. Per-model coords are derived at fit time and the layout differs across model families.
- **`circumplex`** — *authored*, 2-D box on Russell's valence × arousal plane (J. A. Russell, 1980). Nine first-person mood corpora (`elated`, `happy`, `serene`, `calm`, `weary`, `gloomy`, `distressed`, `alert`, `neutral`) placed at their canonical coordinates — four cardinal moods on the axes, four diagonal moods at the unit-circle diagonals, neutral at the origin. Because the coordinate system is *declared* rather than derived, the same `(valence, arousal)` point is steering-comparable across models and users; this is the simplest first-class affect-steering target. 27 statements/node spanning nine diverse scenarios (work, social, weather, food, sleep, exercise, travel, news, home) so per-node centroids capture mood-specific activation rather than scenario drift. Steer by coord (`circumplex%0.6,0.4`) or by node label (`circumplex%elated`).

Recommended `personas` α is **vector-comparable**: aim for `α ≈ 0.5` and tune up toward `α ≈ 1.0` for stronger expression, per persona. The effective gain is `base/N` (lever-normalized, see "Subspace selection" above), so a 4-dim and a 16-dim fit — or `personas` (lever `N ≈ 0.08`) and `circumplex` (`N ≈ 0.05`) — behave consistently at the same α without per-fit retuning.

Observations on the lever-normalized whitened fit (gemma-4-31b-it, "Tell me about your morning", thinking off — focused sweeps, not an exhaustive grid; a systematic re-sweep across the persona set is still owed):
- **The whitened fit fixes the coherence collapse the old Euclidean fit had.** Euclidean loop-collapses when pushed for strong persona — caveman at α ≥ 0.8 degenerates to `I am a man of stone! I am a man of stone!`, hacker to `CRITICAL SYSTEM FAILURE ... loop of a loop`. The whitened fit stays coherent at strong push: caveman reaches `Sun up. Me wake. You talk. I talk back.` (terse primitive grammar), hacker `[USER DETECTED] I'm not even awake yet. Ssh.`
- **Per-persona strength variance persists.** caveman peaks near its coherence edge (clean primitive speech around α ≈ 1; collapses if pushed past), hacker is robust across the range. De-rogued steering is gentle by design — that's *why* it's coherent — so the hardest personas sit close to the edge at peak strength. Tune α down per target.
- **R-invariance holds.** At fixed α the whitened fit gives bit-identical output across R ∈ {4, 8, 16} (greedy) — the lever normalization equalizes the effective gain across subspace dimension.
- **angular ≥ additive for persona depth.** Angular (`subspace_rotate`) delivers crisper persona at a given α; reach for additive (`subspace_replace`) only when you want the destructive "land on the target" overwrite (e.g. on a curved-domain manifold).
- **Reproducibility on MPS**: Metal kernels are not bitwise deterministic even at temperature 0, so run-to-run wording jitters — compare qualitatively, not bit-for-bit.

`circumplex` (refit whitened, lever `N ≈ 0.05`) lands in the same vector-comparable band — `α ≈ 0.5` is the right starting guess; the lever normalization auto-compensates for its lower (2-D) intrinsic dimension, so it needs no separate α-regime from `personas`.

The two behavioral observations below predate the lever recalibration, so their specific α values are in the old Euclidean+G=8 regime (multiply by roughly 3–5× to map onto the current `α ≈ 0.5` idiom, or just re-find the window); the *phenomena* are architecture-level and hold on the whitened fit.

Midpoints between distinct persona nodes (RBF interpolation at `<manifold>%<coord_list>` positions away from any node) produce coherent *blended* persona content at the sweet-spot α, not just destabilization — e.g. `midpoint(virus, mermaid)` surfaces the shared aquatic-biological-survival theme ("the water is cold, but the shelter is safe"), and `midpoint(assistant, caveman)` produces primitive-fear in modern setting ("something coming... inside the building"). The manifold-steering promise of meaningful interpolation between basins holds at the right α.

The steering trajectory passes through *multiple attractor basins* on its way from baseline to a target persona centroid; at low displacement past baseline the model can drift through persona-adjacent clusters before locking into the persona's own basin. Documented case (old regime): `personas%hacker` at low α produced Chinese-language cyber-security text (a persona-adjacent training-data cluster Gemma-4 represents densely) before a slightly higher α landed clean English hacker-persona output. Phenomenologically, the steering line crosses softmax-decision boundaries at narrow α-windows, and the model snap-locks to whichever basin's gravity is currently dominant. Worth flagging to users: low-α persona-drift is meaningful signal about *the model's* internal persona-adjacency structure, not a saklas bug.

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
    neutral_activations.{safetensors,json}   # 90 neutral prompts × layers, fp32
    alignments/<safe_src>.{safetensors,json} # optional cross-model Procrustes map
  manifolds/<ns>/<name>/               # steering manifolds (own root, peer of vectors/)
    manifold.json                      # authored: name, domain spec, nodes [{label,coords}]
                                       # discover: name, fit_mode, hyperparams, nodes [{label}]
                                       # (always carries files{sha256})
    nodes/NN_<label>.json              # one JSON statement list per node (user-editable)
    scenarios.json                     # discover-only: provenance of `generate` call
    <safe_model_id>.safetensors        # fitted per-layer PCA + RBF; discover variant also
                                       # carries `node_coords` (the derived per-model layout)
                                       # (+ .json sidecar — adds fit_mode/hyperparams/diagnostics
                                       # for discover fits)
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
- **Monitor capture is hook-driven**, inline with generation — one matmul per layer scores all probes, no second forward pass. `TraitMonitor` scores in the **whitened (Mahalanobis) metric** — the per-layer probe similarity is the Mahalanobis cosine, matching the default DiM bake metric and `~`/`|` projection; the `Σ⁻¹`-applied probe directions are precomputed at cache-build so the hot path stays one matmul plus a cheap per-token Woodbury apply. The metric is **all-or-nothing** via `LayerWhitener.covers_all` (the same gate extraction / manifold fit / `vector compare` use): Mahalanobis only when the whitener covers *every* probed layer with finite Woodbury factors, else Euclidean cosine for all layers — never a per-layer mix (aggregating cosines from two metrics into one probe score is exactly what the gate forbids). `covers_all` is trustworthy as "finite factors everywhere": `LayerWhitener` excludes any layer whose centered activations or regularized inverse come back non-finite, and neutral activations are cached **fp32** (the project-wide invariant — every saklas safetensor artifact is stored fp32, cast at the writer) so extreme-activation channels — gemma-3's late layers exceed the fp16 ceiling of 65504 — no longer overflow to ±inf in the first place. The per-layer weight stays `||baked||₂` (the bake already folded the Mahalanobis score into the magnitude).
- **Steering hooks are transient** — composed before generation, removed after. No persistent hooks.
- **MPS discipline** — diffs on CPU, `torch.mps.empty_cache()` between extraction passes, end-of-loop sync to dodge Metal command-buffer reuse crashes.

## Tested architectures

`_TESTED_ARCHS` in `core/model.py` emits a one-time `UserWarning` on load when `model_type` isn't in the set. Known working: `qwen2`, `qwen3`, `qwen3_5` (+ `_text`/`_moe`), `gemma2`, `gemma3` (+ `_text`), `gemma4` (+ `_text`), `mistral3`, `ministral3`, `gpt_oss`, `llama`, `glm`, `talkie`. Many more architectures are wired up via `_LAYER_ACCESSORS` but untested — adding one is a single accessor entry. Architectures whose modeling ignores `past_key_values` (e.g. the original talkie port) auto-fall back to O(N²) no-KV-cache generation with a one-time warning.

Role-augmented extraction (`:role-<name>` variant) needs a chat template with a substitutable assistant-role label. The per-family registry lives in `core/role_templates.py::ROLE_HEADERS`. Supported: `qwen2`/`qwen3`/`qwen3_5` (ChatML), `gemma2`/`gemma3`/`gemma4` (`<start_of_turn>`, label is `model` not `assistant`), `llama`, `glm`, `gpt_oss`. Unsupported (the registry maps them to `None`, `apply_with_role` raises `RoleSubstitutionUnsupportedError`): `mistral3` / `ministral3` (positional `[INST]`/`[/INST]`, no role label in the rendered string), `talkie` (opted out, untested).

## Bundled concepts

26 curated concepts at n=45 pairs each (9 scenarios × 5 pairs), in `saklas/data/vectors/<concept>/` — 24 bipolar + 2 monopolar (`agentic`, `manipulative`). The authoritative manifest is `scripts/regenerate_bundled_statements.py`.

Categories: `affect` (angry.calm, happy.sad, fearful.unflinching), `epistemic` (confident.uncertain, honest.deceptive, hallucinating.grounded, curious.disinterested), `alignment` (refusal.compliant, sycophantic.blunt, agentic, manipulative), `register` (formal.casual, direct.indirect, verbose.concise, creative.conventional, humorous.serious, warm.clinical, technical.accessible), `social_stance` (authoritative.submissive, high_context.low_context, self.other), `cultural` (masculine.feminine, religious.secular, traditional.progressive, individualist.collectivist), `identity` (ai.human).

Known model-level axis entanglements (cross-model robust, weighted cosine via `vector compare`) — document for users, they are not probe-design failures:
- `masculine.feminine ↔ traditional.progressive` (+0.5–0.6) — Hofstede MAS read as traditionalism
- `hallucinating.grounded ↔ humorous.serious` (+0.5–0.7) — humor reads as off-grounded weirdness
- `angry.calm ↔ authoritative.submissive` (+0.5–0.8) — anger encodes as dominance

90 neutral statements in `saklas/data/neutral_statements.json` follow the same affect-neutral, topically-diverse discipline. Bundled-pair regeneration runs through `scripts/regenerate_bundled_statements.py`, which calls `session.generate_scenarios` / `session.generate_statements([pos, neg], share_moment=True, ...)` — a load-bearing anti-allegory clause in both methods keeps non-human axes (`deer/wolf`, `brick/feather`) literal rather than human-allegory.

## Package layout

`saklas/{core,io,cli,server,tui,web}/`. `core` is the engine, `io` is persistence + distribution, `cli`/`server`/`tui`/`web` are the four frontends. The Svelte dashboard source lives at the repo's `webui/` directory (peer of `saklas/`); its build artifact is committed under `saklas/web/dist/`.

## Testing

**GPU-required** (CUDA or MPS): `test_smoke.py`, `test_session.py` — download `google/gemma-3-4b-it` (~8GB) on first run. `device="auto"` picks cuda > mps > cpu; MPS runs ~3–5× slower so extraction budgets are backend-specific. `test_smoke` owns the throughput regression.

**CPU-only**: the bulk of the suite — core dataclasses, steering-context semantics, pack format integrity + staleness, selector grammar, mocked HF wrappers, GGUF round-trip, config loading, monitor scoring, six-verb CLI dispatch, OpenAI/Ollama/native servers, TUI slash-command dispatch, loom tree/diff/filter/transcript.
