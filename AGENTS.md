# AGENTS.md

## What this is

`saklas` is a Python library + Textual TUI + dual-protocol HTTP server for
activation steering and trait monitoring on HuggingFace causal LMs. It runs
OpenAI `/v1/*` and Ollama `/api/*` on one port, plus a native `/saklas/v1/*` API
and a Svelte dashboard at `/`. Steering signal comes from representation
engineering: difference-of-means vectors, multi-node affine subspaces, and curved
RBF manifolds. Every steering term — vectors, poles, `~`/`|` projections, `!`
ablations, and `%` manifold positions — lowers at generation time to one unified
per-layer injection (the along/onto subspace kernel, `core/manifold.py::subspace_inject`).
Per-call coefficients, no model mutation. Three frontends over one engine:
`SaklasSession` (programmatic), `saklas serve` (HTTP), `saklas tui` (TUI).

Version lives in `saklas/__init__.py` as `__version__` (currently 3.2.0).
`pyproject.toml` reads it via `version = {attr = "saklas.__version__"}`, so there
is one place to bump. Do not bump it as part of feature work — version bumps are
user-owned.

Releases: merge a version bump to `main` → `.github/workflows/release.yml` tags
`v$VERSION`, builds, publishes via trusted publishing, and cuts a GitHub release.
A push without a bump is a no-op.

The cross-cutting design — how extraction, composition, injection, and reads fit
together — lives in the repo-root `ARCHITECTURE.md`. Read it before touching the
engine.

## Subtree docs

Deep internals live in subtree `AGENTS.md` files — Claude Code auto-loads each
when you work in that directory. Consult them only when editing that layer.

- `saklas/core/AGENTS.md` — model loading, extraction, the subspace/manifold fit +
  injection, monitor, session, generation loop, loom tree
- `saklas/io/AGENTS.md` — packs, manifolds, HF distribution, GGUF, cloning,
  alignment, paths/selectors
- `saklas/cli/AGENTS.md` — eight-verb dispatch, config loading, flags
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
saklas tui <model_id> [--projection-metric {mahalanobis,euclidean}] [--no-dls]
saklas serve <model_id> [--no-web] [--steer/-S EXPR]
saklas pack install <target> [-s|-a NS/N|-f]    # HF coord or folder; -s = statements only
saklas pack refresh <selector> [-m MODEL]       # re-pull; `refresh neutrals` is reserved
saklas pack clear <selector> [-m MODEL] [--variant raw|sae|all]   # delete per-model tensors
saklas pack rm <selector> [-y]                  # remove folder (bundled respawns)
saklas pack ls [selector] [-j|-v]               # LOCAL installed packs only
saklas pack search <query> [-j|-v]              # search HF hub for saklas-pack repos
saklas pack push <selector> [-a OWNER/NAME] [-m MODEL] [--variant raw|sae|all] ...
saklas pack export gguf <selector> [-m MODEL] [-o PATH] [--model-hint HINT]
saklas subspace extract <concept>|<pos> <neg> [-m MODEL] [--sae RELEASE] [--role SLUG] [--namespace NS] [-f]
saklas subspace merge <name> <expression> [-m]    # shared grammar: "0.3 ns/a + 0.5 ns/b~ns/c"
saklas subspace clone <corpus> -N NAME [-m MODEL] [-n N_PAIRS] [--seed S]
saklas subspace compare <concepts...> -m MODEL [--metric mahalanobis|euclidean]
saklas subspace why <concept> -m MODEL [-j]       # per-layer ||baked|| as a 16-bucket histogram
saklas subspace transfer <concept> --from SRC --to TGT [-f]   # cross-model Procrustes transfer
saklas manifold fit <folder> [-m MODEL] [--sae REL]  # fit an authored manifold
saklas manifold discover <name> [-m MODEL] [--method pca|spectral] [--max-dim N] ...
saklas manifold generate <name> --concepts C... [--n-scenarios N] [--statements-per-concept K] [--seed S]
saklas manifold install <target> [-a NS/N] [-f]      # HF coord or local folder
saklas manifold search <query> [-j|-v]               # search HF hub for saklas-manifold repos
saklas manifold merge <name> <src...> [-f]           # union discover-mode node corpora
saklas manifold push <name> [-a OWNER/N] [-m MODEL] [--variant raw|sae|all]
saklas manifold transfer <name> --from SRC --to TGT [-f]   # cross-model Procrustes (discover coords)
saklas manifold clear <name> [-m MODEL] [--variant raw|sae|all]   # delete per-model fitted tensors
saklas manifold rm <name> [-y]                       # remove folder (bundled respawns)
saklas manifold refresh <name> [-m MODEL]            # re-pull (hf) / re-fit (-m scoped)
saklas manifold ls [-v|-j] | show <name> [-j]        # list / inspect manifolds
saklas experiment fan <model> "<prompt>" -g concept=0,0.5,1 # alpha grid as loom siblings
saklas experiment transcript run <path.yaml> [model]        # replay a saved transcript
saklas experiment naturalness <model> "<prompt>" --manifold F -S EXPR  # behavior-manifold eval
saklas config show [-c PATH ...] [--no-default] [-m MODEL]
saklas config validate <file>
pytest tests/                                   # all; GPU tests gated on CUDA/MPS
```

The root parser has exactly eight verbs: `tui`, `serve`, `pack`, `subspace`,
`manifold`, `vector`, `experiment`, `config`. `vector` is a deprecated 3.x alias
for `subspace` plus the old nested `vector manifold` tree; new commands should
use top-level `subspace` / `manifold`. No `argv[0]` peeking, no bare-TUI fallback
— `saklas google/gemma-2-2b-it` is an argparse error. Bare `saklas` / `saklas
pack` / `saklas subspace` / `saklas manifold` / `saklas vector` / `saklas
experiment` / `saklas config` print help and exit 0.

Every subcommand that takes `-c/--config` auto-loads `~/.saklas/config.yaml`
first, then composes explicit `-c` files on top (later overrides earlier). The
`vectors:` YAML key is a single steering expression parsed by
`saklas.core.steering_expr.parse_expr`. `cli/AGENTS.md` has the full per-verb flag
set. There is no `--steer-mode`/`--theta-max`/`--method`/`--legacy` surface — the
injection kernel is unified, difference-of-means is the only vector extraction
method, and the only steering knobs at the CLI are `--projection-metric` and
`--no-dls`.

## Selector grammar

Shared across surfaces: `<name>`, `<ns>/<name>`, `tag:<t>`, `namespace:<ns>`,
`model:<m>`, `default`, `all`, optionally suffixed `:<variant>` where `<variant>`
is `raw` (canonical DiM), `sae`, `sae-<release>`, `from-<safe_src>` (cross-model
transfer), or `role-<name>` (role-augmented extraction — the contrast pairs were
generated under a chat template whose assistant-role label was substituted, and
the same substitution is auto-applied at steering time so extract baseline equals
steer baseline). There is no `pca` variant. Bare names resolve cross-namespace and
raise `AmbiguousSelectorError` on collision; a bare `:sae` raises
`AmbiguousVariantError` when a concept has multiple SAE releases. Bare poles alias
to installed bipolar concepts: `wolf` → `deer.wolf @ -0.5` (caller multiplies user
alpha by the sign), via `io.selectors.resolve_pole`. A steering expression
composing role-augmented terms must agree on role; plain `:raw` terms compose with
role terms but emit a one-time `RoleBaselineMismatchWarning`.

Canonical naming: `canonical_concept_name` (module-level in `core/session.py`)
slugs poles via `[^a-z0-9]+ → _` and joins bipolar poles with `BIPOLAR_SEP = "."`,
so `/steer happy . sad` and `/steer happy.sad` resolve to the same vector.
`NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$`; `@` is forbidden (HF revision separator),
and `.` is used over `~` because HF repo names reject `~`.

## Steering expression grammar

Every input surface — Python, YAML, HTTP, TUI, `vector merge` — speaks the
grammar in `saklas.core.steering_expr`. `parse_expr(text)` → `Steering`;
`format_expr` round-trips it back.

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

`+`/`-` add terms, `*` attaches a coefficient (omit → 0.5), `~` projects onto a
direction (keep the shared component), `|` projects orthogonal (remove it), `!`
mean-ablates the concept (`h' = h − α(h·d̂ − μ·d̂)d̂`; bare `!x` is α=1.0).
`@<preset>` overrides a term's trigger; `@when:<probe><op><num>` is a probe gate
that fires only on decode steps where the monitor reading satisfies the comparison
(implicit `prompt=False`). `!` cannot compose with `~`/`|`. Compound triggers
(`@after&when:…`) are programmatic-only.

Probe gates accept three identifier shapes against the merged scalars the session
writes into `TriggerContext.probe_scores`. Vector probes are bare concept names
(`@when:angry.calm > 0.4`), matching `TraitMonitor.score_single_token` keys.
Manifold subspace-fraction gates write the `:fraction` channel suffix
(`@when:pad:fraction > 0.5`) — the share of the centered activation living
in that manifold's subspace, in `[0, 1]`. Manifold label-similarity gates write
the `@<label>` suffix (`@when:pad@elated > -0.1`) — the negated distance to
a named node (larger = closer), so the natural threshold range is negative. All
three flow through one `ProbeGate` (full namespaced string stored verbatim,
matching the keys `ManifoldMonitor.flat_scalars` emits) and round-trip
byte-for-byte.

`%` is the manifold operator: `<manifold> % <position>` places a generation at a
point of a fitted manifold. `<position>` is `<coord_list>` (a comma-separated list
of authoring coordinates, one per intrinsic dimension, e.g.
`0.7 pad%0.3,0.8,0.0@response`) or `<label>` (sugar for "the coords of the node
labeled `pirate`", e.g. `0.5 persona%pirate`). The coefficient slot is
`along[,onto]` — `along` is the slide fraction toward the position, `onto` (curved
manifolds only) collapses the off-surface in-subspace residual. The former third
`toward` slot is removed. An affine `%` term (e.g. `personas%pirate`) joins the
merged subspace as a push fragment; a curved `%` term (e.g. `pad%…`) gets
its own injection term. A `%` term doesn't compose with `~`/`|`/`!`. Arity (coord
form) and label existence (label form) are validated at manifold-load time. Two
*curved* manifolds at one layer must be (near-)orthogonal or `OverlappingManifoldError`
raises; the merged affine subspace is always orthogonalized against the curved
spans. See "Manifold steering" below.

A bare slug (`pirate`) resolves through `io.selectors.resolve_bare_name`: first as
an installed bipolar pole, then as a manifold node label (synthesizing a
label-form `ManifoldTerm`, e.g. `local/persona%pirate`). Cross-tier ambiguity
raises `AmbiguousSelectorError`. Namespace-qualified or variant-suffixed forms
(`alice/pirate`, `pirate:role-x`, `civilian.pirate`) skip the manifold-label tier.

## Extraction

`extract_difference_of_means` (`core/vectors.py`) is the only vector extractor: the
per-layer direction is the mean of contrastive diffs (fp32), in plain or SAE
feature space. `core/extraction.py::ExtractionPipeline` orchestrates the cache-miss
path (statements → scenarios → pairs → extract → save tensor); a tensor cache hit
short-circuits upstream.

Share-baking folds the per-layer score into the tensor magnitude — `stored =
unit_direction × ref_norm × score / Σ scores` over the DLS-retained layers — so
sidecars carry no separate `scores` field and llama.cpp's uniform GGUF scalar
reproduces the per-layer weighting for free.

Discriminative Layer Selection (Selective Steering, Dang & Ngo 2026 Eq. 9): a
layer/axis is kept iff the pos/neg projections relative to neutral have opposite
signs — same-side layers encode concept *intensity*, not *polarity*. `--no-dls`
opts out. The N-node generalization (`compute_dls_axes`) backs the subspace fit.

Bake metric: `‖mean_diff‖_M / ref_norm` when a `LayerWhitener` covers every scored
layer (Mahalanobis, the session-driven default), `‖·‖₂ / ref_norm` otherwise. The
choice is all-or-nothing (`covers_all`) — the two scores live on different scales
(`‖·‖_M` carries a per-layer `1/√λ_L` factor that doesn't cancel from the
cross-layer-normalized share). The sidecar `bake` field records which. The
`LayerWhitener` (`core/mahalanobis.py`) is built lazily from cached neutral
activations and also drives Mahalanobis `~`/`|` projection (closed-form LEACE),
the whitened/Fisher subspace fit, the whitened monitor reads, and
`vector compare --metric mahalanobis`.

## Injection

One injection kernel — `core/manifold.py::subspace_inject` — for every steering
term. There is no angular/additive mode, no `injection_mode`/`theta_max`, no
`_STEER_GAIN`. At dispatch (`session._compose_steering_entries`) each term is
classified and lowered: vectors / poles / `~`/`|` projections / `!` ablations /
affine `%` are composed into **one merged affine subspace per trigger group** via
`synthesize_subspace`; each curved `%` gets its own term. `apply_to_model` then
attaches per-layer `subspace_inject` groups.

The kernel decomposes `h = mean + h_par + h_perp` against the layer's affine
subspace and applies two ops: **along** slides the in-subspace foot toward the
target (geodesically; on a curved surface it transports the off-surface residual
`H_n` to stay normal at the new foot), **onto** scales `H_n` by `(1 − o)` (vacuous
when the surface fills its subspace, i.e. for every flat/affine term). The
off-subspace residual `h_perp` is always kept verbatim — the old `toward` op that
scaled it is removed, which is what lets a vector and N orthogonal manifolds
compose with zero cross-talk. A flat (affine) subspace takes an analytic shortcut
(foot = the projected coord, no Gauss-Newton / RBF), load-bearing for throughput;
a curved manifold runs a warm-started per-token nearest-point foot-follower. fp32
throughout; the only norm guard is the soft cap `‖h_new‖ ≤ 3·‖h‖`.

Gain (`core/hooks.py`): the per-layer share is normalized to **mean 1** (`Σ_L
share_L = n_layers`) and `eff_along_L = share_L · _MANIFOLD_GAIN` (one constant,
`= 1.0`). For an affine term the coefficient α is folded into the slide *target*
by `synthesize_subspace`, so α scales *where* the slide lands and `share·base`
scales *how far*; for a curved term α is the (clamped `[0,1]`) `along` fraction.
There is **no lever / N correction** and **no `[0,1]` clamp / water-fill on
`along`** (a high-signal layer is meant to overshoot the target; the de-rogued
whitened coords keep it controlled and `norm_cap` bounds it). `onto` stays clamped
`[0,1]`. A steered layer always runs the slow (ctx-consulting) hook, so per-step
triggers and probe gates work uniformly; `torch.compile`/StaticCache graph capture
stays available only for unsteered generation.

## Manifold steering

Manifold steering (Goodfire, arXiv 2605.05115): instead of a single linear
direction, fit an interpolant through per-concept activation *centroids* in a
low-dim subspace, then steer by moving the running activation's in-subspace
component onto a point of that surface. A straight A→B vector cuts through
low-density off-manifold regions; the manifold stays on the learned surface.

Geometry is a `ManifoldDomain` — an embedding of an n-D intrinsic manifold into
R^m plus a distance function. `BoxDomain` (per-axis open or periodic) covers
boxes/disks, cylinders, n-tori; `SphereDomain` covers S^n (chordal); `CustomDomain`
is the explicit-immersion escape hatch (and the identity carrier for discover
coords and synthesized affine subspaces). The per-layer interpolant is one `r³`
polyharmonic RBF; at n=1 over an open axis it reproduces the natural cubic spline.

A manifold is its own artifact type — labeled nodes at authoring coordinates on a
domain, each a small statement corpus, under `~/.saklas/manifolds/<ns>/<name>/`
(not a `ConceptFolder`). Authored as `manifold.json` (domain spec + per-node
`{label, coords}`) + `nodes/*.json` — by hand or via the webui builder
(`io.manifolds.create_manifold_folder`). `manifold fit`, the webui fit
action, and `POST .../fit` all run `ManifoldExtractionPipeline`: pool each node's
centroid, embed coords through the domain, fit a per-layer subspace (flat
`fit_affine_subspace` for `fit_mode=pca`, curved `fit_layer_subspace` for
authored/spectral — whitened/Fisher PCA basis when the whitener covers all fit
layers, ordinary PCA otherwise), bake the per-layer Mahalanobis share, write the
per-model tensor. `--sae <release>` reconstructs each centroid through the SAE
before the fit; the fitted subspace is always model-space so the hook never
touches the SAE. `min_nodes(n) = 2n+1`; authored nodes must be *poised* (affinely
span the embedding).

Per-node `role` (slug `[a-z0-9._-]+`): the centroid is pooled under a
chat-template substitution that replaces the assistant-role label, so the fit
lives in role-baselined (persona) activation space. At steer time
`nearest_node_role` pipes the closest node's role through `session._active_role` so
the generation prefill applies the same substitution. Distinct implied roles
compose under soft-warn + highest-coefficient-wins; family-unsupported (Mistral-3,
talkie) raises `RoleSubstitutionUnsupportedError` at fit time.

**Whitened/Fisher subspace selection.** When the whitener covers every fit layer
the basis is selected by whitened PCA — maximize `vᵀS_b v / vᵀΣv` (the LDA
objective) rather than raw between-node variance, which on real LMs chases the
massive-activation (rogue) channels and leaves the subspace, its `mean`, and the
steering direction rogue-dominated. The Fisher ratio divides each direction by its
background variance, so rogue dims cancel (the same cancellation DiM gets by
differencing); the de-rogued subspace barely overlaps the rogue-dominated `mean`,
so the running-`‖h‖` artifact collapses for free. Solved as the generalized
eigenproblem via the whitener's Woodbury Σ⁻¹, re-expressed Euclidean-orthonormal so
the hot path is unchanged. Gated all-or-nothing on `covers_all`; absent coverage →
Euclidean PCA.

### Discover mode (auto-fit from a heap of corpora)

`manifold.json::fit_mode` is the discriminator: `"authored"` (user supplies
domain + per-node coords) vs *discover* — `"pca"`/`"spectral"`, where the user
supplies labeled corpora only and coordinates are derived per-model at fit time
from the pooled centroids. PCA picks the smallest prefix whose cumulative variance
crosses `var_threshold` (default 0.70), capped at `max_dim` (default 8); spectral
runs Laplacian eigenmaps on a symmetric k-NN graph and picks `k` by the
eigenvalue-ratio cliff. `fit_mode=pca` produces a flat affine subspace (no RBF);
`spectral` (and `authored`) produces a curved RBF surface. Per-model coordinates
are the architectural shift: a Gemma fit and a Qwen fit produce different node
layouts for the same heap (stored as `node_coords` in the per-model safetensors).

`manifold generate <name> --concepts ...` LLM-authors a discover folder via
`session.generate_statements` — one call for shared scenarios, then per-(scenario,
concept) cells. Scenario-sharing across the row is load-bearing (else per-concept
centroids mix concept and scenario signal); the anti-allegory (literal-concept)
directive keeps non-human axes literal. `manifold discover <name>` then
fits — the two steps are deliberate (a flaky generation leaves inspectable
corpora). Cross-model Procrustes alignment for discover coords is deferred (TODO
in `io/manifolds.py`). The naturalness eval (`experiment naturalness`) fits a
behavior-space manifold over node output distributions in Hellinger space and
reports the per-step Bhattacharyya distance of a steered trajectory to it (low =
natural; `--compare-linear` scores a straight-chord baseline alongside).

### Bundled manifolds + coefficient regime

Two bundled manifolds ship under `saklas/data/manifolds/`, materializing into
`~/.saklas/manifolds/default/` on session start via `materialize_bundled_manifolds()`
(process-scope no-op after the first call):

- **`personas`** — discover-mode `fit_mode=pca` (a flat ~rank-8 affine subspace),
  101 persona nodes (100 archetypes `assistant`…`vandal` + a `default` anchor) in
  assistant-baselined activation space; from Anthropic's Assistant Axis paper
  (arXiv 2601.10387). Per-model coords derived at fit time.
- **`pad`** — *authored*, curved 3-D box on the PAD (pleasure × arousal ×
  dominance) space, each axis `[−1, 1]`. 15 first-person mood corpora: the nine
  valence × arousal moods at dominance 0 (cardinal moods on the axes, diagonal
  moods at the unit-circle diagonals, `neutral` at the origin), the
  `dominant`/`submissive` dominance poles, and four dominance-differentiated
  affects (`furious`/`fearful`/`triumphant`/`humiliated`); 27 statements/node.
  Steer by coord (`pad%0.6,0.4,0.0` — three coords) or label (`pad%elated`).
  Supersedes the old 2-D `circumplex`.

Recommended α is vector-comparable: aim for `α ≈ 0.5`, tune up toward `α ≈ 1.0`
for stronger expression (α clamps to `[0,1]`, so `_MANIFOLD_GAIN = 1.0` is the
strength ceiling). Because the share is normalized to mean 1 and the lever is
gone, a low-dim and a high-dim fit — and `personas` vs `pad` — land in the
same α-band without per-fit retuning. Architecture-level behavioral notes (hold
across model families; α values are qualitative, MPS is not bitwise deterministic
so compare qualitatively):

- The whitened fit stays coherent at strong push where a Euclidean fit
  loop-collapses; caveman reaches terse primitive grammar, hacker a guarded
  intrusion register.
- Per-persona strength variance persists — a hard persona peaks near its coherence
  edge at α ≈ 1 where a robust one still has room; tune α down per target.
- Midpoints between distinct persona nodes (RBF interpolation at off-node coords)
  produce coherent *blended* persona content at the sweet-spot α — the
  interpolation-between-basins promise holds.
- The steering trajectory can pass through persona-adjacent attractor basins at low
  displacement (e.g. `personas%hacker` surfacing a cyber-security training cluster
  before locking into the clean persona). Low-α persona-drift is meaningful signal
  about the *model's* internal structure, not a saklas bug.

> **Open frontiers** (see `ARCHITECTURE.md` §10): the fitted `personas` subspace is
> a near-1-D "persona-ness" fan, so distinct personas can express the same generic
> intense register (a steering-access problem, not the rogue problem — whitening
> verifiably worked); `_MANIFOLD_GAIN`'s exact value is provisional and coupled to
> that; and converging vector concepts onto the 2-node-affine-manifold representation
> would close the last `vectors/`-vs-`manifolds/` seam.

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
- `generate` / `generate_stream` / `session.steering()` accept `str | Steering |
  None` only — dicts raise `TypeError`. A string is a steering expression.
- `generate`, `generate_batch`, `generate_sweep` always return `RunSet` — list-like,
  carrying `node_ids`/`grid`, with `.first` (the underlying `GenerationResult`) and
  common attributes delegating to it. `session.last_result` is the `GenerationResult`.
- `extract()` returns `(name, Profile)`. `Profile` wraps `dict[int, Tensor]`
  (mapping interface plus `layers`, `metadata`, `save`/`load`, `merged`,
  `projected_away`, `cosine_similarity`).
- `Steering` is frozen; its only per-call override field is `projection_metric`
  (`None` = inherit the session default). There is no `injection_mode`/`theta_max`.
- `SaklasSession.__init__` takes a pre-loaded `PreTrainedModel`; use `from_pretrained`
  for HF loads. There is no `cache_dir=` — set `$SAKLAS_HOME` to relocate paths.
- Every saklas exception subclasses `SaklasError` while preserving its stdlib MRO,
  so `except SaklasError` catches the family and `except ValueError`/`RuntimeError`
  at existing sites still works.
- `GenerationResult.applied_steering` carries the canonical expression string
  (round-trips through `parse_expr`).
- `saklas/__init__.py` pins the public surface (`SaklasSession`, `Profile`,
  `Steering`, `SamplingConfig`, `Trigger`, `LayerWhitener`, the `RunSet`/
  `TokenEvent`/`ResultCollector` result types, the `EventBus` + event dataclasses,
  the `LoomTree`/`Recipe`/`Transcript` suites, `DataSource`, and their error
  types). `from saklas import X` is stable; private submodule paths are not.

## Cache layout

All state under `~/.saklas/` (override via `$SAKLAS_HOME`):

```
~/.saklas/
  neutral_statements.json              # user-editable (copy-on-miss from package)
  vectors/
    default/<concept>/                 # bundled (source=bundled)
      pack.json                        # name, description, tags, files{sha256}
      scenarios.json  statements.json
      <safe_model_id>.safetensors      # baked DiM tensor (+ .json slim sidecar)
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
    nodes/NN_<label>.json              # one JSON statement list per node (user-editable)
    scenarios.json                     # discover-only: provenance of `generate`
    <safe_model_id>.safetensors        # fitted per-layer subspaces; discover variant
                                       # also carries node_coords (the derived layout)
                                       # (+ .json sidecar with fit_mode/hyperparams/diagnostics)
  conversations/<name>.json            # explicit loom-tree saves (no autosave)
```

`pack.json.files` / `manifold.json.files` are sha256 maps verified on load. A
concept folder can hold multiple baked tensors per model, distinguished by
filename suffix: `<safe>.safetensors` (canonical DiM), `_sae-<release>`,
`_from-<safe_src>` (transfer), `_role-<slug>` — at most one kind per file (no `pca`
suffix). `tensor_filename` / `parse_tensor_filename` in `io/paths.py` round-trip
them. Safetensors win over GGUF on a same-stem conflict. `materialize_bundled` is
copy-on-miss but auto-upgrades bundled concepts in place on a stale `format_version`.

## Performance invariants

These gate `test_smoke.py::test_throughput_regression` (steered ≥ 85% of vanilla
tok/s):

- **Hot-path hooks**: no Python allocation, no `.item()`, no CPU sync, in-place
  only. The whole generation loop is wrapped in `torch.inference_mode()`.
- **Norms use fp32** — fp16 sum-of-squares overflows at hidden_dim ≥ 2048. Applies
  to extraction-time direction norms and the per-position norms inside
  `subspace_inject`. Contrastive diffs are differenced in fp32 too.
- **One injection kernel.** Every steered layer runs `subspace_inject` (the slow,
  ctx-consulting hook path), so per-step triggers and probe gates work uniformly;
  there is no composed-tensor fast path. `torch.compile`/StaticCache graph capture
  is eligible only for unsteered generation. The affine analytic shortcut keeps the
  common case (folded vectors / flat subspaces) cheap; curved manifolds pay a
  warm-started O(R) per-token foot solve.
- **Share baked at extraction / fit**, normalized to mean 1 at apply; the
  subspace foot slides by `share_L · _MANIFOLD_GAIN` toward a target that already
  carries the coefficient. No norm preservation (onto is meant to shrink `‖h‖`);
  `norm_cap = 3·‖h‖` is the only bound.
- **Top-p via `torch.topk`**, not full-vocab sort; `top_k` (default 1024 cap) is a
  hard candidate-pool cap applied before top-p (llama.cpp/Ollama order).
- **Monitor capture is hook-driven**, inline with generation — one matmul per layer
  scores all probes, no second forward pass. `TraitMonitor` scores in the whitened
  (Mahalanobis) cosine when the whitener covers every probed layer, else Euclidean
  for all (all-or-nothing via `covers_all`; never a per-layer mix). The Σ⁻¹-applied
  probe directions + device-resident Woodbury factors are precomputed at cache
  build, so the hot path stays one matmul plus a cheap per-token Woodbury apply.
  `covers_all` is trustworthy as "finite factors everywhere": any non-finite layer
  is excluded, and neutral activations are cached fp32 (so gemma-3's late layers
  don't overflow the fp16 65504 ceiling to ±inf). The probe weight stays `‖baked‖₂`
  (the bake already folded the Mahalanobis score in).
- **Steering hooks are transient** — composed before generation, removed after.
- **MPS discipline** — diffs on CPU, `torch.mps.empty_cache()` between extraction
  passes, end-of-loop sync to dodge Metal command-buffer reuse crashes.

## Tested architectures

`_TESTED_ARCHS` in `core/model.py` emits a one-time `UserWarning` on load when
`model_type` isn't in the set. Known working: `qwen2`, `qwen3`, `qwen3_5`
(+ `_text`/`_moe`), `gemma2`, `gemma3` (+ `_text`), `gemma4` (+ `_text`),
`mistral3`, `ministral3`, `gpt_oss`, `llama`, `glm`, `talkie`. Many more are wired
via `_LAYER_ACCESSORS` but untested — adding one is a single accessor entry.
Architectures whose modeling ignores `past_key_values` (e.g. the original talkie
port) auto-fall back to O(N²) no-KV-cache generation with a one-time warning.

Role-augmented extraction (`:role-<name>` variant) and persona manifolds need a
chat template with a substitutable assistant-role label (`core/role_templates.py::
ROLE_HEADERS`). Supported: `qwen2`/`qwen3`/`qwen3_5` (ChatML), `gemma2`/`gemma3`/
`gemma4` (`<start_of_turn>`, label is `model`), `llama`, `glm`, `gpt_oss`.
Unsupported (mapped to `None`, `apply_with_role` raises
`RoleSubstitutionUnsupportedError`): `mistral3` / `ministral3` (positional
`[INST]`, no role label in the rendered string), `talkie` (opted out).

## Bundled concepts

26 curated concepts at n=45 pairs each (9 scenarios × 5 pairs), in
`saklas/data/vectors/<concept>/` — 24 bipolar + 2 monopolar (`agentic`,
`manipulative`). The authoritative manifest is
`scripts/regenerate_bundled_statements.py`.

Categories: `affect` (angry.calm, happy.sad, fearful.unflinching), `epistemic`
(confident.uncertain, honest.deceptive, hallucinating.grounded,
curious.disinterested), `alignment` (refusal.compliant, sycophantic.blunt,
agentic, manipulative), `register` (formal.casual, direct.indirect,
verbose.concise, creative.conventional, humorous.serious, warm.clinical,
technical.accessible), `social_stance` (authoritative.submissive,
high_context.low_context, self.other), `cultural` (masculine.feminine,
religious.secular, traditional.progressive, individualist.collectivist),
`identity` (ai.human).

Known model-level axis entanglements (cross-model robust, weighted cosine via
`vector compare`) — document for users, not probe-design failures:
- `masculine.feminine ↔ traditional.progressive` (+0.5–0.6) — Hofstede MAS read as
  traditionalism
- `hallucinating.grounded ↔ humorous.serious` (+0.5–0.7) — humor reads as
  off-grounded weirdness
- `angry.calm ↔ authoritative.submissive` (+0.5–0.8) — anger encodes as dominance

90 neutral statements in `saklas/data/neutral_statements.json` follow the same
affect-neutral, topically-diverse discipline. Bundled-pair regeneration runs
through `scripts/regenerate_bundled_statements.py`, which calls
`session.generate_scenarios` / `session.generate_statements` — DiM pairs are the
two per-concept corpora zipped pos/neg (the moment-paired generation path was
removed; the centroid difference is identical either way). A load-bearing
anti-allegory clause keeps non-human axes (`deer/wolf`, `brick/feather`) literal.

## Package layout

`saklas/{core,io,cli,server,tui,web}/`. `core` is the engine, `io` is persistence
+ distribution, `cli`/`server`/`tui`/`web` are the four frontends. The Svelte
dashboard source lives at the repo's `webui/` directory (peer of `saklas/`); its
build artifact is committed under `saklas/web/dist/`.

## Testing

**GPU-required** (CUDA or MPS): `test_smoke.py`, `test_session.py` — download
`google/gemma-3-4b-it` (~8GB) on first run. `device="auto"` picks cuda > mps >
cpu; MPS runs ~3–5× slower so extraction budgets are backend-specific. `test_smoke`
owns the throughput regression.

**CPU-only**: the bulk of the suite — core dataclasses, steering-context
semantics, pack/manifold format integrity + staleness, selector grammar, mocked HF
wrappers, GGUF round-trip, config loading, monitor scoring, six-verb CLI dispatch,
OpenAI/Ollama/native servers, TUI slash-command dispatch, loom tree/diff/filter/
transcript.
