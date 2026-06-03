# AGENTS.md

## What this is

`saklas` is a Python library + Textual TUI + dual-protocol HTTP server for
activation steering and trait monitoring on HuggingFace causal LMs. It runs
OpenAI `/v1/*` and Ollama `/api/*` on one port, plus a native `/saklas/v1/*` API
and a Svelte dashboard at `/`. Steering signal comes from representation
engineering, unified under a single artifact family — the **manifold**: labeled
nodes placed on a domain, fit to a per-layer subspace. A difference-of-means
steering vector is the 2-node flat case; `personas` is a 101-node flat fan; `pad`
is a curved 3-D box. Every steering term — vectors, poles, `~`/`|` projections,
`!` ablations, and `%` manifold positions — lowers at generation time to one
unified per-layer injection (the along/onto subspace kernel,
`core/manifold.py::subspace_inject`). Per-call coefficients, no model mutation.
Three frontends over one engine: `SaklasSession` (programmatic), `saklas serve`
(HTTP), `saklas tui` (TUI).

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

- `saklas/core/AGENTS.md` — model loading, the manifold/subspace fit + injection,
  monitor, session, generation loop, loom tree
- `saklas/io/AGENTS.md` — manifold format, HF distribution, GGUF, merge, cloning,
  alignment, paths/selectors
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
saklas tui <model_id> [--no-dls]
saklas serve <model_id> [--no-web] [--steer/-S EXPR]
saklas subspace extract <concept>|<pos> <neg> [-m MODEL] [--sae RELEASE] [--role SLUG] [--namespace NS] [-f]
saklas subspace merge <name> <expression> [-m]    # shared grammar: "0.3 ns/a + 0.5 ns/b|ns/c"
saklas subspace compare <concepts...> -m MODEL [--ridge-scale R]
saklas subspace why <concept> -m MODEL [-j]       # per-layer ||baked|| as a 16-bucket histogram
saklas subspace transfer <concept> --from SRC --to TGT [-f]   # cross-model Procrustes transfer
saklas manifold fit <folder> [-m MODEL] [--sae REL]  # fit an authored manifold
saklas manifold discover <name> [-m MODEL] [--method pca|spectral] [--max-dim N] ...
saklas manifold generate <name> --concepts C... [--kind abstract|concrete] [--samples-per-prompt K] [--seed S]
saklas manifold install <target> [-a NS/N] [-f]      # HF coord or local folder (also ports legacy saklas-packs)
saklas manifold search <query> [-j|-v]               # search HF hub for saklas-manifold repos
saklas manifold merge <name> <src...> [-f]           # union discover-mode node corpora
saklas manifold push <name> [-a OWNER/N] [-m MODEL] [--variant raw|sae|all]
saklas manifold transfer <name> --from SRC --to TGT [-f]   # cross-model Procrustes
saklas manifold clear <name> [-m MODEL] [--variant raw|sae|all]   # delete per-model fitted tensors
saklas manifold rm <name> [-y]                       # remove folder (bundled respawns)
saklas manifold refresh <name> [-m MODEL]            # re-pull (hf) / re-fit (-m scoped)
saklas manifold export gguf <name> [-m MODEL] [-o PATH] [--model-hint HINT]   # fold a 2-node pca manifold to a control-vector GGUF
saklas manifold ls [-v|-j] | show <name> [-j]        # list / inspect manifolds
saklas experiment fan <model> "<prompt>" -g concept=0,0.5,1 # alpha grid as loom siblings
saklas experiment transcript run <path.yaml> [model]        # replay a saved transcript
saklas experiment naturalness <model> "<prompt>" --manifold F -S EXPR  # behavior-manifold eval
saklas config show [-c PATH ...] [--no-default] [-m MODEL]
saklas config validate <file>
pytest tests/                                   # all; GPU tests gated on CUDA/MPS
```

The root parser has exactly six verbs: `tui`, `serve`, `subspace`, `manifold`,
`experiment`, `config`. There is no `pack` verb and no `vector` alias — pack
distribution folded into the manifold artifact, so install via `manifold install`
and export via `manifold export gguf`. `subspace` is the flat-artifact
computation surface (extract/merge/compare/why/transfer); `manifold` owns
the full steering-manifold tree. No `argv[0]` peeking, no bare-TUI fallback —
`saklas google/gemma-2-2b-it` is an argparse error. Bare `saklas` / `saklas
subspace` / `saklas manifold` / `saklas experiment` / `saklas config` print help
and exit 0.

Every subcommand that takes `-c/--config` auto-loads `~/.saklas/config.yaml`
first, then composes explicit `-c` files on top (later overrides earlier). The
`vectors:` YAML key is a single steering expression parsed by
`saklas.core.steering_expr.parse_expr`. `cli/AGENTS.md` has the full per-verb flag
set. There is no `--steer-mode`/`--theta-max`/`--method`/`--legacy`/
`--projection-metric` surface — the injection kernel is unified, difference-of-means
is the only vector extraction method, `~`/`|` projection is Mahalanobis-only, and
the only steering knob at the CLI is `--no-dls`.

## Selector grammar

Shared across surfaces: `<name>`, `<ns>/<name>`, `tag:<t>`, `namespace:<ns>`,
`model:<m>`, `default`, `all`, optionally suffixed `:<variant>` where `<variant>`
is `raw` (canonical DiM), `sae`, `sae-<release>`, `from-<safe_src>` (cross-model
transfer), or `role-<name>` (role-augmented extraction — the contrast pairs were
generated under a chat template whose assistant-role label was substituted, and
the same substitution is auto-applied at steering time so extract baseline equals
steer baseline). There is no `pca` variant. Bare names resolve cross-namespace and
raise `AmbiguousSelectorError` on collision. Concepts *are* manifolds now
(`io.selectors` walks `manifolds_dir()`): a bare pole resolves through the
manifold tier — `resolve_bare_name` tries the bipolar-pole node first
(`resolve_manifold_label`/`resolve_manifold_name` over the installed 2-node `pca`
manifolds), then a multi-node manifold node label, raising on cross-tier
collision. A steering expression composing role-augmented terms must agree on
role; plain `:raw` terms compose with role terms but emit a one-time
`RoleBaselineMismatchWarning`.

Canonical naming: `canonical_concept_name` (module-level in `core/session.py`)
slugs poles via `[^a-z0-9]+ → _` and joins bipolar poles with `BIPOLAR_SEP = "."`,
so `/steer happy . sad` and `/steer happy.sad` resolve to the same manifold.
`NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$`; `@` is forbidden (HF revision separator),
and `.` is used over `~` because HF repo names reject `~`.

## Steering expression grammar

Every input surface — Python, YAML, HTTP, TUI, `subspace merge` — speaks the
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
labeled `pirate`", e.g. `0.5 personas%pirate`). The coefficient slot is
`along[,onto]` — `along` is the slide fraction toward the position, `onto` (curved
manifolds only) collapses the off-surface in-subspace residual. An affine `%` term
(flat manifold, e.g. `personas%pirate`) joins the merged subspace as a push
fragment; a curved `%` term (e.g. `pad%…`) gets its own injection term. A `%`
term doesn't compose with `~`/`|`/`!`. Arity (coord form) and label existence
(label form) are validated at manifold-load time. Two *curved* manifolds at one
layer must be (near-)orthogonal or `OverlappingManifoldError` raises; the merged
affine subspace is always orthogonalized against the curved spans. See "Manifold
steering" below.

A bare slug (`pirate`) resolves through `io.selectors.resolve_bare_name`: first as
a bipolar-pole node of a 2-node `pca` manifold, then as a multi-node manifold node
label (synthesizing a label-form `ManifoldTerm`, e.g. `local/personas%pirate`).
Cross-tier ambiguity raises `AmbiguousSelectorError`. Namespace-qualified or
variant-suffixed forms (`alice/pirate`, `pirate:role-x`, `civilian.pirate`) skip
the manifold-label tier.

## Extraction

There is **one** extraction pipeline. A steering vector is the K=2 flat case of a
manifold, so `session.extract(concept, baseline, *, kind="abstract")` authors a
2-node discover-`pca` manifold under `manifolds/<ns>/<name>/` — node 0 the positive
pole, node 1 the negative — and fits it via
`core/extraction.py::ManifoldExtractionPipeline`. There is no separate
`{positive, negative}` / `statements.json` / baked-DiM artifact; the corpus lives
as the manifold's two node groups. `extract()` returns `(canonical_name, Profile)`
where the `Profile` is the **folded** per-layer direction view of the fitted 2-node
manifold (`core/vectors.py::folded_vector_directions`) — the in-memory
steering-vector shape callers expect, with the manifold the single on-disk
artifact. A tensor cache hit (sidecar `nodes_sha256` matches the folder)
short-circuits the forward passes.

**Conversational corpora (4.0 / A2).** Each node's corpus is generated by
`session.generate_responses`: the model answers a fixed set of shared baseline
prompts *in character* — the concept rides a system prompt (by `kind`:
abstract → "someone {c}", concrete → "{art} {c}") and a swapped assistant-role
elicitation label — and extraction pools the swapped-back `[user: prompt,
assistant: response]` pairs in **standard-assistant space**. The corpus is a
`list[str]` of responses aligned `response[i] ↔ baseline_prompt[i % k]`
(`saklas/data/baseline_prompts.json`, 64 prompts; length must be a multiple of
`k`). `kind` is recorded per node (generation-time provenance; the fit never
consumes it). `--role`/`role=` opts into a persona-baselined fit (the explicit
role overrides the kind-derived label at both generation and capture). A
**monopolar** concept (`baseline=None`) authors a genuinely **1-node** `pca`
folder. The pipeline recognizes the single-node shape (a flat `pca` fit otherwise
needs `k+1 ≥ 2` poised nodes) and folds `concept − ν` — ν = the model's neutral
activation mean (`layer_means`) — into a 1-node neutral-anchored ray via
`fold_directions_to_subspace` (the same primitive `subspace merge` uses): no
discover-coords, no DLS, no synthetic second node. Neutral is the **implicit**
negative pole, sourced per-model at fit (never a stored corpus); `concept − ν`
already cancels common-mode like DiM, so the raw δ̂ basis is appropriate.
`0.5 <concept>` steers neutral → concept, resolving through the bare-label tier
exactly like a bipolar pole. The fit needs ν available (`layer_means`); when the
neutral corpus is stale it raises the same prerequisite the whitener does.

The per-layer fit derives the basis by **μ-centered PCA**; at K=2 the sole axis is
exactly the unit difference-of-means `δ̂` (PCA@2 ≡ DiM). The fitted manifold stores
each layer's `LayerSubspace` (mean, basis, real `node_coords`): the activation-space
magnitude lives in the neutral-anchored real coords (`coord_pos − coord_neg =
‖δ_L‖`), and a separate per-layer Mahalanobis `share` carries the cross-layer
budget. The in-memory steering vector is the folded view `δ̂_L · share_L`
(`folded_vector_directions`); for GGUF export llama.cpp's uniform control-vector
scalar reproduces the relative per-layer weighting from those magnitudes.

Discriminative Layer Selection (Selective Steering, Dang & Ngo 2026 Eq. 9): a
layer/axis is kept iff the node projections relative to neutral straddle zero
(both signs present) — same-side layers encode concept *intensity*, not
*polarity*. `--no-dls` opts out. `compute_dls_axes` (N-node straddle) backs the
flat fit; at K=2 it is exactly the pos/neg opposite-sign test. Curved manifolds
have no pos/neg polarity, so they skip DLS — per-layer signal is the apply-time
share alone.

Bake/share metric: always `‖·‖_M` (Mahalanobis). Activation-space fit, `~`/`|`
projection, monitor reads, cross-model transfer, and `subspace compare` require a
`LayerWhitener` covering every scored layer (`covers_all`) and raise `WhitenerError`
otherwise — there is no Euclidean fallback (on real LMs the Euclidean metric is
rogue-dominated, so it would be a wrong answer, not a degraded one). The sidecar
`share_metric` / `subspace_metric` fields are provenance and read `mahalanobis`,
the one exception being a monopolar fit's `subspace_metric`, which labels its
raw-δ̂ basis `euclidean` (a basis-selection label — `concept − ν` cancels
common-mode by differencing, like DiM — not a metric fallback). The `LayerWhitener`
(`core/mahalanobis.py`) is built lazily from cached neutral activations and drives
the closed-form LEACE `~`/`|` projection, the whitened/Fisher subspace fit, the
whitened monitor reads, and `subspace compare`.

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
off-subspace residual `h_perp` is always kept verbatim, which is what lets a
vector and N orthogonal manifolds compose with zero cross-talk. A flat (affine)
subspace takes an analytic shortcut (foot = the projected coord, no Gauss-Newton /
RBF), load-bearing for throughput; a curved manifold runs a warm-started per-token
nearest-point foot-follower. fp32 throughout; the only norm guard is the soft cap
`‖h_new‖ ≤ 3·‖h‖`.

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

A manifold lives under `~/.saklas/manifolds/<ns>/<name>/` as `manifold.json`
(domain spec + per-node `{label, coords}` for authored; `fit_mode` + hyperparams +
`{label}` for discover) + `nodes/NN_<label>.json` corpora — by hand or via the
webui builder (`io.manifolds.create_manifold_folder` /
`create_discover_manifold_folder`). `manifold fit`, the webui fit action, and
`POST .../fit` all run `ManifoldExtractionPipeline`: pool each node's centroid,
embed coords through the domain (or derive them for discover mode), fit a per-layer
subspace (flat `fit_affine_subspace` for `fit_mode=pca`, curved
`fit_layer_subspace` for authored/spectral — whitened/Fisher PCA basis when the
whitener covers all fit layers, ordinary PCA otherwise), bake the per-layer
Mahalanobis share, write the per-model tensor. `--sae <release>` reconstructs each
centroid through the SAE before the fit; the fitted subspace is always model-space
so the hook never touches the SAE. `min_nodes(n) = 2n+1` for a curved fit (a flat
`pca` fit needs only `k+1`); authored nodes must be *poised* (affinely span the
embedding).

`fit_mode` is one of four: `authored` (user supplies domain + coords; curved),
`pca` / `spectral` (discover — labeled corpora only, coords derived per-model;
`pca` is flat, `spectral` curved), and `baked` (corpus-less, a precomputed
direction written by `subspace merge` — see io/AGENTS.md).

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
the hot path is unchanged. Gated all-or-nothing on `covers_all`: an activation-space
fit requires the whitener to cover every fit layer and raises `WhitenerError`
otherwise — no Euclidean fit path. (The low-level `_pca_basis` keeps a Euclidean SVD
branch for the behavior-space naturalness fit, which lives in output-distribution
space where there are no rogue activation dims; the activation-space callers never
reach it.)

### Discover mode (auto-fit from a heap of corpora)

`manifold.json::fit_mode` is the discriminator: `"authored"` (user supplies
domain + per-node coords) vs *discover* — `"pca"`/`"spectral"`, where the user
supplies labeled corpora only and coordinates are derived per-model at fit time
from the pooled centroids. PCA picks the smallest prefix whose cumulative variance
crosses `var_threshold` (default 0.70), capped at `max_dim` (default 8); spectral
runs Laplacian eigenmaps on a symmetric k-NN graph and picks `k` by the
eigenvalue-ratio cliff. `fit_mode=pca` produces a flat affine subspace (no RBF);
`spectral` (and `authored`) produces a curved RBF surface. `anchor_origin` (pca
only) shifts every coord so a named node lands at the origin. Per-model coordinates
are the architectural shift: a Gemma fit and a Qwen fit produce different node
layouts for the same heap (stored as `node_coords` in the per-model safetensors).

`manifold generate <name> --concepts ... [--kind abstract|concrete]` LLM-authors a
discover folder via `session.generate_responses` — each concept answers the shared
baseline prompts in character (one corpus per node). The shared baseline prompts
hold topic common-mode across nodes (response[i] ↔ prompt[i % k]), so the
per-concept centroids stay comparable without a per-manifold scenario set.
`manifold discover <name>` then fits — the two steps are deliberate (a flaky
generation leaves inspectable corpora). Cross-model Procrustes alignment for discover coords is deferred (TODO
in `io/manifolds.py`). The naturalness eval (`experiment naturalness`) fits a
behavior-space manifold over node output distributions in Hellinger space and
reports the per-step Bhattacharyya distance of a steered trajectory to it (low =
natural; `--compare-linear` scores a straight-chord baseline alongside).

### Bundled manifolds + coefficient regime

The bundled artifacts ship under `saklas/data/manifolds/`, materializing into
`~/.saklas/manifolds/default/` on session start via `materialize_bundled_manifolds()`
(process-scope no-op after the first call):

- **26 concept manifolds** — 2-node `fit_mode=pca` axes (24 bipolar + 2 monopolar
  `agentic`/`manipulative`), tagged by category (`affect`, `epistemic`,
  `alignment`, `register`, `social_stance`, `cultural`, `identity`). These are the
  steering vectors: `0.5 angry.calm` steers toward `angry` (node 0).
- **`personas`** — discover `fit_mode=pca` (a flat ~rank-8 affine subspace),
  101 persona nodes (100 archetypes `assistant`…`vandal` + a `default` anchor) in
  assistant-baselined activation space; from Anthropic's Assistant Axis paper
  (arXiv 2601.10387). `anchor_origin` lands `default` at the origin.
- **`cultural`** (9-node) / **`register`** (15-node) — discover `fit_mode=pca`
  multi-axis subspaces grouping the cultural / register concept families into one
  flat subspace each (`anchor_origin`, `max_subspace_dim` 4 / 6).
- **`pad`** — *authored*, curved 3-D box on the PAD (pleasure × arousal ×
  dominance) space, each axis `[−1, 1]`. 15 first-person mood corpora. Steer by
  coord (`pad%0.6,0.4,0.0`) or label (`pad%elated`).

Recommended α is vector-comparable: aim for `α ≈ 0.5`, tune up toward `α ≈ 1.0`
for stronger expression (α clamps to `[0,1]`, so `_MANIFOLD_GAIN = 1.0` is the
strength ceiling). Because the share is normalized to mean 1 and the lever is
gone, a low-dim and a high-dim fit — a 2-node vector and `personas` and `pad` —
land in the same α-band without per-fit retuning. Architecture-level behavioral
notes (hold across model families; α values are qualitative, MPS is not bitwise
deterministic so compare qualitatively):

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
> verifiably worked), and `_MANIFOLD_GAIN`'s exact value is provisional and coupled
> to that.

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
    with session.steering("0.5 wolf"):                 # bare pole → deer.wolf node 1
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
- `extract()` returns `(name, Profile)`; the `Profile` is the folded view of the
  2-node manifold. `extract_vector_from_corpora` is the corpus-in sibling;
  `extract_manifold(folder)` fits a multi-node manifold and returns a `Manifold`.
  `Profile` wraps `dict[int, Tensor]` (mapping interface plus `layers`, `metadata`,
  `save`/`load`, `merged`, `projected_away`, `cosine_similarity`).
- `Steering` is frozen; it carries no per-call metric override (`~`/`|` projection
  is Mahalanobis-only). There is no `injection_mode`/`theta_max`/`projection_metric`.
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
  the `LoomTree`/`Recipe`/`Transcript` suites, and their error types). `from saklas
  import X` is stable; private submodule paths are not.

## Cache layout

All state under `~/.saklas/` (override via `$SAKLAS_HOME`):

```
~/.saklas/
  neutral_statements.json              # user-editable; organic responses to the
                                       # baseline prompts (copy-on-miss from package)
  baseline_prompts.json                # user override for the shared A2 prompts
  manifolds/<ns>/<name>/               # THE concept + steering-manifold root
    manifold.json                      # name, source, fit_mode, per-node {label,kind,role?}, domain/coords or hyperparams, files{sha256}
    nodes/NN_<label>.json              # one JSON response list per node (response[i] ↔ baseline_prompt[i % k])
    <safe_model_id>.safetensors        # fitted per-layer subspaces (+ .json sidecar);
                                       # discover/baked also carry node_coords (the layout)
    <safe_model_id>_sae-<rel>.safetensors    # SAE-space fit
    <safe_model_id>_from-<src>.safetensors   # cross-model transfer
    <safe_model_id>_role-<slug>.safetensors  # role-augmented
  models/<safe_model_id>/
    layer_means.{safetensors,json}     # probe-centering baseline
    neutral_activations.{safetensors,json}   # 90 neutral prompts × layers, fp32
    alignments/<safe_src>.{safetensors,json} # optional cross-model Procrustes map
  vectors/<ns>/<concept>/              # LEGACY (pre-4.0) packs only — ported to
                                       # manifolds/ on first touch; no longer written
  conversations/<name>.json            # explicit loom-tree saves (no autosave)
```

`manifold.json.files` is a sha256 map verified on load. A manifold folder can hold
multiple fitted tensors per model, distinguished by filename suffix:
`<safe>.safetensors` (raw DiM), `_sae-<release>`, `_from-<safe_src>` (transfer),
`_role-<slug>` — at most one kind per file (no `pca` suffix). `tensor_filename` /
`parse_tensor_filename` in `io/paths.py` round-trip them.
`materialize_bundled_manifolds()` is copy-on-miss. A pre-4.0 `vectors/` pack
(`pack.json.format_version < PACK_FORMAT_VERSION = 3`) is *legacy*: ported to a
2-node `pca` manifold on first steer touch (`_port_stale_legacy_vector` /
`scripts/upgrade_packs.py`) and re-fit lazily.

## Performance invariants

These gate `test_smoke.py::test_throughput_regression` (steered ≥ 85% of vanilla
tok/s):

- **Hot-path hooks**: no Python allocation, no `.item()`, no CPU sync, in-place
  only. The whole generation loop is wrapped in `torch.inference_mode()`.
- **Norms use fp32** — fp16 sum-of-squares overflows at hidden_dim ≥ 2048. Applies
  to fit-time direction norms and the per-position norms inside `subspace_inject`.
  Centroid differences are taken in fp32 too.
- **One injection kernel.** Every steered layer runs `subspace_inject` (the slow,
  ctx-consulting hook path), so per-step triggers and probe gates work uniformly;
  there is no composed-tensor fast path. `torch.compile`/StaticCache graph capture
  is eligible only for unsteered generation. The affine analytic shortcut keeps the
  common case (folded vectors / flat subspaces) cheap; curved manifolds pay a
  warm-started O(R) per-token foot solve.
- **Share baked at fit**, normalized to mean 1 at apply; the subspace foot slides
  by `share_L · _MANIFOLD_GAIN` toward a target that already carries the
  coefficient. No norm preservation (onto is meant to shrink `‖h‖`); `norm_cap =
  3·‖h‖` is the only bound.
- **Top-p via `torch.topk`**, not full-vocab sort; `top_k` (default 1024 cap) is a
  hard candidate-pool cap applied before top-p (llama.cpp/Ollama order).
- **Monitor capture is hook-driven**, inline with generation — one matmul per layer
  scores all probes, no second forward pass. `TraitMonitor` scores in the whitened
  (Mahalanobis) cosine — mandatory: the whitener must cover every probed layer
  (`covers_all`), else scoring raises `WhitenerError` (no Euclidean path). The
  Σ⁻¹-applied probe directions + device-resident Woodbury factors are precomputed at
  cache build, so the hot path stays one matmul plus a cheap per-token Woodbury apply.
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

26 curated concepts under `saklas/data/manifolds/<concept>/` — 24 bipolar
(2-node `pca`) + 2 monopolar (`agentic`, `manipulative`). Under 4.0 / A2 each
pole's corpus is conversational responses to the shared baseline prompts.
Monopolar `extract` is now implemented as a genuine **1-node** fold against the
neutral mean ν (see "Extraction"), so a fresh `extract("agentic")` authors a
1-node ray. The *bundled* `agentic`/`manipulative` still ship with their pre-4.0
explicit 2-node `_neg` corpora; re-authoring them as 1-node + A2 is part of the
bundled-regeneration TODO. The pre-4.0 bundled statement-regeneration script is
removed; its manifest is recoverable from git history and a 4.0 bipolar-concept
regeneration script is a TODO (the persona + cultural/register discover scripts
are already on A2).

Categories: `affect` (angry.calm, happy.sad, fearful.unflinching), `epistemic`
(confident.uncertain, honest.deceptive, hallucinating.grounded,
curious.disinterested), `alignment` (refusal.compliant, sycophantic.blunt,
agentic, manipulative), `register` (formal.casual, direct.indirect,
verbose.concise, creative.conventional, humorous.serious, warm.clinical,
technical.accessible), `social_stance` (authoritative.submissive,
high_context.low_context, self.other), `cultural` (masculine.feminine,
religious.secular, traditional.progressive, individualist.collectivist),
`identity` (ai.human). The `cultural` / `register` families also ship as
multi-node discover manifolds of the same name.

Known model-level axis entanglements (cross-model robust, weighted cosine via
`subspace compare`) — document for users, not probe-design failures:
- `masculine.feminine ↔ traditional.progressive` (+0.5–0.6) — Hofstede MAS read as
  traditionalism
- `hallucinating.grounded ↔ humorous.serious` (+0.5–0.7) — humor reads as
  off-grounded weirdness
- `angry.calm ↔ authoritative.submissive` (+0.5–0.8) — anger encodes as dominance

`saklas/data/neutral_statements.json` holds the neutral baseline as organic,
no-system/no-role responses to the same shared baseline prompts (a multiple of the
64-prompt set), regenerated via `session.generate_neutral_responses`; it backs the
probe-centering means + Mahalanobis whitener. `saklas/data/baseline_prompts.json`
(64 affect-neutral, topically-diverse prompts) is the shared elicitation set every
node and the neutral corpus answer.

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
semantics, manifold format integrity + staleness, selector grammar, mocked HF
wrappers, GGUF round-trip, config loading, monitor scoring, six-verb CLI dispatch,
OpenAI/Ollama/native servers, TUI slash-command dispatch, loom tree/diff/filter/
transcript.
