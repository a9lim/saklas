# AGENTS.md

## What this is

`saklas` is a Python library + Textual TUI + dual-protocol HTTP server for
activation steering and trait monitoring on HuggingFace causal LMs. It runs
OpenAI `/v1/*` and Ollama `/api/*` on one port, plus a native `/saklas/v1/*` API
and a Svelte dashboard at `/`. Steering signal comes from representation
engineering, unified under a single artifact family — the **manifold**: labeled
nodes placed on a domain, fit to a per-layer subspace. A difference-of-means
steering vector is the 2-node flat case; `personas` is a 107-node flat fan; `emotions`
is a 20-node affect manifold over PAD. Every steering term — vectors, poles, `~`/`|` projections,
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
- `saklas/io/AGENTS.md` — manifold format, HF distribution, GGUF, merge,
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
saklas manifold extract <concept>|<pos> <neg> [-m MODEL] [--sae RELEASE] [--role SLUG] [--namespace NS] [-f]
saklas manifold generate <name> --concepts C... [--kind abstract|concrete] [--samples-per-prompt K] [--seed S]
saklas manifold from-template <template> [--name MANIFOLD] [--fit-mode auto|pca|spectral] [--max-dim N] [-f]   # derive a discover manifold from a standalone template
saklas manifold fit <name>|<folder> [-m MODEL] [--sae REL] [--method pca|spectral|auto] [--max-dim N] [--var-threshold T] [--k-nn K] [--bandwidth SIGMA] [--max-subspace-dim R] [--smoothing auto|0|LAMBDA] [--persistence-frac F]  # authored or discover-mode (hyperparams apply only to discover folders; --smoothing curved only, --persistence-frac auto only)
saklas manifold bake <name> <expression> [-m]    # shared grammar: "0.3 ns/a + 0.5 ns/b|ns/c"
saklas manifold merge <name> <src...> [-f]           # union discover-mode node corpora
saklas manifold transfer <name> --from SRC --to TGT [-f]   # cross-model Procrustes
saklas manifold compare <concepts...> -m MODEL [--ridge-scale R]
saklas manifold why <concept> -m MODEL [-j]       # per-layer ||baked|| as a 16-bucket histogram
saklas pack ls [-v|-j] | show <name> [-j]            # list / inspect manifolds
saklas pack install <target> [-a NS/N] [-f]          # HF coord or local folder (also ports legacy saklas-packs)
saklas pack search <query> [-j|-v]                   # search HF hub for saklas-manifold repos
saklas pack push <name> [-a OWNER/N] [-m MODEL] [--variant raw|sae|all]
saklas pack rm <name> [-y]                           # remove folder (bundled respawns)
saklas pack clear <name> [-m MODEL] [--variant raw|sae|all]   # delete per-model fitted tensors
saklas pack refresh <name> [-m MODEL]                # re-pull (hf) / re-fit (-m scoped)
saklas pack export gguf <name> [-m MODEL] [-o PATH] [--model-hint HINT]   # fold a 2-node pca manifold to a control-vector GGUF
saklas experiment fan <model> "<prompt>" -g concept=0,0.5,1 # alpha grid as loom siblings
saklas experiment transcript run <path.yaml> [model]        # replay a saved transcript
saklas experiment naturalness <model> "<prompt>" --manifold F -S EXPR  # behavior-manifold eval
saklas template create <name> --slot TOKEN --values V... --contexts FILE [--description TEXT] [-f]
saklas template ls [-j] | show <name> [-j] | rm <name> [-y]
saklas template score <name> -m MODEL [-S EXPR] [--by sum|mean] [-j]   # restricted-choice value distribution
saklas config show [-c PATH ...] [--no-default] [-m MODEL]
saklas config validate <file>
pytest tests/                                   # all; GPU tests gated on CUDA/MPS
```

The root parser has exactly seven verbs: `tui`, `serve`, `manifold`, `pack`,
`experiment`, `config`, `template`. There is no `vector` alias. `manifold` is the
unified compute surface
(extract/generate/from-template/fit/bake/merge/transfer/compare/why); `pack` owns
lifecycle and distribution (ls/show/install/search/push/rm/clear/refresh/export
gguf), so install via `pack install` and export via `pack export gguf`; `template`
owns the standalone templated-completion artifact (create/ls/show/score/rm — a slot
+ candidate values + multi-turn contexts, read by both the completion **scorer**
and a `manifold from-template` fit). No `argv[0]` peeking, no bare-TUI fallback —
`saklas google/gemma-2-2b-it` is an argparse error. Bare `saklas` / `saklas
manifold` / `saklas pack` / `saklas experiment` / `saklas config` / `saklas
template` print help and exit 0.

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
(`io.selectors` walks `manifolds_dir()`): a bare slug resolves through the steering
grammar's bare-atom tier (`core/steering_expr`) — the bipolar-pole/name resolvers
(`resolve_pole`/`resolve_manifold_name` over the installed 2-node `pca` manifolds)
first, then `io.selectors.resolve_bare_name` for a multi-node manifold node label,
raising on cross-tier collision. A steering expression composing role-augmented terms must agree on
role; plain `:raw` terms compose with role terms but emit a one-time
`RoleBaselineMismatchWarning`.

Canonical naming: `canonical_concept_name` (module-level in `core/session.py`)
slugs poles via `[^a-z0-9]+ → _` and joins bipolar poles with `BIPOLAR_SEP = "."`,
so `/steer formal . casual` and `/steer formal.casual` resolve to the same manifold.
`NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$`; `@` is forbidden (HF revision separator),
and `.` is used over `~` because HF repo names reject `~`.

## Steering expression grammar

Every input surface — Python, YAML, HTTP, TUI, `manifold bake` — speaks the
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
probe_atom  := NAME ["." NAME] ["[" INT "]"]       # vector probe (e.g. confident.uncertain),
                                                   # optional coord axis (personas[3])
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

Probe gates accept these identifier shapes against the merged scalars the session
writes into `TriggerContext.probe_scores`. Vector probes are concept names whose
bare form reads coordinate axis 0 (`@when:confident.uncertain > 0.4`) and whose `[i]` form
reads a specific coordinate axis of a multi-axis fit (`@when:personas[3] > 0.4`);
both match the keys `Monitor.flat_scalars` emits. The coordinate is
domain-frame (pole-normalized at rank-1: `1.0` at the positive node), so a 2-node
concept gate reads the same single coordinate it always did.
Manifold subspace-fraction gates write the `:fraction` channel suffix
(`@when:emotions:fraction > 0.5`) — the share of the centered activation living
in that manifold's subspace, in `[0, 1]`. Manifold label-similarity gates write
the `@<label>` suffix (`@when:emotions@happy > -0.5`) — the negated distance to
a named node (larger = closer), so the natural threshold range is negative. The
distance is reported in units of the probe's **typical label spacing** (the
median node nearest-neighbor whitened distance, a single per-probe scale), so a
threshold reads as "within N typical label-spacings" and **transfers across
probes** — the raw whitened distance spans ~60× by fit (a node sits 1.15..72
from neutral), which made a bare threshold meaningless. The scale is a uniform
per-probe constant, so `nearest` still ranks by raw distance (literally nearest,
distinct from the density-aware `~<label>`). Two
**fuzzy-manifold** channels join them (additive — the `@<label>` distance gate is
untouched): the soft-assignment probability `~<label>` (`@when:personas~hacker >
0.5`) — a normalized, in-`[0,1]` `softmax(−d²/2τ²)` membership over the nodes,
the distributional counterpart to argmax `nearest` — and the tube-fit density
`:membership` (`@when:emotions:membership > 0.6`) — `exp(−residual²/2σ²)` under the
fitted within-node thickness `σ(z)`, high when the activation sits inside the
manifold's learned tube (distinguishes off-surface from on-surface-but-diffuse,
which a hard `residual` threshold can't). The
reserved label `neutral` (`@when:personas@neutral > -0.5`) reads the negated
distance to the frame *anchor* (the per-model neutral mean), in the same
label-spacing units — every fit is
neutral-anchored, so neutral is a point in the same whitened metric as the nodes,
not a stored node. It competes in the nearest-node ranking, so the channel is
present whenever neutral lands in a probe's top-N; suppressed only when a manifold
already carries a real node named `neutral`. Every shape flows through one
`ProbeGate` (full namespaced string stored verbatim, matching the keys
`Monitor.flat_scalars` emits) and round-trips byte-for-byte.

`%` is the manifold operator: `<manifold> % <position>` places a generation at a
point of a fitted manifold. `<position>` is `<coord_list>` (a comma-separated list
of authoring coordinates, one per intrinsic dimension, e.g.
`0.7 emotions%0.3,0.8,0.0@response`) or `<label>` (sugar for "the coords of the node
labeled `pirate`", e.g. `0.5 personas%pirate`). The coefficient slot is
`along[,onto]` — `along` is the slide fraction toward the position, `onto` (curved
manifolds only) collapses the off-surface in-subspace residual. An affine `%` term
(flat manifold, e.g. `personas%pirate`) joins the merged subspace as a push
fragment; a curved `%` term (e.g. a curved-resolving manifold `m%…`) gets its own injection term. A `%`
term doesn't compose with `~`/`|`/`!`. Arity (coord form) and label existence
(label form) are validated at manifold-load time. Two *curved* manifolds at one
layer must be (near-)orthogonal or `OverlappingManifoldError` raises; the merged
affine subspace is always orthogonalized against the curved spans. See "Manifold
steering" below.

A bare slug (`pirate`) resolves in `core/steering_expr`: first as a bipolar
pole/name of a 2-node `pca` manifold (`resolve_pole`/`resolve_manifold_name`), then
via `io.selectors.resolve_bare_name` as a multi-node manifold node label
(synthesizing a label-form `ManifoldTerm`, e.g. `local/personas%pirate`).
Cross-tier ambiguity raises `AmbiguousSelectorError`. Namespace-qualified or
variant-suffixed forms (`alice/pirate`, `pirate:role-x`, `civilian.pirate`) skip
the manifold-label tier.

## Templated completions + scoring

Some categories you *reference* rather than *embody* — days of the week, months,
durations, directions. "someone who is Tuesday" is nonsense, so the persona-framed
extraction doesn't apply. A **template** is the artifact for these: a `slot` token,
a set of candidate `values`, and one or more multi-turn `contexts` whose final
assistant turn carries the slot (`io/templates.py::TemplateFolder`, on disk at
`~/.saklas/templates/<ns>/<name>/template.json`). Invariant: the slot appears
**exactly once** in each context's final `assistant` string and **never** in a
history turn (history is shared common-mode across the values; the slot lives only
where the value is read). A single-turn template is the degenerate
`turns:[{user}]` case.

The template is a first-class artifact with **two** consumers:

- **The completion scorer** (`core/scoring.py::score_choices`) — for each context,
  the restricted-choice logprob distribution over the values: the model's belief
  about which slot-fill comes next. Forced-choice scoring against the **raw** model
  distribution (temperature 1, no top-k/p truncation), one batched teacher-forced
  forward per chunk; `_shared_prefix_len` absorbs the boundary-token merge
  (`"Monday"` vs `"Tuesday"` may retokenize the last prefix token). Multi-token
  candidates report both `sum_logprob` (the joint `log P(candidate | context)`) and
  `mean_logprob` (length-normalized), each with its own softmax over the set —
  neither view is silently chosen. An optional `steering=` runs the scoring forward
  under a steering expression, so the project's core question answers directly:
  *did steering shift the distribution, not just the argmax?* `session.score_choices(
  messages, choices, …)` / `session.score_template(name, steering=…)` return one
  `ChoiceScores` per context.
- **A manifold fit** — `manifold from-template <tmpl>`
  (`io/manifolds.py::create_manifold_from_template`) resolves the template, expands
  its `values × contexts` into per-value node corpora (the slot-filled assistant
  turns, `corpus[i]` aligned to `contexts[i]`), and writes a discover folder that
  stores the corpus **and** a `template_ref`. At fit time the pipeline resolves the
  ref to use the template's multi-turn contexts as the per-node elicitation
  prefixes (the multi-turn generalization of the A2 `response[i] → prompt[i]`
  alignment). The template is the authoring source of truth; the manifold's corpus
  is its materialization, and the resolved template's content hash folds into
  `nodes_sha256` so a context/value edit re-fits. There is no embedded `template`
  block anymore — `manifold from-template` replaces the former `manifold template`.

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
(`saklas/data/baseline_prompts.json`, 48 prompts; length must be a multiple of
`k`). A shared one-paragraph length directive (`_LENGTH_DIRECTIVE`) leads every
system prompt — the persona at node generation, the sole system for the neutral
baseline, and the sole system at *capture* (`_encode_and_capture_all`; the
persona is generation-only, so node capture sees only the directive). Identical
and symmetric across all four sites, it is common-mode that cancels at extraction
while keeping responses from rambling past the token cap, and it matches capture
framing to generation framing (not OOD). `kind` is recorded per node
(generation-time provenance; the fit never
consumes it). `--role`/`role=` opts into a persona-baselined fit (the explicit
role overrides the kind-derived label at both generation and capture). A
**monopolar** concept (`baseline=None`) authors a genuinely **1-node** `pca`
folder. The pipeline recognizes the single-node shape (a flat `pca` fit otherwise
needs `k+1 ≥ 2` poised nodes) and folds `concept − ν` — ν = the model's neutral
activation mean (`layer_means`) — into a 1-node neutral-anchored ray via
`fold_directions_to_subspace` (the same primitive `manifold bake` uses): no
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
projection, monitor reads, cross-model transfer, and `manifold compare` require a
`LayerWhitener` covering every scored layer (`covers_all`) and raise `WhitenerError`
otherwise — there is no Euclidean fallback (on real LMs the Euclidean metric is
rogue-dominated, so it would be a wrong answer, not a degraded one). The sidecar
`share_metric` / `subspace_metric` fields are provenance and read `mahalanobis`,
the one exception being a monopolar fit's `subspace_metric`, which labels its
raw-δ̂ basis `euclidean` (a basis-selection label — `concept − ν` cancels
common-mode by differencing, like DiM — not a metric fallback). The `LayerWhitener`
(`core/mahalanobis.py`) is built lazily from cached neutral activations and drives
the closed-form LEACE `~`/`|` projection, the whitened/Fisher subspace fit, the
whitened monitor reads, and `manifold compare`.

## Injection

One injection kernel — `core/manifold.py::subspace_inject` — for every steering
term. There is no angular/additive mode, no `injection_mode`/`theta_max`, no
`_STEER_GAIN`. At dispatch (`session._compose_steering_entries`) each term is
classified and lowered: vectors / poles / `~`/`|` projections / `!` ablations /
affine `%` are composed into **one merged affine subspace per trigger group** via
`synthesize_subspace`; each curved `%` gets its own term. `apply_to_model` then
attaches per-layer `subspace_inject` groups.

The kernel decomposes `h = mean + h_par + h_perp` against the layer's affine
subspace and applies two ops: **along** *translates* the in-subspace foot by the
fixed `α·(target−neutral)` offset (preserving per-token spread — not a lerp onto
the absolute target, which loop-collapses at strong push; on a curved surface it
transports the off-surface residual `H_n` to stay normal at the new foot), with a
per-axis κ mask collapsing ablation axes toward 0 instead; **onto** collapses
`H_n` toward the surface (vacuous when the surface fills its subspace, i.e. for
every flat/affine term). With no σ-field it scales `H_n` by `(1 − o)` — `o=1`
lands on the zero-thickness wire. With a **fuzzy-manifold σ-field**
(`LayerSubspace.sigma_at`, curved fits only) it instead shrinks `‖H_n‖` toward the
local within-node thickness `σ(z)` — `o=1` lands one-σ off the wire, a sample-like
point on the surface's *typical set*, direction preserved and never expanding a
residual already inside the tube; `σ=0` reproduces the `(1 − o)` collapse exactly.
The
off-subspace residual `h_perp` is always kept verbatim, which is what lets a
vector and N orthogonal manifolds compose with zero cross-talk. A flat (affine)
subspace takes an analytic shortcut (foot = the projected coord, no Gauss-Newton /
RBF), load-bearing for throughput; a curved manifold runs a warm-started per-token
nearest-point foot-follower. fp32 throughout; the only norm guard is the soft cap
`‖h_new‖ ≤ 3·‖h‖`.

Gain (`core/hooks.py`): the per-layer share is normalized to **mean 1** (`Σ_L
share_L = n_layers`) and `eff_along_L = share_L · _MANIFOLD_ALONG_GAIN`
(`= 16.0` — bumped from `0.125` when the affine target went whitened-unit and
live-recalibrated on gemma-4-12b, see below; the translate-slide gain for both
modes). `_MANIFOLD_GAIN` scales
**onto** only (the curved off-surface collapse). For an affine term the
coefficient α is folded into the translate *target* by `synthesize_subspace`, so α
scales the offset magnitude (unclamped); for a curved term α is the (clamped
`[0,1]`) `along` fraction. **Whitened along-normalization:** when a covering
whitener is present `synthesize_subspace` makes the affine push target a
*whitened-unit* direction (`target = Σᵢ coeffᵢ·(B@dirᵢ)/‖dirᵢ‖_M`) and the share
the *whitened* displacement `‖Δ‖_M` — so the per-layer slide budget is the same
whitened amount for every target (`Σ_L eff_along_L = gain·n_layers`), linear in α.
This replaced steering by the raw-Euclidean node distance, which spanned ~100×
across targets (a tight pole sits ~0.3 from neutral, a far persona centroid ~17),
so one `along` gain couldn't calibrate both — `0.5 formal%formal` did nothing
while `0.5 personas%caveman` slammed the output. `along` is now a scale-stable
strength knob across ranks/targets; per-target *coherence* variance (~2-3×, §10)
remains. There is **no lever / N correction** and **no `[0,1]`
clamp / water-fill on `along`** (a high-signal layer is meant to overshoot the
target; the de-rogued whitened coords keep it controlled and `norm_cap` bounds
it). `onto` stays clamped `[0,1]`. (`_MANIFOLD_ALONG_GAIN` is tagged a prototype —
calibrated so `≈0.5 <concept>` lands at the coherent sweet spot; the whitened
normalization changes the absolute scale, so this constant is **due for
recalibration** against live output.) A steered layer always runs the slow (ctx-consulting) hook, so per-step
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

`fit_mode` is one of five: `authored` (user supplies domain + coords; curved),
`pca` / `spectral` / `auto` (discover — labeled corpora only, coords derived
per-model; `pca` is flat, `spectral` curved, `auto` picks between them), and
`baked` (corpus-less, a precomputed direction written by `manifold bake` — see
io/AGENTS.md). `auto` runs `select_topology` at fit time (`core/manifold.py`):
flat (`pca`) vs curved (`spectral`) chosen by GCV in a shared whitened-reduced
metric, plus periodic (`BoxDomain`) axes detected by Vietoris–Rips H1 persistent
homology (loop counter, robust to ellipse distortion) coordinated off the
spectral eigenpairs, with a guarded single-cycle fallback for faint rings PH's
hole-size threshold misses (days-of-week and the like). The resolved geometry is per-model and recorded in the
sidecar (`resolved_fit_mode` + the ranked `topology_candidates`). Sphere is
authored-only — not an auto candidate. The curved (`spectral`/`auto`) RBF fit is
**penalized**: a GCV-selected smoothing λ (`fit_rbf_smoothed`) regularizes the
surface so it doesn't chase noise in the centroids, generalizing the exact
interpolation `authored` keeps (node = exact steering target). λ=0 reproduces the
interpolant; the hot-path `eval_rbf` is unchanged (only the coefficients shrink).

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
domain + per-node coords) vs *discover* — `"pca"`/`"spectral"`/`"auto"`, where the
user supplies labeled corpora only and coordinates are derived per-model at fit
time. `"auto"` defers the flat-vs-curved-vs-periodic choice to `select_topology`
(see below); `"pca"`/`"spectral"` pin it.
The derivation is **layer-agnostic** — there is no reference layer. Each fit layer
contributes its whitened, node-mean-centered `(K, K)` Gram and the coords come from
their mean (the *consensus Gram*); whitening puts every layer in common units so
the average is signal-weighted — a layer where the nodes aren't separated drops out
on its own, so the layout draws on whichever layers carry the concept. PCA
eigendecomposes the consensus Gram and picks the smallest prefix whose cumulative
variance crosses `var_threshold` (default 0.70), capped at `max_dim` (default 8);
spectral reads pairwise distances off it, runs Laplacian eigenmaps on a symmetric
k-NN graph, and picks `k` by the eigenvalue-ratio cliff (both embed the same
consensus geometry). `fit_mode=pca` produces a flat affine subspace (no RBF);
`spectral` (and `authored`) produces a curved RBF surface. For a flat fit the
per-layer steerable subspace *is* its `max_dim`-capped layout span, so `max_dim`
is the only dim knob — `max_subspace_dim` survives only for the curved spectral
fit (where the RBF subspace can carry off-surface dims). The steer-time origin is
always the projection of the per-model neutral mean onto the subspace (the affine
fit neutral-anchors the frame), so there is no `anchor_origin` knob. Per-model
coordinates are the architectural shift: a Gemma fit and a Qwen fit produce
different node layouts for the same heap (stored as `node_coords` in the per-model
safetensors).

`fit_mode="auto"` chooses the geometry per-model via `select_topology`, in two
decoupled decisions (decoupling is deliberate: a single reconstruction score has a
dimension bias — more spectral coordinates always "fit" better, so the highest-dim
candidate always wins). **(a) flat vs curved** — the flat affine (`pca`) and curved
RBF (`spectral`) fits, each at its own intrinsic dim, are scored by GCV (effective-
dof-penalized) in a shared per-layer whitened/Fisher reduced metric; lower wins.
**(b) periodic axes** — Vietoris–Rips H1 *persistent homology* counts the loops
(topologically robust: a circle, a 6:1 ellipse, and a noisy circle all read as one
loop; a 2-torus as two; a blob/arc/line as zero), and the spectral eigenpairs
supply the angle coordinates, routing a detection to a periodic `BoxDomain`.
Spheres are **authored-only** (a speculative topology that's the least reliable to
detect from few centroids). `--persistence-frac` tunes the loop-significance
threshold. PH counts loops by *hole size*, so a **faint ring** — a small cyclic
modulation on a near-equidistant heap (e.g. day-of-week centroids) — slips under
its threshold; a complementary guarded single-cycle fallback (1-D max-degree +
closed-tour + local-recall + cyclic-grading tests) catches it when PH counts zero
and routes one periodic axis, while rejecting arcs/lines/grids/blobs/persona-fans. The curved fit (`spectral`, or an `auto` resolving to it) is a
**penalized** RBF — `--smoothing auto` GCV-selects a smoothing λ (default), `0` is
exact interpolation, a float is a fixed λ; the penalty regularizes the surface
against centroid noise without touching the hot-path evaluator.

`manifold generate <name> --concepts ... [--kind abstract|concrete]` LLM-authors a
discover folder via `session.generate_responses` — each concept answers the shared
baseline prompts in character (one corpus per node). The shared baseline prompts
hold topic common-mode across nodes (response[i] ↔ prompt[i % k]), so the
per-concept centroids stay comparable without a per-manifold scenario set.
`manifold fit <name>` then fits — the two steps are deliberate (a flaky
generation leaves inspectable corpora). Cross-model Procrustes alignment for discover coords is deferred (TODO
in `io/manifolds.py`). The naturalness eval (`experiment naturalness`) fits a
behavior-space manifold over node output distributions in Hellinger space and
reports the per-step Bhattacharyya distance of a steered trajectory to it (low =
natural; `--compare-linear` scores a straight-chord baseline alongside).

### Bundled manifolds + coefficient regime

Complete bundled artifacts ship under `saklas/data/manifolds/`, materializing into
`~/.saklas/manifolds/default/` on session start via `materialize_bundled_manifolds()`
(process-scope no-op after the first call). The materializer only advertises
folders whose `manifold.json` and declared `nodes/*.json` corpus files are all
present, so an interrupted `scripts/regenerate_bundled.py` run can leave a
partial folder in the package tree without exposing it as a default manifold:

- **17 concept manifolds** — bipolar 2-node `fit_mode=pca` axes, tagged by
  category (`epistemic`, `alignment`, `register`, `cultural`). These are the
  steering vectors: `0.5 formal.casual` steers toward `formal` (node 0). The
  `register` (7) and `cultural` (4) families are independent bipolar axes, *not*
  fused into a discover manifold — each has a designed opposite and the primary
  use is independent signed control. Affect is reserved for `emotions`, not shipped as
  bipolar axes.
- **`personas`** — discover `fit_mode=auto` resolving flat (a low-dim ~rank-8
  affine subspace — the auto selector independently picks flat for the persona
  fan), 107 persona archetype nodes (`assistant`…`vandal`) in assistant-baselined
  activation space; from Anthropic's Assistant Axis paper (arXiv 2601.10387).
  `max_dim` 8.
- **`emotions`** — a discover `fit_mode=auto` affect manifold over PAD
  (pleasure × arousal × dominance), 20 mood nodes. `auto` resolves the geometry
  per-model (flat affine vs curved RBF by GCV, with periodic detection); on
  gemma-4-12B it resolves to a flat 3-D affect subspace. It materializes only after
  all 20 mood corpora exist; incomplete package-data output is skipped.

Recommended α is vector-comparable: aim for `α ≈ 0.5`, tune up toward `α ≈ 1.0`
for stronger expression. (For an affine push term α is unclamped — it sets the
translate-offset magnitude; for a curved `%` term `along` clamps to `[0,1]`.) The
global translate-slide gain `_MANIFOLD_ALONG_GAIN = 16.0` (live-calibrated on
gemma-4-12b — it jumped from `0.125` when the affine target became whitened-unit, a
unit-scale change of ~100×) targets `≈0.5 <concept>` at the coherent band for both
tight concepts and personas (`α ≈ 1.0` is the strong / over-steer zone where hard
personas break — dial down per target, §10). Because
the target is now whitened-unit and the share is normalized
to mean 1 and the lever is gone, a low-dim and a high-dim fit — a 2-node vector,
`personas`, and `emotions` once its corpus is complete — land in the same α-band
without per-fit retuning.
Architecture-level behavioral notes (hold across model families; α values are
qualitative, MPS is not bitwise deterministic so compare qualitatively):

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
> verifiably worked). The translate-slide gain (`_MANIFOLD_ALONG_GAIN`) is
> live-calibrated but still coupled to that: with the geometric scale now whitened
> away, what remains is per-target *coherence* variance (~2× — a hard persona like
> hacker shatters at roughly half the effective gain a robust concept tolerates),
> which a single scalar gain can't unify. `16.0` is the coherence-first compromise.

## Python API

```python
from saklas import SaklasSession, SamplingConfig, Steering, Profile

with SaklasSession.from_pretrained("google/gemma-3-4b-it", device="auto") as session:
    name, profile = session.extract("confident.uncertain")   # returns (canonical_name, Profile)
    result = session.generate(
        "What makes a good day?",
        steering=f"0.3 {name}",
        sampling=SamplingConfig(temperature=0.7, max_tokens=256, seed=42),
    )
    with session.steering("0.5 wolf"):                 # bare label → personas%wolf node
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
  `fit(folder)` fits a multi-node manifold and returns a `Manifold`;
  `bake(name, expression)` lands a corpus-less baked manifold from a steering
  expression (the mirror of `manifold bake`). `Profile` wraps `dict[int, Tensor]`
  (mapping interface plus `layers`, `metadata`, `save`/`load`, `merged`,
  `projected_away`, `cosine_similarity`).
- `session.score_choices(messages, choices, *, assistant_prefix="", steering=None)`
  and `session.score_template(template, *, steering=None)` return the
  restricted-choice completion distribution (`ChoiceScores`, per-context for a
  template) — the logit read; `core/scoring.py`. Steering-aware (the distributional
  before/after). `template` is a `TemplateFolder` or a `<name>`/`<ns>/<name>`
  selector.
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
  templates/<ns>/<name>/               # standalone templated-completion artifact
    template.json                      # slot, values, contexts:[{turns:[{role,content}], assistant}]
  manifolds/<ns>/<name>/               # THE concept + steering-manifold root
    manifold.json                      # name, source, fit_mode, per-node {label,kind,role?}, domain/coords or hyperparams, files{sha256}, template_ref?
    nodes/NN_<label>.json              # one JSON response list per node (response[i] ↔ baseline_prompt[i % k])
    <safe_model_id>.safetensors        # fitted per-layer subspaces (+ .json sidecar);
                                       # discover/baked also carry node_coords (the layout)
    <safe_model_id>_sae-<rel>.safetensors    # SAE-space fit
    <safe_model_id>_from-<src>.safetensors   # cross-model transfer
    <safe_model_id>_role-<slug>.safetensors  # role-augmented
  models/<safe_model_id>/
    neutral_activations.{safetensors,json}   # neutral corpus (mult. of 48) × layers, fp32;
                                       # the single per-model neutral artifact — the
                                       # probe-centering mean is its per-layer X.mean(0),
                                       # the whitener covariance is built from the stack
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
- **One injection kernel.** Every steered layer runs `subspace_inject`. The
  dominant case — one always-active (`Trigger.BOTH`) affine group — takes a
  precomputed fast path (`SteeringHook._single_affine_fast`): one analytic slide +
  in-place write, no ctx read, no foot state. `torch.compile`/StaticCache graph
  capture is eligible for unsteered generation (`all_fast_path`) **and** for that
  static-affine steered case (`static_steerable` — the hook is a fixed tensor-op
  sequence and StaticCache never bypasses forward hooks). Curved / gated / phased
  steering keeps the ctx-consulting general path (DynamicCache, eager); curved
  manifolds pay a warm-started O(R) per-token foot solve. (StaticCache-with-steering
  is CUDA-only, so MPS/CPU are unaffected; the through-the-hook graph capture wants a
  CUDA validation pass.)
- **Share baked at fit**, normalized to mean 1 at apply; the subspace foot
  translates by `share_L · _MANIFOLD_ALONG_GAIN · target` (the target already
  carries the coefficient; `_MANIFOLD_GAIN = 1.0` is now the `onto`-only gain). No
  norm preservation (onto is meant to shrink `‖h‖`); `norm_cap = 3·‖h‖` is the
  only bound.
- **Top-p via `torch.topk`**, not full-vocab sort; `top_k` (default 1024 cap) is a
  hard candidate-pool cap applied before top-p (llama.cpp/Ollama order).
- **Monitor capture is hook-driven**, inline with generation — no second forward
  pass. The unified `Monitor` reads every probe — flat or curved — as a whitened
  **coordinate**, emitting a full `ProbeReading` (`coords` domain-frame + `fraction`
  + `nearest` + `residual`, plus per-layer traces). The full-reading *shape* is the
  research-tool priority; execution is no-redundancy: the whole **flat** roster is
  scored together (`_score_flat_batched` — one `Σ⁻¹h` Woodbury apply + block-diagonal
  matmuls + a single host transfer per layer), **curved** probes run the per-probe
  `invert_parameterization` foot solve (warm-started across decode tokens from the
  previous foot on the sequential live path). Mahalanobis-only: the whitener must
  cover every probed layer (`covers_all`), else scoring raises `WhitenerError` (no
  Euclidean path). `covers_all` is trustworthy as "finite factors everywhere": any
  non-finite layer is excluded, and neutral activations are cached fp32 (so gemma-3's
  late layers don't overflow the fp16 65504 ceiling to ±inf). Per-token scoring is
  **conditional on need**: a gate / live-stream consumer ⇒ **incremental** (each token
  scored live, only the latest per-layer slice + `ProbeReading` rows retained,
  O(layers·D)); probes attached but only the aggregate wanted (no gate / stream — e.g.
  a stateless server gen) ⇒ **aggregate-only** (a bounded tail ring, NO per-token
  scoring, pooled once at finalize, T scorings → 1); `return_hidden` ⇒ full retention
  + `score_per_token`.
- **Steering hooks are transient** — composed before generation, removed after.
- **MPS discipline** — diffs on CPU, `torch.mps.empty_cache()` between extraction
  *batches* (capture is chunked at `_CAPTURE_BATCH`; corpus generation at
  `_CORPUS_GEN_BATCH`), end-of-loop sync to dodge Metal command-buffer reuse crashes.

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

17 curated concepts under `saklas/data/manifolds/<concept>/` — all bipolar
(2-node `pca`). Under 4.0 / A2 each pole's corpus is conversational responses to
the shared baseline prompts. Monopolar `extract` (`baseline=None`) is still a
genuine **1-node** fold against the neutral mean ν (see "Extraction") — a user
`extract("agentic")` authors a 1-node ray — but no monopolar concept ships
bundled anymore (the former `agentic` / `manipulative` were dropped or folded
into bipolar `sincere.manipulative`). Bundled regeneration is unified under
`scripts/regenerate_bundled.py` — one A2 pipeline writing the bipolar axes,
`personas`, and `emotions`; the fit is the separate `manifold fit` step. Partial
generation output is ignored by bundled materialization until every manifest node
has a corpus file.

The 4.0 / A2 regen also dropped the `affect`, `social_stance`, and `identity`
categories as bipolar axes: affect (the former angry.calm / happy.sad /
fearful.unflinching) folds into the `emotions` manifold, and social_stance
(authoritative.submissive, high_context.low_context, self.other), identity
(ai.human), and hallucinating.grounded were cut.

Categories: `epistemic` (confident.uncertain, honest.deceptive,
curious.disinterested), `alignment` (refusing.compliant, sycophantic.blunt,
sincere.manipulative), `register` (formal.casual, direct.indirect,
verbose.concise, creative.conventional, humorous.serious, warm.clinical,
technical.accessible), `cultural` (masculine.feminine,
individualist.collectivist, traditional.progressive, religious.secular).

Known model-level axis entanglements (cross-model robust, weighted cosine via
`manifold compare`) — document for users, not probe-design failures:
- `masculine.feminine ↔ traditional.progressive` (+0.5–0.6) — Hofstede MAS read as
  traditionalism

`saklas/data/neutral_statements.json` holds the neutral baseline as organic,
no-persona/no-role responses to the same shared baseline prompts (a multiple of
the 48-prompt set; the shared one-paragraph length directive is its *only* system
prompt — the same directive that leads the node systems — so the framing it
shares with them cancels at extraction), regenerated via
`session.generate_neutral_responses`; it backs the probe-centering means +
Mahalanobis whitener. `saklas/data/baseline_prompts.json` (48 affect-neutral,
topically-diverse prompts) is the shared elicitation set every node and the
neutral corpus answer. `scripts/regenerate_bundled.py` regenerates it as the
`neutral` pseudo-target (`--neutral-samples`, default 2, sets its density
independently of the per-node `--samples-per-prompt`).

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
