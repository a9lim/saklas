# AGENTS.md

## What this is

`saklas` is a Python library + dual-protocol HTTP server for activation steering
and trait monitoring on HuggingFace causal LMs. It runs OpenAI `/v1/*` and
Ollama `/api/*` on one port, plus a native `/saklas/v1/*` API and a Svelte
dashboard at `/`. Steering signal comes from representation engineering, unified
under a single artifact family — the **manifold**: labeled nodes placed on a
domain, fit to a per-layer subspace. A difference-of-means steering vector is
the 2-node flat case; `personas` is a 107-node flat fan; `emotions` is a 20-node
affect manifold over PAD. Every steering term — vectors, poles, `~`/`|`
projections, `!` ablations, and `%` manifold positions — lowers at generation
time to one unified per-layer injection (the along/onto subspace kernel,
`core/manifold.py::subspace_inject`). Per-call coefficients, no model mutation.
Two frontends over one engine: `SaklasSession` (programmatic) and `saklas serve`
(HTTP APIs plus the web dashboard).

Version lives in `saklas/__init__.py` as `__version__`. `pyproject.toml` reads
it via `version = {attr = "saklas.__version__"}`, so there is one place to bump.
Do not bump it as part of feature work — version bumps are user-owned.

Releases: merge a version bump to `main` → `.github/workflows/release.yml` tags
`v$VERSION`, builds, publishes via trusted publishing, and cuts a GitHub
release. A push without a bump is a no-op.

The cross-cutting design — how extraction, composition, injection, and reads
fit together — lives in the repo-root `ARCHITECTURE.md`. Read it before
touching the engine.

## Subtree docs

Deep internals live in subtree `AGENTS.md` files — Claude Code auto-loads each
when you work in that directory. Consult them only when editing that layer.

- `saklas/core/AGENTS.md` — model loading, the manifold/subspace fit +
  injection, monitor + instruments, session, generation loop, loom tree
- `saklas/io/AGENTS.md` — manifold format, HF distribution, GGUF, merge,
  alignment, paths/selectors, source registries
- `saklas/cli/AGENTS.md` — eight-verb dispatch, config loading, flags
- `saklas/server/AGENTS.md` — OpenAI / Ollama / native routes, WS protocol
- `saklas/web/AGENTS.md` — dashboard mount, wire protocol, Svelte source layout

## Commands

```bash
pip install -e ".[dev]"                         # editable + pytest + SAELens
pip install -e ".[gguf]"                        # llama.cpp GGUF I/O
pip install -e ".[cuda,flash]"                  # bitsandbytes + kernels + tested FlashAttention (Linux/CUDA)
saklas serve <model_id> [--no-web] [--steer/-S EXPR]
saklas manifold extract <concept>|<pos> <neg> [-m MODEL] [--sae RELEASE] [--role SLUG] [--kind abstract|concrete|custom] [--system TEMPLATE] [--namespace NS] [--no-dls] [-f]
saklas manifold generate <name> --concepts C... [--kind abstract|concrete|custom] [--system TEMPLATE] [--samples-per-prompt K] [--seed S]
saklas manifold from-template <template> [--name MANIFOLD] [--fit-mode auto|pca|spectral] [--max-dim N] [--var-threshold T] [--description TEXT] [-f]
saklas manifold fit <name>|<folder> [-m MODEL] [--sae REL] [--layers L1,L2|workspace|all] [--method pca|spectral|auto] [--max-dim N] [--min-dim N] [--var-threshold T] [--k-nn K] [--bandwidth SIGMA] [--max-subspace-dim R] [--smoothing auto|0|LAMBDA] [--persistence-frac F] [--no-dls]
saklas manifold bake <name> <expression> [-m]    # additive subset: "0.3 ns/a + 0.5 ns/b"
saklas manifold merge <name> <src...> [-f]           # union discover-mode node corpora
saklas manifold transfer <name> --from SRC --to TGT [-f]   # cross-model Procrustes
saklas manifold compare <concepts...> -m MODEL [--ridge-scale R]
saklas manifold why <concept> -m MODEL [-j]       # per-layer ||baked|| as a 16-bucket histogram
saklas pack ls [-v|-j] | show <name> [-j]            # list / inspect manifolds
saklas pack install <target> [-a NS/N] [-f]          # HF coord or current local manifold folder
saklas pack search <query> [-j|-v]                   # search HF hub for saklas-manifold repos
saklas pack push <name> [-a OWNER/N] [-m MODEL] [--variant raw|sae|all] [--private] [--dry-run]
saklas pack rm <name> -y                             # remove folder (bundled respawns); -y required
saklas pack clear <name> [-m MODEL] [--variant raw|sae|all]   # delete per-model fitted tensors
saklas pack refresh <name> [-m MODEL]                # re-pull (hf) / re-fit (-m scoped)
saklas pack export gguf <name> [-m MODEL] [-o PATH] [--model-hint HINT]   # fold a 2-node pca manifold to a control-vector GGUF
saklas experiment fan <model> "<prompt>" -g concept=0,0.5,1 [-S EXPR] [--max-tokens N]   # alpha grid as loom siblings
saklas experiment transcript run <path.yaml> [model] [--max-tokens N]     # replay a saved transcript
saklas experiment naturalness <model> "<prompt>" --manifold F -S EXPR [--max-tokens N]   # behavior-manifold eval
saklas template create <name> --slot TOKEN --values V... --contexts FILE [--description TEXT] [-f]
saklas template ls [-j] | show <name> [-j] | rm <name> [-y]
saklas template score <name> -m MODEL [-S EXPR] [--by sum|mean] [-j]   # restricted-choice value distribution
saklas lens fit <model> [--corpus FILE] [--prompts N] [--seq-len T] [--dim-batch K] [--prompt-batch B] [-f]   # per-model Jacobian lens (backward passes; resumes by default)
saklas lens fetch <model> [neuronpedia] | ls <model> | show <model> [source] | use <model> <source> | rm <model> [source] [-y]
saklas lens top <model> "<prompt>" [-k K] [--layers L1,L2] [--position P] [-j]   # workspace readout on a raw prompt
saklas lens decompose <selector> -m MODEL [-k K] [--layers L1,L2] [-j]   # J-space share + tokens of a direction
saklas sae train <model> <name> [--corpus FILE] [--layer L] [--tokens N] [-f]
saklas sae fetch <model> saelens:<release> [--layer L] [--revision REV]  # pure IO: validates via config, writes the binding, loads no weights
saklas sae ls <model> | show <model> [source] | use <model> <source> | rm <model> [source] [-y]
saklas config show [-c PATH ...] [--no-default] [-m MODEL]
saklas config validate <file>
pytest tests/                                   # all; GPU tests gated on CUDA/MPS
```

The root parser has exactly eight verbs: `serve`, `manifold`, `pack`,
`experiment`, `config`, `template`, `lens`, `sae`. `manifold` is the unified
compute surface (extract/generate/from-template/fit/bake/merge/transfer/
compare/why); `pack` owns lifecycle and distribution (ls/show/install/search/
push/rm/clear/refresh/export gguf); `template` owns the standalone
templated-completion artifact (create/ls/show/score/rm — a slot + candidate
values + multi-turn contexts, read by both the completion scorer and a
`manifold from-template` fit); `lens` owns local fitting plus source-aware
lifecycle/readout (fit/fetch/ls/show/use/top/decompose/rm); `sae` owns the
parallel local/external lifecycle (train/fetch/ls/show/use/rm). `lens fetch`
and `sae fetch` are both pure IO — neither loads model weights; both keep
provider-owned payloads in provider caches and store only pinned bindings
locally. No `argv[0]` peeking or bare-model fallback — `saklas
google/gemma-2-2b-it` is an argparse error. Bare `saklas` or any bare verb
group prints its dispatch menu and exits 0.

Every subcommand that takes `-c/--config` auto-loads `~/.saklas/config.yaml`
first, then composes explicit `-c` files on top (later overrides earlier). The
`vectors:` YAML key is a single steering expression parsed by
`saklas.core.steering_expr.parse_expr`. Flag precedence is CLI > YAML > runner
default throughout (including `--max-tokens`). `cli/AGENTS.md` has the full
per-verb flag set and the reserved short-flag table.

Deliberately absent surfaces (rejected designs — do not re-propose): no
`vector` verb alias, no `--steer-mode`/`--theta-max`/`--legacy`/
`--projection-metric`, no Euclidean fallback anywhere, no extraction method
besides difference-of-means (PCA@2 ≡ DiM). `--method` belongs only to
discover-mode manifold fitting/merging. The only steering knob at the CLI
beyond coefficients is `--no-dls`.

## Selector grammar

Shared across surfaces: `<name>`, `<ns>/<name>`, `tag:<t>`, `namespace:<ns>`,
`model:<m>`, `default`, `all`, optionally suffixed `:<variant>` where
`<variant>` is `raw` (canonical DiM), `sae`, `sae-<release>`, `from-<safe_src>`
(cross-model transfer), or `role-<name>` (role-augmented extraction — the
contrast pairs were generated under a chat template whose assistant-role label
was substituted, and the same substitution is auto-applied at steering time so
extract baseline equals steer baseline). There is no `pca` variant. Bare names
resolve cross-namespace and raise `AmbiguousSelectorError` on collision.
Concepts *are* manifolds (`io.selectors` walks `manifolds_dir()`): a bare slug
resolves through the steering grammar's bare-atom tier (`core/steering_expr`) —
the bipolar-pole/name resolvers (`resolve_pole`/`resolve_manifold_name` over
the installed 2-node `pca` manifolds) first, then
`io.selectors.resolve_manifold_label` for a multi-node manifold node label,
raising on cross-tier collision. A steering expression composing role-augmented
terms must agree on role; plain `:raw` terms compose with role terms but emit a
one-time `RoleBaselineMismatchWarning`.

Canonical naming: `canonical_concept_name` (module-level in `core/naming.py`)
slugs poles via `[^a-z0-9]+ → _` and joins bipolar poles with `BIPOLAR_SEP =
"."`, so `/steer formal . casual` and `/steer formal.casual` resolve to the
same manifold. `NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$`; `@` is forbidden (HF
revision separator), and `.` is used over `~` because HF repo names reject `~`.

## Steering expression grammar

Every live steering surface — Python, YAML, HTTP, and CLI — speaks the grammar
in `saklas.core.steering_expr`. `manifold bake` parses the same syntax but
accepts only namespace-qualified additive/subtractive scalar terms: dynamic
`!`/`%`, triggers, multi-coefficients, and Mahalanobis `~`/`|` projections
require a live model and are rejected offline. `parse_expr(text)` → `Steering`;
`format_expr` round-trips it back. Every `Trigger` a public factory can build
has a grammar form; `format_expr` raises `SteeringExprError` on a
programmatically-constructed trigger with no canonical spelling rather than
emitting an unparseable sentinel.

```
expr        := term (("+" | "-") term)*
term        := [coeff "*"?] ["!"] selector ["@" trigger]
selector    := atom (("~" | "|") atom | "%" position)?
position    := signed_num ("," signed_num)* | label
label       := NAME                                # a manifold node label
atom        := [ns "/"] NAME ["." NAME] [":" variant] | "sae" "/" INT
trigger     := phase ["&" gate] | gate
phase       := preset | ("first" | "after") ":" INT
preset      := before | after | both | thinking | response | prompt | generated
gate        := "when" ":" probe_atom op NUM        # op ∈ > >= < <=
probe_atom  := [ns "/"] NAME ["." NAME] ["[" INT "]"]  # vector probe (e.g. confident.uncertain,
                                                   # jlens/fake); optional coord axis (personas[3])
             | [ns "/"] NAME ":" ("fraction" | "membership")
             | [ns "/"] NAME ("@" | "~") NAME      # distance / assignment
             | "sae" "/" INT                       # resident SAE feature strength (activation /
                                                   # Neuronpedia maxActApprox when cached; raw otherwise)
```

`+`/`-` add terms, `*` attaches a coefficient (omit → 0.5), `~` projects onto a
direction (keep the shared component), `|` projects orthogonal (remove it), `!`
mean-ablates the concept (`h' = h − α(h·d̂ − μ·d̂)d̂`; bare `!x` is α=1.0).
`@<preset>` overrides a term's trigger; `@first:N`/`@after:N` are counted
decode windows (the grammar forms of `Trigger.first`/`Trigger.after`);
`@when:<probe><op><num>` is a probe gate that fires only on decode steps where
the monitor reading satisfies the comparison (implicit `prompt=False`); a
compound `@<phase>&when:<gate>` narrows a gate to a phase (e.g.
`@after&when:angry.calm>0.4`). `!` cannot compose with `~`/`|`.

Probe gates accept these identifier shapes against the merged scalars the
session writes into `TriggerContext.probe_scores`. Every shape also takes an
optional leading `<ns>/` segment — a J-lens token probe (`@when:jlens/fake >
0.01`, the readout-channel mean fitted-layer **probability**) or a probe
attached under a qualified selector (`@when:default/emotions@happy > -0.5`) —
stored verbatim. Vector probes are concept names whose bare form reads
coordinate axis 0 (`@when:confident.uncertain > 0.4`) and whose `[i]` form
reads a specific coordinate axis of a multi-axis fit (`@when:personas[3] >
0.4`); both match the keys `Monitor.flat_scalars` emits. The coordinate is
domain-frame (pole-normalized at rank-1: `1.0` at the positive node).
Manifold subspace-fraction gates write the `:fraction` channel
(`@when:emotions:fraction > 0.5`) — the share of the centered activation living
in that manifold's subspace, in `[0, 1]`. Label-similarity gates write the
`@<label>` suffix (`@when:emotions@happy > -0.5`) — the negated distance to a
named node (larger = closer), so the natural threshold range is negative. The
distance is reported in units of the probe's **typical label spacing** (the
median node nearest-neighbor whitened distance, a single per-probe scale), so a
threshold reads as "within N typical label-spacings" and transfers across
probes; `nearest` still ranks by raw distance. Two **fuzzy-manifold** channels
complement it: the soft-assignment probability `~<label>`
(`@when:personas~hacker > 0.5`) — a normalized in-`[0,1]`
`softmax(−d²/(2τ²) − R·log(τ))` posterior over the nodes (including the
Gaussian log-volume correction), the distributional counterpart to argmax
`nearest` — and the tube-fit density `:membership` (`@when:emotions:membership
> 0.6`) — `exp(−residual²/2σ²)` under the fitted within-node thickness `σ(z)`,
high when the activation sits inside the manifold's learned tube. The reserved
label `neutral` (`@when:personas@neutral > -0.5`) reads the negated distance to
the frame *anchor* (the per-model neutral mean) in the same label-spacing units
— every fit is neutral-anchored, so neutral is a point in the same whitened
metric as the nodes, not a stored node. It competes in the nearest-node
ranking, so the channel is present whenever neutral lands in a probe's top-N;
suppressed only when a manifold already carries a real node named `neutral`.
Every shape flows through one `ProbeGate` (full namespaced string stored
verbatim, matching the keys `Monitor.flat_scalars` emits) and round-trips
byte-for-byte. At generation preflight the composer channel-validates every
gate through `parse_gate_ref` against the attached instrument's supported
channels (`validate_gate_channels`): a gate on a channel the family can never
produce — e.g. `@when:sae/123:membership` (SAE has only the strength channel) —
raises `UnsupportedProbeChannelError` (400) instead of sitting silently
inactive. An SAE probe emits only its real normalized-strength channel
(`sae/<id>` / `sae/<id>[0]`); a lens probe only its mean-probability strength.

`%` is the manifold operator: `<manifold> % <position>` places a generation at
a point of a fitted manifold. `<position>` is `<coord_list>` (a comma-separated
list of authoring coordinates, one per intrinsic dimension, e.g. `0.7
emotions%0.3,0.8,0.0@response`) or `<label>` (sugar for "the coords of the node
labeled `pirate`", e.g. `0.5 personas%pirate`). The coefficient slot is
`along[,onto]` — `along` is the slide fraction toward the position, `onto`
(curved manifolds only) collapses the off-surface in-subspace residual. An
affine `%` term (flat manifold, e.g. `personas%pirate`) joins the merged
subspace as a push fragment; a curved `%` term gets its own injection term. A
`%` term doesn't compose with `~`/`|`/`!`. Arity (coord form) and label
existence (label form) are validated at manifold-load time. Two *curved*
manifolds at one layer must be (near-)orthogonal or `OverlappingManifoldError`
raises; the merged affine subspace is always orthogonalized against the curved
spans. See "Manifold steering" below.

A bare slug (`pirate`) resolves in `core/steering_expr`: first as a bipolar
pole/name of a 2-node `pca` manifold (`resolve_pole`/`resolve_manifold_name`),
then via `io.selectors.resolve_manifold_label` as a multi-node manifold node
label (synthesizing a label-form `ManifoldTerm`, e.g. `local/personas%pirate`).
Cross-tier ambiguity raises `AmbiguousSelectorError`. Namespace-qualified or
variant-suffixed forms (`alice/pirate`, `pirate:role-x`, `civilian.pirate`)
skip the manifold-label tier.

## Templated completions + scoring

Some categories you *reference* rather than *embody* — days of the week,
months, durations, directions. "someone who is Tuesday" is nonsense, so the
persona-framed extraction doesn't apply. A **template** is the artifact for
these: a `slot` token, a set of candidate `values`, and one or more multi-turn
`contexts` whose final assistant turn carries the slot
(`io/templates.py::TemplateFolder`, on disk at
`~/.saklas/templates/<ns>/<name>/template.json`). Invariant: the slot appears
**exactly once** in each context's final `assistant` string and **never** in a
history turn (history is shared common-mode across the values; the slot lives
only where the value is read). A single-turn template is the degenerate
`turns:[{user}]` case. Templates can ship **bundled** —
`saklas/data/templates/<name>/template.json` materializes to `default/<name>`
on session start (the io bootstrap runs the template materializer before the
manifold one, so a template-derived bundled manifold can `template_ref` it).
None ship at present; the machinery is dormant by design.

The template is a first-class artifact with **two** consumers:

- **The completion scorer** (`core/scoring.py::score_choices`) — for each
  context, the restricted-choice logprob distribution over the values: the
  model's belief about which slot-fill comes next. Forced-choice scoring
  against the **raw** model distribution (temperature 1, no top-k/p
  truncation), one batched teacher-forced forward per chunk;
  `_shared_prefix_len` absorbs the boundary-token merge. Multi-token candidates
  report both `sum_logprob` (the joint `log P(candidate | context)`) and
  `mean_logprob` (length-normalized), each with its own softmax over the set —
  neither view is silently chosen. An optional `steering=` runs the scoring
  forward under a steering expression, so the project's core question answers
  directly: *did steering shift the distribution, not just the argmax?*
  `session.score_choices(messages, choices, …)` / `session.score_template(name,
  steering=…)` return one `ChoiceScores` per context.
- **A manifold fit** — `manifold from-template <tmpl>`
  (`io/manifold_authoring.py::create_manifold_from_template`) resolves the
  template, expands its `values × contexts` into per-value node corpora (the
  slot-filled assistant turns, `corpus[i]` aligned to `contexts[i]`), and
  writes a discover folder that stores the corpus **and** a `template_ref`. At
  fit time the pipeline resolves the ref to use the template's multi-turn
  contexts as the per-node elicitation prefixes. The template is the authoring
  source of truth; the manifold's corpus is its materialization, and the
  resolved template's content hash folds into `nodes_sha256` so a context/value
  edit re-fits.

## Jacobian lens

The third artifact family, **per-model** rather than per-concept (Gurnee et
al., "Verbalizable Representations Form a Global Workspace in Language Models",
Transformer Circuits 2026). The lens is one matrix per source layer, `J_l =
E[∂h_final/∂h_l]` — the average first-order effect of a layer's residual on the
final-layer residual over positions and a web-text corpus — stored as immutable
per-layer fp32 shards. Saklas-fitted lenses live at
`models/<safe_model_id>/jlens/local/default/manifest.json`
(`LENS_FORMAT_VERSION = 6`, required exactly). Official Neuronpedia lenses stay
in the Hugging Face cache behind a commit-pinned
`jlens/bindings/neuronpedia.json`; `jlens/active.json` selects the runtime
source. The sidecar records the immutable corpus spec + token-id sha256, exact
source/live model identities, and one payload sha256 per layer. `lens fit` runs
the estimator (`core/jlens_fit.py::fit_jacobian_lens` — the **only** backward
passes in saklas; resumable, checkpointed, OOM-adaptive — see
`core/AGENTS.md`). The fit is compute-bound; `--layers` restriction is the one
real wall-time lever. ~100 prompts is usable, 1000 is paper-parity; the default
corpus is a commit-pinned FineWeb-Edu stream via the optional `datasets`
dependency (`pip install 'saklas[hf]'`).

Three read surfaces over either source, plus local fit and external fetch:

- **Readout** — `lens top` / `session.jlens_readout`: `softmax(W_U ·
  norm(J_l h))` ranks the vocabulary by what an intermediate activation is
  disposed to make the model say. Every readout surface also carries the
  **layer-aggregated** view (`core/jlens.py::aggregate_readout`): per-layer
  softmax calibrates away the cross-layer logit scale, then per token
  `strength = mean_l p_l(v)` (mean fitted-layer probability, 0..1) and a depth
  center of mass `com` (+ `spread`) weighted by the same per-layer probability
  — one channel backs every readout statistic, and the probability `p_l` is the
  one unit behind every lens surface (probe readings, gate scalars, the live
  wire, and `lens top -j`'s per-layer `strength` key). Top-k selection runs on
  the aggregated full-vocab strengths. `session.set_live(...)` on the lens
  instrument streams the per-layer matrix + aggregate chips every decode step
  (`TokenEvent.measurements` → `instruments.lens.readout`); the reader consumes
  the capture's latest slices post-forward at the token tap — no new forward
  hooks, so steering fast-path/compile eligibility is untouched. The default
  live layer set is every fitted layer, and `saklas serve` auto-enables the
  live lens at startup when the artifact exists (serve-side policy; the library
  stays opt-in). The dashboard's J-LENS tab is the server frontend: SOURCE
  (local/neuronpedia switch, fetch, background fit), STEER (`α jlens/<word>`
  cards), and PROBE (the merged workspace readout — pinned token probes above
  the live top-k aggregate cards, one card shape, live toggle in the header).
  `instrument.token_readout(node_id, raw_index)` is the loom-anchored replay
  behind the dashboard token drilldown: rebuild the node's prompt render + raw
  decode prefix, one capture forward under the recipe steering, the full
  fitted-layer top-k matrix + aggregate at the producing position
  (`steered=false` reads the unsteered counterfactual). Geometry and SAE have
  the same replay surface.
- **Steering atoms** — `jlens/<word>` is an ordinary `ns/name` atom; the
  direction for vocab id v at layer l is `W_U[v] @ J_l`, registered lazily
  across every fitted layer. `0.3 jlens/orange` pushes, `!jlens/fake` ablates,
  and it composes with every other term. Lens atoms run **hotter** than concept
  vectors: on gemma-3-4b α≈0.3 is the coherent sweet spot and α≥0.5 over-steers
  into repetition (a single sharp token direction, not a distributed contrast).
  Single-token words only — a multi-token word raises `MultiTokenWordError`
  listing the pieces. The `jlens` manifold namespace is reserved (authoring
  under it raises).
- **Probes + gates** — a `jlens/<word>` probe is a **readout-channel** probe,
  not a linear probe: it lands in the lens instrument's registry (never the
  Monitor — no whitener, no direction fold), and the reading is the token's
  standing in `softmax(W_U · norm(J_l h))` over all fitted layers. The reading
  is a `ScalarReading` with one channel — the mean fitted-layer probability
  `mean_l p_l(v)` ∈ [0,1] (`@when:jlens/fake > 0.01`) — plus the per-layer
  `p_l` trace and a probability-mass-weighted depth summary. Scoring is
  post-forward on the lens path: per-step readings ride the live-lens step's
  own logits, gate scalars are computed once per forward and stashed for the
  display step to reuse, and the finalize aggregate pools the last content
  token from the capture tail ring exactly like the monitor roster. A lens gate
  forces its per-step lens computation regardless of the live toggles; a
  lens-only gate does NOT force per-token *monitor* scoring.
- **Decomposition** — `lens decompose` / `session.jspace_decompose`: greedy
  sparse nonnegative pursuit of any steerable direction against the lens
  dictionary `W_U J_l` (never materialized), reporting the per-layer variance
  share — how verbalizable the direction is — plus the contributing tokens.

## Extraction

There is **one** extraction pipeline. A steering vector is the K=2 flat case of
a manifold, so `session.extract(concept, baseline, *, kind="abstract",
custom_system=None, …)` authors a 2-node discover-`pca` manifold under
`manifolds/<ns>/<name>/` — node 0 the positive pole, node 1 the negative — and
fits it via `core/extraction.py::ManifoldExtractionPipeline`. The corpus lives
as the manifold's two node groups. `extract()` returns `(canonical_name,
Profile)` where the `Profile` is the **folded** per-layer direction view of the
fitted 2-node manifold (`core/capture.py::folded_directions`) — the in-memory
steering-vector shape callers expect, with the manifold the single on-disk
artifact. A tensor cache hit (sidecar `nodes_sha256` matches the folder)
short-circuits the forward passes.

**Conversational corpora.** Each node's corpus is generated by
`session.generate_responses`: the model answers a fixed set of shared baseline
prompts *in character* — the concept rides a system prompt (by `kind`: abstract
→ "someone {c}", concrete → "{art} {c}", custom → the caller's `--system`
template) and a swapped assistant-role elicitation label — and extraction pools
the swapped-back `[user: prompt, assistant: response]` pairs in
**standard-assistant space**. The corpus is a `list[str]` of responses aligned
`response[i] ↔ baseline_prompt[i % k]` (`saklas/data/baseline_prompts.json`, 48
prompts; length must be a multiple of `k`). A shared one-paragraph length
directive (`_LENGTH_DIRECTIVE`) leads every system prompt — the persona at node
generation, the sole system for the neutral baseline, and the sole system at
capture — identical and symmetric across all four sites, so it is common-mode
that cancels at extraction while keeping responses from rambling past the token
cap, and it matches capture framing to generation framing. `kind` is recorded
per node (generation-time provenance; the fit never consumes it).
`--role`/`role=` opts into a persona-baselined fit (the explicit role overrides
the kind-derived label at both generation and capture). A **monopolar** concept
(`baseline=None`) authors a genuinely **1-node** `pca` folder; the pipeline
folds `concept − ν` — ν = the model's neutral activation mean (`layer_means`) —
into a 1-node neutral-anchored ray via `fold_directions_to_subspace`: no
discover-coords, no DLS, no synthetic second node. Neutral is the implicit
negative pole, sourced per-model at fit; `concept − ν` already cancels
common-mode like DiM, so the raw δ̂ basis is appropriate. `0.5 <concept>`
steers neutral → concept, resolving through the bare-label tier exactly like a
bipolar pole. The fit needs ν available (`layer_means`); when the neutral
corpus is stale it raises the same prerequisite the whitener does.

The per-layer fit derives the basis by **μ-centered PCA**; at K=2 the sole axis
is exactly the unit difference-of-means `δ̂` (PCA@2 ≡ DiM). The fitted manifold
stores each layer's `LayerSubspace` (mean, basis, real `node_coords`): the
activation-space magnitude lives in the neutral-anchored real coords
(`coord_pos − coord_neg = ‖δ_L‖`), and a separate per-layer Mahalanobis `share`
carries the cross-layer budget. The in-memory steering vector is the folded
view `δ̂_L · share_L` (`folded_directions`); for GGUF export llama.cpp's
uniform control-vector scalar reproduces the relative per-layer weighting from
those magnitudes.

Discriminative Layer Selection (Selective Steering, Dang & Ngo 2026 Eq. 9): a
layer/axis is kept iff the node projections relative to neutral straddle zero
(both signs present) — same-side layers encode concept *intensity*, not
*polarity*. `--no-dls` (available on `manifold extract` and `manifold fit`, and
as `dls=False` on the session) opts out. `compute_dls_axes` (N-node straddle)
backs the flat fit; at K=2 it is exactly the pos/neg opposite-sign test. Curved
manifolds have no pos/neg polarity, so they skip DLS — per-layer signal is the
apply-time share alone.

Bake/share metric: always `‖·‖_M` (Mahalanobis). Activation-space fit, `~`/`|`
projection, monitor reads, cross-model transfer, and `manifold compare` require
a `LayerWhitener` covering every scored layer (`covers_all`) and raise
`WhitenerError` otherwise — there is no Euclidean fallback (on real LMs the
Euclidean metric is rogue-dominated, so it would be a wrong answer, not a
degraded one). The sidecar `share_metric` / `subspace_metric` fields are
provenance and read `mahalanobis`, the one exception being a monopolar fit's
`subspace_metric`, which labels its raw-δ̂ basis `euclidean` (a basis-selection
label — `concept − ν` cancels common-mode by differencing, like DiM — not a
metric fallback). The `LayerWhitener` (`core/mahalanobis.py`) is built lazily
from cached neutral activations and drives the closed-form LEACE `~`/`|`
projection, the whitened/Fisher subspace fit, the whitened monitor reads, and
`manifold compare`.

## Injection

One injection kernel — `core/manifold.py::subspace_inject` — for every steering
term. There is no angular/additive mode, no `injection_mode`/`theta_max`, no
`_STEER_GAIN`. At dispatch (`SteeringComposer.compose_steering_entries`) each
term is classified and lowered: vectors / poles / `~`/`|` projections / `!`
ablations / affine `%` are composed into **one merged affine subspace per
trigger group** via `synthesize_subspace`; each curved `%` gets its own term.
`apply_to_model` then attaches per-layer `subspace_inject` groups.

The kernel decomposes `h = mean + h_par + h_perp` against the layer's affine
subspace and applies two ops: **along** *translates* the in-subspace foot by
the fixed `α·(target−neutral)` offset (preserving per-token spread — not a lerp
onto the absolute target, which loop-collapses at strong push; on a curved
surface it transports the off-surface residual `H_n` to stay normal at the new
foot), with a per-axis κ mask collapsing ablation axes toward 0 instead;
**onto** collapses `H_n` toward the surface (vacuous when the surface fills its
subspace, i.e. for every flat/affine term). With no σ-field it scales `H_n` by
`(1 − o)` — `o=1` lands on the zero-thickness wire. With a **fuzzy-manifold
σ-field** (`LayerSubspace.sigma_at`, curved fits only) it instead shrinks
`‖H_n‖` toward the local within-node thickness `σ(z)` — `o=1` lands one-σ off
the wire, a sample-like point on the surface's *typical set*, direction
preserved and never expanding a residual already inside the tube; `σ=0`
reproduces the `(1 − o)` collapse exactly. The off-subspace residual `h_perp`
is always kept verbatim, which is what lets a vector and N orthogonal manifolds
compose with zero cross-talk. A flat (affine) subspace takes an analytic
shortcut (foot = the projected coord, no Gauss-Newton / RBF), load-bearing for
throughput; a curved manifold runs a warm-started per-token nearest-point
foot-follower. fp32 throughout; the soft cap `‖h_new‖ ≤ 3·‖h‖` is the only norm
guard, and it rides the **curved path only** — the affine fast path skips it (a
flat fit can't extrapolate off-domain, so the bounded displacement can't blow
the norm).

Gain (`core/hooks.py`): the per-layer share is normalized to **mean 1** (`Σ_L
share_L = n_layers`) and `eff_along_L = share_L · gain`. The along-gain is
**path-specific** — `_SUBSPACE_GAIN = 16.0` on the affine path (whitened-unit
target, free push magnitude, overshoot-safe; live-calibrated on gemma-4-12b so
`α ≈ 0.5` lands the coherent band across concepts and personas) and
`_MANIFOLD_ALONG_GAIN = 4.0` on the curved path (the curved target is raw node
coords, so `eff_along` is a *fraction to the node* — `1.0` lands on it,
`norm_cap` bounds off-domain RBF extrapolation; calibrated stateless on
`months_loop%january` so `along=1.0` lands the vivid coherent sweet spot). For
**periodic `BoxDomain` fits** `eff_along` is clamped and share-weighting
dropped: `eff_along = max(0, min(1, along·_MANIFOLD_ALONG_GAIN))`, uniform per
layer, so no layer wraps past the target node. Non-periodic curved fits keep
the share-weighted unclamped path. `_MANIFOLD_ONTO_GAIN = 0.5` scales **onto**
only (the curved off-surface collapse). For an affine term the coefficient α is
folded into the translate *target* by `synthesize_subspace`, so α scales the
offset magnitude (unclamped); for a curved term α is the (clamped `[0,1]`)
`along` fraction. **Whitened along-normalization:** when a covering whitener is
present, `synthesize_subspace` makes the affine push target a *whitened-unit*
direction (`target = Σᵢ coeffᵢ·(B@dirᵢ)/‖dirᵢ‖_M`) and the share the *whitened*
displacement `‖Δ‖_M` — so the per-layer slide budget is the same whitened
amount for every target (`Σ_L eff_along_L = gain·n_layers`), linear in α.
Steering by raw-Euclidean node distance would span ~100× across targets (a
tight pole sits ~0.3 from neutral, a far persona centroid ~17), so one `along`
gain could never calibrate both; the whitened target makes `along` a
scale-stable strength knob across ranks/targets. Per-target *coherence*
variance (~2-3×, ARCHITECTURE §10 / "Open frontiers" below) remains. There is
**no lever / N correction** and, except for periodic domains, **no `[0,1]`
clamp / water-fill on `along`** (a high-signal layer is meant to overshoot the
target; the de-rogued whitened coords keep it controlled, with `norm_cap`
guarding the non-periodic curved path while the affine path relies on the
bounded target). `onto` stays clamped `[0,1]`. The affine lowering (share
normalization → eff_along → offset) is computed once
(`hooks.py::_lower_affine_subspaces`) and consumed by both the transient-hook
and compiled-offset paths, which a parity test pins numerically identical. The
dominant always-active affine case uses `SteeringHook._single_affine_fast`, a
fixed tensor-op sequence eligible for StaticCache and `torch.compile`. Curved,
phased, or gated steering uses the ctx-consulting general path so per-step
triggers and probe gates remain dynamic.

## Manifold steering

Manifold steering (Goodfire, arXiv 2605.05115): instead of a single linear
direction, fit an interpolant through per-concept activation *centroids* in a
low-dim subspace, then steer by moving the running activation's in-subspace
component onto a point of that surface. A straight A→B vector cuts through
low-density off-manifold regions; the manifold stays on the learned surface.

Geometry is a `ManifoldDomain` — an embedding of an n-D intrinsic manifold into
R^m plus a distance function. `BoxDomain` (per-axis open or periodic) covers
boxes/disks, cylinders, n-tori; `SphereDomain` covers S^n (chordal);
`CustomDomain` is the explicit-immersion escape hatch (and the identity carrier
for discover coords and synthesized affine subspaces). The per-layer
interpolant is one `r³` polyharmonic RBF; at n=1 over an open axis it
reproduces the natural cubic spline.

A manifold lives under `~/.saklas/manifolds/<ns>/<name>/` as `manifold.json`
(domain spec + per-node `{label, coords}` for authored; `fit_mode` +
hyperparams + `{label}` for discover) + `nodes/NN_<label>.json` corpora — by
hand or via the webui builder (`io.manifold_authoring`). `manifold fit`, the
webui fit action, and `POST .../fit` all run `ManifoldExtractionPipeline`: pool
each node's centroid, embed coords through the domain (or derive them for
discover mode), fit a per-layer subspace (flat `fit_affine_subspace` for
`fit_mode=pca`, curved `fit_layer_subspace` for authored/spectral —
Mahalanobis/Fisher PCA; the whitener must cover every selected layer), bake the
per-layer Mahalanobis share, write the per-model tensor. `--sae <release>`
reconstructs each centroid through the SAE before the fit; SAE weights load one
layer at a time and the fitted subspace is always model-space, so the hook
never touches the SAE. Discover fit-mode/hyperparameter overrides are merged
into `manifold.json` inside the same manifest lock that derives the cache key
and publishes the fit. Capture is tokenized/length-bucketed once, terminates at
the last requested transformer block, and writes curved-fit rows to a
source-dtype layer-major spool; geometry-only refits reuse a digested,
size-bounded token-exact per-model activation cache. `--layers` can restrict
the artifact to explicit indices or the workspace band. `min_nodes(n) = 2n+1`
for a curved fit (a flat `pca` fit needs only `k+1`); authored nodes must be
*poised* (affinely span the embedding).

`fit_mode` is one of five: `authored` (user supplies domain + coords; curved),
`pca` / `spectral` / `auto` (discover — labeled corpora only, coords derived
per-model; `pca` is flat, `spectral` curved, `auto` picks between them), and
`baked` (corpus-less, a precomputed direction written by `manifold bake`).
`auto` runs `select_topology` (`core/topology.py`) at fit time: flat vs curved
chosen by GCV in a shared whitened-reduced metric, plus periodic (`BoxDomain`)
axes detected by Vietoris–Rips H1 persistent homology coordinated off the
spectral eigenpairs, with a guarded single-cycle fallback for faint or
clustered rings PH's hole-size threshold misses (days-of-week and the like).
The resolved geometry is per-model and recorded in the sidecar
(`resolved_fit_mode` + the ranked `topology_candidates`). Sphere is
authored-only. The curved (`spectral`/`auto`) RBF fit is **penalized**: a
GCV-selected smoothing λ (`--smoothing auto`, the default) regularizes the
surface against centroid noise, `0` is exact interpolation, a float is a fixed
λ; the hot-path `eval_rbf` is unchanged (only the coefficients shrink).
`authored` stays exact (node = exact steering target).

Per-node `role` (slug `[a-z0-9._-]+`): the centroid is pooled under a
chat-template substitution that replaces the assistant-role label, so the fit
lives in role-baselined (persona) activation space. At steer time
`nearest_node_role` pipes the closest node's role through `session._active_role`
so the generation prefill applies the same substitution. Distinct implied roles
compose under soft-warn + highest-coefficient-wins; family-unsupported
(Mistral) raises `RoleSubstitutionUnsupportedError` at fit time.

**Whitened/Fisher subspace selection.** When the whitener covers every fit
layer the basis is selected by whitened PCA — maximize `vᵀS_b v / vᵀΣv` (the
LDA objective) rather than raw between-node variance, which on real LMs chases
the massive-activation (rogue) channels and leaves the subspace, its `mean`,
and the steering direction rogue-dominated. The Fisher ratio divides each
direction by its background variance, so rogue dims cancel (the same
cancellation DiM gets by differencing); the de-rogued subspace barely overlaps
the rogue-dominated `mean`, so the running-`‖h‖` artifact collapses for free.
Solved as the generalized eigenproblem via the whitener's Woodbury Σ⁻¹,
re-expressed Euclidean-orthonormal so the hot path is unchanged. Gated
all-or-nothing on `covers_all`: an activation-space fit requires the whitener
to cover every fit layer and raises `WhitenerError` otherwise. (The low-level
`_pca_basis` keeps a Euclidean SVD branch for the behavior-space naturalness
fit, which lives in output-distribution space where there are no rogue
activation dims; the activation-space callers never reach it.)

### Discover mode (auto-fit from a heap of corpora)

`manifold.json::fit_mode` is the discriminator: `"authored"` (user supplies
domain + per-node coords) vs *discover* — `"pca"`/`"spectral"`/`"auto"`, where
the user supplies labeled corpora only and coordinates are derived per-model at
fit time. `"auto"` defers the flat-vs-curved-vs-periodic choice to
`select_topology`; `"pca"`/`"spectral"` pin it. The derivation is
**layer-agnostic** — there is no reference layer. Each fit layer contributes
its whitened, node-mean-centered `(K, K)` Gram and the coords come from their
mean (the *consensus Gram*); whitening puts every layer in common units so the
average is signal-weighted — a layer where the nodes aren't separated drops out
on its own, so the layout draws on whichever layers carry the concept. PCA
eigendecomposes the consensus Gram and picks the smallest prefix whose
cumulative variance crosses `var_threshold` (default 0.70), capped at `max_dim`
(default 8); spectral reads pairwise distances off it, runs Laplacian eigenmaps
on a symmetric k-NN graph, and picks `k` by the eigenvalue-ratio cliff.
`fit_mode=pca` produces a flat affine subspace (no RBF); `spectral` (and
`authored`) produces a curved RBF surface. For a flat fit the per-layer
steerable subspace *is* its `max_dim`-capped layout span, so `max_dim` is the
only dim knob — `max_subspace_dim` applies only to the curved spectral fit. The
steer-time origin is always the projection of the per-model neutral mean onto
the subspace (the affine fit neutral-anchors the frame). The shared display
layout (`node_coords`) is **neutral-centered** to match: the flat fit
re-anchors it on neutral's landmark-MDS projection into the layout
(`neutral_layout_coord`) — a pure translation that leaves steering untouched
and makes `% 0,…,0` read as neutral, so the rack sliders and the probe-geometry
plot share neutral as their origin. Per-model coordinates are the
architectural shift: a Gemma fit and a Qwen fit produce different node layouts
for the same heap (stored as `node_coords` in the per-model safetensors).

`manifold generate <name> --concepts ... [--kind abstract|concrete|custom]
[--system TEMPLATE]` LLM-authors a discover folder via
`session.generate_responses` — each concept answers the shared baseline prompts
in character (one corpus per node; `--kind custom` rides the `--system`
template — `{c}` = concept — with no role swap, the system-only frame that
works on every model family). The shared baseline prompts hold topic
common-mode across nodes (`response[i] ↔ prompt[i % k]`), so the per-concept
centroids stay comparable without a per-manifold scenario set. `manifold fit
<name>` then fits — the two steps are deliberate (a flaky generation leaves
inspectable corpora). `manifold transfer` maps fitted layers through
neutral-derived Procrustes factors and re-bakes them in the target Mahalanobis
metric. The naturalness eval (`experiment naturalness`) fits a behavior-space
manifold over node output distributions in Hellinger space and reports the
per-step Bhattacharyya distance of a steered trajectory to it (low = natural;
`--compare-linear` scores a straight-chord baseline alongside).

### Bundled manifolds + coefficient regime

Complete bundled artifacts ship under `saklas/data/manifolds/`, materializing
into `~/.saklas/manifolds/default/` on session start via the io bootstrap
(`io/bootstrap.py`; process-scope no-op after the first call). The materializer
only advertises folders whose `manifold.json` and declared `nodes/*.json`
corpus files are all present, so a partial folder in the package tree is never
exposed as a default manifold:

- **17 concept manifolds** — bipolar 2-node `fit_mode=pca` axes, tagged by
  category (`epistemic`, `alignment`, `register`, `cultural`). These are the
  steering vectors: `0.5 formal.casual` steers toward `formal` (node 0). The
  `register` (7) and `cultural` (4) families are independent bipolar axes, not
  fused into a discover manifold — each has a designed opposite and the primary
  use is independent signed control. Affect is reserved for `emotions`.
- **`personas`** — discover `fit_mode=auto` resolving flat (a low-dim ~rank-8
  affine subspace — the auto selector independently picks flat for the persona
  fan), 107 persona archetype nodes (`assistant`…`vandal`) in
  assistant-baselined activation space; from Anthropic's Assistant Axis paper
  (arXiv 2601.10387). `max_dim` 8.
- **`emotions`** — a discover `fit_mode=auto` affect manifold over PAD
  (pleasure × arousal × dominance), 20 mood nodes. `auto` resolves the geometry
  per-model; on gemma-4-12B it resolves to a flat 3-D affect subspace. It
  materializes only after all 20 mood corpora exist.
- **`months`** — an **authored** periodic 1-D `BoxDomain` loop, 12 first-person
  month nodes (`january`…`december` at coords 0…11, December wrapping to
  January). The corpus is `kind=custom` seasonal-embodiment ("I am January…")
  pooled in standard-assistant space. The cyclic geometry is authored, not
  auto-discovered — the year is a known cycle, and a per-model auto-fit
  GCV-prefers a flat subspace, so the closed ring is declared in
  `manifold.json` (regen fills the corpora only). The corpus carries a
  warm↔cold seasonal axis and a period-2 solstice/equinox "extremeness" axis
  that lift the ring into a saddle.

Recommended α is vector-comparable: aim for `α ≈ 0.5`, tune up toward `α ≈ 1.0`
for stronger expression. (For an affine push term α is unclamped — it sets the
translate-offset magnitude; for a curved `%` term `along` clamps to `[0,1]`.)
Because the target is whitened-unit, the share is normalized to mean 1, and
there is no lever, a low-dim and a high-dim fit — a 2-node vector, `personas`,
`emotions` — land in the same α-band without per-fit retuning.
Architecture-level behavioral notes (hold across model families; α values are
qualitative, MPS is not bitwise deterministic so compare qualitatively):

- The whitened fit stays coherent at strong push where a Euclidean fit
  loop-collapses; caveman reaches terse primitive grammar, hacker a guarded
  intrusion register.
- Per-persona strength variance persists — a hard persona peaks near its
  coherence edge at α ≈ 1 where a robust one still has room; tune α down per
  target.
- Midpoints between distinct persona nodes (RBF interpolation at off-node
  coords) produce coherent *blended* persona content at the sweet-spot α — the
  interpolation-between-basins promise holds.
- The steering trajectory can pass through persona-adjacent attractor basins at
  low displacement (e.g. `personas%hacker` surfacing a cyber-security training
  cluster before locking into the clean persona). Low-α persona-drift is
  meaningful signal about the *model's* internal structure, not a saklas bug.

> **Open frontiers** (see `ARCHITECTURE.md` §10): the fitted `personas`
> subspace is a near-1-D "persona-ness" fan, so distinct personas can express
> the same generic intense register (a steering-access problem, not the rogue
> problem — whitening verifiably worked). What a single scalar gain can't
> unify is per-target *coherence* variance (~2× — a hard persona like hacker
> shatters at roughly half the effective gain a robust concept tolerates);
> `_SUBSPACE_GAIN = 16.0` is the coherence-first compromise, tagged a
> prototype.

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
- **Cast model** (`core/scene.py`): rendering goes through the per-session
  **scene grammar** (template autopsy + byte-exact round-trip validation;
  `session.scene_grammar`, None = raw-marker fallback). `generate(...,
  gen_seat="user")` has the model speak the user seat (the node lands
  `role="user"` with a recipe — generatedness is provenance, not a seat);
  `generate(None, ...)` continues from the current leaf with no committed turn
  (a/a, u/u sequences); per-turn `role_label`s are cast labels. Seats stay
  binary; arbitrary seat *sequences* and labels are free on validated families
  (gemma-2/3/4, llama, qwen, talkie), raw-marker fallback otherwise. WS:
  `generate_seat` on the generate frame. **Cast roster**: `LoomTree.cast` maps
  label → `CastMember` (`recipe` + `notes`; `session.set_cast_member(label,
  steering=…)` validates at authoring time). At generation the gen label's
  member recipe is the *weakest* tier — fills only unset call kwargs
  (`steering=""` = explicit unsteered; sampling merges field-wise via
  `merged_with`; regen overrides still win) — and the effective values are what
  land on the node's Recipe. Rides tree save (`cast` key, `tree_format` 2) and
  transcript v2. A steering scope whose role baseline differs from the send's
  `assistant_role` warns (`RoleBaselineMismatchWarning`; the steering role
  still wins). **Committed thinking**: `LoomNode.thinking_text` (commits via
  `append_*_turn(thinking=…)` / WS `commit_thinking` on the submit frame;
  generated nodes stamped at finalize when the family's think delimiters can
  re-render it) rides `messages_for(with_labels=True)` as a `"thinking"` key
  and renders through the stitcher under the family history policy; commit is
  refused up front (`SceneThinkingUnsupportedError`) on families that can't
  carry it. Generating into a non-assistant seat appends the seat's close
  segment as a stop string when it differs from the assistant's. Loom trees
  and transcripts load only their current formats (`tree_format` 2, transcript
  v2) — older files are rejected, not migrated.
- `generate`, `generate_batch`, `generate_sweep` always return `RunSet` —
  list-like, carrying `node_ids`/`grid`, with `.first` (the underlying
  `GenerationResult`) and common attributes delegating to it.
  `session.last_result` is the `GenerationResult`.
- `extract()` returns `(name, Profile)`; the `Profile` is the folded view of
  the 2-node manifold, and it round-trips through `Profile.save`/`load`.
  `extract_from_corpora` is the corpus-in sibling; `fit(folder)` fits a
  multi-node manifold and returns a `Manifold`; `bake(name, expression)` lands
  a corpus-less baked manifold from qualified additive scalar terms — the one
  implementation behind both `manifold bake` (CLI, offline) and `POST
  .../profiles/bake` (which adds the loaded-model fingerprint guard). `Profile`
  wraps `dict[int, Tensor]` (mapping interface plus `layers`, `metadata`,
  `save`/`load`, `merged`, `projected_away`, `cosine_similarity`).
- **Instruments**: `session.geometry` / `session.lens` / `session.sae` are the
  three read-side facades over one contract (`core/instruments/protocol.py`),
  registered in `session.instruments`. Uniform per-family surface:
  `attach`/`detach`/`names`/`specs`, `set_live(enabled, **extras) → LiveState`
  / `live_state`, `active_source`, `validate_gate`, `probe_hash`, and
  `token_readout(node_id, raw_index, …)` returning the finished
  `scope="replay"` measurements envelope. `session.add_probe(selector)` routes
  a reserved `jlens/<word>` or `sae/<id>` selector to its family and everything
  else to the Monitor. Lens/SAE probe readings are `ScalarReading`s (value +
  explicit unit + per-layer trace + depth summary); geometry readings are full
  `ProbeReading`s.
- `session.score_choices(messages, choices, *, assistant_prefix="",
  steering=None)` and `session.score_template(template, *, steering=None)`
  return the restricted-choice completion distribution (`ChoiceScores`,
  per-context for a template). Steering-aware (the distributional
  before/after). `template` is a `TemplateFolder` or a `<name>`/`<ns>/<name>`
  selector.
- `from_pretrained(..., on_progress=...)` narrates the load: model load,
  whitener build, neutral capture, and the bundled-probe bootstrap all report
  through one callback (the CLI passes a printer; the library default is
  silent).
- `Steering` is frozen; it carries no per-call metric override (`~`/`|`
  projection is Mahalanobis-only). There is no
  `injection_mode`/`theta_max`/`projection_metric`.
- `SaklasSession.__init__` takes a pre-loaded `PreTrainedModel`; use
  `from_pretrained` for HF loads. There is no `cache_dir=` — set `$SAKLAS_HOME`
  to relocate paths.
- Every saklas exception subclasses `SaklasError` while preserving its stdlib
  MRO, so `except SaklasError` catches the family and `except
  ValueError`/`RuntimeError` at existing sites still works (an invariant test
  walks every public module and enforces it). `user_message()` maps each to
  `(http_status, text)`; composition-time steering errors
  (`ManifoldArityError`, `OverlappingManifoldError`) are
  `SteeringCompositionError` at 422, parse errors `SteeringExprError` at 400.
- `GenerationResult.applied_steering` carries the canonical expression string
  (round-trips through `parse_expr`).
- `saklas/__init__.py` pins the public surface (`SaklasSession`, `Profile`,
  `Steering`, `SamplingConfig`, `Trigger`, `LayerWhitener`, the
  `RunSet`/`TokenEvent`/`ResultCollector` result types, the `EventBus` + event
  dataclasses, the `LoomTree`/`Recipe`/`Transcript` suites, their error types,
  the term types `ManifoldTerm`/`ProjectedTerm`/`AblationTerm`,
  `parse_expr`/`format_expr`, `ChoiceScores`/`ChoiceScore`, the selector errors,
  the Jacobian-lens suite, and the instrument facades
  `GeometryInstrument`/`LensInstrument`/`SaeInstrument` with
  `UnsupportedProbeChannelError`). `from saklas import X` is stable; private
  submodule paths are not.

## Cache layout

All state under `~/.saklas/` (override via `$SAKLAS_HOME`):

```
~/.saklas/
  neutral_statements.json              # user-editable; organic responses to the
                                       # baseline prompts (read-through from package)
  baseline_prompts.json                # user override for the shared prompts
  templates/<ns>/<name>/               # standalone templated-completion artifact
    template.json                      # slot, values, contexts:[{turns:[{role,content}], assistant}]
  manifolds/<ns>/<name>/               # THE concept + steering-manifold root
    manifold.json                      # name, source, fit_mode, per-node {label,kind,role?}, domain/coords or hyperparams, files{sha256}, template_ref?
    nodes/NN_<label>.json              # one JSON response list per node (response[i] ↔ baseline_prompt[i % k])
    <safe_model_id>.safetensors        # fitted per-layer subspaces (+ .json sidecar);
                                       # discover/baked also carry node_coords (the layout)
    <safe_model_id>_sae-<rel>.safetensors    # SAE-space fit
    <safe_model_id>_from-<src>.safetensors   # cross-model transfer
  models/<safe_model_id>/
    neutral_activations.{safetensors,json}   # neutral corpus (mult. of 48) × layers, fp32;
                                       # the single per-model neutral artifact — the
                                       # probe-centering mean is its per-layer X.mean(0),
                                       # the whitener covariance is built from the stack
    alignments/<safe_src>.{safetensors,json} # optional cross-model Procrustes map
    jlens/
      active.json                      # selected local/external J-lens source
      bindings/<provider>.json         # commit-pinned external source metadata
      local/default/
        manifest.json                  # atomic immutable-shard pointer
        checkpoint.json                # present only during a resumable local fit
        jlens*.gen-*.safetensors       # immutable fp32 J_l layer shards
    sae/
      active.json                      # selected local/SAELens source
      bindings/<release>.json          # provider binding + optional feature metadata
      local/<name>/                    # Saklas-trained weights + manifest
```

Conversation exports are browser-downloaded JSON files or explicit
`LoomTree.save(path)` targets; there is no `$SAKLAS_HOME/conversations`
autosave tree.

`manifold.json.files` is a sha256 map verified on load. A manifold folder can
hold multiple fitted tensors per model, distinguished by filename suffix:
`<safe>.safetensors` (raw DiM), `_sae-<release>`, `_from-<safe_src>`
(transfer) — at most one kind per file (no `pca` suffix). `tensor_filename` /
`parse_tensor_filename` in `io/paths.py` round-trip these. Bundled artifacts
materialize copy-on-miss through `io/bootstrap.py::materialize_bundled_artifacts`
(templates before manifolds — the single entry point every bootstrap site uses;
a test enforces it). io mutators invalidate the selector cache themselves;
callers need not.

## Performance invariants

These gate `test_smoke.py::test_throughput_regression` (steered ≥ 85% of
vanilla tok/s):

- **Hot-path hooks**: no Python allocation, no `.item()`, no CPU sync, in-place
  only. The whole generation loop is wrapped in `torch.inference_mode()`.
- **Norms use fp32** — fp16 sum-of-squares overflows at hidden_dim ≥ 2048.
  Applies to fit-time direction norms and the per-position norms inside
  `subspace_inject`. Centroid differences are taken in fp32 too.
- **One injection kernel.** Every steered layer runs `subspace_inject`. The
  dominant case — one always-active (`Trigger.BOTH`) affine group — takes a
  precomputed fast path (`SteeringHook._single_affine_fast`): one analytic
  slide + in-place write, no ctx read, no foot state.
  `torch.compile`/StaticCache graph capture is eligible for unsteered
  generation (`all_fast_path`) **and** for that static-affine steered case
  (`static_steerable` — the hook is a fixed tensor-op sequence and StaticCache
  never bypasses forward hooks). The compiled-offset and transient-hook
  lowerings share one implementation (`_lower_affine_subspaces`), pinned
  numerically identical by test. Curved / gated / phased steering keeps the
  ctx-consulting general path (DynamicCache, eager); curved manifolds pay a
  warm-started O(R) per-token foot solve. (StaticCache + the static
  single-affine steered fast path compile on MPS too — inductor's MPS backend
  fuses the per-layer kernels; only CUDA-graph capture (`reduce-overhead`)
  stays CUDA-only.) Curved `%` steering is materially the slowest path — per
  token it runs a Gauss-Newton foot solve + frame-rotation transport, an `n≥2`
  fit hops to CPU for the SVD on MPS, and it forfeits
  StaticCache/`torch.compile`. Prefer affine/flat terms where coherence allows.
- **Share baked at fit**, normalized to mean 1 at apply; gain constants and
  their calibration live in `core/hooks.py` (see "Injection").
- **Top-p via `torch.topk`**, not full-vocab sort; `top_k` (default 1024 cap)
  is a hard candidate-pool cap applied before top-p (llama.cpp/Ollama order).
- **Monitor capture is hook-driven**, inline with generation — no second
  forward pass. The unified `Monitor` reads every probe — flat or curved — as a
  whitened **coordinate**, emitting a full `ProbeReading` (coords + fraction +
  nearest + residual + assignment/membership + per-layer traces + depth stats).
  Execution is no-redundancy: the whole **flat** roster is scored together
  (one `Σ⁻¹h` Woodbury apply + block-diagonal matmuls + a single host transfer
  per layer), **curved** probes run the per-probe `invert_parameterization`
  foot solve (warm-started across decode tokens on the live path).
  Mahalanobis-only: the whitener must cover every probed layer (`covers_all`),
  else `WhitenerError`. Neutral activations are cached fp32 so `covers_all` is
  trustworthy as "finite factors everywhere". Per-token scoring runs
  **post-forward**, not inside the capture hook, so the host-side score read
  doesn't drain the device pipeline mid-forward. It is also **conditional on
  need** — the capture mode (incremental / lean-incremental / gating-subset /
  aggregate-only / full) trades when scoring runs against what consumes it;
  `core/AGENTS.md` has the mode table. A generation with probes attached but
  no per-token consumer pays zero per-token scoring (bounded tail ring, one
  aggregate pool at finalize).
- **Steering hooks are transient** — composed before generation, removed after
  (persistent compiled-offset buffers are zeroed between generations and
  detached at `session.close()`).
- **MPS discipline** — diffs on CPU, `torch.mps.empty_cache()` between
  extraction *batches*, end-of-loop sync to dodge Metal command-buffer reuse
  crashes. Long MPS loops need periodic queue drains: Metal reports queue
  exhaustion as an asynchronous command-buffer error that silently zeroes work
  instead of raising (the J-lens fit loop and its zero-row guard are the
  reference pattern).

## Tested architectures

`_ARCH_PROFILES` in `core/model.py` is the single per-architecture registry
(layer accessor + tested flag; `_LAYER_ACCESSORS`/`_TESTED_ARCHS` are derived
views, and an invariant test keeps them coherent). A one-time `UserWarning`
fires on load when `model_type` isn't known-working. Known working: `qwen2`,
`qwen3`, `qwen3_5` (+ `_text`/`_moe`), `gemma2`, `gemma3` (+ `_text`), `gemma4`
(+ `_text`/`_unified`/`_unified_text`), `mistral` (the text sub-model Mistral-3
and Ministral-3 actually load as), `gpt_oss`, `llama`, `glm`, `talkie`. Many
more are wired via accessors but untested — adding one is a single registry
entry. Architectures whose modeling ignores `past_key_values` auto-fall back to
O(N²) no-KV-cache generation with a one-time warning.

Role-augmented extraction (`:role-<name>` variant) and persona manifolds need a
chat template with a substitutable assistant-role label
(`core/role_templates.py::SEAT_HEADERS` — one registry for both seats, with
derived per-seat views). Supported: `qwen2`/`qwen3`/`qwen3_5` (ChatML),
`gemma2`/`gemma3`/`gemma4` (`<start_of_turn>`, label is `model`), `llama`,
`glm`, `gpt_oss`, `talkie` (`<|role|>` markers, GLM-shaped). Unsupported
(mapped to label-free; `apply_with_role` raises
`RoleSubstitutionUnsupportedError`): `mistral` / `ministral3` (positional
`[INST]`, no role label in the rendered string).

## Bundled concepts

17 curated concepts under `saklas/data/manifolds/<concept>/` — all bipolar
(2-node `pca`), each pole's corpus conversational responses to the shared
baseline prompts. Monopolar `extract` (`baseline=None`) is a genuine 1-node
fold against the neutral mean ν — a user `extract("agentic")` authors a 1-node
ray — but no monopolar concept ships bundled. Model-driven bundled regeneration
is unified under `scripts/regenerate_bundled.py` — one pipeline writing the
bipolar axes, `personas`, `emotions`, the authored `months` corpora, and the
neutral baseline; the fit is the separate `manifold fit` step. Partial
generation output is ignored by bundled materialization until every manifest
node has a corpus file.

Categories: `epistemic` (confident.uncertain, honest.deceptive,
curious.disinterested), `alignment` (refusing.compliant, sycophantic.blunt,
sincere.manipulative), `register` (formal.casual, direct.indirect,
verbose.concise, creative.conventional, humorous.serious, warm.clinical,
technical.accessible), `cultural` (masculine.feminine,
individualist.collectivist, traditional.progressive, religious.secular).

Known model-level axis entanglements (cross-model robust, weighted cosine via
`manifold compare`) — document for users, not probe-design failures:
- `masculine.feminine ↔ traditional.progressive` (+0.5–0.6) — Hofstede MAS read
  as traditionalism

`saklas/data/neutral_statements.json` holds the neutral baseline as organic,
no-persona/no-role responses to the same shared baseline prompts (a multiple of
the 48-prompt set; the shared one-paragraph length directive is its *only*
system prompt, so the framing it shares with the node corpora cancels at
extraction), regenerated via `session.generate_neutral_responses`; it backs the
probe-centering means + Mahalanobis whitener. `saklas/data/baseline_prompts.json`
(48 affect-neutral, topically-diverse prompts) is the shared elicitation set
every node and the neutral corpus answer.

## Package layout

`saklas/{core,io,cli,server,web,notebook}/` plus `saklas/data/` (bundled
artifacts) and `saklas/__main__.py`. `core` is the engine, `io` is persistence
+ distribution, `cli`/`server`/`web` are the interface layers, and `notebook`
holds the plotly/pandas figure helpers over the public result types (optional
`[notebook]` extra). The Svelte dashboard source lives at the repo's `webui/`
directory (peer of `saklas/`); its build artifact is committed under
`saklas/web/dist/` and must be rebuilt whenever `webui/src` changes.

## Testing

**GPU-required** (CUDA or MPS): `test_smoke.py`, `test_session.py`,
`test_jlens_gpu.py` — download `google/gemma-3-4b-it` (~8GB) on first run.
`device="auto"` picks cuda > mps > cpu; MPS runs ~3–5× slower so extraction
budgets are backend-specific. `test_smoke` owns the throughput regression.

**CPU-only**: the bulk of the suite — core dataclasses, steering-context
semantics, manifold format integrity + staleness, selector grammar, mocked HF
wrappers, GGUF round-trip, config loading, monitor scoring, instrument
protocol + golden measurement envelopes, eight-verb CLI dispatch,
OpenAI/Ollama/native servers, and loom tree/diff/filter/transcript.
