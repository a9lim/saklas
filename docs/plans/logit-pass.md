# logits, distributional surfacing, continuation mode

Plan for a UX pass that makes saklas more useful to the cyborgism-adjacent
audience without pulling it off its core "steering research" identity.
Audience priority is **steering > mechinterp > loom**: every decision
optimizes for "did my steering actually shift the distribution, or just
the argmax?" first, layer-aware probe surfacing second, base-model
multiverse third.

The unifying frame is **soft-everywhere applied to generation**. Saklas's
introspection-via-kaomoji methodology pivot retired hard-categorical
self-report in favor of distribution-vs-distribution scoring; the same
move at the generation layer is "stop showing the user just the chosen
token, ship the shape of the distribution that produced it." Logits on
the wire, distributional comparison in NodeCompareDrawer, distributional
deltas in fan results — all the same insight, applied wherever the UI
currently collapses a distribution into a single point.

Out of scope (deferred to separate plans):

- **Logit lens / tuned lens.** Per-layer logit projection wants a
  different hook architecture and (for tuned lens) trained translators.
  Worth its own plan when there's appetite.
- **Full N-up multiverse fan view.** The audit flagged this as the
  single biggest framing gap; it's deferred because (a) it's weeks of
  Chat-layout surgery and (b) logits + better fan-result rendering
  cover ~70% of the use case. Reassess after Phase 6 lands.
- **Tree-as-primary view.** The "TUI loom screen but in the webui"
  reframing is the right long-term shape but a separate effort.

---

## Implementation status

**Phase 1 — DONE** (2026-05-14). Engine + wire + data model + CLI/YAML +
tests. Full non-GPU suite green (1425 passing, 6 skipped, 0 failed). No
visible UI change until phase 2+ consumes the wire. Committed as a
hard-break refactor per the project's "no dual shapes" discipline.

**Phases 2–6 — DONE** (2026-05-14). Webui Svelte surfaces + a new
server route for phase 5's lazy joint-logprobs join. Full non-GPU suite
green (1474 passing, 1 skipped, 0 failed — 15 new tests for the joint-
logprobs math). Webui `dist/` rebuilt and committed per the project's
source-of-truth discipline.

- **Phase 2**: TokenDrilldownDrawer gains a `[probes | logits]` tab
  strip; logits tab renders the ranked top-K alts table with chosen-row
  tinted plus an empty state linking to the SamplingStrip's new `alts`
  toggle.  `SamplingState.return_top_k` flips between 0 and 8
  (Decision 1).  Pipes `logprob`/`top_alts`/`mean_logprob` through the
  WS shape and the chat snapshot.
- **Phase 3**: `surprise` highlight mode for Chat.svelte token tinting
  via `-logprob / (1 - logprob)` mapping through the existing
  `scoreToRgb` ramp (Decisions 4 + 5).  `compare-two` also accepts
  surprise as the B stripe.  TUI parity: a three-state highlight-mode
  cycle on `Ctrl+Y` / `Ctrl+Shift+Y` (`off → probe → surprise`) plus
  the `SURPRISE_PROBE` sentinel on `chat_panel`'s markup-cache path;
  per-token `event.logprob` rides through the `_ui_token_queue` (the
  trailing `is_shadow` flag shifts to tuple index 7).  (Phase 3 first
  shipped this as a `/surprise` slash command — see the deviations
  note below.)
- **Phase 4**: Loom edge weighting + sibling sort + filter help.
  `loomUiState.weightMode` (`"none"|"confidence"|"surprise"`) drives
  `LoomEdge`'s stroke-width / opacity scaling; `LoomNode` gains a
  numeric `mean_logprob` badge when weight mode is on.
  `applyTreeFilter` peels `sort:surprise`/`sort:confidence` directives
  off client-side and reorders sibling DFS order.  `?` button surfaces
  the filter grammar inline (Decision 8).
- **Phase 5**: NodeCompareDrawer per-aligned-token cross-evaluation
  table.  New core module `saklas.core.joint_logprobs` runs one forward
  pass per branch and surfaces self / cross logprobs, rank-1-change
  flag, and a top-K-truncated approx KL (Decision 9 — lazy / cached on
  the session keyed by sorted `(a_id, b_id)`).  New route
  `POST /saklas/v1/sessions/{id}/tree/joint_logprobs` held under
  `acquire_session_lock`.
- **Phase 6**: Siblings rollup at the top of NodeCompareDrawer — one
  row per id with `mean_logprob`, rank-1 % unchanged, and mean approx-
  KL vs the baseline (left-most node).  Renders for both N=2 and the
  "compare children" N-way mode (the v2.3 sweep replacement surface).

### Phase 2-6 deviations from spec

- **Vitest tests not landed.** No Vitest harness exists in the webui;
  setup is a separate scope.  Type safety is enforced via
  `svelte-check` (zero new errors); the only failures are two pre-
  existing `string | null` typing issues unrelated to this work.
- **TUI surprise mode: `/surprise` → highlight-mode cycle.** Phase 3
  first shipped a `/surprise [off]` slash command — the TUI's
  highlight surface was then a binary `Ctrl+Y` toggle over a single
  probe name, with no mode-cycle infrastructure for `surprise` to slot
  into.  A later pass added the three-state highlight-mode cycle
  (`off → probe → surprise` on `Ctrl+Y` forward / `Ctrl+Shift+Y`
  backward), folded the binary toggle into it, and **removed
  `/surprise`** — the cycle is now the sole entry to surprise mode.
  (`Ctrl+H`, the originally-natural chord, is unusable: terminals send
  `0x08` for it, which Textual maps to `backspace` before binding
  resolution.)  Cursor-token + status-footer `lp=-2.34` readout
  deferred — the TUI has no cursor-token concept and building one is
  separate scope.
- **`sort:` grammar is client-side.** The server filter grammar
  doesn't grow a sort directive; `applyTreeFilter` peels `sort:` terms
  out before sending to the server.  The sibling DFS in the sidebar
  reorders locally.  Server route grammar stays unchanged.
- **Phase 6 server-side sweep test deferred.** The plan calls for
  `tests/test_sweep_metrics.py` (server-side monotonic-KL check on a
  fan).  The CPU test surface for sweeps requires a real model
  load — out of scope for this CPU-only test suite.  The KL math is
  covered indirectly by `tests/test_joint_logprobs.py::test_approx_kl_*`.
- **Surprise scale swapped to `1 - probability`.** Decision 4's
  `-logprob / (1 - logprob)` was a smooth rational mapping but obscured
  what the number meant; the replacement `surprise = 1 - exp(logprob)`
  is the same `[0, 1)` range with a direct reading ("how unlikely was
  this token"), so researchers don't have to mentally invert a ratio.
  Applied uniformly: `surpriseScore` (inline tint), `_surprise_score`
  (TUI mirror), `LoomEdge` intensity, and `ActivationAtlasDrawer`
  picker spark (which had been on a separate `-logprob / 12` linear
  ramp — now folded into the same formula).

**Phase 7 — deferred to its own plan** as the spec calls for.

### Phase 1 deviations from spec

A few things landed slightly off-spec; all are documented inline in the
code and align with the project's discipline guidelines.

- **Engine knob composition rather than rename.** The plan calls for
  `SamplingConfig.return_top_k` as the only knob; OpenAI's `logprobs`
  was already wired through the same engine path. Both fields coexist
  on `SamplingConfig` and `_generate_core` composes
  `lp_count = max(return_top_k, logprobs or 0)`. The OpenAI route still
  uses `logprobs` (back-compat at the request shape level); the
  loom/webui path uses `return_top_k`. Single engine knob, two surface
  expressions of the same intent.
- **Chosen-logprob capture widened** to fire whenever the engine's
  log_softmax already runs (any `on_token` consumer or explicit
  `logprobs` request). Costs one extra `.item()` per step on the
  streaming-without-logprobs path; well below the throughput
  invariant's noise floor since the loom path already pays the entropy
  `.item()` and the `token_id.item()` per step. This is what makes the
  plan's "K=0 = logprob only, minimal additive cost" promise true in
  practice.
- **Hard-break shape change** on `TokenEvent.top_logprobs → top_alts`
  and `GenerationResult.logprobs` inner tuple. Per
  `~/.claude/CLAUDE.md`'s "delete legacy shapes in same change"
  guidance and matching the project's stated preference (per
  `AGENTS.md`'s history of hard-break refactors).
  `_render_logprobs_chat`/`_completions` in the OpenAI route updated
  to consume `alt.text` directly — no redundant retokenization.
- **Session-level default for `return_top_k`** stored as
  `self._default_return_top_k`. Per-call `SamplingConfig.return_top_k`
  > 0 wins; K=0 inherits this session-level value. There's no way to
  explicitly request K=0 over a non-zero session default through this
  knob — the inheritance semantic is intentional and documented.
  Callers who need single-call suppression set
  `sampling.logprobs=0` (chosen-only) instead.
- **`mean_logprob` computed inline** in `_generate_core` from the
  accumulator inside `_token_tap`, not at `tree.commit_node` time
  reading from `node.tokens`. Reason: `LoomTree.append_token` is
  defined but **never called** from the engine path — `node.tokens`
  stays `None` for live gens. The chosen-token logprob lives in
  `logprobs_list` / the `_token_tap` accumulator, which `_generate_core`
  has scope on; `_finalize_generation` receives the pre-computed
  floats as kwargs. Same shape on the wire and in the tree.
- **`_need_tap` gate unchanged.** When `return_top_k > 0` but
  `on_token is None` and no other consumer is live, the engine still
  computes top-K alts that go nowhere. Decided to keep this as a
  small wasted-work edge case rather than adding another branch in
  the composition; loom paths always have `on_token`.

### Surfaces touched (file list for reviewers)

```
saklas/core/results.py         # TokenAlt + TokenEvent.top_alts + GenerationResult.logprobs shape
saklas/core/sampling.py        # SamplingConfig.return_top_k + clamp
saklas/core/generation.py      # decode top-K texts inline, emit list[TokenAlt]
saklas/core/session.py         # _token_tap shape, lp_count composition, mean_logprob inline, _default_return_top_k
saklas/core/loom.py            # LoomNode.mean_logprob/mean_surprise + finalize_assistant kwargs
saklas/server/saklas_api.py    # WS token event surfaces logprob/top_alts; done event carries mean_logprob; WSSamplingParams.return_top_k
saklas/server/app.py           # OpenAI render functions consume alt.text directly
saklas/cli/config_file.py      # YAML return_top_k:
saklas/cli/parsers.py          # --top-k-alts on tui/serve
saklas/cli/runners.py          # YAML/CLI value → SaklasSession.from_pretrained(return_top_k=...)
saklas/__init__.py             # re-export TokenAlt
saklas/core/AGENTS.md          # docstring update for new TokenEvent/GenerationResult shapes
tests/test_logits.py           # 32 new tests (CPU-only)
tests/test_results.py          # updated for top_alts rename
tests/test_tui_commands.py     # stub _Event updated
```

---

## Architectural choice: top-K on the wire, opt-in by default

Three options for shipping distributional info per token:

1. **Full vocab logits per step.** Truthful, useless. 250k floats × 256
   tokens × 1 turn = 256 MB on the wire per generation; persisting it
   to localStorage is a non-starter.
2. **Top-K alternatives + chosen-token logprob.** Compact (K=8 ≈ 60 KB
   per turn), lossy in a controlled way — exact for the head of the
   distribution, summary statistics for the tail. Already cached by the
   sampling path (top-p uses `torch.topk` with `k = min(config.top_k or
   1024, vocab)`), so the engine cost is one slice.
3. **Chosen-token logprob only.** Cheapest, but kills the "see what
   else the model was considering" use case that's the whole loom
   point. Rejected.

Pick (2) with K configurable via `SamplingConfig.return_top_k`. Default
K=0 (logprob only — minimal additive cost over today). K>0 enables the
top-alternatives surface. `SaklasSession.from_pretrained` exposes the
default; CLI `--top-k-alts N` on `tui` / `serve`; YAML key
`return_top_k:`. The webui's chat header gets a "show alts" toggle
that flips the live SamplingConfig to K=8 and re-arms hooks for
subsequent generations.

The opt-in default keeps the existing wire/storage cost identical for
users who don't care, and bounds the cost when they do.

### What "logprob" means here

Chosen-token natural-log probability under the post-temperature,
post-top-p, post-top-k distribution that sampling actually drew from.
That's the calibrated quantity for the question "how surprising was
this token *to the configured sampler*." We don't ship the
pre-temperature softmax — temperature is a researcher knob and the
displayed surprise should reflect what they configured, not a
counterfactual sampler.

(For mechinterp users who want the raw-model probability, add a
`raw_logprob: float` field alongside `logprob` in a follow-up; cheap
to compute since the unmodified logits are right there.)

---

## Core data

```python
# saklas/core/types.py (new dataclass; merged into existing TokenScore module)

@dataclass(frozen=True)
class TokenAlt:
    """One alternative the model considered at this position."""
    id: int                          # token id
    text: str                        # decoded text
    logprob: float                   # post-sampler log-probability

@dataclass(frozen=False)
class TokenScore:
    """Existing — extended for logits."""
    text: str
    id: int
    logprob: float | None = None     # NEW; chosen-token logprob
    top_alts: list[TokenAlt] | None = None  # NEW; len == return_top_k or None
    per_layer_scores: dict[str, dict[str, float]] | None = None  # existing
    probes: dict[str, float] | None = None  # existing aggregate readings
    thinking: bool = False
```

Two new fields, both optional. `logprob` populates whenever sampling
ran (essentially free); `top_alts` populates only when
`return_top_k > 0`. Both are `None` for legacy tokens (replayed from
old transcripts), so renderers must `?? null`-guard.

### LoomNode aggregate

Add to `LoomNode`:

```python
mean_logprob: float | None = None     # over all assistant tokens
mean_surprise: float | None = None    # -mean_logprob, cached for sort
```

Computed at `tree.commit_node` time from the assembled token list.
`None` when logprob wasn't captured (back-compat replay). The sidebar
gets a "sort by surprise" mode that reads this field — the
"which branches was the model most confident in?" question becomes a
keyboard sort, not a manual scan.

### Wire protocol

WS `token` event extends with `logprob`/`top_alts` fields, both
optional. WS `done` event extends with `mean_logprob` per-turn rollup.
Subscribers that don't care ignore the new fields. No version bump
needed (additive).

REST `GET /saklas/v1/sessions/{id}/tree/{node_id}` returns the full
node payload including new fields, same shape.

REST `POST /saklas/v1/sessions/{id}/experiments/fan` extends
result rows with `mean_logprob`, `kl_from_baseline`, `n_rank1_changed`
(see Phase 6).

---

## Phase 1 — Engine-side capture + WS emission

**Why first:** without logits on the wire, every later phase has nothing
to render. No visible UI change after this phase alone; the field is
populated and ignored downstream until Phase 2.

### Changes

- `SamplingConfig.return_top_k: int = 0` — new field. Validated in
  `SamplingConfig.__post_init__` (clamp to `[0, 256]` — top_alts
  beyond 256 is data the user can't act on and bytes that nobody wants).
- `saklas/core/session.py::generate_steered` — at decode time, after
  the existing `torch.topk` for sampling, pull `chosen_logprob =
  log_softmax(logits)[chosen_id]` and (if `return_top_k > 0`) the K
  highest-probability alternatives. Both are already in cache from
  the sampling path; this is one extra `gather` and one slice.
- `EventBus` `TokenStreamed` event — extend with `logprob`,
  `top_alts` fields.
- `LoomNode.mean_logprob` — populate at `tree.commit_node`, average
  the non-None per-token logprobs over the assistant span (skip
  thinking tokens — surprise during thinking is a different signal
  and conflating them muddies the per-turn rollup).

### Tests

- `tests/test_logits.py::test_chosen_logprob_matches_softmax` — for
  a fixed seed + greedy sampling, the captured logprob equals
  `log_softmax(raw_logits)[chosen_id]` to fp32 precision.
- `tests/test_logits.py::test_top_alts_sum_le_1` — softmax of the
  top-K alts sums to ≤ 1 (sanity that we're not double-applying
  log/exp).
- `tests/test_logits.py::test_disabled_means_no_alts` — `K=0`
  produces `top_alts is None`, not `top_alts == []`.
- `tests/test_logits.py::test_mean_logprob_skips_thinking` — for a
  thinking-mode generation, mean_logprob covers only the response
  span.

### Backward compat

- Old transcripts replay with `logprob=None`, `top_alts=None`,
  `mean_logprob=None`. Every UI surface has to `?? null`-guard.
- `SamplingConfig` is frozen but the new field has a default, so
  existing call sites that build a `SamplingConfig` by keyword don't
  break.
- `EventBus` subscribers receive the new fields as added kwargs;
  this is additive and doesn't break existing handlers.

---

## Phase 2 — Token drilldown: logits tab

**Why second:** the drilldown is the natural place for the dense
distributional view, and it ships the soft-everywhere display before
we commit to inline pixels.

### Changes

`drawers/TokenDrilldownDrawer.svelte` gains a tab strip at the top:

```
[ probes (per-layer) ]  [ logits ]
```

Probes tab is the existing per-layer × per-probe heatmap, unchanged.
Logits tab renders:

```
chosen: " hello"  logprob = -2.34  (rank 1 of 8)

rank   token        logprob    p          Δ from rank 1
1   *  " hello"     -2.34      0.0961     —
2      " hi"        -2.78      0.0620     -0.44
3      " hey"       -3.12      0.0440     -0.78
...
```

The chosen-token row gets a `*` and a background tint. If `top_alts`
is None (not requested or replayed-legacy), the tab renders an empty
state with a "enable in sampling header" link that flips the toggle.

Each alternative row has a "branch from this alt" button (Phase 4
follow-up — wire to `loomBranch` with the alt-text pre-filled in the
text buffer).

### Tests

- `tests/test_drilldown_logits` — Vitest component test: render the
  drilldown with synthetic `top_alts`, assert chosen row tinted,
  ranks correct, deltas correct.

---

## Phase 3 — Inline surprise indicator on chat tokens

**Why third:** answers the steering-researcher question "where in this
turn did the model do something unusual?" without requiring a click
per token. Subtle by default — researchers who don't care don't see it.

### Changes

`panels/Chat.svelte` token-rendering layer gets a new highlight mode in
the existing highlight dropdown: alongside `none / aggregate / compare-two`,
add `surprise`. When selected:

- Token background tint = `scoreToRgb(-logprob / (1 - logprob))` per
  Decisions 4 + 5: the `1 / (-logprob + 1)` smoothing maps `[0, ∞) →
  (0, 1]` with no arbitrary cap, then we invert (`1 - that`) so high
  surprise = high tint, then feed the positive half of the existing
  diverging probe color scale. No new color ramp, no new module —
  reuses `lib/tokens.ts::scoreToRgb`.
- Hovering a token shows a tooltip `logprob = -2.34, rank 1 of 8`
  if `top_alts` is present, just `logprob = -2.34` otherwise.
- Compatible with existing per-token click → drilldown.

The `compare-two` mode also gains a logit option: pick "logprob" as
one of the two stripe sources, get a side-by-side "probe reading vs.
surprise" tint.

### TUI parity

The TUI gets the surprise highlight mode too — the engine work in
Phase 1 already populates `logprob` on tokens regardless of frontend,
and the TUI's existing token-coloring path (the highlight selector in
the chat screen) extends symmetrically. Adds `surprise` to the
TUI's highlight-mode cycle and renders the same `1 / (-logprob + 1)`
mapping over the TUI's existing 256-color rich-text scale (whichever
of the diverging palettes maps cleanest to a `[0, 1]` input —
implementation detail, but the shape mirrors the webui's
`scoreToRgb` reuse). Tooltip-on-hover doesn't apply in the TUI; the
status footer was to get a `lp=-2.34` readout for the cursor-token
instead (deferred — see the deviations note: the TUI has no
cursor-token concept). The surprise mode is reached through the
three-state highlight cycle (`Ctrl+Y` / `Ctrl+Shift+Y`).

The drilldown / fan / NodeCompare / loom-edge polish is webui-only
in this pass — the TUI's loom screen and compare drawer are simpler
surfaces and a parity pass for them is a separate effort. The
inline surprise mode is the one piece worth porting now because
it's a one-line color-table change with disproportionate
researcher value.

### Tests

- Vitest snapshot of the surprise-mode chat render with synthetic
  tokens at logprobs spanning the scale.
- Reactivity: changing the highlight mode dropdown re-renders without
  a full panel remount.

---

## Phase 4 — Loom edge weighting by mean log-prob

**Why fourth:** answers "which branches was the model most confident
in?" at a glance, in the place a loom-native user is already looking
(the tree).

### Changes

`panels/loom/LoomEdge.svelte` gains an optional `weight` prop driven
by `loomTree.weightMode`:

- `none` (default): edges render as today.
- `confidence`: edge stroke-width and opacity scale with parent →
  child `mean_logprob` (low surprise = thick / opaque; high surprise =
  thin / faint).
- `surprise`: inverse of confidence — thick for surprising children
  (the "this branch went somewhere weird" view).

The mode toggle is a small dropdown in the sidebar's filter strip,
next to the existing filter input. State lives in `loomTree.weightMode`
and persists with the chat snapshot.

`panels/loom/LoomNode.svelte` gains a numeric badge in the corner:
`mean_logprob` formatted to 2 decimal places, only visible in
`weightMode != "none"`. Researchers can read the actual number, not
just eyeball the edge.

Sort grammar: the sidebar's existing filter input gets a sort suffix
`sort:surprise` / `sort:confidence` that rearranges siblings by
`mean_logprob`. Documented in the filter help (Decision 8 below
unblocks this).

### Tests

- Vitest: render LoomEdge with weight prop, snapshot SVG attributes.
- Vitest: filter `sort:confidence` reorders siblings deterministically.

---

## Phase 5 — NodeCompareDrawer logit-shift columns

**Why fifth:** this is the steering-researcher kill-shot. "Did honest+0.5
actually shift the distribution at the contentious tokens, or just the
argmax?" becomes one drawer.

### Changes

`drawers/NodeCompareDrawer.svelte` already renders per-token alignment
between two sibling completions. Extend the per-token row with two
new columns:

```
A: " kind"      (rank 1, lp=-1.23)     <hover: top-3 alts>
B: " honest"    (rank 1, lp=-2.41)     <hover: top-3 alts>
Δ rank-1: changed   Δ logp(A's token in B): -3.12   approx KL: 1.84
```

`Δ logp(A's token in B)` answers "what would B have given the token A
chose here?" — requires that B's `top_alts` includes A's chosen token
OR that we extended the wire to ship the chosen token's logprob under
B's distribution as a join key. Cleanest: a server route that, given
two node ids, re-evaluates each branch's logits at the other's chosen
token positions and returns the joint logprobs. **Lazy / on-demand
per Decision 9** — fired only when the user opens NodeCompareDrawer
for a specific pair, results cached keyed by `(node_a_id, node_b_id)`
for the session lifetime. The eager fan path stays cheap; the join
cost is paid only for pairs the researcher actually inspects.

`approx KL` = top-K-truncated KL between the two distributions, using
overlap of the top alts. Documented in the drawer as approximate
because the tail isn't observed; this is signal not measurement.

The N-way compare path (multi-sibling) gets the same columns pairwise
against the leftmost column.

### Tests

- `tests/test_node_compare_logits` — Vitest, synthetic tokens, assert
  KL is symmetric to within ε and that "rank-1 changed" highlights
  correctly.
- `tests/test_server_joint_logprobs.py` — server-side join over two
  recorded sibling completions.

### Decisions deferred from this phase

- Whether the inline "approx KL" should be JSD instead. JSD is bounded
  and symmetric, better for visual scaling; KL is what the literature
  uses. Pick after Phase 5 ships and we see how it reads.

---

## Phase 6 — Sweep / fan: distributional metrics in the result table

**Why sixth:** the existing fan results show text only. With logits on
the wire, every fan result already carries the data; surface it in the
sweep result table.

### Changes

The sweep/fan result rendering (now living as loom siblings under one
shared anchor per AGENTS.md sweep deprecation note) gets per-sibling
rollup columns in the sidebar's "compare children" drawer:

| α    | text preview | mean_logprob | rank-1 % unchanged | KL from baseline |
|------|--------------|--------------|--------------------|--------------------|
| 0.0  | (baseline)   | -1.84        | —                  | 0.00               |
| 0.3  | …            | -1.92        | 0.83               | 0.41               |
| 0.5  | …            | -2.18        | 0.61               | 1.12               |
| 0.7  | …            | -2.84        | 0.42               | 2.31               |

Baseline = α=0 sibling if present, else first sibling. KL is approx-KL
as in Phase 5.

This makes the soft-everywhere insight literal: the researcher sees
"my α=0.5 changed the distribution at 39% of positions" rather than
just reading text and guessing. The argmax-only failure mode becomes
visible as "rank-1 % unchanged is high but text reads steered" — that's
sampling noise, not actual intervention.

### Tests

- `tests/test_sweep_metrics.py` — server-side: a fan over a fixed
  prompt with known steering vectors produces monotonic KL.
- Vitest: sweep result table renders with sortable columns.

---

## Phase 7 — Continuation mode (deferred to its own pass)

**Why scoped here but not built in this pass:** continuation mode is
the loom-native shape; not building it leaves a real gap for the
base-model audience, and it _will_ get built — just not stapled to
this one. The reason for the split: base models work meaningfully
differently from chat-templated ones (no role tags, no system
prompt, no thinking-mode chrome, raw text in / raw text out, tree
nodes are buffer chunks rather than turns), so the design surface
deserves its own focused pass rather than being negotiated alongside
the logit/distributional work. Phases 1-6 are the load-bearing work
of this plan; Phase 7's sketch lives here as the seed of a follow-up
plan (`docs/plans/continuation-mode.md` when it opens).

### Sketch

`SaklasSession.from_pretrained(continuation=True)` skips
`apply_chat_template`. Input is raw text appended to the running
buffer; output is generated text appended to the buffer. No system
prompt, no role tags, no thinking-mode chrome.

CLI: `saklas tui --continuation`, `saklas serve --continuation`.
Webui: a topbar "mode" toggle "Chat / Continuation" that swaps the
chat panel for a single editable text buffer. Tree nodes become text
chunks rather than turns; loom ops (regen, branch, fan) work the same
but on chunks. Steering, probes, logits — all unchanged.

The whole thing is a separate `panels/Continuation.svelte` view that
shares the loom sidebar and the steering/probe racks. Persistence
keys off `saklas.continuation.v1.<model_id>` so the two modes don't
clobber each other.

### Specific decisions to lock down before building Phase 7

- Does the loom tree share between modes, or is there one tree per
  mode? (Default: per-mode tree.)
- How does the steering recipe attach to a continuation chunk? (Same
  as today's assistant-turn `recipe`, but the "user prompt" anchor
  doesn't exist — anchor is the buffer position.)
- Token drilldown semantics: same as chat mode.

---

## Decisions

1. **Default K for `return_top_k` when enabled: 8.** Enough to see the
   head of the distribution, fits in a tooltip, ~60 KB per turn at 256
   tokens. _Note: K=4 is a reasonable fallback for models whose
   distributions drop off fast (Gemma-style sharp posteriors); revisit
   the default if empirical telemetry shows the K=5..8 alts almost
   always have negligible mass._ _Status: locked._

2. **Wire protocol: piggyback on `token` event** with optional
   `logprob`/`top_alts` fields. Keeps subscribers' state machine
   simple, avoids ordering races between `token` and a parallel
   `logits` event. _Status: locked._

3. **Storage on chat snapshot (`localStorage`): persist everything,
   including `top_alts`.** Top_alts is exploration data the researcher
   may want to revisit on a historical token, not just live; the
   ~60 KB/turn cost stays well inside the existing 5 MB
   `_LOCALSTORAGE_SOFT_BUDGET` for typical session lengths, and the
   existing budget toast handles the long-session pathology. The
   "re-fetch on demand" alternative was a premature optimization —
   server-side regen of historical logits requires re-running the
   forward pass, not just a cache lookup, so persisting beats
   recomputing. _Status: locked._

4. **Inline surprise scale: `surprise_value = 1 / (-logprob + 1)`,
   reusing the existing color machinery.** Maps `[0, ∞)` smoothly to
   `(0, 1]` with no arbitrary cap — confident tokens (logprob → 0)
   land at 1, vanishingly unlikely tokens (logprob → -∞) approach 0.
   Plugs straight into `lib/tokens.ts::scoreToRgb` without a new color
   ramp. Polarity choice for the renderer: invert to `1 -
   1/(-logprob + 1) = -logprob / (1 - logprob)` so high surprise =
   high tint (the actionable signal — researchers are looking for
   the surprising spans, not the predictable ones). _Status: locked._

5. **Inline surprise color: reuse `scoreToRgb` (positive half of the
   diverging probe scale).** Supersedes the earlier amber single-hue
   proposal — Decision 4 gave us a bounded `[0, 1]` value, so reusing
   the existing color machinery saves a code path and keeps the chat
   tinting visually consistent. The "diverging scale implies a
   midpoint" concern from the original proposal is sidestepped because
   we only ever feed it the positive half. _Status: locked._

6. **`mean_logprob` skips thinking tokens.** Expose
   `mean_logprob_thinking` separately for users who care. Surprise
   during thinking has a different distribution than during response;
   conflating them muddies the per-turn rollup. _Status: locked._

7. **`raw_logprob` (pre-sampler) field: deferred.** The post-sampler
   logprob is what answers "what was this sampler likely to draw?" —
   the more common research question. Add `raw_logprob` as a follow-up
   if mechinterp users ask for it. _Status: locked (deferred)._

8. **Filter grammar help affordance: a `?` button** next to the filter
   input opens a small popover with the grammar (`agg:<probe> <op>
   <num>`, `sort:surprise`, `sort:confidence`, `starred`,
   `text:<query>`) and 3 example queries. Lifts the
   undiscoverable-feature problem flagged in the audit. Lands as part
   of Phase 4 since it adds the new sort grammar. _Status: locked._

9. **Joint-logprobs server-side join: lazy / on-demand.** Compute
   `joint_logprobs` only when the user opens NodeCompareDrawer for a
   given pair; do not pre-compute for every fan sibling pair. The
   server route force-replays each branch token-by-token under the
   node's stamped recipe: steering hooks, recipe override effects,
   probe gates, penalties, logit bias, and sampler transform all match
   generation. It caches the result keyed by `(node_a_id, node_b_id)`
   until a tree edit/delete/finalize/reset invalidates the cache.
   Keeps the eager fan path cheap; pays the cost only for pairs the
   researcher actually inspects. _Status: locked._

---

## Migration

No migration needed for Phases 1-6 — every new field is optional,
every UI surface guards on `?? null`. Old transcripts replay without
logit info; the inline surprise mode shows "no logit data" empty
state on legacy tokens, the drilldown logits tab shows the
"enable in header" empty state, NodeCompareDrawer falls back to its
existing text/reading-only diff.

Phase 7 (continuation mode) introduces a new persistence key and a
new panel; users who never enable it see no change.

---

## Audit followups absorbed

Items from the UX audit that this plan addresses or explicitly defers:

| Audit item | Status |
|---|---|
| No fan view (N-up multiverse) | Deferred to separate plan; partially mitigated by Phase 6 distributional metrics on existing fan |
| No logits / no token probabilities | Phases 1-3 |
| Per-layer readings aggregate-only on chat tokens | Out of scope here; tracked separately as a probe-surfacing pass |
| Loom node decoration single-probe-only | Out of scope here |
| Filter grammar undiscoverable | Decision 8 (lands with Phase 4) |
| Transcript export/import CLI-only | Out of scope here; tracked separately as a transcript-drawer pass |
| Pin-for-comparison only via right-click | Out of scope here |
| Loom keyboard nav (`j/k/h/l`) only with sidebar focus | Out of scope here |
| Chat-as-primary framing | Phase 7 partially addresses for continuation mode; full reframing is a separate plan |

The follow-up plans worth opening when this one ships:

- **Continuation mode** (`docs/plans/continuation-mode.md`). Phase 7's
  sketch above as a full plan — base-model raw-text mode, separate
  Chat view, per-mode tree, CLI `--continuation` flag. Committed to
  ship; just deferred from this pass because base models warrant
  their own design surface.
- **Probe surfacing pass.** Per-layer inline tinting on chat tokens,
  multi-probe loom-node rings, transcript drawer, pin-from-chat.
- **Tree-as-primary view.** The webui's TUI-loom-screen analogue:
  full-screen tree + detail pane + chat-as-viewport. The biggest
  framing change from the audit; worth its own plan.
