# web/

Static Svelte 5 + Vite dashboard mounted at `/` by `saklas serve`. CLI default is on (`--no-web` opts out); `create_app(..., web=False)` is the library default so embedded API surfaces don't pick up the dashboard.

## Layout

```
saklas/web/
  __init__.py        # re-exports: register_web_routes, dist_path
  routes.py          # mount logic + SPA fallback
  dist/              # COMMITTED build artifact, ships in the wheel
    index.html
    assets/index.css
    assets/saklas.js
```

The Svelte source lives at the repo's `webui/` directory (peer of `saklas/`). `cd webui && npm run build` emits straight to `saklas/web/dist/` — no intermediate copy. The committed bundle is the source of truth; CI source-vs-dist drift gating is wired but disabled by default.

## Mount

`register_web_routes(app)` mounts `/assets/*` on `StaticFiles` (content-hashed, safe to cache hard), registers `GET /` → `index.html`, and a catch-all `GET /{full_path:path}` that serves allowlisted top-level dist files (favicon, etc.) and otherwise falls back to `index.html` for SPA routing. The catch-all is registered last by `create_app` so it never shadows `/v1/*`, `/api/*`, `/saklas/v1/*`. `full_path` is only ever used as a dict key, never a path component — `..` traversal is structurally impossible.

`dist_path()` resolves through `importlib.resources` (editable + wheel). `WebUINotBuilt` is raised on mount when the dist directory is empty — only fires in source installs that haven't run `npm run build`.

## Wire protocol

The dashboard speaks the native `/saklas/v1/*` API (`saklas/server/saklas_api.py`):

- **WS `/saklas/v1/sessions/{id}/stream`** — token + probe co-stream. With probes loaded, the `token` event carries `scores` (`dict[str, float]`, the magnitude-weighted `score_single_token` aggregate the TUI also tints with), `per_layer_scores` (`dict[str, dict[str, float]]`, string-keyed, feeds the token-drilldown heatmap), and `raw_index` (decode-step index into the backing node's `raw_token_ids`). `done` carries `per_token_probes`. A `generate` message with `fork_node_id`/`fork_raw_index`/`fork_alt_token_id` is the **logit fork** — replays the node's raw decode prefix with one token swapped, resampling the continuation as a sibling. A `generate` with `prefill_node_id`/`prefill_text` is **answer-prefill** — seeds an assistant reply under a *user* node. A `generate` with `commit_role`/`commit_text` is **commit (no-generation send)** — `commit_role="user"` lands a user turn under `parent_node_id`/active node; `commit_role="assistant"` lands an authored assistant turn under the user node `parent_node_id` (required), with `commit_text` as the whole turn. Short-circuits the streaming machinery: one `started` (node_id=null) + one `done` (`result.kind="commit"`, `result.node_id`, `result.role`, `result.text`), no token frames. Mutually exclusive with fork and prefill. Driven from `Chat.svelte` by `Ctrl+Enter` / `Cmd+Enter` (also `Ctrl`/`Cmd`-click on the send button); live `ctrlHeld` window listener swaps the send-button label between `send`/`prefill` and `commit user`/`commit assistant` while the modifier is held.
- **GET `/sessions/{id}/correlation[?names=…]`** — N×N magnitude-weighted cosine matrix; default pool unions steering vectors + active probes.
- **GET `/sessions/{id}/vectors/pairwise?a=…&b=…`** — cross-layer cosine matrix between two named vectors / probes. Distinct from `correlation`: one pair, two-axis matrix indexed by layer rather than by name; backs the pairwise-compare analysis drawer. Registered *before* `GET /vectors/{name}` so the literal path wins the routing match.
- **GET `/sessions/{id}/vectors/{name}/diagnostics`** — 16-bucket layer-magnitude histogram + per-layer magnitudes; falls back to monitor profiles when `name` is a probe.
- **GET `/packs`**, **GET `/packs/search?q=…`**, **POST `/packs`** — installed packs, HF Hub search proxy, install.
- **`/manifolds`** — `GET` list (with `fitted_for_session` / `stale` per manifold), `GET /{ns}/{name}` detail (nodes + statements), `POST` create, `PATCH /{ns}/{name}`, `DELETE /{ns}/{name}`, `POST /{ns}/{name}/fit` (SSE). Steering a fitted manifold is just a `%` term in the WS `steering` string — no separate route.
- **POST `/sessions/{id}/vectors/merge`**, **POST `/sessions/{id}/vectors/clone`** (SSE progress on `Accept: text/event-stream`), **POST `/sessions/{id}/extract`** (SSE). **POST `/sessions/{id}/extract/preview`** returns the LLM-generated contrastive pairs without committing, so ExtractDrawer can show them in an editable table.
- **POST `/sessions/{id}/experiments/fan`** — alpha grid as loom siblings, JSON `RunSet` summary.
- **Loom tree** under `/sessions/{id}/tree` — `tree`/`tree/active` GETs; `navigate`/`edit`/`branch`/`delete`/`star`/`note`/`reset` mutations; `edge_label`, `filter`, `diff`, `joint_logprobs`; `transcript` export/import.
- **GET `/sessions/{id}/traits/stream`** — live per-token probe SSE.

## Source layout

```
webui/src/
  main.ts                     # bootstrap: mounts <App /> via Svelte 5 mount()
  App.svelte                  # shell + drawer switch; NARROW_DRAWERS size class
  lib/
    api.ts                    # typed REST + WS + SSE clients
    stores.svelte.ts          # runes-based shared state + cross-cutting WS/tree state
    stores/                   # split slices: drawers, inputHistory, toasts (.svelte.ts)
    types.ts                  # shared interfaces; DrawerName union
    expression.ts             # parse/serialize the steering grammar
    concepts.ts               # concept-catalog helpers (category / poles / recommended α)
    tokens.ts                 # HIGHLIGHT_SAT + scoreToRgb + twoStripeStyle
    charts.ts                 # bucketize() port of saklas.core.histogram
    charts/{Bar,Sparkline,Histogram,HeatmapCell}.svelte
    Slider.svelte             # shared range slider
    Toaster.svelte            # toast host (bottom-right, TTL-dismissed)
    style/{tokens.css,global.css}
  panels/
    WorkspaceRail.svelte      # left rail: category fly-outs
    InspectorPanel.svelte     # right rack: steering + probe racks (full column height)
    WorkbenchCard.svelte      # active-workbench card (model + device); bottom of threads column
    StatusFooter.svelte       # gen progress · t/s · elapsed · ppl + pending-queue count badge; mounted inside Chat above the input row
    PendingBubbles.svelte     # ghosted bubbles for queued sends/commits/mutations + per-item × cancel; mounted between StatusFooter and the input row
    Chat.svelte               # thinking-collapsible, probe-tinted tokens, inline actions (clear/save/load/transcript), status footer, pending bubbles
    SamplingStrip.svelte      # T / P / K / max / pres / freq / seed / thinking / alts + advanced/system-prompt buttons; foot of the threads column, below WorkbenchCard
    SteeringRack.svelte       # vector + manifold strips + "+ add steering"/"+ add manifold"
    VectorStrip.svelte        # enable + α slider + trigger + variant + projection modal
    ManifoldStrip.svelte      # racked manifold term: position picker + blend slider + trigger
    manifold/XYPad.svelte     # position picker: 2D xy-pad / 1D-3D sliders, periodic axes wrap
    RawBuffer.svelte          # base-model flat completion buffer (replaces Chat's bubbles)
    ProbeRack.svelte          # probe strips + sort + "+ add probe"
    ProbeStrip.svelte         # select-for-highlight + sparkline + per-layer reading strip
    loom/{LoomSidebar,LoomNode,LoomEdge}.svelte  # permanent "threads" column
  drawers/
    {Load,SaveConversation,LoadConversation,Compare,SystemPrompt,
     Help,Export,Pack,Merge,Clone,Vectors,Extract,Manifold,ManifoldBuilder,
     TokenDrilldown,ExperimentLab,ActivationAtlas,RecipeBuilder,
     AdvancedSampling,Health,SessionAdmin,Correlation,LayerNorms,
     NodeCompare,Transcript}Drawer.svelte
    index.ts                  # barrel re-exports for App.svelte's switch
```

`VectorsDrawer` is the unified vector management surface — both rack `+ add` buttons (steering and probe) and `RecipeBuilderDrawer`'s "browse…" route here.  It reads `packsState.infos` reactively (populated by `refreshPacks`) and splits into two sections on the server's `LocalPackInfo.has_tensor` flag:

- **Extracted** — packs with a baked tensor for the loaded model.  Per row: `[steer] [probe] [delete]` toggles.  The steer/probe buttons reflect current `vectorRack`/`probeRack.active` state and toggle on click (mutate the corresponding store; no server roundtrip for removal, an extract-with-cache-hit before adding so the session has the profile registered).
- **Statements only** — packs with `statements.json` + `scenarios.json` but no tensor for this model.  Per row: `[extract] [delete]`.  Extract reuses the cached statements — one-click, no form, drives the same sticky-toast progress flow as ExtractDrawer.  On success, `refreshPacks()` reshuffles the row up into the "Extracted" section without remount.

Delete uses a 2-step confirm — first click flips the button label to `confirm?` for ~3 s, second commits.  No native dialog.

`ExtractDrawer` is the custom-vector form, reached from the `+ custom vector` button at the top of VectorsDrawer.  It has a `poles` / `custom` input-mode switch.  **Poles mode**: two concept slots (A required, B optional) plus an advanced-options collapsible (method / SAE / DLS); a `generate previews` button calls `/extract/preview` and drops the LLM-generated pairs into an editable pos/neg table, which the user edits before committing.  **Custom mode**: the pair table starts empty and the user types pos/neg statements directly.  Either way `submit()` sends `source: {pairs}` (empty rows trimmed) — the legacy slug-string `source` is kept only for a poles-mode submit where the user never previewed.  `params.seed_a` pre-fills concept A from the unmatched search query.  Submission closes the form and reopens VectorsDrawer while the sticky progress toast tracks extraction in the background.

`ManifoldDrawer` is the manifold browser (parallel to VectorsDrawer) — Fitted manifolds get `[+steer] [delete]`, Unfitted get `[fit] [delete]` (fit drives an SSE progress toast), and a `+ build manifold` launcher opens `ManifoldBuilderDrawer`.  The builder authors a manifold in two steps: a domain step (Box 1D/2D/3D with per-axis lo/hi + open/periodic toggles, or a Sphere dim) and a node editor (label + intrinsic-dim coords + an expandable per-node statement list), validated live against `min_nodes = 2n+1` and in-domain coordinates; Save posts to `POST /manifolds`.  A racked manifold renders as a `ManifoldStrip` in `SteeringRack` (below the vector strips) carrying an `XYPad` position picker, a blend slider, and a trigger pill; the position serializes into the steering expression as a `%` term.

`RawBuffer` is the base-model surface.  `SessionInfo.is_base_model` (a non-chat model has no chat template) drives `genUiMode.effectiveRawMode()`; the `genUiMode.override` (`auto`/`chat`/`raw`, persisted per `model_id`, set from the AdvancedSamplingDrawer control or the cycling badge in the Chat header) wins when not `auto`.  In raw mode `Chat.svelte` renders `<RawBuffer />` instead of role bubbles — one continuous editable `pre-wrap` surface with the loom active path joined as plain text, no roles.  "continue" generates with `raw: true` from the active leaf (the whole buffer is a prefill); a mid-buffer edit lands as a `loomEdit`/`loomBranch` first.  Toggling the mode never mutates the tree.  Per-token tinting rides a read-only mirror layer behind the transparent-text textarea (a textarea can't tint spans) and shows only when not actively editing.

`lib/expression.ts` parses `%` manifold terms (`<manifold>%<coord_list>`, e.g. `0.7 circumplex%0.3,0.8@response`) alongside vector terms — `Term` is a discriminated union, `serializeExpression(vectorRack, manifoldRack?)` emits both, `parseExpression` returns `{vectors, manifolds}`.

(`lib/stores.ts` is a dead legacy file — not imported anywhere; ignore it.)

Adding a panel: write the `.svelte`, wire state into the smallest matching `lib/stores/` slice (or `stores.svelte.ts` for cross-cutting WS/tree/chat state), mount from `App.svelte`, `npm run build`, commit the regenerated `dist/`. Adding a drawer: write it under `drawers/`, add the name to the `DrawerName` union in `lib/types.ts` (and to `NARROW_DRAWERS` in `App.svelte` for forms/pickers), add an `App.svelte` switch branch, re-export from `drawers/index.ts`, and add it to a `WorkspaceRail.svelte` category fly-out.

## Pending queue

Submissions during an in-flight gen (or behind earlier queued items) defer rather than racing the WS — same semantics as the TUI. `sendGenerate` / `sendCommit` / `sendPrefill` check `isPendingBusy()` (gen active OR `pendingActions.queue.length > 0`) and, when busy, append a `PendingAction` (defined in `lib/types.ts`) carrying a `rebuild` factory the `↑`-pull-and-edit path uses to re-encode the same kind/role/target with new text. Instant mutations from the chat header (`/clear`, regen) and the rack/sampling sites also queue via `enqueuePending` with `awaitsGen: false` so the drain chains through them without waiting on a `done` that never fires.

Queued rack mutations coalesce. `enqueueOrApply` tags each rack-mutation item with `coalesceKey: "rack"`; when the queue tail already carries that key, a fresh mutation chains its `apply` onto the tail item rather than appending a new slot, and the bubble's label updates to the latest action. A slider drag firing 30+ intermediate `setVectorAlpha` calls mid-gen therefore leaves one queued bubble carrying the net effect. Coalescing stops at any non-rack item — rack changes before and after a queued send form distinct groups, so FIFO ordering relative to the send holds.

The WS `done` / `error` handlers call `drainNextPendingAction()` — one item per event — instead of the old v1 `applyPendingActions` (which drained everything at once). `PendingBubbles.svelte` renders the live queue above the input as dim chips; the per-bubble `×` calls `cancelPendingAction(id)` to remove a single slot. The bubble whose slot the user is currently editing via ↑ gets the `.editing` class — brighter amber background, thicker border, full-strength text, and a leading `✎` marker — driven off `inputHistory.pulledSlot`. The StatusFooter shows a `N queued` readout but no "apply now" button — under the FIFO model there's no skip-ahead semantics.

Up-arrow walks the combined ring `[editable pending (most-recent first), input history (newest first)]`. Pulling a queued item sets `inputHistory.pulledSlot`; Chat.svelte forwards that to `sendGenerate` / `sendCommit` / `sendPrefill` as `replaceSlot` so a re-edited send lands at its original slot. `Esc` while pulled cancels the edit (slot stays, input restores the stash); empty `Enter` while pulled removes the slot — keyboard equivalent of the `×` button. Non-editable items (`rebuild === null`, e.g. queued `/clear` and regen) sit in the queue but the up-arrow walks past them.

## Reactivity gotcha

Svelte 5's `$state` does NOT track `Map.set` / `Set.add` / inner-object property writes inside collections. Cross-component collections in `stores.svelte.ts` use `SvelteMap` / `SvelteSet` from `svelte/reactivity`. Inner-object mutations on map values are still untracked, so every rack mutator reassigns: `entries.set(name, {...e, alpha})` — vectorRack and probeRack alike. `updateProbeFromScores` (driven by every WS `token` event) is the hot path here — a bare `entry.current = val` would freeze probe sparklines at zero through a whole generation.

## Persistence

The server loom tree is authoritative. The browser keeps a first-paint cache of the latest `LoomTreeJSON` plus `highlightState` in `localStorage` under `saklas.chat.v2.<model_id>`; v1 flat `ChatTurn[]` logs auto-migrate. Saves are debounced ~250 ms after mutations. `refreshLoomTree()` overwrites the cache with server state once the tree endpoint responds. `schedulePersist` measures payload size against a 5 MB soft budget and fires a once-per-session advisory toast (suggesting transcript export + tree clear) above it — the write isn't hard-stopped. `pendingIndex` is force-cleared on restore so an in-flight turn from a killed tab can't ghost the UI.

## Per-token highlighting

Highlighting lives on the chat token spans, driven by a single highlight-probe dropdown in the chat header with an optional two-stripe compare-two mode. It tints **live** as tokens stream: the WS `token` event's `scores` aggregate feeds the same `scoreToRgb` ramp the post-generation pass uses, so streaming and finalized tints match (and match the TUI). Clicking any token opens the `token_drilldown` drawer with the per-layer × per-probe heatmap regardless of whether a highlight probe is selected.

## Toasts

`lib/stores/toasts.svelte.ts` toasts carry `kind`, `message`, optional dim `detail` sub-line, and `ttlMs: number | null` — `null` is sticky (no auto-dismiss; caller owns dismissal).  `updateToast(id, patch)` mutates a live toast in place so long-running async work (extract / clone) can drive a single chip from kickoff to completion without spawning new ones.  `Toaster.svelte` only schedules a dismissal timer the first time it sees a non-null TTL, so flipping sticky → ttl mid-flight is a no-op; callers that want a finite TTL at the end should `dismissToast` + `pushToast` instead.  ExtractDrawer is the canonical user — sticky progress on submit, dismissed and replaced with a 6 s success toast (or sticky error toast) when the SSE `done` / `error` lands.

## Out of scope

- True multi-session switching — server URL-paths support it; the client still assumes `default`. `SessionAdminDrawer` inspects the collection and sets an in-memory bearer key but is not a session router.
- Persistent credential management — the bearer key stays in memory for the page session, never written to `localStorage`.
- Mobile / touch-first layout — desktop research tool, min-width 1280px.
- Combobox autocomplete on the projection-target picker (free-form name input).
- Pagination on HF pack search (capped at 20 results).
