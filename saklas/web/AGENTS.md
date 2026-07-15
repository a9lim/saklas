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

`register_web_routes(app)` mounts `/assets/*` on `StaticFiles` with hard cache headers, registers `GET /` → `index.html`, and a catch-all `GET /{full_path:path}` that serves the allowlisted top-level dist files (every top-level file except `index.html` — currently just `favicon.ico`) and otherwise falls back to `index.html` for SPA routing. (Caveat: the bundle filenames are *fixed* — `assets/saklas.js` / `assets/index.css`, not content-hashed — so a rebuilt bundle can stale-cache under the hard headers; a shipping concern, not a mount bug.) The catch-all is registered last by `create_app` so it never shadows `/v1/*`, `/api/*`, `/saklas/v1/*`. `full_path` is only ever used as a dict key, never a path component — `..` traversal is structurally impossible.

`dist_path()` resolves through `importlib.resources` (editable + wheel). `WebUINotBuilt` is raised on mount when the dist directory is empty — only fires in source installs that haven't run `npm run build`.

## Wire protocol

The dashboard speaks the native `/saklas/v1/*` API (registered by
`saklas/server/native_routes.py`):

**Current composer contract (supersedes the legacy generate/seat wording
below):** `Chat.svelte` sends `type:"submit"` with explicit native
`authored_role` / `generated_role` (`user|assistant`). Two visible role controls
select those roles independently while displaying their chat-template labels;
the continuation control also accepts `none` for authored-only append, and a
one-shot swap action exchanges the current pair. The template-channel seat stays
an implementation detail in the composer. Selected-node role never changes
composer semantics. Empty text generates the selected continuation role. The
visible actions remain the compact `send`, `generate`, and `append`; there is no
append modifier shortcut.
Matching structural role + effective role label coalesces into one message.
`type:"generate"` remains for specialist fork/prefill and compatibility.
Result actions/rendering key off artifacts (`recipe`, token rows,
`raw_token_ids`, logprobs), not assistant role. The cast is structural
user/assistant plus labels observed anywhere in the tree, overlaid by configured
recipes/notes; effective roster rows carry `origin`.

- **WS `/saklas/v1/sessions/{id}/stream`** — token + probe co-stream. Every measured `token` event carries a canonical `captured` record that is byte-for-byte the same JSON-safe object stored on its loom token row: optional `probes` (`scores`, `per_layer_scores`, full `readings`), `lens` (`source`, recipe `steering`, endpoint-shaped `layers[].tokens[]` with token/id/logprob, plus `aggregate`), and `sae` (`source`, recipe `steering`, resident `layer`, `features`). Each channel is marked `provenance:"captured"`. The older top-level `scores`, `per_layer_scores`, `probe_readings`, `lens_readout`, `lens_aggregate`, and `sae_readout` fields remain compatibility aliases; new WebUI code reads `captured`. `raw_index` is the decode-step join into the backing node's `raw_token_ids`. The `done` event's `result` carries aggregate `probe_readings` of the same `ProbeReading` shape. The composer uses `type:"submit"`; authored-only submissions return `result.kind="append"`. A `generate` message with `fork_node_id`/`fork_raw_index`/`fork_alt_token_id` is the **logit fork** — replays the node's raw decode prefix with one token swapped, resampling the continuation as a sibling. A `generate` with `prefill_node_id`/`prefill_text` is the compatibility **answer-prefill** path. Legacy `commit_role`/`commit_text` fields remain accepted for older clients and keep their legacy `result.kind="commit"`; new UI code must not use them for the composer.
- **GET `/sessions/{id}/correlation[?names=…]`** — N×N magnitude-weighted cosine matrix; default pool unions steering vectors + active probes.
- **GET `/sessions/{id}/vectors/pairwise?a=…&b=…`** — cross-layer cosine matrix between two named vectors / probes. Distinct from `correlation`: one pair, two-axis matrix indexed by layer rather than by name; backs the pairwise-compare analysis drawer. Registered *before* `GET /vectors/{name}` so the literal path wins the routing match.
- **GET `/sessions/{id}/vectors/{name}/diagnostics`** — 16-bucket layer-magnitude histogram + per-layer magnitudes; falls back to monitor profiles when `name` is a probe.
- **GET `/sessions/{id}/vectors`** — registered steering vectors. Vector authoring rides the session routes (`POST /extract`, `/vectors/merge`, `/vectors/bake`) + the manifold install/browse routes. There is **no `/saklas/v1/packs*` surface**, **no `/manifold-probes`**, **no `/extract/preview`**, and **no `/vectors/clone`** — all removed in the 4.0 collapse.
- **`/manifolds`** — `GET` list (each row carries `tags` (category) alongside `fit_mode` / `is_discover` / `hyperparams` / `fitted_for_session` / `stale`), `GET /{ns}/{name}` detail (nodes + statements; discover-mode fits carry derived coords + `diagnostics`), `POST` create (authored), `POST /discover` create (discover-mode, label-only nodes), `POST /from-template` derives a discover manifold from an existing standalone template, `POST /generate` (SSE; LLM-authors a discover folder via `session.generate_responses` — A2 conversational extraction, body `kind: "abstract"|"concrete"` + `samples_per_prompt`, not scenario counts), `PATCH /{ns}/{name}`, `DELETE /{ns}/{name}`, `POST /{ns}/{name}/fit` (SSE; accepts discover-mode `fit_mode` / `hyperparams` overrides on discover folders). Steering a fitted manifold is just a `%` term in the WS `steering` string — no separate route.
- **`/templates`** — the standalone templated-completion artifact. `GET` list, `GET /{ns}/{name}` detail (incl. `contexts`), `POST` create (`{slot, values, contexts:[{turns, assistant}]}`), `DELETE /{ns}/{name}`, `POST /{ns}/{name}/score` (`{steering?}` → per-context restricted-choice value distribution, under the session lock). `apiTemplates` in `lib/api.ts`; backs `TemplateLabDrawer`.
- **`/sessions/{id}/probes`** — one unified read-side probe collection. The pre-4.0 split of vector probes vs manifold probes collapsed onto a single route over the session's single `Monitor`. `GET` lists every attached probe as a `ProbeInfo` (`name`, `manifold`, `top_n`, `layers`, `node_labels`, `node_count`, `domain`, `intrinsic_dim`, `feature_space`, `is_affine`, `node_coords`); `GET /defaults` returns the default roster; `POST` body `{selector, name?, top_n?}` attaches any probe shape by selector (a 2-node concept axis is the rank-1 case); `DELETE /{name}` detaches. `is_affine` is the flat-vs-curved discriminator the client classifies subspace-vs-manifold on; `node_coords` backs the 2-D mini-map. Per-token (WS `token`) and aggregate (WS `done` `result`) readings ride the `probe_readings` field — the same `ProbeReading` shape for both.
- **POST `/sessions/{id}/vectors/merge`** (JSON), **POST `/sessions/{id}/vectors/bake`** (JSON), **POST `/sessions/{id}/extract`** (SSE progress on `Accept: text/event-stream`, JSON otherwise).
- **POST `/sessions/{id}/experiments/fan`** — alpha grid as loom siblings, JSON `RunSet` summary.
- **Loom tree** under `/sessions/{id}/tree` — full-tree GET plus atomic full-tree PUT restore (model-bound, the dashboard v4 save/load inverse), `tree/active` GET; `navigate`/`edit`/`branch`/`delete`/`star`/`note`/`reset` mutations; `edge_label`, `filter`, `diff`, `joint_logprobs`; `transcript` export/import.
- **GET `/sessions/{id}/traits/stream`** — live per-token probe SSE.
- **GET `/sessions/{id}/lens/token-readout?node_id=&raw_index=…`** — the J-lens
  readout at one decode step: per-layer top-k matrix
  (`layers: [{layer, tokens:[{token, id, logprob}]}]`) plus the
  layer-aggregated `aggregate: [{token, strength, com, spread}]` block
  (per-layer softmax → mean fitted-layer probability + probability-mass-weighted
  depth center of mass across all requested layers) at the forward that produced
  the clicked token,
  recomputed on demand server-side (node prompt render + raw prefix replay
  under the node's recipe steering; `steered=false` for the unsteered
  counterfactual, `raw=true` for flat-buffer nodes — the client's render mode
  supplies raw-ness). Backs `TokenDrilldownDrawer`'s **j-lens** tab via
  `apiLens.tokenReadout`; gated on the session-info `jlens_fitted` flag
  (unfitted → the tab shows the `saklas lens fit` hint).
- **POST `/sessions/{id}/lens/token/validate`** — read-only `{word}` →
  `{word, token_id}` single-token check. Both J-LENS add forms call it before
  mutating the steering rack or attaching a probe; a rejected word remains in
  the input and is surfaced as an error toast.
- **POST `/sessions/{id}/lens/live`** — toggle the *live* J-lens readout
  (`{enabled, layers?}` → `{enabled, layers}`; layers omitted picks every
  fitted layer). The generation's logit-alternative `return_top_k` also sets
  the live lens width. While on, each WS `token` frame carries
  the `lens_readout` matrix + the `lens_aggregate` chip list and session info
  reports `live_lens_layers` (`null` while off — the J-LENS tab's rehydration
  read). `apiLens.setLive`; backs the live toggle on `JLensPanel`'s merged
  PROBE section.
- **POST `/sessions/{id}/probes/live`** — the CAA live toggle (`{enabled}` →
  `{enabled}`): off ⇒ per-token monitor scoring is disabled for UI/trait/loom
  consumers (probes settle to end-of-gen aggregates; gates still fire).
  Session info reports `live_probe_scores` (default-on). `apiProbes.setLive` +
  `probesLiveState`/`setLiveProbes`; backs the live toggle on `ProbeRack`'s
  PROBE header — with it and the lens toggle both off, a compute-constrained
  session pays no per-token scoring at all.
- **J-LENS source lifecycle** — `GET /sessions/{id}/lens/sources`, `POST
  /lens/use`, and polled `POST/GET /lens/fetch` back the tab's SOURCE section.
  Fetch leaves official payloads in the Hugging Face cache and writes only the
  binding; use/fetch activation turns the live workspace on. The existing
  polled/cancellable `/lens/fit` is the Saklas-owned local sibling.
- **SAE source lifecycle** — `GET /sessions/{id}/sae/sources`, background
  `POST/GET /sae/load`, and polled/cancellable `POST/GET/DELETE /sae/train` back
  the symmetric SOURCE section. Load accepts `local:<name>` or
  `saelens:<release>` plus an optional hook `layer`; the SAE tab's second
  dropdown selects that resident measurement layer, and the selection is
  restored from cached provider metadata. Local training reports token progress
  and activates the resulting Saklas-owned source. In both instrument tabs, local fit/train fields
  are hidden until the synthetic `local` selector option is chosen; that same
  source-row action changes from fetch/use to fit/train, while prepared
  `local:<name>` sources remain separately selectable. Prepared cached sources
  win the default selector choice; the duplicate lower SOURCE label and passive
  source-summary lines are omitted so SOURCE flows directly into STEER.

## Source layout

```
webui/src/
  main.ts                     # bootstrap: mounts <App /> via Svelte 5 mount()
  App.svelte                  # shell + drawer switch; NARROW_DRAWERS size class
  lib/
    api.ts                    # typed REST + WS + SSE clients
    stores.svelte.ts          # runes-based shared state + cross-cutting WS/tree state
    tooltips.ts               # delegated Saklas tooltip layer; `title` remains
                              # the authoring API, native bubbles are suppressed
    stores/                   # split slices: drawers, inputHistory, toasts (.svelte.ts)
    types.ts                  # shared interfaces; DrawerName union
    expression.ts             # parse/serialize the steering grammar
    concepts.ts               # concept-catalog helpers (category / poles / recommended α)
    tokens.ts                 # HIGHLIGHT_SAT + scoreToRgb + twoStripeStyle
    charts/{Bar,Sparkline,HeatmapCell}.svelte
    charts/probeGeometry.ts   # hand-rolled canvas renderer for the probe-inspector geometry plot
    manifolds/diagnostics.ts  # pure helpers — classify/bars/summary for discover fits
    manifolds/DiagnosticsPanel.svelte  # variance-bars / eigenvalue-spectrum renderer
    Slider.svelte             # shared range slider
    Select.svelte             # themed dropdown — replaces native <select> everywhere
    Checkbox.svelte           # themed checkbox — square accent fill on checked
    Radio.svelte              # themed radio — generic over value, bind:group
    NumberInput.svelte        # themed numeric input — hover-revealed steppers,
                              # exported focus()/select() for modal hosts,
                              # onkeydown forwarding, onchange + oninput
    Disclosure.svelte         # themed collapsible — ▸/▾ caret pattern
    builder/                  # shared layout helpers for the two builder drawers
      ModeTabs.svelte         # the auto-generated / custom-statements|nodes tab strip
      AdvancedSection.svelte  # collapsible "Advanced options" wrapper around Disclosure
      ValidationBlock.svelte  # "not ready to <verb>:" + bulleted reasons
    Toaster.svelte            # toast host (bottom-right, TTL-dismissed)
    style/{fonts.css,tokens.css,global.css}
                              # fonts.css: Recursive VF self-hosted (src/assets/
                              # fonts/), ONE woff2 exposed as TWO families —
                              # "Recursive Sans" (MONO 0, CASL .35) for chrome,
                              # "Recursive Mono" (MONO 1) for data — axes pinned
                              # per-family in @font-face, so a font-family switch
                              # alone flips the voice
    ui/                       # v2 primitives (Observatory+ redesign, 2026-07-10):
      Button.svelte           # solid / ghost / danger; ``accent`` retints to a pillar hue
      Chip.svelte             # mono capsule, hue wash; recipe terms, badges, tags
      SegmentedTabs.svelte    # pillar tab strip (hue dot + glass active)
      GlassCard.svelte        # the v2 card material; exposes --card-accent like RackCard
  styleguide/StyleGuide.svelte # living design-system page at /styleguide —
                              # main.ts routes on pathname (no App bootstrap/WS
                              # there); every specimen is the real component
                              # reading the real tokens
  panels/
    CommandPalette.svelte     # ⌘K palette — THE tool launcher (the former left
                              # rail is gone; a ⌘K chip in the chat header is the
                              # visible hint): instrument-tab jumps + every registry
                              # tool (lib/commands.ts) + pages; state in
                              # lib/stores/palette.svelte.ts
    RecipeBar.svelte          # persistent cross-pillar expression bar atop the
                              # inspector: every racked term as a pillar-colored
                              # chip (click → jump to its tab, × → remove, ⧉ →
                              # copy the canonical expression)
    SaePanel.svelte           # SAE pillar — SOURCE picker + provider fetch and
                              # cancellable local train; loaded: identity +
                              # decoder-row STEER cards + pinned/live-discovery
                              # feature PROBE cards (▲ unpins / △ pins — same
                              # merged-PROBE shape and strength/name sort as the
                              # lens tab), cards in
                              # rack/{SaeSteerCard,SaeProbeCard} (gold, ▲/△).
                              # Cards read normalized strength — activation /
                              # Neuronpedia maxActApprox, absolute 0..1 bar
                              # like the lens cards (saeState.meta, fed by
                              # token-frame max_act + the between-generation
                              # POST sae/features/metadata backfill in
                              # backfillSaeMeta); metadata-less features fall
                              # back to raw activation on a panel-shared
                              # scale so bars always rank with the numbers
    InspectorPanel.svelte     # right rack: THE INSTRUMENT STACK — RecipeBar +
                              # four pillar tabs (subspace/manifold/sae/lens via
                              # ui/SegmentedTabs). subspace + manifold tabs each
                              # render SteeringRack+ProbeRack filtered by a
                              # family prop (the tab IS the group; the old CAA
                              # two-group split is gone), sae = SaePanel,
                              # lens = JLensPanel. InspectorTab union is the four
                              # pillar names (the pre-4.2 "probes"/"jlens" values
                              # are gone). Hovering a transcript/raw-buffer token
                              # overlays that token's probe/J-LENS/SAE readings;
                              # every channel emitted by a live frame is retained
                              # for the page session; channels absent at generation
                              # use the existing historical token endpoints
    JLensPanel.svelte         # J-LENS pillar — SOURCE picker + official fetch
                              # and cancellable local fit; then STEER + merged
                              # PROBE cards with live workspace discovery
    rack/InstrumentSourceSection.svelte # one shared prepared/provider selector
                              # plus the canonical labelled custom lifecycle
                              # row for the SAE and J-LENS pillars
    WorkbenchCard.svelte      # active-workbench card (model + device); bottom of threads column
    StatusFooter.svelte       # gen progress · t/s · elapsed · ppl + pending-queue count badge; mounted inside Chat above the input row
    PendingBubbles.svelte     # ghosted bubbles for queued sends/commits/mutations + per-item × cancel; mounted between StatusFooter and the input row
    Chat.svelte               # cast-model turn cards (neutral glass, role chips,
                              # ppl provenance, system = stage directions),
                              # thinking-collapsible, probe-tinted tokens, the
                              # cast row (speaking as / reply as chips), inline
                              # actions, status footer, pending bubbles
    SamplingStrip.svelte      # T / P / K / max / pres / freq / seed / thinking /
                              # alts + advanced/system-prompt buttons (role boxes
                              # moved to Chat's cast row); foot of the threads
                              # column, below WorkbenchCard
    SteeringRack.svelte       # STEER section of one instrument tab — takes a
                              # ``family`` prop ("subspace" | "manifold") and renders
                              # only that family's cards (+ the shared "subspace
                              # along" master on the subspace tab); footer keeps the
                              # family's single "+ … steer" launcher
    ProbeRack.svelte          # PROBE section of one instrument tab — same ``family``
                              # prop (subspace ⇔ info.is_affine); header live toggle
                              # (probesLiveState → apiProbes.setLive — the ONE
                              # monitor per-token scoring switch, rendered in both
                              # tabs driving the same state); footer keeps the
                              # family's single "+ … probe" launcher
    rack/RackCard.svelte      # shared card chrome: statline on top, controls/meters stacked below
    rack/RackSectionHeader.svelte # shared STEER/PROBE title + live/count/sort row across all pillars
    rack/RackMarker.svelte    # optically matched 18 px family markers: ●/◆/■/▲
    rack/ProbePinButton.svelte # shared 24 px pin/unpin action for every probe family
    rack/ProbeHighlightButton.svelte # pinned SAE/J-LENS transcript-highlight action
    rack/LayerStrip.svelte    # shared borderless, high-contrast per-layer cells + endcaps
    rack/ProbeReadingRow.svelte # canonical four-column label/bar/context/value meter grid
    rack/SteerCard.svelte     # unified steer card — branches on entry.mode (snap-to-node + XYPad both):
                              #   subspace → position only (magnitude = shared subspace-along master)
                              #   manifold → position + per-card along / onto
    rack/ProbeCard.svelte     # unified probe card — branches on is_affine / node_count===2:
                              #   bipolar bar OR scalar/fraction + nearest; curved adds coords / residual / mini-map
    manifold/XYPad.svelte     # position picker: 2D xy-pad / 1D-3D sliders, periodic axes wrap
    RawBuffer.svelte          # base-model flat completion buffer (replaces Chat's bubbles)
    manifold/ManifoldMiniMap.svelte  # SVG mini-map: node labels at authoring coords + per-token trajectory polyline + bold settled-aggregate dot
    loom/{LoomSidebar,LoomNode,LoomEdge}.svelte  # permanent "threads" column
  drawers/
    {SaveConversation,LoadConversation,Compare,SystemPrompt,
     Help,Export,Rack,ManifoldBuilder,ProbeInspector,TemplateLab,
     ManifoldMerge,ManifoldPacks,TokenDrilldown,ExperimentLab,ActivationAtlas,
     RecipeBuilder,AdvancedSampling,Health,SessionAdmin,Correlation,LayerNorms,
     NodeCompare,Transcript}Drawer.svelte
    index.ts                  # barrel re-exports for App.svelte's switch
```

The rack split is **subspace (flat/affine) vs manifold (curved)**, not vector vs manifold — the post-4.0 vocabulary where a steering vector is the 2-node flat case of a manifold and "manifold" means a genuinely curved fit. Every row in either rack wears one `RackCard` chrome — since the v2 redesign that chrome is the dense glass material (translucent fill lit from above, `--radius-lg`, borderless at rest — hover lifts the fill, and the border slot exists only for the `active` hue-ring/glow "alive" state the highlight-selected probe wears; no backdrop blur — cards sit on an opaque panel): a statline on top (marker glyph · name · status chips · actions, all one row) with the controls (steer cards) or meters (probe cards) stacked **vertically below** — never inline. The cards differ only by accent colour + marker glyph: subspace is `--accent` + ●/○, manifold is `--accent-purple` + ◆/◇ — and the other two pillars continue the vocabulary (lens `--accent-blue` + ■/□, sae `--pillar-sae` + ▲/△), so every card in the instrument stack reads family from hue + marker alone. That sameness is the steer/probe + subspace/manifold harmonisation made visible.

Classification. The steer rack is one `steerRack` of tagged `SteerEntry` (`mode: "subspace" | "manifold"`) — every term is a position on a fitted geometry (a steering vector is the K=2 flat case of a manifold), so all terms share one rack, one card, one serializer. `mode` is the geometry family itself: **subspace** (a flat affine fit — a 2-node bipolar axis through the rank-8 `personas` fan) vs **manifold** (a curved fit). It is set at add time (`RackDrawer` reads the catalog `fit_mode`: `pca`/`baked` → subspace, `spectral`/`authored` → manifold) and at parse time, never flipped in the UI. `SteeringRack` groups by `entry.mode` directly (no catalog lookup) — the subspace group then the manifold group (empty groups hidden), sorted within each. Every subspace term shares the rack-level **`subspaceAlong`** master (the merged affine subspace slides once) — one slider in the subspace group header, default 0.5; manifold terms keep a per-card along/onto. A probe entry is subspace iff `info.is_affine` (probes are unchanged by the 4.1 steer-rack collapse).

`SteerCard` (every `steerRack` entry; accent + glyph follow the subspace/manifold family — subspace `--accent` ●/○, manifold `--accent-purple` ◆/◇). It branches its body on `entry.mode` via narrowed `s` / `m` deriveds (so svelte-check enforces mode-correct field access). Statline (both modes): enable glyph · name · `unfitted`/`stale` warn chip · trigger pill · `✕` remove. Body: a snap-to-node `Select` (with a `(free position)` escape hatch) + the `XYPad` (rendered `locked` while a label is bound, so the pad reads out but rejects input until `(free position)` unsnaps). **Subspace** adds no per-card magnitude control — its strength is the shared `subspaceAlong` master in the group header (a hint line points there), and relative weight between subspace terms lives in how far each position sits from neutral. **Manifold** adds a per-card `along` slider plus an `onto` slider (the second coefficient that collapses the off-surface in-subspace residual onto the surface). Picking a node sets the entry's `label` (label form, `<m>%<label>`); pulling the pad authors a free coord and clears it. The `XYPad` renders a 2-D draggable pad at intrinsic dim 2 and one slider per axis otherwise (so a 2-node concept is one slider, `personas`'s 8-D fit is eight sliders). The pre-4.1 `~`/`|` projection, `!` ablation, and the `:variant` chip are gone from the card — the grammar forbids `%` composing with projection/ablation, and `:variant` survives on the entry (round-trip) but isn't authored here. (Reconciliation note: a 2-node concept renders the generic 1-axis XYPad + pole snap, not a dedicated bipolar signed slider — a candidate follow-up.)

`ProbeCard` (every `probeRack` entry; the store normalises `current` / `sparkline` / `perLayer` per family, so the card reads those uniformly and branches only on presentation). Statline: highlight-select glyph (●/○ flat, ◆/◇ curved — filled when this probe is the highlight target; click the identity cluster to toggle it) · name · sparkline · `✕` detach. The identity cluster is highlight-selectable for **every** family — the per-token score map is keyed by probe name regardless of geometry (the top-bar dropdown already lists curved probes, and this wires the click too). Body branches: a 2-node flat probe (`is_affine && node_count === 2`) renders a signed **bipolar reading bar** (axis-0 `coords[0]`) with poles; every higher-rank flat fan and every curved fit renders a scalar/fraction bar + nearest-node readout. A curved probe additionally shows the settled `coords` + `residual` meta (from the end-of-gen aggregate) and a `ManifoldMiniMap` for 2-D `BoxDomain` probes with attached node coords (`intrinsic_dim == 2`, `domain.type == "box"`, `node_coords` populated; higher-dim and sphere / custom probes skip the map). The per-layer heatmap strip sits below for all.

The probe data layer is one `probeRack` keyed by registered name (the pre-4.0 `probeRack` + `manifoldProbeRack` fused into one). Each `ProbeRackEntry` carries the `ProbeInfo` (with `is_affine`, and `lens: true` for a pinned J-lens token probe), a sparkline of the primary scalar (signed `coords[0]` for a flat probe, the `[0,1]` readout strength `coords[0]` for a lens probe, `fraction` for a curved one), the latest per-token `reading` + the end-of-gen `aggregate` (both the single `ProbeReadingJSON` shape), the recent `nearest`, and a 2-D `trajectory`. `refreshProbeList` / `attachProbe(selector, opts?)` / `detachProbe(name)` cover the REST surface over `apiProbes`; `updateProbesFromReadings` (WS `token`) / `setProbeAggregates` (WS `done`) read `msg.probe_readings`; `scores` feeds the highlight tinting only. Peripheral drawers reading `probeRack.active` / `.entries` are untouched by the fusion.

**Sheet interiors (v2).** The drawer host (`App.svelte`) paints the floating sheet — `--bg-alt` fill, `--glass-line` hairline, `--radius-lg`, overlay shadow — so every drawer interior is `background: transparent` with **no** `border-left` (the pre-v2 full-height-wall chrome is gone). Chrome speaks sans (`--font-ui`); every value / identifier / expression / number sits in `--font-mono` locally. The shared grammar, with `ProbeInspectorDrawer` and `TokenDrilldownDrawer` as the reference implementations: header = eyebrow (tracked caps `--text-xs` / `--weight-medium` / `0.08em`) over a name-row (mono `--text-md` subject + `--fg-subtle` meta), a 26 px circle glass-fill close button, body padding `--space-5/--space-6`, and data wells — tables / matrices / plots recessed into a `--bg` container with `--radius`, sticky label cells kept opaque `--bg` so they occlude scrolled content. **Borderless (Phase 6):** none of that carries a resting hairline — the fill step against the sheet's `--bg-alt` is the well boundary, spacing + eyebrow typography are the header/footer seams, and sticky header/label cells separate with `--shadow-sticky` (elevation that only reads once content scrolls under) instead of a rule. Hairlines survive only as meaning: focus rings, active/selected/invalid state rings, floating-surface outer edges (the sheet itself, popover menus, toasts), dashed pending-ghosts, and control glyph strokes. Exclusive-choice strips are `lib/ui/SegmentedTabs`, actions are `lib/ui/Button` (ghost default, `sm` in dense rows), micro-scrubbers stay custom pills. Hue stays ontological: the one pillar-owned surface in a drawer may carry its hue (the drilldown's j-lens tab dot, the inspector's family accent); everything else is achromatic.

`ProbeInspectorDrawer` (`probe_inspector`, launched from the `ProbeCard` statline's `[ⓘ]`) is the per-probe geometry inspector — it subsumes the layer-norms view for probes and adds a rank-aware plot in the whitened (Mahalanobis) frame, rendered by the hand-rolled canvas in `lib/charts/probeGeometry.ts` (zero 3-D dep). The plot branches on the selected layer's subspace `rank`: rank 1 → a line (poles + neutral + sliding live dot), rank 2 → a 2-D node scatter (+ a curve overlay when `intrinsic_dim == 1`), rank 3+ → a drag-orbit 3-D scatter on the top-3 PCs (+ curve / wireframe-surface overlay). Geometry is fetched once for all layers (`Monitor.probe_geometry(name)` → `GET /sessions/{id}/probes/{name}/geometry`, returning per-layer `node_white` / `neutral_white`, a top-3 PCA rotation at rank ≥ 3, and the sampled overlay, all whitened) and reprojected client-side on scrub; the per-layer `‖share‖` bars double as the layer scrubber (the former redundant layer `Select` is gone — the rows are the picker), with the active layer echoed by a chip pinned to the plot's top-left. **Camera (rank ≥ 3):** the scale is a rotation-invariant constant derived once from the static framing set (nodes + neutral + overlay) with the orbit a rigid spin about the **neutral anchor** (the whitened origin), so neutral sits at the plot center, the cloud shows its real displacement from neutral, and dragging never changes the cloud's size — this replaced the per-frame rescale off the rotated 2-D silhouette that made the view zoom while rotating. (Pivoting on neutral rather than the node centroid pairs with the backend's neutral-centered `node_coords` layout — both surfaces share neutral as origin.) Zoom is now an explicit `orbit.zoom` driven by the scroll wheel (clamped `0.3–6×`); rank 2 likewise frames from the static geometry so a moving live point can't rescale it. **Colour** follows the hue ontology: node centroids wear the probe's *family* hue — flat = subspace white, curved = manifold violet, the same `is_affine` split the racks use — supplied by the drawer through the `--geom-node` custom property on the plot well (the canvas renderer reads its palette off CSS vars, so `probeGeometry.ts` stays hue-agnostic; `--geom-neutral` likewise). The neutral anchor is a hollow *grey* ring (neutral is the absence of concept, so it carries no hue; the old amber is gone), the live hidden-state point a white core in a green halo with a soft green bloom (`liveDot` — glow is reserved for what is alive right now), and the fading trajectory trail a green time ramp (along-axis gradient allowed: the gradient *is* the data). Canvas labels render in Recursive Mono. The drawer itself is the v2 sheet-interior reference (see §Sheet interiors below): the family hue also accents the header dot, the active share row, the bars, and the plot well's hairline + ambient.

The live point + trail ride the probe's per-token `subspace_coords_per_layer` (the whitened query coords, same frame as `node_white`), gated on by the `persist_subspace_coords` sampling flag — `buildSamplingPayload` sets it only while this drawer is open, so the default generate path keeps the cheaper reading shape. Each token's coords are stored across **all** probed layers in `ProbeRackEntry.subspaceTrail` (`MAX_SUBSPACE_TRAIL` cap, reset per generation), so scrubbing layers is a pure read; the live point is the last sample at the scrubbed layer, the trail the rest. A small `N trail pts` readout on the plot surfaces the trail depth (and doubles as a flow check — climbing means coords are arriving). The inspector's `‖share‖` bars source `mahalanobis_share` from the geometry payload, not `/diagnostics` (which only resolves rank-1 probes).

**Default probes.** `saklas serve` auto-attaches the bundled manifolds (`default/personas`, `default/emotions`, plus any fitted bundled concept axis) as probes on startup when the dashboard is mounted (`_attach_default_manifold_probes` in `cli/runners.py`, gated on `web_enabled` so `--no-web` skips it). Only manifolds already *fitted* for the loaded model attach; an unfitted bundled manifold is skipped with a one-line startup hint (fitting runs a forward pass per node and would block startup, so it stays a manual dashboard action). The client doesn't initiate this — `refreshProbeList()` simply fetches the server's already-attached set on load, so the rack opens populated.

`RackDrawer` is the one shared browser for both families — the former `VectorsDrawer` + `ManifoldDrawer` folded into a single component parameterised by `family: "subspace" | "manifold"` (read off `params`, default subspace) and `mode: "steer" | "probe"`. Both rack `+ …` launchers and `RecipeBuilderDrawer`'s "browse…" route here. It lists `steerRack.catalog` filtered by family — subspace admits every flat affine fit (`fit_mode` `pca`/`baked`, so the 2-node concept axes *and* higher-rank flats like `personas`), manifold admits only curved fits (`spectral`/`authored`) — then splits on `fitted_for_session` into Fitted vs Unfitted and groups by `categoryOf(m.tags)`. `family` drives one `--family-accent` custom property (white `--accent` vs purple `--accent-purple`) colouring the title / section headers / row actions / launcher, plus the title text (`subspace` / `manifold`, `· probe` in probe mode) and the authoring launcher, which now routes to `ManifoldBuilderDrawer` for both families (`+ build manifold` — a flat fit is just a `pca` manifold, so there is no separate vector-extraction form).

Per Fitted row: `[ⓘ] [+steer] [+probe] [re-fit] [delete]`; Unfitted: `[fit] [delete]`. `+steer` routes by the drawer `family`: a flat fit joins as a subspace term (`addSubspaceToRack` — a 2-node concept defaults to its positive pole, a higher-rank flat to the domain centroid; magnitude is the shared `subspaceAlong` master), a curved fit as a manifold term (`addManifoldToRack`, own along/onto). A node-chip click (`onSteerNode`) racks-and-pins to that label the same way, by family (`setSubspaceLabel` / `setManifoldLabel`). `[+probe]` calls `attachProbe("ns/name")` for both; the steer/probe buttons short-circuit when already racked. `[ⓘ]` renders `lib/manifolds/DiagnosticsPanel.svelte` inline (PCA variance bars / spectral eigenvalue spectrum, picked-k cut in accent) plus the metadata + a `show layer norms →` deep-link to `LayerNormsDrawer`; in `probe` mode the `+probe`/`-probe` actions and the `attach by selector` form go through `attachProbe`/`detachProbe`/`probeRack`. Delete is a 2-step confirm (the button flips to `confirm?` for ~3 s). The pre-4.0 pack-backed `packsState` / `has_tensor` plumbing is gone — the drawer reads the live manifold catalog directly.

Flat-vector authoring forms were removed (the former `ExtractDrawer` *build vector* + `MergeDrawer` *merge vector*): a 2-node concept axis is just a `pca` manifold, so authoring folds into `ManifoldBuilderDrawer`'s auto-generated (`pca`) path, and the subspace `RackDrawer`'s `+ build manifold` launcher routes there. The backend `POST /extract` / `/vectors/merge` / `/vectors/bake` routes are unchanged — they simply have no dedicated UI form; SAE / `role` / `namespace` / DLS / overwrite options live on `ManifoldBuilderDrawer`'s `AdvancedSection`. CLI `saklas manifold extract` remains the scripted path.

`ManifoldMergeDrawer` and `ManifoldPacksDrawer` are the peer manifold tools (`PackDrawer` and `CloneDrawer` are gone — packs and cloning were removed in 4.0). Both launch from the ⌘K palette's `steering manifolds` group; the `RackDrawer` browser itself stays reachable only from the rack `+` launchers. **`ManifoldMergeDrawer`** is the discover-only node-union surface — checkbox list of discover-mode catalog rows, target-name input, and a `fit_mode` picker that defaults to inherit-from-sources when they agree and requires an explicit override when they don't, with a yellow warning naming the mixed modes. Submits `apiManifolds.merge` and closes back to the catalog on success. **`ManifoldPacksDrawer`** is the local-catalog + HF-search surface — two tabs (installed local rows from the shared `steerRack.catalog` store; HF `saklas-manifold` search debounced at 300 ms against `apiManifolds.search`, per-row `install` button calling `apiManifolds.install` and refreshing the local list on success). Both drawers' purple action accents echo the manifold-family colour the rack uses.

`ManifoldBuilderDrawer` has top-of-form `auto-generated` / `custom nodes` tabs (the internal `AuthoringMode` values stay `discover` / `authored` — only the labels rename). **Custom nodes** is the user-authored path: node editor with labels + statements + optional per-node coords, validated live against `min_nodes = 2n+1` and in-domain coordinates; Save posts to `POST /manifolds`. **Auto-generated** is concept-list + method-picker: a textarea of whitespace/comma-separated concept slugs, a `kind` (abstract / concrete) radio + a `samples_per_prompt` count (the A2 conversational corpus knobs — no scenario counts), a `pca` / `spectral` toggle. Both tabs share an `AdvancedSection` carrying hyperparams (`max_dim`, `var_threshold` for PCA / `k_nn` + `bandwidth` blank-means-auto for spectral) and toggles (`fit immediately after generating corpora`, `persona manifold (use each concept slug as that node's role)`, `overwrite an existing manifold`). Auto-generated save calls `apiManifoldGenerateStream` (SSE-streamed corpus generation) then optionally chains `apiManifoldFitStream` with the same hyperparams — both legs use sticky progress toasts. The diagnostics renderer in `lib/manifolds/diagnostics.ts` is pure helpers (`classifyDiagnostics`, `pcaBars`, `spectralBars`, `diagnosticsSummary`, `pickDiscoverFit`); the `DiagnosticsPanel.svelte` shell consumes them.

The custom-nodes tab grows an `auto-domain (let the fitter derive coords from corpora)` checkbox at the top. **Unchecked** (default) is the historical authored flow — box/sphere domain picker → per-node coord inputs → `POST /manifolds`. **Checked** hides the domain picker + the per-node coord inputs and exposes a `pca` / `spectral` fit-method radio (same hyperparams as the auto-generated tab, exposed via `AdvancedSection`); Save posts to `POST /manifolds/discover` with `{name, namespace, description, fit_mode, nodes: [{label, statements, role?}], hyperparams}` and the fitter derives node coordinates per-model. Reuses the same `discoverFitMode` / `discoverMaxDim` / `discoverVarThreshold` / `discoverKNN` / `discoverBandwidth` state slots as the auto-generated tab. The submit label changes to `build manifold (auto-domain <fit_mode>) → return to list` so the routing is visible.

`SteerCard`'s `XYPad` per-axis bounds for non-box (sphere / custom) domains are symmetric `[-R, R]` where `R = max(1, ceil(max|v|))` over each axis's `manifold.node_coords` column (clean whole-number ceiling over the per-axis max magnitude). 0 sits at the visual center, so the pad's crosshair gridlines + the slider midpoint align with the (0, 0, ...) **neutral** origin (the flat `node_coords` layout is neutral-centered at fit time, so the node centroid sits off-center and a persona node lands wherever its displacement from neutral falls — e.g. `c0 = -10` at the left endpoint with `R = 10`). Falls back to `[-1, 1]` when no fitted coords are on the wire (the unfitted-manifold pre-fit state). The store side: `setSubspaceLabel` / `setManifoldLabel(name, label|null)` toggle label-form, `formatSubspaceTerm` / `formatManifoldTerm` emit `<m>%<label>` when `label` is set otherwise the comma-joined coord list, and `lib/expression.ts`'s parser produces a `ManifoldTerm.label` (string) or `ManifoldTerm.coords` (number[]) that rides into the rack entry. Each snap-to-node option shows the node's optional `role` tag (`pirate [role=pirate]`) when set, surfacing persona manifolds.

Rack layer views are one `rack/LayerStrip.svelte` across CAA and pinned/live
J-LENS cards. Its `HeatmapCell` data marks have no outlines: a one-pixel gap
separates layers, neutral stays visible against the card, and the full-scale
green/red/J-LENS-blue endpoints stay at least 3:1 from neutral as well as the
card. The same cells in matrix drawers use one-pixel physical table spacing so
borderless neighbors never merge. A focused strip is an arrow-key layer scrubber
with a visible exact-value readout; the style guide renders the real component.
Every probe meter uses `rack/ProbeReadingRow.svelte`, which owns the exact shared
four-column grid across CAA, J-LENS, and SAE cards. `Select.svelte`, `RackCard`,
`SteerCard`, and `XYPad` all carry explicit `min-width: 0` shrink boundaries, so
switching a snapped position to `(free position)` cannot expand the rack or clip
past the inspector edge. Each `Select` instance owns unique option ids, lets Tab
follow native focus order, and places its listbox in the browser top layer with
viewport-clamped fixed coordinates so rack scrollers cannot crop it. App drawers
and the command palette are modal dialogs: the bench is inert while one is open,
focus is trapped inside and restored to the launcher on close. Global
`:focus-visible` owns one opaque `--focus-ring` that local dense-field styles
cannot suppress. Shared pointer controls have a 24 px minimum target.

The leading marker on every PROBE card means the same thing: persistence. It
pins/unpins J-LENS/SAE discovery cards and detaches an attached CAA/manifold
probe; transcript highlighting is a separate labelled action on CAA/manifold
cards. Disabled steer cards keep readable text and the re-enable action at full
contrast, using the hollow marker, struck name, and quieter surface instead of
whole-card opacity. Starting a background J-LENS fit is a two-step confirmation,
and a running fit exposes the server's cooperative cancel route while preserving
its resumable checkpoint.

`ManifoldBuilderDrawer`'s custom-nodes tab grows a per-node `role` input alongside coords (validated client-side against the same `[a-z0-9._-]+` slug regex the engine uses); the auto-generated tab grows a "persona manifold (use each concept slug as that node's role)" checkbox inside `AdvancedSection` that sets `GenerateManifoldRequest.role_per_node`. `RackDrawer` rows carry a `persona` `fit-badge` when any node in `node_roles` is non-null, parallel to the `pca` / `spectral` badges.

`TemplateLabDrawer` (⌘K `templates…`, `template_lab`) is the standalone templated-completion surface, two tabs. **score** — pick a template, optional steering expression, `rank by` sum/mean; runs `apiTemplates.score` twice (baseline + steered when an expression is set) and renders per-context distribution bars (base accent + steered purple, `N% → M%`), rows sorted by max prob. **build** — author a template: name, slot, values textarea (newline/comma split), and a multi-turn contexts editor (per context: role-tagged turn rows with add/remove + the slotted assistant field), validated client-side against the same invariants the server enforces (slot once in the assistant, not in a history turn, last turn user, ≥2 values). An `installed` catalog footer lists/deletes templates. The `ManifoldBuilderDrawer`'s `templated` tab remains as a one-step *single-turn* shortcut: it creates canonical single-turn contexts through `apiTemplates.create`, then derives the manifold through `apiManifolds.createFromTemplate`; `TemplateLabDrawer` is the multi-turn + scoring surface.

`RawBuffer` is the base-model surface.  `SessionInfo.is_base_model` (a non-chat model has no chat template) drives `genUiMode.effectiveRawMode()`; the `genUiMode.override` (`auto`/`chat`/`raw`, persisted per `model_id`, set from the AdvancedSamplingDrawer control or the cycling badge in the Chat header) wins when not `auto`.  In raw mode `Chat.svelte` renders `<RawBuffer />` instead of role bubbles — one continuous editable `pre-wrap` surface with the loom active path joined as plain text, no roles.

Flat mode is non-linear: editing text anywhere in the buffer and appending past its end are the *same* operation. `resolveDivergence()` diffs the draft against the settled buffer, finds the first changed character, and the tail from there becomes one new span. **send** submits that tail as `user` and generates `assistant`; a clean buffer shows **generate** and omits the authored half; the explicit **append** button submits only the `user` tail. There is no append modifier shortcut. All three use the same native `sendSubmit(..., {raw:true})` contract as chat. The divergence node and its subtree are preserved as the original branch—an edit never overwrites a generated span in place. The internal `committing` latch holds the buffer→draft sync across the server round trip so the typed tail does not flash out; it releases on a content check (`bufferText.startsWith(draft)`) and suppresses the tinted mirror while held. Toggling the mode never mutates the tree. Per-token tinting rides a read-only mirror layer behind the transparent-text textarea (a textarea cannot tint spans) and shows only when not actively editing.

`lib/expression.ts` — every term is a `%` position. `serializeExpression(rack, subspaceAlong = 1)` emits subspace terms first (each at the shared `subspaceAlong` magnitude) then manifold terms (each at its own `along[,onto]`), picking the production from `entry.mode`. `parseExpression(expr, { isFlat? })` returns `{ rack, subspaceAlong, warnings }`: a `%` term with an `onto` coeff or a non-flat catalog `fit_mode` lands `manifold`, else `subspace` (magnitude collected into `subspaceAlong`; a later subspace term whose magnitude differs folds onto the shared value with a warning); a bare-pole term (`0.5 formal.casual`) becomes a label-form subspace term toward the signed pole. The pre-4.1 `~`/`|` projection and `!` ablation still **parse** (so pasted expressions don't throw) but the operator is dropped with a `warnings` entry. The `%` coefficient slot is `along[,onto]`: serialize emits `<along>,<onto>` when `onto > 0` and `<along>` alone otherwise; the parser reads a pre-selector comma-run of ≤ 2 (mirroring the engine's `coeff := signed_float ("," signed_float)?`), lexically unambiguous from the post-`%` coord commas. `:variant` rides the atom (`name:sae%pos`) and survives on every term shape; keep the variant list in parity with the Python grammar (`raw`, `sae`, `sae-*`, `role`, `role-*`, `from`, `from-*`) — there is no `pca` variant. The round-trip invariant is parse(serialize(rack, G)) reproducing `rack` + `G` for any serializer output.

The current composer is role-neutral. Its two structural roles are always
`user` and `assistant`; `samplingState.user_role` / `assistant_role` are the editable
chat-template labels for those roles and seed once per model from
`SessionInfo.default_user_role` / `default_assistant_role` (Gemma therefore
shows `user` / `model`). Only genuine overrides lower to protocol `user_role` /
`assistant_role` in `buildSamplingPayload`. `CastDrawer` owns those two editable
labels through the Saklas `Combobox`, whose themed list combines both model
defaults with every genuinely custom role observed in the auto-derived tree cast;
the structural `user` / `assistant` roster keys are presented through those model
defaults instead of leaking into the list as extra roles. The composer
shows the resulting labels in ordinary `Select` controls; never reintroduce a native
`datalist`.

`Chat.svelte` exposes a visible two-part turn plan. **you write** selects the
authored role; **model writes** independently selects a generated role or `none`
for an authored-only append. Each visible role option carries its canonical
user/assistant role internally. A one-shot `⇄` exchanges the two selected roles; it
is not a persistent mode. Non-empty text with a generated role is **send**, empty
text with a generated role is **generate**, and selecting `none` is **append**.
There is no append shortcut. Selection chooses the branch anchor but never
changes these meanings. Scene mode gates nonstandard role choices because legacy
templates cannot open a user-role generation header or freely commit
assistant-role text.

Same effective roles coalesce throughout the engine. `append_user_turn` /
`append_assistant_turn` append authored text in place when the selected leaf has
the same structural role and `role_label`; `_generate_core` reuses such a leaf
for a one-shot generation, forcing its existing text as the prefix and sampling
the tail into the same message. This is the general prefill mechanism: append a
model-role prefix, then generate the model role. Explicit fan-out, regeneration,
logit forks, and the compatibility prefill endpoint retain sibling semantics.
An authored tail clears stale node-level generation receipts; a generated tail
rebuilds the receipt for the whole forced-prefix result. Exact draft whitespace
is preserved—only the emptiness test trims.

The composer footer has only the green primary action and red stop action. Both
use the same tinted-glass `Button` treatment; the action label changes among
`send` / `generate` / `append`. Regeneration is message-local: every non-system
bubble renders a small flat `↻` button immediately after its role chip and calls
`loomRegenerateNode(node_id)`. The replacement is a generated sibling in the
same structural role, so every non-system message is rerollable without
consulting provenance.

Every speaker still renders as one neutral glass card with identity in the role
chip; system nodes are stage directions. The cast manager (`cast…`) owns the
tree-scoped roster and standing recipes, and `+ thinking` drafts an authored
thinking block for the next line. Composer actions use shared Saklas controls
(`Button`, `Checkbox`, `Combobox`), as does the raw buffer's action surface.

Adding a panel: write the `.svelte`, wire state into the smallest matching `lib/stores/` slice (or `stores.svelte.ts` for cross-cutting WS/tree/chat state), mount from `App.svelte`, `npm run build`, commit the regenerated `dist/`. Adding a drawer: write it under `drawers/`, add the name to the `DrawerName` union in `lib/types.ts` (and to `NARROW_DRAWERS` in `App.svelte` for forms/pickers), add an `App.svelte` switch branch, re-export from `drawers/index.ts`, and add it to a `RAIL_CATEGORIES` group in `lib/commands.ts` so the ⌘K palette can launch it.

## Pending queue

Submissions during an in-flight gen (or behind earlier queued items) defer rather than racing the WS — same semantics as the TUI. `sendGenerate` / `sendCommit` / `sendPrefill` check `isPendingBusy()` (gen active OR `pendingActions.queue.length > 0`) and, when busy, append a `PendingAction` (defined in `lib/types.ts`) carrying a `rebuild` factory the `↑`-pull-and-edit path uses to re-encode the same kind/role/target with new text. Instant mutations from the chat header (`/clear`, regen) and the rack/sampling sites also queue via `enqueuePending` with `awaitsGen: false` so the drain chains through them without waiting on a `done` that never fires.

Queued rack mutations coalesce. `enqueueOrApply` tags each rack-mutation item with `coalesceKey: "rack"`; when the queue tail already carries that key, a fresh mutation chains its `apply` onto the tail item rather than appending a new slot, and the bubble's label updates to the latest action. A slider drag firing 30+ intermediate `setSubspaceAlong` calls mid-gen therefore leaves one queued bubble carrying the net effect. Coalescing stops at any non-rack item — rack changes before and after a queued send form distinct groups, so FIFO ordering relative to the send holds.

The WS `done` / `error` handlers call `drainNextPendingAction()` — one item per event — instead of the old v1 `applyPendingActions` (which drained everything at once). `PendingBubbles.svelte` renders the live queue above the input as dim chips; the per-bubble `×` calls `cancelPendingAction(id)` to remove a single slot. The bubble whose slot the user is currently editing via ↑ gets the `.editing` class — brighter amber background, thicker border, full-strength text, and a leading `✎` marker — driven off `inputHistory.pulledSlot`. The StatusFooter shows a `N queued` readout but no "apply now" button — under the FIFO model there's no skip-ahead semantics.

Up-arrow walks the combined ring `[editable pending (most-recent first), input history (newest first)]`. Pulling a queued item sets `inputHistory.pulledSlot`; Chat.svelte forwards that to `sendGenerate` / `sendCommit` / `sendPrefill` as `replaceSlot` so a re-edited send lands at its original slot. `Esc` while pulled cancels the edit (slot stays, input restores the stash); empty `Enter` while pulled removes the slot — keyboard equivalent of the `×` button. Non-editable items (`rebuild === null`, e.g. queued `/clear` and regen) sit in the queue but the up-arrow walks past them.

## Reactivity gotcha

Svelte 5's `$state` does NOT track `Map.set` / `Set.add` / inner-object property writes inside collections. Cross-component collections in `stores.svelte.ts` use `SvelteMap` / `SvelteSet` from `svelte/reactivity`. Inner-object mutations on map values are still untracked, so every rack mutator reassigns: `entries.set(name, {...e, coords})` — steerRack and probeRack alike (the `setSubspace*` / `setManifold*` setters are mode-guarded `mutateSubspace` / `mutateManifold` wrappers over the one `steerRack.entries`; the shared `subspaceAlong` is a scalar on the slice). `updateProbeFromScores` (driven by every WS `token` event) is the hot path here — a bare `entry.current = val` would freeze probe sparklines at zero through a whole generation.

## Persistence

The server loom tree is authoritative. The browser keeps a first-paint cache of the latest `LoomTreeJSON` plus `highlightState` in `localStorage` under `saklas.chat.v3.<model_id>`; only that exact version and the server's list-shaped `nodes` wire are accepted. There is no flat-log migration or synthetic tree hydration. Saves are debounced ~250 ms after mutations, and `refreshLoomTree()` overwrites the cache with server state once the required tree endpoint responds. `pendingIndex` is force-cleared on restore so an in-flight turn from a killed tab can't ghost the UI.

Downloaded conversations likewise use one exact schema (`version: 5`): complete Loom tree, visual steering rack (or an authoritative custom full-grammar expression), probe rack, highlight, and sampling sections are required. Loading an older or partial file is a visible error; the client never guesses at missing state.

## Per-token highlighting

Highlighting lives on the chat token spans, driven by a single highlight-probe dropdown in the chat header with an optional two-stripe compare-two mode. It tints **live** as tokens stream: the WS `token` event's `scores` aggregate feeds the same `scoreToRgb` ramp the post-generation pass uses, so streaming and finalized tints match. Authored spans gain the same token rows when their next generation prefill consumes them: the engine emits a `capture_authored` tree mutation carrying their original probe/J-LENS/SAE payloads, so user-written and model-written text share hover, highlight, and drawer behavior without an extra browser path. An authored-only append remains plain until a later generation actually forward-passes it. (v2 ramp note: `scoreToRgb` emits **constant-hue alpha ramps** — tint strength = opacity, hue = meaning — with `signed` green/red probe poles, blue surprise/J-LENS, and gold SAE families. Pinned SAE/J-LENS cards expose the same explicit `highlight` action as geometry probes; both read their native `[0,1]` strength on a unit saturation scale. The TUI still runs the old opaque ramp — a parity pass is deliberately deferred.) Hover and drawer history read the loom-owned `token.captured` channels directly, so refreshes, source switches, and explicit loom save/load preserve the original generation measurements without a browser retention cache or token-count cap. A channel disabled during that generation may still use its loom replay endpoint after the hover dwell; the drawer also replays for the explicit unsteered J-LENS counterfactual. Replayed values are never written back or mislabeled as original capture. Clicking any token opens the `token_drilldown` drawer regardless of whether a highlight probe is selected — four `SegmentedTabs` on one toolbar row (only **j-lens** and **sae** carry their pillar hue; the steered/unsteered A/B branch toggle sits right on the same row when the turn has an `abPair`): **probes** (the per-layer × per-probe heatmap), **logits** (ranked top-K alts + logit fork), **sae** (captured resident-hook top features, replay fallback), and **j-lens** (the captured all-fitted-layer readout — aggregate chip row then per-layer matrix — with replay fallback and an `apply recipe steering` checkbox for the unsteered counterfactual). Captured rows show their provenance and source even if that instrument is no longer active. A **token scrubber** in the drawer header (`◀ N / M ▶`, or `←`/`→` anywhere in the drawer outside a focusable field) walks the inspected position along the turn's token list while preserving the selected tab/branch; a fresh token click snaps back to its own index.

## Toasts

`lib/stores/toasts.svelte.ts` toasts carry `kind`, `message`, optional dim `detail` sub-line, and `ttlMs: number | null` — `null` is sticky (no auto-dismiss; caller owns dismissal).  `updateToast(id, patch)` mutates a live toast in place so long-running async work (manifold generate / fit) can drive a single chip from kickoff to completion without spawning new ones.  `Toaster.svelte` only schedules a dismissal timer the first time it sees a non-null TTL, so flipping sticky → ttl mid-flight is a no-op; callers that want a finite TTL at the end should `dismissToast` + `pushToast` instead.  `ManifoldBuilderDrawer`'s generate / fit legs are the canonical users — sticky progress on submit, dismissed and replaced with a 6 s success toast (or sticky error toast) when the SSE `done` / `error` lands.

## Out of scope

- True multi-session switching — server URL-paths support it; the client still assumes `default`. `SessionAdminDrawer` inspects the collection and sets an in-memory bearer key but is not a session router.
- Persistent credential management — the bearer key stays in memory for the page session, never written to `localStorage`.
- Mobile / touch-first layout — desktop research tool, min-width 1280px.
- Combobox autocomplete on the projection-target picker (free-form name input).
- Pagination on HF pack search (capped at 20 results).
