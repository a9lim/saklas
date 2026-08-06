# web/

Static Svelte 5 + Vite dashboard mounted at `/` by `saklas serve`. The CLI mounts
it by default (`--no-web` opts out); `create_app(..., web=False)` is the library
default so embedded API surfaces don't pick up the dashboard.

## Package + build

```
saklas/web/
  __init__.py        # re-exports: register_web_routes, dist_path
  routes.py          # mount logic + SPA fallback
  dist/              # COMMITTED build artifact, ships in the wheel
    index.html  favicon.ico  LICENSE-Recursive.txt
    assets/{index.css, saklas.js, Recursive_VF.woff2}
```

Svelte source lives at the repo's `webui/` (peer of `saklas/`); `cd webui && npm
run build` emits straight into `saklas/web/dist/` — no intermediate copy. CI runs
`npm ci`, `npm run check`, `npm run build`, then `git diff --exit-code
saklas/web/dist/`, so a source change that isn't accompanied by a rebuilt bundle
fails. `npm run check` is `svelte-check` plus `scripts/check-theme.mjs`, which
fails when a referenced CSS custom property has no declaration in the style
tokens.

## Mount (`routes.py`)

`register_web_routes(app)` mounts `/assets/*` on `StaticFiles` with its default
ETag / Last-Modified revalidation — Vite emits fixed filenames
(`assets/saklas.js`, `assets/index.css`), not content-hashed ones, so an
immutable long-lived cache would serve a stale rebuild. It registers `GET /` →
`index.html` and a catch-all `GET /{full_path:path}` that 404s `styleguide[/…]`
and anything under `api`/`v1`/`saklas`, serves the allowlisted top-level dist
files (every top-level file except `index.html` — `favicon.ico` and the Recursive
licence), and otherwise falls back to `index.html` for SPA routing. `full_path` is
only ever a dict key, never a path component, so `..` traversal is structurally
impossible. `create_app` registers the catch-all last so it never shadows
`/v1/*`, `/api/*`, `/saklas/v1/*`. Repeat calls append a second mount and
duplicate routes that the first ones shadow — harmless, not rejected.

`dist_path()` resolves through `importlib.resources` (editable + wheel).
`WebUINotBuilt` raises on mount when the dist directory is empty — only in source
installs that haven't run `npm run build`.

## Wire protocol

The dashboard is the sole client of the native `/saklas/v1/*` API;
`server/AGENTS.md` is the route catalogue. `lib/api.ts` is a client for the routes
the dashboard actually calls, **not** an exhaustive mirror — an absent method
means "no surface needs it yet", never "the server lacks it". Session id is the
literal `default`; the bearer key is read once from `<meta name="api-key">` and
kept in memory.

**WS `/sessions/{id}/stream`** carries generation. Three client frames:

- `submit` — the composer's one door: `{text, authored_role, generated_role,
  steering, sampling, thinking, authored_thinking?, raw, parent_node_id?, n?,
  recipe_override?}`. `generated_role: null` is an authored-only append; `text:
  null` continues from the selected leaf. Both roles are structural
  (`user | assistant`) and independent.
- `generate` — the specialist door: a bare continuation (`input: null`, used by
  regenerate / continue-from-committed), an explicit `WSInputMessage[]` replay
  (`{role, content, label?}` — the unsteered A/B shadow), the **logit fork**
  (`fork_node_id` / `fork_raw_index` / `fork_alt_token_id` travel together and
  replay the node's raw decode prefix with one token swapped), and
  `generate_seat`. The client has no commit/prefill fields to emit.
- `stop`.

Server frames are `started` / `token` / `done` / `error` / `tree_mutated`.

**The measurements envelope** is the single read-side record. Every measured
`token` frame carries `measurements`: `version`, `scope`, `provenance`, the flat
cross-family `scores` / `per_layer_scores` views, and per-family `instruments` —
`geometry` (attached-probe `readings`), `lens` and `sae` (`readings` plus the
native `readout` discovery block and a `binding` recording source + recipe
steering, plus the resident `layer` for sae). It is byte-for-byte the object
stored on the backing loom token row, and `done.result.measurements` is its
`scope: "aggregate"` sibling; the native `done` frame carries no flat
`probe_readings`. `_mergedReadings` folds the three families' `readings` into one
name-keyed dict, because the probe rack keys by name across families.

Reading shapes differ by family and the shape is the discriminator
(`isScalarReading`): geometry emits `ProbeReadingJSON` (`coords` / `fraction` /
`nearest` / `residual` + per-layer traces), lens and SAE emit `ScalarReadingJSON`
(`value`, an explicit `unit` — `mean_token_probability` /
`activation_over_max` / `raw_activation` — `per_layer`, and `depth: {center,
spread, basis}`). `_primaryScalar` / `_primaryPerLayer` branch on that shape, not
on probe-info flags. `raw_index` is the decode-step join into the node's
`raw_token_ids`.

**Instruments** live under `/sessions/{id}/instruments/{family}/…`, `family ∈
geometry | lens | sae`: `POST live` (geometry = the per-token monitor scoring
switch, lens = the workspace readout with an optional layer list, sae = feature
discovery), `GET sources`, `PUT source` (lens), the polled/cancellable
`POST/GET/DELETE preparations` job (lens `fetch` | `fit`, sae `fetch` | `train`),
`GET token-readout` (a `scope="replay"` measurements envelope for any family),
plus the family extras `POST lens/token/validate`, `POST sae/features/validate`
and `POST sae/features/metadata`. `apiInstruments` is the single client;
`lib/stores/preparations.svelte.ts` adapts the job resource.

**Session info** carries `instruments: InstrumentFamilyBlock[]` — `{family, live,
source, probes, capabilities}`, the same block `GET .../instruments` lists, so
there is one representation of instrument state. `instrumentFamily(f)` reads it;
`refreshSession` rehydrates `lensState.layers`, `saeState.live` / `release` /
`layer`, and `probesLiveState` from it, resetting SAE discovery + metadata when
the resident release or layer changes. `saeLoaded()` is "the sae block has a
source". `jlens_fitted` stays its own flat key — it is a sidecar/weight-identity
path check rather than instrument state, and it gates the drilldown's j-lens tab.

**Probe rows** (`GET /probes`) are a `family`-discriminated union:
`GeometryProbeInfo` (`manifold`, `top_n`, `layers`, `node_labels`, `node_count`,
`domain`, `intrinsic_dim`, `feature_space`, `is_affine`, `node_coords`),
`LensProbeInfo` (`layers`, `word`, `token_id`), `SaeProbeInfo` (`layers`,
`feature_id`, `label`, `max_act`). The readout families carry no invented
geometry fields. `is_affine` is the flat-vs-curved discriminator the client
classifies subspace-vs-manifold on; `node_coords` backs the 2-D mini-map.

## Source layout

```
webui/src/
  main.ts                     # bootstrap: mount <App />, install the tooltip layer
  App.svelte                  # shell + drawer host; renders the DRAWERS registry row
  lib/
    api.ts                    # typed REST + WS + SSE clients; ApiError + describeError
    stores.svelte.ts          # THE barrel: re-exports every stores/ slice, nothing else
    stores/                   # one slice per concern; slices import each other directly
      session.svelte.ts       # the mirror of GET /sessions/current
      sampling.svelte.ts      # sampling controls + the wire payload they build
      steering.svelte.ts      # the unified steer rack (subspace / manifold / atom terms)
      probes.svelte.ts        # the unified probe rack + the transcript-highlight target
      instruments.svelte.ts   # lens / SAE / geometry live state, sources, preparations
      ws.svelte.ts            # the singleton socket, its dispatcher, the send primitives
      loom.svelte.ts          # the client mirror of the server's LoomTree
      loomUi.svelte.ts        # threads column: modals, edge labels, filter, selection
      chat.svelte.ts          # the chat-log projection + per-generation counters
      ab.svelte.ts            # the A/B shadow generation + auto-regen recipe-override
      pending.svelte.ts       # the deferred-mutation queue
      persistence.svelte.ts   # the localStorage half of the presentation preferences
      bootstrap.svelte.ts     # the mount-time fan-out over all of the above
      drawers, inputHistory, palette, preparations, toasts
    types.ts                  # THE import surface: re-exports types.gen.ts, owns WS + request + UI types + the DrawerName union
    types.gen.ts              # GENERATED REST response types (scripts/generate_webui_types.py) — do not edit
    highlight.ts              # THE per-token highlight implementation (score lookup + style)
    tokens.ts                 # pure ramp math: HIGHLIGHT_SAT, scoreToRgb, two-stripe/blend
    templates.ts              # THE client mirror of the template artifact's invariants
    expression.ts             # serialize the steer rack to a steering expression
    readouts.ts               # the one shared readout top-k resolution
    concepts.ts               # concept-catalog helpers (category / poles / recommended α)
    commands.ts               # ⌘K index: paletteCommands() over the drawer registry
    tooltips.ts               # delegated tooltip layer; `title` stays the authoring API
    charts/{Bar,Sparkline,HeatmapCell}.svelte
    charts/probeGeometry.ts   # hand-rolled canvas renderer for the probe-inspector plot
    manifolds/diagnostics.ts  # pure classify/bars/summary helpers for discover fits
    manifolds/DiagnosticsPanel.svelte
    {Slider,Select,Checkbox,Radio,Combobox,NumberInput,Disclosure,Toaster}.svelte
    builder/{ModeTabs,AdvancedSection,ValidationBlock}.svelte
    ui/{Button,Chip,SegmentedTabs,DrawerCloseButton}.svelte
    style/{fonts.css,tokens.css,global.css}
                              # fonts.css: Recursive VF self-hosted, ONE woff2 as TWO
                              # families — "Recursive Sans" (MONO 0, CASL .35) for chrome,
                              # "Recursive Mono" (MONO 1) for data; axes pinned per-family
                              # in @font-face, so a font-family switch flips the voice
  panels/
    InspectorPanel.svelte     # the instrument stack: RecipeBar + four pillar tabs
    RecipeBar.svelte          # persistent expression bar — every term as a pillar chip
    SteeringRack.svelte       # STEER section of one geometry tab (family prop)
    ProbeRack.svelte          # PROBE section of one geometry tab (family prop)
    JLensPanel.svelte         # lens pillar: SOURCE + STEER + merged PROBE
    SaePanel.svelte           # sae pillar: same shape, gold
    CommandPalette.svelte     # ⌘K — THE tool launcher (state in lib/stores/palette)
    Chat.svelte               # cast-model turn cards, composer, highlight controls
    RawBuffer.svelte          # base-model flat completion buffer
    StatusFooter.svelte       # gen progress · t/s · elapsed · ppl + queued-count badge
    PendingBubbles.svelte     # ghosted queued-action chips + per-item × cancel
    WorkbenchCard.svelte      # model + device card, foot of the threads column
    SamplingStrip.svelte      # T / P / K / max / penalties / seed / thinking / alts
    loom/{LoomSidebar,LoomNode,LoomEdge}.svelte  # the permanent "threads" column
    manifold/{XYPad,ManifoldMiniMap}.svelte
    rack/
      RackCard.svelte         # shared card chrome (statline over stacked controls/meters)
      RackSectionHeader.svelte, RackMarker.svelte, LayerStrip.svelte
      ProbeReadingRow.svelte, ProbePinButton.svelte, ProbeHighlightButton.svelte
      SteerCard.svelte        # geometry steer card (subspace | manifold)
      AtomSteerCard.svelte    # jlens/sae steer card (one component, family table)
      ProbeCard.svelte        # geometry probe card
      JLensProbeCard.svelte, SaeProbeCard.svelte   # pinned + discovery rows
      InstrumentSourceSection.svelte  # shared SOURCE selector + custom lifecycle row
      probeRows.ts            # mergeInstrumentProbeRows(pinned, discovered, sort)
      triggers.ts             # trigger pill labels + cycle order
  drawers/
    {Rack,ManifoldBuilder,ManifoldMerge,ManifoldPacks,TemplateLab,ProbeInspector,
     TokenDrilldown,Correlation,Compare,NodeCompare,Transcript,SaveConversation,
     LoadConversation,Cast,SystemPrompt,AdvancedSampling,Health,SessionAdmin,
     Help}Drawer.svelte
    manifold/                 # builder internals (ManifoldBuilderDrawer is the shell)
      shared.ts               # identity slugs + the shared discover tuning/hyperparams
      form.css                # the builder's field/step/node rules, scoped `.mb-form`
      FitMethodPicker.svelte, DiscoverTuningFields.svelte
      {Authored,Discover,Templated}Form.svelte
    token/                    # drilldown internals (TokenDrilldownDrawer is the shell)
      cursor.ts               # conversation-walking cursor: segments, step/jump/clamp
      readout.svelte.ts       # ReplayReadout — one captured-or-replay resource per family
      drilldown.svelte.ts     # page-session sticky tab state (default: lens)
      InstrumentHeader.svelte # shared provenance · source · steering · apply-recipe row
      DetailCardHeader.svelte, DetailSection.svelte, EvidenceChips.svelte
      TokenRibbon.svelte, PinnedReadings.svelte, EmptyState.svelte
      {Geometry,Logits,Sae,Lens}Tab.svelte
    index.ts                  # THE drawer registry: Record<DrawerName, DrawerEntry>
```

`lib/stores.svelte.ts` is a re-export barrel and nothing else — every component
keeps importing `from "./lib/stores.svelte"` and sees one flat surface, while the
state lives one slice per concern under `lib/stores/`. Slices import each other
*directly*, never through the barrel: the graph is genuinely cyclic in places
(loom ↔ ws, chat ↔ pending), which ES modules handle because every cross-slice
reference happens inside a function body — no module-level initializer reads an
imported binding. When adding state, add a slice and a barrel line rather than
growing an unrelated one.

## Shell + layout

`App.svelte` is three permanent columns — threads `376px` · chat `minmax(0,1fr)` ·
rack `432px` — with a 1280px design floor (1280 − 376 − 432 ≈ 472px of chat).
Below `1279px` the same three surfaces become explicit full-width views behind a
`threads / chat / instruments` nav, so every workflow stays reachable without
shrinking dense scientific controls into illegibility. It also owns the boot
gate, the global accelerators (Esc priority: palette → loom modal → drawer → stop
gen; ⌘K; ⌘R/E/B/N/D loom ops), and the drawer host.

Drawers are modal: the bench columns are `inert` while one is open, focus is
trapped inside and restored to the launcher on close (`lib/stores/drawers`
remembers the opener). The host paints the floating sheet — `--bg-alt` fill,
`--glass-line` hairline, `--radius-lg`, overlay shadow — so every drawer interior
is transparent with no border of its own.

Every drawer is one row of `drawers/index.ts`'s `DRAWERS: Record<DrawerName,
DrawerEntry>` — component, fixed `params` (how one `RackDrawer` serves both the
`subspace` and `manifolds` names), `narrow` sizing (forms and pickers get
`min(480px, 92%)` instead of the wide analysis panel), and reachability. The host
renders `DRAWERS[drawerState.open]` through one dynamic component; there is no
per-drawer branch, and a name added to the `DrawerName` union without a row is a
compile error. Reachability is a typed either/or: a `launcher` (palette group +
label + keywords) or `launcher: null` plus a `via` string naming the surface that
opens it. `RAIL_CATEGORIES` is **derived** from those launchers, so a drawer
cannot be declared, rendered, and still unreachable — the failure the
hand-maintained list allowed.

Sheet-interior grammar (`ProbeInspectorDrawer` and `TokenDrilldownDrawer` are the
reference implementations): header = eyebrow (tracked caps `--text-xs` /
`--weight-medium` / `0.08em`) over a name row (mono `--text-md` subject +
`--fg-subtle` meta) with a 26px circle close button; body padding
`--space-5/--space-6`; data wells recessed into a `--bg` container. Chrome speaks
`--font-ui`; every value / identifier / expression / number sits in `--font-mono`.
**Borderless:** the fill step against the sheet is the well boundary, spacing +
eyebrow typography are the seams, and sticky header/label cells separate with
`--shadow-sticky`. Hairlines survive only as meaning — focus rings,
active/selected/invalid state, floating-surface outer edges, dashed pending
ghosts, control glyph strokes. Exclusive-choice strips are `lib/ui/SegmentedTabs`,
actions are `lib/ui/Button`. Hue stays ontological: at most one pillar-owned
surface per drawer carries its hue; everything else is achromatic.

## The instrument stack

`InspectorPanel` is four coequal pillars over the ONE steering expression and
probe roster, each with the same verbs (observe / steer / gate) and its own hue +
marker: **subspace** (flat/affine fits — concept axes, `personas`;
`--pillar-subspace`, ●/○), **manifold** (curved fits; `--pillar-manifold`, ◆/◇),
**sae** (`--pillar-sae`, ▲/△), **lens** (`--pillar-lens`, ■/□). `InspectorTab` is
exactly those four names. The subspace and manifold tabs render
`SteeringRack` + `ProbeRack` filtered by a `family` prop — the tab *is* the group;
`sae` is `SaePanel` and `lens` is `JLensPanel`. `RecipeBar` sits above the tabs and
keeps every racked term visible as a pillar-colored chip regardless of which tab
is open (click → jump to its pillar, × → remove, ⧉ → copy the canonical
expression, i.e. the exact text the WS `steering` field carries). Hovering a
transcript or raw-buffer token overlays that token's readings into the same cards.

The rack split is **subspace (flat/affine) vs manifold (curved)** — a steering
vector is the 2-node flat case of a manifold, and "manifold" means a genuinely
curved fit. Every row wears one `RackCard`: dense quiet glass (`--radius-lg`,
borderless at rest; hover lifts the fill, the border slot exists only for the
`active` hue ring; no backdrop blur — cards sit on an opaque panel), a statline on
top (marker · name · status chips · actions, one row) with controls or meters
stacked **vertically below**, never inline. `RackMarker` gives all four glyph
shapes one optical box so family reads from hue + marker alone.

**Steer classification.** One `steerRack` of tagged entries keyed by name.
`mode` is the family itself: `subspace` / `manifold` (geometry positions) and
`jlens` / `sae` (single-direction atoms). It is set at add time — `RackDrawer`
reads the catalog `fit_mode` (`pca`/`baked` → subspace, `spectral`/`authored` →
manifold) — and never flipped in the UI. `SteeringRack` groups by `entry.mode`
directly. Every subspace term shares the rack-level **`subspaceAlong`** master
(the merged affine subspace slides once) — one slider in the subspace group
header; manifold terms keep a per-card `along` + `onto`.

`SteerCard` (geometry) branches its body on `entry.mode` via narrowed `s` / `m`
deriveds so svelte-check enforces mode-correct field access. Statline: enable
glyph · name · `unfitted`/`stale` chip · trigger pill · ✕. Body: a snap-to-node
`Select` with a `(free position)` escape hatch plus the `XYPad` (rendered `locked`
while a label is bound). Subspace adds no per-card magnitude — a hint line points
at the shared master. Manifold adds `along` + `onto`. Picking a node sets the
entry's `label` (`<m>%<label>`); pulling the pad authors free coords and clears it.
`XYPad` is a 2-D draggable pad at intrinsic dim 2 and one slider per axis
otherwise, so a 2-node concept is one slider and an 8-D `personas` fit is eight.
Non-box (sphere / custom) axes get symmetric `[-R, R]` bounds where
`R = max(1, ceil(max|v|))` over that axis's `node_coords` column, so 0 sits at the
visual center — the flat layout is neutral-centered at fit time, so a persona node
lands wherever its displacement from neutral falls. `[-1, 1]` is the fallback
before a fit exists.

`AtomSteerCard` is the one card for both single-direction families (`jlens/<word>`,
`sae/<id>`); a small family table supplies the accent, marker and noun. Statline:
enable marker · atom id · trigger pill · ✕; body: one per-card α slider. Atoms run
hotter than concept vectors — `ATOM_DEFAULT_ALPHA = 0.3` is the coherent sweet
spot, `≥ 0.5` over-steers into repetition — which is why they carry their own dial
rather than the shared master. `atomActions(mode)` returns the five mutators;
`addJLensToRack(word)` / `addSaeToRack(id)` stay family-specific because they
differ in key construction and validation.

**Probes.** One `probeRack` keyed by registered name across all three families.
Each entry carries the server `ProbeInfo`, a sparkline of the primary scalar, the
latest per-token `reading`, the end-of-gen `aggregate`, the recent `nearest`, and
a 2-D `trajectory`. `refreshProbeList` / `attachProbe(selector, opts?)` /
`detachProbe(name)` cover the REST surface; `updateProbesFromReadings` (WS
`token`) and `setProbeAggregates` (WS `done`) both read the merged
`instruments.*.readings`. `probeEntryForDisplay(name)` is the hover overlay,
synthesizing a family-native reading when only a flat score row is available.

`ProbeCard` renders the **geometry** family only (`ProbeRack` filters the readout
families out to their own tabs). Statline: pin marker (click detaches) · name ·
`@com ±spread` · ⓘ inspect · highlight action · sparkline. Body: the subspaceness
row (white 0→1 `fraction` bar · nearest node), then one signed bar per intrinsic
axis — pole labels only when the fit is rank-1 with ≤ 2 nodes, `c0…cR-1`
otherwise — then the curved-only `residual` meta, the per-layer `LayerStrip`, and a
`ManifoldMiniMap` for 2-D `BoxDomain` probes with node coords. The heatmap /
tint saturation scale is the axis-0 node extent for a flat probe and the fixed
unit scale for a curved probe's `[0,1]` fraction.

`JLensProbeCard` and `SaeProbeCard` render both the pinned probe rows and the
unpinned live-discovery rows of their pillar — pinning changes persistence and
actions, never shape or position. `mergeInstrumentProbeRows(pinned, discovered,
sortMode)` in `rack/probeRows.ts` is the one merge + sort for both panels
(`strength` / `name` / `depth`, with natural-number name collation for SAE
feature ids so `sae/9` precedes `sae/10`; rows with no depth CoM sort last).

`LayerStrip` is the one per-layer view across every pillar: `HeatmapCell` marks
with no outlines, a one-pixel gap between layers, endpoints at least 3:1 from
neutral and from the card. A focused strip is an arrow-key layer scrubber with an
exact-value readout. `ProbeReadingRow` owns the shared four-column
label/bar/context/value grid. `Select`, `RackCard`, `SteerCard` and `XYPad` all
carry explicit `min-width: 0` shrink boundaries so unsnapping a position cannot
expand the rack past the inspector edge; each `Select` places its listbox in the
browser top layer with viewport-clamped fixed coordinates so rack scrollers can't
crop it. Global `:focus-visible` owns one opaque `--focus-ring` that dense local
field styles cannot suppress; shared pointer controls have a 24px minimum target.

**Live toggles.** The geometry PROBE header carries the CAA live switch
(`probesLiveState` → `apiInstruments.setLive("geometry")`, rendered in both
geometry tabs driving the same state): off ⇒ no per-token monitor scoring for UI /
loom consumers (probes settle to end-of-gen aggregates; gates still force the
subset they need). The lens and SAE PROBE headers carry their own. With all three
off a compute-constrained session pays no per-token read cost at all.

**Instrument SOURCE sections.** `InstrumentSourceSection` is one shared
prepared/provider selector plus the canonical labelled custom lifecycle row, used
by both JLensPanel and SaePanel; it owns the `selectedSource` reconciliation
because it owns the options list. Behind it,
`createPreparationSlice(family, operation, {label, intervalMs, successMessage,
onSettled})` builds all four background jobs (lens `fetch`/`fit`, sae
`fetch`/`train`) on one contract: `start` posts, reflects the returned status and
fires the poll loop without awaiting completion; the loop is re-entrancy-guarded
so N panels asking share one interval; `cancel` no-ops unless running; `check` is
the mount-time resume probe and only reflects a job when it is *this* operation
(the two operations of a family share one resource); settle toasts are uniform
(cancelled → info, error → sticky, else the success line) and run after
`onSettled` so the panel is consistent first. Local fit/train fields stay hidden
until the synthetic `local` selector option is chosen, and prepared cached sources
win the default selection.

## Steering expression

`lib/expression.ts` is **serialize-only** — the dashboard emits the grammar and
the server parses it; there is no client-side parser. `serializeExpression(rack,
subspaceAlong)` emits subspace terms first (each at the shared master), then
jlens, then sae, then manifold terms (each at its own `along[,onto]` — `onto`
rides the coefficient slot only when > 0), picking the production from
`entry.mode` and skipping disabled entries. Position is the label form
(`personas%hacker`) when `entry.label` is set, else the comma-joined coord list.
`:variant` rides the atom and survives round-trip; keep the variant list in parity
with the Python grammar (`raw`, `sae`, `sae-*`, `role`, `role-*`, `from`,
`from-*`). `currentSteeringExpression()` returns `steerRack.customExpression` when
one is set, else the serialized rack; `applyCustomSteeringExpression` clears the
visual rack, because a rack that can't represent every projection/gate form would
lie about what generation uses.

## Drawers

`RackDrawer` is the one catalog browser for both geometry families, parameterised
by `family` (read off `params`, default subspace) and reached from every rack `+`
launcher. It filters `steerRack.catalog` — subspace admits every flat affine fit
(`pca`/`baked`, so 2-node concept axes *and* higher-rank flats like `personas`),
manifold only curved fits (`spectral`/`authored`) — splits on `fitted_for_session`
into Fitted vs Unfitted, groups by `categoryOf(m.tags)`, and searches over
manifolds and their node labels. `family` drives one `--family-accent` custom
property plus the title text. Per Fitted row: `[ⓘ] [+steer] [+probe] [re-fit]
[delete]`; Unfitted: `[fit] [delete]` (delete is a two-step confirm). `+steer`
routes by family (`addSubspaceToRack` — a 2-node concept defaults to its positive
pole, a higher-rank flat to the domain centroid — vs `addManifoldToRack`); a node
chip click racks-and-pins that label. `[ⓘ]` renders `DiagnosticsPanel` inline
(PCA variance bars / spectral eigenvalue spectrum, picked-`k` cut in accent). The
`+ build manifold` launcher routes to `ManifoldBuilderDrawer` for **both**
families — a flat fit is just a `pca` manifold, so there is no separate
vector-extraction form in the dashboard; `POST /extract` remains the server route
and `saklas manifold extract` the scripted path.

`ManifoldBuilderDrawer` is a shell — header, `auto` / `template` / `custom` mode
tabs, and the shared identity fields (namespace / name / description) — over three
disjoint sibling forms in `drawers/manifold/`. The internal `AuthoringMode` values
are `discover` / `templated` / `authored`; the shell hands each form the raw
identity and each slugs it at submit. `manifold/form.css` carries the field / step
/ node-card rules under the `.mb-form` class on the shell's body, so the forms
share one visual surface without duplicating it, and `FitMethodPicker` +
`DiscoverTuningFields` are the two blocks more than one form needs.

**`AuthoredForm`** (custom) is the authored path — label + statements + optional
per-node coords and `role`, validated live against `min_nodes = 2n+1` and
in-domain coordinates, `POST /manifolds` — with an `auto-domain` checkbox that
hides the domain picker and coord inputs, exposes the `pca`/`spectral`/`auto`
radio, and posts to `POST /manifolds/discover` so the fitter derives coords
per-model. **`DiscoverForm`** (auto) is a concept-slug textarea plus a `kind`
(abstract/concrete/custom) radio and a `samples_per_prompt` count, calling
`apiManifoldGenerateStream` (SSE) and optionally chaining `apiManifoldFitStream`
with the same hyperparams; both legs drive one sticky progress toast.
**`TemplatedForm`** picks an existing template and derives a manifold from it
(`apiManifolds.createFromTemplate` + the optional fit). It does **not** author
templates — `TemplateLabDrawer`'s build tab is the one editor, deep-linked from
here via `openDrawer("template_lab", { tab: "build" })`. A per-form
`AdvancedSection` carries the hyperparams (`max_dim`, `var_threshold`, `k_nn`,
`bandwidth`) and the fit-immediately / persona-manifold / overwrite toggles;
`manifold/shared.ts` owns the slugging and the tuning → hyperparams mapping so
the two discover-routed forms build the same request body.

`ManifoldMergeDrawer` unions discover-mode node corpora (checkbox list, target
name, a `fit_mode` picker defaulting to inherit-from-sources and demanding an
explicit override when sources disagree). `ManifoldPacksDrawer` is the local
catalog plus HF `saklas-manifold` search (debounced 300ms, server-capped at 20
rows), installing through `apiManifoldInstallStream` and rendering the SSE stage
line under the row. Both launch from the ⌘K palette's Steering group.

`TemplateLabDrawer` is the standalone templated-completion surface. **score** —
pick a template, an optional steering expression, `rank by` sum/mean; runs
`apiTemplates.score` twice (baseline + steered) and renders per-context
distribution bars. **build** — THE template editor: name, slot, values and a
multi-turn contexts editor, opened directly or deep-linked with `params: { tab:
"build" }` from the builder's template form. It validates
through the one `lib/templates.ts::validateTemplateDraft`, which mirrors
`saklas/io/templates.py::_validate_body` / `_validate_context` one-for-one — slot
exactly once in the final assistant turn and absent from history turns, last turn
`user`, ≥ 2 values, distinct non-empty value slugs under the engine's
`_slug_value` — so a draft that passes locally is a draft the server accepts.

`ProbeInspectorDrawer` (`probe_inspector`, from a `ProbeCard`'s `[ⓘ]`) is the
per-probe geometry inspector in the whitened (Mahalanobis) frame, rendered by the
hand-rolled canvas in `lib/charts/probeGeometry.ts` (zero 3-D dependency). The
plot branches on the selected layer's subspace rank: 1 → a line (poles + neutral +
sliding live dot), 2 → a node scatter (+ curve overlay at `intrinsic_dim == 1`),
3+ → a drag-orbit 3-D scatter on the top-3 PCs (+ curve / wireframe overlay).
Geometry is fetched once for all layers (`GET /probes/{name}/geometry` →
per-layer `node_white` / `neutral_white`, a top-3 PCA rotation at rank ≥ 3, the
sampled overlay, all whitened) and reprojected client-side on scrub; the per-layer
`‖share‖` bars are the layer scrubber. At rank ≥ 3 the scale is a
rotation-invariant constant from the static framing set and the orbit is a rigid
spin about the **neutral anchor**, so neutral sits at the plot center, the cloud
shows its real displacement, and dragging never resizes it; zoom is an explicit
scroll-wheel `orbit.zoom` clamped `0.3–6×`. Colour follows the hue ontology: node
centroids wear the probe's family hue via the `--geom-node` custom property (the
renderer reads its palette off CSS vars and stays hue-agnostic), neutral is a
hollow grey ring (neutral is the absence of concept, so it carries no hue), the
live point is a white core with a green rim, and the trail is a green time ramp —
the one place a gradient is allowed, because the gradient *is* the data. The live
point and trail ride the probe's per-token `subspace_coords_per_layer`, gated on
by the `persist_subspace_coords` sampling flag that `buildSamplingPayload` sends
whenever any probe is attached — so opening the inspector *after* a generation
still shows that run's path. Samples are stored across **all** probed layers in
`subspaceTrail` (a bounded ring, reset per generation), so scrubbing layers is a
pure read.

Other drawers: `CastDrawer` (tree-scoped roster of labels + standing recipes, and
the two editable structural-role labels through `Combobox`), `Correlation` (N×N
magnitude-weighted cosine), `Compare` (cross-layer pairwise cosine),
`NodeCompare` (cross-branch diff, from the loom sidebar), `Transcript`
(export/import), `Save`/`LoadConversation` (the v6 whole-tree file),
`SystemPrompt` and `AdvancedSampling` (from `SamplingStrip`), `Health`,
`SessionAdmin`, `Help`.

## Composer, cast, raw buffer

The composer is role-neutral. Its two structural roles are always `user` and
`assistant`; `samplingState.user_role` / `assistant_role` are the editable
chat-template labels, seeded once per model from `SessionInfo.default_user_role` /
`default_assistant_role` (Gemma therefore shows `user` / `model`), and only
genuine overrides lower to protocol `user_role` / `assistant_role` in
`buildSamplingPayload`. `CastDrawer` owns those two labels through the Saklas
`Combobox`, whose list combines the model defaults with every genuinely custom
label observed in the auto-derived tree cast; the composer shows the resulting
labels in ordinary `Select` controls (never a native `datalist`).

`Chat.svelte` exposes a visible two-part turn plan: **you write** selects the
authored role, **model writes** independently selects a generated role or `none`
for an authored-only append, and a one-shot `⇄` exchanges the two (not a
persistent mode). Non-empty text with a generated role is **send**, empty text
with a generated role is **generate**, `none` is **append**; there is no append
modifier shortcut, and node selection chooses the branch anchor without changing
those meanings. Scene mode gates nonstandard role choices, because a family
without a validated scene grammar cannot open a user-role generation header or
freely commit assistant-role text. `+ thinking` drafts an authored thinking block
for the next line. Every speaker renders as one neutral glass card with identity
in the role chip; system nodes are stage directions. Regeneration is
message-local — a small
`↻` after each non-system role chip calls `loomRegenerateNode(node_id)` and the
replacement is a generated sibling in the same structural role. The footer has
only the green primary action (label `send` / `generate` / `append`) and the red
stop action.

Same effective roles coalesce throughout the engine: authored text appends in
place when the selected leaf shares its structural role and `role_label`, and a
one-shot generation reuses such a leaf by forcing its text as the prefix — the
general prefill mechanism. Explicit fan-out, regeneration, logit forks and the
compatibility prefill endpoint keep sibling semantics.

`RawBuffer` is the base-model surface. `SessionInfo.is_base_model` drives
`genUiMode.effectiveRawMode()`; the `genUiMode.override` (`auto`/`chat`/`raw`,
persisted per `model_id`, set from `AdvancedSamplingDrawer` or the cycling badge
in the chat header) wins when not `auto`. In raw mode `Chat.svelte` renders
`<RawBuffer />` instead of role bubbles — one continuous editable `pre-wrap`
surface with the loom active path joined as plain text. Flat mode is non-linear:
editing anywhere in the buffer and appending past its end are the same operation.
`resolveDivergence()` diffs the draft against the settled buffer and the tail from
the first changed character becomes one new span — **send** submits it as `user`
and generates `assistant`, a clean buffer shows **generate**, and **append**
submits only the tail; all three go through the same `sendSubmit(..., {raw:true})`
contract as chat. The divergence node and its subtree survive as the original
branch — an edit never overwrites a generated span in place. The internal
`committing` latch holds the buffer→draft sync across the round trip so the typed
tail doesn't flash out, releasing on a content check. Toggling render mode never
mutates the tree. Per-token tinting rides a read-only mirror behind the
transparent-text textarea (a textarea cannot tint spans) and shows only when not
actively editing.

## Pending queue

Submissions during an in-flight gen (or behind earlier queued items) defer rather
than racing the WS. `sendSubmit` checks `isPendingBusy()` (gen active OR a
non-empty queue) and appends a `PendingAction` carrying a `rebuild` factory the
`↑`-pull-and-edit path uses to re-encode the same plan with new text. Rack and
sampling mutations queue through `enqueueOrApply` with `awaitsGen: false`, so the
drain chains through them without waiting on a `done` that never fires.

Queued rack mutations coalesce: each is tagged `coalesceKey: "rack"`, and when the
queue tail already carries that key a fresh mutation chains its `apply` onto the
tail item instead of taking a new slot, with the bubble's label updating to the
latest action. A slider drag firing 30+ intermediate `setSubspaceAlong` calls
mid-gen therefore leaves one bubble carrying the net effect. Coalescing stops at
any non-rack item, so rack changes before and after a queued send form distinct
groups and FIFO ordering relative to the send holds.

The WS `done` / `error` handlers call `drainNextPendingAction()` — one item per
event; a failed apply surfaces as a system turn rather than stalling the queue.
`PendingBubbles.svelte` renders the live queue above the input as dim chips with a
per-bubble `×` (`cancelPendingAction(id)`); the bubble whose slot the user is
editing via `↑` gets the `.editing` treatment off `inputHistory.pulledSlot`.
`StatusFooter` shows an `N queued` readout and no "apply now" — under FIFO there
are no skip-ahead semantics.

`↑` walks the combined ring `[editable pending (most-recent first), input history
(newest first)]`, with an edge-only multi-line policy (recall only when the cursor
is on the draft's first/last line). Pulling a queued item sets
`inputHistory.pulledSlot`, forwarded to `sendSubmit` as `replaceSlot` so a
re-edited send lands at its original slot. `Esc` while pulled cancels the edit
(slot stays, input restores the stash); an empty `Enter` while pulled removes the
slot. Non-editable items (`rebuild === null`) sit in the queue and `↑` walks past
them.

## Per-token highlighting and the drilldown

`lib/highlight.ts` is THE per-token highlight implementation — the chat
transcript, the raw-buffer mirror and the drilldown's context ribbon all render
through `highlightStyleFor` / `highlightStyleString`, so they agree on the score
lookup, the per-probe scale, the hue family and compare-two. `lib/tokens.ts` owns
the pure ramp math and stays store-free. `highlightScoreFor` resolves a target in
order: the surprise sentinel → live per-axis coords (`<probe>[i]` mirrors the gate
grammar) → the collapsed axis-0 `probes` row → the deepest per-layer row → the
cached single-probe score, returning `undefined` so the caller renders
transparent.

Highlighting is driven by one probe dropdown in the chat header with an optional
compare-two mode, and it tints **live** as tokens stream: the `token` frame's
`scores` view feeds the same ramp the settled pass uses. `scoreToRgb` emits
**constant-hue alpha ramps** — tint strength is opacity, hue is meaning — with
`signed` green/red probe poles, blue surprise / J-lens and gold SAE. Pinned
SAE/J-LENS cards expose the same explicit `highlight` action as geometry probes;
both read their native `[0,1]` strength on a unit saturation scale. Authored spans
gain token rows when a later generation's prefill consumes them: the engine emits
a `capture_authored` tree mutation carrying their measurements, so user-written
and model-written text share hover, highlight and drawer behavior. An
authored-only append stays plain until something actually forward-passes it.

Hover and drawer history read the loom-owned `token.measurements` envelopes
directly, so refreshes, source switches and explicit loom save/load preserve the
original generation's measurements with no browser retention cache and no
token-count cap. A channel that was off during that generation can still use its
replay endpoint after the hover dwell; replayed values are never written back or
relabelled as capture.

Clicking any token opens `token_drilldown` regardless of whether a highlight probe
is selected. The drawer is a shell over four tabs — **geometry** (the whitened
Monitor readings: coords, fraction, residual, membership, nearest/assignment
chips, per-layer strip, depth CoM), **logits** (ranked top-K alts with
absolute-probability bars + the logit fork), **sae**, **j-lens** (each: pinned
readings from `instruments.<family>.readings` when captured live, then the native
readout — SAE feature meters in the panel's strength unit, lens aggregate chips +
the all-fitted-layer matrix). Only j-lens and sae carry a pillar hue; the
steered/unsteered A/B branch toggle sits on the tab row when the turn has an
`abPair`. The selected tab is **sticky for the page session**
(`drilldown.svelte.ts`, default `lens`). All three replay families share one
`ReplayReadout` (captured-envelope-preferred, request-sequenced so a stale
response can't clobber a newer view) and one `InstrumentHeader` provenance row —
origin · source · steering chip · `apply recipe steering` toggle — so every
replay-capable tab has the unsteered counterfactual. Captured rows show their
provenance and source even when that instrument is not currently active.

Every evidence-card statline goes through `DetailCardHeader`: a fixed 24px leading
slot puts ranks and family markers on one x-origin, and the primary identifier is
uniformly `--text-sm` with one line-height and baseline across all four tabs.
`EvidenceChips` gives geometry distances / tube facts and SAE
activation / maxActApprox one supporting-fact grammar. Statlines omit the value
already present beside the canonical meter; that meter value carries the pillar
hue instead. Geometry cards use the same responsive two-column desktop grid as the
other card views (one column below 820px). The J-lens layer × vocabulary matrix
has no fixed height and no nested scroller — it expands to full table height, the
drawer body owns the scroll, and the matrix header stays sticky in that outer
flow.

The shell header carries the token's identity chips (turn · role · segment — the
segment chip jumps thinking ⇄ response — vocabulary `id`, `raw` decode index or a
`no replay` marker, and the chosen `p / logp / rank` when captured) over a
**context ribbon** (a windowed, highlight-tinted, clickable strip of the
surrounding segment). Navigation is the conversation-walking cursor
(`token/cursor.ts`): the whole conversation is ONE walkable token sequence — each
turn contributes a thinking and a response segment — so `◀ ▶` / `←`/`→` step
tokens and roll across segment and turn boundaries exactly as the raw buffer
shows them, `▲ ▼` / `↑`/`↓` jump turns, Home/End jump segment bounds, and `↩`
snaps back to the clicked anchor. Keys are ignored inside focusable fields and
`role="slider"` layer strips.

## Persistence

The server loom tree is authoritative and is never first-painted from cache: after
a server restart the cached node ids are invalid, and rendering them fires
requests against nonexistent edges before the authoritative fetch lands.
`localStorage` therefore holds only lightweight presentation preferences —
`saklas.chat.v4.<model_id>`, a `{version: 4, model_id, saved_at, highlight}`
snapshot, accepted only at that exact version and shape and dropped otherwise.
Writes are debounced ~250ms; the bootstrap read also reaps the older tree-bearing
key for the same model, which could occupy most of the origin quota.

Durable conversation persistence is the explicit Save/Load flow: a browser-
downloaded JSON file at **`version: 6`** carrying the complete loom tree
(`tree_format`, `saklas_version`, `root_id`, `active_node_id`, `rev`, `nodes`,
`children_of`, `cast`), the visual steer rack plus `subspaceAlong` (or an
authoritative custom full-grammar expression), the probe rack, highlight and
sampling sections. Every field is required and validated exactly — including a
key-for-key match on the sampling section — and loading an older or partial file
is a visible error. The client never guesses at missing state.

## Reactivity gotcha

Svelte 5's `$state` does NOT track `Map.set` / `Set.add` / inner-object property
writes inside collections. Cross-component collections use `SvelteMap` /
`SvelteSet` from `svelte/reactivity`. Inner-object mutations on map values are
still untracked, so every rack mutator reassigns — `entries.set(name, {...e,
coords})` — for `steerRack` and `probeRack` alike (the `mutateSubspace` /
`mutateManifold` / atom `mutate` wrappers are mode-guarded views over the one
`steerRack.entries`; the shared `subspaceAlong` is a scalar on the slice). The
per-token probe update is the hot path here: a bare `entry.current = val` would
freeze every sparkline at zero for a whole generation.

## Toasts

`lib/stores/toasts.svelte.ts` toasts carry `kind`, `message`, an optional dim
`detail` sub-line, and `ttlMs: number | null` — `null` is sticky and the caller
owns dismissal. `updateToast(id, patch)` mutates a live toast in place so
long-running async work drives one chip from kickoff to completion.
`Toaster.svelte` only schedules a dismissal timer the first time it sees a
non-null TTL, so flipping sticky → ttl mid-flight is a no-op; a caller wanting a
finite TTL at the end should `dismissToast` + `pushToast`.
`ManifoldBuilderDrawer`'s generate / fit legs are the canonical users. When a
message is rendered from a failed request, use `describeError(e)` from
`lib/api.ts` — it lives next to `ApiError` because it is the only thing that
knows the error body shape, and it always prefixes `<status>: ` since
400/404/409/503 carry distinct meaning across this API. A surface that only ever
sees local failures can keep the plain `e instanceof Error ? e.message :
String(e)` fallback.

## Adding a panel or a drawer

**Panel:** write the `.svelte`, wire state into the smallest matching
`lib/stores/` slice (add a slice + a barrel line rather than growing an unrelated
one), mount it from `App.svelte`, `npm run build`, commit the regenerated `dist/`.

**Drawer:** write it under `drawers/`, add the name to the `DrawerName` union in
`lib/types.ts` (a client-local type, so it stays hand-written — the generated
half is REST responses only), then add its row to `DRAWERS` in
`drawers/index.ts` — the compiler demands it. The row carries `narrow: true`
for forms/pickers and either a `launcher` (which lands it in the palette
automatically) or `launcher: null` with a `via` string naming the surface that
opens it. Nothing else needs touching: there is no switch branch, no barrel
export, and no separate palette list.

## Deliberately absent

- **No client-side expression parser.** `lib/expression.ts` serializes only; the
  server's grammar is the single parser.
- **No composer commit/prefill wire fields.** `submit` is the one authored-turn
  door and `WSGenerateRequest` has no `commit_*` / `prefill_*` to emit.
- **No flat per-token aliases.** `measurements` is the only read-side record on
  the `token` and `done` frames — no top-level `captured` / `scores` /
  `probe_readings` / `lens_readout` / `lens_aggregate` / `sae_readout` beside it.
  (The flat `scores` / `per_layer_scores` views live *inside* the envelope.)
- **No flat instrument keys in session info.** One `instruments` block list.
- **No `~`/`|` projection or `!` ablation authoring in the rack.** The grammar
  forbids composing them with `%`, so the cards don't offer them.
- **No pack-backed catalog plumbing in the drawers** — they read the live manifold
  catalog. There is no `/saklas/v1/packs*`, `/manifold-probes`, `/extract/preview`
  or `/profiles/clone` surface, and no traits SSE stream (the dashboard reads
  every per-token channel off the WS).
- **No true multi-session switching.** The server's URL paths support it; the
  client assumes `default`. `SessionAdminDrawer` inspects the collection and sets
  an in-memory bearer key — it is not a session router.
- **No persistent credential storage.** The bearer key lives in memory for the
  page session only.
- **No touch-first layout.** This is a desktop research cockpit designed at a
  1280px floor; below that the columns become full-width views, but the dense
  controls are not resized for touch.
- **No pagination on HF pack search** (server-capped at 20 rows).
