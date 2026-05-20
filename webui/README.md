# saklas web UI

Svelte 5 + Vite source tree for the saklas interpretability cockpit. The build emits to `../saklas/web/dist/`, which the Python package `saklas.web` mounts at `/` on every `saklas serve` (pass `--no-web` to skip).

## Development

```bash
cd webui
npm install
# In another terminal: saklas serve <model>  (so the proxy has a target)
npm run dev
```

`npm run dev` starts Vite on port 5173 with the dev server proxying `/saklas`, `/v1`, and `/api` (including the WS upgrade) to `localhost:8000`. Edit `src/` files and HMR reloads the browser.

## Production build

```bash
npm run build
```

Wipes `../saklas/web/dist/` and writes the compiled bundle there. The committed bundle ships in the wheel, so please commit the regenerated bundle alongside any source changes you make. The CI job that would diff the source tree against the committed bundle is stubbed in `.github/workflows/ci.yml` and disabled by default.

## Product shape

This is a desktop research cockpit, not a landing page and not a thin chat wrapper. The first viewport should answer five questions without mode-switching: which branch am I on, what prompt/sampling state will run next, what steering recipe is active, what probes are reading, and where do I open deeper analysis. The fixed shell is:

- far-left workspace rail — a loom-sidebar toggle plus three category icons (vectors / analysis / session) that open fly-out tool lists; this is the single home for the ~19 tool drawers
- optional loom sidebar for tree navigation and filtering
- center branch canvas plus chat/token surface — the chat panel header carries a ⋮ menu for the live conversation actions (clear / rewind / transcript / auto-regen) and the input row carries send / stop / regen
- right inspector — one flat panel split into two edge-to-edge sections, the steering rack above the probe rack, divided by a single hairline
- a thin topbar holding only the brand, session status, and the pending-actions badge
- drawer overlays for tools that need width or dense tables — sized narrow (~480px) for forms and pickers, wide (~980px) for analysis views

The visual language is one flat dark theme, dark mode only. Surfaces separate by background shade rather than outlines; the only border is a single faint hairline, and corners stay a sharp 2px throughout. A blue accent is the one chrome hue — selection, focus, and primary actions — while green, amber, red, yellow, and purple are reserved for state and data, never decoration. Type comes from the `--text-*` scale, spacing from `--space-*`, weight from `--weight-*`; all of it lives in `src/lib/style/tokens.css`, which is the source of truth and documents the rules in its header. Colour values and motion curves still mirror a9l.im's dark palette (`shared-tokens.js`), but the webui drops the site's magma-red accent and its film-grain texture for a calmer, fully flat surface. Keep the interface information-rich, but keep controls visible and directly operable with the mouse.

## Layout

```
src/
  main.ts                # bootstrap: mounts <App /> via Svelte 5's mount()
  App.svelte             # shell: topbar / rail / loom / chat-canvas / inspector / footer / drawers
  lib/
    api.ts               # typed REST + WS + SSE clients for /saklas/v1/*
    stores.svelte.ts     # shared state barrel + WS/tree coordination
    stores/              # split slices: drawers, inputHistory, toasts
    types.ts             # every shared interface (DrawerName, ChatTurn, …)
    expression.ts        # parse/serialize the steering-expression grammar
    concepts.ts          # concept-catalog helpers — category + bipolar poles
    tokens.ts            # per-token highlight RGB mapping (mirrors TUI)
    charts.ts            # bucketize() port of saklas.core.histogram
    charts/              # SVG primitives — Bar, Sparkline, Histogram, HeatmapCell
    Segmented.svelte     # shared segmented control (animated indicator)
    Slider.svelte        # shared range slider — one thumb/track everywhere
    style/               # tokens.css (design tokens) + global.css (resets)
  panels/
    Topbar.svelte                # thin strip: brand · session status · pending-actions badge
    WorkspaceRail.svelte         # left rail: loom toggle + vectors/analysis/session fly-outs
    BranchCanvas.svelte          # active path + sibling/child lanes + fan/compare actions
    InspectorPanel.svelte        # runtime meters + sampling + steering/probe racks
    StatusFooter.svelte          # ● gen N/M [bar] · t/s · elapsed · ppl
    Chat.svelte                  # thinking-collapsible + live probe-tinted tokens + ⋮ actions menu + auto-regen/pinned split
    SamplingStrip.svelte         # T / P / K / max / seed / thinking + segmented apply-mode
    SteeringRack.svelte          # vector strips + "+ add steering" + canonical expression
    VectorStrip.svelte           # ●/○ enable + pole-framed α slider + α display + trigger / variant menu / ⋮ menu / ✕ + inline projection modal
    ProbeRack.svelte             # sort + "+ add probe"
    ProbeStrip.svelte            # ●/○ select (whole-row click) + name + right-aligned sparkline + bar + value + ✕ + per-layer cells
    loom/                        # LoomSidebar + LoomNode + LoomEdge
  drawers/
    Load / Compare / SystemPrompt / ModelInfo / Help / Export
    SaveConversation / LoadConversation
    VectorPicker / ProbePicker
    Pack / Merge / Clone
    ExperimentLab / ActivationAtlas / RecipeBuilder / AdvancedSampling
    Health / SessionAdmin / TokenDrilldown / Correlation / LayerNorms
    NodeCompare / Transcript
    _SearchableConceptList.svelte  # shared categorized catalog for both pickers
    index.ts             # barrel re-exports for App.svelte's drawer switch
```

## Adding a panel

1. New `src/panels/Foo.svelte`.
2. Wire any new state into the smallest matching file under `lib/stores/`, or into `lib/stores.svelte.ts` only when it genuinely crosses WS/tree/chat boundaries. Use Svelte 5 runes (`$state`) and `SvelteMap` / `SvelteSet` for collections.
3. Mount it from `App.svelte`.
4. `npm run build`, then commit the regenerated `../saklas/web/dist/` so the wheel picks up the new entrypoint.

## Adding a drawer

1. New `src/drawers/FooDrawer.svelte`. Take `params: unknown` via `$props()` — the host forwards `drawerState.params`.
2. Add the name to the `DrawerName` union in `lib/types.ts`; add it to `NARROW_DRAWERS` in `App.svelte` if it is a form/picker rather than an analysis view.
3. Add a branch to `App.svelte`'s drawer switch and re-export from `drawers/index.ts` when the host references it through the barrel.
4. Decide how users reach it: add it to the matching category in `WorkspaceRail.svelte`'s fly-out lists, or an inline button in the panel that owns the workflow.

## Reactivity gotcha

Svelte 5 `$state` doesn't track plain `Map.set` / `Set.add` or inner-object property writes inside collections. The store uses `SvelteMap` / `SvelteSet` from `svelte/reactivity` for cross-component collections; rack mutators reassign via `entries.set(name, {...e, alpha})` instead of `e.alpha = alpha` so subscribers see the change.

## Wire protocol

The dashboard speaks the existing `/saklas/v1/*` native API:

* `GET /saklas/v1/sessions/default` — `SessionInfo`
* `GET/POST/DELETE /saklas/v1/sessions/default/vectors[/{name}]` — list / load-from-disk / drop, with per-layer `||baked||` on the GET
* `GET /saklas/v1/sessions/default/vectors/{name}/diagnostics` — 16-bucket layer-magnitude histogram + summary metrics (powers `saklas vector why`; falls back to monitor profiles for probes)
* `GET /saklas/v1/sessions/default/correlation[?names=a,b]` — N×N cosine
* `GET/POST/DELETE /saklas/v1/sessions/default/probes[/{name}]` — list / activate / deactivate
* `POST /saklas/v1/sessions/default/extract` — JSON or SSE-progress when `Accept: text/event-stream`
* `POST /saklas/v1/sessions/default/experiments/fan` — alpha grid as loom siblings
* `POST /saklas/v1/sessions/default/vectors/{merge,clone}` — register a derived vector
* `GET /saklas/v1/packs[/search]`, `POST /saklas/v1/packs` — pack browse + install
* `GET /saklas/v1/sessions/default/tree` and `/tree/active` — full loom tree or active path
* `POST /saklas/v1/sessions/default/tree/{navigate,edit,branch,star,note,reset}` and `DELETE /tree/{node_id}` — loom mutations
* `GET /saklas/v1/sessions/default/tree/{edge_label,filter}` — branch labels and search/filter support
* `POST /saklas/v1/sessions/default/tree/{diff,joint_logprobs,transcript,transcript/load}` — compare branches and import/export transcripts
* `WS /saklas/v1/sessions/default/stream` — token + probe co-stream; the `token` event carries optional `scores` (magnitude-weighted aggregate, drives the live inline highlight) + `per_layer_scores` (per-layer heatmap for the drilldown), and the `done` event carries `per_token_probes`
* `GET /saklas/v1/sessions/default/traits/stream` — live per-token probe SSE

See `saklas/server/saklas_api.py` for the Python side.
