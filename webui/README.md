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

- no launcher chrome — the ⌘K command palette is the single home for the ~19 tool drawers (a ⌘K chip in the chat header is the visible hint; the former far-left workspace rail is gone)
- optional loom sidebar for tree navigation and filtering
- center branch canvas plus chat/token surface — the chat panel header carries a ⋮ menu for the live conversation actions (clear / rewind / transcript / auto-regen) and the input row carries send / stop / regen
- right inspector — one flat panel split into two edge-to-edge sections, the steering rack above the probe rack, divided by a single hairline
- a thin topbar holding only the brand, session status, and the pending-actions badge
- drawer overlays for tools that need width or dense tables — sized narrow (~480px) for forms and pickers, wide (~980px) for analysis views

The visual language is the **Observatory+ system** (tokens v2, 2026-07-10 — dark only, by standing decision): a deep blue-black ground with a faint ambient field, translucent glass cards lit from above, and a hue ontology where **color answers "which space am I looking at?"** — subspace white (shared with chrome, deliberately), manifold violet, SAE gold (the only saturated warm), lens + surprise blue (the logit-space family), green for live/probe+, red for error/probe−. Roles carry no hue (the cast model). Gradients are material, never data: fills sheen *across* the value axis; along-axis gradients only where the gradient *is* the data (depth strips, time trails). Glow is reserved for what is alive right now. The surface is **borderless by default** (Phase 6): separation comes from space, then fill contrast on the surface ladder, then light (top-light, shadow, `--shadow-sticky` under sticky headers) — hairlines survive only as meaning (focus rings, state rings, floating-surface edges, dashed pending-ghosts, control glyph strokes), and inputs recess to `--input-well` with a ring on focus only. Type is Recursive in both voices — one self-hosted variable woff2 exposed as "Recursive Sans" (chrome) and "Recursive Mono" (data) with axes pinned per-family in `src/lib/style/fonts.css`. Tokens live in `src/lib/style/tokens.css` (source of truth, rules in its header); v2 primitives in `src/lib/ui/`; the living specimen page is served at **`/styleguide`**. Keep the interface information-rich, but keep controls visible and directly operable with the mouse.

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
    BranchCanvas.svelte          # active path + sibling/child lanes + fan/compare actions
    InspectorPanel.svelte        # runtime meters + sampling + steering/probe racks
    StatusFooter.svelte          # ● gen N/M [bar] · t/s · elapsed · ppl
    Chat.svelte                  # thinking-collapsible + live probe-tinted tokens + ⋮ actions menu + auto-regen/pinned split
    SamplingStrip.svelte         # T / P / K / max / seed / thinking + segmented apply-mode
    SteeringRack.svelte          # subspace group (+ shared "subspace along" master) + manifold group of steer cards + "+ subspace/manifold steer"
    ProbeRack.svelte             # subspace + manifold groups of probe cards + sort + "+ subspace/manifold probe"
    rack/RackCard.svelte         # shared card chrome (statline on top, controls/meters below)
    rack/SteerCard.svelte        # unified steer card — branches on entry.mode (snap-to-node + XYPad both); subspace = position only (shared along master) | manifold = position + per-card along / onto
    rack/ProbeCard.svelte        # unified probe card — bipolar bar | scalar/fraction + nearest; curved adds coords / residual / mini-map
    loom/                        # LoomSidebar + LoomNode + LoomEdge
  drawers/
    Load / Compare / SystemPrompt / Help / Export
    SaveConversation / LoadConversation
    Vectors / Extract       # unified picker + manage drawer, plus custom-vector form
    Pack / Merge / Clone
    ExperimentLab / ActivationAtlas / RecipeBuilder / AdvancedSampling
    Health / SessionAdmin / TokenDrilldown / Correlation / LayerNorms
    NodeCompare / Transcript
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
4. Decide how users reach it: add it to the matching `RAIL_CATEGORIES` group in `lib/commands.ts` (the ⌘K palette), or an inline button in the panel that owns the workflow.

## Reactivity gotcha

Svelte 5 `$state` doesn't track plain `Map.set` / `Set.add` or inner-object property writes inside collections. The store uses `SvelteMap` / `SvelteSet` from `svelte/reactivity` for cross-component collections; rack mutators reassign via `entries.set(name, {...e, alpha})` instead of `e.alpha = alpha` so subscribers see the change.

## Wire protocol

The dashboard speaks the existing `/saklas/v1/*` native API:

* `GET /saklas/v1/sessions/default` — `SessionInfo`
* `GET /saklas/v1/sessions/default/correlation[?names=a,b]` — N×N cosine over active analytics profiles
* `GET/POST/DELETE /saklas/v1/sessions/default/probes[/{name}]` — list / activate / deactivate probes
* `POST /saklas/v1/sessions/default/extract` — JSON or SSE-progress when `Accept: text/event-stream`
* `GET /saklas/v1/manifolds`, `POST /saklas/v1/manifolds`, `PATCH/DELETE /saklas/v1/manifolds/{ns}/{name}` — author, edit, and remove manifold folders
* `POST /saklas/v1/manifolds/generate`, `POST /saklas/v1/manifolds/{ns}/{name}/fit`, `POST /saklas/v1/manifolds/merge` — generate discover corpora, fit tensors, and merge discover sources
* `GET /saklas/v1/packs[/search]`, `POST /saklas/v1/packs`, `DELETE /saklas/v1/packs/{ns}/{name}` — pack browse, install, and remove. The list response carries a session-relative `has_tensor` flag so the rack can split rows by fitted availability; DELETE detaches the manifold before removing the folder.
* `GET /saklas/v1/sessions/default/tree` and `/tree/active` — full loom tree or active path
* `POST /saklas/v1/sessions/default/tree/{navigate,edit,branch,star,note,reset}` and `DELETE /tree/{node_id}` — loom mutations
* `GET /saklas/v1/sessions/default/tree/{edge_label,filter}` — branch labels and search/filter support
* `POST /saklas/v1/sessions/default/tree/{diff,joint_logprobs,transcript,transcript/load}` — compare branches and import/export transcripts
* `WS /saklas/v1/sessions/default/stream` — token + probe co-stream; the `token` event carries optional `scores` (magnitude-weighted aggregate, drives the live inline highlight) + `per_layer_scores` (per-layer heatmap for the drilldown), and the `done` event carries `per_token_probes`
* `GET /saklas/v1/sessions/default/traits/stream` — live per-token probe SSE

See `saklas/server/native_routes.py` for the Python route registrar.
