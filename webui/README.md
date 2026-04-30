# saklas web UI

Svelte 5 + Vite source tree for the v2.0 interpretability cockpit. The build emits to `../saklas/web/dist/`, which the Python package `saklas.web` mounts at `/` on every `saklas serve` (pass `--no-web` to skip).

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

## Layout

```
src/
  main.ts                # bootstrap: mounts <App /> via Svelte 5's mount()
  App.svelte             # shell — topbar / two-column main / status footer / drawer host
  lib/
    api.ts               # typed REST + WS + SSE clients for /saklas/v1/*
    stores.svelte.ts     # Svelte 5 runes-based shared state (SvelteMap-backed)
    types.ts             # every shared interface (DrawerName, ChatTurn, …)
    expression.ts        # parse/serialize the steering-expression grammar
    tokens.ts            # per-token highlight RGB mapping (mirrors TUI)
    charts.ts            # bucketize() port of saklas.core.histogram
    charts/              # SVG primitives — Bar, Sparkline, Histogram, HeatmapCell
    style/               # tokens.css (design tokens) + global.css (resets)
  panels/
    Topbar.svelte                # model · device · clear/rewind/regen · tools · stop
    StatusFooter.svelte          # ● gen N/M [bar] · t/s · elapsed · ppl
    Chat.svelte                  # thinking-collapsible + probe-tinted tokens + A/B
    SamplingStrip.svelte         # T / P / K / max / seed / thinking
    SteeringRack.svelte          # vector strips + canonical EXPR + "+ steer"
    VectorStrip.svelte           # enable / α slider / trigger / variant / ⋮ menu / ✕
    ProbeRack.svelte             # highlight + compare-two + sort + "+ probe"
    ProbeStrip.svelte            # radio + sparkline + value bar + WHY histogram
    ReferenceCollapsibles.svelte # ▶ correlation N×N · ▶ layer norms per vector
  drawers/
    Extract / Load / Compare / SystemPrompt / ModelInfo / Help / Export
    SaveConversation / LoadConversation
    VectorPicker / ProbePicker
    Sweep / Pack / Merge / Clone
    TokenDrilldown
    _SearchableConceptList.svelte
    index.ts             # barrel re-exports for App.svelte's drawer switch
```

## Adding a panel

1. New `src/panels/Foo.svelte`.
2. Wire any new state into `lib/stores.svelte.ts` — Svelte 5 runes (`$state`), exported as a slice. Use `SvelteMap` / `SvelteSet` (not plain `Map` / `Set`) for collections.
3. Mount it from `App.svelte`.
4. `npm run build`, then commit the regenerated `../saklas/web/dist/` so the wheel picks up the new entrypoint.

## Adding a drawer

1. New `src/drawers/FooDrawer.svelte`. Take `params: unknown` via `$props()` — the host forwards `drawerState.params`.
2. Add the name to the `DrawerName` union in `lib/types.ts`.
3. Add a branch to `App.svelte`'s drawer switch and (optionally) re-export from `drawers/index.ts` to ship through the topbar tools menu.

## Reactivity gotcha

Svelte 5 `$state` doesn't track plain `Map.set` / `Set.add` or inner-object property writes inside collections. The store uses `SvelteMap` / `SvelteSet` from `svelte/reactivity` for cross-component collections; rack mutators reassign via `entries.set(name, {...e, alpha})` instead of `e.alpha = alpha` so subscribers see the change.

## Wire protocol

The dashboard speaks the existing `/saklas/v1/*` native API:

* `GET /saklas/v1/sessions/default` — `SessionInfo`
* `GET/POST/DELETE /saklas/v1/sessions/default/vectors[/{name}]` — list / load-from-disk / drop, with per-layer `||baked||` on the GET
* `GET /saklas/v1/sessions/default/vectors/{name}/diagnostics` — 16-bucket WHY histogram (falls back to monitor profiles for probes)
* `GET /saklas/v1/sessions/default/correlation[?names=a,b]` — N×N cosine
* `GET/POST/DELETE /saklas/v1/sessions/default/probes[/{name}]` — list / activate / deactivate
* `POST /saklas/v1/sessions/default/extract` — JSON or SSE-progress when `Accept: text/event-stream`
* `POST /saklas/v1/sessions/default/sweep` — alpha-grid SSE
* `POST /saklas/v1/sessions/default/vectors/{merge,clone}` — register a derived vector
* `GET /saklas/v1/packs[/search]`, `POST /saklas/v1/packs` — pack browse + install
* `WS /saklas/v1/sessions/default/stream` — token + probe co-stream; the `token` event carries optional `per_layer_scores` and the `done` event carries `per_token_probes`
* `GET /saklas/v1/sessions/default/traits/stream` — live per-token probe SSE

See `saklas/server/saklas_api.py` for the Python side.
