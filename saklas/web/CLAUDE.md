# web/

Static dashboard mounted at `/` by default when `saklas serve` runs. Opt out with `--no-web`. Library callers using `create_app(..., web=False)` directly default-off — only the CLI presumes the casual-user UX.

## Layout

```
saklas/web/
  __init__.py        # public re-exports: register_web_routes, dist_path
  routes.py          # the actual mount logic + SPA fallback
  dist/              # COMMITTED build artifact, ships in the wheel
    index.html
    assets/index.css
    assets/saklas.js
```

The Svelte 5 + Vite source lives at the repo's `webui/` directory (peer of `saklas/`). `cd webui && npm run build` emits to `saklas/web/dist/` directly — no intermediate copy step. CI gating on source-vs-dist drift is wired but disabled by default in `.github/workflows/ci.yml` because the committed bundle is the source of truth.

## Mount

`register_web_routes(app)` mounts `/assets/*` on `StaticFiles` for the bundled CSS/JS, registers `GET /` to return `index.html`, and a catch-all `GET /{full_path:path}` that serves real files when present and falls back to `index.html` for SPA-owned routes. `server.create_app` registers the dashboard last so the catch-all doesn't shadow `/v1/*`, `/api/*`, `/saklas/v1/*`. CLI default is `web=True`; the library default is `web=False` so embedded API surfaces don't pick up the dashboard accidentally.

`dist_path()` resolves the bundled assets through `importlib.resources` so it works for both editable and wheel installs. `WebUINotBuilt` is raised on mount when the dist directory is empty — only fires in source-tree installs that haven't run `npm run build` yet.

## What the dashboard is

v2.0 reframed the dashboard from "chat with diagnostics on the side" to a mouse-first interpretability cockpit. Three concerns must be answerable at any moment without leaving the main view: what am I telling the model (input + system prompt + sampling), how am I steering it (vector rack), what is it doing internally (probe rack + per-token tinting + correlation + layer norms + ppl).

Per-token highlighting now lives on the chat tokens themselves, driven by a single highlight-probe dropdown with an optional two-stripe compare-two mode. The v1.6 always-on per-token × per-layer × per-probe heatmap is gone; the deep drilldown is opt-in via clicking a single token (`token_drilldown` drawer).

## Wire protocol

The dashboard speaks the existing `/saklas/v1/*` native API plus six routes added in v2.0:

1. **WS `/saklas/v1/sessions/{id}/stream`** — token + probe co-stream. The `token` event carries optional `per_layer_scores` (`dict[str, dict[str, float]]`, string-keyed) when probes are loaded; `done` carries `per_token_probes` assembled from `session._last_per_token_scores`. Drives chat tinting + click-token drilldown.
2. **GET `/saklas/v1/sessions/{id}/correlation[?names=…]`** — N×N magnitude-weighted cosine matrix. Drives the correlation collapsible.
3. **GET `/saklas/v1/sessions/{id}/vectors/{name}/diagnostics`** — 16-bucket layer-magnitude histogram + summary metrics. Falls back to `session._monitor.profiles` when the name is a probe rather than a registered steering vector. Drives the WHY histogram inside expanded probe strips.
4. **GET `/saklas/v1/packs`** — locally installed packs, JSON shape. Drives the vector picker and probe picker.
5. **GET `/saklas/v1/packs/search?q=…`** — HF Hub proxy. Drives the Pack drawer's search tab.
6. **POST `/saklas/v1/packs`** — install pack from HF coord or local folder.
7. **POST `/saklas/v1/sessions/{id}/vectors/merge`** — register a merged-expression vector.
8. **POST `/saklas/v1/sessions/{id}/vectors/clone`** — corpus-based clone, SSE progress branch on `Accept: text/event-stream`.

POST `/sweep` (the alpha-grid SSE) and the `traits/stream` SSE existed before v2.0 and remain.

## Source layout

```
webui/src/
  main.ts                     # bootstrap: mounts <App /> via Svelte 5 mount()
  App.svelte                  # shell — topbar / two-column main / status footer / drawer host
  lib/
    api.ts                    # typed REST + WS + SSE clients
    stores.svelte.ts          # Svelte 5 runes-based shared state (SvelteMap-backed)
    types.ts                  # every shared interface
    expression.ts             # parse/serialize the steering grammar
    tokens.ts                 # HIGHLIGHT_SAT + scoreToRgb + twoStripeStyle
    charts.ts                 # bucketize() port of saklas.core.histogram
    charts/{Bar,Sparkline,Histogram,HeatmapCell}.svelte
    style/{tokens.css,global.css}
  panels/
    Topbar.svelte             # model + device + clear/rewind/regen + tools menu + stop
    StatusFooter.svelte       # ● gen N/M [bar] · t/s · elapsed · ppl
    Chat.svelte               # thinking-collapsible, probe-tinted tokens, A/B split
    SamplingStrip.svelte      # T / P / K / max / seed / thinking / session-vs-one-shot
    SteeringRack.svelte       # one strip per loaded vector + canonical EXPR + "+ steer"
    VectorStrip.svelte        # enable / α slider / trigger pill / variant / projection / ✕
    ProbeRack.svelte          # highlight + compare-two dropdowns + sort + "+ probe"
    ProbeStrip.svelte         # radio + sparkline + value bar + WHY histogram on expand
    ReferenceCollapsibles.svelte # ▶ correlation N×N + ▶ layer norms (per vector)
  drawers/
    {Extract,Load,SaveConversation,LoadConversation,Compare,
     SystemPrompt,ModelInfo,Help,Export,Sweep,Pack,Merge,Clone,
     VectorPicker,ProbePicker,TokenDrilldown}Drawer.svelte
    _SearchableConceptList.svelte  # shared between picker drawers
    index.ts                  # barrel re-exports for App.svelte's switch
```

Adding a panel: write the .svelte file, wire any new state into `stores.svelte.ts`, mount from App.svelte's grid, `npm run build`, commit the regenerated `saklas/web/dist/`. Adding a drawer: write the .svelte file under `drawers/`, add the name to the `DrawerName` union in `lib/types.ts`, add a branch to App.svelte's drawer switch, re-export from `drawers/index.ts` if it should ship in the topbar tools menu.

## Reactivity gotcha

Svelte 5's `$state` does NOT track plain `Map.set` / `Set.add` / inner-object property writes inside collections. Cross-component state collections in `stores.svelte.ts` use `SvelteMap` / `SvelteSet` from `svelte/reactivity`. Inner-object mutations (e.g. `e.alpha = 0.5`) on map values are still untracked, so every rack mutator reassigns via `entries.set(name, {...e, alpha})` — applies to vectorRack and probeRack alike.

`updateProbeFromScores` (driven by every WS `token` event) is the hot path that depends on this — bare `entry.current = val` would freeze probe sparklines at zero throughout a generation.

## Persistence

`chatLog.turns` and `highlightState` (target / compareTarget / compareTwo) persist to `localStorage` under `saklas.chat.v1.<model_id>`. Saves debounced via `$effect.root` ~250 ms after mutations so token streams don't beat the disk. Restored after `refreshSession()` in `bootstrap` so the storage key resolves.

Server-restart guard: if the persisted snapshot has user turns but the fresh session reports `history_length === 0`, drop the snapshot. Replaying would lie about generation context (next gen would see empty server-side history).

`pendingIndex` is force-cleared on restore so an in-flight turn from a killed tab doesn't ghost the UI.

## Steering / probe pickers

The picker drawers (`VectorPickerDrawer`, `ProbePickerDrawer`) mirror the TUI's `/steer 0.5 honest` ergonomics: the user picks a concept name from `GET /packs`, the server's extract endpoint short-circuits to the cached profile when the model already has it, the rack lands. `addVectorToRack` defaults α to 0.5 (matches `DEFAULT_COEFF` in `saklas.core.steering_expr`). The picker's footer keeps the advanced affordances — extract from pos/neg, load from disk path.

## Out of scope (v2.0)

- Multi-session UI (server URL-paths support it; the client assumes `default`).
- Auth UI for `SAKLAS_API_KEY` — the underlying Bearer middleware applies; no dedicated UI surface.
- Mobile / touch-first responsive layout. Saklas is a desktop research tool; min-width 1280px.
- Combobox autocomplete on the projection-target picker (currently `window.prompt`).
- Pagination on HF pack search (capped at 20 results).
