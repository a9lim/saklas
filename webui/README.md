# saklas web UI

Svelte 4 + Vite + TypeScript source tree for the analytics dashboard.
The build emits to `../saklas/web/dist/`, which the Python package
`saklas.web` mounts at `/` when `saklas serve --web` is passed.

## Development

```bash
cd webui
npm install
# In another terminal: saklas serve <model>  (so the proxy has a target)
npm run dev
```

`npm run dev` starts Vite on port 5173 with the dev server proxying
`/saklas`, `/v1`, and `/api` (including the WS upgrade) to
`localhost:8000`. Edit `src/` files; HMR reloads the browser.

## Production build

```bash
npm run build
```

Wipes `../saklas/web/dist/` and writes the compiled bundle there.
The committed bundle ships in the wheel — CI verifies the source tree
hasn't drifted from the on-disk artifact.

## Layout

```
src/
  main.ts                # bootstrap: mounts <App /> on #app
  App.svelte             # 4-panel dashboard layout + status bar
  lib/
    api.ts               # REST + WS clients for /saklas/v1/*
    stores.ts            # writable stores backing the panels
  panels/
    Chat.svelte          # left column: streamed chat with WS-driven inspector
    Inspector.svelte     # center: per-token × per-layer × per-probe heatmap
    Correlation.svelte   # right top: N×N magnitude-weighted cosine
    LayerNorms.svelte    # right bottom: full-resolution per-layer ||baked||
```

## Adding a panel

1. New `src/panels/Foo.svelte`.
2. Wire any state into `lib/stores.ts` (writable store + a `refreshFoo`
   thunk that fetches from `/saklas/v1/...`).
3. Mount it from `App.svelte`'s grid, adjusting `grid-template-rows`
   if the right column needs another tile.
4. `npm run build` — the committed bundle in `../saklas/web/dist/`
   updates with the new entrypoint.

## Wire protocol

The dashboard speaks the existing native API; no protocol versioning
matters as long as both sides agree on these shapes:

* `GET /saklas/v1/sessions/default` → `SessionInfo`
* `GET /saklas/v1/sessions/default/vectors` → list with `per_layer_norms`
* `GET /saklas/v1/sessions/default/correlation[?names=a,b]` → matrix
* `WS /saklas/v1/sessions/default/stream` — `started` / `token` / `done`
  events; the `token` event carries optional `per_layer_scores` when
  the session has probes loaded.

See `saklas/server/saklas_api.py` for the Python side.
