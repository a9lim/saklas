# saklas web UI

Svelte 5 and Vite source tree for the analytics dashboard. The build emits to `../saklas/web/dist/`, which the Python package `saklas.web` mounts at `/` on every `saklas serve` (pass `--no-web` to skip).

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

Wipes `../saklas/web/dist/` and writes the compiled bundle there. The committed bundle ships in the wheel, so please commit the regenerated bundle alongside any source changes you make. The CI job that would diff the source tree against the committed bundle is stubbed in `.github/workflows/ci.yml` and disabled by default; please re-enable it when the source stabilizes if you want CI to enforce the match.

## Layout

```
src/
  main.ts                # bootstrap: mounts <App /> on #app via Svelte 5's mount()
  App.svelte             # 4-panel dashboard layout and status bar
  lib/
    api.ts               # REST and WS clients for /saklas/v1/*
    stores.ts            # writable stores backing the panels
  panels/
    Chat.svelte          # left column: streamed chat with WS-driven inspector
    Inspector.svelte     # center: per-token by per-layer by per-probe heatmap
    Correlation.svelte   # right top: NxN magnitude-weighted cosine
    LayerNorms.svelte    # right bottom: full-resolution per-layer ||baked||
```

## Adding a panel

1. New `src/panels/Foo.svelte`.
2. Wire any state into `lib/stores.ts` (writable store plus a `refreshFoo` thunk that fetches from `/saklas/v1/...`).
3. Mount it from `App.svelte`'s grid, adjusting `grid-template-rows` if the right column needs another tile.
4. `npm run build`, then commit the regenerated `../saklas/web/dist/` so the wheel picks up the new entrypoint.

## Wire protocol

The dashboard speaks the existing native API. No protocol versioning matters as long as both sides agree on these shapes:

* `GET /saklas/v1/sessions/default` returns `SessionInfo`
* `GET /saklas/v1/sessions/default/vectors` returns a list with `per_layer_norms`
* `GET /saklas/v1/sessions/default/correlation[?names=a,b]` returns the matrix
* `WS /saklas/v1/sessions/default/stream` carries `started`, `token`, and `done` events; the `token` event carries optional `per_layer_scores` when the session has probes loaded.

See `saklas/server/saklas_api.py` for the Python side.
