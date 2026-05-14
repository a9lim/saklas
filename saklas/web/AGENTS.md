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
2. **GET `/saklas/v1/sessions/{id}/correlation[?names=…]`** — N×N magnitude-weighted cosine matrix. Default pool unions registered steering vectors AND active probes (deduplicated by name). Drives the correlation drawer overlay.
3. **GET `/saklas/v1/sessions/{id}/vectors/{name}/diagnostics`** — 16-bucket layer-magnitude histogram + per-layer magnitudes + (optional) probe-quality metrics. Falls back to `session._monitor.profiles` when the name is a probe rather than a registered steering vector. Drives the layer-norms drawer overlay (and `saklas vector why` from the CLI).
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
    Chat.svelte               # thinking-collapsible, probe-tinted tokens, auto-regen / pin split
    SamplingStrip.svelte      # T / P / K / max / seed / thinking / session-vs-one-shot
    SteeringRack.svelte       # one strip per loaded vector + canonical EXPR + "+ steer"
    VectorStrip.svelte        # ●/○ enable + α slider + α display + trigger pill + variant chip + ⋮ menu + ✕ + inline projection modal
    ProbeRack.svelte          # highlight + compare-two dropdowns + sort + "+ probe"
    ProbeStrip.svelte         # ●/○ select-for-highlight (whole-row click target) + name + right-aligned sparkline + value bar + α display + ✕ + always-visible per-layer reading strip
  drawers/
    {Extract,Load,SaveConversation,LoadConversation,Compare,
     SystemPrompt,ModelInfo,Help,Export,Sweep,Pack,Merge,Clone,
     VectorPicker,ProbePicker,TokenDrilldown,
     Correlation,LayerNorms}Drawer.svelte
    _SearchableConceptList.svelte  # shared between picker drawers
    index.ts                  # barrel re-exports for App.svelte's switch
```

Adding a panel: write the .svelte file, wire any new state into `stores.svelte.ts`, mount from App.svelte's grid, `npm run build`, commit the regenerated `saklas/web/dist/`. Adding a drawer: write the .svelte file under `drawers/`, add the name to the `DrawerName` union in `lib/types.ts`, add a branch to App.svelte's drawer switch, re-export from `drawers/index.ts` if it should ship in the topbar tools menu.

## Reactivity gotcha

Svelte 5's `$state` does NOT track plain `Map.set` / `Set.add` / inner-object property writes inside collections. Cross-component state collections in `stores.svelte.ts` use `SvelteMap` / `SvelteSet` from `svelte/reactivity`. Inner-object mutations (e.g. `e.alpha = 0.5`) on map values are still untracked, so every rack mutator reassigns via `entries.set(name, {...e, alpha})` — applies to vectorRack and probeRack alike.

`updateProbeFromScores` (driven by every WS `token` event) is the hot path that depends on this — bare `entry.current = val` would freeze probe sparklines at zero throughout a generation.

## Persistence

`chatLog.turns` and `highlightState` (target / compareTarget / compareTwo) persist to `localStorage` under `saklas.chat.v2.<model_id>` (v2 is the loom-tree shape; v1 single-branch logs auto-migrate via `hydrateV1ChatAsLinearTree`). Saves debounced via `$effect.root` ~250 ms after mutations so token streams don't beat the disk. Restored after `refreshSession()` in `bootstrap` so the storage key resolves.

**Size budget toast.** `schedulePersist` measures `JSON.stringify(payload).length` against `_LOCALSTORAGE_SOFT_BUDGET = 5 MB` per Decision 18 in `docs/plans/loom.md`. Above the threshold, `pushToast` fires a once-per-session advisory suggesting transcript export + tree clear. Write isn't hard-stopped; `_sizeWarned` guards against re-firing on every subsequent save. `Toaster.svelte` is the host (bottom-right, TTL-dismissed), mounted alongside `StatusFooter` in `App.svelte`.

Server-restart guard: if the persisted snapshot has user turns but the fresh session reports `history_length === 0`, drop the snapshot. Replaying would lie about generation context (next gen would see empty server-side history).

`pendingIndex` is force-cleared on restore so an in-flight turn from a killed tab doesn't ghost the UI.

## Steering / probe pickers

The picker drawers (`VectorPickerDrawer`, `ProbePickerDrawer`) mirror the TUI's `/steer 0.5 honest` ergonomics: the user picks a concept name from `GET /packs`, the server's extract endpoint short-circuits to the cached profile when the model already has it, the rack lands. `addVectorToRack` defaults α to 0.5 (matches `DEFAULT_COEFF` in `saklas.core.steering_expr`). The picker's footer keeps the advanced affordances — extract from pos/neg, load from disk path.

## Click semantics

Probe rows: clicking anywhere on the row body (not the ✕) toggles highlight selection — first click anchors the probe as the chat-token highlight target, click again on the same row to deselect. ✕ is `stopPropagation`'d so removal doesn't trip toggle on the way out. The row uses `role="button"` + `aria-pressed` so the toggle is keyboard-reachable (Enter/Space).

Chat tokens: every per-token span is clickable regardless of whether a highlight probe is selected — the click opens `token_drilldown` with `{turnIdx, tokenIdx, isThinking}`. The `isThinking` flag routes the drilldown to `turn.thinkingTokens` vs `turn.tokens` so clicks inside the thinking-collapsible body resolve to the right token row. With no highlight target, tokens render bare but the hover outline still gives a click affordance.

Vector strip projection picker: the `⋮` menu's "project onto (~)…" / "project orthogonal (|)…" entries open an inline modal at `--z-modal` (above drawers) — text input autofocuses, Enter confirms, Escape / click-outside / cancel cancel. Replaces the v1 `window.prompt`. Re-clicking the same operator with an existing projection clears it (no dialog).

## Input history (↑/↓ recall)

Shell-style history on the chat textarea. Every line submitted via `doSend` lands in `inputHistory.entries` through `pushInputHistory` (chat messages and slash commands alike); ↑/↓ in `Chat.svelte::onKeydown` call `navigateInputHistory(±1, currentInput)`. Edge-only multi-line policy: ↑ recalls only when the cursor sits on the first line (`shouldRecallUp`); ↓ goes forward only on the last line (`shouldRecallDown`) and is a no-op when no recall is in flight, so multi-line editing inside the draft isn't hijacked. First ↑ stashes the in-progress draft; ↓ past the newest entry restores it. `INPUT_HISTORY_MAX = 200` caps the ring. **In-memory only** — no `localStorage` persistence, matches the TUI's process-scoped shape and avoids leaking command lines to disk. Mirrors `saklas/tui/app.py::_history_navigate` / `_push_input_history` semantics; bash-style dedupe (collapses immediate repeats, preserves ping-pong).

## Auto-regen (replaces A/B in v2.3)

The standalone A/B button is gone — Decision 13's "Ctrl+A toggles auto-regen" landed end-to-end. `Topbar.svelte`'s auto-regen toggle is the canonical entry, with a ⚙ popover for mode selection (`unsteered` / `inverted` / `reseed` / `cool` / `hot` / `custom: <partial recipe>`). `mode="unsteered"` is bit-identical to the old A/B flow.

`Chat.svelte`'s `twoColumns` derived OR-folds `pinnedActive || autoRegenActive` — pin generalizes the right column and auto-regen drives it when nothing is pinned. The WS `done` handler in `stores.svelte.ts` branches by mode: `unsteered` → `_sendShadowGenerate(steeredIdx)` (legacy shadow path, untouched); any other override → `loomRegenerateActive(mode)` + auto-pin to the new sibling. `toggleAutoRegen` retroactively fires a shadow for the most recent assistant turn when flipped on with `mode === "unsteered"` (mirrors the old A/B retroactive fire).

The legacy A/B compare flow follows:

`abState.enabled` toggles two-column rendering. The shadow gen (unsteered) runs after the steered turn finishes via `_sendShadowGenerate(steeredIdx)`, which:

1. Builds a messages list from `chatLog.turns[0..steeredIdx-1]` via `_buildShadowMessages` — past steered assistant turns ride along as context, the trailing user turn is what the unsteered model responds to.
2. Sends `input: <messages list>` + `stateless: true` over the WS so the server's `prior=[]` (stateless) and the messages list is the *only* context — no contamination from server-side history.

Toggling A/B from off→on while the chat already has steered turns immediately fires a shadow for the most recent assistant turn that doesn't have an `abPair` (skipped if a generation is in flight — the `done` handler will fire its own shadow). This is the "play the conversation back to the unsteered agent" flow: previously A/B only worked when regenerating the first turn.

## Thinking semantics

The thinking checkbox is a strict binary. Initial `samplingState.thinking` is `false` (not legacy `null` "auto"). Both `sendGenerate` and `_sendShadowGenerate` send `thinking: samplingState.thinking ?? false` so any null write that leaked through still serialises as explicit-off — the previous null path could leak through to the chat-template `enable_thinking=null` ambiguity and have the model think anyway.

## Loom sidebar (v2.3)

`panels/loom/LoomSidebar.svelte` is the collapsible left panel. Tree rendering is a flat DFS at depth-indented rows so nested recursive components don't stack frames. Decoration ring on each `LoomNode` follows Decision 10: `ringFor(node)` reads `highlightState.target` × `node.aggregate_readings[target]` and returns a reading value clamped to [-1, 1]; null when no probe is selected. The ring is the visual cue for "which branches the highlight probe lights up on" — independent of the dead-branch dim (`.node.dead { opacity: 0.3 }`) and the filter dim (`.tree-row.filtered-out { opacity: 0.5 }`) per Decision 18's three-channel separation.

Esc inside the sidebar defocuses the active element (search input / context menu) rather than collapsing the panel — collapse is a topbar action only. Context menu covers all five core ops + decorations (star/note) + pin + select-for-compare + fan-out + compare-children (user nodes with ≥2 assistant children).

## NodeCompareDrawer real-token rendering (v2.3)

`drawers/NodeCompareDrawer.svelte` renders per-token spans by iterating `diff.per_token` directly — each entry carries `{a_text, b_text, a_index, b_index, reading_deltas}` from the server's real model tokenization, so reading-delta tooltips and cross-pane hover highlight (`hoveredAnchorIdx` ↔ `aligned?.a_index`) align exactly with token boundaries. The earlier word-split-via-`/(\s+)/` shape mis-indexed alignment maps; the whitespace split is retained only as a fallback for nodes loaded without tokens (transcripts). Multi-select renders N columns with per-pair diffs; cross-pane highlight is gated `diffIdx === 0` (B-vs-A only) since N>2 pairings don't fit one hover target.

## Out of scope (v2.0)

- Multi-session UI (server URL-paths support it; the client assumes `default`).
- Auth UI for `SAKLAS_API_KEY` — the underlying Bearer middleware applies; no dedicated UI surface.
- Mobile / touch-first responsive layout. Saklas is a desktop research tool; min-width 1280px.
- Combobox autocomplete on the projection-target picker — the modal takes a free-form name; no live name-completion against the loaded-vector list.
- Pagination on HF pack search (capped at 20 results).
