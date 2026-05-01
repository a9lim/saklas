<script lang="ts">
  // Per-token drilldown drawer — opens when a chat token is clicked.
  //
  // Replaces the v1.6 always-on inspector (which forced every user to look
  // at a per-token × per-layer × per-probe heatmap whether they wanted it
  // or not) with an on-demand surface keyed to a specific token.  Drawer
  // params come in via openDrawer("token_drilldown", { turnIdx, tokenIdx,
  // isThinking? }).  Click data flows through chatLog.turns[turnIdx] —
  // either turn.tokens[tokenIdx] (response stream, default) or
  // turn.thinkingTokens[tokenIdx] when isThinking is true.
  //
  // Layout: header with the token text + coordinates, body grid with
  // sortable layer (rows ascending) × probe (cols A-Z) cells.  Each cell
  // is a HeatmapCell tinted via tokens.scoreToRgb so highlight saturation
  // matches the chat tokens themselves.  Sticky labels keep orientation
  // when the grid scrolls.  A/B mode adds a steered/unsteered toggle when
  // turn.abPair exists.

  import { drawerState, closeDrawer, chatLog } from "../lib/stores.svelte";
  import type { ChatTurn, TokenScore } from "../lib/types";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import { HIGHLIGHT_SAT } from "../lib/tokens";

  interface DrawerParams {
    turnIdx: number;
    tokenIdx: number;
    /** When true the click came from the thinking-collapsible body, so
     * the source list is ``turn.thinkingTokens`` instead of
     * ``turn.tokens``.  Defaults to false (response stream). */
    isThinking?: boolean;
  }

  // ---- params + branch selection ---------------------------------------

  // Drawer host forwards { params } — but we read off the store via
  // $derived below since drawerState.params is the source of truth.
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  const params = $derived(drawerState.params as DrawerParams | null);
  const turnIdx = $derived(params?.turnIdx ?? -1);
  const tokenIdx = $derived(params?.tokenIdx ?? -1);
  const isThinking = $derived(params?.isThinking === true);

  /** Which branch we're inspecting — "primary" is the steered turn,
   * "shadow" is the unsteered abPair when available.  Local UI state
   * because the drawer toggles between them without changing the click
   * target.  The explicit ``$state<...>`` widens the rune's literal-
   * inferred type so reassignment to ``"shadow"`` later doesn't trip
   * a "comparison has no overlap" narrowing error. */
  type Branch = "primary" | "shadow";
  let branch: Branch = $state<Branch>("primary");

  /** Reset the branch when the click target changes — opening the drawer
   * on a new token should always start on the primary side. */
  $effect(() => {
    void turnIdx;
    void tokenIdx;
    branch = "primary";
  });

  const turn = $derived<ChatTurn | null>(
    turnIdx >= 0 && turnIdx < chatLog.turns.length
      ? chatLog.turns[turnIdx]
      : null,
  );

  /** The actual ChatTurn whose tokens we're inspecting — primary or the
   * abPair when ``branch === "shadow"``.  Null when the indices are
   * stale (e.g. the user cleared the chat after opening the drawer). */
  const inspected = $derived<ChatTurn | null>(
    branch === "shadow" ? (turn?.abPair ?? null) : turn,
  );

  const tokenList = $derived<TokenScore[]>(
    (isThinking ? inspected?.thinkingTokens : inspected?.tokens) ?? [],
  );

  const token = $derived<TokenScore | null>(
    tokenIdx >= 0 && tokenIdx < tokenList.length ? tokenList[tokenIdx] : null,
  );

  // ---- per-layer × per-probe grid --------------------------------------

  /** Layer indices sorted ascending.  Numeric sort over the string-keyed
   * layer map — TypeScript leaves Object.keys typed as string[] but the
   * server emits zero-padded integers, so a Number() cast is safe. */
  const layerKeys = $derived<string[]>(
    token?.perLayerScores
      ? Object.keys(token.perLayerScores).sort(
          (a, b) => Number(a) - Number(b),
        )
      : [],
  );

  /** Probe names sorted alphabetically (case-insensitive).  Source the
   * union over every layer's probe row so a probe with sparse coverage
   * still gets a column.  Most layers carry the same set in practice
   * but we don't enforce it. */
  const probeKeys = $derived.by<string[]>(() => {
    const pls = token?.perLayerScores;
    if (!pls) return [];
    const seen = new Set<string>();
    for (const layer of Object.keys(pls)) {
      for (const probe of Object.keys(pls[layer] ?? {})) {
        seen.add(probe);
      }
    }
    return [...seen].sort((a, b) =>
      a.localeCompare(b, undefined, { sensitivity: "base" }),
    );
  });

  function cellValue(layer: string, probe: string): number | null {
    const pls = token?.perLayerScores;
    if (!pls) return null;
    const row = pls[layer];
    if (!row) return null;
    const v = row[probe];
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  }

  function cellTooltip(layer: string, probe: string): string {
    const v = cellValue(layer, probe);
    if (v === null) return `L${layer} · ${probe} · —`;
    const sign = v >= 0 ? "+" : "";
    return `L${layer} · ${probe} · ${sign}${v.toFixed(3)}`;
  }

  // ---- cell sizing (responsive-ish) -----------------------------------

  /** Cell pixel size.  Capped on the high end so wide grids don't push
   * the drawer beyond reasonable widths; floored on the low end so the
   * tints remain visible.  ~22px reads cleanly with the optional value
   * label off; we keep value labels off because the cell count gets
   * large enough that text becomes noise. */
  const CELL_SIZE = 22;

  // ---- drawer chrome ---------------------------------------------------

  function onClose(): void {
    closeDrawer();
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      onClose();
    }
  }

  const hasAbPair = $derived(turn?.abPair != null);
  const isEmpty = $derived(layerKeys.length === 0 || probeKeys.length === 0);
</script>

<svelte:window onkeydown={onKeydown} />

<aside
  class="drawer"
  aria-label="Token drilldown"
>
  <header class="drawer-header">
    <div class="title">
      {#if token}
        <span class="label">token</span>
        <code class="tok-text">{JSON.stringify(token.text)}</code>
        <span class="coord">
          turn {turnIdx} · {isThinking ? "thinking" : "response"} token {tokenIdx}
        </span>
      {:else}
        <span class="label">token</span>
        <span class="coord">no token at ({turnIdx}, {tokenIdx})</span>
      {/if}
    </div>
    <button type="button" class="close" onclick={onClose} aria-label="Close drawer">
      ×
    </button>
  </header>

  {#if hasAbPair}
    <div class="branch-toggle" role="tablist" aria-label="Select branch">
      <button
        type="button"
        role="tab"
        aria-selected={branch === "primary"}
        class:active={branch === "primary"}
        onclick={() => (branch = "primary")}
      >steered</button>
      <button
        type="button"
        role="tab"
        aria-selected={branch === "shadow"}
        class:active={branch === "shadow"}
        onclick={() => (branch = "shadow")}
      >unsteered</button>
    </div>
  {/if}

  <div class="body">
    {#if !token}
      <div class="empty">
        Token not found.  The chat log may have been cleared or rewound
        since the drawer was opened.
      </div>
    {:else if isEmpty}
      <div class="empty">
        No per-layer scores captured for this token (probes not loaded?).
      </div>
    {:else}
      <div class="grid-scroll">
        <table class="grid" style="--cell: {CELL_SIZE}px;">
          <thead>
            <tr>
              <th class="corner" scope="col">L \ probe</th>
              {#each probeKeys as probe (probe)}
                <th class="col-label" scope="col" title={probe}>
                  <span>{probe}</span>
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each layerKeys as layer (layer)}
              <tr>
                <th class="row-label" scope="row" title={`Layer ${layer}`}>
                  L{layer}
                </th>
                {#each probeKeys as probe (probe)}
                  {@const v = cellValue(layer, probe)}
                  <td class="cell-td">
                    <HeatmapCell
                      value={v}
                      size={CELL_SIZE}
                      title={cellTooltip(layer, probe)}
                    />
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>

  <footer class="drawer-footer">
    <span class="hint">
      Tints map score / {HIGHLIGHT_SAT} clamped to ±1 — green = +pole, red =
      −pole, transparent ≈ 0.
    </span>
  </footer>
</aside>

<style>
  .drawer {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
    border-left: 1px solid var(--border);
  }

  .drawer-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.6em;
    padding: 0.6em 0.8em;
    border-bottom: 1px solid var(--border-dim);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: 0.2em;
    min-width: 0;
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .tok-text {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    background: var(--bg-alt);
    padding: 0.1em 0.4em;
    border: 1px solid var(--border);
    word-break: break-all;
    max-width: 28ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .coord {
    color: var(--fg-dim);
    font-size: var(--font-size-small);
  }
  .close {
    background: transparent;
    color: var(--fg-muted);
    border: 1px solid var(--border);
    padding: 0 0.5em;
    font: inherit;
    font-size: 1.1em;
    cursor: pointer;
    line-height: 1.4;
  }
  .close:hover {
    color: var(--fg-strong);
    border-color: var(--fg-muted);
  }

  .branch-toggle {
    display: flex;
    gap: 0.3em;
    padding: 0.4em 0.8em;
    border-bottom: 1px dashed var(--border-dim);
  }
  .branch-toggle button {
    background: transparent;
    color: var(--fg-dim);
    border: 1px solid var(--border);
    padding: 0.2em 0.7em;
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
    text-transform: lowercase;
  }
  .branch-toggle button:hover {
    color: var(--fg-strong);
    border-color: var(--fg-muted);
  }
  .branch-toggle button.active {
    color: var(--accent-blue);
    border-color: var(--accent-blue);
    background: rgba(88, 166, 255, 0.08);
  }

  .body {
    flex: 1 1 auto;
    overflow: auto;
    min-height: 0;
    padding: 0.6em 0.8em;
  }
  .empty {
    color: var(--fg-muted);
    font-style: italic;
    padding: 1em 0;
    line-height: 1.4;
  }

  .grid-scroll {
    overflow: auto;
    max-height: 100%;
    border: 1px solid var(--border-dim);
    background: var(--bg-alt);
  }
  .grid {
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
  }
  .grid th,
  .grid td {
    padding: 0;
    margin: 0;
    background: var(--bg-alt);
  }
  /* Sticky row + column labels so orientation survives long scrolls. */
  .grid thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    border-bottom: 1px solid var(--border);
  }
  .grid .row-label {
    position: sticky;
    left: 0;
    z-index: 1;
    text-align: right;
    padding: 0 0.5em 0 0.4em;
    color: var(--fg-dim);
    font-size: var(--font-size-tiny);
    border-right: 1px solid var(--border);
    white-space: nowrap;
  }
  .grid .corner {
    position: sticky;
    top: 0;
    left: 0;
    z-index: 3;
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
    text-align: left;
    padding: 0.2em 0.5em;
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }
  .grid .col-label {
    color: var(--fg-dim);
    font-size: var(--font-size-tiny);
    padding: 0;
    /* Rotate compact column labels so they fit narrow cells.  Wrap the
     * inner span so the rotation pivots around the cell box, not the
     * letterform. */
    height: 6em;
    vertical-align: bottom;
    width: var(--cell);
    min-width: var(--cell);
    max-width: var(--cell);
  }
  .grid .col-label > span {
    display: inline-block;
    transform: rotate(-60deg);
    transform-origin: left bottom;
    white-space: nowrap;
    padding-bottom: 0.4em;
  }
  .grid .cell-td {
    line-height: 0; /* HeatmapCell brings its own box; remove text leading. */
  }

  .drawer-footer {
    border-top: 1px solid var(--border-dim);
    padding: 0.4em 0.8em;
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
  }
  .hint {
    line-height: 1.4;
  }
</style>
