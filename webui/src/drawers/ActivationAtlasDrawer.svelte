<script lang="ts">
  // Activation atlas — token × layer × probe inspection across the
  // active conversation.  Three stacked sections share the flat drawer
  // chrome used elsewhere in the project (CorrelationDrawer,
  // LayerNormsDrawer): a token picker at the top, the layer/probe
  // heatmap in the middle, and the distribution lens (top alternatives
  // for the selected token) at the bottom.
  //
  // The heatmap reuses the project's canonical primitive,
  // ``HeatmapCell`` — the same diverging red/green scoreToRgb ramp, here
  // normalized per-probe by the node-coordinate extent (``probeAxisScale``)
  // so a column matches that probe's rack strip and chat-token tint rather
  // than the fixed ±HIGHLIGHT_SAT cutoff the cosine matrices keep.  The grid
  // is laid out exactly like ``CorrelationDrawer``'s matrix: sticky-header
  // table, rotated column labels for long probe names, scroll-area bounded
  // by the drawer body.

  import {
    closeDrawer,
    chatLog,
    highlightState,
    probeRack,
    probeAxisScale,
  } from "../lib/stores.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import type { TokenScore } from "../lib/types";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  // ----------------------------------------------------- token list ---

  interface AtlasToken {
    idx: number;
    turn: number;
    token: number;
    text: string;
    score: TokenScore;
  }

  const tokens = $derived.by<AtlasToken[]>(() => {
    const out: AtlasToken[] = [];
    for (let turn = 0; turn < chatLog.turns.length; turn++) {
      const t = chatLog.turns[turn];
      if (t.role !== "assistant") continue;
      for (let token = 0; token < (t.tokens ?? []).length; token++) {
        const score = t.tokens![token];
        out.push({ idx: out.length, turn, token, text: score.text, score });
      }
    }
    return out;
  });

  let selectedIdx = $state(0);
  const selected = $derived(tokens[selectedIdx] ?? tokens[0] ?? null);

  // -------------------------------------------------- heatmap shape ---

  // Layer keys (rows) — sorted ascending.  The drawer doesn't pad with
  // dropped layers the way LayerNormsDrawer does: the per-layer probe
  // map carries readings only for the layers the active probes were
  // sliced on, and that subset is exactly what's informative here.
  const layerKeys = $derived.by(() =>
    Object.keys(selected?.score.perLayerScores ?? {}).sort(
      (a, b) => Number(a) - Number(b),
    ),
  );

  // Probe keys (columns) — union of active rack names + every probe
  // referenced in this token's per-layer map.  Sorted alphabetically so
  // the column order is stable across selections.
  const probeKeys = $derived.by<string[]>(() => {
    const names = new Set<string>(probeRack.active);
    for (const row of Object.values(selected?.score.perLayerScores ?? {})) {
      for (const k of Object.keys(row)) names.add(k);
    }
    return [...names].sort();
  });

  function cell(layer: string, probe: string): number | null {
    const v = selected?.score.perLayerScores?.[layer]?.[probe];
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  }

  function cellTitle(layer: string, probe: string, v: number | null): string {
    return `${probe} L${layer}: ${v == null ? "—" : v.toFixed(3)}`;
  }

  // ----------------------------------------------------- summary -----

  const logprobStats = $derived.by(() => {
    const vals = tokens
      .map((t) => t.score.logprob)
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    if (vals.length === 0) return null;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    return {
      count: vals.length,
      mean,
      min: Math.min(...vals),
      max: Math.max(...vals),
    };
  });

  /** Surprise glyph width for the token-picker spark — logprob → [0, 1]
   * surprise via ``1 - probability``, scaled into a thin bar along the
   * bottom of each picker chip. */
  function surpriseWidth(tok: TokenScore): string {
    if (typeof tok.logprob !== "number") return "width: 4%;";
    const surprise = Math.min(1, Math.max(0, 1 - Math.exp(tok.logprob)));
    return `width: ${Math.max(4, surprise * 100)}%;`;
  }

  // ------------------------------------------------------ chrome -----

  function onClose(): void {
    closeDrawer();
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      onClose();
    }
  }

  /** Cell pixel size — matches ``CorrelationDrawer`` exactly so the
   * two heatmaps read at the same density. */
  const CELL_SIZE = 26;
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" aria-label="Activation atlas">
  <header class="drawer-header">
    <div class="title">
      <span class="eyebrow">activation atlas</span>
      <div class="name-row">
        <span class="meta">
          {tokens.length} {tokens.length === 1 ? "token" : "tokens"}
          {#if selected} · T{selected.turn}:{selected.token}{/if}
        </span>
      </div>
    </div>
    <button type="button" class="close" onclick={onClose} aria-label="Close drawer">
      ×
    </button>
  </header>

  <!-- Summary strip — same flat row pattern as the rest of the drawer. -->
  <div class="summary">
    <div class="stat">
      <span class="stat-label">highlight</span>
      <strong>{highlightState.target ?? "off"}</strong>
    </div>
    <div class="stat">
      <span class="stat-label">probes</span>
      <strong>{probeRack.active.length}</strong>
    </div>
    <div class="stat">
      <span class="stat-label">logprob mean</span>
      <strong>{logprobStats ? logprobStats.mean.toFixed(2) : "—"}</strong>
    </div>
    <div class="stat">
      <span class="stat-label">selected</span>
      <strong>{selected ? `T${selected.turn}:${selected.token}` : "—"}</strong>
    </div>
  </div>

  <div class="body">
    <!-- Token picker — horizontal scroll-strip of every assistant token. -->
    <section class="section">
      <div class="section-head">
        <h3>token timeline</h3>
        <span>{tokens.length} response tokens</span>
      </div>
      {#if tokens.length === 0}
        <div class="empty">
          generate with probes or top alternatives enabled to populate the atlas
        </div>
      {:else}
        <div class="token-grid" role="listbox" aria-label="atlas tokens">
          {#each tokens as tok (tok.idx)}
            <button
              type="button"
              class="chip"
              class:selected={tok.idx === selectedIdx}
              role="option"
              aria-selected={tok.idx === selectedIdx}
              onclick={() => (selectedIdx = tok.idx)}
              title={`turn ${tok.turn}, token ${tok.token}, logprob ${tok.score.logprob ?? "—"}`}
            >
              <span class="spark" style={surpriseWidth(tok.score)}></span>
              <code>{tok.text}</code>
            </button>
          {/each}
        </div>
      {/if}
    </section>

    <!-- Layer × probe heatmap.  Grid structure mirrors
         CorrelationDrawer.svelte (sticky thead/leftmost-col, rotated
         col-labels, HeatmapCell with the canonical scoreToRgb palette). -->
    <section class="section">
      <div class="section-head">
        <h3>layer / probe heatmap</h3>
        <span>
          {selected ? JSON.stringify(selected.text) : "no token selected"}
        </span>
      </div>
      {#if !selected || layerKeys.length === 0 || probeKeys.length === 0}
        <div class="empty">
          no per-layer probe scores on the selected token — load probes and
          generate a fresh turn to populate this grid
        </div>
      {:else}
        <div class="grid-scroll">
          <table class="grid" style="--cell: {CELL_SIZE}px;">
            <thead>
              <tr>
                <th class="corner" scope="col">layer</th>
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
                  <th class="row-label" scope="row" title={`layer ${layer}`}>
                    L{layer}
                  </th>
                  {#each probeKeys as probe (probe)}
                    {@const v = cell(layer, probe)}
                    <td class="cell-td">
                      <HeatmapCell
                        value={v}
                        scale={probeAxisScale(probe)}
                        size={CELL_SIZE}
                        title={cellTitle(layer, probe, v)}
                      />
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </section>

    <!-- Distribution lens — top alternatives for the selected token. -->
    <section class="section">
      <div class="section-head">
        <h3>distribution lens</h3>
        <span>{selected?.score.topAlts?.length ?? 0} alternatives</span>
      </div>
      {#if selected?.score.topAlts?.length}
        <div class="alts">
          {#each selected.score.topAlts as alt (alt.id)}
            <div class="alt-row">
              <code>{JSON.stringify(alt.text)}</code>
              <span class="alt-id">#{alt.id}</span>
              <strong class="alt-logprob">{alt.logprob.toFixed(3)}</strong>
            </div>
          {/each}
        </div>
      {:else}
        <div class="empty">
          enable top alternatives in advanced sampling to see token counterfactuals
        </div>
      {/if}
    </section>
  </div>

  <footer class="drawer-footer">
    <span class="hint">
      Per-token per-layer probe readings under the active rack.  Heatmap
      colors match the correlation matrix and probe strip — diverging
      red/green via the project's scoreToRgb scale.
    </span>
  </footer>
</aside>

<style>
  /* v2 sheet interior — the host paints the sheet surface (glass hairline,
   * radius, --bg-alt fill), so the root is transparent; chrome speaks sans
   * and every value/identifier/expression sits in mono. */
  .drawer {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    background: transparent;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }

  .drawer-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--space-5) var(--space-6);
    border-bottom: 1px solid var(--glass-line);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .eyebrow {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .name-row {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .meta {
    color: var(--fg-subtle);
    font-size: var(--text-sm);
    white-space: nowrap;
  }
  .close {
    background: transparent;
    color: var(--fg-muted);
    border: 1px solid var(--border);
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font: inherit;
    font-size: var(--text-md);
    line-height: 1;
    cursor: pointer;
    flex: none;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .close:hover {
    color: var(--fg);
    background: var(--bg-hover);
    border-color: var(--fg-muted);
  }

  /* Summary strip — flat row, no card backgrounds; same border-bottom
   * pattern as the loom side bar headers (single hairline separator). */
  .summary {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--space-5);
    padding: var(--space-3) var(--space-6);
    border-bottom: 1px solid var(--glass-line);
  }
  .stat {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .stat-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .stat strong {
    color: var(--fg);
    font-family: var(--font-mono);
    font-weight: var(--weight-medium);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }

  .body {
    flex: 1 1 auto;
    overflow: auto;
    min-height: 0;
  }
  .section {
    padding: var(--space-5) var(--space-6);
    border-bottom: 1px solid var(--glass-line);
  }
  .section:last-child {
    border-bottom: 0;
  }
  .section-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-4);
    margin-bottom: var(--space-3);
  }
  .section-head h3 {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .section-head span {
    color: var(--fg-subtle);
    font-size: var(--text-xs);
  }

  .empty {
    color: var(--fg-muted);
    padding: var(--space-6) 0;
    line-height: 1.5;
    max-width: 62ch;
    margin: 0 auto;
    text-align: center;
  }

  /* Token picker chips — minimal pills.  Selected = accent border +
   * accent-subtle fill, matching the loom-node selected state. */
  .token-grid {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    align-content: flex-start;
    max-height: 12rem;
    overflow: auto;
  }
  .chip {
    position: relative;
    max-width: 11rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: transparent;
    color: var(--fg);
    padding: var(--space-2) var(--space-3);
    overflow: hidden;
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .chip:hover {
    background: var(--bg-hover);
  }
  .chip.selected {
    border-color: var(--accent);
    background: var(--accent-subtle);
  }
  /* Surprise spark — a vocabulary-distribution quantity, so it reads in
   * logit-space blue (the lens family), not the demoted amber. */
  .spark {
    position: absolute;
    inset: auto auto 0 0;
    height: 2px;
    background: var(--highlight-surprise);
    opacity: 0.8;
  }
  code {
    position: relative;
    font-family: var(--font-mono);
    white-space: pre-wrap;
  }

  /* Heatmap grid — same well chrome as CorrelationDrawer so the two
   * surfaces read identically (sticky headers, rotated col labels,
   * hairline glass borders, opaque sticky cells). */
  .grid-scroll {
    overflow: auto;
    max-height: 32rem;
    border: 1px solid var(--glass-line);
    border-radius: var(--radius);
    background: var(--bg);
  }
  .grid {
    border-collapse: separate;
    border-spacing: 0;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }
  .grid th,
  .grid td {
    padding: 0;
    margin: 0;
    background: var(--bg);
  }
  .grid thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    border-bottom: 1px solid var(--glass-line);
  }
  .grid .row-label {
    position: sticky;
    left: 0;
    z-index: 1;
    text-align: right;
    padding: 0 var(--space-3) 0 var(--space-2);
    color: var(--fg-dim);
    font-size: var(--text-xs);
    border-right: 1px solid var(--glass-line);
    white-space: nowrap;
  }
  .grid .corner {
    position: sticky;
    top: 0;
    left: 0;
    z-index: 3;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-align: left;
    padding: var(--space-1) var(--space-3);
    border-right: 1px solid var(--glass-line);
    border-bottom: 1px solid var(--glass-line);
  }
  .grid .col-label {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    padding: 0;
    height: 7em;
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
    padding-bottom: var(--space-2);
  }
  .grid .cell-td {
    line-height: 0;
  }

  /* Distribution lens — flat row list, same hairline idiom as the loom-
   * node chip / nav button. */
  .alts {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    max-height: 18rem;
    overflow: auto;
  }
  .alt-row {
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: var(--space-4);
    align-items: center;
    padding: var(--space-2) var(--space-3);
    border-bottom: 1px solid var(--glass-line);
    font-size: var(--text-xs);
  }
  .alt-row:last-child {
    border-bottom: 0;
  }
  .alt-id {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }
  /* A vocabulary-distribution logprob — logit-space blue, same family as
   * the lens, not the demoted amber. */
  .alt-logprob {
    color: var(--pillar-lens);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }

  .drawer-footer {
    border-top: 1px solid var(--glass-line);
    padding: var(--space-3) var(--space-6);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .hint {
    line-height: 1.5;
  }
</style>
