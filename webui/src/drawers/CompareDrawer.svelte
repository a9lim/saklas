<script lang="ts">
  // Pairwise compare drawer — cross-layer cosine matrix between two
  // named steering vectors / probes.  Two dropdowns pick the pair (same
  // pool the layer-norms drawer uses: registered vectors ∪ active
  // probes); the body renders an L_A × L_B heatmap structurally akin to
  // the correlation matrix, but indexed by layer rather than by name.
  //
  // Data: GET /vectors/pairwise?a=&b= — the server falls back to monitor
  // profiles when a name isn't a registered steering vector, so probe
  // names resolve cleanly without a new endpoint.

  import { apiVectors, ApiError } from "../lib/api";
  import {
    closeDrawer,
    probeRack,
    vectorsState,
    refreshVectorList,
  } from "../lib/stores.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import type { PairwiseCompareResponse } from "../lib/types";

  // Drawer host forwards { params } — unused here, but the prop must
  // exist so the host's switch can pass it uniformly.
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  // Picker source — union of registered vectors and active probes,
  // sorted case-insensitively.  Mirrors LayerNormsDrawer so both
  // analysis tools share the same name space.
  const names = $derived.by<string[]>(() => {
    const set = new Set<string>();
    for (const v of vectorsState.names) set.add(v);
    for (const p of probeRack.active) set.add(p);
    return [...set].sort((a, b) =>
      a.localeCompare(b, undefined, { sensitivity: "base" }),
    );
  });

  let conceptA = $state<string>("");
  let conceptB = $state<string>("");
  let data = $state<PairwiseCompareResponse | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Refresh the rack on mount so newly extracted vectors show up in the
  // picker without requiring a full drawer reopen.  Cheap idempotent.
  $effect(() => {
    void refreshVectorList().catch(() => {/* non-fatal */});
  });

  // Auto-pick: first two distinct names when nothing is selected yet
  // (or when the prior selections drop out of the pool).  A drives B's
  // default to "next available != A" so the matrix renders on open
  // instead of waiting for a second click.
  $effect(() => {
    if (names.length === 0) {
      conceptA = "";
      conceptB = "";
      return;
    }
    if (!conceptA || !names.includes(conceptA)) {
      conceptA = names[0];
    }
    if (!conceptB || !names.includes(conceptB)) {
      conceptB = names.find((n) => n !== conceptA) ?? conceptA;
    }
  });

  async function load(a: string, b: string): Promise<void> {
    if (!a || !b) {
      data = null;
      return;
    }
    loading = true;
    error = null;
    try {
      data = await apiVectors.pairwise(a, b);
    } catch (e) {
      if (e instanceof ApiError) {
        const detail =
          e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message;
        error = `${e.status}: ${detail}`;
      } else {
        error = e instanceof Error ? e.message : String(e);
      }
      data = null;
    } finally {
      loading = false;
    }
  }

  // Re-fetch when either selection changes.  Idempotent server-side; no
  // need to dedupe identical (a, b) pairs.
  $effect(() => {
    void load(conceptA, conceptB);
  });

  function cellTitle(la: number, lb: number, v: number | null): string {
    return `${conceptA} L${la} × ${conceptB} L${lb}: ${v == null ? "—" : v.toFixed(3)}`;
  }

  function onClose(): void {
    closeDrawer();
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      onClose();
    }
  }

  /** Cell pixel size — matches the correlation matrix.  Typical model
   * is ~30 layers so the matrix lands ~900px square; the scroll
   * container handles larger models. */
  const CELL_SIZE = 26;

  const matrix = $derived(data?.matrix ?? null);
  const layersA = $derived<number[]>(data?.layers_a ?? []);
  const layersB = $derived<number[]>(data?.layers_b ?? []);
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" aria-label="Pairwise compare">
  <header class="drawer-header">
    <div class="title">
      <span class="label">pairwise compare</span>
      <span class="coord">
        {#if data}
          {layersA.length} × {layersB.length} layers · model {data.model ?? "—"}
        {:else if names.length < 2}
          need at least two vectors or probes
        {:else}
          pick two names
        {/if}
      </span>
    </div>
    <button type="button" class="close" onclick={onClose} aria-label="Close drawer">×</button>
  </header>

  <div class="picker-row">
    <label class="picker">
      <span class="picker-label">a</span>
      <select bind:value={conceptA} disabled={names.length === 0}>
        {#if names.length === 0}
          <option value="">(empty)</option>
        {:else}
          {#each names as name (name)}
            <option value={name}>{name}</option>
          {/each}
        {/if}
      </select>
    </label>
    <label class="picker">
      <span class="picker-label">b</span>
      <select bind:value={conceptB} disabled={names.length === 0}>
        {#if names.length === 0}
          <option value="">(empty)</option>
        {:else}
          {#each names as name (name)}
            <option value={name}>{name}</option>
          {/each}
        {/if}
      </select>
    </label>
  </div>

  <div class="body">
    {#if error}
      <div class="empty err">error: {error}</div>
    {:else if loading && !matrix}
      <div class="empty">loading…</div>
    {:else if !matrix || layersA.length === 0 || layersB.length === 0}
      <div class="empty">no layer data for the selected pair</div>
    {:else}
      <div class="grid-scroll">
        <table class="grid" style="--cell: {CELL_SIZE}px;">
          <thead>
            <tr>
              <th class="corner" scope="col">
                <span class="axis-a">{conceptA}</span>
                <span class="axis-sep">/</span>
                <span class="axis-b">{conceptB}</span>
              </th>
              {#each layersB as lb (lb)}
                <th class="col-label" scope="col" title="{conceptB} L{lb}">
                  <span>L{lb}</span>
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each layersA as la, i (la)}
              <tr>
                <th class="row-label" scope="row" title="{conceptA} L{la}">L{la}</th>
                {#each layersB as lb, j (lb)}
                  {@const v = matrix[i]?.[j] ?? null}
                  <td class="cell-td">
                    <HeatmapCell
                      value={v}
                      size={CELL_SIZE}
                      title={cellTitle(la, lb, v)}
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
      Per-layer cosine similarity, ``a``'s layers down rows, ``b``'s
      layers across columns.  Diagonal lights up when the two profiles
      track the same direction at the matching layer; off-diagonal
      structure shows how the concept "rotates" across depth.
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
    font-size: var(--text);
    border-left: 1px solid var(--border);
  }

  .drawer-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-4);
    padding: var(--space-4) var(--space-4);
    border-bottom: 1px solid var(--border);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0;
  }
  .coord {
    color: var(--fg-dim);
    font-size: var(--text-sm);
  }
  .close {
    background: transparent;
    color: var(--fg-muted);
    border: 1px solid var(--border);
    padding: 0 var(--space-3);
    font: inherit;
    font-size: var(--text-md);
    cursor: pointer;
    line-height: 1.4;
  }
  .close:hover {
    color: var(--fg-strong);
    border-color: var(--fg-muted);
  }

  .picker-row {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-4);
    border-bottom: 1px solid var(--border);
  }
  .picker {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex: 1 1 0;
    min-width: 0;
  }
  .picker-label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    text-transform: lowercase;
  }
  .picker select {
    background: var(--bg-alt);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: var(--space-1) var(--space-2);
    font: inherit;
    font-size: var(--text-sm);
    flex: 1 1 auto;
    min-width: 0;
  }
  .picker select:focus {
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }

  .body {
    flex: 1 1 auto;
    overflow: auto;
    min-height: 0;
    padding: var(--space-4) var(--space-4);
  }
  .empty {
    color: var(--fg-muted);
    font-style: italic;
    padding: var(--space-5) 0;
    line-height: 1.4;
  }
  .empty.err {
    color: var(--accent-error);
    font-style: normal;
  }

  .grid-scroll {
    overflow: auto;
    max-height: 100%;
    border: 1px solid var(--border);
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
    padding: 0 var(--space-3) 0 var(--space-2);
    color: var(--fg-dim);
    font-size: var(--text-xs);
    border-right: 1px solid var(--border);
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
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  .corner .axis-a,
  .corner .axis-b {
    color: var(--fg-strong);
  }
  .corner .axis-sep {
    color: var(--fg-dim);
    padding: 0 var(--space-1);
  }
  .grid .col-label {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    padding: 0;
    height: 3em;
    vertical-align: bottom;
    width: var(--cell);
    min-width: var(--cell);
    max-width: var(--cell);
    text-align: center;
  }
  .grid .col-label > span {
    display: inline-block;
    padding-bottom: var(--space-2);
  }
  .grid .cell-td {
    line-height: 0;
  }

  .drawer-footer {
    border-top: 1px solid var(--border);
    padding: var(--space-2) var(--space-4);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .hint {
    line-height: 1.4;
  }
</style>
