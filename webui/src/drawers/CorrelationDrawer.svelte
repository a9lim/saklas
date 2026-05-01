<script lang="ts">
  // Correlation overlay — N×N magnitude-weighted cosine matrix across
  // every registered steering vector AND every active probe.  Replaces
  // the v1.7 inline ReferenceCollapsibles' correlation section with a
  // drawer-style overlay so the rack zone reclaims that vertical space
  // and so the matrix has room to breathe at larger N.
  //
  // Data: GET /correlation with no ``names=`` filter — the server-side
  // pool unions session.vectors and monitor.probe_names so probes that
  // were never registered as steering vectors still show up.
  //
  // Layout mirrors TokenDrilldownDrawer: header (title + ✕) · sticky-
  // header table body · footer hint.  Cells reuse <HeatmapCell showValue>
  // for the printed cosine — same color mapping as the click-token grid
  // so reading a row across both surfaces stays consistent.

  import { closeDrawer, refreshCorrelation, vectorRack } from "../lib/stores.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";

  // Drawer host forwards { params } — unused here, but the prop must
  // exist so the host's switch can pass it uniformly.
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  // Lazy-fetch on mount when no snapshot exists; reopens reuse the
  // cached matrix so the drawer lands instantly.
  let loading = $state(false);
  let error = $state<string | null>(null);

  async function reload(): Promise<void> {
    loading = true;
    error = null;
    try {
      // ``refreshCorrelation()`` with no names → server unions vectors +
      // probes and returns the full matrix.
      await refreshCorrelation(null);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    // Trigger initial load only if the snapshot is empty.  Subsequent
    // reopens read straight off ``vectorRack.correlation``.
    if (!vectorRack.correlation) void reload();
  });

  const data = $derived(vectorRack.correlation);
  const names = $derived<string[]>(data?.names ?? []);

  function cellTitle(a: string, b: string, v: number | null): string {
    return `${a} vs ${b}: ${v == null ? "—" : v.toFixed(3)}`;
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

  /** Cell pixel size — wider than the click-drilldown's grid because
   * we want to read the printed cosine value inside each cell, and
   * narrow column count (typical N=20-40) leaves room. */
  const CELL_SIZE = 26;
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" aria-label="Correlation matrix">
  <header class="drawer-header">
    <div class="title">
      <span class="label">correlation</span>
      <span class="coord">
        {names.length} {names.length === 1 ? "name" : "names"}
        {#if names.length > 0} · vectors + probes{/if}
      </span>
    </div>
    <div class="actions">
      <button
        type="button"
        class="refresh"
        onclick={() => void reload()}
        disabled={loading}
        title="Re-fetch the correlation matrix"
      >{loading ? "…" : "refresh"}</button>
      <button type="button" class="close" onclick={onClose} aria-label="Close drawer">
        ×
      </button>
    </div>
  </header>

  <div class="body">
    {#if error}
      <div class="empty err">error: {error}</div>
    {:else if loading && !data}
      <div class="empty">loading…</div>
    {:else if !data || names.length === 0}
      <div class="empty">no vectors or probes registered</div>
    {:else}
      <div class="grid-scroll">
        <table class="grid" style="--cell: {CELL_SIZE}px;">
          <thead>
            <tr>
              <th class="corner" scope="col">name</th>
              {#each names as col (col)}
                <th class="col-label" scope="col" title={col}>
                  <span>{col}</span>
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each names as a (a)}
              <tr>
                <th class="row-label" scope="row" title={a}>{a}</th>
                {#each names as b (b)}
                  {@const v = data.matrix[a]?.[b] ?? null}
                  <td class="cell-td">
                    <HeatmapCell
                      value={v}
                      size={CELL_SIZE}
                      showValue
                      title={cellTitle(a, b, v)}
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
      Magnitude-weighted cosine across shared layers.  Diagonal pinned
      at +1.  Sources: registered steering vectors ∪ active probes.
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
  .coord {
    color: var(--fg-dim);
    font-size: var(--font-size-small);
  }
  .actions {
    display: flex;
    gap: 0.4em;
    align-items: center;
  }
  .refresh {
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.15em 0.6em;
    font: inherit;
    font-size: 0.85em;
    text-transform: lowercase;
    cursor: pointer;
  }
  .refresh:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .refresh:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
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
  .empty.err {
    color: var(--accent-error);
    font-style: normal;
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
    padding-bottom: 0.4em;
  }
  .grid .cell-td {
    line-height: 0;
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
