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

  import { closeDrawer, refreshCorrelation, steerRack } from "../lib/stores.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import Button from "../lib/ui/Button.svelte";

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
    // reopens read straight off ``steerRack.correlation``.
    if (!steerRack.correlation) void reload();
  });

  const data = $derived(steerRack.correlation);
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
      <span class="eyebrow">correlation matrix</span>
      <div class="name-row">
        <span class="meta">
          {names.length} {names.length === 1 ? "name" : "names"}
          {#if names.length > 0} · steering + probes{/if}
        </span>
      </div>
    </div>
    <div class="actions">
      <Button
        size="sm"
        onclick={() => void reload()}
        disabled={loading}
        title="refresh"
      >{loading ? "…" : "refresh"}</Button>
      <button type="button" class="close" onclick={onClose} aria-label="Close drawer">
        ✕
      </button>
    </div>
  </header>

  <div class="body">
    {#if error}
      <div class="empty err">error: {error}</div>
    {:else if loading && !data}
      <div class="empty">loading…</div>
    {:else if !data || names.length === 0}
      <div class="empty">no data</div>
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
    <span class="hint">magnitude-weighted cosine · shared layers</span>
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
  .actions {
    display: flex;
    gap: var(--space-3);
    align-items: center;
    flex: none;
  }
  .close {
    background: var(--glass);
    color: var(--fg-muted);
    border: 1px solid transparent;
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
      background var(--dur-fast) var(--ease-out);
  }
  .close:hover {
    color: var(--fg);
    background: var(--glass-strong);
  }

  .body {
    flex: 1 1 auto;
    overflow: auto;
    min-height: 0;
    padding: var(--space-5) var(--space-6);
  }
  .empty {
    color: var(--fg-muted);
    padding: var(--space-6) 0;
    line-height: 1.5;
    max-width: 62ch;
  }
  .empty.err {
    color: var(--accent-error);
  }

  /* Data well — recessed matrix.  Every column/row label here IS an
   * identifier (vector/probe name), not a category header, so the whole
   * grid speaks mono (matches the probes tab in TokenDrilldownDrawer). */
  .grid-scroll {
    overflow: auto;
    max-height: 100%;
    border-radius: var(--radius);
    background: var(--bg);
  }
  .grid {
    border-collapse: separate;
    border-spacing: 1px;
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
    box-shadow: var(--shadow-sticky);
  }
  .grid .row-label {
    position: sticky;
    left: 0;
    z-index: 1;
    text-align: right;
    padding: 0 var(--space-3) 0 var(--space-2);
    color: var(--fg-dim);
    font-size: var(--text-xs);
    box-shadow: 2px 0 8px rgba(0, 0, 0, 0.45);
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
    box-shadow: var(--shadow-sticky), 2px 0 8px rgba(0, 0, 0, 0.45);
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

  .drawer-footer {
    padding: var(--space-3) var(--space-6);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .hint {
    line-height: 1.5;
  }
</style>
