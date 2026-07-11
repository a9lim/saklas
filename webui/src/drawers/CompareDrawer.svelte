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
  import Select from "../lib/Select.svelte";
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

  /** Options for the A / B pickers — same list both sides; the
   *  "(empty)" fallback only renders when the catalog is empty. */
  const nameOptions = $derived(
    names.length === 0
      ? [{ value: "", label: "(empty)" }]
      : names.map((n) => ({ value: n, label: n })),
  );
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
      <span class="eyebrow">pairwise compare</span>
      <div class="name-row">
        {#if data}
          <code class="name">{conceptA} × {conceptB}</code>
          <span class="meta">{layersA.length} × {layersB.length} layers · model {data.model ?? "—"}</span>
        {:else if names.length < 2}
          <span class="meta">need at least two vectors or probes</span>
        {:else}
          <span class="meta">pick two names</span>
        {/if}
      </div>
    </div>
    <button type="button" class="close" onclick={onClose} aria-label="Close drawer">✕</button>
  </header>

  <div class="picker-row">
    <label class="picker">
      <span class="picker-label">a</span>
      <Select
        bind:value={conceptA}
        options={nameOptions}
        disabled={names.length === 0}
        ariaLabel="Concept A"
      />
    </label>
    <label class="picker">
      <span class="picker-label">b</span>
      <Select
        bind:value={conceptB}
        options={nameOptions}
        disabled={names.length === 0}
        ariaLabel="Concept B"
      />
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
  /* v2 sheet interior — the host paints the sheet surface, so the root
   * stays transparent and chrome speaks sans (data stays mono). */
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
    flex-wrap: wrap;
  }
  .name {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .meta {
    color: var(--fg-subtle);
    font-size: var(--text-sm);
    white-space: nowrap;
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

  .picker-row {
    display: flex;
    align-items: center;
    gap: var(--space-5);
    padding: var(--space-4) var(--space-6);
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
  }
  /* The themed Select owns its own chrome — the picker label provides
   * the only host styling. */

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

  /* Data well — sticky label cells stay OPAQUE (they occlude scrolled
   * cells), so they paint --bg rather than glass. */
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
    padding: var(--space-3) var(--space-6);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .hint {
    line-height: 1.5;
  }
</style>
