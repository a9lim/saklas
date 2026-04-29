<script lang="ts">
  // Reference collapsibles — two independently-foldable panels stacked
  // beneath the rack:
  //   1. correlation N×N — cosine similarity between every loaded vector
  //   2. layer norms (per vector) — bar chart of ||baked|| per layer for
  //      a user-picked vector
  //
  // Both sections are closed by default; opening one lazy-fetches the
  // data it needs.  The correlation matrix reuses any already-bootstrapped
  // ``vectorRack.correlation`` snapshot when present so the "open it for
  // the first time" interaction lands instantly when bootstrap already
  // populated it.
  //
  // Restyled from the v1.6 panels — same math, design tokens instead of
  // inline GitHub-dark hex, and the new <HeatmapCell>/<Bar> primitives in
  // place of hand-rolled rgb/style bookkeeping.

  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import Bar from "../lib/charts/Bar.svelte";
  import {
    vectorRack,
    vectorsState,
    refreshCorrelation,
    refreshVector,
  } from "../lib/stores.svelte";

  // ----------------------------------------------------- correlation --

  let correlationOpen = $state(false);
  let correlationLoading = $state(false);
  let correlationError = $state<string | null>(null);

  // First-open lazy fetch — but only when we don't already have a
  // bootstrap-populated snapshot.  Reopening a previously-fetched
  // section reuses the cached matrix without a refetch.
  async function openCorrelation(): Promise<void> {
    correlationOpen = true;
    if (vectorRack.correlation) return;
    await reloadCorrelation();
  }

  async function reloadCorrelation(): Promise<void> {
    correlationLoading = true;
    correlationError = null;
    try {
      await refreshCorrelation();
    } catch (e) {
      correlationError = e instanceof Error ? e.message : String(e);
    } finally {
      correlationLoading = false;
    }
  }

  function onCorrelationToggle(ev: Event): void {
    const el = ev.currentTarget as HTMLDetailsElement;
    if (el.open && !correlationOpen) {
      void openCorrelation();
    } else if (!el.open) {
      correlationOpen = false;
    }
  }

  const correlationNames = $derived(vectorRack.correlation?.names ?? []);

  // ----------------------------------------------------- layer norms --

  let layerNormsOpen = $state(false);
  let selectedVector = $state<string>("");
  let layerNormsLoading = $state(false);
  let layerNormsError = $state<string | null>(null);

  // Profile lookup — drawn straight off the rack's per-vector cache.
  // Refresh is a no-op when the cache already has the entry; we still
  // call refreshVector lazily on selection changes so a freshly-loaded
  // vector lands without needing a bootstrap re-run.
  const selectedProfile = $derived(
    selectedVector ? vectorRack.profiles.get(selectedVector) ?? null : null,
  );

  async function ensureSelectedProfile(): Promise<void> {
    if (!selectedVector) return;
    if (vectorRack.profiles.has(selectedVector)) return;
    layerNormsLoading = true;
    layerNormsError = null;
    try {
      await refreshVector(selectedVector);
    } catch (e) {
      layerNormsError = e instanceof Error ? e.message : String(e);
    } finally {
      layerNormsLoading = false;
    }
  }

  // Re-fetch on selection change (no-op when cached).  $effect runs after
  // the binding update so the network call doesn't fight reactivity.
  $effect(() => {
    if (layerNormsOpen && selectedVector) {
      void ensureSelectedProfile();
    }
  });

  function onLayerNormsToggle(ev: Event): void {
    const el = ev.currentTarget as HTMLDetailsElement;
    layerNormsOpen = el.open;
    // Auto-pick first registered vector on first open so the panel has
    // something to render rather than an empty placeholder.
    if (
      el.open &&
      !selectedVector &&
      vectorsState.names.length > 0
    ) {
      selectedVector = vectorsState.names[0];
    }
  }

  // Sort layers numerically so the bar chart reads top-to-bottom in
  // model order rather than string-lexicographic order.
  const sortedLayers = $derived.by(() => {
    if (!selectedProfile) return [] as number[];
    return Object.keys(selectedProfile.per_layer_norms)
      .map((k) => parseInt(k, 10))
      .filter((n) => !Number.isNaN(n))
      .sort((a, b) => a - b);
  });

  const maxNorm = $derived.by(() => {
    if (!selectedProfile) return 0;
    let m = 0;
    for (const l of sortedLayers) {
      const v = selectedProfile.per_layer_norms[String(l)];
      if (typeof v === "number" && Math.abs(v) > m) m = Math.abs(v);
    }
    return m;
  });
</script>

<section class="reference">
  <details class="block" ontoggle={onCorrelationToggle}>
    <summary>
      <span class="caret" aria-hidden="true">{correlationOpen ? "▼" : "▶"}</span>
      <span class="title">correlation N×N</span>
      {#if correlationOpen}
        <button
          type="button"
          class="refresh"
          onclick={(ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            void reloadCorrelation();
          }}
          disabled={correlationLoading}
          title="Re-fetch the correlation matrix"
        >
          {correlationLoading ? "…" : "refresh"}
        </button>
      {/if}
    </summary>

    <div class="body">
      {#if correlationError}
        <div class="hint err">error: {correlationError}</div>
      {:else if correlationLoading && !vectorRack.correlation}
        <div class="hint">loading…</div>
      {:else if !vectorRack.correlation || correlationNames.length === 0}
        <div class="hint">no vectors loaded</div>
      {:else}
        <div
          class="matrix"
          style="grid-template-columns: auto repeat({correlationNames.length}, 18px);"
        >
          <div class="corner"></div>
          {#each correlationNames as name (name)}
            <div class="col-label" title={name}>{name}</div>
          {/each}
          {#each correlationNames as a (a)}
            <div class="row-label" title={a}>{a}</div>
            {#each correlationNames as b (b)}
              {@const v = vectorRack.correlation?.matrix[a]?.[b] ?? null}
              <HeatmapCell
                value={v}
                size={18}
                showValue
                title={`${a} vs ${b}: ${v == null ? "—" : v.toFixed(3)}`}
              />
            {/each}
          {/each}
        </div>
      {/if}
    </div>
  </details>

  <details class="block" ontoggle={onLayerNormsToggle}>
    <summary>
      <span class="caret" aria-hidden="true">{layerNormsOpen ? "▼" : "▶"}</span>
      <span class="title">layer norms (per vector)</span>
    </summary>

    <div class="body">
      {#if vectorsState.names.length === 0}
        <div class="hint">no vectors loaded</div>
      {:else}
        <label class="picker">
          <span class="picker-label">vector</span>
          <select bind:value={selectedVector}>
            <option value="" disabled>— pick a vector —</option>
            {#each vectorsState.names as name (name)}
              <option value={name}>{name}</option>
            {/each}
          </select>
        </label>

        {#if layerNormsError}
          <div class="hint err">error: {layerNormsError}</div>
        {:else if !selectedVector}
          <div class="hint">no vector selected</div>
        {:else if layerNormsLoading && !selectedProfile}
          <div class="hint">loading…</div>
        {:else if !selectedProfile}
          <div class="hint">no profile data</div>
        {:else if sortedLayers.length === 0}
          <div class="hint">no layers</div>
        {:else}
          <div class="bars">
            {#each sortedLayers as l (l)}
              {@const v = selectedProfile.per_layer_norms[String(l)] ?? 0}
              <div class="row">
                <span class="label">L{l}</span>
                <Bar value={v} max={maxNorm} width={144} height={8} />
                <span class="value">{v.toFixed(3)}</span>
              </div>
            {/each}
          </div>
        {/if}
      {/if}
    </div>
  </details>
</section>

<style>
  .reference {
    display: flex;
    flex-direction: column;
    gap: 0.4em;
  }
  .block {
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    background: var(--bg-alt);
    overflow: hidden;
  }
  .block + .block {
    /* Subtle separation, no double-border because each block already
     * has its own thin border. */
    margin-top: 0;
  }
  summary {
    list-style: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.5em;
    padding: 0.4em 0.6em;
    font-size: 0.85em;
    color: var(--fg-subtle);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    user-select: none;
  }
  summary::-webkit-details-marker {
    display: none;
  }
  summary:hover {
    background: var(--bg-elev);
    color: var(--fg-strong);
  }
  .caret {
    color: var(--fg-muted);
    font-size: 0.85em;
    width: 0.9em;
    display: inline-block;
  }
  .title {
    flex: 0 1 auto;
  }
  .refresh {
    margin-left: auto;
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.1em 0.5em;
    font: inherit;
    font-size: 0.8em;
    text-transform: none;
    letter-spacing: 0;
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
  .body {
    padding: 0.5em 0.6em 0.7em;
    border-top: 1px solid var(--border-dim);
    background: var(--bg);
  }
  .hint {
    color: var(--fg-muted);
    font-size: 0.85em;
    padding: 0.3em 0.1em;
  }
  .hint.err {
    color: var(--accent-error);
  }

  /* ----- correlation matrix ----- */
  .matrix {
    display: grid;
    overflow: auto;
    font-size: var(--font-size-tiny);
    align-items: center;
    column-gap: 1px;
    row-gap: 1px;
    max-height: 240px;
  }
  .corner {
    /* Empty top-left cell — left blank to keep the grid aligned. */
    width: 100%;
    height: 100%;
  }
  .col-label {
    color: var(--fg-muted);
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    font-size: 9px;
    padding: 0.1em 0;
    text-align: left;
    /* Keep column labels visible while the matrix scrolls vertically. */
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 1;
  }
  .row-label {
    color: var(--fg-muted);
    text-align: right;
    padding-right: 0.4em;
    font-size: var(--font-size-tiny);
    /* Sticky so row labels stay visible while the matrix scrolls
     * horizontally on narrow racks. */
    position: sticky;
    left: 0;
    background: var(--bg);
    z-index: 1;
    white-space: nowrap;
  }

  /* ----- layer norms ----- */
  .picker {
    display: flex;
    align-items: center;
    gap: 0.5em;
    margin-bottom: 0.5em;
  }
  .picker-label {
    color: var(--fg-muted);
    font-size: 0.85em;
  }
  .picker select {
    background: var(--bg-alt);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: 0.2em 0.4em;
    font: inherit;
    font-size: 0.85em;
    flex: 1 1 auto;
  }
  .picker select:focus {
    outline: 1px solid var(--accent-blue);
    outline-offset: -1px;
  }
  .bars {
    display: flex;
    flex-direction: column;
    gap: 1px;
    font-size: var(--font-size-tiny);
    max-height: 240px;
    overflow: auto;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.5em;
  }
  .label {
    color: var(--fg-muted);
    width: 3em;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .value {
    color: var(--fg-dim);
    width: 4.5em;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
</style>
