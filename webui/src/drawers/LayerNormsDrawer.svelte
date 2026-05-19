<script lang="ts">
  // Layer-norms overlay — per-layer ``||baked||`` bar chart for any
  // registered steering vector OR active probe.  Replaces the v1.7
  // inline ReferenceCollapsibles' layer-norms section.  Picker spans
  // both registries because the structural question ("how concentrated
  // is this concept across layers?") applies to probes as well as
  // steering vectors.
  //
  // Data: GET /vectors/{name}/diagnostics — the server falls back to
  // monitor.profiles when the name isn't a registered steering vector,
  // so probe names resolve cleanly without a new endpoint.  Optional
  // ``params.name`` pre-selects the picker (used when launching from a
  // per-strip "show layer norms" affordance, even if no caller does
  // that today).

  import { closeDrawer, drawerState, probeRack, vectorsState } from "../lib/stores.svelte";
  import { apiVectors, ApiError } from "../lib/api";
  import Bar from "../lib/charts/Bar.svelte";
  import type { VectorDiagnosticsResponse } from "../lib/types";

  interface DrawerParams {
    /** Optional name to pre-select.  Falls back to the first available
     * vector or probe when null. */
    name?: string;
  }

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  const params = $derived(drawerState.params as DrawerParams | null);

  // Picker source — union of registered vectors and active probes,
  // sorted case-insensitively for the dropdown.
  const names = $derived.by<string[]>(() => {
    const set = new Set<string>();
    for (const v of vectorsState.names) set.add(v);
    for (const p of probeRack.active) set.add(p);
    return [...set].sort((a, b) =>
      a.localeCompare(b, undefined, { sensitivity: "base" }),
    );
  });

  let selected = $state<string>("");
  let data = $state<VectorDiagnosticsResponse | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Auto-pick: explicit param wins; else first available name; else "".
  $effect(() => {
    const wanted = params?.name;
    if (wanted && names.includes(wanted)) {
      selected = wanted;
      return;
    }
    if (!selected || !names.includes(selected)) {
      selected = names[0] ?? "";
    }
  });

  async function load(name: string): Promise<void> {
    if (!name) {
      data = null;
      return;
    }
    loading = true;
    error = null;
    data = null;
    try {
      data = await apiVectors.diagnostics(name);
    } catch (e) {
      if (e instanceof ApiError) {
        error = `${e.status}`;
      } else {
        error = e instanceof Error ? e.message : String(e);
      }
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    void load(selected);
  });

  // Full-range layer view: every layer 0..total_layers-1, with the
  // profile's per-layer ``||baked||`` filled in where present and 0 for
  // layers DLS dropped (or that never landed in the profile).  Showing
  // the full strip lets the user *see* the discriminative-layer-select
  // pattern rather than infer it from the gaps in a sparse row list —
  // the dropped layers read as flat-zero bars between the active ones.
  const sortedLayers = $derived.by<{ layer: number; magnitude: number }[]>(() => {
    const present = new Map<number, number>();
    for (const e of data?.layers ?? []) present.set(e.layer, e.magnitude);
    const total = data?.total_layers ?? 0;
    if (total <= 0) {
      return [...present.entries()]
        .map(([layer, magnitude]) => ({ layer, magnitude }))
        .sort((a, b) => a.layer - b.layer);
    }
    const rows: { layer: number; magnitude: number }[] = [];
    for (let i = 0; i < total; i++) {
      rows.push({ layer: i, magnitude: present.get(i) ?? 0 });
    }
    return rows;
  });

  const maxMagnitude = $derived(
    sortedLayers.reduce(
      (m, e) => (Math.abs(e.magnitude) > m ? Math.abs(e.magnitude) : m),
      0,
    ),
  );

  const stoplight = $derived(data?.diagnostics_summary?.stoplight ?? null);

  function onClose(): void {
    closeDrawer();
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      onClose();
    }
  }
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" aria-label="Layer norms">
  <header class="drawer-header">
    <div class="title">
      <span class="label">layer norms</span>
      <span class="coord">
        {#if data}
          {data.total_layers} layers · model {data.model}
        {:else if names.length === 0}
          no vectors or probes registered
        {:else}
          pick a name
        {/if}
      </span>
    </div>
    <button type="button" class="close" onclick={onClose} aria-label="Close drawer">×</button>
  </header>

  <div class="picker-row">
    <label class="picker">
      <span class="picker-label">name</span>
      <select bind:value={selected} disabled={names.length === 0}>
        {#if names.length === 0}
          <option value="">(empty)</option>
        {:else}
          {#each names as name (name)}
            <option value={name}>{name}</option>
          {/each}
        {/if}
      </select>
    </label>
    {#if stoplight}
      <span class="stoplight {stoplight}" title="probe quality">
        {stoplight}
      </span>
    {/if}
  </div>

  <div class="body">
    {#if error}
      <div class="empty err">error: {error}</div>
    {:else if loading}
      <div class="empty">loading…</div>
    {:else if !selected}
      <div class="empty">nothing selected</div>
    {:else if sortedLayers.length === 0}
      <div class="empty">no layer data for {selected}</div>
    {:else}
      <div class="bars">
        {#each sortedLayers as e (e.layer)}
          <div class="row">
            <span class="layer">L{e.layer}</span>
            <Bar value={e.magnitude} max={maxMagnitude || 1} width={280} height={8} />
            <span class="value">{e.magnitude.toFixed(3)}</span>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <footer class="drawer-footer">
    <span class="hint">
      Per-layer ‖baked‖.  Bar length encodes magnitude relative to the
      max layer for this concept.  Sources: registered steering vectors
      ∪ active probes.
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
    flex: 1 1 auto;
  }
  .picker-label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .picker select {
    background: var(--bg-alt);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: var(--space-1) var(--space-2);
    font: inherit;
    font-size: var(--text-sm);
    flex: 1 1 auto;
  }
  .picker select:focus {
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }
  .stoplight {
    font-size: var(--text-xs);
    text-transform: lowercase;
    letter-spacing: 0;
    padding: var(--space-1) var(--space-3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--fg-dim);
  }
  .stoplight.solid {
    color: var(--accent-green);
    border-color: var(--accent-green);
  }
  .stoplight.shaky {
    color: var(--accent-yellow);
    border-color: var(--accent-yellow);
  }
  .stoplight.poor {
    color: var(--accent-red);
    border-color: var(--accent-red);
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

  .bars {
    display: flex;
    flex-direction: column;
    gap: 1px;
    font-size: var(--text-xs);
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--space-4);
  }
  .layer {
    color: var(--fg-muted);
    width: 3em;
    text-align: right;
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .value {
    color: var(--fg-dim);
    width: 5em;
    text-align: right;
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
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
