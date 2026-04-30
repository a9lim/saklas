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

  // Sorted-by-layer view of ``data.layers``.  The server already sorts
  // ascending but we re-sort defensively because cheap.
  const sortedLayers = $derived<{ layer: number; magnitude: number }[]>(
    [...(data?.layers ?? [])].sort((a, b) => a.layer - b.layer),
  );

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

  .picker-row {
    display: flex;
    align-items: center;
    gap: 0.6em;
    padding: 0.5em 0.8em;
    border-bottom: 1px dashed var(--border-dim);
  }
  .picker {
    display: flex;
    align-items: center;
    gap: 0.5em;
    flex: 1 1 auto;
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
  .stoplight {
    font-size: var(--font-size-tiny);
    text-transform: lowercase;
    letter-spacing: 0.08em;
    padding: 0.1em 0.5em;
    border: 1px solid var(--border);
    border-radius: 2px;
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

  .bars {
    display: flex;
    flex-direction: column;
    gap: 1px;
    font-size: var(--font-size-tiny);
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.6em;
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
    border-top: 1px solid var(--border-dim);
    padding: 0.4em 0.8em;
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
  }
  .hint {
    line-height: 1.4;
  }
</style>
