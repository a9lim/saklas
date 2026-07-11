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
  import Select from "../lib/Select.svelte";
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
      <span class="eyebrow">layer norms</span>
      <div class="name-row">
        {#if selected}
          <code class="name" title={selected}>{selected}</code>
          {#if data}
            <span class="meta">{data.total_layers} layers · model {data.model}</span>
          {/if}
        {:else}
          <span class="meta">
            {names.length === 0 ? "no vectors or probes registered" : "pick a name"}
          </span>
        {/if}
      </div>
    </div>
    <button type="button" class="close" onclick={onClose} aria-label="Close drawer">×</button>
  </header>

  <div class="toolbar">
    <label class="picker">
      <span class="picker-label">name</span>
      <Select
        bind:value={selected}
        options={names.length === 0
          ? [{ value: "", label: "(empty)" }]
          : names.map((n) => ({ value: n, label: n }))}
        disabled={names.length === 0}
        ariaLabel="Vector name"
      />
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
  /* v2 sheet interior — the host paints the sheet surface (glass hairline,
   * radius, --bg-alt fill), so the root is transparent; chrome speaks sans
   * and every value/identifier sits in mono. */
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

  .toolbar {
    display: flex;
    align-items: center;
    gap: var(--space-5);
    padding: var(--space-3) var(--space-6);
  }
  .picker {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex: 1 1 auto;
  }
  .picker-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  /* Themed Select fills its picker host; Select owns its own chrome. */
  .picker :global(.sk-select) {
    flex: 1 1 auto;
  }
  .stoplight {
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    text-transform: lowercase;
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    padding: 1px var(--space-4);
    color: var(--fg-dim);
  }
  .stoplight.solid {
    color: var(--accent-green);
    background: color-mix(in srgb, var(--accent-green) 16%, var(--glass));
  }
  .stoplight.shaky {
    color: var(--accent-yellow);
    background: color-mix(in srgb, var(--accent-yellow) 16%, var(--glass));
  }
  .stoplight.poor {
    color: var(--accent-red);
    background: color-mix(in srgb, var(--accent-red) 16%, var(--glass));
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

  .bars {
    display: flex;
    flex-direction: column;
    gap: 1px;
    font-family: var(--font-mono);
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
    padding: var(--space-3) var(--space-6);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .hint {
    line-height: 1.5;
  }
</style>
