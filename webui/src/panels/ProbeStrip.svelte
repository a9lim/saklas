<script lang="ts">
  // One row in the probe rack — radio + name + sparkline + value bar +
  // ✕.  Clicking the strip body (anywhere outside the radio / ✕) toggles
  // an inline WHY histogram below the row.  The histogram lazy-fetches
  // /vectors/{name}/diagnostics on first expand and falls back to a
  // client-side bucketize over the cached profile if the endpoint 404s
  // (probe registered without a steering profile).
  //
  // Mirrors saklas/tui/trait_panel.py — same sparkline + bar visual
  // rhythm, same WHY footer in HIST_BUCKETS=16 buckets.

  import Bar from "../lib/charts/Bar.svelte";
  import Sparkline from "../lib/charts/Sparkline.svelte";
  import Histogram from "../lib/charts/Histogram.svelte";
  import { bucketize, type HistogramBucket } from "../lib/charts";
  import { apiVectors, ApiError } from "../lib/api";
  import {
    deactivateProbe,
    highlightState,
    probeRack,
    setHighlightTarget,
    vectorRack,
  } from "../lib/stores.svelte";
  import type { VectorDiagnosticsResponse } from "../lib/types";

  interface Props {
    name: string;
  }

  let { name }: Props = $props();

  // Live entry view — re-reads from the rack on every paint so live
  // sparkline updates from updateProbeFromScores propagate.
  const entry = $derived(probeRack.entries.get(name));
  const current = $derived(entry?.current ?? 0);
  const sparkline = $derived(entry?.sparkline ?? []);
  const isHighlight = $derived(highlightState.target === name);

  // Expander + WHY state.  The diagnostics fetch is one-shot per probe
  // life — once successful, we cache and never re-issue.  A failure
  // (404 or network) flips ``loadError`` and the render falls back to
  // client-side bucketize.
  let expanded = $state(false);
  let diagnostics = $state<VectorDiagnosticsResponse | null>(null);
  let loading = $state(false);
  let loadError = $state<string | null>(null);

  async function loadDiagnostics(): Promise<void> {
    if (diagnostics !== null || loading) return;
    loading = true;
    loadError = null;
    try {
      diagnostics = await apiVectors.diagnostics(name);
    } catch (e) {
      // 404 is the common branch (probe registered without a registered
      // steering profile by that name).  Lump every failure into a
      // single fallback path — the client-side bucketize over the
      // cached vectorRack profile still gives a usable histogram.
      if (e instanceof ApiError) {
        loadError = `${e.status}`;
      } else {
        loadError = e instanceof Error ? e.message : String(e);
      }
    } finally {
      loading = false;
    }
  }

  // The rendered histogram buckets — server's pre-bucketed list when
  // diagnostics landed; client-side bucketize() over the cached profile
  // otherwise.  Returns [] when neither source has data, which the
  // <Histogram> component renders as "no data".
  const buckets = $derived.by<HistogramBucket[]>(() => {
    if (diagnostics?.histogram?.length) {
      return diagnostics.histogram.map((b) => ({
        lo: b.lo,
        hi: b.hi,
        value: b.value,
        label: b.label,
      }));
    }
    // Two fallback sources: the diagnostics response's per_layer_norms
    // (when the server returned the response but with empty histogram —
    // shouldn't happen but cheap to handle) and the cached profile from
    // vectorRack.profiles (populated by refreshVectorList).
    const norms =
      diagnostics?.per_layer_norms ??
      vectorRack.profiles.get(name)?.per_layer_norms;
    if (!norms) return [];
    return bucketize(norms);
  });

  function onSelectHighlight(ev: Event): void {
    ev.stopPropagation();
    setHighlightTarget(name);
  }

  function onRemove(ev: MouseEvent): void {
    ev.stopPropagation();
    void deactivateProbe(name);
  }

  function onToggleExpand(): void {
    expanded = !expanded;
    if (expanded) void loadDiagnostics();
  }

  // Keyboard a11y — Enter/Space on the body toggles, matching the
  // mouse click.  The radio + remove button keep their native semantics.
  function onBodyKey(ev: KeyboardEvent): void {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      onToggleExpand();
    }
  }
</script>

<div class="strip" class:selected={isHighlight} class:expanded>
  <div
    class="row"
    role="button"
    tabindex="0"
    aria-expanded={expanded}
    aria-label="Toggle WHY histogram for {name}"
    onclick={onToggleExpand}
    onkeydown={onBodyKey}
  >
    <label class="radio" title="Highlight {name} on chat tokens">
      <input
        type="radio"
        name="probe-highlight"
        checked={isHighlight}
        value={name}
        onclick={onSelectHighlight}
        onchange={onSelectHighlight}
        aria-label="Set {name} as highlight target"
      />
      <span class="radio-glyph" aria-hidden="true">{isHighlight ? "●" : "○"}</span>
    </label>

    <span class="name" title={name}>{name}</span>

    <Sparkline points={sparkline} width={56} height={14} />

    <Bar value={current} max={1} width={96} height={8} />

    <span class="value" class:pos={current > 0} class:neg={current < 0}>
      {current >= 0 ? "+" : ""}{current.toFixed(2)}
    </span>

    <button
      type="button"
      class="remove"
      aria-label="Remove probe {name}"
      title="Remove probe"
      onclick={onRemove}
    >✕</button>
  </div>

  {#if expanded}
    <div class="why" aria-label="WHY histogram for {name}">
      {#if loading}
        <div class="why-status">loading…</div>
      {:else if buckets.length === 0}
        <div class="why-status">
          {loadError ? `no diagnostics (${loadError})` : "no data"}
        </div>
      {:else}
        {#if loadError}
          <div class="why-status fallback">
            client-side bucketize ({loadError})
          </div>
        {/if}
        <Histogram {buckets} barWidth={140} barHeight={6} />
      {/if}
    </div>
  {/if}
</div>

<style>
  .strip {
    border: 1px solid var(--border-dim);
    border-radius: 3px;
    background: var(--bg-alt);
    transition: border-color 0.1s ease;
  }
  .strip.selected {
    border-color: var(--accent-blue);
  }
  .strip.expanded {
    background: var(--bg);
  }

  .row {
    display: grid;
    grid-template-columns: auto 1fr auto auto auto auto;
    align-items: center;
    gap: var(--row-gap);
    height: 40px;
    padding: 0 0.6em;
    cursor: pointer;
    user-select: none;
  }
  .row:hover {
    background: var(--bg-elev);
  }
  .row:focus-visible {
    outline: 1px solid var(--accent-blue);
    outline-offset: -1px;
  }

  .radio {
    display: inline-flex;
    align-items: center;
    cursor: pointer;
    width: 1.2em;
  }
  .radio input {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
  }
  .radio-glyph {
    color: var(--fg-dim);
    font-size: 0.95em;
    line-height: 1;
  }
  .strip.selected .radio-glyph {
    color: var(--accent-blue);
  }

  .name {
    color: var(--fg-strong);
    font-size: 0.9em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .strip.selected .name {
    color: var(--accent-blue);
    font-weight: bold;
  }

  .value {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    font-variant-numeric: tabular-nums;
    min-width: 3.5em;
    text-align: right;
  }
  .value.pos {
    color: var(--accent-green);
  }
  .value.neg {
    color: var(--accent-red);
  }

  .remove {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: 0.85em;
    line-height: 1;
    padding: 0.2em 0.3em;
    border-radius: 2px;
  }
  .remove:hover {
    color: var(--accent-red);
    background: var(--bg-elev);
  }

  .why {
    padding: 0.4em 0.6em 0.6em 0.6em;
    border-top: 1px solid var(--border-dim);
    /* Total expanded height ~140px: 40 row + ~100 histogram block. */
    max-height: 110px;
    overflow-y: auto;
  }
  .why-status {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    padding: 0.2em 0;
  }
  .why-status.fallback {
    color: var(--accent-yellow);
  }
</style>
