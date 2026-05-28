<script lang="ts">
  // Manifold-probe rack — peer to ProbeRack, reading-side counterpart to
  // manifold steering.  Each row is one attached probe (see
  // ManifoldProbeStrip).  The header carries an attach form: a small
  // selector input + optional ``name`` override + optional ``top_n``, with
  // a one-click "rack this" shortcut per fitted catalog entry so the
  // common case (attach circumplex / personas straight from the catalog)
  // stays single-click.
  //
  // The whole rack hides itself when the server is too old to expose the
  // manifold-probe routes (``manifoldProbeRack.unavailable``) — the
  // InspectorPanel reads that flag and skips this section entirely.

  import ManifoldProbeStrip from "./ManifoldProbeStrip.svelte";
  import Select from "../lib/Select.svelte";
  import NumberInput from "../lib/NumberInput.svelte";
  import {
    attachManifoldProbe,
    manifoldProbeRack,
    manifoldRack,
    refreshManifoldList,
    refreshManifoldProbeList,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import { onMount } from "svelte";
  import type { ManifoldInfo } from "../lib/types";

  // Local form state.  Selector free-text so the user can type a bare
  // slug or a ns/name not yet in the catalog; the catalog dropdown is a
  // convenience shortcut that fills the selector + immediately attaches.
  let selector = $state("");
  let aliasName = $state("");
  let topN: number | null = $state(3);
  let attaching = $state(false);

  // Sorted view — alphabetical by name for stable order under live
  // attach/detach.
  const sorted = $derived.by(() => {
    return [...manifoldProbeRack.entries.keys()].sort((a, b) =>
      a.localeCompare(b),
    );
  });
  const count = $derived(sorted.length);

  // Catalog options for the "attach from catalog" select — only fitted
  // manifolds are valid attach targets (the engine requires a fitted
  // tensor to read).  Empty when the server has no manifolds installed.
  const catalogOptions = $derived.by(() => {
    const out: { value: string; label: string }[] = [];
    for (const m of manifoldRack.catalog) {
      if (!m.fitted_for_session) continue;
      const key = `${m.namespace}/${m.name}`;
      out.push({ value: key, label: key });
    }
    return out;
  });

  onMount(() => {
    // Be defensive — InspectorPanel might mount before bootstrap's first
    // catalog fetch lands.  Cheap on subsequent mounts (404 / empty).
    void refreshManifoldList();
    void refreshManifoldProbeList();
  });

  async function doAttach(
    sel: string,
    opts: { name?: string; top_n?: number } = {},
  ): Promise<void> {
    if (!sel.trim()) {
      pushToast("selector required", { kind: "error" });
      return;
    }
    attaching = true;
    try {
      const info = await attachManifoldProbe(sel.trim(), opts);
      pushToast(`attached manifold probe ${info.name}`, { kind: "info" });
      selector = "";
      aliasName = "";
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      pushToast(`attach failed — ${msg}`, { kind: "error", ttlMs: null });
    } finally {
      attaching = false;
    }
  }

  function onCatalogPick(value: string): void {
    if (!value) return;
    void doAttach(value, topN ? { top_n: topN } : {});
  }

  function onFormSubmit(ev: SubmitEvent): void {
    ev.preventDefault();
    const opts: { name?: string; top_n?: number } = {};
    if (aliasName.trim()) opts.name = aliasName.trim();
    if (topN && topN > 0) opts.top_n = topN;
    void doAttach(selector, opts);
  }

  function isAlreadyAttached(m: ManifoldInfo): boolean {
    const key = `${m.namespace}/${m.name}`;
    return (
      manifoldProbeRack.entries.has(key) || manifoldProbeRack.entries.has(m.name)
    );
  }

  /** Catalog-picker options minus probes already attached, with a
   *  leading "pick a manifold to attach…" placeholder. */
  const pickerOptions = $derived.by(() => {
    const out = [{ value: "", label: "pick a manifold to attach…" }];
    for (const m of manifoldRack.catalog) {
      if (!m.fitted_for_session) continue;
      const key = `${m.namespace}/${m.name}`;
      if (isAlreadyAttached(m)) continue;
      out.push({ value: key, label: key });
    }
    return out;
  });
</script>

{#if !manifoldProbeRack.unavailable}
  <section class="rack" aria-label="Manifold probe rack">
    <header class="header">
      <div class="header-text">
        <span class="title">MANIFOLD PROBES</span>
      </div>
      <span class="count" aria-live="polite">
        {count} attached
      </span>
    </header>

    <div class="strips" class:is-empty={count === 0} role="list">
      {#if count === 0}
        <div class="empty">
          <p class="empty-copy">
            Manifold probes read the inferred ``(coords, fraction,
            nearest)`` of generation against a fitted steering manifold.
            Attach one to watch.
          </p>
        </div>
      {:else}
        {#each sorted as probeName (probeName)}
          <div role="listitem">
            <ManifoldProbeStrip name={probeName} />
          </div>
        {/each}
      {/if}
    </div>

    <form class="attach" onsubmit={onFormSubmit}>
      {#if catalogOptions.length > 0}
        <div class="row picker-row">
          <span class="row-label">catalog</span>
          <span class="picker">
            <Select
              value=""
              options={pickerOptions}
              onchange={onCatalogPick}
              ariaLabel="Attach a manifold probe from the catalog"
              disabled={attaching}
            />
          </span>
        </div>
      {/if}

      <div class="row selector-row">
        <input
          type="text"
          class="text-input selector"
          placeholder="selector (ns/name)"
          aria-label="Manifold selector"
          bind:value={selector}
          disabled={attaching}
        />
        <input
          type="text"
          class="text-input alias"
          placeholder="name (optional)"
          aria-label="Registered probe name (optional)"
          bind:value={aliasName}
          disabled={attaching}
        />
      </div>

      <div class="row submit-row">
        <label class="top-n">
          <span class="row-label">top_n</span>
          <NumberInput
            value={topN}
            min={1}
            max={32}
            step={1}
            onchange={(v) => (topN = typeof v === "number" ? v : null)}
            ariaLabel="Per-token nearest-node list length"
          />
        </label>
        <button
          type="submit"
          class="add"
          disabled={attaching || !selector.trim()}
          title="Attach by selector"
        >+ attach</button>
      </div>

      {#if manifoldProbeRack.error}
        <p class="error" role="alert">{manifoldProbeRack.error}</p>
      {/if}
    </form>
  </section>
{/if}

<style>
  /* Same flat-section anatomy as ProbeRack — no own border, no own bg.
   * Fixed chrome + one scrollable middle.  The hairline above this rack
   * is owned by the SteeringRack's border-bottom; this rack owns its
   * own border-top so the InspectorPanel can stack three sections
   * cleanly. */
  .rack {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-5);
    background: transparent;
    border-top: 1px solid var(--border);
    min-height: 0;
    max-height: 100%;
    overflow: hidden;
  }
  .header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: var(--space-3);
  }
  .header-text {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .title {
    font-weight: var(--weight-bold);
    color: var(--accent-purple);
    font-size: var(--text-sm);
    letter-spacing: 0;
    text-transform: uppercase;
  }
  .count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    flex: 0 0 auto;
  }

  .strips {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    flex: 1 1 0;
    min-height: 2.4rem;
    max-height: 100%;
    overflow-y: auto;
    padding-right: var(--space-1);
  }
  .strips.is-empty {
    align-items: center;
    justify-content: center;
  }
  .empty {
    padding: var(--space-4) var(--space-4);
    text-align: center;
  }
  .empty-copy {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.5;
    max-width: 32ch;
  }

  .attach {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    border-top: 1px solid var(--border);
    padding-top: var(--space-3);
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
  }
  .picker-row {
    align-items: baseline;
  }
  .picker {
    flex: 1 1 auto;
    min-width: 0;
    display: inline-flex;
  }
  .row-label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    flex: 0 0 4em;
  }
  .text-input {
    flex: 1 1 0;
    min-width: 0;
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  .text-input:focus-visible {
    outline: 1px solid var(--accent-purple);
    outline-offset: -1px;
  }
  .text-input:disabled {
    opacity: 0.6;
  }
  .submit-row {
    justify-content: space-between;
  }
  .top-n {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .add {
    flex: 0 0 auto;
    background: rgba(167, 139, 250, 0.10);
    color: var(--accent-purple);
    border: 1px solid var(--border);
    padding: var(--space-2) var(--space-5);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .add:hover:not(:disabled) {
    background: rgba(167, 139, 250, 0.18);
  }
  .add:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .error {
    color: var(--accent-error);
    font-size: var(--text-xs);
    margin: 0;
  }
</style>
