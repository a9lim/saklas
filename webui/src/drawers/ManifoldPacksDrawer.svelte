<script lang="ts">
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // ManifoldPacksDrawer — manifold-side counterpart to PackDrawer.
  //
  // Two tabs at the top: "Installed" lists what saklas knows locally
  // (proxied through GET /saklas/v1/manifolds); "Search HF" hits
  // GET /saklas/v1/manifolds/search with a debounced query and offers
  // an Install button per row that POSTs the install request and
  // refreshes the local list on success.
  //
  // Pair-drawer with ManifoldDrawer: that one is the *workspace* surface
  // (per-row steer / probe / fit / delete on the active session); this
  // one is the *catalog* surface (list local, browse HF, install).
  // Mirrors how VectorsDrawer and PackDrawer split the same concerns
  // for steering vectors.
  //
  // Reachable from the workspace rail's "manifolds → packs…" entry,
  // parallel to "vectors → packs…".

  import { onMount } from "svelte";
  import { ApiError, apiManifolds } from "../lib/api";
  import {
    closeDrawer,
    steerRack,
    refreshManifoldList,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { ManifoldInfo, RemoteManifoldInfo } from "../lib/types";

  type Tab = "installed" | "search";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  let tab: Tab = $state("installed");

  // ----- search-tab state ----------------------------------------------
  let query = $state("");
  let searchResults: RemoteManifoldInfo[] = $state([]);
  let searchLoading = $state(false);
  let searchError: string | null = $state(null);
  let installing: string | null = $state(null);

  // Debounce: redo searches 300ms after the user stops typing — mirrors
  // PackDrawer's cadence so the two feel identical.
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  function scheduleSearch(): void {
    if (debounceTimer) clearTimeout(debounceTimer);
    const q = query.trim();
    if (!q) {
      searchResults = [];
      searchError = null;
      searchLoading = false;
      return;
    }
    searchLoading = true;
    debounceTimer = setTimeout(() => {
      void runSearch(q);
    }, 300);
  }

  async function runSearch(q: string): Promise<void> {
    try {
      const r = await apiManifolds.search(q, 20);
      searchResults = r.results ?? [];
      searchError = null;
    } catch (e) {
      searchResults = [];
      if (e instanceof ApiError) {
        if (e.status === 503) {
          searchError =
            "huggingface_hub isn't installed on the server. Run `pip install -e \".[serve]\"` and restart.";
        } else if (e.status === 502) {
          searchError = `HF transport error: ${e.message}`;
        } else {
          searchError = e.message;
        }
      } else {
        searchError = e instanceof Error ? e.message : String(e);
      }
    } finally {
      searchLoading = false;
    }
  }

  async function installRow(row: RemoteManifoldInfo): Promise<void> {
    const target = `${row.namespace}/${row.name}`;
    installing = target;
    try {
      await apiManifolds.install({ target });
      await refreshManifoldList();
      tab = "installed";
      pushToast(`installed manifold ${target}`, { kind: "info" });
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.status === 503) {
          pushToast(
            "huggingface_hub isn't installed on the server",
            { kind: "error", ttlMs: null },
          );
        } else if (e.status === 502) {
          pushToast(`HF transport error: ${e.message}`, {
            kind: "error", ttlMs: null,
          });
        } else if (e.status === 409) {
          pushToast(`${target} is already installed`, {
            kind: "error", ttlMs: null,
          });
        } else {
          pushToast(`install '${target}' failed — ${e.message}`, {
            kind: "error", ttlMs: null,
          });
        }
      } else {
        const msg = e instanceof Error ? e.message : String(e);
        pushToast(`install '${target}' failed — ${msg}`, {
          kind: "error", ttlMs: null,
        });
      }
    } finally {
      installing = null;
    }
  }

  // ----- installed-tab state -------------------------------------------
  function selectorOf(row: ManifoldInfo | RemoteManifoldInfo): string {
    return `${row.namespace}/${row.name}`;
  }
  function fitBadge(row: ManifoldInfo | RemoteManifoldInfo): string | null {
    if (row.fit_mode && row.fit_mode !== "authored") return row.fit_mode;
    return null;
  }

  // The store keeps the local catalog hot — refresh on mount in case
  // we landed here without having visited ManifoldDrawer first.
  onMount(() => {
    void refreshManifoldList();
  });
</script>

<div class="drawer-shell">
  <header class="header">
    <span class="title">packs</span>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="tabs" role="tablist">
    <button
      type="button"
      role="tab"
      aria-selected={tab === "installed"}
      class:active={tab === "installed"}
      onclick={() => (tab = "installed")}
    >installed</button>
    <button
      type="button"
      role="tab"
      aria-selected={tab === "search"}
      class:active={tab === "search"}
      onclick={() => (tab = "search")}
    >hf</button>
  </div>

  <div class="body">
    {#if tab === "installed"}
      {#if steerRack.loading && steerRack.catalog.length === 0}
        <p class="muted">loading manifolds…</p>
      {:else if steerRack.catalog.length === 0}
        <p class="muted">
          none installed
        </p>
      {:else}
        <ul class="rows" role="list">
          {#each steerRack.catalog as m (selectorOf(m))}
            {@const key = selectorOf(m)}
            {@const badge = fitBadge(m)}
            <li class="row" title={m.description || key}>
              <div class="meta">
                <span class="row-name">{key}</span>
                <span class="row-sub">
                  {m.domain_label} · {m.node_count} nodes
                  {#if badge}
                    <span class="fit-badge fit-{badge}">{badge}</span>
                  {/if}
                  {#if m.fitted_for_session}
                    <span class="fit-tag">fitted</span>
                  {/if}
                  {#if m.stale}
                    <span class="stale">stale</span>
                  {/if}
                </span>
              </div>
            </li>
          {/each}
        </ul>
      {/if}
    {:else}
      <div class="search">
        <label class="search-label">
          <span class="vh">search query</span>
          <input
            type="search"
            placeholder="search HF…"
            aria-label="Search HF for saklas-manifold repos"
            bind:value={query}
            oninput={scheduleSearch}
          />
        </label>
        {#if !query.trim()}
          <p class="muted">
            <code>saklas-manifold</code> repos
          </p>
        {:else if searchLoading}
          <p class="muted">searching…</p>
        {:else if searchError}
          <p class="error" role="alert">{searchError}</p>
        {:else if searchResults.length === 0}
          <p class="muted">no matches</p>
        {:else}
          <ul class="rows" role="list">
            {#each searchResults as row (selectorOf(row))}
              {@const target = selectorOf(row)}
              {@const inFlight = installing === target}
              {@const badge = fitBadge(row)}
              <li class="row" title={row.description || target}>
                <div class="meta">
                  <span class="row-name">{target}</span>
                  <span class="row-sub">
                    {row.domain_label} · {row.node_count} nodes
                    {#if badge}
                      <span class="fit-badge fit-{badge}">{badge}</span>
                    {/if}
                    {#if row.tensor_models.length > 0}
                      <span class="hf-fit-count">
                        · {row.tensor_models.length} fit{row.tensor_models.length === 1 ? "" : "s"}
                      </span>
                    {/if}
                  </span>
                </div>
                <div class="actions">
                  <button
                    type="button"
                    class="act install"
                    disabled={inFlight}
                    onclick={() => void installRow(row)}
                    title={`install ${target}`}
                  >{inFlight ? "…" : "install"}</button>
                </div>
              </li>
            {/each}
          </ul>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* Two-tab layout mirrors PackDrawer pixel-for-pixel so the two
   * pack browsers feel like one family — same tabs, same row look,
   * same colour rhythm.  Only the install action's accent goes
   * purple (the manifold family colour); the steering-vector side
   * keeps its blue. */
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-5) var(--space-6);
  }
  .title {
    color: var(--accent);
    letter-spacing: 0;
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }

  .tabs {
    display: flex;
    gap: 0;
    padding: 0 var(--space-5);
  }
  .tabs button {
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: var(--space-3) var(--space-4);
    color: var(--fg-dim);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition:
      color var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out);
  }
  .tabs button.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }

  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-5) var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
  }

  .search {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .search-label {
    display: flex;
    flex-direction: column;
  }
  .vh {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
  }
  .search input[type="search"] {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  .search input[type="search"]:focus-visible {
    outline: 1px solid var(--pillar-manifold);
    outline-offset: -1px;
  }

  .muted {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    margin: 0;
    line-height: 1.4;
  }
  .muted code {
    font-family: var(--font-mono);
    color: var(--fg-strong);
    font-size: var(--text-sm);
  }
  .error {
    margin: 0;
    color: var(--accent-red);
    font-size: var(--text-sm);
    word-break: break-word;
  }

  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .row {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: var(--space-3);
    background: var(--bg-deep);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    transition: background var(--dur) var(--ease-out);
  }
  .row:hover { background: color-mix(in srgb, var(--pillar-manifold) 8%, var(--bg-deep)); }
  .meta {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .row-name {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row-sub {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .fit-badge {
    display: inline-block;
    margin-left: var(--space-2);
    padding: 0 var(--space-2);
    border-radius: var(--radius);
    text-transform: uppercase;
    font-size: var(--text-2xs);
    letter-spacing: 0.04em;
    border: 1px solid transparent;
    color: var(--pillar-manifold);
    background: color-mix(in srgb, var(--pillar-manifold) 12%, transparent);
  }
  .fit-spectral {
    color: var(--accent);
    background: var(--accent-subtle);
  }
  .fit-tag {
    color: var(--accent-green);
    margin-left: var(--space-2);
    font-size: var(--text-2xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .stale {
    color: var(--accent-yellow);
    margin-left: var(--space-2);
  }
  .hf-fit-count {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }

  .actions {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .act {
    background: var(--glass);
    color: var(--pillar-manifold);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .act:hover:not(:disabled) {
    background: color-mix(in srgb, var(--pillar-manifold) 12%, transparent);
  }
  .act:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
</style>
