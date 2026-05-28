<script lang="ts">
  // PackDrawer — browse locally installed vector packs and search HF
  // hub.  Pair-drawer with ManifoldPacksDrawer (the manifold-side
  // counterpart); the two share one layout — Convention-A header, the
  // same two-tab installed/search shell, single-column rows, and a
  // toast-driven install action.  Only the accent family differs
  // (vectors blue, manifolds purple).
  //
  // Two tabs: "Installed" lists what saklas knows locally (GET
  // /saklas/v1/packs); "Search HF" hits GET /saklas/v1/packs/search
  // with a debounced query and offers an Install button per row that
  // POSTs the install request and refreshes the local list on success.
  //
  // The shared store keeps only ``packsState.installed`` (name-only) —
  // this drawer fetches the full rows itself so it can render
  // description, source, tags, tensor count.

  import { onMount } from "svelte";
  import { ApiError, apiPacks } from "../lib/api";
  import {
    closeDrawer,
    refreshPacks,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";

  // The server returns rows shaped per saklas.io.cache_ops.ConceptRow /
  // HfRow.  Types.ts exports LocalPackInfo / RemotePackInfo with
  // ``[key: string]: unknown`` passthroughs — we narrow at the field
  // sites below since not every field is required.
  interface LocalRow {
    name: string;
    namespace: string;
    status?: string;
    recommended_alpha?: number;
    tags?: string[];
    description?: string;
    source?: string;
    tensor_models?: string[];
    error?: string;
  }
  interface HfRow {
    name: string;
    namespace: string;
    recommended_alpha?: number;
    tags?: string[];
    description?: string;
    tensor_models?: string[];
  }

  type Tab = "installed" | "search";

  // Drawer host forwards { params } — unused (drawer reads from store).
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  let tab: Tab = $state("installed");

  // ----- installed tab state -----
  let local: LocalRow[] = $state([]);
  let localLoading = $state(false);
  let localError: string | null = $state(null);

  async function loadInstalled(): Promise<void> {
    localLoading = true;
    localError = null;
    try {
      const r = await apiPacks.list();
      local = (r.packs as unknown as LocalRow[]) ?? [];
      // Mirror into the shared store so other panels stay in sync.
      void refreshPacks();
    } catch (e) {
      localError = e instanceof Error ? e.message : String(e);
    } finally {
      localLoading = false;
    }
  }

  // ----- search tab state -----
  let query = $state("");
  let searchResults: HfRow[] = $state([]);
  let searchLoading = $state(false);
  let searchError: string | null = $state(null);
  let installing: string | null = $state(null);

  // Debounce: redo searches 300ms after the user stops typing.
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
      const r = await apiPacks.search(q, 20);
      searchResults = (r.results as unknown as HfRow[]) ?? [];
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

  async function installRow(row: HfRow): Promise<void> {
    const target = `${row.namespace}/${row.name}`;
    installing = target;
    try {
      await apiPacks.install({ target });
      // Pull the fresh local list, then bounce to the installed tab so
      // the user sees the new row.
      await loadInstalled();
      tab = "installed";
      pushToast(`installed ${target}`, { kind: "info" });
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
          pushToast(
            `${target} is already installed (use force in the CLI to overwrite)`,
            { kind: "error", ttlMs: null },
          );
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

  function fileCount(row: LocalRow | HfRow): number {
    return Array.isArray(row.tensor_models) ? row.tensor_models.length : 0;
  }

  function selectorOf(row: LocalRow | HfRow): string {
    return `${row.namespace}/${row.name}`;
  }

  onMount(() => {
    // Fetch the full rows once on open so descriptions/tags/etc. render
    // — the shared store is name-only.  ``loadInstalled`` mirrors back
    // into it via ``refreshPacks``.
    void loadInstalled();
  });
</script>

<div class="drawer-shell">
  <header class="header">
    <span class="title">packs</span>
    <button
      type="button"
      class="close"
      aria-label="Close"
      onclick={closeDrawer}
    >✕</button>
  </header>

  <div class="tabs" role="tablist">
    <button
      type="button"
      role="tab"
      aria-selected={tab === "installed"}
      class:active={tab === "installed"}
      onclick={() => (tab = "installed")}
    >installed{local.length ? ` (${local.length})` : ""}</button>
    <button
      type="button"
      role="tab"
      aria-selected={tab === "search"}
      class:active={tab === "search"}
      onclick={() => (tab = "search")}
    >search hf</button>
  </div>

  <div class="body">
    {#if tab === "installed"}
      {#if localLoading && local.length === 0}
        <p class="muted">loading…</p>
      {:else if localError}
        <p class="error" role="alert">{localError}</p>
      {:else if local.length === 0}
        <p class="muted">
          no packs installed locally — search HF above, or use
          <strong>+ build vector</strong> from the vectors drawer.
        </p>
      {:else}
        <ul class="rows" role="list">
          {#each local as row (selectorOf(row))}
            {@const key = selectorOf(row)}
            <li class="row" title={row.description || key}>
              <div class="meta">
                <span class="row-name">{key}</span>
                <span class="row-sub">
                  {row.source ?? "—"} · {fileCount(row)} tensor{fileCount(row) === 1 ? "" : "s"}
                  {#if row.recommended_alpha !== undefined}
                    · α {row.recommended_alpha.toFixed(2)}
                  {/if}
                  {#if row.status && row.status !== "installed"}
                    <span class="status">{row.status}</span>
                  {/if}
                  {#if row.tags && row.tags.length}
                    · {row.tags.join(", ")}
                  {/if}
                  {#if row.error}
                    <span class="row-error">{row.error}</span>
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
            placeholder="lying, persona, owner/name…"
            aria-label="Search HF for saklas-pack repos"
            bind:value={query}
            oninput={scheduleSearch}
          />
        </label>
        {#if !query.trim()}
          <p class="muted">type to search the HF hub.</p>
        {:else if searchLoading}
          <p class="muted">searching hf hub…</p>
        {:else if searchError}
          <p class="error" role="alert">{searchError}</p>
        {:else if searchResults.length === 0}
          <p class="muted">no results for "{query}".</p>
        {:else}
          <ul class="rows" role="list">
            {#each searchResults as row (selectorOf(row))}
              {@const target = selectorOf(row)}
              {@const inFlight = installing === target}
              <li class="row" title={row.description || target}>
                <div class="meta">
                  <span class="row-name">{target}</span>
                  <span class="row-sub">
                    {fileCount(row)} tensor{fileCount(row) === 1 ? "" : "s"}
                    {#if row.recommended_alpha !== undefined}
                      · α {row.recommended_alpha.toFixed(2)}
                    {/if}
                    {#if row.tags && row.tags.length}
                      · {row.tags.join(", ")}
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
  /* Layout mirrors ManifoldPacksDrawer so the two pack browsers read as
   * one family — same header, tabs, row look, and toast-driven install.
   * Only the accent goes blue here (the steering-vector family colour);
   * the manifold side keeps purple. */
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-4) var(--space-5);
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent);
    letter-spacing: 0;
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    font-size: var(--text);
    line-height: 1;
    padding: var(--space-2) var(--space-3);
    cursor: pointer;
    transition: color var(--dur) var(--ease-out);
  }
  .close:hover { color: var(--accent-red); }

  .tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
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
    padding: var(--space-4) var(--space-5) var(--space-5);
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
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  .search input[type="search"]:focus-visible {
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }

  .muted {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    margin: 0;
    line-height: 1.4;
  }
  .muted strong {
    color: var(--accent);
    font-weight: var(--weight-medium);
  }
  .error {
    margin: 0;
    color: var(--accent-error);
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
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    transition: border-color var(--dur) var(--ease-out);
  }
  .row:hover { border-color: var(--accent); }
  .meta {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .row-name {
    color: var(--fg-strong);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row-sub {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    word-break: break-word;
  }
  .status {
    color: var(--accent-yellow);
    margin-left: var(--space-2);
  }
  .row-error {
    color: var(--accent-error);
    margin-left: var(--space-2);
  }

  .actions {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .act {
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .act:hover:not(:disabled) {
    background: var(--accent-subtle);
    border-color: var(--accent);
  }
  .act:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
</style>
