<script lang="ts">
  // PackDrawer — browse locally installed packs and search HF hub.
  //
  // Two tabs at the top: "Installed" lists what saklas knows locally
  // (proxied through GET /saklas/v1/packs); "Search HF" hits
  // GET /saklas/v1/packs/search with a debounced query and offers
  // an Install button per row that POSTs the install request and
  // refreshes the local list on success.
  //
  // The store keeps only ``packsState.installed`` (list of namespace/name
  // strings) — this drawer fetches the full pack rows itself so it can
  // render description, source, tags, tensor count without bloating the
  // shared store.

  import { onMount } from "svelte";
  import { ApiError, apiPacks } from "../lib/api";
  import {
    closeDrawer,
    packsState,
    refreshPacks,
  } from "../lib/stores.svelte";

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
  let selected: LocalRow | null = $state(null);

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
  let installError: string | null = $state(null);
  let installNotice: string | null = $state(null);

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
            "huggingface_hub not installed on the server — `pip install -e \".[serve]\"` and restart.";
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
    installError = null;
    installNotice = null;
    try {
      await apiPacks.install({ target });
      // Pull the fresh local list, then bounce to the installed tab so
      // the user sees the new row.
      await loadInstalled();
      tab = "installed";
      installNotice = `installed ${target}`;
      // Try to highlight the freshly-installed row.
      const just = local.find(
        (p) => p.namespace === row.namespace && p.name === row.name,
      );
      if (just) selected = just;
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.status === 503) {
          installError =
            "huggingface_hub not installed on the server — `pip install -e \".[serve]\"` and restart.";
        } else if (e.status === 502) {
          installError = `HF transport error: ${e.message}`;
        } else if (e.status === 409) {
          installError = `already installed (use force in the CLI to overwrite): ${e.message}`;
        } else {
          installError = e.message;
        }
      } else {
        installError = e instanceof Error ? e.message : String(e);
      }
    } finally {
      installing = null;
    }
  }

  function fileCount(row: LocalRow): number {
    return Array.isArray(row.tensor_models) ? row.tensor_models.length : 0;
  }

  function selectorOf(row: LocalRow | HfRow): string {
    return `${row.namespace}/${row.name}`;
  }

  onMount(() => {
    if (packsState.installed.length === 0) {
      void loadInstalled();
    } else {
      // Even when the store has names, fetch the full rows once on open
      // so descriptions/tags/etc. render — the store is name-only.
      void loadInstalled();
    }
  });
</script>

<div class="drawer-shell">
  <header class="head">
    <h2>packs</h2>
    <button
      type="button"
      class="close"
      aria-label="Close drawer"
      onclick={closeDrawer}>✕</button
    >
  </header>

  <div class="tabs" role="tablist">
    <button
      type="button"
      role="tab"
      aria-selected={tab === "installed"}
      class:active={tab === "installed"}
      onclick={() => (tab = "installed")}
    >
      installed{local.length ? ` (${local.length})` : ""}
    </button>
    <button
      type="button"
      role="tab"
      aria-selected={tab === "search"}
      class:active={tab === "search"}
      onclick={() => (tab = "search")}
    >
      search hf
    </button>
  </div>

  <div class="body">
    {#if tab === "installed"}
      <div class="installed">
        <div class="list-pane">
          {#if localLoading}
            <p class="muted">loading…</p>
          {:else if localError}
            <p class="error">{localError}</p>
          {:else if local.length === 0}
            <p class="muted">no packs installed locally.</p>
          {:else}
            <ul class="rows" role="listbox">
              {#each local as row (selectorOf(row))}
                {@const sel = selectorOf(row)}
                <li>
                  <button
                    type="button"
                    class="row"
                    class:selected={selected &&
                      selectorOf(selected) === sel}
                    onclick={() => (selected = row)}
                  >
                    <div class="row-top">
                      <span class="row-name">{sel}</span>
                      <span class="row-source" title="source"
                        >{row.source ?? "—"}</span
                      >
                    </div>
                    {#if row.description}
                      <p class="row-desc">{row.description}</p>
                    {/if}
                    <div class="row-bot">
                      <span class="row-files">
                        {fileCount(row)} tensor{fileCount(row) === 1 ? "" : "s"}
                      </span>
                      {#if row.status && row.status !== "installed"}
                        <span class="row-status">{row.status}</span>
                      {/if}
                    </div>
                  </button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
        <aside class="preview" aria-label="Pack preview">
          {#if selected}
            <h3>{selectorOf(selected)}</h3>
            {#if selected.description}
              <p class="preview-desc">{selected.description}</p>
            {/if}
            <dl>
              <dt>source</dt>
              <dd>{selected.source ?? "—"}</dd>
              <dt>status</dt>
              <dd>{selected.status ?? "—"}</dd>
              <dt>recommended α</dt>
              <dd>
                {selected.recommended_alpha !== undefined
                  ? selected.recommended_alpha.toFixed(2)
                  : "—"}
              </dd>
              <dt>tags</dt>
              <dd>
                {#if selected.tags && selected.tags.length}
                  {selected.tags.join(", ")}
                {:else}
                  —
                {/if}
              </dd>
              <dt>tensor models</dt>
              <dd>
                {#if selected.tensor_models && selected.tensor_models.length}
                  <ul class="model-list">
                    {#each selected.tensor_models as m}
                      <li>{m}</li>
                    {/each}
                  </ul>
                {:else}
                  none baked
                {/if}
              </dd>
              {#if selected.error}
                <dt>error</dt>
                <dd class="error">{selected.error}</dd>
              {/if}
            </dl>
          {:else}
            <p class="muted">select a pack to preview metadata.</p>
          {/if}
        </aside>
      </div>
    {:else}
      <div class="search">
        <label class="query">
          <span class="vh">search query</span>
          <input
            type="search"
            placeholder="lying, persona, owner/name…"
            bind:value={query}
            oninput={scheduleSearch}
          />
        </label>
        {#if installNotice}
          <p class="notice">{installNotice}</p>
        {/if}
        {#if installError}
          <p class="error">{installError}</p>
        {/if}
        {#if !query.trim()}
          <p class="muted">type to search the HF hub.</p>
        {:else if searchLoading}
          <p class="muted">searching hf hub…</p>
        {:else if searchError}
          <p class="error">{searchError}</p>
        {:else if searchResults.length === 0}
          <p class="muted">no results for "{query}".</p>
        {:else}
          <ul class="rows">
            {#each searchResults as row (selectorOf(row))}
              {@const sel = selectorOf(row)}
              <li class="row search-row">
                <div class="row-top">
                  <span class="row-name">{sel}</span>
                  {#if row.recommended_alpha !== undefined}
                    <span class="row-alpha"
                      >α {row.recommended_alpha.toFixed(2)}</span
                    >
                  {/if}
                </div>
                {#if row.description}
                  <p class="row-desc">{row.description}</p>
                {/if}
                <div class="row-bot">
                  <span class="row-files">
                    {Array.isArray(row.tensor_models)
                      ? row.tensor_models.length
                      : 0} tensor{Array.isArray(row.tensor_models) &&
                    row.tensor_models.length === 1
                      ? ""
                      : "s"}
                  </span>
                  {#if row.tags && row.tags.length}
                    <span class="row-tags">{row.tags.join(", ")}</span>
                  {/if}
                  <button
                    type="button"
                    class="install"
                    disabled={installing !== null}
                    onclick={() => installRow(row)}
                  >
                    {#if installing === sel}
                      <span class="spinner" aria-hidden="true"></span>
                      installing…
                    {:else}
                      install
                    {/if}
                  </button>
                </div>
              </li>
            {/each}
          </ul>
        {/if}
      </div>
    {/if}
  </div>

  <footer class="foot">
    <button type="button" class="secondary" onclick={closeDrawer}>close</button>
  </footer>
</div>

<style>
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6em 1em;
    border-bottom: 1px solid var(--border);
  }
  .head h2 {
    margin: 0;
    font-size: 1em;
    color: var(--accent-blue);
    letter-spacing: 0.04em;
    text-transform: lowercase;
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    font-size: 1em;
    line-height: 1;
    padding: 0.25em 0.4em;
    cursor: pointer;
  }
  .close:hover {
    color: var(--accent-red);
  }

  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
    background: var(--bg-deep);
  }
  .tabs button {
    flex: 1;
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    padding: 0.5em 1em;
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }
  .tabs button:hover {
    color: var(--fg);
  }
  .tabs button.active {
    color: var(--accent-blue);
    border-bottom-color: var(--accent-blue);
    background: var(--bg-alt);
  }

  .body {
    flex: 1;
    overflow-y: auto;
    padding: 0.6em 1em;
    min-height: 0;
  }

  .installed {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
    gap: 0.8em;
    height: 100%;
  }
  .list-pane,
  .preview {
    min-height: 0;
    overflow-y: auto;
  }
  .preview {
    background: var(--bg-deep);
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    padding: 0.6em 0.8em;
  }
  .preview h3 {
    margin: 0 0 0.3em 0;
    font-size: 0.95em;
    color: var(--accent-green);
  }
  .preview-desc {
    color: var(--fg-strong);
    margin: 0 0 0.6em 0;
    font-size: 0.9em;
  }
  .preview dl {
    margin: 0;
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.2em 0.7em;
    font-size: var(--font-size-small);
  }
  .preview dt {
    color: var(--fg-muted);
    text-transform: lowercase;
  }
  .preview dd {
    color: var(--fg);
    margin: 0;
    word-break: break-word;
  }
  .model-list {
    margin: 0;
    padding-left: 1em;
  }

  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.4em;
  }
  .row {
    display: block;
    width: 100%;
    text-align: left;
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    padding: 0.5em 0.7em;
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
    cursor: pointer;
  }
  .row:hover {
    border-color: var(--border);
    background: var(--bg-alt);
  }
  .row.selected {
    border-color: var(--accent-blue);
    background: var(--bg-alt);
  }
  .row-top {
    display: flex;
    justify-content: space-between;
    gap: 0.6em;
  }
  .row-name {
    color: var(--accent-green);
    font-weight: 600;
  }
  .row-source,
  .row-alpha,
  .row-status,
  .row-tags,
  .row-files {
    color: var(--fg-dim);
    font-size: var(--font-size-small);
  }
  .row-desc {
    color: var(--fg-strong);
    margin: 0.3em 0 0 0;
    font-size: 0.9em;
  }
  .row-bot {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5em;
    margin-top: 0.4em;
    flex-wrap: wrap;
  }
  .row-status {
    color: var(--accent-yellow);
  }

  .search {
    display: flex;
    flex-direction: column;
    gap: 0.6em;
  }
  .query {
    display: block;
  }
  .query input {
    width: 100%;
    box-sizing: border-box;
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.45em 0.6em;
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
  }
  .query input:focus {
    outline: 1px solid var(--accent-blue);
    border-color: var(--accent-blue);
  }
  .search-row {
    cursor: default;
  }
  .install {
    background: transparent;
    color: var(--accent-blue);
    border: 1px solid var(--accent-blue);
    padding: 0.25em 0.7em;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
  }
  .install:hover:not(:disabled) {
    background: rgba(88, 166, 255, 0.12);
  }
  .install:disabled {
    opacity: 0.55;
    cursor: progress;
  }
  .spinner {
    width: 0.7em;
    height: 0.7em;
    border-radius: 50%;
    border: 1.5px solid var(--accent-blue);
    border-right-color: transparent;
    animation: spin 0.7s linear infinite;
    display: inline-block;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .foot {
    border-top: 1px solid var(--border);
    padding: 0.5em 1em;
    display: flex;
    justify-content: flex-end;
    gap: 0.5em;
  }
  .secondary {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-dim);
    padding: 0.35em 0.9em;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    cursor: pointer;
  }
  .secondary:hover {
    border-color: var(--fg);
    color: var(--fg);
  }

  .muted {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    margin: 0;
  }
  .notice {
    color: var(--accent-green);
    font-size: var(--font-size-small);
    margin: 0;
  }
  .error {
    color: var(--accent-error);
    font-size: var(--font-size-small);
    margin: 0;
    word-break: break-word;
  }
  .vh {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
