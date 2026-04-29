<script lang="ts">
  // Shared list shell for the vector + probe picker drawers.  Renders a
  // search box, a scrollable filtered list of locally installed concepts,
  // and an "extract on the fly" affordance when the typed query doesn't
  // match any local row.
  //
  // Calls ``apiPacks.list()`` on mount so the rows have description /
  // tags / source — the shared store keeps name-only entries.  We keep
  // the full rows local to the picker so we don't bloat the global state
  // (mirrors PackDrawer's local-list approach).
  //
  // The picker reports user intent through three callbacks:
  //   - ``onPick(row)``: a row was clicked
  //   - ``onExtractFly(query)``: the typed query has no local match and
  //     the user clicked "extract on the fly"
  //   - ``onClear()`` is implicit — the parent owns close behavior.
  //
  // Visuals match the PackDrawer row idiom (mono row with name + source,
  // optional description, tag chips on the bottom row).

  import { onMount } from "svelte";
  import { ApiError, apiPacks } from "../lib/api";
  import type { LocalPackInfo } from "../lib/types";

  interface Props {
    /** Header label above the search box ("add steering vector" etc).
     * Decorative only — the parent owns the drawer header. */
    placeholder?: string;
    /** Disable the "extract on the fly" affordance — used by the probe
     * picker, where extracting a brand-new concept isn't the point. */
    allowExtractFly?: boolean;
    /** "Add to rack" / "Activate" — the primary verb shown on each row. */
    actionLabel: string;
    /** Optional secondary hint shown below the search box when no rows
     * match (e.g. "install a pack via Tools > Packs"). */
    emptyHint?: string;
    /** The submit callback.  Receives the picked row. */
    onPick: (row: LocalPackInfo) => void;
    /** When the user clicks "extract on the fly" with a typed query that
     * doesn't match any local row.  Receives the raw query string. */
    onExtractFly?: (query: string) => void;
    /** Names already in flight (showing a spinner on their row).  Empty
     * by default. */
    busy?: ReadonlySet<string>;
  }

  const {
    placeholder = "type a concept name…",
    allowExtractFly = true,
    actionLabel,
    emptyHint,
    onPick,
    onExtractFly,
    busy,
  }: Props = $props();

  let rows: LocalPackInfo[] = $state([]);
  let loading = $state(false);
  let error: string | null = $state(null);
  let query = $state("");
  let searchInputRef: HTMLInputElement | null = $state(null);

  async function load(): Promise<void> {
    loading = true;
    error = null;
    try {
      const r = await apiPacks.list();
      rows = (r.packs as unknown as LocalPackInfo[]) ?? [];
    } catch (e) {
      if (e instanceof ApiError) {
        error = `${e.status}: ${e.message}`;
      } else {
        error = e instanceof Error ? e.message : String(e);
      }
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    void load();
    queueMicrotask(() => searchInputRef?.focus());
  });

  function rowKey(r: LocalPackInfo): string {
    return `${r.namespace}/${r.name}`;
  }

  function rowMatches(r: LocalPackInfo, q: string): boolean {
    if (!q) return true;
    const n = q.toLowerCase();
    if (r.name.toLowerCase().includes(n)) return true;
    if (r.namespace.toLowerCase().includes(n)) return true;
    if (rowKey(r).toLowerCase().includes(n)) return true;
    if (r.description && r.description.toLowerCase().includes(n)) return true;
    if (Array.isArray(r.tags)) {
      for (const t of r.tags) {
        if (typeof t === "string" && t.toLowerCase().includes(n)) return true;
      }
    }
    return false;
  }

  // Filtered rows — case-insensitive substring match across name /
  // namespace / qualified selector / description / tags.
  const filtered = $derived.by(() => {
    const q = query.trim();
    return rows.filter((r) => rowMatches(r, q));
  });

  // True when the typed query has no exact-name match anywhere in the
  // local list — gates the "extract on the fly" affordance below the
  // search box.
  const hasExactMatch = $derived.by(() => {
    const q = query.trim();
    if (!q) return true;
    return rows.some(
      (r) =>
        r.name.toLowerCase() === q.toLowerCase() ||
        rowKey(r).toLowerCase() === q.toLowerCase(),
    );
  });

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Enter") {
      ev.preventDefault();
      const q = query.trim();
      if (!q) return;
      // Pick the first filtered row when one matches; else try the
      // extract-on-the-fly path if allowed.
      if (filtered.length > 0) {
        onPick(filtered[0]);
        return;
      }
      if (allowExtractFly && onExtractFly) {
        onExtractFly(q);
      }
    }
  }
</script>

<div class="picker">
  <div class="search-row">
    <input
      type="search"
      class="search"
      bind:this={searchInputRef}
      bind:value={query}
      placeholder={placeholder}
      autocomplete="off"
      spellcheck="false"
      onkeydown={onKeydown}
    />
    <button
      type="button"
      class="refresh"
      onclick={() => void load()}
      disabled={loading}
      title="re-fetch local packs"
      aria-label="Refresh"
    >{loading ? "…" : "↻"}</button>
  </div>

  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}

  {#if loading && rows.length === 0}
    <p class="muted">loading local packs…</p>
  {:else if rows.length === 0}
    <p class="muted">no packs installed locally{emptyHint ? ` — ${emptyHint}` : ""}.</p>
  {:else if filtered.length === 0}
    <div class="no-match">
      <p class="muted">no local match for "{query}".</p>
      {#if allowExtractFly && onExtractFly}
        <button
          type="button"
          class="fly"
          onclick={() => onExtractFly?.(query.trim())}
          disabled={!query.trim()}
          title="run extract — server short-circuits on cache hit"
        >
          extract '{query.trim() || "…"}' on the fly
        </button>
      {:else if emptyHint}
        <p class="hint">{emptyHint}</p>
      {/if}
    </div>
  {:else}
    <ul class="rows" role="listbox">
      {#each filtered as row (rowKey(row))}
        {@const sel = rowKey(row)}
        {@const inFlight = busy?.has(row.name) || busy?.has(sel) || false}
        <li>
          <button
            type="button"
            class="row"
            disabled={inFlight}
            onclick={() => onPick(row)}
            title="{actionLabel}: {sel}"
          >
            <div class="row-top">
              <span class="row-name">{row.name}</span>
              <span class="row-meta">
                <span class="row-ns">{row.namespace}</span>
                <span class="row-sep">·</span>
                <span class="row-source">{row.source ?? "—"}</span>
              </span>
            </div>
            {#if row.description}
              <p class="row-desc">{row.description}</p>
            {/if}
            <div class="row-bot">
              {#if Array.isArray(row.tags) && row.tags.length}
                <span class="chips">
                  {#each row.tags.slice(0, 6) as tag (tag)}
                    <span class="chip">{tag}</span>
                  {/each}
                </span>
              {/if}
              <span class="row-action">
                {inFlight ? "…" : actionLabel}
              </span>
            </div>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .picker {
    display: flex;
    flex-direction: column;
    gap: 0.5em;
    min-height: 0;
    flex: 1 1 auto;
  }
  .search-row {
    display: flex;
    gap: 0.4em;
    align-items: stretch;
  }
  .search {
    flex: 1 1 auto;
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.45em 0.6em;
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
  }
  .search:focus {
    outline: 1px solid var(--accent-blue);
    border-color: var(--accent-blue);
  }
  .refresh {
    flex: 0 0 auto;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-dim);
    padding: 0 0.7em;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
    cursor: pointer;
  }
  .refresh:hover:not(:disabled) {
    border-color: var(--fg-muted);
    color: var(--fg);
  }
  .refresh:disabled {
    opacity: 0.5;
    cursor: progress;
  }

  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.35em;
    overflow-y: auto;
    min-height: 0;
    flex: 1 1 auto;
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
  .row:hover:not(:disabled) {
    border-color: var(--accent-blue);
    background: var(--bg-alt);
  }
  .row:disabled {
    opacity: 0.55;
    cursor: progress;
  }
  .row-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.6em;
  }
  .row-name {
    color: var(--accent-green);
    font-weight: 600;
  }
  .row-meta {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    display: inline-flex;
    gap: 0.35em;
  }
  .row-sep {
    color: var(--fg-muted);
  }
  .row-ns,
  .row-source {
    color: var(--fg-dim);
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
  .chips {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 0.25em;
  }
  .chip {
    background: var(--bg-alt);
    color: var(--fg-dim);
    border: 1px solid var(--border-dim);
    border-radius: 999px;
    padding: 0.05em 0.55em;
    font-size: var(--font-size-tiny);
  }
  .row-action {
    color: var(--accent-blue);
    font-size: var(--font-size-small);
    text-transform: lowercase;
    letter-spacing: 0.04em;
  }

  .no-match {
    display: flex;
    flex-direction: column;
    gap: 0.4em;
    padding: 0.5em 0.2em;
  }
  .fly {
    background: transparent;
    color: var(--accent-blue);
    border: 1px solid var(--accent-blue);
    border-radius: 3px;
    padding: 0.4em 0.7em;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    cursor: pointer;
    align-self: flex-start;
  }
  .fly:hover:not(:disabled) {
    background: rgba(88, 166, 255, 0.12);
  }
  .fly:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .muted {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    margin: 0;
  }
  .hint {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
    margin: 0;
  }
  .error {
    color: var(--accent-error);
    font-size: var(--font-size-small);
    margin: 0;
    word-break: break-word;
  }
</style>
