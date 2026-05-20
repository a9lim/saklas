<script lang="ts">
  // Shared catalog list for the steering + probe pickers.  Presents the
  // installed concepts the way the catalog is actually organised — grouped
  // into the seven fixed categories, each row framed as its bipolar axis
  // (``calm ◄──●──► angry``) with a non-interactive slider preview parked
  // at the concept's recommended α.
  //
  // See docs/plans/webui-overhaul.md §"The picker".  Calls
  // ``apiPacks.list()`` on mount for the description / tags / source rows
  // (the shared store keeps name-only entries).
  //
  // ``query`` is bindable so the parent can react to the no-match case —
  // the steering picker pre-fills its custom-extraction name field with
  // whatever the user typed when nothing in the catalog matches.

  import { onMount } from "svelte";
  import { SvelteSet, SvelteMap } from "svelte/reactivity";
  import { ApiError, apiPacks } from "../lib/api";
  import Slider from "../lib/Slider.svelte";
  import type { LocalPackInfo } from "../lib/types";
  import {
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    DEFAULT_EXPANDED,
    categoryOf,
    polesOf,
    recommendedAlpha,
    type Category,
  } from "../lib/concepts";

  interface Props {
    /** Search-box placeholder. */
    placeholder?: string;
    /** Primary verb shown on each row ("add" / "watch"). */
    actionLabel: string;
    /** Show the per-row strength slider + α readout.  False for the
     * probe picker — a probe observes, it has no steering strength. */
    showStrength?: boolean;
    /** Secondary hint shown when no packs are installed at all. */
    emptyHint?: string;
    /** Row clicked — receives the row and its recommended α. */
    onPick: (row: LocalPackInfo, alpha: number) => void;
    /** Names already in flight (spinner on their row). */
    busy?: ReadonlySet<string>;
    /** Current search string — bindable so the parent can read it. */
    query?: string;
    /** Number of catalog rows matching ``query`` — bindable readout so the
     * steering picker can flow the no-match case into custom extraction. */
    matchCount?: number;
    /** Own the vertical scroll internally (probe picker).  False lets the
     * parent scroll the whole body (steering picker, where the catalog
     * shares space with the custom-extraction section). */
    scroll?: boolean;
    /** Focus the search box on mount.  Default true. */
    autofocusSearch?: boolean;
  }

  let {
    placeholder = "search concepts…",
    actionLabel,
    showStrength = true,
    emptyHint,
    onPick,
    busy,
    query = $bindable(""),
    matchCount = $bindable(0),
    scroll = true,
    autofocusSearch = true,
  }: Props = $props();

  let rows: LocalPackInfo[] = $state([]);
  let loading = $state(false);
  let error: string | null = $state(null);
  let searchInputRef: HTMLInputElement | null = $state(null);

  // Per-category expansion.  Searching overrides this — every section with
  // a match renders open so a query never hides a hit behind a collapsed
  // header.  State is intentionally not persisted (cheap to re-open).
  const expanded = new SvelteSet<Category>(DEFAULT_EXPANDED);

  async function load(): Promise<void> {
    loading = true;
    error = null;
    try {
      const r = await apiPacks.list();
      rows = (r.packs as unknown as LocalPackInfo[]) ?? [];
    } catch (e) {
      error = e instanceof ApiError ? `${e.status}: ${e.message}`
        : e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    void load();
    // ``preventScroll`` keeps the browser from scrolling the drawer's
    // ancestors to bring the input into view — the drawer enters mid-
    // animation at ``translateX(100%)``, so without this the focus call
    // bumps the layout's ``scrollLeft`` and the chat/rack content visibly
    // shifts left then snaps back when the slide-in completes.
    if (autofocusSearch)
      queueMicrotask(() => searchInputRef?.focus({ preventScroll: true }));
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

  const searching = $derived(query.trim().length > 0);

  // Filtered rows grouped by category, each group name-sorted.  Iterated
  // in the fixed CATEGORY_ORDER (+ "other" last) at render time.
  const grouped = $derived.by(() => {
    const q = query.trim();
    const m = new Map<Category, LocalPackInfo[]>();
    for (const r of rows) {
      if (!rowMatches(r, q)) continue;
      const c = categoryOf(r.tags);
      const list = m.get(c);
      if (list) list.push(r);
      else m.set(c, [r]);
    }
    for (const list of m.values()) {
      list.sort((a, b) => a.name.localeCompare(b.name));
    }
    return m;
  });

  const sections = $derived.by(() => {
    const order: Category[] = [...CATEGORY_ORDER, "other"];
    return order
      .filter((c) => (grouped.get(c)?.length ?? 0) > 0)
      .map((c) => ({ cat: c, items: grouped.get(c) as LocalPackInfo[] }));
  });

  const totalMatches = $derived(
    sections.reduce((n, s) => n + s.items.length, 0),
  );

  // Mirror the match count out to the bindable so the parent can react.
  $effect(() => {
    matchCount = totalMatches;
  });

  function toggle(cat: Category): void {
    if (expanded.has(cat)) expanded.delete(cat);
    else expanded.add(cat);
  }

  function isOpen(cat: Category): boolean {
    return searching || expanded.has(cat);
  }

  // Per-row draft α — the user can set the strength on the slider
  // *before* committing, then [add] drops it on the rack at that value.
  // Lazily seeded from the pack's recommended α on first read.
  const drafts = new SvelteMap<string, number>();

  function draftAlpha(row: LocalPackInfo): number {
    const key = rowKey(row);
    const d = drafts.get(key);
    return d ?? recommendedAlpha(row);
  }

  function setDraft(row: LocalPackInfo, v: number): void {
    drafts.set(rowKey(row), v);
  }

  function formatAlpha(a: number): string {
    if (a === 0) return "0.00";
    return `${a > 0 ? "+" : "-"}${Math.abs(a).toFixed(2)}`;
  }

  function alphaColor(a: number): string {
    if (a > 0) return "var(--accent-green)";
    if (a < 0) return "var(--accent-red)";
    return "var(--fg-muted)";
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key !== "Enter") return;
    ev.preventDefault();
    // Enter adds the single match when the query narrows to exactly one.
    if (totalMatches === 1) {
      const r = sections[0].items[0];
      onPick(r, draftAlpha(r));
    }
  }
</script>

<div class="picker" class:flow={!scroll}>
  <div class="search-row">
    <input
      type="search"
      class="search"
      bind:this={searchInputRef}
      bind:value={query}
      {placeholder}
      autocomplete="off"
      spellcheck="false"
      onkeydown={onKeydown}
    />
    <button
      type="button"
      class="refresh"
      onclick={() => void load()}
      disabled={loading}
      title="re-fetch installed packs"
      aria-label="Refresh"
    >{loading ? "…" : "↻"}</button>
  </div>

  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}

  {#if loading && rows.length === 0}
    <p class="muted">loading concepts…</p>
  {:else if rows.length === 0}
    <p class="muted">
      no packs installed locally{emptyHint ? `, ${emptyHint}` : ""}.
    </p>
  {:else if sections.length === 0}
    <p class="muted">no concept matches "{query.trim()}".</p>
  {:else}
    <div class="catalog">
      {#each sections as { cat, items } (cat)}
        {@const open = isOpen(cat)}
        <section class="category">
          <button
            type="button"
            class="cat-header"
            class:open
            aria-expanded={open}
            onclick={() => toggle(cat)}
            disabled={searching}
          >
            <span class="caret" aria-hidden="true">{open ? "▾" : "▸"}</span>
            <span class="cat-name">{CATEGORY_LABELS[cat]}</span>
            <span class="cat-count">{items.length}</span>
          </button>

          {#if open}
            <ul class="rows" role="list" aria-label={CATEGORY_LABELS[cat]}>
              {#each items as row (rowKey(row))}
                {@const sel = rowKey(row)}
                {@const inFlight =
                  busy?.has(row.name) || busy?.has(sel) || false}
                {@const poles = polesOf(row.name)}
                {@const mono = poles.negative === null}
                {@const a = draftAlpha(row)}
                <li
                  class="row"
                  class:compact={!showStrength}
                  title={row.description
                    ? `${sel}: ${row.description}`
                    : sel}
                >
                  {#if showStrength}
                    <span class="pole neg">{poles.negative ?? ""}</span>
                    <Slider
                      value={a}
                      min={mono ? 0 : -1}
                      max={1}
                      step={0.05}
                      disabled={inFlight}
                      ariaLabel="strength for {row.name}"
                      oninput={(v) => setDraft(row, v)}
                    />
                    <span class="pole pos">{poles.positive}</span>
                    <span class="alpha" style:color={alphaColor(a)}>
                      {formatAlpha(a)}
                    </span>
                  {:else}
                    <span class="concept">
                      {#if !mono}
                        <span class="pole neg">{poles.negative}</span>
                        <span class="axis-sep" aria-hidden="true">↔</span>
                      {/if}
                      <span class="pole pos">{poles.positive}</span>
                    </span>
                  {/if}
                  <button
                    type="button"
                    class="add-btn"
                    disabled={inFlight}
                    onclick={() => onPick(row, a)}
                    title={showStrength
                      ? `${actionLabel} ${sel} at α ${formatAlpha(a)}`
                      : `${actionLabel} ${sel}`}
                  >
                    {inFlight ? "…" : actionLabel}
                  </button>
                </li>
              {/each}
            </ul>
          {/if}
        </section>
      {/each}
    </div>
  {/if}
</div>

<style>
  .picker {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    min-height: 0;
    flex: 1 1 auto;
  }
  .search-row {
    display: flex;
    gap: var(--space-3);
    align-items: stretch;
  }
  .search {
    flex: 1 1 auto;
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    font-family: var(--font-mono);
    font-size: var(--text);
    transition: border-color var(--dur) var(--ease-out);
  }
  .search:focus {
    outline: none;
    border-color: var(--accent);
  }
  .refresh {
    flex: 0 0 auto;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-dim);
    padding: 0 var(--space-4);
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: var(--text);
    cursor: pointer;
    transition: color var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out);
  }
  .refresh:hover:not(:disabled) {
    border-color: var(--fg-muted);
    color: var(--fg);
  }
  .refresh:disabled {
    opacity: 0.5;
    cursor: progress;
  }

  .picker.flow {
    flex: 0 0 auto;
  }
  .catalog {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    overflow-y: auto;
    min-height: 0;
    flex: 1 1 auto;
    padding-right: var(--space-1);
  }
  /* Parent-scroll mode: let the catalog grow to its natural height. */
  .picker.flow .catalog {
    overflow: visible;
    flex: 0 0 auto;
  }
  .category {
    display: flex;
    flex-direction: column;
  }

  .cat-header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    width: 100%;
    text-align: left;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--border);
    padding: var(--space-3) var(--space-1) var(--space-2);
    color: var(--fg-muted);
    cursor: pointer;
    transition: color var(--dur) var(--ease-out);
  }
  .cat-header:hover:not(:disabled) {
    color: var(--fg-strong);
  }
  .cat-header:disabled {
    cursor: default;
  }
  .caret {
    font-size: var(--text-xs);
    color: var(--fg-muted);
  }
  .cat-name {
    flex: 1 1 auto;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
  }
  .cat-header.open .cat-name {
    color: var(--accent);
  }
  .cat-count {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
  }

  .rows {
    list-style: none;
    margin: 0;
    padding: var(--space-2) 0 var(--space-2);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  /* Pole · slider · pole · α readout · add.  The bipolar axis frames the
   * row; the slider is live so the strength is set before committing. */
  .row {
    display: grid;
    grid-template-columns:
      minmax(2.6em, 1fr) minmax(70px, 2.4fr) minmax(2.6em, 1fr) auto auto;
    align-items: center;
    gap: var(--space-3);
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    font-family: var(--font-mono);
    font-size: var(--text);
    transition: border-color var(--dur) var(--ease-out);
  }
  .row:hover {
    border-color: var(--accent);
  }
  /* Probe picker — no strength slider, just the concept axis + action. */
  .row.compact {
    grid-template-columns: 1fr auto;
  }
  .concept {
    display: inline-flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .axis-sep {
    color: var(--fg-muted);
    flex: 0 0 auto;
  }

  .pole {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: var(--text-sm);
  }
  .pole.neg {
    color: var(--fg-muted);
    text-align: right;
  }
  .pole.pos {
    color: var(--fg-strong);
  }

  .alpha {
    font-variant-numeric: tabular-nums;
    font-size: var(--text-sm);
    min-width: 3.4em;
    text-align: right;
  }

  .add-btn {
    background: var(--accent-subtle);
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .add-btn:hover:not(:disabled) {
    background: var(--accent-glow);
  }
  .add-btn:disabled {
    opacity: 0.55;
    cursor: progress;
  }

  .muted {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    margin: 0;
  }
  .error {
    color: var(--accent-error);
    font-size: var(--text-sm);
    margin: 0;
    word-break: break-word;
  }
</style>
