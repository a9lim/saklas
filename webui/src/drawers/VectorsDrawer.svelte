<script lang="ts">
  // Unified vector management drawer — replaces VectorPickerDrawer +
  // ProbePickerDrawer.  One surface, two sections split on the
  // server-supplied ``has_tensor`` flag:
  //
  //   * Extracted      — packs with a baked tensor for the loaded
  //                      model.  Per row: [+steer] [+probe] [delete].
  //                      Steer/probe are pure-add buttons (no
  //                      toggling); when the concept is already in
  //                      the rack / active as a probe the button
  //                      disables.  Removal happens through the
  //                      rack strip's own ✕ (mirrors the TUI's
  //                      asymmetry — picker adds, rack manages).
  //   * Statements only — packs with statements + scenarios on disk
  //                      but no tensor for this model.  Per row:
  //                      [extract] [delete].  Extract reuses the
  //                      cached statements — one-click action, no
  //                      form needed.
  //
  // The "+ custom vector" button at the top opens ExtractDrawer for
  // concepts the catalog doesn't carry at all.  Delete uses a 2-step
  // confirm (first click flips to "confirm?", second commits; auto-
  // resets after ~3s).

  import { onMount } from "svelte";
  import { SvelteSet } from "svelte/reactivity";
  import { ApiError, apiExtractStream, apiPacks } from "../lib/api";
  import type { ExtractRequest } from "../lib/types";
  import {
    activateProbe,
    addVectorToRack,
    closeDrawer,
    openDrawer,
    packsState,
    probeRack,
    refreshPacks,
    refreshVectorList,
    vectorRack,
  } from "../lib/stores.svelte";
  import {
    dismissToast,
    pushToast,
    updateToast,
  } from "../lib/stores/toasts.svelte";
  import {
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    DEFAULT_EXPANDED,
    categoryOf,
    polesOf,
    recommendedAlpha,
    type Category,
  } from "../lib/concepts";
  import type { LocalPackInfo } from "../lib/types";

  let { params: _params }: { params?: unknown } = $props();
  $effect(() => { void _params; });

  let query = $state("");
  let errorMsg: string | null = $state(null);
  let searchInputRef: HTMLInputElement | null = $state(null);

  // Per-row in-flight + delete-confirm state.  Both are name-keyed
  // (``ns/name``).  ``confirmTimers`` is a plain Map — timer ids are
  // not reactive state.
  const busyKeys = new SvelteSet<string>();
  const confirmKeys = new SvelteSet<string>();
  const confirmTimers = new Map<string, number>();
  // Per-row inspect open state — parallels ManifoldDrawer's inspector.
  // Toggling ⓘ reveals a small in-row metadata + "deep diagnostics"
  // affordance without committing to opening LayerNormsDrawer.
  const inspectKeys = new SvelteSet<string>();

  onMount(() => {
    void refreshPacks();
    queueMicrotask(() => searchInputRef?.focus({ preventScroll: true }));
    return () => {
      // Cancel any pending confirm timers — drawer unmounting means
      // no one is watching anymore.
      for (const t of confirmTimers.values()) window.clearTimeout(t);
      confirmTimers.clear();
    };
  });

  // ----- helpers ------------------------------------------------------

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

  function describeError(e: unknown): string {
    if (e instanceof ApiError) {
      const detail =
        e.body && typeof e.body === "object" && "detail" in (e.body as object)
          ? String((e.body as { detail: unknown }).detail)
          : e.message;
      return `${e.status}: ${detail}`;
    }
    return e instanceof Error ? e.message : String(e);
  }

  // ----- grouping -----------------------------------------------------

  const searching = $derived(query.trim().length > 0);

  const filtered = $derived(
    packsState.infos.filter((r) => rowMatches(r, query.trim())),
  );
  const extracted = $derived(filtered.filter((r) => r.has_tensor === true));
  const statementsOnly = $derived(
    filtered.filter((r) => r.has_tensor !== true),
  );

  // Section-prefixed category expansion ("ex:affect" / "st:affect") so
  // both sections track open state independently.
  const expanded = new SvelteSet<string>(
    [...DEFAULT_EXPANDED].flatMap((c) => [`ex:${c}`, `st:${c}`]),
  );

  function groupedByCategory(rows: LocalPackInfo[]): Map<Category, LocalPackInfo[]> {
    const m = new Map<Category, LocalPackInfo[]>();
    for (const r of rows) {
      const c = categoryOf(r.tags);
      const list = m.get(c);
      if (list) list.push(r);
      else m.set(c, [r]);
    }
    for (const list of m.values()) {
      list.sort((a, b) => a.name.localeCompare(b.name));
    }
    return m;
  }

  function sectionsOf(g: Map<Category, LocalPackInfo[]>) {
    const order: Category[] = [...CATEGORY_ORDER, "other"];
    return order
      .filter((c) => (g.get(c)?.length ?? 0) > 0)
      .map((c) => ({ cat: c, items: g.get(c) as LocalPackInfo[] }));
  }

  const exSections = $derived(sectionsOf(groupedByCategory(extracted)));
  const stSections = $derived(sectionsOf(groupedByCategory(statementsOnly)));

  function toggle(key: string): void {
    if (expanded.has(key)) expanded.delete(key);
    else expanded.add(key);
  }
  function isOpen(key: string): boolean {
    return searching || expanded.has(key);
  }

  // ----- per-row state ------------------------------------------------

  function isInRack(name: string): boolean {
    return vectorRack.entries.has(name);
  }
  function isActiveProbe(name: string): boolean {
    return probeRack.active.includes(name);
  }

  function setBusy(key: string, busy: boolean): void {
    if (busy) busyKeys.add(key);
    else busyKeys.delete(key);
  }

  // ----- actions ------------------------------------------------------

  async function onSteer(row: LocalPackInfo): Promise<void> {
    const name = row.name;
    // Pure-add: clicking an already-racked concept is a no-op.  The
    // button is disabled in that state, but the no-op here keeps the
    // handler defensive against double-clicks during the in-flight
    // window before the rack store updates.
    if (isInRack(name)) return;
    const key = rowKey(row);
    setBusy(key, true);
    errorMsg = null;
    try {
      // Ensure the profile is registered server-side.  Tensor-on-disk
      // doesn't mean it's loaded in the session; the extract endpoint
      // short-circuits on cache hit (cheap profile load).
      await apiExtractStream({ name, register: true }, () => {});
      await refreshVectorList();
      addVectorToRack(name, recommendedAlpha(row));
    } catch (e) {
      errorMsg = describeError(e);
    } finally {
      setBusy(key, false);
    }
  }

  async function onProbe(row: LocalPackInfo): Promise<void> {
    const name = row.name;
    if (isActiveProbe(name)) return;
    const key = rowKey(row);
    setBusy(key, true);
    errorMsg = null;
    try {
      await apiExtractStream({ name, register: true }, () => {});
      await activateProbe(name);
    } catch (e) {
      errorMsg = describeError(e);
    } finally {
      setBusy(key, false);
    }
  }

  /** Statements-only extract.  The concept has a name + cached
   *  pairs/scenarios; this kicks off the same toast-driven flow as
   *  the custom-vector form, no user input required.
   *
   *  ``force=true`` is the re-fit path: the row already has a tensor
   *  for this model, but the user wants to bypass the tensor cache
   *  and re-run the contrastive forward passes (parity with the
   *  manifold drawer's "re-fit" action).  The toast wording branches
   *  so the user sees "re-extracting" vs "extracting". */
  function onExtract(row: LocalPackInfo, opts: { force?: boolean } = {}): void {
    const name = row.name;
    const key = rowKey(row);
    setBusy(key, true);
    const verb = opts.force ? "re-extracting" : "extracting";
    const toastId = pushToast(`${verb} '${name}'…`, {
      kind: "info",
      ttlMs: null,
    });
    const req: ExtractRequest = { name, register: true };
    if (opts.force) req.force = true;
    void (async () => {
      try {
        const result = await apiExtractStream(
          req,
          (ev) => {
            if (ev.event === "progress") {
              const m =
                ev.data && typeof ev.data === "object"
                  ? (ev.data as { message?: string }).message
                  : null;
              if (m) updateToast(toastId, { detail: m });
            }
          },
        );
        // Refresh both so the row jumps from "Statements only" up to
        // "Extracted" without remount, and the rack autocomplete sees
        // the newly-registered profile.
        await Promise.all([refreshVectorList(), refreshPacks()]);
        dismissToast(toastId);
        const past = opts.force ? "re-extracted" : "extracted";
        pushToast(`${past} ${result.canonical}`, { kind: "info" });
      } catch (e) {
        dismissToast(toastId);
        const verbed = opts.force ? "re-extract" : "extract";
        pushToast(`${verbed} '${name}' failed — ${describeError(e)}`, {
          kind: "error",
          ttlMs: null,
        });
      } finally {
        setBusy(key, false);
      }
    })();
  }

  /** Two-step delete confirm: first click flips the button to
   *  "confirm?", second click within 3 s commits.  Auto-resets via
   *  timer so a stray click doesn't leave the button armed forever. */
  function onDeleteClick(row: LocalPackInfo): void {
    const key = rowKey(row);
    if (confirmKeys.has(key)) {
      const t = confirmTimers.get(key);
      if (t !== undefined) {
        window.clearTimeout(t);
        confirmTimers.delete(key);
      }
      confirmKeys.delete(key);
      void doDelete(row);
      return;
    }
    confirmKeys.add(key);
    const t = window.setTimeout(() => {
      confirmKeys.delete(key);
      confirmTimers.delete(key);
    }, 3000);
    confirmTimers.set(key, t);
  }

  async function doDelete(row: LocalPackInfo): Promise<void> {
    const key = rowKey(row);
    setBusy(key, true);
    errorMsg = null;
    try {
      const r = await apiPacks.delete(row.namespace, row.name);
      // Server already detached the concept from the session for us.
      await Promise.all([refreshPacks(), refreshVectorList()]);
      const msg = r.rematerializes_on_restart
        ? `deleted ${row.namespace}/${row.name} — bundled, respawns on restart`
        : `deleted ${row.namespace}/${row.name}`;
      pushToast(msg, { kind: "info" });
    } catch (e) {
      errorMsg = describeError(e);
    } finally {
      setBusy(key, false);
    }
  }

  function gotoCustom(): void {
    openDrawer("extract", { seed_a: query.trim() || undefined });
  }

  /** Toggle the per-row inspect body.  Light by design — the deep
   *  diagnostics + layer-norms view lives one click further in via the
   *  LayerNormsDrawer, matching ManifoldDrawer's "ⓘ + drill-down"
   *  split. */
  function toggleInspect(row: LocalPackInfo): void {
    const key = rowKey(row);
    if (inspectKeys.has(key)) inspectKeys.delete(key);
    else inspectKeys.add(key);
  }

  /** Open the layer-norms drawer pre-selected on this concept.  Used
   *  by the inspect body's "show layer norms →" link as the bridge
   *  from the cheap inline panel to the heavier per-layer diagnostics
   *  view. */
  function gotoLayerNorms(row: LocalPackInfo): void {
    openDrawer("layer_norms", { name: row.name });
  }

  /** Tensor-models field is a loose pass-through on LocalPackInfo —
   *  returns the list of safe-model-id stems this concept is extracted
   *  for.  We surface it in the inspect body so the user sees which
   *  other models the tensor is available on. */
  function tensorModels(row: LocalPackInfo): string[] {
    const raw = (row as { tensor_models?: unknown }).tensor_models;
    if (!Array.isArray(raw)) return [];
    return raw.filter((m): m is string => typeof m === "string");
  }
</script>

<section class="drawer-shell" aria-label="Vectors">
  <header class="header">
    <span class="title">vectors</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button>
  </header>

  <div class="body">
    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}

    <button type="button" class="custom-btn" onclick={gotoCustom}>
      <span class="plus" aria-hidden="true">+</span>
      <span class="custom-label">custom vector</span>
      <span class="custom-hint">extract a concept the catalog doesn't carry</span>
    </button>

    <div class="search-row">
      <input
        type="search"
        class="search"
        bind:this={searchInputRef}
        bind:value={query}
        placeholder="search concepts…"
        autocomplete="off"
        spellcheck="false"
      />
      <button
        type="button"
        class="refresh"
        onclick={() => void refreshPacks()}
        disabled={packsState.loading}
        title="re-fetch installed packs"
        aria-label="Refresh"
      >{packsState.loading ? "…" : "↻"}</button>
    </div>

    {#if packsState.error}
      <p class="error" role="alert">{packsState.error}</p>
    {/if}

    {#if packsState.loading && packsState.infos.length === 0}
      <p class="muted">loading concepts…</p>
    {:else if packsState.infos.length === 0}
      <p class="muted">
        no packs installed locally — use + custom vector above, or
        install one via the rail › vectors › packs.
      </p>
    {:else if filtered.length === 0}
      <p class="muted">no concept matches "{query.trim()}".</p>
    {:else}
      {#if exSections.length > 0}
        <h2 class="section-title">Extracted</h2>
        <div class="catalog">
          {#each exSections as { cat, items } (cat)}
            {@const key = `ex:${cat}`}
            {@const open = isOpen(key)}
            <section class="category">
              <button
                type="button"
                class="cat-header"
                class:open
                aria-expanded={open}
                onclick={() => toggle(key)}
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
                    {@const inFlight = busyKeys.has(sel)}
                    {@const inRack = isInRack(row.name)}
                    {@const active = isActiveProbe(row.name)}
                    {@const poles = polesOf(row.name)}
                    {@const mono = poles.negative === null}
                    {@const confirming = confirmKeys.has(sel)}
                    {@const inspecting = inspectKeys.has(sel)}
                    {@const models = tensorModels(row)}
                    <li class="row" title={row.description || sel}>
                      <div class="row-line">
                        <span class="concept">
                          {#if !mono}
                            <span class="pole neg">{poles.negative}</span>
                            <span class="axis-sep" aria-hidden="true">↔</span>
                          {/if}
                          <span class="pole pos">{poles.positive}</span>
                        </span>
                        <div class="actions">
                          <button
                            type="button"
                            class="act inspect"
                            aria-expanded={inspecting}
                            onclick={() => toggleInspect(row)}
                            title={inspecting ? "hide details" : "inspect pack metadata"}
                          >{inspecting ? "▾" : "ⓘ"}</button>
                          <button
                            type="button"
                            class="act steer"
                            disabled={inFlight || inRack}
                            onclick={() => void onSteer(row)}
                            title={inRack
                              ? `${sel} is already in the steering rack — remove via the rack strip's ✕`
                              : `steer ${sel} at α ${recommendedAlpha(row).toFixed(2)}`}
                          >+steer</button>
                          <button
                            type="button"
                            class="act probe"
                            disabled={inFlight || active}
                            onclick={() => void onProbe(row)}
                            title={active
                              ? `${sel} is already an active probe — remove via the probe strip's ✕`
                              : `activate probe ${sel}`}
                          >+probe</button>
                          <button
                            type="button"
                            class="act refit"
                            disabled={inFlight}
                            onclick={() => onExtract(row, { force: true })}
                            title={`re-extract ${sel} for the current model (bypass tensor cache)`}
                          >{inFlight ? "…" : "re-extract"}</button>
                          <button
                            type="button"
                            class="act del"
                            class:confirm={confirming}
                            disabled={inFlight}
                            onclick={() => onDeleteClick(row)}
                            title={confirming
                              ? "click again to confirm"
                              : `delete ${sel}`}
                          >{confirming ? "confirm?" : "delete"}</button>
                        </div>
                      </div>
                      {#if inspecting}
                        <div class="inspect-body">
                          <dl class="meta-list">
                            <dt>description</dt>
                            <dd>{row.description || "—"}</dd>
                            <dt>source</dt>
                            <dd>{row.source}</dd>
                            <dt>recommended α</dt>
                            <dd>{recommendedAlpha(row).toFixed(2)}</dd>
                            {#if row.tags && row.tags.length > 0}
                              <dt>tags</dt>
                              <dd>{row.tags.join(", ")}</dd>
                            {/if}
                            {#if models.length > 0}
                              <dt>tensor models</dt>
                              <dd class="model-list">{models.join(", ")}</dd>
                            {/if}
                          </dl>
                          <button
                            type="button"
                            class="deep-link"
                            onclick={() => gotoLayerNorms(row)}
                            title="open the per-layer ||baked|| histogram + extraction diagnostics"
                          >show layer norms →</button>
                        </div>
                      {/if}
                    </li>
                  {/each}
                </ul>
              {/if}
            </section>
          {/each}
        </div>
      {/if}

      {#if stSections.length > 0}
        <h2 class="section-title">
          Statements only
          <span class="section-hint">no tensor for this model yet</span>
        </h2>
        <div class="catalog">
          {#each stSections as { cat, items } (cat)}
            {@const key = `st:${cat}`}
            {@const open = isOpen(key)}
            <section class="category">
              <button
                type="button"
                class="cat-header"
                class:open
                aria-expanded={open}
                onclick={() => toggle(key)}
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
                    {@const inFlight = busyKeys.has(sel)}
                    {@const poles = polesOf(row.name)}
                    {@const mono = poles.negative === null}
                    {@const confirming = confirmKeys.has(sel)}
                    <li class="row" title={row.description || sel}>
                      <div class="row-line">
                        <span class="concept">
                          {#if !mono}
                            <span class="pole neg">{poles.negative}</span>
                            <span class="axis-sep" aria-hidden="true">↔</span>
                          {/if}
                          <span class="pole pos">{poles.positive}</span>
                        </span>
                        <div class="actions">
                          <button
                            type="button"
                            class="act extract"
                            disabled={inFlight}
                            onclick={() => onExtract(row)}
                            title={`extract ${sel} for the current model`}
                          >{inFlight ? "…" : "extract"}</button>
                          <button
                            type="button"
                            class="act del"
                            class:confirm={confirming}
                            disabled={inFlight}
                            onclick={() => onDeleteClick(row)}
                            title={confirming
                              ? "click again to confirm"
                              : `delete ${sel}`}
                          >{confirming ? "confirm?" : "delete"}</button>
                        </div>
                      </div>
                    </li>
                  {/each}
                </ul>
              {/if}
            </section>
          {/each}
        </div>
      {/if}
    {/if}
  </div>
</section>

<style>
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
  .close:hover {
    color: var(--accent-red);
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
  .error {
    color: var(--accent-error);
    font-size: var(--text-sm);
    margin: 0;
    word-break: break-word;
  }
  .muted {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    margin: 0;
  }

  /* ---- custom vector launcher ---- */
  .custom-btn {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    width: 100%;
    text-align: left;
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .custom-btn:hover {
    background: var(--bg-elev);
    border-color: var(--accent);
    color: var(--accent);
  }
  .plus {
    color: var(--accent);
    font-size: var(--text);
    font-weight: var(--weight-medium);
  }
  .custom-label {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
  }
  .custom-hint {
    flex: 1 1 auto;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-align: right;
  }

  /* ---- search row ---- */
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
    transition:
      color var(--dur) var(--ease-out),
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

  /* ---- section headings ---- */
  .section-title {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    margin: var(--space-3) 0 0;
    padding: 0;
    color: var(--accent);
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .section-hint {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-normal);
    text-transform: none;
    letter-spacing: 0;
  }

  /* ---- catalog ---- */
  .catalog {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
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

  /* ---- rows ---- */
  .rows {
    list-style: none;
    margin: 0;
    padding: var(--space-2) 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .row {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
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
  /* Top line of a row — keeps the historical concept-on-left,
   * actions-on-right shape so existing rows look unchanged when the
   * inspect body is collapsed. */
  .row-line {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: var(--space-3);
  }
  .concept {
    display: inline-flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .pole {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: var(--text-sm);
  }
  .pole.neg {
    color: var(--fg-muted);
  }
  .pole.pos {
    color: var(--fg-strong);
  }
  .axis-sep {
    color: var(--fg-muted);
    flex: 0 0 auto;
  }

  .actions {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .act {
    background: transparent;
    color: var(--fg-dim);
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
    color: var(--fg-strong);
    border-color: var(--fg-muted);
  }
  .act:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .act.extract,
  .act.refit {
    color: var(--accent);
    border-color: var(--border);
  }
  .act.extract:hover:not(:disabled),
  .act.refit:hover:not(:disabled) {
    background: var(--accent-subtle);
    border-color: var(--accent);
  }
  .act.inspect {
    color: var(--fg-dim);
    min-width: 2.2em;
  }
  .act.inspect:hover:not(:disabled) {
    color: var(--accent);
    border-color: var(--accent);
  }
  .act.del:hover:not(:disabled) {
    color: var(--accent-red);
    border-color: var(--accent-red);
  }
  .act.del.confirm {
    color: var(--accent-red);
    border-color: var(--accent-red);
    background: color-mix(in srgb, var(--accent-red) 12%, transparent);
  }

  /* ---- inspect body ---- */
  .inspect-body {
    padding-top: var(--space-2);
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .meta-list {
    margin: 0;
    display: grid;
    grid-template-columns: max-content 1fr;
    column-gap: var(--space-4);
    row-gap: var(--space-1);
    font-size: var(--text-xs);
  }
  .meta-list dt {
    color: var(--fg-muted);
    text-transform: lowercase;
  }
  .meta-list dd {
    margin: 0;
    color: var(--fg-strong);
    word-break: break-word;
  }
  .meta-list dd.model-list {
    color: var(--fg-dim);
    font-variant-numeric: tabular-nums;
  }
  .deep-link {
    align-self: flex-start;
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    cursor: pointer;
    transition: border-color var(--dur) var(--ease-out),
      background var(--dur) var(--ease-out);
  }
  .deep-link:hover {
    border-color: var(--accent);
    background: var(--accent-subtle);
  }
</style>
