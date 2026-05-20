<script lang="ts">
  // Steering selector — the catalog, presented the way it's organised.
  //
  // Three things in one narrow surface (docs/plans/webui-overhaul.md
  // §"The picker"):
  //   1. a categorized, pole-framed concept menu (SearchableConceptList)
  //   2. inline custom extraction for when the catalog hasn't got it
  //   3. a "load from disk" link for the genuinely separate file path
  //
  // Picking a catalog row extracts on miss / loads from cache, then lands
  // it on the rack at the pack's recommended α.  A search query that
  // matches nothing flows into the custom-extraction name field.

  import { apiExtractStream, ApiError, apiVectors } from "../lib/api";
  import {
    addVectorToRack,
    closeDrawer,
    openDrawer,
    refreshPacks,
    refreshVectorList,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { ExtractRequest, LocalPackInfo } from "../lib/types";
  import SearchableConceptList from "./_SearchableConceptList.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  // Names currently being extracted — keys both ``name`` and the
  // qualified ``ns/name`` so the catalog shows a spinner on the right row.
  let busy: Set<string> = $state(new Set());
  let errorMsg: string | null = $state(null);

  // Bound out of the catalog list.
  let query = $state("");
  let matchCount = $state(0);

  // Refresh the global pack name list so racks elsewhere reflect anything
  // installed since bootstrap.  Failures don't block the drawer.
  void refreshPacks();

  function markBusy(...names: string[]): void {
    const next = new Set(busy);
    for (const n of names) next.add(n);
    busy = next;
  }
  function clearBusy(...names: string[]): void {
    const next = new Set(busy);
    for (const n of names) next.delete(n);
    busy = next;
  }

  function reportError(e: unknown): void {
    if (e instanceof ApiError) {
      const detail =
        e.body && typeof e.body === "object" && "detail" in (e.body as object)
          ? String((e.body as { detail: unknown }).detail)
          : e.message;
      errorMsg = `${e.status}: ${detail}`;
    } else {
      errorMsg = e instanceof Error ? e.message : String(e);
    }
  }

  /** Extract (server short-circuits on cache hit) and add to the rack at
   * the concept's recommended α. */
  async function pickAndAdd(name: string, alpha: number): Promise<void> {
    if (!name) return;
    errorMsg = null;
    markBusy(name);
    try {
      const r = await apiVectors.extract({ name, register: true });
      await refreshVectorList();
      addVectorToRack(r.canonical, alpha);
      closeDrawer();
    } catch (e) {
      reportError(e);
    } finally {
      clearBusy(name);
    }
  }

  function onPick(row: LocalPackInfo, alpha: number): void {
    void pickAndAdd(row.name, alpha);
  }

  // ---------------- custom extraction ----------------

  let customOpen = $state(false);
  let cName = $state("");
  let cPositive = $state("");
  let cNegative = $state("");
  let cMethod: "dim" | "pca" = $state("dim");
  let cDls = $state(true);
  let cSae = $state("");
  let cBusy = $state(false);
  let cLog: string[] = $state([]);
  let logEl: HTMLDivElement | null = $state(null);

  // A query that matches no catalog row flows into custom extraction —
  // open the section and seed the name field with what the user typed.
  let _lastSeeded = "";
  $effect(() => {
    const q = query.trim();
    if (q && matchCount === 0 && q !== _lastSeeded) {
      _lastSeeded = q;
      customOpen = true;
      if (!cName.trim()) cName = q;
    }
  });

  // name required; negative requires positive (nothing to contrast).
  const cValid = $derived.by(() => {
    const n = cName.trim();
    const p = cPositive.trim();
    const ng = cNegative.trim();
    if (!n) return { ok: false, reason: "name is required" } as const;
    if (ng && !p)
      return {
        ok: false,
        reason: "negative needs a positive, or leave both blank for single-concept",
      } as const;
    return { ok: true, reason: null } as const;
  });

  function appendLog(line: string): void {
    cLog = [...cLog, line];
    queueMicrotask(() => {
      if (logEl) logEl.scrollTop = logEl.scrollHeight;
    });
  }

  async function runExtract(): Promise<void> {
    if (!cValid.ok || cBusy) return;
    cBusy = true;
    errorMsg = null;
    cLog = [];

    const req: ExtractRequest = { name: cName.trim(), register: true };
    const p = cPositive.trim();
    const ng = cNegative.trim();
    if (p && ng) req.source = { positive: p, negative: ng };
    else if (p) req.source = p;
    req.method = cMethod;
    req.dls = cDls;
    const sae = cSae.trim();
    if (sae) req.sae = sae;

    try {
      const result = await apiExtractStream(req, (ev) => {
        if (ev.event === "progress") {
          const m =
            ev.data && typeof ev.data === "object"
              ? (ev.data as { message?: string }).message
              : null;
          appendLog(m ?? JSON.stringify(ev.data));
        } else if (ev.event === "done") {
          appendLog("done");
        } else if (ev.event === "error") {
          const m =
            ev.data && typeof ev.data === "object"
              ? (ev.data as { message?: string }).message
              : null;
          appendLog(`error: ${m ?? "unknown"}`);
        }
      });
      await refreshVectorList();
      addVectorToRack(result.canonical);
      pushToast(`extracted ${result.canonical}, added to rack`, {
        kind: "info",
      });
      closeDrawer();
    } catch (e) {
      reportError(e);
    } finally {
      cBusy = false;
    }
  }

  function gotoLoad(): void {
    openDrawer("load");
  }
</script>

<section class="drawer-shell" aria-label="Add steering">
  <header class="header">
    <span class="title">add steering</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button>
  </header>

  <div class="body">
    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}

    <SearchableConceptList
      placeholder="search concepts…"
      actionLabel="add"
      emptyHint="install one via the rail › vectors › packs"
      scroll={false}
      bind:query
      bind:matchCount
      {busy}
      {onPick}
    />

    <!-- Custom extraction — the catalog's escape hatch. -->
    <section class="custom" class:open={customOpen}>
      <button
        type="button"
        class="custom-header"
        aria-expanded={customOpen}
        onclick={() => (customOpen = !customOpen)}
      >
        <span class="caret" aria-hidden="true">{customOpen ? "▾" : "▸"}</span>
        <span class="custom-name">Custom extraction</span>
        <span class="custom-hint">positive / negative contrast pair</span>
      </button>

      {#if customOpen}
        <form
          class="form"
          onsubmit={(ev) => {
            ev.preventDefault();
            void runExtract();
          }}
        >
          <label class="field">
            <span class="label">name</span>
            <input
              type="text"
              class="input"
              bind:value={cName}
              disabled={cBusy}
              placeholder="my_concept"
              autocomplete="off"
              spellcheck="false"
            />
          </label>
          <label class="field">
            <span class="label">positive</span>
            <input
              type="text"
              class="input"
              bind:value={cPositive}
              disabled={cBusy}
              placeholder="contrastive positive text"
              autocomplete="off"
              spellcheck="false"
            />
          </label>
          <label class="field">
            <span class="label">negative</span>
            <input
              type="text"
              class="input"
              bind:value={cNegative}
              disabled={cBusy}
              placeholder="contrastive negative text"
              autocomplete="off"
              spellcheck="false"
            />
          </label>
          <p class="field-hint">
            leave both blank to extract from a statements pack named above.
          </p>

          <fieldset class="field method">
            <legend class="label">method</legend>
            <label class="radio">
              <input
                type="radio"
                bind:group={cMethod}
                value="dim"
                disabled={cBusy}
              />
              <span>difference-of-means</span>
            </label>
            <label class="radio">
              <input
                type="radio"
                bind:group={cMethod}
                value="pca"
                disabled={cBusy}
              />
              <span>contrastive PCA</span>
            </label>
          </fieldset>

          <label class="field">
            <span class="label">SAE release <span class="opt">optional</span></span>
            <input
              type="text"
              class="input"
              bind:value={cSae}
              disabled={cBusy}
              placeholder="e.g. gemma-scope-2b-pt-res"
              autocomplete="off"
              spellcheck="false"
            />
          </label>

          <label class="check">
            <input type="checkbox" bind:checked={cDls} disabled={cBusy} />
            <span>centered DLS layer selection</span>
          </label>

          {#if !cValid.ok}
            <p class="validation">{cValid.reason}</p>
          {/if}

          <button
            type="submit"
            class="extract-btn"
            disabled={!cValid.ok || cBusy}
          >
            {cBusy ? "extracting…" : "extract → add to rack"}
          </button>

          {#if cLog.length > 0}
            <div class="log" bind:this={logEl} aria-label="Extraction progress">
              {#each cLog as line, i (i)}
                <div class="log-line">{line}</div>
              {/each}
            </div>
          {/if}
        </form>
      {/if}
    </section>

    <button type="button" class="disk-link" onclick={gotoLoad}>
      load a vector from disk…
    </button>
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
    color: var(--accent-blue);
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

  /* ---- custom extraction ---- */
  .custom {
    border-top: 1px solid var(--border);
    padding-top: var(--space-3);
    display: flex;
    flex-direction: column;
  }
  .custom-header {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    width: 100%;
    text-align: left;
    background: transparent;
    border: 0;
    padding: var(--space-2) var(--space-1);
    color: var(--fg-muted);
    cursor: pointer;
    transition: color var(--dur) var(--ease-out);
  }
  .custom-header:hover {
    color: var(--fg-strong);
  }
  .caret {
    font-size: var(--text-xs);
  }
  .custom-name {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
  }
  .custom.open .custom-name {
    color: var(--accent-blue);
  }
  .custom-hint {
    flex: 1 1 auto;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-align: right;
  }

  .form {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-1) var(--space-1);
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    letter-spacing: 0;
  }
  .opt {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-style: italic;
  }
  .input {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    transition: border-color var(--dur) var(--ease-out);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .input:disabled {
    opacity: 0.6;
  }
  .field-hint {
    margin: calc(-1 * var(--space-1)) 0 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }

  .method {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    margin: 0;
  }
  .method legend {
    padding: 0 var(--space-2);
  }
  .radio,
  .check {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    color: var(--fg-strong);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .radio input,
  .check input {
    accent-color: var(--accent-blue);
  }

  .validation {
    color: var(--accent-yellow);
    font-size: var(--text-sm);
    margin: 0;
  }
  .extract-btn {
    background: var(--accent);
    color: var(--text-on-accent);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out);
  }
  .extract-btn:hover:not(:disabled) {
    background: var(--accent-light);
    border-color: var(--accent-light);
  }
  .extract-btn:disabled {
    background: var(--bg-elev);
    color: var(--fg-muted);
    border-color: var(--border);
    cursor: not-allowed;
  }

  .log {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-3);
    max-height: 180px;
    overflow-y: auto;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.4;
    white-space: pre-wrap;
  }
  .log-line {
    word-break: break-word;
  }

  .disk-link {
    align-self: flex-start;
    background: transparent;
    border: 0;
    border-top: 1px solid var(--border);
    width: 100%;
    text-align: left;
    color: var(--fg-dim);
    padding: var(--space-4) var(--space-1) var(--space-1);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition: color var(--dur) var(--ease-out);
  }
  .disk-link:hover {
    color: var(--accent-blue);
  }
</style>
