<script lang="ts">
  // Vector picker drawer — lists locally installed concepts and lets the
  // user add one to the steering rack with a single click.  Mirrors the
  // TUI's ``/steer 0.5 honest`` ergonomics: pick a name, the session
  // resolves it (loads from cache or extracts on miss), it lands on the
  // rack at the canonical name.
  //
  // Footer keeps "extract new (pos/neg)…" and "load from disk…" affordances
  // for the rare advanced paths — these route into the existing extract /
  // load drawers.

  import { ApiError, apiVectors } from "../lib/api";
  import {
    addVectorToRack,
    closeDrawer,
    openDrawer,
    refreshPacks,
    refreshVectorList,
  } from "../lib/stores.svelte";
  import type { LocalPackInfo } from "../lib/types";
  import SearchableConceptList from "./_SearchableConceptList.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  // Names currently being extracted — keys both the LocalPackInfo's
  // ``name`` field and the qualified ``ns/name`` selector so the picker
  // child component shows a spinner on the right row.
  let busy: Set<string> = $state(new Set());
  let errorMsg: string | null = $state(null);

  // Refresh the global pack name list when the drawer opens so racks
  // elsewhere (autocomplete, status bar) reflect anything that was
  // installed since the last bootstrap.  Failures don't block the
  // drawer — the inner list fetches its own copy anyway.
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

  /** Fire ``apiVectors.extract`` against the chosen concept name and add
   * the resulting canonical to the rack.  The server short-circuits to
   * the cached profile when one already exists for the loaded model —
   * matches the TUI's /steer 0.5 <name> semantics. */
  async function pickAndAdd(name: string): Promise<void> {
    if (!name) return;
    errorMsg = null;
    markBusy(name);
    try {
      const r = await apiVectors.extract({ name, register: true });
      // refreshVectorList caches the profile in vectorRack.profiles so
      // the strip's expander, layer-norms collapsible, and correlation
      // matrix all see the new vector without their own refetch.
      await refreshVectorList();
      addVectorToRack(r.canonical);
      closeDrawer();
    } catch (e) {
      if (e instanceof ApiError) {
        const detail =
          e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message;
        errorMsg = `${e.status}: ${detail}`;
      } else {
        errorMsg = e instanceof Error ? e.message : String(e);
      }
    } finally {
      clearBusy(name);
    }
  }

  function onPick(row: LocalPackInfo): void {
    void pickAndAdd(row.name);
  }

  function onExtractFly(query: string): void {
    void pickAndAdd(query);
  }

  function gotoExtract(): void {
    openDrawer("extract");
  }

  function gotoLoad(): void {
    openDrawer("load");
  }
</script>

<section class="drawer-shell" aria-label="Vector picker drawer">
  <header class="header">
    <span class="title">add steering vector</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button>
  </header>

  <div class="body">
    <p class="hint">
      pick a concept — saklas extracts on miss, loads from cache otherwise.
      mirrors the TUI's <code>/steer 0.5 &lt;name&gt;</code>.
    </p>

    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}

    <SearchableConceptList
      placeholder="filter local concepts (or type a name to extract)…"
      actionLabel="add to rack"
      allowExtractFly
      emptyHint="install one via Tools › Packs"
      busy={busy}
      onPick={onPick}
      onExtractFly={onExtractFly}
    />
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={gotoLoad}>
      load from disk…
    </button>
    <button type="button" class="btn" onclick={gotoExtract}>
      extract new (pos/neg)…
    </button>
    <button type="button" class="btn primary" onclick={closeDrawer}>
      done
    </button>
  </footer>
</section>

<style>
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6em 1em;
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent-blue);
    text-transform: lowercase;
    letter-spacing: 0.04em;
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

  .body {
    flex: 1 1 auto;
    overflow: hidden;
    padding: 0.7em 1em;
    display: flex;
    flex-direction: column;
    gap: 0.6em;
    min-height: 0;
  }
  .hint {
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    margin: 0;
    line-height: 1.4;
  }
  .hint code {
    color: var(--accent-blue);
    background: var(--bg-alt);
    padding: 0.05em 0.3em;
    border-radius: 2px;
  }
  .error {
    color: var(--accent-error);
    font-size: var(--font-size-small);
    margin: 0;
    word-break: break-word;
  }

  .footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.5em;
    padding: 0.6em 1em;
    border-top: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .btn {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.4em 0.9em;
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    cursor: pointer;
    border-radius: 3px;
  }
  .btn:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .btn.primary {
    color: var(--accent-blue);
    border-color: var(--accent-blue);
  }
  .btn.primary:hover:not(:disabled) {
    background: rgba(88, 166, 255, 0.1);
  }
</style>
