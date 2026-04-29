<script lang="ts">
  // Probe picker drawer — mirror of VectorPickerDrawer but the primary
  // action activates the concept as a probe (POST /probes/{name}) rather
  // than landing it on the steering rack.  Unlike the vector picker, no
  // "extract on the fly" path: probes only attach to concepts that have
  // already been extracted (the server's probe activation route assumes
  // a registered profile).  The empty-hint points at the pack drawer for
  // installing more.

  import { ApiError, apiVectors } from "../lib/api";
  import {
    activateProbe,
    closeDrawer,
    openDrawer,
    refreshPacks,
  } from "../lib/stores.svelte";
  import type { LocalPackInfo } from "../lib/types";
  import SearchableConceptList from "./_SearchableConceptList.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  let busy: Set<string> = $state(new Set());
  let errorMsg: string | null = $state(null);

  void refreshPacks();

  function markBusy(name: string): void {
    const next = new Set(busy);
    next.add(name);
    busy = next;
  }
  function clearBusy(name: string): void {
    const next = new Set(busy);
    next.delete(name);
    busy = next;
  }

  /** Activate the picked concept as a probe.  Probe activation expects a
   * registered profile, so we extract first (server short-circuits on
   * cache hit, identical to the steer path) — that way picking a bundled
   * concept that was never extracted on this model still works.  Then
   * activate via POST /probes/{name}.  ``activateProbe`` also seeds the
   * highlight target on first add, mirroring the TUI's /probe behavior. */
  async function pickAndActivate(name: string): Promise<void> {
    if (!name) return;
    errorMsg = null;
    markBusy(name);
    try {
      const extracted = await apiVectors.extract({ name, register: true });
      await activateProbe(extracted.canonical);
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
    void pickAndActivate(row.name);
  }

  function gotoPack(): void {
    openDrawer("pack");
  }
</script>

<section class="drawer-shell" aria-label="Probe picker drawer">
  <header class="header">
    <span class="title">add probe</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button>
  </header>

  <div class="body">
    <p class="hint">
      pick a concept to monitor — saklas activates it as a probe (mirrors
      the TUI's <code>/probe &lt;name&gt;</code>).  if it isn't installed
      locally, install a pack first.
    </p>

    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}

    <SearchableConceptList
      placeholder="filter local concepts…"
      actionLabel="activate"
      allowExtractFly={false}
      emptyHint="install a pack via Tools › Packs"
      busy={busy}
      onPick={onPick}
    />
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={gotoPack}>
      packs…
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
