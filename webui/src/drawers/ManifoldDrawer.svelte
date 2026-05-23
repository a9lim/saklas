<script lang="ts">
  // Manifold browser — the "+ add manifold" surface from the steering
  // rack.  Two sections split on whether a tensor exists for the loaded
  // model:
  //
  //   * Fitted    — manifolds with a baked tensor for the current
  //                 model.  Per row: [+steer] [fit] [delete].
  //   * Unfitted  — manifolds without a tensor for this model.  Per
  //                 row: [fit] [delete] — they must be fitted before
  //                 they can steer.
  //
  // The "+ build manifold" button at the top opens ManifoldBuilderDrawer
  // for authoring a fresh manifold.  Fit drives an SSE progress toast;
  // delete uses a 2-step confirm.

  import { onMount } from "svelte";
  import { SvelteMap, SvelteSet } from "svelte/reactivity";
  import { ApiError, apiManifolds, apiManifoldFitStream } from "../lib/api";
  import {
    addManifoldToRack,
    closeDrawer,
    manifoldRack,
    openDrawer,
    refreshManifoldList,
  } from "../lib/stores.svelte";
  import {
    dismissToast,
    pushToast,
    updateToast,
  } from "../lib/stores/toasts.svelte";
  import type { ManifoldInfo } from "../lib/types";
  import DiagnosticsPanel from "../lib/manifolds/DiagnosticsPanel.svelte";

  let { params: _params }: { params?: unknown } = $props();
  $effect(() => { void _params; });

  let errorMsg: string | null = $state(null);

  const busyKeys = new SvelteSet<string>();
  const confirmKeys = new SvelteSet<string>();
  const confirmTimers = new Map<string, number>();

  // Per-row inspector state.  The list endpoint omits per-tensor
  // diagnostics (the JSON would balloon on big heaps), so the
  // inspector lazily GETs the detail shape the first time the user
  // expands a row.  ``inspectKeys`` drives the open/closed flag;
  // ``detailCache`` keeps the fetched payload alive for subsequent
  // re-opens within the same drawer mount.
  const inspectKeys = new SvelteSet<string>();
  const detailCache = new SvelteMap<string, ManifoldInfo>();
  const detailLoading = new SvelteSet<string>();
  const detailErrors = new SvelteMap<string, string>();

  onMount(() => {
    void refreshManifoldList();
    return () => {
      for (const t of confirmTimers.values()) window.clearTimeout(t);
      confirmTimers.clear();
    };
  });

  function rowKey(m: ManifoldInfo): string {
    return `${m.namespace}/${m.name}`;
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

  const fitted = $derived(
    manifoldRack.catalog.filter((m) => m.fitted_for_session),
  );
  const unfitted = $derived(
    manifoldRack.catalog.filter((m) => !m.fitted_for_session),
  );

  function isRacked(m: ManifoldInfo): boolean {
    return manifoldRack.entries.has(rowKey(m)) ||
      manifoldRack.entries.has(m.name);
  }

  function onSteer(m: ManifoldInfo): void {
    if (isRacked(m)) return;
    addManifoldToRack(rowKey(m));
    closeDrawer();
  }

  function onFit(m: ManifoldInfo): void {
    const key = rowKey(m);
    busyKeys.add(key);
    errorMsg = null;
    const toastId = pushToast(`fitting manifold '${key}'…`, {
      kind: "info",
      ttlMs: null,
    });
    void (async () => {
      try {
        await apiManifoldFitStream(m.namespace, m.name, {}, (ev) => {
          if (ev.event === "progress") {
            const msg =
              ev.data && typeof ev.data === "object"
                ? (ev.data as { message?: string }).message
                : null;
            if (msg) updateToast(toastId, { detail: msg });
          }
        });
        await refreshManifoldList();
        dismissToast(toastId);
        pushToast(`fitted ${key}`, { kind: "info" });
      } catch (e) {
        dismissToast(toastId);
        pushToast(`fit '${key}' failed — ${describeError(e)}`, {
          kind: "error",
          ttlMs: null,
        });
      } finally {
        busyKeys.delete(key);
      }
    })();
  }

  function onDeleteClick(m: ManifoldInfo): void {
    const key = rowKey(m);
    if (confirmKeys.has(key)) {
      const t = confirmTimers.get(key);
      if (t !== undefined) {
        window.clearTimeout(t);
        confirmTimers.delete(key);
      }
      confirmKeys.delete(key);
      void doDelete(m);
      return;
    }
    confirmKeys.add(key);
    const t = window.setTimeout(() => {
      confirmKeys.delete(key);
      confirmTimers.delete(key);
    }, 3000);
    confirmTimers.set(key, t);
  }

  async function doDelete(m: ManifoldInfo): Promise<void> {
    const key = rowKey(m);
    busyKeys.add(key);
    errorMsg = null;
    try {
      await apiManifolds.delete(m.namespace, m.name);
      await refreshManifoldList();
      pushToast(`deleted manifold ${key}`, { kind: "info" });
    } catch (e) {
      errorMsg = describeError(e);
    } finally {
      busyKeys.delete(key);
    }
  }

  function gotoBuilder(): void {
    openDrawer("manifold_builder");
  }

  function fitModeBadge(m: ManifoldInfo): string | null {
    if (m.fit_mode && m.fit_mode !== "authored") return m.fit_mode;
    return null;
  }

  function isDiscoverMode(m: ManifoldInfo): boolean {
    return m.fit_mode === "pca" || m.fit_mode === "spectral";
  }

  async function toggleInspect(m: ManifoldInfo): Promise<void> {
    const key = rowKey(m);
    if (inspectKeys.has(key)) {
      inspectKeys.delete(key);
      return;
    }
    inspectKeys.add(key);
    if (detailCache.has(key)) return;
    detailLoading.add(key);
    detailErrors.delete(key);
    try {
      const detail = await apiManifolds.get(m.namespace, m.name);
      detailCache.set(key, detail);
    } catch (e) {
      detailErrors.set(key, describeError(e));
    } finally {
      detailLoading.delete(key);
    }
  }
</script>

<section class="drawer-shell" aria-label="Manifolds">
  <header class="header">
    <span class="title">manifolds</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button>
  </header>

  <div class="body">
    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}

    <button type="button" class="build-btn" onclick={gotoBuilder}>
      <span class="plus" aria-hidden="true">+</span>
      <span class="build-label">build manifold</span>
      <span class="build-hint">author a domain and node corpus</span>
    </button>

    {#if manifoldRack.unavailable}
      <p class="muted">
        this server doesn't expose the manifold API — update saklas to
        author and fit steering manifolds.
      </p>
    {:else if manifoldRack.loading && manifoldRack.catalog.length === 0}
      <p class="muted">loading manifolds…</p>
    {:else if manifoldRack.catalog.length === 0}
      <p class="muted">
        no manifolds yet — use + build manifold above to author one.
      </p>
    {:else}
      {#if fitted.length > 0}
        <h2 class="section-title">Fitted</h2>
        <ul class="rows" role="list">
          {#each fitted as m (rowKey(m))}
            {@const key = rowKey(m)}
            {@const busy = busyKeys.has(key)}
            {@const confirming = confirmKeys.has(key)}
            {@const inspecting = inspectKeys.has(key)}
            {@const badge = fitModeBadge(m)}
            <li class="row" title={m.description || key}>
              <div class="row-line">
                <div class="meta">
                  <span class="row-name">{key}</span>
                  <span class="row-sub">
                    {m.domain_label} · {m.node_count} nodes
                    {#if badge}<span class="fit-badge fit-{badge}">{badge}</span>{/if}
                    {#if m.stale}<span class="stale">stale</span>{/if}
                  </span>
                </div>
                <div class="actions">
                  <button
                    type="button"
                    class="act inspect"
                    aria-expanded={inspecting}
                    onclick={() => void toggleInspect(m)}
                    title={inspecting ? "hide diagnostics" : "inspect fit"}
                  >{inspecting ? "▾" : "ⓘ"}</button>
                  <button
                    type="button"
                    class="act steer"
                    disabled={busy || isRacked(m)}
                    onclick={() => onSteer(m)}
                    title={isRacked(m)
                      ? `${key} is already racked`
                      : `rack ${key} for steering`}
                  >+steer</button>
                  <button
                    type="button"
                    class="act fit"
                    disabled={busy}
                    onclick={() => onFit(m)}
                    title={`re-fit ${key} for the current model`}
                  >{busy ? "…" : "re-fit"}</button>
                  <button
                    type="button"
                    class="act del"
                    class:confirm={confirming}
                    disabled={busy}
                    onclick={() => onDeleteClick(m)}
                    title={confirming ? "click again to confirm" : `delete ${key}`}
                  >{confirming ? "confirm?" : "delete"}</button>
                </div>
              </div>
              {#if inspecting}
                <div class="inspect-body">
                  {#if detailLoading.has(key)}
                    <p class="muted">loading…</p>
                  {:else if detailErrors.has(key)}
                    <p class="error">{detailErrors.get(key)}</p>
                  {:else if detailCache.has(key)}
                    {@const detail = detailCache.get(key)!}
                    {#if isDiscoverMode(detail)}
                      <DiagnosticsPanel manifold={detail} />
                    {:else}
                      <p class="muted">
                        authored manifold — no discover-mode diagnostics
                        ({detail.domain_label}, dim={detail.intrinsic_dim})
                      </p>
                    {/if}
                  {/if}
                </div>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}

      {#if unfitted.length > 0}
        <h2 class="section-title">
          Unfitted
          <span class="section-hint">no tensor for this model yet</span>
        </h2>
        <ul class="rows" role="list">
          {#each unfitted as m (rowKey(m))}
            {@const key = rowKey(m)}
            {@const busy = busyKeys.has(key)}
            {@const confirming = confirmKeys.has(key)}
            {@const badge = fitModeBadge(m)}
            <li class="row" title={m.description || key}>
              <div class="row-line">
                <div class="meta">
                  <span class="row-name">{key}</span>
                  <span class="row-sub">
                    {m.domain_label} · {m.node_count} nodes
                    {#if badge}<span class="fit-badge fit-{badge}">{badge}</span>{/if}
                  </span>
                </div>
                <div class="actions">
                  <button
                    type="button"
                    class="act fit"
                    disabled={busy}
                    onclick={() => onFit(m)}
                    title={`fit ${key} for the current model`}
                  >{busy ? "…" : "fit"}</button>
                  <button
                    type="button"
                    class="act del"
                    class:confirm={confirming}
                    disabled={busy}
                    onclick={() => onDeleteClick(m)}
                    title={confirming ? "click again to confirm" : `delete ${key}`}
                  >{confirming ? "confirm?" : "delete"}</button>
                </div>
              </div>
            </li>
          {/each}
        </ul>
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
    line-height: 1.4;
  }
  .build-btn {
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
  .build-btn:hover {
    background: var(--bg-elev);
    border-color: var(--accent-purple);
    color: var(--accent-purple);
  }
  .plus {
    color: var(--accent-purple);
    font-weight: var(--weight-medium);
  }
  .build-label {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
  }
  .build-hint {
    flex: 1 1 auto;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-align: right;
  }
  .section-title {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    margin: var(--space-3) 0 0;
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
  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
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
    transition: border-color var(--dur) var(--ease-out);
  }
  .row:hover {
    border-color: var(--accent-purple);
  }
  .row-line {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: var(--space-3);
  }
  .fit-badge {
    display: inline-block;
    margin-left: var(--space-2);
    padding: 0 var(--space-2);
    border-radius: var(--radius);
    text-transform: uppercase;
    font-size: var(--text-2xs);
    letter-spacing: 0.04em;
    border: 1px solid var(--border);
    color: var(--accent-purple);
    background: color-mix(in srgb, var(--accent-purple) 12%, transparent);
  }
  .fit-spectral {
    color: var(--accent);
    background: var(--accent-subtle);
  }
  .inspect-body {
    padding-top: var(--space-2);
    border-top: 1px solid var(--border);
  }
  .error {
    color: var(--accent-error);
    font-size: var(--text-xs);
    margin: 0;
  }
  .act.inspect {
    color: var(--fg-dim);
    min-width: 2.2em;
  }
  .act.inspect:hover:not(:disabled) {
    color: var(--accent-purple);
    border-color: var(--accent-purple);
  }
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
  }
  .stale {
    color: var(--accent-yellow);
    margin-left: var(--space-2);
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
  .act.steer {
    color: var(--accent-purple);
  }
  .act.steer:hover:not(:disabled) {
    background: rgba(167, 139, 250, 0.12);
    border-color: var(--accent-purple);
  }
  .act.fit {
    color: var(--accent);
  }
  .act.fit:hover:not(:disabled) {
    background: var(--accent-subtle);
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
</style>
