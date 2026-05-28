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
    attachManifoldProbe,
    closeDrawer,
    detachManifoldProbe,
    manifoldProbeRack,
    manifoldRack,
    openDrawer,
    refreshManifoldList,
    refreshManifoldProbeList,
    setManifoldLabel,
  } from "../lib/stores.svelte";
  import {
    dismissToast,
    pushToast,
    updateToast,
  } from "../lib/stores/toasts.svelte";
  import type { ManifoldInfo } from "../lib/types";
  import DiagnosticsPanel from "../lib/manifolds/DiagnosticsPanel.svelte";

  // ``mode`` selects the visual emphasis: ``"steer"`` (default —
  // matches the original drawer behaviour) or ``"probe"`` (opened from
  // ProbeRack's "+ add manifold probe"; lands the user on the probe
  // attach action and the inline custom-probe form).  ``params`` is
  // typed as ``unknown`` so it round-trips through ``drawerState.params``
  // (also typed loosely so each drawer owns its own shape); the mode
  // derivation pulls the field defensively.
  let { params }: { params?: unknown } = $props();
  const mode = $derived.by<"steer" | "probe">(() => {
    if (
      params && typeof params === "object"
      && (params as { mode?: unknown }).mode === "probe"
    ) {
      return "probe";
    }
    return "steer";
  });

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

  // Free-text filter over manifolds *and* their node labels (mirrors
  // VectorsDrawer's search).  ``searching`` auto-expands every node
  // list so matches are visible without a manual unfold.
  let query: string = $state("");
  let searchInputRef: HTMLInputElement | null = $state(null);

  // Per-row node-list collapse.  Node lists default OPEN, so a key
  // present here means the user explicitly folded that row's nodes.
  const collapsedNodes = new SvelteSet<string>();

  onMount(() => {
    void refreshManifoldList();
    // Pull the attached-probe list too — the +probe / detach-button
    // affordance is keyed on whether a fitted manifold is already
    // attached, so the drawer needs this fresh on mount.
    void refreshManifoldProbeList();
    queueMicrotask(() => searchInputRef?.focus({ preventScroll: true }));
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

  const searching = $derived(query.trim().length > 0);

  function nodeMatches(label: string, q: string): boolean {
    return label.toLowerCase().includes(q);
  }

  /** A manifold matches the query on its key/name/namespace/description
   *  OR on any of its node labels — so searching a node name surfaces
   *  the card that owns it. */
  function rowMatches(m: ManifoldInfo, q: string): boolean {
    if (!q) return true;
    const n = q.toLowerCase();
    if (rowKey(m).toLowerCase().includes(n)) return true;
    if (m.name.toLowerCase().includes(n)) return true;
    if (m.namespace.toLowerCase().includes(n)) return true;
    if (m.description && m.description.toLowerCase().includes(n)) return true;
    return (m.node_labels ?? []).some((l) => nodeMatches(l, n));
  }

  /** Node labels to render for a row under the active query.  When
   *  searching, narrow to matching labels — but if the row matched only
   *  on its name/description (no node hit), keep the full list so the
   *  card never renders an empty node block. */
  function visibleNodes(m: ManifoldInfo): string[] {
    const labels = m.node_labels ?? [];
    const q = query.trim().toLowerCase();
    if (!q) return labels;
    const hits = labels.filter((l) => nodeMatches(l, q));
    return hits.length > 0 ? hits : labels;
  }

  /** Per-node role slug (role-baselined manifolds), aligned with
   *  ``node_labels``.  ``null`` when the manifold carries no roles. */
  function nodeRoleFor(m: ManifoldInfo, label: string): string | null {
    const roles = m.node_roles;
    if (!roles) return null;
    const idx = (m.node_labels ?? []).indexOf(label);
    return idx >= 0 ? (roles[idx] ?? null) : null;
  }

  function nodesOpen(key: string): boolean {
    return searching || !collapsedNodes.has(key);
  }
  function toggleNodes(key: string): void {
    if (collapsedNodes.has(key)) collapsedNodes.delete(key);
    else collapsedNodes.add(key);
  }

  /** Rack the manifold (if not already) and pin it to ``label`` as a
   *  label-form term, then close — the one-click "steer to this node"
   *  affordance.  A second node click on an already-racked manifold
   *  just re-targets the existing term via ``setManifoldLabel``. */
  function onSteerNode(m: ManifoldInfo, label: string): void {
    if (!m.fitted_for_session) return;
    const key = rowKey(m);
    addManifoldToRack(key);
    setManifoldLabel(key, label);
    closeDrawer();
  }

  const fitted = $derived(
    manifoldRack.catalog.filter(
      (m) => m.fitted_for_session && rowMatches(m, query.trim()),
    ),
  );
  const unfitted = $derived(
    manifoldRack.catalog.filter(
      (m) => !m.fitted_for_session && rowMatches(m, query.trim()),
    ),
  );

  function isRacked(m: ManifoldInfo): boolean {
    return manifoldRack.entries.has(rowKey(m)) ||
      manifoldRack.entries.has(m.name);
  }

  function isProbed(m: ManifoldInfo): boolean {
    return manifoldProbeRack.entries.has(rowKey(m)) ||
      manifoldProbeRack.entries.has(m.name);
  }

  function onSteer(m: ManifoldInfo): void {
    if (isRacked(m)) return;
    addManifoldToRack(rowKey(m));
    closeDrawer();
  }

  async function onProbe(m: ManifoldInfo): Promise<void> {
    if (isProbed(m)) return;
    const key = rowKey(m);
    busyKeys.add(key);
    try {
      const info = await attachManifoldProbe(key);
      pushToast(`attached manifold probe ${info.name}`, { kind: "info" });
      closeDrawer();
    } catch (e) {
      pushToast(`attach failed — ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      busyKeys.delete(key);
    }
  }

  // Custom-attach form state — for selectors not in the catalog
  // (typed slugs, ns/name overrides, etc.).  Only rendered in probe
  // mode so it doesn't clutter the steer flow.
  let customSelector: string = $state("");
  let customAlias: string = $state("");
  let customTopN: number = $state(3);
  let customAttaching: boolean = $state(false);

  async function onCustomAttachSubmit(ev: SubmitEvent): Promise<void> {
    ev.preventDefault();
    const sel = customSelector.trim();
    if (!sel) {
      pushToast("selector required", { kind: "error" });
      return;
    }
    customAttaching = true;
    try {
      const opts: { name?: string; top_n?: number } = {};
      if (customAlias.trim()) opts.name = customAlias.trim();
      if (customTopN && customTopN > 0) opts.top_n = customTopN;
      const info = await attachManifoldProbe(sel, opts);
      pushToast(`attached manifold probe ${info.name}`, { kind: "info" });
      customSelector = "";
      customAlias = "";
    } catch (e) {
      pushToast(`attach failed — ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      customAttaching = false;
    }
  }

  async function onProbeRemove(m: ManifoldInfo): Promise<void> {
    const key = rowKey(m);
    busyKeys.add(key);
    try {
      // Probe may have been registered under either qualified or bare
      // name depending on the original selector — detach both forms
      // defensively.  The store helper deletes the entry locally; the
      // server route only knows the registered name, so try the
      // qualified key first and fall back to the bare name.
      const target = manifoldProbeRack.entries.has(key) ? key : m.name;
      await detachManifoldProbe(target);
      pushToast(`detached manifold probe ${target}`, { kind: "info" });
    } catch (e) {
      pushToast(`detach failed — ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      busyKeys.delete(key);
    }
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

  /** True iff at least one node carries a non-null role — this is a
   *  persona / role-paired manifold. */
  function isRoleAugmented(m: ManifoldInfo): boolean {
    return (m.node_roles ?? []).some((r) => r);
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
    <span class="title">{mode === "probe" ? "manifolds · probe" : "manifolds"}</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button>
  </header>

  <div class="body">
    {#snippet nodeSection(m: ManifoldInfo, steerable: boolean)}
      {@const key = rowKey(m)}
      {@const total = (m.node_labels ?? []).length}
      {@const nodes = visibleNodes(m)}
      {#if total > 0}
        <div class="nodes-block">
          <button
            type="button"
            class="nodes-toggle"
            aria-expanded={nodesOpen(key)}
            onclick={() => toggleNodes(key)}
            title={nodesOpen(key) ? "collapse nodes" : "expand nodes"}
          >
            <span class="caret" aria-hidden="true">{nodesOpen(key) ? "▾" : "▸"}</span>
            <span class="nodes-label">nodes</span>
            <span class="nodes-count"
              >{nodes.length}{#if nodes.length !== total}/{total}{/if}</span>
          </button>
          {#if nodesOpen(key)}
            <ul class="node-chips" role="list">
              {#each nodes as label (label)}
                {@const role = nodeRoleFor(m, label)}
                <li>
                  <button
                    type="button"
                    class="node-chip"
                    disabled={!steerable}
                    onclick={() => onSteerNode(m, label)}
                    title={steerable
                      ? `steer ${key} → ${label}`
                      : `fit ${key} first to steer to ${label}`}
                  >
                    <span class="node-name">{label}</span>
                    {#if role}<span class="node-role" title="role-baselined node">{role}</span>{/if}
                  </button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      {/if}
    {/snippet}

    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}

    {#if mode === "probe"}
      <p class="mode-hint">
        Attach a fitted manifold as a read-side probe.
        Use <strong>+probe</strong> on a fitted row, or attach by
        selector below.
      </p>
      <details class="custom-attach">
        <summary>attach by selector</summary>
        <form class="attach-form" onsubmit={onCustomAttachSubmit}>
          <label class="row-label">
            <span>selector</span>
            <input
              type="text"
              class="text-input"
              placeholder="ns/name"
              aria-label="Manifold selector"
              bind:value={customSelector}
              disabled={customAttaching}
            />
          </label>
          <label class="row-label">
            <span>name <span class="optional">(optional)</span></span>
            <input
              type="text"
              class="text-input"
              placeholder="registered name"
              aria-label="Registered probe name (optional)"
              bind:value={customAlias}
              disabled={customAttaching}
            />
          </label>
          <label class="row-label">
            <span>top_n</span>
            <input
              type="number"
              class="text-input small"
              min="1"
              max="32"
              step="1"
              aria-label="Per-token nearest-node list length"
              bind:value={customTopN}
              disabled={customAttaching}
            />
          </label>
          <button
            type="submit"
            class="act probe"
            disabled={customAttaching || !customSelector.trim()}
          >+ attach</button>
        </form>
      </details>
    {/if}

    <button type="button" class="build-btn" onclick={gotoBuilder}>
      <span class="plus" aria-hidden="true">+</span>
      <span class="build-label">build manifold</span>
      <span class="build-hint">author a domain and node corpus</span>
    </button>

    {#if !manifoldRack.unavailable && manifoldRack.catalog.length > 0}
      <div class="search-row">
        <input
          type="search"
          class="search"
          bind:this={searchInputRef}
          bind:value={query}
          placeholder="search manifolds & nodes…"
          autocomplete="off"
          spellcheck="false"
        />
        <button
          type="button"
          class="refresh"
          onclick={() => void refreshManifoldList()}
          disabled={manifoldRack.loading}
          title="re-fetch manifolds"
          aria-label="Refresh"
        >{manifoldRack.loading ? "…" : "↻"}</button>
      </div>
    {/if}

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
                    {#if isRoleAugmented(m)}<span class="fit-badge fit-persona" title="persona / role-paired manifold">persona</span>{/if}
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
                  {#if !manifoldProbeRack.unavailable}
                    {#if isProbed(m)}
                      <button
                        type="button"
                        class="act probe attached"
                        disabled={busy}
                        onclick={() => void onProbeRemove(m)}
                        title={`detach manifold probe ${key}`}
                      >−probe</button>
                    {:else}
                      <button
                        type="button"
                        class="act probe"
                        disabled={busy}
                        onclick={() => void onProbe(m)}
                        title={`attach ${key} as a read-side manifold probe`}
                      >+probe</button>
                    {/if}
                  {/if}
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
              {@render nodeSection(m, true)}
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
              {@render nodeSection(m, false)}
            </li>
          {/each}
        </ul>
      {/if}

      {#if fitted.length === 0 && unfitted.length === 0}
        <p class="muted">no manifold or node matches "{query.trim()}".</p>
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
  /* Probe action mirrors steer's purple family so manifold surfaces
   * (steering vs probing) keep one accent.  Attached state inverts so
   * the row reads "active probe — click to detach". */
  .act.probe {
    color: var(--accent-purple);
  }
  .act.probe:hover:not(:disabled) {
    background: rgba(167, 139, 250, 0.12);
    border-color: var(--accent-purple);
  }
  .act.probe.attached {
    background: rgba(167, 139, 250, 0.18);
    border-color: var(--accent-purple);
  }
  .act.probe.attached:hover:not(:disabled) {
    color: var(--accent-red);
    border-color: var(--accent-red);
    background: color-mix(in srgb, var(--accent-red) 12%, transparent);
  }

  /* Probe-mode hint banner — sits at the top of the drawer when the
   * user landed here from the probe rack, naming the difference
   * between +steer and +probe. */
  .mode-hint {
    margin: 0;
    padding: var(--space-3) var(--space-4);
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.4;
    background: rgba(167, 139, 250, 0.06);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent-purple);
    border-radius: var(--radius);
  }
  .mode-hint strong {
    color: var(--accent-purple);
    font-weight: var(--weight-medium);
  }

  /* Custom-attach disclosure — collapsed by default so it doesn't
   * compete with the catalog rows.  Opens to the same three-field
   * form the old standalone ManifoldProbeRack carried. */
  .custom-attach {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-deep);
  }
  .custom-attach > summary {
    cursor: pointer;
    padding: var(--space-3) var(--space-4);
    color: var(--fg-dim);
    font-size: var(--text-sm);
    list-style: none;
  }
  .custom-attach > summary::-webkit-details-marker { display: none; }
  .custom-attach > summary::before {
    content: "▸ ";
    color: var(--fg-muted);
  }
  .custom-attach[open] > summary::before { content: "▾ "; }
  .attach-form {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: var(--space-3) var(--space-4) var(--space-4);
    border-top: 1px solid var(--border);
  }
  .attach-form .row-label {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    font-size: var(--text-sm);
    color: var(--fg-muted);
  }
  .attach-form .row-label > span {
    flex: 0 0 5.5em;
  }
  .attach-form .optional {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .text-input {
    flex: 1 1 0;
    min-width: 0;
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  .text-input.small {
    flex: 0 0 5em;
  }
  .text-input:focus-visible {
    outline: 1px solid var(--accent-purple);
    outline-offset: -1px;
  }
  .text-input:disabled {
    opacity: 0.6;
  }
  .attach-form .act.probe {
    align-self: flex-end;
    margin-top: var(--space-2);
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

  /* ---- search row (mirrors VectorsDrawer) ---- */
  .search-row {
    display: flex;
    gap: var(--space-3);
    align-items: stretch;
  }
  .search {
    flex: 1 1 auto;
    min-width: 0;
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
    border-color: var(--accent-purple);
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
    opacity: 0.45;
    cursor: not-allowed;
  }

  /* ---- per-row node list ---- */
  .nodes-block {
    padding-top: var(--space-2);
    border-top: 1px solid var(--border);
  }
  .nodes-toggle {
    display: inline-flex;
    align-items: baseline;
    gap: var(--space-2);
    background: transparent;
    border: 0;
    padding: 0;
    color: var(--fg-dim);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    cursor: pointer;
    transition: color var(--dur) var(--ease-out);
  }
  .nodes-toggle:hover {
    color: var(--accent-purple);
  }
  .caret {
    color: var(--fg-muted);
  }
  .nodes-label {
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .nodes-count {
    color: var(--fg-muted);
  }
  .node-chips {
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .node-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    background: var(--bg-alt);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .node-chip:hover:not(:disabled) {
    color: var(--accent-purple);
    border-color: var(--accent-purple);
    background: rgba(167, 139, 250, 0.12);
  }
  .node-chip:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .node-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 16em;
  }
  .node-role {
    padding: 0 var(--space-2);
    border-radius: var(--radius);
    font-size: var(--text-2xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--accent-purple);
    background: color-mix(in srgb, var(--accent-purple) 12%, transparent);
  }
</style>
