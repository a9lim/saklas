<script lang="ts">
  // ManifoldMergeDrawer — manifold-side counterpart to MergeDrawer.
  //
  // Vector merge composes a new vector from a steering expression; the
  // manifold analogue unions *node corpora* and lets the next autofit
  // derive coords from the combined heap.  Restricted to discover-mode
  // sources by design — authored manifolds carry user-declared
  // geometry that isn't mergeable without a shared coordinate system.
  //
  // Reached from the workspace rail's "manifolds → merge…" entry,
  // parallel to "vectors → merge vector…".

  import { onMount } from "svelte";
  import { SvelteSet } from "svelte/reactivity";
  import { ApiError, apiManifolds } from "../lib/api";
  import {
    closeDrawer,
    manifoldRack,
    refreshManifoldList,
  } from "../lib/stores.svelte";
  import { dismissToast, pushToast } from "../lib/stores/toasts.svelte";
  import type { ManifoldInfo } from "../lib/types";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

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

  function rowKey(m: ManifoldInfo): string {
    return `${m.namespace}/${m.name}`;
  }
  function isDiscoverMode(m: ManifoldInfo): boolean {
    return m.fit_mode === "pca" || m.fit_mode === "spectral";
  }

  // ----- form state ----------------------------------------------------
  const selected = new SvelteSet<string>();
  let targetName: string = $state("");
  let fitMode: "" | "pca" | "spectral" = $state("");
  let merging: boolean = $state(false);

  /** Discover-only catalog — merge sources must be autofitted. */
  const discoverManifolds = $derived(
    manifoldRack.catalog.filter((m) => isDiscoverMode(m)),
  );

  function toggleSource(key: string): void {
    if (selected.has(key)) selected.delete(key);
    else selected.add(key);
  }

  /** Reconcile preview — when all selected sources share a fit_mode the
   *  merge inherits it; otherwise an explicit override is required. */
  const sourceModes = $derived.by<string[]>(() => {
    const modes = new Set<string>();
    for (const key of selected) {
      const m = manifoldRack.catalog.find((x) => rowKey(x) === key);
      if (m && m.fit_mode) modes.add(m.fit_mode);
    }
    return [...modes].sort();
  });

  const canSubmit = $derived(
    !merging
    && selected.size >= 2
    && targetName.trim().length > 0
    && (sourceModes.length <= 1 || fitMode !== "")
  );

  async function onSubmit(ev: SubmitEvent): Promise<void> {
    ev.preventDefault();
    const target = targetName.trim();
    if (selected.size < 2) {
      pushToast("pick >= 2 discover manifolds to merge", { kind: "error" });
      return;
    }
    if (!target) {
      pushToast("target name required", { kind: "error" });
      return;
    }
    const sources = [...selected].map((k) => {
      const [namespace, name] = k.split("/");
      return { namespace, name };
    });
    merging = true;
    const toastId = pushToast(
      `merging ${sources.length} manifolds into '${target}'…`,
      { kind: "info", ttlMs: null },
    );
    try {
      await apiManifolds.merge({
        name: target,
        sources,
        fit_mode: fitMode || undefined,
      });
      await refreshManifoldList();
      dismissToast(toastId);
      pushToast(
        `merged into local/${target} — run fit next`,
        { kind: "info" },
      );
      closeDrawer();
    } catch (e) {
      dismissToast(toastId);
      pushToast(`merge failed — ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      merging = false;
    }
  }

  onMount(() => {
    void refreshManifoldList();
  });
</script>

<section class="drawer-shell" aria-label="Merge manifolds">
  <header class="header">
    <span class="title">merge manifolds</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button>
  </header>

  <div class="body">
    <p class="hint">
      Union the <strong>nodes</strong> of two or more discover-mode
      manifolds into a fresh discover folder, then re-fit on the
      combined heap.  Authored manifolds (declared domain + coords)
      aren't mergeable — only their <em>autofitted</em> siblings.
    </p>

    {#if manifoldRack.unavailable}
      <p class="muted">
        this server doesn't expose the manifold API — update saklas to
        author and fit steering manifolds.
      </p>
    {:else if discoverManifolds.length < 2}
      <p class="muted">
        need ≥ 2 discover-mode (pca / spectral) manifolds in the
        catalog to merge.  Author one via <strong>+ build manifold</strong>
        in the manifold drawer, or pull one from <strong>manifold packs →
        search hf</strong>.
      </p>
    {:else}
      <form class="form" onsubmit={onSubmit}>
        <fieldset class="sources">
          <legend>sources <span class="optional">(pick ≥ 2)</span></legend>
          <ul class="source-list" role="list">
            {#each discoverManifolds as m (rowKey(m))}
              {@const key = rowKey(m)}
              <li class="source">
                <label class="source-label">
                  <input
                    type="checkbox"
                    checked={selected.has(key)}
                    onchange={() => toggleSource(key)}
                    disabled={merging}
                  />
                  <span class="row-name">{key}</span>
                  <span class="row-sub">
                    {m.domain_label} · {m.node_count} nodes ·
                    {m.fit_mode}
                  </span>
                </label>
              </li>
            {/each}
          </ul>
        </fieldset>

        <label class="row-label">
          <span>target name</span>
          <input
            type="text"
            class="text-input"
            placeholder="combined"
            aria-label="merged manifold name (under local/)"
            bind:value={targetName}
            disabled={merging}
          />
        </label>

        <label class="row-label">
          <span>fit mode</span>
          <select
            class="text-input"
            aria-label="merged manifold fit mode"
            bind:value={fitMode}
            disabled={merging || selected.size === 0}
          >
            {#if sourceModes.length <= 1}
              <option value="">inherit ({sourceModes[0] ?? "—"})</option>
            {/if}
            <option value="pca">pca</option>
            <option value="spectral">spectral</option>
          </select>
        </label>
        {#if sourceModes.length > 1}
          <p class="warn">
            sources have mixed fit_modes ({sourceModes.join(", ")})
            — pick one explicitly above.
          </p>
        {/if}

        <div class="actions">
          <button
            type="submit"
            class="act submit"
            disabled={!canSubmit}
            title={selected.size < 2
              ? "pick >= 2 sources"
              : !targetName.trim()
                ? "target name required"
                : "merge into a fresh discover folder"}
          >
            {merging
              ? "merging…"
              : `merge ${selected.size} → local/${targetName.trim() || "…"}`}
          </button>
        </div>
      </form>
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
  .close:hover { color: var(--accent-red); }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-4) var(--space-5) var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
  }
  .hint {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.5;
  }
  .hint strong {
    color: var(--accent-purple);
    font-weight: var(--weight-medium);
  }
  .muted {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
    line-height: 1.4;
  }
  .muted strong {
    color: var(--accent-purple);
    font-weight: var(--weight-medium);
  }
  .form {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
  .sources {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-deep);
    padding: var(--space-3) var(--space-4);
    margin: 0;
  }
  .sources legend {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    padding: 0 var(--space-2);
  }
  .optional {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .source-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    max-height: 18em;
    overflow-y: auto;
  }
  .source-label {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    cursor: pointer;
    padding: var(--space-2) var(--space-2);
    border-radius: var(--radius);
    font-size: var(--text-sm);
  }
  .source-label:hover { background: var(--bg-elev); }
  .row-name {
    color: var(--fg-strong);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row-sub {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .row-label {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    font-size: var(--text-sm);
    color: var(--fg-muted);
  }
  .row-label > span {
    flex: 0 0 6em;
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
  .text-input:focus-visible {
    outline: 1px solid var(--accent-purple);
    outline-offset: -1px;
  }
  .warn {
    margin: 0;
    color: var(--accent-yellow);
    font-size: var(--text-xs);
  }
  .actions {
    display: flex;
    justify-content: flex-end;
  }
  .act {
    background: transparent;
    color: var(--accent-purple);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .act:hover:not(:disabled) {
    background: rgba(167, 139, 250, 0.12);
    border-color: var(--accent-purple);
  }
  .act:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
</style>
