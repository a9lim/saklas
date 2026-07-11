<script lang="ts">
  // ManifoldMergeDrawer — discover-mode node-union merge.
  //
  // Unions the *node corpora* of two or more discover-mode manifolds and
  // lets the next autofit derive coords from the combined heap.  Restricted
  // to discover-mode sources by design — authored manifolds carry
  // user-declared geometry that isn't mergeable without a shared coordinate
  // system.
  //
  // Reached from the workspace rail's "manifolds → merge manifolds…" entry.

  import { onMount } from "svelte";
  import { SvelteSet } from "svelte/reactivity";
  import { ApiError, apiManifolds } from "../lib/api";
  import {
    closeDrawer,
    steerRack,
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
    steerRack.catalog.filter((m) => isDiscoverMode(m)),
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
      const m = steerRack.catalog.find((x) => rowKey(x) === key);
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

    {#if steerRack.unavailable}
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

        <label class="field">
          <span class="label">target name</span>
          <input
            type="text"
            placeholder="combined"
            aria-label="merged manifold name (under local/)"
            bind:value={targetName}
            disabled={merging}
            autocomplete="off"
            spellcheck="false"
          />
        </label>

        <label class="field">
          <span class="label">fit mode</span>
          <select
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

        <footer class="foot">
          <button type="button" class="secondary" onclick={closeDrawer}
            >cancel</button>
          <button
            type="submit"
            class="primary"
            disabled={!canSubmit}
            title={selected.size < 2
              ? "pick >= 2 sources"
              : !targetName.trim()
                ? "target name required"
                : "merge into a fresh discover folder"}
          >
            {#if merging}
              <span class="spinner" aria-hidden="true"></span> merging…
            {:else}
              merge {selected.size} → local/{targetName.trim() || "…"}
            {/if}
          </button>
        </footer>
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
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-5) var(--space-6);
    border-bottom: 1px solid var(--glass-line);
  }
  .title {
    color: var(--accent);
    letter-spacing: 0;
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  .close {
    background: transparent;
    color: var(--fg-muted);
    border: 1px solid var(--border);
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font: inherit;
    font-size: var(--text-md);
    line-height: 1;
    cursor: pointer;
    flex: none;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .close:hover {
    color: var(--fg);
    background: var(--bg-hover);
    border-color: var(--fg-muted);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-5) var(--space-6);
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
    flex: 1 1 auto;
    min-height: 0;
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
    font-family: var(--font-mono);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row-sub {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  /* Stacked form fields — label above control. */
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .label {
    color: var(--fg-dim);
    font-size: var(--text-sm);
    text-transform: lowercase;
    letter-spacing: 0;
    padding: 0;
  }
  input[type="text"],
  select {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    font-family: var(--font-mono);
    font-size: var(--text);
    box-sizing: border-box;
    width: 100%;
  }
  input[type="text"]:focus,
  select:focus {
    outline: 1px solid var(--accent-purple);
    border-color: var(--accent-purple);
  }
  .warn {
    margin: 0;
    color: var(--accent-yellow);
    font-size: var(--text-xs);
  }

  /* Footer: cancel (secondary) + primary submit pinned to the bottom.
   * Only the primary accent goes purple. */
  .foot {
    border-top: 1px solid var(--glass-line);
    padding-top: var(--space-3);
    margin-top: auto;
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
  }
  .primary {
    background: var(--accent-purple);
    color: var(--text-on-accent);
    border: 1px solid var(--accent-purple);
    padding: var(--space-2) var(--space-5);
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: var(--space-3);
  }
  .primary:hover:not(:disabled) {
    filter: brightness(1.1);
  }
  .primary:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .secondary {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-dim);
    padding: var(--space-2) var(--space-5);
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .secondary:hover {
    border-color: var(--fg);
    color: var(--fg);
  }
  .spinner {
    width: 0.7em;
    height: 0.7em;
    border-radius: 50%;
    border: 1px solid var(--bg-deep);
    border-right-color: transparent;
    animation: spin 0.7s linear infinite;
    display: inline-block;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
