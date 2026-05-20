<script lang="ts">
  // Transcript export / import drawer — phase 5.  Two tabs:
  //
  //   * export — render the path ending at the chosen node as
  //     transcript YAML and offer it as a .yaml download.  Defaults
  //     to the active node; the user can pick any node id via a
  //     short id-prefix.
  //   * import — paste YAML or upload a file, pick a mode
  //     (default / here / merge), tick strict, fire the load.
  //     Guard warnings (model / system-prompt / probe drift) surface
  //     in a banner with the diff list.

  import { apiTree, ApiError } from "../lib/api";
  import {
    closeDrawer,
    loomTree,
    refreshLoomTree,
  } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  type Tab = "export" | "import";
  let tab: Tab = $state("export");

  // -------------------------------------------------- export state ---

  let exportTargetId = $state(""); // empty = active node
  let exportYaml = $state("");
  let exportError: string | null = $state(null);
  let exportBusy = $state(false);
  let exportLeafId: string | null = $state(null);

  async function runExport(): Promise<void> {
    if (exportBusy) return;
    exportBusy = true;
    exportError = null;
    exportYaml = "";
    exportLeafId = null;
    try {
      const id = exportTargetId.trim();
      let resolved: string | null = null;
      if (id) {
        const r = resolveByPrefix(id);
        if (r.id) {
          resolved = r.id;
        } else if (r.matches.length === 0) {
          exportError = `no node matches prefix "${id}"`;
          return;
        } else {
          const preview = r.matches.slice(0, 6).map((s) => s.slice(0, 8)).join(", ");
          exportError =
            `ambiguous: ${r.matches.length} matches (${preview}` +
            (r.matches.length > 6 ? ", …" : "") + ")";
          return;
        }
      }
      const r = await apiTree.transcriptExport(resolved);
      exportYaml = r.yaml;
      exportLeafId = r.node_id;
    } catch (e) {
      exportError = describeError(e);
    } finally {
      exportBusy = false;
    }
  }

  function downloadYaml(): void {
    if (!exportYaml) return;
    const blob = new Blob([exportYaml], {
      type: "application/yaml;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const tsId = exportLeafId ? exportLeafId.slice(0, 8) : "active";
    a.download = `saklas-transcript-${tsId}.yaml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async function copyYaml(): Promise<void> {
    if (!exportYaml) return;
    try {
      await navigator.clipboard.writeText(exportYaml);
    } catch {
      // Clipboard unavailable — leave the textarea content for the user.
    }
  }

  interface PrefixResolution {
    /** Single unambiguous match — caller can use immediately. */
    id: string | null;
    /** All node ids whose ulid starts with the prefix; ``length > 1``
     *  signals ambiguity. */
    matches: string[];
  }

  function resolveByPrefix(prefix: string): PrefixResolution {
    const p = prefix.trim();
    if (!p) return { id: null, matches: [] };
    if (p === "root") {
      const root = loomTree.root_id;
      return root ? { id: root, matches: [root] } : { id: null, matches: [] };
    }
    const matches: string[] = [];
    for (const id of loomTree.nodes.keys()) {
      if (id === p) return { id, matches: [id] };
      if (id.startsWith(p)) matches.push(id);
    }
    if (matches.length === 1) return { id: matches[0], matches };
    return { id: null, matches };
  }

  // -------------------------------------------------- import state ---

  let importYaml = $state("");
  let importMode: "default" | "here" | "merge" = $state("default");
  let importStrict = $state(false);
  let importError: string | null = $state(null);
  let importBusy = $state(false);
  let importGuards: string[] = $state([]);
  let importLeafId: string | null = $state(null);

  let fileInputRef: HTMLInputElement | null = $state(null);

  async function onFileChange(ev: Event): Promise<void> {
    const target = ev.target as HTMLInputElement;
    const file = target.files?.[0];
    if (!file) return;
    importYaml = await file.text();
  }

  async function runImport(): Promise<void> {
    if (importBusy) return;
    const yaml = importYaml.trim();
    if (!yaml) {
      importError = "paste a transcript YAML or upload a file";
      return;
    }
    importBusy = true;
    importError = null;
    importGuards = [];
    importLeafId = null;
    try {
      const r = await apiTree.transcriptLoad(yaml, importMode, importStrict);
      importGuards = r.guards;
      importLeafId = r.leaf_id;
      // Refresh the tree so the new branch shows up in the sidebar.
      await refreshLoomTree();
    } catch (e) {
      importError = describeError(e);
    } finally {
      importBusy = false;
    }
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
</script>

<section class="drawer-shell" aria-label="Transcript drawer">
  <header class="header">
    <span class="title">transcript</span>
    <div class="tabs" role="tablist">
      <button
        type="button"
        class="tab"
        class:active={tab === "export"}
        role="tab"
        aria-selected={tab === "export"}
        onclick={() => (tab = "export")}
      >export</button>
      <button
        type="button"
        class="tab"
        class:active={tab === "import"}
        role="tab"
        aria-selected={tab === "import"}
        onclick={() => (tab = "import")}
      >import</button>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button>
  </header>

  <div class="body">
    {#if tab === "export"}
      <p class="hint">
        Export the path ending at the chosen node as a saklas transcript
        YAML. Leave the field blank to export the active path.
      </p>

      <label class="field">
        <span class="label">node id (or prefix)</span>
        <input
          type="text"
          class="input"
          bind:value={exportTargetId}
          placeholder={`active: ${loomTree.active_node_id?.slice(0, 12) ?? "—"}`}
          autocomplete="off"
          spellcheck="false"
        />
      </label>

      <div class="form-actions">
        <button
          type="button"
          class="btn primary"
          onclick={runExport}
          disabled={exportBusy}
        >{exportBusy ? "rendering…" : "render YAML"}</button>
      </div>

      {#if exportError}
        <p class="error" role="alert">{exportError}</p>
      {/if}

      {#if exportYaml}
        <textarea
          class="yaml"
          readonly
          rows="20"
          value={exportYaml}
        ></textarea>
        <div class="form-actions">
          <button type="button" class="btn" onclick={copyYaml}>copy</button>
          <button type="button" class="btn primary" onclick={downloadYaml}
            >download .yaml</button>
        </div>
      {/if}
    {:else}
      <p class="hint">
        Paste YAML below or upload a file. Pick a mode and run.
      </p>

      <div class="field">
        <span class="label">mode</span>
        <label class="mode-opt">
          <input
            type="radio"
            name="import-mode"
            value="default"
            checked={importMode === "default"}
            onchange={() => (importMode = "default")}
          />
          <span><strong>default</strong>: attach at root (fresh branch)</span>
        </label>
        <label class="mode-opt">
          <input
            type="radio"
            name="import-mode"
            value="here"
            checked={importMode === "here"}
            onchange={() => (importMode = "here")}
          />
          <span><strong>here</strong>: attach at the active node</span>
        </label>
        <label class="mode-opt">
          <input
            type="radio"
            name="import-mode"
            value="merge"
            checked={importMode === "merge"}
            onchange={() => (importMode = "merge")}
          />
          <span><strong>merge</strong>: deepest user-turn prefix match</span>
        </label>
      </div>

      <label class="mode-opt">
        <input
          type="checkbox"
          bind:checked={importStrict}
        />
        <span>strict (refuse on probe-hash drift)</span>
      </label>

      <div class="field">
        <span class="label">file (optional)</span>
        <input
          type="file"
          accept=".yaml,.yml,application/x-yaml,text/yaml"
          bind:this={fileInputRef}
          onchange={onFileChange}
        />
      </div>

      <textarea
        class="yaml"
        rows="14"
        bind:value={importYaml}
        placeholder="paste transcript YAML here…"
        spellcheck="false"
      ></textarea>

      <div class="form-actions">
        <button
          type="button"
          class="btn primary"
          onclick={runImport}
          disabled={importBusy}
        >{importBusy ? "importing…" : "import"}</button>
      </div>

      {#if importError}
        <p class="error" role="alert">{importError}</p>
      {/if}

      {#if importGuards.length > 0}
        <div class="banner">
          <span class="banner-title">guards triggered</span>
          <ul>
            {#each importGuards as g (g)}
              <li>{g}</li>
            {/each}
          </ul>
          <p class="hint">
            These have been stamped as notes on the imported branch's
            root node so the sidebar can flag them in context.
          </p>
        </div>
      {/if}
      {#if importLeafId && !importError}
        <p class="ok">
          imported successfully, leaf node
          <code>{importLeafId.slice(0, 12)}</code>
        </p>
      {/if}
    {/if}
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={closeDrawer}>close</button>
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
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-5) var(--space-6);
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent-blue);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .tabs {
    display: flex;
    gap: var(--space-2);
    flex: 1 1 auto;
    justify-content: center;
  }
  .tab {
    background: transparent;
    color: var(--fg-dim);
    border: 1px solid var(--border);
    padding: var(--space-1) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .tab:hover {
    color: var(--fg-strong);
  }
  .tab.active {
    color: var(--accent-blue);
    border-color: var(--accent);
    background: var(--accent-subtle);
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    cursor: pointer;
    padding: var(--space-2) var(--space-2);
    font-size: var(--text);
    line-height: 1;
  }
  .close:hover {
    color: var(--accent-red);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    min-height: 0;
  }
  .hint {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    margin: 0;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    text-transform: lowercase;
  }
  .input {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .yaml {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    resize: vertical;
    min-height: 8em;
    white-space: pre;
    overflow: auto;
  }
  .mode-opt {
    display: inline-flex;
    gap: var(--space-2);
    align-items: center;
    color: var(--fg-strong);
    font-size: var(--text-sm);
    margin: var(--space-1) 0;
  }
  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
  }
  .error {
    color: var(--accent-red);
    font-size: var(--text-sm);
    margin: 0;
  }
  .ok {
    color: var(--accent-green);
    font-size: var(--text-sm);
    margin: 0;
  }
  .banner {
    background: rgba(210, 153, 34, 0.12);
    border: 1px solid var(--accent-yellow);
    padding: var(--space-4) var(--space-4);
    color: var(--accent-yellow);
    font-size: var(--text-sm);
  }
  .banner-title {
    text-transform: lowercase;
    font-weight: var(--weight-medium);
    display: block;
    margin-bottom: var(--space-1);
  }
  .banner ul {
    margin: var(--space-1) 0 var(--space-1) 1em;
    padding: 0;
  }
  .footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
    padding: var(--space-5) var(--space-6);
    border-top: 1px solid var(--border);
  }
  .btn {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-2) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .btn:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
  .btn.primary {
    background: var(--accent);
    color: var(--text-on-accent);
    border-color: var(--accent);
  }
  .btn.primary:hover:not(:disabled) {
    background: var(--accent-light);
    border-color: var(--accent-light);
  }
  .btn.primary:disabled {
    background: var(--bg-elev);
  }
</style>
