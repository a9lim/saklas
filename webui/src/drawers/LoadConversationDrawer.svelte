<script lang="ts">
  // Load-conversation drawer — restore from a previously-saved JSON blob.
  // Mirrors SaveConversationDrawer's v4 whole-tree wire shape exactly.
  // Older active-path-only files are rejected rather than pretending to
  // restore state the server does not own.

  import {
    addSubspaceToRack,
    setSubspaceLabel,
    setSubspaceCoords,
    setSubspaceVariant,
    setSubspaceTrigger,
    setSubspaceEnabled,
    setSubspaceAlong,
    addManifoldToRack,
    setManifoldLabel,
    setManifoldCoords,
    setManifoldBlend,
    setManifoldOnto,
    setManifoldTrigger,
    setManifoldEnabled,
    samplingState,
    setSampling,
    setHighlightTarget,
    setCompareTarget,
    highlightState,
    closeDrawer,
    refreshVectorList,
    refreshLoomTree,
    vectorsState,
    steerRack,
    probeRack,
    attachProbe,
    detachProbe,
    setProbeSortMode,
  } from "../lib/stores.svelte";
  import { apiTree } from "../lib/api";
  import type { LoomTreeJSON, ProbeSortMode, Trigger, Variant } from "../lib/types";
  import type { SamplingState } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  // Local state for the picker.
  let fileInputEl: HTMLInputElement | null = $state(null);
  let parsed: SnapshotShape | null = $state(null);
  let parseError: string | null = $state(null);
  let warnings: string[] = $state([]);
  let appliedSummary: string | null = $state(null);
  let applying = $state(false);

  interface SteerRowShape {
    name: string;
    mode: "subspace" | "manifold";
    coords: number[];
    label: string | null;
    variant?: Variant;
    blend?: number;
    onto?: number;
    trigger: Trigger;
    enabled: boolean;
  }

  interface SnapshotShape {
    version: 4;
    savedAt: string;
    model_id: string;
    session_id: string;
    tree: LoomTreeJSON;
    steerRack: SteerRowShape[];
    subspaceAlong: number;
    probeRack: {
      sortMode: string;
      active: string[];
      entries: Array<{
        name: string;
        sparkline: number[];
        current: number;
        previous: number;
      }>;
    };
    highlightState: {
      target: string | null;
      compareTarget: string | null;
      compareTwo: boolean;
      smoothBlend: boolean;
    };
    samplingState: SamplingState;
  }

  function isSnapshotShape(v: unknown): v is SnapshotShape {
    if (!v || typeof v !== "object") return false;
    const obj = v as Record<string, unknown>;
    if (obj.version !== 4 || typeof obj.savedAt !== "string") return false;
    if (typeof obj.model_id !== "string" || typeof obj.session_id !== "string") return false;
    if (!obj.tree || typeof obj.tree !== "object" || !Array.isArray(obj.steerRack)) return false;
    const tree = obj.tree as Record<string, unknown>;
    if (typeof tree.tree_format !== "number"
      || typeof tree.saklas_version !== "string"
      || typeof tree.root_id !== "string"
      || typeof tree.active_node_id !== "string"
      || typeof tree.rev !== "number"
      || tree.model_id !== obj.model_id
      || !Array.isArray(tree.nodes)
      || !tree.children_of || typeof tree.children_of !== "object"
      || !tree.cast || typeof tree.cast !== "object") return false;
    if (typeof obj.subspaceAlong !== "number") return false;
    if (!obj.probeRack || typeof obj.probeRack !== "object") return false;
    const probes = obj.probeRack as Record<string, unknown>;
    if (!(probes.sortMode === "name" || probes.sortMode === "value" || probes.sortMode === "change")) return false;
    if (!Array.isArray(probes.active) || !probes.active.every((x) => typeof x === "string")) return false;
    if (!Array.isArray(probes.entries) || !probes.entries.every((raw) => {
      if (!raw || typeof raw !== "object") return false;
      const entry = raw as Record<string, unknown>;
      return typeof entry.name === "string"
        && Array.isArray(entry.sparkline)
        && entry.sparkline.every((x) => typeof x === "number")
        && typeof entry.current === "number"
        && typeof entry.previous === "number";
    })) return false;
    if (!obj.highlightState || typeof obj.highlightState !== "object") return false;
    const highlight = obj.highlightState as Record<string, unknown>;
    if (!(typeof highlight.target === "string" || highlight.target === null)
      || !(typeof highlight.compareTarget === "string" || highlight.compareTarget === null)
      || typeof highlight.compareTwo !== "boolean"
      || typeof highlight.smoothBlend !== "boolean") return false;
    if (!obj.samplingState || typeof obj.samplingState !== "object") return false;
    const sampling = obj.samplingState as Record<string, unknown>;
    const expectedSamplingKeys = Object.keys(samplingState).sort();
    if (Object.keys(sampling).sort().join("\0") !== expectedSamplingKeys.join("\0")) return false;
    if (!obj.steerRack.every((raw) => {
      if (!raw || typeof raw !== "object") return false;
      const row = raw as Record<string, unknown>;
      const common = typeof row.name === "string"
        && (row.mode === "subspace" || row.mode === "manifold")
        && Array.isArray(row.coords)
        && row.coords.every((x) => typeof x === "number")
        && (typeof row.label === "string" || row.label === null)
        && typeof row.trigger === "string"
        && typeof row.enabled === "boolean";
      if (!common) return false;
      return row.mode === "subspace"
        ? typeof row.variant === "string"
        : typeof row.blend === "number" && typeof row.onto === "number";
    })) return false;
    return true;
  }

  async function onFileChange(ev: Event): Promise<void> {
    parseError = null;
    parsed = null;
    warnings = [];
    appliedSummary = null;
    const target = ev.currentTarget as HTMLInputElement;
    const file = target.files?.[0] ?? null;
    if (!file) return;
    let text: string;
    try {
      text = await file.text();
    } catch (e) {
      parseError = `read failed: ${e instanceof Error ? e.message : String(e)}`;
      return;
    }
    let json: unknown;
    try {
      json = JSON.parse(text);
    } catch (e) {
      parseError = `parse failed: ${e instanceof Error ? e.message : String(e)}`;
      return;
    }
    if (!isSnapshotShape(json)) {
      parseError =
        "unsupported conversation file: expected the complete saklas conversation schema version 4";
      return;
    }
    parsed = json;
  }

  async function applySnapshot(): Promise<void> {
    if (!parsed) return;
    warnings = [];
    appliedSummary = null;
    let appliedTurns = 0;
    let appliedTerms = 0;
    let skippedTerms = 0;
    let appliedSampling = 0;

    applying = true;
    try {
      await apiTree.restore(parsed.tree);
      await refreshLoomTree();
      await refreshVectorList();
      appliedTurns = parsed.tree.nodes.filter((node) => node.parent_id !== null).length;

      const steerRows = parsed.steerRack;
      steerRack.entries.clear();
      setSubspaceAlong(parsed.subspaceAlong);
      for (const row of steerRows) {
        const name = row.name;
        if (row.mode === "manifold") {
          addManifoldToRack(name);
          if (row.label !== null) setManifoldLabel(name, row.label);
          else setManifoldCoords(name, row.coords);
          if (typeof row.blend === "number") setManifoldBlend(name, row.blend);
          if (typeof row.onto === "number") setManifoldOnto(name, row.onto);
          setManifoldTrigger(name, row.trigger);
          if (row.enabled === false) setManifoldEnabled(name, false);
          appliedTerms++;
          continue;
        }
        addSubspaceToRack(name);
        if (row.label !== null) setSubspaceLabel(name, row.label);
        else setSubspaceCoords(name, row.coords);
        if (typeof row.variant === "string") setSubspaceVariant(name, row.variant as Variant);
        setSubspaceTrigger(name, row.trigger);
        if (row.enabled === false) setSubspaceEnabled(name, false);
        appliedTerms++;
      }
      // Sanity-check against the server's known set; surface a warning
      // for terms that aren't currently registered.  Informational only —
      // the rack carries them as-is.
      try {
        const known = vectorsState.names;
        for (const row of steerRows) {
          if (typeof row?.name !== "string") continue;
          if (known.length > 0 && !known.includes(row.name)) {
            skippedTerms++;
            warnings.push(
              `'${row.name}' not registered server-side; present in rack but won't apply at gen time`,
            );
          }
        }
      } catch {
        /* ignore */
      }
      for (const [k, v] of Object.entries(parsed.samplingState)) {
        if (k in samplingState) {
          setSampling(k as keyof SamplingState, v as never);
          appliedSampling++;
        }
      }

      const hs = parsed.highlightState;
      setHighlightTarget(hs.target);
      setCompareTarget(hs.compareTarget);
      highlightState.compareTwo = hs.compareTwo;
      highlightState.smoothBlend = hs.smoothBlend;

      for (const name of [...probeRack.active]) await detachProbe(name);
      setProbeSortMode(parsed.probeRack.sortMode as ProbeSortMode);
      const savedProbeEntries = new Map(
        parsed.probeRack.entries.map((entry) => [entry.name, entry]),
      );
      for (const name of parsed.probeRack.active) {
        await attachProbe(name);
        const entry = probeRack.entries.get(name);
        const saved = savedProbeEntries.get(name);
        if (entry && saved) {
          probeRack.entries.set(name, {
            ...entry,
            sparkline: [...saved.sparkline],
            current: saved.current,
            previous: saved.previous,
          });
        }
      }

      appliedSummary = `restored ${appliedTurns} turn${appliedTurns === 1 ? "" : "s"}, ${appliedTerms} term${appliedTerms === 1 ? "" : "s"}${skippedTerms ? ` (${skippedTerms} not server-known)` : ""}, ${appliedSampling} sampling field${appliedSampling === 1 ? "" : "s"}`;
    } catch (e) {
      parseError = `restore failed: ${e instanceof Error ? e.message : String(e)}`;
    } finally {
      applying = false;
    }
  }
</script>

<section class="drawer-shell" aria-label="Load conversation drawer">
  <header class="header">
    <span class="title">load conversation</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    <p class="hint">model must match</p>

    <label class="field">
      <span class="label">file</span>
      <input
        type="file"
        accept=".json,application/json"
        bind:this={fileInputEl}
        onchange={onFileChange}
        class="file"
      />
    </label>

    {#if parseError}
      <p class="error" role="alert">{parseError}</p>
    {/if}

    {#if parsed}
      <div class="parsed-info">
        <span class="meta">
          saved {parsed.savedAt} · model {parsed.model_id}
        </span>
        <ul class="counts">
          <li>turns: {parsed.tree.nodes.filter((node) => node.parent_id !== null).length}</li>
          <li>terms: {parsed.steerRack.length}</li>
          <li>probes: {parsed.probeRack.active.length}</li>
          <li>sampling: {Object.keys(parsed.samplingState).length}</li>
        </ul>
      </div>
    {/if}

    {#if warnings.length > 0}
      <div class="warnings" role="alert">
        <span class="label">warnings</span>
        {#each warnings as w, i (i)}
          <div class="warn-line">· {w}</div>
        {/each}
      </div>
    {/if}

    {#if appliedSummary}
      <p class="success">{appliedSummary}</p>
    {/if}
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={closeDrawer}>
      {appliedSummary ? "done" : "cancel"}
    </button>
    <button
      type="button"
      class="btn primary"
      disabled={!parsed || applying}
      onclick={applySnapshot}
    >{applying ? "restoring…" : "restore"}</button>
  </footer>
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
  }
  .title {
    color: var(--accent);
    text-transform: lowercase;
    letter-spacing: 0;
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  .close {
    background: var(--glass);
    color: var(--fg-muted);
    border: 1px solid transparent;
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
      background var(--dur-fast) var(--ease-out);
  }
  .close:hover {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
  }
  .hint {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.4;
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
  .file {
    color: var(--fg);
    font: inherit;
    font-family: var(--font-mono);
  }
  .error {
    color: var(--accent-error);
    font-size: var(--text-sm);
    margin: 0;
    word-break: break-word;
  }
  .success {
    color: var(--accent-green);
    font-size: var(--text-sm);
    margin: 0;
  }
  .parsed-info {
    background: var(--bg-deep);
    padding: var(--space-3) var(--space-4);
  }
  .meta {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  .counts {
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    color: var(--fg-strong);
    font-size: var(--text-sm);
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-1) var(--space-5);
  }
  .warnings {
    background: color-mix(in srgb, var(--accent-yellow) 12%, var(--bg-deep));
    padding: var(--space-2) var(--space-4);
    color: var(--accent-yellow);
    font-size: var(--text-sm);
    line-height: 1.4;
    max-height: 180px;
    overflow-y: auto;
  }
  .warn-line {
    margin-top: var(--space-1);
  }
  .footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-6);
    color: var(--fg-muted);
  }
  .btn {
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    padding: var(--space-3) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    background: var(--glass-strong);
  }
  .btn:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
  .btn.primary {
    background: var(--accent);
    color: var(--text-on-accent);
    border-color: transparent;
  }
  .btn.primary:hover:not(:disabled) {
    background: var(--accent-light);
    border-color: transparent;
  }
  .btn.primary:disabled {
    background: var(--bg-elev);
  }
</style>
