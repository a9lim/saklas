<script lang="ts">
  // Load-conversation drawer — restore from a previously-saved JSON blob.
  // Mirrors SaveConversationDrawer's wire shape exactly.  Unknown / missing
  // sections are tolerated; warnings surface inline so the user knows what
  // didn't apply.

  import {
    chatLog,
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
    vectorsState,
  } from "../lib/stores.svelte";
  import { polesOf } from "../lib/concepts";
  import type { ChatTurn, Trigger, Variant } from "../lib/types";
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

  /** One persisted steer row — the v2 ``steerRack`` shape (mode + position),
   *  with the legacy v1 ``vectorRack`` vector fields (``alpha`` / ``projection``
   *  / ``ablate``) kept optional so old saves still load. */
  interface SteerRowShape {
    name: string;
    mode?: "subspace" | "manifold";
    coords?: number[];
    label?: string | null;
    variant?: Variant;
    blend?: number;
    onto?: number;
    trigger?: Trigger;
    enabled?: boolean;
    // legacy v1 (vector) fields:
    alpha?: number;
    projection?: unknown;
    ablate?: boolean;
  }

  /** Loose snapshot shape — matches the writer's output but every field
   * is optional so partial / older saves still load opportunistically. */
  interface SnapshotShape {
    version?: number;
    savedAt?: string;
    model_id?: string | null;
    chatLog?: ChatTurn[];
    /** v2: the full steer rack (subspace + manifold entries). */
    steerRack?: SteerRowShape[];
    /** v2: the shared subspace-along master. */
    subspaceAlong?: number;
    /** v1 legacy: vector-only rack array. */
    vectorRack?: SteerRowShape[];
    probeRack?: {
      sortMode?: string;
      active?: string[];
      entries?: Array<{
        name: string;
        sparkline?: number[];
        current?: number;
        previous?: number;
      }>;
    };
    highlightState?: {
      target?: string | null;
      compareTarget?: string | null;
      compareTwo?: boolean;
      smoothBlend?: boolean;
    };
    samplingState?: Partial<SamplingState>;
  }

  function isSnapshotShape(v: unknown): v is SnapshotShape {
    if (!v || typeof v !== "object") return false;
    const obj = v as Record<string, unknown>;
    // Tolerate missing chatLog as long as at least one of the recognized
    // sections is present — older saves might be sampling-only.
    return (
      "chatLog" in obj ||
      "steerRack" in obj ||
      "vectorRack" in obj ||
      "probeRack" in obj ||
      "samplingState" in obj ||
      "highlightState" in obj
    );
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
        "unrecognized format, expected a saklas conversation JSON with at least one of {chatLog, vectorRack, probeRack, samplingState, highlightState}";
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

    // Refresh server-known vectors so we can warn about missing ones
    // before we try to add them.  This call is best-effort — failure
    // shouldn't block the local restore.
    try {
      await refreshVectorList();
    } catch {
      /* ignore — restore will still attempt with whatever is local */
    }

    if (Array.isArray(parsed.chatLog)) {
      chatLog.turns = parsed.chatLog;
      appliedTurns = parsed.chatLog.length;
    } else {
      warnings.push("chatLog missing or invalid, skipped");
    }

    // v2 ``steerRack`` (subspace + manifold) wins; fall back to the legacy
    // v1 ``vectorRack`` (vector-only) and convert each pole to a subspace
    // term toward its signed pole, the magnitude → the shared master.
    const steerRows = Array.isArray(parsed.steerRack)
      ? parsed.steerRack
      : Array.isArray(parsed.vectorRack)
        ? parsed.vectorRack
        : null;
    if (steerRows) {
      if (typeof parsed.subspaceAlong === "number") {
        setSubspaceAlong(parsed.subspaceAlong);
      }
      let firstLegacyAlong = true;
      for (const row of steerRows) {
        if (!row || typeof row !== "object" || typeof row.name !== "string") {
          warnings.push("steerRack: skipped malformed entry");
          continue;
        }
        const name = row.name;
        if (row.mode === "manifold") {
          addManifoldToRack(name);
          if (typeof row.label === "string") setManifoldLabel(name, row.label);
          else if (Array.isArray(row.coords)) setManifoldCoords(name, row.coords);
          if (typeof row.blend === "number") setManifoldBlend(name, row.blend);
          if (typeof row.onto === "number") setManifoldOnto(name, row.onto);
          if (typeof row.trigger === "string") setManifoldTrigger(name, row.trigger as Trigger);
          if (row.enabled === false) setManifoldEnabled(name, false);
          appliedTerms++;
          continue;
        }
        // subspace (v2) OR legacy v1 vector row.
        addSubspaceToRack(name);
        if (row.mode === "subspace") {
          if (typeof row.label === "string") setSubspaceLabel(name, row.label);
          else if (Array.isArray(row.coords)) setSubspaceCoords(name, row.coords);
        } else {
          // legacy vector → subspace toward the signed pole; |alpha| → master.
          if (typeof row.alpha === "number") {
            if (firstLegacyAlong) {
              setSubspaceAlong(Math.abs(row.alpha));
              firstLegacyAlong = false;
            }
            if (row.alpha < 0) {
              const neg = polesOf(name).negative;
              if (neg) setSubspaceLabel(name, neg);
            }
          }
          if (row.projection || row.ablate) {
            warnings.push(
              `'${name}': projection/ablation dropped (no longer authorable in the rack)`,
            );
          }
        }
        if (typeof row.variant === "string") setSubspaceVariant(name, row.variant as Variant);
        if (typeof row.trigger === "string") setSubspaceTrigger(name, row.trigger as Trigger);
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
    } else {
      warnings.push("steerRack / vectorRack missing or invalid, skipped");
    }

    if (parsed.samplingState && typeof parsed.samplingState === "object") {
      for (const [k, v] of Object.entries(parsed.samplingState)) {
        if (k in samplingState) {
          // setSampling is typed; cast at the boundary.
          setSampling(k as keyof SamplingState, v as never);
          appliedSampling++;
        }
      }
    }

    if (parsed.highlightState && typeof parsed.highlightState === "object") {
      const hs = parsed.highlightState;
      if (hs.target !== undefined) setHighlightTarget(hs.target ?? null);
      if (hs.compareTarget !== undefined) setCompareTarget(hs.compareTarget ?? null);
      if (typeof hs.compareTwo === "boolean")
        highlightState.compareTwo = hs.compareTwo;
      if (typeof hs.smoothBlend === "boolean")
        highlightState.smoothBlend = hs.smoothBlend;
    }

    appliedSummary = `restored ${appliedTurns} turn${appliedTurns === 1 ? "" : "s"}, ${appliedTerms} term${appliedTerms === 1 ? "" : "s"}${skippedTerms ? ` (${skippedTerms} not server-known)` : ""}, ${appliedSampling} sampling field${appliedSampling === 1 ? "" : "s"}`;
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
    <p class="hint">
      restore from a saklas conversation JSON file.  Vectors must already be
      registered on the server for steering to take effect; missing names
      stay in the rack but won't apply at gen time.
    </p>

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
          {parsed.savedAt ? `saved ${parsed.savedAt}` : "saved (no timestamp)"}
          {#if parsed.model_id} · model {parsed.model_id}{/if}
        </span>
        <ul class="counts">
          <li>turns: {parsed.chatLog?.length ?? 0}</li>
          <li>terms: {parsed.steerRack?.length ?? parsed.vectorRack?.length ?? 0}</li>
          <li>probes: {parsed.probeRack?.active?.length ?? 0}</li>
          <li>sampling fields: {parsed.samplingState ? Object.keys(parsed.samplingState).length : 0}</li>
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
      disabled={!parsed}
      onclick={applySnapshot}
    >restore</button>
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
    border-bottom: 1px solid var(--glass-line);
  }
  .title {
    color: var(--accent);
    text-transform: lowercase;
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
    border: 1px solid var(--border);
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
    background: var(--bg-deep);
    border: 1px solid var(--accent-yellow);
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
    border-top: 1px solid var(--glass-line);
    color: var(--fg-muted);
  }
  .btn {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-5);
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
    border-color: var(--border);
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
