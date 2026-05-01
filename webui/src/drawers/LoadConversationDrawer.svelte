<script lang="ts">
  // Load-conversation drawer — restore from a previously-saved JSON blob.
  // Mirrors SaveConversationDrawer's wire shape exactly.  Unknown / missing
  // sections are tolerated; warnings surface inline so the user knows what
  // didn't apply.

  import {
    chatLog,
    addVectorToRack,
    setVectorAlpha,
    setVectorTrigger,
    setVectorVariant,
    setVectorProjection,
    setVectorAblate,
    setVectorEnabled,
    samplingState,
    setSampling,
    setHighlightTarget,
    setCompareTarget,
    highlightState,
    closeDrawer,
    refreshVectorList,
  } from "../lib/stores.svelte";
  import type {
    ChatTurn,
    ProjectionSpec,
    Trigger,
    Variant,
  } from "../lib/types";
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

  /** Loose snapshot shape — matches the writer's output but every field
   * is optional so partial / older saves still load opportunistically. */
  interface SnapshotShape {
    version?: number;
    savedAt?: string;
    model_id?: string | null;
    chatLog?: ChatTurn[];
    vectorRack?: Array<{
      name: string;
      alpha?: number;
      trigger?: Trigger;
      variant?: Variant;
      projection?: ProjectionSpec | null;
      ablate?: boolean;
      enabled?: boolean;
    }>;
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
        "unrecognized format — expected a saklas conversation JSON with at least one of {chatLog, vectorRack, probeRack, samplingState, highlightState}";
      return;
    }
    parsed = json;
  }

  async function applySnapshot(): Promise<void> {
    if (!parsed) return;
    warnings = [];
    appliedSummary = null;
    let appliedTurns = 0;
    let appliedVectors = 0;
    let skippedVectors = 0;
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
      warnings.push("chatLog missing or invalid — skipped");
    }

    if (Array.isArray(parsed.vectorRack)) {
      for (const row of parsed.vectorRack) {
        if (!row || typeof row !== "object" || typeof row.name !== "string") {
          warnings.push("vectorRack: skipped malformed entry");
          continue;
        }
        const name = row.name;
        const alpha = typeof row.alpha === "number" ? row.alpha : 0;
        const trigger: Trigger = (row.trigger ?? "BOTH") as Trigger;
        addVectorToRack(name, alpha, trigger);
        // Apply additional attributes if present.
        if (typeof row.alpha === "number") setVectorAlpha(name, row.alpha);
        if (typeof row.trigger === "string") setVectorTrigger(name, row.trigger as Trigger);
        if (typeof row.variant === "string")
          setVectorVariant(name, row.variant as Variant);
        if (row.projection !== undefined)
          setVectorProjection(name, (row.projection as ProjectionSpec | null) ?? null);
        if (typeof row.ablate === "boolean") setVectorAblate(name, row.ablate);
        if (typeof row.enabled === "boolean")
          setVectorEnabled(name, row.enabled);
        appliedVectors++;
      }
      // Sanity-check against the server's known set; surface a warning
      // for vectors that aren't currently registered.  This is purely
      // informational — the rack carries them as-is.
      try {
        const known = await import("../lib/stores.svelte").then(
          (m) => m.vectorsState.names,
        );
        for (const row of parsed.vectorRack) {
          if (typeof row?.name !== "string") continue;
          if (known.length > 0 && !known.includes(row.name)) {
            skippedVectors++;
            warnings.push(
              `vector '${row.name}' not registered server-side — present in rack but won't apply at gen time`,
            );
          }
        }
      } catch {
        /* ignore */
      }
    } else {
      warnings.push("vectorRack missing or invalid — skipped");
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

    appliedSummary = `restored ${appliedTurns} turn${appliedTurns === 1 ? "" : "s"}, ${appliedVectors} vector${appliedVectors === 1 ? "" : "s"}${skippedVectors ? ` (${skippedVectors} not server-known)` : ""}, ${appliedSampling} sampling field${appliedSampling === 1 ? "" : "s"}`;
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
      registered on the server for steering to take effect — missing names
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
          <li>vectors: {parsed.vectorRack?.length ?? 0}</li>
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
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px;
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
    cursor: pointer;
    padding: 0.25em 0.4em;
  }
  .close:hover {
    color: var(--accent-red);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 0.7em;
    min-height: 0;
  }
  .hint {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    line-height: 1.4;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.25em;
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    text-transform: lowercase;
  }
  .file {
    color: var(--fg);
    font: inherit;
    font-family: var(--font-mono);
  }
  .error {
    color: var(--accent-error);
    font-size: var(--font-size-small);
    margin: 0;
    word-break: break-word;
  }
  .success {
    color: var(--accent-green);
    font-size: var(--font-size-small);
    margin: 0;
  }
  .parsed-info {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    padding: 0.5em 0.6em;
  }
  .meta {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
  }
  .counts {
    list-style: none;
    margin: 0.4em 0 0;
    padding: 0;
    color: var(--fg-strong);
    font-size: var(--font-size-small);
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.2em 1em;
  }
  .warnings {
    background: var(--bg-deep);
    border: 1px solid var(--accent-yellow);
    padding: 0.4em 0.6em;
    color: var(--accent-yellow);
    font-size: var(--font-size-small);
    line-height: 1.4;
    max-height: 180px;
    overflow-y: auto;
  }
  .warn-line {
    margin-top: 0.2em;
  }
  .footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.5em;
    padding: 16px;
    border-top: 1px solid var(--border);
  }
  .btn {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.4em 0.9em;
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
    border-color: var(--border-dim);
    cursor: not-allowed;
  }
  .btn.primary {
    color: var(--accent-blue);
    border-color: var(--accent-blue);
  }
  .btn.primary:hover:not(:disabled) {
    background: rgba(88, 166, 255, 0.1);
  }
</style>
