<script lang="ts">
  // Save-conversation drawer — serialize the live chat log + rack state +
  // probe rack + sampling + highlight settings to a JSON blob and offer
  // it as a browser download.  Mirrors the TUI's ``/save`` but client-
  // side only — no server round-trip.
  //
  // Shape: {version, savedAt, model_id?, chatLog, steerRack, subspaceAlong,
  //         probeRack, highlightState, samplingState}.  Steer / probe rack
  //         Maps are serialized as plain arrays (Map → tuples) for JSON
  //         safety.  (v1 saves used a vector-only ``vectorRack`` array.)

  import {
    chatLog,
    steerRack,
    probeRack,
    highlightState,
    samplingState,
    sessionState,
    closeDrawer,
  } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  // Snapshot once at mount — saving while a generation is in flight
  // would otherwise capture a partial turn.  User can re-open the drawer
  // to refresh.
  const snapshot = $derived.by(() => ({
    version: 2 as const,
    savedAt: new Date().toISOString(),
    model_id: sessionState.info?.model_id ?? null,
    session_id: sessionState.info?.id ?? null,
    chatLog: chatLog.turns,
    // Full steer rack — every term (subspace + manifold) plus the shared
    // subspace-along master.  (v1 saves carried a vector-only ``vectorRack``
    // array; the loader still accepts that shape for back-compat.)
    steerRack: [...steerRack.entries.entries()].map(([name, entry]) => ({
      name,
      ...entry,
    })),
    subspaceAlong: steerRack.subspaceAlong,
    probeRack: {
      sortMode: probeRack.sortMode,
      active: [...probeRack.active],
      entries: [...probeRack.entries.entries()].map(([name, e]) => ({
        name,
        sparkline: e.sparkline,
        current: e.current,
        previous: e.previous,
      })),
    },
    highlightState: { ...highlightState },
    samplingState: { ...samplingState },
  }));

  const previewText = $derived(JSON.stringify(snapshot, null, 2));

  // Cap preview at ~200 lines (~16k chars) so a runaway log doesn't lock
  // the textarea.  The downloaded blob is always the full snapshot.
  const previewLines = $derived(previewText.split("\n"));
  const previewTruncated = $derived(previewLines.length > 200);
  const previewDisplay = $derived(
    previewTruncated
      ? previewLines.slice(0, 200).join("\n") +
          `\n… (${previewLines.length - 200} more lines)`
      : previewText,
  );

  let filename = $state("");

  const defaultFilename = $derived.by(() => {
    const ts = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .replace("T", "_")
      .slice(0, 19);
    return `saklas-conversation-${ts}.json`;
  });

  function effectiveFilename(): string {
    let n = filename.trim();
    if (!n) n = defaultFilename;
    if (!n.endsWith(".json")) n += ".json";
    return n;
  }

  function download(): void {
    const blob = new Blob([previewText], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = effectiveFilename();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
</script>

<section class="drawer-shell" aria-label="Save conversation drawer">
  <header class="header">
    <span class="title">save conversation</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    <p class="hint">
      writes a JSON blob containing the chat log, vector rack, probe rack,
      sampling state and highlight settings.  Re-open via the load-
      conversation drawer to restore.
    </p>

    <label class="field">
      <span class="label">filename</span>
      <input
        type="text"
        class="input"
        bind:value={filename}
        placeholder={defaultFilename}
        autocomplete="off"
        spellcheck="false"
      />
    </label>

    <div class="preview-block">
      <span class="label">preview</span>
      <pre class="preview" aria-label="Preview JSON">{previewDisplay}</pre>
      <span class="meta">
        {chatLog.turns.length} turn{chatLog.turns.length === 1 ? "" : "s"} ·
        {snapshot.steerRack.length} term{snapshot.steerRack.length === 1 ? "" : "s"} ·
        {probeRack.active.length} probe{probeRack.active.length === 1 ? "" : "s"}
      </span>
    </div>
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={closeDrawer}>cancel</button>
    <button type="button" class="btn primary" onclick={download}>
      download
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
  .input {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    padding: var(--space-3) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .preview-block {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-height: 0;
  }
  .preview {
    background: var(--bg-deep);
    padding: var(--space-3) var(--space-4);
    margin: 0;
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.4;
    max-height: 360px;
    overflow: auto;
    white-space: pre;
  }
  .meta {
    color: var(--fg-muted);
    font-size: var(--text-xs);
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
