<script lang="ts">
  // Export drawer — JSONL or CSV download of the last assistant turn.
  // Mirrors the TUI's ``/export <path>``: the TUI dumps
  // ``session.last_result`` through ``ResultCollector``.  The web side
  // doesn't keep ``last_result`` in scope, so we serialize the last
  // assistant turn from ``chatLog`` — same fields the rest of the UI
  // already shows (text, applied_steering, aggregateReadings, finish
  // reason, perplexity, sampling).
  //
  // Empty result (no assistant turn yet, or last turn had no readings)
  // → renders a notice and disables the download button.

  import { chatLog, samplingState, closeDrawer } from "../lib/stores.svelte";
  import type { ChatTurn, TokenScore } from "../lib/types";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  type Format = "jsonl" | "csv";
  let format: Format = $state("jsonl");
  let filename = $state("");

  /** Locate the most recent assistant turn — that's "the last result"
   * for export purposes.  Returns null if there isn't one. */
  const lastTurn: ChatTurn | null = $derived.by(() => {
    for (let i = chatLog.turns.length - 1; i >= 0; i--) {
      if (chatLog.turns[i].role === "assistant") return chatLog.turns[i];
    }
    return null;
  });

  const defaultFilename = $derived.by(() => {
    const ts = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .slice(0, 19);
    return `saklas-result-${ts}.${format}`;
  });

  function effectiveFilename(): string {
    let n = filename.trim();
    if (!n) n = defaultFilename;
    const ext = "." + format;
    if (!n.endsWith(ext)) n += ext;
    return n;
  }

  /** Build the structured payload for the last turn.  Lifts probe
   * readings (aggregate) onto the row; per-token data lives nested
   * inside so JSONL stays one-row-per-result. */
  function buildRecord(turn: ChatTurn): Record<string, unknown> {
    const sampling = {
      temperature: samplingState.temperature,
      top_p: samplingState.top_p,
      top_k: samplingState.top_k,
      max_tokens: samplingState.max_tokens,
      seed: samplingState.seed,
    };
    return {
      role: turn.role,
      text: turn.text ?? "",
      thinking: turn.thinking ?? false,
      applied_steering: turn.appliedSteering ?? null,
      finish_reason: turn.finishReason ?? null,
      tokens: turn.tokensSoFar ?? turn.tokens?.length ?? 0,
      max_tokens: turn.maxTokens ?? null,
      tok_per_sec: turn.tokPerSec ?? null,
      elapsed_sec: turn.elapsedSec ?? null,
      perplexity: turn.perplexity ?? null,
      readings: turn.aggregateReadings ?? {},
      sampling,
      per_token: tokenRowsForExport(turn.tokens ?? []),
      thinking_tokens: tokenRowsForExport(turn.thinkingTokens ?? []),
    };
  }

  function tokenRowsForExport(tokens: TokenScore[]): unknown[] {
    return tokens.map((t, i) => ({
      idx: i,
      text: t.text,
      thinking: t.thinking,
      probes: t.probes ?? null,
    }));
  }

  function buildJsonl(turn: ChatTurn): string {
    return JSON.stringify(buildRecord(turn)) + "\n";
  }

  /** CSV: one column per top-level scalar field plus one column per
   * probe reading (prefixed ``readings.``).  Per-token nested arrays
   * are JSON-encoded into a single cell since CSV can't represent them
   * structurally. */
  function buildCsv(turn: ChatTurn): string {
    const rec = buildRecord(turn);
    const readings = (rec.readings ?? {}) as Record<string, number>;
    const sampling = (rec.sampling ?? {}) as Record<string, unknown>;
    const cols: { key: string; value: unknown }[] = [
      { key: "role", value: rec.role },
      { key: "text", value: rec.text },
      { key: "thinking", value: rec.thinking },
      { key: "applied_steering", value: rec.applied_steering },
      { key: "finish_reason", value: rec.finish_reason },
      { key: "tokens", value: rec.tokens },
      { key: "max_tokens", value: rec.max_tokens },
      { key: "tok_per_sec", value: rec.tok_per_sec },
      { key: "elapsed_sec", value: rec.elapsed_sec },
      { key: "perplexity", value: rec.perplexity },
    ];
    for (const [k, v] of Object.entries(sampling)) {
      cols.push({ key: `sampling.${k}`, value: v });
    }
    for (const [k, v] of Object.entries(readings)) {
      cols.push({ key: `readings.${k}`, value: v });
    }
    cols.push({ key: "per_token_json", value: JSON.stringify(rec.per_token) });
    cols.push({
      key: "thinking_tokens_json",
      value: JSON.stringify(rec.thinking_tokens),
    });
    const head = cols.map((c) => csvEscape(c.key)).join(",");
    const row = cols.map((c) => csvEscape(c.value)).join(",");
    return `${head}\n${row}\n`;
  }

  function csvEscape(v: unknown): string {
    if (v === null || v === undefined) return "";
    let s: string;
    if (typeof v === "string") s = v;
    else if (typeof v === "number" || typeof v === "boolean") s = String(v);
    else s = JSON.stringify(v);
    if (/[",\n\r]/.test(s)) {
      s = '"' + s.replace(/"/g, '""') + '"';
    }
    return s;
  }

  function download(): void {
    if (!lastTurn) return;
    const text = format === "jsonl" ? buildJsonl(lastTurn) : buildCsv(lastTurn);
    const mime = format === "jsonl" ? "application/x-ndjson" : "text/csv";
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = effectiveFilename();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  const hasReadings = $derived.by(() => {
    if (!lastTurn) return false;
    const r = lastTurn.aggregateReadings;
    return r ? Object.keys(r).length > 0 : false;
  });
</script>

<section class="drawer-shell" aria-label="Export drawer">
  <header class="header">
    <span class="title">export last result</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    {#if !lastTurn}
      <p class="dim">no assistant result yet — generate something first.</p>
    {:else}
      <p class="hint">
        exports the last assistant turn (text, applied_steering, sampling,
        per-token data, and aggregate probe readings if present).
      </p>

      <div class="mode-row" role="radiogroup" aria-label="Format">
        <label class="mode-opt">
          <input
            type="radio"
            name="exp-fmt"
            value="jsonl"
            checked={format === "jsonl"}
            onchange={() => (format = "jsonl")}
          />
          <span>JSONL</span>
        </label>
        <label class="mode-opt">
          <input
            type="radio"
            name="exp-fmt"
            value="csv"
            checked={format === "csv"}
            onchange={() => (format = "csv")}
          />
          <span>CSV</span>
        </label>
      </div>

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

      <div class="meta-block">
        <p class="meta-row">
          <span class="meta-key">tokens</span>
          <span class="meta-val">
            {lastTurn.tokensSoFar ?? lastTurn.tokens?.length ?? 0}
          </span>
        </p>
        <p class="meta-row">
          <span class="meta-key">finish reason</span>
          <span class="meta-val">{lastTurn.finishReason ?? "—"}</span>
        </p>
        <p class="meta-row">
          <span class="meta-key">applied steering</span>
          <span class="meta-val">
            {lastTurn.appliedSteering ?? "—"}
          </span>
        </p>
        <p class="meta-row">
          <span class="meta-key">readings</span>
          <span class="meta-val">
            {hasReadings
              ? `${Object.keys(lastTurn.aggregateReadings ?? {}).length} probe(s)`
              : "none attached"}
          </span>
        </p>
      </div>

      {#if !hasReadings}
        <p class="warn">
          last turn carries no aggregate probe readings; the export will
          still succeed with an empty <code>readings</code> field.
        </p>
      {/if}
    {/if}
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={closeDrawer}>cancel</button>
    <button
      type="button"
      class="btn primary"
      onclick={download}
      disabled={!lastTurn}
    >download</button>
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
  .dim {
    color: var(--fg-muted);
  }
  .hint {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    line-height: 1.4;
  }
  .mode-row {
    display: flex;
    gap: 1.2em;
  }
  .mode-opt {
    display: inline-flex;
    align-items: center;
    gap: 0.35em;
    color: var(--fg-strong);
    font-size: var(--font-size-small);
    cursor: pointer;
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
  .input {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: 0.4em 0.5em;
    font: inherit;
    font-family: var(--font-mono);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent-blue);
  }
  .meta-block {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    padding: 0.5em 0.6em;
    display: flex;
    flex-direction: column;
    gap: 0.2em;
  }
  .meta-row {
    margin: 0;
    font-size: var(--font-size-small);
    display: grid;
    grid-template-columns: 11em 1fr;
    gap: 0.5em;
  }
  .meta-key {
    color: var(--fg-muted);
  }
  .meta-val {
    color: var(--fg-strong);
    word-break: break-word;
  }
  .warn {
    color: var(--accent-yellow);
    font-size: var(--font-size-small);
    margin: 0;
  }
  .warn code {
    color: var(--accent-yellow);
    background: rgba(210, 153, 34, 0.1);
    padding: 0 0.2em;
    border-radius: 2px;
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
