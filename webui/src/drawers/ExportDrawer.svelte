<script lang="ts">
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // Export drawer — JSONL or CSV download of the last generated turn.
  // Mirrors the TUI's ``/export <path>``: the TUI dumps
  // ``session.last_result`` through ``ResultCollector``.  The web side
  // doesn't keep ``last_result`` in scope, so we serialize the last
  // generated turn from ``chatLog`` — same fields the rest of the UI
  // already shows (text, applied_steering, aggregateReadings, finish
  // reason, perplexity, sampling).
  //
  // Empty result (no generated turn yet, or last turn had no readings)
  // → renders a notice and disables the download button.

  import {
    chatLog,
    samplingState,
    closeDrawer,
    roleDisplayLabel,
  } from "../lib/stores.svelte";
  import type { ChatTurn, TokenScore } from "../lib/types";
  import Radio from "../lib/Radio.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  type Format = "jsonl" | "csv";
  let format: Format = $state("jsonl");
  let filename = $state("");

  /** Locate the most recent generated turn — that's "the last result"
   * for export purposes.  Returns null if there isn't one. */
  const lastTurn: ChatTurn | null = $derived.by(() => {
    for (let i = chatLog.turns.length - 1; i >= 0; i--) {
      if (chatLog.turns[i].generated) return chatLog.turns[i];
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
      structural_role: turn.role,
      role: roleDisplayLabel(turn.role, turn.roleLabel),
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
      { key: "structural_role", value: rec.structural_role },
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
    <span class="title">export</span>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="body">
    {#if !lastTurn}
      <p class="dim">no result</p>
    {:else}

      <div class="mode-row" role="radiogroup" aria-label="Format">
        <Radio bind:group={format} value="jsonl" label="JSONL" />
        <Radio bind:group={format} value="csv" label="CSV" />
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
          <span class="meta-key">finish</span>
          <span class="meta-val">{lastTurn.finishReason ?? "—"}</span>
        </p>
        <p class="meta-row">
          <span class="meta-key">steering</span>
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
        <p class="warn">no readings</p>
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
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
  }
  .dim {
    color: var(--fg-muted);
  }
  .mode-row {
    display: flex;
    gap: var(--space-6);
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
  .meta-block {
    background: var(--bg-deep);
    padding: var(--space-3) var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .meta-row {
    margin: 0;
    font-size: var(--text-sm);
    display: grid;
    grid-template-columns: 11em 1fr;
    gap: var(--space-3);
  }
  .meta-key {
    color: var(--fg-muted);
  }
  .meta-val {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    word-break: break-word;
  }
  .warn {
    color: var(--accent-yellow);
    font-size: var(--text-sm);
    margin: 0;
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
