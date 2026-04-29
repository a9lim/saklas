<script lang="ts">
  // SweepDrawer — single-axis alpha sweep over the SSE
  // ``POST /sessions/{id}/sweep`` route.  Mirrors the TUI's ``/sweep`` —
  // pick a vector, supply an alpha list (comma list, ``linspace(...)``,
  // or ``start:stop:step``), optionally pin a base steering expression
  // for the rest of the rack, then stream rows into a sortable table.
  //
  // Lifecycle: AbortController cancels the SSE on close + stop.  Rows
  // persist across drawer reopens — the user can come back to a finished
  // sweep without losing it.  The store's ``ingestSweepEvent('started',
  // ...)`` clears prior rows on the next sweep.

  import { apiSweep, ApiError, consumeSse } from "../lib/api";
  import {
    closeDrawer,
    currentSteeringExpression,
    ingestSweepEvent,
    samplingState,
    sweepState,
    vectorRack,
  } from "../lib/stores.svelte";
  import type { SweepEvent, SweepRequest } from "../lib/api";

  // --------------------------------------------------------- props ---
  // Drawer host passes ``params`` (currently unused — sweep is fully
  // self-driven from the form).  Declaring it keeps the App-side prop
  // type-checker happy.
  let { params }: { params: unknown } = $props();
  $effect(() => {
    void params;
  });

  // --------------------------------------------------------- form ---
  let prompt = $state("");
  let vectorName = $state("");
  let alphaInput = $state("0.0, 0.3, 0.5, 0.7, 1.0");
  let baseSteering = $state("");

  // --------------------------------------------------- table state ---
  type SortKey =
    | "idx"
    | "alpha"
    | "tok_count"
    | "tok_per_sec"
    | "finish_reason"
    | "applied_steering"
    | "text";
  let sortKey: SortKey = $state("idx");
  let sortDir: "asc" | "desc" = $state("asc");
  let expandedIdx: number | null = $state(null);

  // -------------------------------------------------- abort handle ---
  let abortCtrl: AbortController | null = $state(null);
  let lastError: string | null = $state(null);

  // ------------------------------------------------- rack options ---
  // Picker options come from the live rack.  Use the rack key (atom
  // display form) as the value sent to the server — that's the same
  // shape ``Steering.alphas`` keys off.
  const rackNames = $derived([...vectorRack.entries.keys()].sort());

  // ----------------------------------------------- alpha parsing ---

  /** Parse the alpha-list field into ``number[]``.  Three accepted
   * forms; whitespace permissive.  Returns ``{ values, error }`` so
   * the form can show inline red error and disable Run. */
  interface ParseResult {
    values: number[];
    error: string | null;
  }

  function parseAlphaList(raw: string): ParseResult {
    const trimmed = raw.trim();
    if (!trimmed) return { values: [], error: "alpha list is empty" };

    // linspace(start, stop, count)
    const linspaceMatch = trimmed.match(
      /^linspace\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,)]+?)\s*\)\s*$/i,
    );
    if (linspaceMatch) {
      const start = Number(linspaceMatch[1]);
      const stop = Number(linspaceMatch[2]);
      const count = Number(linspaceMatch[3]);
      if (!Number.isFinite(start) || !Number.isFinite(stop)) {
        return { values: [], error: "linspace bounds must be numbers" };
      }
      if (!Number.isInteger(count) || count < 1) {
        return {
          values: [],
          error: "linspace count must be a positive integer",
        };
      }
      if (count === 1) return { values: [start], error: null };
      const step = (stop - start) / (count - 1);
      const out: number[] = [];
      for (let i = 0; i < count; i++) out.push(start + step * i);
      return { values: out, error: null };
    }

    // start:stop:step
    if (trimmed.includes(":")) {
      const parts = trimmed.split(":").map((p) => p.trim());
      if (parts.length !== 3) {
        return {
          values: [],
          error: "range form is start:stop:step (three values)",
        };
      }
      const start = Number(parts[0]);
      const stop = Number(parts[1]);
      const step = Number(parts[2]);
      if (![start, stop, step].every(Number.isFinite)) {
        return { values: [], error: "range values must be numbers" };
      }
      if (step === 0) {
        return { values: [], error: "range step must be non-zero" };
      }
      if ((stop - start) * step < 0) {
        return {
          values: [],
          error: "range step direction disagrees with start→stop",
        };
      }
      const out: number[] = [];
      const eps = Math.abs(step) * 1e-9;
      const ascending = step > 0;
      let v = start;
      // Cap iterations defensively so a typo doesn't lock the UI.
      let guard = 0;
      while (
        (ascending ? v <= stop + eps : v >= stop - eps) &&
        guard++ < 10000
      ) {
        out.push(roundTrim(v));
        v += step;
      }
      if (guard >= 10000) {
        return { values: [], error: "range produced too many values" };
      }
      return { values: out, error: null };
    }

    // Comma list.
    const out: number[] = [];
    for (const part of trimmed.split(",")) {
      const t = part.trim();
      if (!t) continue;
      const v = Number(t);
      if (!Number.isFinite(v)) {
        return { values: [], error: `'${t}' is not a number` };
      }
      out.push(v);
    }
    if (out.length === 0) {
      return { values: [], error: "no values parsed" };
    }
    return { values: out, error: null };
  }

  /** Round a stepped value to 12 sig-figs to drop fp noise like
   * ``0.30000000000000004`` showing up in the table. */
  function roundTrim(v: number): number {
    if (v === 0) return 0;
    return Number.parseFloat(v.toPrecision(12));
  }

  // ----------------------------------------------- live validation ---

  const parsedAlphas = $derived(parseAlphaList(alphaInput));

  const validity = $derived.by(() => {
    if (sweepState.active) {
      return { ok: false, reason: "sweep in progress" } as const;
    }
    if (!prompt.trim()) {
      return { ok: false, reason: "prompt is required" } as const;
    }
    if (!vectorName.trim()) {
      return { ok: false, reason: "pick a vector to sweep" } as const;
    }
    if (rackNames.length === 0) {
      return {
        ok: false,
        reason: "rack is empty — extract or load a vector first",
      } as const;
    }
    if (parsedAlphas.error) {
      return { ok: false, reason: parsedAlphas.error } as const;
    }
    if (parsedAlphas.values.length === 0) {
      return { ok: false, reason: "alpha list is empty" } as const;
    }
    return { ok: true, reason: null } as const;
  });

  // ----------------------------------------------- table rendering ---

  /** Sort a copy of the live rows per the chosen column.  ``alpha``
   * sorts on the swept axis (``vectorName``); other axes (in the
   * unlikely event someone reuses this drawer for a multi-axis sweep
   * via base steering) are tie-broken on idx asc. */
  const sortedRows = $derived.by(() => {
    const rows = [...sweepState.rows];
    const cmp = (a: typeof rows[number], b: typeof rows[number]): number => {
      let x: number | string;
      let y: number | string;
      switch (sortKey) {
        case "idx":
          x = a.idx;
          y = b.idx;
          break;
        case "alpha":
          x = a.alpha_values[vectorName] ?? 0;
          y = b.alpha_values[vectorName] ?? 0;
          break;
        case "tok_count":
          x = a.token_count;
          y = b.token_count;
          break;
        case "tok_per_sec":
          x = a.tok_per_sec;
          y = b.tok_per_sec;
          break;
        case "finish_reason":
          x = a.finish_reason;
          y = b.finish_reason;
          break;
        case "applied_steering":
          x = a.applied_steering ?? "";
          y = b.applied_steering ?? "";
          break;
        case "text":
          x = a.text;
          y = b.text;
          break;
      }
      let d: number;
      if (typeof x === "number" && typeof y === "number") {
        d = x - y;
      } else {
        d = String(x).localeCompare(String(y));
      }
      if (d === 0) d = a.idx - b.idx;
      return sortDir === "asc" ? d : -d;
    };
    rows.sort(cmp);
    return rows;
  });

  /** Aggregate stats for the progress strip: average tok/s and total
   * elapsed across received rows. */
  const aggStats = $derived.by(() => {
    const rows = sweepState.rows;
    if (rows.length === 0) return { avgTps: 0, totalElapsed: 0 };
    let tpsSum = 0;
    let elapsedSum = 0;
    for (const r of rows) {
      tpsSum += r.tok_per_sec;
      elapsedSum += r.elapsed;
    }
    return {
      avgTps: tpsSum / rows.length,
      totalElapsed: elapsedSum,
    };
  });

  function setSort(key: SortKey): void {
    if (sortKey === key) {
      sortDir = sortDir === "asc" ? "desc" : "asc";
    } else {
      sortKey = key;
      sortDir = key === "idx" ? "asc" : "desc";
    }
  }

  function sortIndicator(key: SortKey): string {
    if (sortKey !== key) return "";
    return sortDir === "asc" ? " ↑" : " ↓";
  }

  function truncate(s: string, n: number): string {
    if (s.length <= n) return s;
    return s.slice(0, n - 1) + "…";
  }

  function toggleExpand(idx: number): void {
    expandedIdx = expandedIdx === idx ? null : idx;
  }

  // --------------------------------------------------- run / stop ---

  /** Build the ``SweepRequest`` body and start the SSE stream.  Defaults
   * to a one-shot ``stateless: true`` so the sweep doesn't pollute the
   * session's chat history, and forwards the user's sampling defaults
   * the same way ``sendGenerate`` does. */
  async function runSweep(): Promise<void> {
    if (!validity.ok || sweepState.active) return;
    lastError = null;
    expandedIdx = null;

    // Compose base_steering: if the user typed an explicit override use
    // it verbatim (including blank for "no base"); else mirror the live
    // rack expression so the rest of the user's setup carries through
    // the sweep.  Empty string suppresses base steering server-side.
    const explicit = baseSteering.trim();
    const base =
      baseSteering.length > 0
        ? explicit || null
        : currentSteeringExpression() || null;

    const sweepBody: Record<string, number[]> = {
      [vectorName]: parsedAlphas.values,
    };

    const sampling = samplingState.oneShotOverride
      ? {
          temperature: samplingState.temperature,
          top_p: samplingState.top_p,
          top_k: samplingState.top_k,
          max_tokens: samplingState.max_tokens,
          seed: samplingState.seed,
        }
      : null;

    const req: SweepRequest = {
      prompt: prompt.trim(),
      sweep: sweepBody,
      base_steering: base,
      sampling,
      thinking: samplingState.thinking,
      stateless: true,
      raw: false,
    };

    const ctrl = new AbortController();
    abortCtrl = ctrl;
    try {
      const response = await apiSweep.start(req);
      if (!response.body) {
        throw new Error("sweep: server returned no SSE body");
      }
      // The fetch in apiSweep doesn't accept the abort signal, so we
      // cancel the stream by tearing the underlying ReadableStream.
      const stream = response.body;
      ctrl.signal.addEventListener(
        "abort",
        () => {
          try {
            void stream.cancel();
          } catch {
            /* ignore */
          }
        },
        { once: true },
      );

      for await (const evt of consumeSse(stream)) {
        if (ctrl.signal.aborted) break;
        if (!evt.data || typeof evt.data !== "object") continue;
        // Sweep frames are bare ``data:`` lines with the type field
        // inside the payload — no SSE ``event:`` line.
        const payload = evt.data as SweepEvent;
        if (
          typeof payload !== "object" ||
          payload === null ||
          typeof (payload as { type?: unknown }).type !== "string"
        ) {
          continue;
        }
        ingestSweepEvent(payload);
        if (payload.type === "error") {
          lastError = payload.message;
        }
      }
    } catch (e) {
      if (ctrl.signal.aborted) {
        // User-initiated stop — store already cleared ``active`` via
        // the last ingested event, or we tear it down below.
      } else if (e instanceof ApiError) {
        const detail =
          e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message;
        lastError = `${e.status}: ${detail}`;
        // Mark inactive via synthesized error event so the store stays
        // consistent with what the user sees.
        ingestSweepEvent({ type: "error", message: lastError });
      } else {
        lastError = e instanceof Error ? e.message : String(e);
        ingestSweepEvent({ type: "error", message: lastError });
      }
    } finally {
      // If the stream ended without a ``done`` or ``error`` event (e.g.
      // user aborted), force the active flag off so the form re-enables.
      if (sweepState.active) {
        sweepState.active = false;
      }
      if (abortCtrl === ctrl) abortCtrl = null;
    }
  }

  function stopSweep(): void {
    if (abortCtrl) {
      abortCtrl.abort();
    }
  }

  // -------------------------------------------------- close hook ---

  /** When the drawer unmounts (close), abort any in-flight stream.
   * We don't reset ``sweepState.rows`` here — coming back to a finished
   * sweep should still show its table. */
  $effect(() => {
    return () => {
      if (abortCtrl) {
        try {
          abortCtrl.abort();
        } catch {
          /* ignore */
        }
      }
    };
  });
</script>

<section class="drawer-shell" aria-label="Sweep drawer">
  <header class="header">
    <span class="title">alpha sweep</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    <!-- form -->
    <form
      class="form"
      onsubmit={(ev) => {
        ev.preventDefault();
        void runSweep();
      }}
    >
      <label class="field">
        <span class="label">prompt</span>
        <textarea
          class="input textarea"
          bind:value={prompt}
          rows="3"
          disabled={sweepState.active}
          placeholder="prompt to sweep across alpha"
          spellcheck="false"
        ></textarea>
      </label>

      <label class="field">
        <span class="label">vector</span>
        <select
          class="input"
          bind:value={vectorName}
          disabled={sweepState.active || rackNames.length === 0}
        >
          {#if rackNames.length === 0}
            <option value="" disabled selected>(no vectors on rack)</option>
          {:else}
            <option value="" disabled selected={vectorName === ""}
              >— pick one —</option
            >
            {#each rackNames as n (n)}
              <option value={n}>{n}</option>
            {/each}
          {/if}
        </select>
      </label>

      <label class="field">
        <span class="label">alphas</span>
        <input
          type="text"
          class="input"
          bind:value={alphaInput}
          disabled={sweepState.active}
          placeholder="0.0, 0.3, 0.5, 0.7, 1.0  ·  linspace(-1, 1, 9)  ·  0.0:1.0:0.1"
          autocomplete="off"
          spellcheck="false"
        />
        {#if parsedAlphas.error && alphaInput.trim()}
          <span class="hint err">{parsedAlphas.error}</span>
        {:else if parsedAlphas.values.length > 0}
          <span class="hint"
            >{parsedAlphas.values.length} value{parsedAlphas.values.length === 1
              ? ""
              : "s"}: {parsedAlphas.values
              .slice(0, 12)
              .map((v) => v.toFixed(3))
              .join(", ")}{parsedAlphas.values.length > 12 ? ", …" : ""}</span
          >
        {/if}
      </label>

      <label class="field">
        <span class="label">base steering (optional)</span>
        <textarea
          class="input textarea"
          bind:value={baseSteering}
          rows="2"
          disabled={sweepState.active}
          placeholder="e.g. 0.3 honest + 0.2 warm — held during sweep; blank uses current rack"
          spellcheck="false"
        ></textarea>
      </label>

      {#if !validity.ok && !sweepState.active}
        <p class="validation">{validity.reason}</p>
      {/if}

      <div class="form-actions">
        {#if sweepState.active}
          <button type="button" class="btn danger" onclick={stopSweep}
            >stop sweep</button
          >
        {:else}
          <button
            type="submit"
            class="btn primary"
            disabled={!validity.ok}
          >run sweep</button>
        {/if}
      </div>
    </form>

    <!-- progress strip -->
    {#if sweepState.active}
      <div class="progress" role="status">
        sweep in progress: {sweepState.completed}/{sweepState.total} completed
        {#if sweepState.completed > 0}
          ({aggStats.avgTps.toFixed(1)} t/s avg, {aggStats.totalElapsed.toFixed(
            2,
          )}s elapsed)
        {/if}
      </div>
    {/if}

    {#if lastError}
      <p class="error" role="alert">{lastError}</p>
    {/if}
    {#if sweepState.error && sweepState.error !== lastError}
      <p class="error" role="alert">{sweepState.error}</p>
    {/if}

    <!-- result table -->
    {#if sweepState.rows.length > 0}
      <div class="table-wrap">
        <table class="results">
          <thead>
            <tr>
              <th class="th-num">
                <button type="button" onclick={() => setSort("idx")}
                  >idx{sortIndicator("idx")}</button
                >
              </th>
              <th class="th-num">
                <button type="button" onclick={() => setSort("alpha")}
                  >α{sortIndicator("alpha")}</button
                >
              </th>
              <th class="th-num">
                <button type="button" onclick={() => setSort("tok_count")}
                  >tok{sortIndicator("tok_count")}</button
                >
              </th>
              <th class="th-num">
                <button type="button" onclick={() => setSort("tok_per_sec")}
                  >tok/s{sortIndicator("tok_per_sec")}</button
                >
              </th>
              <th>
                <button type="button" onclick={() => setSort("finish_reason")}
                  >finish{sortIndicator("finish_reason")}</button
                >
              </th>
              <th>
                <button
                  type="button"
                  onclick={() => setSort("applied_steering")}
                  >steering{sortIndicator("applied_steering")}</button
                >
              </th>
              <th>
                <button type="button" onclick={() => setSort("text")}
                  >text{sortIndicator("text")}</button
                >
              </th>
            </tr>
          </thead>
          <tbody>
            {#each sortedRows as row (row.idx)}
              {@const alphaVal = row.alpha_values[vectorName] ?? 0}
              {@const isExpanded = expandedIdx === row.idx}
              <tr
                class="row"
                class:expanded={isExpanded}
                onclick={() => toggleExpand(row.idx)}
              >
                <td class="num">{row.idx}</td>
                <td class="num">{alphaVal.toFixed(3)}</td>
                <td class="num">{row.token_count}</td>
                <td class="num">{row.tok_per_sec.toFixed(1)}</td>
                <td>{row.finish_reason}</td>
                <td class="ellipsis"
                  >{row.applied_steering
                    ? truncate(row.applied_steering, 32)
                    : "—"}</td
                >
                <td class="ellipsis">{truncate(row.text, 80)}</td>
              </tr>
              {#if isExpanded}
                <tr class="expand-row">
                  <td colspan="7">
                    <div class="expand-inner">
                      <div class="expand-block">
                        <span class="block-label">applied_steering</span>
                        <code class="block-code"
                          >{row.applied_steering ?? "(none)"}</code
                        >
                      </div>
                      <div class="expand-block">
                        <span class="block-label">text</span>
                        <pre class="block-pre">{row.text}</pre>
                      </div>
                      {#if Object.keys(row.readings).length > 0}
                        <div class="expand-block">
                          <span class="block-label">readings</span>
                          <div class="readings">
                            {#each Object.entries(row.readings) as [name, val] (name)}
                              <div class="reading-row">
                                <span class="reading-name">{name}</span>
                                <div class="reading-bar-track">
                                  <div
                                    class="reading-bar-fill"
                                    style="width: {Math.min(
                                      100,
                                      Math.abs(val) * 100,
                                    )}%; background: {val >= 0
                                      ? 'var(--accent-green)'
                                      : 'var(--accent-red)'};"
                                  ></div>
                                </div>
                                <span class="reading-val"
                                  >{val >= 0 ? "+" : ""}{val.toFixed(3)}</span
                                >
                              </div>
                            {/each}
                          </div>
                        </div>
                      {/if}
                    </div>
                  </td>
                </tr>
              {/if}
            {/each}
          </tbody>
        </table>
      </div>
    {:else if !sweepState.active}
      <p class="empty">no sweep results yet</p>
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
    font-size: 1em;
    line-height: 1;
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

  .form {
    display: flex;
    flex-direction: column;
    gap: 0.55em;
    flex: 0 0 auto;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.8em;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.2em;
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    text-transform: lowercase;
    letter-spacing: 0.04em;
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
  .input:disabled {
    opacity: 0.6;
  }
  .textarea {
    resize: vertical;
    min-height: 2.4em;
  }
  .hint {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
  }
  .hint.err {
    color: var(--accent-red);
  }

  .validation {
    color: var(--accent-yellow);
    font-size: var(--font-size-small);
    margin: 0;
  }
  .error {
    color: var(--accent-red);
    font-size: var(--font-size-small);
    margin: 0;
    word-break: break-word;
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5em;
    margin-top: 0.2em;
  }

  .progress {
    color: var(--accent-blue);
    font-size: var(--font-size-small);
    background: var(--bg-alt);
    border: 1px solid var(--border);
    padding: 0.4em 0.6em;
  }

  .empty {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    margin: 0;
    font-style: italic;
  }

  .table-wrap {
    flex: 1 1 auto;
    overflow: auto;
    border: 1px solid var(--border);
    background: var(--bg-deep);
    min-height: 0;
  }
  .results {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--font-size-small);
    table-layout: fixed;
  }
  .results th,
  .results td {
    padding: 0.3em 0.5em;
    text-align: left;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .results th {
    color: var(--fg-muted);
    background: var(--bg-alt);
    font-weight: normal;
    text-transform: lowercase;
    position: sticky;
    top: 0;
    z-index: 1;
  }
  .results th button {
    all: unset;
    cursor: pointer;
    color: inherit;
    width: 100%;
    text-align: inherit;
  }
  .results th button:hover {
    color: var(--fg-strong);
  }
  .th-num {
    width: 5em;
  }
  .num {
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .row {
    cursor: pointer;
  }
  .row:hover {
    background: var(--bg-alt);
  }
  .row.expanded {
    background: var(--bg-alt);
  }
  .ellipsis {
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .expand-row td {
    background: var(--bg-deep);
    border-bottom: 1px solid var(--border);
    padding: 0.6em 0.8em;
    white-space: normal;
  }
  .expand-inner {
    display: flex;
    flex-direction: column;
    gap: 0.6em;
  }
  .expand-block {
    display: flex;
    flex-direction: column;
    gap: 0.2em;
  }
  .block-label {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
    text-transform: lowercase;
    letter-spacing: 0.04em;
  }
  .block-code {
    color: var(--accent-green);
    word-break: break-word;
  }
  .block-pre {
    margin: 0;
    color: var(--fg);
    white-space: pre-wrap;
    word-break: break-word;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    line-height: 1.4;
  }

  .readings {
    display: flex;
    flex-direction: column;
    gap: 0.2em;
  }
  .reading-row {
    display: grid;
    grid-template-columns: minmax(8em, 14em) 1fr 4em;
    align-items: center;
    gap: 0.5em;
    font-size: var(--font-size-small);
  }
  .reading-name {
    color: var(--fg-strong);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .reading-bar-track {
    background: var(--bg-elev);
    height: 6px;
    border-radius: 1px;
    position: relative;
    overflow: hidden;
  }
  .reading-bar-fill {
    height: 100%;
    background: var(--accent-green);
  }
  .reading-val {
    text-align: right;
    color: var(--fg-strong);
    font-variant-numeric: tabular-nums;
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
    border-color: var(--border);
    cursor: not-allowed;
  }
  .btn.primary {
    color: var(--accent-blue);
    border-color: var(--accent-blue);
  }
  .btn.primary:hover:not(:disabled) {
    background: rgba(88, 166, 255, 0.1);
  }
  .btn.danger {
    color: var(--accent-red);
    border-color: var(--accent-red);
  }
  .btn.danger:hover:not(:disabled) {
    background: rgba(255, 99, 99, 0.1);
  }
</style>
