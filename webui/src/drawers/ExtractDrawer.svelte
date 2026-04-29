<script lang="ts">
  // Extract drawer — kicks off a contrastive-PCA extraction over the SSE
  // ``/extract`` route, streams the server's progress messages into a
  // scrolling log, and on ``done`` adds the resulting profile to the rack.
  //
  // Form shape mirrors ``ExtractRequest`` in saklas/server/saklas_api.py:
  //   - name             → req.name (required)
  //   - positive/negative → req.source (string for single-concept, or
  //                                     {pos, neg} pair when both filled)
  //   - baseline         → req.baseline (optional)
  //   - auto_register    → req.register (default true)
  //   - SAE release       → not yet wired into ExtractRequest server-side;
  //                          we emit it as a hint inside source-payload
  //                          when set, with an inline note for users.

  import { apiExtractStream, ApiError } from "../lib/api";
  import {
    addVectorToRack,
    closeDrawer,
    refreshVectorList,
  } from "../lib/stores.svelte";
  import type { ExtractRequest, VectorInfo } from "../lib/api";

  // Drawer prop contract: ``{ params }`` is forwarded by the drawer host
  // when opening this drawer.  Extract doesn't read it today (the form
  // is fully self-driven) but we still declare it so the orchestrator
  // can pass it without a Svelte unknown-prop warning.
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let name = $state("");
  let positive = $state("");
  let negative = $state("");
  let baseline = $state("");
  let autoRegister = $state(true);
  let saeRelease = $state("");

  let busy = $state(false);
  let errorMsg: string | null = $state(null);
  let progressLog: string[] = $state([]);
  let succeeded: { canonical: string; profile: VectorInfo } | null =
    $state(null);

  let logEl: HTMLDivElement | null = $state(null);

  /** Validation: name always required.  At least one of {name,positive}
   * has content — the bare-name path is single-concept extraction (uses
   * the on-disk statements pack for ``name``); pos/neg pair runs the
   * curated-pair path.  Negative without positive is invalid (nothing
   * to contrast against). */
  const validity = $derived.by(() => {
    const n = name.trim();
    const p = positive.trim();
    const ng = negative.trim();
    if (!n) return { ok: false, reason: "name is required" } as const;
    if (ng && !p)
      return {
        ok: false,
        reason: "negative requires positive (or leave both blank for single-concept)",
      } as const;
    return { ok: true, reason: null } as const;
  });

  function buildSource(): unknown {
    const p = positive.trim();
    const ng = negative.trim();
    if (p && ng) return { positive: p, negative: ng };
    if (p) return p;
    return null;
  }

  function reset(): void {
    succeeded = null;
    errorMsg = null;
    progressLog = [];
    name = "";
    positive = "";
    negative = "";
    baseline = "";
    saeRelease = "";
    autoRegister = true;
  }

  async function run(): Promise<void> {
    if (!validity.ok || busy) return;
    busy = true;
    errorMsg = null;
    progressLog = [];
    succeeded = null;

    const req: ExtractRequest = {
      name: name.trim(),
      register: autoRegister,
    };
    const src = buildSource();
    if (src !== null) req.source = src;
    const bl = baseline.trim();
    if (bl) req.baseline = bl;
    // SAE release: server's ExtractRequest doesn't carry an explicit
    // field today.  We surface the value in progress so the user sees
    // what they typed, but extraction proceeds as raw-PCA until the
    // backend wires the sidecar field.  When that lands, this is where
    // the extra payload goes.
    const sae = saeRelease.trim();
    if (sae) progressLog.push(`(note) sae release '${sae}' is informational; raw extraction will run`);

    try {
      const result = await apiExtractStream(req, (ev) => {
        // Surface every named SSE event line.  The ``progress`` event
        // carries a ``{message: string}`` payload; the ``done`` event
        // carries ``{canonical, profile}``; ``error`` carries
        // ``{message}``.  Everything else gets logged opportunistically.
        if (ev.event === "progress") {
          const m =
            ev.data && typeof ev.data === "object"
              ? (ev.data as { message?: string }).message
              : null;
          if (m) appendLog(m);
          else appendLog(JSON.stringify(ev.data));
        } else if (ev.event === "done") {
          appendLog("done");
        } else if (ev.event === "error") {
          const m =
            ev.data && typeof ev.data === "object"
              ? (ev.data as { message?: string }).message
              : null;
          appendLog(`error: ${m ?? "unknown"}`);
        } else {
          appendLog(`[${ev.event}] ${typeof ev.data === "string" ? ev.data : JSON.stringify(ev.data)}`);
        }
      });
      succeeded = result;
      // Auto-register on the rack only if the user asked for it, since
      // server-side ``register: false`` already keeps the session clean.
      if (autoRegister) {
        addVectorToRack(result.canonical);
        await refreshVectorList();
      }
    } catch (e) {
      if (e instanceof ApiError) {
        errorMsg = `${e.status}: ${
          e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message
        }`;
      } else {
        errorMsg = e instanceof Error ? e.message : String(e);
      }
    } finally {
      busy = false;
    }
  }

  function appendLog(line: string): void {
    progressLog = [...progressLog, line];
    // Defer scroll so DOM has the new line.
    queueMicrotask(() => {
      if (logEl) logEl.scrollTop = logEl.scrollHeight;
    });
  }
</script>

<section class="drawer-shell" aria-label="Extract drawer">
  <header class="header">
    <span class="title">extract vector</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    {#if succeeded}
      <div class="success" role="status">
        <p class="success-title">extracted <code>{succeeded.canonical}</code></p>
        <p class="success-meta">
          {succeeded.profile.layers.length} layer{succeeded.profile.layers.length === 1 ? "" : "s"}
          {#if autoRegister} · added to rack{/if}
        </p>
      </div>
    {:else}
      <form
        class="form"
        onsubmit={(ev) => {
          ev.preventDefault();
          void run();
        }}
      >
        <label class="field">
          <span class="label">name</span>
          <input
            type="text"
            class="input"
            bind:value={name}
            disabled={busy}
            placeholder="e.g. honest.deceptive or my_concept"
            autocomplete="off"
            spellcheck="false"
          />
        </label>

        <label class="field">
          <span class="label">positive</span>
          <input
            type="text"
            class="input"
            bind:value={positive}
            disabled={busy}
            placeholder="(optional — pole text or pos in pair)"
            autocomplete="off"
            spellcheck="false"
          />
        </label>

        <label class="field">
          <span class="label">negative</span>
          <input
            type="text"
            class="input"
            bind:value={negative}
            disabled={busy}
            placeholder="(optional — pair partner)"
            autocomplete="off"
            spellcheck="false"
          />
          <span class="hint">
            leave both blank for single-concept extraction from the
            statements pack named above
          </span>
        </label>

        <label class="field">
          <span class="label">baseline</span>
          <input
            type="text"
            class="input"
            bind:value={baseline}
            disabled={busy}
            placeholder="(optional)"
            autocomplete="off"
            spellcheck="false"
          />
        </label>

        <label class="field field-row">
          <input
            type="checkbox"
            bind:checked={autoRegister}
            disabled={busy}
          />
          <span class="label">auto-register on rack</span>
        </label>

        <label class="field">
          <span class="label">SAE release</span>
          <input
            type="text"
            class="input"
            bind:value={saeRelease}
            disabled={busy}
            placeholder="(optional, e.g. gemma-scope-2b-pt-res)"
            autocomplete="off"
            spellcheck="false"
          />
          <span class="hint">
            leave blank for raw PCA; SAE pipeline routing happens
            server-side once the field is wired
          </span>
        </label>

        {#if !validity.ok}
          <p class="validation">{validity.reason}</p>
        {/if}
      </form>
    {/if}

    {#if progressLog.length > 0}
      <div class="log" bind:this={logEl} aria-label="Extraction progress">
        {#each progressLog as line, i (i)}
          <div class="log-line">{line}</div>
        {/each}
      </div>
    {/if}

    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}
  </div>

  <footer class="footer">
    {#if succeeded}
      <button
        type="button"
        class="btn"
        onclick={reset}
      >extract another</button>
      <button
        type="button"
        class="btn primary"
        onclick={closeDrawer}
      >done</button>
    {:else}
      <button
        type="button"
        class="btn"
        onclick={closeDrawer}
        disabled={busy}
      >cancel</button>
      <button
        type="button"
        class="btn primary"
        onclick={run}
        disabled={!validity.ok || busy}
      >{busy ? "extracting…" : "extract"}</button>
    {/if}
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
    font-size: 1em;
    line-height: 1;
    padding: 0.25em 0.4em;
    cursor: pointer;
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
    gap: 0.75em;
    min-height: 0;
  }

  .form {
    display: flex;
    flex-direction: column;
    gap: 0.65em;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.25em;
  }
  .field-row {
    flex-direction: row;
    align-items: center;
    gap: 0.5em;
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
  .hint {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
  }

  .validation {
    color: var(--accent-yellow);
    font-size: var(--font-size-small);
    margin: 0;
  }
  .error {
    color: var(--accent-error);
    font-size: var(--font-size-small);
    margin: 0;
    word-break: break-word;
  }

  .log {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    padding: 0.4em 0.5em;
    max-height: 240px;
    overflow-y: auto;
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    line-height: 1.4;
    white-space: pre-wrap;
  }
  .log-line {
    word-break: break-word;
  }

  .success {
    border: 1px solid var(--accent-green);
    padding: 0.6em 0.8em;
    color: var(--fg-strong);
    background: rgba(126, 231, 135, 0.06);
  }
  .success-title {
    margin: 0 0 0.2em;
  }
  .success-title code {
    color: var(--accent-green);
  }
  .success-meta {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--font-size-small);
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
