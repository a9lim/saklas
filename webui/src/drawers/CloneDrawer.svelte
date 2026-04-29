<script lang="ts">
  // CloneDrawer — corpus-based persona clone.  Wraps
  // POST /vectors/clone (SSE) — the underlying clone path emits only
  // ``done`` and ``error`` events (no per-step progress callbacks),
  // so the log area shows status pings + timestamps until the final
  // event lands.

  import { ApiError, apiCloneStream } from "../lib/api";
  import {
    addVectorToRack,
    closeDrawer,
  } from "../lib/stores.svelte";

  // Drawer host forwards { params } — unused.
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  // ----- form state -----
  let name = $state("");
  let corpusPath = $state("");
  let nPairs = $state<number | null>(90);
  let seed = $state<number | null>(null);
  let baseline = $state("");

  // ----- streaming state -----
  let running = $state(false);
  let logLines: { ts: string; text: string; kind: "info" | "ok" | "err" }[] =
    $state([]);
  let serverError: string | null = $state(null);

  // Heartbeat timer drives an "elapsed" line every second so the user
  // sees forward motion even though the server emits no mid-clone
  // events.  Cleared on done/error.
  let heartbeat: ReturnType<typeof setInterval> | null = null;
  let startedAt: number | null = null;

  function tsNow(): string {
    const d = new Date();
    return (
      d.getHours().toString().padStart(2, "0") +
      ":" +
      d.getMinutes().toString().padStart(2, "0") +
      ":" +
      d.getSeconds().toString().padStart(2, "0")
    );
  }

  function appendLog(text: string, kind: "info" | "ok" | "err" = "info"): void {
    logLines = [...logLines, { ts: tsNow(), text, kind }];
  }

  function startHeartbeat(): void {
    startedAt = performance.now();
    heartbeat = setInterval(() => {
      if (startedAt === null) return;
      const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
      appendLog(`cloning… (${elapsed}s elapsed)`, "info");
    }, 1500);
  }

  function stopHeartbeat(): void {
    if (heartbeat !== null) {
      clearInterval(heartbeat);
      heartbeat = null;
    }
    startedAt = null;
  }

  // ----- validation -----
  const canSubmit = $derived(
    name.trim().length > 0 && corpusPath.trim().length > 0 && !running,
  );

  function randomizeSeed(): void {
    seed = Math.floor(Math.random() * 2 ** 31);
  }

  async function submit(): Promise<void> {
    if (!canSubmit) return;
    running = true;
    serverError = null;
    logLines = [];
    appendLog(`starting clone of ${corpusPath.trim()}`, "info");
    appendLog(
      `name=${name.trim()}` +
        (nPairs !== null ? `, n_pairs=${nPairs}` : "") +
        (seed !== null ? `, seed=${seed}` : ""),
      "info",
    );
    startHeartbeat();
    try {
      const final = await apiCloneStream(
        {
          name: name.trim(),
          corpus_path: corpusPath.trim(),
          n_pairs: nPairs ?? undefined,
          seed: seed ?? undefined,
          baseline: baseline.trim() || null,
        },
        (ev) => {
          if (ev.event === "error") {
            const msg =
              (ev.data as { message?: string } | null)?.message ??
              "clone failed";
            appendLog(msg, "err");
          } else if (ev.event === "done") {
            const d = ev.data as { canonical?: string } | null;
            appendLog(`done: ${d?.canonical ?? "(unknown)"}`, "ok");
          } else {
            // Keep unknown event names visible — future-proofs against
            // the server gaining real progress events.
            appendLog(`${ev.event}`, "info");
          }
        },
      );
      stopHeartbeat();
      appendLog(`registered ${final.canonical}`, "ok");
      addVectorToRack(final.canonical);
      // Brief pause so the "registered" line is visible before the
      // drawer closes.  150ms = single frame at 60fps × ~10 — short
      // enough not to feel sluggish.
      setTimeout(() => closeDrawer(), 350);
    } catch (e) {
      stopHeartbeat();
      if (e instanceof ApiError) {
        if (e.status === 404) {
          serverError = `corpus not found: ${e.message}`;
        } else if (e.status === 400) {
          serverError = `clone rejected: ${e.message}`;
        } else {
          serverError = e.message;
        }
      } else {
        serverError = e instanceof Error ? e.message : String(e);
      }
      appendLog(serverError ?? "clone failed", "err");
    } finally {
      running = false;
    }
  }

  // Cleanup on unmount — stop the heartbeat if the drawer closes mid-clone.
  // Svelte 5: $effect with a cleanup return.
  $effect(() => {
    return () => stopHeartbeat();
  });
</script>

<div class="drawer-shell">
  <header class="head">
    <h2>clone from corpus</h2>
    <button
      type="button"
      class="close"
      aria-label="Close drawer"
      onclick={closeDrawer}>✕</button
    >
  </header>

  <form
    class="body"
    onsubmit={(ev) => {
      ev.preventDefault();
      void submit();
    }}
  >
    <label class="field">
      <span class="label">name <em>required</em></span>
      <input
        type="text"
        placeholder="tone"
        bind:value={name}
        autocomplete="off"
        spellcheck="false"
        disabled={running}
      />
    </label>

    <label class="field">
      <span class="label">corpus path <em>required</em></span>
      <input
        type="text"
        placeholder="/path/to/corpus.txt"
        bind:value={corpusPath}
        autocomplete="off"
        spellcheck="false"
        disabled={running}
      />
      <p class="muted">
        server-side absolute path; one utterance per line.
      </p>
    </label>

    <div class="row">
      <label class="field grow">
        <span class="label">n_pairs</span>
        <input
          type="number"
          min="2"
          step="1"
          placeholder="90"
          bind:value={nPairs}
          disabled={running}
        />
      </label>
      <label class="field grow seed-field">
        <span class="label">seed</span>
        <div class="seed-row">
          <input
            type="number"
            step="1"
            placeholder="(random)"
            bind:value={seed}
            disabled={running}
          />
          <button
            type="button"
            class="dice"
            title="Randomize seed"
            aria-label="Randomize seed"
            onclick={randomizeSeed}
            disabled={running}>🎲</button
          >
        </div>
      </label>
    </div>

    <label class="field">
      <span class="label">baseline <em>optional</em></span>
      <input
        type="text"
        placeholder="(reserved — not used today)"
        bind:value={baseline}
        autocomplete="off"
        spellcheck="false"
        disabled={running}
      />
    </label>

    {#if serverError}
      <p class="error" role="alert">{serverError}</p>
    {/if}

    {#if logLines.length > 0 || running}
      <section class="log" aria-live="polite">
        {#if running}
          <div class="log-head">
            <span class="spinner" aria-hidden="true"></span>
            <span>cloning, please wait — clone has no per-step progress.</span>
          </div>
        {/if}
        <ol class="log-lines">
          {#each logLines as line (line.ts + line.text)}
            <li class="log-line {line.kind}">
              <span class="log-ts">[{line.ts}]</span>
              <span class="log-text">{line.text}</span>
            </li>
          {/each}
        </ol>
      </section>
    {/if}

    <footer class="foot">
      <button
        type="button"
        class="secondary"
        onclick={closeDrawer}
        disabled={running}>cancel</button
      >
      <button type="submit" class="primary" disabled={!canSubmit}>
        {#if running}
          <span class="spinner small" aria-hidden="true"></span> cloning…
        {:else}
          clone
        {/if}
      </button>
    </footer>
  </form>
</div>

<style>
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6em 1em;
    border-bottom: 1px solid var(--border);
  }
  .head h2 {
    margin: 0;
    font-size: 1em;
    color: var(--accent-blue);
    letter-spacing: 0.04em;
    text-transform: lowercase;
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
    flex: 1;
    overflow-y: auto;
    padding: 0.8em 1em;
    display: flex;
    flex-direction: column;
    gap: 0.7em;
    min-height: 0;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 0.25em;
  }
  .label {
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    text-transform: lowercase;
    letter-spacing: 0.04em;
  }
  .label em {
    color: var(--fg-muted);
    font-style: normal;
    margin-left: 0.5em;
  }
  input[type="text"],
  input[type="number"] {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.45em 0.6em;
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
    box-sizing: border-box;
    width: 100%;
  }
  input:focus {
    outline: 1px solid var(--accent-blue);
    border-color: var(--accent-blue);
  }
  input:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .row {
    display: flex;
    gap: 0.6em;
  }
  .grow {
    flex: 1 1 0;
    min-width: 0;
  }
  .seed-field {
    flex: 1.2 1 0;
  }
  .seed-row {
    display: flex;
    gap: 0.3em;
  }
  .dice {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg);
    padding: 0 0.5em;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1em;
  }
  .dice:hover:not(:disabled) {
    border-color: var(--accent-blue);
  }
  .dice:disabled {
    opacity: 0.5;
  }

  .muted {
    color: var(--fg-muted);
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
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    background: var(--bg-deep);
    padding: 0.5em 0.7em;
    max-height: 18em;
    overflow-y: auto;
  }
  .log-head {
    display: flex;
    align-items: center;
    gap: 0.5em;
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    margin-bottom: 0.4em;
  }
  .log-lines {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.15em;
  }
  .log-line {
    font-size: var(--font-size-small);
    display: flex;
    gap: 0.5em;
  }
  .log-line .log-ts {
    color: var(--fg-muted);
    flex-shrink: 0;
  }
  .log-line .log-text {
    color: var(--fg-strong);
    word-break: break-word;
  }
  .log-line.ok .log-text {
    color: var(--accent-green);
  }
  .log-line.err .log-text {
    color: var(--accent-error);
  }

  .foot {
    border-top: 1px solid var(--border);
    padding-top: 0.6em;
    margin-top: auto;
    display: flex;
    justify-content: flex-end;
    gap: 0.5em;
  }
  .primary {
    background: var(--accent-blue);
    color: var(--bg-deep);
    border: 1px solid var(--accent-blue);
    padding: 0.35em 1em;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
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
    padding: 0.35em 0.9em;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    cursor: pointer;
  }
  .secondary:hover:not(:disabled) {
    border-color: var(--fg);
    color: var(--fg);
  }
  .secondary:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .spinner {
    width: 0.85em;
    height: 0.85em;
    border-radius: 50%;
    border: 1.5px solid var(--accent-blue);
    border-right-color: transparent;
    animation: spin 0.7s linear infinite;
    display: inline-block;
  }
  .spinner.small {
    width: 0.7em;
    height: 0.7em;
    border-color: var(--bg-deep);
    border-right-color: transparent;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
