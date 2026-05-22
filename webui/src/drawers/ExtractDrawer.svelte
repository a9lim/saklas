<script lang="ts">
  // Custom-vector extraction — the "+ custom vector" launcher from the
  // unified VectorsDrawer.  Two input modes:
  //
  //   * poles   — type concept A (required) and B (optional); saklas
  //               auto-generates contrastive statements.  A "generate
  //               previews" button surfaces those statements in an
  //               editable pos/neg table before they commit; the user
  //               can edit and commit, or skip preview and submit the
  //               bare slug.
  //   * custom  — supply the contrastive pairs directly.  The table
  //               starts with one empty row; no preview call.
  //
  // Submission closes immediately and reopens the vectors drawer so
  // the user lands back in the list while the sticky progress toast
  // tracks extraction in the background.

  import { untrack } from "svelte";
  import { apiExtractStream, apiVectors, ApiError } from "../lib/api";
  import {
    closeDrawer,
    openDrawer,
    refreshPacks,
    refreshVectorList,
  } from "../lib/stores.svelte";
  import {
    dismissToast,
    pushToast,
    updateToast,
  } from "../lib/stores/toasts.svelte";
  import type { ExtractRequest, StatementPair } from "../lib/types";

  interface ExtractParams {
    /** Optional pre-fill for concept A — seeded when the vectors
     *  drawer's search query matches no catalog row and the user
     *  clicks through. */
    seed_a?: string;
  }

  // ``params`` is typed loosely (``unknown``) at the drawer boundary —
  // App.svelte hands every drawer the same untyped slot.  Narrow once
  // up here so the rest of the component speaks the typed shape.
  let { params }: { params?: unknown } = $props();
  function narrow(p: unknown): ExtractParams {
    return p && typeof p === "object" ? (p as ExtractParams) : {};
  }

  type InputMode = "poles" | "custom";
  let inputMode: InputMode = $state("poles");

  let conceptA = $state(untrack(() => narrow(params).seed_a ?? ""));
  let conceptB = $state("");
  let advancedOpen = $state(false);
  let method: "dim" | "pca" = $state("dim");
  let dls = $state(true);
  let sae = $state("");
  let errorMsg: string | null = $state(null);

  // Editable contrastive-pair table.  In poles mode it fills after a
  // "generate previews" call; in custom mode it starts with one empty
  // row.  ``previewed`` tracks whether the poles-mode user has ever
  // generated previews — drives whether submit sends pairs or the bare
  // slug string.
  let pairs: StatementPair[] = $state([]);
  let previewed = $state(false);
  let previewing = $state(false);

  // Custom mode opens with one empty row to fill in.
  $effect(() => {
    if (inputMode === "custom" && pairs.length === 0 && !previewed) {
      pairs = [{ positive: "", negative: "" }];
    }
  });

  function addPairRow(): void {
    pairs = [...pairs, { positive: "", negative: "" }];
  }
  function removePairRow(idx: number): void {
    pairs = pairs.filter((_, i) => i !== idx);
  }
  function setPairField(
    idx: number,
    key: keyof StatementPair,
    value: string,
  ): void {
    pairs = pairs.map((p, i) => (i === idx ? { ...p, [key]: value } : p));
  }

  /** Trimmed pairs with at least one non-empty side dropped. */
  function cleanPairs(): StatementPair[] {
    return pairs
      .map((p) => ({
        positive: p.positive.trim(),
        negative: p.negative.trim(),
      }))
      .filter((p) => p.positive || p.negative);
  }

  async function generatePreviews(): Promise<void> {
    if (!valid.ok || previewing) return;
    previewing = true;
    errorMsg = null;
    try {
      const r = await apiVectors.previewPairs({ concept: canonicalName });
      pairs = r.pairs.map((p) => ({
        positive: p.positive,
        negative: p.negative,
      }));
      previewed = true;
    } catch (e) {
      errorMsg = describeError(e);
    } finally {
      previewing = false;
    }
  }

  /** Slugify a free-text concept into the ``NAME_REGEX`` shape the
   *  server accepts: lowercase, ``[^a-z0-9] → _``, no leading/trailing
   *  underscores.  Empty in / empty out. */
  function slug(s: string): string {
    return s
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
  }

  const canonicalName = $derived.by(() => {
    const a = slug(conceptA);
    if (!a) return "";
    const b = slug(conceptB);
    return b ? `${a}.${b}` : a;
  });

  const valid = $derived.by(() => {
    if (!conceptA.trim()) {
      return {
        ok: false as const,
        reason:
          inputMode === "custom"
            ? "a name (concept A) is required to save the vector"
            : "concept A is required",
      };
    }
    if (!canonicalName) {
      return {
        ok: false as const,
        reason: "concept A needs at least one letter or digit",
      };
    }
    if (canonicalName.length > 64) {
      return {
        ok: false as const,
        reason: "combined slug must be ≤ 64 characters",
      };
    }
    if (inputMode === "custom" && cleanPairs().length === 0) {
      return {
        ok: false as const,
        reason: "custom mode needs at least one non-empty pair",
      };
    }
    return { ok: true as const, reason: null };
  });

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

  /** Background driver — invoked after we've already closed and
   *  reopened the vectors drawer, so the form sits closed while the
   *  toast tracks extraction.  Lifecycle: sticky progress toast
   *  spawn → live-update on SSE ``progress`` events → dismiss +
   *  replace with a 6 s success toast (or sticky error toast) when
   *  ``done`` / ``error`` lands.  Also refreshes packs so the newly
   *  extracted row pops into the vectors drawer reactively. */
  async function driveExtract(req: ExtractRequest): Promise<void> {
    const toastId = pushToast(`extracting '${req.name}'…`, {
      kind: "info",
      ttlMs: null,
    });
    try {
      const result = await apiExtractStream(req, (ev) => {
        if (ev.event === "progress") {
          const m =
            ev.data && typeof ev.data === "object"
              ? (ev.data as { message?: string }).message
              : null;
          if (m) updateToast(toastId, { detail: m });
        }
      });
      await Promise.all([refreshVectorList(), refreshPacks()]);
      dismissToast(toastId);
      pushToast(`extracted ${result.canonical}`, { kind: "info" });
    } catch (e) {
      dismissToast(toastId);
      pushToast(`extract '${req.name}' failed — ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    }
  }

  function submit(): void {
    if (!valid.ok) return;
    errorMsg = null;
    const req: ExtractRequest = {
      name: canonicalName,
      register: true,
      method,
      dls,
    };
    const saeTrim = sae.trim();
    if (saeTrim) req.sae = saeTrim;

    // Source resolution:
    //   * custom mode, or poles mode where the user generated and
    //     possibly edited previews → send the explicit pair list, which
    //     runs the server's generation-skipping fast path.
    //   * poles mode with no preview ever generated → leave ``source``
    //     unset so the server slug-resolves and auto-generates pairs.
    const cleaned = cleanPairs();
    if (inputMode === "custom" || previewed) {
      req.source = { pairs: cleaned };
    }

    // Close this drawer and reopen the vectors drawer so the user
    // lands back in the list while extraction runs in the background.
    // ``driveExtract`` runs on the module scope, so the component
    // unmount here doesn't cancel it.
    closeDrawer();
    openDrawer("vectors");
    void driveExtract(req);
  }

  function cancel(): void {
    // Esc / ✕ — back to the vectors drawer rather than no drawer, so
    // the user doesn't get stranded after backing out of the form.
    closeDrawer();
    openDrawer("vectors");
  }
</script>

<section class="drawer-shell" aria-label="Custom vector extraction">
  <header class="header">
    <span class="title">custom vector</span>
    <button type="button" class="close" aria-label="Close" onclick={cancel}
      >✕</button>
  </header>

  <div class="body">
    <div class="mode-switch" role="tablist" aria-label="Input mode">
      <button
        type="button"
        role="tab"
        class="mode-btn"
        class:active={inputMode === "poles"}
        aria-selected={inputMode === "poles"}
        onclick={() => (inputMode = "poles")}
      >poles</button>
      <button
        type="button"
        role="tab"
        class="mode-btn"
        class:active={inputMode === "custom"}
        aria-selected={inputMode === "custom"}
        onclick={() => (inputMode = "custom")}
      >custom statements</button>
    </div>

    <p class="hint">
      {#if inputMode === "poles"}
        enter one or two concepts.  saklas auto-generates contrastive
        statements — generate previews to review and edit them before
        extraction, or submit straight away to use them as-is.
      {:else}
        supply the contrastive pairs directly.  each row's positive and
        negative statement is differenced to build the steering vector.
      {/if}
    </p>

    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}

    <form
      class="form"
      onsubmit={(ev) => {
        ev.preventDefault();
        submit();
      }}
    >
      <label class="field">
        <span class="label">
          concept A <span class="req">required</span>
          {#if inputMode === "custom"}
            <span class="opt">— names the saved vector</span>
          {/if}
        </span>
        <input
          type="text"
          class="input"
          bind:value={conceptA}
          placeholder="e.g. dog"
          autocomplete="off"
          spellcheck="false"
        />
      </label>

      <label class="field">
        <span class="label">concept B <span class="opt">optional</span></span>
        <input
          type="text"
          class="input"
          bind:value={conceptB}
          placeholder="e.g. cat — leave blank for a monopolar vector"
          autocomplete="off"
          spellcheck="false"
        />
      </label>

      {#if canonicalName}
        <p class="canonical">
          saved as <code>{canonicalName}</code>
        </p>
      {/if}

      {#if inputMode === "poles"}
        <button
          type="button"
          class="preview-btn"
          disabled={!valid.ok || previewing}
          onclick={generatePreviews}
        >
          {previewing ? "generating previews…" : "generate previews"}
        </button>
      {/if}

      {#if inputMode === "custom" || pairs.length > 0}
        <section class="pairs">
          <div class="pairs-head">
            <span class="label">contrastive pairs</span>
            <span class="pairs-count">{cleanPairs().length} used</span>
          </div>
          <div class="pairs-table">
            <div class="pair-row pair-header">
              <span>positive</span>
              <span>negative</span>
              <span></span>
            </div>
            {#each pairs as pair, idx (idx)}
              <div class="pair-row">
                <textarea
                  class="pair-input"
                  rows="2"
                  value={pair.positive}
                  oninput={(ev) =>
                    setPairField(
                      idx,
                      "positive",
                      (ev.currentTarget as HTMLTextAreaElement).value,
                    )}
                  placeholder="positive statement"
                ></textarea>
                <textarea
                  class="pair-input"
                  rows="2"
                  value={pair.negative}
                  oninput={(ev) =>
                    setPairField(
                      idx,
                      "negative",
                      (ev.currentTarget as HTMLTextAreaElement).value,
                    )}
                  placeholder="negative statement"
                ></textarea>
                <button
                  type="button"
                  class="pair-remove"
                  onclick={() => removePairRow(idx)}
                  aria-label="remove pair {idx + 1}"
                  title="remove pair"
                >✕</button>
              </div>
            {/each}
          </div>
          <button type="button" class="add-pair" onclick={addPairRow}>
            + add pair
          </button>
        </section>
      {/if}

      <section class="advanced" class:open={advancedOpen}>
        <button
          type="button"
          class="advanced-header"
          aria-expanded={advancedOpen}
          onclick={() => (advancedOpen = !advancedOpen)}
        >
          <span class="caret" aria-hidden="true"
            >{advancedOpen ? "▾" : "▸"}</span>
          <span class="advanced-name">Advanced options</span>
        </button>

        {#if advancedOpen}
          <div class="advanced-body">
            <fieldset class="field method">
              <legend class="label">method</legend>
              <label class="radio">
                <input type="radio" bind:group={method} value="dim" />
                <span>difference-of-means</span>
              </label>
              <label class="radio">
                <input type="radio" bind:group={method} value="pca" />
                <span>contrastive PCA</span>
              </label>
            </fieldset>

            <label class="field">
              <span class="label"
                >SAE release <span class="opt">optional</span></span>
              <input
                type="text"
                class="input"
                bind:value={sae}
                placeholder="e.g. gemma-scope-2b-pt-res"
                autocomplete="off"
                spellcheck="false"
              />
            </label>

            <label class="check">
              <input type="checkbox" bind:checked={dls} />
              <span>centered DLS layer selection</span>
            </label>
          </div>
        {/if}
      </section>

      {#if !valid.ok}
        <p class="validation">{valid.reason}</p>
      {/if}

      <button type="submit" class="extract-btn" disabled={!valid.ok}>
        extract → return to vectors
      </button>
    </form>
  </div>
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
    justify-content: space-between;
    padding: var(--space-4) var(--space-5);
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent);
    letter-spacing: 0;
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    font-size: var(--text);
    line-height: 1;
    padding: var(--space-2) var(--space-3);
    cursor: pointer;
    transition: color var(--dur) var(--ease-out);
  }
  .close:hover {
    color: var(--accent-red);
  }

  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-4) var(--space-5) var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
  }
  .hint {
    color: var(--fg-dim);
    font-size: var(--text-sm);
    margin: 0;
    line-height: 1.4;
  }
  .error {
    color: var(--accent-error);
    font-size: var(--text-sm);
    margin: 0;
    word-break: break-word;
  }

  .form {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    letter-spacing: 0;
  }
  .req {
    color: var(--accent-red);
    font-size: var(--text-xs);
    font-style: italic;
    margin-left: var(--space-2);
  }
  .opt {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-style: italic;
    margin-left: var(--space-2);
  }
  .input {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    transition: border-color var(--dur) var(--ease-out);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .canonical {
    margin: calc(-1 * var(--space-2)) 0 0;
    color: var(--fg-dim);
    font-size: var(--text-xs);
  }
  .canonical code {
    color: var(--accent);
    background: var(--bg-alt);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
  }

  .advanced {
    border-top: 1px solid var(--border);
    padding-top: var(--space-3);
    display: flex;
    flex-direction: column;
  }
  .advanced-header {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    width: 100%;
    text-align: left;
    background: transparent;
    border: 0;
    padding: var(--space-2) var(--space-1);
    color: var(--fg-muted);
    cursor: pointer;
    transition: color var(--dur) var(--ease-out);
  }
  .advanced-header:hover {
    color: var(--fg-strong);
  }
  .advanced.open .advanced-name {
    color: var(--accent);
  }
  .caret {
    font-size: var(--text-xs);
  }
  .advanced-name {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
  }
  .advanced-body {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-1) var(--space-1);
  }

  .method {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    margin: 0;
  }
  .method legend {
    padding: 0 var(--space-2);
  }
  .radio,
  .check {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    color: var(--fg-strong);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .radio input,
  .check input {
    accent-color: var(--accent);
  }

  .validation {
    color: var(--accent-yellow);
    font-size: var(--text-sm);
    margin: 0;
  }

  /* ---- input-mode switch ---- */
  .mode-switch {
    display: flex;
    gap: var(--space-1);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-1);
  }
  .mode-btn {
    flex: 1 1 0;
    background: transparent;
    color: var(--fg-dim);
    border: 0;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .mode-btn:hover {
    color: var(--fg-strong);
  }
  .mode-btn.active {
    background: var(--accent-subtle);
    color: var(--accent);
  }

  /* ---- preview button ---- */
  .preview-btn {
    background: var(--accent-subtle);
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
    align-self: flex-start;
  }
  .preview-btn:hover:not(:disabled) {
    background: var(--accent-glow);
  }
  .preview-btn:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }

  /* ---- pairs table ---- */
  .pairs {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    border-top: 1px solid var(--border);
    padding-top: var(--space-3);
  }
  .pairs-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  .pairs-count {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .pairs-table {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .pair-row {
    display: grid;
    grid-template-columns: 1fr 1fr auto;
    gap: var(--space-2);
    align-items: start;
  }
  .pair-header {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .pair-input {
    width: 100%;
    box-sizing: border-box;
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    resize: vertical;
    line-height: 1.4;
  }
  .pair-input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .pair-remove {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text);
    cursor: pointer;
    padding: var(--space-2);
  }
  .pair-remove:hover {
    color: var(--accent-red);
  }
  .add-pair {
    background: var(--accent-subtle);
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    align-self: flex-start;
  }
  .add-pair:hover {
    background: var(--accent-glow);
  }
  .extract-btn {
    background: var(--accent);
    color: var(--text-on-accent);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out);
  }
  .extract-btn:hover:not(:disabled) {
    background: var(--accent-light);
    border-color: var(--accent-light);
  }
  .extract-btn:disabled {
    background: var(--bg-elev);
    color: var(--fg-muted);
    border-color: var(--border);
    cursor: not-allowed;
  }
</style>
