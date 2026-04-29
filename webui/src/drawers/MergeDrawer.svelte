<script lang="ts">
  // MergeDrawer — combine vectors via a steering expression and register
  // the merged result as a new local pack.
  //
  // Expression is live-validated in the browser using parseExpression
  // (mirrors the server parser).  On submit, POST /vectors/merge wraps
  // saklas.io.merge.merge_into_pack server-side; the response carries
  // the canonical name + profile, which we drop into the rack.
  //
  // Variant picker covers raw / sae / sae-<release>.  The merge endpoint
  // body shape is { name, expression } only — no variant field on the
  // wire today.  Variant choice is encoded by appending ":<variant>" to
  // each atom as the user types (or via the chip below the textarea
  // which appends a default-variant suffix to the expression).

  import { ApiError, apiVectors } from "../lib/api";
  import {
    addVectorToRack,
    closeDrawer,
    vectorRack,
  } from "../lib/stores.svelte";
  import {
    ExpressionParseError,
    parseExpression,
    serializeExpression,
  } from "../lib/expression";
  import type { Variant, VectorRackEntry } from "../lib/types";

  // Drawer host forwards { params } — unused.
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  // ----- form state -----
  let name = $state("");
  let expression = $state("");
  let variant: Variant = $state("raw");
  let saeRelease = $state("");

  // ----- validation -----
  let parseError: string | null = $state(null);
  let parseCol: number | null = $state(null);
  let preview: string | null = $state(null);
  let unknownTerms: string[] = $state([]);

  // Datalist source: the user's current rack.
  const rackNames = $derived(Array.from(vectorRack.entries.keys()));

  // Live validator — re-runs every keystroke; cheap.
  $effect(() => {
    const expr = expression.trim();
    if (!expr) {
      parseError = null;
      parseCol = null;
      preview = null;
      unknownTerms = [];
      return;
    }
    try {
      const rack: Map<string, VectorRackEntry> = parseExpression(expr);
      preview = serializeExpression(rack);
      parseError = null;
      parseCol = null;
      // Soft check: which atoms aren't present in the user's rack?
      // Pure heuristic — the server may know vectors the rack hasn't
      // loaded yet, so this is informational, not a hard block.
      const known = new Set(rackNames);
      const unknown: string[] = [];
      for (const key of rack.keys()) {
        if (!known.has(key)) unknown.push(key);
      }
      unknownTerms = unknown;
    } catch (e) {
      preview = null;
      unknownTerms = [];
      if (e instanceof ExpressionParseError) {
        parseError = e.message;
        parseCol = e.col;
      } else {
        parseError = e instanceof Error ? e.message : String(e);
        parseCol = null;
      }
    }
  });

  // ----- variant pill: write the chosen variant suffix into a fresh
  // expression so users don't have to type ":sae" by hand.  Only fires
  // on the chip's onclick — preserves manual edits.
  function applyVariantToExpression(): void {
    const expr = expression.trim();
    if (!expr) return;
    let rack: Map<string, VectorRackEntry>;
    try {
      rack = parseExpression(expr);
    } catch {
      return;
    }
    for (const entry of rack.values()) {
      entry.variant = effectiveVariant();
    }
    expression = serializeExpression(rack);
  }

  function effectiveVariant(): Variant {
    if (variant === "raw" || variant === "sae") return variant;
    // sae-<release>; if the user typed a release, honor it.
    const r = saeRelease.trim();
    return (r ? `sae-${r}` : "sae") as Variant;
  }

  // ----- submit -----
  let submitting = $state(false);
  let serverError: string | null = $state(null);

  const canSubmit = $derived(
    name.trim().length > 0 &&
      expression.trim().length > 0 &&
      parseError === null &&
      !submitting,
  );

  async function submit(): Promise<void> {
    if (!canSubmit) return;
    submitting = true;
    serverError = null;
    try {
      const r = await apiVectors.merge({
        name: name.trim(),
        expression: expression.trim(),
      });
      addVectorToRack(r.canonical ?? r.profile?.name ?? name.trim());
      closeDrawer();
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.status === 400) {
          serverError = `merge rejected: ${e.message}`;
        } else if (e.status === 404) {
          serverError = `vector not found: ${e.message}`;
        } else {
          serverError = e.message;
        }
      } else {
        serverError = e instanceof Error ? e.message : String(e);
      }
    } finally {
      submitting = false;
    }
  }
</script>

<div class="drawer-shell">
  <header class="head">
    <h2>merge vectors</h2>
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
      <span class="label">name</span>
      <input
        type="text"
        placeholder="noble"
        bind:value={name}
        autocomplete="off"
        spellcheck="false"
      />
    </label>

    <label class="field">
      <span class="label">expression</span>
      <textarea
        rows="4"
        placeholder={`0.5 honest + 0.3 warm~confident\n— or paste any canonical steering expression.`}
        bind:value={expression}
        spellcheck="false"
        autocomplete="off"
      ></textarea>
      {#if parseError}
        <p class="error">
          {parseError}{parseCol !== null ? "" : ""}
        </p>
      {:else if preview}
        <p class="preview">preview: <code>{preview}</code></p>
      {/if}
      {#if unknownTerms.length > 0}
        <p class="warn">
          not in rack:
          {#each unknownTerms as t, i}
            <code>{t}</code>{#if i < unknownTerms.length - 1},{" "}{/if}
          {/each}
          — server will resolve from installed packs.
        </p>
      {/if}
    </label>

    <fieldset class="field variant">
      <legend class="label">variant</legend>
      <label class="radio">
        <input
          type="radio"
          name="variant"
          value="raw"
          checked={variant === "raw"}
          onchange={() => (variant = "raw")}
        />
        raw
      </label>
      <label class="radio">
        <input
          type="radio"
          name="variant"
          value="sae"
          checked={variant === "sae"}
          onchange={() => (variant = "sae")}
        />
        sae (unique)
      </label>
      <label class="radio">
        <input
          type="radio"
          name="variant"
          value="sae-release"
          checked={variant !== "raw" && variant !== "sae"}
          onchange={() => (variant = ("sae-" + saeRelease.trim()) as Variant)}
        />
        sae-
        <input
          type="text"
          class="release-input"
          placeholder="gemma-scope-2b-pt-res"
          bind:value={saeRelease}
          oninput={() => {
            if (variant !== "raw" && variant !== "sae") {
              variant = ("sae-" + saeRelease.trim()) as Variant;
            }
          }}
        />
      </label>
      <button
        type="button"
        class="apply-variant"
        onclick={applyVariantToExpression}
        disabled={!expression.trim() || parseError !== null}
        title="Rewrite the expression with this variant on every atom."
      >
        apply to all atoms
      </button>
    </fieldset>

    <details class="hint">
      <summary>grammar reference</summary>
      <p class="muted">
        <code>0.3 honest</code> · <code>0.4 warm@after</code> ·
        <code>0.5 wolf~deer</code> (project onto) ·
        <code>0.5 sycophantic|honest</code> (project away) ·
        <code>!hallucinating</code> (mean-ablate) ·
        <code>0.3 honest:sae</code> (SAE variant)
      </p>
    </details>

    <datalist id="rack-names-merge">
      {#each rackNames as n}
        <option value={n}></option>
      {/each}
    </datalist>

    {#if serverError}
      <p class="error" role="alert">{serverError}</p>
    {/if}

    <footer class="foot">
      <button type="button" class="secondary" onclick={closeDrawer}
        >cancel</button
      >
      <button type="submit" class="primary" disabled={!canSubmit}>
        {#if submitting}
          <span class="spinner" aria-hidden="true"></span> merging…
        {:else}
          merge
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
  .label,
  .variant legend {
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    text-transform: lowercase;
    letter-spacing: 0.04em;
    padding: 0;
  }
  input[type="text"],
  textarea {
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
  textarea {
    resize: vertical;
    min-height: 4em;
  }
  input[type="text"]:focus,
  textarea:focus {
    outline: 1px solid var(--accent-blue);
    border-color: var(--accent-blue);
  }

  .variant {
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    padding: 0.5em 0.7em;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4em 1em;
    align-items: center;
  }
  .variant legend {
    padding: 0 0.3em;
  }
  .radio {
    display: inline-flex;
    align-items: center;
    gap: 0.3em;
    color: var(--fg);
    font-size: var(--font-size-small);
  }
  .release-input {
    width: 12em;
    padding: 0.2em 0.4em;
    font-size: var(--font-size-small);
  }
  .apply-variant {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-dim);
    padding: 0.25em 0.6em;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    cursor: pointer;
  }
  .apply-variant:hover:not(:disabled) {
    color: var(--fg);
    border-color: var(--accent-blue);
  }
  .apply-variant:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .preview {
    margin: 0;
    color: var(--fg-strong);
    font-size: var(--font-size-small);
  }
  .preview code {
    color: var(--accent-green);
    background: var(--bg-deep);
    padding: 0.05em 0.3em;
    border-radius: 2px;
  }
  .warn {
    margin: 0;
    color: var(--accent-yellow);
    font-size: var(--font-size-small);
  }
  .warn code {
    color: inherit;
  }
  .error {
    margin: 0;
    color: var(--accent-error);
    font-size: var(--font-size-small);
    word-break: break-word;
  }
  .muted {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    margin: 0;
  }
  .hint summary {
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    cursor: pointer;
    list-style: revert;
  }
  .hint summary:hover {
    color: var(--fg-dim);
  }
  .hint code {
    color: var(--accent-blue);
    background: var(--bg-deep);
    padding: 0.05em 0.3em;
    border-radius: 2px;
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
  .secondary:hover {
    border-color: var(--fg);
    color: var(--fg);
  }
  .spinner {
    width: 0.7em;
    height: 0.7em;
    border-radius: 50%;
    border: 1.5px solid var(--bg-deep);
    border-right-color: transparent;
    animation: spin 0.7s linear infinite;
    display: inline-block;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
