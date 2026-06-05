<script lang="ts">
  // Custom-concept extraction — the "+ extract subspace" launcher from
  // the shared rack drawer's subspace family.  Two input modes:
  //
  //   * poles   — type concept A (required) and B (optional); the bare
  //               slug ``source`` is submitted to ``/extract`` and saklas
  //               auto-generates the contrastive corpus server-side.
  //   * custom  — supply the contrastive pairs directly.  The table
  //               starts with one empty row.
  //
  // Submission closes immediately and reopens the subspace drawer so
  // the user lands back in the list while the sticky progress toast
  // tracks extraction in the background.

  import { untrack } from "svelte";
  import { apiExtractStream, ApiError } from "../lib/api";
  import {
    closeDrawer,
    openDrawer,
    refreshVectorList,
  } from "../lib/stores.svelte";
  import {
    dismissToast,
    pushToast,
    updateToast,
  } from "../lib/stores/toasts.svelte";
  import type { ExtractRequest, StatementPair } from "../lib/types";
  import Checkbox from "../lib/Checkbox.svelte";
  import ModeTabs from "../lib/builder/ModeTabs.svelte";
  import AdvancedSection from "../lib/builder/AdvancedSection.svelte";
  import ValidationBlock from "../lib/builder/ValidationBlock.svelte";

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
  let dls = $state(true);
  let sae = $state("");
  // Role-augmented extraction (`:role-<slug>` variant).  Empty = raw
  // extraction.  Validated client-side against the same slug regex
  // the server uses (``[a-z0-9._-]+``); a non-empty role with
  // ``sae`` set is refused at submit time (engine-mutually-exclusive).
  let role = $state("");
  // Destination namespace + force-overwrite.  Parity with the manifold
  // builder's identity-block controls; both live in Advanced because
  // the defaults (``local`` / no-overwrite) are what 90% of users
  // want — the controls are escape hatches.
  let namespace = $state("local");
  let force = $state(false);
  let errorMsg: string | null = $state(null);

  const roleTrim = $derived(role.trim());
  const ROLE_SLUG_RE = /^[a-z0-9._-]+$/;
  const roleValid = $derived(
    roleTrim === "" || ROLE_SLUG_RE.test(roleTrim),
  );

  // Editable contrastive-pair table — custom mode only.  Starts with one
  // empty row; the auto-generated (poles) tab submits the bare slug and
  // lets saklas build the corpus server-side, so it never touches this.
  let pairs: StatementPair[] = $state([]);

  // Custom mode opens with one empty row to fill in.
  $effect(() => {
    if (inputMode === "custom" && pairs.length === 0) {
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
    if (!roleValid) {
      return {
        ok: false as const,
        reason: "role slug must match [a-z0-9._-]+ (lowercase only)",
      };
    }
    if (roleTrim && sae.trim()) {
      return {
        ok: false as const,
        reason: "role and SAE release are mutually exclusive",
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
   *  ``done`` / ``error`` lands.  Also refreshes the vector list so the
   *  newly extracted row pops into the vectors drawer reactively. */
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
      await refreshVectorList();
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
      dls,
    };
    const saeTrim = sae.trim();
    if (saeTrim) req.sae = saeTrim;
    if (roleTrim) req.role = roleTrim;
    const namespaceTrim = namespace.trim();
    if (namespaceTrim && namespaceTrim !== "local") {
      req.namespace = namespaceTrim;
    }
    if (force) req.force = true;

    // Source resolution:
    //   * custom mode → send the explicit pair list, which runs the
    //     server's generation-skipping fast path.
    //   * poles mode → leave ``source`` unset so the server slug-resolves
    //     the canonical name and auto-generates the contrastive corpus.
    if (inputMode === "custom") {
      req.source = { pairs: cleanPairs() };
    }

    // Close this drawer and reopen the subspace drawer so the user
    // lands back in the list while extraction runs in the background.
    // ``driveExtract`` runs on the module scope, so the component
    // unmount here doesn't cancel it.
    closeDrawer();
    openDrawer("subspace");
    void driveExtract(req);
  }

  function cancel(): void {
    // Esc / ✕ — back to the subspace drawer rather than no drawer, so
    // the user doesn't get stranded after backing out of the form.
    closeDrawer();
    openDrawer("subspace");
  }
</script>

<section class="drawer-shell" aria-label="Build vector">
  <header class="header">
    <span class="title">build vector</span>
    <button type="button" class="close" aria-label="Close" onclick={cancel}
      >✕</button>
  </header>

  <div class="body">
    <ModeTabs
      bind:value={inputMode}
      tabs={[
        { value: "poles", label: "auto-generated" },
        { value: "custom", label: "custom statements" },
      ]}
      ariaLabel="Input mode"
    />

    <p class="hint">
      {#if inputMode === "poles"}
        enter one or two concepts.  saklas auto-generates the contrastive
        corpus server-side and extracts the steering vector.
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

      {#if inputMode === "custom"}
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

      <AdvancedSection bind:expanded={advancedOpen}>
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

        <label class="field">
          <span class="label">
            role <span class="opt">optional — role-augmented extraction</span>
          </span>
          <input
            type="text"
            class="input"
            bind:value={role}
            placeholder="e.g. pirate — replaces the assistant-role label"
            autocomplete="off"
            spellcheck="false"
          />
          <span class="field-hint">
            Renders contrastive pairs with the chat template's
            assistant role substituted by this slug, and re-applies
            the substitution at steer time so baselines match.
            Steer the result with the matching <code>:role-{
              roleTrim || "&lt;slug&gt;"
            }</code> variant. Mistral-3 / talkie families don't
            support role substitution.
          </span>
        </label>

        <label class="field">
          <span class="label">
            namespace <span class="opt">optional — default "local"</span>
          </span>
          <input
            type="text"
            class="input"
            bind:value={namespace}
            placeholder="local"
            autocomplete="off"
            spellcheck="false"
          />
          <span class="field-hint">
            Destination folder under <code>~/.saklas/vectors/</code>.
            Parallel to the manifold builder's namespace control.
          </span>
        </label>

        <span class="check">
          <Checkbox
            bind:checked={dls}
            label="centered DLS layer selection"
          />
        </span>
        <span class="check">
          <Checkbox
            bind:checked={force}
            label="overwrite an existing vector with this name"
          />
        </span>
      </AdvancedSection>

      <ValidationBlock
        verb="extract"
        messages={valid.ok ? [] : [valid.reason]}
      />

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

  .check {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    color: var(--fg-strong);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .field-hint {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    line-height: 1.4;
  }
  .field-hint code {
    color: var(--accent);
    background: var(--bg-alt);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    font-family: var(--font-mono);
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
