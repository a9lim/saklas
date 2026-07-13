<script lang="ts">
  // Flat completion buffer — the chat surface for base (non-chat)
  // models.  No bubbles, no role labels: the loom active path is joined
  // into a single continuous editable text surface, ``white-space:
  // pre-wrap``.  Per-token tinting is preserved for the assistant span
  // when a highlight probe is selected.
  //
  // "continue" generates with ``raw: true`` from the active leaf — the
  // whole buffer is effectively a prefill.  A mid-buffer edit lands as
  // a ``loomEdit`` (active leaf) or ``loomBranch`` (interior node)
  // first; toggling render mode never mutates the tree, only generation
  // and explicit edits do.

  import { onMount } from "svelte";
  import StatusFooter from "./StatusFooter.svelte";
  import PendingBubbles from "./PendingBubbles.svelte";
  import {
    chatLog,
    loomTree,
    sendCommit,
    genStatus,
    sendGenerate,
    sendStop,
    highlightState,
    openDrawer,
    highlightScale,
  } from "../lib/stores.svelte";
  import type { ChatTurn, TokenScore } from "../lib/types";
  import {
    scoreToRgb,
    highlightHue,
    surpriseScore,
    SURPRISE_TARGET,
    probeScoreForTarget,
  } from "../lib/tokens";

  // ---------- buffer text ----------

  /** The conversation flattened to one string — every turn's text
   *  concatenated in order.  This is what the model sees as a prefill
   *  on "continue". */
  const bufferText = $derived(
    chatLog.turns.map((t) => t.text ?? "").join(""),
  );

  // Local editable mirror.  Synced from ``bufferText`` whenever the
  // tree changes and the user isn't mid-edit; user edits write here
  // and "continue" / "commit edit" land them on the tree.
  let draft = $state("");
  let dirty = $state(false);
  // Set the moment a continue/commit is fired: the tree hasn't caught
  // up to the locally-edited draft yet, so the buffer→draft sync below
  // is held until ``bufferText`` reaches the draft — without this the
  // draft would briefly snap back to the pre-edit text (a flash of the
  // user's typed tail vanishing) for the server round-trip.
  let committing = $state(false);
  let textareaRef: HTMLTextAreaElement | null = $state(null);

  // Re-sync the draft from the tree when the buffer changes and the
  // user has no pending edit.  ``dirty`` guards against clobbering an
  // in-progress edit; ``committing`` holds the just-sent draft until
  // the server-side node lands, after which streamed tokens flow in.
  $effect(() => {
    const text = bufferText;
    if (committing) {
      // Hold until the tree genuinely carries the committed draft as a
      // prefix — a *content* check, not a length one.  A length compare
      // releases early when a fast-streaming continuation fills the
      // buffer to the right size before the committed span has landed
      // (or under a transient wrong-parent active path), snapping the
      // draft onto text that doesn't contain what the user wrote.
      if (!text.startsWith(draft)) return;
      committing = false;
    }
    if (!dirty) {
      draft = text;
    }
  });

  function onInput(ev: Event): void {
    draft = (ev.currentTarget as HTMLTextAreaElement).value;
    dirty = draft !== bufferText;
    // Any keystroke supersedes a pending commit — resume live sync.
    committing = false;
  }

  // ---------- continue / commit ----------
  //
  // Flat mode is non-linear: editing text anywhere in the buffer is the
  // same operation as appending to the end.  Both collapse to a single
  // "divergence" — the first character that differs from the committed
  // tree text — and everything from there becomes one new span.  The
  // node containing the divergence (and its whole subtree) is preserved
  // untouched as the original branch; the edited tail is recorded as a
  // fresh span branched at that point.  Appending is just the special
  // case where the divergence sits at the very end of the buffer.

  /** The active leaf node — what a clean continuation extends. */
  const activeLeaf = $derived(
    loomTree.loaded ? (loomTree.active_node_id ?? null) : null,
  );

  interface Divergence {
    /** New text — from the divergence offset to the end of the draft. */
    tail: string;
    /** Where the new span hangs: the diverging node's parent for a
     *  mid-buffer edit, the active leaf for a pure append.
     *  ``undefined`` means the buffer is clean — continue from the leaf
     *  with no new span at all. */
    parentNodeId: string | null | undefined;
  }

  /** Diff the draft against the committed buffer and locate the single
   *  span the change collapses to. */
  function resolveDivergence(): Divergence {
    if (!dirty) return { tail: "", parentNodeId: undefined };
    const turns = chatLog.turns;
    let startOffset = 0;
    for (let i = 0; i < turns.length; i++) {
      const turnText = turns[i].text ?? "";
      const slice = draft.slice(startOffset, startOffset + turnText.length);
      if (slice !== turnText) {
        // Divergence inside turns[i].  That node + its subtree stay as
        // the original branch; the tail (this node's start → end of
        // draft) becomes a new span branched as its sibling — i.e. a
        // child of the node's parent.
        const nid = turns[i].nodeId ?? null;
        const parentNodeId = nid
          ? (loomTree.nodes.get(nid)?.parent_id ?? null)
          : (activeLeaf ?? null);
        return { tail: draft.slice(startOffset), parentNodeId };
      }
      startOffset += turnText.length;
    }
    // No node diverged — the draft runs past the joined buffer.  The
    // appended tail hangs under the active leaf as a fresh child.
    return { tail: draft.slice(startOffset), parentNodeId: activeLeaf ?? null };
  }

  function continueGen(): void {
    if (genStatus.active) return;
    const d = resolveDivergence();
    // The divergence tail rides as raw input: the engine records it as
    // a node and continues from the flat active-path text.  A clean
    // buffer sends "" — a bare continuation from the active leaf.
    committing = true;
    dirty = false;
    void sendGenerate(d.tail, { raw: true, parent_node_id: d.parentNodeId });
  }

  // ---------- edit (commit without generating) ----------

  /** Land the pending edit on the tree without generating — the same
   *  divergence branch ``continueGen`` would take, minus the decode. */
  async function commitEdit(): Promise<void> {
    if (!dirty) return;
    const d = resolveDivergence();
    dirty = false;
    if (d.tail === "") {
      // Pure truncation back to a node boundary — nothing new to land;
      // the boundary node already holds the committed text.
      return;
    }
    committing = true;
    await sendCommit("user", d.parentNodeId ?? null, d.tail, { raw: true });
  }

  function revertEdit(): void {
    draft = bufferText;
    dirty = false;
    committing = false;
  }

  function onKeydown(ev: KeyboardEvent): void {
    // Bare Enter continues; Ctrl/Cmd/Option-Enter commits the edit
    // without generating; Shift-Enter is a literal newline; Escape
    // stops an in-flight gen.
    if (ev.key === "Enter") {
      if (ev.shiftKey) return;
      ev.preventDefault();
      if (ev.ctrlKey || ev.metaKey || ev.altKey) void commitEdit();
      else continueGen();
      return;
    }
    if (ev.key === "Escape" && genStatus.active) {
      ev.preventDefault();
      sendStop();
    }
  }

  // ---------- per-token tint overlay ----------
  //
  // The textarea itself can't be per-token tinted, so when a highlight
  // probe is selected we render a read-only tinted mirror above the
  // (transparent-text) textarea.  Editing drops back to the plain
  // textarea — tinting is a read affordance.

  // A coordinate-preserving view of the buffer's tokens.  The flat
  // ``allTokens`` list below (used by the edit-mode tint mirror) throws
  // away which turn/index each token came from, but inspect mode needs
  // those coordinates to open the same token-drilldown drawer chat mode
  // uses — so the alternatives table and logit-fork work identically.
  interface RawToken {
    text: string;
    /** Loom coordinates for ``openDrawer("token_drilldown", …)``.
     *  ``null`` for prompt text (user / system turns), which renders as
     *  an inert span. */
    turnIdx: number | null;
    tokenIdx: number;
    isThinking: boolean;
    /** Backing score row — drives tint + alt availability.  ``null`` for
     *  inert prompt spans. */
    tok: TokenScore | null;
  }

  const tokenViews = $derived.by<RawToken[]>(() => {
    const out: RawToken[] = [];
    chatLog.turns.forEach((t, turnIdx) => {
      if (t.thinkingTokens) {
        t.thinkingTokens.forEach((tok, tokenIdx) =>
          out.push({ text: tok.text, turnIdx, tokenIdx, isThinking: true, tok }),
        );
      }
      if (t.tokens) {
        t.tokens.forEach((tok, tokenIdx) =>
          out.push({ text: tok.text, turnIdx, tokenIdx, isThinking: false, tok }),
        );
      } else if (t.role !== "assistant") {
        // User / system turns have no token rows — one inert span so the
        // rendered text matches the buffer exactly.
        out.push({
          text: t.text ?? "",
          turnIdx: null,
          tokenIdx: 0,
          isThinking: false,
          tok: null,
        });
      }
    });
    return out;
  });

  const allTokens = $derived<TokenScore[]>(
    tokenViews.map((v) => v.tok ?? { text: v.text, thinking: false }),
  );

  /** True when a tinted read-only mirror should render — a highlight
   *  probe is selected and the buffer isn't being actively edited. */
  // The tinted mirror is built from ``chatLog.turns`` (the committed
  // tree), so while ``committing`` holds an edit the tree hasn't caught
  // up to, the mirror would lag the textarea — show the plain textarea
  // (which carries the held draft) until the tree reconciles.
  const showTint = $derived(
    highlightState.target !== null &&
      !dirty &&
      !committing &&
      allTokens.length > 0,
  );

  function tokenScore(t: TokenScore): number | undefined {
    const target = highlightState.target;
    if (!target) return undefined;
    if (target === SURPRISE_TARGET) return surpriseScore(t.logprob);
    const s = probeScoreForTarget(t, target);
    return s !== undefined ? s : t.score;
  }

  function tintStyle(t: TokenScore): string {
    const bg = scoreToRgb(
      tokenScore(t),
      highlightScale(highlightState.target),
      highlightHue(highlightState.target),
    );
    return bg === "transparent" ? "" : `background-color: ${bg}`;
  }

  // ---------- edit / inspect mode ----------
  //
  // Edit mode is the free-text textarea (plus the optional tint mirror).
  // Inspect mode swaps in a read-only clickable token view: each
  // generated token opens the drilldown drawer where alternatives can be
  // picked and forked, exactly as in chat mode.  Inspect is read-only —
  // editing always happens in edit mode — so an uncommitted draft can
  // never desync from the committed tree the inspect view reads.
  type Mode = "edit" | "inspect";
  let mode: Mode = $state<Mode>("edit");

  /** Whether any clickable (generated) token exists to inspect. */
  const hasClickableTokens = $derived(
    tokenViews.some((v) => v.turnIdx !== null),
  );

  /** Inspect is offered only on a settled buffer — a pending edit or an
   *  in-flight commit would make the read-only view lag the textarea. */
  const canInspect = $derived(hasClickableTokens && !dirty && !committing);

  /** Fall back to edit when nothing remains to inspect (buffer cleared,
   *  conversation reset) so the surface never strands on an empty view. */
  $effect(() => {
    if (mode === "inspect" && !hasClickableTokens) mode = "edit";
  });

  function inspectTooltip(v: RawToken): string {
    const tok = v.tok;
    if (!tok) return "";
    const parts: string[] = [];
    const sc = tokenScore(tok);
    if (sc !== undefined && highlightState.target) {
      const label =
        highlightState.target === SURPRISE_TARGET
          ? "surprise"
          : highlightState.target;
      parts.push(`${label} ${sc >= 0 ? "+" : ""}${sc.toFixed(3)}`);
    }
    const n = tok.topAlts?.length ?? 0;
    parts.push(n > 0 ? `click — ${n} alternatives` : "click — token details");
    return parts.join(" · ");
  }

  function openToken(v: RawToken): void {
    if (v.turnIdx === null) return;
    openDrawer("token_drilldown", {
      turnIdx: v.turnIdx,
      tokenIdx: v.tokenIdx,
      isThinking: v.isThinking,
    });
  }

  let logRef: HTMLDivElement | null = $state(null);
  let scrolledUp = $state(false);
  function onScroll(ev: Event): void {
    const el = ev.currentTarget as HTMLElement;
    scrolledUp = el.scrollHeight - el.scrollTop - el.clientHeight >= 8;
  }
  $effect(() => {
    void bufferText;
    if (!scrolledUp && logRef) {
      queueMicrotask(() => {
        if (logRef) logRef.scrollTop = logRef.scrollHeight;
      });
    }
  });

  onMount(() => {
    draft = bufferText;
  });
</script>

<div class="raw-buffer" aria-label="Completion buffer">
  <div class="raw-head">
    <span class="head-label">buffer</span>
    <div class="mode-toggle" role="tablist" aria-label="Buffer mode">
      <button
        type="button"
        role="tab"
        aria-selected={mode === "edit"}
        class:active={mode === "edit"}
        onclick={() => (mode = "edit")}
        title="edit"
      >edit</button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === "inspect"}
        class:active={mode === "inspect"}
        disabled={mode !== "inspect" && !canInspect}
        onclick={() => (mode = "inspect")}
        title={canInspect || mode === "inspect"
          ? "inspect tokens — click any token for alternatives + fork"
          : dirty || committing
            ? "commit or revert the edit to inspect tokens"
            : "generate tokens first"}
      >inspect</button>
    </div>
  </div>

  <div class="surface" bind:this={logRef} onscroll={onScroll}>
    {#if mode === "inspect"}
      <div class="inspect" aria-label="Completion tokens">
        {#each tokenViews as v, i (i)}
          {#if v.turnIdx === null}
            <span class="seg plain">{v.text}</span>
          {:else}
            <span
              class="seg tok clickable"
              class:tinted={highlightState.target !== null}
              class:has-alts={(v.tok?.topAlts?.length ?? 0) > 0}
              style={v.tok ? tintStyle(v.tok) : ""}
              title={inspectTooltip(v)}
              role="button"
              tabindex="-1"
              onclick={() => openToken(v)}
              onkeydown={(ev) => {
                if (ev.key === "Enter" || ev.key === " ") {
                  ev.preventDefault();
                  openToken(v);
                }
              }}
            >{v.text}</span>
          {/if}
        {/each}
      </div>
    {:else}
      {#if showTint}
        <div class="tint-mirror" aria-hidden="true">
          {#each allTokens as tok, i (i)}
            <span class="tok" style={tintStyle(tok)}>{tok.text}</span>
          {/each}
        </div>
      {/if}
      <textarea
        class="buffer-text"
        class:has-tint={showTint}
        bind:this={textareaRef}
        value={draft}
        oninput={onInput}
        onkeydown={onKeydown}
        spellcheck="false"
        placeholder="prompt…"
        aria-label="Editable completion buffer"
      ></textarea>
    {/if}
  </div>

  <StatusFooter />
  <PendingBubbles />

  <div class="actions">
    {#if dirty}
      <span class="dirty-flag" title="uncommitted">
        edited
      </span>
      <button
        type="button"
        class="act commit"
        onclick={() => void commitEdit()}
        title="⌃⏎ commit"
      >
        commit edit
      </button>
      <button type="button" class="act revert" onclick={revertEdit}>
        revert
      </button>
    {/if}
    <button
      type="button"
      class="act continue"
      onclick={continueGen}
      disabled={genStatus.active}
      title="⏎ continue"
    >continue</button>
    <button
      type="button"
      class="act stop"
      onclick={sendStop}
      disabled={!genStatus.active}
      title="Esc"
    >stop</button>
  </div>
</div>

<style>
  .raw-buffer {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    gap: var(--space-3);
    font-family: var(--font-mono);
    font-size: var(--text);
    color: var(--fg);
  }
  .surface {
    flex: 1 1 auto;
    position: relative;
    overflow-y: auto;
    min-height: 0;
    /* Recessed input well — this is the editable completion surface. */
    background: var(--input-well);
    border: 1px solid transparent;
    border-radius: var(--radius);
  }
  /* The tinted read-only mirror and the textarea share identical
   * geometry so the transparent-text textarea's caret/selection line up
   * with the mirror's glyphs. */
  .tint-mirror,
  .buffer-text,
  .inspect {
    margin: 0;
    padding: var(--space-4);
    font-family: var(--font-mono);
    font-size: var(--text);
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    box-sizing: border-box;
  }
  .tint-mirror {
    position: absolute;
    inset: 0;
    color: var(--fg-strong);
    pointer-events: none;
    overflow: hidden;
  }
  .tint-mirror .tok {
    border-radius: var(--radius);
  }

  /* Inspect mode — read-only clickable token view.  Same geometry as the
   * textarea / mirror (shared rule above) so wrapping matches; each
   * generated token is a button that opens the drilldown drawer. */
  .inspect {
    min-height: 100%;
    color: var(--fg-strong);
  }
  .inspect .plain {
    /* Prompt text (user / system turns) — present but not interactive. */
    color: var(--fg);
  }
  .inspect .tok {
    border-radius: var(--radius);
  }
  .inspect .clickable {
    cursor: pointer;
  }
  .inspect .clickable:hover,
  .inspect .clickable:focus-visible {
    outline: 1px solid var(--fg-muted);
    outline-offset: -1px;
    border-radius: var(--radius);
  }
  /* Tokens that captured top-K alternatives get a faint underline so the
   * forkable ones are discoverable at a glance. */
  .inspect .has-alts {
    text-decoration: underline dotted var(--fg-dim);
    text-underline-offset: 2px;
  }

  /* Header: label + edit/inspect toggle. */
  .raw-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .head-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0;
  }
  /* Segmented control — mirrors the house ``.sk-mode-tabs`` (ModeTabs)
   * and render-mode shape: borderless segments inside one bordered
   * ``--bg-elev`` track, --text-sm, accent-subtle active fill, fast
   * transitions.  Kept local (not the shared component) so the inspect
   * segment can carry a conditional ``disabled`` + dynamic tooltip. */
  .mode-toggle {
    display: flex;
    gap: var(--space-1);
    padding: var(--space-1);
    background: var(--bg-elev);
    border: 1px solid transparent;
    border-radius: var(--radius);
  }
  .mode-toggle button {
    background: transparent;
    color: var(--fg-dim);
    border: 0;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-4);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    text-transform: lowercase;
    transition:
      background var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out);
  }
  .mode-toggle button:hover:not(:disabled):not(.active) {
    color: var(--fg);
    background: var(--bg-hover);
  }
  .mode-toggle button.active {
    color: var(--accent);
    background: var(--accent-subtle);
  }
  .mode-toggle button:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 1px;
  }
  .mode-toggle button:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
    opacity: 0.6;
  }
  .buffer-text {
    position: relative;
    width: 100%;
    min-height: 100%;
    background: transparent;
    color: var(--fg-strong);
    border: 0;
    resize: none;
  }
  .buffer-text:focus {
    outline: none;
  }
  /* When the tinted mirror is showing, hide the textarea's own glyphs
   * so only the mirror paints — the textarea still owns the caret. */
  .buffer-text.has-tint {
    color: transparent;
    caret-color: var(--accent);
  }
  .buffer-text.has-tint::selection {
    background: var(--accent-glow);
  }

  .actions {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    justify-content: flex-end;
  }
  .dirty-flag {
    color: var(--accent-yellow);
    font-size: var(--text-xs);
    margin-right: auto;
  }
  .act {
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    padding: var(--space-2) var(--space-5);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
    border-radius: var(--radius);
  }
  .act:hover:not(:disabled) {
    background: var(--glass-strong);
  }
  .act:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
  .act.continue {
    color: var(--accent-green);
  }
  .act.stop {
    color: var(--accent-red);
  }
  .act.commit {
    color: var(--accent);
  }
  .act.revert {
    color: var(--fg-dim);
  }
</style>
