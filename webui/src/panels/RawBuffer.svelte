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
    loomEdit,
    loomBranch,
    genStatus,
    sendGenerate,
    sendStop,
    highlightState,
  } from "../lib/stores.svelte";
  import type { ChatTurn, TokenScore } from "../lib/types";
  import { scoreToRgb, surpriseScore, SURPRISE_TARGET } from "../lib/tokens";

  // ---------- buffer text ----------

  /** The conversation flattened to one string — every turn's text
   *  concatenated in order.  This is what the model sees as a prefill
   *  on "continue". */
  const bufferText = $derived(
    chatLog.turns.map((t) => t.text ?? "").join(""),
  );

  // Local editable mirror.  Synced from ``bufferText`` whenever the
  // tree changes and the user isn't mid-edit; user edits write here
  // and a "commit edit" lands them on the tree.
  let draft = $state("");
  let dirty = $state(false);
  let textareaRef: HTMLTextAreaElement | null = $state(null);

  // Re-sync the draft from the tree when the buffer changes and the
  // user has no pending edit.  ``dirty`` guards against clobbering an
  // in-progress edit when a token stream updates ``bufferText``.
  $effect(() => {
    const text = bufferText;
    if (!dirty) {
      draft = text;
    }
  });

  function onInput(ev: Event): void {
    draft = (ev.currentTarget as HTMLTextAreaElement).value;
    dirty = draft !== bufferText;
  }

  // ---------- continue ----------

  /** The active leaf node — what a raw continuation extends. */
  const activeLeaf = $derived(
    loomTree.rev > 0 ? (loomTree.active_node_id ?? null) : null,
  );

  function continueGen(): void {
    if (genStatus.active) return;
    // If the buffer carries an uncommitted edit, land it first so the
    // continuation extends the edited text, not the stale leaf.
    if (dirty) {
      void commitEdit().then(() => fireContinue());
      return;
    }
    fireContinue();
  }

  function fireContinue(): void {
    // The whole buffer is the prefill; an empty input string with
    // ``raw: true`` continues from the active leaf's accumulated text.
    void sendGenerate("", { raw: true });
  }

  // ---------- edit ----------

  /** Commit the edited buffer to the tree.
   *
   *  The buffer is the join of every turn's text, so an edit anywhere
   *  in it has to be mapped back to the loom node(s) it actually
   *  touched — silently keeping only the tail would discard a prefix
   *  edit, and a flat completion buffer means the *whole* buffer is the
   *  editable context.
   *
   *  Strategy: walk the turns accumulating their original start offsets
   *  and find the first turn whose contribution to the buffer diverges
   *  from the draft.  From that divergence point on, turn boundaries no
   *  longer line up (an edit shifts every later offset), so the edited
   *  tail — everything from the diverging turn's start to the end of
   *  the draft — is treated as one continuous completion and committed
   *  against that turn's node:
   *
   *    - Leaf, unchanged subtree → ``loomEdit`` in place.
   *    - Otherwise (interior node, or a node with children) → branch a
   *      sibling carrying the edited tail so no subtree is orphaned.
   *
   *  Branching is always-sibling and ``make_active`` on the server, so
   *  the new node becomes the active leaf and a follow-up continuation
   *  extends the edited text. */
  async function commitEdit(): Promise<void> {
    if (!dirty) return;
    const turns = chatLog.turns;
    if (turns.length === 0) return;

    // Find the first turn whose text region diverges from the draft.
    // ``startOffset`` is that turn's offset into the original buffer
    // (== its offset into the draft up to the divergence point, since
    // everything before it matches byte-for-byte).
    let startOffset = 0;
    let divergeIdx = -1;
    for (let i = 0; i < turns.length; i++) {
      const turnText = turns[i].text ?? "";
      const draftSlice = draft.slice(startOffset, startOffset + turnText.length);
      if (draftSlice !== turnText) {
        divergeIdx = i;
        break;
      }
      startOffset += turnText.length;
    }

    if (divergeIdx === -1) {
      // No per-turn slice diverged, yet ``dirty`` is set — the draft is
      // strictly longer than the joined buffer (the user appended past
      // the final turn).  That tail belongs on the active leaf.
      const leaf = activeLeaf;
      if (!leaf) {
        dirty = false;
        return;
      }
      const leafTurn = turns[turns.length - 1];
      const leafStart = startOffset - (leafTurn.text ?? "").length;
      const leafText = draft.slice(leafStart);
      const children = loomTree.children_of.get(leaf) ?? [];
      if (children.length > 0) {
        await loomBranch(leaf, leafText);
      } else {
        await loomEdit(leaf, leafText);
      }
      dirty = false;
      return;
    }

    // The diverging turn's node owns the edit.  Everything from its
    // start offset to the end of the draft is the new (flat) text.
    const target = turns[divergeIdx];
    const targetNode = target.nodeId ?? null;
    if (!targetNode) {
      // No backing node id (legacy single-path render with rev 0) —
      // fall back to editing the active leaf so the edit isn't lost.
      const leaf = activeLeaf;
      if (leaf) await loomEdit(leaf, draft.slice(startOffset));
      dirty = false;
      return;
    }
    const newText = draft.slice(startOffset);
    const children = loomTree.children_of.get(targetNode) ?? [];
    const isLeaf = children.length === 0 && targetNode === activeLeaf;
    if (isLeaf) {
      // Editing the tail node in place — no subtree to orphan.
      await loomEdit(targetNode, newText);
    } else {
      // The edit reaches an interior turn (it has children, or later
      // turns follow it).  Branch a sibling carrying the whole edited
      // tail so the original subtree is preserved.
      await loomBranch(targetNode, newText);
    }
    dirty = false;
  }

  function revertEdit(): void {
    draft = bufferText;
    dirty = false;
  }

  function onKeydown(ev: KeyboardEvent): void {
    // Ctrl/Cmd+Enter continues; Escape stops an in-flight gen.
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault();
      continueGen();
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

  const allTokens = $derived.by<TokenScore[]>(() => {
    const out: TokenScore[] = [];
    for (const t of chatLog.turns) {
      if (t.thinkingTokens) out.push(...t.thinkingTokens);
      if (t.tokens) out.push(...t.tokens);
      else if (!t.tokens && t.role !== "assistant") {
        // User / system turns have no token rows — represent them as a
        // single untinted span so the mirror text matches the buffer.
        out.push({ text: t.text ?? "", thinking: false });
      }
    }
    return out;
  });

  /** True when a tinted read-only mirror should render — a highlight
   *  probe is selected and the buffer isn't being actively edited. */
  const showTint = $derived(
    highlightState.target !== null && !dirty && allTokens.length > 0,
  );

  function tokenScore(t: TokenScore): number | undefined {
    const target = highlightState.target;
    if (!target) return undefined;
    if (target === SURPRISE_TARGET) return surpriseScore(t.logprob);
    if (t.probes && target in t.probes) return t.probes[target];
    return t.score;
  }

  function tintStyle(t: TokenScore): string {
    const bg = scoreToRgb(tokenScore(t));
    return bg === "transparent" ? "" : `background-color: ${bg}`;
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
  <div class="surface" bind:this={logRef} onscroll={onScroll}>
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
      placeholder="(empty completion buffer — type a prompt and continue)"
      aria-label="Editable completion buffer"
    ></textarea>
  </div>

  <StatusFooter />
  <PendingBubbles />

  <div class="actions">
    {#if dirty}
      <span class="dirty-flag" title="the buffer has uncommitted edits">
        edited
      </span>
      <button type="button" class="act commit" onclick={() => void commitEdit()}>
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
      title="⌃⏎ — generate a raw continuation from the buffer"
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
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }
  /* The tinted read-only mirror and the textarea share identical
   * geometry so the transparent-text textarea's caret/selection line up
   * with the mirror's glyphs. */
  .tint-mirror,
  .buffer-text {
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
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-2) var(--space-5);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
    border-radius: var(--radius);
  }
  .act:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
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
