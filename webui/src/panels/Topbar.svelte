<script lang="ts">
  // Topbar: brand · model/device/dtype · action buttons + tools dropdown +
  // pending-actions badge + stop.  All mutating buttons route through the
  // store's ``enqueueOrApply`` machinery — clicking "clear" mid-gen drops
  // a pending action onto the queue rather than racing the WS.

  import {
    sessionState,
    genStatus,
    pendingActions,
    clearSessionHistory,
    rewindSession,
    sendStop,
    sendGenerate,
    applyPendingActions,
    openDrawer,
    chatLog,
    enqueuePending,
    onWsMessage,
  } from "../lib/stores.svelte";
  import type { DrawerName } from "../lib/types";
  import { onMount } from "svelte";

  let toolsOpen = $state(false);
  let toolsRef: HTMLDivElement | null = $state(null);

  // Close tools menu on outside click or Escape.
  function onDocClick(ev: MouseEvent) {
    if (!toolsOpen) return;
    if (toolsRef && !toolsRef.contains(ev.target as Node)) toolsOpen = false;
  }
  function onDocKey(ev: KeyboardEvent) {
    if (ev.key === "Escape" && toolsOpen) toolsOpen = false;
  }

  onMount(() => {
    document.addEventListener("click", onDocClick);
    document.addEventListener("keydown", onDocKey);
    return () => {
      document.removeEventListener("click", onDocClick);
      document.removeEventListener("keydown", onDocKey);
    };
  });

  // Loaded = session info has populated.  Buttons gate on this so a fresh
  // page doesn't flicker with disabled→enabled mid-bootstrap.
  const loaded = $derived(sessionState.info !== null);

  const status = $derived.by(() => {
    const info = sessionState.info;
    if (!info) return "connecting…";
    const thinking = info.supports_thinking ? "thinking" : "no-thinking";
    return `${info.model_id} · ${info.device}/${info.dtype} · ${thinking}`;
  });

  const pendingCount = $derived(pendingActions.queue.length);

  // Last user input — used by "regen" to re-issue the message.  Pulled
  // off chatLog rather than tracked separately so we don't re-implement
  // the "find the last user turn" walk in two places.
  function lastUserInput(): string | null {
    for (let i = chatLog.turns.length - 1; i >= 0; i--) {
      if (chatLog.turns[i].role === "user") return chatLog.turns[i].text;
    }
    return null;
  }

  /** Proper regen: rewind one user→assistant pair on both server and
   * local log, then re-send the user input.  Without the rewind, the
   * server's history is unchanged and the new gen lands as a fresh
   * appended turn — looking visually like a duplicate-then-continue. */
  async function regen(input: string): Promise<void> {
    await rewindSession();
    void sendGenerate(input);
  }

  function clear(): void {
    if (genStatus.active) {
      enqueuePending({ label: "clear", apply: () => void clearSessionHistory() });
    } else {
      void clearSessionHistory();
    }
  }

  function rewind(): void {
    if (genStatus.active) {
      enqueuePending({ label: "rewind", apply: () => void rewindSession() });
    } else {
      void rewindSession();
    }
  }

  function regenAction(): void {
    // Capture the input *now* — by the time a queued action fires, the
    // local chat log may have shifted (e.g. another regen got queued
    // first).  The rewind+resend pair is anchored to the message the
    // user actually intended to regenerate.
    const input = lastUserInput();
    if (input === null) return;
    if (genStatus.active) {
      enqueuePending({ label: "regen", apply: () => void regen(input) });
    } else {
      void regen(input);
    }
  }

  function pickTool(name: DrawerName): void {
    toolsOpen = false;
    openDrawer(name);
  }

  // Stop-then-flush.  Subscribes once to the WS done event, sends stop,
  // then drains the queue when done lands.  Off the happy path: if no gen
  // is active, just flush immediately.
  function applyNow(): void {
    if (!genStatus.active) {
      applyPendingActions();
      return;
    }
    const off = onWsMessage((msg) => {
      if (msg.type === "done" || msg.type === "error") {
        // Store's own handler already drains the queue on done/error.
        off();
      }
    });
    sendStop();
  }
</script>

<header class="topbar" aria-label="saklas top bar">
  <div class="left">
    <span class="brand">saklas</span>
  </div>

  <div class="center">
    <span class="status" title={sessionState.error ?? undefined}>
      {status}
    </span>
  </div>

  <div class="right">
    {#if pendingCount > 0}
      <button
        class="pending-badge"
        type="button"
        onclick={applyNow}
        title="Apply queued rack/sampling changes; interrupts in-flight gen if needed"
      >
        {pendingCount} change{pendingCount === 1 ? "" : "s"} pending — apply now
      </button>
    {/if}

    <button
      type="button"
      class="action"
      disabled={!loaded}
      onclick={clear}
      title="Clear chat history (Ctrl/Cmd-K)"
    >
      clear
    </button>
    <button
      type="button"
      class="action"
      disabled={!loaded}
      onclick={rewind}
      title="Rewind to before the last user turn"
    >
      rewind
    </button>
    <button
      type="button"
      class="action"
      disabled={!loaded || lastUserInput() === null}
      onclick={regenAction}
      title="Re-issue the last user message (Cmd-Shift-R)"
    >
      regen
    </button>

    <div class="tools-wrap" bind:this={toolsRef}>
      <button
        type="button"
        class="action tools"
        disabled={!loaded}
        aria-haspopup="menu"
        aria-expanded={toolsOpen}
        onclick={(ev) => {
          ev.stopPropagation();
          toolsOpen = !toolsOpen;
        }}
      >
        tools
        <span class="caret" aria-hidden="true">▾</span>
      </button>
      {#if toolsOpen}
        <div class="tools-menu" role="menu">
          <button type="button" role="menuitem" onclick={() => pickTool("extract")}>extract vector…</button>
          <button type="button" role="menuitem" onclick={() => pickTool("load")}>load vector…</button>
          <button type="button" role="menuitem" onclick={() => pickTool("merge")}>merge vector…</button>
          <button type="button" role="menuitem" onclick={() => pickTool("clone")}>clone vector…</button>
          <button type="button" role="menuitem" onclick={() => pickTool("compare")}>compare vectors…</button>
          <button type="button" role="menuitem" onclick={() => pickTool("sweep")}>sweep…</button>
          <button type="button" role="menuitem" onclick={() => pickTool("pack")}>packs…</button>
          <hr />
          <button type="button" role="menuitem" onclick={() => pickTool("system_prompt")}>system prompt…</button>
          <button type="button" role="menuitem" onclick={() => pickTool("model_info")}>model info…</button>
          <hr />
          <button type="button" role="menuitem" onclick={() => pickTool("save_conversation")}>save conversation…</button>
          <button type="button" role="menuitem" onclick={() => pickTool("load_conversation")}>load conversation…</button>
          <hr />
          <button type="button" role="menuitem" onclick={() => pickTool("help")}>help / shortcuts…</button>
        </div>
      {/if}
    </div>

    <button
      type="button"
      class="action stop"
      disabled={!genStatus.active}
      onclick={sendStop}
      title="Stop current generation (Esc)"
      aria-label="Stop generation"
    >
      stop <span class="stop-glyph" aria-hidden="true">⏹</span>
    </button>
  </div>
</header>

<style>
  .topbar {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: 1.5em;
    padding: 0.5em 1em;
    background: var(--bg-deep);
    border-bottom: 1px solid var(--border);
    min-height: 36px;
  }
  .left {
    display: flex;
    align-items: center;
  }
  .center {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .right {
    display: flex;
    align-items: center;
    gap: 0.4em;
  }
  .brand {
    font-weight: bold;
    color: var(--accent-green);
    letter-spacing: 0.05em;
    font-size: 0.95em;
  }
  .status {
    color: var(--fg-dim);
    font-size: 0.9em;
  }
  .action {
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.25em 0.6em;
    border-radius: 3px;
    font-size: 0.85em;
    line-height: 1.3;
    transition: background 0.1s ease, border-color 0.1s ease;
  }
  .action:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .action:disabled {
    color: var(--fg-muted);
    border-color: var(--border-dim);
    cursor: not-allowed;
  }
  .stop {
    border-color: var(--accent-red);
    color: var(--accent-red);
  }
  .stop:disabled {
    border-color: var(--border-dim);
    color: var(--fg-muted);
  }
  .stop:hover:not(:disabled) {
    background: rgba(248, 81, 73, 0.12);
  }
  .stop-glyph {
    font-size: 0.9em;
    margin-left: 0.15em;
  }
  .caret {
    margin-left: 0.25em;
    color: var(--fg-muted);
  }
  .pending-badge {
    background: rgba(210, 153, 34, 0.15);
    color: var(--accent-yellow);
    border: 1px solid var(--accent-yellow);
    padding: 0.25em 0.6em;
    border-radius: 3px;
    font-size: 0.85em;
    margin-right: 0.4em;
  }
  .pending-badge:hover {
    background: rgba(210, 153, 34, 0.28);
  }
  .tools-wrap {
    position: relative;
  }
  .tools-menu {
    position: absolute;
    right: 0;
    top: calc(100% + 4px);
    min-width: 200px;
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.25em 0;
    z-index: var(--z-modal);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    display: flex;
    flex-direction: column;
  }
  .tools-menu button {
    background: transparent;
    border: 0;
    text-align: left;
    padding: 0.4em 0.8em;
    color: var(--fg-strong);
    font-size: 0.85em;
  }
  .tools-menu button:hover {
    background: var(--bg-elev);
    color: var(--accent-blue);
  }
  .tools-menu hr {
    border: 0;
    border-top: 1px solid var(--border-dim);
    margin: 0.2em 0;
  }
</style>
