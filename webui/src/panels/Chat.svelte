<script lang="ts">
  // Chat panel — v1.7 rewrite.  Replaces the v1.6 ChatPlaceholder/legacy
  // shape with the full feature set: thinking-collapsible per assistant
  // turn, per-token tinted spans driven by a top-bar highlight dropdown,
  // optional compare-two stripe overlay, click-token drilldown, send /
  // stop / stateless toggle, and an A/B split-view container.
  //
  // Single source of truth for state lives in ``lib/stores.svelte`` —
  // this file is presentation + local-only UI bits (textarea state,
  // scroll bookkeeping, per-turn collapse state).  The WS lifecycle and
  // gen-status accounting belongs to the store.
  //
  // Mirrors saklas/tui/chat_panel.py for the visual rhythm: leading
  // whitespace strip after </think>, role-coloured left borders, plain
  // text fall-through when no probe is selected.

  import { onMount, untrack } from "svelte";
  import { SvelteMap } from "svelte/reactivity";
  import {
    chatLog,
    highlightState,
    setHighlightTarget,
    setCompareTarget,
    toggleCompareTwo,
    abState,
    toggleAb,
    probeRack,
    sendGenerate,
    sendStop,
    genStatus,
    openDrawer,
  } from "../lib/stores.svelte";
  import type { ChatTurn, TokenScore } from "../lib/types";
  import {
    scoreToRgb,
    twoStripeStyle,
    twoBlendStyle,
    formatScoreTooltip,
  } from "../lib/tokens";

  // --------------------------------------------------------------- input --

  let input = $state("");
  let textareaRef: HTMLTextAreaElement | null = $state(null);
  /** Per-call stateless override.  When on the next ``send`` is dispatched
   * with ``stateless: true`` so the server never appends to history.
   * Mirrors the TUI's ``◯/●`` indicator described in the plan. */
  let statelessNext = $state(false);

  /** Auto-grow the textarea between 1 and 6 rows (≈ 132px at 13px line-h). */
  function autosize(): void {
    const el = textareaRef;
    if (!el) return;
    el.style.height = "auto";
    const rowHeight = 22; // mono line-height fudge — matches font-size-base
    const maxH = rowHeight * 6;
    el.style.height = `${Math.min(el.scrollHeight, maxH)}px`;
  }

  $effect(() => {
    // Run autosize whenever ``input`` changes — bind:value + an effect is
    // simpler than wiring an oninput handler that has to cooperate with
    // bind:.
    void input;
    autosize();
  });

  function doSend(): void {
    const text = input.trim();
    if (!text) return;
    const stateless = statelessNext;
    statelessNext = false;
    input = "";
    // Defer the actual send so the textarea clears before the WS round-
    // trip — feels less like the UI froze.
    void sendGenerate(text, { stateless });
    // Force-scroll to bottom on send regardless of where the user was.
    scrolledUp = false;
    queueScrollToBottom();
    // Snap textarea height back.
    queueMicrotask(autosize);
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Enter") {
      // Cmd/Ctrl-Enter always sends; bare Enter sends; Shift-Enter newline.
      if (ev.shiftKey) return;
      ev.preventDefault();
      doSend();
      return;
    }
    if (ev.key === "Escape" && genStatus.active) {
      ev.preventDefault();
      sendStop();
    }
  }

  // ------------------------------------------------------------- highlight --

  /** All probe names available to the highlight dropdowns.  Sourced from
   * the live probe-rack — same source the ProbeRack panel uses. */
  const probeNames = $derived([...probeRack.active]);

  function onHighlightChange(ev: Event): void {
    const value = (ev.currentTarget as HTMLSelectElement).value;
    setHighlightTarget(value === "" ? null : value);
  }

  function onCompareChange(ev: Event): void {
    const value = (ev.currentTarget as HTMLSelectElement).value;
    setCompareTarget(value === "" ? null : value);
  }

  function onCompareToggle(): void {
    toggleCompareTwo();
  }

  // ------------------------------------------------------------- A/B split --

  /** True when AB is on AND we have any turn carrying an ``abPair``.  Even
   * if no ab pair exists we still want to render two columns when the
   * mode is on, so ``abVisible`` is just the toggle. */
  const abEnabled = $derived(abState.enabled);

  // ----------------------------------------------------------- per-turn UI --

  /** Per-turn thinking-collapsed state.  Keyed by turn index so a re-
   * render of the chat log preserves user-explicit collapse choices.  We
   * default to "collapsed" on creation and auto-expand when the first
   * thinking token lands; on done the turn collapses again unless the
   * user manually expanded.
   *
   * SvelteMap (not plain Map) — Svelte 5's $state doesn't track plain
   * Map mutations, so a bare Map.set wouldn't re-render the toggle UI. */
  const collapsedThinking: SvelteMap<number, boolean> = $state(new SvelteMap());

  function turnCollapsed(turnIdx: number, turn: ChatTurn): boolean {
    const explicit = collapsedThinking.get(turnIdx);
    if (explicit !== undefined) return explicit;
    // Default: expanded while in-flight (so the user can watch it
    // generate), collapsed once the gen lands.
    const inFlight =
      chatLog.pendingIndex === turnIdx && (turn.thinkingTokens?.length ?? 0) > 0;
    return !inFlight;
  }

  function toggleThinking(turnIdx: number): void {
    const cur = collapsedThinking.get(turnIdx) ?? true;
    collapsedThinking.set(turnIdx, !cur);
  }

  // ------------------------------------------------------ scroll bookkeeping --

  let logRef: HTMLDivElement | null = $state(null);
  /** True iff the user has manually scrolled up — freezes auto-scroll
   * until they hit the bottom again.  Mirrors the TUI's "scroll_end on
   * append unless user is mid-scroll" pattern. */
  let scrolledUp = $state(false);

  function onScroll(ev: Event): void {
    const el = ev.currentTarget as HTMLElement;
    // 8px slop so a scrollbar that doesn't quite hit the floor still
    // counts as "at bottom".
    const atBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 8;
    scrolledUp = !atBottom;
  }

  function scrollToBottom(): void {
    const el = logRef;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }

  let _scrollScheduled = false;
  function queueScrollToBottom(): void {
    if (_scrollScheduled) return;
    _scrollScheduled = true;
    queueMicrotask(() => {
      _scrollScheduled = false;
      if (!scrolledUp) scrollToBottom();
    });
  }

  // Auto-scroll on new turns or token deltas.  Reads (not writes) the
  // length-aggregates that drive the chat — Svelte 5 tracks these via
  // the runes graph.
  $effect(() => {
    // Touch the things we care about so the effect re-runs on changes.
    void chatLog.turns.length;
    const lastTurn = chatLog.turns[chatLog.turns.length - 1];
    void lastTurn?.tokens?.length;
    void lastTurn?.thinkingTokens?.length;
    void lastTurn?.text;
    untrack(() => queueScrollToBottom());
  });

  onMount(() => {
    autosize();
    scrollToBottom();
    textareaRef?.focus();
  });

  // ----------------------------------------------------------- token render --

  /** Score lookup for a single token against the currently-selected
   * probe.  Falls back to the cached single-probe ``score`` field when
   * the full ``probes`` map is missing (live tokens before done). */
  function pickScore(t: TokenScore, probe: string | null): number | undefined {
    if (!probe) return undefined;
    if (t.probes && probe in t.probes) return t.probes[probe];
    // The store's ``handleWsMessage`` sets ``t.score`` against the
    // current highlight target, so it's safe to use as a hot-path fall-
    // back when the per-probe map hasn't been recorded yet.
    return t.score;
  }

  /** Build the inline-style object for one token's background.  Compare-
   * two needs both probes set; if only one of the two is configured we
   * gracefully fall back to single-probe rendering. */
  function tokenStyle(
    t: TokenScore,
  ): { backgroundColor?: string; backgroundImage?: string } {
    const a = highlightState.target;
    if (!a) return {};
    const aScore = pickScore(t, a);
    if (highlightState.compareTwo && highlightState.compareTarget) {
      const bScore = pickScore(t, highlightState.compareTarget);
      return highlightState.smoothBlend
        ? twoBlendStyle(aScore, bScore)
        : twoStripeStyle(aScore, bScore);
    }
    const bg = scoreToRgb(aScore);
    return bg === "transparent" ? {} : { backgroundColor: bg };
  }

  function styleString(style: {
    backgroundColor?: string;
    backgroundImage?: string;
  }): string {
    const parts: string[] = [];
    if (style.backgroundColor) {
      parts.push(`background-color: ${style.backgroundColor}`);
    }
    if (style.backgroundImage) {
      parts.push(`background-image: ${style.backgroundImage}`);
    }
    return parts.join(";");
  }

  function tooltipFor(t: TokenScore): string {
    if (t.probes) return formatScoreTooltip(t.probes);
    if (t.score !== undefined && highlightState.target) {
      return `${highlightState.target} ${
        t.score >= 0 ? "+" : ""
      }${t.score.toFixed(3)}`;
    }
    return "";
  }

  /** Apply the TUI's leading-whitespace strip — drops whitespace-only
   * tokens from the head of the response so the gap below ``</think>``
   * goes away in plain-text mode too.  Returns the surviving slice
   * starting at the first non-whitespace token. */
  function stripLeadingWhitespace(tokens: TokenScore[]): TokenScore[] {
    let i = 0;
    while (i < tokens.length && !tokens[i].text.trim()) i++;
    return i === 0 ? tokens : tokens.slice(i);
  }

  function tokenClicked(
    turnIdx: number,
    tokenIdx: number,
    ev: MouseEvent,
    isThinking: boolean = false,
  ): void {
    ev.stopPropagation();
    // Pass ``isThinking`` through so the drilldown drawer reads from
    // ``turn.thinkingTokens`` when the click came from the thinking
    // body (otherwise it would index the response stream and either
    // miss or surface the wrong token).
    openDrawer("token_drilldown", { turnIdx, tokenIdx, isThinking });
  }

  // The bare-text form for plain (no-highlight) rendering.  We still want
  // the leading-whitespace strip so the chat surface matches the TUI
  // even when no probe is selected.
  function plainResponseText(turn: ChatTurn): string {
    if (!turn.tokens || turn.tokens.length === 0) {
      // Fall back to the accumulated text if the per-token list is
      // unset (extremely early in a stream, or for non-streamed loads).
      return (turn.text ?? "").replace(/^\s+/, "");
    }
    return stripLeadingWhitespace(turn.tokens)
      .map((t) => t.text)
      .join("");
  }
</script>

<div class="chat" aria-label="Chat">
  <header class="chat-header">
    <label class="ctl">
      <span class="ctl-label">highlight</span>
      <select
        class="ctl-select"
        value={highlightState.target ?? ""}
        onchange={onHighlightChange}
        aria-label="Highlight probe"
      >
        <option value="">(off)</option>
        {#each probeNames as name (name)}
          <option value={name}>{name}</option>
        {/each}
      </select>
    </label>

    <label class="ctl ctl-inline">
      <input
        type="checkbox"
        checked={highlightState.compareTwo}
        onchange={onCompareToggle}
      />
      <span class="ctl-label">compare-two</span>
    </label>

    {#if highlightState.compareTwo}
      <label class="ctl">
        <span class="ctl-label">vs.</span>
        <select
          class="ctl-select"
          value={highlightState.compareTarget ?? ""}
          onchange={onCompareChange}
          disabled={!highlightState.compareTwo}
          aria-label="Compare probe"
        >
          <option value="">(off)</option>
          {#each probeNames as name (name)}
            {#if name !== highlightState.target}
              <option value={name}>{name}</option>
            {/if}
          {/each}
        </select>
      </label>
    {/if}

    <button
      type="button"
      class="ctl-toggle"
      class:active={abEnabled}
      onclick={toggleAb}
      title="A/B compare: render unsteered shadow alongside steered"
      aria-pressed={abEnabled}
    >A/B</button>
  </header>

  <div
    class="log"
    class:ab={abEnabled}
    bind:this={logRef}
    onscroll={onScroll}
    role="log"
    aria-live="polite"
  >
    {#if abEnabled}
      <!-- A/B split: two columns of the same conversation, primary on
           left, unsteered shadow on right.  When ``abPair`` is unset on
           an assistant turn the right column shows a placeholder so the
           grid stays aligned. -->
      <div class="ab-grid">
        <div class="ab-col ab-primary">
          {#each chatLog.turns as turn, turnIdx (turnIdx)}
            {@render bubble(turn, turnIdx, false)}
          {/each}
        </div>
        <div class="ab-col ab-shadow">
          {#each chatLog.turns as turn, turnIdx (turnIdx)}
            {#if turn.role === "user" || turn.role === "system"}
              {@render bubble(turn, turnIdx, false)}
            {:else if turn.abPair}
              {@render bubble(turn.abPair, turnIdx, true)}
            {:else}
              <div class="msg assistant placeholder" aria-hidden="true">
                <span class="role">assistant (unsteered)</span>
                <span class="placeholder-text">— pending —</span>
              </div>
            {/if}
          {/each}
        </div>
      </div>
    {:else}
      {#each chatLog.turns as turn, turnIdx (turnIdx)}
        {@render bubble(turn, turnIdx, false)}
      {/each}
    {/if}
  </div>

  <form class="input-row" onsubmit={(ev) => { ev.preventDefault(); doSend(); }}>
    <textarea
      class="input"
      bind:this={textareaRef}
      bind:value={input}
      onkeydown={onKeydown}
      placeholder="message…  (enter to send · shift-enter newline · cmd/ctrl-enter also sends)"
      rows="1"
      aria-label="Chat input"
    ></textarea>
    <div class="input-actions">
      <button
        type="button"
        class="stateless"
        class:on={statelessNext}
        onclick={() => (statelessNext = !statelessNext)}
        title="Stateless: send next message without appending to history"
        aria-pressed={statelessNext}
      >{statelessNext ? "●" : "◯"} stateless</button>
      <button
        type="submit"
        class="send"
        disabled={!input.trim()}
        title="Enter or Cmd/Ctrl-Enter"
      >send</button>
      <button
        type="button"
        class="stop"
        onclick={sendStop}
        disabled={!genStatus.active}
        title="Esc"
      >stop</button>
    </div>
  </form>
</div>

{#snippet bubble(turn: ChatTurn, turnIdx: number, isShadow: boolean)}
  <div
    class="msg {turn.role}"
    class:shadow={isShadow}
    class:streaming={chatLog.pendingIndex === turnIdx}
  >
    <span class="role">
      {#if turn.role === "user"}user{:else if turn.role === "system"}system{:else}assistant{#if isShadow} (unsteered){/if}{/if}
    </span>

    {#if turn.role === "assistant"}
      {#if (turn.thinkingTokens?.length ?? 0) > 0 || turn.thinking}
        <div class="thinking-block" class:collapsed={turnCollapsed(turnIdx, turn)}>
          <button
            type="button"
            class="thinking-toggle"
            onclick={() => toggleThinking(turnIdx)}
            aria-expanded={!turnCollapsed(turnIdx, turn)}
          >
            <span class="caret">{turnCollapsed(turnIdx, turn) ? "▶" : "▼"}</span>
            <span>thinking{turnCollapsed(turnIdx, turn) ? "…" : ""}</span>
          </button>
          {#if !turnCollapsed(turnIdx, turn)}
            <div class="thinking-body">
              {#if (turn.thinkingTokens?.length ?? 0) > 0}
                {#each turn.thinkingTokens ?? [] as tok, tokenIdx (tokenIdx)}
                  <span
                    class="tok"
                    class:tinted={highlightState.target !== null}
                    style={styleString(tokenStyle(tok))}
                    title={tooltipFor(tok)}
                    onclick={(ev) => tokenClicked(turnIdx, tokenIdx, ev, true)}
                    onkeydown={(ev) => {
                      if (ev.key === "Enter" || ev.key === " ") {
                        ev.preventDefault();
                        ev.stopPropagation();
                        openDrawer("token_drilldown", { turnIdx, tokenIdx, isThinking: true });
                      }
                    }}
                    role="button"
                    tabindex="-1"
                  >{tok.text}</span>
                {/each}
              {:else}
                <span class="plain">{turn.text ?? ""}</span>
              {/if}
            </div>
          {/if}
        </div>
      {/if}

      <div class="response-body">
        {#if (turn.tokens?.length ?? 0) > 0}
          {#each stripLeadingWhitespace(turn.tokens ?? []) as tok, tokenIdx (tokenIdx)}
            <span
              class="tok"
              class:tinted={highlightState.target !== null}
              style={styleString(tokenStyle(tok))}
              title={tooltipFor(tok)}
              onclick={(ev) => tokenClicked(turnIdx, tokenIdx, ev, false)}
              onkeydown={(ev) => {
                if (ev.key === "Enter" || ev.key === " ") {
                  ev.preventDefault();
                  ev.stopPropagation();
                  openDrawer("token_drilldown", { turnIdx, tokenIdx });
                }
              }}
              role="button"
              tabindex="-1"
            >{tok.text}</span>
          {/each}
        {:else}
          <span class="plain">{plainResponseText(turn)}</span>
        {/if}
      </div>
    {:else}
      <div class="response-body"><span class="plain">{turn.text}</span></div>
    {/if}
  </div>
{/snippet}

<style>
  .chat {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    gap: 0.5em;
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
    color: var(--fg);
  }

  .chat-header {
    display: flex;
    align-items: center;
    gap: 0.8em;
    flex-wrap: wrap;
    padding-bottom: 0.4em;
    border-bottom: 1px solid var(--border-dim);
    color: var(--fg-dim);
    font-size: var(--font-size-small);
  }
  .ctl {
    display: inline-flex;
    align-items: center;
    gap: 0.35em;
  }
  .ctl-inline {
    cursor: pointer;
    user-select: none;
  }
  .ctl-label {
    color: var(--fg-muted);
    text-transform: lowercase;
    letter-spacing: 0.04em;
  }
  .ctl-select {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.15em 0.45em;
    font: inherit;
    font-family: var(--font-mono);
  }
  .ctl-select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .ctl-toggle {
    margin-left: auto;
    background: transparent;
    color: var(--fg-dim);
    border: 1px solid var(--border);
    padding: 0.2em 0.6em;
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .ctl-toggle:hover {
    border-color: var(--fg-muted);
    color: var(--fg-strong);
  }
  .ctl-toggle.active {
    color: var(--accent-blue);
    border-color: var(--accent-blue);
    background: rgba(88, 166, 255, 0.08);
  }

  .log {
    flex: 1 1 auto;
    overflow-y: auto;
    overflow-x: hidden;
    display: flex;
    flex-direction: column;
    gap: 0.7em;
    min-height: 0;
    padding-right: 0.4em;
  }

  .log.ab {
    /* Container itself stays vertical scroll; the inner ab-grid handles
     * the two-column rendering so each column shares the same scroll
     * surface. */
    display: block;
  }
  .ab-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8em;
  }
  .ab-col {
    display: flex;
    flex-direction: column;
    gap: 0.7em;
    min-width: 0;
  }
  .ab-col.ab-shadow .msg.assistant {
    border-left-color: var(--accent-purple);
  }

  .msg {
    border-left: 2px solid var(--border);
    padding: 0.1em 0.6em;
    display: flex;
    flex-direction: column;
    gap: 0.25em;
    min-width: 0;
    word-break: break-word;
  }
  .msg .role {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
    text-transform: lowercase;
    letter-spacing: 0.08em;
  }
  .msg.user {
    border-left-color: var(--accent-blue);
  }
  .msg.user .role {
    color: var(--accent-blue);
  }
  .msg.assistant {
    border-left-color: var(--accent-green);
  }
  .msg.assistant .role {
    color: var(--accent-green);
  }
  .msg.system {
    border-left-color: var(--accent-red);
    color: var(--accent-error);
  }
  .msg.system .role {
    color: var(--accent-red);
  }
  .msg.streaming {
    /* Subtle pulse on the in-flight turn so the user can see what's
     * actively being written. */
    background: rgba(126, 231, 135, 0.04);
  }
  .msg.placeholder {
    color: var(--fg-muted);
    font-style: italic;
    opacity: 0.6;
  }
  .placeholder-text {
    font-size: var(--font-size-small);
  }

  .response-body {
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--fg-strong);
    line-height: 1.45;
  }
  .plain {
    white-space: pre-wrap;
  }

  /* Thinking-collapsible block — visible-only header when collapsed,
   * with the body indented when expanded. */
  .thinking-block {
    border-top: 1px dashed var(--border-dim);
    border-bottom: 1px dashed var(--border-dim);
    margin-bottom: 0.2em;
  }
  .thinking-toggle {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    font: inherit;
    font-family: var(--font-mono);
    padding: 0.15em 0;
    cursor: pointer;
    text-align: left;
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
    width: 100%;
  }
  .thinking-toggle:hover {
    color: var(--fg-strong);
  }
  .thinking-toggle .caret {
    color: var(--fg-muted);
    width: 1ch;
    display: inline-block;
  }
  .thinking-body {
    padding: 0.2em 0 0.4em 1.6em;
    color: var(--fg-dim);
    font-style: italic;
    white-space: pre-wrap;
    line-height: 1.4;
  }
  .thinking-body .tok {
    font-style: italic;
  }

  /* Tokens — minimal padding so the tinted span hugs the glyph.  The
   * click handler attaches regardless of highlight state; ``.tinted``
   * marks rows whose background is being painted by the score so the
   * untinted hover outline only fires when there's no other visual.
   * Hover outline gives the click affordance even when highlighting is
   * off (matches the user-visible click contract). */
  .tok {
    cursor: pointer;
    border-radius: 1px;
  }
  .tok:hover {
    outline: 1px solid var(--fg-muted);
  }

  .input-row {
    display: flex;
    gap: 0.5em;
    align-items: flex-end;
    border-top: 1px solid var(--border-dim);
    padding-top: 0.5em;
  }
  .input {
    flex: 1 1 auto;
    background: var(--bg-alt);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: 0.4em 0.6em;
    font: inherit;
    font-family: var(--font-mono);
    resize: none;
    min-height: 1.6em;
    max-height: 132px;
    line-height: 1.45;
  }
  .input:focus {
    outline: none;
    border-color: var(--accent-blue);
  }
  .input-actions {
    display: flex;
    gap: 0.3em;
    align-items: center;
  }
  .input-actions button {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.4em 0.8em;
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
  }
  .input-actions button:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .input-actions button:disabled {
    color: var(--fg-muted);
    border-color: var(--border-dim);
    cursor: not-allowed;
  }
  .input-actions .send {
    color: var(--accent-green);
  }
  .input-actions .stop {
    color: var(--accent-red);
  }
  .input-actions .stateless {
    color: var(--fg-dim);
    font-size: var(--font-size-small);
  }
  .input-actions .stateless.on {
    color: var(--accent-blue);
    border-color: var(--accent-blue);
  }
</style>
