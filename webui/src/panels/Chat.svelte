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
    autoRegenState,
    chatLog,
    highlightState,
    loomTree,
    pinnedComparison,
    setHighlightTarget,
    setCompareTarget,
    toggleCompareTwo,
    unpinComparison,
    probeRack,
    sendGenerate,
    sendStop,
    genStatus,
    openDrawer,
    inputHistory,
    pushInputHistory,
    navigateInputHistory,
    clearSessionHistory,
    rewindSession,
    enqueuePending,
    toggleAutoRegen,
    setAutoRegenMode,
    setAutoRegenCustom,
  } from "../lib/stores.svelte";
  import type { AutoRegenMode } from "../lib/stores.svelte";
  import type { ChatTurn, TokenScore } from "../lib/types";
  import {
    scoreToRgb,
    twoStripeStyle,
    twoBlendStyle,
    formatScoreTooltip,
    surpriseScore,
    SURPRISE_TARGET,
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
    // Push to ↑/↓ recall before clearing — covers both chat messages
    // and slash commands (every line typed in here is recallable).
    pushInputHistory(text);
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

  /** Edge-only multi-line policy: ↑ recalls history when the cursor
   *  sits on the first line of the draft; ↓ goes forward only on the
   *  last line.  In-between lines fall through to the textarea's
   *  native cursor nav so multi-line editing isn't hijacked. */
  function shouldRecallUp(ta: HTMLTextAreaElement): boolean {
    const value = ta.value;
    const cursor = ta.selectionStart ?? 0;
    const firstNL = value.indexOf("\n");
    return firstNL === -1 || cursor <= firstNL;
  }

  function shouldRecallDown(ta: HTMLTextAreaElement): boolean {
    const value = ta.value;
    const cursorEnd = ta.selectionEnd ?? value.length;
    const lastNL = value.lastIndexOf("\n");
    return lastNL === -1 || cursorEnd > lastNL;
  }

  function applyRecalled(text: string): void {
    input = text;
    // Defer cursor placement past the bind:value flush so the textarea
    // reflects the new value before we set the selection.
    queueMicrotask(() => {
      const el = textareaRef;
      if (el) {
        el.setSelectionRange(el.value.length, el.value.length);
        autosize();
      }
    });
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
      return;
    }
    if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
      const ta = textareaRef;
      if (!ta) return;
      const goingUp = ev.key === "ArrowUp";
      if (goingUp ? !shouldRecallUp(ta) : !shouldRecallDown(ta)) return;
      // ↓ at the live slot (no recall in flight) is a no-op — leave
      // the keystroke for the textarea so it can move within an empty
      // last line or trigger the browser's native end-of-input nudge.
      if (!goingUp && inputHistory.index === null) return;
      const recalled = navigateInputHistory(goingUp ? -1 : +1, input);
      if (recalled === null) return;
      ev.preventDefault();
      applyRecalled(recalled);
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

  // -------------------------------------------------- conversation actions --
  //
  // clear / rewind / regen / transcript / auto-regen used to live on the
  // Topbar; they act on the conversation, so they belong here.  The
  // mutating ones route through ``enqueuePending`` so clicking them mid-
  // gen queues rather than racing the WS.

  const AUTO_REGEN_MODES: { value: AutoRegenMode; label: string }[] = [
    { value: "unsteered", label: "unsteered" },
    { value: "inverted", label: "inverted" },
    { value: "reseed", label: "reseed" },
    { value: "cool", label: "cool" },
    { value: "hot", label: "hot" },
    { value: "custom", label: "custom…" },
  ];

  /** Last user input — used by regen to re-issue the message. */
  function lastUserInput(): string | null {
    for (let i = chatLog.turns.length - 1; i >= 0; i--) {
      if (chatLog.turns[i].role === "user") return chatLog.turns[i].text;
    }
    return null;
  }

  /** Rewind one user→assistant pair then re-send the captured input —
   * without the rewind the new gen lands as an appended duplicate. */
  async function regen(input: string): Promise<void> {
    await rewindSession();
    void sendGenerate(input);
  }

  function clearChat(): void {
    if (genStatus.active) {
      enqueuePending({ label: "clear", apply: () => void clearSessionHistory() });
    } else {
      void clearSessionHistory();
    }
  }

  function rewindChat(): void {
    if (genStatus.active) {
      enqueuePending({ label: "rewind", apply: () => void rewindSession() });
    } else {
      void rewindSession();
    }
  }

  function regenAction(): void {
    // Capture the input now — a queued action fires later, by which point
    // the local log may have shifted.
    const input = lastUserInput();
    if (input === null) return;
    if (genStatus.active) {
      enqueuePending({ label: "regen", apply: () => void regen(input) });
    } else {
      void regen(input);
    }
  }

  const canRegen = $derived(lastUserInput() !== null);

  function openTranscript(): void {
    openDrawer("transcript");
  }

  function onAutoRegenModeChange(ev: Event): void {
    setAutoRegenMode(
      (ev.currentTarget as HTMLSelectElement).value as AutoRegenMode,
    );
  }

  // ------------------------------------------------------------- A/B split --

  /** v2.3: the standalone A/B toggle is gone.  The right column renders
   *  either a pinned sibling's path or — when auto-regen is on — the
   *  most recent auto-generated shadow / sibling.  The
   *  ``autoRegenState.enabled`` flag drives both branches; mode
   *  ``"unsteered"`` is the bit-identical fold of the old A/B. */
  const autoRegenActive = $derived(autoRegenState.enabled);

  /** Phase-5: the right column renders either the pinned sibling's
   *  subtree path or — when pinning is off — the auto-regen shadow.
   *  Auto-regen overwrites the pin with each new auto-generated
   *  sibling, so the same pane shows whichever sibling is "the other
   *  one" at this moment. */
  const pinnedActive = $derived(
    pinnedComparison.nodeId !== null &&
    loomTree.nodes.has(pinnedComparison.nodeId),
  );

  /** Render the conversation up to (and including) the pinned node by
   *  walking parent pointers from the pinned id back to root.  Skips
   *  the synthetic root.  Used by the right column when pinned. */
  const pinnedPath = $derived.by<ChatTurn[]>(() => {
    if (!pinnedActive || !pinnedComparison.nodeId) return [];
    const out: ChatTurn[] = [];
    let cursor: string | null = pinnedComparison.nodeId;
    const seen = new Set<string>();
    while (cursor && !seen.has(cursor)) {
      seen.add(cursor);
      const node = loomTree.nodes.get(cursor);
      if (!node) break;
      // Skip the synthetic root.
      if (!(node.parent_id === null && node.role === "system" && !node.text)) {
        out.push({
          role: node.role,
          text: node.text ?? "",
          nodeId: node.id,
          appliedSteering: node.applied_steering ?? null,
          aggregateReadings: node.aggregate_readings ?? undefined,
          finishReason: node.finish_reason ?? undefined,
        });
      }
      cursor = node.parent_id;
    }
    return out.reverse();
  });

  /** The right column is visible when EITHER auto-regen is on (which
   *  subsumes the v1.x A/B toggle) or a node is pinned for comparison. */
  const twoColumns = $derived(pinnedActive || autoRegenActive);

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
   * highlight target.  Handles the logit-pass ``SURPRISE_TARGET`` sentinel
   * by routing to ``surpriseScore``; for real probe names, reads
   * ``t.probes`` first and falls back to the cached single-probe
   * ``score`` field (live tokens before done). */
  function latestLayerScores(
    t: TokenScore,
  ): Record<string, number> | undefined {
    const pls = t.perLayerScores;
    if (!pls) return undefined;
    const layers = Object.keys(pls).sort((a, b) => Number(a) - Number(b));
    const last = layers[layers.length - 1];
    return last === undefined ? undefined : pls[last];
  }

  function pickScore(t: TokenScore, target: string | null): number | undefined {
    if (!target) return undefined;
    if (target === SURPRISE_TARGET) return surpriseScore(t.logprob);
    if (t.probes && target in t.probes) return t.probes[target];
    const latest = latestLayerScores(t);
    if (latest && target in latest) return latest[target];
    return t.score;
  }

  /** Build the inline-style object for one token's background.  Compare-
   * two needs both probes set; if only one of the two is configured we
   * gracefully fall back to single-probe rendering.  ``SURPRISE_TARGET``
   * works in either slot — the resulting score feeds the same
   * ``scoreToRgb`` ramp (positive half by construction). */
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

  /** Format the logprob suffix for the surprise-mode tooltip.  Includes
   *  the rank-of-K readout when ``top_alts`` was captured for this
   *  position so researchers can read "this is rank 1 of 8" at a glance. */
  function surpriseTooltip(t: TokenScore): string {
    if (t.logprob == null || !Number.isFinite(t.logprob)) {
      return "no logprob data";
    }
    const lp = `logprob = ${t.logprob.toFixed(3)}`;
    const alts = t.topAlts;
    if (!alts || alts.length === 0) return lp;
    // Look up the chosen token's rank within the captured alts.  Falls
    // back to text equality when ``tokenId`` is missing (legacy shape).
    let rank: number | null = null;
    for (let i = 0; i < alts.length; i++) {
      const a = alts[i];
      if (t.tokenId != null ? a.id === t.tokenId : a.text === t.text) {
        rank = i + 1;
        break;
      }
    }
    return rank !== null
      ? `${lp}, rank ${rank} of ${alts.length}`
      : `${lp}, chosen not in top-${alts.length}`;
  }

  function tooltipFor(t: TokenScore): string {
    // Logit-pass: surprise mode owns the tooltip when active so the
    // surprise number is what hovers on the inline tint.
    if (highlightState.target === SURPRISE_TARGET) return surpriseTooltip(t);
    if (
      highlightState.compareTwo &&
      highlightState.compareTarget === SURPRISE_TARGET
    ) {
      // compare-two with surprise as the B stripe — prefer the probe
      // tooltip but append the surprise number so hover gives both.
      const probeTip = t.probes
        ? formatScoreTooltip(t.probes)
        : t.score !== undefined && highlightState.target
          ? `${highlightState.target} ${t.score >= 0 ? "+" : ""}${t.score.toFixed(3)}`
          : "";
      const sup = surpriseTooltip(t);
      return probeTip ? `${probeTip}\n${sup}` : sup;
    }
    if (t.probes) return formatScoreTooltip(t.probes);
    const latest = latestLayerScores(t);
    if (latest) return formatScoreTooltip(latest);
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
  interface VisibleToken {
    tok: TokenScore;
    originalIdx: number;
  }

  function visibleResponseTokens(tokens: TokenScore[]): VisibleToken[] {
    let i = 0;
    while (i < tokens.length && !tokens[i].text.trim()) i++;
    return tokens.slice(i).map((tok, offset) => ({
      tok,
      originalIdx: i + offset,
    }));
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
    return visibleResponseTokens(turn.tokens)
      .map(({ tok }) => tok.text)
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
        <!-- Logit-pass: ``surprise`` tints tokens by ``-logprob /
             (1 - logprob)`` per Decision 4.  Sentinel value sits next to
             real probe names in the same picker so a single dropdown
             covers both axes. -->
        <option value={SURPRISE_TARGET}>surprise (logprob)</option>
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
          <!-- Allow surprise as the B-stripe target too — "probe X vs.
               surprise" is a useful axis ("does probe X light up at the
               surprising tokens?"). -->
          {#if highlightState.target !== SURPRISE_TARGET}
            <option value={SURPRISE_TARGET}>surprise (logprob)</option>
          {/if}
          {#each probeNames as name (name)}
            {#if name !== highlightState.target}
              <option value={name}>{name}</option>
            {/if}
          {/each}
        </select>
      </label>
    {/if}

    <!-- Conversation actions — clear / rewind / transcript / auto-regen
         sit inline at the end of the header so they're one click away. -->
    <div class="header-actions">
      <button type="button" class="hbtn" onclick={clearChat}>
        clear chat
      </button>
      <button type="button" class="hbtn" onclick={rewindChat}>
        rewind last turn
      </button>
      <button type="button" class="hbtn" onclick={openTranscript}>
        transcript…
      </button>
      <label class="ctl ctl-inline">
        <input
          type="checkbox"
          checked={autoRegenState.enabled}
          onchange={toggleAutoRegen}
        />
        <span class="ctl-label">auto-regen</span>
      </label>
      {#if autoRegenState.enabled}
        <select
          class="ctl-select"
          value={autoRegenState.mode}
          onchange={onAutoRegenModeChange}
          aria-label="Auto-regen mode"
        >
          {#each AUTO_REGEN_MODES as opt (opt.value)}
            <option value={opt.value}>{opt.label}</option>
          {/each}
        </select>
        {#if autoRegenState.mode === "custom"}
          <input
            type="text"
            class="ctl-input"
            value={autoRegenState.custom}
            oninput={(ev) =>
              setAutoRegenCustom(
                (ev.currentTarget as HTMLInputElement).value,
              )}
            placeholder="seed=42, temperature=1.5"
            aria-label="Custom auto-regen recipe"
          />
        {/if}
      {/if}
    </div>
  </header>

  <div
    class="log"
    class:ab={twoColumns}
    bind:this={logRef}
    onscroll={onScroll}
    role="log"
    aria-live="polite"
  >
    {#if twoColumns}
      <!-- Two-column split.  Right column is the *pinned* sibling's
           subtree path when pinning is on, the auto-regen output's
           path when auto-regen is on (auto-regen pins on done), or the
           legacy A/B shadow when only A/B is on. -->
      <div class="ab-grid">
        <div class="ab-col ab-primary">
          {#each chatLog.turns as turn, turnIdx (turnIdx)}
            {@render bubble(turn, turnIdx, false)}
          {/each}
        </div>
        <div class="ab-col ab-shadow">
          {#if pinnedActive}
            <header class="pin-header">
              <span class="pin-tag">pinned</span>
              <code class="pin-id">{pinnedComparison.nodeId?.slice(0, 12)}</code>
              <button
                type="button"
                class="pin-unpin"
                onclick={unpinComparison}
                title="Unpin"
              >unpin</button>
            </header>
            {#each pinnedPath as turn, idx (idx)}
              {@render bubble(turn, idx, true)}
            {/each}
          {:else}
            {#each chatLog.turns as turn, turnIdx (turnIdx)}
              {#if turn.role === "user" || turn.role === "system"}
                {@render bubble(turn, turnIdx, false)}
              {:else if turn.abPair}
                {@render bubble(turn.abPair, turnIdx, true)}
              {:else}
                <div class="msg assistant placeholder" aria-hidden="true">
                  <span class="role">assistant (alt)</span>
                  <span class="placeholder-text">— pending —</span>
                </div>
              {/if}
            {/each}
          {/if}
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
      <button
        type="button"
        class="regen"
        onclick={regenAction}
        disabled={!canRegen}
        title="Rewind and re-issue the last user message"
      >regen</button>
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
          {#each visibleResponseTokens(turn.tokens ?? []) as { tok, originalIdx } (originalIdx)}
            <span
              class="tok"
              class:tinted={highlightState.target !== null}
              style={styleString(tokenStyle(tok))}
              title={tooltipFor(tok)}
              onclick={(ev) => tokenClicked(turnIdx, originalIdx, ev, false)}
              onkeydown={(ev) => {
                if (ev.key === "Enter" || ev.key === " ") {
                  ev.preventDefault();
                  ev.stopPropagation();
                  openDrawer("token_drilldown", { turnIdx, tokenIdx: originalIdx });
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
  .ctl-inline input {
    accent-color: var(--accent-blue);
  }
  .ctl-label {
    color: var(--fg-muted);
    text-transform: lowercase;
    letter-spacing: 0;
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

  /* Conversation-actions strip — inline, pushed to the right edge of
   * the header.  Wraps onto a second row on narrow layouts. */
  .header-actions {
    display: flex;
    align-items: center;
    gap: 0.5em;
    flex-wrap: wrap;
    margin-left: auto;
  }
  .hbtn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--fg-dim);
    padding: 0.15em 0.55em;
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .hbtn:hover {
    background: var(--bg-elev);
    border-color: var(--accent-blue);
    color: var(--accent-blue);
  }
  .ctl-input {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.15em 0.45em;
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    min-width: 14em;
  }
  .ctl-input:focus {
    outline: none;
    border-color: var(--accent-blue);
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
  .pin-header {
    display: flex;
    align-items: center;
    gap: 0.4em;
    padding: 0.3em 0.5em;
    background: rgba(167, 139, 250, 0.10);
    border: 1px solid var(--accent-purple);
    border-radius: var(--radius);
    color: var(--accent-purple);
    font-size: var(--font-size-tiny);
    margin-bottom: 0.4em;
  }
  .pin-tag {
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .pin-id {
    color: var(--accent-yellow);
    flex: 1 1 auto;
  }
  .pin-unpin {
    background: transparent;
    color: var(--fg-dim);
    border: 1px solid var(--border);
    padding: 0.1em 0.4em;
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--font-size-tiny);
    cursor: pointer;
  }
  .pin-unpin:hover {
    color: var(--accent-red);
    border-color: var(--accent-red);
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
    letter-spacing: 0;
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
  .input-actions .regen {
    color: var(--accent-blue);
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
