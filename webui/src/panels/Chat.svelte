<script lang="ts">
  // Chat panel — v1.7 rewrite.  Replaces the v1.6 ChatPlaceholder/legacy
  // shape with the full feature set: thinking-collapsible per assistant
  // turn, per-token tinted spans driven by a top-bar highlight dropdown,
  // optional compare-two stripe overlay, click-token drilldown, send /
  // stop, and an A/B split-view container.
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
  import StatusFooter from "./StatusFooter.svelte";
  import PendingBubbles from "./PendingBubbles.svelte";
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
    inputRestore,
    pushInputHistory,
    navigateInputHistory,
    cancelInputPull,
    consumePulledSlot,
    resetChatToRoot,
    rewindSession,
    sendPrefill,
    sendCommit,
    loomRegenerateFromUser,
    enqueuePending,
    pendingActions,
    cancelPendingAction,
    predictedQueueEndOnUserNode,
    isPendingBusy,
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

  /** Auto-grow the textarea between 1 and 6 rows (≈ 132px at 13px line-h).
   *  With ``box-sizing: border-box`` set in CSS, ``el.scrollHeight``
   *  includes top/bottom padding — which is exactly what we want to write
   *  back into ``style.height``, so a one-line draft sits flush with no
   *  residual scrollbar.  The vertical scrollbar is suppressed unless the
   *  content actually overflows the 6-row cap. */
  function autosize(): void {
    const el = textareaRef;
    if (!el) return;
    el.style.height = "auto";
    const rowHeight = 22; // mono line-height fudge — matches font-size-base
    const maxH = rowHeight * 6;
    const next = Math.min(el.scrollHeight, maxH);
    el.style.height = `${next}px`;
    // Only show the scrollbar once we've actually hit the cap.  Without
    // this the browser's "always reserve a scrollbar gutter" heuristic
    // paints a 1-2px up/down nub on single-line input.
    el.style.overflowY = el.scrollHeight > maxH ? "auto" : "hidden";
  }

  $effect(() => {
    // Run autosize whenever ``input`` changes — bind:value + an effect is
    // simpler than wiring an oninput handler that has to cooperate with
    // bind:.
    void input;
    autosize();
  });

  // --- Role-aware input -------------------------------------------------
  // The selected loom node's role decides what the input box composes.
  // On an assistant / root node you write the next *user* message (the
  // normal chat flow).  On a *user* node the turn below it is the
  // assistant's — so the input composes the assistant reply instead:
  //   empty + send → generate a fresh assistant child (re-roll / fan)
  //   text  + send → answer-prefill — seed the reply with that text
  const activeNodeId = $derived(
    loomTree.rev > 0 ? (loomTree.active_node_id ?? null) : null,
  );
  const activeNode = $derived(
    activeNodeId ? (loomTree.nodes.get(activeNodeId) ?? null) : null,
  );
  const liveOnUserNode = $derived(activeNode?.role === "user");
  // Queue-aware role: a queued ``commit user`` lands a user node before
  // this submission gets to run, so the next message should already be
  // in prefill / commit-assistant mode.  Walks the queue tail-first;
  // falls back to the live active node when nothing in the queue
  // changes the role.  ``pendingActions.queue.length`` is touched so
  // the derived re-runs on queue mutations.
  const onUserNode = $derived.by(() => {
    void pendingActions.queue.length;
    const predicted = predictedQueueEndOnUserNode();
    return predicted === null ? liveOnUserNode : predicted;
  });

  // --- Commit modifier (Ctrl / Cmd / Option) ----------------------------
  // Any of Ctrl, Cmd (⌘), or Option (⌥) held flips the input into
  // "commit" mode: the typed text lands as the next turn but no
  // generation runs.  On an assistant/root node, that turn is a new user
  // node; on a user node, it's an authored assistant turn (the full
  // reply, not a prefilled seed).  Tracked at the window level so the
  // state survives textarea blur and we can swap the send-button caption
  // the moment the modifier comes down — without needing the user to
  // type anything first.
  let modHeld = $state(false);
  /** True whenever the modifier is held, regardless of input content —
   *  so the label flips and gives the user visual confirmation that the
   *  modifier registered.  The button's disabled gate (below) still
   *  refuses to submit an empty commit. */
  const commitMode = $derived(modHeld);
  /** Empty input in commit mode is a no-op (we can't commit nothing).
   *  Used by both the disabled gate and tryCommit's early return. */
  const canCommit = $derived(commitMode && input.trim() !== "");

  /** Four-state placeholder, same `<verb>…  <bind list>` shape across
   *  TUI and GUI.  Verb names the action that bare `⏎` (or `⌃⏎` in
   *  commit mode) will perform; the trailing key list is the minimum
   *  cheatsheet for the bindings that change behavior. */
  const inputPlaceholder = $derived(
    commitMode
      ? (onUserNode
          ? "commit assistant…  ⌃⏎ send · ⇧⏎ newline"
          : "commit user…  ⌃⏎ send · ⇧⏎ newline")
      : (onUserNode
          ? "prefill reply…  ⏎ send · ⌃⏎ commit · ⇧⏎ newline"
          : "message…  ⏎ send · ⌃⏎ commit · ⇧⏎ newline"),
  );
  /** Send-button caption tracks the role-aware action; any held commit
   *  modifier overrides both prefill and send with a "commit" register. */
  const sendLabel = $derived(
    commitMode
      ? (onUserNode ? "commit assistant" : "commit user")
      : (onUserNode ? (input.trim() ? "prefill" : "generate") : "send"),
  );

  /** Shared commit dispatch — used by both Ctrl/Cmd/Option+Enter and a
   *  modified-click on the send button.  Returns true when it claimed
   *  the action (including the empty-input no-op), so the caller knows
   *  not to fall through to the normal send/prefill path: the modifier
   *  explicitly means "don't generate," so an empty commit silently
   *  consumes rather than degrading to a regenerate. */
  function tryCommit(): boolean {
    const text = input.trim();
    // Forward the pulled slot (if any) so a re-edited queued commit
    // lands at its original position rather than appending to the
    // tail.  ``consumePulledSlot`` clears the pull state in one call.
    const replaceSlot = consumePulledSlot();
    if (!text) {
      // Empty commit on a pulled slot is the cancel gesture — drop
      // the slot from the queue.
      if (replaceSlot !== null) {
        cancelPendingAction(pendingActions.queue[replaceSlot]?.id ?? "");
      }
      return true;  // consumed
    }
    // When the queue holds an action that flips the active role (e.g. a
    // queued ``commit user`` puts us in predicted-prefill mode), the
    // live ``activeNodeId`` doesn't yet point at the parent the new
    // commit should hang under.  Pass the ``"active@drain"`` sentinel
    // so the pending action resolves the parent at drain time — by
    // which point the earlier queue items have landed the right node.
    const parent = isPendingBusy()
      ? ("active@drain" as const)
      : activeNodeId;
    if (onUserNode) {
      if (!parent) return true;
      pushInputHistory(text);
      input = "";
      void sendCommit("assistant", parent, text, { replaceSlot });
    } else {
      // Active node is root/assistant.  Pass it as the parent so the
      // server anchors the new user node under it (active-node fall-
      // through would do the same, but explicit avoids races with any
      // mid-flight active-node swap).
      pushInputHistory(text);
      input = "";
      void sendCommit("user", parent, text, { replaceSlot });
    }
    scrolledUp = false;
    queueScrollToBottom();
    queueMicrotask(autosize);
    return true;
  }

  function doSend(commit: boolean = false): void {
    // Modifier-held path: commit the text as the next turn without
    // running a decode.  tryCommit always consumes the action when
    // commit is true — empty input no-ops silently.
    if (commit && tryCommit()) return;
    // Role-aware branch: on a user node the input seeds the assistant
    // reply rather than appending a new user turn.
    if (onUserNode && (activeNodeId || isPendingBusy())) {
      // Keep the raw value — a trailing space in a prefill is meaningful
      // (it decides whether the continuation starts a fresh word).
      const raw = input;
      const trimmed = raw.trim();
      const replaceSlot = consumePulledSlot();
      input = "";
      // Deferred resolution when busy (see tryCommit for the rationale).
      const target = isPendingBusy()
        ? ("active@drain" as const)
        : activeNodeId!;
      if (trimmed) {
        pushInputHistory(trimmed);
        void sendPrefill(target, raw, { replaceSlot });
      } else if (replaceSlot !== null) {
        // Empty prefill on a pulled slot cancels the queued item.
        cancelPendingAction(pendingActions.queue[replaceSlot]?.id ?? "");
      } else if (activeNodeId) {
        // Empty + not pulled + live user node → re-roll the assistant.
        // We can't re-roll a not-yet-existing queued user node, so this
        // branch is gated to live ids only.
        void loomRegenerateFromUser(activeNodeId);
      }
      scrolledUp = false;
      queueScrollToBottom();
      queueMicrotask(autosize);
      return;
    }
    const text = input.trim();
    const replaceSlot = consumePulledSlot();
    if (!text) {
      // Empty send on a pulled slot cancels the queued item; otherwise
      // a no-op.
      if (replaceSlot !== null) {
        cancelPendingAction(pendingActions.queue[replaceSlot]?.id ?? "");
      }
      return;
    }
    // Push to ↑/↓ recall before clearing — covers both chat messages
    // and slash commands (every line typed in here is recallable).
    pushInputHistory(text);
    input = "";
    // Defer the actual send so the textarea clears before the WS round-
    // trip — feels less like the UI froze.
    void sendGenerate(text, { replaceSlot });
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
      // Shift-Enter is a newline; Ctrl/Cmd/Option-Enter is the commit
      // modifier (no generation); bare Enter is the normal send/prefill
      // path.  Reading the modifier flags off the event directly is
      // more reliable than ``modHeld`` (which lags on focus-blur edge
      // cases) — at the moment of Enter the event carries the truth.
      if (ev.shiftKey) return;
      ev.preventDefault();
      doSend(ev.ctrlKey || ev.metaKey || ev.altKey);
      return;
    }
    if (ev.key === "Escape") {
      // Esc is context-sensitive (mirrors the TUI):
      //   1. Gen in flight → stop the gen.  Queue keeps its items.
      //   2. No gen, pull in flight → cancel the pull (restore the
      //      stash, leave the queued slot untouched).
      //   3. Otherwise → fall through to default Escape behavior.
      if (genStatus.active) {
        ev.preventDefault();
        sendStop();
        return;
      }
      if (inputHistory.pulledSlot !== null) {
        ev.preventDefault();
        const stash = cancelInputPull();
        if (stash !== null) {
          input = stash;
          queueMicrotask(autosize);
        }
        return;
      }
    }
    if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
      const ta = textareaRef;
      if (!ta) return;
      const goingUp = ev.key === "ArrowUp";
      if (goingUp ? !shouldRecallUp(ta) : !shouldRecallDown(ta)) return;
      // ↓ at the live slot (no recall in flight, no pending pull) is
      // a no-op — leave the keystroke for the textarea so it can move
      // within an empty last line or trigger the browser's native
      // end-of-input nudge.  Checking only ``index`` misses the
      // pulled-pending case (``pulledSlot`` is what tracks that),
      // which would otherwise strand the user inside an edited
      // pending item with no ↓ exit.
      if (
        !goingUp
        && inputHistory.index === null
        && inputHistory.pulledSlot === null
      ) return;
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
  // clear / regen / transcript / auto-regen used to live on the Topbar;
  // they act on the conversation, so they belong here.  The mutating
  // ones route through ``enqueuePending`` so clicking them mid-gen
  // queues rather than racing the WS.  (regen still rewinds internally —
  // the standalone "rewind" button was vestigial and was removed.)

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
    if (genStatus.active || pendingActions.queue.length > 0) {
      enqueuePending({
        label: "/clear",
        text: null,
        apply: () => void resetChatToRoot(),
        awaitsGen: false,
        rebuild: null,
        // /clear navigates to the synthetic root (system role) — not a
        // user node, so the next submission goes through "message" mode
        // and lands a fresh user branch off root.
        endsOnUserNode: false,
      });
    } else {
      void resetChatToRoot();
    }
  }

  function regenAction(): void {
    // Capture the input now — a queued action fires later, by which point
    // the local log may have shifted.
    const input = lastUserInput();
    if (input === null) return;
    if (genStatus.active || pendingActions.queue.length > 0) {
      enqueuePending({
        label: "regen",
        text: null,
        apply: () => void regen(input),
        awaitsGen: true,
        rebuild: null,
        // Regen rewinds to the user node then sends — lands a new
        // assistant sibling, so the post-drain active is assistant.
        endsOnUserNode: false,
      });
    } else {
      void regen(input);
    }
  }

  const canRegen = $derived(lastUserInput() !== null);

  function openTranscript(): void {
    openDrawer("transcript");
  }

  // Save / load act on the whole conversation tree; they live here at
  // the chat's edge rather than buried in a rail menu.  Regenerate-N and
  // fan-out used to sit here too — both were redundant (the loom right-
  // click menu carries "regenerate…" and "fan out…", and the experiment
  // lab is one click away in the analysis menu) so they were removed.
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

  // Cross-component input restore: when a queue drain pops the slot
  // the user was currently editing, ``drainNextPendingAction`` parks
  // the stash on ``inputRestore`` and bumps ``rev``.  This $effect
  // copies it back into the textarea on the next tick.
  let _restoreRev = $state(0);
  $effect(() => {
    if (inputRestore.rev !== _restoreRev) {
      _restoreRev = inputRestore.rev;
      input = inputRestore.text;
      queueMicrotask(autosize);
    }
  });

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

    // Track any commit modifier at the window level so the send-button
    // label flips the moment the user presses it, not only when they
    // hit Enter.  We read all three flags off the event so the modifier
    // works across platforms and key layouts:
    //   Ctrl → ``ctrlKey``  — Linux / Windows / Mac Ctrl
    //   Cmd  → ``metaKey``  — Mac (⌘)
    //   Option / Alt → ``altKey``  — Mac (⌥) / non-Mac Alt
    // Browsers report all three correctly for both modifier-only
    // keydown and the keydown of a non-modifier key while the modifier
    // is held — and they go false on keyup of the modifier.
    const setHeld = (ev: KeyboardEvent) => {
      modHeld = ev.ctrlKey || ev.metaKey || ev.altKey;
    };
    const clearHeld = () => {
      modHeld = false;
    };
    window.addEventListener("keydown", setHeld);
    window.addEventListener("keyup", setHeld);
    // ``blur`` covers tab-out / window-switch where the keyup never
    // fires — without it the label sticks in "commit" mode after the
    // user Cmd-Tabs away mid-modifier.
    window.addEventListener("blur", clearHeld);
    return () => {
      window.removeEventListener("keydown", setHeld);
      window.removeEventListener("keyup", setHeld);
      window.removeEventListener("blur", clearHeld);
    };
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

    <!-- Conversation actions — clear / save / load / transcript / auto-
         regen sit inline at the end of the header so they're one click
         away. -->
    <div class="header-actions">
      <button type="button" class="hbtn" onclick={clearChat}>
        clear chat
      </button>
      <button
        type="button"
        class="hbtn"
        onclick={() => openDrawer("save_conversation")}
        title="Save this conversation tree to disk"
      >
        save conversation…
      </button>
      <button
        type="button"
        class="hbtn"
        onclick={() => openDrawer("load_conversation")}
        title="Load a saved conversation tree"
      >
        load conversation…
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

  <StatusFooter />

  <PendingBubbles />

  <form class="input-row" onsubmit={(ev) => { ev.preventDefault(); doSend(modHeld); }}>
    <textarea
      class="input"
      class:prefill-mode={onUserNode}
      bind:this={textareaRef}
      bind:value={input}
      onkeydown={onKeydown}
      placeholder={inputPlaceholder}
      rows="1"
      aria-label={onUserNode ? "Assistant prefill input" : "Chat input"}
    ></textarea>
    <div class="input-actions">
      <button
        type="submit"
        class="send"
        disabled={!input.trim() && (commitMode || !onUserNode)}
        title={onUserNode
          ? "⏎ prefill reply (empty = generate fresh) · ⌃-click commit assistant · ⇧⏎ newline"
          : "⏎ send · ⌃-click commit user · ⇧⏎ newline"}
      >{sendLabel}</button>
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
    gap: var(--space-3);
    font-family: var(--font-mono);
    font-size: var(--text);
    color: var(--fg);
  }

  .chat-header {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    flex-wrap: wrap;
    padding-bottom: var(--space-2);
    border-bottom: 1px solid var(--border);
    color: var(--fg-dim);
    font-size: var(--text-sm);
  }
  .ctl {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .ctl-inline {
    cursor: pointer;
    user-select: none;
  }
  .ctl-inline input {
    accent-color: var(--accent);
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
    padding: var(--space-1) var(--space-3);
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
    gap: var(--space-3);
    flex-wrap: wrap;
    margin-left: auto;
  }
  .hbtn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--fg-dim);
    padding: var(--space-1) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .hbtn:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--accent);
    color: var(--accent);
  }
  .hbtn:disabled {
    color: var(--fg-muted);
    border-color: var(--border);
    cursor: not-allowed;
  }
  .ctl-input {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    min-width: 14em;
  }
  .ctl-input:focus {
    outline: none;
    border-color: var(--accent);
  }

  .log {
    flex: 1 1 auto;
    overflow-y: auto;
    overflow-x: hidden;
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
    padding-right: var(--space-2);
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
    gap: var(--space-4);
  }
  .ab-col {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-width: 0;
  }
  .pin-header {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    background: rgba(167, 139, 250, 0.10);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--accent-purple);
    font-size: var(--text-xs);
    margin-bottom: var(--space-2);
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
    padding: var(--space-1) var(--space-2);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
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
    padding: var(--space-1) var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
    word-break: break-word;
  }
  .msg .role {
    color: var(--fg-muted);
    font-size: var(--text-xs);
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
  .msg.placeholder {
    color: var(--fg-muted);
    font-style: italic;
    opacity: 0.6;
  }
  .placeholder-text {
    font-size: var(--text-sm);
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
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    margin-bottom: var(--space-1);
  }
  .thinking-toggle {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    font: inherit;
    font-family: var(--font-mono);
    padding: var(--space-1) 0;
    cursor: pointer;
    text-align: left;
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
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
    /* 1.6em left pad is a hanging indent tuned to the caret width — kept raw. */
    padding: var(--space-1) 0 var(--space-2) 1.6em;
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
    border-radius: var(--radius);
  }
  .tok:hover {
    outline: 1px solid var(--fg-muted);
  }

  .input-row {
    display: flex;
    gap: var(--space-3);
    align-items: flex-end;
    /* No border-top — the status footer directly above already caps
     * the input region with its own hairline. */
  }
  .input {
    flex: 1 1 auto;
    background: var(--bg-alt);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: var(--space-2) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    resize: none;
    /* border-box lets autosize() write ``scrollHeight`` straight into
     * ``style.height`` without a padding/border double-count — without
     * this the one-line draft height was off by ~6px and the textarea's
     * vertical scrollbar leaked through as a tiny up/down nub. */
    box-sizing: border-box;
    overflow-y: hidden;
    min-height: 2.4em;
    max-height: 132px;
    line-height: 1.45;
  }
  .input:focus {
    outline: none;
    border-color: var(--accent);
  }
  /* Prefill mode: the active loom node is a user turn, so this box
     composes the assistant reply.  Tint the border to signal the role
     shift before the user starts typing. */
  .input.prefill-mode {
    border-color: var(--accent);
    background: rgba(167, 139, 250, 0.06);
  }
  .input.prefill-mode:focus {
    border-color: var(--accent);
  }
  .input-actions {
    display: flex;
    gap: var(--space-2);
    align-items: center;
  }
  .input-actions button {
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-2) var(--space-5);
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
    border-color: var(--border);
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
</style>
