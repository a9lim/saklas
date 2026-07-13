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
  // Turn surface follows the CAST MODEL (docs/plans/dynamic-roles.md):
  // every speaker gets one neutral glass card — roles aren't a "space",
  // so they carry no hue — with identity in the role chip (glyph letter +
  // label, arbitrary strings first-class) and provenance as the ppl badge
  // generated turns carry. System turns render as stage directions (a
  // note about the scene, not a speaker). The ``speaking as`` chips in
  // the composer are the promoted SamplingStrip role boxes — same client
  // state, same wire block.
  //
  // Mirrors saklas/tui/chat_panel.py for the token rhythm: leading
  // whitespace strip after </think>, plain text fall-through when no
  // probe is selected. (The TUI still draws role-coloured borders — its
  // cast-model parity pass is deferred.)

  import { onMount, untrack } from "svelte";
  import { SvelteMap } from "svelte/reactivity";
  import StatusFooter from "./StatusFooter.svelte";
  import PendingBubbles from "./PendingBubbles.svelte";
  import RawBuffer from "./RawBuffer.svelte";
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
    rewindSession,
    sendPrefill,
    sendCommit,
    enqueuePending,
    pendingActions,
    cancelPendingAction,
    predictedQueueEndOnUserNode,
    isPendingBusy,
    toggleAutoRegen,
    setAutoRegenMode,
    setAutoRegenCustom,
    effectiveRawMode,
    genUiMode,
    setGenUiMode,
    roleDisplayLabel,
    roleGlyphLetter,
    samplingState,
    sessionState,
    castState,
    highlightScale,
  } from "../lib/stores.svelte";
  import type { AutoRegenMode } from "../lib/stores.svelte";
  import { togglePalette } from "../lib/stores/palette.svelte";
  import type { ChatTurn, TokenScore } from "../lib/types";
  import {
    scoreToRgb,
    twoStripeStyle,
    twoBlendStyle,
    formatScoreTooltip,
    surpriseScore,
    SURPRISE_TARGET,
    probeScoreForTarget,
  } from "../lib/tokens";
  import Select from "../lib/Select.svelte";
  import Checkbox from "../lib/Checkbox.svelte";

  // --------------------------------------------------------------- input --

  let input = $state("");
  let textareaRef: HTMLTextAreaElement | null = $state(null);

  // ---------------------------------------------------------- cast row --
  //
  // The ``speaking as`` / ``reply as`` chips — the SamplingStrip role
  // boxes promoted into the composer (the visible step toward the cast
  // model).  Same client state (``samplingState.user_role`` /
  // ``assistant_role``), same wire block, same support gates: base /
  // completion models and label-free families (Mistral / talkie) can't
  // relabel turns, raw mode has no roles.
  const CAST_SLUG_RE = /^[a-z0-9._-]+$/;
  const castReady = $derived(sessionState.info !== null);
  const castUserSupported = $derived(
    sessionState.info?.is_base_model === false &&
      !effectiveRawMode() &&
      sessionState.info?.user_role_supported === true,
  );
  const castAsstSupported = $derived(
    sessionState.info?.is_base_model === false &&
      !effectiveRawMode() &&
      sessionState.info?.role_substitution_supported === true,
  );
  const castUserValid = $derived(
    samplingState.user_role.trim() === "" ||
      CAST_SLUG_RE.test(samplingState.user_role.trim()),
  );
  const castAsstValid = $derived(
    samplingState.assistant_role.trim() === "" ||
      CAST_SLUG_RE.test(samplingState.assistant_role.trim()),
  );

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
  //   text  + send → answer-prefill — seed the reply with that text
  //   empty        → no-op (send button is grayed, mirroring an
  //                  assistant node).  Re-rolling / fanning a user node
  //                  lives on the loom sidebar's regenerate / fan-out
  //                  menu and the dedicated regen button, not here.
  // Raw (flat completion) mode — base models, or an explicit override.
  // In raw mode the role-aware commit derivations short-circuit: there
  // are no roles, so the input box never enters prefill / commit mode.
  const rawMode = $derived.by(() => {
    void genUiMode.mode;
    return effectiveRawMode();
  });

  const activeNodeId = $derived(
    loomTree.loaded ? (loomTree.active_node_id ?? null) : null,
  );
  const activeNode = $derived(
    activeNodeId ? (loomTree.nodes.get(activeNodeId) ?? null) : null,
  );
  const liveOnUserNode = $derived(!rawMode && activeNode?.role === "user");
  // Queue-aware role: a queued ``commit user`` lands a user node before
  // this submission gets to run, so the next message should already be
  // in prefill / commit-assistant mode.  Walks the queue tail-first;
  // falls back to the live active node when nothing in the queue
  // changes the role.  ``pendingActions.queue.length`` is touched so
  // the derived re-runs on queue mutations.
  const onUserNode = $derived.by(() => {
    if (rawMode) return false;
    void pendingActions.queue.length;
    const predicted = predictedQueueEndOnUserNode();
    return predicted === null ? liveOnUserNode : predicted;
  });

  // ---------------------------------------------------------- cast seats --
  //
  // Scene mode (the validated stitcher grammar) frees the seat the model
  // generates into: the seat toggle picks it, and an empty send becomes
  // an explicit *continue* (``input: null`` — no committed turn, the
  // model speaks next from the leaf: the a/a and u-continue shapes).
  // Continue is deliberately NOT offered on a user-node leaf with the
  // assistant seat — that shape is regen, which lives on the loom
  // sidebar, not the input bar.
  const sceneMode = $derived(sessionState.info?.scene_mode ?? false);
  let genSeat = $state<"assistant" | "user">("assistant");
  $effect(() => {
    // Fallback families have no seat freedom — snap the toggle back.
    if (!sceneMode && genSeat !== "assistant") genSeat = "assistant";
  });
  const userSeatActive = $derived(sceneMode && genSeat === "user");
  const canContinue = $derived(
    sceneMode &&
      !genStatus.active &&
      input.trim() === "" &&
      loomTree.loaded &&
      (userSeatActive || !onUserNode),
  );

  // Committed-thinking input: a block the next *commit* (⌃⏎) carries,
  // rendered through the family think delimiters.  Strip families keep
  // it for one turn only — the warning under the box says so before
  // submit (a9 convention 3).
  const thinkingInputSupported = $derived(
    sessionState.info?.thinking_input_supported ?? false,
  );
  const stripsHistoryThinking = $derived(
    sessionState.info?.strips_history_thinking ?? false,
  );
  let thinkingOpen = $state(false);
  let thinkingDraft = $state("");

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

  /** Keep the composer prompt to the active action; shortcuts live in help. */
  const inputPlaceholder = $derived(
    commitMode
      ? (onUserNode
          ? "commit assistant…"
          : "commit user…")
      : userSeatActive
        ? "user seat…"
        : (onUserNode
            ? "prefill…"
            : "message…"),
  );
  /** Send-button caption tracks the role-aware action; any held commit
   *  modifier overrides both prefill and send with a "commit" register.
   *  In scene mode an empty draft reads "continue" — the explicit
   *  ``input: null`` no-committed-turn generation. */
  const sendLabel = $derived(
    commitMode
      ? (onUserNode ? "commit assistant" : "commit user")
      : userSeatActive
        ? (input.trim() ? "send ⇢ user" : "continue ⇢ user")
        : canContinue && !input.trim()
          ? "continue"
          : (onUserNode ? "prefill" : "send"),
  );

  /** Shared commit dispatch — used by both Ctrl/Cmd/Option+Enter and a
   *  modified-click on the send button.  Returns true when it claimed
   *  the action (including the empty-input no-op), so the caller knows
   *  not to fall through to the normal send/prefill path: the modifier
   *  explicitly means "don't generate," so an empty commit silently
   *  consumes rather than falling through. */
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
    // A drafted thinking block rides this commit (the only surface that
    // carries one) and is consumed by it — one block, one turn.
    const thinking = thinkingDraft.trim() !== "" ? thinkingDraft : null;
    if (thinking !== null) {
      thinkingDraft = "";
      thinkingOpen = false;
    }
    if (onUserNode) {
      if (!parent) return true;
      pushInputHistory(text);
      input = "";
      void sendCommit("assistant", parent, text, { replaceSlot, thinking });
    } else {
      // Active node is root/assistant.  Pass it as the parent so the
      // server anchors the new user node under it (active-node fall-
      // through would do the same, but explicit avoids races with any
      // mid-flight active-node swap).
      pushInputHistory(text);
      input = "";
      void sendCommit("user", parent, text, { replaceSlot, thinking });
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
    // User-seat path (scene mode): the model speaks the user seat.
    // Text commits as a user turn first (the engine's input-string
    // contract), so this covers u/u; empty is the explicit continue.
    // Bypasses prefill mode entirely — prefill seeds an assistant
    // reply, which isn't what this seat means.
    if (userSeatActive) {
      const text = input.trim();
      const replaceSlot = consumePulledSlot();
      if (!text) {
        if (replaceSlot !== null) {
          cancelPendingAction(pendingActions.queue[replaceSlot]?.id ?? "");
          return;
        }
        if (!canContinue) return;
        void sendGenerate(null, { generate_seat: "user" });
      } else {
        pushInputHistory(text);
        input = "";
        void sendGenerate(text, { replaceSlot, generate_seat: "user" });
      }
      scrolledUp = false;
      queueScrollToBottom();
      queueMicrotask(autosize);
      return;
    }
    // Empty-draft continue (scene mode, assistant seat): generate with
    // no committed turn from the current leaf — the a/a shape.  Only
    // offered off user-node leaves (that shape is regen — sidebar).
    if (!input.trim() && canContinue && !onUserNode) {
      const replaceSlot = consumePulledSlot();
      if (replaceSlot !== null) {
        cancelPendingAction(pendingActions.queue[replaceSlot]?.id ?? "");
        return;
      }
      void sendGenerate(null, {});
      scrolledUp = false;
      queueScrollToBottom();
      return;
    }
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
      }
      // Empty + not pulled → no-op.  The send button is grayed in this
      // state (mirroring an assistant node); re-rolling the assistant
      // for a selected user node lives on the loom sidebar's regenerate /
      // fan-out menu, so the input bar no longer doubles as a regen.
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

  function onHighlightChange(value: string): void {
    setHighlightTarget(value === "" ? null : value);
  }

  function onCompareChange(value: string): void {
    setCompareTarget(value === "" ? null : value);
  }

  function onCompareToggle(): void {
    toggleCompareTwo();
  }

  /** Highlight options for one probe.  A rank-1 flat probe (a 2-node concept
   *  axis) and every curved probe stay a single bare-name option — the bare
   *  channel is the pole coordinate / subspace fraction.  A multi-axis flat
   *  probe (the ``personas`` fan, a flat ``emotions``) fans out into one
   *  option per coordinate so a token can be tinted by each PC; axis 0 keeps
   *  the bare-name value (the channel that survives reload) while axis ``i``
   *  uses the ``name[i]`` form that lines up with the ``@when:name[i]`` gate. */
  function axisOptionsFor(name: string): { value: string; label: string }[] {
    const info = probeRack.entries.get(name)?.info;
    const dim = info?.intrinsic_dim ?? 0;
    const flat = info?.is_affine ?? true;
    // Labels strip the namespace prefix (``default/emotions`` → ``emotions``)
    // to match the probe cards; the option value keeps the full registered
    // name so lookups stay unambiguous.
    const display = name.split("/").pop() ?? name;
    if (flat && dim > 1) {
      return Array.from({ length: dim }, (_, i) => ({
        value: i === 0 ? name : `${name}[${i}]`,
        label: `${display}[${i}]`,
      }));
    }
    return [{ value: name, label: display }];
  }

  /** Highlight-target picker options: "(off)" + surprise sentinel + live
   *  probe names, fanned out per coordinate axis for multi-axis probes. */
  const highlightOptions = $derived.by<{ value: string; label: string }[]>(
    () => {
      const opts: { value: string; label: string }[] = [
        { value: "", label: "(off)" },
        { value: SURPRISE_TARGET, label: "surprise (logprob)" },
      ];
      for (const name of probeNames) opts.push(...axisOptionsFor(name));
      return opts;
    },
  );

  /** Compare-target picker — same shape but filtered so the A and B targets
   *  don't pick the same axis.  Distinct axes of one probe (PC0 vs PC1) are
   *  allowed — that's a useful two-stripe compare. */
  const compareOptions = $derived.by<{ value: string; label: string }[]>(() => {
    const opts: { value: string; label: string }[] = [
      { value: "", label: "(off)" },
    ];
    if (highlightState.target !== SURPRISE_TARGET) {
      opts.push({ value: SURPRISE_TARGET, label: "surprise (logprob)" });
    }
    for (const name of probeNames) {
      for (const opt of axisOptionsFor(name)) {
        if (opt.value !== highlightState.target) opts.push(opt);
      }
    }
    return opts;
  });

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
  function onAutoRegenModeChange(v: AutoRegenMode): void {
    setAutoRegenMode(v);
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
    // Probe / per-axis lookup: the live per-PC coords first, then the
    // axis-0 ``probes`` row (the channel ``done`` + reload restore).
    const direct = probeScoreForTarget(t, target);
    if (direct !== undefined) return direct;
    const latest = latestLayerScores(t);
    if (latest && target in latest) return latest[target];
    return t.score;
  }

  /** Build the inline-style object for one token's background.  Compare-
   * two needs both probes set; if only one of the two is configured we
   * gracefully fall back to single-probe rendering.  ``SURPRISE_TARGET``
   * works in either slot — it reads in the logit-space blue ramp
   * (distinct from any probe's signed green/red), per-slot. */
  function hueFor(target: string | null): "signed" | "surprise" {
    return target === SURPRISE_TARGET ? "surprise" : "signed";
  }

  function tokenStyle(
    t: TokenScore,
  ): { backgroundColor?: string; backgroundImage?: string } {
    const a = highlightState.target;
    if (!a) return {};
    const aScore = pickScore(t, a);
    const scaleA = highlightScale(a);
    if (highlightState.compareTwo && highlightState.compareTarget) {
      const b = highlightState.compareTarget;
      const bScore = pickScore(t, b);
      const scaleB = highlightScale(b);
      return highlightState.smoothBlend
        ? twoBlendStyle(aScore, bScore, scaleA, scaleB, hueFor(a), hueFor(b))
        : twoStripeStyle(aScore, bScore, scaleA, scaleB, hueFor(a), hueFor(b));
    }
    const bg = scoreToRgb(aScore, scaleA, hueFor(a));
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
    // Look up the chosen token's rank by its current wire identity.
    if (t.tokenId == null) return lp;
    let rank: number | null = null;
    for (let i = 0; i < alts.length; i++) {
      const a = alts[i];
      if (a.id === t.tokenId) {
        rank = i + 1;
        break;
      }
    }
    return rank !== null
      ? `${lp}, rank ${rank} of ${alts.length}`
      : `${lp}, chosen not in top-${alts.length}`;
  }

  /** A dedicated tooltip line for an axis highlight target (``personas[3]``)
   *  whose value isn't already in the bare-name ``probes`` row.  Returns null
   *  for axis 0 / a plain probe (already shown) or when no value is known. */
  function axisTooltipLine(t: TokenScore, target: string | null): string | null {
    if (!target || target === SURPRISE_TARGET) return null;
    if (t.probes && target in t.probes) return null;
    const v = probeScoreForTarget(t, target);
    if (v === undefined) return null;
    return `${target} ${v >= 0 ? "+" : ""}${v.toFixed(3)}`;
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
    if (t.probes) {
      // Lead with the selected axis target(s) so a per-PC tint reports its
      // own value, then the full axis-0 probe row underneath.
      const extra: string[] = [];
      const la = axisTooltipLine(t, highlightState.target);
      if (la) extra.push(la);
      if (highlightState.compareTwo) {
        const lb = axisTooltipLine(t, highlightState.compareTarget);
        if (lb) extra.push(lb);
      }
      const base = formatScoreTooltip(t.probes);
      return extra.length ? `${extra.join("\n")}\n${base}` : base;
    }
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
      <!-- Logit-pass: ``surprise`` tints tokens by ``-logprob /
           (1 - logprob)`` per Decision 4.  Sentinel value sits next to
           real probe names in the same picker so a single dropdown
           covers both axes. -->
      <span class="ctl-select">
        <Select
          value={highlightState.target ?? ""}
          options={highlightOptions}
          onchange={onHighlightChange}
          ariaLabel="Highlight probe"
        />
      </span>
    </label>

    <span class="ctl ctl-inline">
      <Checkbox
        checked={highlightState.compareTwo}
        onchange={onCompareToggle}
        ariaLabel="compare-two"
      />
      <span class="ctl-label">compare</span>
    </span>

    {#if highlightState.compareTwo}
      <label class="ctl">
        <span class="ctl-label">vs.</span>
        <!-- Allow surprise as the B-stripe target too — "probe X vs.
             surprise" is a useful axis ("does probe X light up at the
             surprising tokens?"). -->
        <span class="ctl-select">
          <Select
            value={highlightState.compareTarget ?? ""}
            options={compareOptions}
            onchange={onCompareChange}
            disabled={!highlightState.compareTwo}
            ariaLabel="Compare probe"
          />
        </span>
      </label>
    {/if}

    <!-- Render-mode badge — raw/chat toggle.  The mode is seeded from
         the model (base → raw, chat → chat) the first time it is seen,
         then it's a plain two-state toggle; the same control also sits
         in the advanced sampling drawer. -->
    <button
      type="button"
      class="mode-badge"
      class:raw={rawMode}
      onclick={() => setGenUiMode(rawMode ? "chat" : "raw")}
      title={`render: ${genUiMode.mode}`}
    >
      {genUiMode.mode}
    </button>

    <!-- Conversation actions — transcript + auto-regen.  Clear / save /
         load moved up to the threads-column header (they act on the
         whole tree, not on the active chat path). -->
    <div class="header-actions">
      <!-- The workspace rail is gone — the palette is the tool launcher,
           and this chip is its one persistent visible hint. -->
      <button
        type="button"
        class="hbtn kbd-hint"
        onclick={togglePalette}
        title="tools"
      >
        ⌘K
      </button>
      <button type="button" class="hbtn" onclick={openTranscript}>
        transcript
      </button>
      <span class="ctl ctl-inline">
        <Checkbox
          checked={autoRegenState.enabled}
          onchange={toggleAutoRegen}
          ariaLabel="auto-regen"
        />
        <span class="ctl-label">auto</span>
      </span>
      {#if autoRegenState.enabled}
        <span class="ctl-select">
          <Select
            value={autoRegenState.mode}
            options={AUTO_REGEN_MODES}
            onchange={onAutoRegenModeChange}
            ariaLabel="Auto-regen mode"
          />
        </span>
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

  {#if rawMode}
    <RawBuffer />
  {:else}
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
                <div class="msg placeholder" aria-hidden="true">
                  <div class="who">
                    <span class="role-chip">
                      <b>{roleGlyphLetter("assistant")}</b>
                      {roleDisplayLabel("assistant")}
                    </span>
                    <span class="who-meta">(alt)</span>
                  </div>
                  <span class="placeholder-text">pending…</span>
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

  <div class="cast-row" aria-label="Speaking roles">
    <label
      class="cast"
      title={castUserSupported
        ? "your role"
        : "unavailable"}
    >
      <span class="cast-label">you</span>
      <input
        class="cast-input"
        class:invalid={!castUserValid}
        bind:value={samplingState.user_role}
        disabled={!castReady || !castUserSupported}
        placeholder="user"
        spellcheck="false"
        aria-label="user role label"
      />
    </label>
    <label
      class="cast"
      title={castAsstSupported
        ? "model role"
        : "unavailable"}
    >
      <span class="cast-label">model</span>
      <input
        class="cast-input"
        class:invalid={!castAsstValid}
        bind:value={samplingState.assistant_role}
        disabled={!castReady || !castAsstSupported}
        placeholder={roleDisplayLabel("assistant")}
        spellcheck="false"
        aria-label="assistant role label"
        list="cast-roster-labels"
      />
    </label>
    {#if sceneMode}
      <div
        class="seat-toggle"
        role="radiogroup"
        aria-label="generation seat"
        title="generation seat"
      >
        <span class="cast-label">seat</span>
        <button
          type="button"
          class="seat"
          class:active={genSeat === "assistant"}
          aria-pressed={genSeat === "assistant"}
          onclick={() => (genSeat = "assistant")}
        >assistant</button>
        <button
          type="button"
          class="seat"
          class:active={genSeat === "user"}
          aria-pressed={genSeat === "user"}
          onclick={() => (genSeat = "user")}
        >user</button>
      </div>
    {/if}
    <button
      type="button"
      class="cast-manage"
      title="cast"
      onclick={() => openDrawer("cast")}
    >cast…</button>
    <datalist id="cast-roster-labels">
      {#each Object.keys(castState.roster) as slug (slug)}
        <option value={slug}></option>
      {/each}
    </datalist>
  </div>

  {#if thinkingInputSupported}
    <div class="thinking-row">
      <button
        type="button"
        class="thinking-toggle"
        class:open={thinkingOpen}
        class:drafted={thinkingDraft.trim() !== ""}
        onclick={() => (thinkingOpen = !thinkingOpen)}
        title="next commit"
      >{thinkingOpen ? "− thinking" : "+ thinking"}</button>
      {#if thinkingOpen}
        <div class="thinking-box">
          <textarea
            class="thinking-input"
            bind:value={thinkingDraft}
            placeholder="thinking…"
            rows="2"
            spellcheck="false"
            aria-label="committed thinking block"
          ></textarea>
          {#if stripsHistoryThinking}
            <p class="thinking-warn" role="note">one turn only</p>
          {/if}
        </div>
      {/if}
    </div>
  {/if}

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
        disabled={!input.trim() && !canContinue}
        title={userSeatActive
          ? "⏎ user seat"
          : onUserNode
            ? "⏎ prefill"
            : canContinue && !input.trim()
              ? "⏎ continue"
              : "⏎ send"}
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
        title="regenerate"
      >regen</button>
    </div>
  </form>
  {/if}
</div>

{#snippet bubble(turn: ChatTurn, turnIdx: number, isShadow: boolean)}
  {#if turn.role === "system"}
    <!-- Stage direction — a note about the scene, not a speaker. -->
    <div
      class="stage"
      class:shadow={isShadow}
      title="system prompt"
    >{turn.text}</div>
  {:else}
  <div
    class="msg"
    class:shadow={isShadow}
  >
    <div class="who">
      <span class="role-chip" title="{turn.role}{turn.roleLabel ? ` as ${turn.roleLabel}` : ''}">
        <b>{roleGlyphLetter(turn.role, turn.roleLabel)}</b>
        {roleDisplayLabel(turn.role, turn.roleLabel)}
      </span>
      {#if isShadow && !pinnedActive}<span class="who-meta">(unsteered)</span>{/if}
      {#if turn.role === "assistant" && turn.meanLogprob != null && Number.isFinite(turn.meanLogprob)}
        <span
          class="prov"
          title="sequence perplexity"
        >seq ppl {Math.exp(-turn.meanLogprob).toFixed(1)}</span>
      {/if}
    </div>

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
  {/if}
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
  .ctl-label {
    color: var(--fg-muted);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  /* Layout host for the themed Select — Select owns its own theme. */
  .ctl-select {
    display: inline-flex;
    min-width: 9em;
  }

  /* Render-mode badge — compact raw/chat pill.  Sits between the
   * highlight controls and the conversation actions. */
  .mode-badge {
    background: var(--accent-subtle);
    color: var(--accent);
    border: 0;
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    text-transform: lowercase;
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .mode-badge:hover {
    background: var(--accent-glow);
  }
  .mode-badge.raw {
    background: rgba(167, 139, 250, 0.12);
    color: var(--accent-purple);
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
  /* Borderless workhorse — glass fill is the shape, hover lifts it. */
  .hbtn {
    background: var(--glass);
    border: 0;
    border-radius: var(--radius);
    color: var(--fg-dim);
    padding: var(--space-1) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .hbtn:hover:not(:disabled) {
    background: var(--glass-strong);
    color: var(--accent);
  }
  .hbtn:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
  /* The palette hint reads as a key, not a word — slightly tighter. */
  .kbd-hint {
    font-size: var(--text-xs);
    letter-spacing: 0.06em;
    padding: var(--space-1) var(--space-3);
  }
  .ctl-input {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
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
    background: var(--glass);
    color: var(--fg-dim);
    border: 0;
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-2);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    cursor: pointer;
  }
  .pin-unpin:hover {
    color: var(--accent-red);
    background: color-mix(in srgb, var(--accent-red) 12%, transparent);
  }
  /* The cast model: every speaker wears ONE neutral glass card — role
   * identity lives in the chip, provenance in the ppl badge, and hue
   * stays reserved for spaces and states.  Borderless: the glass fill +
   * top-light are the card; the border slot exists only so the A/B
   * shadow column can wear its violet-tinted hairline (a comparison
   * marker — state, not chrome). */
  .msg {
    border: 1px solid transparent;
    border-radius: var(--radius-lg);
    background: var(--glass);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    padding: var(--space-3) var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
    word-break: break-word;
  }
  .msg.shadow {
    border-color: color-mix(in srgb, var(--accent-purple) 26%, transparent);
  }

  .who {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .role-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--space-3);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fg-dim);
    padding: 2px 9px 2px 3px;
    border-radius: var(--radius-pill);
    background: var(--glass-strong);
    max-width: 40%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .role-chip b {
    font-weight: var(--weight-bold);
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.09);
    color: var(--fg);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    flex: none;
  }
  .who-meta {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .prov {
    margin-left: auto;
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    flex: none;
  }

  /* Stage direction — the system prompt as a note about the scene. */
  .stage {
    font-style: italic;
    color: var(--fg-subtle);
    font-size: var(--text-sm);
    line-height: 1.5;
    padding: var(--space-1) var(--space-5);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .stage.shadow {
    opacity: 0.7;
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
   * with the body indented when expanded.  Borderless: the caret + the
   * italic dim body delimit it; no rules. */
  .thinking-block {
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

  /* Cast row — compact chip-inputs above the composer.  Quiet until
   * hovered/filled; disabled (unsupported family / raw mode) reads as
   * plain dim text. */
  .cast-row {
    display: flex;
    align-items: center;
    gap: var(--space-5);
    padding: 0 var(--space-1);
  }
  .cast {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .cast-label {
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fg-muted);
  }
  .cast-input {
    width: 9em;
    background: var(--glass-strong);
    color: var(--fg-dim);
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    padding: 2px var(--space-4);
    transition: border-color var(--dur-fast) var(--ease-out);
  }
  .cast-input:focus-visible {
    outline: none;
    border-color: var(--accent-glow);
    color: var(--fg);
  }
  .cast-input.invalid {
    border-color: var(--accent-red);
  }
  .cast-input:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .seat-toggle {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .seat {
    background: var(--glass);
    color: var(--fg-muted);
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    padding: 2px var(--space-3);
    cursor: pointer;
    transition:
      background var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out);
  }
  .seat:hover {
    color: var(--fg);
  }
  .seat.active {
    background: var(--glass-strong);
    color: var(--fg-strong);
  }
  .cast-manage {
    margin-left: auto;
    background: none;
    border: none;
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    padding: 2px var(--space-2);
    transition: color var(--dur-fast) var(--ease-out);
  }
  .cast-manage:hover {
    color: var(--fg);
  }

  .thinking-row {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: 0 var(--space-1);
  }
  .thinking-toggle {
    align-self: flex-start;
    background: none;
    border: none;
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    padding: 0 var(--space-1);
    transition: color var(--dur-fast) var(--ease-out);
  }
  .thinking-toggle:hover,
  .thinking-toggle.open {
    color: var(--fg);
  }
  .thinking-toggle.drafted {
    color: var(--fg-strong);
  }
  .thinking-box {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .thinking-input {
    background: var(--input-well);
    color: var(--fg-dim);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    padding: var(--space-2) var(--space-3);
    resize: vertical;
    min-height: 44px;
  }
  .thinking-input:focus-visible {
    outline: none;
    border-color: var(--accent-glow);
    color: var(--fg);
  }
  .thinking-warn {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-style: italic;
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
    /* Borderless input: recessed well fill; the accent ring on focus. */
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
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
    background: var(--glass);
    color: var(--fg-strong);
    border: 0;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-5);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
    transition: background var(--dur-fast) var(--ease-out);
  }
  .input-actions button:hover:not(:disabled) {
    background: var(--glass-strong);
  }
  .input-actions button:disabled {
    color: var(--fg-muted);
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
