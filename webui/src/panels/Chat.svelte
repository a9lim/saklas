<script lang="ts">
  // Chat panel — v1.7 rewrite.  Replaces the v1.6 ChatPlaceholder/legacy
  // shape with the full feature set: thinking-collapsible per turn
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
  // label, arbitrary strings first-class). Analysis badges appear only when
  // their backing artifacts exist. System turns render as stage directions (a
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
    sendSubmit,
    sendStop,
    genStatus,
    openDrawer,
    inputHistory,
    inputRestore,
    pushInputHistory,
    navigateInputHistory,
    cancelInputPull,
    consumePulledSlot,
    loomRegenerateNode,
    enqueuePending,
    pendingActions,
    cancelPendingAction,
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
    highlightHue,
    twoStripeStyle,
    twoBlendStyle,
    formatScoreTooltip,
    surpriseScore,
    SURPRISE_TARGET,
    probeScoreForTarget,
  } from "../lib/tokens";
  import Select from "../lib/Select.svelte";
  import Checkbox from "../lib/Checkbox.svelte";
  import Combobox from "../lib/Combobox.svelte";
  import Button from "../lib/ui/Button.svelte";

  // --------------------------------------------------------------- input --

  let input = $state("");
  let textareaRef: HTMLTextAreaElement | null = $state(null);

  // ---------------------------------------------------------- cast row --
  //
  // The ``speaking as`` / ``reply as`` chips — the SamplingStrip role
  // boxes promoted into the composer (the visible step toward the cast
  // model).  Same client state (``samplingState.human_role`` /
  // ``model_role``), lowered to protocol role labels at the wire boundary.
  // Same support gates: base /
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
    samplingState.human_role.trim() === "" ||
      CAST_SLUG_RE.test(samplingState.human_role.trim()),
  );
  const castAsstValid = $derived(
    samplingState.model_role.trim() === "" ||
      CAST_SLUG_RE.test(samplingState.model_role.trim()),
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

  // --- Unified submission -----------------------------------------------
  // The composer has exactly two states: ordinary and swapped.  Selection
  // chooses the branch anchor only; it never changes what the box means.
  // Same-role adjacency coalesces in the engine, so prefill is no longer a
  // separate composer state: append a model prefix, then generate model.
  const rawMode = $derived.by(() => {
    void genUiMode.mode;
    return effectiveRawMode();
  });

  const activeNodeId = $derived(
    loomTree.loaded ? (loomTree.active_node_id ?? null) : null,
  );
  const sceneMode = $derived(sessionState.info?.scene_mode ?? false);
  let swapSeats = $state(false);
  $effect(() => {
    // A non-scene renderer cannot open a generation prompt on the human
    // seat.  Keep that compatibility boundary localized to the checkbox.
    if ((!sceneMode || rawMode) && swapSeats) swapSeats = false;
  });
  const authoredSeat = $derived<"human" | "model">(
    swapSeats ? "model" : "human",
  );
  const generatedSeat = $derived<"human" | "model">(
    swapSeats ? "human" : "model",
  );
  const humanLabel = $derived(
    samplingState.human_role.trim()
      || sessionState.info?.default_user_role
      || "human",
  );
  const modelLabel = $derived(
    samplingState.model_role.trim()
      || sessionState.info?.default_assistant_role
      || "model",
  );
  const authoredLabel = $derived(authoredSeat === "human" ? humanLabel : modelLabel);

  // Authored-thinking input: a block the next *append* (modifier+Enter) carries,
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

  // --- Append modifier (Ctrl / Cmd / Option) ----------------------------
  // Any of Ctrl, Cmd (⌘), or Option (⌥) appends a non-empty draft without
  // generation. Empty input always generates in the generated seat; the
  // modifier is deliberately irrelevant when there is nothing to append.
  // Tracked at the window level so the
  // state survives textarea blur and we can swap the send-button caption
  // the moment the modifier comes down — without needing the user to
  // type anything first.
  let modHeld = $state(false);
  const hasText = $derived(input.trim() !== "");
  const appendMode = $derived(modHeld && hasText);

  /** Keep the composer prompt to the active action; shortcuts live in help. */
  const inputPlaceholder = $derived(
    appendMode
      ? `append ${authoredLabel}…`
      : `message as ${authoredLabel}…`,
  );
  const sendLabel = $derived(
    appendMode ? "append" : hasText ? "send" : "generate",
  );

  const castRoleOptions = $derived(
    [...new Set([
      sessionState.info?.default_user_role,
      sessionState.info?.default_assistant_role,
      ...Object.keys(castState.roster),
    ].filter((value): value is string => !!value))]
      .sort((a, b) => a.localeCompare(b))
      .map((value) => ({ value, label: value })),
  );

  function doSend(append: boolean = false): void {
    const text = hasText ? input : "";
    const replaceSlot = consumePulledSlot();
    if (!text) {
      if (replaceSlot !== null) {
        cancelPendingAction(pendingActions.queue[replaceSlot]?.id ?? "");
        return;
      }
    }
    const parent = isPendingBusy()
      ? ("active@drain" as const)
      : activeNodeId;
    const thinking = text && thinkingDraft.trim() !== "" ? thinkingDraft : null;
    if (thinking !== null) {
      thinkingDraft = "";
      thinkingOpen = false;
    }
    if (text) {
      pushInputHistory(text);
      input = "";
    }
    const appendOnly = append && text !== "";
    void sendSubmit(
      text || null,
      text ? authoredSeat : null,
      appendOnly ? null : generatedSeat,
      {
        parent_node_id: parent,
        replaceSlot,
        raw: rawMode,
        authored_thinking: thinking,
      },
    );
    scrolledUp = false;
    queueScrollToBottom();
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
      // Shift-Enter is a newline; Ctrl/Cmd/Option-Enter is the append
      // modifier (no generation); bare Enter sends or generates according
      // to whether the box has text. Reading the modifier flags directly is
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
  // clear / transcript / auto-regen used to live on the Topbar;
  // they act on the conversation, so they belong here.  The mutating
  // ones route through ``enqueuePending`` so clicking them mid-gen
  // queues rather than racing the WS.

  const AUTO_REGEN_MODES: { value: AutoRegenMode; label: string }[] = [
    { value: "unsteered", label: "unsteered" },
    { value: "inverted", label: "inverted" },
    { value: "reseed", label: "reseed" },
    { value: "cool", label: "cool" },
    { value: "hot", label: "hot" },
    { value: "custom", label: "custom…" },
  ];

  function regenMessage(turn: ChatTurn): void {
    const nodeId = turn.nodeId;
    if (!nodeId) return;
    if (isPendingBusy()) {
      enqueuePending({
        label: "regen",
        text: null,
        apply: () => void loomRegenerateNode(nodeId, 1),
        awaitsGen: true,
        rebuild: null,
        endsOnUserNode: turn.role === "user",
      });
    } else {
      void loomRegenerateNode(nodeId, 1);
    }
  }

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
          roleLabel: node.role_label,
          nodeId: node.id,
          generated: node.recipe !== null,
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

    // Track any append modifier at the window level so the send-button
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
    // fires — without it the label sticks in "append" mode after the
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
        ? twoBlendStyle(aScore, bScore, scaleA, scaleB, highlightHue(a), highlightHue(b))
        : twoStripeStyle(aScore, bScore, scaleA, scaleB, highlightHue(a), highlightHue(b));
    }
    const bg = scoreToRgb(aScore, scaleA, highlightHue(a));
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
              {#if !turn.generated || turn.role === "system"}
                {@render bubble(turn, turnIdx, false)}
              {:else if turn.abPair}
                {@render bubble(turn.abPair, turnIdx, true)}
              {:else}
                <div class="msg placeholder" aria-hidden="true">
                  <div class="who">
                    <span class="role-chip">
                      <b>{roleGlyphLetter(turn.role, turn.roleLabel)}</b>
                      {roleDisplayLabel(turn.role, turn.roleLabel)}
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
      <span class="cast-label">human</span>
      <Combobox
        bind:value={samplingState.human_role}
        options={castRoleOptions}
        disabled={!castReady || !castUserSupported}
        invalid={!castUserValid}
        placeholder={sessionState.info?.default_user_role ?? "human"}
        spellcheck={false}
        ariaLabel="human role label"
      />
    </label>
    <label
      class="cast"
      title={castAsstSupported
        ? "model role"
        : "unavailable"}
    >
      <span class="cast-label">model</span>
      <Combobox
        bind:value={samplingState.model_role}
        options={castRoleOptions}
        disabled={!castReady || !castAsstSupported}
        invalid={!castAsstValid}
        placeholder={sessionState.info?.default_assistant_role ?? "model"}
        spellcheck={false}
        ariaLabel="model role label"
      />
    </label>
    <div
      class="seat-toggle"
      title={sceneMode && !rawMode
        ? "swap who writes and who replies"
        : "seat swapping requires scene mode"}
    >
      <Checkbox
        bind:checked={swapSeats}
        disabled={!sceneMode || rawMode}
        label="swap seats"
        ariaLabel="swap human and model seats"
      />
    </div>
    <div class="cast-manage">
      <Button
        size="sm"
        title="cast"
        onclick={() => openDrawer("cast")}
      >cast…</Button>
    </div>
  </div>

  {#if thinkingInputSupported}
    <div class="thinking-row">
      <div class="thinking-control">
        <Button
          variant="flat"
          size="sm"
          accent={thinkingDraft.trim() !== "" ? "var(--accent-violet)" : undefined}
          onclick={() => (thinkingOpen = !thinkingOpen)}
          title="next append"
        >{thinkingOpen ? "− thinking" : "+ thinking"}</Button>
      </div>
      {#if thinkingOpen}
        <div class="thinking-box">
          <textarea
            class="thinking-input"
            bind:value={thinkingDraft}
            placeholder="thinking…"
            rows="2"
            spellcheck="false"
            aria-label="authored thinking block"
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
      bind:this={textareaRef}
      bind:value={input}
      onkeydown={onKeydown}
      placeholder={inputPlaceholder}
      rows="1"
      aria-label={`Compose as ${authoredLabel}`}
    ></textarea>
    <div class="input-actions">
      <Button
        type="submit"
        accent="var(--accent-green)"
        disabled={!loomTree.loaded}
        title={appendMode ? "modifier + ⏎ append" : "⏎ submit"}
      >{sendLabel}</Button>
      <Button
        variant="danger"
        onclick={sendStop}
        disabled={!genStatus.active}
        title="Esc"
      >stop</Button>
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
      {#if turn.nodeId}
        <Button
          size="sm"
          onclick={() => regenMessage(turn)}
          title="reroll this message"
          ariaLabel={`Reroll ${roleDisplayLabel(turn.role, turn.roleLabel)} message`}
        >↻</Button>
      {/if}
      {#if isShadow && !pinnedActive}<span class="who-meta">(unsteered)</span>{/if}
      {#if turn.meanLogprob != null && Number.isFinite(turn.meanLogprob)}
        <span
          class="prov"
          title="sequence perplexity"
        >seq ppl {Math.exp(-turn.meanLogprob).toFixed(1)}</span>
      {/if}
    </div>

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
  /* Every speaker wears ONE neutral glass card — role identity lives in
   * the chip, while optional analysis artifacts live beside it. Hue stays
   * reserved for spaces and states.  Borderless: the glass fill +
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
    width: 13em;
    min-width: 0;
  }
  .cast-label {
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fg-muted);
  }
  .seat-toggle {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .cast-manage {
    margin-left: auto;
  }

  @media (max-width: 600px) {
    /* Keep both editable role labels legible on phones.  Four controls in one
     * flex row truncated the model value (often to ``mode``) and compressed
     * the seat toggle into an ambiguous sliver. */
    .cast-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: var(--space-2) var(--space-4);
    }
    .cast {
      width: auto;
    }
    .seat-toggle {
      grid-column: 1;
    }
    .cast-manage {
      grid-column: 2;
      justify-self: end;
      margin-left: 0;
    }
  }

  .thinking-row {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: 0 var(--space-1);
  }
  .thinking-control {
    align-self: flex-start;
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
  .input-actions {
    display: flex;
    gap: var(--space-2);
    align-items: center;
  }
</style>
