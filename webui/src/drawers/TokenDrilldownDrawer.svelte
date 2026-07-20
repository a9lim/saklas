<script lang="ts">
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // Per-token drilldown drawer — opens when a chat / raw-buffer token is
  // clicked.
  //
  // The drawer is a shell over four family tabs (drawers/token/):
  // geometry (Monitor readings), logits (top-K alts + fork), sae (gold),
  // j-lens (blue) — each speaking one shared grammar: the shell's
  // identity header (token text + id / raw / logprob chips), the context
  // ribbon, then a per-tab InstrumentHeader (provenance · source ·
  // steering · apply-recipe toggle) over the body.  The selected tab is
  // STICKY for the page session (drilldown.svelte.ts); j-lens is the
  // default.
  //
  // Navigation is a conversation-walking cursor (drawers/token/cursor.ts):
  // ←/→ step tokens and ROLL ACROSS segment + turn boundaries (thinking →
  // response → next turn), so chat mode walks the same flat stream the
  // raw buffer shows; ↑/↓ jump turns, Home/End jump segment bounds, and
  // a fresh token click snaps the cursor back to its anchor.  Drawer
  // params still come in via openDrawer("token_drilldown", { turnIdx,
  // tokenIdx, isThinking? }) and index chatLog.turns.

  import {
    drawerState,
    closeDrawer,
    chatLog,
    loomTree,
    samplingState,
    sessionState,
    effectiveRawMode,
    probeRack,
    lensSourceState,
    saeSourceState,
  } from "../lib/stores.svelte";
  import type {
    ChatTurn,
    LensTokenReadoutJSON,
    ProbeReadingJSON,
    SaeTokenReadoutJSON,
    TokenScore,
  } from "../lib/types";
  import SegmentedTabs from "../lib/ui/SegmentedTabs.svelte";
  import {
    clampCursor,
    jumpTurn,
    segmentTokens,
    segmentsOf,
    stepCursor,
    type SegmentKind,
    type TokenCursor,
  } from "./token/cursor";
  import {
    ReplayReadout,
    type GeometryTokenReadout,
  } from "./token/readout.svelte";
  import { drilldownUi, type DrilldownTab } from "./token/drilldown.svelte";
  import TokenRibbon from "./token/TokenRibbon.svelte";
  import GeometryTab from "./token/GeometryTab.svelte";
  import LogitsTab from "./token/LogitsTab.svelte";
  import SaeTab from "./token/SaeTab.svelte";
  import LensTab from "./token/LensTab.svelte";
  import { resolveReadoutTopK } from "../lib/readouts";

  interface DrawerParams {
    turnIdx: number;
    tokenIdx: number;
    /** When true the click came from the thinking-collapsible body, so
     * the anchor segment is ``turn.thinkingTokens``. */
    isThinking?: boolean;
  }

  // ---- params → anchor cursor -------------------------------------------

  // Drawer host forwards { params } — but we read off the store via
  // $derived below since drawerState.params is the source of truth.
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  const params = $derived(drawerState.params as DrawerParams | null);

  /** The position the user actually CLICKED — the drawer's anchor.  A
   *  fresh click (params object identity changes) snaps the cursor back
   *  here; the ↩ action does the same. */
  const anchor = $derived.by<TokenCursor | null>(() => {
    if (!params) return null;
    return {
      turnIdx: params.turnIdx,
      seg: params.isThinking ? "thinking" : "response",
      tokenIdx: params.tokenIdx,
    };
  });

  /** The walking cursor — every view reads the clamped ``effCursor``
   *  below; mutations write here. */
  let cursor = $state<TokenCursor | null>(null);

  /** Which branch we're inspecting — "primary" is the steered turn,
   * "shadow" the unsteered abPair when available.  Applies to the
   * cursor's current turn; walking into a different turn resets it. */
  type Branch = "primary" | "shadow";
  let branch: Branch = $state<Branch>("primary");

  const BRANCH_ITEMS: Array<{ value: Branch; label: string; title: string }> = [
    { value: "primary", label: "steered", title: "The steered (primary) turn" },
    { value: "shadow", label: "unsteered", title: "The unsteered A/B shadow turn" },
  ];

  $effect(() => {
    const a = anchor;
    cursor = a ? { ...a } : null;
    branch = "primary";
  });

  // ---- cursor resolution --------------------------------------------------

  /** The conversation with the shadow turn swapped in at the cursor's
   *  turn when the unsteered branch is selected. */
  const turnsView = $derived<ChatTurn[]>(
    chatLog.turns.map((t, i) =>
      i === (cursor?.turnIdx ?? -1) && branch === "shadow" && t.abPair
        ? t.abPair
        : t,
    ),
  );

  const segments = $derived(segmentsOf(turnsView));

  /** The effective inspected position — the cursor clamped onto the
   *  current segment list (token lists change while streaming). */
  const effCursor = $derived(cursor ? clampCursor(segments, cursor) : null);

  /** Reset the branch when the cursor crosses turns — the shadow pair is
   *  a per-turn artifact. */
  let lastBranchTurn = -1;
  $effect(() => {
    const t = effCursor?.turnIdx ?? -1;
    if (t !== lastBranchTurn) {
      lastBranchTurn = t;
      if (branch !== "primary") branch = "primary";
    }
  });

  /** The primary turn at the cursor (for abPair detection). */
  const turn = $derived<ChatTurn | null>(
    effCursor != null &&
      effCursor.turnIdx >= 0 &&
      effCursor.turnIdx < chatLog.turns.length
      ? chatLog.turns[effCursor.turnIdx]
      : null,
  );

  /** The turn actually inspected (shadow-swapped when selected). */
  const inspected = $derived<ChatTurn | null>(
    effCursor ? (turnsView[effCursor.turnIdx] ?? null) : null,
  );

  const tokenList = $derived<TokenScore[]>(
    effCursor ? segmentTokens(inspected, effCursor.seg) : [],
  );

  const token = $derived<TokenScore | null>(
    effCursor != null &&
      effCursor.tokenIdx >= 0 &&
      effCursor.tokenIdx < tokenList.length
      ? tokenList[effCursor.tokenIdx]
      : null,
  );

  const loomNodeId = $derived.by(() => {
    if (!effCursor) return null;
    if (branch === "shadow") {
      return inspected?.nodeId ?? null;
    }
    if (turn?.nodeId) return turn.nodeId;
    if (effCursor.turnIdx < 0 || loomTree.activePath.length === 0) return null;
    const visible = loomTree.activePath
      .map((id) => loomTree.nodes.get(id))
      .filter(Boolean)
      .filter((n) => !(n!.parent_id === null && n!.role === "system" && !n!.text));
    return visible[effCursor.turnIdx]?.id ?? null;
  });

  const hasReplayContext = $derived(
    loomNodeId != null && token?.rawIndex != null,
  );

  /** The durable generation record behind the inspected token. The
   *  detail view keeps this recipe beside the token rather than making
   *  users reconstruct it from the global controls. */
  const loomNode = $derived(
    loomNodeId ? (loomTree.nodes.get(loomNodeId) ?? null) : null,
  );
  const recipeSampling = $derived(loomNode?.recipe?.sampling ?? null);
  const recipeSteering = $derived(
    loomNode?.recipe?.steering ?? inspected?.appliedSteering ?? null,
  );
  function fmtSetting(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return "—";
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }

  const recipeChips = $derived.by<string[]>(() => {
    const sampling = recipeSampling;
    const recipe = loomNode?.recipe;
    if (!sampling && !recipe) return [];
    const chips = [
      `T ${fmtSetting(sampling?.temperature)}`,
      `top-p ${fmtSetting(sampling?.top_p)}`,
      `top-k ${fmtSetting(sampling?.top_k)}`,
      `max ${fmtSetting(sampling?.max_tokens)}`,
    ];
    const seed = recipe?.seed ?? sampling?.seed;
    if (seed != null) chips.push(`seed ${seed}`);
    if (sampling?.presence_penalty) chips.push(`presence ${fmtSetting(sampling.presence_penalty)}`);
    if (sampling?.frequency_penalty) chips.push(`frequency ${fmtSetting(sampling.frequency_penalty)}`);
    if (sampling?.return_top_k != null) chips.push(`alts ${sampling.return_top_k}`);
    if (recipe?.thinking != null) chips.push(recipe.thinking ? "thinking on" : "thinking off");
    if ((recipe?.probes.length ?? 0) > 0) chips.push(`${recipe!.probes.length} recipe probes`);
    return chips;
  });

  // ---- navigation ---------------------------------------------------------

  function moveTo(next: TokenCursor | null): void {
    if (next) cursor = next;
  }

  const canStepBack = $derived(
    effCursor != null && stepCursor(segments, effCursor, -1) !== null,
  );
  const canStepFwd = $derived(
    effCursor != null && stepCursor(segments, effCursor, 1) !== null,
  );
  const canPrevTurn = $derived(
    effCursor != null && jumpTurn(segments, effCursor, -1) !== null,
  );
  const canNextTurn = $derived(
    effCursor != null && jumpTurn(segments, effCursor, 1) !== null,
  );

  function step(delta: 1 | -1): void {
    if (effCursor) moveTo(stepCursor(segments, effCursor, delta));
  }
  function turnHop(delta: 1 | -1): void {
    if (effCursor) moveTo(jumpTurn(segments, effCursor, delta));
  }
  function segEdge(where: "home" | "end"): void {
    if (!effCursor) return;
    cursor = {
      ...effCursor,
      tokenIdx: where === "home" ? 0 : Math.max(0, tokenList.length - 1),
    };
  }

  const atAnchor = $derived(
    effCursor != null &&
      anchor != null &&
      effCursor.turnIdx === anchor.turnIdx &&
      effCursor.seg === anchor.seg &&
      effCursor.tokenIdx === anchor.tokenIdx,
  );

  function resetToAnchor(): void {
    if (anchor) cursor = { ...anchor };
  }

  /** The counterpart segment of the current turn, when it exists —
   *  clicking the segment badge jumps to its start. */
  const otherSeg = $derived.by<SegmentKind | null>(() => {
    if (!effCursor) return null;
    const other: SegmentKind =
      effCursor.seg === "thinking" ? "response" : "thinking";
    return segments.some(
      (s) => s.turnIdx === effCursor.turnIdx && s.seg === other,
    )
      ? other
      : null;
  });

  function toggleSeg(): void {
    if (!effCursor || !otherSeg) return;
    cursor = { turnIdx: effCursor.turnIdx, seg: otherSeg, tokenIdx: 0 };
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      closeDrawer();
      return;
    }
    // Never steal keys from a focusable field or the layer-strip
    // scrubbers (role="slider" owns its own arrow keys).
    const t = ev.target as HTMLElement | null;
    if (
      t &&
      (t.tagName === "INPUT" ||
        t.tagName === "TEXTAREA" ||
        t.tagName === "SELECT" ||
        t.isContentEditable ||
        t.closest('[role="slider"]'))
    ) {
      return;
    }
    switch (ev.key) {
      case "ArrowLeft":
        step(-1);
        break;
      case "ArrowRight":
        step(1);
        break;
      case "ArrowUp":
        turnHop(-1);
        break;
      case "ArrowDown":
        turnHop(1);
        break;
      case "Home":
        segEdge("home");
        break;
      case "End":
        segEdge("end");
        break;
      default:
        return;
    }
    ev.preventDefault();
  }

  // ---- identity chips -------------------------------------------------

  const roleLabel = $derived(
    inspected ? (inspected.roleLabel ?? inspected.role) : "",
  );

  /** Rank of the chosen token within its captured alts, when present. */
  const chosenRank = $derived.by<number | null>(() => {
    const alts = token?.topAlts;
    if (!alts || alts.length === 0 || token?.tokenId == null) return null;
    const i = alts.findIndex((a) => a.id === token.tokenId);
    return i >= 0 ? i + 1 : null;
  });

  function fmtP(p: number): string {
    if (!Number.isFinite(p)) return "—";
    if (p >= 0.001) return p.toFixed(3);
    return p.toExponential(1);
  }

  // ---- tabs ----------------------------------------------------------

  const tabItems = $derived<Array<{
    value: DrilldownTab;
    label: string;
    meta: string;
    color?: string;
    title: string;
  }>>([
    {
      value: "geometry",
      label: "geometry",
      meta: String(Object.keys(token?.measurements?.instruments.geometry?.readings ?? {}).length),
      color: "var(--fg-dim)",
      title: "Activation geometry",
    },
    {
      value: "logits",
      label: "logits",
      meta: String(token?.topAlts?.length ?? 0),
      title: "Sampling alternatives",
    },
    {
      value: "sae",
      label: "sae",
      meta: String(token?.measurements?.instruments.sae?.readout?.features.length ?? 0),
      color: "var(--pillar-sae)",
      title: "Sparse feature field",
    },
    {
      value: "lens",
      label: "j-lens",
      meta: String(token?.measurements?.instruments.lens?.readout?.layers.length ?? 0),
      color: "var(--pillar-lens)",
      title: "J-lens workspace",
    },
  ]);

  const hasAbPair = $derived(turn?.abPair != null);

  // ---- instrument readouts (captured-or-replay) ------------------------
  //
  // The token's loom-owned ``measurements`` envelope is authoritative;
  // replay covers old/missing channels and the explicit unsteered
  // counterfactual.  One ReplayReadout per family — the shell owns them
  // (and the steered flags) so tab switches don't lose state; the tabs
  // are presentational.

  const jlensFitted = $derived(sessionState.info?.jlens_fitted === true);
  const saeLoaded = $derived(sessionState.info?.sae_loaded === true);

  // Share the logit-alternative width. Zero means the ordinary logit
  // capture is off, so retain the canonical eight-wide read-side view.
  const readoutTopK = $derived(resolveReadoutTopK(samplingState.return_top_k));

  const lensReadout = new ReplayReadout<LensTokenReadoutJSON>();
  const saeReadout = new ReplayReadout<SaeTokenReadoutJSON>();
  const geometryReadout = new ReplayReadout<GeometryTokenReadout>();
  let lensSteered = $state(true);
  let saeSteered = $state(true);
  let geometrySteered = $state(true);

  /** ≥1 attached Monitor probe (anything in the rack that isn't a lens
   *  or SAE readout probe) — the geometry replay 400s on an empty roster. */
  const hasGeometryProbes = $derived(
    probeRack.active.some((name) => {
      const info = probeRack.entries.get(name)?.info;
      return !!info && !info.lens && !info.sae;
    }),
  );

  const lensPinned = $derived<Record<string, ProbeReadingJSON> | null>(
    token?.measurements?.instruments.lens?.readings ?? null,
  );
  const saePinned = $derived<Record<string, ProbeReadingJSON> | null>(
    token?.measurements?.instruments.sae?.readings ?? null,
  );

  const capturedLensData = $derived.by<LensTokenReadoutJSON | null>(() => {
    const lens = token?.measurements?.instruments.lens;
    if (!lens?.readout || !token) return null;
    return {
      node_id: loomNodeId ?? "",
      raw_index: token.rawIndex ?? -1,
      token_id: token.tokenId ?? -1,
      token_text: token.text,
      steering: lens.binding.steering,
      aggregate: lens.readout.aggregate,
      layers: lens.readout.layers,
    };
  });

  const capturedSaeData = $derived.by<SaeTokenReadoutJSON | null>(() => {
    const sae = token?.measurements?.instruments.sae;
    if (!sae?.readout || !token) return null;
    return {
      node_id: loomNodeId ?? "",
      raw_index: token.rawIndex ?? -1,
      token_id: token.tokenId ?? -1,
      token_text: token.text,
      steering: sae.binding.steering,
      layer: sae.binding.layer ?? -1,
      features: sae.readout.features,
    };
  });

  const capturedGeometryData = $derived.by<GeometryTokenReadout | null>(() => {
    const geometry = token?.measurements?.instruments.geometry;
    if (!geometry || Object.keys(geometry.readings ?? {}).length === 0) {
      return null;
    }
    return {
      steering: geometry.binding?.steering ?? null,
      readings: geometry.readings,
    };
  });

  $effect(() => {
    if (drilldownUi.tab !== "lens") return;
    const captured = capturedLensData;
    if (lensSteered && captured) {
      lensReadout.adopt(
        captured,
        token?.measurements?.instruments.lens?.binding.source ?? null,
      );
      return;
    }
    lensReadout.clear();
    if (!jlensFitted) return;
    const nodeId = loomNodeId;
    const rawIndex = token?.rawIndex;
    if (!nodeId || rawIndex == null) return;
    lensReadout.replay(
      "lens",
      nodeId,
      rawIndex,
      { topK: readoutTopK, steered: lensSteered, raw: effectiveRawMode(), layers: "all" },
      (m) => {
        const lens = m.instruments.lens;
        return {
          data: {
            node_id: nodeId,
            raw_index: rawIndex,
            token_id: token?.tokenId ?? -1,
            token_text: token?.text ?? "",
            steering: lens?.binding.steering ?? null,
            aggregate: lens?.readout?.aggregate ?? [],
            layers: lens?.readout?.layers ?? [],
          },
          source:
            lens?.binding.source ??
            lensSourceState.sources.find((source) => source.active)?.source ??
            null,
        };
      },
    );
  });

  $effect(() => {
    if (drilldownUi.tab !== "sae") return;
    const captured = capturedSaeData;
    if (saeSteered && captured) {
      saeReadout.adopt(
        captured,
        token?.measurements?.instruments.sae?.binding.source ?? null,
      );
      return;
    }
    saeReadout.clear();
    if (!saeLoaded) return;
    const nodeId = loomNodeId;
    const rawIndex = token?.rawIndex;
    if (!nodeId || rawIndex == null) return;
    saeReadout.replay(
      "sae",
      nodeId,
      rawIndex,
      { topK: readoutTopK, steered: saeSteered, raw: effectiveRawMode() },
      (m) => {
        const sae = m.instruments.sae;
        return {
          data: {
            node_id: nodeId,
            raw_index: rawIndex,
            token_id: token?.tokenId ?? -1,
            token_text: token?.text ?? "",
            steering: sae?.binding.steering ?? null,
            layer: sae?.binding.layer ?? -1,
            features: sae?.readout?.features ?? [],
          },
          source:
            sae?.binding.source ??
            saeSourceState.sources.find((source) => source.active)?.source ??
            sessionState.info?.sae_info?.release ??
            null,
        };
      },
    );
  });

  $effect(() => {
    if (drilldownUi.tab !== "geometry") return;
    const captured = capturedGeometryData;
    if (geometrySteered && captured) {
      geometryReadout.adopt(captured, null);
      return;
    }
    geometryReadout.clear();
    if (!hasGeometryProbes) return;
    const nodeId = loomNodeId;
    const rawIndex = token?.rawIndex;
    if (!nodeId || rawIndex == null) return;
    geometryReadout.replay(
      "geometry",
      nodeId,
      rawIndex,
      { steered: geometrySteered, raw: effectiveRawMode() },
      (m) => {
        const geometry = m.instruments.geometry;
        return {
          data: {
            steering: geometry?.binding?.steering ?? null,
            readings: geometry?.readings ?? {},
          },
          source: null,
        };
      },
    );
  });
</script>

<svelte:window onkeydown={onKeydown} />

<aside
  class="drawer"
  aria-label="Token drilldown"
>
  <header class="drawer-header">
    <div class="title">
      <span class="eyebrow">token drilldown</span>
      {#if token && effCursor}
        <div class="name-row">
          <code class="tok-text" title={`generated token ${JSON.stringify(token.text)}`}>
            {JSON.stringify(token.text)}
          </code>
          <button
            type="button"
            class="kv-chip seg-chip"
            disabled={!otherSeg}
            onclick={toggleSeg}
            title={otherSeg
              ? `jump to this turn's ${otherSeg} tokens`
              : "turn segment"}
          >
            turn {effCursor.turnIdx} · {roleLabel} · {effCursor.seg}
          </button>
          {#if token.tokenId != null}
            <span class="kv-chip" title="vocabulary id">id {token.tokenId}</span>
          {/if}
          {#if token.rawIndex != null}
            <span class="kv-chip" title="raw decode-step index — the fork / replay join key">
              raw {token.rawIndex}
            </span>
          {:else}
            <span
              class="kv-chip warn"
              title="no raw decode record — logit forks and instrument replay are unavailable for this token"
            >
              no replay
            </span>
          {/if}
          {#if token.logprob != null}
            <span
              class="kv-chip"
              title="chosen-token probability under the post-temperature / post-top-p / post-top-k distribution the sampler drew from"
            >
              p {fmtP(Math.exp(token.logprob))} · logp {token.logprob.toFixed(3)}{chosenRank !== null
                ? ` · rank ${chosenRank}/${token.topAlts?.length ?? 0}`
                : ""}
            </span>
          {/if}
        </div>
        <div class="nav-row">
          <span class="scrub" title="Walk token sequence">
            <button
              type="button"
              class="scrub-btn"
              disabled={!canStepBack}
              onclick={() => step(-1)}
              aria-label="Previous token"
            >◀</button>
            <span class="scrub-pos">{effCursor.tokenIdx + 1} / {tokenList.length}</span>
            <button
              type="button"
              class="scrub-btn"
              disabled={!canStepFwd}
              onclick={() => step(1)}
              aria-label="Next token"
            >▶</button>
          </span>
          <span class="scrub" title="Jump turns">
            <button
              type="button"
              class="scrub-btn"
              disabled={!canPrevTurn}
              onclick={() => turnHop(-1)}
              aria-label="Previous turn"
            >▲</button>
            <span class="scrub-pos">turn {effCursor.turnIdx}</span>
            <button
              type="button"
              class="scrub-btn"
              disabled={!canNextTurn}
              onclick={() => turnHop(1)}
              aria-label="Next turn"
            >▼</button>
          </span>
          {#if !atAnchor}
            <button
              type="button"
              class="scrub-btn scrub-home"
              onclick={resetToAnchor}
              title="back to the clicked token"
            >↩</button>
          {/if}
        </div>
        <div class="generation-context">
          <div class="recipe">
            <span class="recipe-label">generation recipe</span>
            <code class="recipe-steering" title={recipeSteering ?? "no steering"}>
              {recipeSteering ?? "unsteered"}
            </code>
            {#each recipeChips as chip (chip)}
              <span>{chip}</span>
            {/each}
          </div>
        </div>
      {:else}
        <div class="name-row">
          <span class="coord">
            no token at (turn {params?.turnIdx ?? "?"}, {params?.tokenIdx ?? "?"})
          </span>
        </div>
      {/if}
    </div>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  {#if token && effCursor}
    <TokenRibbon
      tokens={tokenList}
      index={effCursor.tokenIdx}
      onjump={(i) => {
        if (effCursor) cursor = { ...effCursor, tokenIdx: i };
      }}
    />
  {/if}

  <!-- View tabs + the steered/unsteered branch toggle when this turn has
       an A/B pair.  Tabs always render so users see the other views
       exist even when their capture is off. -->
  <div class="toolbar">
    <SegmentedTabs items={tabItems} bind:value={drilldownUi.tab} ariaLabel="Token detail view" />
    {#if hasAbPair}
      <SegmentedTabs items={BRANCH_ITEMS} bind:value={branch} ariaLabel="Token branch" />
    {/if}
  </div>

  <div class="body">
    {#if !token || !effCursor}
      <div class="empty">token unavailable</div>
    {:else if drilldownUi.tab === "geometry"}
      <GeometryTab
        readout={geometryReadout}
        bind:steered={geometrySteered}
        {hasGeometryProbes}
        {hasReplayContext}
      />
    {:else if drilldownUi.tab === "logits"}
      <LogitsTab {token} nodeId={loomNodeId} />
    {:else if drilldownUi.tab === "sae"}
      <SaeTab
        readout={saeReadout}
        bind:steered={saeSteered}
        {saeLoaded}
        {hasReplayContext}
        pinned={saePinned}
      />
    {:else}
      <LensTab
        readout={lensReadout}
        bind:steered={lensSteered}
        {jlensFitted}
        {hasReplayContext}
        pinned={lensPinned}
        modelId={sessionState.info?.model_id ?? null}
      />
    {/if}
  </div>

  <footer class="drawer-footer">
    <span class="hint">
      {#if drilldownUi.tab === "geometry"}
        Full whitened Monitor readings at the forward that produced this
        token: coords are domain-frame subspace coordinates, fraction the
        in-subspace share, nearest the whitened node distances. Replay
        scores the current roster post-hoc, so aggregate-only generations
        and probes attached after the fact still read.
      {:else if drilldownUi.tab === "logits"}
        Logprob is the chosen-token natural-log probability under the
        post-temperature / post-top-p / post-top-k distribution the sampler
        drew from. Bars show absolute probability.
      {:else if drilldownUi.tab === "sae"}
        Post-nonlinearity sparse-feature activations from the resident SAE's
        hook layer. Strength is activation / Neuronpedia maxActApprox — the
        absolute 0..1 unit gates read; features without metadata show raw
        activation on this readout's own scale.
      {:else}
        Each row ranks softmax(W_U · norm(J_l·h)) at the forward that
        produced this token — what that layer's residual was disposed to
        make the model say. Cell tint = probability; highlighted cells match
        the produced token. All fitted layers are shown because the
        informative depth range is model-dependent.
      {/if}
    </span>
  </footer>
</aside>

<style>
  /* v2 sheet interior — the host paints the sheet surface (glass hairline,
   * radius, --bg-alt fill), so the root is transparent; chrome speaks sans
   * and every value/token/expression sits in mono. */
  .drawer {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    background: transparent;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }

  .drawer-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--space-5) var(--space-6) var(--space-3);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    min-width: 0;
    flex: 1 1 auto;
  }
  .eyebrow {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .name-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
    flex-wrap: wrap;
  }
  .tok-text {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-md);
    background: var(--glass-strong);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    word-break: break-all;
    max-width: 28ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .coord {
    color: var(--fg-subtle);
    font-size: var(--text-sm);
    white-space: nowrap;
  }

  /* Identity chips — quiet mono capsules; the segment chip doubles as
   * the thinking/response jump when the turn has both. */
  .kv-chip {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    padding: var(--space-1) var(--space-3);
    white-space: nowrap;
  }
  .kv-chip.warn {
    color: var(--fg-muted);
    font-style: italic;
  }
  button.seg-chip {
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    cursor: pointer;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  button.seg-chip:hover:not(:disabled) {
    color: var(--fg);
    background: var(--glass-strong);
  }
  button.seg-chip:disabled {
    cursor: default;
  }

  .nav-row {
    display: flex;
    align-items: center;
    gap: var(--space-5);
    flex-wrap: wrap;
  }
  .scrub {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .scrub-btn {
    background: var(--glass);
    color: var(--fg-muted);
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    font: inherit;
    font-size: var(--text-2xs);
    line-height: 1;
    padding: var(--space-2) var(--space-4);
    cursor: pointer;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .scrub-btn:hover:not(:disabled) {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .scrub-btn:disabled {
    color: var(--fg-muted);
    opacity: 0.35;
    cursor: default;
  }
  .scrub-pos {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .scrub-home {
    color: var(--accent);
    font-size: var(--text-xs);
  }

  .generation-context {
    min-width: 0;
  }
  .recipe {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    flex-wrap: wrap;
    min-width: 0;
    margin: 0;
    padding: var(--space-3);
    border-radius: var(--radius);
    background: var(--input-well);
  }
  .recipe-label {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    white-space: nowrap;
  }
  .recipe code {
    background: transparent;
    font-family: var(--font-mono);
  }
  .recipe {
    gap: var(--space-2) var(--space-3);
  }
  .recipe > span:not(.recipe-label),
  .recipe-steering {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .recipe-steering {
    color: var(--fg);
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 30ch;
  }

  /* Toolbar — view tabs left, branch toggle right (when A/B). */
  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--space-3) var(--space-6);
  }

  .body {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-6);
    overflow: auto;
    min-height: 0;
    padding: var(--space-5) var(--space-6);
  }
  .empty {
    color: var(--fg-muted);
    padding: var(--space-6) 0;
    line-height: 1.5;
    max-width: 62ch;
  }

  .drawer-footer {
    padding: var(--space-3) var(--space-6);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .hint {
    line-height: 1.5;
  }

  @media (max-width: 820px) {
    .toolbar {
      align-items: flex-start;
      flex-direction: column;
    }
  }
</style>
