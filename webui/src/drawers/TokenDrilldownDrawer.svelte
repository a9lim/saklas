<script lang="ts">
  // Per-token drilldown drawer — opens when a chat token is clicked.
  //
  // Replaces the v1.6 always-on inspector (which forced every user to look
  // at a per-token × per-layer × per-probe heatmap whether they wanted it
  // or not) with an on-demand surface keyed to a specific token.  Drawer
  // params come in via openDrawer("token_drilldown", { turnIdx, tokenIdx,
  // isThinking? }).  Click data flows through chatLog.turns[turnIdx] —
  // either turn.tokens[tokenIdx] (response stream, default) or
  // turn.thinkingTokens[tokenIdx] when isThinking is true.
  //
  // Layout: header with the token text + coordinates + position scrubber,
  // a toolbar with the view tabs (probes / logits / j-lens; only the lens
  // carries a hue dot — it is the one pillar-owned surface here) and the
  // steered/unsteered branch toggle, then the per-tab body.  Heatmap cells
  // tint via tokens.scoreToRgb so highlight saturation matches the chat
  // tokens themselves; the j-lens matrix tints in the lens family blue.

  import {
    drawerState,
    closeDrawer,
    chatLog,
    loomTree,
    sendFork,
    samplingState,
    sessionState,
    effectiveRawMode,
    probeAxisScale,
  } from "../lib/stores.svelte";
  import { ApiError, apiLens, apiSae } from "../lib/api";
  import type {
    ChatTurn,
    LensTokenReadoutJSON,
    SaeTokenReadoutJSON,
    TokenScore,
  } from "../lib/types";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import SegmentedTabs from "../lib/ui/SegmentedTabs.svelte";
  import Button from "../lib/ui/Button.svelte";

  interface DrawerParams {
    turnIdx: number;
    tokenIdx: number;
    /** When true the click came from the thinking-collapsible body, so
     * the source list is ``turn.thinkingTokens`` instead of
     * ``turn.tokens``.  Defaults to false (response stream). */
    isThinking?: boolean;
  }

  // ---- params + branch selection ---------------------------------------

  // Drawer host forwards { params } — but we read off the store via
  // $derived below since drawerState.params is the source of truth.
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  const params = $derived(drawerState.params as DrawerParams | null);
  const turnIdx = $derived(params?.turnIdx ?? -1);
  /** The token index the user actually CLICKED — the drawer's anchor.
   *  Tab / branch resets key off this, never the scrubbed position, so
   *  walking the scrubber doesn't kick the user off their tab. */
  const paramTokenIdx = $derived(params?.tokenIdx ?? -1);
  const isThinking = $derived(params?.isThinking === true);

  /** Scrubber override — walks the inspected position along the turn's
   *  token list without re-opening the drawer.  ``null`` means "at the
   *  clicked token"; any fresh click (params object identity changes)
   *  snaps back to it. */
  let scrubTokenIdx = $state<number | null>(null);
  $effect(() => {
    void params;
    scrubTokenIdx = null;
  });

  /** The effective inspected token index — every view (probes heatmap,
   *  logits table, j-lens matrix, fork actions) reads this one. */
  const tokenIdx = $derived(scrubTokenIdx ?? paramTokenIdx);

  /** Which branch we're inspecting — "primary" is the steered turn,
   * "shadow" is the unsteered abPair when available.  Local UI state
   * because the drawer toggles between them without changing the click
   * target.  The explicit ``$state<...>`` widens the rune's literal-
   * inferred type so reassignment to ``"shadow"`` later doesn't trip
   * a "comparison has no overlap" narrowing error. */
  type Branch = "primary" | "shadow";
  let branch: Branch = $state<Branch>("primary");
  let branchingRank: number | null = $state(null);
  let branchError: string | null = $state(null);

  const BRANCH_ITEMS: Array<{ value: Branch; label: string; title: string }> = [
    { value: "primary", label: "steered", title: "The steered (primary) turn" },
    { value: "shadow", label: "unsteered", title: "The unsteered A/B shadow turn" },
  ];

  /** Reset the branch when the click target changes — opening the drawer
   * on a new token should always start on the primary side. */
  $effect(() => {
    void turnIdx;
    void paramTokenIdx;
    branch = "primary";
  });

  const turn = $derived<ChatTurn | null>(
    turnIdx >= 0 && turnIdx < chatLog.turns.length
      ? chatLog.turns[turnIdx]
      : null,
  );

  /** The actual ChatTurn whose tokens we're inspecting — primary or the
   * abPair when ``branch === "shadow"``.  Null when the indices are
   * stale (e.g. the user cleared the chat after opening the drawer). */
  const inspected = $derived<ChatTurn | null>(
    branch === "shadow" ? (turn?.abPair ?? null) : turn,
  );

  const tokenList = $derived<TokenScore[]>(
    (isThinking ? inspected?.thinkingTokens : inspected?.tokens) ?? [],
  );

  const token = $derived<TokenScore | null>(
    tokenIdx >= 0 && tokenIdx < tokenList.length ? tokenList[tokenIdx] : null,
  );

  const loomNodeId = $derived.by(() => {
    if (branch === "shadow") {
      return inspected?.nodeId ?? null;
    }
    if (turn?.nodeId) return turn.nodeId;
    if (turnIdx < 0 || loomTree.activePath.length === 0) return null;
    const visible = loomTree.activePath
      .map((id) => loomTree.nodes.get(id))
      .filter(Boolean)
      .filter((n) => !(n!.parent_id === null && n!.role === "system" && !n!.text));
    return visible[turnIdx]?.id ?? null;
  });

  // ---- per-layer × per-probe grid --------------------------------------

  /** Layer indices sorted ascending.  Numeric sort over the string-keyed
   * layer map — TypeScript leaves Object.keys typed as string[] but the
   * server emits zero-padded integers, so a Number() cast is safe. */
  const layerKeys = $derived<string[]>(
    token?.perLayerScores
      ? Object.keys(token.perLayerScores).sort(
          (a, b) => Number(a) - Number(b),
        )
      : [],
  );

  /** Probe names sorted alphabetically (case-insensitive).  Source the
   * union over every layer's probe row so a probe with sparse coverage
   * still gets a column.  Most layers carry the same set in practice
   * but we don't enforce it. */
  const probeKeys = $derived.by<string[]>(() => {
    const pls = token?.perLayerScores;
    if (!pls) return [];
    const seen = new Set<string>();
    for (const layer of Object.keys(pls)) {
      for (const probe of Object.keys(pls[layer] ?? {})) {
        seen.add(probe);
      }
    }
    return [...seen].sort((a, b) =>
      a.localeCompare(b, undefined, { sensitivity: "base" }),
    );
  });

  function cellValue(layer: string, probe: string): number | null {
    const pls = token?.perLayerScores;
    if (!pls) return null;
    const row = pls[layer];
    if (!row) return null;
    const v = row[probe];
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  }

  function cellTooltip(layer: string, probe: string): string {
    const v = cellValue(layer, probe);
    if (v === null) return `L${layer} · ${probe} · —`;
    const sign = v >= 0 ? "+" : "";
    return `L${layer} · ${probe} · ${sign}${v.toFixed(3)}`;
  }

  // ---- cell sizing (responsive-ish) -----------------------------------

  /** Cell pixel size.  Capped on the high end so wide grids don't push
   * the drawer beyond reasonable widths; floored on the low end so the
   * tints remain visible.  ~22px reads cleanly with the optional value
   * label off; we keep value labels off because the cell count gets
   * large enough that text becomes noise. */
  const CELL_SIZE = 22;

  // ---- logits tab (logit-pass) -----------------------------------------
  //
  // Layout: ranked table of ``top_alts`` with rank / token / logprob /
  // probability / Δ-from-rank-1.  The chosen token gets a ``*`` glyph
  // and a background tint when present in the alts.  Empty state
  // routes the user to the SamplingStrip ``alts`` toggle when capture
  // wasn't on.

  type Tab = "probes" | "logits" | "lens" | "sae";
  let tab: Tab = $state<Tab>("probes");

  const TAB_ITEMS: Array<{ value: Tab; label: string; color?: string; title: string }> = [
    { value: "probes", label: "probes", title: "Per-layer × per-probe readings" },
    { value: "logits", label: "logits", title: "Ranked top-K alternatives at this position" },
    {
      value: "sae",
      label: "sae",
      color: "var(--pillar-sae)",
      title: "Sparse feature activations at the resident SAE hook layer",
    },
    {
      value: "lens",
      label: "j-lens",
      color: "var(--pillar-lens)",
      title: "Workspace readout — what each layer was disposed to say",
    },
  ];

  /** Reset to probes tab when the CLICK target changes (scrubbing keeps
   *  the tab).  Drilldown stays on whatever tab the user had open within
   *  a single token view, but a fresh open should always show the
   *  default surface. */
  $effect(() => {
    void turnIdx;
    void paramTokenIdx;
    tab = "probes";
  });

  interface RankRow {
    rank: number;
    id: number;
    text: string;
    logprob: number;
    /** ``exp(logprob)`` — the post-sampler probability. */
    p: number;
    /** ``logprob - logprob_rank1`` — zero for rank 1. */
    delta: number;
    /** True iff this row's ``id`` matches the chosen token id. */
    chosen: boolean;
  }

  /** Build ranked rows from the token's ``top_alts``.  Server emits the
   *  list in descending-logprob order, so ``index + 1`` is the rank.
   *  Chosen-row identification falls back to text equality when
   *  ``token.tokenId`` is null (legacy / replayed shape). */
  const rankRows = $derived.by<RankRow[]>(() => {
    const alts = token?.topAlts;
    if (!alts || alts.length === 0) return [];
    const lp0 = alts[0]?.logprob ?? 0;
    return alts.map((a, i) => ({
      rank: i + 1,
      id: a.id,
      text: a.text,
      logprob: a.logprob,
      p: Math.exp(a.logprob),
      delta: a.logprob - lp0,
      chosen:
        token?.tokenId != null
          ? a.id === token.tokenId
          : a.text === token?.text,
    }));
  });

  /** Rank of the chosen token within the captured alts, or null when the
   *  chosen token didn't make the top-K cut (rare unless K is very small
   *  or the distribution is very flat). */
  const chosenRank = $derived<number | null>(
    rankRows.find((r) => r.chosen)?.rank ?? null,
  );

  /** Format a logprob for the column.  Always-sign so positive vs
   *  negative is unambiguous; "—" for null/NaN. */
  function fmtLogprob(v: number | null | undefined): string {
    if (v == null || !Number.isFinite(v)) return "—";
    return v.toFixed(3);
  }
  function fmtProb(p: number): string {
    if (!Number.isFinite(p)) return "—";
    if (p >= 0.001) return p.toFixed(4);
    return p.toExponential(2);
  }
  function fmtDelta(d: number, rank: number): string {
    if (rank === 1) return "—";
    if (!Number.isFinite(d)) return "—";
    return d.toFixed(3);
  }

  /** Flip the SamplingStrip's alts toggle from the empty state.  Sets
   *  ``return_top_k`` to the canonical 8 (Decision 1); takes effect on
   *  the next generation.  Doesn't backfill — the current token's alts
   *  weren't captured and there's no way to recover them. */
  function enableAlts(): void {
    if (samplingState.return_top_k === 0) {
      samplingState.return_top_k = 8;
    }
  }

  /** Logit fork — regenerate this node as a sibling with the clicked
   *  token swapped for ``row``'s alternative.  Unlike the old branch
   *  (which spliced a single token into otherwise-stale text), this
   *  replays the node's raw decode prefix and *resamples* the
   *  continuation conditioned on the swapped token, so everything
   *  downstream of the fork is coherent.  Works for thinking tokens
   *  too — the engine forces the thinking prefix and lets the model
   *  close the channel itself. */
  async function branchFromAlt(row: RankRow): Promise<void> {
    branchError = null;
    const nodeId = loomNodeId;
    if (!nodeId) {
      branchError = "no loom assistant node is available for this token";
      return;
    }
    if (token == null || token.rawIndex == null) {
      branchError =
        "this token has no raw-decode index; forking needs a node " +
        "generated in this session (legacy / replayed turns can't fork)";
      return;
    }
    branchingRank = row.rank;
    try {
      await sendFork(nodeId, token.rawIndex, row.id);
      closeDrawer();
    } catch (e) {
      branchError = e instanceof Error ? e.message : String(e);
    } finally {
      branchingRank = null;
    }
  }

  // ---- j-lens tab (workspace readout) -----------------------------------
  //
  // On-demand recompute, not stored stream data: the server rebuilds the
  // node's prompt render + raw decode prefix up to this token and reads
  // the per-layer J-lens top-k at the forward that produced it — so any
  // in-session token drills down, steered generations read their steered
  // workspace (recipe replay), and nothing is paid per-token at gen time.
  // Responses are cached per (node, raw_index, steered) while the drawer
  // lives; the steered toggle refetches the unsteered counterfactual.

  const LENS_TOP_K = 8;

  let lensData = $state<LensTokenReadoutJSON | null>(null);
  let lensLoading = $state(false);
  let lensError = $state<string | null>(null);
  /** Replay under the node's recipe steering (server default).  Flipping
   *  it off reads the unsteered counterfactual workspace of the same
   *  token stream.  Sticky across tokens within one drawer life. */
  let lensSteered = $state(true);
  const lensCache = new Map<string, LensTokenReadoutJSON>();

  const jlensFitted = $derived(sessionState.info?.jlens_fitted === true);
  const saeLoaded = $derived(sessionState.info?.sae_loaded === true);
  let saeData = $state<SaeTokenReadoutJSON | null>(null);
  let saeLoading = $state(false);
  let saeError = $state<string | null>(null);
  const saeCache = new Map<string, SaeTokenReadoutJSON>();

  $effect(() => {
    if (tab !== "sae" || !saeLoaded) return;
    const nodeId = loomNodeId;
    const rawIndex = token?.rawIndex;
    if (!nodeId || rawIndex == null) return;
    const key = `${nodeId}:${rawIndex}:${effectiveRawMode() ? 1 : 0}`;
    const hit = saeCache.get(key);
    if (hit) { saeData = hit; saeError = null; return; }
    saeLoading = true; saeError = null; saeData = null;
    apiSae.tokenReadout(nodeId, rawIndex, { raw: effectiveRawMode() })
      .then((data) => { saeCache.set(key, data); saeData = data; })
      .catch((error) => { saeError = error instanceof Error ? error.message : String(error); })
      .finally(() => { saeLoading = false; });
  });

  const lensKey = $derived.by<string | null>(() => {
    const nodeId = loomNodeId;
    const rawIndex = token?.rawIndex;
    const raw = effectiveRawMode();
    if (!nodeId || rawIndex == null) return null;
    return `${nodeId}:${rawIndex}:${lensSteered ? 1 : 0}:${raw ? 1 : 0}:workspace`;
  });

  $effect(() => {
    if (tab !== "lens" || !jlensFitted) return;
    const key = lensKey;
    const nodeId = loomNodeId;
    const rawIndex = token?.rawIndex;
    if (!key || !nodeId || rawIndex == null) return;
    const hit = lensCache.get(key);
    if (hit) {
      lensData = hit;
      lensError = null;
      return;
    }
    lensLoading = true;
    lensError = null;
    lensData = null;
    apiLens
      .tokenReadout(nodeId, rawIndex, {
        topK: LENS_TOP_K,
        steered: lensSteered,
        raw: effectiveRawMode(),
        layers: "workspace",
      })
      .then((d) => {
        lensCache.set(key, d);
        if (key === lensKey) lensData = d;
      })
      .catch((e) => {
        if (key !== lensKey) return;
        const detail =
          e instanceof ApiError &&
          typeof (e.body as { detail?: unknown } | null)?.detail === "string"
            ? (e.body as { detail: string }).detail
            : null;
        lensError = detail ?? (e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (key === lensKey) lensLoading = false;
      });
  });

  /** Whether to offer the steered/unsteered toggle: the node actually has
   *  recipe steering (the steered fetch reported an expression), or the
   *  user already flipped to unsteered and needs the way back. */
  const lensHasSteering = $derived(
    (lensData?.steering ?? null) !== null || !lensSteered,
  );

  function lensCellStyle(logprob: number): string {
    const p = Math.min(1, Math.exp(logprob));
    const pct = Math.round(p * 60);
    return `background: color-mix(in srgb, var(--pillar-lens) ${pct}%, transparent);`;
  }

  function lensCellTitle(layer: number, t: { token: string; logprob: number }): string {
    const p = Math.exp(t.logprob);
    const pTxt = p >= 0.001 ? p.toFixed(4) : p.toExponential(2);
    return `L${layer} · ${JSON.stringify(t.token)} · p=${pTxt} · logprob=${t.logprob.toFixed(3)}`;
  }

  function lensCellText(t: { token: string }): string {
    const trimmed = t.token.trim();
    return trimmed.length > 0 ? trimmed : JSON.stringify(t.token);
  }

  // ---- drawer chrome ---------------------------------------------------

  function onClose(): void {
    closeDrawer();
  }

  // ---- token scrubber ----------------------------------------------------
  //
  // ◀ ▶ in the header (or ←/→ anywhere in the drawer) walk the inspected
  // position along the turn's token list.  Every tab follows — the probes
  // heatmap and logits table read stream-captured data (instant), the
  // j-lens tab refetches per position (cached per (node, raw_index,
  // steered), so a revisit is instant).  A fresh token click snaps back.

  function scrubTo(i: number): void {
    if (i < 0 || i >= tokenList.length) return;
    scrubTokenIdx = i === paramTokenIdx ? null : i;
  }

  const canScrubBack = $derived(tokenIdx > 0);
  const canScrubFwd = $derived(tokenIdx < tokenList.length - 1);

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      onClose();
      return;
    }
    // Arrow scrubbing — but never steal arrows from a focusable field
    // (the chat input lives outside the drawer and must keep caret keys).
    const t = ev.target as HTMLElement | null;
    if (
      t &&
      (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)
    ) {
      return;
    }
    if (ev.key === "ArrowLeft" && canScrubBack) {
      ev.preventDefault();
      scrubTo(tokenIdx - 1);
    } else if (ev.key === "ArrowRight" && canScrubFwd) {
      ev.preventDefault();
      scrubTo(tokenIdx + 1);
    }
  }

  const hasAbPair = $derived(turn?.abPair != null);
  const isEmpty = $derived(layerKeys.length === 0 || probeKeys.length === 0);
</script>

<svelte:window onkeydown={onKeydown} />

<aside
  class="drawer"
  aria-label="Token drilldown"
>
  <header class="drawer-header">
    <div class="title">
      <span class="eyebrow">token drilldown</span>
      {#if token}
        <div class="name-row">
          <code class="tok-text">{JSON.stringify(token.text)}</code>
          <span class="coord">
            turn {turnIdx} · {isThinking ? "thinking" : "response"}
          </span>
          <span class="scrub" title="Walk the inspected token along this turn (← / →); every tab follows">
            <button
              type="button"
              class="scrub-btn"
              disabled={!canScrubBack}
              onclick={() => scrubTo(tokenIdx - 1)}
              aria-label="Previous token"
            >◀</button>
            <span class="scrub-pos">{tokenIdx + 1} / {tokenList.length}</span>
            <button
              type="button"
              class="scrub-btn"
              disabled={!canScrubFwd}
              onclick={() => scrubTo(tokenIdx + 1)}
              aria-label="Next token"
            >▶</button>
            {#if scrubTokenIdx !== null}
              <button
                type="button"
                class="scrub-btn scrub-home"
                onclick={() => (scrubTokenIdx = null)}
                title="Snap back to the clicked token"
              >↩ clicked</button>
            {/if}
          </span>
        </div>
      {:else}
        <div class="name-row">
          <span class="coord">no token at ({turnIdx}, {tokenIdx})</span>
        </div>
      {/if}
    </div>
    <button type="button" class="close" onclick={onClose} aria-label="Close drawer">
      ✕
    </button>
  </header>

  <!-- View tabs (probes / logits / j-lens) + the steered/unsteered branch
       toggle when this turn has an A/B pair.  Tabs always render so users
       see the other views exist even when their capture is off. -->
  <div class="toolbar">
    <SegmentedTabs items={TAB_ITEMS} bind:value={tab} ariaLabel="Token detail view" />
    {#if hasAbPair}
      <SegmentedTabs items={BRANCH_ITEMS} bind:value={branch} ariaLabel="Token branch" />
    {/if}
  </div>

  <div class="body">
    {#if !token}
      <div class="empty">
        Token not found.  The chat log may have been cleared or rewound
        since the drawer was opened.
      </div>
    {:else if tab === "probes"}
      {#if isEmpty}
        <div class="empty">
          No per-layer scores captured for this token (probes not loaded?).
        </div>
      {:else}
        <div class="grid-scroll probe-grid">
          <table class="grid" style="--cell: {CELL_SIZE}px;">
            <thead>
              <tr>
                <th class="corner" scope="col">L \ probe</th>
                {#each probeKeys as probe (probe)}
                  <th class="col-label" scope="col" title={probe}>
                    <span>{probe}</span>
                  </th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each layerKeys as layer (layer)}
                <tr>
                  <th class="row-label" scope="row" title={`Layer ${layer}`}>
                    L{layer}
                  </th>
                  {#each probeKeys as probe (probe)}
                    {@const v = cellValue(layer, probe)}
                    <td class="cell-td">
                      <HeatmapCell
                        value={v}
                        scale={probeAxisScale(probe)}
                        size={CELL_SIZE}
                        title={cellTooltip(layer, probe)}
                      />
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    {:else if tab === "logits"}
      <!-- Logits tab.  Three states: ranked rows present, alts captured
           but empty (degenerate / stop token), nothing captured at all. -->
      {#if rankRows.length > 0}
        <div class="tab-summary">
          <div>
            chosen: <code class="tok-inline">{JSON.stringify(token.text)}</code>
            <span class="kv">
              logprob = <strong>{fmtLogprob(token.logprob)}</strong>
            </span>
            <span class="kv">
              {#if chosenRank !== null}
                (rank {chosenRank} of {rankRows.length})
              {:else}
                (chosen token not in top-{rankRows.length})
              {/if}
            </span>
          </div>
        </div>
        <div class="grid-scroll">
          <table class="logits-table">
            <thead>
              <tr>
                <th class="num">rank</th>
                <th class="tok">token</th>
                <th class="num">logprob</th>
                <th class="num">p</th>
                <th class="num">Δ from rank 1</th>
                <th class="num">branch</th>
              </tr>
            </thead>
            <tbody>
              {#each rankRows as row (row.rank)}
                <tr class:chosen={row.chosen}>
                  <td class="num">
                    {row.chosen ? "* " : ""}{row.rank}
                  </td>
                  <td class="tok">
                    <code>{JSON.stringify(row.text)}</code>
                  </td>
                  <td class="num">{fmtLogprob(row.logprob)}</td>
                  <td class="num">{fmtProb(row.p)}</td>
                  <td class="num">{fmtDelta(row.delta, row.rank)}</td>
                  <td class="num">
                    <Button
                      size="sm"
                      disabled={row.chosen || branchingRank !== null}
                      onclick={() => branchFromAlt(row)}
                      title="Fork a sibling branch: swap in this token and resample the continuation"
                    >
                      {branchingRank === row.rank ? "…" : "fork"}
                    </Button>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
        {#if branchError}
          <p class="branch-error">{branchError}</p>
        {/if}
      {:else if token.logprob != null}
        <div class="empty">
          <p>
            Chosen-token logprob captured:
            <strong>{fmtLogprob(token.logprob)}</strong>, but no top-K
            alternatives were requested for this generation.
          </p>
          <p>
            <Button
              onclick={enableAlts}
              disabled={samplingState.return_top_k > 0}
            >
              {samplingState.return_top_k > 0
                ? "alts on (effective next gen)"
                : "enable alts (next generation)"}
            </Button>
          </p>
        </div>
      {:else}
        <div class="empty">
          <p>
            No logprob data captured for this token. Either it was replayed
            from a transcript that pre-dates logit capture, or capture wasn't
            live for the run.
          </p>
          <p>
            <Button
              onclick={enableAlts}
              disabled={samplingState.return_top_k > 0}
            >
              {samplingState.return_top_k > 0
                ? "alts on (effective next gen)"
                : "enable alts (next generation)"}
            </Button>
          </p>
        </div>
      {/if}
    {:else if tab === "lens"}
      <!-- J-lens tab.  The workspace readout matrix — rows are lens
           layers (ascending; workspace-band rows marked), cells the
           top-K vocabulary tokens each layer's residual was disposed to
           say at the forward that produced this token.  Recomputed
           on demand server-side (node prompt render + raw prefix replay),
           so it works for any in-session token, no per-token gen cost. -->
      {#if !jlensFitted}
        <div class="empty">
          <p>
            No Jacobian lens is fitted for
            <code>{sessionState.info?.model_id ?? "this model"}</code>.
          </p>
          <p>
            Fit one from a terminal —
            <code>saklas lens fit {sessionState.info?.model_id ?? "<model>"}</code>
            — then reopen this drawer.
          </p>
        </div>
      {:else if token.rawIndex == null}
        <div class="empty">
          This token has no raw-decode index — the readout needs a node
          generated in this session (legacy / replayed turns can't be
          re-read, same constraint as forking).
        </div>
      {:else if !loomNodeId}
        <div class="empty">
          No loom assistant node is available for this token.
        </div>
      {:else if lensLoading}
        <div class="empty">
          computing workspace readout (one prefix forward)…
        </div>
      {:else if lensError}
        <div class="empty">
          <p>Workspace readout failed: {lensError}</p>
        </div>
      {:else if lensData}
        <div class="tab-summary">
          <div>
            produced: <code class="tok-inline">{JSON.stringify(lensData.token_text)}</code>
            {#if lensData.steering !== null}
              <span class="kv steer-chip" title="The replay ran under the node's recipe steering">
                steered: <code>{lensData.steering}</code>
              </span>
            {:else if !lensSteered}
              <span class="kv">unsteered counterfactual</span>
            {/if}
            {#if lensHasSteering}
              <label class="kv steer-toggle" title="Replay under the node's recipe steering vs the unsteered counterfactual of the same token stream">
                <input type="checkbox" bind:checked={lensSteered} />
                apply recipe steering
              </label>
            {/if}
          </div>
        </div>
        {#if (lensData.aggregate ?? []).length > 0}
          <!-- Layer-aggregated view of the same logits (band-restricted):
               strength = mean band probability, com = probability-mass-
               weighted depth center of mass (0 = first block, 1 = last). -->
          <div class="lens-agg" role="list" aria-label="Aggregate lens tokens">
            <span class="lens-agg-label">aggregate</span>
            {#each lensData.aggregate ?? [] as chip, i (i)}
              <span
                class="lens-agg-chip"
                role="listitem"
                title={`"${chip.token}" — strength ${chip.strength.toFixed(3)} · com ${chip.com.toFixed(2)} ±${chip.spread.toFixed(2)}`}
              >
                <span class="lens-agg-tok">{chip.token.trim() || JSON.stringify(chip.token)}</span>
                <span class="lens-agg-com">@{chip.com.toFixed(2)}</span>
              </span>
            {/each}
          </div>
        {/if}
        <div class="grid-scroll">
          <table class="lens-table">
            <thead>
              <tr>
                <th class="corner">L \ rank</th>
                {#each { length: LENS_TOP_K } as _, i (i)}
                  <th class="num">{i + 1}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each lensData.layers as row (row.layer)}
                <tr class:off-band={!row.in_band}>
                  <th
                    class="row-label"
                    class:band={row.in_band}
                    title={row.in_band
                      ? `Layer ${row.layer} — workspace band (40–90% depth)`
                      : `Layer ${row.layer} — outside the workspace band`}
                  >
                    L{row.layer}
                  </th>
                  {#each row.tokens as cell (cell.id)}
                    <td
                      class="lens-cell"
                      class:hit={cell.id === lensData.token_id}
                      style={lensCellStyle(cell.logprob)}
                      title={lensCellTitle(row.layer, cell)}
                    >
                      {lensCellText(cell)}
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    {:else}
      {#if !saeLoaded}
        <div class="empty">No SAE is loaded for this session. Load one from the SAE inspector tab.</div>
      {:else if token.rawIndex == null || !loomNodeId}
        <div class="empty">This token has no in-session raw decode record for SAE replay.</div>
      {:else if saeLoading}
        <div class="empty">computing SAE readout (one prefix forward)…</div>
      {:else if saeError}
        <div class="empty">SAE readout failed: {saeError}</div>
      {:else if saeData}
        <div class="tab-summary">resident SAE · L{saeData.layer} · produced <code>{JSON.stringify(saeData.token_text)}</code></div>
        <div class="grid-scroll">
          <table class="logits-table">
            <!-- strength = activation / Neuronpedia maxActApprox — the
                 normalized 0..1 unit the probe cards + gates read; "—"
                 when no metadata is cached for the feature. -->
            <thead><tr><th class="num">rank</th><th class="tok">feature</th><th class="tok">label</th><th class="num">strength</th><th class="num">activation</th></tr></thead>
            <tbody>
              {#each saeData.features as feature, index (feature.id)}
                <tr>
                  <td class="num">{index + 1}</td>
                  <td class="tok"><code>sae/{feature.id}</code></td>
                  <td class="tok">{feature.label ?? "—"}</td>
                  <td class="num">{feature.max_act != null && feature.max_act > 0 ? (feature.activation / feature.max_act).toFixed(2) : "—"}</td>
                  <td class="num">{feature.activation.toFixed(3)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    {/if}
  </div>

  <footer class="drawer-footer">
    <span class="hint">
      {#if tab === "probes"}
        Tints map each probe's coordinate to its node extent (full color at
        the most extreme node), clamped to ±1. Green = +pole, red = −pole,
        transparent ≈ 0.
      {:else if tab === "logits"}
        Logprob is the chosen-token natural-log probability under the
        post-temperature / post-top-p / post-top-k distribution the sampler
        drew from.
      {:else if tab === "lens"}
        Each row ranks softmax(W_U · norm(J_l·h)) at the forward that
        produced this token — what that layer's residual was disposed to
        make the model say. Cell tint = probability; blue row labels mark
        the 40–90% workspace band (early rows are noise, late rows converge
        on the raw logits). Highlighted cells match the produced token.
      {:else}
        Post-nonlinearity sparse-feature activations from the resident SAE's
        hook layer. Values are comparable across tokens for one feature, not
        calibrated across different features.
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
    padding: var(--space-5) var(--space-6);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
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
    gap: var(--space-4);
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
  .close {
    background: var(--glass);
    color: var(--fg-muted);
    border: 1px solid transparent;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font: inherit;
    font-size: var(--text-md);
    line-height: 1;
    cursor: pointer;
    flex: none;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .close:hover {
    color: var(--fg);
    background: var(--glass-strong);
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
  .empty code {
    font-family: var(--font-mono);
    color: var(--fg-dim);
  }
  .empty strong {
    color: var(--fg-strong);
  }

  /* Data wells — tables recess into a deeper glass window.  Sticky
   * label cells must stay OPAQUE (they occlude scrolled cells), so they
   * paint the well tone rather than glass. */
  .grid-scroll {
    overflow: auto;
    max-height: 100%;
    border-radius: var(--radius);
    background: var(--bg);
  }
  /* A narrow probe roster should not paint a full-drawer black well around
     a five-column heatmap. Hug the matrix and center it; once the roster is
     wider than the drawer, the same wrapper becomes the horizontal scroller. */
  .probe-grid {
    width: max-content;
    max-width: 100%;
    margin-inline: auto;
  }
  .grid {
    border-collapse: separate;
    border-spacing: 1px;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }
  .grid th,
  .grid td {
    padding: 0;
    margin: 0;
    background: var(--bg);
  }
  /* Sticky row + column labels so orientation survives long scrolls. */
  .grid thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    box-shadow: var(--shadow-sticky);
  }
  .grid .row-label {
    position: sticky;
    left: 0;
    z-index: 1;
    text-align: right;
    padding: 0 var(--space-3) 0 var(--space-2);
    color: var(--fg-dim);
    font-size: var(--text-xs);
    font-weight: var(--weight-normal);
    box-shadow: 2px 0 8px rgba(0, 0, 0, 0.45);
    white-space: nowrap;
  }
  .grid .corner {
    position: sticky;
    top: 0;
    left: 0;
    z-index: 3;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-normal);
    text-align: left;
    padding: var(--space-1) var(--space-3);
    box-shadow: var(--shadow-sticky), 2px 0 8px rgba(0, 0, 0, 0.45);
  }
  .grid .col-label {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    font-weight: var(--weight-normal);
    padding: 0;
    /* Rotate compact column labels so they fit narrow cells.  Wrap the
     * inner span so the rotation pivots around the cell box, not the
     * letterform. */
    height: 6em;
    vertical-align: bottom;
    width: var(--cell);
    min-width: var(--cell);
    max-width: var(--cell);
  }
  .grid .col-label > span {
    display: inline-block;
    transform: rotate(-60deg);
    transform-origin: left bottom;
    white-space: nowrap;
    padding-bottom: var(--space-2);
  }
  .grid .cell-td {
    line-height: 0; /* HeatmapCell brings its own box; remove text leading. */
  }

  /* Logits tab — chosen-row summary line and the ranked alts table. */
  .tab-summary {
    padding: 0 0 var(--space-4) 0;
    color: var(--fg);
    font-size: var(--text-sm);
    line-height: 1.6;
  }
  .tab-summary .kv {
    color: var(--fg-dim);
    margin-left: var(--space-4);
  }
  .tab-summary strong {
    color: var(--fg-strong);
    font-family: var(--font-mono);
  }
  .tok-inline {
    color: var(--fg);
    font-family: var(--font-mono);
    background: var(--glass-strong);
    border-radius: var(--radius-sm);
    padding: 0 var(--space-2);
  }
  .logits-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
    font-size: var(--text-sm);
  }
  .logits-table th,
  .logits-table td {
    padding: var(--space-2) var(--space-4);
    text-align: left;
    background: var(--bg);
  }
  .logits-table thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    color: var(--fg-muted);
    font-family: var(--font-ui);
    font-weight: var(--weight-medium);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    box-shadow: var(--shadow-sticky);
  }
  .logits-table td {
    font-family: var(--font-mono);
  }
  .logits-table td.num,
  .logits-table th.num {
    text-align: right;
    width: 1%;
    white-space: nowrap;
  }
  .logits-table td.tok code {
    color: var(--fg-strong);
    background: transparent;
    word-break: break-all;
  }
  .logits-table tbody tr:hover td {
    background: color-mix(in srgb, var(--bg-hover) 60%, var(--bg));
  }
  /* Chosen row gets a soft accent wash + a heavier color so it reads at
     a glance. */
  .logits-table tr.chosen td,
  .logits-table tbody tr.chosen:hover td {
    background: var(--accent-subtle);
    color: var(--fg-strong);
  }
  /* J-lens tab — the layer-aggregated chip row above the matrix. */
  .lens-agg {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-2);
    margin-bottom: var(--space-4);
  }
  .lens-agg-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-right: var(--space-2);
  }
  .lens-agg-chip {
    display: inline-flex;
    align-items: baseline;
    gap: var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    background: color-mix(in srgb, var(--pillar-lens) 16%, var(--glass));
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    padding: 1px var(--space-4);
  }
  .lens-agg-tok {
    color: var(--pillar-lens);
  }
  .lens-agg-com {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
  }
  /* J-lens tab — the workspace readout matrix.  Same well chrome as the
     logits table; cells carry an inline probability tint in the lens
     family blue, so the static styles stay layout-only. */
  .lens-table {
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
    font-size: var(--text-sm);
  }
  .lens-table th,
  .lens-table td {
    padding: var(--space-1) var(--space-3);
    text-align: left;
    background: var(--bg);
  }
  .lens-table thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    color: var(--fg-muted);
    font-family: var(--font-ui);
    font-weight: var(--weight-medium);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    box-shadow: var(--shadow-sticky);
  }
  .lens-table .corner {
    position: sticky;
    left: 0;
    z-index: 3;
    box-shadow: var(--shadow-sticky), 2px 0 8px rgba(0, 0, 0, 0.45);
  }
  .lens-table .row-label {
    position: sticky;
    left: 0;
    z-index: 1;
    text-align: right;
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-weight: var(--weight-normal);
    font-size: var(--text-xs);
    box-shadow: 2px 0 8px rgba(0, 0, 0, 0.45);
    white-space: nowrap;
  }
  .lens-table .row-label.band {
    color: var(--pillar-lens);
  }
  .lens-table tr.off-band td {
    opacity: 0.55;
  }
  .lens-cell {
    font-family: var(--font-mono);
    color: var(--fg-strong);
    white-space: nowrap;
    max-width: 12ch;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .lens-cell.hit {
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }
  .steer-chip code {
    color: var(--fg-strong);
    background: transparent;
    font-family: var(--font-mono);
  }
  .steer-toggle {
    cursor: pointer;
    user-select: none;
  }
  .steer-toggle input {
    accent-color: var(--accent);
    vertical-align: middle;
    margin-right: var(--space-1);
  }

  .branch-error {
    color: var(--accent-error);
    font-size: var(--text-sm);
    margin: var(--space-4) 0 0;
  }

  .drawer-footer {
    padding: var(--space-3) var(--space-6);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .hint {
    line-height: 1.5;
  }
</style>
