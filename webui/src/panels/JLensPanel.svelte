<script lang="ts">
  // J-LENS — the inspector column's Jacobian-lens tab: two sections,
  // card-based and symmetric with the CAA tab (every row wears RackCard;
  // the j-lens family accent is blue, marker ■/□).
  //
  //   STEER — one card per ``α jlens/<word>`` token atom in the ONE
  //           steering expression (the engine folds the lens direction
  //           over all fitted layers with whitened shares, exactly
  //           like a concept vector).  Per-card α slider + trigger
  //           pill (lens atoms run hotter than concept vectors;
  //           default 0.3).
  //   PROBE — the J-lens readout, pinned and unpinned in ONE section
  //           (pinning a token just makes its card persistent and
  //           gate-able — both card kinds are the same shape: strength
  //           bar + per-layer strength strip; the pinned card's ■ unpins,
  //           the unpinned card's □ pins).  Pinned ``jlens/<word>`` token
  //           probes first, then the live open-vocab aggregate cards for
  //           the top-k tokens not already pinned.  The card list owns
  //           the scroll (header + add form stay anchored, like the CAA
  //           racks).  The header's live toggle is the lens live switch:
  //           off ⇒ no per-step lens computation — pinned probes settle
  //           to the end-of-gen aggregate, discovery cards go quiet.  The
  //           full per-layer ranking lives in the transcript drilldown.
  //
  // SOURCE is the shared lifecycle shell: one selector uses or fetches an
  // artifact, followed by the same labelled custom row as SAE. Successful
  // preparation activates the source and live readout.

  import Bar from "../lib/charts/Bar.svelte";
  import Button from "../lib/ui/Button.svelte";
  import { onMount } from "svelte";
  import InstrumentSourceSection from "./rack/InstrumentSourceSection.svelte";
  import RackSectionHeader from "./rack/RackSectionHeader.svelte";
  import JLensProbeCard from "./rack/JLensProbeCard.svelte";
  import { mergeInstrumentProbeRows } from "./rack/probeRows";
  import type { InstrumentProbeRow } from "./rack/probeRows";
  import AtomSteerCard from "./rack/AtomSteerCard.svelte";
  import { apiInstruments, describeError } from "../lib/api";
  import {
    addJLensToRack,
    activeProbeNames,
    attachProbe,
    lensFetch,
    lensFit,
    lensSourceState,
    lensState,
    lensAggregateForDisplay,
    lensReadoutForDisplay,
    probeRack,
    probeEntryForDisplay,
    refreshLensSources,
    seedProbeDisplay,
    sessionState,
    setLensWorkspaceSortMode,
    setLiveLens,
    steerRack,
    tokenHoverState,
    useLensSource,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { JLensSteerEntry, ProbeRackEntry } from "../lib/types";
  import type { LensWorkspaceSortMode } from "../lib/stores.svelte";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const liveOn = $derived(lensState.layers !== null);
  const displayReadout = $derived(lensReadoutForDisplay());
  const displayAggregate = $derived(lensAggregateForDisplay());
  const displayLayers = $derived.by(() => {
    if (!tokenHoverState.active) return lensState.layers ?? [];
    return Object.keys(displayReadout ?? {}).map(Number).sort((a, b) => a - b);
  });
  const sourceBusy = $derived(
    lensSourceState.loading || lensSourceState.busy ||
      lensFetch.state.running || lensFit.state.running,
  );
  const LENS_PROVIDER_OPTIONS = [
    { value: "neuronpedia", label: "neuronpedia" },
  ];
  let fitPrompts = $state(100);
  let fitLayers = $state("all");
  let fitConfirm = $state(false);
  let selectedSource = $state("");
  const fitReady = $derived(
    Number.isInteger(fitPrompts) && fitPrompts >= 1 && fitPrompts <= 5000 &&
      fitLayers.trim().length > 0,
  );
  const fitIsPreparing = $derived(
    (lensFit.state.message ?? "").startsWith("streaming "),
  );

  function requestFit(): void {
    if (!fitConfirm) {
      fitConfirm = true;
      return;
    }
    fitConfirm = false;
    void lensFit.start({
      prompts: fitPrompts,
      layers: fitLayers.trim(),
    });
  }

  // Resume-visibility: a page reload mid-fit should pick the progress
  // polling back up (the fit runs server-side regardless of the client).
  onMount(() => {
    void lensFit.check();
    void lensFetch.check();
    void refreshLensSources();
  });

  $effect(() => {
    if (selectedSource !== "local") fitConfirm = false;
  });

  // ---------- STEER: jlens-mode rack entries (alphabetical) ----------
  const steerCards = $derived.by(() => {
    const arr = [...steerRack.entries.entries()].filter(
      (kv): kv is [string, JLensSteerEntry] => kv[1].mode === "jlens",
    );
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });

  let steerInput = $state("");
  let steerBusy = $state(false);

  function bareWord(value: string): string {
    return value.trim().replace(/^jlens\//, "");
  }

  async function onAddSteer(ev: SubmitEvent): Promise<void> {
    ev.preventDefault();
    const submitted = steerInput;
    const word = bareWord(submitted);
    if (!word || steerBusy) return;
    steerBusy = true;
    try {
      const validated = await apiInstruments.validateLensToken(word);
      addJLensToRack(validated.word);
      if (steerInput === submitted) steerInput = "";
    } catch (e) {
      pushToast(`steer jlens/${word} failed — ${describeError(e)}`, {
        kind: "error",
      });
    } finally {
      steerBusy = false;
    }
  }

  // ---------- PROBE: pinned probe cards + unpinned aggregate cards ----------
  // One merged section — pinning a workspace token just makes its card
  // persistent (and gate-able); both card families share the one sort
  // control (strength / name / depth).

  /** One card's props plus the shared sort keys — pinned and discovery
   *  rows produce the identical shape, so the roster is one list of one
   *  card. */
  interface WorkspaceCard extends InstrumentProbeRow {
    token: string;
    strength: number;
    com: number | null;
    spread: number | null;
    series: number[];
    cells: { layer: number; p: number | null }[];
    pinned: boolean;
  }

  const SORT_OPTIONS: {
    value: LensWorkspaceSortMode;
    label: string;
  }[] = [
    { value: "strength", label: "strength" },
    { value: "name", label: "name" },
    { value: "depth", label: "depth" },
  ];

  /** Per-layer cells for a pinned probe — the store's axis-0 per-layer map. */
  function pinnedCells(
    entry: ProbeRackEntry,
  ): { layer: number; p: number | null }[] {
    const perLayer = entry.perLayer;
    if (!perLayer) return [];
    return Object.keys(perLayer)
      .sort((a, b) => Number(a) - Number(b))
      .map((layer) => ({ layer: Number(layer), p: perLayer[layer] ?? null }));
  }

  /** Per-layer cells for a discovery token — its softmax probability in
   *  each streamed readout row; ``null`` = below that layer's top-k. */
  function readoutCells(token: string): { layer: number; p: number | null }[] {
    const trimmed = token.trim() || JSON.stringify(token);
    return displayLayers.map((layer) => {
      const pairs = displayReadout?.[String(layer)];
      if (!pairs || pairs.length === 0) return { layer, p: null };
      const hit =
        pairs.find(([text]) => text === token) ??
        pairs.find(([text]) => text.trim() === trimmed);
      return { layer, p: hit ? hit[1] : null };
    });
  }

  const pinnedCards = $derived.by((): WorkspaceCard[] => {
    const rows: WorkspaceCard[] = [];
    for (const name of activeProbeNames()) {
      if (!name.startsWith("jlens/")) continue;
      const entry = probeEntryForDisplay(name);
      if (!entry) continue;
      const latest = entry.aggregate ?? entry.reading;
      const word = name.slice("jlens/".length);
      rows.push({
        key: name,
        sortName: word,
        token: word,
        strength: latest?.coords?.[0] ?? entry.current ?? 0,
        com: latest?.depth_com?.[0] ?? null,
        spread: latest?.depth_spread?.[0] ?? null,
        series: entry.sparkline ?? [],
        cells: pinnedCells(entry),
        pinned: true,
      });
    }
    return rows;
  });

  const aggRows = $derived.by((): WorkspaceCard[] => {
    const rows = displayAggregate;
    if (!rows || rows.length === 0) return [];
    const hist = tokenHoverState.active ? [] : lensState.aggHistory;
    // Pinned tokens already have a persistent card — the aggregate group
    // carries only the unpinned remainder of the top-k.
    return rows
      .filter(([token]) => !probeRack.active.includes(`jlens/${token.trim()}`))
      .map(([token, strength, com, spread]) => ({
        key: `aggregate:${token}`,
        sortName: token.trim(),
        token,
        strength,
        com,
        spread,
        series: tokenHoverState.active
          ? [strength]
          : hist.map((frame) => frame.find(([t]) => t === token)?.[1] ?? 0),
        cells: readoutCells(token),
        pinned: false,
      }));
  });

  /** Pinned and discovered tokens are one visual roster. Persistence is an
   *  action/state difference, not a hidden first sort key. */
  const workspaceCards = $derived(
    mergeInstrumentProbeRows(
      pinnedCards,
      liveOn || tokenHoverState.active ? aggRows : [],
      lensState.workspaceSortMode,
    ),
  );

  let probeInput = $state("");
  let probeBusy = $state(false);

  async function pinWord(word: string): Promise<boolean> {
    const bare = bareWord(word);
    if (!bare || probeBusy) return false;
    const selector = `jlens/${bare}`;
    if (probeRack.active.includes(selector)) return true;
    probeBusy = true;
    try {
      const validated = await apiInstruments.validateLensToken(bare);
      const validatedSelector = `jlens/${validated.word}`;
      const live = lensState.aggregate?.find(
        ([token]) => token.trim() === validated.word,
      );
      await attachProbe(validatedSelector);
      if (live) {
        const [token, strength, com, spread] = live;
        const perLayer: Record<string, number> = {};
        const coordsPerLayer: Record<string, number[]> = {};
        for (const layer of lensState.layers ?? []) {
          const pairs = lensState.readout?.[String(layer)] ?? [];
          const hit = pairs.find(([text]) => text === token) ??
            pairs.find(([text]) => text.trim() === validated.word);
          const value = hit?.[1] ?? 0;
          perLayer[String(layer)] = value;
          coordsPerLayer[String(layer)] = [value];
        }
        const reading = {
          fraction: 0,
          nearest: [] as [string, number][],
          coords: [strength],
          residual: 0,
          fraction_per_layer: {},
          coords_per_layer: coordsPerLayer,
          residual_per_layer: {},
          depth_com: [com],
          depth_spread: [spread],
        };
        const series = lensState.aggHistory.map(
          (frame) => frame.find(([text]) => text === token)?.[1] ?? 0,
        );
        seedProbeDisplay(validatedSelector, {
          current: strength,
          sparkline: series,
          perLayer,
          reading,
          aggregate: reading,
        });
      }
      pushToast(`pinned ${validatedSelector}`, { kind: "info" });
      return true;
    } catch (e) {
      pushToast(`pin ${selector} failed — ${describeError(e)}`, {
        kind: "error",
      });
      return false;
    } finally {
      probeBusy = false;
    }
  }

  async function onAddProbe(ev: SubmitEvent): Promise<void> {
    ev.preventDefault();
    const submitted = probeInput;
    if (await pinWord(submitted)) {
      if (probeInput === submitted) probeInput = "";
    }
  }

  function onToggleLive(): void {
    void setLiveLens(!liveOn);
  }
</script>

<div class="jlens" aria-label="Jacobian-lens inspector">
  <InstrumentSourceSection
    ready={fitted}
    sources={lensSourceState.sources}
    bind:value={selectedSource}
    busy={sourceBusy}
    accent="var(--pillar-lens)"
    sourceError={lensSourceState.error}
    working={lensFetch.state.running || lensFit.state.running}
    onuse={(source) => void useLensSource(source)}
    providerOptions={LENS_PROVIDER_OPTIONS}
    providerPlaceholder="lens provider"
    onfetch={(source) => void lensFetch.start({ source })}
    localActionLabel={fitConfirm ? "confirm fit" : "fit"}
    localActionDisabled={sourceBusy || !fitReady}
    onlocal={requestFit}
  >
    {#snippet localControls()}
      <label class="setup-field setup-field-medium">
        <span class="setup-field-label">prompts</span>
        <input
          class="add-input"
          type="number"
          min="1"
          max="5000"
          step="25"
          bind:value={fitPrompts}
          placeholder="100"
          aria-label="J-lens corpus prompts"
          title="1–5000"
        />
      </label>
      <label class="setup-field setup-field-wide">
        <span class="setup-field-label">layers</span>
        <input
          class="add-input"
          bind:value={fitLayers}
          placeholder="workspace | all | 13,14,…"
          aria-label="J-lens source layers"
          title="workspace, all, or layer ids"
        />
      </label>
    {/snippet}
    {#snippet progress()}
      {#if lensFetch.state.running}
        <p class="work-status" role="status" aria-live="polite">
          {lensFetch.state.message ?? "fetching official lens…"}
        </p>
      {:else}
        <div
          class="fit-progress"
          role="status"
          aria-live="polite"
          aria-label="Lens fit progress"
        >
          <div class="fit-line">
            <span class="fit-msg">{lensFit.state.message ?? "fitting…"}</span>
            {#if lensFit.state.total > 0}
              <span class="fit-count">
                {lensFit.state.current}/{lensFit.state.total}
              </span>
            {/if}
          </div>
          <div
            class="fit-bar"
            role="progressbar"
            aria-label="J-lens prompts fitted"
            aria-valuemin="0"
            aria-valuemax={Math.max(lensFit.state.total, 1)}
            aria-valuenow={lensFit.state.current}
          >
            <Bar
              value={lensFit.state.current}
              max={Math.max(lensFit.state.total, 1)}
              width={160}
              height={8}
              color="var(--pillar-lens)"
            />
          </div>
          <p class="hint">
            {#if lensFit.state.cancelling}
              stopping background work…
            {:else if fitIsPreparing}
              generation available during corpus setup
            {:else}
              generation paused during model fitting
            {/if}
          </p>
          <Button
            size="sm"
            variant="danger"
            disabled={lensFit.state.cancelling}
            onclick={() => void lensFit.cancel()}
          >
            {lensFit.state.cancelling ? "cancelling…" : "cancel"}
          </Button>
        </div>
      {/if}
    {/snippet}
    {#snippet warning()}
      {#if fitConfirm && !lensFit.state.running}
        <p class="hint fit-warning" role="alert">
          Blocks generation; may take hours. Confirm again.
        </p>
      {/if}
    {/snippet}
    {#snippet messages()}
      {#if lensFit.state.error}
        <p class="hint fit-error" role="alert">local fit: {lensFit.state.error}</p>
      {/if}
      {#if lensFetch.state.error}
        <p class="hint fit-error" role="alert">official fetch: {lensFetch.state.error}</p>
      {/if}
    {/snippet}
  </InstrumentSourceSection>

  {#if fitted}
    <!-- STEER — token-atom cards in the shared steering expression. -->
    <section class="section steer">
      <RackSectionHeader
        title="STEER"
        count={`${steerCards.length} term${steerCards.length === 1 ? "" : "s"}`}
      />

      {#if steerCards.length > 0}
        <div class="cards steer-cards" role="list">
          {#each steerCards as [name, entry] (name)}
            <div role="listitem">
              <AtomSteerCard mode="jlens" {name} {entry} />
            </div>
          {/each}
        </div>
      {/if}

      <form class="add-form" onsubmit={onAddSteer}>
        <input
          class="add-input"
          type="text"
          placeholder="word (single token)"
          bind:value={steerInput}
          aria-label="Add a J-lens steering token"
        />
        <button
          type="submit"
          class="add-btn"
          disabled={steerBusy || !steerInput.trim()}
        >
          + steer
        </button>
      </form>
    </section>

    <!-- PROBE — the merged workspace readout: pinned token-probe cards
         (persistent, gate-able) + the unpinned live aggregate cards.
         The card list owns the scroll; header + add form stay anchored
         (the CAA racks' fixed-chrome / scrollable-middle shape). -->
    <section class="section probe">
      <RackSectionHeader
        title="PROBE"
        count={`${pinnedCards.length} pinned`}
        live={liveOn}
        liveBusy={lensState.busy}
        liveTitle={liveOn
          ? "disable live readout"
          : "enable live readout"}
        onLiveToggle={onToggleLive}
        sortValue={lensState.workspaceSortMode}
        sortOptions={SORT_OPTIONS}
        sortAriaLabel="Sort J-lens probe tokens by"
        onSortChange={setLensWorkspaceSortMode}
      />

      <div class="scroll">
        {#if workspaceCards.length > 0}
          <div class="cards" role="list" aria-label="J-lens probe tokens">
            {#each workspaceCards as card (card.key)}
              <div role="listitem">
                <JLensProbeCard
                  token={card.token}
                  strength={card.strength}
                  com={card.com}
                  spread={card.spread}
                  series={card.series}
                  cells={card.cells}
                  pinned={card.pinned}
                  busy={probeBusy}
                  onpin={pinWord}
                />
              </div>
            {/each}
          </div>
        {/if}

        {#if tokenHoverState.active}
          {#if tokenHoverState.lensLoading}
            <p class="hint">reading hovered token…</p>
          {:else if aggRows.length === 0}
            <p class="hint">no J-lens score for this token</p>
          {/if}
        {:else if liveOn}
          {#if aggRows.length > 0}
            <p class="hint drill-hint">click a token for layers</p>
          {:else}
            <p class="hint">run to discover</p>
          {/if}
        {:else}
          <p class="hint">pinned only · end of run</p>
        {/if}
      </div>

      <form class="add-form anchored" onsubmit={onAddProbe}>
        <input
          class="add-input"
          type="text"
          placeholder="word (single token)"
          bind:value={probeInput}
          aria-label="Pin a J-lens token probe"
        />
        <button
          type="submit"
          class="add-btn"
          disabled={probeBusy || !probeInput.trim()}
        >
          + pin
        </button>
      </form>
    </section>
  {/if}
</div>

<style>
  /* Fixed-chrome column, matching the CAA rack-grid: STEER sizes to its
     content up to half the inspector, PROBE takes the rest and scrolls
     internally so the header + add form stay visible. */
  .jlens {
    display: flex;
    flex-direction: column;
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
  }

  /* Flat borderless sections — typography + padding carry the divide,
     matching the rack chrome. */
  .section {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-5);
  }
  .work-status {
    margin: 0;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  .section.steer {
    flex: 0 1 auto;
    min-height: 0;
    max-height: 50%;
    overflow: hidden;
  }
  /* A populated steer-card pile scrolls inside its own half-column cap
     rather than eating the probe section's share of the inspector. */
  .steer-cards {
    overflow-y: auto;
    min-height: 0;
  }
  .section.probe {
    flex: 1 1 0;
    min-height: 0;
    overflow: hidden;
  }
  /* The scrollable middle — cards + hints; header and add form stay put. */
  .scroll {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    flex: 1 1 0;
    min-height: 2.4rem;
    overflow-y: auto;
    padding-right: var(--space-1);
  }
  /* Anchored footer — borderless, same padding treatment as the CAA
     racks' actions row. */
  .add-form.anchored {
    flex: 0 0 auto;
    padding-top: var(--space-3);
  }

  .hint {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .drill-hint {
    font-size: var(--text-xs);
    color: var(--fg-dim);
  }

  .fit-error {
    color: var(--accent-red);
  }
  .fit-warning {
    color: var(--accent-yellow);
  }
  .fit-progress {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .fit-line {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .fit-msg {
    color: var(--fg);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .fit-count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .fit-bar :global(.bar) {
    width: 100%;
    height: var(--data-bar-height);
    display: block;
  }

  /* Card stack — same rhythm as the probe rack's strips. */
  .cards {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  /* ----- add forms ----- */
  .add-form {
    display: flex;
    gap: var(--space-2);
  }
  .add-input {
    flex: 1 1 auto;
    min-width: 0;
    /* Borderless input: recessed well fill; ring on focus only. */
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    padding: 2px var(--space-3);
    transition: border-color var(--dur-fast) var(--ease-out);
  }
  .add-input:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 1px;
    border-color: var(--accent-strong);
  }
  .add-btn {
    min-height: var(--control-target);
    background: color-mix(in srgb, var(--pillar-lens) 10%, transparent);
    color: var(--pillar-lens);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-size: var(--text-sm);
    padding: 1px var(--space-3);
    cursor: pointer;
    flex: 0 0 auto;
  }
  .add-btn:hover:not(:disabled) {
    background: color-mix(in srgb, var(--pillar-lens) 18%, transparent);
  }
  .add-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }

</style>
