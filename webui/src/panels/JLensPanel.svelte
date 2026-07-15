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
  import JLensSteerCard from "./rack/JLensSteerCard.svelte";
  import JLensTokenCard from "./rack/JLensTokenCard.svelte";
  import { ApiError, apiLens } from "../lib/api";
  import {
    addJLensToRack,
    activeProbeNames,
    attachProbe,
    checkLensFetch,
    cancelLensFit,
    checkLensFit,
    lensFitState,
    lensFetchState,
    lensSourceState,
    lensState,
    probeRack,
    refreshLensSources,
    seedProbeDisplay,
    sessionState,
    setLensWorkspaceSortMode,
    setLiveLens,
    startLensFetch,
    startLensFit,
    steerRack,
    useLensSource,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { JLensSteerEntry, ProbeRackEntry } from "../lib/types";
  import type { LensWorkspaceSortMode } from "../lib/stores.svelte";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const liveOn = $derived(lensState.layers !== null);
  const sourceBusy = $derived(
    lensSourceState.loading || lensSourceState.busy ||
      lensFetchState.running || lensFitState.running,
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
    (lensFitState.message ?? "").startsWith("streaming "),
  );

  function requestFit(): void {
    if (!fitConfirm) {
      fitConfirm = true;
      return;
    }
    fitConfirm = false;
    void startLensFit({
      prompts: fitPrompts,
      layers: fitLayers.trim(),
    });
  }

  // Resume-visibility: a page reload mid-fit should pick the progress
  // polling back up (the fit runs server-side regardless of the client).
  onMount(() => {
    void checkLensFit();
    void checkLensFetch();
    void refreshLensSources().then(() => {
      const active = lensSourceState.sources.find((source) => source.active);
      if (active) selectedSource = active.source;
    });
  });

  $effect(() => {
    const active = lensSourceState.sources.find((source) => source.active);
    const known = lensSourceState.sources.some(
      (source) => source.source === selectedSource,
    ) || LENS_PROVIDER_OPTIONS.some((source) => source.value === selectedSource) ||
      selectedSource === "local";
    if (
      !selectedSource ||
      !known
    ) {
      selectedSource = active?.source ?? lensSourceState.sources[0]?.source ??
        LENS_PROVIDER_OPTIONS[0]?.value ?? "";
    }
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

  function describeError(e: unknown): string {
    if (e instanceof ApiError) {
      return e.body && typeof e.body === "object" && "detail" in e.body
        ? String((e.body as { detail: unknown }).detail)
        : e.message;
    }
    return e instanceof Error ? e.message : String(e);
  }

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
      const validated = await apiLens.validateToken(word);
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

  interface PinnedRow {
    name: string;
    entry: ProbeRackEntry;
    /** Sort keys mirroring the aggregate rows' (axis 0 = strength). */
    value: number;
    com: number;
  }

  interface AggRow {
    /** Raw vocabulary token text (untrimmed — the strip matches on it). */
    token: string;
    strength: number;
    com: number;
    spread: number;
    /** Recent strength history (0 where the token fell below top-k). */
    series: number[];
  }

  const SORT_OPTIONS: {
    value: LensWorkspaceSortMode;
    label: string;
  }[] = [
    { value: "strength", label: "strength" },
    { value: "name", label: "name" },
    { value: "depth", label: "depth" },
  ];

  const pinnedCards = $derived.by((): PinnedRow[] => {
    const names = activeProbeNames().filter((n) => n.startsWith("jlens/"));
    const rows: PinnedRow[] = [];
    for (const name of names) {
      const entry = probeRack.entries.get(name);
      if (!entry) continue;
      const latest = entry.aggregate ?? entry.reading;
      rows.push({
        name,
        entry,
        value: latest?.coords?.[0] ?? entry.current ?? 0,
        com: latest?.depth_com?.[0] ?? 0,
      });
    }
    if (lensState.workspaceSortMode === "name") {
      rows.sort((a, b) => a.name.localeCompare(b.name));
    } else if (lensState.workspaceSortMode === "depth") {
      rows.sort((a, b) => a.com - b.com || b.value - a.value);
    } else {
      rows.sort((a, b) => b.value - a.value || a.name.localeCompare(b.name));
    }
    return rows;
  });

  const aggRows = $derived.by((): AggRow[] => {
    const rows = lensState.aggregate;
    if (!rows || rows.length === 0) return [];
    const hist = lensState.aggHistory;
    // Pinned tokens already have a persistent card above — the aggregate
    // group carries only the unpinned remainder of the top-k.
    const out = rows
      .filter(([token]) => !probeRack.active.includes(`jlens/${token.trim()}`))
      .map(([token, strength, com, spread]) => ({
        token,
        strength,
        com,
        spread,
        series: hist.map(
          (frame) => frame.find(([t]) => t === token)?.[1] ?? 0,
        ),
      }));
    if (lensState.workspaceSortMode === "name") {
      out.sort((a, b) =>
        a.token.trim().localeCompare(b.token.trim()) || b.strength - a.strength,
      );
    } else if (lensState.workspaceSortMode === "depth") {
      out.sort((a, b) => a.com - b.com || b.strength - a.strength);
    } else {
      out.sort((a, b) => b.strength - a.strength);
    }
    return out;
  });

  type WorkspaceCard =
    | {
        kind: "pinned";
        key: string;
        row: PinnedRow;
        sortName: string;
        strength: number;
        com: number;
      }
    | {
        kind: "aggregate";
        key: string;
        row: AggRow;
        sortName: string;
        strength: number;
        com: number;
      };

  /** Pinned and discovered tokens are one visual roster. Persistence is an
   *  action/state difference, not a hidden first sort key. */
  const workspaceCards = $derived.by((): WorkspaceCard[] => {
    const rows: WorkspaceCard[] = pinnedCards.map((row) => ({
      kind: "pinned",
      key: row.name,
      row,
      sortName: row.name.slice("jlens/".length),
      strength: row.value,
      com: row.com,
    }));
    if (liveOn) {
      rows.push(...aggRows.map((row) => ({
        kind: "aggregate" as const,
        key: `aggregate:${row.token}`,
        row,
        sortName: row.token.trim(),
        strength: row.strength,
        com: row.com,
      })));
    }
    if (lensState.workspaceSortMode === "name") {
      rows.sort((a, b) => a.sortName.localeCompare(b.sortName) ||
        b.strength - a.strength);
    } else if (lensState.workspaceSortMode === "depth") {
      rows.sort((a, b) => a.com - b.com || b.strength - a.strength);
    } else {
      rows.sort((a, b) => b.strength - a.strength ||
        a.sortName.localeCompare(b.sortName));
    }
    return rows;
  });

  let probeInput = $state("");
  let probeBusy = $state(false);

  async function pinWord(word: string): Promise<boolean> {
    const bare = bareWord(word);
    if (!bare || probeBusy) return false;
    const selector = `jlens/${bare}`;
    if (probeRack.active.includes(selector)) return true;
    probeBusy = true;
    try {
      const validated = await apiLens.validateToken(bare);
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
    accent="var(--accent-blue)"
    sourceError={lensSourceState.error}
    working={lensFetchState.running || lensFitState.running}
    onuse={(source) => void useLensSource(source)}
    providerOptions={LENS_PROVIDER_OPTIONS}
    providerPlaceholder="lens provider"
    onfetch={(source) => void startLensFetch(source)}
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
      {#if lensFetchState.running}
        <p class="work-status" role="status" aria-live="polite">
          {lensFetchState.message ?? "fetching official lens…"}
        </p>
      {:else}
        <div
          class="fit-progress"
          role="status"
          aria-live="polite"
          aria-label="Lens fit progress"
        >
          <div class="fit-line">
            <span class="fit-msg">{lensFitState.message ?? "fitting…"}</span>
            {#if lensFitState.promptsTotal > 0}
              <span class="fit-count">
                {lensFitState.promptsDone}/{lensFitState.promptsTotal}
              </span>
            {/if}
          </div>
          <div
            class="fit-bar"
            role="progressbar"
            aria-label="J-lens prompts fitted"
            aria-valuemin="0"
            aria-valuemax={Math.max(lensFitState.promptsTotal, 1)}
            aria-valuenow={lensFitState.promptsDone}
          >
            <Bar
              value={lensFitState.promptsDone}
              max={Math.max(lensFitState.promptsTotal, 1)}
              width={160}
              height={8}
              color="var(--accent-blue)"
            />
          </div>
          <p class="hint">
            {#if lensFitState.cancelling}
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
            disabled={lensFitState.cancelling}
            onclick={() => void cancelLensFit()}
          >
            {lensFitState.cancelling ? "cancelling…" : "cancel"}
          </Button>
        </div>
      {/if}
    {/snippet}
    {#snippet warning()}
      {#if fitConfirm && !lensFitState.running}
        <p class="hint fit-warning" role="alert">
          Blocks generation; may take hours. Confirm again.
        </p>
      {/if}
    {/snippet}
    {#snippet messages()}
      {#if lensFitState.error}
        <p class="hint fit-error" role="alert">local fit: {lensFitState.error}</p>
      {/if}
      {#if lensFetchState.error}
        <p class="hint fit-error" role="alert">official fetch: {lensFetchState.error}</p>
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
              <JLensSteerCard {name} {entry} />
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
                {#if card.kind === "pinned"}
                  <JLensProbeCard
                    name={card.row.name}
                    entry={card.row.entry}
                  />
                {:else}
                  <JLensTokenCard
                    token={card.row.token}
                    strength={card.row.strength}
                    com={card.row.com}
                    spread={card.row.spread}
                    series={card.row.series}
                    layers={lensState.layers ?? []}
                    readout={lensState.readout}
                    pinned={false}
                    busy={probeBusy}
                    onpin={pinWord}
                  />
                {/if}
              </div>
            {/each}
          </div>
        {/if}

        {#if liveOn}
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
    height: 8px;
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
    border-color: var(--accent-glow);
  }
  .add-btn {
    min-height: 24px;
    background: color-mix(in srgb, var(--accent-blue) 10%, transparent);
    color: var(--accent-blue);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-size: var(--text-sm);
    padding: 1px var(--space-3);
    cursor: pointer;
    flex: 0 0 auto;
  }
  .add-btn:hover:not(:disabled) {
    background: color-mix(in srgb, var(--accent-blue) 18%, transparent);
  }
  .add-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }

</style>
