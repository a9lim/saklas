<script lang="ts">
  // SAE — the inspector column's sparse-autoencoder tab: two sections,
  // card-based and symmetric with the other three pillars (every row wears
  // RackCard; the SAE family accent is gold, marker ▲/△).
  //
  //   STEER — one card per ``α sae/<id>`` decoder-row atom in the ONE
  //           steering expression.  Per-card α slider + trigger pill.
  //   PROBE — pinned ``sae/<id>`` feature probes (persistent, gate-able)
  //           first, then the live discovery cards for the per-step top-k
  //           features not already pinned — both the same card shape,
  //           exactly like the lens workspace readout (the pinned card's ▲
  //           unpins, the unpinned card's △ pins).  The card list owns the
  //           scroll; header + add form stay anchored.  The header's live
  //           toggle is the SAE live switch: off ⇒ no per-step feature
  //           readout — pinned probes settle to the end-of-gen activation,
  //           discovery cards go quiet.
  //
  // SOURCE mirrors J-LENS exactly: one selector uses or fetches an artifact,
  // followed by a labelled custom row. Successful preparation makes the source
  // resident and turns live readout on.

  import Bar from "../lib/charts/Bar.svelte";
  import Select from "../lib/Select.svelte";
  import Button from "../lib/ui/Button.svelte";
  import { onMount } from "svelte";
  import SaeProbeCard from "./rack/SaeProbeCard.svelte";
  import SaeSteerCard from "./rack/SaeSteerCard.svelte";
  import InstrumentSourceSection from "./rack/InstrumentSourceSection.svelte";
  import RackSectionHeader from "./rack/RackSectionHeader.svelte";
  import { apiInstruments } from "../lib/api";
  import {
    activeProbeNames,
    addSaeToRack,
    attachProbe,
    cancelSaeTrain,
    checkSaeTrain,
    loadSae,
    probeRack,
    probeEntryForDisplay,
    saeState,
    saeSourceState,
    saeTrainState,
    saeRawFallbackScale,
    saeReadoutForDisplay,
    seedProbeDisplay,
    sessionState,
    setLiveSae,
    setSaeSortMode,
    startSaeTrain,
    steerRack,
    tokenHoverState,
    refreshSaeSources,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { SaeSteerEntry } from "../lib/types";
  import type { SaeSortMode } from "../lib/stores.svelte";

  const loaded = $derived(sessionState.info?.sae_loaded === true);
  const displayReadout = $derived(saeReadoutForDisplay());
  const info = $derived(sessionState.info?.sae_info ?? null);
  let selectedSource = $state("");
  let selectedLayer = $state("");
  let layerSource = $state("");
  let localName = $state("my-sae");
  let trainTokens = $state(1_000_000);
  let trainLayer = $state("");
  let trainConfirm = $state(false);
  let releases = $state<{
    release: string;
    layers: number[];
    source?: "local" | "saelens";
  }[]>([]);
  let discoverError = $state<string | null>(null);
  const providerOptions = $derived(releases.map((row) => ({
    value: `saelens:${row.release}`,
    label: row.release,
  })));
  const sourceBusy = $derived(
    saeSourceState.loading || saeState.loading || saeTrainState.running,
  );
  const selectedPreparedSource = $derived(
    saeSourceState.sources.find((source) => source.source === selectedSource),
  );
  const selectedRelease = $derived(
    selectedSource.startsWith("saelens:")
      ? selectedSource.slice("saelens:".length)
      : selectedSource,
  );
  const availableLayers = $derived.by(() => {
    const registry = releases.find((row) => row.release === selectedRelease);
    const layers = registry?.layers ?? (
      selectedPreparedSource?.layer == null ? [] : [selectedPreparedSource.layer]
    );
    return [...new Set(layers)].sort((a, b) => a - b);
  });
  const layerOptions = $derived(availableLayers.map((layer) => ({
    value: String(layer),
    label: `layer ${layer}`,
  })));
  const sourceMatchesLoaded = $derived(
    loaded && info !== null && selectedRelease === info.release,
  );
  const selectedLayerNumber = $derived(
    selectedLayer === "" ? null : Number(selectedLayer),
  );
  const sourceSelectionCurrent = $derived(
    sourceMatchesLoaded && selectedLayerNumber === info?.layer,
  );

  onMount(() => {
    void refreshSaeSources().then(() => {
      const active = saeSourceState.sources.find((source) => source.active);
      if (active) selectedSource = active.source;
    });
    void checkSaeTrain();
  });

  $effect(() => {
    const active = saeSourceState.sources.find((source) => source.active);
    const known = saeSourceState.sources.some(
      (source) => source.source === selectedSource,
    ) || providerOptions.some((source) => source.value === selectedSource) ||
      selectedSource === "local";
    if (
      !selectedSource ||
      !known
    ) {
      selectedSource = active?.source ?? saeSourceState.sources[0]?.source ??
        providerOptions[0]?.value ?? "";
    }
  });

  $effect(() => {
    const source = selectedSource;
    const layers = availableLayers;
    if (source !== layerSource) {
      layerSource = source;
      const resident = sourceMatchesLoaded ? info?.layer : null;
      const cached = selectedPreparedSource?.layer ?? null;
      const preferred = resident != null && layers.includes(resident)
        ? resident
        : cached != null && layers.includes(cached)
        ? cached
        : preferredLayer(layers);
      selectedLayer = preferred == null ? "" : String(preferred);
    } else if (
      layers.length > 0 &&
      (selectedLayer === "" || !layers.includes(Number(selectedLayer)))
    ) {
      const preferred = preferredLayer(layers);
      selectedLayer = preferred == null ? "" : String(preferred);
    }
  });

  $effect(() => {
    if (selectedSource !== "local") trainConfirm = false;
  });

  $effect(() => {
    if (releases.length > 0) return;
    void apiInstruments.sources("sae").then((result) => {
      releases = (result.releases ?? []).filter((row) => row.source !== "local");
      if (!selectedSource && releases.length > 0) {
        selectedSource = `saelens:${releases[0].release}`;
      }
    }).catch((error) => {
      discoverError = error instanceof Error ? error.message : String(error);
    });
  });

  function requestTrain(): void {
    if (!localName.trim() || saeTrainState.running) return;
    if (!trainConfirm) {
      trainConfirm = true;
      return;
    }
    trainConfirm = false;
    const parsedLayer = trainLayer.trim() === "" ? null : Number(trainLayer);
    void startSaeTrain({
      name: localName.trim(),
      tokens: trainTokens,
      layer: parsedLayer != null && Number.isInteger(parsedLayer)
        ? parsedLayer
        : null,
    });
  }

  function preferredLayer(layers: number[]): number | null {
    if (layers.length === 0) return null;
    const depth = Math.max(...layers, 1);
    const band = layers.filter((layer) => {
      const fraction = layer / depth;
      return fraction >= 0.4 && fraction <= 0.9;
    });
    const pool = band.length > 0 ? band : layers;
    const target = 0.65 * depth;
    return [...pool].sort((a, b) =>
      Math.abs(a - target) - Math.abs(b - target) || a - b
    )[0] ?? null;
  }

  function loadSelectedSae(source: string): void {
    void loadSae(source, selectedLayerNumber);
  }

  // ---------- STEER: sae-mode rack entries (by feature id) ----------
  const steerCards = $derived.by(() => {
    const rows = [...steerRack.entries.entries()].filter(
      (row): row is [string, SaeSteerEntry] => row[1].mode === "sae",
    );
    rows.sort((a, b) => Number(a[0].slice(4)) - Number(b[0].slice(4)));
    return rows;
  });

  // ---------- PROBE: pinned probe cards + unpinned discovery cards ----------
  const pinnedBase = $derived.by(() => activeProbeNames()
    .filter((name) => name.startsWith("sae/"))
    .map((name) => ({ name, entry: probeEntryForDisplay(name) }))
    .filter((row) => row.entry !== undefined));

  const discoveryBase = $derived.by(() => displayReadout
    .filter((row) => !probeRack.active.includes(`sae/${row.id}`))
    .map((row) => {
      // Metadata merges from the row's server-cached values and the
      // between-generation backfill (saeState.meta).
      const meta = saeState.meta.get(row.id);
      return {
        ...row,
        label: row.label ?? meta?.label ?? null,
        max_act: row.max_act ?? meta?.max_act ?? null,
      };
    }));

  /** Panel-shared raw scale for metadata-less cards: the max raw reading
   *  across the visible cards that lack a maxActApprox unit, so their
   *  bars and numbers rank identically.  Cards WITH the unit render on
   *  the absolute 0..1 strength scale instead. */
  const fallbackScale = $derived(saeRawFallbackScale());

  const SORT_OPTIONS: { value: SaeSortMode; label: string }[] = [
    { value: "strength", label: "strength" },
    { value: "name", label: "name" },
  ];

  function visibleStrength(value: number, maxAct: number | null): number {
    return maxAct != null && maxAct > 0 ? value / maxAct : value / fallbackScale;
  }

  const pinned = $derived.by(() => {
    const rows = [...pinnedBase];
    if (saeState.sortMode === "name") {
      rows.sort((a, b) => {
        const aid = Number(a.name.slice(4));
        const bid = Number(b.name.slice(4));
        const an = a.entry?.info.label || String(aid);
        const bn = b.entry?.info.label || String(bid);
        return an.localeCompare(bn, undefined, { numeric: true });
      });
    } else {
      rows.sort((a, b) => {
        const ae = a.entry!;
        const be = b.entry!;
        const av = ae.aggregate?.coords?.[0] ?? ae.reading?.coords?.[0] ?? ae.current;
        const bv = be.aggregate?.coords?.[0] ?? be.reading?.coords?.[0] ?? be.current;
        // Pinned values with max_act are already normalized by the server.
        const as = ae.info.max_act != null ? av : av / fallbackScale;
        const bs = be.info.max_act != null ? bv : bv / fallbackScale;
        return bs - as;
      });
    }
    return rows;
  });

  const discovery = $derived.by(() => {
    const rows = [...discoveryBase];
    if (saeState.sortMode === "name") {
      rows.sort((a, b) =>
        (a.label || String(a.id)).localeCompare(
          b.label || String(b.id),
          undefined,
          { numeric: true },
        ),
      );
    } else {
      rows.sort((a, b) =>
        visibleStrength(b.activation, b.max_act ?? null) -
        visibleStrength(a.activation, a.max_act ?? null),
      );
    }
    return rows;
  });

  type VisibleProbeCard =
    | {
        kind: "pinned";
        key: string;
        name: string;
        entry: NonNullable<(typeof pinnedBase)[number]["entry"]>;
        sortName: string;
        strength: number;
      }
    | {
        kind: "discovery";
        key: string;
        feature: (typeof discoveryBase)[number];
        sortName: string;
        strength: number;
      };

  /** One visible list, one sort order. Pinning changes persistence/actions,
   *  never a card's position outside the selected sort. */
  const probeCards = $derived.by((): VisibleProbeCard[] => {
    const rows: VisibleProbeCard[] = pinned.map((row) => {
      const entry = row.entry!;
      const value = entry.aggregate?.coords?.[0] ??
        entry.reading?.coords?.[0] ?? entry.current;
      const id = Number(row.name.slice(4));
      return {
        kind: "pinned",
        key: row.name,
        name: row.name,
        entry,
        sortName: entry.info.label || String(id),
        strength: entry.info.max_act != null ? value : value / fallbackScale,
      };
    });
    if (saeState.live || tokenHoverState.active) {
      rows.push(...discovery.map((feature) => ({
        kind: "discovery" as const,
        key: `sae/${feature.id}`,
        feature,
        sortName: feature.label || String(feature.id),
        strength: visibleStrength(feature.activation, feature.max_act ?? null),
      })));
    }
    rows.sort(saeState.sortMode === "name"
      ? (a, b) => a.sortName.localeCompare(
          b.sortName, undefined, { numeric: true },
        ) || b.strength - a.strength
      : (a, b) => b.strength - a.strength ||
          a.sortName.localeCompare(b.sortName, undefined, { numeric: true }));
    return rows;
  });

  let steerInput = $state("");
  let probeInput = $state("");
  let featureBusy = $state(false);

  async function validateInput(raw: string): Promise<number | null> {
    const id = Number(raw.trim().replace(/^sae\//, ""));
    if (!Number.isInteger(id) || id < 0) {
      pushToast("feature id must be a non-negative integer", { kind: "error" });
      return null;
    }
    const validated = await apiInstruments.validateSaeFeature(id);
    return validated.id;
  }

  async function addSteer(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    if (featureBusy) return;
    featureBusy = true;
    try {
      const id = await validateInput(steerInput);
      if (id !== null) {
        addSaeToRack(id);
        steerInput = "";
      }
    } catch (error) {
      pushToast(error instanceof Error ? error.message : String(error), { kind: "error" });
    } finally {
      featureBusy = false;
    }
  }

  async function pin(id: number): Promise<void> {
    if (featureBusy || probeRack.active.includes(`sae/${id}`)) return;
    featureBusy = true;
    try {
      await apiInstruments.validateSaeFeature(id);
      const live = discoveryBase.find((row) => row.id === id);
      const meta = saeState.meta.get(id);
      const preAttachMaxAct = live?.max_act ?? meta?.max_act ?? null;
      const attached = await attachProbe(`sae/${id}`);
      if (live) {
        // Attachment may discover Neuronpedia metadata that the live row
        // did not have yet. Seed in the unit the attached probe declares,
        // not the stale pre-attach unit.
        const maxAct = attached.max_act ?? preAttachMaxAct;
        const value = maxAct != null && maxAct > 0
          ? live.activation / maxAct
          : live.activation;
        const series = saeState.history.get(id) ?? [];
        seedProbeDisplay(`sae/${id}`, {
          current: value,
          sparkline: maxAct != null && maxAct > 0
            ? series.map((v) => v / maxAct)
            : series,
        });
      }
    } catch (error) {
      pushToast(error instanceof Error ? error.message : String(error), { kind: "error" });
    } finally {
      featureBusy = false;
    }
  }

  async function addProbe(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    if (featureBusy) return;
    featureBusy = true;
    try {
      const id = await validateInput(probeInput);
      if (id !== null) {
        await attachProbe(`sae/${id}`);
        probeInput = "";
      }
    } catch (error) {
      pushToast(error instanceof Error ? error.message : String(error), { kind: "error" });
    } finally {
      featureBusy = false;
    }
  }
</script>

<div class="sae" aria-label="Sparse-autoencoder inspector">
  <InstrumentSourceSection
    ready={loaded}
    sources={saeSourceState.sources}
    bind:value={selectedSource}
    busy={sourceBusy}
    accent="var(--pillar-sae)"
    sourceError={saeSourceState.error}
    working={saeTrainState.running}
    selectionCurrent={sourceSelectionCurrent}
    onuse={loadSelectedSae}
    providerOptions={providerOptions}
    providerPlaceholder="SAELens release"
    onfetch={loadSelectedSae}
    localActionLabel={trainConfirm ? "confirm train" : "train"}
    localActionDisabled={!localName.trim() || sourceBusy}
    onlocal={requestTrain}
  >
    {#snippet sourceControls()}
      <Select
        bind:value={selectedLayer}
        options={layerOptions}
        placeholder="layer"
        disabled={sourceBusy || layerOptions.length === 0}
        ariaLabel="SAE measurement layer"
      />
    {/snippet}
    {#snippet localControls()}
      <label class="setup-field setup-field-wide">
        <span class="setup-field-label">name</span>
        <input
          class="add-input"
          bind:value={localName}
          placeholder="name"
          aria-label="Local SAE name"
        />
      </label>
      <label class="setup-field setup-field-medium">
        <span class="setup-field-label">tokens</span>
        <input
          class="add-input"
          type="number"
          min="1"
          step="10000"
          bind:value={trainTokens}
          aria-label="SAE training tokens"
          title="tokens"
        />
      </label>
      <label class="setup-field setup-field-narrow">
        <span class="setup-field-label">layer</span>
        <input
          class="add-input"
          inputmode="numeric"
          bind:value={trainLayer}
          placeholder="auto"
          aria-label="Residual layer (blank for automatic)"
        />
      </label>
    {/snippet}
    {#snippet progress()}
      <div class="train-progress" role="status" aria-live="polite">
        <div class="train-line">
          <span class="work-status">{saeTrainState.message ?? "training…"}</span>
          <span class="train-count">
            {saeTrainState.tokensDone.toLocaleString()}/{saeTrainState.tokensTotal.toLocaleString()}
          </span>
        </div>
        <Bar
          value={saeTrainState.tokensDone}
          max={Math.max(saeTrainState.tokensTotal, 1)}
          width={160}
          height={8}
          color="var(--pillar-sae)"
        />
        <Button
          size="sm"
          variant="danger"
          disabled={saeTrainState.cancelling}
          onclick={() => void cancelSaeTrain()}
        >
          {saeTrainState.cancelling ? "cancelling…" : "cancel"}
        </Button>
      </div>
    {/snippet}
    {#snippet warning()}
      {#if trainConfirm}
        <p class="hint train-warning" role="alert">
          Blocks generation; uses FineWeb-Edu. Confirm again.
        </p>
      {/if}
    {/snippet}
    {#snippet messages()}
      {#if saeState.loading && saeState.loadMessage}
        <p class="hint" role="status" aria-live="polite">{saeState.loadMessage}</p>
      {/if}
      {#if saeState.loadError}
        <p class="hint load-error" role="alert">{saeState.loadError}</p>
      {/if}
      {#if saeTrainState.error}
        <p class="hint load-error" role="alert">local train: {saeTrainState.error}</p>
      {/if}
      {#if discoverError}
        <p class="hint" role="alert">registry: {discoverError}</p>
      {/if}
    {/snippet}
  </InstrumentSourceSection>

  {#if loaded}

    <!-- STEER — decoder-row atom cards in the shared steering expression. -->
    <section class="section steer">
      <RackSectionHeader
        title="STEER"
        count={`${steerCards.length} term${steerCards.length === 1 ? "" : "s"}`}
      />

      {#if steerCards.length > 0}
        <div class="cards steer-cards" role="list">
          {#each steerCards as [name, entry] (name)}
            <div role="listitem">
              <SaeSteerCard {name} {entry} />
            </div>
          {/each}
        </div>
      {/if}

      <form class="add-form" onsubmit={addSteer}>
        <input
          class="add-input"
          type="text"
          placeholder="feature id"
          bind:value={steerInput}
          aria-label="Add an SAE steering feature"
        />
        <button
          type="submit"
          class="add-btn"
          disabled={featureBusy || !steerInput.trim()}
        >
          + steer
        </button>
      </form>
    </section>

    <!-- PROBE — pinned feature probes + the live discovery top-k.  The
         card list owns the scroll; header + add form stay anchored (the
         other pillars' fixed-chrome / scrollable-middle shape). -->
    <section class="section probe">
      <RackSectionHeader
        title="PROBE"
        count={`${pinned.length} pinned`}
        live={saeState.live}
        liveBusy={saeState.busy}
        liveTitle={saeState.live
          ? "disable live readout"
          : "enable live readout"}
        onLiveToggle={() => void setLiveSae(!saeState.live)}
        sortValue={saeState.sortMode}
        sortOptions={SORT_OPTIONS}
        sortAriaLabel="Sort SAE probe features by"
        onSortChange={setSaeSortMode}
      />

      <div class="scroll">
        {#if probeCards.length > 0}
          <div class="cards" role="list" aria-label="SAE feature probes">
            {#each probeCards as row (row.key)}
              <div role="listitem">
                {#if row.kind === "pinned"}
                  {@const reading = row.entry.aggregate ?? row.entry.reading}
                  <!-- A pinned probe's channel (coords, sparkline, gates) is
                       already normalized server-side when max_act is set. -->
                  <SaeProbeCard
                    id={Number(row.name.slice(4))}
                    label={row.entry.info.label}
                    layer={info?.layer ?? null}
                    value={reading?.coords?.[0] ?? row.entry.current ?? 0}
                    maxAct={row.entry.info.max_act ?? null}
                    valueIsStrength={row.entry.info.max_act != null}
                    {fallbackScale}
                    series={row.entry.sparkline}
                    pinned={true}
                  />
                {:else}
                  <SaeProbeCard
                    id={row.feature.id}
                    label={row.feature.label}
                    layer={info?.layer ?? null}
                    value={row.feature.activation}
                    maxAct={row.feature.max_act}
                    {fallbackScale}
                    series={saeState.history.get(row.feature.id) ?? []}
                    pinned={false}
                    busy={featureBusy}
                    onpin={(id) => void pin(id)}
                  />
                {/if}
              </div>
            {/each}
          </div>
        {/if}

        {#if tokenHoverState.active}
          {#if tokenHoverState.saeLoading}
            <p class="hint">reading hovered token…</p>
          {:else if probeCards.length === 0}
            <p class="hint">no SAE score for this token</p>
          {/if}
        {:else if saeState.live}
          {#if discovery.length === 0}
            <p class="hint">run to discover</p>
          {/if}
        {:else}
          <p class="hint">pinned only · end of run</p>
        {/if}
      </div>

      <form class="add-form anchored" onsubmit={addProbe}>
        <input
          class="add-input"
          type="text"
          placeholder="feature id"
          bind:value={probeInput}
          aria-label="Pin an SAE feature probe"
        />
        <button
          type="submit"
          class="add-btn"
          disabled={featureBusy || !probeInput.trim()}
        >
          + pin
        </button>
      </form>
    </section>
  {/if}
</div>

<style>
  /* Fixed-chrome column, matching the other pillar tabs: STEER sizes to
     its content up to half the inspector, PROBE takes the rest and scrolls
     internally so the header + add form stay visible. */
  .sae {
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
    min-height: 0;
  }
  .train-line {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .train-progress {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-2);
  }
  .train-line {
    width: 100%;
    justify-content: space-between;
  }
  .work-status {
    margin: 0;
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  .work-status {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .train-count {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .train-warning {
    color: var(--accent-yellow);
  }
  .section.steer {
    flex: 0 1 auto;
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
  /* Anchored footer — borderless, same padding treatment as the racks'
     actions row. */
  .add-form.anchored {
    flex: 0 0 auto;
    padding-top: var(--space-3);
  }

  .hint {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .load-error {
    color: var(--accent-red);
  }

  /* Card stack — same rhythm as the other racks' strips. */
  .cards {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  /* ----- add / load forms ----- */
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
    /* Family-tinted fill — the gold sibling of the racks' launchers. */
    background: color-mix(in srgb, var(--pillar-sae) 10%, transparent);
    color: var(--pillar-sae);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-size: var(--text-sm);
    padding: 1px var(--space-3);
    cursor: pointer;
    flex: 0 0 auto;
    transition: background var(--dur) var(--ease-out);
  }
  .add-btn:hover:not(:disabled) {
    background: color-mix(in srgb, var(--pillar-sae) 18%, transparent);
  }
  .add-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }

</style>
