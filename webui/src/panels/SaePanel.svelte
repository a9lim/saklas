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
  // With no resident SAE (``session_info.sae_loaded``) the tab renders the
  // release picker instead — loading pins one SAELens release + hook layer
  // for the session, so the absent channel advertises itself (the same
  // move as the lens tab's "fit j-lens" button).

  import SaeProbeCard from "./rack/SaeProbeCard.svelte";
  import SaeSteerCard from "./rack/SaeSteerCard.svelte";
  import RackSectionHeader from "./rack/RackSectionHeader.svelte";
  import { apiSae } from "../lib/api";
  import {
    activeProbeNames,
    addSaeToRack,
    attachProbe,
    loadSae,
    probeRack,
    saeState,
    seedProbeDisplay,
    sessionState,
    setLiveSae,
    setSaeSortMode,
    steerRack,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { SaeSteerEntry } from "../lib/types";
  import type { SaeSortMode } from "../lib/stores.svelte";

  const loaded = $derived(sessionState.info?.sae_loaded === true);
  const info = $derived(sessionState.info?.sae_info ?? null);
  let release = $state("");
  let releases = $state<{ release: string; layers: number[] }[]>([]);
  let discoverError = $state<string | null>(null);

  $effect(() => {
    if (loaded || releases.length > 0) return;
    void apiSae.releases().then((result) => {
      releases = result.releases;
      if (!release && releases.length > 0) release = releases[0].release;
    }).catch((error) => {
      discoverError = error instanceof Error ? error.message : String(error);
    });
  });

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
    .map((name) => ({ name, entry: probeRack.entries.get(name) }))
    .filter((row) => row.entry !== undefined));

  const discoveryBase = $derived.by(() => saeState.readout
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
  const fallbackScale = $derived.by(() => {
    let max = 0;
    for (const row of pinnedBase) {
      const entry = row.entry!;
      if (entry.info.max_act != null) continue;
      const reading = entry.aggregate ?? entry.reading;
      max = Math.max(max, reading?.coords?.[0] ?? entry.current ?? 0);
    }
    for (const feature of discoveryBase) {
      if (feature.max_act != null) continue;
      max = Math.max(max, feature.activation);
    }
    return Math.max(max, 1);
  });

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
    if (saeState.live) {
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
    const validated = await apiSae.validateFeature(id);
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
      await apiSae.validateFeature(id);
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
  {#if !loaded}
    <section class="section">
      <RackSectionHeader title="SAE" />
      <p class="hint">
        no resident SAE in this session — feature atoms, probes, and the
        live discovery readout all need one loaded SAELens release
      </p>
      <form class="load-form" onsubmit={(event) => { event.preventDefault(); void loadSae(release); }}>
        <input
          class="add-input"
          list="sae-releases"
          bind:value={release}
          placeholder="SAELens release"
          aria-label="SAE release"
        />
        <datalist id="sae-releases">
          {#each releases as row (row.release)}
            <option value={row.release}>{row.layers.map((layer) => `L${layer}`).join(", ")}</option>
          {/each}
        </datalist>
        <button class="add-btn" disabled={!release.trim() || saeState.loading}>
          {saeState.loading ? "loading…" : "load SAE"}
        </button>
      </form>
      <p class="hint">
        the selected hook layer stays resident; weights use the normal
        Hugging Face cache
      </p>
      {#if saeState.loadMessage}
        <p class="hint" role="status" aria-live="polite">{saeState.loadMessage}</p>
      {/if}
      {#if saeState.loadError}
        <p class="hint load-error" role="alert">{saeState.loadError}</p>
      {/if}
      {#if discoverError}
        <p class="hint" role="alert">registry suggestions unavailable: {discoverError}</p>
      {/if}
    </section>
  {:else}
    <!-- Identity strip — which SAE is resident (release · hook layer ·
         dictionary width).  Session identity, not a section. -->
    <div class="identity">
      <span class="release" title="resident SAELens release">{info?.release}</span>
      <span class="chip">L{info?.layer}</span>
      <span class="chip">{info?.width?.toLocaleString()} features</span>
    </div>

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
          ? "Stop the per-step feature readout (pinned probes settle to the end-of-gen activation)"
          : "Stream the SAE feature readout live during generation (pinned probes + discovery top-k)"}
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

        {#if saeState.live}
          {#if discovery.length === 0}
            <p class="hint">feature discovery streams on the next generation</p>
          {/if}
        {:else}
          <p class="hint">
            live off — pinned features settle to the end-of-generation activation
          </p>
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

  /* Identity strip — borderless: the mono/muted register and its own
     padding separate it from the STEER section below. */
  .identity {
    display: flex;
    gap: var(--space-3);
    align-items: center;
    padding: var(--space-3) var(--space-5) 0;
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  .release {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .chip {
    background: var(--glass);
    border-radius: var(--radius);
    padding: 0 var(--space-2);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
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
  .add-form,
  .load-form {
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
