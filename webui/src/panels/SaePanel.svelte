<script lang="ts">
  import Bar from "../lib/charts/Bar.svelte";
  import Sparkline from "../lib/charts/Sparkline.svelte";
  import Slider from "../lib/Slider.svelte";
  import RackCard from "./rack/RackCard.svelte";
  import { apiSae } from "../lib/api";
  import {
    activeProbeNames,
    addSaeToRack,
    attachProbe,
    detachProbe,
    loadSae,
    probeRack,
    removeSaeFromRack,
    saeState,
    sessionState,
    setLiveSae,
    setSaeAlpha,
    setSaeEnabled,
    setSaeTrigger,
    steerRack,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { SaeSteerEntry, Trigger } from "../lib/types";
  import { TRIGGER_LABEL, TRIGGER_WORD, nextTrigger } from "./rack/triggers";

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

  const steerCards = $derived.by(() => {
    const rows = [...steerRack.entries.entries()].filter(
      (row): row is [string, SaeSteerEntry] => row[1].mode === "sae",
    );
    rows.sort((a, b) => Number(a[0].slice(4)) - Number(b[0].slice(4)));
    return rows;
  });

  const pinned = $derived.by(() => activeProbeNames()
    .filter((name) => name.startsWith("sae/"))
    .map((name) => ({ name, entry: probeRack.entries.get(name) }))
    .filter((row) => row.entry !== undefined));

  const discovery = $derived(saeState.readout.filter(
    (row) => !probeRack.active.includes(`sae/${row.id}`),
  ));

  let featureInput = $state("");
  let featureBusy = $state(false);

  async function validateInput(): Promise<number | null> {
    const id = Number(featureInput.trim().replace(/^sae\//, ""));
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
      const id = await validateInput();
      if (id !== null) {
        addSaeToRack(id);
        featureInput = "";
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
      await attachProbe(`sae/${id}`);
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
      const id = await validateInput();
      if (id !== null) {
        await attachProbe(`sae/${id}`);
        featureInput = "";
      }
    } catch (error) {
      pushToast(error instanceof Error ? error.message : String(error), { kind: "error" });
    } finally {
      featureBusy = false;
    }
  }

  function featureLabel(id: number, label?: string | null): string {
    return label ? `${id} · ${label}` : String(id);
  }
</script>

<div class="sae" aria-label="Sparse-autoencoder inspector">
  {#if !loaded}
    <section class="section empty">
      <header><span class="title">SAE</span></header>
      <p>Load one SAELens release into this session. The selected hook layer stays resident; weights use the normal Hugging Face cache.</p>
      <form class="load-form" onsubmit={(event) => { event.preventDefault(); void loadSae(release); }}>
        <input list="sae-releases" bind:value={release} placeholder="SAELens release" aria-label="SAE release" />
        <datalist id="sae-releases">
          {#each releases as row (row.release)}
            <option value={row.release}>{row.layers.map((layer) => `L${layer}`).join(", ")}</option>
          {/each}
        </datalist>
        <button disabled={!release.trim() || saeState.loading}>{saeState.loading ? "loading…" : "load SAE"}</button>
      </form>
      {#if saeState.loadMessage}<p class="hint">{saeState.loadMessage}</p>{/if}
      {#if saeState.loadError}<p class="error">{saeState.loadError}</p>{/if}
      {#if discoverError}<p class="hint">registry suggestions unavailable: {discoverError}</p>{/if}
    </section>
  {:else}
    <div class="identity">
      <span>{info?.release}</span>
      <span class="chip">L{info?.layer}</span>
      <span class="chip">{info?.width?.toLocaleString()} features</span>
    </div>

    <section class="section steer">
      <header><span class="title">STEER</span><span class="count">{steerCards.length} terms</span></header>
      <div class="scroll">
        {#each steerCards as [name, entry] (name)}
          <RackCard accent="--pillar-sae" disabled={!entry.enabled}>
            {#snippet statline()}
              <button class="glyph" onclick={() => setSaeEnabled(name, !entry.enabled)}>{entry.enabled ? "■" : "□"}</button>
              <span class="name">{name}</span><span class="spacer"></span>
              <button class="trigger" title={TRIGGER_LABEL[entry.trigger]} onclick={() => setSaeTrigger(name, nextTrigger(entry.trigger) as Trigger)}>{TRIGGER_WORD[entry.trigger]}</button>
              <button class="remove" onclick={() => removeSaeFromRack(name)}>✕</button>
            {/snippet}
            {#snippet body()}
              <div class="alpha"><span>α</span><Slider value={entry.alpha} min={0} max={1} step={0.05} ariaLabel={`alpha for ${name}`} oninput={(value) => setSaeAlpha(name, value)} /><span>{entry.alpha.toFixed(2)}</span></div>
            {/snippet}
          </RackCard>
        {/each}
      </div>
      <form class="add" onsubmit={addSteer}><input bind:value={featureInput} placeholder="feature id" /><button disabled={featureBusy}>+ steer</button></form>
    </section>

    <section class="section probe">
      <header>
        <span class="title">PROBE</span>
        <button class:on={saeState.live} class="toggle" disabled={saeState.busy} onclick={() => void setLiveSae(!saeState.live)}>live: {saeState.live ? "on" : "off"}</button>
        <span class="count">{pinned.length} pinned</span>
      </header>
      <div class="scroll">
        {#each pinned as row (row.name)}
          {@const entry = row.entry!}
          {@const id = Number(row.name.slice(4))}
          {@const reading = entry.aggregate ?? entry.reading}
          {@const value = reading?.coords?.[0] ?? entry.current ?? 0}
          <RackCard accent="--pillar-sae">
            {#snippet statline()}
              <button class="glyph" onclick={() => void detachProbe(row.name)} title="unpin">■</button>
              <span class="name">{featureLabel(id, entry.info.label)}</span><span class="spacer"></span><span class="chip">L{info?.layer}</span>
            {/snippet}
            {#snippet body()}
              <div class="meter"><span>activation</span><Bar value={Math.max(value, 0)} max={Math.max(...entry.sparkline, value, 1)} width={160} height={8} color="var(--pillar-sae)" /><span>{value.toFixed(2)}</span><Sparkline points={entry.sparkline} width={56} height={14} color="var(--pillar-sae)" /></div>
            {/snippet}
          </RackCard>
        {/each}
        {#if saeState.live}
          {#each discovery as feature (feature.id)}
            <RackCard accent="--pillar-sae">
              {#snippet statline()}
                <button class="glyph" disabled={featureBusy} onclick={() => void pin(feature.id)} title="pin">□</button>
                <span class="name">{featureLabel(feature.id, feature.label)}</span><span class="spacer"></span><span class="chip">L{info?.layer}</span>
              {/snippet}
              {#snippet body()}
                <div class="meter"><span>activation</span><Bar value={Math.max(feature.activation, 0)} max={Math.max(...(saeState.history.get(feature.id) ?? []), feature.activation, 1)} width={160} height={8} color="var(--pillar-sae)" /><span>{feature.activation.toFixed(2)}</span><Sparkline points={saeState.history.get(feature.id) ?? []} width={56} height={14} color="var(--pillar-sae)" /></div>
              {/snippet}
            </RackCard>
          {/each}
        {:else}
          <p class="hint">live off — pinned features settle to the end-of-generation activation</p>
        {/if}
      </div>
      <form class="add" onsubmit={addProbe}><input bind:value={featureInput} placeholder="feature id" /><button disabled={featureBusy}>+ pin</button></form>
    </section>
  {/if}
</div>

<style>
  .sae { display:flex; flex-direction:column; min-height:0; height:100%; overflow:hidden; }
  .identity { display:flex; gap:var(--space-3); align-items:center; padding:var(--space-3) var(--space-5); border-bottom:1px solid var(--border); color:var(--fg-muted); font-family:var(--font-mono); font-size:var(--text-xs); }
  .section { display:flex; flex-direction:column; gap:var(--space-3); padding:var(--space-5); border-bottom:1px solid var(--border); min-height:0; }
  .section.steer { flex:0 1 45%; } .section.probe { flex:1 1 0; overflow:hidden; }
  .empty { overflow:auto; }
  header { display:flex; align-items:center; gap:var(--space-3); border-bottom:1px solid var(--border); padding-bottom:var(--space-3); }
  .title { color:var(--pillar-sae); font-weight:var(--weight-bold); } .count,.hint { color:var(--fg-muted); font-size:var(--text-sm); }
  .scroll { display:flex; flex-direction:column; gap:var(--space-2); overflow-y:auto; min-height:0; }
  .add,.load-form { display:flex; gap:var(--space-2); } input { flex:1; min-width:0; background:var(--bg); color:var(--fg); border:1px solid var(--border); border-radius:var(--radius); padding:var(--space-2) var(--space-3); }
  button { color:var(--pillar-sae); background:transparent; border:1px solid var(--border); border-radius:var(--radius); cursor:pointer; padding:var(--space-1) var(--space-3); } button:disabled { opacity:.5; cursor:default; }
  .glyph,.remove { border:0; padding:0 var(--space-1); } .name { font-family:var(--font-mono); color:var(--fg-strong); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; } .spacer { flex:1; }
  .trigger,.toggle,.chip { font-size:var(--text-xs); } .toggle.on { color:var(--pillar-sae); border-color:var(--pillar-sae); }
  .alpha { display:grid; grid-template-columns:2em 1fr 3em; gap:var(--space-2); align-items:center; color:var(--fg-muted); }
  .meter { display:grid; grid-template-columns:minmax(4em,1fr) minmax(60px,2fr) 4em 56px; gap:var(--space-2); align-items:center; color:var(--fg-muted); font-size:var(--text-xs); }
  .error { color:var(--accent-red); }
</style>
