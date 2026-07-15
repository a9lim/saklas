<script lang="ts">
  // The living style guide — mounted at /styleguide (main.ts routes on
  // pathname; the server's SPA fallback already serves index.html there).
  // Every specimen below is the real component rendering real tokens, so
  // this page IS the design system's ground truth: if it looks wrong
  // here, it is wrong everywhere.
  //
  // Spec lineage: the locked direction board (2026-07-10) — Observatory+
  // (A + B-as-live-layer + gradients-as-material), the hue ontology, the
  // constant-hue alpha ramps, Recursive in both voices.

  import Slider from "../lib/Slider.svelte";
  import Select from "../lib/Select.svelte";
  import Checkbox from "../lib/Checkbox.svelte";
  import Radio from "../lib/Radio.svelte";
  import NumberInput from "../lib/NumberInput.svelte";
  import Disclosure from "../lib/Disclosure.svelte";
  import Bar from "../lib/charts/Bar.svelte";
  import Sparkline from "../lib/charts/Sparkline.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import Button from "../lib/ui/Button.svelte";
  import Chip from "../lib/ui/Chip.svelte";
  import SegmentedTabs from "../lib/ui/SegmentedTabs.svelte";
  import GlassCard from "../lib/ui/GlassCard.svelte";
  import LayerStrip from "../panels/rack/LayerStrip.svelte";
  import { scoreToRgb, type TintHue } from "../lib/tokens";

  // ---- the hue ontology, as data --------------------------------------
  const PILLARS = [
    { value: "subspace", label: "subspace", color: "var(--pillar-subspace)" },
    { value: "manifold", label: "manifold", color: "var(--pillar-manifold)" },
    { value: "sae", label: "sae", color: "var(--pillar-sae)" },
    { value: "lens", label: "lens", color: "var(--pillar-lens)" },
  ] as const;

  const ONTOLOGY = [
    {
      swatch: "var(--pillar-subspace)",
      hex: "#EDEFF7",
      name: "subspace · chrome",
      role: "flat geometry + chrome",
    },
    {
      swatch: "var(--pillar-manifold)",
      hex: "#A78BFA",
      name: "manifold",
      role: "curved geometry",
    },
    {
      swatch: "var(--pillar-sae)",
      hex: "#F2C94C",
      name: "sae",
      role: "feature space",
    },
    {
      swatch: "var(--pillar-lens)",
      hex: "#6BA6F8",
      name: "lens · surprise",
      role: "logit space + surprise",
    },
    {
      swatch: "var(--live)",
      hex: "#34D399",
      name: "live",
      role: "live state + positive pole",
    },
    {
      swatch: "var(--accent-red)",
      hex: "#E5544F",
      name: "error",
      role: "errors + negative pole",
    },
  ];

  // ---- live demo state -------------------------------------------------
  let pillar = $state<(typeof PILLARS)[number]["value"]>("manifold");
  const pillarColor = $derived(
    PILLARS.find((p) => p.value === pillar)?.color ?? "var(--accent)",
  );

  // A fake probe reading that drifts like a real one — feeds the
  // streaming sparkline, the fraction bar, and the layer strip so the
  // card specimen visibly LIVES (the point of the motion system).
  let stream = $state<number[]>([0.31, 0.35, 0.3, 0.42, 0.4]);
  let streaming = $state(true);
  $effect(() => {
    if (!streaming) return;
    const id = setInterval(() => {
      const prev = stream[stream.length - 1] ?? 0.4;
      const next = Math.max(
        -0.9,
        Math.min(0.9, prev + (Math.random() - 0.48) * 0.22),
      );
      stream.push(next);
      if (stream.length > 48) stream.shift();
    }, 400);
    return () => clearInterval(id);
  });
  const current = $derived(stream[stream.length - 1] ?? 0);

  const LAYER_STRIP = [4, 8, 14, 22, 40, 66, 88, 74, 52, 61, 38, 20, 12, 6];
  const STYLE_LAYER_CELLS = LAYER_STRIP.map((value, layer) => ({
    layer,
    value,
    title: `L${layer} · ${value}%`,
  }));

  // Tint playground: one slider score, both ramps, over a sample line.
  let tintScore = $state(0.55);
  let tintHue = $state<TintHue>("signed");
  const SAMPLE_TOKENS = [
    { t: "The", w: 0.18 },
    { t: "tide", w: 0.44 },
    { t: "forgives", w: 0.7 },
    { t: "no", w: 0.26 },
    { t: "schedule", w: 0.55 },
    { t: ",", w: 0 },
    { t: "it", w: -0.3 },
    { t: "keeps", w: -0.55 },
    { t: "its own", w: 0.8 },
    { t: "ledger", w: 1.0 },
  ];

  // Control specimens (bound so they demonstrably work).
  let selVal = $state("wistful");
  let chkVal = $state(true);
  let radVal = $state("pca");
  let numVal = $state<number | null>(0.5);
  let discOpen = $state(false);
  let drawKey = $state(0); // re-keys the draw-in sparkline to replay it
</script>

<div class="guide">
  <div class="wrap">
    <header>
      <div class="eyebrow">saklas · design system</div>
      <h1><span class="wordmark">saklas</span> / design</h1>
      <p class="lede">
        Live components and tokens.
      </p>
    </header>

    <!-- ================= ontology ================= -->
    <section>
      <div class="sec-head">
        <span class="eyebrow">01 · color</span>
        <h2>Hue by space</h2>
      </div>
      <div class="onto">
        {#each ONTOLOGY as o (o.name)}
          <div class="onto-row">
            <span class="sw" style:background={o.swatch}></span>
            <span class="onto-name">{o.name} <code>{o.hex}</code></span>
            <span class="onto-role">{o.role}</span>
          </div>
        {/each}
      </div>
      <p class="note">
        Achromatic chrome · hue marks data space · gradients mark depth or time.
      </p>
    </section>

    <!-- ================= type ================= -->
    <section>
      <div class="sec-head">
        <span class="eyebrow">02 · type</span>
        <h2>Recursive, both voices</h2>
      </div>
      <div class="type-rows">
        <div class="type-row">
          <span class="type-role">chrome — Recursive Sans (MONO 0, CASL .35)</span>
          <span class="type-sample sans">Instruments — subspace · manifold · sae · lens</span>
        </div>
        <div class="type-row">
          <span class="type-role">data — Recursive Mono (MONO 1)</span>
          <span class="type-sample mono">0.7 emotions%0.3,0.8,0 @when:jlens/fake &gt; 0.01</span>
        </div>
        <div class="type-row">
          <span class="type-role">labels — tracked caps</span>
          <span class="type-sample caps">probe · live · @0.54 ±0.11</span>
        </div>
        <div class="type-row">
          <span class="type-role">weights — wght 420 / 560 / 680</span>
          <span class="type-sample sans">
            <span style:font-weight="var(--weight-normal)">normal</span>
            &nbsp;·&nbsp;
            <span style:font-weight="var(--weight-medium)">medium</span>
            &nbsp;·&nbsp;
            <span style:font-weight="var(--weight-bold)">bold</span>
          </span>
        </div>
      </div>
    </section>

    <!-- ================= surfaces ================= -->
    <section>
      <div class="sec-head">
        <span class="eyebrow">03 · surfaces</span>
        <h2>Elevation</h2>
      </div>
      <div class="tiles">
        <div class="tile" style:background="var(--bg-deep)"><span>--bg-deep</span></div>
        <div class="tile" style:background="var(--bg)"><span>--bg</span></div>
        <div class="tile" style:background="var(--bg-alt)"><span>--bg-alt</span></div>
        <div class="tile" style:background="var(--bg-elev)"><span>--bg-elev</span></div>
        <div class="tile" style:background="var(--surface-hi)"><span>--surface-hi</span></div>
      </div>
      <div class="card-states">
        <GlassCard>
          <div class="cs-label">glass · idle</div>
        </GlassCard>
        <GlassCard accent="var(--pillar-manifold)" active>
          <div class="cs-label">glass · active</div>
        </GlassCard>
        <GlassCard disabled>
          <div class="cs-label">glass · disabled</div>
        </GlassCard>
      </div>
    </section>

    <!-- ================= primitives ================= -->
    <section>
      <div class="sec-head">
        <span class="eyebrow">04 · primitives</span>
        <h2>Controls</h2>
      </div>

      <div class="spec-row">
        <span class="spec-label">button</span>
        <div class="spec-items">
          <Button variant="solid">generate</Button>
          <Button>browse manifolds</Button>
          <Button variant="danger">delete</Button>
          <Button size="sm">re-fit</Button>
          <Button size="sm" title="Saklas tooltip">tooltip</Button>
          <Button size="sm" accent="var(--pillar-sae)">+ pin feature</Button>
          <Button size="sm" accent="var(--pillar-lens)">+ steer token</Button>
          <Button disabled>unavailable</Button>
        </div>
      </div>

      <div class="spec-row">
        <span class="spec-label">chip</span>
        <div class="spec-items">
          <Chip color="var(--pillar-subspace)">0.5 personas%pirate</Chip>
          <Chip color="var(--pillar-manifold)" onremove={() => {}}>0.7 emotions%0.3,0.8,0</Chip>
          <Chip color="var(--pillar-sae)">1.2 sae/9143</Chip>
          <Chip color="var(--pillar-lens)">0.3 jlens/orange</Chip>
          <Chip color="var(--pillar-manifold)" muted>0.4 months%january</Chip>
          <Chip>@response</Chip>
        </div>
      </div>

      <div class="spec-row">
        <span class="spec-label">tabs</span>
        <div class="spec-items grow">
          <SegmentedTabs
            items={PILLARS.map((p) => ({ ...p }))}
            bind:value={pillar}
          />
        </div>
      </div>

      <div class="spec-row">
        <span class="spec-label">controls</span>
        <div class="spec-items controls">
          <div class="ctl"><Slider value={0.5} ariaLabel="specimen slider" /></div>
          <Select
            bind:value={selVal}
            options={[
              { value: "wistful", label: "wistful" },
              { value: "elated", label: "elated" },
              { value: "furious", label: "furious" },
            ]}
            ariaLabel="specimen select"
          />
          <Checkbox bind:checked={chkVal} label="fit now" />
          <Radio bind:group={radVal} value="pca" label="pca" name="sg-fit" />
          <Radio bind:group={radVal} value="spectral" label="spectral" name="sg-fit" />
          <NumberInput bind:value={numVal} min={0} max={2} step={0.05} ariaLabel="specimen number" />
        </div>
      </div>

      <div class="spec-row">
        <span class="spec-label">disclosure</span>
        <div class="spec-items grow">
          <Disclosure bind:expanded={discOpen} summary="advanced">
            <p class="note" style:margin="8px 0 0">
              max_dim · var_threshold · k_nn · bandwidth
            </p>
          </Disclosure>
        </div>
      </div>
    </section>

    <!-- ================= ramps ================= -->
    <section>
      <div class="sec-head">
        <span class="eyebrow">05 · ramps</span>
        <h2>Hue × strength</h2>
      </div>
      <div class="ramp-play">
        <div class="ramp-controls">
          <div class="ctl">
            <Slider bind:value={tintScore} min={-1} max={1} step={0.01} ariaLabel="tint score" />
          </div>
          <span class="mono readout">{tintScore >= 0 ? "+" : ""}{tintScore.toFixed(2)}</span>
          <Radio bind:group={tintHue} value={"signed" as TintHue} label="probe (signed)" name="sg-hue" />
          <Radio bind:group={tintHue} value={"surprise" as TintHue} label="surprise (blue)" name="sg-hue" />
        </div>
        <div class="stream mono">
          {#each SAMPLE_TOKENS as tok (tok.t)}
            <span
              class="tok"
              style:background-color={scoreToRgb(tok.w * tintScore, 0.5, tintHue)}
            >{tok.t}</span>{" "}
          {/each}
          <span class="caret"></span>
        </div>
        <div class="ramp-scales">
          <div class="scale">
            <span>−pole</span>
            <div class="bar signed"></div>
            <span>+pole</span>
          </div>
          <div class="scale">
            <span>expected</span>
            <div class="bar surprise"></div>
            <span>surprising</span>
          </div>
        </div>
      </div>
    </section>

    <!-- ================= charts + card grammar ================= -->
    <section>
      <div class="sec-head">
        <span class="eyebrow">06 · data</span>
        <h2>Charts + cards</h2>
      </div>
      <div class="data-grid">
        <div class="data-col">
          <div class="spec-row">
            <span class="spec-label">sparkline</span>
            <div class="spec-items">
              {#key drawKey}
                <Sparkline points={stream} width={160} height={28} color="var(--accent-green)" />
              {/key}
              <Button size="sm" onclick={() => (drawKey += 1)}>replay draw-in</Button>
              <Button size="sm" onclick={() => (streaming = !streaming)}>
                {streaming ? "pause stream" : "resume stream"}
              </Button>
            </div>
          </div>
          <div class="spec-row">
            <span class="spec-label">bar</span>
            <div class="spec-items">
              <Bar value={Math.abs(current)} max={1} width={140} color="#a78bfa" />
              <Bar value={current} max={1} width={140} bipolar color="#34d399" />
            </div>
          </div>
          <div class="spec-row">
            <span class="spec-label">heatmap</span>
            <div class="spec-items">
              {#each [-0.5, -0.31, -0.12, 0, 0.18, 0.4, 0.5] as v (v)}
                <HeatmapCell value={v} showValue size={40} />
              {/each}
            </div>
          </div>
        </div>

        <!-- The specimen card: statline on top, meters below, per-layer
             strip at the foot — the one grammar every pillar's cards
             wear, accent driven by the tabs in §04. -->
        <GlassCard accent={pillarColor} active={streaming}>
          <div class="spec-card">
            <div class="statline">
              <span class="glyph" style:color={pillarColor}>◆</span>
              <span class="name">emotions</span>
              <Chip color={pillarColor}>@0.54 ±0.11</Chip>
              <span class="spacer"></span>
              <Sparkline points={stream} width={72} height={20} color={pillarColor} />
            </div>
            <div class="meter mono">
              <span class="k">fraction</span>
              <div class="meter-bar">
                <div
                  class="meter-fill"
                  style:width="{Math.round(Math.abs(current) * 100)}%"
                  style:--mf={pillarColor}
                ></div>
              </div>
              <span class="v">{Math.abs(current).toFixed(2)}</span>
            </div>
            <div class="meter mono">
              <span class="k">nearest</span>
              <span class="v-dim">wistful · d 0.8 spacings</span>
            </div>
            <LayerStrip
              cells={STYLE_LAYER_CELLS}
              scale={100}
              ariaLabel="Per-layer share specimen"
            />
          </div>
        </GlassCard>
      </div>
    </section>

    <!-- ================= motion ================= -->
    <section>
      <div class="sec-head">
        <span class="eyebrow">07 · motion</span>
        <h2>Live motion</h2>
      </div>
      <div class="spec-row">
        <span class="spec-label">live</span>
        <div class="spec-items">
          <span class="live-dot"></span>
          <span class="caret"></span>
          <span class="note-inline">
            <code>--live</code> · <code>--ease-spring</code> · reduced motion
          </span>
        </div>
      </div>
    </section>

    <footer>
      <span class="mono">saklas design system · phase 1 · tokens v2 · /styleguide</span>
    </footer>
  </div>
</div>

<style>
  .guide {
    position: fixed;
    inset: 0;
    overflow-y: auto;
  }
  .wrap {
    max-width: 1060px;
    margin: 0 auto;
    padding: 56px 32px 96px;
  }

  .eyebrow {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--fg-muted);
  }
  h1 {
    font-size: 34px;
    font-weight: var(--weight-medium);
    letter-spacing: -0.01em;
    margin: 12px 0 10px;
  }
  /* The one text-gradient moment in the whole system. */
  .wordmark {
    font-family: var(--font-mono);
    background: linear-gradient(
      100deg,
      var(--pillar-subspace),
      var(--pillar-manifold) 40%,
      var(--pillar-lens) 75%,
      var(--pillar-sae)
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .lede {
    color: var(--fg-dim);
    max-width: 62ch;
    margin: 0;
    line-height: 1.6;
  }

  section {
    margin-top: 64px;
  }
  .sec-head {
    display: flex;
    align-items: baseline;
    gap: 16px;
    padding-bottom: 10px;
    margin-bottom: 22px;
  }
  .sec-head .eyebrow {
    flex: none;
  }
  h2 {
    font-size: var(--text-lg);
    font-weight: var(--weight-medium);
    margin: 0;
  }
  .note {
    color: var(--fg-dim);
    max-width: 68ch;
    line-height: 1.6;
    margin: 18px 0 0;
  }
  .note-inline {
    color: var(--fg-dim);
    font-size: var(--text-sm);
    max-width: 52ch;
  }
  code {
    font-size: var(--text-sm);
    color: var(--fg);
    background: var(--accent-subtle);
    padding: 1px 5px;
    border-radius: var(--radius-sm);
  }
  .mono {
    font-family: var(--font-mono);
  }

  /* ontology */
  .onto {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .onto-row {
    display: grid;
    grid-template-columns: 56px 260px 1fr;
    gap: 16px;
    align-items: center;
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: 10px 14px;
  }
  .sw {
    width: 40px;
    height: 24px;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
  }
  .onto-name {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
  }
  .onto-name code {
    font-size: var(--text-2xs);
    color: var(--fg-muted);
    background: none;
    padding: 0;
  }
  .onto-role {
    font-size: var(--text-sm);
    color: var(--fg-dim);
  }

  /* type */
  .type-rows {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .type-row {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 16px;
    align-items: baseline;
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: 12px 14px;
  }
  .type-role {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--fg-muted);
  }
  .type-sample.sans {
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  .type-sample.mono {
    font-family: var(--font-mono);
    font-size: var(--text);
  }
  .type-sample.caps {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--fg-dim);
  }

  /* surfaces */
  .tiles {
    display: flex;
    gap: 8px;
    margin-bottom: 14px;
  }
  /* Kept bordered by design: this demonstrates the --bg/--bg-deep surface
   * tokens themselves, one of which matches the page's own canvas — the
   * hairline is the only thing that makes that swatch legible as a tile
   * rather than invisible against the page. */
  .tile {
    flex: 1;
    height: 64px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    display: flex;
    align-items: flex-end;
    padding: 8px;
  }
  .tile span {
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    color: var(--fg-muted);
  }
  .card-states {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  .cs-label {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--fg-dim);
  }

  /* spec rows — borderless; padding alone carries the rhythm between
   * static documentation rows (no hover wash needed here). */
  .spec-row {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 12px 0;
  }
  .spec-label {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--fg-muted);
    min-width: 92px;
    flex: none;
  }
  .spec-items {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .spec-items.grow {
    flex: 1;
  }
  .spec-items.controls {
    row-gap: 14px;
  }
  .ctl {
    width: 160px;
  }

  /* ramps */
  .ramp-play {
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius-lg);
    padding: 18px 20px;
  }
  .ramp-controls {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 14px;
    flex-wrap: wrap;
  }
  .readout {
    font-size: var(--text-sm);
    color: var(--fg);
    min-width: 52px;
    font-variant-numeric: tabular-nums;
  }
  .stream {
    font-size: var(--text);
    line-height: 2;
  }
  .tok {
    border-radius: var(--radius-sm);
    padding: 1px 3px;
  }
  .ramp-scales {
    display: flex;
    gap: 24px;
    margin-top: 14px;
  }
  .scale {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .scale span {
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    color: var(--fg-muted);
  }
  .scale .bar {
    flex: 1;
    height: 8px;
    border-radius: var(--radius-pill);
  }
  .scale .bar.signed {
    background: linear-gradient(
      90deg,
      rgba(229, 84, 79, 0.65),
      rgba(229, 84, 79, 0) 49%,
      rgba(52, 211, 153, 0) 51%,
      rgba(52, 211, 153, 0.65)
    );
  }
  .scale .bar.surprise {
    background: linear-gradient(
      90deg,
      rgba(107, 166, 248, 0),
      rgba(107, 166, 248, 0.7)
    );
  }

  /* data grid + specimen card */
  .data-grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 20px;
    align-items: start;
  }
  .data-col {
    display: flex;
    flex-direction: column;
  }
  .spec-card .statline {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
  }
  .spec-card .glyph {
    font-size: var(--text-sm);
  }
  .spec-card .name {
    font-weight: var(--weight-medium);
  }
  .spec-card .spacer {
    flex: 1;
  }
  .meter {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
    font-size: var(--text-sm);
  }
  .meter .k {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    min-width: 52px;
  }
  .meter-bar {
    flex: 1;
    height: 7px;
    border-radius: var(--radius-pill);
    background: var(--glass-strong);
    overflow: hidden;
  }
  /* Lit material: sheen runs ACROSS the value axis (vertical), value
   * edge stays honest; width transitions spring so live readings move. */
  .meter-fill {
    height: 100%;
    border-radius: var(--radius-pill);
    background: linear-gradient(
      180deg,
      color-mix(in srgb, var(--mf) 80%, white),
      var(--mf)
    );
    box-shadow: 0 0 10px color-mix(in srgb, var(--mf) 30%, transparent);
    transition: width var(--dur-slow) var(--ease-spring);
  }
  .meter .v {
    min-width: 44px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .meter .v-dim {
    color: var(--fg-dim);
    font-size: var(--text-sm);
  }
  /* motion */
  .live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--live);
    animation: sg-pulse 1.6s ease-in-out infinite;
  }
  @keyframes sg-pulse {
    0%,
    100% {
      box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.45);
    }
    50% {
      box-shadow: 0 0 0 6px transparent;
    }
  }
  .caret {
    display: inline-block;
    width: 8px;
    height: 15px;
    border-radius: 2px;
    background: var(--live);
    vertical-align: text-bottom;
    box-shadow: var(--glow-live);
    animation: sg-blink 1.1s steps(2, start) infinite;
  }
  @keyframes sg-blink {
    to {
      visibility: hidden;
    }
  }

  footer {
    margin-top: 64px;
    padding-top: 16px;
  }
  footer .mono {
    font-size: var(--text-xs);
    color: var(--fg-muted);
  }
</style>
