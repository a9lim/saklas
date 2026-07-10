<script lang="ts">
  // The SAE instrument — fourth pillar, shipped ahead of its engine.  The
  // runtime (resident SAE, per-step top-k features, ``sae/<id>`` atoms +
  // probes + gates) is specced in docs/plans/sae-pillar.md and deliberately
  // deferred; this tab is the designed placeholder that gates on the
  // session-info capability flag once the server grows one.
  //
  // TODO(sae-runtime): when ``SessionInfo.sae_loaded`` exists, branch here —
  // loaded ⇒ the JLensPanel-shaped STEER + merged PROBE surface (gold
  // accent); unloaded ⇒ a release picker driving the background load route.
  // Until then the panel is honest about what exists today: the fit-time
  // ``:sae`` variant reachable through the manifold builder.

  import Button from "../lib/ui/Button.svelte";
  import Chip from "../lib/ui/Chip.svelte";
  import { openDrawer } from "../lib/stores.svelte";
</script>

<section class="sae" aria-label="SAE instrument">
  <div class="hero">
    <span class="glyph">▲</span>
    <h2>feature space</h2>
    <Chip color="var(--pillar-sae)">runtime: specced · deferred</Chip>
  </div>

  <p class="copy">
    The SAE instrument reads and steers the model through sparse-autoencoder
    features — the same three verbs as its siblings:
  </p>

  <ul class="verbs">
    <li>
      <span class="k">observe</span> live top-k feature activations per
      decode step, pinnable as feature probes
    </li>
    <li>
      <span class="k">steer</span> decoder-row atoms —
      <code>1.2 sae/9143</code> — composing with every other term
    </li>
    <li>
      <span class="k">gate</span> conditional triggers —
      <code>@when:sae/9143 &gt; 3.0</code>
    </li>
  </ul>

  <p class="copy dim">
    The engine work mirrors the J-lens architecture (resident release,
    readout-channel probes, reserved <code>sae/</code> namespace) and lands
    as its own release — the full sketch lives in
    <code>docs/plans/sae-pillar.md</code>.
  </p>

  <div class="today">
    <h3>available today</h3>
    <p class="copy dim">
      Fit-time SAE-space manifolds: reconstruct node centroids through an
      SAE before the fit (the <code>:sae</code> tensor variant).
    </p>
    <Button
      accent="var(--pillar-sae)"
      onclick={() => openDrawer("manifold_builder")}
    >
      build an SAE-space manifold…
    </Button>
  </div>
</section>

<style>
  .sae {
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
    padding: var(--space-6) var(--space-6);
    overflow-y: auto;
    min-height: 0;
  }

  .hero {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding-bottom: var(--space-4);
    border-bottom: 1px solid var(--border);
  }
  .glyph {
    color: var(--pillar-sae);
    font-size: var(--text-lg);
    text-shadow: 0 0 14px color-mix(in srgb, var(--pillar-sae) 40%, transparent);
  }
  h2 {
    margin: 0;
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
    flex: 1;
  }
  h3 {
    margin: 0 0 var(--space-3);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--fg-muted);
  }

  .copy {
    margin: 0;
    font-size: var(--text-sm);
    line-height: 1.6;
    color: var(--fg-dim);
    max-width: 46ch;
  }
  .copy.dim {
    color: var(--fg-subtle);
  }

  .verbs {
    margin: 0;
    padding: 0;
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .verbs li {
    font-size: var(--text-sm);
    line-height: 1.55;
    color: var(--fg-dim);
    padding-left: var(--space-5);
    position: relative;
  }
  .verbs li::before {
    content: "▸";
    position: absolute;
    left: 0;
    color: color-mix(in srgb, var(--pillar-sae) 65%, transparent);
  }
  .verbs .k {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--pillar-sae);
    margin-right: var(--space-2);
  }

  code {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--fg);
    background: var(--accent-subtle);
    padding: 1px 5px;
    border-radius: var(--radius-sm);
  }

  .today {
    border: 1px solid var(--glass-line);
    background: var(--glass);
    border-radius: var(--radius-lg);
    padding: var(--space-5) var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    align-items: flex-start;
  }
</style>
