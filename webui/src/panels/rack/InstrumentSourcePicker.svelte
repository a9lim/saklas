<script lang="ts">
  // Shared source selector for the SAE and J-LENS pillars. The source string
  // shown here is the exact identifier their CLI/API ``use`` action accepts.

  import Select from "../../lib/Select.svelte";
  import Button from "../../lib/ui/Button.svelte";
  import type { InstrumentSourceJSON } from "../../lib/types";

  let {
    sources,
    value = $bindable(),
    busy = false,
    accent,
    onuse,
  }: {
    sources: InstrumentSourceJSON[];
    value: string;
    busy?: boolean;
    accent: string;
    onuse: (source: string) => void;
  } = $props();

  const options = $derived(sources.map((source) => ({
    value: source.source,
    label: `${source.active ? "● " : ""}${source.source}`,
  })));
  const selected = $derived(sources.find((source) => source.source === value));
  const active = $derived(sources.find((source) => source.active));
</script>

{#if sources.length > 0}
  <div class="source-row">
    <div class="source-select">
      <Select
        bind:value
        {options}
        ariaLabel="Artifact source"
        disabled={busy}
      />
    </div>
    <Button
      size="sm"
      variant="solid"
      {accent}
      disabled={busy || !value || selected?.active === true}
      onclick={() => onuse(value)}
    >
      {busy ? "switching…" : selected?.active ? "active" : "use source"}
    </Button>
  </div>
  <p class="source-note">
    {#if active}
      <code>{active.source}</code>
      · {active.kind === "local" ? "Saklas-owned artifact" : "provider-owned cache"}
    {:else}
      choose an available source
    {/if}
  </p>
{:else}
  <p class="source-note">no prepared sources yet</p>
{/if}

<style>
  .source-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .source-select {
    flex: 1 1 0;
    min-width: 0;
  }
  .source-note {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  code {
    color: var(--fg-dim);
    font-family: var(--font-mono);
  }
</style>
