<script lang="ts">
  // Canonical SOURCE section for the SAE and J-LENS pillars. Lifecycle
  // semantics differ, but the information hierarchy and row geometry do not.

  import type { Snippet } from "svelte";
  import type { InstrumentSourceJSON } from "../../lib/types";
  import Select from "../../lib/Select.svelte";
  import Button from "../../lib/ui/Button.svelte";
  import RackSectionHeader from "./RackSectionHeader.svelte";
  import InstrumentSourcePicker from "./InstrumentSourcePicker.svelte";

  interface ProviderOption {
    value: string;
    label: string;
    disabled?: boolean;
  }

  let {
    ready,
    sources,
    value = $bindable(),
    busy = false,
    accent,
    sourceError = null,
    working = false,
    onuse,
    providerValue = $bindable(),
    providerOptions,
    providerPlaceholder = "provider source",
    onfetch,
    localControls,
    localAction,
    progress,
    warning,
    summary,
    messages,
  }: {
    ready: boolean;
    sources: InstrumentSourceJSON[];
    value: string;
    busy?: boolean;
    accent: string;
    sourceError?: string | null;
    working?: boolean;
    onuse: (source: string) => void;
    providerValue: string;
    providerOptions: ProviderOption[];
    providerPlaceholder?: string;
    onfetch: (source: string) => void;
    localControls: Snippet;
    localAction: Snippet;
    progress?: Snippet;
    warning?: Snippet;
    summary: Snippet;
    messages?: Snippet;
  } = $props();
</script>

<section class="source-section">
  <RackSectionHeader title="SOURCE" count={ready ? "ready" : "required"} />
  <InstrumentSourcePicker
    {sources}
    bind:value
    {busy}
    {accent}
    {onuse}
  />

  {#if sourceError}
    <p class="source-error" role="alert">{sourceError}</p>
  {/if}

  {#if working && progress}
    <div class="progress">{@render progress()}</div>
  {:else}
    <div class="setup-stack">
      <div class="setup-row provider-row">
        <div class="setup-controls">
          <Select
            bind:value={providerValue}
            options={providerOptions}
            placeholder={providerPlaceholder}
            disabled={busy || providerOptions.length === 0}
            ariaLabel="Provider source"
          />
        </div>
        <div class="setup-action">
          <Button
            size="sm"
            variant="solid"
            {accent}
            disabled={busy || !providerValue}
            onclick={() => onfetch(providerValue)}
          >fetch provider</Button>
        </div>
      </div>
      <div class="setup-row local-row">
        <div class="setup-controls">{@render localControls()}</div>
        <div class="setup-action">{@render localAction()}</div>
      </div>
    </div>
  {/if}

  {#if warning}
    <div class="warning">{@render warning()}</div>
  {/if}

  <div class="summary">{@render summary()}</div>

  {#if messages}
    <div class="messages">{@render messages()}</div>
  {/if}
</section>

<style>
  .source-section {
    display: flex;
    flex: 0 0 auto;
    flex-direction: column;
    gap: var(--space-3);
    min-width: 0;
    padding: var(--space-5) var(--space-5) var(--space-3);
  }
  .setup-stack,
  .progress,
  .warning,
  .summary,
  .messages {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .setup-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: stretch;
    gap: var(--space-2);
    min-width: 0;
  }
  .setup-controls {
    display: flex;
    align-items: stretch;
    gap: var(--space-2);
    min-width: 0;
  }
  .setup-action {
    display: flex;
    align-items: stretch;
  }
  .setup-action :global(button) {
    height: 100%;
  }
  .setup-controls :global(input),
  .setup-controls :global(.source-field) {
    min-width: 0;
  }
  .source-error {
    margin: 0;
    color: var(--accent-red);
    font-size: var(--text-sm);
  }
</style>
