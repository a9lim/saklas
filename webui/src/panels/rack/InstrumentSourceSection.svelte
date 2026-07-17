<script lang="ts">
  // Canonical SOURCE section for the SAE and J-LENS pillars. Prepared and
  // provider-backed artifacts share one selector. Local authoring is an
  // explicit selector mode: its fields stay out of the way until chosen, and
  // the source-row action becomes the pillar's fit/train action.

  import type { Snippet } from "svelte";
  import type { InstrumentSourceJSON } from "../../lib/types";
  import Select from "../../lib/Select.svelte";
  import Button from "../../lib/ui/Button.svelte";
  import RackSectionHeader from "./RackSectionHeader.svelte";

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
    selectionCurrent = true,
    onuse,
    providerOptions,
    providerPlaceholder = "provider source",
    onfetch,
    localControls,
    localActionLabel,
    localActionDisabled = false,
    onlocal,
    sourceControls,
    progress,
    warning,
    messages,
  }: {
    ready: boolean;
    sources: InstrumentSourceJSON[];
    value: string;
    busy?: boolean;
    accent: string;
    sourceError?: string | null;
    working?: boolean;
    /** Whether secondary source settings match the resident runtime. */
    selectionCurrent?: boolean;
    onuse: (source: string) => void;
    providerOptions: ProviderOption[];
    providerPlaceholder?: string;
    onfetch: (source: string) => void;
    localControls: Snippet;
    localActionLabel: string;
    localActionDisabled?: boolean;
    onlocal: () => void;
    /** Optional controls that sit directly below the source picker. */
    sourceControls?: Snippet;
    progress?: Snippet;
    warning?: Snippet;
    messages?: Snippet;
  } = $props();

  const options = $derived.by(() => {
    const providers = new Map(providerOptions.map((option) => [option.value, option]));
    const result: ProviderOption[] = sources.map((source) => ({
      value: source.source,
      label: providers.get(source.source)?.label ?? source.source,
    }));
    const prepared = new Set(sources.map((source) => source.source));
    for (const option of providerOptions) {
      if (!prepared.has(option.value)) result.push(option);
    }
    if (!prepared.has("local")) result.push({ value: "local", label: "local" });
    return result;
  });
  const localSelected = $derived(value === "local");
  const selectedSource = $derived(sources.find((source) => source.source === value));
  const selectedProviderOption = $derived(
    providerOptions.find((option) => option.value === value),
  );
  const selectedOption = $derived(options.find((option) => option.value === value));

  function applySource(): void {
    if (!value) return;
    if (localSelected) {
      onlocal();
      return;
    }
    if (selectedSource) {
      // A prepared external source and its provider intentionally share one
      // identifier (for example ``neuronpedia``).  Treat it as prepared first;
      // only re-enter the provider fetch when the active binding is not usable.
      if (selectedSource.active && !ready && selectedProviderOption) onfetch(value);
      else onuse(value);
    }
    else onfetch(value);
  }
</script>

<section class="source-section">
  <RackSectionHeader title="SOURCE" count={ready ? "ready" : "required"} />

  {#if sourceError}
    <p class="source-error" role="alert">{sourceError}</p>
  {/if}

  {#if working && progress}
    <div class="progress">{@render progress()}</div>
  {:else}
    <div class="setup-stack">
      <div class="setup-group">
        <div class="setup-row source-row">
          <div class="setup-controls">
            <Select
              bind:value
              {options}
              placeholder={providerPlaceholder}
              disabled={busy || options.length === 0}
              ariaLabel="Artifact source"
            />
          </div>
          <div class="setup-action">
            <Button
              size="sm"
              variant="solid"
              {accent}
              disabled={busy || !value ||
                (localSelected
                  ? localActionDisabled
                  : selectedOption?.disabled === true ||
                    (selectedSource?.active === true &&
                      selectionCurrent &&
                      (ready || selectedProviderOption === undefined)))}
              onclick={applySource}
            >
              {busy
                ? "working…"
                : localSelected
                  ? localActionLabel
                  : selectedSource?.active
                  ? ready && selectionCurrent
                    ? "active"
                    : ready
                    ? "use"
                    : selectedProviderOption
                    ? "repair"
                      : "unavailable"
                  : selectedSource
                    ? "use"
                    : "fetch"}
            </Button>
          </div>
        </div>
        {#if sourceControls}
          <div class="setup-row supplemental-row">
            <div class="setup-controls">{@render sourceControls()}</div>
          </div>
        {/if}
      </div>
      {#if localSelected}
        <div class="setup-group local-group">
          <span class="setup-label">local</span>
          <div class="setup-row local-row">
          <div class="setup-controls">{@render localControls()}</div>
          </div>
        </div>
      {/if}
    </div>
  {/if}

  {#if warning}
    <div class="warning">{@render warning()}</div>
  {/if}

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
  .setup-group,
  .progress,
  .warning,
  .messages {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .setup-group {
    gap: var(--space-1);
  }
  .setup-label {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-family: var(--font-ui);
    font-weight: var(--weight-medium);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .setup-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: stretch;
    gap: var(--space-2);
    min-width: 0;
  }
  .local-row {
    grid-template-columns: minmax(0, 1fr);
  }
  .supplemental-row {
    grid-template-columns: minmax(0, 1fr);
  }
  .setup-controls {
    display: flex;
    align-items: stretch;
    gap: var(--space-2);
    min-width: 0;
  }
  .setup-action {
    display: flex;
    align-items: flex-end;
  }
  .setup-action :global(button) {
    height: var(--control-compact);
  }
  .setup-controls :global(input),
  .setup-controls :global(.source-field) {
    min-width: 0;
  }
  .setup-controls :global(.setup-field) {
    display: flex;
    flex: 1 1 0;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .setup-controls :global(.setup-field-narrow) {
    flex: 0.65 1 4.5rem;
  }
  .setup-controls :global(.setup-field-medium) {
    flex: 0.85 1 6.5rem;
  }
  .setup-controls :global(.setup-field-wide) {
    flex: 1.3 1 8rem;
  }
  .setup-controls :global(.setup-field-label) {
    color: var(--fg-muted);
    font-family: var(--font-ui);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    letter-spacing: 0.06em;
    line-height: 1;
    text-transform: uppercase;
  }
  .setup-controls :global(.setup-field input) {
    box-sizing: border-box;
    min-height: var(--control-compact);
    width: 100%;
  }
  .source-error {
    margin: 0;
    color: var(--accent-red);
    font-size: var(--text-sm);
  }
  /* Optional snippets often render no DOM at all. Empty flex children still
     consumed a section gap, producing the large SOURCE→STEER void. */
  .warning:empty,
  .messages:empty {
    display: none;
  }
</style>
