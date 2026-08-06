<script lang="ts">
  // The method-specific hyperparameter fields that live inside an
  // advanced flyout.  Which ones show follows ``tuning.fitMode``: the
  // variance threshold belongs to the pca layout selection, k-NN and
  // bandwidth to the spectral embedding, and ``auto`` accepts the union.

  import NumberInput from "../../lib/NumberInput.svelte";
  import type { DiscoverTuning } from "./shared";

  let { tuning }: { tuning: DiscoverTuning } = $props();
</script>

<div class="grid2">
  <label class="field">
    <span class="label">max dim</span>
    <NumberInput
      value={tuning.maxDim}
      min={1}
      step={1}
      oninput={(v) => {
        if (v !== null) tuning.maxDim = v;
      }}
    />
  </label>
  {#if tuning.fitMode === "pca" || tuning.fitMode === "auto"}
    <label class="field">
      <span class="label">variance</span>
      <NumberInput
        value={tuning.varThreshold}
        min={0}
        max={1}
        step={0.05}
        oninput={(v) => {
          if (v !== null) tuning.varThreshold = v;
        }}
      />
    </label>
  {:else}
    <label class="field">
      <span class="label">k-NN</span>
      <NumberInput
        value={tuning.kNN}
        min={1}
        step={1}
        allowEmpty
        placeholder="max(5, ⌈log K⌉)"
        oninput={(v) => { tuning.kNN = v; }}
      />
    </label>
  {/if}
</div>
{#if tuning.fitMode === "spectral"}
  <label class="field">
    <span class="label">bandwidth σ</span>
    <NumberInput
      value={tuning.bandwidth}
      min={0}
      step={0.01}
      allowEmpty
      placeholder="median(k-NN edges)"
      oninput={(v) => { tuning.bandwidth = v; }}
    />
  </label>
{/if}
