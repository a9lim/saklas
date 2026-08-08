<script lang="ts">
  // The flat / curved / auto choice, shared by every tab that routes
  // through a discover fit.  ``tuning`` is a $state object owned by the
  // form, so writing ``tuning.fitMode`` here propagates back.

  import Radio from "../../lib/Radio.svelte";
  import type { DiscoverTuning } from "./shared";

  let {
    tuning,
    /** ``spectral``'s note differs where node counts are the practical
     *  constraint (the generated tab) from where they aren't. */
    spectralNote = "curved",
  }: { tuning: DiscoverTuning; spectralNote?: string } = $props();
</script>

<section class="step">
  <h2 class="step-title">fit method</h2>
  <div class="radio-row">
    <Radio bind:group={tuning.fitMode} value="auto" label="auto" />
    <Radio bind:group={tuning.fitMode} value="pca" label="pca" />
    <Radio bind:group={tuning.fitMode} value="spectral" label="spectral" />
  </div>
  <p class="dim-note">
    {#if tuning.fitMode === "auto"}
      choose per model
    {:else if tuning.fitMode === "pca"}
      flat
    {:else}
      {spectralNote}
    {/if}
  </p>
</section>
