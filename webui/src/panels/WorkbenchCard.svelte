<script lang="ts">
  // Active-workbench card — model id and device/dtype.  Lives at the
  // bottom of the threads column.  The tok/s · ppl · tree meters were
  // removed as redundant: the status footer already carries t/s and ppl.

  import { sessionState } from "../lib/stores.svelte";

  const model = $derived(sessionState.info?.model_id ?? "no session");
  const device = $derived(
    sessionState.info
      ? `${sessionState.info.device}/${sessionState.info.dtype}`
      : "offline",
  );
</script>

<section class="workbench" aria-label="Active workbench">
  <h2 title={model}>{model}</h2>
  <p class="sub">{device}</p>
</section>

<style>
  .workbench {
    /* margin-top:auto pins the card to the column floor even when the
     * tree above it is short (empty / error states don't flex-grow).
     * Flat — border-top is the separator, no own fill. */
    margin-top: auto;
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--border);
  }

  .sub {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0;
  }

  h2 {
    margin: var(--space-1) 0 0;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.25;
    color: var(--fg-strong);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
