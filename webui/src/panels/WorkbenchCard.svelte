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
  <span class="sub" title={device}>{device}</span>
</section>

<style>
  .workbench {
    /* margin-top:auto pins the card to the column floor even when the
     * tree above it is short (empty / error states don't flex-grow).
     * Borderless — the gap above carries the separation.  Model id and
     * device/dtype sit on one baseline row (model left, device right)
     * rather than stacked. */
    margin-top: auto;
    flex: 0 0 auto;
    display: flex;
    flex-direction: row;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-4);
  }

  .sub {
    margin: 0;
    flex: 0 0 auto;
    padding: 1px 7px;
    border-radius: var(--radius-pill);
    background: var(--glass-strong);
    border: 1px solid transparent;
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
  }

  h2 {
    margin: 0;
    /* Take the row's slack and ellipsize so a long model id never
     * pushes the device readout off the right edge. */
    flex: 1 1 auto;
    min-width: 0;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.25;
    color: var(--fg-strong);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
