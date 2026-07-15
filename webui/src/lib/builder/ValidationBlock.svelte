<script lang="ts">
  // Builder-drawer validation block — the "not ready to <verb>" header
  // plus a bulleted reasons list.  One consistent shape across extract
  // and manifold drawers; reasons come from each drawer's $derived
  // validation function.
  //
  // Renders nothing when ``messages`` is empty so callers can drop it in
  // unconditionally.

  interface Props {
    /** What action would run if validation passed — completes the
     *  "not ready to ____:" header sentence.  Examples: "extract",
     *  "build", "discover". */
    verb: string;
    messages: string[];
  }

  let { verb, messages }: Props = $props();
</script>

{#if messages.length > 0}
  <div class="sk-validation" role="alert">
    <p class="sk-validation-head">{verb} blocked</p>
    <ul>
      {#each messages as m (m)}
        <li>{m}</li>
      {/each}
    </ul>
  </div>
{/if}

<style>
  .sk-validation {
    border-left: 2px solid var(--accent-yellow);
    background: rgba(153, 135, 0, 0.06);
    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius);
    color: var(--fg-strong);
    font-size: var(--text-sm);
  }
  .sk-validation-head {
    margin: 0 0 var(--space-2);
    color: var(--accent-yellow);
    font-weight: var(--weight-medium);
  }
  .sk-validation ul {
    margin: 0;
    padding-left: var(--space-6);
    color: var(--fg-dim);
  }
  .sk-validation li {
    line-height: 1.4;
  }
</style>
