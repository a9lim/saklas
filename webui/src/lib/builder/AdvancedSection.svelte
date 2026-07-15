<script lang="ts">
  // Advanced-options collapsible — the builder-drawer wrapper around
  // ``Disclosure`` with a fixed "Advanced options" summary and a body
  // layout that matches the extract / manifold forms.  Used in both
  // drawers so the secondary-fields surface reads identically.

  import type { Snippet } from "svelte";
  import Disclosure from "../Disclosure.svelte";

  interface Props {
    expanded?: boolean;
    /** Override the default "Advanced options" summary text. */
    summary?: string;
    children: Snippet;
  }

  let {
    expanded = $bindable(false),
    summary = "advanced",
    children,
  }: Props = $props();
</script>

<div class="sk-advanced">
  <Disclosure bind:expanded {summary} flush>
    <div class="sk-advanced-body">
      {@render children()}
    </div>
  </Disclosure>
</div>

<style>
  .sk-advanced {
    /* Match the extract/manifold "field" rhythm — a small bottom margin
     * keeps the section anchored above the submit button. */
    margin: var(--space-3) 0;
  }
  .sk-advanced-body {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
</style>
