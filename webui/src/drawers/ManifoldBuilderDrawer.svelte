<script lang="ts">
  // Manifold authoring shell — reached from "+ build manifold" in the
  // rack drawer.
  //
  // Three disjoint authoring paths share this drawer, and the only thing
  // they genuinely share is the identity (namespace / name / description)
  // and the mode tabs; everything below is per-path, so each is its own
  // sibling under ``drawers/manifold/``:
  //
  //   * auto      — DiscoverForm: hand the model a flat concept list, the
  //                 generator writes per-concept corpora, the fitter
  //                 derives coords per-model.  No coords, no domain.
  //   * template  — TemplatedForm: derive a manifold from a standalone
  //                 template (slot + values + multi-turn contexts).  The
  //                 tool for categories one references rather than
  //                 embodies (days, months, …).  Deterministic, no model.
  //   * custom    — AuthoredForm: user-supplied corpora, with coordinates
  //                 either hand-placed on a picked domain or derived by
  //                 the fitter (the ``auto-domain`` switch).
  //
  // All three build the same on-disk manifold artifact; the path shows up
  // as ``manifold.json::fit_mode``.  Inspector + steering are unchanged
  // from there on.
  //
  // The shared field/step/node styling lives in ``manifold/form.css``,
  // scoped under the ``.mb-form`` class on the body below, so the three
  // forms read as one surface without each carrying a copy.

  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import ModeTabs from "../lib/builder/ModeTabs.svelte";
  import { closeDrawer, openDrawer } from "../lib/stores.svelte";
  import AuthoredForm from "./manifold/AuthoredForm.svelte";
  import DiscoverForm from "./manifold/DiscoverForm.svelte";
  import TemplatedForm from "./manifold/TemplatedForm.svelte";
  import type { ManifoldIdentity } from "./manifold/shared";
  import "./manifold/form.css";

  let { params: _params }: { params?: unknown } = $props();
  $effect(() => { void _params; });

  type AuthoringMode = "authored" | "discover" | "templated";
  let authoringMode: AuthoringMode = $state("authored");

  const identity: ManifoldIdentity = $state({
    namespace: "local",
    name: "",
    description: "",
  });

  function cancel(): void {
    closeDrawer();
    openDrawer("manifolds");
  }
</script>

<section class="drawer-shell" aria-label="Build manifold">
  <header class="header">
    <span class="title">build manifold</span>
    <DrawerCloseButton onclick={cancel} />
  </header>

  <div class="body mb-form">
    <ModeTabs
      bind:value={authoringMode}
      tabs={[
        { value: "discover", label: "auto" },
        { value: "templated", label: "template" },
        { value: "authored", label: "custom" },
      ]}
      ariaLabel="Authoring mode"
    />

    <!-- identity — shared by all three paths -->
    <div class="grid2">
      <label class="field">
        <span class="label">namespace</span>
        <input
          type="text"
          class="input"
          bind:value={identity.namespace}
          placeholder="local"
          autocomplete="off"
          spellcheck="false"
        />
      </label>
      <label class="field">
        <span class="label">name *</span>
        <input
          type="text"
          class="input"
          bind:value={identity.name}
          placeholder="circumplex"
          autocomplete="off"
          spellcheck="false"
        />
      </label>
    </div>
    <label class="field">
      <span class="label">description</span>
      <input
        type="text"
        class="input"
        bind:value={identity.description}
        placeholder="description"
        autocomplete="off"
      />
    </label>

    {#if authoringMode === "authored"}
      <AuthoredForm {identity} />
    {:else if authoringMode === "discover"}
      <DiscoverForm {identity} />
    {:else}
      <TemplatedForm {identity} />
    {/if}
  </div>
</section>

<style>
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-5) var(--space-6);
  }
  .title {
    color: var(--accent);
    letter-spacing: 0;
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-5) var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
  }
</style>
