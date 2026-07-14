<script lang="ts">
  // Cast manager (phase 3 of the cast model) — the tree's roster of
  // named labels, each with a standing steering recipe.  A member's
  // recipe is the *weakest* tier at generation: it fills only fields
  // the send left unset, so the rack and per-send controls always win.
  // Identity is auto-derived from labels observed anywhere in the tree;
  // configuration adds only a standing recipe/notes layer.  Mutations and
  // new observed labels reconcile through the inlined effective roster.
  //
  // A steering surface, not a chat feature: labels here are the same
  // slugs the composer's "speaking as" / "reply as" chips take, and a
  // turn labeled with a member's slug generates under its recipe.

  import { apiTree } from "../lib/api";
  import { castState, pushToast } from "../lib/stores.svelte";
  import { closeDrawer } from "../lib/stores/drawers.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  const SLUG_RE = /^[a-z][a-z0-9._-]*$/;

  let label = $state("");
  let steering = $state("");
  let notes = $state("");
  let editing = $state<string | null>(null);
  let busy = $state(false);
  let err = $state<string | null>(null);

  const labelValid = $derived(
    label.trim() === "" || SLUG_RE.test(label.trim()),
  );
  const canSave = $derived(
    !busy && label.trim() !== "" && SLUG_RE.test(label.trim()),
  );

  const roster = $derived(
    Object.entries(castState.roster).sort(([a], [b]) => a.localeCompare(b)),
  );

  function loadMember(slug: string): void {
    const m = castState.roster[slug];
    if (!m) return;
    editing = slug;
    label = slug;
    steering = m.recipe?.steering ?? "";
    notes = m.notes ?? "";
    err = null;
  }

  function clearForm(): void {
    editing = null;
    label = "";
    steering = "";
    notes = "";
    err = null;
  }

  async function save(): Promise<void> {
    const slug = label.trim();
    if (!canSave) return;
    busy = true;
    err = null;
    try {
      const r = await apiTree.castPut(slug, {
        steering: steering.trim() === "" ? null : steering.trim(),
        notes: notes.trim(),
      });
      // Optimistic merge — the ``op="cast"`` frame confirms shortly.
      castState.roster = { ...castState.roster, [slug]: r.member };
      clearForm();
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  async function remove(slug: string): Promise<void> {
    try {
      await apiTree.castDelete(slug);
      if (editing === slug) clearForm();
    } catch (e) {
      pushToast(
        `remove cast member: ${e instanceof Error ? e.message : String(e)}`,
        { kind: "error" },
      );
    }
  }
</script>

<section class="drawer-shell" aria-label="Cast manager drawer">
  <header class="header">
    <span class="title">cast</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}
      >✕</button
    >
  </header>

  <div class="body">
    <p class="hint">roles appear from conversation history · recipes are saved</p>

    {#if roster.length === 0}
      <p class="empty">none</p>
    {:else}
      <ul class="roster" aria-label="Cast roster">
        {#each roster as [slug, member] (slug)}
          <li class="member" class:editing={editing === slug}>
            <button
              type="button"
              class="member-main"
              title="edit {slug}"
              onclick={() => loadMember(slug)}
            >
              <span class="glyph">{slug.slice(0, 1).toUpperCase()}</span>
              <span class="member-text">
                <span class="member-label">{slug}</span>
                {#if member.recipe?.steering}
                  <span class="member-recipe">{member.recipe.steering}</span>
                {/if}
                {#if member.notes}
                  <span class="member-notes">{member.notes}</span>
                {/if}
              </span>
            </button>
            {#if member.origin === "configured"}
              <button
                type="button"
                class="remove"
                aria-label="clear configuration for {slug}"
                title="clear recipe and notes"
                onclick={() => void remove(slug)}
              >✕</button>
            {/if}
          </li>
        {/each}
      </ul>
    {/if}

    <div class="form" aria-label={editing ? `Edit ${editing}` : "Add cast member"}>
      <span class="form-title">{editing ? `edit ${editing}` : "add member"}</span>
      <label class="field">
        <span class="label">label</span>
        <input
          class="input mono"
          class:invalid={!labelValid}
          bind:value={label}
          placeholder="deer"
          spellcheck="false"
          autocomplete="off"
        />
      </label>
      <label class="field">
        <span class="label">recipe</span>
        <input
          class="input mono"
          bind:value={steering}
          placeholder="0.5 personas%deer"
          spellcheck="false"
          autocomplete="off"
        />
      </label>
      <label class="field">
        <span class="label">notes</span>
        <input
          class="input"
          bind:value={notes}
          placeholder="—"
          spellcheck="false"
          autocomplete="off"
        />
      </label>
      {#if err}
        <p class="error" role="alert">{err}</p>
      {/if}
    </div>
  </div>

  <footer class="footer">
    {#if editing}
      <button type="button" class="btn" onclick={clearForm}>new</button>
    {/if}
    <button type="button" class="btn" onclick={closeDrawer}>close</button>
    <button
      type="button"
      class="btn primary"
      disabled={!canSave}
      onclick={() => void save()}
    >{editing ? "save" : "add"}</button>
  </footer>
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
    text-transform: lowercase;
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  .close {
    background: var(--glass);
    color: var(--fg-muted);
    border: 1px solid transparent;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font: inherit;
    font-size: var(--text-md);
    line-height: 1;
    cursor: pointer;
    flex: none;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .close:hover {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
  }
  .hint {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.4;
  }
  .empty {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .roster {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .member {
    display: flex;
    align-items: stretch;
    gap: var(--space-2);
    background: var(--glass);
    border-radius: var(--radius);
  }
  .member.editing {
    background: var(--glass-strong);
  }
  .member-main {
    flex: 1 1 auto;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    background: none;
    border: none;
    color: inherit;
    font: inherit;
    text-align: left;
    padding: var(--space-3) var(--space-4);
    cursor: pointer;
    min-width: 0;
  }
  .glyph {
    flex: none;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.09);
    color: var(--fg-strong);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  .member-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }
  .member-label {
    font-family: var(--font-mono);
    color: var(--fg-strong);
  }
  .member-recipe {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--fg-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .member-notes {
    font-size: var(--text-xs);
    color: var(--fg-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .remove {
    flex: none;
    align-self: center;
    background: none;
    border: none;
    color: var(--fg-muted);
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
    padding: var(--space-2) var(--space-3);
  }
  .remove:hover {
    color: var(--accent-red);
  }
  .form {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding-top: var(--space-3);
  }
  .form-title {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    text-transform: lowercase;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    text-transform: lowercase;
  }
  .input {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    padding: var(--space-3) var(--space-3);
    font: inherit;
  }
  .input.mono {
    font-family: var(--font-mono);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .input.invalid {
    border-color: var(--accent-red);
  }
  .error {
    margin: 0;
    color: var(--accent-red);
    font-size: var(--text-sm);
  }
  .footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-6);
    color: var(--fg-muted);
  }
  .btn {
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    padding: var(--space-3) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    background: var(--glass-strong);
  }
  .btn.primary {
    background: var(--accent);
    color: var(--text-on-accent);
  }
  .btn.primary:hover:not(:disabled) {
    background: var(--accent-light);
  }
  .btn.primary:disabled {
    background: var(--bg-elev);
    color: var(--fg-muted);
    cursor: default;
  }
</style>
