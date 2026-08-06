<script lang="ts">
  // Derive a manifold from a standalone template.
  //
  // Some categories you *reference* rather than embody — days, months,
  // durations — so the persona-framed generation doesn't apply.  A
  // template (slot + candidate values + multi-turn contexts) materializes
  // those deterministically: no model call, the node corpus is just the
  // slot-filled assistant turns.
  //
  // This tab does NOT author templates.  It used to, with its own
  // single-user-turn editor, which was a strict subset of the template
  // lab's multi-turn editor and had already drifted away from the server's
  // validation once.  Authoring now lives in exactly one place — the
  // template lab's build tab, linked below — and this tab picks one of
  // those templates and runs the derivation the lab has no verb for.

  import { onMount } from "svelte";
  import {
    apiManifoldFitStream,
    apiManifolds,
    apiTemplates,
    describeError,
  } from "../../lib/api";
  import { closeDrawer, openDrawer, refreshManifoldList } from "../../lib/stores.svelte";
  import { dismissToast, pushToast, updateToast } from "../../lib/stores/toasts.svelte";
  import type {
    CreateManifoldFromTemplateRequest,
    TemplateSummary,
  } from "../../lib/types";
  import Checkbox from "../../lib/Checkbox.svelte";
  import NumberInput from "../../lib/NumberInput.svelte";
  import Select from "../../lib/Select.svelte";
  import AdvancedSection from "../../lib/builder/AdvancedSection.svelte";
  import ValidationBlock from "../../lib/builder/ValidationBlock.svelte";
  import FitMethodPicker from "./FitMethodPicker.svelte";
  import {
    defaultTuning,
    identitySlugs,
    slug,
    type ManifoldIdentity,
  } from "./shared";

  let { identity }: { identity: ManifoldIdentity } = $props();

  const tuning = $state(defaultTuning());
  let templates: TemplateSummary[] = $state([]);
  let loadingTemplates = $state(true);
  let selectedKey = $state("");
  let maxDim: number | null = $state(null);
  let alsoFit = $state(true);
  let advancedOpen = $state(false);
  let submitting = $state(false);

  onMount(async () => {
    try {
      templates = (await apiTemplates.list()).templates;
    } catch (e) {
      pushToast(`couldn't load templates: ${describeError(e)}`, {
        kind: "error",
      });
    } finally {
      loadingTemplates = false;
    }
  });

  const options = $derived(
    templates.map((t) => ({
      value: `${t.namespace}/${t.name}`,
      label: `${t.namespace}/${t.name} · ${t.slot} · ${t.n_values}×${t.n_contexts}`,
    })),
  );

  const selected = $derived(
    templates.find((t) => `${t.namespace}/${t.name}` === selectedKey) ?? null,
  );

  const validation = $derived.by<{ ok: boolean; messages: string[] }>(() => {
    const messages: string[] = [];
    if (!slug(identity.name)) messages.push("name required");
    if (!selected) messages.push("template required");
    if (maxDim !== null && maxDim < 1) messages.push("max dim ≥1");
    return { ok: messages.length === 0, messages };
  });

  /** Hand authoring to the one editor that speaks the full multi-turn
   *  context shape, then come back here to derive. */
  function openTemplateLab(): void {
    closeDrawer();
    openDrawer("template_lab", { tab: "build" });
  }

  async function save(): Promise<void> {
    if (!validation.ok || !selected || submitting) return;
    submitting = true;
    const { namespace, name, description } = identitySlugs(identity);
    const hyperparams: Record<string, number> = {};
    if (maxDim !== null && maxDim >= 1) hyperparams.max_dim = maxDim;
    const req: CreateManifoldFromTemplateRequest = {
      namespace,
      name,
      description,
      fit_mode: tuning.fitMode,
      template_ref: `${selected.namespace}/${selected.name}`,
      hyperparams,
    };
    const toastId = pushToast(`authoring ${namespace}/${name}…`, {
      kind: "info",
      ttlMs: null,
    });
    try {
      await apiManifolds.createFromTemplate(req);
      dismissToast(toastId);
      if (alsoFit) {
        const fitToastId = pushToast(`fitting ${namespace}/${name}…`, {
          kind: "info",
          ttlMs: null,
        });
        try {
          await apiManifoldFitStream(
            namespace,
            name,
            { fit_mode: tuning.fitMode, hyperparams },
            (ev) => {
              if (ev.event !== "progress") return;
              const msg =
                ev.data && typeof ev.data === "object"
                  ? (ev.data as { message?: string }).message
                  : null;
              if (msg) updateToast(fitToastId, { detail: msg });
            },
          );
          dismissToast(fitToastId);
          pushToast(`fit ${namespace}/${name} (${tuning.fitMode})`, {
            kind: "info",
          });
        } catch (e) {
          dismissToast(fitToastId);
          pushToast(`fit failed — ${describeError(e)}`, {
            kind: "error",
            ttlMs: null,
          });
        }
      } else {
        pushToast(
          `authored ${namespace}/${name} — open manifolds drawer to fit`,
          { kind: "info" },
        );
      }
      await refreshManifoldList();
      closeDrawer();
      openDrawer("manifolds");
    } catch (e) {
      dismissToast(toastId);
      pushToast(`author failed — ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      submitting = false;
    }
  }
</script>

<div class="form-stack">
  <section class="step">
    <h2 class="step-title">template</h2>
    {#if loadingTemplates}
      <p class="muted">loading…</p>
    {:else if templates.length === 0}
      <p class="muted">no templates yet</p>
    {:else}
      <label class="field">
        <span class="label">source *</span>
        <Select
          value={selectedKey}
          options={[{ value: "", label: "— pick a template —" }, ...options]}
          ariaLabel="Template"
          onchange={(v) => {
            selectedKey = String(v);
          }}
        />
      </label>
      {#if selected}
        <p class="dim-note">
          slot <strong>{selected.slot}</strong> ·
          <strong>{selected.n_values}</strong> values ×
          <strong>{selected.n_contexts}</strong> contexts
        </p>
      {/if}
    {/if}
    <button type="button" class="add-node" onclick={openTemplateLab}>
      author a template…
    </button>
  </section>

  <FitMethodPicker {tuning} />

  <AdvancedSection bind:expanded={advancedOpen}>
    <label class="field">
      <span class="label">max dim</span>
      <NumberInput
        value={maxDim}
        min={1}
        step={1}
        allowEmpty
        placeholder="auto"
        oninput={(v) => {
          maxDim = v;
        }}
      />
    </label>
    <div class="check-stack">
      <Checkbox bind:checked={alsoFit} label="fit now" />
    </div>
  </AdvancedSection>

  <ValidationBlock verb="author" messages={validation.messages} />

  <button
    type="button"
    class="save-btn"
    disabled={!validation.ok || submitting}
    onclick={save}
  >
    {submitting ? "building…" : alsoFit ? "build + fit" : "build"}
  </button>
</div>
