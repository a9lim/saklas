<script lang="ts">
  // Auto-generated authoring: hand the model a flat concept list, the
  // K-tuple generator produces per-concept corpora against the shared
  // baseline prompts, then the fitter derives coords per-model via PCA or
  // spectral embedding.  No coords to author; no domain to pick.
  //
  // Generate and fit are deliberately two server calls — a flaky
  // generation leaves inspectable corpora — but "fit now" chains them so
  // the common case is one gesture.

  import {
    apiManifoldFitStream,
    apiManifoldGenerateStream,
    describeError,
  } from "../../lib/api";
  import { closeDrawer, openDrawer, refreshManifoldList } from "../../lib/stores.svelte";
  import { dismissToast, pushToast, updateToast } from "../../lib/stores/toasts.svelte";
  import type { GenerateManifoldRequest } from "../../lib/types";
  import Checkbox from "../../lib/Checkbox.svelte";
  import NumberInput from "../../lib/NumberInput.svelte";
  import Radio from "../../lib/Radio.svelte";
  import AdvancedSection from "../../lib/builder/AdvancedSection.svelte";
  import ValidationBlock from "../../lib/builder/ValidationBlock.svelte";
  import DiscoverTuningFields from "./DiscoverTuningFields.svelte";
  import FitMethodPicker from "./FitMethodPicker.svelte";
  import {
    defaultTuning,
    identitySlugs,
    parseTokens,
    progressMessage,
    slug,
    tuningHyperparams,
    tuningMessages,
    type ManifoldIdentity,
  } from "./shared";

  let { identity }: { identity: ManifoldIdentity } = $props();

  type DiscoverKind = "abstract" | "concrete" | "custom";

  const tuning = $state(defaultTuning());
  let conceptsText = $state("");
  // Conversational corpus knobs: ``kind`` frames each concept's system
  // prompt (abstract → "someone {c}", concrete → "{article} {c}");
  // ``samplesPerPrompt`` is the in-character responses generated per
  // shared baseline prompt.
  let kind: DiscoverKind = $state("abstract");
  let customSystem = $state("You are {c}.");
  let samplesPerPrompt = $state(1);
  // Persona-manifold opt-in: when set, each concept slug doubles as the
  // matching node's assistant-role substitution at fit time, producing a
  // role-paired manifold (steering through it implies the nearest node's
  // role at decode time).  The slug regex matches the engine's role
  // validation — concepts that pass ``slug()`` are already in
  // ``[a-z0-9._-]+``.
  let rolePerNode = $state(false);
  let force = $state(false);
  let saeRelease = $state("");
  let alsoFit = $state(true);
  let advancedOpen = $state(false);
  let progress: string | null = $state(null);
  let submitting = $state(false);

  const concepts = $derived(parseTokens(conceptsText));

  const validation = $derived.by<{ ok: boolean; messages: string[] }>(() => {
    const messages: string[] = [];
    if (!slug(identity.name)) {
      messages.push("name required");
    }
    if (concepts.length < 2) {
      messages.push(`concepts: ${concepts.length} / 2`);
    }
    const seen = new Set<string>();
    for (const c of concepts) {
      const s = slug(c);
      if (!s) {
        messages.push(`invalid concept "${c}"`);
      } else if (seen.has(s)) {
        messages.push(`duplicate concept "${s}"`);
      } else {
        seen.add(s);
      }
    }
    if (samplesPerPrompt <= 0) {
      messages.push("samples / prompt > 0");
    }
    if (kind === "custom") {
      const template = customSystem.trim();
      if (!template) {
        messages.push("system template required");
      } else if (!template.includes("{c}")) {
        messages.push('system template needs "{c}"');
      }
    }
    messages.push(...tuningMessages(tuning));
    return { ok: messages.length === 0, messages };
  });

  async function save(): Promise<void> {
    if (!validation.ok || submitting) return;
    submitting = true;
    progress = null;
    const { namespace, name, description } = identitySlugs(identity);
    const hyperparams = tuningHyperparams(tuning);
    const req: GenerateManifoldRequest = {
      namespace,
      name,
      description,
      concepts: concepts.map((c) => slug(c)),
      kind,
      custom_system: kind === "custom" ? customSystem.trim() : undefined,
      samples_per_prompt: samplesPerPrompt,
      fit_mode: tuning.fitMode,
      hyperparams,
      force,
      role_per_node: rolePerNode,
    };
    const toastId = pushToast(`generating ${namespace}/${name} corpora…`, {
      kind: "info",
      ttlMs: null,
    });
    try {
      await apiManifoldGenerateStream(req, (ev) => {
        if (ev.event !== "progress") return;
        const msg = progressMessage(ev);
        if (msg) {
          progress = msg;
          updateToast(toastId, { detail: msg });
        }
      });
      dismissToast(toastId);
      if (alsoFit) {
        // Chain the fit immediately — the user opted into the two-step.
        // The fit endpoint accepts the discover-mode hyperparams as an
        // override; passing them here keeps the sidecar metadata in sync
        // even if the folder already had matching values from generate.
        const fitToastId = pushToast(`fitting ${namespace}/${name}…`, {
          kind: "info",
          ttlMs: null,
        });
        try {
          await apiManifoldFitStream(
            namespace,
            name,
            {
              sae: saeRelease.trim() || null,
              fit_mode: tuning.fitMode,
              hyperparams,
            },
            (ev) => {
              if (ev.event !== "progress") return;
              const msg = progressMessage(ev);
              if (msg) {
                progress = msg;
                updateToast(fitToastId, { detail: msg });
              }
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
          `generated ${namespace}/${name} — open manifolds drawer to fit`,
          { kind: "info" },
        );
      }
      await refreshManifoldList();
      closeDrawer();
      openDrawer("manifolds");
    } catch (e) {
      dismissToast(toastId);
      pushToast(`generate failed — ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      submitting = false;
      progress = null;
    }
  }
</script>

<div class="form-stack">
  <section class="step">
    <h2 class="step-title">concepts</h2>
    <label class="field">
      <span class="label">concepts * · ≥2</span>
      <textarea
        class="input"
        rows="4"
        placeholder="pirate caveman assistant scholar robot"
        bind:value={conceptsText}
        spellcheck="false"
      ></textarea>
      <span class="dim-note">
        <strong>{concepts.length}</strong> parsed
      </span>
    </label>
    <div class="grid2">
      <div class="field">
        <span class="label">kind</span>
        <div class="radio-row">
          <Radio bind:group={kind} value="abstract" label="abstract" />
          <Radio bind:group={kind} value="concrete" label="concrete" />
          <Radio bind:group={kind} value="custom" label="custom" />
        </div>
      </div>
      <label class="field">
        <span class="label">samples / prompt</span>
        <NumberInput
          value={samplesPerPrompt}
          min={1}
          step={1}
          oninput={(v) => {
            if (v !== null) samplesPerPrompt = v;
          }}
        />
      </label>
    </div>
    {#if kind === "custom"}
      <label class="field">
        <span class="label">system template</span>
        <textarea
          class="input"
          rows="3"
          bind:value={customSystem}
          spellcheck="false"
        ></textarea>
      </label>
    {/if}
  </section>

  <FitMethodPicker {tuning} spectralNote="curved · best with ≥50 nodes" />

  <AdvancedSection bind:expanded={advancedOpen}>
    <label class="field">
      <span class="label">SAE release</span>
      <!-- ``text-input`` has never had a rule in this drawer, so the field
           renders unstyled.  Preserved verbatim: the visual language is
           locked for this pass, and correcting it to ``input`` would be a
           look change, not a structural one. -->
      <input
        class="text-input"
        bind:value={saeRelease}
        placeholder="e.g. gemma-scope-2-4b-it-res"
        spellcheck="false"
      />
    </label>
    <DiscoverTuningFields {tuning} />
    <div class="check-stack">
      <Checkbox bind:checked={alsoFit} label="fit now" />
      <Checkbox bind:checked={rolePerNode} label="node roles" />
      {#if rolePerNode}
        <p class="role-hint">
          uses each concept as its assistant role; unsupported models fail at fit
        </p>
      {/if}
      <Checkbox bind:checked={force} label="overwrite" />
    </div>
  </AdvancedSection>

  {#if progress}
    <p class="progress">{progress}</p>
  {/if}

  <ValidationBlock verb="generate" messages={validation.messages} />

  <button
    type="button"
    class="save-btn"
    disabled={!validation.ok || submitting}
    onclick={save}
  >
    {submitting ? "generating…" : alsoFit ? "generate + fit" : "generate"}
  </button>
</div>
