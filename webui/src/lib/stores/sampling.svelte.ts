// Sampling controls + the wire payload they build.
//
// ``samplingState`` is the editable mirror of the server's session
// defaults: ``hydrateSamplingFromInfo`` pulls the server's config down on
// every session refresh, ``patchSessionDefaults`` pushes the numeric
// controls back so they become defaults for future calls, and
// ``buildSamplingPayload`` snapshots whatever is visible in the UI onto
// each generation — the PATCH is deliberately asynchronous, so echoing
// the per-call values is what keeps "change a field, press Send"
// honest.

import { apiSessions } from "../api";
import type { ChatRole, WSSampling } from "../types";
import { probeRack, sessionState } from "../stores.svelte";

export interface SamplingState {
  temperature: number | null;
  top_p: number | null;
  top_k: number | null;
  max_tokens: number;
  /** ``null`` = use the WS default (no seed sent).  Numeric value pinned. */
  seed: number | null;
  system_prompt: string;
  /** One per line stop sequences; parsed into SamplingConfig.stop. */
  stop_sequences: string;
  /** Raw logit bias map. Accepts JSON {"123": -4} or lines "123: -4". */
  logit_bias_text: string;
  presence_penalty: number;
  frequency_penalty: number;
  /** ``null`` = auto, true/false = explicit override. */
  thinking: boolean | null;
  /** Logit-pass: top-K alternatives to capture per token (``0`` = off,
   *  matches the engine's chosen-only mode).  When ``> 0`` the WS ``token``
   *  event carries ``top_alts`` and the drilldown's logits tab + the
   *  inline ``surprise`` highlight mode populate.  Flipped via the
   *  "show alts" toggle in ``SamplingStrip``; the canonical "on" value
   *  is ``8``. */
  return_top_k: number;
  /** Active chat-template labels for the structural user/assistant roles. Sticky
   *  client state like ``seed`` — whatever's in the boxes rides the next
   *  submission and is stamped onto that turn's loom node. */
  user_role: string;
  assistant_role: string;
}

export const samplingState: SamplingState = $state({
  temperature: null,
  top_p: null,
  top_k: null,
  max_tokens: 256,
  seed: null,
  system_prompt: "",
  stop_sequences: "",
  logit_bias_text: "",
  presence_penalty: 0,
  frequency_penalty: 0,
  user_role: "user",
  assistant_role: "assistant",
  // Initial thinking state: explicit ``false`` so an unchecked checkbox
  // on first paint actually sends ``thinking: false`` to the server.
  // The previous ``null`` (auto) state silently fell through to whatever
  // the model template defaults to — for thinking-capable templates that
  // meant the model thought even though the box was visually off.
  thinking: false,
  // Logit-pass: top-K alternatives on by default — the drilldown logits
  // tab and the inline surprise highlight want them.  The SamplingStrip's
  // "alts" toggle flips this between 0 and 8.
  return_top_k: 8,
});

export function setSampling<K extends keyof SamplingState>(
  key: K,
  value: SamplingState[K],
): void {
  samplingState[key] = value;
}

// ------------------------------------------ session-defaults mirror ---

/** Mirror the server's session.config defaults into the local
 * ``samplingState``.  The local store was previously pre-seeded with its
 * own constants (``max_tokens: 256`` etc.) which drifted away from the
 * server's actual ``session.config.max_new_tokens`` (= 1024 by default),
 * so the gen-status footer rendered ``gen N/256`` even when the engine
 * was running against a 1024-token cap.  Sync once on every refresh so
 * the displayed cap matches what generation actually used. */
let _roleDefaultsModelId: string | null = null;

export function hydrateSamplingFromInfo(): void {
  const info = sessionState.info;
  // The editable values are the actual labels used by this model family's
  // chat template. Seed once per
  // loaded model so ordinary refreshes never erase a custom cast label.
  if (info && _roleDefaultsModelId !== info.model_id) {
    _roleDefaultsModelId = info.model_id;
    samplingState.user_role = info.default_user_role ?? "user";
    samplingState.assistant_role = info.default_assistant_role ?? "assistant";
  }
  const cfg = info?.config;
  if (!cfg) return;
  if (typeof cfg.max_tokens === "number" && Number.isFinite(cfg.max_tokens)) {
    samplingState.max_tokens = cfg.max_tokens;
  }
  if (typeof cfg.temperature === "number") {
    samplingState.temperature = cfg.temperature;
  }
  if (typeof cfg.top_p === "number") {
    samplingState.top_p = cfg.top_p;
  }
  // ``null`` is meaningful: no top-k cutoff.  Assign it rather than keeping a
  // stale local number after the user clears the field.
  samplingState.top_k = cfg.top_k;
  if (typeof cfg.system_prompt === "string") {
    samplingState.system_prompt = cfg.system_prompt;
  }
  if (typeof cfg.thinking === "boolean") {
    samplingState.thinking = cfg.thinking;
  }
}

export async function patchSessionDefaults(
  body: Partial<{
    temperature: number;
    top_p: number;
    top_k: number | null;
    max_tokens: number;
    system_prompt: string;
    thinking: boolean;
  }>,
): Promise<void> {
  const info = await apiSessions.patch(body);
  sessionState.info = info;
  sessionState.lastRefresh = Date.now();
  hydrateSamplingFromInfo();
}

// ------------------------------------------------- wire payload ------

function parsedStopSequences(): string[] | null {
  const lines = samplingState.stop_sequences
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  return lines.length > 0 ? lines : null;
}

function parsedLogitBias(): Record<string, number> | null {
  const raw = samplingState.logit_bias_text.trim();
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const out: Record<string, number> = {};
      for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
        const n = Number(v);
        if (Number.isFinite(n)) out[String(Number(k))] = n;
      }
      return Object.keys(out).length > 0 ? out : null;
    }
  } catch {
    /* fall through to line parser */
  }
  const out: Record<string, number> = {};
  for (const line of raw.split(/\r?\n/)) {
    const m = line.match(/^\s*(-?\d+)\s*[:=,\s]\s*(-?\d+(?:\.\d+)?)\s*$/);
    if (!m) continue;
    out[String(Number(m[1]))] = Number(m[2]);
  }
  return Object.keys(out).length > 0 ? out : null;
}

function nonDefaultSamplingOverrides(): Partial<WSSampling> {
  const stop = parsedStopSequences();
  const logit_bias = parsedLogitBias();
  return {
    ...(stop ? { stop } : {}),
    ...(logit_bias ? { logit_bias } : {}),
    ...(samplingState.presence_penalty !== 0
      ? { presence_penalty: samplingState.presence_penalty }
      : {}),
    ...(samplingState.frequency_penalty !== 0
      ? { frequency_penalty: samplingState.frequency_penalty }
      : {}),
    ...(samplingState.return_top_k > 0
      ? { return_top_k: samplingState.return_top_k }
      : {}),
    // Per-message role labels (roleplay scaffold) ride every send like
    // ``seed`` — trimmed.  Empty = standard role, omitted.  A value equal to
    // the family's standard label (the box's seeded default) is *also* a
    // no-op, so we omit it too: the node isn't stamped with a redundant label
    // and the bubble keeps its structural heading.  Only a genuine override
    // (a label the user changed away from the default) is sent.
    ...roleOverride(
      samplingState.user_role,
      sessionState.info?.default_user_role,
      "user",
      "user_role",
    ),
    ...roleOverride(
      samplingState.assistant_role,
      sessionState.info?.default_assistant_role,
      "assistant",
      "assistant_role",
    ),
  };
}

/** ``{key: value}`` when ``raw`` is a non-empty label that differs from the
 *  family default, else ``{}`` (treated as "use the standard role"). */
function roleOverride(
  raw: string,
  fallback: string | null | undefined,
  structural: ChatRole,
  key: "user_role" | "assistant_role",
): Partial<WSSampling> {
  const value = raw.trim();
  if (!value || value === structural || value === fallback) return {};
  return { [key]: value };
}

export function buildSamplingPayload(): WSSampling | null {
  // The strip PATCHes these numeric controls so they become defaults for
  // future calls, but that persistence request is deliberately asynchronous.
  // Echo the values visible in the UI on every generation as well: otherwise
  // changing a field and immediately pressing Send can race the PATCH and run
  // with the previous server default while the footer claims the new cap.
  // Per-call values are the authoritative snapshot for this generation.
  const payload: WSSampling = {
    temperature: samplingState.temperature,
    top_p: samplingState.top_p,
    top_k: samplingState.top_k,
    max_tokens: samplingState.max_tokens,
    persist_per_layer_scores: true,
    // The probe-inspector live point + fading trail need per-layer whitened
    // subspace coords on each token reading.  Sent whenever a probe is attached
    // so the trajectory is always captured — opening the inspector after any
    // generation shows the run's path with no prior opt-in.  (The Python
    // SamplingConfig default stays off, so non-webui callers and the throughput
    // benchmark are unaffected.)
    ...(probeRack.active.length > 0
      ? { persist_subspace_coords: true }
      : {}),
    ...nonDefaultSamplingOverrides(),
    ...(samplingState.seed !== null ? { seed: samplingState.seed } : {}),
  };
  return Object.keys(payload).length > 0 ? payload : null;
}
