// The A/B comparison: an auto-regen of the just-finished turn, shown in
// the chat's right column.
//
// ``autoRegenState`` is the whole control surface (there is no standalone
// A/B toggle).  With ``mode === "unsteered"`` the pairing runs here as a
// *shadow* generation: ``abState.processingAb`` routes the next
// ``started``/``token``/``done`` stream into
// ``chatLog.turns[pendingTurnIdx].abPair`` instead of appending a fresh
// top-level turn.  Every other mode is an ordinary loom regen with a
// recipe-override, dispatched from the WS ``done`` handler.
//
// The shadow's prompt is reconstructed from ``chatLog.turns`` at fire
// time, so the comparison works for any turn, not only the just-sent one.

import type { WSClientMessage } from "../api";
import { chatLog, genStatus } from "./chat.svelte";
import { buildSamplingPayload, samplingState } from "./sampling.svelte";
import { ensureWebSocket } from "./ws.svelte";

/** Transient routing state for the unsteered-shadow generation.
 *
 *  The shadow is ``autoRegenState`` with ``mode === "unsteered"``; there
 *  is no standalone toggle.  ``processingAb`` / ``pendingTurnIdx`` are
 *  load-bearing for the WS dispatcher — while ``processingAb`` is set the
 *  next ``started``/``token``/``done`` stream routes into
 *  ``chatLog.turns[pendingTurnIdx].abPair`` instead of appending a fresh
 *  top-level turn.  ``pendingTurnIdx`` is set when the shadow gen is
 *  dispatched and cleared on its ``done`` / ``error``.
 *
 *  The shadow's prompt is reconstructed from ``chatLog.turns`` at fire
 *  time (see ``_buildShadowMessages``), so the comparison works for any
 *  turn, not only the just-sent one.  Turning auto-regen off mid-flight
 *  lets the in-flight shadow finish writing into ``abPair`` — the turn is
 *  harmless when not rendered, and tearing the WS state down mid-stream
 *  is more error-prone than letting it complete; it only prevents the
 *  *next* steered gen from spawning a shadow. */
export interface AbState {
  pendingTurnIdx: number | null;
  processingAb: boolean;
  pendingRole: "user" | "assistant" | null;
  pendingRoleLabel: string | null;
}

export const abState: AbState = $state({
  pendingTurnIdx: null,
  processingAb: false,
  pendingRole: null,
  pendingRoleLabel: null,
});

/** Build the conversation as a messages list to replay through the
 * unsteered shadow.  Walks ``chatLog.turns[0..steeredIdx-1]`` (excluding
 * ``steeredIdx`` itself, which is the generated response we
 * don't want the shadow to inherit), filtering out system / error turns
 * that aren't real conversation context.
 *
 * The unsteered model sees prior steered turns as if they
 * happened naturally — that's the user's "play the conversation back"
 * contract. Scene rendering permits arbitrary seat sequences, so the target
 * turn itself—not the final history role—selects the prompt that follows. */
function _buildShadowMessages(
  steeredIdx: number,
): Array<{
  role: "user" | "assistant";
  content: string;
  label?: string | null;
}> {
  const out: Array<{
    role: "user" | "assistant";
    content: string;
    label?: string | null;
  }> = [];
  for (let i = 0; i < steeredIdx; i++) {
    const t = chatLog.turns[i];
    if (!t) continue;
    if (t.role !== "user" && t.role !== "assistant") continue; // skip system / errors
    // Use the accumulated text — generated turns already exclude their
    // thinking content (only response tokens land in ``turn.text``), so
    // replaying them through ``enable_thinking=False`` is well-formed.
    out.push({ role: t.role, content: t.text ?? "", label: t.roleLabel });
  }
  return out;
}

/** Internal: dispatch the unsteered shadow generate that pairs with the
 * just-finished steered turn at index ``steeredIdx``.  Sends the full
 * conversation as a ``messages`` list instead of a bare input string +
 * server-side history — the shadow runs ``stateless: true`` so the
 * server doesn't append to history (the steered branch already did) and
 * the messages list is the *only* context the unsteered model sees.
 * That makes the comparison work for any turn, not just the first. */
export async function sendShadowGenerate(steeredIdx: number): Promise<void> {
  const target = chatLog.turns[steeredIdx];
  if (!target?.generated || target.role === "system") return;
  const messages = _buildShadowMessages(steeredIdx);
  const sock = await ensureWebSocket();
  // Shadow path mirrors ``sendGenerate``'s sampling-payload build so the
  // ``return_top_k`` opt-in rides shadow / auto-regen runs too (matches
  // the steered turn's wire-shape, keeps logit captures comparable across
  // siblings).
  const sampling = buildSamplingPayload() ?? {};
  if (target.roleLabel) {
    if (target.role === "user") sampling.user_role = target.roleLabel;
    else sampling.assistant_role = target.roleLabel;
  }
  // Mark the WS reception path before the request lands so the
  // ``started`` event routes into the abPair and not a fresh turn.
  abState.pendingTurnIdx = steeredIdx;
  abState.processingAb = true;
  abState.pendingRole = target.role;
  abState.pendingRoleLabel = target.roleLabel ?? null;
  const payload: WSClientMessage = {
    type: "generate",
    // ``input`` accepts ``Any`` server-side; a list goes straight through
    // to ``session._prepare_input`` which dispatches on isinstance(list).
    input: messages,
    // Empty steering string == unsteered shadow per the WS protocol
    // (the server treats "" as "no expression").
    steering: "",
    sampling,
    thinking: samplingState.thinking ?? false,
    // Stateless so the shadow doesn't pollute server-side history; the
    // steered turn already populated history.  Combined with the
    // explicit messages list this means the shadow's prompt is exactly
    // the conversation up to (but not including) the steered response.
    stateless: true,
    raw: false,
    generate_seat: target.role,
  };
  const send = () => sock.send(JSON.stringify(payload));
  if (sock.readyState === WebSocket.OPEN) send();
  else sock.addEventListener("open", send, { once: true });
}

// --------------------------------- auto-regen recipe-override -------

/** Built-in auto-regen modes from the engine. */
export type AutoRegenMode =
  | "unsteered"
  | "inverted"
  | "reseed"
  | "cool"
  | "hot"
  | "custom";

export interface AutoRegenState {
  /** Master toggle (replaces the old A/B toggle one-for-one).  Default
   *  off — the previous A/B behaviour resumed by toggling on with mode
   *  ``"unsteered"``. */
  enabled: boolean;
  mode: AutoRegenMode;
  /** Custom-mode body — a partial-recipe expression (e.g. ``"seed=42,
   *  temperature=1.5"``).  Ignored when ``mode != "custom"``. */
  custom: string;
}

export const autoRegenState: AutoRegenState = $state({
  enabled: false,
  mode: "unsteered",
  custom: "",
});

export function toggleAutoRegen(): void {
  const wasOff = !autoRegenState.enabled;
  autoRegenState.enabled = !autoRegenState.enabled;
  // Off → on with the "unsteered" mode: replay the conversation through
  // the unsteered model for the most recent generated turn that doesn't
  // already carry an ``abPair``, so users who flip the toggle on
  // after-the-fact see the right column populate immediately rather
  // than waiting for the next send.  Other modes use the loom-regen
  // path — they take effect on the next ``done`` event by design.
  if (!wasOff) return;
  if (genStatus.active) return; // ``done`` handler will fire its own
  if (currentRecipeOverride() !== "unsteered") return;
  for (let i = chatLog.turns.length - 1; i >= 0; i--) {
    const t = chatLog.turns[i];
    if (!t) continue;
    if (!t.generated || t.role === "system") continue;
    if (t.abPair) break;
    void sendShadowGenerate(i);
    break;
  }
}

export function setAutoRegenMode(mode: AutoRegenMode): void {
  autoRegenState.mode = mode;
}

export function setAutoRegenCustom(text: string): void {
  autoRegenState.custom = text;
}

/** Render the configured recipe-override the engine consumes.  Returns
 *  ``null`` when auto-regen is off — callers shouldn't dispatch a
 *  shadow regen in that case. */
export function currentRecipeOverride(): string | null {
  if (!autoRegenState.enabled) return null;
  if (autoRegenState.mode === "custom") {
    const v = autoRegenState.custom.trim();
    return v || null;
  }
  return autoRegenState.mode;
}
