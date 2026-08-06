// The chat surface's own state — the projection of the loom tree the
// transcript renders, and the live counters that ride a generation.
//
// ``chatLog.turns`` is NOT authoritative: the server's loom tree is, and
// ``syncChatLogFromTree`` re-derives the turns from its active path after
// every mutation.  What lives here is the projection plus the things only
// the chat panel cares about — the role display labels, the clear-chat
// gesture, the per-generation token/perplexity counters, and the
// per-model chat-vs-raw render mode.

import type { ChatTurn, GenStatus, TokenScore } from "../types";
import { loomNavigate, loomTree, sessionState } from "../stores.svelte";
import { enqueuePending, pendingActions } from "./pending.svelte";
import { safeLocalStorageGet, safeLocalStorageSet } from "./persistence.svelte";

export interface ChatLogState {
  turns: ChatTurn[];
  /** Index of the in-flight assistant turn, when one exists.  Null
   * between gens.  Used by the WS event handlers to attach streamed
   * tokens to the right turn. */
  pendingIndex: number | null;
}

export const chatLog: ChatLogState = $state({
  turns: [],
  pendingIndex: null,
});

// ------------------------------------------------ role labels --------

/** Display label for a turn, honoring its per-message role-substitution
 *  label (the roleplay scaffold stamped at send time).  ``roleLabel`` —
 *  the node's ``role_label`` — wins when set; otherwise the structural
 *  ``role`` falls back to the model family's actual chat-template label. */
export function roleDisplayLabel(
  role: string,
  roleLabel?: string | null,
): string {
  if (roleLabel) return roleLabel;
  if (role === "user") {
    return sessionState.info?.default_user_role ?? "user";
  }
  if (role === "assistant") {
    return sessionState.info?.default_assistant_role ?? "assistant";
  }
  return role;
}

/** Single-character glyph for the loom node badge — first char of the
 *  display label, uppercased.  Default roles reduce to ``U`` / ``A`` / ``S``;
 *  a per-turn ``captain`` label yields ``C``. */
export function roleGlyphLetter(
  role: string,
  roleLabel?: string | null,
): string {
  const label = roleDisplayLabel(role, roleLabel);
  return (label.charAt(0) || role.charAt(0) || "?").toUpperCase();
}

/** "Clear chat" — preserves the loom tree and just navigates back to the
 *  synthetic system root.  The next submission lands a fresh user turn
 *  as a sibling branch off root rather than the legacy behaviour of
 *  destroying every existing branch.  Earlier conversation paths stay
 *  reachable from the sidebar.
 *
 */
export async function resetChatToRoot(): Promise<void> {
  const root = loomTree.root_id;
  if (root !== null) {
    // ``loomNavigate`` runs the navigate REST call and refreshes the
    // tree; ``syncChatLogFromTree`` then projects the [root]-only
    // active path to an empty ``chatLog.turns`` because the synthetic
    // system root is filtered out of the chat view.
    await loomNavigate(root);
    return;
  }
  throw new Error("Cannot clear chat before the tree root is loaded");
}

/** Clear the chat back to root.  Lifted out of Chat.svelte so the
 *  threads-column action button can call the same code path the chat
 *  header used to — queue-aware when generation is in flight, direct
 *  when idle. */
export function clearChat(): void {
  if (genStatus.active || pendingActions.queue.length > 0) {
    enqueuePending({
      label: "/clear",
      text: null,
      apply: () => void resetChatToRoot(),
      awaitsGen: false,
      rebuild: null,
      // /clear navigates to the synthetic root (system role) — not a
      // user node, so the next submission goes through "message" mode
      // and lands a fresh user branch off root.
      endsOnUserNode: false,
    });
  } else {
    void resetChatToRoot();
  }
}

// ----------------------------------------- live token / gen status ---

/** Captures the in-flight generation's per-token scores so the chat
 * renderer can highlight live before the WS ``done`` event lands.  Reset
 * on each ``started``. */
export interface LiveTokenStream {
  responseTokens: TokenScore[];
  thinkingTokens: TokenScore[];
}

export const liveTokenStream: LiveTokenStream = $state({
  responseTokens: [],
  thinkingTokens: [],
});

export const genStatus: GenStatus = $state({
  active: false,
  tokensSoFar: 0,
  maxTokens: 0,
  startedAt: null,
  tokPerSec: 0,
  ppl: { logSum: 0, count: 0, mean: null },
  finishReason: null,
});

/** Geometric-mean perplexity assembled from per-token TokenEvent.perplexity
 * values using ``exp(sum(log(ppl)) / count)``. Pure
 * function — caller passes the slice so it can also be used on ad-hoc
 * accumulators (e.g. an A/B side's separate perplexity buffer). */
export function geometricMeanPpl(state: GenStatus): number | null {
  if (state.ppl.count <= 0) return null;
  return Math.exp(state.ppl.logSum / state.ppl.count);
}

// ------------------------------------------- generation UI mode ------
//
// Base (non-chat) models have no chat template — the engine handles
// them as flat completion.  ``genUiMode`` decides whether the chat panel
// renders bubbles + roles (chat) or a single flat completion buffer
// (raw).  It is a plain two-state toggle: the default is seeded from the
// model's ``is_base_model`` flag (base → raw, chat → chat) the first
// time a model is seen, then the user's explicit choice is persisted
// per ``model_id`` and survives reloads.

export interface GenUiModeState {
  /** Which surface the chat panel renders — ``"chat"`` (bubbles +
   *  roles) or ``"raw"`` (a single flat completion buffer). */
  mode: "chat" | "raw";
}

export const genUiMode: GenUiModeState = $state({ mode: "chat" });

/** Resolve the effective rendering mode — true means flat raw buffer. */
export function effectiveRawMode(): boolean {
  return genUiMode.mode === "raw";
}

const GENUI_KEY_PREFIX = "saklas.genui.v1.";

function genUiKey(): string | null {
  const id = sessionState.info?.model_id;
  return id ? GENUI_KEY_PREFIX + id : null;
}

/** Load the per-model render mode.  Called from ``bootstrap`` once the
 *  model id is known.  A stored preference wins; with none, the mode is
 *  seeded from the model's nature — a base model defaults to ``raw``, a
 *  chat model to ``chat``. */
export function loadGenUiMode(): void {
  const key = genUiKey();
  const stored = key ? safeLocalStorageGet(key) : null;
  if (stored === "chat" || stored === "raw") {
    genUiMode.mode = stored;
  } else {
    genUiMode.mode =
      sessionState.info?.is_base_model === true ? "raw" : "chat";
  }
}

/** Set (and persist) the render mode.  Toggling mode never mutates the
 *  loom tree — only generation does. */
export function setGenUiMode(mode: "chat" | "raw"): void {
  genUiMode.mode = mode;
  const key = genUiKey();
  if (key) safeLocalStorageSet(key, mode);
}
