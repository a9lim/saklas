// Trigger-pill vocabulary shared by every steer card (subspace / manifold
// / j-lens).  One cycle order, one display word, one tooltip per preset —
// extracted from SteerCard so the J-lens steer card's pill stays in
// lock-step with the concept cards'.

import type { Trigger } from "../../lib/types";

export const TRIGGER_ORDER: Trigger[] = [
  "BOTH",
  "BEFORE",
  "AFTER",
  "THINKING",
  "RESPONSE",
  "PROMPT",
  "GENERATED",
];

export const TRIGGER_WORD: Record<Trigger, string> = {
  BOTH: "both",
  BEFORE: "before",
  AFTER: "after",
  THINKING: "thinking",
  RESPONSE: "response",
  PROMPT: "prompt",
  GENERATED: "generated",
};

export const TRIGGER_LABEL: Record<Trigger, string> = {
  BOTH: "both: steer the whole turn (default)",
  BEFORE: "before: steer thinking and response",
  AFTER: "after: steer the after-thinking response only",
  THINKING: "thinking: steer the chain-of-thought only",
  RESPONSE: "response: steer the generated response only",
  PROMPT: "prompt (alias of before)",
  GENERATED: "generated (alias of response)",
};

/** The next trigger in the cycle order — the pill's click step. */
export function nextTrigger(current: Trigger): Trigger {
  const idx = TRIGGER_ORDER.indexOf(current);
  return TRIGGER_ORDER[(idx + 1) % TRIGGER_ORDER.length];
}
