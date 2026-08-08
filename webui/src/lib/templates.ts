// Client-side mirror of the standalone template artifact's invariants.
//
// THE one validator for every template-authoring surface (TemplateLabDrawer's
// build tab, ManifoldBuilderDrawer's templated tab).  It reproduces
// ``saklas/io/templates.py::_validate_body`` / ``_validate_context``
// one-for-one so a draft that passes here is a draft the server accepts —
// the point is that the form goes red where it is wrong instead of filling
// out cleanly and 400-ing on submit.
//
// Keep this in lockstep with the engine when the invariants move.  The
// server stays the authority; this only saves the round trip.

/** Template value → node label.  Mirrors ``io/templates.py::_slug_value``:
 *  ``[^a-z0-9]+ -> _`` over the lowercased value, edge underscores
 *  trimmed.  Note this is stricter than the manifold *name* slug — a
 *  template value's slug becomes a node label, which admits no ``.``. */
export function slugTemplateValue(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

/** Mirrors ``io/templates.py::_LABEL_REGEX``. */
const LABEL_REGEX = /^[a-z][a-z0-9_-]{0,63}$/;

export type TemplateDraftRole = "system" | "user" | "assistant";

export interface TemplateDraftTurn {
  role: TemplateDraftRole;
  content: string;
}

/** One authored context: the shared history plus the slotted final
 *  assistant turn the value is read from. */
export interface TemplateDraftContext {
  turns: TemplateDraftTurn[];
  assistant: string;
}

export interface TemplateDraft {
  slot: string;
  values: string[];
  contexts: TemplateDraftContext[];
}

export interface ValidateTemplateOpts {
  /** Noun the messages use for a context row — the builder's templated tab
   *  calls them "template", TemplateLab calls them "context". */
  contextLabel?: string;
}

/** Validate an authored template draft against the engine's invariants.
 *
 *  Returns one message per violation, empty when the draft is submittable.
 *  Callers add their own surface-specific checks (artifact name, fit
 *  hyperparams) — those are not template invariants.
 */
export function validateTemplateDraft(
  draft: TemplateDraft,
  opts: ValidateTemplateOpts = {},
): string[] {
  const label = opts.contextLabel ?? "context";
  const errs: string[] = [];
  const slot = draft.slot.trim();
  if (!slot) errs.push("slot required");

  const values = draft.values.filter((v) => v.trim());
  if (values.length < 2) errs.push("≥ 2 values");
  const seen = new Set<string>();
  for (const value of values) {
    const node = slugTemplateValue(value);
    if (!LABEL_REGEX.test(node)) {
      errs.push(`value "${value}" is not a valid node label`);
    } else if (seen.has(node)) {
      errs.push(`value "${value}" collides with another value's label`);
    } else {
      seen.add(node);
    }
  }

  if (draft.contexts.length === 0) errs.push(`at least one ${label} required`);
  draft.contexts.forEach((ctx, i) => {
    const where = `${label} ${i + 1}`;
    if (ctx.turns.length === 0) {
      errs.push(`${where}: needs a history turn`);
    } else {
      if (ctx.turns.some((t) => !t.content.trim())) {
        errs.push(`${where}: every history turn needs text`);
      }
      if (ctx.turns[ctx.turns.length - 1].role !== "user") {
        errs.push(`${where}: last turn must be user`);
      }
      // History is shared common-mode across the values, so the slot lives
      // only in the final assistant turn.
      if (slot && ctx.turns.some((t) => t.content.includes(slot))) {
        errs.push(`${where}: slot must not appear in a history turn`);
      }
    }
    if (!ctx.assistant.trim()) {
      errs.push(`${where}: assistant turn required`);
    } else if (slot) {
      // Exactly once — the value is read at a single position.
      const n = ctx.assistant.split(slot).length - 1;
      if (n !== 1) {
        errs.push(`${where}: slot must appear once in the assistant turn`);
      }
    }
  });
  return errs;
}
