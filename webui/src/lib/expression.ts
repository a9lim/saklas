// Bidirectional steering-expression handling.
//
// Mirrors saklas/core/steering_expr.py:
//
//   expr     := term (("+" | "-") term)*
//   term     := [coeff "*"?] ["!"] selector ["@" trigger]
//   selector := atom (("~" | "|") atom)?
//   atom     := [ns "/"] NAME ["." NAME] [":" variant]
//   trigger  := before|after|both|thinking|response|prompt|generated
//   variant  := raw | sae | sae-<release>
//
// The rack is the source of truth — VectorRackEntry stores α, trigger,
// variant, projection, ablate, enabled.  serializeExpression emits the
// canonical string the server's parser accepts; parseExpression hydrates
// a fresh rack Map from a pasted expression.
//
// Round-trip property: parseExpression(serializeExpression(rack)) === rack
// for any disabled-stripped rack the parser can express.

import type { ProjectionSpec, Trigger, Variant, VectorRackEntry } from "./types";

// ----------------------------------------------- triggers ----------------

/** Default coefficient for plain additive terms when the user omits the
 * number (matches DEFAULT_COEFF in steering_expr.py). */
export const DEFAULT_COEFF = 0.5;
/** Default coefficient for ablation (!x) terms — fully replace. */
export const DEFAULT_ABLATION_COEFF = 1.0;

/** Map UI Trigger values to the keyword the formatter emits.  Aliases
 * (PROMPT->before, GENERATED->response) collapse onto the canonical
 * render so round-trips are deterministic.  BOTH is the default and is
 * omitted from the output. */
const TRIGGER_TO_KEYWORD: Record<Trigger, string | null> = {
  BOTH: null,
  BEFORE: "before",
  AFTER: "after",
  THINKING: "thinking",
  RESPONSE: "response",
  PROMPT: "before",
  GENERATED: "response",
};

/** Inverse map for parsing — accepts every grammar keyword and lands on
 * the canonical UI Trigger so re-serialize emits the canonical form. */
const KEYWORD_TO_TRIGGER: Record<string, Trigger> = {
  both: "BOTH",
  before: "BEFORE",
  after: "AFTER",
  thinking: "THINKING",
  response: "RESPONSE",
  prompt: "BEFORE",
  generated: "RESPONSE",
};

// ----------------------------------------------- error type --------------

export class ExpressionParseError extends Error {
  readonly col: number | null;
  readonly source: string;
  constructor(msg: string, source: string, col: number | null) {
    super(col !== null ? `${msg} (col ${col})` : msg);
    this.name = "ExpressionParseError";
    this.col = col;
    this.source = source;
  }
}

// ----------------------------------------------- formatter ---------------

/** Render the rack as a canonical expression string.  Disabled entries
 * are skipped.  Empty/all-disabled rack returns "". */
export function serializeExpression(
  rack: Map<string, VectorRackEntry>,
): string {
  const parts: string[] = [];
  for (const [name, entry] of rack) {
    if (!entry.enabled) continue;
    parts.push(formatTerm(name, entry));
  }
  if (parts.length === 0) return "";
  // First term keeps its leading sign verbatim; subsequent terms join
  // with " + " or " - " depending on coefficient sign.
  let out = parts[0];
  for (let i = 1; i < parts.length; i++) {
    const p = parts[i];
    if (p.startsWith("-")) out += " - " + p.slice(1).trimStart();
    else out += " + " + p;
  }
  return out;
}

function formatTerm(name: string, entry: VectorRackEntry): string {
  const coeff = entry.alpha;
  const triggerSuffix = formatTriggerSuffix(entry.trigger);
  if (entry.ablate) {
    // !x (coeff=1) and -!x (coeff=-1) are the bare canonical forms;
    // anything else carries the explicit number.  Parser canonicalizes
    // 1.0 !x back to !x on round-trip.
    let body: string;
    if (coeff === 1.0) body = `!${nameWithVariant(name, entry.variant)}`;
    else if (coeff === -1.0) body = `-!${nameWithVariant(name, entry.variant)}`;
    else body = `${formatCoeff(coeff)} !${nameWithVariant(name, entry.variant)}`;
    return body + triggerSuffix;
  }
  const head = nameWithVariant(name, entry.variant);
  let selector = head;
  if (entry.projection) {
    selector = `${head}${entry.projection.op}${entry.projection.target}`;
  }
  return `${formatCoeff(coeff)} ${selector}${triggerSuffix}`;
}

function nameWithVariant(name: string, variant: Variant): string {
  return variant === "raw" ? name : `${name}:${variant}`;
}

function formatCoeff(coeff: number): string {
  // Number.toString trims trailing zeros while preserving precision.
  // The leading sign rides the term; the joiner above splits +/-.
  if (Number.isNaN(coeff) || !Number.isFinite(coeff)) return "0";
  return String(coeff);
}

function formatTriggerSuffix(trigger: Trigger): string {
  const kw = TRIGGER_TO_KEYWORD[trigger];
  return kw ? `@${kw}` : "";
}

// ----------------------------------------------- lexer -------------------

interface Tok {
  kind: TokKind;
  value: string | number;
  col: number;
}

type TokKind =
  | "NUM"
  | "IDENT"
  | "DOT"
  | "SLASH"
  | "COLON"
  | "STAR"
  | "PLUS"
  | "MINUS"
  | "AT"
  | "TILDE"
  | "ORTHO"
  | "BANG"
  | "EOF";

const SINGLE_CHAR: Record<string, TokKind> = {
  ".": "DOT",
  "/": "SLASH",
  ":": "COLON",
  "*": "STAR",
  "+": "PLUS",
  "-": "MINUS",
  "@": "AT",
  "~": "TILDE",
  "|": "ORTHO",
  "!": "BANG",
};

const NUM_RE = /^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/;
const IDENT_START_RE = /[A-Za-z]/;
const IDENT_CHAR_RE = /[A-Za-z0-9_]/;

function lex(text: string): Tok[] {
  const out: Tok[] = [];
  const n = text.length;
  let i = 0;
  while (i < n) {
    const c = text[i];
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    // Number — must precede the DOT branch because .25 starts with a dot.
    if (/[0-9]/.test(c) || (c === "." && i + 1 < n && /[0-9]/.test(text[i + 1]))) {
      const m = NUM_RE.exec(text.slice(i));
      if (!m) throw new ExpressionParseError(`malformed number at ${c}`, text, i);
      out.push({ kind: "NUM", value: parseFloat(m[0]), col: i });
      i += m[0].length;
      continue;
    }
    if (c in SINGLE_CHAR) {
      out.push({ kind: SINGLE_CHAR[c], value: c, col: i });
      i++;
      continue;
    }
    if (IDENT_START_RE.test(c)) {
      const start = i;
      i++;
      while (i < n) {
        const ch = text[i];
        if (IDENT_CHAR_RE.test(ch)) {
          i++;
          continue;
        }
        // Internal dash (sae-release) needs an ident-char on both sides.
        if (
          ch === "-" &&
          i + 1 < n &&
          IDENT_CHAR_RE.test(text[i + 1])
        ) {
          i++;
          continue;
        }
        break;
      }
      out.push({ kind: "IDENT", value: text.slice(start, i), col: start });
      continue;
    }
    if (c === '"' || c === "'") {
      throw new ExpressionParseError(
        "quoted identifiers are not supported; use underscores for multi-word concept names",
        text,
        i,
      );
    }
    throw new ExpressionParseError(`unexpected character ${c}`, text, i);
  }
  out.push({ kind: "EOF", value: "", col: n });
  return out;
}

// ----------------------------------------------- parser ------------------

interface Atom {
  namespace: string | null;
  concept: string; // may contain a single '.' joining two poles
  variant: Variant;
  col: number;
}

interface Selector {
  base: Atom;
  operator: "~" | "|" | null;
  onto: Atom | null;
}

interface Term {
  coeff: number;
  selector: Selector;
  trigger: string | null;
  explicitCoeff: boolean;
  ablation: boolean;
}

class Parser {
  private pos = 0;
  constructor(private toks: Tok[], private src: string) {}

  private peek(off = 0): Tok {
    return this.toks[this.pos + off];
  }
  private consume(): Tok {
    return this.toks[this.pos++];
  }
  private expect(kind: TokKind): Tok {
    const t = this.peek();
    if (t.kind !== kind) {
      throw new ExpressionParseError(
        `expected ${kind}, got ${t.kind} (${t.value})`,
        this.src,
        t.col,
      );
    }
    return this.consume();
  }

  parse(): Term[] {
    let sign = 1;
    if (this.peek().kind === "PLUS" || this.peek().kind === "MINUS") {
      sign = this.consume().kind === "MINUS" ? -1 : 1;
    }
    const terms = [this.term(sign)];
    while (this.peek().kind === "PLUS" || this.peek().kind === "MINUS") {
      const opSign = this.consume().kind === "MINUS" ? -1 : 1;
      terms.push(this.term(opSign));
    }
    if (this.peek().kind !== "EOF") {
      const t = this.peek();
      throw new ExpressionParseError(
        `unexpected token ${t.kind} (${t.value}) after a complete term`,
        this.src,
        t.col,
      );
    }
    return terms;
  }

  private term(sign: number): Term {
    let explicit = false;
    let coeff = sign * DEFAULT_COEFF;
    if (this.peek().kind === "NUM") {
      coeff = sign * (this.consume().value as number);
      explicit = true;
      if (this.peek().kind === "STAR") this.consume();
    }
    let ablation = false;
    if (this.peek().kind === "BANG") {
      this.consume();
      ablation = true;
      if (!explicit) coeff = sign * DEFAULT_ABLATION_COEFF;
    }
    const selector = this.selector();
    let trigger: string | null = null;
    if (this.peek().kind === "AT") {
      this.consume();
      const tok = this.expect("IDENT");
      trigger = String(tok.value);
      if (!(trigger in KEYWORD_TO_TRIGGER)) {
        const valid = Object.keys(KEYWORD_TO_TRIGGER).sort().join(", ");
        throw new ExpressionParseError(
          `unknown trigger '@${trigger}'; valid: ${valid}`,
          this.src,
          tok.col,
        );
      }
    }
    return { coeff, selector, trigger, explicitCoeff: explicit, ablation };
  }

  private selector(): Selector {
    const base = this.atom();
    if (this.peek().kind === "TILDE" || this.peek().kind === "ORTHO") {
      const opTok = this.consume();
      const operator: "~" | "|" = opTok.kind === "TILDE" ? "~" : "|";
      const onto = this.atom();
      if (this.peek().kind === "TILDE" || this.peek().kind === "ORTHO") {
        const t = this.peek();
        throw new ExpressionParseError(
          "chained projection is not allowed; use one '~' or '|' per term",
          this.src,
          t.col,
        );
      }
      return { base, operator, onto };
    }
    return { base, operator: null, onto: null };
  }

  private atom(): Atom {
    const first = this.expect("IDENT");
    const col = first.col;
    let namespace: string | null = null;
    let concept = String(first.value);
    if (this.peek().kind === "SLASH") {
      this.consume();
      const second = this.expect("IDENT");
      namespace = concept;
      concept = String(second.value);
    }
    if (this.peek().kind === "DOT") {
      this.consume();
      const rhs = this.expect("IDENT");
      concept = `${concept}.${rhs.value}`;
    }
    let variant: Variant = "raw";
    if (this.peek().kind === "COLON") {
      this.consume();
      const v = this.expect("IDENT");
      const vs = String(v.value);
      if (vs === "sae" || vs === "raw") {
        variant = vs as Variant;
      } else if (vs.startsWith("sae-")) {
        variant = vs as Variant;
      } else {
        throw new ExpressionParseError(
          `unknown variant ':${vs}'; expected raw, sae, or sae-<release>`,
          this.src,
          v.col,
        );
      }
    }
    return { namespace, concept, variant, col };
  }
}

// ----------------------------------------------- fold to rack ------------

function atomKey(atom: Atom): string {
  // ns/ prefix preserved if user typed it; concept may contain '.'.
  return atom.namespace ? `${atom.namespace}/${atom.concept}` : atom.concept;
}

/** Hydrate a rack Map from an expression string.  Rack keys use the
 * atom's display form (ns/foo, happy.sad) — variants live on the
 * entry, not in the key, so two terms differing only by variant collide
 * (matching the saklas parser's Steering.alphas semantics; users wanting
 * both should ablate-or-merge explicitly). */
export function parseExpression(expr: string): Map<string, VectorRackEntry> {
  if (!expr || !expr.trim()) {
    throw new ExpressionParseError("empty steering expression", expr, null);
  }
  const toks = lex(expr);
  const terms = new Parser(toks, expr).parse();
  const rack = new Map<string, VectorRackEntry>();

  for (const term of terms) {
    const sel = term.selector;
    const baseKey = atomKey(sel.base);
    const trigger: Trigger = term.trigger
      ? KEYWORD_TO_TRIGGER[term.trigger]
      : "BOTH";

    if (term.ablation) {
      if (sel.operator !== null) {
        throw new ExpressionParseError(
          "ablation does not compose with projection operators",
          expr,
          sel.base.col,
        );
      }
      mergeAblation(rack, baseKey, term.coeff, trigger, sel.base.variant);
      continue;
    }

    if (sel.operator !== null) {
      // Projection term: base + (~|) + onto.
      const onto = sel.onto!;
      const projection: ProjectionSpec = {
        op: sel.operator,
        target: atomKey(onto),
      };
      mergeProjected(
        rack,
        baseKey,
        term.coeff,
        trigger,
        sel.base.variant,
        projection,
      );
      continue;
    }

    mergePlain(rack, baseKey, term.coeff, trigger, sel.base.variant);
  }
  return rack;
}

function mergePlain(
  rack: Map<string, VectorRackEntry>,
  key: string,
  coeff: number,
  trigger: Trigger,
  variant: Variant,
): void {
  const existing = rack.get(key);
  if (!existing) {
    rack.set(key, {
      alpha: coeff,
      trigger,
      variant,
      projection: null,
      ablate: false,
      enabled: true,
    });
    return;
  }
  if (existing.ablate) {
    throw new ExpressionParseError(
      `concept '${key}' appears as both plain and ablation`,
      key,
      null,
    );
  }
  if (existing.projection) {
    throw new ExpressionParseError(
      `concept '${key}' appears both as plain and projection target; use distinct names`,
      key,
      null,
    );
  }
  if (existing.trigger !== trigger) {
    throw new ExpressionParseError(
      `concept '${key}' appears with conflicting triggers`,
      key,
      null,
    );
  }
  if (existing.variant !== variant) {
    throw new ExpressionParseError(
      `concept '${key}' appears with conflicting variants`,
      key,
      null,
    );
  }
  existing.alpha += coeff;
}

function mergeProjected(
  rack: Map<string, VectorRackEntry>,
  baseKey: string,
  coeff: number,
  trigger: Trigger,
  variant: Variant,
  projection: ProjectionSpec,
): void {
  const existing = rack.get(baseKey);
  if (!existing) {
    rack.set(baseKey, {
      alpha: coeff,
      trigger,
      variant,
      projection,
      ablate: false,
      enabled: true,
    });
    return;
  }
  if (existing.ablate || !existing.projection) {
    throw new ExpressionParseError(
      `projection '${baseKey}' conflicts with a plain entry of the same name`,
      baseKey,
      null,
    );
  }
  if (
    existing.projection.op !== projection.op ||
    existing.projection.target !== projection.target ||
    existing.trigger !== trigger ||
    existing.variant !== variant
  ) {
    throw new ExpressionParseError(
      `projection '${baseKey}' appears with conflicting attributes`,
      baseKey,
      null,
    );
  }
  existing.alpha += coeff;
}

function mergeAblation(
  rack: Map<string, VectorRackEntry>,
  baseKey: string,
  coeff: number,
  trigger: Trigger,
  variant: Variant,
): void {
  const existing = rack.get(baseKey);
  if (!existing) {
    rack.set(baseKey, {
      alpha: coeff,
      trigger,
      variant,
      projection: null,
      ablate: true,
      enabled: true,
    });
    return;
  }
  if (!existing.ablate) {
    throw new ExpressionParseError(
      `ablation '!${baseKey}' conflicts with a non-ablation entry of the same name`,
      baseKey,
      null,
    );
  }
  if (existing.trigger !== trigger || existing.variant !== variant) {
    throw new ExpressionParseError(
      `ablation '!${baseKey}' appears with conflicting attributes`,
      baseKey,
      null,
    );
  }
  existing.alpha += coeff;
}
