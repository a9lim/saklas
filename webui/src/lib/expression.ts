// Bidirectional steering-expression handling.
//
// Mirrors saklas/core/steering_expr.py:
//
//   expr     := term (("+" | "-") term)*
//   term     := [coeff "*"?] ["!"] selector ["@" trigger]
//   selector := atom (("~" | "|") atom | "%" coord_list)?
//   coord_list := signed_num ("," signed_num)*
//   atom     := [ns "/"] NAME ["." NAME] [":" variant]
//   trigger  := before|after|both|thinking|response|prompt|generated
//   variant  := raw | sae | sae-<release>
//
// A ``%`` selector is the manifold operator — ``<manifold> % a,b,...``
// places generation at a point of a fitted steering manifold.  The
// coefficient is the blend fraction.
//
// The racks are the source of truth — VectorRackEntry stores α, trigger,
// variant, projection, ablate, enabled; ManifoldRackEntry stores blend,
// coords, trigger, enabled.  serializeExpression emits the canonical
// string the server's parser accepts; parseExpression hydrates fresh
// rack Maps from a pasted expression.
//
// Round-trip property: parseExpression(serializeExpression(v, m)) ===
// {v, m} for any disabled-stripped racks the parser can express.

import type {
  ManifoldRackEntry,
  ProjectionSpec,
  Trigger,
  Variant,
  VectorRackEntry,
} from "./types";

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

/** Render the racks as a canonical expression string.  Disabled
 * entries are skipped; manifold terms are emitted after vector terms.
 * Empty/all-disabled racks return "". */
export function serializeExpression(
  rack: Map<string, VectorRackEntry>,
  manifolds?: Map<string, ManifoldRackEntry>,
): string {
  const parts: string[] = [];
  for (const [name, entry] of rack) {
    if (!entry.enabled) continue;
    parts.push(formatTerm(name, entry));
  }
  if (manifolds) {
    for (const [name, entry] of manifolds) {
      if (!entry.enabled) continue;
      parts.push(formatManifoldTerm(name, entry));
    }
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

/** Render one manifold rack entry — ``<coeff> <name>%<c0,c1,...>`` plus
 *  an optional ``@trigger`` suffix.  The coefficient is the blend
 *  fraction; ``-`` rides the joiner like vector terms. */
function formatManifoldTerm(name: string, entry: ManifoldRackEntry): string {
  const coords = entry.coords.map((c) => formatCoeff(c)).join(",");
  const selector = `${name}%${coords}`;
  return `${formatCoeff(entry.blend)} ${selector}${formatTriggerSuffix(entry.trigger)}`;
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
  | "PERCENT"
  | "COMMA"
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
  "%": "PERCENT",
  ",": "COMMA",
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

/** A vector term — the classic coeff·selector·trigger shape. */
interface VectorTerm {
  kind: "vector";
  coeff: number;
  selector: Selector;
  trigger: string | null;
  explicitCoeff: boolean;
  ablation: boolean;
}

/** A manifold term — ``<coeff> <name>%<coords>``.  Discriminated off
 *  ``kind`` so ``parseExpression`` can fold the two streams into their
 *  respective racks. */
interface ManifoldTerm {
  kind: "manifold";
  /** Display name of the manifold (``ns/name`` or bare ``name``). */
  name: string;
  coeff: number;
  coords: number[];
  trigger: string | null;
}

type Term = VectorTerm | ManifoldTerm;

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
    // Parse the base atom up front so a trailing ``%`` can fork into the
    // manifold production before the vector projection grammar runs.
    const base = this.atom();

    if (this.peek().kind === "PERCENT") {
      if (ablation) {
        const t = this.peek();
        throw new ExpressionParseError(
          "manifold terms ('%') do not compose with ablation ('!')",
          this.src,
          t.col,
        );
      }
      this.consume(); // '%'
      const coords = this.coordList();
      const trigger = this.optTrigger();
      return {
        kind: "manifold",
        name: atomKey(base),
        coeff,
        coords,
        trigger,
      };
    }

    const selector = this.selector(base);
    const trigger = this.optTrigger();
    return {
      kind: "vector",
      coeff,
      selector,
      trigger,
      explicitCoeff: explicit,
      ablation,
    };
  }

  /** Parse an optional ``@trigger`` suffix. */
  private optTrigger(): string | null {
    if (this.peek().kind !== "AT") return null;
    this.consume();
    const tok = this.expect("IDENT");
    const trigger = String(tok.value);
    if (!(trigger in KEYWORD_TO_TRIGGER)) {
      const valid = Object.keys(KEYWORD_TO_TRIGGER).sort().join(", ");
      throw new ExpressionParseError(
        `unknown trigger '@${trigger}'; valid: ${valid}`,
        this.src,
        tok.col,
      );
    }
    return trigger;
  }

  /** Parse a non-empty comma-separated list of signed numbers — the
   *  manifold authoring coordinates.  A leading ``-`` is a sign on the
   *  number, not a term joiner, since this only runs right after ``%``. */
  private coordList(): number[] {
    const out: number[] = [this.signedNum()];
    while (this.peek().kind === "COMMA") {
      this.consume();
      out.push(this.signedNum());
    }
    return out;
  }

  private signedNum(): number {
    let sign = 1;
    if (this.peek().kind === "MINUS") {
      this.consume();
      sign = -1;
    } else if (this.peek().kind === "PLUS") {
      this.consume();
    }
    const tok = this.expect("NUM");
    return sign * (tok.value as number);
  }

  private selector(base: Atom): Selector {
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

/** Result of {@link parseExpression} — the two racks a pasted
 *  expression hydrates: classic vector terms plus manifold (``%``)
 *  terms. */
export interface ParsedExpression {
  vectors: Map<string, VectorRackEntry>;
  manifolds: Map<string, ManifoldRackEntry>;
}

/** Hydrate the vector + manifold racks from an expression string.
 *  Vector rack keys use the atom's display form (ns/foo, happy.sad) —
 *  variants live on the entry, not in the key, so two terms differing
 *  only by variant collide (matching the saklas parser's
 *  Steering.alphas semantics; users wanting both should ablate-or-merge
 *  explicitly).  Manifold rack keys are the manifold display name. */
export function parseExpression(expr: string): ParsedExpression {
  if (!expr || !expr.trim()) {
    throw new ExpressionParseError("empty steering expression", expr, null);
  }
  const toks = lex(expr);
  const terms = new Parser(toks, expr).parse();
  const rack = new Map<string, VectorRackEntry>();
  const manifolds = new Map<string, ManifoldRackEntry>();

  for (const term of terms) {
    if (term.kind === "manifold") {
      const trigger: Trigger = term.trigger
        ? KEYWORD_TO_TRIGGER[term.trigger]
        : "BOTH";
      const existing = manifolds.get(term.name);
      if (existing) {
        throw new ExpressionParseError(
          `manifold '${term.name}' appears more than once`,
          expr,
          null,
        );
      }
      manifolds.set(term.name, {
        blend: term.coeff,
        coords: term.coords,
        trigger,
        enabled: true,
      });
      continue;
    }
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
  return { vectors: rack, manifolds };
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
