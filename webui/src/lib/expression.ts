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
//   variant  := raw | sae | sae-<release> | role | role-<name>
//             | from | from-<safe-source-model>
//
// A ``%`` selector is the manifold operator — ``<manifold> % a,b,...``
// places generation at a point of a fitted steering manifold.  The
// coefficient is the blend fraction.
//
// One unified steer rack is the source of truth — a ``Map<string,
// SteerEntry>`` where each entry is ``mode: "subspace"`` (flat: coords,
// label, variant, trigger, enabled — magnitude is the rack-level
// ``subspaceAlong`` master) or ``mode: "manifold"`` (curved: blend, onto,
// coords, label, variant, trigger, enabled).  Both serialize as ``%`` terms;
// the magnitude slot is the shared ``subspaceAlong`` for subspace terms and
// the per-card ``along[,onto]`` for manifold terms.
//
// ``parseExpression`` returns ``{ rack, subspaceAlong, warnings }``: a curved
// ``%`` (or any ``onto`` coeff) lands ``manifold``, else ``subspace``; a
// bare-pole term (``0.5 formal.casual``) is converted to a label-form
// subspace term toward the signed pole, its magnitude collected into
// ``subspaceAlong``.  The pre-4.1 ``~``/``|`` projection and ``!`` ablation
// no longer have a rack home: they parse (so pasted expressions don't throw)
// but the operator is dropped with a warning.  ``:variant`` survives on the
// atom.
//
// Round-trip property: parse(serialize(rack, G)) reproduces ``rack`` (and
// ``G``) for any disabled-stripped rack the serializer emits.

import type {
  ManifoldSteerEntry,
  SteerEntry,
  SubspaceSteerEntry,
  Trigger,
  Variant,
} from "./types";

/** Result of ``parseExpression`` — the hydrated rack plus the recovered
 *  ``subspaceAlong`` master and any soft warnings (dropped projection /
 *  ablation operators) the caller should surface. */
export interface ParsedExpression {
  rack: Map<string, SteerEntry>;
  subspaceAlong: number;
  warnings: string[];
}

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

/** Render the rack as a canonical expression string.  Disabled entries are
 * skipped; subspace (flat) terms emit first, then manifold (curved) terms.
 * Every term is a ``%`` position; the coefficient slot is the shared
 * ``subspaceAlong`` master for subspace terms and the per-card
 * ``along[,onto]`` for manifold terms.  Empty/all-disabled racks return "". */
export function serializeExpression(
  rack: Map<string, SteerEntry>,
  subspaceAlong = 1,
): string {
  const parts: string[] = [];
  for (const [name, entry] of rack) {
    if (entry.enabled && entry.mode === "subspace") {
      parts.push(formatSubspaceTerm(name, entry, subspaceAlong));
    }
  }
  for (const [name, entry] of rack) {
    if (entry.enabled && entry.mode === "manifold") {
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

/** Render one subspace (flat) term — ``<subspaceAlong> name[:variant]%<pos>``
 *  plus an optional ``@trigger``.  All subspace terms share the one
 *  ``subspaceAlong`` magnitude (the merged affine subspace slides once);
 *  relative weight between them lives in how far each position sits from
 *  neutral.  ``<pos>`` is the label form (``personas%hacker``) when
 *  ``entry.label`` is set, else the comma-joined coord list. */
function formatSubspaceTerm(
  name: string,
  entry: SubspaceSteerEntry,
  subspaceAlong: number,
): string {
  const position = entry.label ?? entry.coords.map((c) => formatCoeff(c)).join(",");
  const selector = `${nameWithVariant(name, entry.variant)}%${position}`;
  return `${formatCoeff(subspaceAlong)} ${selector}${formatTriggerSuffix(entry.trigger)}`;
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

/** Render one manifold (curved) rack entry — ``<along[,onto]>
 *  <name>[:variant]%<position>`` plus an optional ``@trigger`` suffix.
 *  Coefficient is the per-card blend fraction (``-`` rides the joiner).
 *  ``<position>`` is the label form (``emotions%happy``) when ``entry.label`` is
 *  set; otherwise the comma-joined coord list (``emotions%0.3,0.8``). */
function formatManifoldTerm(name: string, entry: ManifoldSteerEntry): string {
  const position = entry.label
    ? entry.label
    : entry.coords.map((c) => formatCoeff(c)).join(",");
  const selector = `${nameWithVariant(name, entry.variant)}%${position}`;
  // Coefficient slot: ``along`` alone, or ``along,onto`` when onto is set
  // (> 0) — the curved-manifold residual-collapse fraction.
  const coeff =
    (entry.onto ?? 0) > 0
      ? `${formatCoeff(entry.blend)},${formatCoeff(entry.onto)}`
      : formatCoeff(entry.blend);
  return `${coeff} ${selector}${formatTriggerSuffix(entry.trigger)}`;
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

/** A manifold term — ``<coeff> <name>%<position>``.  Discriminated off
 *  ``kind`` so ``parseExpression`` can fold the two streams into their
 *  respective racks.  ``label`` is the label-form payload
 *  (``persona%pirate`` — Phase B sugar); ``coords`` is the coord-form
 *  payload.  Exactly one is non-null. */
interface ManifoldTerm {
  kind: "manifold";
  /** Display name of the manifold (``ns/name`` or bare ``name``). */
  name: string;
  /** ``along`` — the first (and representative) coefficient. */
  coeff: number;
  /** ``onto`` — the second coefficient when the slot was a comma-run
   *  (``along,onto``); null for a single-coeff term. */
  onto: number | null;
  coords: number[] | null;
  label: string | null;
  /** Tensor variant off the atom (``personas:sae%hacker``). */
  variant: Variant;
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
    // The coefficient slot is a comma-run of <= 2 (along[,onto]); the second
    // value is the curved-manifold ``onto``.  Lexically unambiguous from the
    // post-``%`` position commas — this run precedes the selector.  Each
    // value carries the term's leading sign (mirrors the engine grammar).
    const coeffs: number[] = [coeff];
    if (this.peek().kind === "NUM") {
      coeff = sign * (this.consume().value as number);
      explicit = true;
      coeffs[0] = coeff;
      while (this.peek().kind === "COMMA") {
        this.consume();
        coeffs.push(sign * this.signedNum());
        if (coeffs.length > 2) {
          const bad = this.peek(-1);
          throw new ExpressionParseError(
            "a manifold coefficient slot takes at most 2 comma-separated " +
              "values (along, onto)",
            this.src,
            bad.col,
          );
        }
      }
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
      // Label form (``persona%pirate``) vs coord form
      // (``persona%0.3,0.8``).  Disambiguate on the next token: a
      // bare identifier is a node label, everything else (NUM, signed
      // NUM) starts a coord list.  Mirrors the Python grammar parser.
      let coords: number[] | null = null;
      let label: string | null = null;
      if (this.peek().kind === "IDENT") {
        label = String(this.consume().value);
      } else {
        coords = this.coordList();
      }
      const trigger = this.optTrigger();
      return {
        kind: "manifold",
        name: atomKey(base),
        coeff,
        onto: coeffs.length > 1 ? coeffs[1] : null,
        coords,
        label,
        variant: base.variant,
        trigger,
      };
    }

    if (coeffs.length > 1) {
      throw new ExpressionParseError(
        "a comma coefficient (along,onto) is only valid on a manifold '%' term",
        this.src,
        base.col,
      );
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
      if (
        vs === "raw" ||
        vs === "sae" ||
        vs === "role" ||
        vs === "from"
      ) {
        variant = vs as Variant;
      } else if (
        vs.startsWith("sae-") ||
        vs.startsWith("role-") ||
        vs.startsWith("from-")
      ) {
        variant = vs as Variant;
      } else {
        throw new ExpressionParseError(
          `unknown variant ':${vs}'; expected raw, sae, sae-<release>, role, role-<name>, from, or from-<safe-source-model>`,
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

/** Hydrate the unified steer rack from an expression string.  Returns
 *  ``{ rack, subspaceAlong, warnings }``.  Rack keys use the atom's display
 *  form (``ns/foo``, ``happy.sad``, ``personas``) — variants live on the
 *  entry, not the key.  Classification:
 *
 *   - A ``%`` term with an ``onto`` coeff, or whose ``isFlat(name)`` is
 *     ``false`` (a curved fit per the catalog), lands a ``manifold`` entry
 *     with its per-card ``along``/``onto``; otherwise a ``subspace`` entry
 *     whose magnitude is collected into the shared ``subspaceAlong``.
 *   - A bare-pole term (``0.5 formal.casual``) becomes a label-form
 *     ``subspace`` entry toward the signed pole, magnitude → ``subspaceAlong``.
 *   - ``~``/``|`` projection and ``!`` ablation no longer have a rack home:
 *     they parse (so pasted expressions don't throw) but the operator is
 *     dropped and a warning recorded; the base concept stays as a subspace
 *     term.
 *
 *  ``opts.isFlat`` lets the caller resolve flat-vs-curved off the live
 *  catalog (``manifoldByName(...).fit_mode``); the default treats unknown
 *  manifolds as flat (subspace). */
export function parseExpression(
  expr: string,
  opts: { isFlat?: (name: string) => boolean } = {},
): ParsedExpression {
  if (!expr || !expr.trim()) {
    throw new ExpressionParseError("empty steering expression", expr, null);
  }
  const isFlat = opts.isFlat ?? (() => true);
  const toks = lex(expr);
  const terms = new Parser(toks, expr).parse();
  const rack = new Map<string, SteerEntry>();
  const warnings: string[] = [];
  let subspaceAlong: number | null = null;

  // The merged affine subspace has one slide, so subspace terms share one
  // ``subspaceAlong``.  The first subspace term sets it; a later one whose
  // magnitude differs folds onto it with a warning (lossy, but the
  // serializer only ever emits uniform output, so round-trips are exact).
  function noteSubspaceAlong(name: string, mag: number): void {
    if (subspaceAlong === null) {
      subspaceAlong = mag;
    } else if (Math.abs(mag - subspaceAlong) > 1e-9) {
      warnings.push(
        `'${name}': magnitude ${formatCoeff(mag)} folded onto the shared ` +
          `subspace-along ${formatCoeff(subspaceAlong)} (subspace terms share one along)`,
      );
    }
  }

  function ensureUnique(name: string): void {
    if (rack.has(name)) {
      throw new ExpressionParseError(`'${name}' appears more than once`, expr, null);
    }
  }

  for (const term of terms) {
    if (term.kind === "manifold") {
      const trigger: Trigger = term.trigger ? KEYWORD_TO_TRIGGER[term.trigger] : "BOTH";
      ensureUnique(term.name);
      // Curved iff a 2-coeff (along,onto) slot was given, or the catalog
      // says this fit is non-flat.  Else flat (subspace).
      const curved = term.onto !== null || isFlat(term.name) === false;
      if (curved) {
        rack.set(term.name, {
          mode: "manifold",
          blend: term.coeff,
          onto: term.onto ?? 0,
          coords: term.coords ?? [],
          label: term.label,
          variant: term.variant,
          trigger,
          enabled: true,
        });
      } else {
        noteSubspaceAlong(term.name, term.coeff);
        rack.set(term.name, {
          mode: "subspace",
          coords: term.coords ?? [],
          label: term.label,
          variant: term.variant,
          trigger,
          enabled: true,
        });
      }
      continue;
    }

    // A "vector" AST term — a flat concept addressed as a pole/DiM atom.
    // It lowers to a label-form subspace term toward the signed pole; the
    // pre-4.1 projection / ablation operators are dropped with a warning.
    const sel = term.selector;
    const baseKey = atomKey(sel.base);
    const trigger: Trigger = term.trigger ? KEYWORD_TO_TRIGGER[term.trigger] : "BOTH";
    if (term.ablation) {
      warnings.push(
        `'!${baseKey}': ablation is no longer authorable in the rack; steering toward the concept instead`,
      );
    }
    if (sel.operator !== null) {
      warnings.push(
        `'${baseKey}${sel.operator}${atomKey(sel.onto!)}': projection is no longer authorable in the rack; the '${sel.operator}' operator was dropped`,
      );
    }
    ensureUnique(baseKey);
    // Signed pole → node label; magnitude → the shared master.  Bipolar
    // concepts (``a.b``) pick ``a`` for coeff ≥ 0 and ``b`` for coeff < 0;
    // a monopolar concept keeps the (signed) coeff on the master.
    const poles = sel.base.concept.split(".");
    const negative = poles.length > 1 ? poles[1] : null;
    const label = negative !== null ? (term.coeff >= 0 ? poles[0] : negative) : poles[0];
    const mag = negative !== null ? Math.abs(term.coeff) : term.coeff;
    noteSubspaceAlong(baseKey, mag);
    rack.set(baseKey, {
      mode: "subspace",
      coords: [],
      label,
      variant: sel.base.variant,
      trigger,
      enabled: true,
    });
  }
  return { rack, subspaceAlong: subspaceAlong ?? DEFAULT_COEFF, warnings };
}
