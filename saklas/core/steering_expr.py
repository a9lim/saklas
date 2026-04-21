"""Unified steering-expression grammar and IR compiler.

One parser + formatter for the steering expression language shared across
every saklas input surface (Python, YAML, HTTP, TUI, CLI). Every surface
turns a user-supplied string into the same :class:`Steering` IR.

Grammar::

    expr     := term (("+" | "-") term)*
    term     := [coeff ["*"]] selector ["@" trigger]
    selector := atom (("~" | "|") atom)?
    atom     := [ns "/"] NAME ["." NAME] [":" variant]
    trigger  := "before" | "after" | "both" | "thinking" | "response"
              | "prompt" | "generated"
    coeff    := signed_float   (optional; defaults to DEFAULT_COEFF = 0.5)
    variant  := "raw" | "sae" | "sae-" ID

Concept names are ASCII identifiers: letter followed by any of
``[a-z0-9_-]``.  Multi-word concepts use underscores
(``artificial_intelligence``) — spaces separate tokens, so
``artificial intelligence`` errors with an underscore hint.  Quoted
identifiers are rejected.  Bipolar pairs join with ``.``
(``human.artificial_intelligence``).

Pole aliases (``wolf`` on top of an installed ``deer.wolf``) resolve via
:func:`saklas.cli.selectors.resolve_pole`; the sign flip folds into the
user-supplied coefficient before the term lands in
``Steering.alphas``.  Projection terms produce :class:`ProjectedTerm`
values; the session materializes them into derived profiles on scope
entry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING, cast

from saklas.core.errors import SaklasError
from saklas.core.triggers import Trigger

if TYPE_CHECKING:
    from saklas.core.steering import AlphaEntry, Steering


# Default coefficient when a term omits the explicit number.  Matches the
# ``recommended_alpha`` field on bundled packs — the observed coherent-α
# sweet spot post-share-baking.  Future hook: when a per-vector default
# alpha becomes available on the profile/pack metadata side, ``_fold``
# should consult it and fall back to this constant.  ``_Term.explicit_coeff``
# preserves the "user typed a number" signal so that late resolution can
# tell a defaulted ``honest`` from an explicit ``0.5 honest``.
DEFAULT_COEFF = 0.5


_TRIGGER_PRESETS: dict[str, Trigger] = {
    "both": Trigger.BOTH,
    "after": Trigger.AFTER_THINKING,
    "before": Trigger.PROMPT_ONLY,
    "thinking": Trigger.THINKING_ONLY,
    "response": Trigger.GENERATED_ONLY,
    "prompt": Trigger.PROMPT_ONLY,
    "generated": Trigger.GENERATED_ONLY,
}

# Preferred render string per preset.  Multiple grammar tokens alias onto
# the same Trigger (``@before`` == ``@prompt``); we pick one canonical form
# so round-tripping is deterministic.
_TRIGGER_CANONICAL: dict[Trigger, str] = {
    Trigger.BOTH: "both",
    Trigger.AFTER_THINKING: "after",
    Trigger.PROMPT_ONLY: "before",
    Trigger.THINKING_ONLY: "thinking",
    Trigger.GENERATED_ONLY: "response",
}


@dataclass(frozen=True)
class ProjectedTerm:
    """Runtime projection entry in ``Steering.alphas``.

    The session materializes a derived profile on scope entry (combining
    the ``base`` and ``onto`` profiles via ``project_profile``), registers
    it under a synthetic name ``"<base><op><onto>"``, and feeds it to the
    usual hook path.  Stored as a value inside ``Steering.alphas``; the
    key matches the synthetic name so duplicate references compose.
    """
    coeff: float
    trigger: Trigger
    operator: Literal["~", "|"]
    base: str
    onto: str


class SteeringExprError(ValueError, SaklasError):
    """Raised when a steering expression string cannot be parsed."""

    def __init__(self, msg: str, *, col: int | None = None) -> None:
        self.col = col
        if col is not None:
            msg = f"{msg} (col {col})"
        super().__init__(msg)


# ---------------------------------------------------------------- lexer ---

_NUM_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_IDENT_START_RE = re.compile(r"[A-Za-z]")
_IDENT_CHAR_RE = re.compile(r"[A-Za-z0-9_]")

_SINGLE_CHAR_TOKENS = {
    ".": "DOT", "/": "SLASH", ":": "COLON", "*": "STAR",
    "+": "PLUS", "-": "MINUS", "@": "AT", "~": "TILDE",
    "|": "ORTHO",
}


@dataclass
class _Tok:
    kind: str
    value: str | float
    col: int


def _lex(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        # Number detection must precede the single-char DOT branch, since
        # ``.25`` starts with a DOT that is actually a decimal point.
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            m = _NUM_RE.match(text, i)
            if m is None:  # pragma: no cover — isdigit guard
                raise SteeringExprError(f"malformed number at {c!r}", col=i)
            toks.append(_Tok("NUM", float(m.group()), i))
            i = m.end()
            continue
        if c in _SINGLE_CHAR_TOKENS:
            toks.append(_Tok(_SINGLE_CHAR_TOKENS[c], c, i))
            i += 1
            continue
        if _IDENT_START_RE.match(c):
            start = i
            i += 1
            while i < n:
                ch = text[i]
                if _IDENT_CHAR_RE.match(ch):
                    i += 1
                    continue
                # Dash-joined segments inside an ident (e.g. sae-release)
                # require a valid ident-char on each side; otherwise the
                # dash is a MINUS operator.
                if (
                    ch == "-"
                    and i + 1 < n
                    and _IDENT_CHAR_RE.match(text[i + 1])
                ):
                    i += 1
                    continue
                break
            toks.append(_Tok("IDENT", text[start:i], start))
            continue
        if c in ('"', "'"):
            raise SteeringExprError(
                "quoted identifiers are not supported; use underscores "
                "for multi-word concept names "
                "(e.g. 'artificial_intelligence') and '.' for bipolar "
                "pairs (e.g. 'human.artificial_intelligence')",
                col=i,
            )
        raise SteeringExprError(f"unexpected character {c!r}", col=i)
    toks.append(_Tok("EOF", "", n))
    return toks


# ------------------------------------------------------------------ ast ---

@dataclass
class _Atom:
    namespace: Optional[str]
    concept: str  # may contain a single '.' joining two poles
    variant: str  # 'raw' (default) | 'sae' | 'sae-<release>'
    col: int


@dataclass
class _Selector:
    base: _Atom
    operator: Optional[str]  # None | '~' | '|'
    onto: Optional[_Atom]


@dataclass
class _Term:
    coeff: float
    selector: _Selector
    trigger: Optional[str]  # raw trigger keyword; None = fall through
    # True iff the user typed a numeric coefficient; False when the parser
    # substituted ``DEFAULT_COEFF``.  Internal — lets a future resolver step
    # swap in per-vector defaults without re-parsing the expression.
    explicit_coeff: bool


# --------------------------------------------------------------- parser ---

class _Parser:
    def __init__(self, toks: list[_Tok]) -> None:
        self._toks = toks
        self._pos = 0

    def _peek(self, off: int = 0) -> _Tok:
        return self._toks[self._pos + off]

    def _consume(self) -> _Tok:
        t = self._toks[self._pos]
        self._pos += 1
        return t

    def _expect(self, kind: str) -> _Tok:
        t = self._peek()
        if t.kind != kind:
            raise SteeringExprError(
                f"expected {kind}, got {t.kind} ({t.value!r})", col=t.col,
            )
        return self._consume()

    def parse(self) -> list[_Term]:
        sign = +1
        if self._peek().kind in ("PLUS", "MINUS"):
            sign = -1 if self._consume().kind == "MINUS" else +1
        terms = [self._term(sign)]
        while self._peek().kind in ("PLUS", "MINUS"):
            op_sign = -1 if self._consume().kind == "MINUS" else +1
            terms.append(self._term(op_sign))
        if self._peek().kind != "EOF":
            t = self._peek()
            if t.kind == "IDENT":
                raise SteeringExprError(
                    f"unexpected identifier {t.value!r} after a complete "
                    f"term; multi-word concept names use underscores "
                    f"(e.g. 'artificial_intelligence', not "
                    f"'artificial intelligence'), and bipolar pairs join "
                    f"with '.' (e.g. 'human.artificial_intelligence')",
                    col=t.col,
                )
            raise SteeringExprError(
                f"unexpected token {t.kind} ({t.value!r})", col=t.col,
            )
        return terms

    def _term(self, sign: int) -> _Term:
        explicit = False
        coeff = float(sign) * DEFAULT_COEFF
        if self._peek().kind == "NUM":
            coeff = float(sign) * float(self._consume().value)
            explicit = True
            if self._peek().kind == "STAR":
                self._consume()
        selector = self._selector()
        trigger: str | None = None
        if self._peek().kind == "AT":
            self._consume()
            tok = self._expect("IDENT")
            trigger = str(tok.value)
            if trigger not in _TRIGGER_PRESETS:
                valid = ", ".join(sorted(_TRIGGER_PRESETS.keys()))
                raise SteeringExprError(
                    f"unknown trigger '@{trigger}'; valid: {valid}. "
                    f"Note: '@' is for triggers only; HF revisions are not "
                    f"accepted inside steering expressions.",
                    col=tok.col,
                )
        return _Term(
            coeff=coeff, selector=selector, trigger=trigger,
            explicit_coeff=explicit,
        )

    def _selector(self) -> _Selector:
        base = self._atom()
        if self._peek().kind in ("TILDE", "ORTHO"):
            op_tok = self._consume()
            op = "~" if op_tok.kind == "TILDE" else "|"
            onto = self._atom()
            if self._peek().kind in ("TILDE", "ORTHO"):
                nxt = self._peek()
                raise SteeringExprError(
                    "chained projection is not allowed; "
                    "use one '~' or '|' per term",
                    col=nxt.col,
                )
            return _Selector(base=base, operator=op, onto=onto)
        return _Selector(base=base, operator=None, onto=None)

    def _atom(self) -> _Atom:
        first = self._expect("IDENT")
        col = first.col
        namespace: str | None = None
        concept = str(first.value)
        if self._peek().kind == "SLASH":
            self._consume()
            second = self._expect("IDENT")
            namespace = concept
            concept = str(second.value)
        if self._peek().kind == "DOT":
            self._consume()
            rhs = self._expect("IDENT")
            concept = f"{concept}.{rhs.value}"
        variant = "raw"
        if self._peek().kind == "COLON":
            self._consume()
            v = self._expect("IDENT")
            variant = str(v.value)
        return _Atom(
            namespace=namespace, concept=concept, variant=variant, col=col,
        )


# ------------------------------------------------------------ resolve/fold ---

def _with_variant(canonical: str, variant: str) -> str:
    return canonical if variant == "raw" else f"{canonical}:{variant}"


def _resolve_atom(
    atom: _Atom, default_namespace: Optional[str],
) -> tuple[str, int]:
    """Return ``(alphas_key, sign_flip)`` for an atom.

    ``alphas_key`` is the key under which this atom lands in
    ``Steering.alphas``: the canonical concept name from
    ``resolve_pole``, suffixed with ``:<variant>`` when the variant is
    anything other than ``raw``.  ``sign_flip`` is +1 or -1 per
    ``resolve_pole``; callers multiply their user-supplied coefficient
    by this flip.
    """
    from saklas.cli.selectors import resolve_pole

    raw = atom.concept
    if atom.variant != "raw":
        raw = f"{raw}:{atom.variant}"
    ns = atom.namespace if atom.namespace is not None else default_namespace
    canonical, sign, _match, variant = resolve_pole(raw, namespace=ns)
    return _with_variant(canonical, variant), sign


def _merge_plain(
    alphas: "dict[str, AlphaEntry]",
    key: str,
    coeff: float,
    trig: Optional[Trigger],
) -> None:
    if key not in alphas:
        alphas[key] = coeff if trig is None else (coeff, trig)
        return
    existing = alphas[key]
    if isinstance(existing, ProjectedTerm):
        raise SteeringExprError(
            f"concept '{key}' appears both as a plain term and as a "
            f"projection target; use distinct references"
        )
    prev_coeff: float
    prev_trig: Optional[Trigger]
    if isinstance(existing, tuple):
        prev_coeff = float(existing[0])
        prev_trig = existing[1]
    else:
        prev_coeff = float(cast(float, existing))
        prev_trig = None
    if prev_trig is None and trig is None:
        alphas[key] = prev_coeff + coeff
        return
    if prev_trig is not None and trig is not None and prev_trig == trig:
        alphas[key] = (prev_coeff + coeff, trig)
        return
    raise SteeringExprError(
        f"concept '{key}' appears with conflicting triggers; "
        f"merge triggers explicitly or split into separate Steering entries"
    )


def _merge_projected(
    alphas: "dict[str, AlphaEntry]",
    key: str,
    op: Literal["~", "|"],
    base: str,
    onto: str,
    coeff: float,
    trig: Trigger,
) -> None:
    if key not in alphas:
        alphas[key] = ProjectedTerm(
            coeff=coeff, trigger=trig, operator=op, base=base, onto=onto,
        )
        return
    existing = alphas[key]
    if not isinstance(existing, ProjectedTerm):
        raise SteeringExprError(
            f"projection '{key}' conflicts with a plain entry of the same name"
        )
    if existing.trigger != trig:
        raise SteeringExprError(
            f"projection '{key}' appears with conflicting triggers"
        )
    alphas[key] = ProjectedTerm(
        coeff=existing.coeff + coeff,
        trigger=trig, operator=op, base=base, onto=onto,
    )


def _fold(terms: list[_Term], *, namespace: Optional[str]) -> "Steering":
    from saklas.core.steering import Steering

    alphas: "dict[str, AlphaEntry]" = {}
    for term in terms:
        sel = term.selector
        base_key, base_sign = _resolve_atom(sel.base, namespace)
        coeff = term.coeff * base_sign
        trig: Trigger | None
        trig = _TRIGGER_PRESETS[term.trigger] if term.trigger is not None else None
        if sel.operator is None:
            _merge_plain(alphas, base_key, coeff, trig)
            continue
        # Projection terms.  Projection math is insensitive to the sign of
        # the onto direction (``a|b`` yields the same result as
        # ``a|(-b)``); the base sign is already folded into ``coeff``.
        assert sel.onto is not None
        onto_key, _onto_sign = _resolve_atom(sel.onto, namespace)
        effective_trig = trig if trig is not None else Trigger.BOTH
        op: Literal["~", "|"] = cast(Literal["~", "|"], sel.operator)
        syn_key = f"{base_key}{op}{onto_key}"
        _merge_projected(
            alphas, syn_key, op, base_key, onto_key,
            coeff, effective_trig,
        )
    return Steering(alphas=alphas)


# ---------------------------------------------------------------- public ---

def parse_expr(
    text: str, *, namespace: Optional[str] = None,
) -> "Steering":
    """Parse a steering expression string into a :class:`Steering` IR.

    ``namespace`` scopes bare pole resolution to a single namespace; when
    ``None``, :func:`saklas.cli.selectors.resolve_pole` raises
    :class:`~saklas.cli.selectors.AmbiguousSelectorError` if a bare pole
    matches concepts across multiple namespaces.
    """
    if not text or not text.strip():
        raise SteeringExprError("empty steering expression")
    toks = _lex(text)
    terms = _Parser(toks).parse()
    return _fold(terms, namespace=namespace)


def format_expr(steering: "Steering") -> str:
    """Render a :class:`Steering` back into canonical expression form.

    Round-trips with :func:`parse_expr` for any IR produced by the parser.
    Entries whose trigger equals :data:`Trigger.BOTH` omit the ``@`` tag;
    entries whose coefficient is negative are emitted via ``-`` separators
    (first term carries the sign verbatim).
    """
    default_trig = steering.trigger
    parts: list[str] = []
    for name, val in steering.alphas.items():
        if isinstance(val, ProjectedTerm):
            parts.append(_fmt_projected(val))
            continue
        if isinstance(val, tuple):
            coeff, trig = float(val[0]), val[1]
        else:
            coeff, trig = float(val), default_trig
        parts.append(_fmt_plain(name, coeff, trig))
    if not parts:
        return ""
    out = parts[0]
    for p in parts[1:]:
        if p.startswith("-"):
            out += " - " + p[1:].lstrip()
        else:
            out += " + " + p
    return out


def _fmt_plain(name: str, coeff: float, trig: Trigger) -> str:
    body = f"{coeff:g} {name}"
    if trig != Trigger.BOTH:
        body += "@" + _trigger_name(trig)
    return body


def _fmt_projected(p: ProjectedTerm) -> str:
    body = f"{p.coeff:g} {p.base}{p.operator}{p.onto}"
    if p.trigger != Trigger.BOTH:
        body += "@" + _trigger_name(p.trigger)
    return body


def _trigger_name(trig: Trigger) -> str:
    if trig in _TRIGGER_CANONICAL:
        return _TRIGGER_CANONICAL[trig]
    return "custom"


def referenced_selectors(
    text: str,
) -> list[tuple[Optional[str], str, str]]:
    """Return every ``(namespace, concept, variant)`` referenced in ``text``.

    Walks the AST before pole resolution so namespace prefixes survive —
    useful at install time, when the CLI needs to know which pack to fetch
    for each atom.  Projection terms contribute two entries (base + onto).
    """
    if not text or not text.strip():
        return []
    toks = _lex(text)
    terms = _Parser(toks).parse()
    out: list[tuple[Optional[str], str, str]] = []
    for term in terms:
        sel = term.selector
        out.append((sel.base.namespace, sel.base.concept, sel.base.variant))
        if sel.onto is not None:
            out.append((sel.onto.namespace, sel.onto.concept, sel.onto.variant))
    return out


__all__ = [
    "DEFAULT_COEFF",
    "ProjectedTerm",
    "SteeringExprError",
    "parse_expr",
    "format_expr",
    "referenced_selectors",
]
