"""Unified steering-expression grammar and IR compiler.

One parser + formatter for the steering expression language shared across
every saklas input surface (Python, YAML, HTTP, TUI, CLI). Every surface
turns a user-supplied string into the same :class:`Steering` IR.

Grammar::

    expr        := term (("+" | "-") term)*
    term        := [coeff ["*"]] ["!"] selector ["@" trigger]
    selector    := atom (("~" | "|") atom | "%" position)?
    position    := signed_num ("," signed_num)* | label   # coord list | node label
    label       := NAME                                # a manifold node label
    atom        := [ns "/"] NAME ["." NAME] [":" variant]
    trigger     := preset | gate
    preset      := "before" | "after" | "both" | "thinking" | "response"
                 | "prompt" | "generated"
    gate        := "when" ":" probe_atom op NUM
    probe_atom  := NAME ["." NAME]                     # vector probe (e.g. "angry.calm")
                 | NAME ":" "fraction"                 # manifold subspace fraction
                 | NAME "@" NAME                       # manifold label similarity
    op          := ">" | ">=" | "<" | "<="
    coeff       := signed_float ("," signed_float){0,2}  # comma-run of ≤3;
                                                       # >1 value is valid
                                                       # ONLY on a "%" term.
                                                       # 1→along (onto/toward
                                                       # off), 2→along,onto,
                                                       # 3→along,onto,toward.
                                                       # Optional; defaults
                                                       # to DEFAULT_COEFF = 0.5
    variant     := "raw" | "pca" | "sae" | "sae-" ID
                 | "role" | "role-" ID | "from" | "from-" ID

``!`` mean-ablates the selector (``h' = h − α(h·d̂ − μ·d̂)d̂``; bare
``!x`` is α=1.0); it does not compose with ``~`` / ``|`` / ``%``.  The
``%`` operator places a generation on a fitted manifold — the
``position`` is either a comma-separated list of authoring coordinates
(one per intrinsic dimension) or a single node-label string (sugar for
that node's coords); the parser only collects the payload, arity /
label-existence is validated at manifold-load time.

Probe gates (v2.1): ``@when:<probe><op><threshold>`` fires the term only
on decode steps where the named probe's last reading satisfies the
comparison.  Implicit ``prompt=False`` (no probe reading during prefill).
Compose with other windows via the programmatic surface — the v1
grammar accepts a single ``@`` clause per term.

Manifold-probe gates extend the same shape over the two scalar channels
:class:`ManifoldMonitor` exposes: ``@when:<manifold>:fraction <op> N``
fires on the subspace-fraction reading (the share of the centered
activation living in the manifold's PCA subspace), and
``@when:<manifold>@<label> <op> N`` fires on the negated distance to a
named node (larger = closer; label-similarity gates routinely use
negative thresholds).  The gate's probe string is stored verbatim
(``"circumplex:fraction"``, ``"circumplex@elated"``) so it matches the
key ``ManifoldMonitor.flat_scalars`` already merges into
``TriggerContext.probe_scores``; no runtime gate machinery changes.

Concept names are ASCII identifiers: letter followed by any of
``[a-z0-9_-]``.  Multi-word concepts use underscores
(``artificial_intelligence``) — spaces separate tokens, so
``artificial intelligence`` errors with an underscore hint.  Quoted
identifiers are rejected.  Bipolar pairs join with ``.``
(``human.artificial_intelligence``).

Pole aliases (``wolf`` on top of an installed ``deer.wolf``) resolve via
:func:`saklas.io.selectors.resolve_pole`; the sign flip folds into the
user-supplied coefficient before the term lands in
``Steering.alphas``.  Projection terms produce :class:`ProjectedTerm`
values; the session materializes them into derived profiles on scope
entry.

Manifold steering: the ``%`` infix operator places a generation at a
point of a fitted manifold — ``manifold % coord_list``, e.g.
``0.7 circumplex%0.3,0.8@response``.  The left operand is a manifold name
(not a concept; no pole resolution), the right is a comma-separated list
of authoring coordinates — one per intrinsic dimension of the manifold's
domain.  The parser only collects the coordinate tuple; arity and range
are validated at manifold-load time, when the domain is known.  A ``%``
selector does not compose with the ``~`` / ``|`` projection operators or
with ``!`` ablation.  Manifold terms produce :class:`ManifoldTerm` values;
the session loads the manifold artifact and the hook does a soft
subspace-replace.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING, cast

from saklas.core.errors import (
    ManifoldArityError,  # noqa: F401  — re-exported for back-compat
    OverlappingManifoldError,  # noqa: F401  — re-exported for back-compat
    SaklasError,  # noqa: F401  — re-exported for back-compat
    SteeringExprError,
)
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

# Shared message for the two structurally-parallel "manifold doesn't
# compose" parse guards (the ``%``-with-projection / second-``%`` guard
# and the ``!``-with-``%`` guard).  Consolidated so both surfaces report
# the same constraint in the same words.
_MANIFOLD_COMPOSE_MSG = (
    "a manifold term does not compose with projection ('~'/'|') or "
    "ablation ('!')"
)


# Default coefficient for ablation terms.  Bare ``!x`` means "fully
# replace the component along d̂ with the neutral-baseline mean" — the
# intuitive "ablation" semantics from the literature.  Partial ablation
# (``0.5 !x``) blends the original component toward the mean.  This
# intentionally differs from ``DEFAULT_COEFF`` (0.5), which applies to
# additive terms.
DEFAULT_ABLATION_COEFF = 1.0


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


@dataclass(frozen=True)
class AblationTerm:
    """Mean-replacement ablation entry in ``Steering.alphas``.

    The session resolves ``target`` through the auto-load fast path at
    ``_SteeringContext`` entry, then hands ``(target_profile, coeff,
    trigger, layer_means)`` to ``SteeringManager.add_ablation``.  Stored
    as a value inside ``Steering.alphas``; the key is ``"!<target>"`` so
    ablation entries never collide with plain-term entries on the same
    concept.
    """
    coeff: float
    trigger: Trigger
    target: str


@dataclass(frozen=True)
class ManifoldTerm:
    """Three-op manifold-steering entry in ``Steering.alphas``.

    Produced by a ``manifold % position`` term.  The session loads the
    named :class:`~saklas.core.manifold.Manifold` artifact on scope entry
    and hands the hook a per-layer subspace + domain + the authoring
    ``position`` coords; the hook runs the unified along/onto/toward
    injection (:func:`~saklas.core.manifold.inject_three_op`).  Stored as a
    value inside ``Steering.alphas`` under the key ``"<manifold>%<position>"``
    so two positions on the same manifold compose as distinct entries and
    never collide with plain / projected / ablation keys.

    The three coefficients (each clamped to ``[0, 1]`` at apply time):

    - ``along`` — slide the projected foot toward ``position`` geodesically
      in coordinate space (the principled "directional" op).
    - ``onto`` — collapse the off-manifold, in-subspace residual onto the
      surface (vacuous when the surface fills its subspace).
    - ``toward`` — collapse the off-subspace residual onto the subspace (the
      R-dim generalization of the ``~`` / ``|`` projection operators).

    The grammar's coefficient slot expands to a comma-run of ≤ 3: one coeff
    ⇒ ``along = onto = toward``; two ⇒ ``along, onto = toward``; three ⇒
    explicit.  ``manifold`` is the registry key — namespace-qualified when
    the user typed a namespace, variant-suffixed when not ``raw``.

    ``position`` is either:

    - A tuple of authoring coordinates — one float per intrinsic
      dimension of the manifold's domain (the historical shape, e.g.
      ``(0.3, 0.8)`` on a 2-D affect manifold).
    - A node label string — sugar for "the coords of the node labeled
      <s> on this manifold", resolved at scope entry via
      :meth:`Manifold.resolve_position`.  The label form makes
      ``persona%pirate`` a first-class steering term parallel to
      ``persona%0.3,0.8``.

    Round-trip via :func:`format_expr` preserves the authored form.
    """
    along: float
    onto: float
    toward: float
    trigger: Trigger
    manifold: str
    position: tuple[float, ...] | str

    @property
    def coeff(self) -> float:
        """Representative scalar — the ``along`` (directional) coefficient.

        Read by the uniform telemetry / snapshot / role-aggregation sites
        that want one "how strongly is this term steering" number; ``along``
        is the directional strength that picks the node, so it is the right
        representative.  Not a stored field — there is no dual shape.
        """
        return self.along


# ``SteeringExprError`` now lives in :mod:`saklas.core.errors` (alongside its
# new manifold-specific subclasses ``ManifoldArityError`` /
# ``OverlappingManifoldError``); it is imported above and re-exported here so
# ``from saklas.core.steering_expr import SteeringExprError`` keeps working.


# ---------------------------------------------------------------- lexer ---

_NUM_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_IDENT_START_RE = re.compile(r"[A-Za-z]")
_IDENT_CHAR_RE = re.compile(r"[A-Za-z0-9_]")

_SINGLE_CHAR_TOKENS = {
    ".": "DOT", "/": "SLASH", ":": "COLON", "*": "STAR",
    "+": "PLUS", "-": "MINUS", "@": "AT", "~": "TILDE",
    "|": "ORTHO", "!": "BANG", "%": "PERCENT", ",": "COMMA",
}

# Comparison-op tokens.  Two-char ops (``>=``, ``<=``) take precedence
# over the single-char ones; the lexer peeks one char ahead before
# emitting GT / LT.  ``=`` alone is not a token — equality (``==``) is
# not part of the gate grammar in v2.1 (probe scores are continuous
# floats; equality is meaningless).
_COMPARE_OPS: dict[str, str] = {
    ">=": "GTE", "<=": "LTE",
    ">": "GT", "<": "LT",
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
        # Comparison ops: two-char first, then single-char.
        if c in (">", "<"):
            two = text[i:i + 2]
            if two in _COMPARE_OPS:
                toks.append(_Tok(_COMPARE_OPS[two], two, i))
                i += 2
                continue
            toks.append(_Tok(_COMPARE_OPS[c], c, i))
            i += 1
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
    variant: str  # 'raw' (default) or an io.selectors tensor variant
    col: int


@dataclass
class _Selector:
    base: _Atom
    operator: Optional[str]  # None | '~' | '|'
    onto: Optional[_Atom]
    # Set when the selector is a manifold position term
    # (``base % NUM (, NUM)*``); mutually exclusive with
    # ``operator`` / ``onto``.
    # ``tuple[float, ...]`` for coord-form (``%0.3,0.8``); ``str`` for
    # label-form (``%pirate``).  ``None`` when this selector is not a
    # manifold term.
    manifold_position: Optional[tuple[float, ...] | str] = None


@dataclass
class _Term:
    coeff: float
    selector: _Selector
    # Pre-resolved Trigger object (or ``None`` to fall through to the
    # session/Steering default).  Preset triggers map through
    # :data:`_TRIGGER_PRESETS`; probe gates (``@when:angry>0.4``)
    # produce a ``Trigger(prompt=False, gate=ProbeGate(...))`` directly
    # at parse time.  Rendered back to canonical form by ``_fmt_*``.
    trigger: Optional[Trigger]
    # True iff the user typed a numeric coefficient; False when the parser
    # substituted ``DEFAULT_COEFF``.  Internal — lets a future resolver step
    # swap in per-vector defaults without re-parsing the expression.
    explicit_coeff: bool
    ablation: bool = False  # True iff term was prefixed with `!`
    # The full comma-run of coefficients the user typed.  A plain term
    # carries a 1-tuple; a manifold ``%`` term may carry up to 3 mapping
    # to (along, onto, toward) via :func:`_expand_three_op_coeffs`.
    # ``coeff`` always equals ``coeffs[0]`` (the representative scalar all
    # non-manifold code paths read), so existing single-coeff behavior is
    # bit-identical; ``> 1`` on a non-manifold term is a parse-fold error.
    coeffs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.coeffs:
            self.coeffs = (self.coeff,)


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
        coeffs: tuple[float, ...] = (coeff,)
        if self._peek().kind == "NUM":
            coeff = float(sign) * float(self._consume().value)
            explicit = True
            # Comma-run of coefficients: the TERM-level coeff slot may carry
            # up to 3 values mapping to (along, onto, toward) on a manifold
            # ``%`` term.  This run is lexically unambiguous from the
            # post-``%`` position commas — those follow the manifold IDENT
            # and ``%``, this precedes the selector.  Each subsequent value
            # carries the term's leading sign (so ``-0.6,0.3`` is
            # ``(-0.6, -0.3)``, the natural read of a signed term).  A 4th
            # value is a parse error; whether a comma-run is even legal for
            # the selector kind (manifold-only) is enforced in ``_fold``.
            run = [coeff]
            while self._peek().kind == "COMMA":
                self._consume()
                run.append(float(sign) * self._signed_num())
                if len(run) > 3:
                    raise SteeringExprError(
                        "a manifold coefficient slot takes at most 3 "
                        "comma-separated values (along, onto, toward); "
                        "got more",
                        col=self._peek(-1).col,
                    )
            coeffs = tuple(run)
            if self._peek().kind == "STAR":
                self._consume()
        ablation = False
        if self._peek().kind == "BANG":
            self._consume()
            ablation = True
            if not explicit:
                # Bare `!x` defaults to fully-replace (coeff=1.0).
                coeff = float(sign) * DEFAULT_ABLATION_COEFF
                coeffs = (coeff,)
        selector = self._selector()
        trigger: Trigger | None = None
        if self._peek().kind == "AT":
            self._consume()
            tok = self._expect("IDENT")
            kw = str(tok.value)
            if kw == "when":
                # Probe-gated trigger: ``when:<probe><op><threshold>``.
                # The trailing COLON disambiguates from preset names that
                # happen to start with "when" (none ship today, but the
                # check keeps future namespace freedom).
                if self._peek().kind != "COLON":
                    raise SteeringExprError(
                        f"trigger '@when' requires ':' followed by a "
                        f"probe gate (e.g. '@when:angry.calm>0.4'); "
                        f"got {self._peek().kind}",
                        col=self._peek().col,
                    )
                self._consume()  # COLON
                trigger = self._parse_when_gate(tok.col)
            elif kw in _TRIGGER_PRESETS:
                trigger = _TRIGGER_PRESETS[kw]
            else:
                valid = ", ".join(sorted(_TRIGGER_PRESETS.keys()))
                raise SteeringExprError(
                    f"unknown trigger '@{kw}'; valid: {valid}, or "
                    f"'when:<probe><op><threshold>'.  Note: '@' is for "
                    f"triggers only; HF revisions are not accepted "
                    f"inside steering expressions.",
                    col=tok.col,
                )
        return _Term(
            coeff=coeff, selector=selector, trigger=trigger,
            explicit_coeff=explicit, ablation=ablation, coeffs=coeffs,
        )

    def _parse_when_gate(self, when_col: int) -> Trigger:
        """Parse the ``when:`` payload into a probe-gated :class:`Trigger`.

        Already consumed: ``@`` ``when`` ``:``.  Expects a probe
        identifier, one of ``>``/``>=``/``<``/``<=``, then a numeric
        threshold.

        Three probe identifier shapes are accepted:

        - Vector probe: ``IDENT`` optionally followed by ``.IDENT`` for
          a bipolar concept (e.g. ``angry.calm``).  No namespace prefix
          in v2.1 (the monitor is keyed by canonical concept name
          regardless of which pack the probe came from), and no SAE
          variant suffix — gates fire on the live monitor reading,
          which is already a single number per probe.
        - Manifold subspace-fraction probe:
          ``<manifold>:fraction`` — fires on the fraction of the
          centered activation that lives in the manifold's PCA
          subspace.  Stored verbatim as the gate's probe string
          (e.g. ``"circumplex:fraction"``); the session's
          :class:`ManifoldMonitor.flat_scalars` already emits a
          matching namespaced key, so ``Trigger.active`` looks it up
          unchanged.
        - Manifold label-similarity probe:
          ``<manifold>@<label>`` — fires on the negated distance to
          the named node (larger = closer).  Stored verbatim as the
          gate's probe string (e.g. ``"circumplex@elated"``); same
          ``flat_scalars`` correspondence.

        The discriminator on the trailing IDENT: a ``COLON`` after the
        manifold name routes to the fraction form, an ``AT`` to the
        label form, a ``DOT`` to the bipolar vector form, anything
        else is a plain vector probe.  Within the ``when:`` body the
        ``AT`` token cannot be a fresh trigger marker (one ``@`` clause
        per term, already consumed), so the disambiguation is
        unambiguous.
        """
        probe_tok = self._expect("IDENT")
        probe = str(probe_tok.value)
        nxt = self._peek().kind
        if nxt == "COLON":
            # Manifold subspace-fraction gate: ``<manifold>:fraction``.
            # The suffix after the colon must be the literal slug
            # ``fraction`` — the only fraction-channel scalar
            # ``ManifoldMonitor.flat_scalars`` emits.  Any other slug
            # would silently never match a probe score, so we surface
            # the typo here rather than letting the gate report
            # inactive forever.
            self._consume()
            suffix_tok = self._expect("IDENT")
            suffix = str(suffix_tok.value)
            if suffix != "fraction":
                raise SteeringExprError(
                    f"unknown manifold gate channel "
                    f"'{probe}:{suffix}'; expected "
                    f"'<manifold>:fraction' or "
                    f"'<manifold>@<label>'",
                    col=suffix_tok.col,
                )
            probe = f"{probe}:{suffix}"
        elif nxt == "AT":
            # Manifold label-similarity gate: ``<manifold>@<label>``.
            # Label existence is not validated at parse time — the
            # manifold artifact may not be loaded yet, and a stale
            # gate against a removed label should report inactive (no
            # matching key in ``flat_scalars``), not raise.
            self._consume()
            label_tok = self._expect("IDENT")
            probe = f"{probe}@{label_tok.value}"
        elif nxt == "DOT":
            self._consume()
            rhs = self._expect("IDENT")
            probe = f"{probe}.{rhs.value}"
        op_tok = self._consume()
        op_kind = op_tok.kind
        if op_kind not in ("GT", "GTE", "LT", "LTE"):
            raise SteeringExprError(
                f"expected comparison op (>, >=, <, <=) in '@when:' "
                f"clause, got {op_kind} ({op_tok.value!r})",
                col=op_tok.col,
            )
        op_str = {"GT": ">", "GTE": ">=", "LT": "<", "LTE": "<="}[op_kind]
        # Allow a leading ``-`` on the threshold so ``@when:x>-0.5``
        # parses as ``threshold = -0.5``.
        threshold = self._signed_num()
        return Trigger.when(probe, op_str, threshold)  # pyright: ignore[reportArgumentType]  # dict[str,str] widens the literal op value; runtime is always a valid ProbeGateOp

    def _signed_num(self) -> float:
        """Parse an optionally signed numeric literal: ``[+|-] NUM``."""
        sign = +1.0
        if self._peek().kind == "MINUS":
            self._consume()
            sign = -1.0
        elif self._peek().kind == "PLUS":
            self._consume()
        num_tok = self._expect("NUM")
        return sign * float(num_tok.value)

    def _selector(self) -> _Selector:
        base = self._atom()
        if self._peek().kind == "PERCENT":
            # Manifold position term: ``base % NUM(,NUM)*`` (coord form)
            # OR ``base % IDENT`` (label form, sugar for "the coords of
            # the node labeled <s>"; resolved at scope entry).  The
            # parser only collects the position payload — arity (against
            # the domain's intrinsic dimension) for the coord form, and
            # label existence + role-paired lookup for the label form,
            # are checked at manifold-load time when both the domain
            # and the node labels are known.
            self._consume()
            head = self._peek()
            position: tuple[float, ...] | str
            if head.kind == "IDENT":
                # Label form: a single bare identifier.  No comma list
                # — a manifold node label is one slug, not a tuple.
                # Variant suffixes (``%pirate:role-pirate``) are not
                # meaningful on a position and so not part of the
                # grammar; if the user typed one, it lexed as a
                # separate ``:`` token and will fall out as a parse
                # error on the trailing arm below.
                tok = self._consume()
                position = str(tok.value)
            else:
                coords: list[float] = [self._signed_num()]
                while self._peek().kind == "COMMA":
                    self._consume()
                    coords.append(self._signed_num())
                position = tuple(coords)
            if self._peek().kind in ("TILDE", "ORTHO", "PERCENT"):
                raise SteeringExprError(
                    _MANIFOLD_COMPOSE_MSG,
                    col=self._peek().col,
                )
            return _Selector(
                base=base, operator=None, onto=None,
                manifold_position=position,
            )
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
    ``resolve_pole``, prefixed with ``<namespace>/`` when the user
    explicitly typed a namespace (so two installed packs sharing a
    concept name — ``alice/foo`` vs ``bob/foo`` — stay distinct
    through the registry-key path), and suffixed with ``:<variant>``
    when the variant is anything other than ``raw``.  ``sign_flip``
    is +1 or -1 per ``resolve_pole``; callers multiply their
    user-supplied coefficient by this flip.

    Bare references (no user-typed namespace) keep the canonical
    name as the key — matches v2.0 behavior and lets cross-namespace
    bare-pole collisions surface at parse time via
    :class:`AmbiguousSelectorError` rather than silently picking
    one.
    """
    from saklas.io.selectors import resolve_pole

    raw = atom.concept
    if atom.variant != "raw":
        raw = f"{raw}:{atom.variant}"
    ns = atom.namespace if atom.namespace is not None else default_namespace
    canonical, sign, _match, variant = resolve_pole(raw, namespace=ns)
    if atom.namespace is not None:
        canonical = f"{atom.namespace}/{canonical}"
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


def _merge_ablation(
    alphas: "dict[str, AlphaEntry]",
    target_key: str,
    coeff: float,
    trig: Trigger,
) -> None:
    key = f"!{target_key}"
    if key not in alphas:
        alphas[key] = AblationTerm(coeff=coeff, trigger=trig, target=target_key)
        return
    existing = alphas[key]
    if not isinstance(existing, AblationTerm):
        raise SteeringExprError(  # pragma: no cover — key namespace is disjoint
            f"ablation '{key}' conflicts with a non-ablation entry of the same name"
        )
    if existing.trigger != trig:
        raise SteeringExprError(
            f"ablation '!{target_key}' appears with conflicting triggers; "
            f"merge triggers explicitly or split into separate Steering entries"
        )
    alphas[key] = AblationTerm(
        coeff=existing.coeff + coeff, trigger=trig, target=target_key,
    )


def _resolve_manifold_atom(atom: _Atom) -> str:
    """Return the registry key for a manifold atom.

    Unlike :func:`_resolve_atom`, manifold atoms do *not* go through
    ``resolve_pole`` — a manifold name is not a concept and must not
    alias to one.  The key is the bare name, namespace-prefixed only when
    the user typed a namespace, variant-suffixed when not ``raw``.
    """
    key = atom.concept
    if atom.namespace is not None:
        key = f"{atom.namespace}/{key}"
    return _with_variant(key, atom.variant)


def _fmt_position(position: tuple[float, ...] | str) -> str:
    """Render a manifold position payload back to grammar form.

    Coord tuple → comma-joined ``%g`` list (``0.3,0.8``).  Label string
    → the slug verbatim (``pirate``).  The two forms are unambiguous
    by shape — a node label is a slug-shaped identifier, a coord list
    starts with a digit / sign.
    """
    if isinstance(position, str):
        return position
    return ",".join(f"{c:g}" for c in position)


def _expand_three_op_coeffs(
    coeffs: tuple[float, ...],
) -> tuple[float, float, float]:
    """Map a 1/2/3-length coefficient run to ``(along, onto, toward)``.

    The manifold ``%`` coefficient slot expands a comma-run with the
    collapse ops defaulting **off** — they are progressively opted into:

    - ``c`` → ``along = c, onto = 0, toward = 0``  (pure directional slide)
    - ``a, o`` → ``along = a, onto = o, toward = 0``
    - ``a, o, t`` → explicit

    ``along`` is *the* steering knob: it slides the projected foot toward
    the target on the manifold and keeps everything else.  ``onto`` (flatten
    the off-manifold in-subspace residual) and ``toward`` (collapse the
    off-*subspace* residual) are aggressive and so off by default — crucially
    ``toward`` scales ``H_o``, which is the bulk of the activation (the lever
    ``‖h_par_c‖/‖h‖`` is ~0.1, so ``H_o`` is ~90% of it), so a nonzero
    ``toward`` guts the activation and is reachable only via the explicit
    3-coeff form.  This is why the single-coeff form maps to along-only, not
    to ``(c, c, c)``.

    A run of length 0 or > 3 is a programming error here — the parser
    rejects the 4-coeff case at parse time and a term always carries at
    least one coeff — but raise rather than index blindly.
    """
    n = len(coeffs)
    if n == 1:
        return (coeffs[0], 0.0, 0.0)
    if n == 2:
        a, o = coeffs
        return (a, o, 0.0)
    if n == 3:
        return (coeffs[0], coeffs[1], coeffs[2])
    raise SteeringExprError(
        "a manifold coefficient slot takes 1, 2, or 3 comma-separated "
        f"values (along, onto, toward); got {n}"
    )


def _merge_manifold(
    alphas: "dict[str, AlphaEntry]",
    manifold: str,
    coeffs: tuple[float, float, float],
    position: tuple[float, ...] | str,
    trig: Trigger,
) -> None:
    along, onto, toward = coeffs
    key = f"{manifold}%{_fmt_position(position)}"
    if key not in alphas:
        alphas[key] = ManifoldTerm(
            along=along, onto=onto, toward=toward,
            trigger=trig, manifold=manifold, position=position,
        )
        return
    existing = alphas[key]
    if not isinstance(existing, ManifoldTerm):
        raise SteeringExprError(  # pragma: no cover — key namespace is disjoint
            f"manifold '{key}' conflicts with a non-manifold entry"
        )
    if existing.trigger != trig:
        raise SteeringExprError(
            f"manifold '{key}' appears with conflicting "
            f"triggers; merge triggers explicitly or split into separate "
            f"Steering entries"
        )
    alphas[key] = ManifoldTerm(
        along=existing.along + along,
        onto=existing.onto + onto,
        toward=existing.toward + toward,
        trigger=trig, manifold=manifold, position=position,
    )


def _fold(terms: list[_Term], *, namespace: Optional[str]) -> "Steering":
    from saklas.core.steering import Steering

    alphas: "dict[str, AlphaEntry]" = {}
    for term in terms:
        sel = term.selector
        # Manifold position terms resolve before ``_resolve_atom`` — a
        # manifold name must not run through concept pole-resolution.
        if sel.manifold_position is not None:
            if term.ablation:
                raise SteeringExprError(_MANIFOLD_COMPOSE_MSG)
            mfld_key = _resolve_manifold_atom(sel.base)
            mfld_trig = (
                term.trigger if term.trigger is not None else Trigger.BOTH
            )
            _merge_manifold(
                alphas, mfld_key,
                _expand_three_op_coeffs(term.coeffs),
                sel.manifold_position, mfld_trig,
            )
            continue
        # Bare-name manifold-label fallback (Phase C.2): a plain term
        # whose base is a bare slug (no namespace, no variant suffix,
        # no bipolar ``.``, no projection / ablation) is a candidate
        # for the unified bare-name resolver.  If the slug isn't an
        # installed bipolar pole *and* matches a manifold's node
        # label, synthesize a label-form ManifoldTerm at that node
        # instead of treating the slug as a fresh concept.  Cross-tier
        # collisions (slug matches both a pole and a manifold node)
        # raise ``AmbiguousSelectorError`` inside ``resolve_bare_name``.
        if (
            sel.operator is None
            and not term.ablation
            and sel.base.variant == "raw"
            and sel.base.namespace is None
            and "." not in sel.base.concept
        ):
            from saklas.io.selectors import (
                AmbiguousSelectorError,
                resolve_bare_name,
            )
            try:
                pole_hit, manifold_hit = resolve_bare_name(
                    sel.base.concept, namespace=namespace,
                )
            except AmbiguousSelectorError:
                raise
            except Exception:
                # Errors fall through to the historical resolve_pole
                # path below; if the same error is genuine, it raises
                # there with the canonical surface message.
                pole_hit, manifold_hit = None, None
            if manifold_hit is not None and pole_hit is None:
                mfld_trig = (
                    term.trigger if term.trigger is not None else Trigger.BOTH
                )
                _merge_manifold(
                    alphas, manifold_hit.manifold_key,
                    _expand_three_op_coeffs(term.coeffs),
                    manifold_hit.label, mfld_trig,
                )
                continue
        # Past this point the term is unambiguously a vector (plain /
        # projection / ablation) — the comma-separated coefficient run is a
        # manifold-``%``-only construct, so reject it here rather than
        # silently dropping the extra values.
        if len(term.coeffs) > 1:
            raise SteeringExprError(
                "comma-separated coefficients are only valid for "
                "`manifold % position` terms"
            )
        base_key, base_sign = _resolve_atom(sel.base, namespace)
        coeff = term.coeff * base_sign
        # ``_Term.trigger`` already carries a resolved Trigger object
        # (built by :class:`_Parser._term`) — preset names map through
        # _TRIGGER_PRESETS at parse time, gates produce a fresh
        # Trigger directly.  Pass through verbatim.
        trig: Trigger | None = term.trigger
        if term.ablation:
            if sel.operator is not None:
                raise SteeringExprError(
                    "ablation does not compose with projection operators; "
                    "ablate the base concept and project separately"
                )
            effective_trig = trig if trig is not None else Trigger.BOTH
            _merge_ablation(alphas, base_key, coeff, effective_trig)
            continue
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
    ``None``, :func:`saklas.io.selectors.resolve_pole` raises
    :class:`~saklas.io.selectors.AmbiguousSelectorError` if a bare pole
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
        if isinstance(val, AblationTerm):
            parts.append(_fmt_ablation(val))
            continue
        if isinstance(val, ProjectedTerm):
            parts.append(_fmt_projected(val))
            continue
        if isinstance(val, ManifoldTerm):
            parts.append(_fmt_manifold(val))
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


def _fmt_ablation(a: AblationTerm) -> str:
    if a.coeff == 1.0:
        body = f"!{a.target}"
    elif a.coeff == -1.0:
        body = f"-!{a.target}"
    else:
        body = f"{a.coeff:g} !{a.target}"
    if a.trigger != Trigger.BOTH:
        body += "@" + _trigger_name(a.trigger)
    return body


def _fmt_manifold(m: ManifoldTerm) -> str:
    # Render the shortest coefficient form the (along, onto, toward) triple
    # collapses to — the inverse of ``_expand_three_op_coeffs`` — so the
    # round-trip is byte-for-byte.  The collapse ops default off, so: both
    # zero → one coeff (along); toward zero → two (along, onto); else three.
    pos = _fmt_position(m.position)
    if m.onto == 0.0 and m.toward == 0.0:
        coeff_str = f"{m.along:g}"
    elif m.toward == 0.0:
        coeff_str = f"{m.along:g},{m.onto:g}"
    else:
        coeff_str = f"{m.along:g},{m.onto:g},{m.toward:g}"
    body = f"{coeff_str} {m.manifold}%{pos}"
    if m.trigger != Trigger.BOTH:
        body += "@" + _trigger_name(m.trigger)
    return body


def _trigger_name(trig: Trigger) -> str:
    if trig in _TRIGGER_CANONICAL:
        return _TRIGGER_CANONICAL[trig]
    # Probe gate (v2.1) — round-trip via the ``when:`` syntax.  Gate-
    # only triggers (the canonical shape produced by ``Trigger.when``)
    # render as ``when:<probe><op><threshold>``; any other custom
    # trigger (e.g. user-built with ``first_n=`` plus a gate) falls
    # through to the ``"custom"`` sentinel — programmatic-only,
    # callers don't expect it to round-trip through the grammar.
    if (
        trig.gate is not None
        and trig.prompt is False
        and trig.generated is True
        and trig.thinking is True
        and trig.response is True
        and trig.first_n is None
        and trig.after_n is None
    ):
        g = trig.gate
        return f"when:{g.probe}{g.op}{_fmt_number(g.threshold)}"
    return "custom"


def _fmt_number(x: float) -> str:
    """Render a probe threshold compactly while staying parseable.

    The grammar accepts a leading ``-`` before NUM, so negative
    thresholds render with the minus sign in-line.  Uses ``%g`` so
    integer-valued thresholds drop the trailing ``.0`` and small
    fractions render at full precision without scientific notation.
    """
    if x < 0:
        return f"-{x * -1.0:g}"
    return f"{x:g}"


def referenced_selectors(
    text: str,
) -> list[tuple[Optional[str], str, str]]:
    """Return every ``(namespace, concept, variant)`` referenced in ``text``.

    Walks the AST before pole resolution so namespace prefixes survive —
    useful at install time, when the CLI needs to know which pack to fetch
    for each atom.  Projection terms contribute two entries (base + onto).
    Manifold terms are skipped — a manifold name is not a concept and
    resolves through :func:`referenced_manifolds` instead.
    """
    if not text or not text.strip():
        return []
    toks = _lex(text)
    terms = _Parser(toks).parse()
    out: list[tuple[Optional[str], str, str]] = []
    for term in terms:
        sel = term.selector
        if sel.manifold_position is not None:
            continue
        out.append((sel.base.namespace, sel.base.concept, sel.base.variant))
        if sel.onto is not None:
            out.append((sel.onto.namespace, sel.onto.concept, sel.onto.variant))
    return out


def referenced_manifolds(
    text: str,
) -> list[tuple[Optional[str], str, str]]:
    """Return every ``(namespace, name, variant)`` manifold referenced.

    The manifold-term analogue of :func:`referenced_selectors` — kept a
    separate function so ``referenced_selectors`` keeps its exact 3-tuple
    shape and no concept-install caller has to learn about manifolds.
    Walks the AST before resolution so namespace prefixes survive.
    """
    if not text or not text.strip():
        return []
    toks = _lex(text)
    terms = _Parser(toks).parse()
    out: list[tuple[Optional[str], str, str]] = []
    for term in terms:
        sel = term.selector
        if sel.manifold_position is not None:
            out.append(
                (sel.base.namespace, sel.base.concept, sel.base.variant)
            )
    return out


__all__ = [
    "DEFAULT_COEFF",
    "DEFAULT_ABLATION_COEFF",
    "AblationTerm",
    "ManifoldTerm",
    "ProjectedTerm",
    "SteeringExprError",
    "parse_expr",
    "format_expr",
    "referenced_selectors",
    "referenced_manifolds",
]
