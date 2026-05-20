"""Filter grammar for tree pruning (v2.3 phase 5).

The grammar is **adjacent to** the steering ``@when:`` clause grammar
but deliberately distinct — the underlying scalars are different.
``@when:`` (in :mod:`saklas.core.steering_expr`) gates on *per-step*
probe readings during generation; this module gates on *per-node*
aggregates that the monitor stamped on each assistant node when the gen
finalized.  Reusing one grammar would silently change semantics across
contexts (decision 18 in ``docs/plans/loom.md``).

Grammar::

    filter_clauses := clause ("," clause)*           # multi-clause is AND
    clause         := agg_op ":" probe op threshold
    agg_op         := "agg" | "any" | "last"
                      #   agg  = aggregate (default; ProbeReadings.mean)
                      #   any  = max over per-token scores
                      #   last = last-token score
    op             := > | >= | < | <=
    probe          := <probe name as in @when:>
    threshold      := <float>

Examples::

    agg:angry.calm > 0.4
    any:hallucinating.grounded > 0.7, agg:honest > 0
    last:refusal.compliant < 0

Aggregate semantics:

- ``agg:`` reads from :attr:`LoomNode.aggregate_readings` directly.
- ``any:`` and ``last:`` need the per-token score table; callers must
  pass ``per_token_scores={node_id: {probe: [score_per_token,...]}}``
  to :meth:`FilterClause.evaluate`.  When the table is missing for a
  node (or the probe is missing for that node), the clause evaluates
  to ``False`` per the documented "missing-probe = False, AND
  semantics" rule.

The grammar is intentionally minimal — no parentheses, no ``OR``, no
negation.  Multi-clause AND covers the practical filter cases for
v2.3; more elaborate predicates compose programmatically through
:meth:`LoomTree.filter`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from saklas.core.errors import SaklasError


AggOp = Literal["agg", "any", "last"]
CompareOp = Literal[">", ">=", "<", "<="]


class FilterParseError(ValueError, SaklasError):
    """Raised when a filter expression cannot be parsed."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


# Per-clause numeric helper — accepts ``>``/``>=``/``<``/``<=``.
def _apply_op(op: CompareOp, lhs: float, rhs: float) -> bool:
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<":
        return lhs < rhs
    return lhs <= rhs  # "<="


@dataclass(frozen=True)
class _Clause:
    """One parsed (agg_op, probe, op, threshold) clause."""
    agg: AggOp
    probe: str
    op: CompareOp
    threshold: float


@dataclass(frozen=True)
class FilterClause:
    """A parsed filter expression — one or more AND'd :class:`_Clause`s.

    Build via :func:`parse_filter`; evaluate against a single
    :class:`saklas.core.loom.LoomNode` via :meth:`evaluate`.  Stored as a
    frozen tuple of clauses so the IR is hashable and stable across
    evaluations.
    """

    clauses: tuple[_Clause, ...]

    def evaluate(
        self,
        node: Any,
        *,
        per_token_scores: Mapping[str, Sequence[float]] | None = None,
    ) -> bool:
        """Return ``True`` iff every clause matches against ``node``.

        ``node`` is a :class:`LoomNode` (typed loosely so this module
        stays import-cycle-free against ``saklas.core.loom``).  We read
        :attr:`LoomNode.aggregate_readings` for ``agg:`` clauses and
        ``per_token_scores`` (a flat ``{probe: [scores]}`` mapping for
        this single node) for ``any:`` / ``last:`` clauses.

        Missing-probe semantics (documented contract): when the probe
        key is absent from the relevant table, the clause evaluates to
        ``False``.  Under multi-clause AND a single false clause sinks
        the whole filter.  Callers that want "treat missing as pass"
        should preprocess inputs.
        """
        aggregates: Mapping[str, float] = getattr(
            node, "aggregate_readings", {},
        ) or {}
        ptokens = per_token_scores or {}

        for c in self.clauses:
            if c.agg == "agg":
                if c.probe not in aggregates:
                    return False
                if not _apply_op(c.op, float(aggregates[c.probe]), c.threshold):
                    return False
                continue

            # ``any`` / ``last`` need the per-token table.
            seq = ptokens.get(c.probe) if ptokens else None
            if not seq:
                return False
            if c.agg == "any":
                # Match if *any* per-token score satisfies the comparison.
                # ``>``/``>=`` use max; ``<``/``<=`` use min — picking the
                # extreme on each side gives the cheapest correct check.
                if c.op in (">", ">="):
                    if not _apply_op(c.op, max(float(x) for x in seq), c.threshold):
                        return False
                else:  # "<", "<="
                    if not _apply_op(c.op, min(float(x) for x in seq), c.threshold):
                        return False
                continue
            if c.agg == "last":
                if not _apply_op(c.op, float(seq[-1]), c.threshold):
                    return False
                continue

            raise FilterParseError(  # pragma: no cover — agg is Literal
                f"unknown agg op {c.agg!r}"
            )
        return True


# --- parser ----------------------------------------------------------------

_AGG_OPS: tuple[AggOp, ...] = ("agg", "any", "last")

# Allow the same probe-name shape the steering grammar accepts: ASCII
# identifier, optional dotted second pole, optional embedded ``_``/``-``
# inside an identifier segment.  Multi-word probe names use ``_``.
_PROBE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z][A-Za-z0-9_-]*)?$")

# Compare op precedence: try two-char before single-char.
_COMPARE_OP_RE = re.compile(r"(>=|<=|>|<)")


def _split_top_level(text: str) -> list[str]:
    """Split on top-level commas.

    The grammar has no nesting, so a plain ``.split(",")`` would suffice
    — but we trim each fragment and drop empties so trailing commas /
    extra whitespace don't blow up.
    """
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


def _parse_one_clause(raw: str) -> _Clause:
    """Parse a single ``<agg>:<probe> <op> <num>`` clause.

    The ``agg:`` prefix is optional — bare ``<probe> <op> <num>``
    defaults to ``agg:`` (the per-node aggregate reading), matching
    the plan docs' "agg is the default" wording.  Explicit ``agg:``
    / ``any:`` / ``last:`` keep working unchanged.
    """
    # When the leading colon is followed by an ``agg:`` /``any:`` /
    # ``last:`` token, route into the prefixed path; otherwise default
    # to ``agg:`` and treat the whole clause as ``<probe> <op> <num>``.
    agg: AggOp
    rest: str
    if ":" in raw:
        head, _, tail = raw.partition(":")
        head_stripped = head.strip()
        if head_stripped in _AGG_OPS:
            agg = head_stripped  # type: ignore[assignment]
            rest = tail
        else:
            # Colon inside a probe name (e.g. ``deer.wolf:sae`` — not a
            # legal probe shape here today but reserves the room) or
            # an unknown prefix.  When the head doesn't match an agg
            # op, treat it as malformed-prefix rather than silently
            # defaulting — bare ``foo:bar`` is almost certainly a typo
            # for an agg op.
            raise FilterParseError(
                f"unknown agg op {head_stripped!r}; expected one of "
                f"{', '.join(_AGG_OPS)} (or drop the prefix to use "
                f"'agg:' by default)"
            )
    else:
        agg = "agg"
        rest = raw

    # Find the comparison op — try two-char first.
    m = _COMPARE_OP_RE.search(rest)
    if not m:
        raise FilterParseError(
            f"clause {raw!r} missing comparison op (>, >=, <, <=)"
        )
    probe = rest[: m.start()].strip()
    op_str = m.group(1)
    threshold_str = rest[m.end():].strip()

    if not probe:
        raise FilterParseError(
            f"clause {raw!r} missing probe name before {op_str!r}"
        )
    if not _PROBE_NAME_RE.match(probe):
        raise FilterParseError(
            f"clause {raw!r}: probe {probe!r} is not a valid identifier "
            f"(letter, then [A-Za-z0-9_-], optional .pole)"
        )

    if not threshold_str:
        raise FilterParseError(
            f"clause {raw!r}: missing threshold after {op_str!r}"
        )
    try:
        threshold = float(threshold_str)
    except ValueError:
        raise FilterParseError(
            f"clause {raw!r}: threshold {threshold_str!r} is not a number"
        ) from None

    op: CompareOp = op_str  # type: ignore[assignment]
    return _Clause(agg=agg, probe=probe, op=op, threshold=threshold)


def parse_filter(text: str) -> FilterClause:
    """Parse a filter expression into a :class:`FilterClause`.

    Raises :class:`FilterParseError` on any parse problem (missing
    prefix, unknown agg op, missing operator, malformed threshold).
    Whitespace is collapsed; trailing commas are tolerated.
    """
    if not text or not text.strip():
        raise FilterParseError("empty filter expression")
    parts = _split_top_level(text)
    if not parts:
        raise FilterParseError(f"filter expression {text!r} yielded no clauses")
    clauses = tuple(_parse_one_clause(p) for p in parts)
    return FilterClause(clauses=clauses)


# --- LoomTree integration --------------------------------------------------

def filter_tree(
    tree: Any,
    text: str,
    *,
    per_token_scores: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> set[str]:
    """Apply a filter expression to every node in ``tree``.

    ``tree`` is a :class:`saklas.core.loom.LoomTree`.  Returns the set
    of node ids whose nodes satisfy the parsed filter.  This is the
    free-function form; :meth:`LoomTree.filter_by_expr` calls it.

    ``per_token_scores`` is an optional ``{node_id: {probe: [scores]}}``
    mapping — needed for ``any:`` / ``last:`` clauses.  Callers can
    pull this from a side-cache keyed by node id; absent entries simply
    fail the clause per documented semantics.
    """
    clause = parse_filter(text)
    ptokens = per_token_scores or {}

    def _pred(node: Any) -> bool:
        nid = getattr(node, "id", None)
        per_node = ptokens.get(nid) if nid is not None else None
        return clause.evaluate(node, per_token_scores=per_node)

    return tree.filter(_pred)


__all__ = [
    "FilterClause",
    "FilterParseError",
    "filter_tree",
    "parse_filter",
]
