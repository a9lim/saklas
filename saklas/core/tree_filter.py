"""Filter grammar for tree pruning.

The grammar is **adjacent to** the steering ``@when:`` clause grammar
but deliberately distinct — the underlying scalars are different.
``@when:`` (in :mod:`saklas.core.steering_expr`) gates on *per-step*
probe readings during generation; this module gates on *per-node*
aggregates that the monitor stamped on each assistant node when the gen
finalized.  Reusing one grammar would silently change semantics across
contexts.

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

    agg:confident.uncertain > 0.4
    any:formal.casual > 0.7, agg:honest.deceptive > 0
    last:refusing.compliant < 0

Aggregate semantics:

- ``agg:`` reads from :attr:`LoomNode.aggregate_readings` directly.
- ``any:`` and ``last:`` read the node's own per-token rows —
  :attr:`LoomNode.thinking_tokens` then :attr:`LoomNode.tokens`, in decode
  order — pulling each row's flat probe scores out of its ``measurements``
  envelope (falling back to the row's ``probes`` alias).  Rows that carry no
  reading for the clause's probe are skipped, so ``last:`` is the last row
  that measured it.  When no row measured it at all the clause evaluates to
  ``False`` per the documented "missing-probe = False, AND semantics" rule.

Every clause is therefore a pure function of the node — there is no
caller-supplied side table to plumb, which is what makes ``any:`` / ``last:``
work through every surface that exposes the grammar (notably
``GET /saklas/v1/sessions/{id}/tree/filter``).

The grammar is intentionally minimal — no parentheses, no ``OR``, no
negation.  Multi-clause AND covers the practical filter cases for
v2.3; more elaborate predicates compose programmatically through
:meth:`LoomTree.filter`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

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


def _row_scores(row: Any) -> Mapping[str, Any] | None:
    """Flat ``{probe: axis0}`` view of one persisted token row.

    The 5.x ``measurements`` envelope is canonical; the row-level ``probes``
    key is the flat alias the same writers emit beside it, kept as a fallback
    so pre-envelope rows still filter.
    """
    if not isinstance(row, Mapping):
        return None
    measurements = row.get("measurements")
    if isinstance(measurements, Mapping):
        scores = measurements.get("scores")
        if isinstance(scores, Mapping):
            return scores
    probes = row.get("probes")
    return probes if isinstance(probes, Mapping) else None


def node_token_series(node: Any, probes: frozenset[str]) -> dict[str, list[float]]:
    """Per-probe per-token score series for one node, in decode order.

    Reads the node's own token rows — ``thinking_tokens`` first, then
    ``tokens``, which is the order the decode loop wrote them — and collects
    each requested probe's per-row reading.  A row that carries no reading
    for a probe contributes nothing to that probe's series, so a probe
    attached mid-generation yields the readings it actually has rather than
    a padded lie.  ``node`` is typed loosely so this module stays
    import-cycle-free against :mod:`saklas.core.loom`.
    """
    series: dict[str, list[float]] = {}
    if not probes:
        return series
    for attr in ("thinking_tokens", "tokens"):
        for row in getattr(node, attr, None) or ():
            scores = _row_scores(row)
            if not scores:
                continue
            for probe in probes:
                value = scores.get(probe)
                if value is None:
                    continue
                try:
                    series.setdefault(probe, []).append(float(value))
                except (TypeError, ValueError):
                    continue
    return series


@dataclass(frozen=True)
class FilterClause:
    """A parsed filter expression — one or more AND'd :class:`_Clause`s.

    Build via :func:`parse_filter`; evaluate against a single
    :class:`saklas.core.loom.LoomNode` via :meth:`evaluate`.  Stored as a
    frozen tuple of clauses so the IR is hashable and stable across
    evaluations.
    """

    clauses: tuple[_Clause, ...]

    @property
    def token_probes(self) -> frozenset[str]:
        """Probes whose per-token series this expression needs.

        Only ``any:`` / ``last:`` clauses read token rows, so an
        ``agg:``-only expression skips the per-node row walk entirely.
        """
        return frozenset(c.probe for c in self.clauses if c.agg != "agg")

    def evaluate(self, node: Any) -> bool:
        """Return ``True`` iff every clause matches against ``node``.

        ``node`` is a :class:`LoomNode` (typed loosely so this module
        stays import-cycle-free against ``saklas.core.loom``).  ``agg:``
        clauses read :attr:`LoomNode.aggregate_readings`; ``any:`` /
        ``last:`` clauses read the node's own token rows via
        :func:`node_token_series`.

        Missing-probe semantics (documented contract): when the probe
        key is absent from the relevant table, the clause evaluates to
        ``False``.  Under multi-clause AND a single false clause sinks
        the whole filter.  Callers that want "treat missing as pass"
        should preprocess inputs.
        """
        aggregates: Mapping[str, float] = getattr(
            node, "aggregate_readings", {},
        ) or {}
        ptokens = node_token_series(node, self.token_probes)

        for c in self.clauses:
            if c.agg == "agg":
                if c.probe not in aggregates:
                    return False
                if not _apply_op(c.op, float(aggregates[c.probe]), c.threshold):
                    return False
                continue

            # ``any`` / ``last`` read the node's own per-token series.
            seq = ptokens.get(c.probe)
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
# inside an identifier segment.  Multi-word probe names use ``_``.  An
# optional ``<ns>/`` prefix matches the qualified keys the fitted multi-node
# defaults register under (``default/personas`` / ``default/emotions``), and
# an optional ``[i]`` axis suffix matches a multi-axis probe's coordinate key
# — both land verbatim in ``LoomNode.aggregate_readings``.
_PROBE_NAME_RE = re.compile(
    r"^(?:[A-Za-z][A-Za-z0-9_-]*/)?"
    r"[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z][A-Za-z0-9_-]*)?(?:\[[0-9]+\])?$"
)

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
            agg = head_stripped
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

    op: CompareOp = op_str  # pyright: ignore[reportAssignmentType]  # regex group() is str, not Literal
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

def filter_tree(tree: Any, text: str) -> set[str]:
    """Apply a filter expression to every node in ``tree``.

    ``tree`` is a :class:`saklas.core.loom.LoomTree`.  Returns the set
    of node ids whose nodes satisfy the parsed filter.  This is the
    free-function form; :meth:`LoomTree.filter_by_expr` calls it.

    Every clause reads the node itself — aggregates from
    ``aggregate_readings``, per-token series from the node's own token rows
    — so all three ops work through every surface that exposes the grammar.
    """
    clause = parse_filter(text)
    return tree.filter(clause.evaluate)


__all__ = [
    "FilterClause",
    "FilterParseError",
    "filter_tree",
    "node_token_series",
    "parse_filter",
]
