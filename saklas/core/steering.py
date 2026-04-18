"""Per-call steering configuration for SaklasSession.generate.

A frozen dataclass wrapping ``alphas: {name: alpha_or_entry}`` plus an
optional ``thinking`` override and an optional default ``trigger``.
Callers hand :func:`Steering.from_value` either an expression string
(routed through the shared grammar in
:mod:`saklas.core.steering_expr`) or a pre-built :class:`Steering`.
Dict inputs are no longer accepted — the expression string is the only
input shape for ad-hoc use; programmatic callers construct the dataclass
directly.

An individual entry can carry its own :class:`~saklas.core.triggers.Trigger`
by using a ``(alpha, trigger)`` tuple as the dict value; entries given as
bare floats inherit ``Steering.trigger`` (which itself defaults to
``Trigger.BOTH``, the "steer every token" behavior).  Projection terms
land as :class:`~saklas.core.steering_expr.ProjectedTerm` values and are
materialized into derived profiles by
:class:`~saklas.core.session._SteeringContext` on scope entry.

Pole aliasing is NOT resolved here — that happens inside
``SaklasSession.steering()`` (the canonical resolver site, per the plan).
Callers that pre-resolve a pole via ``cli_selectors.resolve_pole`` can pass
the canonical name directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING, Union

from saklas.core.triggers import Trigger

if TYPE_CHECKING:
    from saklas.core.steering_expr import ProjectedTerm

#: Accepted shapes for a single entry in ``Steering.alphas`` — a bare
#: alpha (inherits ``Steering.trigger``), a ``(alpha, Trigger)`` tuple for
#: a per-entry trigger override, or a
#: :class:`~saklas.core.steering_expr.ProjectedTerm` for runtime
#: projection (materialized into a derived profile by the session).
AlphaEntry = Union[float, "tuple[float, Trigger]", "ProjectedTerm"]


@dataclass(frozen=True)
class Steering:
    """Per-call steering configuration.

    alphas: vector name -> alpha, ``(alpha, Trigger)``, or
        :class:`~saklas.core.steering_expr.ProjectedTerm`.  Pole aliases
        (bare poles of installed bipolar vectors) are resolved when the
        steering is entered via ``SaklasSession.steering()``.
    thinking: per-call thinking override; ``None`` means fall through to the
        caller's ``thinking=`` kwarg / the session default.
    trigger: default trigger for entries that are bare floats. Defaults to
        ``Trigger.BOTH`` — steer every token.  Entries given as
        ``(alpha, Trigger)`` tuples ignore this default; projection
        entries carry their own trigger inside the ``ProjectedTerm``.
    """

    alphas: Mapping[str, AlphaEntry]
    thinking: bool | None = None
    trigger: Trigger = Trigger.BOTH

    @classmethod
    def from_value(
        cls, value: "str | Steering | None",
    ) -> "Steering | None":
        """Coerce a string / Steering / None into a Steering or None.

        Strings parse through the shared expression grammar in
        :mod:`saklas.core.steering_expr`.  ``None`` passes through (the
        caller interprets as "no steering").  Pre-built :class:`Steering`
        instances pass through unchanged.

        Dict inputs are rejected with a :class:`TypeError` carrying a
        migration hint — use an expression string (``"0.5 honest"``) or
        construct :class:`Steering` directly.
        """
        if value is None:
            return None
        if isinstance(value, Steering):
            return value
        if isinstance(value, str):
            from saklas.core.steering_expr import parse_expr
            return parse_expr(value)
        if isinstance(value, Mapping):
            raise TypeError(
                "Steering.from_value no longer accepts dict inputs; pass an "
                "expression string (e.g. \"0.5 honest + 0.3 warm\") or "
                "construct Steering(alphas=...) directly."
            )
        raise TypeError(
            f"Steering.from_value expects str | Steering | None, "
            f"got {type(value).__name__}"
        )

    def normalized_entries(self) -> "dict[str, tuple[float, Trigger]]":
        """Return a plain ``{name: (alpha, trigger)}`` dict.

        Every entry carries an explicit trigger: tuple entries keep their
        per-entry trigger, bare-float entries take ``self.trigger``, and
        :class:`~saklas.core.steering_expr.ProjectedTerm` values flatten
        to ``(coeff, term.trigger)``.  This is the canonical form consumed
        by the session's steering stack and the hook manager — everything
        downstream of pole resolution works in this shape.  Synthetic
        projection keys (``"<base><op><onto>"``) pass through verbatim;
        the session is responsible for materializing the derived profile
        before the manager sees the key.
        """
        from saklas.core.steering_expr import ProjectedTerm

        out: dict[str, tuple[float, Trigger]] = {}
        default = self.trigger
        for name, val in self.alphas.items():
            if isinstance(val, ProjectedTerm):
                out[name] = (float(val.coeff), val.trigger)
                continue
            if isinstance(val, tuple):
                alpha, trig = val
                out[name] = (float(alpha), trig)
                continue
            out[name] = (float(val), default)
        return out

    def __str__(self) -> str:
        from saklas.core.steering_expr import format_expr
        return format_expr(self)
