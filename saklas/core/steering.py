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
Callers that pre-resolve a pole via ``io.selectors.canonicalize_atom`` can
pass the canonical name directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING, Union

from saklas.core.triggers import Trigger

if TYPE_CHECKING:
    from saklas.core.steering_expr import (
        AblationTerm,
        ManifoldTerm,
        ProjectedTerm,
    )

#: Accepted shapes for a single entry in ``Steering.alphas`` — a bare
#: alpha (inherits ``Steering.trigger``), a ``(alpha, Trigger)`` tuple for
#: a per-entry trigger override, a
#: :class:`~saklas.core.steering_expr.ProjectedTerm` for runtime
#: projection (materialized into a derived profile by the session), an
#: :class:`~saklas.core.steering_expr.AblationTerm` for mean-replacement
#: ablation, or a :class:`~saklas.core.steering_expr.ManifoldTerm` for
#: spline-based manifold steering (resolved into a loaded manifold
#: artifact by the session).
AlphaEntry = Union[
    float,
    "tuple[float, Trigger]",
    "ProjectedTerm",
    "AblationTerm",
    "ManifoldTerm",
]


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

    ``~`` / ``|`` projection terms always materialize through the
    closed-form LEACE projector against the session's
    :class:`~saklas.core.mahalanobis.LayerWhitener` — there is no
    Euclidean path and no per-call metric override.
    """

    alphas: Mapping[str, AlphaEntry]
    thinking: bool | None = None
    trigger: Trigger = Trigger.BOTH

    @classmethod
    def from_value(
        cls, value: object, *, profile_names: set[str] | None = None,
    ) -> "Steering | None":
        """Coerce a string / Steering / None into a Steering or None.

        Strings parse through the shared expression grammar in
        :mod:`saklas.core.steering_expr`.  ``None`` passes through (the
        caller interprets as "no steering").  Pre-built :class:`Steering`
        instances pass through unchanged.  Any other input type raises
        ``TypeError``.
        """
        if value is None:
            return None
        if isinstance(value, Steering):
            return value
        if isinstance(value, str):
            from saklas.core.steering_expr import parse_expr
            return parse_expr(value, profile_names=profile_names)
        raise TypeError(
            f"Steering.from_value expects str | Steering | None, "
            f"got {type(value).__name__}"
        )

    def classified(self) -> "SteeringEntries":
        """Split ``alphas`` into its three consumer shapes in one walk.

        The five entry kinds lower to three: plain floats and
        ``(alpha, Trigger)`` tuples and
        :class:`~saklas.core.steering_expr.ProjectedTerm` values all become
        ``{key: (alpha, trigger)}`` additive entries (a projection's
        ``operator``/``base``/``onto`` fields are consumed earlier, when the
        session materializes the derived profile under the synthetic key);
        :class:`~saklas.core.steering_expr.AblationTerm` and
        :class:`~saklas.core.steering_expr.ManifoldTerm` values are kept whole,
        because their extra fields (the ablation target, the manifold's
        ``along``/``onto`` split and position) have no place in the additive
        shape.  Keys stay verbatim and live in disjoint namespaces
        (``<name>`` / ``<base><op><onto>`` / ``!<target>`` /
        ``<manifold>%<position>``), so a caller can merge the three back into
        one dict without collision.
        """
        from saklas.core.steering_expr import (
            AblationTerm,
            ManifoldTerm,
            ProjectedTerm,
        )

        additive: dict[str, tuple[float, Trigger]] = {}
        ablations: dict[str, "AblationTerm"] = {}
        manifolds: dict[str, "ManifoldTerm"] = {}
        default = self.trigger
        for name, val in self.alphas.items():
            if isinstance(val, AblationTerm):
                ablations[name] = val
            elif isinstance(val, ManifoldTerm):
                manifolds[name] = val
            elif isinstance(val, ProjectedTerm):
                additive[name] = (float(val.coeff), val.trigger)
            elif isinstance(val, tuple):
                alpha, trig = val
                additive[name] = (float(alpha), trig)
            else:
                additive[name] = (float(val), default)
        return SteeringEntries(
            additive=additive, ablations=ablations, manifolds=manifolds,
        )

    def normalized_entries(self) -> "dict[str, tuple[float, Trigger]]":
        """Return just the **additive** view: ``{name: (alpha, trigger)}``.

        The additive third of :meth:`classified` — plain, tuple and projection
        entries with an explicit trigger each (bare floats take
        ``self.trigger``).  Ablation and manifold entries are absent; reach for
        :meth:`classified` when you need them, rather than re-walking
        ``alphas`` alongside this view.
        """
        return self.classified().additive

    def __str__(self) -> str:
        from saklas.core.steering_expr import format_expr
        return format_expr(self)


@dataclass(frozen=True)
class SteeringEntries:
    """The three consumer shapes :meth:`Steering.classified` splits into.

    ``additive`` is the ``{key: (alpha, trigger)}`` view every scalar term
    lowers to; ``ablations`` and ``manifolds`` keep their term objects whole
    because their extra fields don't survive that flattening.  Keys are the
    verbatim ``Steering.alphas`` keys and the three groups are key-disjoint.
    """

    additive: dict[str, tuple[float, Trigger]]
    ablations: "dict[str, AblationTerm]"
    manifolds: "dict[str, ManifoldTerm]"
