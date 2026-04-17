"""Per-call steering configuration for SaklasSession.generate.

A frozen dataclass wrapping ``alphas: {name: alpha_or_entry}`` plus an
optional ``thinking`` override and an optional default ``trigger``.  Bare
dicts are accepted at the API boundary via ``Steering.from_value`` —
callers typically pass ``steering={"angry.calm": 0.5}`` without knowing
the class exists.

An individual entry can carry its own :class:`~saklas.core.triggers.Trigger`
by using a ``(alpha, trigger)`` tuple as the dict value; entries given as
bare floats inherit ``Steering.trigger`` (which itself defaults to
``Trigger.BOTH``, the "steer every token" behavior).

Pole aliasing is NOT resolved here — that happens inside
``SaklasSession.steering()`` (the canonical resolver site, per the plan).
Callers that pre-resolve a pole via ``cli_selectors.resolve_pole`` can pass
the canonical name directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

from saklas.core.triggers import Trigger

#: Accepted shapes for a single entry in ``Steering.alphas`` — either a bare
#: alpha (inherits ``Steering.trigger``) or a ``(alpha, Trigger)`` tuple for
#: per-entry override.
AlphaEntry = Union[float, tuple[float, Trigger]]


@dataclass(frozen=True)
class Steering:
    """Per-call steering configuration.

    alphas: vector name -> alpha or ``(alpha, Trigger)``. Pole aliases
        (bare poles of installed bipolar vectors) are resolved when the
        steering is entered via ``SaklasSession.steering()``.
    thinking: per-call thinking override; ``None`` means fall through to the
        caller's ``thinking=`` kwarg / the session default.
    trigger: default trigger for entries that are bare floats. Defaults to
        ``Trigger.BOTH`` — steer every token, matching v1.x behavior.
        Entries given as ``(alpha, Trigger)`` tuples ignore this default.
    """

    alphas: Mapping[str, AlphaEntry]
    thinking: bool | None = None
    trigger: Trigger = Trigger.BOTH

    @classmethod
    def from_value(
        cls, value: "Steering | Mapping[str, AlphaEntry] | None",
    ) -> "Steering | None":
        """Coerce a dict / Steering / None into a Steering or None.

        Bare dicts are promoted to ``Steering(alphas=dict(value))`` — the
        resulting Steering uses ``Trigger.BOTH`` as the default trigger.
        Dict values may be bare floats or ``(alpha, Trigger)`` tuples;
        both pass through verbatim into the returned Steering.
        ``None`` passes through (caller interprets as "no steering").
        """
        if value is None:
            return None
        if isinstance(value, Steering):
            return value
        return cls(alphas=dict(value))

    def normalized_entries(self) -> dict[str, tuple[float, Trigger]]:
        """Return a plain ``{name: (alpha, trigger)}`` dict.

        Every entry carries an explicit trigger: tuple entries keep their
        per-entry trigger, bare-float entries take ``self.trigger``. This
        is the canonical form consumed by the session's steering stack and
        the hook manager — everything downstream of pole resolution works
        in this shape.
        """
        out: dict[str, tuple[float, Trigger]] = {}
        default = self.trigger
        for name, val in self.alphas.items():
            if isinstance(val, tuple):
                alpha, trig = val
                out[name] = (float(alpha), trig)
            else:
                out[name] = (float(val), default)
        return out
