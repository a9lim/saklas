"""Per-call steering configuration for SaklasSession.generate.

A frozen dataclass wrapping ``alphas: {name: alpha}`` plus an optional
``thinking`` override.  Bare dicts are accepted at the API boundary via
``Steering.from_value`` — callers typically pass ``steering={"angry.calm":
0.5}`` without knowing the class exists.

Pole aliasing is NOT resolved here — that happens inside
``SaklasSession.steering()`` (the canonical resolver site, per the plan).
Callers that pre-resolve a pole via ``cli_selectors.resolve_pole`` can pass
the canonical name directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Steering:
    """Per-call steering configuration.

    alphas: vector name -> alpha.  Pole aliases (bare poles of installed
        bipolar vectors) are resolved when the steering is entered via
        ``SaklasSession.steering()``.
    thinking: per-call thinking override; ``None`` means fall through to the
        caller's ``thinking=`` kwarg / the session default.
    """

    alphas: Mapping[str, float]
    thinking: bool | None = None

    @classmethod
    def from_value(
        cls, value: "Steering | Mapping[str, float] | None",
    ) -> "Steering | None":
        """Coerce a dict / Steering / None into a Steering or None.

        Bare dicts are promoted to ``Steering(alphas=dict(value))``.
        ``None`` passes through (caller interprets as "no steering").
        """
        if value is None:
            return None
        if isinstance(value, Steering):
            return value
        return cls(alphas=dict(value))
