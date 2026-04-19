"""Synchronous pub/sub event bus for SaklasSession.

Event types are frozen dataclasses emitted at specific lifecycle points
(extraction, steering enter/exit, probe scoring, generation start/finish).
Subscribers are called on the emit thread — server routes that need to hop
to an event loop should do so inside their callback via
``loop.call_soon_threadsafe``.

Synchronous by design: keeps the hot-path emit cost to a for-loop over
callbacks, and lets us wire event emission from inside the generation
worker thread without needing asyncio plumbing at this layer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Union


@dataclass(frozen=True)
class VectorExtracted:
    name: str
    profile: Any  # Profile — forward ref to avoid import cycle
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SteeringApplied:
    alphas: dict[str, float]
    # Full entries (alpha + trigger) for subscribers that need to know which
    # trigger each alpha flows under. Keys mirror ``alphas`` one-for-one;
    # values are ``(alpha, Trigger)`` tuples.
    entries: dict[str, tuple[float, Any]]


@dataclass(frozen=True)
class SteeringCleared:
    pass


@dataclass(frozen=True)
class ProbeScored:
    readings: dict[str, float]


@dataclass(frozen=True)
class GenerationStarted:
    input: Any
    stateless: bool


@dataclass(frozen=True)
class GenerationFinished:
    result: Any  # GenerationResult


Event = Union[
    VectorExtracted,
    SteeringApplied,
    SteeringCleared,
    ProbeScored,
    GenerationStarted,
    GenerationFinished,
]


class EventBus:
    """Synchronous pub/sub bus.

    Subscribers are called on the emit thread in registration order.
    A subscriber that raises swallows the exception into a warning — event
    delivery must not be able to break the generation path.
    """

    def __init__(self) -> None:
        self._subs: list[Callable[[Event], None]] = []

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Register ``callback`` and return an unsubscribe function."""
        self._subs.append(callback)

        def _unsub() -> None:
            try:
                self._subs.remove(callback)
            except ValueError:
                pass

        return _unsub

    def emit(self, event: Event) -> None:
        # Iterate over a copy so callbacks that unsubscribe (or register
        # new subscribers) can't corrupt the in-progress loop.
        for cb in list(self._subs):
            try:
                cb(event)
            except Exception:
                warnings.warn(
                    f"event subscriber {type(cb).__name__} raised during emit",
                    RuntimeWarning,
                )
