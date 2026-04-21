"""EventBus unit tests — subscribe, unsubscribe, exception isolation."""

import warnings

from saklas.core.events import (
    EventBus,
    GenerationFinished,
    GenerationStarted,
    ProbeScored,
    SteeringApplied,
    SteeringCleared,
    VectorExtracted,
)
from saklas.core.triggers import Trigger


def test_subscribe_and_emit():
    bus = EventBus()
    seen = []
    bus.subscribe(seen.append)
    bus.emit(SteeringCleared())
    assert len(seen) == 1
    assert isinstance(seen[0], SteeringCleared)


def test_unsubscribe():
    bus = EventBus()
    seen = []
    unsub = bus.subscribe(seen.append)
    bus.emit(SteeringCleared())
    unsub()
    bus.emit(SteeringCleared())
    assert len(seen) == 1


def test_unsubscribe_twice_is_noop():
    bus = EventBus()
    unsub = bus.subscribe(lambda e: None)
    unsub()
    unsub()  # must not raise


def test_multiple_subscribers_all_fire():
    bus = EventBus()
    a, b = [], []
    bus.subscribe(a.append)
    bus.subscribe(b.append)
    bus.emit(SteeringApplied(alphas={"x": 0.5}, entries={"x": (0.5, None)}))
    assert len(a) == 1
    assert len(b) == 1


def test_subscriber_exception_does_not_break_emit():
    bus = EventBus()
    seen = []

    def _boom(event):
        raise RuntimeError("boom")

    bus.subscribe(_boom)
    bus.subscribe(seen.append)
    # The exception is swallowed into a warning; the second subscriber
    # still receives the event.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bus.emit(SteeringCleared())
    assert len(seen) == 1
    assert any("subscriber" in str(wi.message) for wi in w)


def test_subscriber_that_unsubscribes_mid_emit():
    bus = EventBus()
    seen = []
    unsubs: list = []

    def _self_unsub(event):
        seen.append(event)
        unsubs[0]()

    unsubs.append(bus.subscribe(_self_unsub))
    bus.emit(SteeringCleared())
    bus.emit(SteeringCleared())
    # First emit sees the event, second emit does not.
    assert len(seen) == 1


def test_vector_extracted_fields():
    ev = VectorExtracted(name="angry.calm", profile={"a": 1}, metadata={"method": "pca"})
    assert ev.name == "angry.calm"
    assert ev.metadata["method"] == "pca"


def test_steering_applied_entries_carries_trigger():
    """``entries`` keys mirror ``alphas`` and carry ``(alpha, Trigger)``."""
    ev = SteeringApplied(
        alphas={"honest": 0.5, "warm": 0.3},
        entries={
            "honest": (0.5, Trigger.BOTH),
            "warm": (0.3, Trigger.AFTER_THINKING),
        },
    )
    assert set(ev.entries.keys()) == set(ev.alphas.keys())
    assert ev.entries["warm"][1] is Trigger.AFTER_THINKING


def test_probe_scored_fields():
    ev = ProbeScored(readings={"honest": 0.8, "warm": -0.2})
    assert ev.readings["honest"] == 0.8


def test_generation_started_fields():
    ev = GenerationStarted(input="hello", stateless=True)
    assert ev.input == "hello"
    assert ev.stateless is True


def test_generation_finished_wraps_result():
    # ``result`` is deliberately typed as Any so the event bus doesn't
    # import the heavy dataclass module; a stand-in with the expected
    # attribute is enough for subscribers.
    class _FakeResult:
        text = "hi"
    ev = GenerationFinished(result=_FakeResult())
    assert ev.result.text == "hi"


def test_events_are_frozen_dataclasses():
    import pytest as _p
    ev = SteeringCleared()
    with _p.raises(Exception):
        ev.foo = "x"  # type: ignore[attr-defined]
