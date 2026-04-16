"""EventBus unit tests — subscribe, unsubscribe, exception isolation."""

import warnings

from saklas.core.events import (
    EventBus,
    SteeringApplied,
    SteeringCleared,
    VectorExtracted,
)


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
    bus.emit(SteeringApplied(alphas={"x": 0.5}))
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
