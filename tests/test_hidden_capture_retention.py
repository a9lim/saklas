"""``HiddenCapture.set_retention`` — the one retention setter.

``set_incremental`` / ``set_aggregate_tail`` / ``set_tail_with_sink`` are the
three named points in the ``(depth, step_sink, tail_layers)`` space
``set_retention`` parameterizes.  These tests pin that each name is exactly its
``set_retention`` call (so the aliases can never drift from the implementation)
and that the two axes stay orthogonal — a deep ring with a sink, and a length-1
buffer without one, are both reachable.

Pure CPU: a toy block stack stands in for the decoder.
"""

from __future__ import annotations

from typing import Callable

import torch

from saklas.core.hooks import HiddenCapture

_D = 4
_LAYERS = [0, 1, 2]


class _Block(torch.nn.Module):
    """Emits a residual whose ``[0, -1, :]`` is ``tag + clock`` (deterministic)."""

    _tag: float
    _clock: list[float]

    def __init__(self, tag: float, clock: list[float]) -> None:
        super().__init__()
        object.__setattr__(self, "_tag", tag)
        object.__setattr__(self, "_clock", clock)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        h = torch.zeros(1, x.shape[1], _D)
        h[0, -1, :] = self._tag + self._clock[0]
        return (h,)


def _toy_stack() -> tuple[torch.nn.ModuleList, list[float]]:
    clock = [0.0]
    return (
        torch.nn.ModuleList([_Block(float(i + 1), clock) for i in _LAYERS]),
        clock,
    )


def _drive(stack: torch.nn.ModuleList, clock: list[float], n: int) -> None:
    for f in range(n):
        clock[0] = float(f)
        x = torch.ones(1, 1, _D)
        for blk in stack:
            x = blk(x)[0]


_State = tuple[
    bool,
    int,
    "frozenset[int] | None",
    "Callable[[int, dict[int, torch.Tensor]], None] | None",
    "int | None",
]


def _state(cap: HiddenCapture) -> _State:
    """The five fields every retention setter writes."""
    return (
        cap._incremental, cap._tail_depth, cap._tail_layers,
        cap._step_sink, cap._max_layer,
    )


def _armed(arm: Callable[[HiddenCapture], None]) -> HiddenCapture:
    stack, _clock = _toy_stack()
    cap = HiddenCapture()
    cap.attach(stack, list(_LAYERS))
    arm(cap)
    return cap


# --------------------------------------------------- the aliases are the setter ---

def test_set_incremental_is_set_retention_depth_one():
    def sink(_step: int, _latest: dict[int, torch.Tensor]) -> None:
        return None

    alias = _armed(lambda cap: cap.set_incremental(sink))
    explicit = _armed(lambda cap: cap.set_retention(depth=1, step_sink=sink))
    assert _state(alias) == _state(explicit)


def test_set_aggregate_tail_is_set_retention_without_a_sink():
    alias = _armed(lambda cap: cap.set_aggregate_tail(8))
    explicit = _armed(lambda cap: cap.set_retention(depth=8))
    assert _state(alias) == _state(explicit)
    assert alias._step_sink is None


def test_set_tail_with_sink_is_set_retention_with_both():
    def sink(_step: int, _latest: dict[int, torch.Tensor]) -> None:
        return None

    alias = _armed(
        lambda cap: cap.set_tail_with_sink(6, sink, tail_layers={0, 2})
    )
    explicit = _armed(
        lambda cap: cap.set_retention(
            depth=6, step_sink=sink, tail_layers={0, 2},
        )
    )
    assert _state(alias) == _state(explicit)
    assert alias._tail_layers == frozenset({0, 2})


def test_depth_floors_at_one():
    cap = _armed(lambda c: c.set_retention(depth=0))
    assert cap._tail_depth == 1
    cap = _armed(lambda c: c.set_retention(depth=-5))
    assert cap._tail_depth == 1


# ------------------------------------------------------- the axes stay orthogonal ---

def test_depth_one_overwrites_in_place_and_keeps_the_latest_slice():
    stack, clock = _toy_stack()
    cap = HiddenCapture()
    cap.attach(stack, list(_LAYERS))
    cap.set_retention(depth=1)
    _drive(stack, clock, 5)

    buckets = cap.per_layer_buckets()
    assert all(len(b) == 1 for b in buckets.values())
    latest = cap.latest_per_layer()
    for L in _LAYERS:
        # tag (L+1) + the last forward index (4).
        assert torch.allclose(latest[L], torch.full((_D,), float(L + 1) + 4.0))
    assert cap._forward_count == 5


def test_deep_ring_with_a_sink_retains_both_the_ring_and_the_per_step_rows():
    stack, clock = _toy_stack()
    cap = HiddenCapture()
    cap.attach(stack, list(_LAYERS))
    steps: list[int] = []

    def sink(step: int, latest: dict[int, torch.Tensor]) -> None:
        steps.append(step)
        assert set(latest) == set(_LAYERS)

    cap.set_retention(depth=3, step_sink=sink)
    for f in range(5):
        clock[0] = float(f)
        x = torch.ones(1, 1, _D)
        for blk in stack:
            x = blk(x)[0]
        cap.fire_step_sink(f)

    assert steps == [0, 1, 2, 3, 4]
    # The ring holds the last three forwards (2, 3, 4).
    assert all(len(b) == 3 for b in cap.per_layer_buckets().values())
    for f in (2, 3, 4):
        slc = cap.tail_slice_at(f)
        for L in _LAYERS:
            assert torch.allclose(
                slc[L], torch.full((_D,), float(L + 1) + float(f)),
            )


def test_tail_layers_restricts_the_deep_ring_only():
    stack, clock = _toy_stack()
    cap = HiddenCapture()
    cap.attach(stack, list(_LAYERS))
    cap.set_retention(depth=4, tail_layers={1})
    _drive(stack, clock, 6)

    buckets = cap.per_layer_buckets()
    assert len(buckets[1]) == 4          # deep ring on the selected layer
    assert len(buckets[0]) == 1          # length-1 latest-slice elsewhere
    assert len(buckets[2]) == 1
    # Only the ringed layer is offered to the finalize pool.
    assert set(cap.tail_slice_at(5)) == {1}


def test_attach_resets_to_full_retention():
    stack, clock = _toy_stack()
    cap = HiddenCapture()
    cap.attach(stack, list(_LAYERS))
    cap.set_retention(depth=4, step_sink=lambda _s, _l: None)
    cap.detach()
    cap.attach(stack, list(_LAYERS))
    assert _state(cap) == (False, 1, None, None, None)
    _drive(stack, clock, 3)
    # Full retention appends a distinct clone per forward.
    assert all(len(b) == 3 for b in cap.per_layer_buckets().values())
    assert cap.stacked()[0].shape == (3, _D)
