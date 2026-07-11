"""Persistent compile-clean capture parity (slice 2).

The compiled MPS path can't register transient per-token capture hooks (they
graph-break / recompile), so probed generation rides always-on pre-compile
``copy_`` hooks (:func:`install_persistent_capture_hooks`) plus a post-forward
:meth:`HiddenCapture.ingest_persistent` accumulation.  These tests assert that
path produces *byte-identical* bucket state (latest slice, tail ring, full
stack) and step-sink rows to the transient hook path it replaces — so every
downstream consumer (``tail_slice_at`` / ``stacked`` / ``latest_per_layer`` /
the gate + trait sinks) is unaffected by which capture path ran.

Pure CPU: a toy block stack stands in for the decoder so the accumulation logic
is exercised without a model download.
"""

from __future__ import annotations

from typing import Callable

import torch

from saklas.core.hooks import (
    HiddenCapture,
    install_persistent_capture_hooks,
)

_D = 8
_LAYERS = [0, 1, 2]


class _Block(torch.nn.Module):
    """Returns a per-forward residual whose ``[0, -1, :]`` is a known value.

    Per-forward state rides a shared ``clock`` list (set via ``object.__setattr__``
    so nn.Module's Tensor/Module-typed ``__setattr__`` doesn't intercept the plain
    Python attributes), keeping the last-token slice deterministic in
    ``(layer tag, forward index)`` for element-wise comparison across paths.
    """

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


def _drive(
    stack: torch.nn.ModuleList, clock: list[float], n_forward: int,
    *, after: "Callable[[], None] | None" = None,
) -> None:
    """Run ``n_forward`` forwards through the stack (each block sees the prior)."""
    for f in range(n_forward):
        clock[0] = float(f)
        x = torch.ones(1, 1, _D)
        for blk in stack:
            x = blk(x)[0]
        if after is not None:
            after()


def test_persistent_ingest_matches_transient_tail_ring():
    """Aggregate-tail (gating / lean / aggregate-only) parity.

    The bounded tail ring drives ``tail_slice_at`` (the full-roster finalize),
    so its contents must match between the in-hook and persistent paths.
    """
    n = 12  # > _AGG_TAIL_DEPTH so the ring rotates

    # Transient reference.
    stack_t, clock_t = _toy_stack()
    cap_t = HiddenCapture()
    cap_t.attach(stack_t, list(_LAYERS))
    cap_t.set_aggregate_tail(8)
    _drive(stack_t, clock_t, n)

    # Persistent path: real installer hooks write the buffers; ingest accumulates.
    stack_p, clock_p = _toy_stack()
    buffers, handles = install_persistent_capture_hooks(
        stack_p, _D, torch.device("cpu"), torch.float32,
    )
    cap_p = HiddenCapture()
    cap_p.attach_persistent(list(_LAYERS), buffers)
    cap_p.set_aggregate_tail(8)
    try:
        _drive(stack_p, clock_p, n, after=cap_p.ingest_persistent)
    finally:
        for h in handles:
            h.remove()

    assert cap_t._forward_count == cap_p._forward_count == n
    for k in range(n):
        slc_t = cap_t.tail_slice_at(k)
        slc_p = cap_p.tail_slice_at(k)
        assert set(slc_t) == set(slc_p) == set(_LAYERS)
        for layer in _LAYERS:
            assert torch.equal(slc_t[layer], slc_p[layer]), (
                f"tail_slice_at({k}) layer {layer} diverged"
            )


def test_persistent_ingest_matches_transient_selective_tail_with_sink():
    """Selective deep-tail layers must match on transient and persistent paths."""
    n = 7
    rows_t: list[dict[int, torch.Tensor]] = []
    rows_p: list[dict[int, torch.Tensor]] = []

    stack_t, clock_t = _toy_stack()
    cap_t = HiddenCapture()
    cap_t.attach(stack_t, list(_LAYERS))
    cap_t.set_tail_with_sink(
        3,
        lambda latest: rows_t.append({
            k: v.clone() for k, v in latest.items()
        }),
        tail_layers={2},
    )
    _drive(stack_t, clock_t, n, after=cap_t.fire_step_sink)

    stack_p, clock_p = _toy_stack()
    buffers, handles = install_persistent_capture_hooks(
        stack_p, _D, torch.device("cpu"), torch.float32,
    )
    cap_p = HiddenCapture()
    cap_p.attach_persistent(list(_LAYERS), buffers)
    cap_p.set_tail_with_sink(
        3,
        lambda latest: rows_p.append({
            k: v.clone() for k, v in latest.items()
        }),
        tail_layers={2},
    )
    try:
        _drive(stack_p, clock_p, n, after=cap_p.ingest_persistent)
    finally:
        for h in handles:
            h.remove()

    assert [len(cap_t._per_layer[layer]) for layer in _LAYERS] == [1, 1, 3]
    assert [len(cap_p._per_layer[layer]) for layer in _LAYERS] == [1, 1, 3]
    assert len(rows_t) == len(rows_p) == n
    for row_t, row_p in zip(rows_t, rows_p):
        assert set(row_t) == set(row_p) == set(_LAYERS)
        for layer in _LAYERS:
            assert torch.equal(row_t[layer], row_p[layer])

    for k in range(n):
        slc_t = cap_t.tail_slice_at(k)
        slc_p = cap_p.tail_slice_at(k)
        assert set(slc_t) == set(slc_p) == {2}
        assert torch.equal(slc_t[2], slc_p[2])


def test_persistent_ingest_matches_transient_incremental_and_sink():
    """Length-1 incremental parity + step-sink rows fire identically."""
    n = 6
    rows_t: list[dict[int, torch.Tensor]] = []
    rows_p: list[dict[int, torch.Tensor]] = []

    stack_t, clock_t = _toy_stack()
    cap_t = HiddenCapture()
    cap_t.attach(stack_t, list(_LAYERS))
    cap_t.set_incremental(lambda latest: rows_t.append(
        {k: v.clone() for k, v in latest.items()}
    ))
    # In-hook incremental accumulates during the forward; the FIX-F1 sink fires
    # post-forward via fire_step_sink.
    _drive(stack_t, clock_t, n, after=cap_t.fire_step_sink)

    stack_p, clock_p = _toy_stack()
    buffers, handles = install_persistent_capture_hooks(
        stack_p, _D, torch.device("cpu"), torch.float32,
    )
    cap_p = HiddenCapture()
    cap_p.attach_persistent(list(_LAYERS), buffers)
    cap_p.set_incremental(lambda latest: rows_p.append(
        {k: v.clone() for k, v in latest.items()}
    ))
    try:
        _drive(stack_p, clock_p, n, after=cap_p.ingest_persistent)
    finally:
        for h in handles:
            h.remove()

    # Length-1 buckets: one slice per layer, the latest.
    assert len(rows_t) == len(rows_p) == n
    for r_t, r_p in zip(rows_t, rows_p):
        for layer in _LAYERS:
            assert torch.equal(r_t[layer], r_p[layer])
    lt, lp = cap_t.latest_per_layer(), cap_p.latest_per_layer()
    for layer in _LAYERS:
        assert torch.equal(lt[layer], lp[layer])


def test_persistent_ingest_full_stack_matches_transient():
    """Non-incremental full-retention parity (return_hidden ``stacked()``)."""
    n = 5
    stack_t, clock_t = _toy_stack()
    cap_t = HiddenCapture()
    cap_t.attach(stack_t, list(_LAYERS))  # append mode (no set_*)
    _drive(stack_t, clock_t, n)

    stack_p, clock_p = _toy_stack()
    buffers, handles = install_persistent_capture_hooks(
        stack_p, _D, torch.device("cpu"), torch.float32,
    )
    cap_p = HiddenCapture()
    cap_p.attach_persistent(list(_LAYERS), buffers)  # append mode
    try:
        _drive(stack_p, clock_p, n, after=cap_p.ingest_persistent)
    finally:
        for h in handles:
            h.remove()

    st, sp = cap_t.stacked(), cap_p.stacked()
    assert set(st) == set(sp) == set(_LAYERS)
    for layer in _LAYERS:
        assert st[layer].shape == sp[layer].shape == (n, _D)
        assert torch.equal(st[layer], sp[layer])


def test_attach_persistent_registers_no_hooks_and_detach_clears_buffers():
    """The persistent path must not register transient hooks (the routing's
    ``not self._capture._handles`` gate keys the compiled clean path off this),
    and ``detach`` must drop the buffer source so a later gen can't read stale
    slices through ``ingest_persistent``.
    """
    buffers = {i: torch.zeros(_D) for i in _LAYERS}
    cap = HiddenCapture()
    cap.attach_persistent(list(_LAYERS), buffers)
    assert cap._handles == []
    assert set(cap._persistent_buffers) == set(_LAYERS)
    cap.detach()
    assert cap._persistent_buffers == {}
    # ingest is a no-op once the buffer source is gone.
    cap.ingest_persistent()
    assert cap._forward_count == 0
