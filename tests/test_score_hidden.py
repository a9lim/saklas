"""CPU tests for hidden-state round-trip scoring.

Covers TraitMonitor.score_stack (Task 3) and SaklasSession.score_hidden
(Task 4) using a synthetic monitor and a mock SaklasSession built without
a real model.
"""
from __future__ import annotations

import pytest
import torch

from saklas.core.monitor import TraitMonitor


def _monitor_with_probe() -> TraitMonitor:
    """Build a monitor with a single probe on layer 0, dim=4.

    Probe direction is [1, 0, 0, 0] with unit norm and unit weight so
    scoring math is easy to reason about.
    """
    profile = {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}
    return TraitMonitor({"x": profile}, layer_means=None)


def test_score_stack_single_layer_per_token_values():
    m = _monitor_with_probe()
    # Three "tokens": first aligned with probe, second anti-aligned, third neutral.
    stack = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    captured = {0: stack}
    agg, per_token = m.score_stack(captured, accumulate=False)
    assert set(per_token.keys()) == {"x"}
    vals = per_token["x"]
    assert len(vals) == 3
    assert vals[0] == pytest.approx(1.0, abs=1e-5)
    assert vals[1] == pytest.approx(-1.0, abs=1e-5)
    assert abs(vals[2]) < 1e-5
    # Default agg_index=None uses the last row — token 2 is near-zero.
    assert abs(agg["x"]) < 1e-5


def test_score_stack_agg_index_selects_row():
    m = _monitor_with_probe()
    stack = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    agg, _ = m.score_stack({0: stack}, agg_index=0, accumulate=False)
    assert agg["x"] == pytest.approx(1.0, abs=1e-5)
    agg2, _ = m.score_stack({0: stack}, agg_index=1, accumulate=False)
    assert abs(agg2["x"]) < 1e-5


def test_score_stack_accumulate_false_leaves_history_untouched():
    m = _monitor_with_probe()
    before = list(m.history["x"])
    m.score_stack(
        {0: torch.tensor([[1.0, 0.0, 0.0, 0.0]])}, accumulate=False,
    )
    assert list(m.history["x"]) == before
    assert m._stats["x"]["count"] == 0


def test_score_stack_accumulate_true_records_aggregate():
    m = _monitor_with_probe()
    m.score_stack(
        {0: torch.tensor([[1.0, 0.0, 0.0, 0.0]])}, accumulate=True,
    )
    assert m._stats["x"]["count"] == 1
    assert list(m.history["x"])[-1] == pytest.approx(1.0, abs=1e-5)


def test_score_stack_empty_inputs():
    m = _monitor_with_probe()
    agg, per_token = m.score_stack({}, accumulate=False)
    assert agg == {"x": 0.0}
    assert per_token == {"x": []}
