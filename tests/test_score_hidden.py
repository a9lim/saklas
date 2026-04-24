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


def test_score_stack_uneven_T_raises_value_error():
    """Mixed T across layers is a caller bug — fail loud, not silently skip."""
    m = _monitor_with_probe()
    bad = {
        0: torch.zeros(3, 4),
        1: torch.zeros(2, 4),
    }
    with pytest.raises(ValueError, match="expected"):
        m.score_stack(bad, accumulate=False)


def test_score_stack_accumulate_false_leaves_pending_flag_clear():
    """The researcher-facing path must not signal pending-per-token when
    the caller explicitly opted out of accumulation."""
    m = _monitor_with_probe()
    assert m.has_pending_per_token() is False
    m.score_stack(
        {0: torch.tensor([[1.0, 0.0, 0.0, 0.0]])}, accumulate=False,
    )
    assert m.has_pending_per_token() is False


# ---------------------------------------------------------------------------
# Task 4 — SaklasSession.score_hidden
# ---------------------------------------------------------------------------

from saklas.core.errors import SaklasError  # noqa: E402
from saklas.core.session import SaklasSession  # noqa: E402


def _mock_session() -> SaklasSession:
    """Build a SaklasSession without touching a real model.

    We bypass __init__ (which requires a PreTrainedModel) and wire up
    only the fields score_hidden reads: _monitor, _device.  Every other
    attribute remains un-set; score_hidden must not touch them.
    """
    s = SaklasSession.__new__(SaklasSession)
    s._monitor = _monitor_with_probe()
    s._device = torch.device("cpu")
    return s


def test_score_hidden_single_state_returns_probe_dict():
    s = _mock_session()
    h = {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}
    scores = s.score_hidden(h)
    assert set(scores.keys()) == {"x"}
    assert scores["x"] == pytest.approx(1.0, abs=1e-5)


def test_score_hidden_stack_aggregate_only():
    s = _mock_session()
    stack = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    scores = s.score_hidden({0: stack})
    # Aggregate pools from last row (neutral wrt probe).
    assert isinstance(scores, dict)
    assert abs(scores["x"]) < 1e-5


def test_score_hidden_stack_per_token_returns_tuple():
    s = _mock_session()
    stack = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
    ])
    agg, per_token = s.score_hidden({0: stack}, per_token=True)
    assert set(agg.keys()) == {"x"}
    assert per_token["x"][0] == pytest.approx(1.0, abs=1e-5)
    assert per_token["x"][1] == pytest.approx(-1.0, abs=1e-5)


def test_score_hidden_empty_dict_raises():
    s = _mock_session()
    with pytest.raises(SaklasError, match="no layers"):
        s.score_hidden({})


def test_score_hidden_mixed_shapes_raises():
    s = _mock_session()
    bad = {
        0: torch.tensor([1.0, 0.0, 0.0, 0.0]),        # [D]
        1: torch.tensor([[1.0, 0.0, 0.0, 0.0]]),      # [T, D]
    }
    with pytest.raises(SaklasError, match="mixed shapes"):
        s.score_hidden(bad)


def test_score_hidden_uneven_T_raises():
    s = _mock_session()
    bad = {
        0: torch.zeros(3, 4),
        1: torch.zeros(2, 4),
    }
    # Both the monitor's ValueError (uneven T) and the session's
    # SaklasError wrapping must surface as SaklasError at the public
    # boundary — callers catching SaklasError must not miss this.
    with pytest.raises(SaklasError):
        s.score_hidden(bad)


def test_score_hidden_bad_ndim_raises():
    """ndim=3 and beyond are not [D] or [T, D]; must raise SaklasError."""
    s = _mock_session()
    bad = {0: torch.zeros(2, 3, 4)}
    with pytest.raises(SaklasError, match="expected \\[D\\] or \\[T, D\\]"):
        s.score_hidden(bad)


def test_score_hidden_dim_mismatch_raises():
    """A tensor with wrong hidden_dim must raise SaklasError, not leak
    a raw torch RuntimeError from the scoring matmul."""
    s = _mock_session()
    # Monitor probe is dim=4 at layer 0; pass dim=8 input.
    bad = {0: torch.zeros(2, 8)}
    with pytest.raises(SaklasError, match="dim mismatch"):
        s.score_hidden(bad)


def test_score_hidden_accumulate_false_does_not_mutate_history():
    s = _mock_session()
    before = list(s._monitor.history["x"])
    s.score_hidden({0: torch.tensor([1.0, 0.0, 0.0, 0.0])})
    assert list(s._monitor.history["x"]) == before
    assert s._monitor._stats["x"]["count"] == 0


def test_score_hidden_accumulate_true_records():
    s = _mock_session()
    s.score_hidden(
        {0: torch.tensor([1.0, 0.0, 0.0, 0.0])}, accumulate=True,
    )
    assert s._monitor._stats["x"]["count"] == 1
