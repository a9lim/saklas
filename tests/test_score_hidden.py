"""CPU tests for hidden-state round-trip scoring.

Covers Monitor.score_stack and SaklasSession.score_hidden using a synthetic
monitor and a mock SaklasSession built without a real model.

Mahalanobis-only (4.0 collapse): probe scoring *requires* a whitener covering
every probed layer — there is no Euclidean path — so every monitor here is
built with a synthetic covering whitener (``tests/_whitener.py``).

Post-unification (the read side is the unified ``Monitor``): a probe is a
:class:`~saklas.core.manifold.Manifold`, and a single steering direction is
attached as a 1-node neutral-anchored ray via ``fold_directions_to_subspace``
(exactly the session's ``_fold_profile_probe`` path).  The read is the
in-subspace **domain coordinate** (``ProbeReading.coords[0]``), not a signed
cosine — but because the fold anchors the node at coord ``1.0`` along the
direction, an aligned hidden still reads ≈ ``+1``, an anti-aligned one ≈ ``-1``,
and an M-orthogonal one ≈ ``0``.  Assertions pin those coordinates with a
coordinate-scale tolerance and the ordering invariant (aligned > ortho > anti).
"""
from __future__ import annotations

import pytest
import torch

from typing import Any

from saklas.core.manifold import Manifold
from saklas.core.monitor import Monitor
from saklas.core.vectors import fold_directions_to_subspace
from tests._whitener import synthetic_means, synthetic_whitener

# A probe along axis 0, layer 0, dim 4 — easy to reason about.  The fold
# anchors the ray on the neutral mean (the whitener's own neutral mean), so
# probe/token vectors below are stated relative to that mean (added back where
# a concrete read value matters).
_DIM = 4
_MEANS0 = synthetic_means([0], _DIM)
_WHIT0 = synthetic_whitener([0], _DIM, means=_MEANS0)


_PROBE0 = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _m_orthogonal_to_probe() -> torch.Tensor:
    """A centered vector deliberately *Mahalanobis*-orthogonal to the probe.

    The whitened inner product is ``d^T Σ⁻¹ v`` — so a vector M-orthogonal to
    the probe ``d`` is any ``v`` Euclidean-orthogonal to ``Σ⁻¹ d``
    (``= whitener.apply_inv(layer, d)``).  An arbitrary axis vector
    (``[0,1,0,0]``) is *Euclidean*-orthogonal to the probe but NOT
    M-orthogonal under an anisotropic Σ, so its whitened coordinate is nonzero —
    hence the explicit construction here.
    """
    sinv_d = _WHIT0.apply_inv(0, _PROBE0).float()
    # Solve sinv_d[0]*a + sinv_d[1]*b = 0 in the first two coords (b = 1).
    a = -float(sinv_d[1]) / float(sinv_d[0])
    return torch.tensor([a, 1.0, 0.0, 0.0])


def _probe_manifold(
    direction: dict[int, torch.Tensor], means: Any, whit: Any,
) -> Manifold:
    """Fold a per-layer steering direction into a 1-node ray Manifold probe."""
    return fold_directions_to_subspace(
        "x", direction, dict(means), whitener=whit,
    )


def _monitor_with_probe() -> Monitor:
    """A monitor with a single axis-0 ray probe on layer 0, covered by a whitener."""
    m = _probe_manifold({0: _PROBE0.clone()}, _MEANS0, _WHIT0)
    return Monitor({"x": m}, layer_means=dict(_MEANS0), whitener=_WHIT0)


def test_score_stack_aligned_beats_anti_aligned():
    """An aligned token reads coord ≈ +1, an anti-aligned one ≈ -1, and a
    deliberately M-orthogonal token reads ≈ 0 — strictly ordered between
    them (the coordinate readout is monotonic in alignment)."""
    m = _monitor_with_probe()
    mu = _MEANS0[0]
    v_ortho = _m_orthogonal_to_probe()
    stack = torch.stack([
        torch.tensor([1.0, 0.0, 0.0, 0.0]) + mu,    # aligned   → +1
        torch.tensor([-1.0, 0.0, 0.0, 0.0]) + mu,   # anti      → -1
        v_ortho + mu,                                # M-orthogonal → 0
    ])
    agg, per_token = m.score_stack({0: stack}, accumulate=False)
    assert set(per_token.keys()) == {"x"}
    vals = per_token["x"]
    assert len(vals) == 3
    assert vals[0] == pytest.approx(1.0, abs=0.1)     # aligned
    assert vals[1] == pytest.approx(-1.0, abs=0.1)    # anti-aligned
    assert vals[2] == pytest.approx(0.0, abs=0.1)     # M-orthogonal
    assert vals[0] > vals[2] > vals[1]                # aligned > ortho > anti
    # ``agg`` (default agg_index=None) pools the last row → the coord readout.
    assert agg["x"].coords[0] == pytest.approx(vals[2], abs=1e-5)


def test_score_stack_agg_index_selects_row():
    m = _monitor_with_probe()
    mu = _MEANS0[0]
    stack = torch.stack([
        torch.tensor([1.0, 0.0, 0.0, 0.0]) + mu,
        torch.tensor([0.0, 1.0, 0.0, 0.0]) + mu,
    ])
    _agg, per_token = m.score_stack({0: stack}, accumulate=False)
    agg0, _ = m.score_stack({0: stack}, agg_index=0, accumulate=False)
    agg1, _ = m.score_stack({0: stack}, agg_index=1, accumulate=False)
    # The selected aggregate row's coord matches that row's per-token value.
    assert agg0["x"].coords[0] == pytest.approx(per_token["x"][0], abs=1e-5)
    assert agg1["x"].coords[0] == pytest.approx(per_token["x"][1], abs=1e-5)


def test_score_stack_accumulate_false_leaves_history_untouched():
    m = _monitor_with_probe()
    before = list(m.history["x"])
    m.score_stack(
        {0: torch.tensor([[1.0, 0.0, 0.0, 0.0]])}, accumulate=False,
    )
    assert list(m.history["x"]) == before


def test_score_stack_accumulate_true_records_aggregate():
    m = _monitor_with_probe()
    m.score_stack(
        {0: (torch.tensor([[1.0, 0.0, 0.0, 0.0]]) + _MEANS0[0])},
        accumulate=True,
    )
    assert len(m.history["x"]) == 1
    # Aligned token ⇒ coord ≈ +1 recorded (history stores the coord tuple).
    assert list(m.history["x"])[-1][0] == pytest.approx(1.0, abs=0.1)


def test_score_stack_empty_inputs():
    m = _monitor_with_probe()
    agg, per_token = m.score_stack({}, accumulate=False)
    # Empty capture → an empty-coords reading per probe, no per-token stream.
    assert set(agg.keys()) == {"x"}
    assert agg["x"].coords == ()
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


def test_score_probes_without_whitener_raises():
    """A probe attach without a whitener can't build its factors — Mahalanobis
    is mandatory, so ``add_probe`` raises rather than reading Euclidean."""
    from saklas.core.mahalanobis import WhitenerError

    m = _probe_manifold({0: _PROBE0.clone()}, _MEANS0, _WHIT0)
    mon = Monitor()  # no whitener wired
    with pytest.raises(WhitenerError, match="whitener"):
        mon.add_probe("x", m)


# ---------------------------------------------------------------------------
# SaklasSession.score_hidden
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
    h = {0: torch.tensor([1.0, 0.0, 0.0, 0.0]) + _MEANS0[0]}
    scores = s.score_hidden(h)
    assert set(scores.keys()) == {"x"}
    # Aligned with the probe ⇒ coord ≈ +1.
    assert scores["x"].coords[0] == pytest.approx(1.0, abs=0.1)


def test_score_hidden_stack_aggregate_only():
    s = _mock_session()
    stack = torch.stack([
        torch.tensor([1.0, 0.0, 0.0, 0.0]) + _MEANS0[0],
        torch.tensor([0.0, 1.0, 0.0, 0.0]) + _MEANS0[0],
    ])
    scores = s.score_hidden({0: stack})
    assert isinstance(scores, dict)
    assert set(scores.keys()) == {"x"}
    c = scores["x"].coords[0]
    assert c == c  # finite


def test_score_hidden_stack_per_token_returns_tuple():
    s = _mock_session()
    stack = torch.stack([
        torch.tensor([1.0, 0.0, 0.0, 0.0]) + _MEANS0[0],
        torch.tensor([-1.0, 0.0, 0.0, 0.0]) + _MEANS0[0],
    ])
    agg, per_token = s.score_hidden({0: stack}, per_token=True)
    assert set(agg.keys()) == {"x"}
    # Aligned token reads ≈ +1, the anti-aligned one ≈ -1; strictly ordered.
    assert per_token["x"][0] == pytest.approx(1.0, abs=0.1)
    assert per_token["x"][1] == pytest.approx(-1.0, abs=0.1)
    assert per_token["x"][0] > per_token["x"][1]


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
    s.score_hidden({0: torch.tensor([1.0, 0.0, 0.0, 0.0]) + _MEANS0[0]})
    assert list(s._monitor.history["x"]) == before


def test_score_hidden_accumulate_true_records():
    s = _mock_session()
    s.score_hidden(
        {0: torch.tensor([1.0, 0.0, 0.0, 0.0]) + _MEANS0[0]}, accumulate=True,
    )
    assert len(s._monitor.history["x"]) == 1


# ============================================ Mahalanobis read metric ===


def _whitener_from_neutrals(
    X: torch.Tensor, mean: torch.Tensor, layer: int = 0,
):
    """Build a single-layer LayerWhitener from synthetic neutrals."""
    from saklas.core.mahalanobis import LayerWhitener

    return LayerWhitener.from_neutral_activations(
        {layer: X}, {layer: mean},
    )


def test_anisotropic_whitened_read_downweights_high_variance_axis():
    """On strongly anisotropic neutrals the whitened coordinate down-weights
    alignment along the high-variance neutral direction relative to a
    probe-aligned read along a low-variance axis."""
    torch.manual_seed(1)
    # Probe along the LOW-variance axis 1.
    base = torch.randn(300, 4)
    base[:, 0] *= 20.0   # huge variance axis 0
    base[:, 1] *= 0.5    # tiny variance axis 1
    mean = base.mean(dim=0)
    whitener = _whitener_from_neutrals(base, mean)
    m = fold_directions_to_subspace(
        "x", {0: torch.tensor([0.0, 1.0, 0.0, 0.0])}, {0: mean},
        whitener=whitener,
    )
    maha = Monitor({"x": m}, layer_means={0: mean}, whitener=whitener)
    # A hidden along the probe (low-var axis) reads strongly; one along the
    # high-var axis reads weakly under the whitened metric.
    on_probe = maha.measure_from_hidden(
        {0: torch.tensor([0.0, 1.0, 0.0, 0.0]) + mean}, accumulate=False,
    )["x"].coords[0]
    off_probe = maha.measure_from_hidden(
        {0: torch.tensor([10.0, 0.0, 0.0, 0.0]) + mean}, accumulate=False,
    )["x"].coords[0]
    assert on_probe > off_probe


def test_set_whitener_invalidates_cache_and_switches_metric():
    """``set_whitener`` swaps the whitener and rebuilds the per-probe factors so
    the next scoring call reads against the new covariance."""
    torch.manual_seed(2)
    base_a = torch.randn(300, 4)
    base_a[:, 0] *= 20.0
    mean = base_a.mean(dim=0)
    whit_a = _whitener_from_neutrals(base_a, mean)

    # A second, differently-anisotropic whitener over the SAME mean (so the
    # centering is unchanged and only the metric differs).
    torch.manual_seed(20)
    base_b = torch.randn(300, 4)
    base_b[:, 1] *= 20.0
    base_b = base_b - base_b.mean(dim=0) + mean
    whit_b = _whitener_from_neutrals(base_b, mean)

    direction = {0: torch.tensor([1.0, 1.0, 0.0, 0.0]) / (2 ** 0.5)}
    m = fold_directions_to_subspace("x", direction, {0: mean}, whitener=whit_a)
    mon = Monitor({"x": m}, layer_means={0: mean}, whitener=whit_a)
    h = {0: torch.tensor([10.0, 0.5, 0.0, 0.0]) + mean}
    before = mon.measure_from_hidden(h, accumulate=False)["x"].coords[0]
    assert mon.whitener is whit_a

    mon.set_whitener(whit_b)
    assert mon.whitener is whit_b
    after = mon.measure_from_hidden(h, accumulate=False)["x"].coords[0]
    # Same instance, same probe, different metric ⇒ different reading.
    assert abs(after - before) > 1e-2

    # Re-setting the identical whitener is a no-op (no exception, metric
    # unchanged).
    again = mon.measure_from_hidden(h, accumulate=False)["x"].coords[0]
    assert again == pytest.approx(after, abs=1e-6)


def test_partial_coverage_raises():
    """All-or-nothing read metric: a whitener covering only some probed
    layers is a hard error (Mahalanobis is mandatory; there is no
    per-layer mix and no Euclidean fallback)."""
    from saklas.core.mahalanobis import WhitenerError

    torch.manual_seed(3)
    X0 = torch.randn(120, 4)
    mean0 = X0.mean(dim=0)
    # Whitener covers layer 0 only; the probe spans layers 0 and 1.
    whitener = _whitener_from_neutrals(X0, mean0, layer=0)
    direction = {
        0: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        1: torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    means = {0: mean0, 1: torch.zeros(4)}
    # Coverage is enforced at construction: a partial whitener cannot produce
    # a current manifold carrying mixed metrics across its layers.
    with pytest.raises(WhitenerError, match="whitener"):
        fold_directions_to_subspace("x", direction, means, whitener=whitener)
