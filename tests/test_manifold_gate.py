"""Manifold-probe gates: parse + runtime semantics.

Phase 2 of the manifold-probes feature.  The grammar work happens in
``steering_expr.py``; this file covers the integration story end to end —
parse a manifold gate expression, drive ``Trigger.active`` with a fake
``TriggerContext.probe_scores`` dict that mirrors what
``ManifoldMonitor.flat_scalars`` produces, and assert the gate fires
exactly when the corresponding scalar crosses its threshold.

The runtime side has zero code changes: ``ProbeGate.probe`` stores the
full namespaced string verbatim (``"circumplex:fraction"`` /
``"circumplex@elated"``), and the session merges the manifold scalars
into ``probe_scores`` under those same keys, so ``Trigger.active``'s
``ctx.probe_scores.get(gate.probe)`` lookup just works.  These tests
lock that contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from saklas.core.steering_expr import parse_expr
from saklas.core.triggers import ProbeGate, TriggerContext
from saklas.io import selectors as sel

if TYPE_CHECKING:
    from saklas.core.steering import Steering


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Run each test against an empty SAKLAS_HOME.

    Bundled bipolar concepts (``angry.calm``) would otherwise resolve
    a bare-pole reference like ``calm`` through ``resolve_pole`` and
    canonicalize it to ``angry.calm``, masking the multi-term
    composition the cross-probe tests are about.  An isolated home
    keeps every bare slug resolving to itself with sign +1.
    """
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    sel.invalidate()
    yield
    sel.invalidate()


def _trigger_of(steering: Steering, key: str) -> Any:
    """Pull the :class:`Trigger` attached to ``key`` in ``steering.alphas``."""
    val = steering.alphas[key]
    if isinstance(val, tuple):
        _, trig = val
        return trig
    return steering.trigger


def _ctx(scores: dict[str, float]) -> TriggerContext:
    """Build a decode-phase TriggerContext with the given probe scores."""
    ctx = TriggerContext()
    ctx.is_prefill = False
    ctx.probe_scores = dict(scores)
    return ctx


# ---------------------------------------------------- parse-side contract ---


class TestParseStoresNamespacedProbe:
    def test_fraction_gate_probe_string(self):
        s = parse_expr("0.3 happy.sad @when:circumplex:fraction > 0.5")
        trig = _trigger_of(s, "happy.sad")
        assert trig.gate == ProbeGate(
            probe="circumplex:fraction", op=">", threshold=0.5,
        )

    def test_label_gate_probe_string(self):
        s = parse_expr("0.3 happy.sad @when:circumplex@elated > 0.7")
        trig = _trigger_of(s, "happy.sad")
        assert trig.gate == ProbeGate(
            probe="circumplex@elated", op=">", threshold=0.7,
        )

    def test_label_gate_with_negative_threshold(self):
        # Label-similarity scalars carry ``-distance`` (larger = closer),
        # so users naturally write negative thresholds.  The signed-NUM
        # path in the parser must accept this.
        s = parse_expr("0.3 happy.sad @when:circumplex@elated > -0.1")
        trig = _trigger_of(s, "happy.sad")
        assert trig.gate.threshold == -0.1
        assert trig.gate.op == ">"


# ----------------------------------------------------- runtime gate firing ---


class TestManifoldFractionGate:
    def test_fires_when_above_threshold(self):
        s = parse_expr("0.3 happy.sad @when:circumplex:fraction > 0.5")
        trig = _trigger_of(s, "happy.sad")
        ctx = _ctx({"circumplex:fraction": 0.7})
        assert trig.active(ctx)

    def test_does_not_fire_when_below_threshold(self):
        s = parse_expr("0.3 happy.sad @when:circumplex:fraction > 0.5")
        trig = _trigger_of(s, "happy.sad")
        ctx = _ctx({"circumplex:fraction": 0.3})
        assert not trig.active(ctx)

    def test_does_not_fire_when_equal_under_strict(self):
        s = parse_expr("0.3 happy.sad @when:circumplex:fraction > 0.5")
        trig = _trigger_of(s, "happy.sad")
        ctx = _ctx({"circumplex:fraction": 0.5})
        assert not trig.active(ctx)

    def test_gte_fires_at_equality(self):
        s = parse_expr("0.3 happy.sad @when:circumplex:fraction >= 0.5")
        trig = _trigger_of(s, "happy.sad")
        assert trig.active(_ctx({"circumplex:fraction": 0.5}))
        assert trig.active(_ctx({"circumplex:fraction": 0.6}))
        assert not trig.active(_ctx({"circumplex:fraction": 0.49}))

    def test_lt_fires_below(self):
        s = parse_expr("0.3 happy.sad @when:circumplex:fraction < 0.2")
        trig = _trigger_of(s, "happy.sad")
        assert trig.active(_ctx({"circumplex:fraction": 0.1}))
        assert not trig.active(_ctx({"circumplex:fraction": 0.2}))

    def test_lte_fires_at_equality(self):
        s = parse_expr("0.3 happy.sad @when:circumplex:fraction <= 0.2")
        trig = _trigger_of(s, "happy.sad")
        assert trig.active(_ctx({"circumplex:fraction": 0.2}))
        assert trig.active(_ctx({"circumplex:fraction": 0.1}))
        assert not trig.active(_ctx({"circumplex:fraction": 0.21}))

    def test_missing_key_does_not_fire(self):
        # Manifold probe not attached at session-side, or the scalar
        # never landed in this step's ``probe_scores`` — the gate must
        # report inactive instead of raising.  Matches the vector-probe
        # missing-key convention.
        s = parse_expr("0.3 happy.sad @when:circumplex:fraction > 0.5")
        trig = _trigger_of(s, "happy.sad")
        ctx = _ctx({"other:fraction": 0.9})
        assert not trig.active(ctx)

    def test_prefill_does_not_fire_even_with_passing_score(self):
        # Probe scores aren't available during prefill (capture runs *in*
        # the forward, scoring runs *after*).  The gate stays inactive
        # even when a stale ``probe_scores`` carries a passing value.
        s = parse_expr("0.3 happy.sad @when:circumplex:fraction > 0.5")
        trig = _trigger_of(s, "happy.sad")
        ctx = _ctx({"circumplex:fraction": 0.8})
        ctx.is_prefill = True
        assert not trig.active(ctx)


class TestManifoldLabelGate:
    def test_fires_above_negative_threshold(self):
        # ``flat_scalars`` stores ``-distance`` so a small distance lands
        # close to zero from below.  ``> -0.1`` means "within 0.1 of the
        # node".
        s = parse_expr("0.3 happy.sad @when:circumplex@elated > -0.1")
        trig = _trigger_of(s, "happy.sad")
        ctx = _ctx({"circumplex@elated": -0.05})
        assert trig.active(ctx)

    def test_does_not_fire_when_too_far(self):
        s = parse_expr("0.3 happy.sad @when:circumplex@elated > -0.1")
        trig = _trigger_of(s, "happy.sad")
        ctx = _ctx({"circumplex@elated": -0.5})
        assert not trig.active(ctx)

    def test_missing_label_key_does_not_fire(self):
        # The named node label isn't present in this step's scalars (the
        # probe ranks top-N; non-top nodes never reach ``flat_scalars``).
        # Gate reports inactive.
        s = parse_expr("0.3 happy.sad @when:circumplex@elated > -0.1")
        trig = _trigger_of(s, "happy.sad")
        ctx = _ctx({"circumplex@calm": -0.05})
        assert not trig.active(ctx)


# ---------------------------------------------- merged-scalar (vector + mfd) ---


class TestVectorAndManifoldGatesCompose:
    """The session merges vector and manifold scalars into one dict.

    A multi-term expression where one term gates on a vector probe and
    another on a manifold scalar must read both keys out of the same
    ``ctx.probe_scores`` without interference.  This is the cross-probe
    convergence point the Phase 1 ``flat_scalars`` adapter exists for.
    """

    def test_both_gates_fire_when_both_keys_above(self):
        text = (
            "0.3 happy.sad@when:angry.calm>0.4 "
            "+ 0.2 calm@when:circumplex:fraction>0.5"
        )
        s = parse_expr(text)
        ctx = _ctx({
            "angry.calm": 0.6,
            "circumplex:fraction": 0.7,
        })
        assert _trigger_of(s, "happy.sad").active(ctx)
        assert _trigger_of(s, "calm").active(ctx)

    def test_only_vector_gate_fires_when_manifold_below(self):
        text = (
            "0.3 happy.sad@when:angry.calm>0.4 "
            "+ 0.2 calm@when:circumplex:fraction>0.5"
        )
        s = parse_expr(text)
        ctx = _ctx({
            "angry.calm": 0.6,
            "circumplex:fraction": 0.3,
        })
        assert _trigger_of(s, "happy.sad").active(ctx)
        assert not _trigger_of(s, "calm").active(ctx)

    def test_only_manifold_gate_fires_when_vector_below(self):
        text = (
            "0.3 happy.sad@when:angry.calm>0.4 "
            "+ 0.2 calm@when:circumplex:fraction>0.5"
        )
        s = parse_expr(text)
        ctx = _ctx({
            "angry.calm": 0.2,
            "circumplex:fraction": 0.7,
        })
        assert not _trigger_of(s, "happy.sad").active(ctx)
        assert _trigger_of(s, "calm").active(ctx)

    def test_mixed_fraction_and_label_against_one_manifold(self):
        # Two gates against different channels of the same manifold —
        # the namespaced-key trick keeps them independent in
        # ``probe_scores`` and so independent in the trigger logic.
        text = (
            "0.3 happy.sad@when:circumplex:fraction>=0.5 "
            "+ 0.2 calm@when:circumplex@elated<-0.2"
        )
        s = parse_expr(text)
        ctx = _ctx({
            "circumplex:fraction": 0.6,
            "circumplex@elated": -0.5,
        })
        assert _trigger_of(s, "happy.sad").active(ctx)
        assert _trigger_of(s, "calm").active(ctx)
