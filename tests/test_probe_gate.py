"""Probe-gated triggers (v2.1): grammar, runtime gate, session detection.

Covers:
- ``ProbeGate.evaluate`` semantics across all four ops.
- ``Trigger.when`` constructor + ``Trigger.active`` gating logic.
- ``TriggerContext.probe_scores`` plumbing + reset behavior.
- Grammar: lex/parse/format round-trip for ``@when:<probe><op><val>``.
- Session: ``_steering_needs_probe_gating`` walks the active stack.

End-to-end gating during ``generate_steered`` is exercised by the
GPU-required ``test_session.py`` suite — model + monitor wiring is too
heavy to stub usefully.  These tests cover the moving parts.
"""

from __future__ import annotations

import pytest

from saklas.core.events import EventBus
from saklas.core.session import SaklasSession
from saklas.core.steering import Steering
from saklas.core.steering_expr import (
    SteeringExprError, format_expr, parse_expr,
)
from saklas.core.triggers import (
    ProbeGate, Trigger, TriggerContext,
)


# ----------------------------------------------------------- ProbeGate ---

class TestProbeGate:
    def test_gt_strict(self):
        g = ProbeGate(probe="x", op=">", threshold=0.4)
        assert g.evaluate(0.5)
        assert not g.evaluate(0.4)
        assert not g.evaluate(0.0)

    def test_gte(self):
        g = ProbeGate(probe="x", op=">=", threshold=0.4)
        assert g.evaluate(0.5)
        assert g.evaluate(0.4)
        assert not g.evaluate(0.39)

    def test_lt_strict(self):
        g = ProbeGate(probe="x", op="<", threshold=-0.2)
        assert g.evaluate(-0.3)
        assert not g.evaluate(-0.2)
        assert not g.evaluate(0.5)

    def test_lte(self):
        g = ProbeGate(probe="x", op="<=", threshold=0.0)
        assert g.evaluate(-0.1)
        assert g.evaluate(0.0)
        assert not g.evaluate(0.0001)

    def test_frozen_dataclass_equality(self):
        a = ProbeGate(probe="angry.calm", op=">", threshold=0.4)
        b = ProbeGate(probe="angry.calm", op=">", threshold=0.4)
        c = ProbeGate(probe="angry.calm", op=">", threshold=0.5)
        assert a == b
        assert a != c
        # Hashable — required for ``Trigger`` (frozen=True) + dict keys.
        assert hash(a) == hash(b)


# ----------------------------------------------------------- Trigger.when ---

class TestTriggerWhen:
    def test_factory_builds_gate_only_trigger(self):
        t = Trigger.when("angry.calm", ">", 0.4)
        assert t.gate == ProbeGate(probe="angry.calm", op=">", threshold=0.4)
        # ``when`` triggers turn off prefill (no probe reading available)
        # and leave the other windows wide open so the gate is the only
        # discriminator on decode steps.
        assert t.prompt is False
        assert t.generated is True
        assert t.thinking is True
        assert t.response is True
        assert t.first_n is None
        assert t.after_n is None

    def test_active_inactive_during_prefill(self):
        t = Trigger.when("angry.calm", ">", 0.4)
        ctx = TriggerContext()
        ctx.is_prefill = True
        # Even with a passing score, prefill keeps the gate inactive —
        # probe scores aren't refreshed until *after* the first forward.
        ctx.probe_scores = {"angry.calm": 0.8}
        assert not t.active(ctx)

    def test_active_when_score_above_threshold(self):
        t = Trigger.when("angry.calm", ">", 0.4)
        ctx = TriggerContext()
        ctx.is_prefill = False
        ctx.probe_scores = {"angry.calm": 0.5}
        assert t.active(ctx)

    def test_inactive_when_score_below(self):
        t = Trigger.when("angry.calm", ">", 0.4)
        ctx = TriggerContext()
        ctx.is_prefill = False
        ctx.probe_scores = {"angry.calm": 0.2}
        assert not t.active(ctx)

    def test_inactive_when_probe_missing(self):
        # Gate names a probe that the monitor doesn't carry.  Should
        # report inactive instead of raising — gating on a non-existent
        # probe is a degenerate ask, but a hard error mid-generation
        # would be worse than silently doing nothing.
        t = Trigger.when("nonexistent", ">", 0.4)
        ctx = TriggerContext()
        ctx.is_prefill = False
        ctx.probe_scores = {"angry.calm": 0.9}
        assert not t.active(ctx)

    def test_inactive_when_probe_scores_empty(self):
        # Pre-first-forward state: probe_scores is empty.
        t = Trigger.when("angry.calm", ">", 0.4)
        ctx = TriggerContext()
        ctx.is_prefill = False
        assert ctx.probe_scores == {}
        assert not t.active(ctx)

    def test_negative_threshold(self):
        t = Trigger.when("angry.calm", "<", -0.2)
        ctx = TriggerContext()
        ctx.is_prefill = False
        ctx.probe_scores = {"angry.calm": -0.5}
        assert t.active(ctx)
        ctx.probe_scores = {"angry.calm": -0.1}
        assert not t.active(ctx)


# ------------------------------------------------------ TriggerContext ---

class TestTriggerContextProbeScores:
    def test_default_empty_dict(self):
        ctx = TriggerContext()
        assert ctx.probe_scores == {}

    def test_reset_clears_scores(self):
        ctx = TriggerContext()
        ctx.probe_scores = {"angry.calm": 0.7}
        ctx.gen_step = 5
        ctx.reset()
        assert ctx.probe_scores == {}
        assert ctx.gen_step == 0

    def test_independent_instances_dont_share_state(self):
        a = TriggerContext()
        b = TriggerContext()
        a.probe_scores["x"] = 0.3
        # Mutable default with field(default_factory=dict) — each
        # instance must get its own dict, not a shared module-level one.
        assert b.probe_scores == {}


# ------------------------------------------------------------- grammar ---

class TestGrammar:
    @pytest.fixture(autouse=True)
    def _isolated_home(self, monkeypatch, tmp_path):
        from saklas.io import selectors as _sel
        monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
        _sel.invalidate()
        yield
        _sel.invalidate()

    def test_parse_basic_gate(self):
        s = parse_expr("0.3 angry.calm@when:angry.calm>0.4")
        assert "angry.calm" in s.alphas
        coeff, trig = s.alphas["angry.calm"]
        assert coeff == 0.3
        assert trig.gate == ProbeGate(
            probe="angry.calm", op=">", threshold=0.4,
        )
        assert trig.prompt is False  # implicit from Trigger.when

    def test_parse_all_four_ops(self):
        for op in (">", ">=", "<", "<="):
            s = parse_expr(f"0.5 angry.calm@when:angry.calm{op}0.3")
            _coeff, trig = s.alphas["angry.calm"]
            assert trig.gate is not None
            assert trig.gate.op == op

    def test_parse_negative_threshold(self):
        s = parse_expr("0.5 angry.calm@when:angry.calm<-0.5")
        _coeff, trig = s.alphas["angry.calm"]
        assert trig.gate is not None
        assert trig.gate.threshold == -0.5

    def test_parse_unipolar_probe_name(self):
        # Probes don't have to be bipolar — ``manipulative`` is monopolar.
        s = parse_expr("0.5 angry.calm@when:manipulative>0.3")
        _coeff, trig = s.alphas["angry.calm"]
        assert trig.gate is not None
        assert trig.gate.probe == "manipulative"

    def test_format_round_trip(self):
        text = "0.3 angry.calm@when:angry.calm>0.4"
        s = parse_expr(text)
        out = format_expr(s)
        assert out == text
        # Re-parse produces an equal IR (alphas dict equality).
        s2 = parse_expr(out)
        assert dict(s.alphas) == dict(s2.alphas)

    def test_format_negative_threshold_round_trip(self):
        text = "0.5 angry.calm@when:angry.calm<=-0.25"
        s = parse_expr(text)
        out = format_expr(s)
        assert out == text

    def test_when_without_colon_rejects(self):
        # ``@when`` alone is invalid — the colon-payload is required.
        with pytest.raises(SteeringExprError, match="when"):
            parse_expr("0.5 angry.calm@when")

    def test_when_missing_op_rejects(self):
        with pytest.raises(SteeringExprError):
            parse_expr("0.5 angry.calm@when:angry.calm 0.4")

    def test_when_missing_threshold_rejects(self):
        with pytest.raises(SteeringExprError):
            parse_expr("0.5 angry.calm@when:angry.calm>")

    def test_unknown_trigger_message_mentions_when(self):
        with pytest.raises(SteeringExprError) as ei:
            parse_expr("0.5 angry.calm@bogus")
        msg = str(ei.value)
        assert "when" in msg

    def test_compose_with_other_terms(self):
        s = parse_expr(
            "0.3 angry.calm@when:angry.calm>0.4 + 0.2 happy.sad"
        )
        # Both terms land in alphas with their respective triggers.
        _c, t1 = s.alphas["angry.calm"]
        assert t1.gate is not None
        e2 = s.alphas["happy.sad"]
        # Bare-float entry — inherits Steering.trigger (= BOTH default).
        assert isinstance(e2, float)


# ------------------------------------------ session probe-gate detection ---

class _StubSession(SaklasSession):
    """Stub for ``_steering_needs_probe_gating`` testing.

    Bypasses model load; only the steering stack helpers are exercised.
    """

    def __init__(self) -> None:  # type: ignore[override]
        from saklas.core.hooks import DEFAULT_THETA_MAX
        self._profiles = {}
        self._steering_stack = []
        self._steering_override_stack = []
        self._injection_mode = "angular"
        self._theta_max = DEFAULT_THETA_MAX
        self._projection_metric = "mahalanobis"
        self._whitener = None
        self._layer_means = {}
        self.events = EventBus()


class TestSessionProbeGateDetection:
    def test_empty_stack_returns_false(self):
        s = _StubSession()
        assert s._steering_needs_probe_gating() is False

    def test_no_gates_returns_false(self):
        s = _StubSession()
        s._steering_stack.append({
            "angry.calm": (0.5, Trigger.BOTH),
        })
        assert s._steering_needs_probe_gating() is False

    def test_after_thinking_no_gate_returns_false(self):
        # Preset triggers without a gate stay false — ``AFTER_THINKING``
        # has gate=None.
        s = _StubSession()
        s._steering_stack.append({
            "angry.calm": (0.5, Trigger.AFTER_THINKING),
        })
        assert s._steering_needs_probe_gating() is False

    def test_gate_in_first_scope_returns_true(self):
        s = _StubSession()
        gate_trig = Trigger.when("angry.calm", ">", 0.4)
        s._steering_stack.append({
            "calm": (0.5, gate_trig),
        })
        assert s._steering_needs_probe_gating() is True

    def test_gate_in_outer_scope_under_inner(self):
        s = _StubSession()
        # Outer has gate, inner doesn't — flatten = inner-wins, but the
        # outer scope's "calm" key is still in the head if not shadowed.
        gate_trig = Trigger.when("angry.calm", ">", 0.4)
        s._steering_stack.append({
            "calm": (0.5, gate_trig),
        })
        s._steering_stack.append({
            "happy.sad": (0.3, Trigger.BOTH),
        })
        assert s._steering_needs_probe_gating() is True

    def test_inner_overrides_outer_gate(self):
        # Inner shadows the outer's gated entry under the same key, so
        # the active stack head has no gate and detection should be
        # False — important for the "remove gate by re-pushing without
        # one" pattern.
        s = _StubSession()
        gate_trig = Trigger.when("angry.calm", ">", 0.4)
        s._steering_stack.append({
            "calm": (0.5, gate_trig),
        })
        s._steering_stack.append({
            "calm": (0.5, Trigger.BOTH),
        })
        assert s._steering_needs_probe_gating() is False


# -------------------------------------------------- Steering integration ---

class TestSteeringIntegration:
    def test_normalized_entries_propagates_gate(self):
        s = parse_expr("0.5 angry.calm@when:angry.calm>0.3")
        out = s.normalized_entries()
        assert "angry.calm" in out
        _alpha, trig = out["angry.calm"]
        assert trig.gate is not None

    def test_steering_str_round_trips_gate(self):
        s = parse_expr("0.3 angry.calm@when:angry.calm>=0.5")
        # ``Steering.__str__`` delegates to ``format_expr``.
        assert str(s) == "0.3 angry.calm@when:angry.calm>=0.5"

    def test_steering_from_value_accepts_gate_string(self):
        s = Steering.from_value("0.5 calm@when:angry.calm>0.4")
        assert s is not None
        # Pole-resolver folds 'calm' → 'angry.calm' with sign flip.
        assert "angry.calm" in s.alphas
