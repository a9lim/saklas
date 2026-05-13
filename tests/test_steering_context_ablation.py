"""CPU tests for ablation dispatch through _SteeringContext / _rebuild_steering_hooks.

Builds a ``SaklasSession`` skeleton by hand (bypassing ``__init__`` via
``__new__``) so no model load is required.  Safe because these tests never
call generate() / extract() / anything that touches the model -- only the
steering-stack manipulation and hook-manager wiring is exercised.
"""
from __future__ import annotations

import pytest
import torch

from saklas.io import selectors as _sel
from saklas.core.events import EventBus
from saklas.core.hooks import SteeringManager
from saklas.core.session import SaklasSession, VectorNotRegisteredError
from saklas.core.steering import Steering
from saklas.core.steering_expr import AblationTerm
from saklas.core.triggers import Trigger


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    """Keep parser pole-resolution from scanning the user's real vectors dir."""
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()
    yield
    _sel.invalidate()


class _NoopModule(torch.nn.Module):
    def forward(self, x):  # type: ignore[override]
        return (x,)


def _skeleton_session() -> SaklasSession:
    import threading
    session = SaklasSession.__new__(SaklasSession)
    session._model = None  # type: ignore[attr-defined]
    session._tokenizer = None  # type: ignore[attr-defined]
    session._layers = torch.nn.ModuleList(
        [_NoopModule(), _NoopModule(), _NoopModule()]
    )
    session._device = torch.device("cpu")
    session._dtype = torch.float32
    session._profiles = {}  # type: ignore[attr-defined]
    session._layer_means = {}  # type: ignore[attr-defined]
    session._steering = SteeringManager()
    session._steering_stack = []
    session._steering_override_stack = []  # type: ignore[attr-defined]
    # v2.2: _push_steering / _pop_steering acquire _gen_lock; skeleton
    # mode never runs gen so the lock is uncontended, but the ``with
    # self._gen_lock:`` block needs the attribute to exist.
    session._gen_lock = threading.RLock()  # type: ignore[attr-defined]
    # Phase guard the push/pop methods read to reject callback
    # reentry — skeleton sessions are always idle.
    from saklas.core.session import GenState
    session._gen_phase = GenState.IDLE  # type: ignore[attr-defined]
    session._internal_steering_pop = False  # type: ignore[attr-defined]
    session._injection_mode = "additive"  # type: ignore[attr-defined]
    session._theta_max = 1.5707963267948966  # type: ignore[attr-defined]
    session._projection_metric = "mahalanobis"  # type: ignore[attr-defined]
    session._whitener = None  # type: ignore[attr-defined]
    # Skeleton session has no real model — stub the lazy whitener
    # property to ``None`` so ``_materialize_projections`` falls back
    # to Euclidean per-layer transparently.
    type(session).whitener = property(lambda _self: None)  # type: ignore[attr-defined]
    session.events = EventBus()
    session._history = []
    return session


def test_session_steering_dispatches_ablation_to_manager():
    session = _skeleton_session()
    session._profiles["refusal"] = {1: torch.tensor([1.0, 0.0, 0.0])}
    session._layer_means[1] = torch.tensor([0.5, 0.0, 0.0])

    steering = Steering(alphas={
        "!refusal": AblationTerm(coeff=1.0, trigger=Trigger.BOTH, target="refusal"),
    })
    with session.steering(steering):
        assert "refusal" in session._steering.ablations
        entry = session._steering.ablations["refusal"]
        assert entry["alpha"] == 1.0
        assert entry["trigger"] == Trigger.BOTH
        assert 1 in session._steering.hooks
        assert session._steering.hooks[1].ablation_groups

    # Post-exit: ablation cleared, hook detached.
    assert not session._steering.ablations
    assert not session._steering.hooks


def test_session_steering_ablation_missing_profile_raises():
    session = _skeleton_session()
    steering = Steering(alphas={
        "!nonexistent": AblationTerm(
            coeff=1.0, trigger=Trigger.BOTH, target="nonexistent",
        ),
    })
    with pytest.raises(VectorNotRegisteredError):
        with session.steering(steering):
            pass


def test_session_steering_string_with_ablation_end_to_end():
    """session.steering('!refusal') string form parses through to the manager."""
    session = _skeleton_session()
    session._profiles["refusal"] = {1: torch.tensor([1.0, 0.0, 0.0])}
    session._layer_means[1] = torch.tensor([0.5, 0.0, 0.0])

    with session.steering("!refusal"):
        assert "refusal" in session._steering.ablations
        entry = session._steering.ablations["refusal"]
        assert entry["alpha"] == 1.0
        assert entry["trigger"] == Trigger.BOTH

    assert not session._steering.ablations
