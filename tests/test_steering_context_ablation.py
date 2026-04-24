"""CPU tests for ablation dispatch through _SteeringContext / _rebuild_steering_hooks.

Builds a ``SaklasSession`` skeleton by hand (bypassing ``__init__`` via
``__new__``) so no model load is required.  Safe because these tests never
call generate() / extract() / anything that touches the model -- only the
steering-stack manipulation and hook-manager wiring is exercised.
"""
from __future__ import annotations

import pytest
import torch

from saklas.cli import selectors as _sel
from saklas.core.events import EventBus
from saklas.core.hooks import SteeringManager
from saklas.core.session import SaklasSession
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
    with pytest.raises(Exception):
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
